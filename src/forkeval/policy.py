"""What decides the agent's next move, given the conversation so far.

A policy is a pure function of the message list (plus its own sampling). It
holds no per-run state, which is what makes forking a run trivial: copy the
messages, fork the sandbox, call the same policy.

Messages are plain JSON-serialisable dicts in the Messages API shape:
  {"role": "user", "content": "<instruction>"}
  {"role": "assistant", "content": [<blocks>]}
  {"role": "user", "content": [{"type": "tool_result", ...}]}
"""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


class BudgetExceeded(RuntimeError):
    """The experiment's allowance (tokens, or a share of a subscription's limit) is spent.
    Nothing more is sent to the model."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: What the provider says the call cost at list price, when it says (the CLI does;
    #: on a subscription nothing is charged, and this is the API-equivalent figure).
    cost_usd: float = 0.0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class Turn:
    #: The assistant message content, to be appended to the conversation as is.
    content: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    #: "tool_use" (the run continues), or why it ended: "end_turn",
    #: "max_tokens", "refusal".
    stop: str
    usage: Usage = field(default_factory=Usage)


class Policy(Protocol):
    name: str

    async def act(self, messages: list[Message]) -> Turn: ...


TOOLS: list[dict[str, Any]] = [
    {
        "name": "shell",
        "description": (
            "Run a shell command in the task's Linux environment and return its exit code, stdout and stderr. "
            "The environment keeps its state between calls: files, and processes started in the background "
            "(for anything that must keep running, detach it and redirect its output, or the call does not "
            "return: nohup CMD >/tmp/CMD.log 2>&1 </dev/null &). Each call is a new shell, so `cd` and "
            "exported variables do not carry over."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command line, run with sh -c."}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content. Parent directories are created.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path."},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

SYSTEM_PROMPT = (
    "You are completing a task in a Linux environment that you reach only through the tools provided. "
    "Work step by step: inspect before you change things, and verify your work before you finish. "
    "When the task is complete, reply with a short summary and no tool call. "
    "You cannot ask the user questions; make reasonable assumptions and proceed."
)


class AnthropicPolicy:
    """A Claude model with the two tools, one tool call per turn."""

    def __init__(self, model: str, *, max_tokens: int = 16000, client: Any = None) -> None:
        import anthropic

        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self._client = client or anthropic.AsyncAnthropic(max_retries=6)

    async def act(self, messages: list[Message]) -> Turn:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            # One call per turn, so "the state after step t" is well defined.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            # The shared prefix of k continuations is read from the cache.
            cache_control={"type": "ephemeral"},
            messages=messages,
        )
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        stop = resp.stop_reason or "end_turn"
        if stop == "tool_use" and not calls:
            stop = "end_turn"
        u = resp.usage
        return Turn(
            content=content,
            tool_calls=calls,
            stop=stop,
            usage=Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ),
        )


_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["shell", "write_file", "done"]},
        "command": {"type": "string", "description": "for shell: the command line, run with sh -c"},
        "path": {"type": "string", "description": "for write_file: absolute path"},
        "content": {"type": "string", "description": "for write_file: the whole file"},
        "summary": {"type": "string", "description": "for done: one or two sentences on what was done"},
    },
    "required": ["action"],
    "additionalProperties": False,
}

_CLI_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " You act one step at a time. Each message gives the task and every step taken so far with its result; "
    "reply with the single next action only. The actions: `shell` runs a command in the task's environment "
    "(state persists between steps: files, and processes started in the background; for anything that must keep "
    "running, detach it and redirect its output, or the step does not return: "
    "nohup CMD >/tmp/CMD.log 2>&1 </dev/null &; each step is a new shell, so `cd` and exported variables do not "
    "carry over); `write_file` creates or overwrites a file; `done` ends the task. You have no other access to "
    "the environment: nothing happens except through these actions."
)


def render_transcript(messages: list[Message]) -> str:
    """The conversation as one prompt, for a model that is called afresh at every step."""
    out = ["# Task", "", str(messages[0]["content"]), "", "# Steps so far", ""]
    step = 0
    for m in messages[1:]:
        for b in m["content"]:
            if b.get("type") == "tool_use":
                step += 1
                out += [f"## Step {step}: {b['name']}", json.dumps(b["input"], ensure_ascii=False, indent=1)]
            elif b.get("type") == "tool_result":
                out += ["Result:", "```", str(b.get("content", "")), "```", ""]
    if step == 0:
        out.append("(none yet)")
    out += ["", "Give the next action."]
    return "\n".join(out)


class ClaudeCodePolicy:
    """A Claude model reached through the Claude Code CLI's headless mode (`claude -p`),
    which runs on a Claude subscription instead of an API key.

    The CLI is used as a bare model call: its system prompt is replaced, its
    tools are off, and it runs in an empty directory so no project
    instructions or memory are loaded. Each step is a fresh process that sees
    the whole transcript as text and returns one action as structured output.

    What differs from `AnthropicPolicy`, and so what results from this policy
    are results about: the model's reasoning is not carried from step to step
    (every step starts cold from the visible transcript), actions come back as
    JSON instead of native tool calls, and the CLI adds a few lines about its
    host (working directory, OS, date) to the context.
    """

    def __init__(self, model: str, *, binary: str = "claude", concurrency: int = 4, timeout: float = 300,
                 max_plan_use: float | None = None, max_week_use: float | None = None) -> None:
        self.model = model
        self.name = f"claude-code:{model}"
        self._binary = binary
        self._timeout = timeout
        #: Stop once any of the subscription's usage windows is this full (0 to 1). The
        #: subscription is shared with whatever else its owner does with Claude, so a
        #: long experiment should leave some of it.
        self._max_plan_use = max_plan_use
        #: The same for the weekly window, which is the scarcer one; without it, the limit above applies to it too.
        self._max_week_use = max_week_use if max_week_use is not None else max_plan_use
        #: Latest share used of each window the CLI reports, e.g. {"five_hour": 0.11, "seven_day": 0.41}.
        self.plan_use: dict[str, float] = {}
        self._slots = asyncio.Semaphore(concurrency)
        self._cwd = tempfile.mkdtemp(prefix="forkeval-cc-")

    #: A failed call is tried again after these waits (seconds): a rate limit or a dropped
    #: connection passes; a spent subscription allowance does not, and then the run stops.
    RETRY_WAITS = (5, 20, 60)

    #: Within this much of a limit nothing new is started, so that the runs in progress
    #: can finish inside it instead of being cut off half-way and thrown away.
    WIND_DOWN = 0.05

    def _at(self, margin: float) -> dict[str, float]:
        """The usage windows that are within `margin` of their limit."""
        if self._max_plan_use is None:
            return {}
        limit = {"seven_day": self._max_week_use}
        return {w: u for w, u in self.plan_use.items() if u >= limit.get(w, self._max_plan_use) - margin}

    @property
    def closing(self) -> bool:
        """True when the runner should start nothing new."""
        return bool(self._at(self.WIND_DOWN))

    async def _call(self, prompt: str) -> dict[str, Any]:
        full = self._at(0.0)
        if full:
            raise BudgetExceeded(f"subscription usage {full} reached its limit "
                                 f"({self._max_plan_use:.2f}; weekly {self._max_week_use:.2f})")
        for wait in (*self.RETRY_WAITS, None):
            try:
                return await self._call_once(prompt)
            except RuntimeError:
                if wait is None:
                    raise
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def _call_once(self, prompt: str) -> dict[str, Any]:
        args = [
            self._binary, "-p", "--model", self.model,
            "--system-prompt", _CLI_SYSTEM_PROMPT,
            "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
            # The event stream, not the single result object: it also carries the subscription's usage.
            "--output-format", "stream-json", "--verbose", "--json-schema", json.dumps(_ACTION_SCHEMA),
        ]
        async with self._slots:
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=self._cwd, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), self._timeout)
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError(f"claude -p did not answer within {self._timeout:.0f}s") from None
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exited {proc.returncode}: {(err or out).decode(errors='replace')[:500]}")
        reply: dict[str, Any] = {}
        for line in out.decode(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "rate_limit_event":
                windows = (event.get("rate_limit_info") or {}).get("unifiedWindows") or {}
                for name, w in windows.items():
                    if isinstance(w, dict) and isinstance(w.get("utilization"), (int, float)):
                        self.plan_use[name] = float(w["utilization"])
            elif event.get("type") == "result" or not reply:
                reply = event
        if reply.get("type") != "result":
            raise RuntimeError(f"claude -p gave no result: {out.decode(errors='replace')[:500]}")
        if reply.get("is_error"):
            raise RuntimeError(f"claude -p: {str(reply.get('result'))[:500]}")
        return reply

    async def act(self, messages: list[Message]) -> Turn:
        prompt = render_transcript(messages)
        reply: dict[str, Any] = {}
        action: Any = None
        used = Usage()
        # A reply without a valid action (the model answered in prose) is asked for once more.
        for _ in range(2):
            reply = await self._call(prompt)
            used = _add(used, _cli_usage(reply))
            action = reply.get("structured_output")
            if isinstance(action, dict) and action.get("action") in ("shell", "write_file", "done"):
                break
            action = None
        if action is None:
            return Turn(content=[{"type": "text", "text": str(reply.get("result", ""))}], tool_calls=[],
                        stop="no_action", usage=used)
        if action["action"] == "done":
            return Turn(content=[{"type": "text", "text": str(action.get("summary", "done"))}], tool_calls=[],
                        stop="end_turn", usage=used)
        if action["action"] == "shell":
            call = ToolCall(f"cc-{uuid.uuid4().hex[:12]}", "shell", {"command": str(action.get("command", ""))})
        else:
            call = ToolCall(f"cc-{uuid.uuid4().hex[:12]}", "write_file",
                            {"path": str(action.get("path", "")), "content": str(action.get("content", ""))})
        return Turn(content=[{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}],
                    tool_calls=[call], stop="tool_use", usage=used)


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(a.input_tokens + b.input_tokens, a.output_tokens + b.output_tokens,
                 a.cache_read_tokens + b.cache_read_tokens, a.cache_write_tokens + b.cache_write_tokens,
                 a.cost_usd + b.cost_usd)


def _cli_usage(reply: dict[str, Any]) -> Usage:
    """Token counts for one CLI call. `modelUsage` covers every request the call made
    (returning structured output takes two); `usage` is the fallback."""
    per_model = reply.get("modelUsage") or {}
    if per_model:
        return Usage(
            input_tokens=sum(int(m.get("inputTokens", 0)) for m in per_model.values()),
            output_tokens=sum(int(m.get("outputTokens", 0)) for m in per_model.values()),
            cache_read_tokens=sum(int(m.get("cacheReadInputTokens", 0)) for m in per_model.values()),
            cache_write_tokens=sum(int(m.get("cacheCreationInputTokens", 0)) for m in per_model.values()),
            cost_usd=float(reply.get("total_cost_usd") or sum(float(m.get("costUSD", 0)) for m in per_model.values())),
        )
    u = reply.get("usage") or {}
    return Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)),
                 int(u.get("cache_read_input_tokens", 0)), int(u.get("cache_creation_input_tokens", 0)))


class GatesPolicy:
    """A synthetic agent with a known answer, for validating the harness.

    The task has `len(p)` gates. At step i the policy opens gate i correctly
    with probability p[i], wrongly otherwise, then stops. The run succeeds only
    if every gate was opened correctly, so the true value of every state is
    known in closed form (`true_value`), and the estimates the harness produces
    from real forks can be checked against it.
    """

    def __init__(self, p: list[float], *, command: str = "python3 /work/gate.py set {i} {v}", seed: int | None = None) -> None:
        self.p = list(p)
        self.name = "gates:" + ",".join(f"{x:g}" for x in self.p)
        self._command = command
        self._rng = random.Random(seed)

    async def act(self, messages: list[Message]) -> Turn:
        i = sum(1 for m in messages if m["role"] == "assistant")
        if i >= len(self.p):
            return Turn(content=[{"type": "text", "text": "done"}], tool_calls=[], stop="end_turn")
        v = "ok" if self._rng.random() < self.p[i] else "bad"
        call = ToolCall(f"syn-{i}", "shell", {"command": self._command.format(i=i, v=v)})
        return Turn(
            content=[{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}],
            tool_calls=[call],
            stop="tool_use",
        )

    def true_value(self, messages: list[Message]) -> float:
        """P(success) from the state these messages describe."""
        done = 0
        for m in messages:
            if m["role"] != "assistant":
                continue
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    if b["input"]["command"].rstrip().endswith(" bad"):
                        return 0.0
                    done += 1
        v = 1.0
        for x in self.p[done:]:
            v *= x
        return v
