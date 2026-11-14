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

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


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
