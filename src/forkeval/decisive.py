"""The step that decided a run, found from the values of the states along it.

A trunk's states are numbered by step: state s is the world after s tool
results. Their values V(s) come from forks. The run's own outcome is one more
point at the end, step T + 1, after the turn that ended the run. The decisive
move of a failed run is the biggest drop in V between two neighbouring measured
points; of a passed run, the biggest rise. When the two points are one step
apart the move is a single action, the one the run took between them: that is
the decisive step. Action numbers match the judge's transcript: action s + 1
leads from state s to state s + 1, and action T + 1 is the turn that ended the run.

Pure functions; no I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

#: The bisection goes on while the biggest move is at least this (2 of 6 continuations).
MIN_CHANGE = 1 / 3
#: A decisive step counts as clear when V moves at least this much across it.
CLEAR_CHANGE = 0.5


@dataclass(frozen=True)
class Move:
    #: Step of the state before the move, and of the state after it (T + 1 for the end of the run).
    before: int
    after: int
    #: Change of V across the move, in the direction of the outcome: the drop for a failed run, the rise for a passed one.
    change: float

    @property
    def actions(self) -> range:
        """The actions the run took between the two states."""
        return range(self.before + 1, self.after + 1)

    @property
    def single(self) -> bool:
        return self.after - self.before == 1

    @property
    def clear(self) -> bool:
        return self.single and self.change >= CLEAR_CHANGE - 1e-9


def biggest_move(steps: Sequence[int], values: Sequence[float | None], outcome: bool,
                 trunk_steps: int) -> Move | None:
    """The biggest move towards the outcome between neighbouring measured points,
    the run's own end included. Ties go to the earlier move."""
    pts = [(s, v) for s, v in zip(steps, values) if v is not None] + [(trunk_steps + 1, float(outcome))]
    best: Move | None = None
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        change = (vb - va) if outcome else (va - vb)
        if best is None or change > best.change + 1e-12:
            best = Move(a, b, change)
    return best


def next_to_measure(steps: Sequence[int], values: Sequence[float | None], available: Sequence[bool],
                    outcome: bool, trunk_steps: int, *, cap: int | None = None,
                    min_change: float = MIN_CHANGE) -> int | None:
    """Which state to measure next (an index into `steps`), or None when done.

    The first state and the last are measured first. Then the move found by
    `biggest_move` is halved, by measuring the available state nearest its
    middle, until it is between two neighbouring snapshots or smaller than
    `min_change` (or `cap` states are measured). `available[i]`: state i has a
    snapshot and has not been measured.
    """
    n = len(steps)
    if n == 0:
        return None
    for i in (0, n - 1):
        if values[i] is None and available[i]:
            return i
    if cap is not None and sum(v is not None for v in values) >= cap:
        return None
    move = biggest_move(steps, values, outcome, trunk_steps)
    if move is None or move.change < min_change - 1e-9:
        return None
    inside = [i for i in range(n) if move.before < steps[i] < move.after and values[i] is None and available[i]]
    if not inside:
        return None
    mid = (move.before + move.after) / 2
    return min(inside, key=lambda i: (abs(steps[i] - mid), steps[i]))


# --- what a judge sees ---------------------------------------------------------

JUDGE_SYSTEM = (
    "You review the transcript of an AI agent that worked on a task in a Linux environment. Each numbered step is "
    "one action the agent took, followed by what the environment returned. The agent did not see the tests that "
    "decided success, and neither do you. Answer with the step that was asked for and a one-sentence reason."
)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "step": {"type": "integer", "description": "the number of the decisive step"},
        "reason": {"type": "string"},
    },
    "required": ["step", "reason"],
}

#: Prompt version 2 asks two questions, about this agent rather than any agent.
JUDGE_SCHEMA_V2: dict[str, Any] = {
    "type": "object",
    "properties": {
        "introduced": {"type": "integer", "description": "the step where the mistake that caused the failure was made"},
        "unrecoverable": {"type": "integer",
                          "description": "the step after which this agent was unlikely to succeed whatever it did next"},
        "reason": {"type": "string"},
    },
    "required": ["introduced", "unrecoverable", "reason"],
}

RESULT_CHARS = 3000


def _clip(text: str) -> str:
    if len(text) <= RESULT_CHARS:
        return text
    half = RESULT_CHARS // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters omitted ...]\n{text[-half:]}"


def judge_prompt(messages: list[dict[str, Any]], version: int = 1) -> tuple[str, int]:
    """A failed run as the judge sees it, and its number of steps (the ending turn included, when there is one).
    Version 1 asks for the step that decided the failure; version 2 asks where the mistake was made and after
    which step this agent, continuing as it was, was unlikely to succeed."""
    out = ["# Task", "", str(messages[0]["content"]), "", "# What the agent did", ""]
    step = 0
    for m in messages[1:]:
        content = m["content"]
        if isinstance(content, str):
            continue
        if _ends(m):
            step += 1
            text = " ".join(str(b.get("text", "")) for b in content if b.get("type") == "text")
            out += [f"## Step {step}: ends the run", text, ""]
            continue
        for b in content:
            if b.get("type") == "tool_use":
                step += 1
                out += [f"## Step {step}: {b['name']}", json.dumps(b["input"], ensure_ascii=False, indent=1)]
            elif b.get("type") == "tool_result":
                out += ["Result:", "```", _clip(str(b.get("content", ""))), "```", ""]
    if version == 2:
        out += [
            "# Questions", "",
            "This run FAILED: the task's tests did not pass at the end. Answer two questions about this agent, the "
            "same model continuing from where it was, without seeing the tests:",
            "1. introduced: at which step was the mistake that caused the failure made?",
            "2. unrecoverable: after which step was this agent unlikely to succeed, whatever it did next?",
            f"The two can be the same step. Any step above is a possible answer, from 1 to {step}; the step that ends "
            "the run counts.",
        ]
        return "\n".join(out), step
    out += [
        "# Question", "",
        "This run FAILED: the task's tests did not pass at the end. Which single step decided the failure, that is, "
        "the step after which the run was unlikely to succeed whatever the agent did next? Any step above is a "
        f"possible answer, from 1 to {step}.",
    ]
    return "\n".join(out), step


def nonzero_exits(messages: list[dict[str, Any]]) -> list[int]:
    """Steps whose command exited non-zero or timed out (a baseline for the judge)."""
    bad, step = [], 0
    for m in messages[1:]:
        content = m["content"]
        if isinstance(content, str):
            continue
        if _ends(m):
            step += 1
            continue
        for b in content:
            if b.get("type") == "tool_use":
                step += 1
            elif b.get("type") == "tool_result":
                head = str(b.get("content", "")).split("\n", 1)[0]
                if head == "timed out" or (head.startswith("exit code: ") and head != "exit code: 0"):
                    bad.append(step)
    return bad


def first_write(messages: list[dict[str, Any]]) -> int | None:
    """The first step that wrote a file (a baseline that needs no model: the first solution attempt)."""
    step = 0
    for m in messages[1:]:
        content = m["content"]
        if isinstance(content, str):
            continue
        if _ends(m):
            step += 1
            continue
        for b in content:
            if b.get("type") == "tool_use":
                step += 1
                if b.get("name") == "write_file":
                    return step
    return None


def cluster_bootstrap(groups: Sequence[Sequence[float]], *, draws: int = 4000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float, float]:
    """Mean of per-case values and a percentile interval from resampling whole groups (tasks) with replacement:
    cases of one task are not independent. Returns (mean, lo, hi)."""
    import random
    groups = [g for g in groups if g]
    flat = [v for g in groups for v in g]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        vals = [v for g in pick for v in g]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    a = (1 - level) / 2
    return sum(flat) / len(flat), stats[int(a * draws)], stats[min(draws - 1, int((1 - a) * draws))]


def _ends(m: dict[str, Any]) -> bool:
    """An assistant turn without a tool call: the one that ended the run."""
    return m["role"] == "assistant" and not any(b.get("type") == "tool_use" for b in m["content"])
