"""Runs: a trunk, the branch points taken along it, and the continuations of each.

A run's state is two things: the message list and a sandbox. One function
(`advance`) moves a state forward by one step; everything else here decides
which states to keep and how many times to continue each.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Callable

from .env import Box, BoxFactory, ExecOutput
from .policy import Message, Policy, ToolCall
from .store import Store
from .tasks import BuildStep, Task

#: A tool result longer than this is cut in the middle; the model sees both ends.
MAX_RESULT_CHARS = 8000


class BudgetExceeded(RuntimeError):
    """The experiment's token budget is spent. Nothing more is sent to the model."""


@dataclass
class Budget:
    store: Store
    experiment: str
    max_tokens: int | None

    def check(self) -> None:
        if self.max_tokens is not None and self.store.tokens_used(self.experiment) >= self.max_tokens:
            raise BudgetExceeded(f"token budget of {self.max_tokens} reached for experiment {self.experiment!r}")


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    half = MAX_RESULT_CHARS // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters omitted ...]\n{text[-half:]}"


def _render(out: ExecOutput) -> str:
    head = "timed out" if out.timed_out else f"exit code: {out.exit_code}"
    parts = [head]
    if out.stdout:
        parts.append(out.stdout.rstrip("\n"))
    if out.stderr:
        parts.append("[stderr]\n" + out.stderr.rstrip("\n"))
    return _clip("\n".join(parts))


async def execute(call: ToolCall, box: Box, task: Task) -> tuple[str, bool]:
    """Run one tool call in the sandbox. Returns (result text, is_error)."""
    try:
        if call.name == "shell":
            out = await box.run(str(call.input["command"]), timeout=task.step_timeout,
                                cwd=task.workdir, env=task.env)
            return _render(out), out.timed_out
        if call.name == "write_file":
            await box.write(str(call.input["path"]), str(call.input["content"]))
            return "written", False
        return f"unknown tool {call.name!r}", True
    except KeyError as e:
        return f"missing argument {e}", True


async def advance(messages: list[Message], box: Box, policy: Policy, task: Task,
                  store: Store, run_id: str, budget: Budget) -> str | None:
    """One step: ask the policy, run its tool calls, append both to `messages`.

    Returns None if the run goes on, or the reason it ended.
    """
    budget.check()
    turn = await policy.act(messages)
    store.add_usage(run_id, turn.usage)
    messages.append({"role": "assistant", "content": turn.content})
    if turn.stop != "tool_use":
        return turn.stop
    results = []
    for call in turn.tool_calls:
        text, is_error = await execute(call, box, task)
        results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": is_error})
    messages.append({"role": "user", "content": results})
    return None


_PYTEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR)\s+\S", re.M)


def pytest_passed(output: str) -> bool:
    """Whether a `pytest -rA` run passed, read from its short summary: at least
    one test, and none failed or errored. A run that never reached pytest has
    no summary and fails."""
    _, sep, summary = output.rpartition("short test summary info")
    if not sep:
        return False
    marks = _PYTEST_LINE.findall(summary)
    return "PASSED" in marks and "FAILED" not in marks and "ERROR" not in marks


async def score(box: Box, task: Task) -> bool:
    """Run the task's test in this sandbox. The test files arrive only now."""
    for path, content in task.test_files.items():
        await box.write(path, content)
    out = await box.run(task.test, timeout=task.test_timeout, cwd=task.workdir, env=task.env)
    if task.test_parser == "pytest":
        return not out.timed_out and pytest_passed(out.stdout)
    return out.ok


async def prepare(box: Box, task: Task) -> None:
    for path, content in task.files.items():
        await box.write(path, content)
    if task.setup:
        out = await box.run(task.setup, timeout=task.step_timeout, cwd=task.workdir, env=task.env)
        if not out.ok:
            raise RuntimeError(f"task {task.id}: setup failed: {_render(out)[:500]}")


#: A build step may install packages or compile; it gets longer than an agent's step.
BUILD_STEP_TIMEOUT = 1800


def built_image_name(task: Task) -> tuple[str, str]:
    """Name and tag of the image a task's build produces. The tag is a hash of
    the base image and every step, so a changed build is a different image and
    an unchanged one is found again."""
    h = hashlib.sha256(task.image.encode())
    for st in task.build:
        for part in (st.run.encode(), st.tar, st.cwd.encode(), repr(st.env).encode()):
            h.update(len(part).to_bytes(8, "big"))
            h.update(part)
    name = "fe-" + re.sub(r"[^a-z0-9]+", "-", task.id.lower()).strip("-")[:48]
    return name, h.hexdigest()[:12]


async def apply_build_step(box: Box, st: BuildStep) -> None:
    if st.tar:
        await box.write(BuildStep.ARCHIVE, st.tar)
    out = await box.run(st.run, timeout=BUILD_STEP_TIMEOUT, cwd=st.cwd, env=dict(st.env))
    if not out.ok:
        raise RuntimeError(f"build step failed: {st.run[:200]!r}: {_render(out)[-800:]}")


def steps_taken(messages: list[Message]) -> int:
    """Tool-result messages so far: the t in "state after step t"."""
    return sum(1 for m in messages[1:] if m["role"] == "user")


async def finish(messages: list[Message], box: Box, policy: Policy, task: Task,
                 store: Store, run_id: str, budget: Budget,
                 after_step: Callable[[int], "asyncio.Future[None] | None"] | None = None) -> None:
    """Drive a state to the end of its run, score it, and record the result."""
    stop: str | None = None
    error: str | None = None
    outcome: bool | None = None
    try:
        while True:
            if steps_taken(messages) >= task.max_steps:
                stop = "step_limit"
                break
            stop = await advance(messages, box, policy, task, store, run_id, budget)
            if stop is not None:
                break
            if after_step is not None:
                await after_step(steps_taken(messages))
        outcome = await score(box, task)
    except BudgetExceeded as e:
        stop, error = "budget", str(e)
        raise
    except Exception as e:  # recorded, then re-raised: an unscored run is visible in the store
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        store.save_transcript(run_id, messages)
        store.end_run(run_id, outcome=outcome, steps=steps_taken(messages), stop=stop, error=error)


def schedule(spec: str) -> Callable[[int], bool]:
    """Which steps get a branch point. "every"; "grid" (0-3, then every third);
    or an explicit list like "0,2,5"."""
    if spec == "every":
        return lambda t: True
    if spec == "grid":
        return lambda t: t <= 3 or t % 3 == 0
    wanted = {int(x) for x in spec.split(",") if x.strip()}
    return lambda t: t in wanted


@dataclass
class _BranchPoint:
    id: str
    box: Box
    messages: list[Message]


class Runner:
    def __init__(self, *, factory: BoxFactory, policy: Policy, store: Store, experiment: str,
                 max_tokens: int | None = None, concurrency: int = 2) -> None:
        self.factory = factory
        self.policy = policy
        self.store = store
        self.experiment = experiment
        self.budget = Budget(store, experiment, max_tokens)
        self._slots = asyncio.Semaphore(concurrency)
        self._images: dict[str, "asyncio.Task[str]"] = {}

    async def image_for(self, task: Task) -> str:
        """The image a run of this task starts from, building it the first time it is needed."""
        if not task.build:
            return task.image
        if task.id not in self._images:
            self._images[task.id] = asyncio.ensure_future(self._build_image(task))
        return await self._images[task.id]

    async def _build_image(self, task: Task) -> str:
        name, tag = built_image_name(task)
        if await self.factory.has_image(f"{name}:{tag}"):
            return f"{name}:{tag}"
        box = await self.factory.create(image=task.image, flavor=task.flavor, ttl=task.ttl,
                                        tags=self._tags(task, "build"))
        try:
            for st in task.build:
                await apply_build_step(box, st)
            return await box.commit(name, tag)
        finally:
            await box.delete()

    def _tags(self, task: Task, kind: str) -> dict[str, str]:
        return {"forkeval": self.experiment, "task": task.id, "kind": kind}

    async def fresh(self, task: Task) -> None:
        """One ordinary run in a sandbox of its own."""
        async with self._slots:
            box = await self.factory.create(image=await self.image_for(task), flavor=task.flavor, ttl=task.ttl,
                                            tags=self._tags(task, "fresh"))
            try:
                await prepare(box, task)
                run_id = self.store.start_run(experiment=self.experiment, task_id=task.id, policy=self.policy.name,
                                              kind="fresh", sandbox_id=box.id)
                messages: list[Message] = [{"role": "user", "content": task.instruction}]
                await finish(messages, box, self.policy, task, self.store, run_id, self.budget)
            finally:
                await box.delete()

    async def check(self, task: Task) -> tuple[bool, bool | None]:
        """Is the task sound here? Returns (untouched environment fails the test,
        environment after the known solution passes it; None without a solution)."""
        async with self._slots:
            image = await self.image_for(task)

            async def outcome(solve: bool) -> bool:
                box = await self.factory.create(image=image, flavor=task.flavor, ttl=task.ttl,
                                                tags=self._tags(task, "check"))
                try:
                    await prepare(box, task)
                    if solve:
                        await box.write("/tmp/.forkeval-solution.sh", task.solution)
                        await box.run("bash /tmp/.forkeval-solution.sh; rm -f /tmp/.forkeval-solution.sh",
                                      timeout=BUILD_STEP_TIMEOUT, cwd=task.workdir, env=task.env)
                    return await score(box, task)
                finally:
                    await box.delete()

            untouched = await outcome(False)
            solved = await outcome(True) if task.solution else None
            return (not untouched, solved)

    async def trunk(self, task: Task, *, k: int, branch_at: Callable[[int], bool]) -> None:
        """One full run with a branch point at every scheduled step, then k
        continuations of each branch point.

        Holds one slot throughout, so at most `concurrency` tasks are in
        flight and the live sandboxes peak near concurrency * k.
        """
        async with self._slots:
            points: list[_BranchPoint] = []
            box = await self.factory.create(image=await self.image_for(task), flavor=task.flavor, ttl=task.ttl,
                                            tags=self._tags(task, "trunk"))
            try:
                try:
                    await prepare(box, task)
                    run_id = self.store.start_run(experiment=self.experiment, task_id=task.id,
                                                  policy=self.policy.name, kind="trunk", sandbox_id=box.id)
                    messages: list[Message] = [{"role": "user", "content": task.instruction}]

                    async def branch(step: int) -> None:
                        if not branch_at(step):
                            return
                        t0 = time.monotonic()
                        (bp_box,) = await box.fork(1)
                        t1 = time.monotonic()
                        # Registered before the suspend, so a failed suspend still gets it deleted.
                        point = _BranchPoint(id="", box=bp_box, messages=copy.deepcopy(messages))
                        points.append(point)
                        await bp_box.suspend()
                        t2 = time.monotonic()
                        true_value = getattr(self.policy, "true_value", None)
                        point.id = self.store.add_branch_point(
                            trunk=run_id, step=step, sandbox_id=bp_box.id,
                            fork_ms=round((t1 - t0) * 1000), suspend_ms=round((t2 - t1) * 1000),
                            true_value=true_value(messages) if true_value else None,
                        )

                    await branch(0)
                    await finish(messages, box, self.policy, task, self.store, run_id, self.budget,
                                 after_step=branch)
                finally:
                    # The branch points hold the trunk's states; its own sandbox is done.
                    await box.delete()
                for point in points:
                    if point.id:
                        await self._continuations(task, point, k)
            finally:
                await asyncio.gather(*(p.box.delete() for p in points), return_exceptions=True)

    async def _continuations(self, task: Task, point: _BranchPoint, k: int) -> None:
        boxes = await point.box.fork(k)
        try:
            results = await asyncio.gather(
                *(self._continue(task, point, b) for b in boxes), return_exceptions=True
            )
        finally:
            await asyncio.gather(*(b.delete() for b in boxes), return_exceptions=True)
        # One continuation erroring is recorded on its own row (outcome NULL)
        # and does not stop its siblings; a spent budget stops everything.
        for r in results:
            if isinstance(r, BudgetExceeded):
                raise r

    async def _continue(self, task: Task, point: _BranchPoint, box: Box) -> None:
        run_id = self.store.start_run(experiment=self.experiment, task_id=task.id, policy=self.policy.name,
                                      kind="cont", branch_point=point.id, sandbox_id=box.id)
        await finish(copy.deepcopy(point.messages), box, self.policy, task, self.store, run_id, self.budget)
