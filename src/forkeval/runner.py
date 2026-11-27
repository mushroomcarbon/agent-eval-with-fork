"""Runs: a trunk, the branch points taken along it, and the continuations of each.

A run's state is two things: the message list and a sandbox. One function
(`advance`) moves a state forward by one step; everything else here decides
which states to keep and how many times to continue each.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Callable

from .decisive import biggest_move, next_to_measure
from .env import Box, BoxFactory, ExecOutput
from .policy import BudgetExceeded, Message, Policy, ToolCall
from .store import Store
from .tasks import BuildStep, Task

#: A tool result longer than this is cut in the middle; the model sees both ends.
MAX_RESULT_CHARS = 8000


@dataclass
class Budget:
    store: Store
    experiment: str
    max_tokens: int | None

    def check(self) -> None:
        if self.max_tokens is not None and self.store.tokens_used(self.experiment) >= self.max_tokens:
            raise BudgetExceeded(f"token budget of {self.max_tokens} reached for experiment {self.experiment!r}")


class Holdout:
    """The same policy under another name, for continuations that check a state's value and never choose it."""

    def __init__(self, base: Policy) -> None:
        self.base = base
        self.name = f"{base.name}#holdout"

    async def act(self, messages: list[Message]) -> Turn:
        return await self.base.act(messages)

    def __getattr__(self, attr: str):
        return getattr(self.base, attr)


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


#: `skipped` values of a branch point: the adaptive schedule did not need it; there was no room to
#: snapshot it; the trunk had more snapshots than --max-states; a kept snapshot was gone when the run
#: resumed; the bisection did not need it.
#: SHARED: a state at step 0, which every trunk of a task shares, already measured from another trunk.
SETTLED, NO_ROOM, OVER_LIMIT, LOST, UNNEEDED, SHARED = 1, 2, 3, 4, 5, 6

#: A branch point is a sandbox with a deadline like any other. One that is going to be measured is
#: given this long from the end of its trunk, enough for every continuation of a long task.
MEASURE_SECONDS = 6 * 3600


def spread(n: int, m: int) -> set[int]:
    """m of the indices 0..n-1, evenly spaced, the first and the last among them."""
    if m >= n:
        return set(range(n))
    if m <= 1:
        return {0}
    return {round(i * (n - 1) / (m - 1)) for i in range(m)}


@dataclass
class _BranchPoint:
    id: str
    box: Box | None
    messages: list[Message]
    kept: bool = False  # left on the platform by an earlier stop
    step: int = 0


#: How long a trunk waits for room to snapshot a step after the first before it goes on without one.
SNAPSHOT_WAIT = 120.0


class _Room:
    """Counts sandboxes against the limit the platform gives a tenant."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._free = limit
        self._changed = asyncio.Condition()

    async def take(self, n: int, wait: float | None = None) -> bool:
        """Room for n sandboxes; waits for it (for ever, or `wait` seconds and then False)."""
        if n > self.limit:
            raise ValueError(f"{n} sandboxes at once is over the limit of {self.limit}")
        async with self._changed:
            try:
                await asyncio.wait_for(self._changed.wait_for(lambda: self._free >= n), wait)
            except asyncio.TimeoutError:
                return False
            self._free -= n
            return True

    async def give(self, n: int) -> None:
        async with self._changed:
            self._free += n
            self._changed.notify_all()


class Runner:
    def __init__(self, *, factory: BoxFactory, policy: Policy, store: Store, experiment: str,
                 max_tokens: int | None = None, concurrency: int = 2, max_sandboxes: int = 45,
                 keep_seconds: int = 0, swap_policy: Policy | None = None, swap_k: int = 6,
                 confirm_k: int = 0, swap_at: str = "all", holdout_k: int = 0) -> None:
        self.factory = factory
        self.policy = policy
        #: A second policy that continues some of the first one's states (with --bisect: step 0, once per
        #: task, and either side of a failed run's decisive step), `swap_k` times each.
        self.swap_policy = swap_policy
        self.swap_k = swap_k
        self._swap_starts: set[str] = set()
        #: "all": step 0 and both sides of the decisive step; "after": only the state just after it.
        if swap_at not in ("all", "after"):
            raise ValueError(f"swap_at must be 'all' or 'after', not {swap_at!r}")
        self.swap_at = swap_at
        #: With --bisect: continuations each of the two states either side of the decisive step end up with.
        self.confirm_k = confirm_k
        #: With --bisect: fresh continuations of the two states either side of a failed run's clear decisive
        #: step, recorded under "<policy>#holdout". They play no part in finding the step, so they give an
        #: estimate of its size that the selection did not touch.
        self.holdout_k = holdout_k
        self.holdout_policy = Holdout(policy)
        self.store = store
        self.experiment = experiment
        self.budget = Budget(store, experiment, max_tokens)
        self._slots = asyncio.Semaphore(concurrency)
        self._spent: BudgetExceeded | None = None
        self._room = _Room(max_sandboxes)
        #: When a budget stops a trunk that has finished, its unmeasured snapshots stay on the
        #: platform this long so that `resume` can measure them later. 0: they are deleted.
        self.keep_seconds = keep_seconds
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
        async with self._slot():
            await self._room.take(1)
            try:
                box = await self.factory.create(image=await self.image_for(task), flavor=task.flavor, ttl=task.ttl,
                                                tags=self._tags(task, "fresh"))
                try:
                    await prepare(box, task)
                    run_id = self.store.start_run(experiment=self.experiment, task_id=task.id,
                                                  policy=self.policy.name, kind="fresh", sandbox_id=box.id)
                    messages: list[Message] = [{"role": "user", "content": task.instruction}]
                    await finish(messages, box, self.policy, task, self.store, run_id, self.budget)
                finally:
                    await box.delete()
            finally:
                await self._room.give(1)

    async def grader(self, task: Task, k: int) -> tuple[int, int]:
        """Is the task's test deterministic? Applies the reference solution once, forks the result k times and
        runs the test in every fork at the same time (as continuations run). Returns (passes, scored)."""
        async with self._slots:
            image = await self.image_for(task)
            box = await self.factory.create(image=image, flavor=task.flavor, ttl=task.ttl,
                                            tags=self._tags(task, "grader"))
            forks: list[Box] = []
            try:
                await prepare(box, task)
                await box.write("/tmp/.forkeval-solution.sh", task.solution or "")
                await box.run("bash /tmp/.forkeval-solution.sh; rm -f /tmp/.forkeval-solution.sh",
                              timeout=BUILD_STEP_TIMEOUT, cwd=task.workdir, env=task.env)
                forks = await box.fork(k)
                got = await asyncio.gather(*(score(f, task) for f in forks), return_exceptions=True)
                ok = [g for g in got if isinstance(g, bool)]
                return sum(ok), len(ok)
            finally:
                await asyncio.gather(*(b.delete() for b in [box, *forks]), return_exceptions=True)

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

    @contextlib.asynccontextmanager
    async def _slot(self):
        """One of the `concurrency` places. Once the budget is spent, jobs still waiting
        for a place end at once instead of each starting a sandbox to find that out."""
        async with self._slots:
            if self._spent is None and self._closing():
                self._spent = BudgetExceeded("the subscription is close to its limit: nothing new is started")
            if self._spent is not None:
                raise self._spent
            try:
                yield
            except BudgetExceeded as e:
                self._spent = e
                raise

    def _closing(self) -> bool:
        return any(getattr(p, "closing", False) for p in (self.policy, self.swap_policy) if p is not None)

    async def _drop(self, point: _BranchPoint) -> None:
        if point.box is not None:
            gone, point.box = point.box, None
            try:
                await gone.delete()
            finally:
                if point.kept:
                    self.store.keep_point(point.id, False)
                await self._room.give(1)

    async def trunk(self, task: Task, *, k: int, branch_at: Callable[[int], bool], adaptive: bool = False,
                    max_states: int | None = None, bisect: bool = False) -> None:
        """One full run with a branch point at every scheduled step, then k
        continuations of each branch point.

        A branch point is a suspended copy of the sandbox and costs no model
        calls; the continuations are where the tokens go. Two things limit them.
        With `max_states`, a trunk with more snapshots than that has only that
        many measured, evenly spaced from the first to the last; the rest are
        recorded as skipped. With `adaptive`, they are measured in step order
        and that stops after the first state whose continuations all end the
        way the trunk did: from there the run looks settled, and the later
        states are recorded as skipped. A run that was never in doubt then
        costs k continuations instead of k per step. What this gives up: a
        skipped state can still be open (all k agreeing does not make the value
        exactly 0 or 1), and `decided_by_step` treats a skipped state like the
        last one measured before it.

        Holds one slot throughout, so at most `concurrency` tasks are in flight.
        Sandboxes are counted against `max_sandboxes` (a tenant has a limit, and
        a suspended branch point counts): a trunk holds room for itself and its
        k continuations from the start, so it can always finish, and takes room
        for each branch point as it goes. When there is none for a step after
        the first within SNAPSHOT_WAIT, that step is recorded as not
        snapshotted and the run goes on. Branch points are deleted as soon as
        they are done with.

        With `bisect`, the states measured are chosen to find the decisive
        step (`decisive.next_to_measure`): the first and the last, then halving
        the biggest move towards the outcome until it is one action wide;
        `max_states` caps how many. The rest are recorded as not needed. Every
        snapshot is held until then, so a trunk holds one sandbox per step.
        """
        async with self._slot():
            await self._room.take(1 + k)
            points: list[_BranchPoint] = []
            try:
                box = await self.factory.create(image=await self.image_for(task), flavor=task.flavor, ttl=task.ttl,
                                                tags=self._tags(task, "trunk"))
                try:
                    await prepare(box, task)
                    run_id = self.store.start_run(experiment=self.experiment, task_id=task.id,
                                                  policy=self.policy.name, kind="trunk", sandbox_id=box.id)
                    messages: list[Message] = [{"role": "user", "content": task.instruction}]
                    true_value = getattr(self.policy, "true_value", None)

                    async def branch(step: int) -> None:
                        if not branch_at(step):
                            return
                        value = true_value(messages) if true_value else None
                        if not await self._room.take(1, None if step == 0 else SNAPSHOT_WAIT):
                            self.store.add_branch_point(trunk=run_id, step=step, sandbox_id=None, fork_ms=None,
                                                        suspend_ms=None, true_value=value, skipped=NO_ROOM)
                            return
                        t0 = time.monotonic()
                        try:
                            (bp_box,) = await box.fork(1)
                        except BaseException:
                            await self._room.give(1)
                            raise
                        t1 = time.monotonic()
                        # Registered before the suspend, so a failed suspend still gets it deleted.
                        point = _BranchPoint(id="", box=bp_box, messages=copy.deepcopy(messages), step=step)
                        points.append(point)
                        await bp_box.suspend()
                        t2 = time.monotonic()
                        point.id = self.store.add_branch_point(
                            trunk=run_id, step=step, sandbox_id=bp_box.id,
                            fork_ms=round((t1 - t0) * 1000), suspend_ms=round((t2 - t1) * 1000), true_value=value,
                        )
                        self.store.save_transcript(point.id, point.messages)

                    await branch(0)
                    await finish(messages, box, self.policy, task, self.store, run_id, self.budget,
                                 after_step=branch)
                finally:
                    # The branch points hold the trunk's states; its own sandbox is done.
                    await box.delete()
                live = [p for p in points if p.id]
                if max_states is not None and not bisect:
                    chosen = spread(len(live), max_states)
                    for i, point in enumerate(live):
                        if i not in chosen:
                            self.store.skip_branch_point(point.id, OVER_LIMIT)
                            await self._drop(point)
                    live = [p for i, p in enumerate(live) if i in chosen]
                await asyncio.gather(*(p.box.extend(MEASURE_SECONDS) for p in live if p.box is not None),
                                     return_exceptions=True)
                if bisect:
                    await self._bisect(task, run_id, live, k, max_states)
                else:
                    await self._measure(task, run_id, live, k, adaptive)
            finally:
                await asyncio.gather(*(self._drop(p) for p in points), return_exceptions=True)
                await self._room.give(1 + k)

    async def resume(self, task: Task, trunk_id: str, *, k: int, adaptive: bool = False, bisect: bool = False,
                     max_states: int | None = None) -> None:
        """Measure what is left of a trunk that a budget stopped after it had
        finished: its unmeasured branch points were kept on the platform
        (`keep_seconds`), and their states' messages are in the store."""
        async with self._slot():
            rows = [r for r in self.store.points(trunk_id) if not r["skipped"] or (bisect and r["skipped"] == SHARED)]
            held = sum(1 for r in rows if r["kept"])
            await self._room.take(k + held)
            points: list[_BranchPoint] = []
            try:
                for r in rows:
                    box = await self.factory.attach(r["sandbox_id"]) if r["kept"] else None
                    if r["kept"] and box is None:
                        await self._room.give(1)
                    messages = self.store.load_transcript(r["id"]) if box is not None else []
                    points.append(_BranchPoint(id=r["id"], box=box, messages=messages, kept=bool(r["kept"]),
                                               step=r["step"]))
                if bisect:
                    await self._bisect(task, trunk_id, points, k, max_states)
                else:
                    await self._measure(task, trunk_id, points, k, adaptive)
            finally:
                await asyncio.gather(*(self._drop(p) for p in points), return_exceptions=True)
                await self._room.give(k)

    async def _measure(self, task: Task, trunk_id: str, points: list[_BranchPoint], k: int, adaptive: bool) -> None:
        """Continuations for the branch points of a finished trunk, in step order.
        A point without a sandbox was measured in an earlier session (or its kept
        snapshot is gone); it still says whether the run was settled by then."""
        outcome = self.store.run_outcome(trunk_id)
        me = self.policy.name
        settled = False
        try:
            for point in points:
                if settled:
                    self.store.skip_branch_point(point.id, SETTLED)
                    await self._drop(point)
                    continue
                if point.box is None and point.kept:
                    self.store.skip_branch_point(point.id, LOST)
                    continue
                if point.box is not None:
                    need = k - self.store.point_result(point.id, me)[1]
                    if need > 0:
                        if self._closing():
                            raise BudgetExceeded("the subscription is close to its limit: nothing new is started")
                        await self._continuations(task, point, need)
                    await self._drop(point)
                if adaptive and outcome is not None:
                    passes, n = self.store.point_result(point.id, me)
                    settled = n >= 2 and passes == (n if outcome else 0)
        except BudgetExceeded:
            if outcome is not None:
                await self._keep(points)
            raise

    async def _keep(self, points: list[_BranchPoint]) -> None:
        """A budget stopped a finished trunk: leave its snapshots on the platform for `resume`."""
        if not self.keep_seconds:
            return
        for point in points:
            if point.box is None:
                continue
            try:
                await point.box.extend(self.keep_seconds)
            except Exception:
                continue  # not kept: the caller deletes it
            self.store.keep_point(point.id, True)
            point.box = None
            await self._room.give(1)

    async def _bisect(self, task: Task, trunk_id: str, points: list[_BranchPoint], k: int,
                      cap: int | None) -> None:
        """Measure the states that find the decisive step of a finished trunk, then (with a
        swap policy) continue some of them with the second policy as well."""
        row = self.store.db.execute("SELECT outcome, steps FROM runs WHERE id = ?", (trunk_id,)).fetchone()
        if row is None or row["outcome"] is None:
            return
        outcome, trunk_steps = bool(row["outcome"]), int(row["steps"])
        me = self.policy.name
        tried: set[int] = set()
        confirmed: set[int] = set()
        try:
            for point in points:
                if point.box is None and point.kept:
                    self.store.skip_branch_point(point.id, LOST)
            points = [p for p in points if not (p.box is None and p.kept)]
            steps = [p.step for p in points]
            # A state a stop interrupted is topped up first.
            for i, point in enumerate(points):
                n = self.store.point_result(point.id, me)[1]
                if point.box is not None and 0 < n < k:
                    await self._run_more(task, point, k - n, self.policy)
                    tried.add(i)
            # Every trunk of a task starts from the same state: one trunk's step 0 serves them all.
            shared = None
            if steps and steps[0] == 0 and self.store.point_result(points[0].id, me)[1] == 0:
                x, n = self.store.start_result(self.experiment, task.id, me)
                if n >= k:
                    shared = (x, n)
                    self.store.skip_branch_point(points[0].id, SHARED)
            while True:
                res = [self.store.point_result(p.id, me) for p in points]
                if shared is not None:
                    res[0] = shared
                values = [x / n if n else None for x, n in res]
                available = [p.box is not None and n == 0 and i not in tried
                             for i, (p, (_, n)) in enumerate(zip(points, res))]
                i = next_to_measure(steps, values, available, outcome, trunk_steps, cap=cap)
                if i is not None:
                    tried.add(i)
                    await self._run_more(task, points[i], k, self.policy)
                    continue
                # Six continuations a side can make a move out of noise: the two states either side of
                # the one found get `confirm_k` each, and the search goes on from the new values.
                move = biggest_move(steps, values, outcome, trunk_steps)
                ends = [] if move is None or not move.single else \
                    [j for j, s in enumerate(steps) if s in (move.before, move.after)
                     and points[j].box is not None and res[j][1] < self.confirm_k and j not in confirmed
                     and not (j == 0 and shared is not None)]
                if not ends:
                    break
                for j in ends:
                    confirmed.add(j)
                    await self._run_more(task, points[j], self.confirm_k - res[j][1], self.policy, batch=k)
            move = biggest_move(steps, values, outcome, trunk_steps)
            targets: set[int] = set()
            if self.swap_policy is not None:
                if self.swap_at == "all" and steps and steps[0] == 0 and values[0] is not None:
                    targets.add(0)
                if not outcome and move is not None and move.clear:
                    wanted = (move.before, move.after) if self.swap_at == "all" else (move.after,)
                    targets |= {i for i, s in enumerate(steps) if s in wanted}
            if self.holdout_k and not outcome and move is not None and move.clear:
                for i, point in enumerate(points):
                    if steps[i] in (move.before, move.after) and point.box is not None:
                        need = self.holdout_k - self.store.point_result(point.id, self.holdout_policy.name)[1]
                        if need > 0:
                            await self._run_more(task, point, need, self.holdout_policy, batch=k)
            for i, point in enumerate(points):
                if values[i] is None:
                    self.store.skip_branch_point(point.id, UNNEEDED)
                if i == 0 and shared is not None and self.swap_policy is None:
                    await self._drop(point)
                if i not in targets:
                    await self._drop(point)
            for i in sorted(targets):
                point = points[i]
                if point.box is None:
                    continue
                swap = self.swap_policy.name
                if steps[i] == 0:
                    need = self.swap_k - self.store.start_result(self.experiment, task.id, swap)[1]
                    if task.id in self._swap_starts:
                        need = 0  # another trunk of the task is doing it
                    elif need > 0:
                        self._swap_starts.add(task.id)
                else:
                    need = self.swap_k - self.store.point_result(point.id, swap)[1]
                if need > 0:
                    await self._run_more(task, point, need, self.swap_policy)
                await self._drop(point)
        except BudgetExceeded:
            await self._keep(points)
            raise

    async def _run_more(self, task: Task, point: _BranchPoint, n: int, policy: Policy,
                        batch: int | None = None) -> None:
        """n more continuations, at most `batch` at a time (the room a trunk holds)."""
        while n > 0:
            if self._closing():
                raise BudgetExceeded("the subscription is close to its limit: nothing new is started")
            m = n if batch is None else min(n, batch)
            await self._continuations(task, point, m, policy)
            n -= m

    async def _continuations(self, task: Task, point: _BranchPoint, k: int, policy: Policy | None = None) -> None:
        boxes = await point.box.fork(k)
        try:
            results = await asyncio.gather(
                *(self._continue(task, point, b, policy or self.policy) for b in boxes), return_exceptions=True
            )
        finally:
            await asyncio.gather(*(b.delete() for b in boxes), return_exceptions=True)
        # One continuation erroring is recorded on its own row (outcome NULL)
        # and does not stop its siblings; a spent budget stops everything.
        for r in results:
            if isinstance(r, BudgetExceeded):
                raise r

    async def _continue(self, task: Task, point: _BranchPoint, box: Box, policy: Policy) -> None:
        run_id = self.store.start_run(experiment=self.experiment, task_id=task.id, policy=policy.name,
                                      kind="cont", branch_point=point.id, sandbox_id=box.id)
        await finish(copy.deepcopy(point.messages), box, policy, task, self.store, run_id, self.budget)
