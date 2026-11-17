import json

import pytest

from fakes import FakeBox, FakeFactory
from forkeval.policy import GatesPolicy, Turn, Usage
from forkeval.runner import BudgetExceeded, Runner, schedule
from forkeval.store import Store
from forkeval.tasks import Task

TASK = Task(id="gates", instruction="Open every gate correctly.", test="python3 /work/gate.py check 3", max_steps=10)


def make(tmp_path, policy, **kw):
    FakeBox.live.clear()
    store = Store(tmp_path)
    return Runner(factory=FakeFactory(), policy=policy, store=store, experiment="x", **kw), store


async def test_trunk_branches_at_every_step_and_cleans_up(tmp_path):
    runner, store = make(tmp_path, GatesPolicy([1.0, 1.0, 1.0], seed=0))
    await runner.trunk(TASK, k=4, branch_at=schedule("every"))
    rows = store.states("x")
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert all(r["n"] == 4 and r["passes"] == 4 for r in rows)      # p = 1: every continuation passes
    assert all(r["trunk_outcome"] == 1 and r["trunk_steps"] == 3 for r in rows)
    assert all(r["true_value"] == 1.0 for r in rows)
    assert FakeBox.live == set()                                    # trunk, branch points, continuations all deleted


async def test_continuations_start_from_the_branch_points_state(tmp_path):
    # Gate 0 always fails, so every state after step 1 has value 0 and step 0 has value 0 too.
    runner, store = make(tmp_path, GatesPolicy([0.0, 1.0, 1.0], seed=0))
    await runner.trunk(TASK, k=3, branch_at=schedule("0,1"))
    rows = {r["step"]: r for r in store.states("x")}
    assert rows[0]["passes"] == 0 and rows[1]["passes"] == 0
    assert rows[1]["true_value"] == 0.0
    # A continuation of step 1 must not redo gate 0: its transcript has the trunk's first step, then 2 more.
    cont = store.db.execute(
        "SELECT id, steps FROM runs WHERE kind = 'cont' AND branch_point = ?", (rows[1]["branch_point"],)
    ).fetchone()
    assert cont["steps"] == 3
    lines = [json.loads(x) for x in (tmp_path / "transcripts" / f"{cont['id']}.jsonl").read_text().splitlines()]
    cmds = [b["input"]["command"] for m in lines if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    assert [c.split()[-2:] for c in cmds] == [["0", "bad"], ["1", "ok"], ["2", "ok"]]


async def test_estimates_track_the_known_value(tmp_path):
    p = [0.8, 0.7, 0.9]
    runner, store = make(tmp_path, GatesPolicy(p, seed=7))
    for _ in range(60):
        await runner.trunk(TASK, k=6, branch_at=schedule("0"))
    rows = store.states("x")
    passes, n = sum(r["passes"] for r in rows), sum(r["n"] for r in rows)
    assert n == 360
    assert passes / n == pytest.approx(0.8 * 0.7 * 0.9, abs=0.08)


async def test_fresh_runs_are_recorded_separately(tmp_path):
    runner, store = make(tmp_path, GatesPolicy([1.0, 1.0, 1.0], seed=0))
    await runner.fresh(TASK)
    (row,) = store.fresh("x")
    assert (row["n"], row["passes"]) == (1, 1)
    assert store.states("x") == []
    assert FakeBox.live == set()


async def test_step_limit_ends_a_run_and_it_is_still_scored(tmp_path):
    task = Task(id="gates", instruction="x", test="python3 /work/gate.py check 3", max_steps=2)
    runner, store = make(tmp_path, GatesPolicy([1.0, 1.0, 1.0], seed=0))
    await runner.fresh(task)
    row = store.db.execute("SELECT outcome, stop, steps FROM runs").fetchone()
    assert (row["outcome"], row["stop"], row["steps"]) == (0, "step_limit", 2)


class Costly(GatesPolicy):
    async def act(self, messages):
        t = await super().act(messages)
        return Turn(t.content, t.tool_calls, t.stop, Usage(input_tokens=100, output_tokens=10))


async def test_budget_stops_the_experiment_and_leaves_nothing_behind(tmp_path):
    runner, store = make(tmp_path, Costly([1.0, 1.0, 1.0], seed=0), max_tokens=500)
    with pytest.raises(BudgetExceeded):
        await runner.trunk(TASK, k=4, branch_at=schedule("every"))
    assert 500 <= store.tokens_used("x") < 500 + 5 * 110        # at most one request per in-flight run past the cap
    assert FakeBox.live == set()
    assert store.db.execute("SELECT COUNT(*) FROM runs WHERE stop = 'budget'").fetchone()[0] >= 1


def test_schedules():
    assert [t for t in range(10) if schedule("grid")(t)] == [0, 1, 2, 3, 6, 9]
    assert [t for t in range(10) if schedule("0, 4")(t)] == [0, 4]
    assert all(schedule("every")(t) for t in range(5))


@pytest.mark.asyncio
async def test_done_counts_only_complete_trunks(tmp_path):
    store = Store(tmp_path)
    def trunk(outcome, conts):
        t = store.start_run(experiment="e", task_id="a", policy="p", kind="trunk")
        store.end_run(t, outcome=outcome, steps=1, stop="end_turn")
        bp = store.add_branch_point(trunk=t, step=0, sandbox_id="s", fork_ms=1, suspend_ms=1, true_value=None)
        for o in conts:
            c = store.start_run(experiment="e", task_id="a", policy="p", kind="cont", branch_point=bp)
            store.end_run(c, outcome=o, steps=1, stop="end_turn")
    trunk(True, [True, False])      # complete
    trunk(None, [True, True])       # the trunk itself was never scored
    trunk(False, [True, None])      # a branch point with one scored continuation
    f = store.start_run(experiment="e", task_id="a", policy="p", kind="fresh")
    store.end_run(f, outcome=True, steps=1, stop="end_turn")
    store.start_run(experiment="e", task_id="a", policy="p", kind="fresh")   # still running
    assert store.done("e") == {"a": (1, 1)}
    assert store.done("other") == {}


async def test_adaptive_spends_nothing_past_a_settled_state(tmp_path):
    # Never in doubt: only step 0 gets continuations, and the trunk still counts as complete.
    runner, store = make(tmp_path, GatesPolicy([1.0, 1.0, 1.0], seed=0))
    await runner.trunk(TASK, k=4, branch_at=schedule("every"), adaptive=True)
    rows = store.states("x")
    assert [(r["step"], r["n"], r["skipped"]) for r in rows] == [(0, 4, 0), (1, 0, 1), (2, 0, 1), (3, 0, 1)]
    assert store.done("x") == {"gates": (1, 0)}
    assert FakeBox.live == set()


async def test_adaptive_follows_a_run_until_it_is_settled(tmp_path):
    # The first gate is a coin flip and the rest are sure: open at step 0, settled from step 1.
    runner, store = make(tmp_path, GatesPolicy([0.5, 1.0, 1.0], seed=3))
    await runner.trunk(TASK, k=8, branch_at=schedule("every"), adaptive=True)
    rows = store.states("x")
    assert 0 < rows[0]["passes"] < 8
    assert rows[1]["n"] == 8 and rows[1]["passes"] == 8 * rows[1]["trunk_outcome"]
    assert [r["skipped"] for r in rows] == [0, 0, 1, 1]
    assert FakeBox.live == set()


async def test_jobs_still_waiting_end_at_once_when_the_budget_is_spent(tmp_path):
    runner, store = make(tmp_path, Costly([1.0, 1.0, 1.0], seed=0), max_tokens=500)
    with pytest.raises(BudgetExceeded):
        await runner.trunk(TASK, k=4, branch_at=schedule("every"))
    created = len(FakeFactory.created)
    with pytest.raises(BudgetExceeded):
        await runner.fresh(TASK)
    assert len(FakeFactory.created) == created      # no sandbox was started to find that out


async def test_sandbox_limit_is_kept_and_a_step_without_room_is_recorded(tmp_path, monkeypatch):
    import forkeval.runner as runner_mod

    monkeypatch.setattr(runner_mod, "SNAPSHOT_WAIT", 0.01)
    peak = 0

    class Watched(set):
        def add(self, x):
            nonlocal peak
            super().add(x)
            peak = max(peak, len(self))

    monkeypatch.setattr(FakeBox, "live", Watched())
    store = Store(tmp_path)
    # Room for the trunk, its 2 continuations and 2 branch points: steps 0 and 1 fit, 2 and 3 do not.
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 1.0, 1.0], seed=0), store=store,
                    experiment="x", max_sandboxes=5)
    await runner.trunk(TASK, k=2, branch_at=schedule("every"))
    rows = store.states("x")
    assert [(r["step"], r["n"], r["skipped"]) for r in rows] == [(0, 2, 0), (1, 2, 0), (2, 0, 2), (3, 0, 2)]
    assert peak <= 5 and len(FakeBox.live) == 0
    assert store.done("x") == {"gates": (1, 0)}
    assert runner._room._free == 5


def test_summary_merges_skipped_steps_and_hides_states_with_no_continuations(tmp_path, capsys):
    from forkeval.cli import main

    store = Store(tmp_path)
    trunk = store.start_run(experiment="x", task_id="a-task-with-quite-a-long-name", policy="p", kind="trunk")
    store.end_run(trunk, outcome=True, steps=4, stop="end_turn")
    for step in range(5):
        bp = store.add_branch_point(trunk=trunk, step=step, sandbox_id="s", fork_ms=1, suspend_ms=1, true_value=None,
                                    skipped=0 if step == 0 else 1)
        if step == 0:
            for _ in range(2):
                c = store.start_run(experiment="x", task_id="a-task-with-quite-a-long-name", policy="p", kind="cont",
                                    branch_point=bp)
                store.end_run(c, outcome=True, steps=1, stop="end_turn")
    cut = store.start_run(experiment="x", task_id="cut", policy="p", kind="trunk")   # stopped by the budget
    store.add_branch_point(trunk=cut, step=0, sandbox_id="s", fork_ms=1, suspend_ms=1, true_value=None)

    assert main(["summary", "--experiment", "x", "--data", str(tmp_path), "--draws", "10"]) == 0
    out = capsys.readouterr().out
    table = out.split("\n\n")[0].splitlines()
    assert len(table) == 3                                   # header, the measured state, one line for steps 1-4
    assert " 1-4 " in table[2] and "settled at an earlier step" in table[2]
    assert "1 state(s) were snapshotted but have no scored continuation yet" in out
    assert not any(line.startswith("cut ") for line in out.splitlines())

    main(["summary", "--experiment", "x", "--data", str(tmp_path), "--draws", "10", "--all"])
    assert any(line.startswith("cut ") for line in capsys.readouterr().out.splitlines())


def test_spread_keeps_the_ends():
    from forkeval.runner import spread

    assert spread(4, 6) == {0, 1, 2, 3}
    assert spread(21, 6) == {0, 4, 8, 12, 16, 20}
    assert spread(10, 2) == {0, 9} and spread(10, 1) == {0}


async def test_max_states_measures_an_even_spread(tmp_path):
    runner, store = make(tmp_path, GatesPolicy([1.0, 1.0, 1.0], seed=0))
    await runner.trunk(TASK, k=2, branch_at=schedule("every"), max_states=2)
    assert [(r["step"], r["n"], r["skipped"]) for r in store.states("x")] == [(0, 2, 0), (1, 0, 3), (2, 0, 3), (3, 2, 0)]
    assert store.done("x") == {"gates": (1, 0)} and FakeBox.live == set()


class StopsAfter(GatesPolicy):
    """Ends the experiment with a spent budget once it has acted `calls` times."""

    def __init__(self, *a, calls, **kw):
        super().__init__(*a, **kw)
        self.calls = calls

    async def act(self, messages):
        self.calls -= 1
        if self.calls < 0:
            raise BudgetExceeded("spent")
        return await super().act(messages)


async def test_a_stopped_trunk_keeps_its_snapshots_and_is_resumed(tmp_path):
    # The trunk takes 4 calls (3 gates and the last word); the two continuations of step 0 take 8.
    # Stopping after 14 leaves step 1 part-measured and steps 2 and 3 untouched.
    runner, store = make(tmp_path, StopsAfter([1.0, 1.0, 1.0], seed=0, calls=14), keep_seconds=3600)
    with pytest.raises(BudgetExceeded):
        await runner.trunk(TASK, k=2, branch_at=schedule("every"))
    kept = store.kept("x")
    assert [r["step"] for r in kept] == [1, 2, 3]
    assert FakeBox.live == {r["sandbox_id"] for r in kept}               # only those are left, each given more time
    assert all(FakeBox.by_id[r["sandbox_id"]].extended >= 3600 for r in kept)
    assert store.done("x") == {}                                         # not complete while snapshots wait
    before = {r["step"]: r["n"] for r in store.states("x")}
    assert before[0] == 2 and before[1] < 2 and before[2] == before[3] == 0

    FakeBox.live.discard(kept[2]["sandbox_id"])                          # one of them expires in the meantime
    again = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 1.0, 1.0], seed=1), store=store, experiment="x",
                   keep_seconds=3600)
    await again.resume(TASK, kept[0]["trunk"], k=2)
    rows = {r["step"]: r for r in store.states("x")}
    assert [rows[t]["n"] for t in (0, 1, 2)] == [2, 2, 2] and all(rows[t]["passes"] == 2 for t in (0, 1, 2))
    assert rows[3]["skipped"] == 4                                       # recorded as lost, not silently missing
    assert store.kept("x") == [] and FakeBox.live == set()
    assert store.done("x") == {"gates": (1, 0)}
    # A continuation of the resumed step 2 starts from that state: two gates already open, one to go.
    cont = store.db.execute("SELECT steps FROM runs WHERE kind = 'cont' AND branch_point = ?",
                            (rows[2]["branch_point"],)).fetchone()
    assert cont["steps"] == 3


async def test_a_resumed_adaptive_trunk_stops_where_it_is_settled(tmp_path):
    runner, store = make(tmp_path, StopsAfter([1.0, 1.0, 1.0], seed=0, calls=4), keep_seconds=3600)
    with pytest.raises(BudgetExceeded):
        await runner.trunk(TASK, k=2, branch_at=schedule("every"), adaptive=True)
    assert [r["step"] for r in store.kept("x")] == [0, 1, 2, 3]
    again = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 1.0, 1.0], seed=1), store=store, experiment="x")
    await again.resume(TASK, store.kept("x")[0]["trunk"], k=2, adaptive=True)
    assert [(r["step"], r["n"], r["skipped"]) for r in store.states("x")] == [(0, 2, 0), (1, 0, 1), (2, 0, 1), (3, 0, 1)]
    assert store.kept("x") == [] and FakeBox.live == set()


class Closing(GatesPolicy):
    closing = False


async def test_nothing_new_starts_once_the_plan_is_close_to_its_limit(tmp_path):
    policy = Closing([1.0, 1.0, 1.0], seed=0)
    runner, store = make(tmp_path, policy, keep_seconds=3600)
    real = runner._continuations

    async def then_close(task, point, k):
        await real(task, point, k)
        policy.closing = True                    # the limit comes near while step 0 is being measured

    runner._continuations = then_close
    with pytest.raises(BudgetExceeded):
        await runner.trunk(TASK, k=2, branch_at=schedule("every"))
    rows = store.states("x")
    assert [(r["step"], r["n"], r["unscored"]) for r in rows] == [(0, 2, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    assert [r["step"] for r in store.kept("x")] == [1, 2, 3]            # no run was cut off half-way
    with pytest.raises(BudgetExceeded):
        await runner.fresh(TASK)
    FakeBox.live.clear()                         # the kept snapshots; later tests start from none
