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
