import json

from fakes import FakeBox, FakeFactory
from forkeval import cli
from forkeval.decisive import Move, biggest_move, judge_prompt, next_to_measure, nonzero_exits
from forkeval.policy import GatesPolicy
from forkeval.runner import UNNEEDED, Runner, schedule
from forkeval.store import Store
from forkeval.tasks import Task

TASK = Task(id="gates", instruction="Open every gate correctly.", test="python3 /work/gate.py check 3", max_steps=10)


def test_biggest_move_counts_the_end_of_the_run():
    # A failed run that looked won at its last state: the turn that ended it decided it.
    m = biggest_move([0, 1, 2], [0.8, 0.8, 1.0], False, 2)
    assert m == Move(2, 3, 1.0) and m.single and m.clear and list(m.actions) == [3]
    # A passed run: the biggest rise.
    m = biggest_move([0, 4, 8], [0.2, 0.3, 1.0], True, 9)
    assert (m.before, m.after) == (4, 8) and not m.single


def test_bisection_halves_the_biggest_move_until_it_is_one_action():
    steps = list(range(9))
    vals = [None] * 9
    avail = [True] * 9
    assert next_to_measure(steps, vals, avail, False, 8) == 0
    vals[0], avail[0] = 0.8, False
    assert next_to_measure(steps, vals, avail, False, 8) == 8
    vals[8], avail[8] = 0.0, False
    assert next_to_measure(steps, vals, avail, False, 8) == 4
    vals[4], avail[4] = 0.7, False
    assert next_to_measure(steps, vals, avail, False, 8) == 6   # the drop is between 4 and 8
    vals[6], avail[6] = 0.0, False
    assert next_to_measure(steps, vals, avail, False, 8) == 5
    vals[5], avail[5] = 0.0, False
    assert next_to_measure(steps, vals, avail, False, 8) is None  # 4 -> 5 is one action
    assert biggest_move(steps, vals, False, 8) == Move(4, 5, 0.7)


def test_bisection_stops_on_a_run_that_was_never_open_and_at_the_cap():
    assert next_to_measure([0, 1, 2, 3], [0.0, None, None, 0.0], [False, True, True, False], False, 3) is None
    vals = [0.8, None, None, None, 0.0]
    assert next_to_measure(list(range(5)), vals, [False, True, True, True, False], False, 4, cap=2) is None


async def test_bisect_finds_the_decisive_step_and_the_swap_policy_continues_around_it(tmp_path):
    FakeBox.live.clear()
    store = Store(tmp_path)
    # Seed 2: the trunk opens gate 1 wrongly, so V drops from 0.9 to 0 at action 2.
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 0.9, 1.0], seed=2), store=store,
                    experiment="x", swap_policy=GatesPolicy([1.0, 1.0, 1.0]), swap_k=3)
    await runner.trunk(TASK, k=6, branch_at=schedule("every"), bisect=True)
    rows = {r["step"]: r for r in store.states("x")}
    assert rows[0]["trunk_outcome"] == 0
    measured = sorted(s for s, r in rows.items() if r["n"])
    assert measured == [0, 1, 2, 3]
    assert rows[2]["passes"] == 0 and rows[1]["passes"] >= 4
    (trunk,) = store.trunks("x")
    _, _, _, move = cli._curve(store, trunk)
    assert (move.before, move.after) == (1, 2) and move.clear
    swapped = {(r["step"], r["policy"]): r["n"] for r in store.by_policy("x") if r["policy"].endswith("1,1,1")}
    assert swapped == {(0, "gates:1,1,1"): 3, (1, "gates:1,1,1"): 3, (2, "gates:1,1,1"): 3}
    assert FakeBox.live == set()


async def test_bisect_skips_states_it_does_not_need(tmp_path):
    FakeBox.live.clear()
    store = Store(tmp_path)
    task = Task(id="g6", instruction="x", test="python3 /work/gate.py check 6", max_steps=10)
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([0.0] * 6, seed=0), store=store, experiment="x")
    await runner.trunk(task, k=3, branch_at=schedule("every"), bisect=True)
    rows = store.states("x")
    assert [r["step"] for r in rows if r["n"]] == [0, 6]          # 0 from the start: nothing to find
    assert all(r["skipped"] == UNNEEDED for r in rows if not r["n"])
    assert FakeBox.live == set()


def _msgs():
    return [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"},
                                          {"type": "tool_use", "id": "a", "name": "shell", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "exit code: 0\nx"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "shell", "input": {"command": "make"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "exit code: 2\nerr"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "all done"}]},
    ]


def test_judge_sees_numbered_steps_with_the_ending_turn_last():
    text, n = judge_prompt(_msgs())
    assert n == 3
    assert "## Step 1: shell" in text and "## Step 2: shell" in text and "## Step 3: ends the run" in text
    assert "from 1 to 3" in text
    assert nonzero_exits(_msgs()) == [2]


async def test_decisive_report_scores_judges_against_the_forks(tmp_path, capsys):
    FakeBox.live.clear()
    store = Store(tmp_path)
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 0.9, 1.0], seed=2), store=store,
                    experiment="x", swap_policy=GatesPolicy([1.0, 1.0, 1.0]), swap_k=3)
    await runner.trunk(TASK, k=6, branch_at=schedule("every"), bisect=True)
    (trunk,) = store.trunks("x")
    store.add_judgment(trunk["id"], "judge-a", 2, "opened gate 1 wrongly", 4)
    store.add_judgment(trunk["id"], "judge-b", 4, "ended too early", 4)
    assert [t["id"] for _, t, _ in cli._judgeable(store, ["x"])] == [trunk["id"]]
    assert cli.main(["decisive", "--experiment", "x", "--data", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "judge-a" in out and "exact 1/1" in out and "exact 0/1" in out
    assert "just after the decisive step" in out and "gates:1,1,1" in out


async def test_a_bisection_stopped_by_a_budget_is_resumed(tmp_path):
    from test_runner import StopsAfter
    import pytest
    from forkeval.runner import BudgetExceeded

    FakeBox.live.clear()
    store = Store(tmp_path)
    # The trunk takes 4 calls, step 0's six continuations 24; at 30 the last state has 2 of its 6.
    runner = Runner(factory=FakeFactory(), policy=StopsAfter([1.0, 0.9, 1.0], seed=2, calls=30), store=store,
                    experiment="x", keep_seconds=3600)
    with pytest.raises(BudgetExceeded):
        await runner.trunk(TASK, k=6, branch_at=schedule("every"), bisect=True)
    assert [r["step"] for r in store.kept("x")] == [0, 1, 2, 3]
    assert store.done("x") == {}
    again = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 0.9, 1.0], seed=5), store=store,
                   experiment="x", keep_seconds=3600)
    await again.resume(TASK, store.kept("x")[0]["trunk"], k=6, bisect=True)
    rows = {r["step"]: r for r in store.states("x")}
    assert [rows[s]["n"] for s in range(4)] == [6, 6, 6, 6]
    assert store.kept("x") == [] and FakeBox.live == set()
    assert store.done("x") == {"gates": (1, 0)}


async def test_confirm_k_tops_up_both_sides_of_the_decisive_step(tmp_path):
    FakeBox.live.clear()
    store = Store(tmp_path)
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([1.0, 0.9, 1.0], seed=2), store=store,
                    experiment="x", confirm_k=12)
    await runner.trunk(TASK, k=6, branch_at=schedule("every"), bisect=True)
    n = {r["step"]: r["n"] for r in store.states("x")}
    assert n == {0: 6, 1: 12, 2: 12, 3: 6}
    assert FakeBox.live == set()
