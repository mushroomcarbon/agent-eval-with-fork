"""Command line: run an experiment, then read it back.

The platform is reached through the customer SDK, which takes its address and
tenant token from SANDBOX_OSS_URL and SANDBOX_OSS_TOKEN (or its config file);
the model through the Anthropic SDK's own credential lookup. Neither is a flag
here, so neither ends up in a shell history or a results directory.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .env import SandboxFactory
from .decisive import JUDGE_SCHEMA, JUDGE_SYSTEM, Move, biggest_move, judge_prompt, nonzero_exits
from .policy import AnthropicPolicy, BudgetExceeded as _Spent, ClaudeCodePolicy, GatesPolicy, Policy
from .runner import BudgetExceeded, Runner, schedule
from .stats import State, bootstrap_decided, decided_by_step, jeffreys_interval, variance_at_start
from .store import Store
from .tasks import Task, load_tasks


def _policy(args: argparse.Namespace, model: str | None = None) -> Policy:
    model = model or args.model
    if args.policy == "gates":
        return GatesPolicy([float(x) for x in args.gates.split(",")])
    if args.policy == "claude-code":
        return ClaudeCodePolicy(model, max_plan_use=args.max_plan_use, max_week_use=args.max_week_use)
    return AnthropicPolicy(model)


def _tasks(args: argparse.Namespace) -> list[Task]:
    only = set(args.only.split(",")) if args.only else None
    if args.format == "tbench":
        from .tbench import load_tbench_tasks

        tasks, skipped = load_tbench_tasks(Path(args.tasks), only, flavor=args.flavor, max_steps=args.max_steps)
        for task_id, why in sorted(skipped.items()):
            print(f"skipped {task_id}: {why}", file=sys.stderr)
        missing = (only or set()) - {t.id for t in tasks} - set(skipped)
    else:
        tasks = [t for t in load_tasks(Path(args.tasks)) if only is None or t.id in only]
        missing = (only or set()) - {t.id for t in tasks}
    if missing:
        raise SystemExit(f"no such task(s) in {args.tasks}: {sorted(missing)}")
    if not tasks:
        raise SystemExit("no tasks to run")
    return tasks


async def _check(args: argparse.Namespace) -> int:
    """Is each task sound on this platform: fails untouched, passes after its reference solution."""
    tasks = _tasks(args)
    runner = Runner(factory=SandboxFactory(), policy=GatesPolicy([]), store=Store(Path(args.data)),
                    experiment="check", concurrency=args.concurrency)

    async def one(task: Task) -> tuple[str, str]:
        try:
            fails_untouched, solved = await runner.check(task)
        except Exception as e:
            return task.id, f"error: {type(e).__name__}: {str(e)[:300]}"
        if not fails_untouched:
            return task.id, "BAD: the test passes with nothing done"
        if solved is None:
            return task.id, "fails untouched; no reference solution to try"
        return task.id, "ok" if solved else "BAD: the test fails after the reference solution"

    results = await asyncio.gather(*(one(t) for t in tasks))
    for task_id, verdict in results:
        print(f"{task_id:<40} {verdict}")
    good = sum(v == "ok" for _, v in results)
    print(f"{good}/{len(results)} tasks sound")
    return 0 if good == len(results) else 1


async def _run(args: argparse.Namespace) -> int:
    tasks = _tasks(args)
    store = Store(Path(args.data))
    policy = _policy(args)
    swap = _policy(args, args.swap_model) if args.swap_model else None
    if swap is not None and not args.bisect:
        raise SystemExit("--swap-model needs --bisect (it continues the states the bisection finds)")
    if args.swap_k > args.k:
        raise SystemExit("--swap-k cannot be more than -k")
    runner = Runner(factory=SandboxFactory(), policy=policy, store=store, experiment=args.experiment,
                    max_tokens=args.max_tokens, concurrency=args.concurrency, max_sandboxes=args.max_sandboxes,
                    keep_seconds=round(args.keep_hours * 3600), swap_policy=swap, swap_k=args.swap_k,
                    confirm_k=args.confirm_k)
    # --trunks and --fresh are targets, not increments: running the same command again after an
    # interruption adds only what is missing. A trunk that a budget stopped after it had finished
    # is resumed from the snapshots kept on the platform; one that died earlier is not continued
    # (its sandboxes are gone): it stays in the store and a new one is run in its place.
    done = store.done(args.experiment)
    kept: dict[str, str] = {}
    for r in store.kept(args.experiment):
        kept.setdefault(r["task_id"], r["trunk"])  # the newest trunk of each task
    resumes, jobs = [], []
    for task in tasks:
        have_trunks, have_fresh = done.get(task.id, (0, 0))
        missing = max(0, args.trunks - have_trunks)
        if missing and task.id in kept:
            resumes.append(runner.resume(task, kept[task.id], k=args.k, adaptive=args.adaptive, bisect=args.bisect,
                                         max_states=args.max_states))
            missing -= 1
        jobs += [runner.trunk(task, k=args.k, branch_at=schedule(args.schedule), adaptive=args.adaptive,
                              max_states=args.max_states, bisect=args.bisect) for _ in range(missing)]
        jobs += [runner.fresh(task) for _ in range(max(0, args.fresh - have_fresh))]
    jobs = resumes + jobs  # unfinished work first
    if any(done.get(t.id) for t in tasks) or resumes:
        print(f"{args.experiment!r} already has results; {len(jobs)} job(s) still to run"
              + (f", {len(resumes)} of them trunks to resume" if resumes else ""), file=sys.stderr)
    results = await asyncio.gather(*jobs, return_exceptions=True)
    failed = [r for r in results if isinstance(r, BaseException)]
    for r in failed:
        kind = "budget" if isinstance(r, BudgetExceeded) else "error"
        print(f"{kind}: {type(r).__name__}: {r}", file=sys.stderr)
    cost = store.cost_usd(args.experiment)
    print(f"{len(results) - len(failed)}/{len(results)} jobs finished; "
          f"{store.tokens_used(args.experiment)} tokens used in {args.experiment!r}"
          + (f" (${cost:.2f} at list price)" if cost else ""))
    plan_use = dict(getattr(policy, "plan_use", None) or {})
    plan_use.update(getattr(swap, "plan_use", None) or {})
    if plan_use:
        print("subscription usage now: " + ", ".join(f"{w} {100 * u:.0f}%" for w, u in sorted(plan_use.items())))
    left = store.kept(args.experiment)
    if left:
        print(f"{len(left)} snapshot(s) of {len({r['trunk'] for r in left})} finished trunk(s) are kept on the platform "
              f"for {args.keep_hours:g} h: the same command measures them, `forkeval drop` deletes them")
    return 1 if failed else 0


async def _drop(args: argparse.Namespace) -> int:
    """Delete the snapshots a stopped run left on the platform."""
    store = Store(Path(args.data))
    factory = SandboxFactory()
    rows = store.kept(args.experiment)
    for r in rows:
        box = await factory.attach(r["sandbox_id"])
        if box is not None:
            await box.delete()
        store.keep_point(r["id"], False)
    print(f"{len(rows)} kept snapshot(s) deleted")
    return 0


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{100 * x:5.1f}"


SKIPPED = {
    1: "settled at an earlier step",
    2: "not snapshotted, no room under the sandbox limit",
    3: "not measured, over --max-states",
    4: "not measured, the kept snapshot was gone",
    5: "not measured, the bisection did not need it",
}


def _summary(args: argparse.Namespace) -> int:
    store = Store(Path(args.data))
    rows = store.states(args.experiment)
    if not rows:
        print(f"no branch points in experiment {args.experiment!r}")
        return 1
    w = max(22, max(len(r["task_id"]) for r in rows))
    print(f"{'task':<{w}} {'trunk':<8} {'step':>4} {'pass/k':>7} {'V':>6} {'95% interval':>15} {'true V':>7} {'trunk':>6}")
    covered = known = unmeasured = 0
    run: list = []  # consecutive skipped steps of one trunk, printed as one line

    def flush() -> None:
        if run:
            first, last = run[0], run[-1]
            why = SKIPPED.get(first["skipped"], "skipped")
            steps_ = str(first["step"]) if first is last else f"{first['step']}-{last['step']}"
            print(f"{first['task_id']:<{w}} {first['trunk'][-6:]:<8} {steps_:>4}   {why}")
            run.clear()

    for r in rows:
        if r["skipped"]:
            if run and (run[-1]["trunk"], run[-1]["skipped"]) != (r["trunk"], r["skipped"]):
                flush()
            run.append(r)
            continue
        flush()
        if not r["n"] and not args.all:
            unmeasured += 1  # snapshotted, but the run stopped (or is still going) before any continuation was scored
            continue
        lo, hi = jeffreys_interval(r["passes"], r["n"])
        v = r["passes"] / r["n"] if r["n"] else float("nan")
        true = r["true_value"]
        mark = ""
        if true is not None and r["n"]:
            known += 1
            inside = lo - 1e-9 <= true <= hi + 1e-9
            covered += inside
            mark = "" if inside else "  <- outside"
        trunk = {1: "pass", 0: "fail", None: "?"}[r["trunk_outcome"]]
        unscored = f"  ({r['unscored']} unscored)" if r["unscored"] else ""
        print(f"{r['task_id']:<{w}} {r['trunk'][-6:]:<8} {r['step']:>4} {r['passes']:>3}/{r['n']:<3} {_pct(v)}% "
              f"[{_pct(lo)}, {_pct(hi)}] {'' if true is None else _pct(true) + '%':>7} {trunk:>6}{mark}{unscored}")
    flush()
    if unmeasured:
        print(f"\n{unmeasured} state(s) were snapshotted but have no scored continuation yet (the run stopped first, or is still going); --all lists them")
    if known:
        print(f"\ntrue value inside the 95% interval: {covered}/{known} states")

    # A trunk that never finished has no last step, so it cannot say when its outcome was settled.
    states = [State(r["task_id"], r["trunk"], r["step"], r["passes"], r["n"], r["trunk_steps"])
              for r in rows if r["trunk_outcome"] is not None]
    steps = sorted({s.step for s in states})
    if any(s.n >= 2 for s in states):
        between, within, mixed, n_tasks = variance_at_start(states)
        if between == between and between + within > 0:
            print(f"\noutcome variance before the first step, over {n_tasks} tasks: "
                  f"{_pct(between / (between + within)).strip()}% is which task it is, "
                  f"{_pct(within / (between + within)).strip()}% is left to the run; "
                  f"{mixed} task(s) had a mixed result at step 0")
            if mixed < 5:
                print("  (decided-by-step below divides up only the second part, and rests on those few tasks)")
        est = decided_by_step(states, steps)
        ci = bootstrap_decided(states, steps, draws=args.draws)
        print("\nshare of the outcome decided by step (bootstrap over tasks, trunks, continuations):")
        for t in steps:
            print(f"  step {t:>2}: {_pct(est[t])}%  [{_pct(ci[t][0])}, {_pct(ci[t][1])}]")

    fresh = {r["task_id"]: r for r in store.fresh(args.experiment)}
    step0: dict[str, list[int]] = {}
    for r in rows:
        if r["step"] == 0:
            agg = step0.setdefault(r["task_id"], [0, 0])
            agg[0] += r["passes"]
            agg[1] += r["n"]
    if fresh:
        print("\nfork against fresh (pass rate from step-0 forks, and from ordinary runs):")
        for task, (x, n) in sorted(step0.items()):
            if task not in fresh or not n or not fresh[task]["n"]:
                continue
            f = fresh[task]
            flo, fhi = jeffreys_interval(f["passes"], f["n"])
            lo, hi = jeffreys_interval(x, n)
            overlap = "overlap" if lo <= fhi and flo <= hi else "DO NOT OVERLAP"
            print(f"  {task:<22} forks {x}/{n} [{_pct(lo)}, {_pct(hi)}]   fresh {f['passes']}/{f['n']} "
                  f"[{_pct(flo)}, {_pct(fhi)}]   intervals {overlap}")
    forks = [r["fork_ms"] for r in rows if r["fork_ms"] is not None]
    if forks:
        forks.sort()
        print(f"\nbranch points: {len(forks)}; fork call median {forks[len(forks) // 2]} ms, max {forks[-1]} ms")
    return 0


def _curve(store: Store, trunk) -> tuple[list[int], list[float | None], list[tuple[int, int]], Move | None]:
    """A finished trunk's measured states (its own policy's continuations) and its decisive move."""
    pts = [r for r in store.points(trunk["id"])]
    res = [store.point_result(r["id"], trunk["policy"]) for r in pts]
    steps = [r["step"] for r in pts]
    values = [x / n if n else None for x, n in res]
    move = biggest_move(steps, values, bool(trunk["outcome"]), trunk["steps"])
    return steps, values, res, move


def _judgeable(store: Store, experiments: list[str]):
    """Failed trunks with a clear decisive step, and that step."""
    for e in experiments:
        for t in store.trunks(e):
            if t["outcome"]:
                continue
            _, _, _, move = _curve(store, t)
            if move is not None and move.clear:
                yield e, t, move


async def _judge(args: argparse.Namespace) -> int:
    """Ask judge models which step decided each failed run that has a clear decisive step."""
    store = Store(Path(args.data))
    judges = [ClaudeCodePolicy(m, max_plan_use=args.max_plan_use, max_week_use=args.max_week_use)
              for m in args.model]
    slots = asyncio.Semaphore(args.concurrency)
    todo = list(_judgeable(store, args.experiment))

    async def one(t, judge: ClaudeCodePolicy) -> None:
        if judge.name in store.judgments(t["id"]):
            return
        prompt, shown = judge_prompt(store.load_transcript(t["id"]))
        async with slots:
            answer, _ = await judge.ask(prompt, system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA)
        step = answer.get("step") if isinstance(answer, dict) else None
        reason = str(answer.get("reason", "")) if isinstance(answer, dict) else ""
        store.add_judgment(t["id"], judge.name, int(step) if isinstance(step, (int, float)) else None, reason, shown)

    results = await asyncio.gather(*(one(t, j) for _, t, _ in todo for j in judges), return_exceptions=True)
    failed = [r for r in results if isinstance(r, BaseException)]
    for r in failed[:5]:
        print(f"{'budget' if isinstance(r, _Spent) else 'error'}: {type(r).__name__}: {str(r)[:300]}", file=sys.stderr)
    print(f"{len(todo)} failed run(s) with a clear decisive step; {len(results) - len(failed)}/{len(results)} "
          f"judgments made or already there")
    return 1 if failed else 0


def _frac(x: int, n: int) -> str:
    if not n:
        return "  -"
    lo, hi = jeffreys_interval(x, n)
    return f"{x}/{n} = {100 * x / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def _decisive(args: argparse.Namespace) -> int:
    """Decisive steps found by forking, what judges said, and the second policy on the same states."""
    store = Store(Path(args.data))
    trunks = [(e, t) for e in args.experiment for t in store.trunks(e)]
    if not trunks:
        print("no finished trunks")
        return 1
    w = max(22, max(len(t["task_id"]) for _, t in trunks))
    print(f"{'task':<{w}} {'trunk':<7} {'end':<4} {'decisive':>8} {'change':>6}  states measured (step: pass/k)")
    failed = clear = final = 0
    hits: dict[str, list[int]] = {}
    base = {"last step": [0, 0], "first failed command": [0, 0]}
    chance = 0.0
    swap_rows = []
    for e, t in trunks:
        steps, values, res, move = _curve(store, t)
        shown = " ".join(f"{s}:{x}/{n}" for s, (x, n) in zip(steps, res) if n)
        end = "pass" if t["outcome"] else "fail"
        dec = "-"
        if move is not None:
            a = move.actions
            dec = str(a[0]) if move.single else f"{a[0]}-{a[-1]}"
        mark = "" if move is None or not move.clear else ("*" if not t["outcome"] else "")
        print(f"{t['task_id']:<{w}} {t['id'][-6:]:<7} {end:<4} {dec:>8}{mark:1} "
              f"{'' if move is None else f'{move.change:+.2f}':>6}  {shown}")
        if t["outcome"]:
            continue
        failed += 1
        if move is None or not move.clear:
            continue
        clear += 1
        truth = move.after
        if truth == t["steps"] + 1:
            final += 1
        messages = store.load_transcript(t["id"])
        _, n_shown = judge_prompt(messages)
        chance += 1 / max(1, n_shown)
        bad = nonzero_exits(messages)
        guesses = {"last step": n_shown, "first failed command": bad[0] if bad else n_shown}
        for name, g in guesses.items():
            base[name][0] += g == truth
            base[name][1] += 1
        for judge, j in store.judgments(t["id"]).items():
            h = hits.setdefault(judge, [0, 0, 0])
            h[0] += j["step"] == truth
            h[1] += j["step"] is not None and abs(j["step"] - truth) <= 1
            h[2] += 1
            print(f"{'':<{w}}   {judge}: step {j['step']}  ({j['reason'][:110]})")
    print(f"\n{failed} failed run(s); {clear} with a clear decisive step (marked *: one action, V moves "
          f"at least 0.5 across it), {final} of them the turn that ended the run")
    if clear:
        print("\nwhich step decided the failure, against the step forking found:")
        for judge, (exact, near, n) in sorted(hits.items()):
            print(f"  {judge:<28} exact {_frac(exact, n)}   within one step {_frac(near, n)}")
        for name, (x, n) in base.items():
            print(f"  {'baseline: ' + name:<28} exact {_frac(x, n)}")
        print(f"  {'baseline: a step at random':<28} exact {100 * chance / clear:.0f}% expected")

    # The second policy on the same states.
    rows = [r for e in args.experiment for r in store.by_policy(e)]
    policies = sorted({r["policy"] for r in rows})
    if len(policies) < 2:
        return 0
    main_of = {t["id"]: t["policy"] for _, t in trunks}
    moves = {}
    for _, t in trunks:
        if not t["outcome"]:
            _, _, _, m = _curve(store, t)
            if m is not None and m.clear:
                moves[t["id"]] = m
    pooled: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        where = None
        if r["step"] == 0:
            where = "step 0 (the task as given)"
        elif r["trunk"] in moves and r["step"] == moves[r["trunk"]].before:
            where = "just before the decisive step"
        elif r["trunk"] in moves and r["step"] == moves[r["trunk"]].after:
            where = "just after the decisive step"
        if where is None:
            continue
        who = "same model" if r["policy"] == main_of.get(r["trunk"]) else r["policy"]
        agg = pooled.setdefault((where, who), [0, 0])
        agg[0] += r["passes"]
        agg[1] += r["n"]
    print("\nthe same states continued by each model (pooled over states; states of one task are not independent):")
    for where in ("step 0 (the task as given)", "just before the decisive step", "just after the decisive step"):
        cells = [(who, x, n) for (wh, who), (x, n) in sorted(pooled.items()) if wh == where]
        if cells:
            print(f"  {where}")
            for who, x, n in cells:
                print(f"    {who:<28} {_frac(x, n)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forkeval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def task_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tasks", required=True, help="directory of tasks")
        p.add_argument("--format", choices=["toml", "tbench"], default="toml",
                       help="toml: this repository's task files; tbench: task directories in the Terminal-Bench layout")
        p.add_argument("--only", help="comma-separated task ids")
        p.add_argument("--flavor", default="medium", help="sandbox size for --format tbench tasks")
        p.add_argument("--max-steps", type=int, default=30, help="step limit for --format tbench tasks")
        p.add_argument("--data", default="data")
        p.add_argument("--concurrency", type=int, default=2, help="tasks in flight at once")

    run = sub.add_parser("run", help="run trunks (with branch points and continuations) and fresh runs")
    task_args(run)
    run.add_argument("--experiment", required=True)
    run.add_argument("--policy", choices=["anthropic", "claude-code", "gates"], default="anthropic",
                     help="anthropic: the Messages API (needs ANTHROPIC_API_KEY); claude-code: the same models "
                          "through `claude -p` on a Claude subscription; gates: a synthetic agent of known value")
    run.add_argument("--model", default="claude-haiku-5-5")
    run.add_argument("--gates", default="0.8,0.7,0.9", help="per-gate success probabilities for --policy gates")
    run.add_argument("--trunks", type=int, default=1, help="complete trunks each task should end up with")
    run.add_argument("--fresh", type=int, default=0, help="ordinary runs each task should end up with, for the fork-against-fresh check")
    run.add_argument("-k", type=int, default=6, help="continuations per branch point")
    run.add_argument("--schedule", default="every", help='"every", "grid", or steps like "0,2,5"')
    run.add_argument("--max-sandboxes", type=int, default=45,
                     help="sandboxes this run may hold at once, suspended branch points included; "
                          "keep it under the tenant's limit")
    run.add_argument("--adaptive", action="store_true",
                     help="run continuations in step order and stop after the first state whose continuations "
                          "all end the way the trunk did; later states are recorded as skipped")
    run.add_argument("--max-states", type=int,
                     help="measure at most this many states of a trunk, evenly spaced from its first snapshot to "
                          "its last; the others are recorded as skipped")
    run.add_argument("--bisect", action="store_true",
                     help="measure the states that find each trunk's decisive step: the first and the last, then "
                          "halving the biggest move in value until it is one action wide (--max-states caps them)")
    run.add_argument("--swap-model",
                     help="with --bisect: also continue step 0 (once per task) and, for a failed trunk, the states "
                          "either side of its decisive step with this model")
    run.add_argument("--swap-k", type=int, default=6, help="continuations per state for --swap-model")
    run.add_argument("--confirm-k", type=int, default=0,
                     help="with --bisect: the two states either side of the decisive step get this many "
                          "continuations each (in batches of at most -k), and the search goes on from the new values")
    
    run.add_argument("--keep-hours", type=float, default=24,
                     help="when a budget stops a trunk that has finished, keep its unmeasured snapshots on the "
                          "platform this long so the same command can measure them later; 0 deletes them")
    run.add_argument("--max-plan-use", type=float, default=0.6,
                     help="for --policy claude-code: stop once the subscription's 5-hour usage window is this full, 0 to 1")
    run.add_argument("--max-week-use", type=float, default=0.6,
                     help="the same for the subscription's weekly window")
    run.add_argument("--max-tokens", type=int, help="stop sending model requests once the experiment has used this many tokens")

    chk = sub.add_parser("check", help="check the tasks themselves: each must fail untouched and pass after "
                                       "its reference solution (no model involved)")
    task_args(chk)

    drp = sub.add_parser("drop", help="delete the snapshots a stopped run left on the platform")
    drp.add_argument("--experiment", required=True)
    drp.add_argument("--data", default="data")

    summ = sub.add_parser("summary", help="value per branch point, decided-by-step, fork against fresh")
    summ.add_argument("--experiment", required=True)
    summ.add_argument("--data", default="data")
    summ.add_argument("--draws", type=int, default=2000)
    summ.add_argument("--all", action="store_true", help="also list states that have no scored continuation yet")

    jdg = sub.add_parser("judge", help="ask judge models which step decided each failed run (needs `claude`)")
    jdg.add_argument("--experiment", action="append", required=True)
    jdg.add_argument("--model", action="append", required=True, help="judge model; may be given more than once")
    jdg.add_argument("--data", default="data")
    jdg.add_argument("--concurrency", type=int, default=4)
    jdg.add_argument("--max-plan-use", type=float, default=0.6)
    jdg.add_argument("--max-week-use", type=float, default=0.6)

    dec = sub.add_parser("decisive", help="decisive steps from forks, the judges' answers, and the swap results")
    dec.add_argument("--experiment", action="append", required=True)
    dec.add_argument("--data", default="data")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_run(args))
    if args.cmd == "check":
        return asyncio.run(_check(args))
    if args.cmd == "drop":
        return asyncio.run(_drop(args))
    if args.cmd == "judge":
        return asyncio.run(_judge(args))
    if args.cmd == "decisive":
        return _decisive(args)
    return _summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
