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
from .policy import AnthropicPolicy, ClaudeCodePolicy, GatesPolicy, Policy
from .runner import BudgetExceeded, Runner, schedule
from .stats import State, bootstrap_decided, decided_by_step, jeffreys_interval
from .store import Store
from .tasks import Task, load_tasks


def _policy(args: argparse.Namespace) -> Policy:
    if args.policy == "gates":
        return GatesPolicy([float(x) for x in args.gates.split(",")])
    if args.policy == "claude-code":
        return ClaudeCodePolicy(args.model)
    return AnthropicPolicy(args.model)


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
    runner = Runner(factory=SandboxFactory(), policy=_policy(args), store=store, experiment=args.experiment,
                    max_tokens=args.max_tokens, concurrency=args.concurrency)
    # --trunks and --fresh are targets, not increments: running the same command again after an
    # interruption adds only what is missing. A trunk that died part-way is not continued (its
    # sandboxes are gone); it stays in the store and a new one is run in its place.
    done = store.done(args.experiment)
    jobs = []
    for task in tasks:
        have_trunks, have_fresh = done.get(task.id, (0, 0))
        jobs += [runner.trunk(task, k=args.k, branch_at=schedule(args.schedule))
                 for _ in range(max(0, args.trunks - have_trunks))]
        jobs += [runner.fresh(task) for _ in range(max(0, args.fresh - have_fresh))]
    if any(done.get(t.id) for t in tasks):
        print(f"{args.experiment!r} already has results; {len(jobs)} job(s) still to run", file=sys.stderr)
    results = await asyncio.gather(*jobs, return_exceptions=True)
    failed = [r for r in results if isinstance(r, BaseException)]
    for r in failed:
        kind = "budget" if isinstance(r, BudgetExceeded) else "error"
        print(f"{kind}: {type(r).__name__}: {r}", file=sys.stderr)
    print(f"{len(results) - len(failed)}/{len(results)} jobs finished; "
          f"{store.tokens_used(args.experiment)} tokens used in {args.experiment!r}")
    return 1 if failed else 0


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{100 * x:5.1f}"


def _summary(args: argparse.Namespace) -> int:
    store = Store(Path(args.data))
    rows = store.states(args.experiment)
    if not rows:
        print(f"no branch points in experiment {args.experiment!r}")
        return 1
    print(f"{'task':<22} {'trunk':<8} {'step':>4} {'pass/k':>7} {'V':>6} {'95% interval':>15} {'true V':>7} {'trunk':>6}")
    covered = known = 0
    for r in rows:
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
        print(f"{r['task_id']:<22} {r['trunk'][-6:]:<8} {r['step']:>4} {r['passes']:>3}/{r['n']:<3} {_pct(v)}% "
              f"[{_pct(lo)}, {_pct(hi)}] {'' if true is None else _pct(true) + '%':>7} {trunk:>6}{mark}{unscored}")
    if known:
        print(f"\ntrue value inside the 95% interval: {covered}/{known} states")

    # A trunk that never finished has no last step, so it cannot say when its outcome was settled.
    states = [State(r["task_id"], r["trunk"], r["step"], r["passes"], r["n"], r["trunk_steps"])
              for r in rows if r["trunk_outcome"] is not None]
    steps = sorted({s.step for s in states})
    if any(s.n >= 2 for s in states):
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
    run.add_argument("--max-tokens", type=int, help="stop sending model requests once the experiment has used this many tokens")

    chk = sub.add_parser("check", help="check the tasks themselves: each must fail untouched and pass after "
                                       "its reference solution (no model involved)")
    task_args(chk)

    summ = sub.add_parser("summary", help="value per branch point, decided-by-step, fork against fresh")
    summ.add_argument("--experiment", required=True)
    summ.add_argument("--data", default="data")
    summ.add_argument("--draws", type=int, default=2000)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_run(args))
    if args.cmd == "check":
        return asyncio.run(_check(args))
    return _summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
