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
from .policy import AnthropicPolicy, GatesPolicy, Policy
from .runner import BudgetExceeded, Runner, schedule
from .stats import State, bootstrap_decided, decided_by_step, jeffreys_interval
from .store import Store
from .tasks import load_tasks


def _policy(args: argparse.Namespace) -> Policy:
    if args.policy == "gates":
        return GatesPolicy([float(x) for x in args.gates.split(",")])
    return AnthropicPolicy(args.model)


async def _run(args: argparse.Namespace) -> int:
    tasks = load_tasks(Path(args.tasks))
    if args.only:
        tasks = [t for t in tasks if t.id in set(args.only.split(","))]
    store = Store(Path(args.data))
    runner = Runner(factory=SandboxFactory(), policy=_policy(args), store=store, experiment=args.experiment,
                    max_tokens=args.max_tokens, concurrency=args.concurrency)
    jobs = []
    for task in tasks:
        jobs += [runner.trunk(task, k=args.k, branch_at=schedule(args.schedule)) for _ in range(args.trunks)]
        jobs += [runner.fresh(task) for _ in range(args.fresh)]
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

    states = [State(r["task_id"], r["trunk"], r["step"], r["passes"], r["n"], r["trunk_steps"]) for r in rows]
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

    run = sub.add_parser("run", help="run trunks (with branch points and continuations) and fresh runs")
    run.add_argument("--tasks", required=True, help="directory of task *.toml files")
    run.add_argument("--only", help="comma-separated task ids")
    run.add_argument("--experiment", required=True)
    run.add_argument("--data", default="data")
    run.add_argument("--policy", choices=["anthropic", "gates"], default="anthropic")
    run.add_argument("--model", default="claude-haiku-5-5")
    run.add_argument("--gates", default="0.8,0.7,0.9", help="per-gate success probabilities for --policy gates")
    run.add_argument("--trunks", type=int, default=1, help="trunks per task")
    run.add_argument("--fresh", type=int, default=0, help="ordinary runs per task, for the fork-against-fresh check")
    run.add_argument("-k", type=int, default=6, help="continuations per branch point")
    run.add_argument("--schedule", default="every", help='"every", "grid", or steps like "0,2,5"')
    run.add_argument("--concurrency", type=int, default=2, help="tasks in flight at once")
    run.add_argument("--max-tokens", type=int, help="stop sending model requests once the experiment has used this many tokens")

    summ = sub.add_parser("summary", help="value per branch point, decided-by-step, fork against fresh")
    summ.add_argument("--experiment", required=True)
    summ.add_argument("--data", default="data")
    summ.add_argument("--draws", type=int, default=2000)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_run(args))
    return _summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
