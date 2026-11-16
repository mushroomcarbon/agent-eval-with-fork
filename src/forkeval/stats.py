"""Estimators and intervals. Pure functions; no I/O, no third-party dependencies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

# --- the Beta distribution, enough of it for a Jeffreys interval -------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


# --- one state -----------------------------------------------------------------


def jeffreys_interval(passes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed Jeffreys interval for a binomial proportion.

    With the usual convention that the lower limit is 0 when passes == 0 and
    the upper limit is 1 when passes == n, which keeps coverage near nominal at
    the ends, where k is small and a normal interval collapses to a point.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= passes <= n:
        raise ValueError("passes must be between 0 and n")
    alpha = 1.0 - level
    a, b = passes + 0.5, n - passes + 0.5
    lo = 0.0 if passes == 0 else beta_quantile(alpha / 2, a, b)
    hi = 1.0 if passes == n else beta_quantile(1 - alpha / 2, a, b)
    return (lo, hi)


def undecided(passes: int, n: int) -> float:
    """Unbiased estimate of V(1 - V) from n pass/fail draws at one state.

    V(1 - V) is the outcome variance still open at the state: 0 when the run is
    already won or lost, 1/4 when it is a coin flip. The plug-in p(1 - p) is
    biased low by a factor (n - 1)/n, which would make every state look more
    decided than it is.
    """
    if n < 2:
        raise ValueError("needs at least 2 continuations")
    p = passes / n
    return n / (n - 1) * p * (1 - p)


# --- along runs ------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """Continuations of one branch point."""

    task: str
    trunk: str
    step: int
    passes: int
    n: int
    #: How many steps the trunk took in all. Past its last step a trunk's
    #: outcome is fixed, so its undecided variance there is exactly 0.
    trunk_steps: int


def decided_by_step(states: Sequence[State], steps: Sequence[int]) -> dict[int, float]:
    """Share of the outcome variance that is settled by each step.

    Along a run V is a martingale ending at 0 or 1, so
        V0 (1 - V0) = E[ V_t (1 - V_t) ] + E[ sum of squared changes up to t ],
    and the share decided by step t is 1 - E[V_t(1 - V_t)] / (V0(1 - V0)).
    Expectations are over trunks of the same task; tasks are averaged with the
    weight of their own step-0 variance (a task that always passes has nothing
    to decide and gets no say).

    A trunk that has no branch point at step t contributes its latest earlier
    one (the schedule skipped the step); a trunk that ended before t contributes 0.
    """
    by_trunk: dict[tuple[str, str], list[State]] = {}
    for s in states:
        if s.n >= 2:
            by_trunk.setdefault((s.task, s.trunk), []).append(s)
    for v in by_trunk.values():
        v.sort(key=lambda s: s.step)

    def open_at(trunk: list[State], t: int) -> float | None:
        if t > trunk[0].trunk_steps:
            return 0.0
        earlier = [s for s in trunk if s.step <= t]
        if not earlier:
            return None
        return undecided(earlier[-1].passes, earlier[-1].n)

    tasks = sorted({k[0] for k in by_trunk})
    out: dict[int, float] = {}
    for t in steps:
        num = den = 0.0
        for task in tasks:
            trunks = [v for (tk, _), v in by_trunk.items() if tk == task]
            at0 = [x for x in (open_at(v, 0) for v in trunks) if x is not None]
            att = [x for x in (open_at(v, t) for v in trunks) if x is not None]
            if not at0 or not att:
                continue
            den += sum(at0) / len(at0)
            num += sum(att) / len(att)
        out[t] = float("nan") if den <= 0 else 1.0 - num / den
    return out


def variance_at_start(states: Sequence[State]) -> tuple[float, float, int, int]:
    """Where the outcome variance sits before any step is taken:
    (between tasks, within a task, tasks with a mixed result, tasks).

    Over a random task and a random run of it, Var(outcome) = Var(V0) + E[V0 (1 - V0)]:
    the first term is which task was drawn, the second is what is left for the run
    to decide, and only the second is what `decided_by_step` divides up. Step-0
    continuations are pooled per task. The spread of the estimated V0 overstates
    Var(V0) by their sampling variance V0(1 - V0)/n, which is taken off.
    """
    pooled: dict[str, list[int]] = {}
    for s in states:
        if s.step == 0:
            agg = pooled.setdefault(s.task, [0, 0])
            agg[0] += s.passes
            agg[1] += s.n
    tasks = [(x, n) for x, n in pooled.values() if n >= 2]
    if len(tasks) < 2:
        return (float("nan"), float("nan"), sum(0 < x < n for x, n in tasks), len(tasks))
    p = [x / n for x, n in tasks]
    open_ = [undecided(x, n) for x, n in tasks]
    mean = sum(p) / len(p)
    spread = sum((v - mean) ** 2 for v in p) / (len(p) - 1)
    noise = sum(u / n for u, (_, n) in zip(open_, tasks)) / len(tasks)
    return (max(0.0, spread - noise), sum(open_) / len(open_), sum(0 < x < n for x, n in tasks), len(tasks))


def bootstrap_decided(states: Sequence[State], steps: Sequence[int], *, draws: int = 2000,
                      level: float = 0.95, seed: int = 0) -> dict[int, tuple[float, float]]:
    """Percentile interval for `decided_by_step`, resampling the whole hierarchy:
    tasks, then trunks within a task, then continuations within a state."""
    rng = random.Random(seed)
    by_task: dict[str, dict[str, list[State]]] = {}
    for s in states:
        by_task.setdefault(s.task, {}).setdefault(s.trunk, []).append(s)
    task_ids = sorted(by_task)
    samples: dict[int, list[float]] = {t: [] for t in steps}
    for _ in range(draws):
        resampled: list[State] = []
        for i in range(len(task_ids)):
            task = rng.choice(task_ids)
            trunk_ids = sorted(by_task[task])
            for j in range(len(trunk_ids)):
                trunk = rng.choice(trunk_ids)
                for s in by_task[task][trunk]:
                    p = s.passes / s.n if s.n else 0.0
                    passes = sum(1 for _ in range(s.n) if rng.random() < p)
                    resampled.append(State(f"{i}", f"{i}.{j}", s.step, passes, s.n, s.trunk_steps))
        est = decided_by_step(resampled, steps)
        for t in steps:
            if not math.isnan(est[t]):
                samples[t].append(est[t])
    out: dict[int, tuple[float, float]] = {}
    alpha = 1.0 - level
    for t in steps:
        xs = sorted(samples[t])
        if not xs:
            out[t] = (float("nan"), float("nan"))
            continue
        out[t] = (xs[int(alpha / 2 * (len(xs) - 1))], xs[int((1 - alpha / 2) * (len(xs) - 1))])
    return out
