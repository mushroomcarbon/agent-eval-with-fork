"""How often does the decisive-step search call a step "clear" when no single step decides the run?

The bisection (`decisive.next_to_measure`) and the confirmation stage are replayed on value paths whose truth is
known, with binomial noise in place of continuations. On a gradual path every action moves V by the same small
amount, so a "clear" move there is a product of noise and of the run's own outcome standing in as a noise-free
point after the last state. Pure functions; no I/O.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .decisive import Move, biggest_move, next_to_measure


def search(path: Sequence[float], rng: random.Random, *, k: int = 6, confirm_k: int = 8, cap: int = 12,
           start_n: int = 18) -> Move | None:
    """Run the runner's search on a failed run whose state s has true value path[s] (states 0..T).

    Mirrors `Runner._bisect`: step 0 is measured once per task with `start_n` continuations, other states get
    `k`; once the search stops, the two states around the move found are topped up to `confirm_k` each and the
    search goes on from the new values."""
    steps = list(range(len(path)))
    trunk_steps = len(path) - 1
    passes = [0] * len(path)
    n = [0] * len(path)
    tried: set[int] = set()
    confirmed: set[int] = set()

    def draw(i: int, m: int) -> None:
        passes[i] += sum(rng.random() < path[i] for _ in range(m))
        n[i] += m

    while True:
        values = [passes[i] / n[i] if n[i] else None for i in steps]
        available = [n[i] == 0 and i not in tried for i in steps]
        i = next_to_measure(steps, values, available, False, trunk_steps, cap=cap)
        if i is not None:
            tried.add(i)
            draw(i, start_n if i == 0 else k)
            continue
        move = biggest_move(steps, values, False, trunk_steps)
        ends = [] if move is None or not move.single else \
            [j for j in (move.before, move.after) if j <= trunk_steps and j != 0 and n[j] < confirm_k
             and j not in confirmed]
        if not ends:
            return move
        for j in ends:
            confirmed.add(j)
            draw(j, confirm_k - n[j])


def gradual(v0: float, v_end: float, steps: int) -> list[float]:
    """V falls by the same amount at every action: no step decides the run."""
    return [v0 + (v_end - v0) * s / steps for s in range(steps + 1)]


def single_drop(v0: float, v_low: float, steps: int, at: int) -> list[float]:
    """V stays at v0 until action `at`, then sits at v_low: action `at` decides the run."""
    return [v0 if s < at else v_low for s in range(steps + 1)]


@dataclass
class Rates:
    clear: float      # share of runs with a clear decisive step
    at_end: float     # ... at the turn that ended the run
    interior: float   # ... at an action inside the run
    right: float      # ... at the action that truly decided it (single_drop paths only)


def rates(make_path, draws: int, seed: int = 0, truth: int | None = None, **kw) -> Rates:
    rng = random.Random(seed)
    clear = end = inside = right = 0
    for _ in range(draws):
        path = make_path()
        m = search(path, rng, **kw)
        if m is None or not m.clear:
            continue
        clear += 1
        if m.after == len(path):
            end += 1
        else:
            inside += 1
        right += truth is not None and m.after == truth
    return Rates(clear / draws, end / draws, inside / draws, right / draws)
