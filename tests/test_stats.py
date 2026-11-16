import math
import random

import pytest

from forkeval.stats import State, beta_cdf, decided_by_step, jeffreys_interval, undecided


def test_beta_cdf_known_values():
    assert beta_cdf(0.5, 1, 1) == pytest.approx(0.5)
    assert beta_cdf(0.25, 2, 1) == pytest.approx(0.0625)        # x^2
    assert beta_cdf(0.5, 0.5, 0.5) == pytest.approx(0.5)        # arcsine law is symmetric
    assert beta_cdf(0.3, 2, 3) == pytest.approx(0.3483, abs=1e-4)


def test_jeffreys_matches_published_values():
    # Brown, Cai & DasGupta (2001), Table 5: n = 10, x = 2 -> (0.044, 0.503); x = 5 -> (0.224, 0.776)
    lo, hi = jeffreys_interval(2, 10)
    assert (lo, hi) == pytest.approx((0.044, 0.503), abs=2e-3)
    lo, hi = jeffreys_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.224, 0.776), abs=2e-3)


def test_jeffreys_ends():
    assert jeffreys_interval(0, 6)[0] == 0.0
    assert jeffreys_interval(6, 6)[1] == 1.0
    assert 0 < jeffreys_interval(0, 6)[1] < 0.5
    assert jeffreys_interval(0, 0) == (0.0, 1.0)


def test_jeffreys_coverage_is_near_nominal():
    rng = random.Random(1)
    for p in (0.1, 0.5, 0.8):
        hits = 0
        for _ in range(4000):
            x = sum(rng.random() < p for _ in range(6))
            lo, hi = jeffreys_interval(x, 6)
            hits += lo <= p <= hi
        assert 0.90 <= hits / 4000 <= 1.0, (p, hits / 4000)


def test_undecided_is_unbiased():
    rng = random.Random(2)
    p, n = 0.3, 6
    mean = sum(undecided(sum(rng.random() < p for _ in range(n)), n) for _ in range(40000)) / 40000
    assert mean == pytest.approx(p * (1 - p), abs=0.004)
    plug_in = sum((x / n) * (1 - x / n) for x in (sum(rng.random() < p for _ in range(n)) for _ in range(40000))) / 40000
    assert plug_in < p * (1 - p) - 0.02     # the plug-in really is biased low


def test_decided_by_step_recovers_a_known_process():
    # Three gates with success probabilities p. Simulate many trunks, with k
    # continuations sampled from the TRUE value at each state, and check the
    # estimate against the closed form 1 - E[V_t(1-V_t)] / (V_0(1-V_0)).
    p = [0.8, 0.7, 0.9]
    rng = random.Random(3)
    k = 6
    states = []
    for trunk in range(3000):
        alive, steps = True, len(p)
        for t in range(len(p) + 1):
            v = math.prod(p[t:]) if alive else 0.0
            passes = sum(rng.random() < v for _ in range(k))
            states.append(State("gates", str(trunk), t, passes, k, steps))
            if t < len(p) and alive:
                alive = rng.random() < p[t]
    est = decided_by_step(states, [0, 1, 2, 3])
    v0 = math.prod(p)

    def truth(t):
        reach = math.prod(p[:t])            # P(still alive at t)
        v = math.prod(p[t:])
        return 1 - reach * v * (1 - v) / (v0 * (1 - v0))

    assert est[0] == pytest.approx(0.0, abs=1e-9)
    for t in (1, 2, 3):
        assert est[t] == pytest.approx(truth(t), abs=0.03), t
    assert est[3] == pytest.approx(1.0, abs=0.03)


def test_decided_handles_trunks_that_end_early():
    states = [State("t", "a", 0, 3, 6, 1), State("t", "a", 1, 6, 6, 1)]
    est = decided_by_step(states, [0, 1, 5])
    assert est[1] == pytest.approx(1.0)
    assert est[5] == pytest.approx(1.0)     # past the trunk's end nothing is open


def test_variance_at_start_separates_task_from_run():
    from forkeval.stats import variance_at_start

    # Tasks whose true pass rate is 0.1, 0.5 or 0.9: Var(V0) = 0.32^2 * 2/3 ... computed below.
    rng = random.Random(1)
    truth = [rng.choice([0.1, 0.5, 0.9]) for _ in range(3000)]
    states = [State(f"t{i}", "a", 0, sum(rng.random() < v for _ in range(6)), 6, 3) for i, v in enumerate(truth)]
    between, within, mixed, n = variance_at_start(states)
    mean = sum(truth) / len(truth)
    assert n == 3000 and 0 < mixed < n
    assert between == pytest.approx(sum((v - mean) ** 2 for v in truth) / len(truth), abs=0.01)
    assert within == pytest.approx(sum(v * (1 - v) for v in truth) / len(truth), abs=0.01)

    # Tasks that always pass or always fail: everything is the task, nothing is left to the run.
    states = [State(f"t{i}", "a", 0, 6 * (i % 2), 6, 3) for i in range(10)]
    between, within, mixed, n = variance_at_start(states)
    assert within == 0 and mixed == 0 and between == pytest.approx(0.25 * 10 / 9)
