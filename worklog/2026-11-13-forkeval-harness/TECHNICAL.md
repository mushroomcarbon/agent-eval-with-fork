# Work report: forkeval, an agent eval harness built on sandbox fork (2026-11-13 to 2026-11-24)

| | |
|---|---|
| **Status** | **Partly done.** Harness built and validated; medium1 complete (62 tasks); the decisive-step experiment `decisive1` is chained across plan windows from 2026-11-24 19:00 UTC |
| **Trigger** | The operator wanted a resume project that uses the platform's new `fork` (2026-11-12): an eval harness that measures where agent runs get decided |
| **Commits** | 24 in `mushroomcarbon/agent-eval-with-fork`, all pushed; `48ff58a` stage 0 … `ef7ed14` `--swap-at`; full list in §3.8 |
| **Deployed** | Nothing deployed. Runs use the dev platform as an ordinary tenant, public SDK only |
| **Customer impact** | None. Dev only. The test tenant's sandbox limit was raised 50 → 200 on dev on 2026-11-17 for these runs (still 200) |
| **Times** | UTC |
| **Prepared by** | Claude Code with the operator |

## 1. Summary

- **What it measures.** V(s_t) = P(the run succeeds | it is in state s_t) for states along real agent runs. A state
  is the conversation so far plus the whole sandbox, processes and memory included. It is estimated by forking the
  sandbox at s_t and letting k continuations finish. A real fork, not a replay, so live processes come along.
- **Validated** against a synthetic agent whose true V is known in closed form: the true value lies inside the 95%
  interval at 64 of 64 states.
- **medium1** (Haiku via `claude -p`, 62 Terminal-Bench medium tasks, one trunk each, k = 6):
  - 45 of 62 trunks passed;
  - 44% of outcome variance is left to the run (56% is which task it is);
  - about 48% of the part left to the run is settled by step 4, 77% by step 7, 88% by step 12;
  - cost 131.7M tokens, $24.31 at the CLI's list figure, 8 plan windows, run unattended.
- **Value review (2026-11-24).** The concept is known (Monte Carlo rollouts from prefixes, as in PRM labelling and
  agent tree search), and fork with memory is parity with a few vendors, not unique. The operator chose two
  directions only a fork can answer:
  1. **Decisive step found by forks vs. LLM-judge attribution.**
  2. **A second model on the first model's exact states:** does Sonnet recover a state just after Haiku's decisive
     mistake?
- **Built for that:**
  - `--bisect`, `--confirm-k`, `--swap-model`/`--swap-at`;
  - step 0 measured once per task;
  - `forkeval judge` and `forkeval decisive`.
- **First signal**, from medium1's existing data (6 failed runs with a clear decisive step):
  - 3 of the 6 were decided by the agent ending the run while forks from its last state passed 5–6/6;
  - Opus and Sonnet judges named the same step as the forks 4/6 times, Haiku 3/6;
  - where they differ, judges named where a bug was introduced and forks where it stopped being recoverable.

## 3. What we did

### 3.1 Design (2026-11-13/14, `DESIGN.md`)

| Decision | Choice | Why |
|---|---|---|
| Agent loop | Outside the sandbox; a small loop of our own (`shell`, `write_file`) | The loop must stop at every tool boundary and hand its state over as a plain value; keeps model credentials out of the guest |
| Branching | `bp = trunk.fork(1); bp.suspend()`, later `bp.fork(k)` | One platform verb; a suspended branch point can be forked again later, which enables adaptive schedules |
| Randomness | Continuations differ only through the model's sampling | Measured: userspace PRNG state is identical in every fork, the kernel's differs |
| Statistics | Jeffreys intervals per state; unbiased k/(k-1)·p(1-p); decided-by-step from the martingale variance decomposition; hierarchical bootstrap (tasks → trunks → continuations) | k is small; the plug-in variance is biased towards "decided" |
| Platform access | Public SDK only (`zillion-asb` ≥ 0.5.0); no platform internals in the repo | The claim is "built on the public API"; the repo may go public |
| Benchmark | Terminal-Bench single-container tasks | Live processes are where a memory fork matters |

### 3.2 Stage 0: pipeline and validation (`48ff58a`, 2026-11-14)

- Task format, agent loop, trunk/branch-point/continuation runner, SQLite store with JSONL transcripts, estimators, CLI.
- Synthetic `gates` agent: three gates opened correctly with probability 0.8, 0.7, 0.9. The task keeps its state only
  in the memory of a daemon started during setup, so a fork that dropped processes would fail every test.
- `results/stage0-validation-gates.txt`:
  - 16 trunks, 64 branch points, 384 continuations, 36 fresh runs;
  - true V inside the 95% interval at 64/64 states;
  - decided-by-step 34% [2, 61] and 77% [53, 96], against true values of 25% and 80%;
  - step-0 forks 42/96 and fresh runs 14/36, against a true pass rate of 50.4%.

### 3.3 Reaching the model on a subscription (`b6a1d1f`…`a8259db`, 2026-11-16)

- `--policy claude-code`:
  - each step is one fresh `claude -p` process, with the transcript as text and one JSON action as structured output;
  - system prompt replaced, tools off, empty working directory;
  - no reasoning is carried between steps. That makes it a different agent from `--policy anthropic`, and results
    name their policy.
- Plan usage is read from the CLI's `rate_limit_event` (`five_hour`, `seven_day` utilisation).
  `--max-plan-use` / `--max-week-use` stop an experiment at a set share of each window, because the subscription is
  shared with the operator's own work.
- Smoke test (`results/smoke-claude-code-haiku.txt`): 30 runs, all scored, 29 passed.
- Seen in the smoke test: the model took to running `python3 -I`, a habit from a line the CLI adds about its host.

### 3.4 Stage 1: Terminal-Bench adapter (`042a8a7`, `01dffee`, 2026-11-16)

- A task's Dockerfile is read as a base image plus steps. The steps are applied once in a sandbox and the result is
  committed as an image (`fe-<task>:<hash>`). No Docker is involved.
- 195 of 241 tasks fit. The rest are skipped with a reason: several containers, multi-stage builds, USER, heredocs,
  and so on.
- `forkeval check`: each task must fail untouched and pass after its reference solution. 84 of 100 were sound on dev
  (`results/tbench-sound-2026-11-16.txt`).
- No benchmark content is copied into the repo.

### 3.5 Making long runs affordable (`5d5c64c`…`06837a6`, 2026-11-16/17)

| Feature | What | Why |
|---|---|---|
| `--adaptive` | Snapshot every step; measure in step order; stop after the first state whose k continuations all match the trunk's outcome | Replayed on pilot1's full grid: 59 of 266 continuations (22%), skipped states mean open variance 0.010 |
| `--max-sandboxes` | Room accounting against the tenant limit; a step with no room in 2 min is recorded as not snapshotted | A suspended branch point counts as a live sandbox; a 20-step trunk holds 21 |
| `--max-states M` | At most M measured states per trunk, evenly spaced | `chess-best-move` sat near 50% for ~20 steps at 370k tokens per continuation, ~44M tokens at every step |
| Keep/resume | When a budget stops a finished trunk, its snapshots get their deadline extended (`--keep-hours`, 24) and the same command measures them later; `forkeval drop` deletes them | A budget stop used to throw away the work in flight |
| Wind-down | Nothing new starts within 5 points of the 5-hour limit or 1 point of the weekly one | First version used 5 points for both and stopped a window 13 minutes in (5 weekly points is five windows of work) |

### 3.6 medium1 (2026-11-16 to 2026-11-20, `9b74cb6`)

- 62 sound medium tasks, one Haiku trunk each, k = 6, `--adaptive`, `--max-states 6` from the third window on.
- Chained unattended across 8 plan windows (5-hour window capped at 50%, weekly at 70%).
- Results (`results/medium1-claude-code-haiku.txt`):

| Measure | Value |
|---|---|
| Trunks passed | 45 / 62 |
| Variance at step 0 | 56.1% which task, 43.9% left to the run; 26 tasks mixed at step 0 |
| Decided by step 1 / 2 / 4 / 7 / 12 / 18 | 10.7% [-19, 38] / 39.1% / 47.9% [23, 72] / 76.5% [53, 95] / 87.5% / 94.4% |
| Branch points | 840; fork call median 2.3 s (the max, 434 s, is the harness waiting on the tenant quota of 50 before it was raised) |
| Cost | 131.7M tokens, $24.31 at the CLI's list figure |

- Read critically:
  - a typical run is about 7 steps, so "77% by step 7" means the outcome is decided gradually, not early;
  - one trunk per task and k = 6 make single curves noisy;
  - the step-1 interval crosses zero;
  - no fork-against-fresh check on these tasks yet.
- Report page (private artifact): https://claude.ai/code/artifact/7db106d6-48d8-438f-a9db-1284d65de936

### 3.7 Decisive steps and the second model (2026-11-24, `2557eb9`, `a1a4bad`, `46e2315`, `ef7ed14`)

- **Decisive move:** the biggest move of V towards the outcome between neighbouring measured points. The run's own
  outcome is a point at step T + 1, so "ended the run too early" can be the decisive step. The move is **clear** when
  it is one action wide and V moves at least 0.5 across it (`src/forkeval/decisive.py`).
- **`--bisect`:** measure the first and last states, then the middle of the biggest move, until it is one action wide
  (about 2 + log2 T states, capped by `--max-states`). Every snapshot is held until then.
- **`--confirm-k 12`:** the two states either side get 12 continuations each, and the search goes on if the move shifts.
- **Step 0 measured once per task:** all trunks of a task start from the same state, so the others reuse it
  (`skipped = 6`).
- **`--swap-model M --swap-at after`:** M continues the state just after each failed run's decisive step.
  `--swap-at all` also covers step 0 and the state just before; it was dropped for cost (below).
- **`forkeval judge`:** a judge model sees a failed run with numbered steps (the ending turn included) and names the
  decisive step.
- **`forkeval decisive`:** scores judges against the forks, with baselines (last step, first failing command, chance)
  and the swap table.
- Resumable across budget stops like everything else.
- **First signal (medium1 data, 3 judges via `claude -p`):**

| | exact | within 1 step |
|---|---|---|
| Opus 5.5 judge | 4/6 | 4/6 |
| Sonnet 5.5 judge | 4/6 | 4/6 |
| Haiku 5.5 judge | 3/6 | 4/6 |
| Baseline: last step | 3/6 | |
| Baseline: first failed command | 1/6 | |
| Baseline: random step | 19% expected | |

- Example, `ode-solver-rk4`:
  - every judge blamed step 2, where a solver making ~8× the minimum function calls was written;
  - forks from the states after step 2 and step 3 passed 6/6, so the bug was still recoverable;
  - the failure was fixed by step 4, the agent ending the run without checking the solver.
- Example, `tree-directory-parser`: forks and the Opus and Sonnet judges agree on the ending turn (the generator was
  never run); the Haiku judge blamed the generator, whose state forks passed 6/6.
- **Pilot of `decisive1`** (3 tasks, 1 trunk each, with Sonnet at step 0 and both sides):
  - 3.87M tokens, $2.33;
  - all three trunks passed;
  - `puzzle-solver` shows a clear decisive good step (5/12 → 12/12 across action 2);
  - Sonnet 11/12 against Haiku 10/18 from the same start (2 tasks only);
  - the 5-hour window hit the wind-down at 47%; 8 snapshots of one trunk kept for resume.
- **Cost finding:**
  - per continuation, Sonnet ~$0.15 against Haiku ~$0.007 (list figures);
  - the full design's Sonnet part would have been ~$45–50, about 15 windows;
  - cut to `--swap-at after`, estimated $10–15 for ~20 failed runs.

### 3.8 Commits

`48ff58a` stage 0 · `1e12f75` design · `b6a1d1f` claude-code policy · `bb86b1b` smoke test · `f1bed72` README ·
`042a8a7` Terminal-Bench adapter · `94c06f7` top-up runs · `1b7dccd` list-price cost · `a8259db` plan guard · `e5b071e`
README · `d1bfd71` variance split, pilot1 · `01dffee` soundness results · `5d5c64c` adaptive · `fb31ad4` weekly guard ·
`c5116fe` sandbox limit · `7fc6900` summary tidy · `0fe3945` max-states, keep/resume · `06837a6` per-window wind-down ·
`9b74cb6` medium1 results · `2557eb9` decisive steps, swap, judge · `a1a4bad` shared step 0 · `46e2315` README ·
`ef7ed14` `--swap-at`.

Platform "files" (CLAUDE.md's six): not applicable. This is a separate repo that only consumes the public SDK; no API
surface changed.

## 4. Verification

| Check | Result |
|---|---|
| Unit tests (`pytest`, fakes; no platform or model) | 57 passed |
| Synthetic validation on real forks | true V inside the 95% interval at 64/64 states |
| Real model, real platform | smoke test 30/30 scored; medium1 62/62 tasks complete |
| Keep/resume across windows | exercised in medium1's chain (8 windows); snapshots survived about 2 days |
| Bisect + swap on the platform | `decisive1` pilot: 3 trunks, Haiku and Sonnet continuations scored, 0 unscored |
| Judge | 18 judgments made on medium1 (3 judges × 6 runs) |

## 5. Status (2026-11-24 ~18:00 UTC)

- `decisive1`:
  - the chain is waiting for the 5-hour reset at 19:00 UTC;
  - it then runs 3 Haiku trunks per task on the 26 tasks mixed at step 0 in medium1 (`--bisect --max-states 12
    --confirm-k 12 --swap-model claude-sonnet-5-5 --swap-at after`, 5-hour 50%, weekly 70%, 400M token cap);
  - it stops on: all done, a non-budget error, a window that used no tokens, or the weekly window at 69%.
- 8 snapshots from the pilot are kept on dev for the chain's first window.
- Plan: weekly window 4% at the start of the chain.
- The dev test tenant's sandbox limit is still 200 (set 2026-11-17; not re-checked today).

## 6. Still to do

1. Let `decisive1` finish. If it gives too few clear failures (target ~40), raise `--trunks` (the same command adds).
2. Run the judges (Haiku, Sonnet, Opus) on every clear case; one call per run per judge.
3. Analysis:
   - agreement with intervals;
   - sort the disagreements into "where the mistake was made" vs. "where it became unrecoverable";
   - Sonnet's recovery rate after Haiku's decisive step.
4. Write-up: results file, the report page rebuilt around the two findings, README headline.
5. Fork-against-fresh check on the medium tasks (fresh runs, about 6M tokens for one per task).
6. **Operator:** roll the dev tenant limit back to the default once the runs end.
7. **Operator:** publish the SDK release with `fork` (Python 0.5.0 / TypeScript 0.6.0) and decide whether the repo goes public. Until then nobody else can run it.
8. A run through `--policy anthropic` once an API key is available, so the results speak about a standard agent loop.

## 7. Operating it

- Launchers and the chain script live in the session scratchpad (`decisive1.sh TRUNKS [TASKS]`, `chain_d1.sh`).
  They set the platform address and read the test tenant's token from its credential file under `/etc`. No token is
  stored in the repo.
- Progress: `forkeval summary --experiment E`, `forkeval decisive --experiment E`.
- Stop: kill the chain, then `forkeval drop --experiment E` deletes kept snapshots.
