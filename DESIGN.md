# forkeval: where agent runs get decided

Design draft, 2026-11-13. Nothing is built. Open decisions are in §9; everything else is a proposal.

## 1. What it measures

An agent run on a task is a sequence of steps (model call, tool call, tool result). After each tool result the
world is in a definite state: the conversation so far, and everything inside the sandbox (files, running
processes, memory). Call that state `s_t`.

The harness estimates **V(s_t) = P(the run ends in success | it is in state s_t)** for the states along real
runs, by forking the sandbox at `s_t` and letting `k` independent continuations finish from it. Each continuation
is scored pass or fail by the task's own tests.

That one quantity gives the three results the project is for:

1. **Value curves.** V along each run, step by step: where it climbs, where it collapses.
2. **How early the outcome is fixed.** Along a run V is a martingale that ends at 0 or 1, so the outcome variance
   of a task splits exactly by step: `V0(1-V0) = sum over t of E[(V_{t+1} - V_t)^2]`. The share of that sum spent
   by step t is "how much of the outcome was decided by step t". This is the headline number ("x% of the outcome
   is fixed by step 5").
3. **The usual numbers, with honest error bars.** V at step 0 is the task's ordinary pass rate, so pass@1, the
   split of variance into task difficulty versus sampling noise, paired model comparisons and a power calculation
   all come from the same data.

Why a fork and not a replay: re-running the same commands from scratch does not reproduce `s_t` when the task has
live processes, timing, or anything the agent started in the background, and it costs the whole prefix again. A
fork is the exact state, in about a second. How often replay fails to reproduce the state is itself measured (§6).

## 2. Decisions already made

- **The agent loop runs outside the sandbox.** The harness process holds the conversation; the sandbox is only
  the environment the tools act on. Forking a state is then: copy the message list, fork the VM. It also keeps the
  model API key out of the guest, and avoids the two things a fork cannot carry (broker and egress sessions).
- **The agent is a small loop of our own** (model + a shell tool + a file-edit tool, both through `exec`), not an
  existing agent product. The loop has to stop at every tool boundary and hand its state over as a plain value.
- **Continuations differ only through the model's sampling.** Measured on the platform: a random generator a
  program seeded before the fork returns the same numbers in every fork; the kernel's does not. So the estimand is
  "probability over the agent's own sampling, given this exact state", at the model's default temperature. The
  harness does not reseed anything in the box; tasks whose tests depend on in-box randomness get flagged.

## 3. How a state is kept and branched

The platform has one verb, `fork(count)`, and every fork starts running. To keep a state to come back to:

```
after tool result t:
    bp = trunk.fork(1)        # exact copy of s_t; the trunk is frozen ~0.4-1.4 s, then continues
    bp.suspend()              # bp is now a frozen branch point for step t
later, any number of times:
    conts = bp.fork(k)        # k running continuations of s_t; bp is untouched
```

A suspended branch point can be forked again later, which matters for §5: samples can be added to a state after
looking at the data, and steps can be chosen adaptively. The copy runs for a moment between `fork` and `suspend`;
whether that drift is visible is checked in stage 0 (fork at step 0 against fresh runs).

Each continuation gets a copy of the trunk's messages up to `t`, runs the same loop to the end, is scored, and is
deleted. Tests are copied into a sandbox only at scoring time, so the agent never sees them. Scoring a state
without continuing it ("is the task already solved at step t?") is the same mechanism: fork one copy, run the
tests in the copy.

Platform limits that shape the harness: forks land on the source's node, so one node's capacity bounds
`k x concurrent tasks`; 32 forks per call; fork chains up to 64 deep; the tenant's concurrent-sandbox limit; a
sandbox's lifetime (`max_session_seconds`, which branch points inherit, so tasks are created with a long one).

## 4. Pieces

| Piece | What it does |
|---|---|
| `tasks/` | A task = image, instruction, setup command, test command, step and time limits. One adapter per benchmark format. |
| `agent.py` | The loop. State is a value: `(messages, sandbox)`. One function advances it by one step. |
| `runner.py` | Runs a trunk, makes branch points on the schedule, runs continuations under a concurrency cap, scores, cleans up. Resumable: every action is recorded before it is taken. |
| `store` | SQLite: tasks, runs, steps, branch points, continuations, outcomes, tokens (cached and uncached), wall times, fork timings. Transcripts as JSONL files. |
| `analysis/` | Estimators and intervals (§5), figures. Pure functions over the store. |
| `report` | One static HTML page: value curves, decided-by-step curve, variance table, model comparison, costs. |

Python, asyncio, the published customer SDK (`zillion-asb`) and the Anthropic SDK. No access to the platform
beyond what a tenant has: the claim is "built on the public API".

## 5. Sampling and statistics

**Structure.** task -> trunk (one full run) -> branch point (a step on the trunk) -> continuation (pass/fail).
Continuations of one state are independent given the state. States on one trunk are not independent, nor are
trunks of one task. All inference across tasks therefore resamples whole tasks.

**Per state.** `V_hat = passes / k` with a Jeffreys interval (k is small; a normal interval is wrong at 0 and k).
The unbiased estimate of `V(1-V)` from k samples is `k/(k-1) * V_hat * (1 - V_hat)`; the decided-by-step curve
is built from that, not from the plug-in, which is biased towards "decided".

**Across tasks.** Hierarchical bootstrap: resample tasks, then trunks within a task, then continuations within a
state. Reported with every aggregate: the interval, the number of tasks, trunks and continuations behind it.

**Built-in validity checks.**
- Fork against fresh: V at step 0 from forks must agree with the pass rate of ordinary fresh runs of the same
  task. If forking biased anything, it shows here.
- Calibration for free: the trunk's own outcome is one more draw from every state on it, and it is not used in
  `V_hat`. Among states with `V_hat` near p, trunks should succeed about p of the time.
- Martingale: the average of V at step t+1 over continuations should equal V at step t.

**Which steps to branch at.** Every step costs `k x (steps remaining)` model calls, so branching everywhere is
quadratic in run length. Three schedules, in order of use:
1. Pilot: every step, short tasks only, to see what the curves look like.
2. Grid: every step for the first few, then every third.
3. Adaptive: keep a suspended branch point at every step (cheap: no model calls), estimate V at the ends, then
   bisect towards the largest change. Finds the deciding step in about log2(T) branch points, and adds samples
   only where the interval is still wide.

**Comparisons.**
- Two models on the same tasks: paired difference in V at step 0, tasks resampled together.
- **Same state, different model**: continue one model's branch point with another model. "Given exactly this
  half-finished state, how often does the stronger model rescue it?" Only a fork can ask this.
- Power: from the estimated variance components, the number of tasks needed to detect a 3-point difference.

## 6. Secondary measurements

- Fork cost against replay: wall time and tokens for k continuations from a fork (the shared prefix is a prompt
  cache hit) against k runs from scratch.
- Replay divergence: re-run a trunk's commands from scratch and compare a hash of the resulting files and process
  list with the branch point's. The fraction of states replay does not reproduce.
- Platform numbers under real use: fork latency, trunk freeze, failures, at the harness's concurrency.

## 7. Stages

| Stage | Content | Exit |
|---|---|---|
| 0 | Skeleton, store, agent loop, 3 hand-written tasks. Fork-against-fresh check. | V at step 0 from forks matches fresh runs within their intervals. |
| 1 | Adapter for one benchmark, 10 tasks, cheap model, schedule 1. | Value curves and the decided-by-step curve for 10 tasks, with intervals. |
| 2 | Adaptive schedule, about 50 tasks, hierarchical bootstrap, report page. | The headline number with its interval; variance table; power calculation. |
| 3 | Second model: paired comparison and same-state swap. | Rescue rate with interval. |
| 4 | Replay divergence, cost comparison, write-up. | README with figures; blog post. |

## 8. Risks

- **Money.** Tokens are the cost, not VMs. The runner takes a hard token budget and stops at it; every stage
  starts with a dry-run estimate from the schedule. Dollar figures are computed from the current price list
  before each stage, not assumed here.
- **Task format.** Benchmarks that need several containers per task do not fit one sandbox; those tasks are
  skipped and counted. Tasks that need the network at run time work only as far as a plain sandbox has it.
- **Capacity.** Everything for one task sits on one node. The concurrency cap is set from the node, and a
  refused fork (node full, quota) is retried, not counted as a failure.
- **Not the first.** VM branching for agents is sold elsewhere, and tree search over agent runs exists on git
  state. The claim is the measurement (value curves from exact-state forks, with honest intervals) on a fork
  primitive built for it, not that forking is new.

## 9. Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| 1 | Benchmark | Terminal-Bench style tasks (live processes are where a memory fork matters), single-container subset. SWE-bench Lite later if wanted for a familiar baseline. |
| 2 | Models | `claude-haiku-5-5` for stages 0-2; `claude-sonnet-5-5` as the second model in stage 3. |
| 3 | k and schedule defaults | k = 6 per state; sequential top-up to 12 where the interval is wide. |
| 4 | Where it runs | Dev (Wnwdax) for stages 0-1; prod as an ordinary tenant for stage 2 on, which needs fork on prod and the SDK published. |
| 5 | Where the code lives | Its own public repository (it is a portfolio piece and must not carry platform internals); this file moves there. |
| 6 | API key and token budget | A key for the harness on the dev box (in `/etc`, not in the repo), and a cap per stage. |
