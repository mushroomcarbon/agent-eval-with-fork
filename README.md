# forkeval

Where do agent runs get decided?

An agent run is a sequence of steps. After each tool result the world is in a definite state: the conversation
so far and everything inside the sandbox (files, running processes, memory). forkeval estimates, for the states
along real runs,

    V(s_t) = P(the run ends in success | it is in state s_t)

by forking the sandbox at `s_t` (an exact copy of the live VM, memory included) and letting `k` independent
continuations finish from it. From that one quantity come value curves along each run, the share of the outcome
that is already settled by each step, and the usual pass rates with honest intervals. The design, the statistics
and the plan are in [DESIGN.md](DESIGN.md).

## Status

Stage 0 of 4: the pipeline works end to end on real sandboxes and real forks, validated against a synthetic
agent whose true V is known in closed form, and smoke-tested with a real model (30 runs, not enough to read
anything into).

| | |
|---|---|
| Built | task format, agent loop (model + `shell` + `write_file`), trunk / branch point / continuation runner, SQLite store with transcripts, Jeffreys intervals, unbiased decided-by-step estimator, hierarchical bootstrap, token budget, CLI |
| Validated | `results/stage0-validation-gates.txt`: 16 trunks, 64 branch points, 384 continuations, 36 fresh runs. True V inside the 95% interval at 64 of 64 states; decided-by-step 34% [2, 61] and 77% [53, 96] against true 25% and 80%; step-0 forks 42/96 and fresh runs 14/36 against a true pass rate of 50.4% |
| Smoke test | `results/smoke-claude-code-haiku.txt`: `claude-haiku-5-5` through `--policy claude-code` on the three hand-written tasks; 30 runs, all scored, 29 passed |
| Since then | the Terminal-Bench adapter, the adaptive schedule, a cap on states per trunk, resuming a trunk that a budget stopped |
| Medium tasks | `results/medium1-claude-code-haiku.txt`: 62 Terminal-Bench tasks, Haiku through `claude -p`, one trunk each |
| Decisive steps | `results/decisive1-claude-code-haiku.txt`: 127 Haiku trunks on the 26 tasks mixed at step 0, bisected to the step that decided each run; 25 failed runs (with medium1's) have a clear one, scored against three judge models and three rules. Review follow-ups (grader check, fresh-run control, reworded judges, null simulation): `results/review-followups-2026-11-28.txt`. `decisive2` (held-out continuations at every clear step) is running |
| Not yet | a run through the API policy; an explanation for fresh runs passing more often than forks |

The validation task keeps its state only in the memory of a daemon started during setup, so a fork that carried
the filesystem but not the process would fail every test.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest                      # unit tests; no platform or model needed

export SANDBOX_OSS_URL=...  SANDBOX_OSS_TOKEN=...    # the sandbox platform, read by its SDK
export ANTHROPIC_API_KEY=...                         # only for --policy anthropic

# the synthetic check: three gates opened correctly with probability 0.8, 0.7, 0.9
.venv/bin/forkeval run --tasks tasks/synthetic --policy gates --gates 0.8,0.7,0.9 \
    --experiment validate --trunks 16 -k 6 --fresh 36 --concurrency 3
.venv/bin/forkeval summary --experiment validate

# a model on the hand-written tasks, with a hard token cap
.venv/bin/forkeval run --tasks tasks/handwritten --model claude-haiku-5-5 \
    --experiment stage0 --trunks 1 -k 6 --fresh 6 --max-tokens 2000000

# the same without an API key: the model is reached through the Claude Code CLI on a Claude subscription
.venv/bin/forkeval run --tasks tasks/handwritten --policy claude-code --model claude-haiku-5-5 \
    --experiment stage0-cc --trunks 1 -k 6 --fresh 6 --max-tokens 2000000
```

## Benchmark tasks

`--format tbench` reads task directories in the Terminal-Bench layout from a local checkout of that benchmark
(nothing of it is copied here; its files ask not to be republished).

```bash
git clone --depth 1 https://github.com/laude-institute/terminal-bench ../terminal-bench
T=../terminal-bench/original-tasks

# are the tasks sound here? each must fail untouched and pass after its own reference solution. No model.
.venv/bin/forkeval check --tasks $T --format tbench --only fix-permissions,hello-world

.venv/bin/forkeval run --tasks $T --format tbench --only fix-permissions,hello-world \
    --policy claude-code --experiment pilot --trunks 2 -k 6 --schedule grid --max-tokens 3000000
```

There is no Docker involved. A task's Dockerfile is read as a base image and a list of steps; the steps are
applied once inside a sandbox started from the base image, and the result is saved as an image that every run
of the task starts from (`fe-<task>:<hash of the base and steps>`). That covers single-stage Dockerfiles made
of FROM, WORKDIR, ENV, RUN, COPY and ADD with one container per task: 195 of the benchmark's 241 tasks. The
rest are skipped with the reason printed (more than one container, multi-stage builds, USER, heredocs, an ENV
that extends the base image's own variables, compose options a sandbox does not have).

## Spending continuations where the outcome is open

A branch point is a suspended copy of the sandbox: it costs a fork (about 1.5 s) and no model calls. The
continuations are where the tokens go, and most of them are wasted on runs that were never in doubt. With
`--adaptive` every scheduled step is still snapshotted, but continuations are run in step order and stop after
the first state whose k continuations all end the way the trunk did; later states are recorded as skipped.

Replayed on the full grid of `pilot1` (8 tasks, 45 states, 266 continuations): the rule would have run 59
continuations (22%), and of the 35 states it skips, 2 were 5/6 and the rest unanimous (mean open variance
0.010, against 0.25 for a coin flip). That is the price: a skipped state is treated as settled when it may
only be nearly so.

Settling early does not help a long run that stays open. One 20-step trunk that sat near 50% the whole way
cost 370k tokens per continuation, about 44M tokens to measure at every step. `--max-states M` measures at most
M states of a trunk, evenly spaced from its first snapshot to its last (6 of 21 are steps 0, 4, 8, 12, 16, 20);
the rest are recorded as not measured, and the estimators treat such a state like the last one measured before
it, so a change in value is dated to the next measured step.

A budget stop used to throw away whatever was in flight. Now, when a limit is reached after a trunk has finished,
its unmeasured snapshots stay on the platform (`--keep-hours`, 24 by default) with their deadlines pushed out,
and the same command measures them later instead of starting the task again; `forkeval drop` deletes them. On
a subscription nothing new is started within 5 points of the 5-hour limit or 1 point of the weekly one, so the runs in progress finish inside
it. A trunk that is stopped before it finishes is still lost, and a new one is run in its place.

A suspended branch point still counts against a tenant's limit on concurrent sandboxes, and a 20-step trunk
holds 21 of them. `--max-sandboxes` (default 45) keeps a run inside that limit: each trunk holds room for itself
and its k continuations, and takes room for branch points as it goes. A step that finds none within two minutes
is recorded as not snapshotted, and the run goes on.

## The step that decided a run

A failed run has a point after which it was unlikely to succeed. Forks can find it: V is high in one state and low
in the next, so the action between them decided the run. The run's own end counts as one more point, so a run that
stopped while every fork from its last state went on to pass is decided by the turn that ended it.

`--bisect` measures the states that find that action, not all of them: the first and the last, then the state in the
middle of the biggest move towards the outcome, until the move is between neighbouring steps (about 2 + log2 T
states, capped by `--max-states`). `--confirm-k 8` gives the two states either side of it 8 continuations each,
since 6 a side can make a move out of noise, and the search goes on if the move shifts. Every snapshot is held until
then, so a trunk holds one sandbox per step. Step 0 is the same state for every trunk of a task and is measured once.

The search picks the biggest move it sees, so the continuations that found it overstate it (regression to the
mean). `--holdout-k 12` runs 12 more continuations at each side of every clear decisive step, recorded under the
policy name plus `#holdout` and never used by the search; `forkeval decisive --replicated` scores only the cases
whose drop the held-out continuations confirm. `forkeval null` replays the search on simulated value paths whose
truth is known: on runs where V falls evenly and no step decides anything, it still calls a step inside the run
clear 5–38% of the time (more the more V falls); a true drop from 0.9 to 0.1 is found at the right action 98% of the time, 0.8 to 0.2 68%.

`forkeval judge --model M` shows a judge model each failed run that has a clear decisive step (one action, V moves at
least 0.5 across it) with numbered steps and asks which step decided the failure. `--prompt-version 2 --samples 3`
asks two questions about this agent instead, where the mistake was made and after which step it was unlikely to
recover, three times each (scored by majority). `forkeval decisive` scores the answers against the forks, next to
baselines: the last step, the first failing command, the first file written, chance. Intervals resample tasks,
since one task gives several runs. `--exclude TASKS` leaves tasks out. For a run that ended on the agent's own
turn, it also splits the continuations from the last state: if a copy that also stopped at once passed, the
test, not the stop, may have decided the run.

`forkeval grader --tasks ... -k 8` checks that each task's test gives the same answer every time: it applies the
reference solution once, forks the result k times and scores every copy.

`--swap-model` continues some of the same states with a second model: step 0 (once per task), and the states just
before and just after each failed run's decisive step (`--swap-at after`: only the state just after).

```bash
.venv/bin/forkeval run ... --bisect --max-states 12 --confirm-k 8 --holdout-k 12
.venv/bin/forkeval judge --experiment E --model claude-haiku-5-5 --model claude-sonnet-5-5 --model claude-opus-5-5
.venv/bin/forkeval judge --experiment E --prompt-version 2 --samples 3 --model ...
.venv/bin/forkeval decisive --experiment E [--exclude TASKS] [--replicated]
.venv/bin/forkeval grader --tasks DIR --format tbench -k 8
.venv/bin/forkeval null
```

The decided-by-step estimator assumes the measured states were not chosen by looking at the data, which the
bisection does; use it on runs without `--bisect`.

### Results so far

`results/decisive1-claude-code-haiku.txt`, `results/review-followups-2026-11-28.txt`. Of 58 failed Haiku runs, 25
have a clear decisive step (13 tasks), 10 of them the turn that ended the run. Two corrections since the first
write-up:

- `gcc-compiler-optimization`'s test is not sound here: its reference solution fails 8 of 8 copies
  (`results/grader-2026-11-27.txt`; every other task passes 8/8). In two of its ending-turn cases, copies that also
  stopped at once passed, so the test decided, not the stop.
- Three of the 10 ending-turn cases are Haiku refusing `tree-directory-parser`.

| Who names the decisive step | all 25 (13 tasks) | without gcc and tree: 18 (11 tasks) |
|---|---|---|
| Sonnet judge | 17 = 68% [41, 92] | 13 = 72% [45, 94] |
| Opus judge | 16 = 64% [39, 88] | 12 = 67% [44, 92] |
| Haiku judge | 14 = 56% [33, 79] | 11 = 61% [33, 84] |
| any judge, asked where the mistake was made (v2, majority of 3) | 15–17 | 13 = 72% |
| first file written (else the last step) | 15 = 60% [36, 84] | 12 = 67% [41, 92] |
| the last step (= the ending turn) | 10 = 40% [13, 68] | 4 = 22% [5, 50] |
| the first failing command | 7 = 28% | 2 = 11% |
| a step at random | 26% expected | 19% expected |

95% intervals resample tasks. On the 18 sound cases the judges beat the last-step rule (Sonnet right where it is
wrong 10 times, the reverse once: +50 points [+8, +79]) but not the rule "the first file the agent wrote" (+6 [−23,
+35]). Asked after which step the agent was unlikely to recover, Sonnet still matches the forks 12/18; Haiku and
Opus almost always answer the last step (4/18). The search has a known false-positive rate on gradual runs (above),
and the held-out continuations of `decisive2` (running) are the check on that.

**Second model.** From Haiku's exact state just after its decisive step, Sonnet finished 14/66 (21%) and Haiku 9/112
(8%). Sonnet's passes all came from the 4 states where Haiku still recovered sometimes (14/24 there, 0/42 on the 7
where it never did). A fresh-run control on those 7 tasks changes the reading: through `claude -p`, Sonnet passes
21/42 fresh runs and Haiku 32/42, so Sonnet is not the stronger agent in this setup and its 0/42 says nothing about
a stronger model. The control also shows Haiku passing 76% of fresh runs on those tasks against 54% of its step-0
forks; why is open (`results/review-followups-2026-11-28.txt`).

## Two ways to reach the model

Both are kept, so a result can be reproduced with whichever access a reader has. They are different agents, and
a result names the one it came from (the `policy` column of every run).

| | `--policy anthropic` | `--policy claude-code` |
|---|---|---|
| Access | `ANTHROPIC_API_KEY`, billed per token | the `claude` CLI logged in to a Claude subscription |
| A step | one Messages API call in a continuing conversation | one fresh `claude -p` process that reads the whole transcript as text |
| Actions | native tool calls | one JSON action per call (structured output) |
| Reasoning | carried from step to step | not carried: each step starts from the visible transcript |
| Context besides ours | none | a few lines the CLI adds about its host (directory, OS, date) |
| Cost of a step | grows with the conversation, mostly cache reads | the same, plus a process start (about 1.5 s) |

The CLI is called with its system prompt replaced, its tools off and an empty working directory, so no project
instructions or memory reach the model. What it adds can still show: in the smoke test the model took to
running `python3 -I`, a habit from one of those added lines.

A subscription has usage windows (a 5-hour one and a weekly one) shared with everything else its owner does
with Claude. The CLI reports how full each is, and `--max-plan-use` (the 5-hour window) and `--max-week-use`
(the weekly one), both 0.6 by default, stop the experiment once a window reaches its share, the same way
`--max-tokens` does. `forkeval run` prints the latest figures when
it ends, and `--trunks`/`--fresh` are targets, so the same command picks up later where it stopped.

Needs `zillion-asb` 0.5.0 or later (the first release with `fork`).

## Layout

| Path | What |
|---|---|
| `src/forkeval/tasks.py` | A task: image, instruction, setup, test, limits. TOML files under `tasks/` |
| `src/forkeval/env.py` | The five operations the harness needs from a sandbox: run, write, fork, suspend, delete |
| `src/forkeval/policy.py` | What picks the next action from the messages: a Claude model through the API, the same through the Claude Code CLI, or the synthetic gates agent |
| `src/forkeval/runner.py` | One step (`advance`), a run to its end (`finish`), and trunks with branch points and continuations |
| `src/forkeval/tbench.py` | Tasks in the Terminal-Bench layout: Dockerfile to base image and build steps, tests, reference solution |
| `src/forkeval/store.py` | SQLite schema and transcripts |
| `src/forkeval/stats.py` | Jeffreys interval, unbiased `V(1-V)`, decided-by-step, bootstrap. No dependencies |
| `src/forkeval/decisive.py` | The decisive move along a trunk, which state the bisection measures next, what a judge is shown |
| `tasks/handwritten/` | Three tasks: a live service to fix, an in-memory queue to drain, a log to summarise (the on-disk control) |

## How a state is branched

```
after tool result t:
    bp = trunk.fork(1)     # exact copy of s_t; the trunk continues
    bp.suspend()           # a frozen branch point
later:
    conts = bp.fork(k)     # k running continuations; bp is untouched
```

The agent loop runs outside the sandbox and holds the conversation, so continuing a state is: copy the messages,
fork the VM, call the same policy. Continuations differ only through the model's own sampling.
