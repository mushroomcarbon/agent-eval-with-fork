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
| Decisive steps | `results/decisive1-claude-code-haiku.txt`: 86 Haiku trunks on the 26 tasks mixed at step 0, bisected to the step that decided each run; 17 failed runs (with medium1's) have a clear one, scored against three judge models; Sonnet continues Haiku's state just after the decisive step |
| In progress | more trunks on the tasks that fail, for more clear cases |
| Not yet | a run through the API policy |

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
states, capped by `--max-states`). `--confirm-k 12` gives the two states either side of it 12 continuations each,
since 6 a side can make a move out of noise, and the search goes on if the move shifts. Every snapshot is held until
then, so a trunk holds one sandbox per step. Step 0 is the same state for every trunk of a task and is measured once.

`forkeval judge --model M` shows a judge model each failed run that has a clear decisive step (one action, V moves at
least 0.5 across it) with numbered steps and asks which step decided the failure; `forkeval decisive` scores the
answers against the forks, next to baselines (the last step, the first failing command, chance). The two can
disagree for a good reason: a judge tends to name where a mistake was made, the forks name where it stopped being
recoverable.

`--swap-model` continues some of the same states with a second model: step 0 (once per task), and the states just
before and just after each failed run's decisive step (`--swap-at after`: only the state just after, which is the
question; Sonnet costs about 20 times as much as Haiku per continuation here). "After Haiku's decisive mistake, how often does Sonnet still
finish the task?" is a question only an exact copy of the state can ask.

```bash
.venv/bin/forkeval run ... --bisect --max-states 12 --confirm-k 8 --swap-model claude-sonnet-5-5 --swap-k 6 --swap-at after
.venv/bin/forkeval judge --experiment E --model claude-haiku-5-5 --model claude-sonnet-5-5 --model claude-opus-5-5
.venv/bin/forkeval decisive --experiment E
```

The decided-by-step estimator assumes the measured states were not chosen by looking at the data, which the
bisection does; use it on runs without `--bisect`.

First results (`results/decisive1-claude-code-haiku.txt`, interim): of 44 failed Haiku runs, 17 have a clear
decisive step, and 8 of those are the turn that ended the run: the agent stopped while forks from its last state
went on to pass. Judges reading the transcript name the same step as the forks in 12/17 (Opus), 11/17 (Sonnet) and
10/17 (Haiku) cases; "the last step" gets 8/17, "the first failing command" 4/17, chance about 29%. The intervals
are wide (Opus 47-88%), so judges beat chance and the first-failure rule, but not yet clearly the last-step rule.
All three judges miss the same 5 cases, and they miss in both directions: twice they name the step where a bug
went in while forks from after it still passed (the run was lost when the agent stopped without checking), and
three times they name a later step (the script that produced the final answer, or a later rewrite) while the forks
put the loss earlier, where the approach that failed was chosen. One of those three comes from medium1, whose states
had 6 continuations and no confirmation stage. From Haiku's exact state just after its decisive step, Sonnet still finishes 11/30 times
(37% [21, 54]) and Haiku 8/64 (12% [6, 22]); from just before it, Haiku finishes 79%. Those 30 Sonnet runs come
from 5 failed trunks.

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
