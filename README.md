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
agent whose true V is known in closed form. No model has been run through it yet.

| | |
|---|---|
| Built | task format, agent loop (model + `shell` + `write_file`), trunk / branch point / continuation runner, SQLite store with transcripts, Jeffreys intervals, unbiased decided-by-step estimator, hierarchical bootstrap, token budget, CLI |
| Validated | `results/stage0-validation-gates.txt`: 16 trunks, 64 branch points, 384 continuations, 36 fresh runs. True V inside the 95% interval at 64 of 64 states; decided-by-step 34% [2, 61] and 77% [53, 96] against true 25% and 80%; step-0 forks 42/96 and fresh runs 14/36 against a true pass rate of 50.4% |
| Not yet | a real model run (needs an API key), a benchmark adapter, the adaptive schedule, resuming an interrupted experiment, the report page |

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
```

Needs `zillion-asb` 0.5.0 or later (the first release with `fork`).

## Layout

| Path | What |
|---|---|
| `src/forkeval/tasks.py` | A task: image, instruction, setup, test, limits. TOML files under `tasks/` |
| `src/forkeval/env.py` | The five operations the harness needs from a sandbox: run, write, fork, suspend, delete |
| `src/forkeval/policy.py` | What picks the next action from the messages: a Claude model, or the synthetic gates agent |
| `src/forkeval/runner.py` | One step (`advance`), a run to its end (`finish`), and trunks with branch points and continuations |
| `src/forkeval/store.py` | SQLite schema and transcripts |
| `src/forkeval/stats.py` | Jeffreys interval, unbiased `V(1-V)`, decided-by-step, bootstrap. No dependencies |
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
