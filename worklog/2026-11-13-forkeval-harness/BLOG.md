# Where do AI agent runs go wrong? Copy the computer mid-run and find out

*2026-11-13 to 2026-11-24. Technical detail, commits and verification: [TECHNICAL.md](TECHNICAL.md).*

**TL;DR.** We built forkeval, a tool for testing AI agents. It answers a question ordinary benchmarks can't: at what
moment did this run become a success or a failure? It works by freezing an agent's computer partway through a task,
making exact copies of it, and letting the agent finish from each copy many times.

- On 62 programming tasks, about half of the uncertainty is gone by the agent's fourth action.
- Many failures come down to one action we can point at.
- Our first look suggests the commonest of those is the agent declaring "done" while the copies show it was one step
  from success.

## The question

An AI agent here is a language model that works on a task by running commands on a computer, one at a time: read a
file, run a test, fix a bug, run it again. A benchmark gives it a task, waits, and checks the result: pass or fail.
That gives you a score, but it says nothing about why or when a run went wrong.

Suppose you could stop the run after step 5 and ask: "from exactly here, what are the chances this ends well?" Ask
again after step 6. If the chance drops from 80% to 10% across step 6, then step 6 is where the run was lost.

The catch is that "exactly here" means the agent's whole situation: everything said so far, plus every file on the
computer, every program left running, everything in memory. Re-running the same commands from scratch doesn't
reproduce that. Programs started in the background, timing and randomness all differ.

## What made it possible

Our sandbox platform runs each agent in its own small virtual machine (a sandbox). On 2026-11-12 it gained a **fork**:
make N exact running copies of a sandbox, memory and running programs included, in about two seconds. forkeval is
built on that one feature.

- After each of the agent's actions, the tool takes a copy of the sandbox and freezes it. That is cheap: no AI calls.
- Later it makes six copies of a frozen state, lets the agent finish the task from each copy, and checks how many pass.
- 4 of 6 passing means "from here, about a 67% chance", with an honest margin of error.

The only difference between the copies is the AI model's own randomness.

## Making sure it measures what it claims

Before trusting it on real agents, we tested it with a fake agent whose odds we knew exactly. The fake agent opens
three "gates" with fixed chances of getting each one right. Its progress lives only in the memory of a background
program, so a copy that lost running programs would fail every time. Across 64 frozen states, the true chance fell
inside the tool's margin of error all 64 times.

## Running it for real

We used Terminal-Bench, a public set of computer tasks such as fixing a broken build, recovering a database or writing
a solver. The model was Claude Haiku, reached through the Claude Code command-line tool on a regular Claude
subscription. The subscription has usage limits, every five hours and every week, so the tool reads how much is left
and stops itself before running out. It also leaves its frozen copies on the platform so it can carry on after the
limit resets. One run of 62 medium tasks took four days and eight usage windows, unattended, and cost about $24 at
list prices.

What it found:

- **45 of 62 runs passed.**
- **Some of the outcome is about the task, some about the run.** About 56% of the variation in outcomes is which task
  it was (some are just hard); the other 44% is down to how that particular run went.
- **Of that 44%, about half is settled by the fourth action and three quarters by the seventh.** A typical run is
  about seven actions long, so the outcome forms gradually. We had half expected the first move to decide everything,
  and it doesn't.
- **Runs can surprise you.** One run failed even though copies taken near its end passed six times out of six.
  Another passed after a point where copies passed zero times out of six.

## Is this new, and is it useful?

We looked at this critically. The basic idea, estimating your chances by finishing from the middle many times, is old.
Researchers use it to train AI models that score individual steps, and to search over agent actions. Some companies
also sell virtual machine copying for AI agents. What's new here is doing it exactly and cheaply on realistic
computer tasks, with careful statistics. The headline number above ("decided gradually") is honest but modest.

So we picked two follow-up questions that only an exact copy can answer.

**1. Can an AI judge find the decisive mistake by reading the transcript?** A common way to debug agents today is to
ask a stronger model to read a failed run and say where it went wrong. Forking gives an answer that doesn't rely on
anyone's reading: the action across which the chance of success fell off a cliff. We built a search that finds that
action cheaply. It measures the start and the end, then keeps halving the stretch where the chance moved most, so
measuring a 20-step run takes about six frozen states, not twenty. Then we compared the result with what AI judges
say.

On the six clear cases in our existing data:

- the two strongest judges named the same step as the copies four times out of six;
- "always blame the last step" also got three out of six.

The disagreements were the interesting part. In one task, every judge blamed the step where the agent wrote an
inefficient solver. But copies taken after that step still passed six out of six: the agent could easily have caught
and fixed it. The run was lost only when the agent declared itself finished without checking. Judges tend to name
where a bug went in; the copies name where it stopped being fixable. Three of the six failures were exactly this:
quitting while the copies show success was one more step away.

**2. Can a stronger model rescue a weaker model's mistake?** Take the exact computer state just after Claude Haiku's
decisive mistake and hand it to Claude Sonnet. Does Sonnet still finish the task? No ordinary benchmark can ask this,
because it can't recreate that state.

## How it went

- **Long runs were expensive.** One task sat near a coin flip for 20 actions, and measuring it at every step would have
  cost about 44 million tokens. We capped how many states we measure per run, and added a rule that stops measuring
  once the outcome looks settled.
- **Usage limits cut runs off and threw work away.** The tool now keeps its frozen copies when a limit hits and picks
  them up after the reset. It also stops starting new work shortly before a limit.
- **Our first "slow down near the limit" rule was too cautious.** It stopped a whole window 13 minutes in, because it
  treated the weekly limit like the five-hour one.
- **Frozen copies count as running sandboxes** against the account's limit, so we had to raise that limit on our test
  account. That is a lesson for the product: if copying is meant for research and training workloads, frozen copies
  probably shouldn't count against the limit on running sandboxes.
- **Sonnet costs about 20 times as much as Haiku per run** on this setup. Our first plan for question 2 would have
  used up about 15 five-hour windows. We cut it to the one place only a copy can answer, the state just after the
  mistake, which costs a quarter to a third as much.

## Next

The full experiment for both questions is running now:

- 26 tasks where Haiku sometimes passes and sometimes fails, three runs each;
- the decisive step pinned down with twelve copies on either side;
- Sonnet handed the state just after each failure's decisive step.

After that we run the judges on every clear case, write up the agreement rate and the rescue rate with their margins
of error, and rebuild the report page around them.
