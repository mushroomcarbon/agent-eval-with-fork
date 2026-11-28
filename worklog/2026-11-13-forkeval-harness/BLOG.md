# Where do AI agent runs go wrong? Copy the computer mid-run and find out

*2026-11-13 to 2026-11-28. Technical detail, commits and verification: [TECHNICAL.md](TECHNICAL.md).*

**TL;DR.** We built forkeval, a tool for testing AI agents. It answers a question ordinary benchmarks can't: at what
moment did this run become a success or a failure? It works by freezing an agent's computer partway through a task,
making exact copies of it, and letting the agent finish from each copy many times.

- On 62 programming tasks, about half of the uncertainty is gone by the agent's fourth action.
- Many failures come down to one action we can point at. Some of those are the agent declaring "done" while
  copies of its computer from just before went on to pass.
- AI judges reading the transcript point at the same action as the copies about two times in three. That beats
  "blame the last step" but not a simple rule, "blame the first file the agent wrote".
- A critical review and a control run found problems with our first write-up: a broken test, a missing
  baseline, and a comparison between models that didn't hold. The story below includes them.
- Aside: Claude Sonnet, the bigger and pricier model, did worse than Claude Haiku on these tasks in our setup.

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

We ran 127 more Haiku runs on the 26 tasks where it sometimes passes and sometimes fails. Together with the earlier
data that gives 58 failed runs, and 25 of them with one clear decisive action. Asked to find that action, Claude
Sonnet as a judge got it exactly right 17 times out of 25, Opus 16, Haiku 14. "Always blame the last step" got 10,
and guessing gets about 6.

That looked good. Then we asked a separate AI reviewer (Opus) to tear the result apart, and it found real problems:

- **One task's test was broken.** We checked every task by applying its official solution and running the test on
  eight copies. One task, a compiler-optimization exercise, failed all eight: in our sandboxes its test fails even
  with the official solution. Four of our "decisive" cases came from it, and in two of them, copies that stopped
  at the same point as the original sometimes passed. The test decided those runs, not the agent.
- **Three more cases were the agent refusing the task**, which is a decisive step of a kind, but not an interesting one.
- **We hadn't tried a smarter dumb rule.** "Blame the first file the agent wrote" gets 15 of 25, almost as many as
  the best judge.
- **The search flatters itself.** It hunts for the biggest drop it can see, so the drop it reports is partly luck.
  We simulated runs where the chance of success fades evenly, with no single decisive step, and the search still
  called a step inside the run "clear" up to a third of the time.

With the broken task and the refusals left out, 18 cases remain. The judges get 11 to 13 of them, clearly better
than "blame the last step" (4), and level with "blame the first file written" (12). So judges are a reasonable way
to find a decisive mistake, but on these runs they don't add much over a one-line rule. We also asked the judges a
better question, about this agent rather than an ideal one: "where was the mistake made?" and "after which step
could it no longer recover?". All three answer the first about as well as before. On the second, Haiku and Opus
mostly just name the last step.

The disagreements are still the interesting part. In one task, every judge blamed the step where the agent wrote an
inefficient solver. But copies taken after that step still passed six out of six: the agent could easily have caught
and fixed it. The run was lost only when the agent declared itself finished without checking. In a sudoku task it
went the other way: the agent picked a method for reading four blurry digits that could never get them right, and
the copies taken just after that step pass far less often, but every judge pointed two steps later, where the agent
wrote down its guess. Judges anchor on visible moments. The copies measure something else: when the run stopped
being rescuable.

To fix the self-flattery, a second experiment is running. For every clear decisive step it finds, it runs twelve
more copies on each side that the search never sees, and only keeps cases those copies confirm.

**2. Can a different model rescue a model's mistake?** Take the exact computer state just after Claude Haiku's
decisive mistake and hand it to Claude Sonnet. No ordinary benchmark can ask this, because it can't recreate that
state.

From those states Sonnet finished 14 times out of 66, Haiku 9 out of 112. Where Haiku's own copies never recovered,
Sonnet never did either: 0 out of 42. We first wrote this up as "a mistake that's fatal for the weaker model is
fatal for the stronger one too". The reviewer asked the obvious question: can Sonnet do these tasks at all? So we
ran both models six times from scratch on those seven tasks.

**An aside: the expensive model lost.** Sonnet passed 21 of 42 fresh runs. Haiku passed 32. Sonnet is the bigger
model, and at list prices each of its runs cost about nine times as much as Haiku's ($0.21 against $0.02). It lost
on four of the seven tasks, and never by crashing: every failure was Sonnet confidently announcing it was done. On
one task it wrote a short program, checked it by eye, and stopped after one or two actions, six times out of six,
failing every time. On a DNA task it found the tool the task called for wasn't installed, computed the answer with
its own formula instead, said so politely, and stopped. Haiku, cheaper and supposedly less capable, plods on for
more steps (8.4 against 6.3 on average), checks more, and passes. Our setup is unusual (the model is driven one
step at a time through the Claude Code command-line tool, with no view of the tests), so this says nothing about
Sonnet in general. It does say "use the smarter model" isn't a safe default for agent work: what matters is how
the model behaves in your loop, and you have to measure that.

So the 0 out of 42 isn't evidence about a stronger model rescuing a weaker one. What survives is narrower: from
the states the copies call lost, a different model didn't rescue them either; from damaged-but-not-dead states, it
sometimes did (14 of 24).

The same control turned up one more thing we can't explain yet. On those seven tasks Haiku passes 76% of runs
started from scratch, but only about 54% of the copies we start from the same initial state. Copies and the
original runs both go through freezing and restoring the computer; runs from scratch don't. Until we know why,
the absolute chances the copies report should be read as possibly a bit low. Comparisons between steps of the same
run are less affected, since both sides are measured the same way.

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
- **Sonnet costs about 9 to 20 times as much as Haiku per run** on this setup, and one batch of 42 Sonnet runs ate
  two-thirds of a five-hour usage window in under an hour. Our first plan for question 2 would have
  used up about 15 five-hour windows. We cut it to the one place only a copy can answer, the state just after the
  mistake, which costs a quarter to a third as much.
- **One task ate a third of the budget.** In a chess task, each copy ran about 20 more actions, and a single run's
  measurements took 14 million tokens. After two usage windows only 12 of 78 runs were done. We dropped that task
  and used fewer extra copies to confirm each decisive step; the remaining runs then cost a third as much each.

## Next

The second experiment, with held-out copies at every decisive step and no Sonnet, runs through next week on the
same subscription. After that: find out why runs from scratch pass more often than copies, and repeat a run through
the regular API instead of the command-line tool.
