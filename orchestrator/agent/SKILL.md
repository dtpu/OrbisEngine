# Running a Wander pipeline

You are the agent for one run. This file is how the job works and does not change; what has
happened on *this* run is in `BRIEF.md` beside it, and in `journal.jsonl`.

The run is this directory. Stages write into it and read each other's output out of it. There
is no artifact store and nothing is copied between steps: if you want to know whether something
worked, look at the files it wrote.

## The loop

1. `wander ready` — what can run now, and what is waiting on what.
2. Start something. `wander step <name>` returns immediately and the step runs on its own.
3. **Stop your turn.** You will be given another when the step finishes.
4. When you come back, look at what it produced before you trust it.

Ending your turn is the normal thing to do, not a failure. A step can take ten minutes on a
GPU, and sitting and watching it is an agent holding a session open to do nothing. You will be
woken when it lands, and again on a timer if nothing lands. Never poll a running step in a
loop, and never wrap a command in a timeout to babysit it — start it and let go.

## Commands

```sh
wander ready [--all]        # what can run now; --all also shows what is blocked, and on what
wander show <step>          # what a step is for, its flags and their current values, its history
wander step <step> [flags]  # start a step; returns at once
wander step <step> --wait   # run it here and block, for something you know is quick
                            # (--wait reads the same before or after the step name)
wander log <step> [--tail]  # what its last run actually said
wander note "..."           # write something down for the next session
wander ask "..."            # something only an operator can decide; then end your turn
wander finish <status> "..."# the run is over
wander status               # everything, at a glance
```

Everything else is an ordinary shell. Read files, decode frames, measure things, write scripts.
Only `wander` commands go into the run's history, so use `wander note` when you learn something
a later session should not have to learn again — a measurement, a dead end, a reason.

## Judging what came out

An exit code of zero means the command ran, not that the result is good. This pipeline's
characteristic failure is a stage that succeeds and produces the wrong thing: an inpainting
pass that takes a railing away with the person, a solve whose camera teleports, a track that
swaps identity between two people. Look at the output. Decode frames, compare them against the
source, measure rather than glance.

Say what you measured. "179 frames, no residual person, the railing survives in all of them" is
a finding; "looks good" is not. Quote body-heights rather than metres, because metres assume a
1.70 m subject. If you could not check something, say that instead of implying you did.

When something is wrong, prefer fixing the input over loosening the check. Each step's flags
are in `wander show`, at the values it is running at now — move one when you can name the thing
you saw and say why that value addresses it. A tolerance or a guard you want relaxed is a
question for an operator, not a flag to set.

## Spending

Steps marked as costing money charge every time they run, and a world generation is 1600
credits. Never submit a paid operation twice: if one has already been submitted, poll it. When
a paid step fails in a way that leaves you unsure whether it charged, stop and `wander ask` —
do not guess, and do not retry to find out.

## Getting stuck

Being stuck is an answer. `wander ask` puts the run in front of an operator with your question
and leaves everything as it is; that is a real outcome and better than a fifth attempt at
something that has failed four times the same way. Say what you tried and what you would need
to know.

Before you repeat anything, read the journal. A step that failed the same way twice is not
going to be fixed by the parameter you already changed.
