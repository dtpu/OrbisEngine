# Quality rubric and retry policy

Criteria for judging a reconstructed replay, and the caps on retrying a failed stage. The
[offline static-world review](quality-judging.md) is the enforced runner gate for world
verification; this rubric is the broader manual/automated standard every candidate is measured
against. "Perfect output" means a useful, recognizable replay meeting the checks below, not exact
recovery of unobserved geometry. Small localized voids or distortion can be acceptable when they do
not damage the main view or walking route. Label invented appearance and unobserved geometry.

## Criteria

| Criterion          | Pass target                                                                                            | Failure examples                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Room layout        | Major stairs, rooms, and objects occupy the right relative places.                                     | Different layout, misplaced stairs, or major objects displaced.                            |
| Scene coverage     | Looking and walking around works without large holes or stretched surfaces dominating important views. | Extensive voids or distortion obscure the main view or route.                              |
| Inpainting         | People are removed while surroundings remain intact.                                                   | Ghosts remain; stairs, ceilings, railings, or other surroundings disappear.                |
| People and objects | The right people/objects appear at the right approximate positions and times with coherent paths.      | Missing person/object, identity swap, or wrong path/timing.                                |
| Walkability        | Floor support and stair ascent/descent work; solid boundaries constrain walking.                       | Falling below floors, clipping into stairs, or walking through substantial solid barriers. |
| Appearance         | Materials and lighting resemble the source enough for scene recognition.                               | Unrecognizable scene or extensive invented appearance in observed regions.                 |

## Judging

- Define stage-specific measurable tolerances and critical checkpoints before testing a candidate.
  Record the rationale and units (body-heights for spatial comparisons), then hold thresholds fixed
  for its baseline/candidate comparison. Do not invent observed scores or silently relax criteria.
- Render batches from the video's own camera positions at matched timestamps and compare side by
  side with real frames; include beginning/middle/end, challenging motion/occlusion, and known
  failures. Add off-axis views and actual walking/collision checks. If camera correspondence is
  unavailable, mark that comparison blocked rather than claiming a matched render.
- Acceptance requires at least **75% passing samples per applicable criterion**, **every critical
  checkpoint passing**, and a **critical-failure veto**: missing people, identity swaps, materially
  wrong layout, destroyed structural surroundings, unsupported floors, or major holes on the intended
  route cannot be averaged away.
- Store a structured result with stage/clip, sample times/views, criterion verdicts, measurements,
  reasons, evidence references, model/prompt/version, parameters, attempt count, elapsed time, and
  cost. Applicable criteria require evidence; record any not-applicable rationale. Missing evidence,
  invalid judge output, or uncalibrated judgments block acceptance; unknown is not a pass.
- Any automated (LLM) judge is a triage signal, not proof of visual quality. Calibrate it against
  clear manual passes and failures before it advances anything, and report false passes/fails.
  Inspect the existing `vlm_judge`/`verify_world.py` integration before adding another service.

## Retries

- **Pass: advance. Fail: diagnose and retry only within the cap.** At most **three quality retries
  after the first execution per clip/stage** (four executions total), each with a changed,
  justified parameter or a new code hypothesis and recorded before/after evidence, timing, and cost.
  Persist attempts across resume and candidate renaming; never reset counters. A reviewing agent
  that wants a further attempt stops and asks the operator rather than deciding for itself; the
  orchestrator enforces this as `AGENT_RETRY_CAP` in `orchestrator/workflows/run.py`.
- A retry must move something the stage actually reads. Each stage declares its overridable
  defaults in its `parameter_schema`, with the value it would otherwise use and a sentence on
  what the knob does; the adapters read the schema, so an attempt's `task.json` shows the value
  the stage really ran with and the reviewing agent is told the same. Anything not declared
  cannot be set, and a stage refuses an attempt that names one. Tolerances and guards are
  deliberately not declared: when the only way past a stage is to loosen one, that is a question
  for the operator, not a retry.
- A review that writes nothing for five minutes is interrupted and resumed in the same session
  and asked what it was waiting on (`WANDER_REVIEW_IDLE`, `WANDER_REVIEW_NUDGES`). Being stuck is
  an answer: `human.ask` leaves the stage waiting for an operator who can resume it, with the
  attempt's outputs intact. Unanswered nudges end the review as stalled, which reaches the
  operator the same way, naming the transcript. A silent review is not a passed one.
- Rerunning a stage invalidates its downstream stages; rerun them.
- Transport failures use a separate, finite request/backoff limit that accounts for SDK retries and
  every possible charge. A transport retry does not grant another quality attempt. Paid launch
  wrappers must never repeat a submitted or ambiguous job.
- Marble stays at one generation per new source clip and zero replacements for existing worlds;
  quality failure does not authorize regeneration.
- An exhausted quality failure remains failed; missing resources or evidence are blocked. Preserve
  the best candidate for inspection without promoting it as passed. The overall time and spend
  ceilings can stop attempts sooner.
- Where a gate is automated, add focused checks for pass/advance, fail/retry, exhausted retries,
  malformed/missing evidence, critical vetoes, and resume/cost limits. `--no-gate` skips the human
  cleaned-frame stop only, not quality checks.
