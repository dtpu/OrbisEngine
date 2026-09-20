# Auto clip: the operator playbook as code

`scripts/auto_clip.py` drives one upload from **video in → best segment → cleaned → cameras →
world → people → packaged → report**. It measures nothing and infers nothing itself: every step
shells out to a script that already exists and is already tested, and the whole point of the file
is that the judgement which used to live in a chat window is now an explicit, named, logged
decision. The OpenAI API is the judge at three places (segment, clean, people), and
[quality-rubric.md](quality-rubric.md) makes every one of those a triage opinion, never proof.

The runner is injected, so `scripts/test_auto_clip.py` drives the whole playbook with fakes and
spends nothing. `--dry-run` prints the exact argv of every command and every decision and starts no
subprocess at all. `--resume` reuses the decisions an earlier invocation recorded.

```sh
uv run --locked python scripts/auto_clip.py run --clip <video> --name <run> --dry-run
uv run --locked python scripts/auto_clip.py run --clip <video> --name <run> \
    [--mode best-shot|sequence] [--truth cuts.json] [--force-window 72 84 --reason "the entrance"] \
    [--judge-segment] [--operator-override REASON] [--people-cap 4] [--sequence-cap 4] \
    [--marble video|image|multi|both|none] [--separate-audio] [--resume]
uv run --locked python scripts/auto_clip.py reconcile --run .context/run/<name> --stage pi3x
uv run --locked python scripts/auto_clip.py preflight [--profile dtpu] [--check] [--json out.json]
uv run --locked python scripts/auto_clip.py judge-segment --request sheets/judge_request.json \
    --out .context/run/<name>/segment-judge.json
```

## The decision log

One file per clip: `.context/run/<name>/auto-log.json`, schema `wander.auto-clip/1`. It is
append-only across resumes.

| Key                 | What it holds                                                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `sourceSha256`      | the original bytes every claim, trim and report is keyed on                                                                |
| `decisions[]`       | `{id, step, rule, inputs, choice, because, evidence, resumed, at}` — the rule or judge opinion that fired and what it read  |
| `commands[]`        | `{decision, argv, cwd, returncode, log, dryRun}` — the exact command, not a description of it                               |
| `costs`             | read from the shared ledger: per logical stage, executions used, executions remaining, `estimatedComputeUSD` (an estimate)  |
| `report`            | the per-clip report below                                                                                                  |

The report states the segment chosen and the coverage left out, per-person seen/selected/
reconstructed/unresolved with reasons, stages passed/failed with measured reasons, what is invented
versus observed, and whether identity was verified. `placement.fitted` is false whenever the scale
gate fell back, and the note says so; the report never claims fitted placement it did not measure.

## The playbook

### 1. Admission

`ffprobe` for codec, pixel format and frame-rate continuity; the letterbox crop comes from
`select_segment.py`'s `video.suggestedCrop`, measured once and applied to the working segment. The
source SHA-256 and the exact source offsets of the window are recorded before anything reads it.

**Stream copy only when it is possible AND safe**: no crop filter, no variable frame rate to
normalise, and the window is the whole file. A copy trim can only cut at a keyframe, so an exact
offset is a re-encode; anything else would make the recorded offsets a lie.

### 2. Segment

`select_segment.py --top 5 --contact-sheet` chooses the window (`--truth` passes hand-labelled cut
boundaries through). Its measured rank 1 wins. Two things can change that, and only two:

- `--force-window START END --reason ...`, refused without a reason, recorded as an override.
- nothing else. With `--judge-segment` and `OPENAI_API_KEY` set, `auto_clip.py judge-segment` buys
  **one receipted opinion** through the selector's `judge_request.json` hook and
  `scripts/vlm_once.py`. It is recorded with `applied: false` and changes no ranking, because
  [segment-selection.md](segment-selection.md) keeps the ranking measured.

`--mode sequence` instead runs `shot_cuts.py`, takes the longest usable shots first up to
`--sequence-cap` (`SEQUENCE_SHOT_CAP`, 4), and merges the per-shot candidates back onto the original
clock with `package_shot_sequence.py`.

### 3. Clean

The mask mode is a measurement, not a preference:

> `people.personPixelFractionMedian` of the chosen window **> `CROWD_PERSON_FRACTION` (0.25)** →
> `--people-mask foreground`, so a stadium crowd stays as background.

That is the same idea and the same number `worker/stages/dense_pi3x.py` uses to switch its own
anchor mask mode; the constant is copied rather than imported (that module pulls in the GPU worker
stack) and `test_auto_clip.py` asserts the two never drift apart.

Then `judge_clean.py` (exit 0 pass / 10 retry / 11 fail). A retry applies its `actionFlags`
**once** (`CLEAN_JUDGE_RETRIES = 1`), merged over the base flags so the measured mask mode survives
unless the judge replaced it. Two mappings are added here:

- `non_human_subject_left` → `--moved-mask` (what is left behind moved with the person, and is not
  a body).
- a frame-filling subject the judge reports in **only** the late or early frames (its
  `smearFraction ≥ SUBJECT_FILL_FRACTION`, or the subject only partly removed) shortens the window
  to the other half, provided the half is still `≥ MIN_SPLIT_SECONDS`. The time dropped is recorded.

Executions never exceed `MAX_STAGE_EXECUTIONS = 3` per stage and window — the same cap
`stage_attempts.py` enforces. The ledger's own refusal ("Paid … was not submitted: …") is surfaced
as a **blocked** stage with the ledger's words, not as a crash.

### 4. Camera solve and the teleport guard

`run_clip.py --only pi3x,frame_align` runs the solve and its pose guard. `SOLVE_RETRIES = 1` retry,
chosen by what the failure says:

- **teleport at `t`** (`pose-guard.json`, `worst.timeSeconds`): split, keep the **longer** of
  `[start, t)` and `(t, end]`, and only if it is still `≥ MIN_SPLIT_SECONDS` (6.0 s, the selector's
  own `MIN_WINDOW_SECONDS`). The dropped span goes into the report.
- **"do not agree on the same scene / Select a shorter segment"**: take the selector's next-best
  window and try that once. The abandoned window is recorded as dropped coverage.

Anything else is a failure this playbook has no rule for, and it stops rather than paying again.

### 5. World

`world_prompt` → gate → **one** marble generation per segment, never a regeneration
(`quality-rubric.md`: one generation per new source clip, zero replacements). `--gate-pass` is
passed only when the clean judge **passed**, or when `--operator-override REASON` was supplied —
and an override is logged as an override, with `OPERATOR OVERRIDE, not a pass` in its reason. With
no pass and no override, nothing is spent.

### 6. People

`tracks` → `judge_people.py` → reconstruct the `selected` main/secondary tracks up to
`--people-cap` (`PEOPLE_CAP`, 4). The same subset reaches the identity audit, because
`run_clip.py --people N` passes `--only-tracks 0..N-1` to `identity_audit.py`.

- **the audit has no power** (`identity.json` `ok: false` with "…has no power…": its swap control
  was missed, which is what look-alike subjects in the same kit do to it) → reconstruct **ONE**
  subject and report `identityVerified: false`. A possible identity swap is never shipped as a pass.
- **prep refuses** ("never fully visible") or **the pose model refuses** ("No source person pose
  detected") → that person is `unresolved` with that exact reason. If every selected subject is
  refused the clip becomes **world-only**, which is a valid outcome and not a stop.

### 7. Reconcile

The recurring time sink: a paid claim left `pending`/`unknown` blocks every retry of its stage.
`auto_clip.py reconcile --run … --stage …` acts only on **provable** failure:

| Evidence                                                           | Verdict                    |
| ------------------------------------------------------------------ | -------------------------- |
| recovery manifest with `status` `failed` or `partial`              | `recovery_manifest`        |
| provider refusal in the saved log, no inference started            | `provider_refusal`         |
| worker pre-inference refusal ("Stage … before inference"), no start| `worker_precheck_refusal`  |
| a refusal, but the log also shows `estimatedComputeUSD` / a call ID| **ambiguous** — stop       |
| no refusal text and no terminal manifest                           | **ambiguous** — stop       |

On provable failure it renames the leftover output directory to `<dir>.failed-attemptN` (never
deletes it), reconciles the claim through `stage_attempts.py --reconcile … --status failed` with
the log or manifest as evidence and a factual reason, moves a stale `identity.json` aside and marks
that stage `stale` in `state.json` with the reason — then continues. On ambiguous evidence it
changes nothing, reconciles nothing and reports. An unknown claim is never reconciled without
evidence.

### 8. Workspace preflight

`auto_clip.py preflight` reads the worker sources — `worker/modal_clean_video.py`,
`worker/modal_multiperson.py`, `worker/modal_lhm.py`, `worker/modal_stage_weights.py` — and prints
the cache volumes, the model paths those workers check for **before** inference, and the exact
`modal_stage_weights.py --phase …` command for each phase. Parsing the sources rather than keeping
a list means a newly demanded weight shows up without anyone remembering to edit this file; a
quoted path counts as a demand only when the code around it checks for it or refuses without it,
which is what keeps the image-build paths out. `--check` issues the read-only `modal volume ls`
listings; with `--dry-run` it prints the checklist and contacts nothing.

### 9. Packaging and report

`scale_fit, place_fit, anchors, verify`, then the report. A failed scale gate keeps the run
**viewable** on the fallback scale and says exactly that; it is never reported as fitted.
`--separate-audio` is a documented follow-up that runs `scripts/separate_audio.py` on the working
segment, and it is off by default.

## Limits and follow-ups

- **`--people N` is a rank count, not a track list.** `judge_people`'s `selected` is ordered by
  measured on-screen prominence, while `run_clip.py` takes track ranks `0..N-1`. When the two
  orders differ the orchestrator records a `people-cap-order` note saying so and reconstructs the
  ranks; giving `run_clip.py` an explicit track list is the fix, and it is not done here.
- **Sequence mode plans and merges, it does not drive the per-shot runs.** `plan_shots` writes the
  shot list and `package_shot_sequence.py` merges it; each shot's own pipeline is still a separate
  `run --name <name>-shotNN`.
- **A dry run walks the happy path.** With no outputs to read it assumes the judge passes and the
  guard is clean, so the printed plan is the shortest route, not every branch.
- **HDR is not tone-mapped into the working segment.** The selector tone-maps for measurement only;
  what the clean worker does with a 10-bit HLG source is an open question, not a solved one.
- Judge verdicts here are `uncalibrated` triage. Nothing in the report is a viewer or headset check.
