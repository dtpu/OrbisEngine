# Tonight

The human rewrites this file before each overnight run.
The agent uses the priorities and time boxes below, then completes the final validation and report
under [RULES.md](RULES.md). The unattended agent does not edit this file unless the human asks.
Progress goes in [LOG.md](LOG.md).

Branch: `daniel/overnight`

Objective: make verified progress on **as many of the pipeline and frontend goals below as
possible within seven hours**, while preserving or improving previously processed clips. Review
the actual videos, not only filenames, manifests, or test output. Favor severe shared failures
and useful completed improvements over polishing one clip or claiming the entire backlog is done.

## Seven-hour execution window

The clock starts when the unattended run begins, not when this brief is edited. Record the UTC
start and deadline in `LOG.md`; resumes and context compaction retain that same deadline.

- **0:00–1:00:** preflight, inventory, source/baseline visual review, flaw list, and pipeline trace.
- **1:00–4:30:** bounded implementation and comparisons, prioritizing pipeline quality and the
  judge. Independent frontend work may proceed in parallel under `AGENTS.md`.
- **4:30–6:30:** freeze feature work; run final validation and real-viewer comparisons. Begin
  long reruns earlier when needed. A late fix requires fresh evidence for affected stages.
- **6:30–7:00:** preserve/publish results, finish the report, push the working branch, and stop
  jobs started by this run. Do not start work that cannot finish within the remaining allowance.

These are planning checkpoints, not a reason to wait. Reallocate implementation time based on
measured costs, while reserving validation and handoff time. At seven hours stop; report unfinished
goals and clips as blocked by time, with partial evidence and next steps. Finish earlier only if
all useful authorized work is complete or blocked. The deadline and spending ceilings override
the full-inventory rerun requirement; missing coverage must remain explicit.

## Resources for this run

- Modal: **$27 maximum additional spend**, using `MODAL_PROFILE=dtpu`.
- OpenAI API: **$393.47 maximum additional spend**. Aayan confirmed $611.53 already used
  out of $1,005. Refresh usage before starting; other activity on the same account reduces
  the available amount. Use the project key shared privately as `OPENAI_API_KEY` in `.env.author`.
- Marble: reuse existing worlds for all existing clips. For each explicitly listed new clip,
  allow at most one generation only after confirming sufficient existing account credits;
  log the intended spend for Aayan before submitting. Unknown or exhausted credits block that
  generation. Do not create/cycle trial accounts or obtain new credits automatically.
- Coding-agent tokens: no numeric cap supplied. Follow the task-based model policy in
  `AGENTS.md`; track host usage when available, keep handoffs concise, and reserve capacity
  for integration, validation, and the final report. API dollars and chat-plan usage are separate
  unless the agent host is explicitly billed to the same API project.

These are ceilings, not spending targets. Apply the smaller of the cap remaining and the
provider balance remaining, accounting for other users and jobs still running. Budget and time limits
take priority over completing the full clip inventory: report unfunded rows as blocked.
No additional spending or automatic top-ups are authorized.

Before executing, follow [the overnight runbook](RUNBOOK.md), record a starting
resource ledger, and verify credentials/caches without launching inference. Keep credentials
out of this file and all tracked logs.

## Backlog

Use this priority order and the measured baseline to select work. Keep every goal visible in the
final report, including deferred ones. Do not wait for an impossible or blocked item before doing
useful independent work; validation and reporting have reserved time regardless of backlog size.

1. **Inventory and review every input and its baseline.** Read the new videos supplied for this
   run and all existing source clips discoverable through S3 viewer snapshots, archive run records,
   and the export manifest, including clips outside the demo picker. Download and watch the actual
   inputs from beginning to end. Manually inspect existing reconstructions in the real viewer
   before changing code; automated scores alone are insufficient. Record every observed flaw with
   the clip name, exact timestamp/range, view or walk path, severity, expected versus observed
   behavior, suspected stage, and matched source/baseline evidence. Use these concrete records as
   worker handoffs. Deduplicate identical source/trim aliases. Generated reels
   and alternative outputs are not new inputs. Recover unavailable records where possible.
   Done when every input has a baseline row or an explicit missing-input/baseline reason, and
   notable problems are ranked by severity, impact across clips, and likely cost to investigate.
2. **Trace the pipeline step by step and improve quality.** Map the actual execution path from
   admission/cuts through camera/depth, detection/tracking, masks/inpainting, image-to-3D assets,
   video-to-pose, world generation/fill, placement, packaging, and viewer ingestion. Record each
   applicable step's inputs/outputs, quality checks, failures, timing, cost, and downstream effects;
   mark absent or separate stages honestly. Follow the investigation policy below and prioritize:
   - Filling negative space and improving scene coverage without misrepresenting invented geometry.
   - Stairs, floor support, railings, and object boundaries: reproduce clipping, falling through
     floors, and walking through solid barriers; test walking up and down stairs. Inspect `stairs2`
     and every other applicable input, without assuming a defect from the filename alone.
   - Image-to-3D quality: silhouettes, shape, proportions, materials, and placement against source.
   - Video-to-pose quality: missing people, identity continuity, jitter, occlusion, foot contact,
     timing, and motion. Respect the measured failed approaches in `docs/known-limits.md`.
   - Object detection and tracking: missing/spurious objects, masks, identities, paths, and timing.
   - Inpainting: remove people/ghosts while preserving stairs, ceilings, railings, and surroundings.

   Preserve baselines and measure selected general fixes on affected and unaffected inputs. Each
   selected problem needs a tested improvement or a documented investigation/blocker.

3. **Implement bounded LLM quality judging and stage decisions.** Use the rubric and retry contract
   below. Evaluate whether the existing OpenAI vision integration can support it; inspect and reuse
   current judge/verification code before adding another service. Compare judge verdicts with manual
   review, including clear failures. Aim for structured stage pass/fail, evidence, and controlled
   retries with different justified parameters. Integrate and test a useful bounded subset if full
   coverage cannot fit; clearly distinguish implemented gates from proposals and uncovered stages.
4. **Improve efficiency and the frontend.** Build on the existing viewer and asset pipeline:
   - Measure output sizes and test compression/packaging improvements, retaining source fidelity,
     original audio words/timing, and acceptable viewer quality.
   - Profile generation time by stage; improve caching, redundant work, or bounded concurrency
     where measurements justify it. Report speed/cost/quality tradeoffs on fixed inputs.
   - Verify pipeline artifacts ingest into a rendered 3D scene end to end, with useful loading,
     progress, empty, and failure states. Improve the HUD and playback/walk/source/audio controls.
   - Measure large-scene load time, bytes transferred, first usable frame, and playback performance;
     improve bottlenecks and rerun the same measurements.
   - Heavily QA small/mobile, ordinary desktop, large/wide, and projector-sized layouts; record
     viewport dimensions, resize behavior, readable HUD, controls, keyboard/pointer behavior, and
     loading/error states. Include a Fergus theatre presentation scenario; actual projector or
     headset performance requires hardware evidence, not a large browser viewport.
   - Build or improve a landing page with a clear path into the demo and accurate capability copy.
     Keep private media private and avoid unsupported quality claims.
5. **Advance lower-priority delivery and admission work if time remains.** Prepare domain-name
   options and a concrete hosting/deployment plan, including private asset access, expected cost,
   and launch steps. No domain or hosting target has been specified: do not purchase a domain or
   provision paid hosting; prepare reviewable configuration and document the remaining decisions.
   Public media publication is not authorized. Investigate seamless cut handling as an optional
   optimization: prefer explicit shot detection/segmentation and preserve source time/audio mappings;
   do not silently delete footage or invent continuous action across a cut.
6. **Validate the final implementation across the complete inventory.** Complete the final rerun
   and review package below within the reserved window, including old S3 inputs and every new
   clip below. Each row needs a final result, source/before/after comparison where available, and a
   reproducible viewer/download path. Time- or budget-blocked rows remain visible, never skipped.

## New clips for this run

### Kitchen cooking (`kitchen-cooking`)

- Source: Aayan's `IMG_5581.MOV`, preserved byte-for-byte as
  `/clips/overnight/kitchen-cooking.mov` in the private S3 viewer catalog.
- Full input: **50.73 seconds**, 1920×1080, approximately 60 fps, HEVC 10-bit HLG/BT.2020,
  with original audio retained. Size: **143,639,523 bytes**.
- SHA-256: `15084804fb92ee6bcf37aa9433452aa534c4ae7f8d43dac3b6dbaf29389d8598`.
- Coverage requested: the complete recording, **0–50.73 seconds**, with no trim selected yet.
  Watch the full source and run admission checks before choosing processing settings. Any
  segmentation or budget-limited excerpt must retain exact source offsets and explicitly account
  for the remaining footage; do not present a short excerpt as a completed full-clip result.
- Sampled frames show a moving camera around a person cooking, handling kitchen objects, and
  opening a refrigerator. Investigate person/object occlusion, hands and handled objects, camera
  motion, and temporal consistency as measured; these observations are not an admission verdict.
- This is a new input with **no prior reconstruction baseline**. Include source-versus-candidate
  evidence and a result row in the final comparison package. Preserve the original MOV; if a
  browser/inference MP4 is needed, document general HDR/color, frame-rate, and audio/timing
  conversion settings and keep its basename aligned with its candidate run name.
- Existing overnight budgets apply, including at most **one new Marble generation total for
  this input**, shared by any derived segments/candidates. No reconstruction has been launched
  as part of registering the clip.

Retrieve it using the pinned snapshot recorded below. The original media stays outside Git;
teammates do not need access to Aayan's local Downloads folder.

```sh
bun run assets:pull --out .context/overnight-kitchen-inputs \
  --snapshot viewer/snapshots/db3be2e3-a4b2-4251-b273-37818b4c5fbc.json \
  --path /clips/overnight/kitchen-cooking.mov
```

Recovered file: `.context/overnight-kitchen-inputs/clips/overnight/kitchen-cooking.mov`.
Upload verification checked the stored full-object SHA-256 and size, plus a byte-range read-back
using the teammate read-only credentials. The snapshot preserves all previously published assets.

### Local test clips (`test1`, `test2`)

| Name    | Source             | Requested trim                                    |
| ------- | ------------------ | ------------------------------------------------- |
| `test1` | `assets/test1.mov` | Full clip, subject to continuity/admission checks |
| `test2` | `assets/test2.mov` | Full clip, subject to continuity/admission checks |

Paths are relative to the repository root. The supplied originals live locally in gitignored
`assets/`; do not commit them or copy them into shared-storage publication trees.
Use these supplied files, not substitute footage. Record hashes, metadata, audio streams, and
continuity before choosing measured processing options. Preserve the originals. Any required
transcode or shot selection needs a recorded mapping to source times; a cut/admission failure is
not permission to silently trim. Use isolated candidate names with matching input basenames as
described in the runbook. Label comparisons **no prior baseline** unless a verified prior output
is recovered. A viewer preset alone is not a source clip.

## Quality rubric and bounded judge

"Perfect output" means a useful, recognizable replay meeting the checks below, not exact recovery
of unobserved geometry. Small localized voids or distortion can be acceptable when they do not
damage the main view or walking route. Label invented appearance and unobserved geometry.

| Criterion          | Pass target                                                                                            | Failure examples                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Room layout        | Major stairs, rooms, and objects occupy the right relative places.                                     | Different layout, misplaced stairs, or major objects displaced.                            |
| Scene coverage     | Looking and walking around works without large holes or stretched surfaces dominating important views. | Extensive voids or distortion obscure the main view or route.                              |
| Inpainting         | People are removed while surroundings remain intact.                                                   | Ghosts remain; stairs, ceilings, railings, or other surroundings disappear.                |
| People and objects | The right people/objects appear at the right approximate positions and times with coherent paths.      | Missing person/object, identity swap, or wrong path/timing.                                |
| Walkability        | Floor support and stair ascent/descent work; solid boundaries constrain walking.                       | Falling below floors, clipping into stairs, or walking through substantial solid barriers. |
| Appearance         | Materials and lighting resemble the source enough for scene recognition.                               | Unrecognizable scene or extensive invented appearance in observed regions.                 |

- Define stage-specific measurable tolerances and critical checkpoints before testing a candidate.
  Record the rationale and units (body-heights for spatial comparisons), then hold thresholds fixed
  for its baseline/candidate comparison. Do not invent observed scores or silently relax criteria.
- Render batches from the video's own camera positions at matched timestamps and compare side by
  side with real frames; include beginning/middle/end, challenging motion/occlusion, and known
  failures. Add off-axis views and actual walking/collision checks. If camera correspondence is
  unavailable, mark that comparison blocked rather than claiming a matched render.
- Start by evaluating the suggested **75% passing samples per applicable criterion**, with **every
  critical checkpoint passing** and a **critical-failure veto**. Missing people, identity swaps,
  materially wrong layout, destroyed structural surroundings, unsupported floors, or major holes
  on the intended route cannot be averaged away. Calibrate the proposal against manual judgments
  before enabling automated advancement; report false passes/fails and the chosen thresholds.
- Store a structured result with stage/clip, sample times/views, criterion verdicts, measurements,
  reasons, evidence references, model/prompt/version, parameters, attempt count, elapsed time, and
  available usage/cost. Applicable criteria require evidence; record any not-applicable rationale.
  Missing evidence, invalid judge output, or uncalibrated judgments block acceptance.
- **Pass: advance. Fail: diagnose and retry only within the cap.** Allow at most **two quality
  retries after the first execution per clip/stage** (three executions total), with changed,
  justified parameters or a new code hypothesis. Persist attempts across resume and candidate
  renaming; never reset counters to bypass the limit. Invalidate/rerun affected downstream stages.
- Set explicit finite request/backoff limits for transport failures, accounting for SDK retries
  and all possible charges. The overall time/spend ceilings can stop attempts sooner. Marble
  remains one generation per new source clip and zero replacements for existing worlds; quality
  failure does not authorize regeneration. Recover known operations instead of resubmitting.
- An exhausted quality failure remains failed; missing resources/evidence are blocked. Preserve
  the best candidate for inspection without promoting it as passed. Add focused checks for
  pass/advance, fail/retry, exhausted retries, malformed/missing evidence, critical vetoes, and
  resume/cost limits. `--no-gate` skips the human cleaned-frame stop, not quality checks.

## Investigate notable clip failures (required)

For both existing and newly supplied clips, a notable failure is an engineering task, not just
a failed inventory row. This applies during development and the final rerun. Examples include
moving water or other dynamic backgrounds, occlusion, lost people, bad depth/scale, or a view
that breaks apart. The example is not a claim that each phenomenon is recoverable from one video.

- Capture the failure in the real viewer against the source. Identify its trigger, severity,
  affected stage, and the class of inputs likely to share it; distinguish missing evidence from
  an implementation defect. Check `docs/known-limits.md` and previous experiments first.
- Have the strong orchestrator investigate and plan. Research relevant primary documentation
  or papers when the cause/approach is uncertain, and cite the useful findings in the log.
  Delegate bounded implementation and checks under the model policy in `AGENTS.md`.
- Form a concrete new hypothesis and a bounded experiment with an expected measurable result,
  estimated cost, and stopping condition. Spend reasonable effort on diagnosis before declaring
  a clip unsupported. A known failed approach needs new evidence or a materially different idea,
  not another unchanged run. Keep the run's time and resource ceilings in force.
- Prefer general solutions based on measured inputs, confidence checks, automatic stage selection,
  or principled fallbacks. Do not branch on clip names or add hand-tuned per-clip runtime constants
  to make one demonstration pass. Do not invent hidden geometry, motion, or dialogue and describe
  it as recovered evidence. Make degraded output or unsupported input clear to the viewer.
- Validate a proposed fix on the triggering clip, another relevant available clip when possible,
  and an unaffected baseline. Add a focused regression check where it can detect the failure.
  If only one suitable example exists, state that generalization remains unverified.
- Rerun affected final-validation stages after a fix. Record what improved and what did not.
  If a sound fix is not feasible within the inputs or remaining budget, preserve the investigation,
  evidence, attempted hypotheses, and next experiment; report the row as failed/blocked with the
  precise limitation. Never hide the failure or mark the clip passed because it was investigated.

## Final pipeline rerun (required)

At the validation checkpoint, run the updated pipeline for **every existing source clip**, plus
every new clip listed above, to the extent the remaining time and resources permit. Start earlier
when necessary; defer unfinished feature work to protect this phase. This is required validation,
not an optional extra after unit tests. Report each unrun clip as blocked with the specific ceiling
or missing prerequisite; do not exceed seven hours to finish the inventory.

- Use the complete inventory established above from S3 and supplied/local sources, including
  clips outside the five-scene picker. Deduplicate aliases for the same source and exact trim.
  Record missing source inputs as blocked rows; do not silently omit them.
- Record the code commit and baseline S3 snapshot/output for each existing clip before rerunning.
  Retain the recorded trim, frame rate, and processing options unless a change is intentional
  and logged. Use separate candidate run names/output paths so the demo and baseline remain usable.
- Actually rerun reconstruction and packaging with the final code. A successful exit that only
  resumes cached completed stages is not a rerun: prefer fresh candidate names/state and record
  which stages ran. `--force` does not invalidate downstream stages; if used, identify and rerun
  every affected descendant explicitly. Reuse downloaded weights and existing generated
  Marble worlds (`--reuse-world`); keep `--marble none` for clips using that lane. Do not buy
  replacement worlds for existing clips.
- Process specified new clips through the applicable pipeline, with at most one new Marble
  generation per new clip. Record any admission failure instead of forcing unsuitable footage
  through. Apply the unattended execution policy in `docs/overnight/RULES.md`.
- Open each completed candidate in the real viewer and compare against its source and baseline.
  Capture representative playback and walk views; check people/placement, objects where present,
  collisions, and original audio timing/availability. Log missing soundtrack as a limitation,
  not invented audio or evidence of spatial dialogue.
- Compare the same source timestamps and matched camera/walk paths before and after. Record
  the metrics appropriate to each failure (for example silhouette overlap/area ratio, floor
  contact, jitter, collision errors, or audio offset) and repeat the same measurement method.
  Label each change improved, unchanged, regressed, or unverified independently of pass/fail.
  New clips without an old output get a source-versus-candidate comparison and an explicit
  "no prior baseline" label; do not invent a before result.
- Preserve runs and evidence in private S3. Only promote a candidate to the normal demo paths
  after its comparison passes. Keep the previous snapshot available for recovery.
- If this phase triggers another code fix, rerun the affected clips/stages against that final
  commit and refresh their evidence. Do not report results from an older implementation as current.

Done when: every inventory row is **passed**, **failed**, or **blocked**, with its source/trim,
commit, rerun stages, output/S3 reference, viewer evidence, and a concrete failure or blocking
reason where needed. Failed or blocked rows mean the full regression has not passed; report
those explicitly in the handoff, alongside any improvements and remaining regressions.

## Final review package (required)

- Write `docs/overnight/RESULTS.md` with one row per inventory input: source/trim/hash, baseline
  and candidate references, final commit/options, stages actually rerun, outcome and change label,
  key measurements, viewer/comparison links, and reasons for any failure or block. Include totals,
  spend with estimates distinguished from confirmed usage, changes made, and remaining priorities.
- Include a goal-by-goal table for every backlog subgoal: completed, partially completed, or
  blocked/deferred, with evidence, implementation limits, and next action. Include the specific
  per-clip flaw inventory, pipeline trace, judge calibration/retry outcomes, compression/speed
  measurements, frontend viewport QA, and landing-page/hosting progress. Separate observed results
  from proposed designs. Record actual elapsed time and why work stopped.
- Create an easy-to-open comparison index with labeled source/before/after screenshots and short
  recorded playback/walk clips. Include every input, even when its row has only baseline evidence
  and a blocked reason. Use matched times/views, include visible failure cases, and preserve audio
  timing where sound exists. Do not present selected stills as proof of temporal quality.
- Keep the generated index and its media outside Git under
  `public/reviews/daniel-overnight/`, with relative asset links. Publish to private S3 when writers
  are idle and record the exact snapshot. After adopting that snapshot, teammates should be able
  to open `http://127.0.0.1:5399/reviews/daniel-overnight/index.html` in their local viewer. Validate
  that path and its assets; provide pinned `assets:pull` recovery instructions in the results too.
- Commit/push code and the text report on the working branch; publish media, run intermediates,
  and detailed evidence to S3. Finish with direct instructions to open the comparisons and test
  candidates, plus the exact branch/commit/snapshots. Verify access; report any publication failure.

## Out of scope

- Unrelated product rewrites or repository/history cleanup; changing provider accounts or raising
  budgets; regenerating existing Marble worlds; overwriting shipping assets before comparisons pass.
- Simulated XR does not establish Quest performance. Physical headset measurements require an
  available connected device; otherwise record that limitation and complete desktop comparisons.
