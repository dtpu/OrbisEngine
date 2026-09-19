# Tonight

The human rewrites this file before each overnight run.
The agent works through the backlog in order, then completes the final pipeline rerun below,
under [RULES.md](RULES.md). The unattended agent does not edit this file unless the human asks.
Progress goes in [LOG.md](LOG.md).

Branch: `overnight-main-1`

Objective: improve the general pipeline for supplied inputs while preserving or improving
previously processed clips. Review the actual videos, not only filenames, manifests, or test output.
No wall-clock deadline is specified: finish with the report below when the work is complete or
further progress is blocked by the authorized resources. Reserve time and budget for comparisons.

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
provider balance remaining, accounting for other users and jobs still running. Budget limits
take priority over completing the full clip inventory: report unfunded rows as blocked.
No additional spending or automatic top-ups are authorized.

Before executing, follow [the overnight runbook](RUNBOOK.md), record a starting
resource ledger, and verify credentials/caches without launching inference. Keep credentials
out of this file and all tracked logs.

## Backlog

Work in this order; use the measured baseline to choose specific implementation work.

1. **Inventory and review every input and its baseline.** Read the new videos supplied for this
   run and all existing source clips discoverable through S3 viewer snapshots, archive run records,
   and the export manifest, including clips outside the demo picker. Download and watch the actual
   inputs. Inspect existing reconstructions before changing code, record failure timestamps and
   matched source/baseline captures, and deduplicate identical source/trim aliases. Generated reels
   and alternative outputs are not new inputs. Recover unavailable records where possible.
   Done when every input has a baseline row or an explicit missing-input/baseline reason, and
   notable problems are ranked by severity, impact across clips, and likely cost to investigate.
2. **Investigate and improve the highest-impact failures with general fixes.** Follow the required
   investigation policy below. Prioritize regressions and severe shared problems; use the strong
   orchestrator to select hypotheses and bounded worker tasks. Preserve baseline assets and compare
   each fix on affected and unaffected inputs. Do not spend the entire allowance optimizing one clip.
   Done when each selected problem has a tested improvement or a documented investigation/blocker,
   with measured evidence and appropriate regression checks; disclose remaining known problems.
3. **Validate the final implementation across the complete inventory.** Complete the final pipeline
   rerun and review package below, including old S3 inputs and any new supplied ones. An old clip
   that still works must be checked for regressions; an old failure is also an opportunity to improve.
   Done when every row has a final result, source/before/after comparison where available, and
   a reproducible viewer/download path. Budget-blocked rows remain visible, never silently skipped.

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

After the code backlog is verified or explicitly blocked, run the updated pipeline for **every
existing source clip**, plus every new clip listed above. This is a final validation phase,
not an optional extra after unit tests.

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
- Create an easy-to-open comparison index with labeled source/before/after screenshots and short
  recorded playback/walk clips. Include every input, even when its row has only baseline evidence
  and a blocked reason. Use matched times/views, include visible failure cases, and preserve audio
  timing where sound exists. Do not present selected stills as proof of temporal quality.
- Keep the generated index and its media outside Git under
  `public/reviews/overnight-main-1/`, with relative asset links. Publish to private S3 when writers
  are idle and record the exact snapshot. After adopting that snapshot, teammates should be able
  to open `http://127.0.0.1:5399/reviews/overnight-main-1/index.html` in their local viewer. Validate
  that path and its assets; provide pinned `assets:pull` recovery instructions in the results too.
- Commit/push code and the text report on the working branch; publish media, run intermediates,
  and detailed evidence to S3. Finish with direct instructions to open the comparisons and test
  candidates, plus the exact branch/commit/snapshots. Verify access; report any publication failure.

## Out of scope

- Unrelated product rewrites or repository/history cleanup; changing provider accounts or raising
  budgets; regenerating existing Marble worlds; overwriting shipping assets before comparisons pass.
- Simulated XR does not establish Quest performance. Physical headset measurements require an
  available connected device; otherwise record that limitation and complete desktop comparisons.
