# Offline static-world review

The runner accepts a world verification stage only after importing an explicit manual review of
the current source, world, camera, registration and image evidence. This is a narrow review of
sampled room layout, scene coverage and appearance. It does not approve an entire scene, geometric
alignment, people, handled objects, motion, collision, walkability, headset performance or unseen
surfaces. A human identity in a JSON document is an operator assertion, not authenticated sign-off.

There is no enabled LLM acceptance or autonomous repair workflow. Importing an `llm` review blocks,
including one claiming successful calibration. The older diagnostic median score and corrected
generation prompt no longer produce acceptance or regeneration recommendations. Other uses of
`worker/stages/vlm_judge.py`, including source-description generation, retain their existing behavior;
this change is not a general API-spending cap or transport replacement.

## Capture and review

The runner's default `verify` invocation captures CPU diagnostics with no model call, retains the
evidence in a fresh `verify/diagnostic-*` directory, then stops with quality status `blocked` until a
review is supplied. Diagnostic completion is not a visual pass. Direct capture is also available:

```sh
uv run --locked python scripts/verify_world.py \
  --clip /absolute/source.mp4 --world /absolute/world.spz \
  --cameras /absolute/cameras.json --scale0 1.0 \
  --out .context/evidence/example-review --no-vlm
```

Use the measured registration scale for the actual candidate. `--no-vlm` is a compatibility flag:
capture is always offline. `--model` and `--pass-score` are rejected. Choose a new output directory;
existing capture files are never overwritten. An explicit `--fit-scale` remains a diagnostic sweep,
not proof of registration; its resulting scale must agree with the scale supplied for import.

Capture writes the exact source PNGs, rendered PNGs, labeled comparison pairs, a contact sheet,
`report.json`, `plan.json` and `review-template.json`. The template deliberately has no reviewer,
false scope acknowledgements and unknown judgments; it cannot pass. Retain the template and author
a separate review JSON only after an actual human inspects every planned pair. Do not fill it with
an invented human approval. Private captures and reviews stay outside Git.

The generated plan includes beginning, middle and end camera samples, all marked critical by
default. Its fixed criteria are `room_layout`, `scene_coverage` and `appearance`; the passing fraction
is fixed at 0.75 and cannot be lowered. Every planned sample requires every criterion. `unknown`
blocks, a critical failure vetoes every average, and any failure on a critical sample fails.
The result must bind the exact plan bytes and each sample's pair hash. The evidence report binds
the source/world/camera hashes, registration scale, capture/validation/render code, exact decoded
source times and all source/render/pair image bytes. Changes invalidate a prior review.

For each criterion, the reviewer supplies `status` (`pass`, `fail` or `unknown`), boolean
`criticalFailure`, and a concrete nonempty `reason`. Inspect the whole source image, including
foreground, both sides, background and structures through openings. Trace visible stairs,
railings, openings and large static objects into the render. Similar foreground colors do not
compensate for missing structure. Keep layout, coverage and appearance judgments independent;
complete but incorrectly arranged surfaces do not establish layout fidelity. Ignore deliberately
removed people for this static-world contract, without approving their later reconstruction.

The review must name its human `reviewer` and set each acknowledgement to true only after the
reviewer understands these limits:

```json
{
  "estimatedCameraCorrespondence": true,
  "unobservedGeometryNotVerified": true,
  "staticWorldOnly": true
}
```

Use the standalone read-only import to check the completed review:

```sh
uv run --locked python scripts/world_quality_review.py \
  --plan .context/evidence/example-review/plan.json \
  --result .context/evidence/example-review/review.json \
  --evidence-root .context/evidence/example-review \
  --clip /absolute/source.mp4 --world /absolute/world.spz \
  --cameras /absolute/cameras.json --scale0 1.0
```

Exit codes are 0 (`passed`), 3 (`failed`) and 4 (`blocked`). This command hashes local files and runs
bounded local FFprobe/FFmpeg source inspection. It never renders the world, calls a provider,
changes a ledger, launches upstream work, publishes assets or writes over review evidence.

Supply the same files to the original runner command with `--quality-plan`, `--quality-result`
and `--quality-evidence-root`. Include `verify` when selecting stages with `--only`. A supplied
review is imported without recapturing or overwriting its images. Cached verification is always
reassessed against current files; `--force verify` is not necessary. Upstream work invalidates a
prior verification status. Rejected or missing evidence cannot become an accepted verification
through a successful subprocess exit or cached stage status. Archival retention and publication
promotion remain separate decisions; this static-world pass is not complete-scene publication
approval.

## Timing and provenance limits

The diagnostic reads actual decoded PTS with FFprobe and selects the decoded source-frame ordinal
with FFmpeg. Each camera's time must match the corresponding source PTS relative to the first
decoded frame within one microsecond. Import probes the source again and compares the exact decoded
source PNG bytes. Nonmonotonic or missing PTS, approximate camera-time mismatches, missing samples,
unsafe evidence paths and changed bytes block. Different decoder/encoder versions can change PNG
bytes and require new evidence even when the images appear similar.

This catches inconsistent labels at this boundary; it does not prove that a camera matrix was
estimated from that exact source frame. New cleaning and multi-image selection also retain actual
decoded source ordinals/PTS and verify selected image hashes; see [the kitchen review](kitchen-review.md).
Camera correspondence and registration remain explicitly estimated and unverified. Existing
approximate reports are not automatically migrated into acceptance evidence. Hashes bind assertions and detect mutation; they
are not signatures or independent proof that a supplied render is authentic. A genuine reviewer
and trustworthy capture environment remain prerequisites.

## Deferred work and exact source lineage

This implementation adapts Austin Jian's offline plan/result and actual-input validation from
`52f211b0ed52db3a531d2bfb17db4372f478defa`. His fail-closed runner behavior originated in
`19b560eb38fba2bcf71ec49ccb82fff4904ab1f1`; the evidence-resume guard from
`b42cd86d1c1afb9d5759358c70f510e849348492` is strengthened here to reassess every cached review.
Daniel Pu's `f95daa0f19f6aad2f4e010f3a9b85b9a887e151c` and
`60e61f7942bd6131251dfd0fb0ae47908f9798a2` informed separate criterion judgments, critical-failure
vetoes, unknown blocking and the decision to disable uncalibrated LLM acceptance. Only the
full-structure inspection guidance from his `614a91df25ef173d8ab2ca6597ff89f5a2a9dfe9` quality-rubric
hunk is used here; none of that commit's motion-packaging/viewer changes are included. The new
manual schemas deliberately differ from both branches' broader acceptance schemas.

Neither branch's `quality_attempts.py` or quality-ledger implementation is integrated. Both used
`wander.quality-attempts/1`, but Daniel's `attempts` list reserves API requests and preserves pending
executions, while Austin's dictionary records offline-assessment histories and policy. Existing
files in either format are not read, rewritten, reset, migrated or counted by this slice. Main's
separate `wander.pipeline-attempts/1` Modal ledger retains its existing scope and allowance. There
is no new offline-assessment cap and no claim that provider requests or future repairs are bounded
by this importer.

Before enabling model acceptance, the next patches must establish a distinct ledger version with
explicit lossless migration of both prior formats, canonical source identity and preserved unknown
or pending reservations; reconcile durable bounded request accounting and transport; bind frozen
human criterion/critical-flag labels and independent positive/negative holdouts to image, prompt,
model and request identities; and demonstrate agreement on those held-out cases. Daniel's archived
six live calibration requests did not establish a valid candidate, Austin's harness evidence was
offline, and this integration's kitchen request returned 429 without a verdict. No new paid call is
authorized or required by this code.

Cleaning and selection now propagate actual source indices/PTS. Camera estimation still needs
equivalent provenance, followed by held-out registration and off-axis/temporal review. Only a subsequent explicitly
bounded design should consider changed-hypothesis repairs or whole-scene promotion. Passing the
synthetic offline tests proves contract behavior, not live judge accuracy or scene quality.
