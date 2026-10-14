# Pipeline reliability validation

The pipeline can submit a clip, retain its source, run stages, recover from a failed stage and
package a viewer scene. Passing these checks does **not** establish fresh provider access or
acceptable reconstruction quality. The doctor reports those separately.

## What is tested

`uv run --locked --group dev pytest` includes:

- Production API, SQLite and filesystem storage: uploads, replay from source artifacts, hash
  verification, extensions, conflicts, failed-submission compensation and restart discovery.
- All five world modes with single/tracked people, objects and fine-tuning: API/agent plans
  use the runner's actual graph. Multi-image cleaning waits for measured cameras, and its
  review gate checks the cleaned selection before generation.
- An integration test that creates an MP4 with audio, uploads it, executes the real runner with
  simulated model providers, injects a failure, resumes, runs the real person packager and
  checks its frame hashes, completion journal and dashboard previews. Successful model stages
  are not rerun. Changed source bytes cannot launch a stage or pass completion.
- Concurrent state writers, paid-stage claims across run names, unresolved-operation blocking,
  retry identity requirements, retained legacy receipts and shared execution allowances.
- Setup diagnostics for missing tools, dependencies, selected credentials/caches, fine-tune
  hosts and discontinuous footage. Remote diagnostics never launch model work.

The admin's `bun test ./tests` checks schema-derived options and queue decisions. Its production
build checks the UI and proxy types. Browser checks should use the real API and existing footage,
not a screenshot-only mock.

The run-control regressions cover pause/cancel during stage admission, a detached child starting
after cancellation, stale agent completion, and journal replay. Canceled runs cannot reopen from
a late stage event; explicit operator resume is required. Already submitted provider work keeps
its completion records.

Source-timing tests compare emitted pixels with the exact decoded source frames, including
variable frame rates and duplicated samples. FFmpeg can split a diagnostic row when another
thread logs an EOF message; only a checksum continuation from the same filter instance completes
that row. Missing, displaced or conflicting records still fail before paid submission.

Placement and anchor regressions use separate run/output roots. The runner passes exact world,
camera, source-video, placement and tracking-mask paths; an invalid explicit path fails instead
of selecting a historical demo. The size gate evaluates the final silhouette fit, and the printed
viewer link enables a successful placement. These checks establish input routing and acceptance
behavior, not visual reconstruction quality.

When enabled, fine-tuning finishes before placement and anchors. Those checks, observation
confidence, static-world review and the viewer link select the same final world. Scale fitting
and fine-tune export/import use the original provider world. Failed or missing fine-tune output
cannot silently fall back to another file. Placement records the world and report hashes plus
an input receipt covering packaged cameras, person manifests and frames, tracking masks and
motion metadata, effective scale/floor, size-gate setting and measurement code. Changed or
missing inputs invalidate its viewer setting and rerun the local placement/anchor stages on
a full resume. Existing placements without these receipts need that local recheck too;
successful paid-stage records remain intact.
Anchor receipts additionally bind the source video, ruler settings/model, measurement code,
anchor report and observation sidecar. Selecting only `verify` cannot accept stale dependencies.
Inputs are compared before and after measurement so edits during execution cannot receive a
current acceptance receipt. Provider metadata follows the selected video, image or multi-image mode.

Before checking observation confidence, the runner rebuilds the sidecar from the current world,
cameras and fitted registration. Tracked runs exclude the source person masks; missing camera
coverage fails instead of silently using the avatar. Single-person runs retain the documented
projected-avatar estimate. Confidence measures visibility and is not proof of accurate geometry.

Metric-depth predictions are cached by source-content hash, model identifier and frame index.
Changing camera poses can reuse a prediction; replacing an upload with the same filename cannot.
Legacy filename-only caches are retained but ignored because their source is unverified.

Silhouette fitting uses the tracker's projected body bounds to flag potentially cropped head/foot
edges even when the segmentation mask stops short of the image border. Those bounds only mark
which edges are unavailable; they never replace source-mask measurements. A cropped person cannot
supply a complete-body height. Motion metadata maps source frames to the original mask samples,
including after sequence subsetting, and malformed mappings fail explicitly.
The ruler and visibility exporter use those same per-track mappings and union masks only for
matching source frames. Missing selected frames fail; legacy tracks without motion metadata
retain the documented manifest-order fallback. Silhouette scoring and detector seed boxes also
require exact source-frame poses. Diagnostic image shifts discard pixels outside the frame
instead of wrapping them to the opposite edge.
The silhouette solver validates each base, perturbed and final avatar projection. Insufficient
points in front of the camera, invalid rows/depth, or nonfinite equations/updates stop the fit
with the affected track and source frame identified, instead of producing a generic exception
or an accepted result from invalid measurements. Valid fits keep the same objective and weights.

## Existing demo replay, 2026-10-12

The private shared snapshot used for this check is
`f87c64c4-7113-43d7-acf9-182e59e65a1d`. The clip is the existing 6.233-second Tears of Steel shot
`/clips/upload-tos-hall-walk-shot00.mp4`, SHA-256
`e632671f7ded630d7260f1016131e3ed8d8bbfb7827d4790f65fccebfc07cca4`.
The retained inference outputs are under `runs/upload-tos-hall-walk/` in its author archive.

The check verified source/track identity, the camera-metadata hash, and all 75 retained animation
frame hashes before rebuilding with the current multiperson packager. The real viewer loaded
75/75 frames, the original 6.233-second video, and the existing 799,996-splat world. Playback
was inspected at the source camera at the start and at three seconds. Browser requests and
JavaScript reported no failures. The admin displayed the source, packaged reports and setup
blockers. This published source video is silent; its external original soundtrack was not part
of this replay. Audio preservation was checked separately with the integration test's audible
fixture. No real-demo audio or headset claim is made.

This was **retained-output replay, not fresh inference**. No paid model work was launched. The
old default Modal credentials were rejected with `AuthError`; another saved workspace profile
authenticated and exposed the model-cache volumes. Normal Hugging Face snapshot symlinks are
reported as unverified rather than missing because metadata alone cannot validate their targets.
The optional local inference environment has been installed separately from the base environment.
Fresh inference remains unverified and is held pending an operator-confirmed run budget.
Changing accounts or providers does not create a separate retry allowance.

The scene is not accepted as a quality pass. The animated person visibly floats above the floor
and disagrees with the source framing. The archive's scale-spread check failed; its old review
report contains rate-limit errors and `pass: false`. Those failures were not bypassed or relabeled
as successful. The replay run deliberately remains incomplete rather than fabricating successful
cleaning, person preparation or fresh reconstruction.

Private screenshots, browser reports, source hashes, download receipts and the replay script
are retained locally under `.context/evidence/pipeline-reliability/`. Media and credentials are
not committed. To repeat the check, use [shared asset recovery](shared-assets.md) to retrieve
the named source and retained run, verify their hashes, and rebuild into a new candidate directory.
Do not copy an old `state.json` wholesale or launch replacement worlds for existing demos.

### Cropped-track diagnosis, 2026-10-14

All 75 tracker records project the lower body outside the image, even though the segmentation
masks stop before its border. The measurement conservatively censors these edges and reports
no complete-body height. This is a tracker-derived classification, not proof that the footage
never shows the feet: at six seconds the boots are visible, with mask bottom near pixel 518
in a 534-pixel image while the projected body extends to about pixel 562. The disagreement
requires investigation of the tracked geometry; the resulting placement is still rejected.

A read-only comparison isolates the placement contribution. At source frames 0, 90 and 185,
the raw packaged person's silhouette IoU is 0.873, 0.816 and 0.718. With the crop-aware
placement it is 0.704, 0.644 and 0.551. Its 12-frame mean area ratio is 1.584, outside the
unchanged 0.85–1.15 target, and its contact residual falls below the noise floor. The live
viewer loads all 75 frames without request or script failures, but source comparisons remain
visibly oversized and world alignment is not accepted. Neither raw geometry nor placement
is promoted as a passing reconstruction. Evidence is retained under
`.context/evidence/production-readiness/placement-isolation.json`, `cropped-quality.json`
and `cropped-placement-browser/`.

A separately authorized local visible-width experiment improved the 12-frame silhouette IoU
from 0.631 to 0.799, but its area ratio of 1.171 still exceeds 1.15 and its contact residual remains
below the noise floor. Width depends on avatar shape and is not an independent ruler. The
prototype remains outside the production runner; this candidate is rejected. Evidence is in
`.context/evidence/production-readiness/width-quality.json` and `width-placement-browser/`.

## Remaining boundaries

- Fresh inference needs confirmed spending controls and compatible model caches. World generation
  additionally needs verified World Labs/OpenAI access; existing demo worlds must be reused.
- The directory ledger bounds paid attempts; it is not a cross-provider dollar meter. The API
  refuses custom monetary/run budgets that this runner cannot enforce.
- A run cannot promise arbitrary videos will reconstruct. Cuts, poor camera geometry, missing
  observations and failed placement/quality checks must remain visible stops.
- Successful stage caches are not yet keyed by every code/parameter input. Use a new candidate
  after changing those inputs while retaining the shared paid ledger.
- Automatic single-person object processing detects thrown objects; 3D object packaging needs
  tracked-person mode. Static-world review is still explicit and does not accept the entire scene.
