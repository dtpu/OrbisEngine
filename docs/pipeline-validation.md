# Pipeline reliability validation

The pipeline can submit a clip, retain its source, run stages, recover from a failed stage and
package a viewer scene. Passing these checks does **not** establish fresh provider access or
acceptable reconstruction quality. The doctor reports those separately.

## What is tested

`uv run --locked --group dev --group inference pytest` includes:

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

The suite needs the locked inference dependencies for local tensor-based observation and
packaging checks. CI installs that group explicitly; these synthetic checks do not download
model weights or submit provider jobs.

The admin's `bun test ./tests` checks schema-derived options and queue decisions. Its production
build checks the UI and proxy types. Browser checks should use the real API and existing footage,
not a screenshot-only mock.

Packaged API runs now have an authenticated, read-only scene preview route. The admin link and
isolated runner summary use that route instead of requesting files from the unrelated shared
public directory. The local Vite proxy supplies credentials server-side and streams source byte
ranges; tests cover authentication, path confinement, selected-world routing and all cast members.
The packaging integration checks delivered frame bytes against their hashes. Optional compact
motion remains an acceleration with the existing PLY fallback, not a prerequisite for preview.
The preview defaults to the packaged camera-0 origin and explicitly leaves placement disabled;
that navigation default does not establish source/world camera correspondence.

Recovered packages can preserve reviewed presentation settings in
`public/worlds/<run-name>-4d/scene.json`, using `wander.viewer-scene/1` with a string-valued
`params` object. Supported settings are `pos`, `floor`, `scale` or `height` (not both),
`feetmode`, `feetlock`, `camdrift`, `fovsrc`, `clamp`, `box`, and `cam`. These defaults override
inferred presentation values; the viewer still honors explicit URL overrides. Asset paths,
scene identity, and `place` cannot be overridden by this file. Invalid settings hide the preview
instead of silently displaying a differently aligned scene. Required exported files are still
checked, and presentation metadata never marks reconstruction quality accepted.

A retained bottle-scene delivery check restored 240 person PLYs byte-for-byte against their
saved hashes and preserved the world, original soundtrack, bottle and backpack models.
The run-scoped viewer matched the existing demo's rendered people and object transforms at
12 timestamps, including all three throws. Original audio playback, video byte ranges,
end-of-clip pose holding, and visible navigation boundaries passed. This checks retained
asset delivery and playback; the run stays paused and does not claim fresh reconstruction
or acceptance of the existing hand/contact geometry.

A subsequent check delivered the same bottle scene through
`wander step --wait package_people --retained-scene CONFIG`, replacing the one-off delivery
helper. It used a fresh isolated API run, 277 verified saved input files, and zero provider
calls. All 240 packaged person frames, the room, original audio, prop models, and presentation
matched the prior delivery. The viewer matched rendered people and object transforms at
12 timestamps, including all three throws; audio playback, tail holding, looping, and
unrestricted exploration passed. Original placement/collision diagnostic reports were excluded,
and no placement or quality acceptance was imported. This proves retained packaging and
playback, not fresh reconstruction, walking support, or improved hand/contact geometry.
The run remains paused. Evidence: `.context/evidence/retained-scene-delivery/final/`.
See [retained scene delivery](paid-recovery.md#deliver-a-retained-tracked-scene-locally).

The bottle scene was also uploaded through the real admin form and delivered by the local
supervisor in a new run. While paused, the supervisor launched no work; after operator resume,
a fixed command harness invoked exactly one retained packaging stage and stopped for review.
Its explicit Python and disabled Modal executables reached the child process. The 277 saved
inputs matched their hashes, the paid-attempt ledger stayed unchanged, and no provider command
ran. Both people and objects matched the prior scene at 12 timestamps; original audio,
end-of-clip holding, looping, and unrestricted exploration passed. This proves upload, local
worker control, retained packaging and playback together; it does not test an autonomous
agent's decisions or fresh inference. Evidence: `.context/evidence/bottle-worker-rehearsal/`.

Dashboard summaries and progress now read the same dynamic stage plan from the run directory.
Saved-input provenance is displayed separately from actual attempt history; presenting a reused
stage as complete never creates an inference attempt or a quality decision.

A local dashboard handoff check asked a question before any agent transcript existed and sent
the answer through the Codex reply box. The run and stage cleared “needs you” automatically;
the stage returned to pending without an attempt or quality decision. The run was then paused.
The paid-attempt ledger was unchanged. Regression coverage preserves successful/failed stage
results, independent review holds, pause/cancel controls, and questions outside the original plan.
Questions now hold the journal without overwriting stage results. Older overwritten nodes are
normalized only in API responses when their event history proves the hold came from a question;
ambiguous or missing history remains held. This verifies the operator reply path, not autonomous
agent execution or fresh inference. Evidence: `.context/evidence/review-answer-handoff/`.

A subsequent local check used the real Codex CLI worker with the retained bottle source.
The agent verified the source/config hashes and asked a question; a reply through the admin
woke the named supervisor watch and resumed the same saved session. The agent recorded the
reply token in 9.1 seconds with the quiet retry timer set to one hour, rechecked both hashes,
and asked for final review. The diagnostic was then paused and its supervisor stopped.
No stage attempt, provider command, or quality acceptance occurred; the paid ledger and
retained hashes were unchanged. This proves the live question/reply/resume path, not fresh
reconstruction. The check used an isolated official CLI 0.163.0 with the existing ChatGPT
sign-in; inherited unrelated MCP sign-in and duplicate-hook warnings did not prevent either
turn. The broken system CLI was then updated to that version and its exec/resume startup
checks passed. Evidence: `.context/evidence/worker-reply-resume/`.

Regression tests cover questions arriving before stage/agent launch, blocked watchdog resumes,
retained transcript/session evidence, and answers arriving between a turn's final journal read
and the supervisor's next pass. Quiet turns do not repeatedly wake on their own status notes.

A real admin-to-viewer check reused the retained six-second scene in an isolated paused run.
The preview loaded all 75 samples, sought to frames 0, 36 and 74, and advanced playback with the
source video. Delivered world and compact-motion hashes matched the retained files. No requests
or uncaught scripts failed; the silent source remained labeled as having no soundtrack. This was
read-only delivery validation, with no inference or placement fit. The avatar and room alignment
remain unaccepted. Evidence is under `.context/evidence/production-readiness/run-preview-delivery/`.

The run-control regressions cover pause/cancel during stage admission, a detached child starting
after cancellation, stale agent completion, and journal replay. Canceled runs cannot reopen from
a late stage event; explicit operator resume is required. Already submitted provider work keeps
its completion records.

Rerunning a stage now invalidates its transitive downstream results before execution. A full
resume rebuilds selected dependents; `--only` leaves excluded dependents pending for a later
resume. This prevents repackaging from deleting object files while their stage still reports
success. Unchanged successful stages and genuinely vacant person slots remain reusable, and
paid outputs and recovery receipts are retained. This is dependency invalidation, not a complete
code/parameter fingerprint for every stage.

Run completion also requires the same packaged files as the dashboard viewer link, including
the selected final world, audio metadata and exported source video. The exported source must
match the submitted SHA-256; malformed base audio metadata or a conflicting source identity
cannot receive a successful completion record. Integration coverage removes and restores an
actual packaged audio manifest to check viewer availability and completion agree. These delivery
checks do not certify spatial audio, reconstruction fidelity or new provider execution.

Source-timing tests compare emitted pixels with the exact decoded source frames, including
variable frame rates and duplicated samples. FFmpeg can split a diagnostic row when another
thread logs an EOF message; only a checksum continuation from the same filter instance completes
that row. Missing, displaced or conflicting records still fail before paid submission.

Avatar input selection retains its front-facing preference when that produces a positive score.
When no front-facing score is positive, positive-quality candidates are ranked by projected-joint
coverage before the existing quality score, with the fallback recorded in prepared metadata. This
avoids preferring a larger cropped torso to a more complete back view. Explicit any-facing mode
retains its quality ranking. Projected coverage is a selection heuristic, not observed body coverage. An explicit source-frame request must exist in the track, and an empty track
fails before image preparation. These checks establish selection behavior, not avatar fidelity.

Preparation preserves the selected full-resolution segmentation as `observed-mask.png` before
dilation, alongside the padded `mask.png` used by LHM. Metadata binds both masks and the source
PNG by hash and records their source-pixel bounds, areas and dilation. The observed mask remains
a segmentation estimate, not ground truth; older outputs without it cannot recover that evidence
by reversing the padding. Model inputs and existing silhouette acceptance targets are unchanged.

Head conditioning uses the same selected-person mask for detection and the expanded head crop.
Previously the pinned [LHM head detector](https://github.com/aigc3d/LHM/blob/4f88aaeb3629249fbbddb4d0784a06962d9e1338/LHM/utils/face_detector.py)
received the full frame and selected its largest detected head, which could belong to a bystander.
Offline regressions reproduce that selection error, check exclusion of pixels outside the supplied
mask even in an expanded crop, and preserve the existing zero-head fallback for unusable detection.
Mask errors and dilation can still include unrelated pixels; this does not establish face fidelity
or fix the retained single-person body's projection mismatch. No new model execution is implied.

Placement and anchor regressions use separate run/output roots. The runner passes exact world,
camera, source-video, placement and tracking-mask paths; an invalid explicit path fails instead
of selecting a historical demo. The size gate evaluates the final silhouette fit, and the printed
viewer link enables a successful placement. These checks establish input routing and acceptance
behavior, not visual reconstruction quality.

The unfitted viewer link keeps the world in raw Marble asset units. World Labs documents
`metric_scale_factor` as metres per raw unit in its
[SPZ rendering guide](https://docs.worldlabs.ai/api/rendering-spz), so the assumed 1.7 m person
height is `1.7 / metric_scale_factor`, consistent with the raw floor at
`-ground_plane_offset / metric_scale_factor`. The earlier multiplication inverted that conversion;
its preview size error was the square of the factor. Tests cover factors below, at and above one,
legacy and nested metadata, and fitted-scale precedence. Correct units do not establish source
camera registration, actual subject stature or reconstruction quality. Previously captured
unfitted views retain their recorded parameters; this correction does not change their verdicts.

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
Reprojecting all 75 retained sets of 3D foot joints through their source intrinsics matches the
stored 2D predictions within 0.86 pixels. The padding/intrinsics conversion also matches the
pinned estimator's contract. This rules out a large post-model export shift; 2D and 3D predictions
are internally consistent but do not establish agreement with the visible boots. Animation fixes
canonical body shape while its motion-joint labels retain the original per-frame shape estimates,
so those labels are not independent measurements of the animated body's geometry either.

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

## Fresh source integration check, 2026-10-14

After explicit approval for four Modal calls within a $10 allowance, candidate
`tos-fresh-source-20261014-01` rebuilt cameras, tracking, canonical appearance and animation from
the same six-second source, reusing the existing world. The workspace usage limit was temporarily
lowered to $9 with separate headroom and restored to its previous $100 after all four owned apps
stopped. No paid retries, replacement world, placement fit, quality promotion or publication ran.
The provider calls used the reviewed implementation at `e60c543`; the local audio packaging
repair described below was discovered during the viewer check afterward.

The fresh track contains 75/75 distinct source samples. Camera continuity passed, and the camera,
source and canonical hashes bind the animation to its inputs. The finalized canonical and motion
checkpoints contain 13 and 83 verified files respectively. All 75 animation PLYs contain 40,000
finite splats; exported poses equal the 75 saved tracker seeds, without missing-pose recovery,
pose interpolation or duplicate output frame hashes. Local canonical tensor inspection could
not import a required model module, so that check remains unverified; the worker loaded the
checkpoint successfully. These are interface and internal-consistency checks, not pose accuracy.

The original fallback chose source frame 28 with the legs cropped. The general coverage ranking
selected frame 185, whose source and mask visibly include both boots. A native-camera Gaussian
render preserves the observed back-view clothing; hidden front/face appearance remains inferred.
All 75 tracked body bounds still project below the image. Their internal 2D/3D consistency does
not resolve that disagreement with visible source pixels.

The real viewer loaded the 799,996-splat world, all 75 fresh animation frames and the 6.233-second
source; seeking and playback advanced together without request or uncaught JavaScript failures.
The first check exposed missing audio metadata: Vite returned HTML for absent `audio.json`, and
the silent clip offered an enable-sound control. Both person packagers now probe the actual input
streams and emit source-audio metadata with the package timeline. This describes embedded audio
availability; it does not synthesize sound or certify soundtrack timing or speaker identity. The
repackaged scene reports no audio error and disables its “No soundtrack available” control; all
75 person frame hashes are unchanged. Conflicting authored audio manifests are refused before
either a direct packager or the runner changes the destination.

The candidate remains rejected for reconstruction quality. Read-only raw silhouette measurements
at 12 source samples give mean IoU 0.836 and mean area ratio 1.123, but only 7/12 samples lie within
0.85–1.15 and the final sample reaches 1.302. Person-only native-camera views confirm the late
silhouette/boot mismatch. Unfitted world views do not align with the source scene. No new placement
or contact acceptance was measured; the earlier failed placement remains failed. The source is
silent and no headset test was performed.

Private receipts, numeric audits, raw measurements and viewer screenshots are retained under
`.context/evidence/production-readiness/tos-fresh-source-20261014-01/`. Generated media and provider
credentials are not committed. The shared ledger now exhausts the ordinary camera and animation
allowances for this original source; candidate renaming does not reset them.

## Remaining boundaries

- The bounded fresh Modal check above verified its selected workspace and models. New inference
  still requires an applicable budget and remaining shared attempt allowance. World generation
  additionally needs verified World Labs/OpenAI access; existing demo worlds must be reused.
- The directory ledger bounds paid attempts; it is not a cross-provider dollar meter. The API
  refuses custom monetary/run budgets that this runner cannot enforce.
- A run cannot promise arbitrary videos will reconstruct. Cuts, poor camera geometry, missing
  observations and failed placement/quality checks must remain visible stops.
- Successful stage caches are not yet keyed by every code/parameter input. Use a new candidate
  after changing those inputs while retaining the shared paid ledger.
- Automatic single-person object processing detects thrown objects; 3D object packaging needs
  tracked-person mode. Static-world review is still explicit and does not accept the entire scene.
