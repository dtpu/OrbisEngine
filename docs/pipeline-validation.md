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

Placement and anchor regressions use separate run/output roots. The runner passes exact world,
camera, source-video, placement and tracking-mask paths; an invalid explicit path fails instead
of selecting a historical demo. The size gate evaluates the final silhouette fit, and the printed
viewer link enables a successful placement. These checks establish input routing and acceptance
behavior, not visual reconstruction quality.

When enabled, fine-tuning finishes before placement and anchors. Those checks, observation
confidence, static-world review and the viewer link select the same final world. Scale fitting
and fine-tune export/import use the original provider world. Failed or missing fine-tune output
cannot silently fall back to another file. Placement records the world and report hashes;
changed files invalidate its viewer setting and rerun the local placement/anchor stages on a
full resume. Existing placements without these hashes need that local recheck too.
Anchor results also bind to the placement and world; selecting only `verify` cannot accept
stale dependencies. Provider metadata follows the selected video, image or multi-image mode.

Before checking observation confidence, the runner rebuilds the sidecar from the current world,
cameras and fitted registration. Tracked runs exclude the source person masks; missing camera
coverage fails instead of silently using the avatar. Single-person runs retain the documented
projected-avatar estimate. Confidence measures visibility and is not proof of accurate geometry.

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
Fresh inference remains unverified and is held pending confirmation of the operator's $30-per-account
workspace limit. No other account or provider receives a separate retry allowance.

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
