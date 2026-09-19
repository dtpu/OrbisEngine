# Kitchen investigation and input review policy

The poor kitchen preview combines an imperfect generated environment, damaged cleaned inputs,
and unvalidated camera/person registration. Export corruption is ruled out for the inspected asset.
A clean-looking origin panorama is not evidence that displaced viewpoints will hold up.

## Findings and limits

| Layer | Evidence | Conclusion |
| --- | --- | --- |
| Provider generation | Archived Marble 1.1 operation completed; the provider supplied a panorama and full-resolution SPZ. | This is a real Marble output, distinct from Daniel's source-coloured reconstruction. Completion is not visual acceptance. |
| Export and serving | Direct provider export, S3 catalog and bytes served by the local viewer have identical SHA-256 and size (28,769,418 bytes). | No swapped, truncated or corrupted SPZ in this chain. |
| Input cleaning | Five SHA-verified cleaned images retain dark/blurred removal fills around recorded scene structure. | Some defects predate Marble. Their contribution to the final geometry is plausible, not quantitatively isolated. |
| Input selection | Eight views across a reported 123.6-degree sweep became five after rejecting smear/blur and inconsistent fridge state. Sink/counter/work-area coverage is visible. | The area cannot simply be dismissed as unseen. No kitchen video-versus-stills comparison was found. |
| Local renderer | Origin checked with 800,000-splat LoD and all 1,920,000 splats; clipping checked at 0.01–500 and 0.001–2000 viewer units; unit scale and Marble up-axis conversion confirmed. | Origin looks substantially cleaner than the user's displaced view. A nearby off-axis camera reproduces the floor smear/black void at both LoD and full detail; more splats alone do not fix it. Full detail and broader clipping do not establish safe arbitrary movement. Diagnostic translations can enter or leave geometry; this preview has no validated collision. |
| Native renderer | The private provider world refused access in a fresh browser. | No matched-camera provider-versus-local rendering verdict is claimed. The user's exact screenshot camera was not recovered. |
| People and registration | Preview explicitly loads only the environment. A separate person sequence exists; archived alignment passes 0/4 held-out views. | Missing people in this preview is expected. Do not place them into Marble with the failed transform. |

The saved generation request says `reconstruct_images=true`, while the returned world says false.
This discrepancy remains unresolved; it does not prove that the provider ignored the request.

The supplied `kitchen-cooking.mov` and the `test1.mov` identified by the person package have different
hashes and frame rates. Archived notes describe the same action, but decoded correspondence is
unverified. They must not share exact frame indices or timing claims without a measured mapping.
Conversely, re-encoding the same recording must not create a fresh paid-stage allowance.

## Correct input provenance before another alignment attempt

The former `worker/modal_clean_video.py` resampled with FFmpeg, then approximated source indices
using rounded average FPS. Recovery of Daniel's two archived framehash files now independently
confirms 609 unique exact pixel-hash matches and 305 incorrect labels. These artifacts preserve
their recorded framehash PTS; they do not recover the original file's native container timebase.

New cleaning runs observe the actual FFmpeg filter selections, cross-check frame checksums before
and after resampling, and join them to decoded source ordinals and FFprobe PTS. Reports and mask
archives retain the source SHA-256 and timebase; each returned cleaned image has its own hash and
source binding. Partial selection preserves the requested order. The full cleaned video is hashed.
Missing/nonmonotonic PTS or unsupported FFmpeg diagnostics fail explicitly.

Multi-image selection now requires source-bound camera metadata and exact retained frame indices
and PTS before paid cleaning. Submission verifies the image hashes and selection receipt. Existing
world reuse and known-operation recovery remain read-only paths without new cleaning prerequisites.
Run `uv run --locked scripts/test_source_timing.py` for the offline pixel-identity and recovery tests.

These changes do not relabel existing outputs or prove camera geometry. A reconstructed historical
mapping needs independent pixel corroboration, and placement still needs held-out registration.
Do not patch clip-specific indices or assume a timing correction repairs generated geometry.

## Video-first policy

The pipeline CLI and standalone `select_world_mode.py` now both default to video. The selector's
former automatic image/multi-image choice and caption-only claim have been removed. Its camera-angle
heuristic is available through explicit `--still-images`; the pipeline uses that only for an explicit
`--marble multi` run. Image and multi-image modes remain reviewable alternatives, not automatic
fallbacks. Angular spread and gradient energy do not establish cleaned structural coverage,
translational parallax or consistent moving fixtures.

The provider documents [video input](https://docs.worldlabs.ai/api) and automatic caption generation
when a prompt is omitted. That does not establish that video pixels are discarded. Video requests now
retain the pipeline's source-grounded description, pin `marble-1.1`, and request private permissions.
The entire cleaned clip at the configured processing frame rate is uploaded; this does not mean every
original source frame or unmodified source pixel is retained. Input bytes/SHA-256 and the exact request
are saved in the existing submission receipts. Existing operation recovery still prevents resubmission.

Offline tests verify video-first decisions, explicit still overrides, full input-byte transfer to the
mock transport, prompt/model/private request fields, input mutation rejection, credential handling,
and recovery. They do **not** establish fewer hallucinations or improved kitchen geometry. No new
video-versus-stills generation was run for this change. Such a comparison must bind the same source
and cleaned inputs, record model/settings, and judge matching held-out and displaced views; a prompt
or a successful API request is not a visual acceptance result.

For moving-camera recordings, review the temporally coherent video first. Require recorded coverage,
cleaning and fixture-consistency evidence before substituting sparse stills. More blurry, occluded or
inconsistently cleaned frames do not guarantee better output. Recover existing outputs and improve
input evidence before buying another generation. The exact initial kitchen multi-mode decision remains
unproven; the five-still rationale is documented, but is not a comparative quality experiment.

## Bounded AI-assisted review

Use labeled source, cleaned and rendered images to flag missing surfaces, removal damage, texture
smear, inconsistent fixtures and camera-dependent failures. Include representative whole-clip frames
and denser samples around occlusions, blur and moving doors. Bind every image to source hash, actual
PTS, world hash and camera parameters outside the model. Review held-out source views and displaced
views separately. A model's visual verdict cannot certify geometric alignment or invent evidence of
unobserved surfaces. [OpenAI image-input documentation](https://developers.openai.com/api/docs/guides/images-vision)
describes multiple-image inputs and limitations of spatial reasoning.

Reconcile Daniel's frozen calibration/holdouts, request accounting and conservative retry handling
with Austin's actual source/world/camera bindings and timestamp validation. Their quality-attempt
ledgers use the same schema string but incompatible list/dictionary structures; migrate old claims
explicitly rather than replacing a ledger or resetting allowances. Neither judge is a demonstrated
autonomous repair-and-promote loop. Proposed repairs need a changed hypothesis, bounded attempt,
retained receipts and independent visual/geometric acceptance.

A bounded live vision-API review was attempted. The system Python trust store failed TLS validation;
a verified Certifi trust store repaired transport, after which the API returned HTTP 429. No model
verdict was obtained and the 429 was not retried. Visual findings above come from direct image
inspection, Playwright and recorded provenance, not a successful external AI approval.

Private source images, screenshots, hashes, browser reports and detailed audits remain in
`.context/evidence/branch-comparison/` and `.context/evidence/integration-final/`, outside Git.
No new Marble generation or GPU inference was launched for this investigation.

## September 19 delivery recovery

A later, separate `kitchen-cooking` run was recovered with 610 detections across two appearance
fragments. [The delivery report](gym-kitchen-delivery.md) describes its source mapping, one duplicate
suppression, 609 retained samples, original audio and full-duration preview. It does not promote
the old `test1.mov` alignment or claim that the new kitchen geometry/object reconstruction passed.
