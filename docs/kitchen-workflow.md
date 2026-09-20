# Workflow for similar kitchen clips

Use this as the default preparation and review procedure for future kitchen clips.
It documents how to apply the reference's lessons; it adds no automated pipeline feature.
The user found the current kitchen useful and good, while identifying the missing Brita
filter pitcher. Its measured door, contact, room and coverage limitations remain recorded
in the [kitchen continuation](kitchen-continuation.md).
Read the [known limits](known-limits.md) before choosing another reconstruction approach.

## Preserve the recording and establish correspondence

Keep the original video bytes and SHA-256, decoded frame ordinals, presentation timestamps
and time base. Bind every sampled output and camera record to that source identity.
For a derivative or constant-rate conversion, retain its separate hash and measured mapping
back to the original; rounded average frame rate is not sufficient correspondence.
Preserve recorded audio words, timing and synchronization. Verify the delivered audio and
video endpoints against the source; do not invent speaker stems or replace recorded sound.

## Inventory props before cleaning

Review the complete source before deciding what to remove from the static reconstruction.
List important props, including a Brita/filter pitcher, pan, bag and handled containers,
and fixtures such as refrigerator/freezer doors, dishwasher fronts, taps and cabinets.
Record identity, source frames, visible shape, hand contact and relevant surfaces for each.
A prop need not be thrown, large or moving for much of the clip to matter to the scene.

Describe each observed phase: resting static, picked up, held, pouring or otherwise used,
handed over, put down, and resting again. Record genuine occlusion and out-of-frame spans.
Reserve independent review frames at transitions and difficult views before fitting motion.
Visible disappearance is a failure; an occlusion label must have evidence in the recording.
The missing Brita is a specific reminder to check this inventory again before delivery.

## Separate motion without erasing the kitchen

Keep the static room, people, movable props and hinged fixtures as distinct reviewed layers.
Use source-supported masks for the changing regions. Preserve unaffected counters, appliances,
handles and background details, and measure what a cleaning or masking change removes.
A prop resting on the counter may belong to the static layer during that observed phase;
coordinate its visibility with the moving layer so it neither doubles nor disappears.
Do not assume automatic object detection feeds dynamic masks into static-world training.
Compare the cleaned result with the original inventory, not only with an empty-room target.

## Keep identity stable and measure each new scene

Use one saved canonical appearance per physical person within a clip, even if tracking splits
that person into fragments. Reuse source-bound poses and retain existing verified outputs.
For an incremental repair, include a repeated control sample, compare native attributes after
the recorded packaging transform, and verify the exact intended sample set before merging.
A sparse successful export is still partial until its coverage is explicitly accounted for.

Solve and review cameras, floor identity and body-height placement for each new recording.
Do not copy this kitchen's coordinates, scales, floor offsets, hinge axis or object dimensions.
A countertop is not evidence of a walking floor. Use visible floor surfaces, source-camera
alignment and actual walking checks; report spatial residuals in body-heights, not assumed metres.
Keep placement changes separate from identity repair so each change has a measurable effect.

## Author unsupported object motion explicitly

The current automatic lane detects thrown objects and packages accepted wrist-connected
flights; it is not a general tracker for held pitchers, pans, bags or hinged appliances.
Held props and fixtures need reviewed manual authoring or another explicitly validated method.
Follow the [object contract](objects.md) for coordinate frames, contiguous baked tracks,
rotation, visibility and appearance loading. Do not hide unsupported motion behind a label.

Derive silhouette, size, orientation, contact and motion from source observations where possible.
Label unseen faces, generated appearance, deformation and approximate proxies as inferred.
A transparent pitcher can have weak edges, changing contents, reflections and hand occlusion;
additional views may be needed, or an explicitly labelled approximate proxy may be appropriate.
The current opaque mesh proxy path does not establish transparent material reconstruction.
A pouring gesture does not prove reconstructed water: water remains a known single-view limit.
If important shape or contact cannot be supported, record the gap rather than inventing evidence.

## Review the rendered result and retain the evidence

Predeclare tolerances and critical times using the [quality rubric](quality-rubric.md),
including pickup, pouring, putdown, fixture motion, occlusion and any identity handover.
Compare the actual viewer at matched source cameras and timestamps with the recording.
Inspect held-out source frames, opposing and displaced views, and walk through the intended
route to test floor support and boundaries. Check object load diagnostics and visible counts.
A plausible screenshot, passing test or numerical control alone does not establish visual quality.
Retain stage-specific failures and missing evidence even when the user finds the result useful.

## Bound paid work and recover existing results

Set a concrete budget and execution limits before paid work; reconcile aliases, earlier attempts
and provider receipts rather than treating a fresh account or candidate name as a fresh allowance.
Inspect completed shared caches and active jobs before setup, without duplicating or stopping
another session's work. Follow [paid accounting and recovery](paid-recovery.md): persist input
hashes and submission identity, preserve partial outputs, and recover with size/hash checks.
An interrupted wait, failed transfer or expected partial-coverage exit never authorizes another
inference submission. Publish a fresh reviewed candidate while preserving accepted assets.
