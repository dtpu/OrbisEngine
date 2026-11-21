# Compact person motion

The default package is **lossless float32**: every recorded frame, splat centre and rotation is
preserved bit-for-bit. The first original PLY supplies colour, opacity, scale and other static
attributes. Packing refuses a sequence if any static attribute changes, any property layout or
splat count differs, or any value is nonfinite. This is a delivery optimization of the existing
reconstruction; it does not improve geometry or registration.

Both `package_person_sequence.py` and `package_multiperson.py` add the lossless track when they
package a run. To add or refresh one without inference or publication:

```sh
uv run --locked scripts/package_person_motion.py public/worlds/<clip>-4d/person
```

Original PLYs remain available. Frame names/order, fps, exact timestamps, duration, source offsets
and other sequence metadata are retained. The packer refreshes `frame_sha256` from the original
PLYs and adds `motion`. A refused repack removes any stale motion descriptor before reporting the
error. It does not remove or rewrite PLYs. Packing and publishing must run after all frame writers
have stopped.

## Format and validation

`sequence.json.motion.schema` is `wander.person-motion/1`. The record contains:

- `file`, `dtype` (`float32` by default), `bytes`, and payload `sha256`.
- `frames`, `splats`, ordered `frameFiles`, and ordered `channels`:
  `x,y,z,rot_0,rot_1,rot_2,rot_3` (rotation is w,x,y,z).
- `base.file`, `base.bytes`, and `base.sha256` binding the track to its exact appearance PLY.
- `min`, `scale` and measured `maxAbsError` per channel; all errors are zero for float32.

Payload order is frame, splat, channel, with little-endian scalar values. Float32 stores the
original values directly. The browser validates schema, dimensions, frame order and byte length,
verifies both SHA-256 hashes, then decodes the complete track before installing any compact keys.
It preserves the normal PLY playback timing and interpolation. Stance stabilization derives complete
predecessor chains, so seek order and frame arrival speed cannot permanently cache incomplete
corrections. Existing viewer flags for explicitly
selecting fewer splats continue to work; compact packaging itself never reduces splats or frames.

Missing, corrupt, mismatched and older unbound motion records fall back to the original PLYs.
The reason is logged and exposed as `wander.people[i].motionFallback`. The interpolated PLY path verifies frame hashes
when `frame_sha256` is present; invalid PLY bodies, mismatched layouts/counts or missing
required frames cause a load error. A failed fallback sets `wander.people[i].loadError`, finishes
its loading attempt (`settled`), and never reports ready. The demo wrapper displays the failure
and offers reload. `?motiontrack=0` forces the original PLY path for comparisons.

## Visibility after occlusion

A person's `visibleSampleRuns` contains inclusive indices into the shared `people.json.timestamps`
grid. Visibility starts at the first indexed timestamp and ends at the next shared timestamp after
the run (or the shared duration). Compact PLY indices cannot identify these intervals after missing
detections or a split track. Malformed or overlapping runs fail explicitly. This affects visibility
only; each sequence retains its own pose timestamps and original PLY geometry.

## Optional quantization

`--quantize` explicitly opts into lossy uint16 storage. The browser also requires
`?motionquantize=1` before using a uint16 track; otherwise it loads the original PLYs. This flag
is for direct viewer review paths. Quantization is never inferred from asset metadata or enabled
by the pipeline. `--lossless` remains an accepted explicit spelling of the default.

For uint16, `value = code * scale[channel] + min[channel]`, rounded to float32 by the viewer.
`maxAbsError` measures that final float32 error. Existing historical byte/speed measurements for
quantized packages do not establish sizes or performance for the new lossless default.

## Published demo verification

Shared snapshot `efe034d2-badf-48c5-9a71-2c0257995b15` enables lossless tracks for stairs and both
people in elevator. The publication changed exactly six paths: three sequence manifests and three
motion payloads. Every original PLY remains available; all unrelated viewer entries and the existing
archive pointer were preserved, including the separately published gym review work.

| Sequence | Frames × splats | Original PLY bytes | Motion + base PLY bytes |
| --- | --- | --- | --- |
| Stairs person | 50 × 40,000 | 136,020,750 | 58,720,415 |
| Elevator person | 120 × 40,000 | 326,449,800 | 137,120,415 |
| Elevator person_01 | 120 × 40,000 | 326,449,800 | 137,120,415 |

All 81.2 million position/rotation values were checked bit-for-bit, with unchanged static appearance,
source-frame hashes and timing metadata. All six published files passed independent fresh-download
SHA-256 and size checks before the conditional snapshot update. Playwright then checked the published
snapshot: all 50/120/120 frames loaded as float32 without fallback or page errors, source playback
advanced, Enter controlled transport, original audio ran, and both elevator objects were present.
Screenshots were visually inspected. This browser check used a warm cache seeded only with bytes
matching the independently verified downloads; its roughly 5-second loads are not cold-network
benchmarks. Earlier matched-camera PLY/compact captures were pixel-identical outside the changing
performance HUD. These checks establish lossless delivery, not improved reconstruction or physical
headset performance. Receipts and private images remain under `.context/evidence/integration-final/`.

## Checks and credits

Run `bun run test:person-motion` for Python packaging checks and real Python-to-TypeScript
round trips, including zero-error float32, static appearance rejection, preserved timing,
SHA-256 failures, wrong keyframes, reordered frames, missing payloads, explicit quantization,
failure propagation when original frames also fail, and stance invariance under seeks/delayed
frames. Tests need no media, network, GPU, compression streams or provider credentials, and run
on Bun 1.2.21.

The packer, decoder, pipeline hooks, PLY parser and round-trip tests were selected from James's
branch commits `d08706d`, `4d3c95b`, `6a34c12`, `265397f`, `6f18244`, `57c8954`, and `5f020e9`.
Their original author Claude and co-author Claude Fable 5.1 remain credited in the ported commits.
Codex adapted the integration for lossless defaults, format identity, integrity verification and
safe fallback. Live rendered comparisons and actual network timing remain separate acceptance
checks; unit tests do not establish visual quality or headset performance.

## Experimental tracking crop recovery

Tracked runs can opt into `--pose-crops` (or the `pose_crops` tracking stage parameter).
It is off by default and requires source camera calibration for every sampled frame and
independent person instance masks. Each instance unmatched by the ordinary full-frame pose
pass gets one additional crop inference. The crop translates the camera principal point,
then applies the estimator's resize and padding; predicted joints are mapped back to source
pixels. Existing full-frame matches are preserved. A crop may claim only its target mask
under the existing IoU threshold.

This adds GPU calls and can increase runtime substantially. It does not automatically retry,
change the stage retry budget, or distinguish pedestrians from people pictured on signs or
statues. `detection-audit.json` records crop bounds, detections, recoveries and pose origins;
track summaries record call counts and motion observations preserve their origin.

The geometry and submission plumbing have CPU tests. In the first city GPU trial, 186 crop
calls produced only five mask-supported poses; none met the unchanged track-start score.
No additional person track survived and source overlays still missed pedestrians. The reviewed
advertisement exclusions worked, but all-person reconstruction failed. Keep crop recovery off
by default; `recoveredCropPoses` counts supported detections, not surviving tracks.

## Reviewed depictions and statues

`--person-exclusions review.json` (or the tracking stage's `person_exclusions` parameter)
accepts explicit source-bound observations of a depiction or statue. It does not classify
objects automatically. Unreviewed observations remain eligible for pose recovery and tracking,
including small or stationary pedestrians. An exclusion must be visually reviewed with source
context; an agent-authored review must identify its author as an agent, not a human approval.

The `wander.person-exclusions/1` JSON includes `sourceSha256`, source `width` and `height`,
`reviewer` (`kind`, `name`), and `entries`. Each entry records `sourceIndex`, a source-pixel
`box` in xyxy order, `label` (`depiction` or `statue`), a specific `reason`, and an `evidence`
reference. Every reviewed box must uniquely match a detector box at IoU strictly above 0.95.
Stale source/dimensions, ambiguous matches and missing detections fail rather than silently
excluding something else. Entries outside the tracking schedule are rejected.

Excluded instances remain competing evidence during pose assignment, so a pose belonging to a
pictured person is not reassigned to its neighbor. They are recorded in `excludedInstances`
separately from unresolved `unmatchedInstances`; the exact review is retained with its SHA-256
and reviewer in the track receipt. The review content participates in the paid-attempt identity.
It does not change retry caps or establish all-person recall, identity or scene acceptance.
