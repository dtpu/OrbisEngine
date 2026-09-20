# Objects

`objects.json` describes moving props separately from people and the static world.
The current writer is `scripts/package_objects.py`; `fourd.html` plays its baked tracks.
The automatic pipeline detects thrown objects. Other motion labels describe the contract,
but do not imply that the pipeline can reconstruct arbitrary doors, bags, or vehicles.

For cooking clips, follow the [kitchen workflow](kitchen-workflow.md): inventory handled props
before cleaning, preserve their source observations, and review each interaction separately.
The current kitchen is a useful reference with recorded gaps, including the missing Brita pitcher.

## Files and discovery

A package normally lives beside the multiperson manifest:

```text
public/worlds/<name>-4d/
  people.json
  cameras.json
  objects.json
  objects/<id>/object.ply
```

The viewer tries `objects.json` beside the selected multiperson manifest. Use
`?objects=/worlds/<name>-4d/objects.json` to select it explicitly, or `?objects=0` to disable it.
Model URLs resolve relative to `objects.json`. Missing manifests are optional; objects without
`bakedTrack.positions` are skipped. Object models and generated manifests stay outside Git.

## Manifest contract

The writer emits `schema: "wander.objects/2"`, plus `clip`, `fps`, `samples`, `sourceIndices`,
`timestamps`, `metresPerWorldUnit`, `coordinates`, `people: "people.json"`, and `objects`.
The top-level sample grid comes from `people.json`; it differs from the baked object's frame grid.

| Object field               | Meaning and current behavior                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `id`, `label`, `prompt`    | Stable object identifier, display name, and segmentation prompt.                            |
| `motion`                   | `attached`, `free`, `handoff`, or `worldDynamic`; descriptive, not a viewer physics switch. |
| `objectClass`              | Descriptive category such as `thrown`.                                                      |
| `appearance`               | Model or proxy, dimensions, color, and provenance; see below.                               |
| `pose.segments`            | Explanation of held spans, ballistic spans, and seam blends.                                |
| `pose.orientation`         | Rotation method and evidence. Baked quaternions take precedence.                            |
| `bakedTrack`               | Positions, orientations, visibility, and timing actually played by the viewer.              |
| `transform.translation`    | Added to the shared placement origin in viewer coordinates.                                 |
| `transform.scale`          | Multiplies the shared scene scale; defaults to 1.                                           |
| `appliesSharedCameraDrift` | Shared vertical drift is applied unless explicitly false.                                   |
| `evidence`                 | Fit errors, flight counts, seam gaps, detection rules, and assumptions.                     |

Positions use the raw SfM/OpenGL world frame shared with the person PLYs. The viewer applies
shared scale and rotation, then the shared placement origin plus the object's translation;
interpolated camera drift adds to world Y. Objects skip person foot locking and floor snapping.
Although the writer includes `transform.quaternionXYZW`, the viewer does not apply that field.
Bake object rotation into `bakedTrack.quaternionsXYZW`.

### Baked tracks and segments

`bakedTrack` contains `fps`, `sourceFrames`, `sampleIndex`, `positions`,
`quaternionsXYZW`, and `visible`. Position and quaternion arrays must have matching lengths.
Quaternions are `[x,y,z,w]`. `sampleIndex` is a floating index into the person sample grid,
used to interpolate shared drift. It is not a timestamp in seconds.

The viewer computes `t * fps - sourceFrames[0]`, clamps to the array range, interpolates
positions linearly, and slerps baked quaternions. It reads visibility from the lower frame.
Therefore tracks must be contiguous and uniformly sampled; arbitrary `sourceFrames` gaps
are not honored. End poses remain visible unless the visibility table says otherwise.

At ingestion, the viewer validates that `fps` is finite and positive, `sourceFrames` is a
contiguous nonnegative integer sequence, positions and optional `sampleIndex`, quaternions, and
visibility arrays have matching lengths, and all vectors/scalars are finite. Quaternions must have
a nonzero norm within 0.1% of unit length; visibility entries must be booleans. When camera drift is
enabled, sample indices must stay within its table. An invalid track is skipped with console and
HUD diagnostics while other objects and the scene continue loading. This rejects malformed data
before it can place a mesh at a nonfinite position; it does not inspect timestamps, infer missing
frames, or repair a variable-rate track. Legacy manifests without optional quaternions, visibility,
or sample indices remain supported.

The detector records source `fps`. The lift step uses that clock (or an explicit `--fps`), and
packaging carries the fitted clock into ballistic positions, spin timing and the baked track.
Both steps cross-check it against camera `time`/`sourceIndex` intervals. Conflicting clocks,
partial timing and nonconstant intervals fail rather than silently packaging incorrectly timed motion.
The 30 fps fallback applies only to legacy files with no source-clock metadata. The person sample
grid can have a different rate. Changing only the packaged output `fps` cannot repair a fit made
with the wrong clock. Variable-rate inputs need a documented constant-rate conversion first.

Segments retain `fromSourceFrame`/`toSourceFrame` and explanatory fields:

- `attached`: parent person, joint index/name, and the held-object assumption.
- `free`: ballistic `p0WorldUnits`, `v0WorldUnitsPerSec`, `gWorldUnitsPerSec2`, reference frame,
  source fps, and fit evidence. These parameters describe drift-corrected positions.
- `blend`: the transition and `offsetWorldUnits`/`offsetMetres` measuring the seam disagreement.

The viewer does not evaluate segments or attach objects to live joints. The packager resolves
those poses first, including smooth seam transitions. Attached spans are assumed held between
observed flights. Their orientation is fixed; free spans follow velocity and optional measured
spin. Seam orientations are interpolated. Without an orientation file, rotation is not measured.

### Appearance and provenance

`appearance.kind: "gaussians"` with `model` loads a Gaussian PLY. The writer copies the model
under `objects/<id>/` and records `modelSha256` and `gaussians`. `object_appearance.py` can
center and orient an input model with its long axis along +Y, then reduce its Gaussian count.

`sizeMetres` is `[x,y,z]` in object coordinates; `sizeWorldUnits` divides these values by
`metresPerWorldUnit`. Gaussian rendering centers the measured bounding box and scales uniformly
from its Y extent. X/Z dimensions do not independently set the rendered width/depth.
The viewer also accepts manually authored `appearance.shape: "mesh"` with inline convex faces
(`meshInline.verticesWorldUnits`, `meshInline.faces`) or `boxHalfWorldUnits`; this is not a GLB loader.
Other appearances use a capsule when `shape` is `capsule`, otherwise a sphere. The current capsule
uses size X for diameter and size Z for length, unlike the Gaussian +Y sizing convention.

`colorSRGB` colors proxies. Proxies render opaque even when `opacity`/`translucent` request
transparency, because of the current splat compositing behavior. Gaussian color/opacity comes
from the PLY. Record `shapeProvenance`, `colourProvenance`, `observedViews`, and
`observedAngularSpreadDeg` when known. Set `invented: true` for generated unseen structure.
Image refinement and image-to-3D are not measured multiview reconstruction. Metre estimates depend
on the person-derived scene scale; seam gaps and per-flight reprojection errors remain evidence.

## Current pipeline

`run_clip.py` enables automatic objects by default; `--no-objects` disables that default.
Explicit `--object` requests still enable the stage. Without a named request, it:

1. Runs `detect_object_flights.py` using camera poses, tracks, masks, and a metre scale.
2. Stops with an empty result if no flights pass. Single-person worlds can record detections,
   but cannot package them through this lane without multiperson floor fits and wrist poses.
3. Runs `lift_object_3d.py`, retaining flights that connect tracked wrists at both ends.
4. Runs `describe_object.py` on observed crops; without `OPENAI_API_KEY` it uses `--no-vlm`.
5. Optionally calls `worker/modal_image_to_3d.py`, then `object_appearance.py`. `--no-shape`
   disables this generation. Shape failures fall back to a proxy; cloud work follows repository rules.
6. Runs `package_objects.py` with accepted flights, dimensions, color, and provenance.

Existing intermediate files are reused. This lane packages the accepted flights as one object;
it does not automatically establish distinct identities for several different thrown props.
It also does not automatically feed `object_segment.py` dynamic masks into static-world training.

For an existing reviewed multiperson run, packaging alone uses real CLI options:

```sh
uv run --locked --group inference scripts/package_objects.py \
  --world public/worlds/example-4d \
  --fit .context/run/example/objects/fit3d.json \
  --tracks-dir .context/run/example/tracks \
  --cameras .context/run/example/pi3x/cameras.json \
  --id prop --label 'tracked prop' --prompt 'a prop' \
  --size-m 0.06,0.12,0.06 --shape-provenance 'Proxy; no reconstructed model'
```

This requires `people.json` with `floorFit`, per-track `source-poses.pt`, and a reviewed
`wander.object-fit/2` fit. A `/1` fit additionally needs `--thrower` and `--catcher`.
Optional `--model`, `--orientation`, and provenance flags add reviewed appearance/rotation.

The named `--object ID:MOTION:PROMPT[:ROI]` lane is manual/experimental: it segments an ROI,
uses fixed thrower/catcher indices for flight fitting, and reuses a supplied model when present.
Attached/worldDynamic requests currently omit the packager's required `--fit`; repeated named
objects overwrite the same manifest. Do not treat this as a general multi-object authoring flow.
Likewise, `--from-rigid-object` conversion does not produce a baked track for converted entries;
those entries need a completed track before the current viewer can display them.

## Explicit static collision surfaces

An optional `?colliders=/path/colliders.json` supplements the walker's splat occupancy grid.
The bench or other furniture remains visually rendered by the room. Its collision geometry is
an independently reviewed set of oriented boxes; loading it does not create a second visible mesh.
Use `?colliderdebug=1` to inspect projected box outlines. They show through the room for review.

The manifest has `schema: "wander.colliders/1"`, `coordinates: "viewer-world"`, a `world` filename,
and a nonempty `bodies` array. Each body requires `id`, `role`, `shape: "box"`, `center`,
positive `halfExtents`, unit `quaternionXYZW`, `walkable`, and descriptive `provenance`.
Coordinates are final viewer coordinates: do not apply person registration or a CV axis flip.
The world filename guard rejects accidental selection of a differently named world; it does not
authenticate file contents or distinguish identically named worlds in different directories.
Explicitly requested missing or invalid manifests fail loading instead of silently losing collision.

A walkable box uses its finite local +Y face as support. Capsule collision and continuous movement
checks also apply to nonwalkable boxes. An inferred conservative frame envelope can block empty
space between real struts; declare that approximation in provenance. With no manifest, the existing
occupancy-grid behavior is retained. Generated geometry and review evidence belong in private storage.

Finite explicit support also establishes walkable floor outside the splat-derived floor hull.
This permits reviewed support across missing floor samples. It does not clear occupied cells,
disable swept collision, or remove step-height checks. Outside the finite support footprint,
the original floor-boundary checks still apply. The occupancy footprint is circular, matching
the walker capsule, so diagonal furniture corners do not acquire square padding.

These colliders constrain the viewer's walker. People and props still follow their baked animation;
they are not rigid bodies and are not automatically pushed onto a seat. Correcting actor contact
requires a separately validated placement using the same measured support surfaces. A walker
collision pass does not establish that the person's pelvis, back, or feet are supported.
