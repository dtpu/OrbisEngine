# Objects: everything in a clip that is neither a body nor the static scene

A clip contains three kinds of content. The **static scene** is what the Gaussian world is trained
on. The **bodies** are lifted out of it and replayed as avatars. Everything else — a thrown bottle, a
worn backpack, a pulled suitcase, a swinging door, a pushed pram — is an **object**, and until this
document existed the pipeline had no representation for one. Objects were either deleted (masked out
of the static training set as part of a person, then never rebuilt) or smeared (left in the static
set while they moved, training as a ghost).

This is the format and the pipeline for them. One format, four motion cases, and an appearance axis
that is solved once for all four.

---

## 0. Why one format

Two formats existed before this document, for the same class of thing:

| | `wander.objects/1` | `wander-rigid-object/1` |
|---|---|---|
| written by | the thrown-bottle session | the worn-object session (branch `aayan/vm-objects`) |
| covers | a thrown object with a ballistic arc | an object worn or carried on a body |
| geometry | a proxy capsule | a Gaussian PLY |
| pose | `segments` of kinds `attached` / `blend` / `free`, plus a baked 30 fps table | one `attachment.transform` on a body anchor, plus `track.per_frame[]` in camera space |
| orientation | `velocityAligned` with a decorative spin | inherited from the anchor frame |

They are the same thing seen from two ends. A held bottle and a worn backpack differ only in **which
joint** they hang from; a thrown bottle and a worn backpack differ only in **how pose is determined
per frame**. Neither difference is a reason for a second schema. `wander.objects/2` is the union,
and §6 gives the field-by-field migration from both.

---

## 1. The four motion cases

Everything an object can do is one of four, and they differ **only** in how pose is determined per
source frame. Appearance, placement, drift correction, visibility and the viewer contract are
identical across all four.

| case | pose is | examples | solved by |
|---|---|---|---|
| **`attached`** | a fixed rigid transform on a body joint | backpack, shoulder bag, hat, headphones, a phone held in a hand, an umbrella held up | least squares for one 6-DOF offset against N frames of the object's observed centroid. No trajectory. Cheapest case. |
| **`free`** | a fitted trajectory under its own dynamics | a thrown bottle, a dropped ball, a kicked can | 2D track → multi-view lift with gravity fixed from the clip's own metre scale. §4. |
| **`handoff`** | `attached` → `free` → `attached` | any throw and catch; putting a bag down; picking a cup up | the two attached fits and the free fit are solved independently and stitched with `blend` segments, which are labelled as the presentation compromise they are. |
| **`worldDynamic`** | its own world track, on no one | a swinging door, a pulled suitcase, a pushed pram, a passing stranger's bike | a baked per-frame world pose. Nothing analytic — the table *is* the model. This is the worst class in the survey (`docs/experiments/worn-objects.md` §1 rows 4, 5, 15) because it is absent from the moving layer **and** smearing the static one. |

`worldDynamic` is the case that needs saying twice. An object with no person to hang from still has
to come out of the static training set, or it trains as a smear. That is the survey's one structural
finding: there is exactly one mask in the codebase, `wander_worker.masks.people_masks`, and it is
used both to *select* the moving layer and to *subtract* moved content from the static one. Those are
different questions. `scripts/object_segment.py` writes a `dynamic/` mask (person ∪ object) precisely
so the static trainer can be given "what moved" instead of "what is a person".

---

## 2. `wander.objects/2`

One file per world, beside `people.json`:

```
public/worlds/<world>/
  people.json
  objects.json          <- this
  objects/
    bottle/object.ply   <- appearance, one per object
```

```jsonc
{
  "schema": "wander.objects/2",
  "clip": "public/clips/elevator.mp4",
  "fps": 12, "samples": 120, "sourceIndices": [...], "timestamps": [...],
  "metresPerWorldUnit": 1.8427,
  "coordinates": "Raw SfM world, OpenGL, camera 0 = identity - the frame the person PLYs use.",
  "people": "people.json",
  "objects": [ /* see below */ ]
}
```

### An object

```jsonc
{
  "id": "bottle",
  "label": "water bottle",
  "prompt": "a plastic water bottle",     // the WORD that segments it. A new object needs this, not new code.
  "motion": "handoff",                    // attached | free | handoff | worldDynamic
  "objectClass": "thrown",                // human-facing only; never branched on

  "appearance": {
    "kind": "gaussians",                  // gaussians | mesh | proxy
    "model": "objects/bottle/object.ply",
    "modelSha256": "...", "gaussians": 41234,
    "frame": "object-local; origin at the model centroid; +y is the long axis, cap at +y; unit scale, sized by sizeMetres",
    "sizeMetres": [0.06, 0.12, 0.06],
    "sizeWorldUnits": [...],
    "colorSRGB": [0.72, 0.78, 0.86],      // proxy fallback and HUD only
    "opacity": 0.9, "translucent": true,
    "shapeProvenance": "...",             // how the geometry came to be, in words
    "colourProvenance": "...",
    "observedViews": 47,
    "observedAngularSpreadDeg": 132.0,
    "invented": false                     // true when a generative model supplied unobserved structure
  },

  "pose": {
    "segments": [ /* §3 */ ],
    "orientation": { /* §5 */ }
  },

  "bakedTrack": {
    "space": "raw SfM world, identical to the person PLYs before transform.translation",
    "fps": 30,
    "sourceFrames": [0, 1, ... 298],
    "sampleIndex": [0.0, 0.4, ...],       // float index into the 12 fps sample grid
    "positions": [[x, y, z], ...],
    "quaternionsXYZW": [[x, y, z, w], ...],
    "visible": [true, ...]
  },

  "transform": { "translation": [0, 0, 0], "quaternionXYZW": [0, 0, 0, 1], "scale": 1 },
  "appliesSharedCameraDrift": true,
  "evidence": { /* whatever was measured; see the bottle's for an example */ }
}
```

Three things are load-bearing.

**`bakedTrack` is the contract; `segments` are the explanation.** Every motion case bakes to the
same table of per-source-frame position and quaternion, so a viewer implements *one* thing and gets
all four cases. The analytic `segments` are kept beside it for anything that wants to evaluate at
continuous time, re-derive the arc, or attach to an avatar that later gains better motion.

**The table is at 30 fps, not on the 12 fps sample grid.** The bottle's whole flight is 5 samples at
12 fps and lerping five points visibly flattens a parabola. The table is keyed on a float
`sampleIndex` so a viewer that already lerps on sample index needs no new time base and no physics.

**`transform` + `appliesSharedCameraDrift` are the same two hooks people have.** The object is stored
raw and the viewer applies shared scale → `transform.translation` → `floorFit.sharedCameraDrift`,
in that order, exactly as for a person. Omitting the drift puts the elevator bottle up to 50 cm off
the two men over the clip.

---

## 3. Segments

A segment covers `[fromSourceFrame, toSourceFrame)` and says how pose is determined there.

```jsonc
// attached - worn, held, carried. Pose = joint pose * offset.
{ "kind": "attached", "fromSourceFrame": 0, "toSourceFrame": 13,
  "parent": "person_01", "joint": 21, "jointName": "rightWrist",
  "offsetLocalMetres": [0.0, -0.03, 0.05],
  "quaternionXYZW": [0, 0, 0, 1],
  "anchorDefinition": "MultiHMR j3d[21], mapped to world by R*diag(1,-1,-1)*j3d*scale + t",
  "evidence": "object segmented in that hand in 11 source frames" }

// free - released, moving under its own dynamics.
{ "kind": "free", "fromSourceFrame": 25, "toSourceFrame": 41,
  "model": "ballistic",
  "p0WorldUnits": [...], "v0WorldUnitsPerSec": [...], "gWorldUnitsPerSec2": [0, -5.3218, 0],
  "t0SourceFrame": 25, "fps": 30, "space": "driftCorrected" }

// blend - the presentation compromise between two segments that do not meet.
{ "kind": "blend", "fromSourceFrame": 13, "toSourceFrame": 25,
  "from_": "attached", "to": "free", "offsetWorldUnits": [...], "offsetMetres": 0.461 }

// worldDynamic - its own world track, on no one. The table is the model.
{ "kind": "worldDynamic", "fromSourceFrame": 0, "toSourceFrame": 299,
  "model": "baked",
  "note": "positions and quaternions come from bakedTrack; there is no closed form" }
```

`blend` exists so nobody can mistake a stitch for a reconstruction. It carries `offsetMetres`: the
distance the two segments actually disagree by at the seam. A viewer that ignores `blend` and
evaluates `attached` + `free` only shows the raw reconstruction with its honest discontinuity, and
that is the right thing to look at when judging a fit.

---

## 4. The pipeline

Five stages. Each is a script, each is skipped when its output exists, and the whole thing is stage
`objects` in `scripts/run_clip.py`.

| stage | script | what it does |
|---|---|---|
| 1. segment | `scripts/object_segment.py` | **text-promptable.** Give it a word. Emits per-frame `object/`, `person/` and `dynamic/` masks plus RGBA cut-outs. |
| 2. track | `scripts/track_object_2d.py` | 2D track of a `free` object through the frames where it is a motion blur and nothing else. Skipped for `attached` and `worldDynamic`. |
| 3. lift | `scripts/lift_object_3d.py` | 2D track → a 3D trajectory, gravity fixed from the clip's own metre scale. |
| 4. orient | `scripts/object_orientation.py` | tumble rate and axis, from the in-flight blobs. Skipped for `attached`. §5. |
| 5. appearance | `scripts/object_appearance.py` | geometry → the object's own frame, at a Gaussian budget the object's on-screen size justifies. §5. |
| 6. package | `scripts/package_objects.py` | everything → `objects.json` in this format, with the baked table. |
| check | `scripts/render_object_check.py` | renders the packaged world offline, so a claim is provable without a browser. |

### Stage 1 is the generalisation that matters

`object_segment.py` is `worn_object_segment.py` (branch `aayan/vm-objects`) with three changes, and
every one of them was forced by a case the worn version could not do:

1. **Region of interest, not a person band.** The worn version scored only inside a dilated person
   mask. A thrown bottle and a swinging door are not inside one. ROI now comes from `--roi-joint`
   (attached / handoff), `--roi-track2d` (free), `--roi-box` (worldDynamic), or the whole frame.
2. **Crop and zoom.** CLIPSeg and SAM run at 352–1024 px. A backpack filling a third of the frame
   survives being resized to that; a 30 px bottle in a 1920×1080 frame becomes 9 px and is invisible.
   Each ROI is now cropped and upsampled before the networks see it.
3. **Proposal-and-score, not heatmap-and-threshold.** `--method clipseg` is the worn tool's path and
   is still right for large objects. It fails on small ones in a way worth recording: on this clip
   CLIPSeg returned **the thrower's hand at peak 0.96** for the prompt "a clear plastic water bottle
   with a blue label", because at 30 px the bottle and the fist are one blob to it.
   `--method sam-clip`, the default, inverts the order — SAM proposes every region in the ROI from a
   point grid, each proposal is cut out onto a neutral field, and full CLIP scores it against the
   prompt and its negatives. Every candidate gets CLIP's whole 224 px input to itself, so a 30 px
   object is judged at 224 px rather than at 9 px.

Adding a new object is a prompt and a motion case. It is not new code.

**Where it stops working, measured on this clip.** The prompt-only path found the thrower's *hand*
at CLIPSeg peak 0.96 and the catcher's *trousers* under `sam-clip`. Adding one seed point per run
(`--seed-json`, or `--seed-from-track2d`, propagated to the rest of the run by optical flow) fixed
the run where the bottle is 26 px against pale trousers — **21 of 25 frames correct at f253-f274,
and the segmentation is clean enough to see the label face give way to the clear side as it turns
in his hand**. It did *not* fix the run where the bottle is 25 px inside a closed fist against a
red-and-white wall: SAM will not split the bottle from the hand there at any prompt. The working
rule from this clip is that text-prompted segmentation needs the object to be **bigger than about
25 px and not enclosed by a hand**; below that a seed point is required, and inside a fist even a
seed point is not enough. That is a limit of the pixels, and the tool says which frames it
rejected rather than guessing them.

---

## 5. Appearance and orientation, once, for all four cases

Appearance is a **separate axis** from motion. Whatever determines an object's pose, its geometry is
built the same way, from whatever frames show it clearly — which are usually **not** the frames where
it is doing the interesting thing.

The order is fixed, and the first one that works, ships:

1. **Multi-view reconstruction** from the segmented cut-outs, in the same Gaussian representation as
   everything else, when the observed angular spread supports it. The worn-object session's result
   stands: a partial surface is honest, and the side the camera never saw should be **absent, not
   invented**.
2. **Image-to-3D** from the single cleanest cut-out when the spread is too narrow. An openly
   licensed model, conditioned on the real crop so the result is *that* object. This path
   **invents** structure, so `appearance.invented` is set true and `shapeProvenance` names the
   model, the revision and the licence.

   In practice this splits in two, and the split matters because the second half invents more:

   - **crop → 3D** works when the crop is big enough and sharp enough to carry the object's shape.
   - **crop → refine → 3D** is needed below that. On the elevator bottle the best crop anywhere in
     the footage is **26 × 25 px**, and fed straight to the 3D model it returned a shapeless dark
     shell with holes. An image-to-image refiner conditioned on the same crop, at high denoise,
     turns it into a clean single-object image that the 3D model can use; the crop then supplies
     the colour and the rough proportions and the refiner supplies the structure. Everything the
     refiner added is invention and must be enumerated in `colourProvenance` /
     `shapeProvenance`, not waved at.
   - Background removal is **off** for a cut-out that already carries alpha and **on** for anything
     a refiner produced, because a generated product shot comes with a cast shadow and the 3D model
     will happily build the shadow as geometry. It did exactly that here: the bottle's cast shadow
     came back as **two extra bottles**. `object_appearance.py --largest-cluster` keeps the biggest
     connected blob on a coarse voxel grid and drops the rest, which is cheaper and far more
     reliable than trying to get the input picture perfect.
3. **A shaded proxy** sized and coloured from the object's own pixels. This is the floor, not a
   destination. A proxy capsule reads as a cylinder, and a cylinder gliding through the air reads as
   fake no matter how good the trajectory under it is.

**Appearance has a budget.** `object_appearance.py` also puts the model into the object's own frame
(`+y` = the long axis, origin at the centroid, long axis spanning 1) and decimates it to a Gaussian
count the object's on-screen size justifies. An image-to-3D model happily returns 359 000 Gaussians
for a thing that is 33 px wide in the reel; 6 000 is already generous.

**Orientation is not optional.** An axis-aligned rigid body sliding along a perfect parabola looks
wrong, and the reason is that real objects tumble. `pose.orientation` supports:

```jsonc
{ "mode": "measuredRate",           // fixed | velocityAligned | measuredRate | measuredKeys
  "spinAxisWorld": [-0.0724, 0.0, -0.9974],
  "spinRevPerSec": 2.727, "spinRevPerSecSigma": 0.668,
  "extentPeriodFrames": 5.5, "crossings": 5,
  "revolutionsOverFlight": 1.45,
  "axisProvenance": "normal of the plane the throw travels in, v0 x g ...",
  "provenance": "rate from the perpendicular extent of the in-flight difference blobs ..." }
```

`fixed` and `velocityAligned` are declarations that nothing was measured. `measuredRate` and
`measuredKeys` bake real quaternions into `bakedTrack.quaternionsXYZW`.

Orientation is baked **per segment**, because the four cases want different things and one rule
over the whole clip gets both halves wrong. An object spinning at 2.7 rev/s while it sits in
someone's hand is as wrong as one gliding through the air axis-aligned. `attached` holds upright
plus the segment's own fixed quaternion; `free` velocity-aligns and adds the measured spin with the
clock starting at that segment's `t0SourceFrame`; `blend` slerps across the seam; `worldDynamic`
velocity-aligns, which is all a baked track on its own can support.

The rate itself comes from a signal that survives motion blur. A thrown elongated object is a
smear, and the smear's extent *along* the image velocity is the blur; its extent *perpendicular*
to the velocity is not, and that one rises and falls as the object turns its long axis and then
its end towards the camera. A rod's projected extent is 180° symmetric, so one period of that
signal is half a revolution. On the elevator bottle: 13 in-flight frames, 5 zero crossings, an
extent period of 5.5 frames, hence **2.73 ± 0.67 rev/s, about 1.45 revolutions over the 0.53 s
flight**. The axis is *assumed*, not measured — `v0 × g`, the normal of the plane the throw travels
in, which is what an ordinary end-over-end throw does — and `axisProvenance` says so in the file.

---

## 6. Migrating the two old formats

### From `wander.objects/1`

| old | new |
|---|---|
| `objectClass` | kept, plus a new `motion` (`thrown` → `handoff` when it is caught, `free` when it is not) |
| `appearance.kind: "proxy"` | unchanged; `"gaussians"` now names a real `model` path |
| `orientation` at object level | moved under `pose.orientation` |
| `segments` at object level | moved under `pose.segments`; `attached` gains `offsetLocalMetres`/`quaternionXYZW` |
| `bakedTrack.positions` | unchanged; `quaternionsXYZW` added |
| `transform`, `appliesSharedCameraDrift`, `evidence` | unchanged |

Nothing in the free-segment fit changes. The elevator bottle's `p0`, `v0`, `g`, `t0`, its 30 fps
table and its 3.56 px RMS reprojection are carried across unmodified.

### From `wander-rigid-object/1`

| old | new |
|---|---|
| `kind: worn \| carried \| thrown` | `motion: attached` (worn, carried) or `handoff`/`free` (thrown) |
| `object.ply`, `model_sha256`, `gaussians` | `appearance.model`, `.modelSha256`, `.gaussians` |
| `frame` | `appearance.frame` |
| `attachment.mode: body-anchor` | an `attached` segment spanning the clip |
| `attachment.anchor`, `anchor_definition` | the segment's `jointName` and `anchorDefinition` |
| `attachment.transform` (16 floats, column-major) | the segment's `offsetLocalMetres` + `quaternionXYZW`. It is a rigid transform, so nothing is lost; a viewer wants translation and rotation, not a matrix to decompose. |
| `attachment.mode: world-trajectory` | a `free` or `worldDynamic` segment |
| `track.per_frame[]` (camera-space anchor) | `bakedTrack` (world-space pose). The camera-space table was there because that session had no world; here there is one. |
| `measured.observed_angular_spread_deg` | `appearance.observedAngularSpreadDeg` |
| `shapeProvenance`, `colourProvenance` | unchanged, under `appearance` |
| `attachment.silhouette_iou_*`, `median_centroid_reprojection_px` | `evidence` |

`package_objects.py --from-rigid-object DIR` performs this conversion.

---

## 7. What the viewer has to do

Identical for all four motion cases, which is the point.

1. **Load** `?objects=<url>`, defaulting to `objects.json` beside the `?people=` manifest when that
   file exists. Absent file → no change to anything.
2. **One drawable per object.** `appearance.kind === "gaussians"` → load `model` as a SplatMesh and
   treat it as a one-frame person. `"mesh"` → a `THREE.Mesh`. `"proxy"` → a capsule with
   `colorSRGB`, `transparent`, `opacity`, sized `sizeWorldUnits × scale0`.
3. **Place it exactly as a person.** `pos = lerp(bakedTrack.positions, sampleIndex)` → `× scale0`
   (the **shared** `scale0` from `manifest.primary`, never a per-object one) → `+ basePos` →
   `+ transform.translation` → `+ sharedCameraDrift(sample)` when `?camdrift=1`.
4. **Orient it** by slerping `bakedTrack.quaternionsXYZW` on the same `sampleIndex`. Objects whose
   orientation is `fixed` bake identity quaternions, so there is no special case.
5. **Visibility** from `bakedTrack.visible`, same rule as `visibleSampleRuns` for people.
6. **Draw it in the same pass as the people**, so it depth-sorts against a body instead of popping
   through a shoulder.
7. **Never let it join the feet/floor snap.** It is not a person and `feetmode=local` must skip it.
8. **HUD**: one line per object, e.g. `bottle · gaussians · handoff · 0.66 m arc · reproj 3.6 px`.
