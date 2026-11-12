# Local completion of unobserved colors

`scripts/complete_masked_colors.py` creates a separate color-only review candidate from a saved
SPZ world and its camera/mask export. It uses actual alpha-composited Gaussian footprints to
distinguish source-visible donors from points rendered only behind person masks. It runs locally
on the CPU and makes no provider requests. The automatic video pipeline does not enable it.

This is an appearance experiment. Colors inferred behind people are not recovered observations;
the tool does not repair geometry, alignment, avatar placement or collision support. A completed
command is not a passed scene. Apply the same [quality criteria](quality-rubric.md) and retain
existing failures when reviewing the candidate.

## Inputs and invocation

Use an SPZ v2, SH-degree-zero world in the same raw coordinate frame as a `finetune_export.py`
export. The tool does not solve alignment. The export must contain `cameras.npz`, `splats.npz`,
`export-receipt.json`, and a binary `masks/NNNN.png` for every selected camera sample. Missing or
mismatched masks stop the command; no avatar-mask or empty-mask fallback is supplied.

Write the intended training sample indices as a JSON integer array. These are export sample
indices, not original video frame numbers. Use the same training/holdout split as the preceding
fit: the tool requires an explicit list and does not choose a split or inspect heldout RGB.

Set `BODY_HEIGHT_NATIVE` to the measured body-height ruler in the export's native coordinates,
then run:

```sh
uv run --locked scripts/complete_masked_colors.py \
  --world /path/to/retained-world.spz \
  --export /path/to/finetune-export \
  --training-frames /path/to/training-frames.json \
  --body-height-native "$BODY_HEIGHT_NATIVE" \
  --out .context/color-completion-review
```

The output directory must be new. Original assets are not overwritten. Keep original source,
world and paid-attempt records; a new output name does not reset retry allowances. The command's
time limit is for bounded local execution, not permission for a new paid stage.

## Selection and color inference

Every selected source camera is replayed with the whole scene participating in front-to-back
occlusion. The tool accumulates each point's rendered mass, person-masked mass, mask-clear
mass and number of views with a mask-clear contribution. A projected center inside a camera
frustum or outside a person mask is not sufficient evidence of rendered exposure.

- Targets have positive rendered mass and exactly zero mask-clear mass across selected views.
  They need not have unchanged training parameters. Completely unrendered points are excluded.
- Donors have mask-clear contributions in at least `--min-clear-views` views, default 3.
  Eligible donors are selected **before** the spatial-neighbor query.
- At most `--neighbors` donors, default 16, are considered within `--radius-body-heights`,
  default 0.25. At least `--min-neighbors`, default 3, must be available. The body-height ruler
  and export scale convert that radius into the stored world's units.
- A spatial Gaussian with sigma half the radius weights a per-channel median of donor RGB.
  Newly inferred colors never become donors in the same execution. Targets with too few donors
  keep their exact original bytes.

Only selected color records are replaced. Header, positions, opacity, scales, rotations and
unselected packed colors remain byte-identical, including colors that decode outside the
display range before clipping. A successful write therefore cannot repair an existing depth or
contact failure, and its preservation of geometry is not evidence that the geometry is correct.

## Evidence and limitations

The result includes the separate `world.spz`, an inferred-point mask, exposure arrays and a JSON
report. The report binds the world, camera export, explicit frame list, masks, relevant code and
lockfile by SHA-256, records settings and counts, and leaves scene acceptance false. Export
metadata and hashes document the supplied inputs; they do not independently prove original
source lineage, physical scale, camera accuracy or background visibility.

The CPU compositor follows the retained diagnostic raster's finite footprint, near-plane,
opacity and transmittance cutoffs. It is an approximation, not a claim of bit-for-bit gsplat
CUDA behavior. Exposure is measured at the stored endpoint, not throughout historical training.
An exact zero is relative to this raster and these selected views. Nearby source-visible colors
can belong to another material or surface; the median does not establish material identity.

Review source-matched beginning/middle/end and critical frames, source-visible regions, lateral
views and playback. Keep original depth, coverage, spread and contact denominators fixed. The
previous [forest probes](forest-observation-diagnostics.md) reduced only part of a remnant;
masked RGB scores alone cannot certify cleanup or a coherent walkable scene.

## Retained forest trial — November 12

The third local appearance candidate used the retained fourth fit directly, the same 83 training
views, and the defaults above. Unlike the preceding probes, target selection included points
whose colors had changed during training, and the spatial query started from eligible donors.
It completed in 96.3 seconds on the local CPU with no provider requests.

| Measurement                                   | Result                                                    |
| --------------------------------------------- | --------------------------------------------------------- |
| World points                                  | 1,920,000                                                 |
| Rendered, entirely masked targets             | 23,026                                                    |
| Donors clear in at least three training views | 417,896                                                   |
| Targets with sufficient donors                | 22,872                                                    |
| Changed packed color records                  | 22,863                                                    |
| Non-color SPZ bytes                           | Identical to the retained fourth fit                      |
| Five fixed alpha/depth/spread arrays          | Exact equality to the prior CPU evaluation                |
| Seven frozen RGB holdouts                     | Worst same-CPU change −0.00000326 dB                      |
| Unchanged geometry checks                     | Depth 0/5, coverage 5/5, spread 5/5                       |
| Unchanged foot-contact checks                 | 47/90, verified reuse from identical geometry and opacity |

Source-matched views at original frames 0, 34, 88, 132 and 177 show a much less visible bright
torso-shaped remnant. A dark lower-body patch remains, along with the road seam and lateral
distortion. These are visual observations, not a measured percentage of person removal. Colors
in masked regions are inferred background. The scene remains failed and is not promoted.

The real local viewer loaded all 90 person frames. Frame stepping at 0/44/89, continuous playback
and pause, a matched 0.53-body-height lateral comparison, and a 1.06-body-height sideways walk
worked on desktop. The road seam and stretched side surfaces were still visible. This is not
headset evidence, an exhaustive collision test, or acceptance of the remaining placement failures.

The input and output world SHA-256 values are
`12a1f4a02f8dc4c242406d92ad1daaf39bedd547897867af0fdd4aeb319fe41d` and
`cb486b9d89ab914da073130de1db1c0789def43fe524eaef105a86170dff2ffe`, respectively.
Private receipts and visual comparisons are retained under
`.context/evidence/masked-color-pr-20261112/`; generated assets are not committed.
