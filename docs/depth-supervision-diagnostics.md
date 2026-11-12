# Depth supervision: masking and retained-scene diagnostics

The ordinary expected-depth objective must ignore unsupported and person-masked pixels before
applying logarithms. Multiplying a residual by zero afterward does not remove an infinity or
NaN. `expected_depth_loss` now selects `supported & valid` first, then computes the mean absolute
log-depth residual. Finite predictions retain the existing 0.05 lower clamp. Opacity does not
change the selected set. Invalid selected targets or nonfinite selected predictions raise an
error; an empty selection yields a differentiable zero.

Source-depth preflight runs when either `--lambda-depth` or `--lambda-surface-depth` is enabled,
before CUDA allocation or output creation. The existing nearest-resampling and source-mask rules
are unchanged. No loss weight, model, quality threshold or pipeline default changes.

## Retained-frame contract check

A full CPU render of the retained forest world at training sample 1 supplied predicted depth
at 672×378. The original export supplies 154,403 selected pixels and 99,613 excluded pixels.
The unchanged ordinary depth loss is 0.0659093484 before and after the correction.

Replacing only excluded target values with zero, NaN or infinity in memory makes the old scalar
nonfinite. The corrected scalar remains exactly 0.0659093484 in all three cases, all prediction
gradients remain finite, and excluded prediction gradients are exactly zero. Unit tests also
cover invalid selected targets, the preserved clamp, empty support, actual training-loop
weighting, pseudo-view exclusion, and rejection before device allocation.

The normal exporter usually fills unsupported holes with finite positive values. More importantly,
the retained fourth forest fit used `--lambda-depth 0`: this latent contract bug did **not** cause
its geometry failure. This check is not another fit or evidence of repaired floor geometry.

## Precision replay

Fifteen read-only renders cover the original world and retained fits 03/04 at the same five fixed
camera views. The old float32 alpha, mean-depth and spread arrays reproduce exactly. Parallel
float64 raw moments and weighted online variance agree within the predeclared numerical checks.

The stable calculation uses the actual sum of contribution weights instead of float32 `1−T`.
Consequently this comparison measures combined accumulation and normalization error, rather
than isolating subtraction cancellation. On the original-compared pixels, the largest change
in a view's spread p90 is 0.0001432 body-heights; no depth-ratio band label changes. The main
depth and spread conclusions remain the same. Existing feet, source depths, masks, denominators,
thresholds and acceptance verdicts are retained.

These are CPU diagnostics, not validation of CUDA accumulation or historical optimizer gradients.
Improved numerical precision does not explain away the rejected forest scene. No new scene,
provider execution, world generation, placement fit or promotion occurred.

## Endpoint objective check

One previously identified, still-supervised ground contributor was evaluated at the saved
unquantized endpoint. The 26 source-supported training views were selected by the preceding
exposure audit and held fixed. Central differences move only that point by ±0.001 body-heights
along each native axis; they do not optimize a scene or use evaluation feet as targets.

Moving the projected footprint changes the objective differently from changing its depth feature
while keeping all compositing weights fixed. On the native-y axis, the full log-depth derivative
is +0.000032589 per body-height, while the frozen-weight derivative is −0.000000033246.
RGB L1 and SSIM oppose much of the full depth derivative, leaving a full objective derivative
of +0.0000035849 after coverage and tether. The spread-regression scalar is unchanged across
all 156 perturbations. This does not establish why it was unchanged or its earlier training role.

The accelerated local rerender exactly matches a full rerender on the declared validation view.
Synthetic scalar components match the historical trainer within 8.9e−16. All 26 raw endpoint
views still have source-supported contribution, and all 63 inputs remain unchanged. Execution
took 91.9 seconds with no provider calls.

This is a CPU endpoint approximation over 26 views, not the complete 83-view objective, CUDA
gradient history or a proposed update. Finite-difference cutoff crossings can be nonsmooth;
no step-size sweep was performed. Native coordinate derivatives do not independently establish
physical height corrections. The result rules out relying on a depth-feature-only argument
for this point; it does not yet justify another loss weight or a scene-quality claim.

Private scripts, hashes and measurements are retained under
`.context/evidence/depth-precision-20261112/`. The first masking smoke check used a sparse retained
raster and is explicitly preserved as `masked-depth-replay-sparse-fixture.*`; the full-frame
result above is recorded by `masked-depth-replay.json` and `full-training-frame-receipt.json`.
