# Experimental surface-spread regularization

The forest refinement improves source-image matching while moving and mixing the supporting
road surface. A separate, opt-in training term penalizes increased depth spread at fixed image
locations. It is an experiment, not an accepted scene correction or a walkability guarantee.

## Why compare each location with its initial render?

The previous objective combines depth bias and spread in a mean log-depth residual. A large
improvement in bias can outweigh worsening spread. Merely penalizing the average of the worst
10% of absolute spreads is also insufficient: improvements in a few extreme errors can outweigh
regressions elsewhere. On the retained forest's five diagnostic views, that latter proposal
missed three of five regressions on mask-clear, fully supported pixels.

The new term compares each location with its own initialization before aggregating. It renders
alpha-composited first and second moments of linear depth divided by an explicit native-unit
ruler. Given the alpha-normalized variance `v`, its smoothed standard deviation is
`sqrt(max(v, 0) + 1e-6) - 0.001`. Each location contributes only the positive increase over its
frozen initial value. The loss averages the largest `ceil(fraction * count)` increases. A decrease
at one location therefore cannot cancel an increase at another.

Selection remains fixed by source depth support and the person-exclusion mask. Current low or
zero opacity does not remove a location from that set. Empty selections return a differentiable
zero for optimization, with zero counts and an unknown reported mean. References are detached
from gradients, generated before optimization, and separated into training and held-out views.
Held-out references are used only for evaluation. The initialization and camera file hashes,
ruler, reference counts and setup time are recorded alongside the run's existing depth provenance.

## Controls

`scripts/finetune_gsplat.py` exposes:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--lambda-surface-spread-regression` | `0` | Weight of the new term; zero retains the existing training path |
| `--surface-spread-unit` | Unset | Explicit positive ruler in native coordinates; use measured body height for body-height comparisons |
| `--surface-spread-tail-fraction` | `0.1` | Fraction of fixed selected locations contributing to the largest-increase mean |

Enabling the term also requires positive `--lambda-surface-depth` and
`--surface-alpha-weight`, retaining target-depth and coverage penalties. Pseudo views receive
no spread-regression supervision. No pipeline default, quality threshold or paid retry allowance
changes. This standalone trainer still requires the normal budget and ownership controls before
any provider launch.

## Evidence and limits

An independent NumPy replay on saved CPU-render summaries gives zero loss at initialization and
positive loss for the rejected corrected forest candidate in all five fixed diagnostic views:
0.7122, 0.5677, 0.4638, 0.4502 and 0.4560 body-heights. The production scalar matches within
`6e-15` when supplied with the same reconstructed float64 moments. This verifies the calculation
and its ability to detect that saved regression; it does not verify CUDA moment accumulation,
optimization improvement, unseen geometry or visual quality.

The initial world is a regression reference, not physical ground truth. A coherent surface can
move to the wrong height without increasing spread. A transparent region has no established
surface even when its mathematical spread is zero; the reference records that limitation.
Coverage penalties do not guarantee opacity, and preserving spread does not guarantee foot
contact. Source-depth, coverage, all fixed foot queries, source-matched images and real-viewer
checks remain independent acceptance requirements.

Moment accumulation in the GPU renderer uses float32. Promoting the subsequent variance
arithmetic to float64 cannot recover precision already lost there. Normalizing before squaring
reduces scale problems but does not eliminate cancellation for distant, thin surfaces.
Initial reference rendering adds setup time and memory when enabled. None is added by default.

Local evidence is under `.context/evidence/surface-tail-20261111/`. Sources, previous rejected
candidates, evaluation references and live run state remain unchanged.

## Approved forest trial — November 11

The explicitly approved fourth fine-tune executed once at commit `b9c15f8`, starting from the
original world and the same corrected export as trial03. It used 800 steps, spread weight `1`,
native body-height ruler `0.6345488724524215` and tail fraction `0.1`. All other optimizer
arguments, source cameras, evaluation rays, masks, foot positions and thresholds stayed fixed.
No world was regenerated. This is one measured configuration, not a parameter sweep.

| Fixed check | Previous trial03 | New trial04 |
| --- | ---: | ---: |
| Alpha-rendered depth ratios | 5/5 | 0/5 |
| Opaque coverage retention | 4/5 | 5/5 |
| Depth-spread non-regression | 0/5 | 5/5 |
| Foot support | 41/90 | 58/90 |
| Foot contact | 31/90 | 47/90 |
| Held-out RGB improvement | 7/7 | 7/7 |

Spread p90 fell to 54–63% of the original world's values across both fixed pixel groups.
Coverage retained 99.013–100% of original opaque pixels. However, all five depth-ratio p10
values remained between 0.827 and 0.841, below the unchanged 0.9 limit. Foot-contact error
was 0.0370 body-heights at the median, 0.1144 at p90 and 0.2232 at worst. The original world
passed all 90 foot queries. Neither trained candidate is accepted.

With the original evaluation masks and quantized renders, mean held-out PSNR improved from
16.9514 to 28.5318 dB, 0.2994 dB below trial03. The separate trainer float/fresh-mask score
was 28.5413 dB. Actual GPU reference setup rendered 83 training and seven held-out views in
0.935 seconds, caching 182.9 MB. Initialization and camera hashes matched, all initial held-out
regression values were zero, and no selected reference pixel had near-zero alpha. The seven
final diagnostic tail means were 0.00172–0.03739 body-heights; these are not acceptance scores.

The real browser viewer loaded all 90 person frames and about 800,000 world splats at its
current level of detail. Play/pause, source inset, final-frame stepping and lateral movement
worked. The source is silent and correctly showed no soundtrack. Static source-camera renders
still contain a person-shaped remnant, and the viewer retains a road-texture seam and distorted
side-view surfaces. Lateral views at approximately 0.53 and 1.57 body-heights are navigation
checks, not source-camera matches or headset/collision validation. The preceding avatar
placement failure is also unchanged.

Offline localization retained all 272,075 original view-ray observations: 95.19% of trial04's
ratios below 0.9 were already below 0.9 in the original world, and 81.43% lie in the furthest
estimated source-depth quartile. On 39,642 rays that were in-band in trial03 and near again in
trial04, median ratio changed from 0.8302 originally to 0.9837 and then 0.8386; median spread
changed from 0.1577 to 0.3072 and then 0.1079 body-heights. This supports narrower distributions
around an original-like near mean, not a unique causal explanation or physical depth proof.
The 43 failed feet divide into 23 spread-only, 11 height-only and nine combined failures;
none fails opacity. All 21 localization input hashes were unchanged. These diagnostic ray
labels do not replace the original quantile acceptance rules.

This trial demonstrates a useful reduction in spread, but also the limitation of preserving a
surface without establishing its correct depth and contact. It remains a rejected diagnostic
preview at `/reviews/forest-spread-regression-04/index.html`; the option stays off by default.
Do not repeat this fit or tune against the fixed evaluation feet. A further hypothesis needs
independent evidence about depth bias and supported ground before another paid execution.

Evidence is under `.context/evidence/surface-tail-20261111/trial-04/`. All 79 evaluation input
hashes were unchanged. The one provider call completed, all 30 outputs were recovered and hash
verified, and its app stopped with zero tasks. Worker elapsed time was 112.825 seconds, with
an estimated $0.08309 compute charge excluding startup, build, storage and egress. The full
$0.75 reservation remains pending billing reconciliation: cumulative conservative usage plus
reserves is $9.90242 of the original $10, not a final invoice. Outputs remain on the private
provider volume and locally; shared-S3 archival is blocked by unavailable author credentials.
