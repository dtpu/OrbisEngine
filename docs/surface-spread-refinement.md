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
