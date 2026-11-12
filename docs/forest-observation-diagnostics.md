# Forest observation and local color diagnostics

After PR #99 merged at `23ab20ca2ea0246570d02f025082c4d357e4034c`, offline investigation and
two local color probes narrowed the remaining failures. The second probe reduces the frozen
upper-body color patch, but neither removes the static person remnant or establishes a useful
end-to-end reconstruction. No pipeline defaults, training objective, source masks, geometry,
acceptance thresholds or live run state changed.

## Retained acceptance

| Candidate                            | Depth checks | Coverage checks | Spread checks | Fixed foot contact |
| ------------------------------------ | -----------: | --------------: | ------------: | -----------------: |
| Previous training trial 03           |          5/5 |             4/5 |           0/5 |              31/90 |
| Training trial 04, PR #99 experiment |          0/5 |             5/5 |           5/5 |              47/90 |
| Local color offset probe 01          |          0/5 |             5/5 |           5/5 |              47/90 |
| Local color replacement probe 02     |          0/5 |             5/5 |           5/5 |              47/90 |

Both color probes preserve every decompressed SPZ byte outside the color stream. Each candidate
also reproduces the five saved alpha, depth and spread arrays exactly. The 90-contact result is
verified reuse from byte-identical geometry/opacity, bound to the prior world and contact receipt;
it is not a new placement fit. Existing avatar-placement failures remain independently unresolved.

## Frozen appearance and the local probes

At trial 04's stored endpoint, 146,012 training-frustum-visible points have color and position
exactly unchanged from initialization. Occlusion-aware CPU replay of all 83 training views finds
129,993 with no rendered contribution, 16,018 rendered only inside the person mask, and one with a
tiny mask-clear RGB contribution. None has mask-clear depth-supported exposure. This is endpoint
evidence, not a claim that an ID never received a historical CUDA gradient.

In three retained source views, virtually all unchanged-color mass within the supplied person
footprint comes from the always-masked category. It contributes 8.85%, 14.47% and 26.24% of the
total rendered weight averaged over that footprint. The mask is a diagnostic region, not an
independent semantic segmentation of every visible ghost pixel.

Two predeclared, local, color-only hypotheses used those 16,018 always-masked IDs as targets:

1. Transfer nearby donors' learned RGB adjustments while retaining each target's initial color.
   This changed 10,603 packed color records without establishing visible ghost removal.
2. Replace the target color using a per-channel weighted median of nearby visible donor colors.
   This changed 11,095 records and reduced the frozen upper-body patch in matched comparisons,
   but left a recognizable outline and leg-shaped remnant. The resulting colors are inferred
   appearance, not recovered observations behind the person.

Both methods query 64 nearest changed-color, training-visible IDs in retained native coordinates,
within 0.25 frozen body-heights. They then filter those candidates to IDs with actual mask-clear
RGB contributions in at least three of the 83 training views. Up to 16 donors are used, with a
minimum of three. Querying before exposure filtering is conservative truncation: eligible donors
beyond the first 64 are not considered. The spatial Gaussian sigma is 0.125 body-heights. Probe 01
also uses an initial-RGB similarity kernel with sigma 0.1; probe 02 omits that kernel because the
target's unobserved initial texture is the suspected defect. Neither selects donors using heldout
RGB frames or evaluation feet. The second probe reuses the first probe's verified exposure data.

Seven frozen heldout images/masks were compared with both worlds rendered through the same CPU
evaluator at 672×378. Probe 01's PSNR deltas were zero; probe 02's largest change was
−0.0000001744 dB on one frame. These are non-regression diagnostics, not substituted CUDA trainer
PNG acceptance. The five fixed source views and three identical lateral-camera comparisons
remain visibly imperfect. The original source is silent; no sound was invented.

## Ground exposure and the movement cap

Only five of 1.92 million trial-04 points ended near the existing 2-body-height displacement cap.
None of the 78,578 contributors to its failing near-depth rays moved even 1.9 body-heights.
Trial 03 passed the depth checks under that same cap. The retained results do not justify a
blanket cap increase, although conditional displacement bounds can matter for individual rays.

The full-footprint ground audit distinguishes height failures from spread-only failures.
Median support mass without any mask-clear depth-supported endpoint contribution is 93.69% for
height-only failures and 96.68% for combined failures, versus 0% in passing and spread-only
cohorts. There are both supervised failures and unexposed passes; missing exposure is neither
necessary nor sufficient for a single bad contact.

A further diagnostic interpolated the full saved original and trial-04 scene at five fractions,
replaying all 83 source training cameras. This is a counterfactual path, not the optimizer's
history. Both endpoint replays reproduce every prior metric and center label exactly. For
preselected IDs 1275811 and 896045, clear-supported footprint exposure disappears between
fractions 0.5–0.75 and 0.75–1 respectively. At the final endpoint their entire potential footprint
alpha lies inside the fixed person masks even before occlusion. Removing occluders would not
restore a usable loss pixel for those footprints.

Actual exposure first increases as occlusion eases, despite shrinking mask-clear footprint area.
A simple count of visible views therefore misses the distinction between available footprint
support and transmittance. Control ID 208732 retains clear-supported contribution in 26 final
views and still fails height. Detecting complete loss of prior mask-clear potential is supported
as a diagnostic trigger; no visibility-preservation guard is validated as a repair, and the
original generated surface is not established as physical truth.

## Evidence and limits

All source, criteria, camera, mask and original/candidate inputs remain unchanged. Independent
review checked target/donor isolation, SPZ offsets, byte preservation and weighted medians. No
provider calls or new forest training executions occurred: the four paid fits and original $10
aggregate allowance remain in force. Neither local probe was promoted or integrated into the
automatic upload path. No headset or comprehensive collision acceptance was established.

Private evidence, scripts, receipts and comparisons are retained outside Git:

- `.context/evidence/forest-observation-20261112/REPORT.md`
- `.context/evidence/forest-hillclimb-20261112/REPORT.md`
- `.context/evidence/forest-hillclimb-20261112/color-01/`
- `.context/evidence/forest-hillclimb-20261112/color-02/`
- `.context/evidence/forest-hillclimb-20261112/ground-path/REPORT.md`
- `.context/evidence/forest-hillclimb-20261112/review/`
- Local previews: `/reviews/forest-color-completion-01/index.html` and
  `/reviews/forest-color-completion-02/index.html` on the existing localhost viewer.

The next correction must address the remaining contaminated appearance and the independent
geometry/placement failures. Repeating either color probe unchanged, increasing the movement
cap, or preserving exposure alone is not supported as an end-to-end fix by these results.
