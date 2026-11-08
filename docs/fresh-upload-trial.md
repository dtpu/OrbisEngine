# Fresh upload trial — November 8, 2026

A new video reached a playable 3D scene through the dashboard, production worker and live
providers. The result is a retained **preview with failed quality checks**, not an accepted
reconstruction or a first-pass automatic success.

The source is Pat Whelen's public [forest walk footage](https://www.pexels.com/video/man-walking-on-road-among-pine-trees-5738706/):
7.424 seconds, 1920×1080, one fully visible person, no soundtrack. Its SHA-256 is
`d2e69e5d13cb784183a9eec840a2df4c732c81e3b58d17eb37d6f9e64d232755`.
The full source was used without a trim. This source had no previous reconstruction baseline.
Options were video-conditioned world generation, one person, no objects and no fine-tuning.

## Execution and recovery

| Part                                      | Observed result                                                                                                                                     |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dashboard upload and worker discovery     | Correct source bytes and options retained; one persistent admission                                                                                 |
| Source description                        | First request timed out after 180 seconds; the failure and possible charge remain recorded                                                          |
| Changed description request               | Same model and six source frames, explicit low reasoning and a 300-second socket timeout; completed in 15.4 seconds                                 |
| Recovery                                  | Operator verified source/request/result hashes and imported that actual description as a separate attempt; no failed history or allowance was reset |
| Person removal                            | First GPU job completed; all 89 frames decoded and received agent triage, with additional source/clean spot comparisons                             |
| Cameras and avatar                        | First camera, frozen-avatar and motion jobs completed; 90 animated samples packaged with matching source ordinals and retained frame hashes         |
| World generation                          | First and only generation completed; 1,600 provider credits consumed                                                                                |
| Scale fitting                             | Failed the unchanged depth-spread check in all five evaluated views                                                                                 |
| Placement, anchors and final verification | Not run because scale fitting failed                                                                                                                |

The automatic run originally ended failed. Recovery reused its completed work and resumed the
same run; it did not repeat cleaning, camera solving or avatar generation. This distinction
matters when assessing whether an unattended new upload has been proven to succeed.

The live trial exposed and prompted fixes for scoped reconstruction caches, bounded source
description requests, selecting Modal from the active locked environment, and preventing new
paid work after a required world prerequisite has failed. The submission CLI also printed the
internal admin port; it now links to the unified app on port 5399, where viewer links resolve.

The motion diagnostic also found that the receipt's supplied-intrinsics flag depended on an
unrelated fixed-scale option. New receipts count fresh and retained poses from the actual
sample records and set that flag only when fresh poses used the supplied cameras. The saved
trial had 90 fresh poses despite its old flag being false. This is a provenance correction;
its historical receipt and generated geometry were not rewritten.

## Viewer and quality evidence

The real run-scoped viewer loaded all 90 samples. Playback, pause, frame stepping, source-video
display and nearby forward/sideways movement worked. The silent source was correctly labelled
as having no soundtrack, and no console errors or warnings were observed. The starting view
and inspected nearby views show a recognizable gravel road and conifer forest. Beginning,
middle and end screenshots were retained, along with walking-mode checks.

These delivery checks do not establish geometric acceptance:

- At the fitted world scale, the five depth-ratio medians were 0.981–1.013, but the tenth
  percentiles were 0.842–0.848, outside the required 0.9–1.1 band. This is the existing
  nearest-center diagnostic, not opacity-rendered surface depth.
- Independent source-camera projections of the actual packaged Gaussians at source frames
  0, 90 and 177 had silhouette IoU 0.526, 0.568 and 0.513. Their lower edges extended at least
  0.323, 0.260 and 0.201 source body-heights below the observed boots. Two projections reached
  the image boundary, so those height errors are lower bounds.
- Raw and packaged projections matched. The frozen reference fit its own observed source mask
  well, while the neighboring motion sample became oversized. This localizes the dominant
  mismatch to the motion-stage camera-space person estimate, before packaging. Changing the
  existing uniform registration scalar cannot fix its source-camera projection.
- The filled road behind the removed person is inferred. The unseen front of the person,
  off-path geometry, full walking support and headset behavior remain unverified. Agent clean
  triage is not human acceptance. Decoder-version differences also prevent a claim of exact
  outside-mask RGB preservation from the local comparisons alone.

The run remains failed with its preview available. No tolerance was relaxed and no human
approval was invented. The previously retained bottle-scene regression separately verified
reopening an identical upload, both people, original audio and playback without provider calls;
it does not establish fresh reconstruction quality for that source.

An offline follow-up supplied the three measured source masks to the existing silhouette
placement solver, with its weights and gates unchanged. The resulting similarity kept the
original articulation and improved those three in-sample IoUs to 0.839–0.847; boot-edge errors
fell to 0.024–0.027 source body-heights. It still failed the camera-height check at 1.303 fitted
body-heights above the generated plane, and retained the contact noise-floor refusal. The
other 87 samples had no mask observations in this fit. Five subsequent held-out frames passed
the same image thresholds without refitting: IoU 0.804–0.881 and maximum boot-edge error
0.072 source body-heights against a 0.08 limit. Another 82 frames remain unmeasured. This is a
promising placement diagnostic, not an accepted transform. A separate localhost viewer labels
the experiment and retained failures; the run's package and default preview remain unchanged.

## Measured placement continuation

A subsequent local-only investigation evaluated all 90 frames of that frozen sparse fit.
All silhouette IoUs exceeded 0.75, but ten frames exceeded the unchanged 0.08-body-height boot
limit. This exposed a default-path omission: only tracked people supplied measured silhouettes
to placement. Single-person uploads skipped the correction entirely.

The corrected single-person path records exact source-indexed, undilated observations without
inventing a tracker identity. One candidate used all 90 FFmpeg-decoded observations, with solver
weights and guards unchanged. Separately OpenCV-decoded observations at the same timestamps
gave mean IoU 0.8465, minimum 0.8057, and 89/90 boot-edge passes. These share the source/model
and are not held-out timestamps or segmentation ground truth. Sample 21, explicitly declared
critical before the solve, measured 0.08209 against the 0.080 limit. The candidate therefore
remains failed despite the improvement; the threshold was not rounded or relaxed.

An independent query of the generated road's Gaussian footprint gave supported contact at all
90 fitted foot centers: absolute residual p90 0.02696, maximum 0.05718 body-heights against the
fixed 0.08 bound. This is a local support diagnostic, not sole contact, collision or physical
scale evidence. Replacing the global floor with local rendered support did not explain the
camera-height failure: the preceding candidate measured approximately 1.303 and 1.306
body-heights respectively. Ground directly below the recorded cameras has no source point-hull
support, and gravity was inferred from the road without walls. The universal handheld-camera
prior conflicts with the source-fitting body, but physical camera height remains unverified.

The original world-depth spread, final size and contact-noise guards still fail. No new provider
request, generation, quality override, live-run mutation or promotion occurred. The actual
single-person extractor, exact cache reuse, solver and all-frame projections were exercised
locally. The automatic pipeline still stops at the preceding scale failure for this run.
Evidence is under `.context/evidence/ground-alignment-20261108/`, including the preserved failed
candidate, both source-mask archives, full-sequence comparisons and local support measurements.

A subsequent local depth audit kept the original five cameras and compared source texels, and
rendered all 96,000 pixels per view with covariance-aware alpha compositing. The lower-tail depth
ratios remained outside the unchanged band: p10 0.816–0.829 instead of the center diagnostic's
0.842–0.848. The existing renderer with packaged cameras independently reproduced the failure.
All generated pixels had high opacity, while source-depth support covered only about 60–63% of
each image. This rules out the nearest-center approximation as a sufficient explanation; it
does not establish Pi3X as physical ground truth. No replacement scale metric or quality override
was introduced. Source-matched images and numerical parity checks are in `world-depth-audit/`.

The continuation also found that the viewer applied recorded placement offsets inside optional
foot effects. Disabling those effects or switching to discrete frame meshes lost the offsets
while retaining the fitted body size. Offsets now apply independently in both playback modes;
additional stance correction defaults off when a measured placement table is present. The real
viewer showed matching displayed person transforms at frames 0 and 25 across these modes,
loaded all 90 samples, and passed pause, stepping, final-frame looping and nearby walking checks.
Discrete mode still uses a different existing walk-eye stature estimate, so these were not
identical camera renders. The diagnostic remains visibly labelled failed. Its iframe wrapper
emitted an intermittent MutationObserver error without a source or stack; no control failure was
observed, and the cause remains unverified.

## Cost and retained evidence

All four owned Modal apps stopped with zero tasks; the admission watcher was stopped after
the terminal outcome. The shared ledger retained 114 settled claims, including its historical
failures. Provider credit usage was 1,600 for the world (US$1.28 at the checked rate). The
account-wide Modal metered increase was US$0.1724, and the completed description used
US$0.06782 in recorded tokens. The timed-out description's charge remains unknown. Gross
metered usage, reporting lag and that uncertainty are retained within the original US$10 cap;
free account credits are not treated as permission to spend more.

Media, provider receipts, screenshots, hashes, local diagnostics, original failed-state snapshots
and the budget reconciliation are ignored local evidence under
`.context/evidence/practical-e2e-20261108/fresh-run/`. They are not committed. The failed run's
outputs remain under `.context/app/runs/pexels-5738706/`; no candidate was promoted or published.
