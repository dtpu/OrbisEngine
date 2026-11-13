# Known limits

Approaches already measured to fail.
Do not retry one without a new idea, and add a line here when you measure a new dead end.

- The forest scale gate fails on distant background, not on the route. Replaying the run's own
  scale diagnostic (unchanged inputs, scale0 0.7414, whole-frame p10 0.842–0.848) split by each
  frame's source-depth quartiles: rays within about 4 body-heights mostly stay inside the unchanged
  0.9–1.1 band (one marginal p90 of 1.102), while the median falls steadily with distance from
  about 1.04 to 0.88 beyond about 7.5 body-heights. A distance-dependent disagreement between the
  generated world and Pi3X cannot be fixed by any single scale, and fitting the world to the far
  Pi3X depth is what moved the road in the rejected refinements. This does not establish which
  depth is physically right and does not pass the gate; `scale-fit.json` now records these
  `distanceBands` and the failure message quotes them. Evidence:
  `.context/evidence/scale-distance-bands-20261112/`.

- The forest clip's camera is elevated: 1.30 body-heights above the road, so the handheld band
  fails without a scale error. place_fit cuts contact from 28.8 to 2.8 cm RMS (in-sample). The size
  check now passes on a measurement of the footage instead of `--allow-bad-size`
  (`scripts/camera_gravity.py`). Gravity from the frames' own tree trunks, pooled over 12 frames
  (80 inlier segments), sits 0.60° from the fitted floor normal. It gives 1.278 body-heights
  against the registration's 1.303. Dead ends measured on the way:
  - The metric-depth ruler is not evidence here. DepthPro's own field of view implies a focal
    length 1.20× the solve's, and correcting it moves the camera from 2.07 m to about 1.72 m and the
    fitted walker to about 1.32 m. The earlier "agrees" was not a measurement.
  - A horizon test from the registration's tilt alone reproduces the fitted 1.303 exactly. The
    silhouette fit already enforces that geometry, so only an independent tilt adds evidence.
  - Single-frame vanishing points on the forest are too noisy: 0.80–1.76 body-heights from 14–26
    segments. Pooled in the track frame, they agree with the registration within 2%.
  - On the lobby and elevator clips, the same estimator found the 9.6° and 8.0° track tilts that
    `frame_align.py` corrects (9.8° and 7.6°). It agreed with the frame-aligned tracks within 0.6°,
    so it does catch a wrong tilt.

  Metres still assume a 1.70 m walker. Do not widen the handheld band or move the floor: a camera
  outside the band passes only when the footage measures it there. With `--allow-scale-spread`
  and `--allow-weak-anchor`, place_fit, anchors and verify pass on a copy of the run. Anchors still
  records four failures:
  - ruler: only one point anchor;
  - silhouette area;
  - contact noise floor;
  - observation mask.

  Evidence: `.context/evidence/forest-size-check-20261113/` and
  `.context/evidence/forest-operator-e2e-20261112/`.

- Baking source-video colours onto the forest world (`bake_video_colours.py` without `--diag`)
  replaces the generated lime-green verges with the source's dry olive grass. It recolours 27% of
  splats, whose colour std across frames has median 0.023. Pass the Pi3X `cameras.json` beside the
  anchors, not the packaged one: the packaged file lacks `framealign.json`, so splats map into the
  wrong frame and only 5–8% are observed. On its own the bake leaves a hard straight edge where
  the source frustum ends. The opt-in `--shift-radius-body-heights 1` carries nearby observed
  colour shifts onto unobserved splats, fading to zero at one body-height (inferred, recorded as
  `inferredShift` in `-bake.npz`), and removes that edge. Neither run is enabled in the pipeline,
  and both leave real defects:
  - mid-distance canopy smears;
  - a generated berm on the far verge now reads as a tan mound;
  - geometry, depth and contact are unchanged.

  Tighter `--tol`/`--front` (0.1) or full-resolution decoding did not fix the canopy smear. Evidence:
  `.context/evidence/forest-colour-bake-20261113/`.

- The forest canopy smear comes from depth disagreement, not decoding or splat size. Canopy
  splats sit at about 0.83× the source depth (p10 0.56), while the ground agrees (median 1.04). A
  splat off the source surface slides across the image as the camera moves, so averaging all 90
  frames paints bright strokes onto it. Its colour still looks consistent (std median 0.033).
  `--tol`/`--front` cannot fix this: they reweight each splat's frames similarly, and the weights
  cancel in its average. The opt-in `--depth-agreement 0.1` bakes only splats whose weighted mean
  |log depth ratio| is within the scale gate's ±10%; the rest keep the generated colour
  (`depthGated` in `-bake.npz`). With it, 82,819 splats keep their generated colour, and the canopy
  matches the generated world in the viewer (detail 2.48 vs 2.63 left, 4.48 vs 4.56 right; the
  ungated bake measured 3.24 and 6.01). The verges keep the source tan. Trade-offs:
  - a rock pile and the far roadside near the crest keep their generated bluish and lime
    speckles, because their depth disagrees too;
  - baked ground still has less fine texture than the generated one;
  - geometry, depth and contact are unchanged.

  Evidence: `.context/evidence/forest-canopy-20261113/`.

- The forest's leftover lime roadside and bluish rocks are depth-gated splats whose hue is
  wrong, while their brightness and detail are fine. Two neighbourhood tints failed:
  - averaging the observed colour shift over 1 body-height moved the rocks toward more blue;
  - restricting it to depth-agreeing splats of similar generated colour doubled dark-canopy
    brightness (0.13 to 0.28), because "dark generated" mixes canopy with bright gravel.

  Local source contrast did not separate canopy from rocks or roadside (median 0.031–0.033), so
  it cannot decide which drift is safe. The opt-in `--gated-chroma` keeps each gated splat's
  generated luminance and scales its observed colour to it, borrowing only chromaticity. With
  `--depth-agreement 0.1 --shift-radius-body-heights 1`, 92% of gated lime splats (4,612 of 5,036)
  and 62% of gated bluish ones (817 of 1,315) lose that hue. The crest and rocks read grey-olive
  and grey at the source pose. Two guards matter:
  - Adding a chroma offset, rather than scaling, tinted dark canopy-top splats purple, because
    they mostly see magenta overcast sky (viewer purple index +0.06–0.07 against −0.02).
  - Splats whose observed luminance differs by more than 2× from the generated luminance now keep
    their colour (22,420 of 82,819 gated). That returns the canopy top to −0.008.

  About 14,000 lime splats the source never observes keep their generated colour. Geometry,
  depth and contact are unchanged. Evidence: `.context/evidence/forest-gated-tint-20261113/`.

- More accurate CPU depth-moment arithmetic does not explain the retained forest failures.
  Fifteen original/trial03/trial04 replays preserve the old arrays exactly; parallel float64
  accumulation and normalization change spread p90 by at most 0.0001432 body-heights on the
  fixed compared pixels and change no depth-ratio band labels. No acceptance verdict was
  replaced. See [the depth-supervision checks](depth-supervision-diagnostics.md).

- Local color completion alone did not remove the forest's static person remnant. Transferring
  nearby visible points' color adjustments preserved the defective texture. Replacing the
  unobserved core with nearby donors' weighted-median colors reduced that core, but the outline
  and leg-shaped remnant remained. Both local probes preserved every non-color SPZ byte;
  depth still passed 0/5 and fixed foot contact 47/90. Seven same-CPU RGB comparisons were
  effectively unchanged; they are not new CUDA trainer acceptance. These inferred colors are
  not source-observed background. Neither probe is promoted or enabled in the pipeline.
  Separately, two raised-support contributors lose all mask-clear potential footprint exposure
  at the saved endpoint, while a supervised control still fails height; preserving visibility
  alone is not a validated repair. See [the diagnosis](forest-observation-diagnostics.md).

- A third local color candidate selected all rendered points with zero mask-clear exposure,
  including points changed during training, and filtered visible donors before finding neighbors.
  This reduced the bright torso-shaped remnant more than the unchanged-point probes, but a dark
  lower-body patch, road seam and lateral distortion remain. All non-color bytes stayed identical;
  depth remains 0/5 and foot contact 47/90. The reusable
  [completion tool](masked-color-completion.md) is opt-in and does not establish scene acceptance.

- Penalizing per-pixel depth-spread increases from frozen initialization improved the forest's
  spread checks from 0/5 to 5/5 and coverage from 4/5 to 5/5, but depth-ratio checks regressed
  from 5/5 to 0/5. Foot contact improved from 31/90 to 47/90, still below the original world's
  90/90; a static person remnant and road seam remained. This explicitly approved fourth fit
  used weight 1 with all previous inputs and acceptance criteria unchanged. Reduced spread
  does not establish correct surface depth or contact; repeating this configuration is not
  justified. No further paid execution is authorized. Evidence:
  `.context/evidence/surface-tail-20261111/trial-04/` and
  [the measured report](surface-spread-refinement.md#approved-forest-trial--november-11).

- Applying conservative masks and full-window SSIM exclusion to the forest refinement did not
  remove the static person outline or repair the road. All seven fixed RGB holdouts improved,
  but all five depth-spread checks failed and only 31/90 unchanged foot contacts passed.
  The fresh export also removed 275 source-anchor points, so this is a paired exclusion test,
  not an isolated SSIM experiment. All three ordinary fine-tune executions are consumed,
  including a failed app initialization; do not repeat the same fit. Evidence:
  `.context/evidence/forest-refinement-next-20261109/trial-03/`.

- The forest refinement's foot-contact regression is actual supporting-splat movement, not a
  coordinate or SPZ import error. Holding horizontal positions fixed still leaves only 58/90
  contacts passing, while horizontal movement alone retains 89/90. All 90 queries remain opaque.
  Average depth error improves while the worst depth spread increases, including on mask-clear,
  source-supported pixels; replacing log depth with a mean linear-depth loss alone is therefore
  not an established remedy. Preserve the fixed feet and evaluation rays. A separate penalty on
  the worst surface spread is a proposed experiment, not a validated fix, and cannot by itself
  guarantee the correct height of a coherent surface. Evidence:
  `.context/evidence/forest-refinement-next-20261109/surface-cause/`.

- A raw-depth refinement of the forest world improved all seven RGB holdouts and passed all five
  alpha-rendered depth-ratio checks, but worsened within-pixel depth spread by 20–33%, lost too much
  opaque coverage in two unsupported regions, and reduced unchanged foot-contact passes from
  90/90 to 49/90. A person-shaped outline remained in the static world despite source-mask exclusion.
  Source-matched image scores and median depth agreement do not establish a coherent supporting
  surface. Keep this candidate failed; do not refit the feet or select favorable pixels to erase the
  regression. The ghost's cause remains unproved. Evidence:
  `.context/evidence/source-geometry-20261108/forest-refinement-01/`.

- Correct avatar appearance does not establish motion depth. On a fresh forest-walk source,
  the frozen reference matched its observed silhouette (IoU 0.898), but the adjacent animated
  sample was oversized (IoU 0.568) with essentially unchanged native body length. The error
  was already present in the camera-space pose; raw and packaged projections matched.
  A fixed registration scale cancels in source-camera projection. A sparse three-mask placement
  fit improved IoU to 0.839–0.847 and boot-edge error to 0.024–0.027 body-heights, but still
  failed camera-height and contact-noise checks. Five held-out frames subsequently passed the
  image thresholds without refitting, but 82 samples remained unmeasured and the generated
  world's depth-spread gate still failed. Do not promote this fitted
  candidate or reverse the upstream log-depth/focal operation order, which matches its training
  target. Evidence: `.context/evidence/practical-e2e-20261108/fresh-run/pose-audit/` and
  `placement-probe/`.

- The forest-walk continuation measured all 90 sparse-fit frames: ten missed the fixed boot-edge
  limit. A source-indexed single-person fit using all 90 observations improved this to 89/90,
  but a predeclared critical frame still measured 0.08209 against the 0.080-body-height limit.
  Local generated-road support passed all 90 foot-center checks without resolving the size,
  contact-noise or world-depth failures. The source contains no ground support directly below
  the cameras, and its gravity estimate came from the road. Neither a higher replacement floor
  nor a wider handheld-camera band is justified. Preserve missing physical calibration as
  unknown and the candidate as failed. Evidence: `.context/evidence/ground-alignment-20261108/`.

- The forest scale failure is not rescued by replacing nearest-center depth with alpha-composited
  visible depth. At the same five source cameras and original compared texels, rendered p10 ratios
  were 0.816–0.829 against the unchanged 0.9–1.1 band; the separate packaged-camera renderer agreed.
  This establishes disagreement with the retained Pi3X depth estimate, not which reconstruction
  is physically correct. Source support covers only about 60–63% of the diagnostic image, and no
  independently labelled road region was available. Do not treat full generated opacity as observed
  geometry or change the scale gate to make this candidate pass. Evidence:
  `.context/evidence/ground-alignment-20261108/world-depth-audit/`.

- Restricting planar training targets to measured lobby floor support fixed the two declared floor-depth checks (p90 0.00336/0.00627 body-heights), but did not establish scene quality. Glass/reflective walls and lateral appearance still distort, the first-frame shoe remains partly occluded, and one optimization holdout loses image agreement. Portal raw-depth p90 also regressed against the preceding candidate on sparse support. Separate training depth from unchanged evaluation depth, preserve the source/region checks, and diagnose the remaining residuals before another refinement. These floor audit views were training views, not unseen geometry. Globally protecting the same measured point IDs across views then reduced portal p90 error from 0.07267 to 0.05876 body-heights, still worse than the preceding raw-depth fit (0.01815); no clear visual repair followed. Image-region exclusions alone do not protect a shared 3D point in other views, and target protection does not isolate neighboring learned Gaussian footprints. Evidence: `.context/evidence/surface-reconstruction-20261107/trial-13/` and `trial-14/`.

- Jointly refining the corrected lobby room with measured source splats improved image agreement and passed all five fixed scale checks, but did not meet the independent planar-floor criterion. After 4,000 iterations, floor ray-plane p90 errors were 0.02309/0.04002 body-heights against a 0.02 limit; the original depth targets themselves had 0.02297/0.03845 error. Further fitting of those same targets cannot be assumed to remove their nonplanarity. Glass regions remain mostly unsupported, and first-frame shoe occlusion persists. Keep raw-depth agreement separate from independent surface quality; no threshold was relaxed. Evidence: `.context/evidence/surface-reconstruction-20261107/trial-12/`.

- Independently aligned Pi3X batches can export inconsistent cameras even when serialization, basis conversion and source-frame selection are correct. On the retained lobby run, early static RGB correspondences had p90 epipolar errors of 7.09, 15.21 and 28.88 pixels at 672×378. Reattaching the shared anchor cloud through those cameras also warped its geometry. An opt-in correction locks each batch to the original shared anchor poses and interpolates the correction in source time; 11 supported held-out pairs then passed median ≤1 and p90 ≤2 pixels, while two pairs lacked enough correspondences. This diagnoses batch consistency, not absolute camera accuracy or scene quality. Preserve the common anchor transform and regenerate dependent artifacts; do not mix corrected cameras with old placement, depths or animation coordinates. Evidence: `.context/evidence/surface-reconstruction-20261107/`.
- Replacing a measured floor with finite, multiview-supported planar splats did not repair the retained lobby scene. The offline prototype kept observed coverage, but only removed 76.5% of the old floor's rendered contribution against a required 90%, and all inspected floor-depth checks failed. The real viewer showed speckles, patch seams and remaining stretched surfaces. Full-detail rendering and a float-versus-SPZ comparison independently ruled out LoD and compression as the main cause. That floor prototype remains rejected; no runtime floor replacement was added. Evidence: `.context/evidence/surface-reconstruction-20261107/`.

- Refining body pose against the footage makes it worse: a single camera leaves a null space.
- Semantic person masks can omit lower bodies against similar-colored floors: the fresh bottle-source clean left black and light trousers in several sampled frames despite a successful worker exit. Tracked video cleaning now unions the existing per-person instance masks and retains their exact source-frame selection; equal FPS and frame count do not imply the same pixels (the observed 30-to-12 FPS tracker and cleaner selected different source frames). Missing, empty, or incomplete track masks and mismatched source/timing metadata block submission. Full mask coverage is not an inpainting-quality pass; inspect the resulting clean for remnants and damaged surroundings. Failure evidence: `.context/evidence/bottle-source-rerun/clean-comparison.jpg`.
- FFmpeg 5.1 rejects a long flat sum of exact-frame selection terms: the 120-frame clean correction failed before inpainting, and the parser failure reproduced in local Debian Bookworm while FFmpeg 8 passed. Balanced expressions preserve the same source indices and pass the full-resolution local Linux reproduction. Keep decoder stderr in failure reports; a successful local decode does not establish remote inpainting quality. Evidence: `.context/evidence/bottle-source-rerun/linux-decoder-check/`.
- Complete tracked person masks can still omit a carried backpack. In the fresh bottle clean, input and output masks matched on all 120 frames, but four inspected backpack regions retained 17–42% unmasked source pixels. A local union with `moved_content_masks` covered those regions but also classified static stair posts and railings as people; its residual-area guard cannot reject semantic person false positives. Do not promote that union as a cleaning fix. Evidence: `.context/evidence/bottle-source-rerun/balanced-clean-review/`.
- Faces below about 64 px of face height are unreadable; per-clip head refits do not help.
- Broader shadow-aware masks are not a general cleanup fix. On the lobby source, the baseline removed parts of bins and a fountain and retained limb traces; enabling moved-content masks with larger dilation removed more fixed fixtures. Both outputs failed review, so world generation stayed blocked at the shared three-execution inpainting cap. Evidence: `.context/evidence/fresh-ten-dollar/clean-comparison-*.jpg` and the run's `clean-review.json`.
- An explicitly approved fourth lobby clean used exact-frame Mask R-CNN instance masks with two-pixel padding. It preserved the sampled bins and fountain, but LaMa left prominent hair, torso, hand, trouser and shoe remnants throughout the 115-frame output; reflected people also remained. Small accurate masks alone are not a person-removal fix. Keep this candidate failed and inspect mask coverage, model context and compositing before another hypothesis. Evidence: `.context/evidence/clean-diagnosis-20261105/instance-clean-review/`.
- An explicitly approved fifth, first-frame-only probe neutralized masked RGB before resizing, covered the filter support in the model mask, and composited fully inside the original mask. Large head/body/trouser ghosts improved substantially, but outlines and shoe remnants still failed review. Context isolation does not repair missing removal-mask coverage or guarantee a plausible fill; no full-video retry or world generation followed. Evidence: `.context/evidence/clean-diagnosis-20261105/isolated-probe-review/` and `isolated-probe-diagnosis/`.
- An offline sweep over eleven difficult lobby frames tested twelve combinations of mask probability (0.5/0.25/0.1), person score (0.5/0.15), and padding (2/4 pixels). No combination covered all marked fragments and reflections while preserving the fixed fixture regions. At probability 0.5, four-pixel padding still missed 340 marked outline/shoe pixels and 7.82% of the marked finger blur. Lower probabilities overlapped the bin, and every setting missed the frame-98 reflection. No full-video mask archive or new inpainting candidate was promoted. Uniform threshold and padding changes alone did not resolve this sample set; this does not rule out other segmentation methods. Evidence: `.context/evidence/clean-mask-refinement-20261106/`.
- Source-frame patches can restore an occluded fixture without repairing its surroundings. Offline lobby cleanup with SAM2 masks, isolated LaMa, and measured source warps restored the fountain and blue bins, but the glass doorway and wall/floor junction still distorted at sample 53. All 115 lossless frames preserved pixels outside the removal masks; that invariant did not establish plausible geometry inside them. The candidate remains failed. Evidence: `.context/evidence/sam2-cleanup-20261106/full-clean-07-review/`.
- A local doorway warp also needs distributed, visible correspondences on the same surface. A lobby probe rejected a donor intersecting the person mask; the mask-clear donor provided only three reciprocal feature pairs, two visibly incorrect. No homography or composite met the predeclared eight-match and two-pixel reprojection criteria. Do not substitute a global room warp for missing doorway evidence. Evidence: `.context/evidence/scene-fidelity-20261107/doorway-probe/`.
- Separately decoding the same source frame is not an exact RGB reference across FFmpeg versions. For the lobby's HLG source, local FFmpeg 8 and retained worker FFmpeg 5.1 evidence differed by small channel values even in untouched areas. Matching source SHA, index and PTS does not remove that decode difference. New isolated-mode workers compare against their own decoded RGB before encoding; historical outputs without that receipt remain unverified for exact outside-mask preservation. Evidence: `.context/evidence/clean-diagnosis-20261105/isolated-probe-diagnosis/`.
- Smoke, fire, and water cannot be reconstructed from one moving camera.
- Clips with too little parallax, or too dark, fail reconstruction.
- A successful Marble video generation can invent major foreground structure despite an accurate prompt. The fresh bottle-source world added closed glazed doors across an open landing and changed the corridor beyond. Eleven local camera views did not recover the source layout; changing scale cannot repair the first-camera view because its translation is zero. Preserve this as a generated-world fidelity failure, not a successful placement or permission to regenerate. Evidence: `.context/evidence/fresh-room-origin-diagnostic-20261018/`.
- A saved world associated with the correct source is not necessarily faithful to it. The retained lobby room renders three blue bins instead of two, alters the fountain, and replaces the recorded interior beyond a doorway with exterior-like greenery. Five source-timestamp camera diagnostics and the existing scale gate failed; a scalar fit cannot repair these appearance and fixture-count errors. Generation lineage is unknown, and no replacement was generated. Evidence: `.context/evidence/sam2-cleanup-20261106/retained-room-diagnostic/`.
- A bounded 800-step fine-tune of that retained room improved held-out masked PSNR from 12.13 to 22.91 dB and restored the two bins, fountain and red interior railing. It still failed structural review: glass doorway and elevator edges smeared, and sideways views stretched the doorway, floor and portal. The unchanged front-center depth gate also failed (last-view p90 increased from 1.639 to 1.785). Exact source masks and per-frame calibration prevent input mismatches; they do not make photometric improvement a geometry or walking-quality pass. Evidence: `.context/evidence/scene-fidelity-20261107/trial-03/`.
- Repeating that bounded training with corrected foreground exclusions in all 115 depth targets reached 23.00 dB but retained the structural failures. Expected depth can hide mixed surfaces: two diagnostic views had median within-pixel depth spread of 0.149 and 0.288 body-heights despite low mean-depth error. Strong center tethering and untethered rotations of elongated splats allowed footprint changes without enough center movement. This is a measured objective limitation, not a new camera-transform mismatch. Evidence: `.context/evidence/upload-e2e-20261107/geometry-next.md` and `trial-05/`.
- An opt-in 800-step trial penalizing per-contribution log-depth residuals, tethering rotations and permitting measured center movement reduced median depth spread to 0.034 and 0.064 body-heights and reached 23.92 dB. It retained over 99% of baseline supported coverage, but late-view p90 expected-depth error rose from 0.472 to 0.579 body-heights. The unchanged scale gate and real viewer still rejected elevator smearing, ghosted glazing and stretched floor/portal surfaces. Only 10–25% of inspected portal interiors had depth support. The new terms remain disabled by default; reducing depth spread does not establish correct surfaces or fill missing observations. Evidence: `.context/evidence/upload-e2e-20261107/trial-06/`.
- A source-only Gaussian seed from eight retained Pi3X anchors recovered the lobby's fixture positions but left large holes at glass, doorway interiors and scene edges. With reviewed exclusion masks, confidence filtering and 391,410 voxel representatives, opacity above 0.5 covered only 49.7–84.6% of the eight source-camera images. The real viewer also showed the missing surfaces and dotted floor. This specific seed is incomplete; replacing generated surfaces with sparse measured points alone did not produce an acceptable room. Evidence: `.context/evidence/scene-fidelity-20261107/source-geometry/`.
- Reducing that source-only cloud without preserving its footprint caused much of the floor stippling. Keeping all 1,365,731 reviewed points at the same opacity and footprint increased the eight main floor regions from 58.2–75.4% to 98.6–100% coverage. It did not recover unsupported glass or portal interiors: one portal still had zero alpha on 8,676 of 9,762 pixels. Sampling density can repair supported surfaces; it does not supply missing depth. These are CPU projection measurements, not scene or headset acceptance. Evidence: `.context/evidence/geometry-support-20261107/source-support-audit/`.
- Combining those full measured points with the trained lobby room sharpened elevator and portal contours but retained warped glass, a hard floor seam and stretched side-view surfaces. Removing generated centers contradicted by five independent anchor depths removed only 2,063 of 1,920,000 generated splats. Complete projected coverage did not establish compatible surfaces: middle source views lost 5.5–6.0 dB of masked PSNR, and independent image review and the real viewer rejected the hybrid. No scene was promoted. Evidence: `.context/evidence/geometry-support-20261107/hybrid-probe/`.
- An explicitly approved image-conditioned replacement restored that open landing, but source-matched beginning/middle/end views still showed a missing staircase through the right portal and an invented raised wedge at the central column. A good first view is not full-clip layout acceptance. Sparse, ambiguous source/render feature matches did not justify a rigid registration correction. Evidence: `.context/evidence/fresh-room-origin-diagnostic-20261018/image-repair-01/`.
- Counting every generated splat in a camera frustum against frontmost source depth mistakes hidden rear geometry for registration error; duplicating hidden splats can even move the fitted median. The scale diagnostic now compares one generated front-center depth per supported source texel, retains the old distribution separately, and reports overlap counts. Synthetic visible near/far errors still fail the unchanged bands. This center z-buffer is an approximation, not opacity rendering or proof of scene coverage; the image replacement's visible geometry still needs independent review.
- GEN3C as a static fill generator lost to the FLUX panorama fill.
- A dominant low horizontal plane can be a countertop rather than a walking floor. Kitchen source-labelled floor tiles lie about 0.61 body-heights below its selected counter plane; validate surface identity before using camera-height or walker-floor estimates.
- A track with no usable complete-body height does not constrain body scale through vertical silhouette edges alone. In the retained hall-walk shot, head/floor fitting increased visible body area and reduced source overlap despite passing the camera-height check. Tracker-derived crop flags are also fallible: at six seconds its projected feet lie outside the image although the source boots are visible. A visible-width experiment improved overlap but still failed area and contact-noise checks. Preserve the failed verdict and investigate tracking disagreement; a size-check pass does not establish accurate placement.
- Main-thread `toBlob` and `OffscreenCanvas.convertToBlob` took about four seconds per spectator image during native Quest XR sessions. Moving JPEG encoding to a dedicated worker reduced observed encoding latency to roughly 25–40 ms; simulated XR did not reproduce the main-thread stall.

- A source MOV containing H.264 video and PCM audio can stall its browser video clock after Web Audio unlock, despite reporting `paused=false`, `readyState=4` and no media error. The same encoded video in a silent MP4 advanced under the same viewer settings. Packaging now stream-copies compatible video to MP4 and the original PCM to an external WAV; packet timestamps and samples must remain identical. This is a delivery compatibility failure, not an animation or inference failure. Source/media proof: `.context/evidence/bottle-source-rerun/media-compatibility/`.
