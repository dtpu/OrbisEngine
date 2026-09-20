# Segment selection

`scripts/select_segment.py` decides **which seconds of an upload the pipeline reconstructs**:
deterministic, CPU-only, no network, no paid call. It measures every candidate window and records
all of them, so the choice is reviewable instead of remembered. `select(video, ...) -> dict` is the
runner's entry point; the CLI writes the same dict as `selection.json`. It cannot know intent —
which play is famous, which line matters — and that comes from the operator (`--force-window`) or
from an optional VLM opinion, recorded as an opinion and never applied.

`shot_cuts.py` scores **whole shots**, only the 12 longest, and `score_shot` treats "exactly one
person" as ideal (two 0.45, more 0.2) — wrong for sports, fights and dance, and unable to find the
best 12 s inside a 3-minute take, where good and bad seconds average into one number. This stage
keeps that file's cut detection, `carry` and boundary margin, replaces the ranking, adds sliding
windows, and imports `shot_cuts` defensively: a half-saved sibling degrades to a local `carry` and
`featuresAvailable.shotCuts: false` instead of crashing the stage.

## Features

One grey decode at `--sample-fps` (6/s) at a fixed 720 px width, plus one RGB pass for the detector;
per-frame results are cached, so 75 %-overlapping windows cost nothing extra. HDR is tone-mapped
first (`video.tonemapped`): decoded flat, a dark HLG phone clip reads as "no near-black pixels".
Letterbox bars are found once and cropped off before anything is measured — left in, a 2.39:1 film's
bars are 26 points of `darkFraction` describing the file rather than the picture, and they inflate
`carry` across a cut and hide it. `video.suggestedCrop` returns that crop in source pixels
(`crop=1920:804:0:138` on the boxing clip) with `barFrameFraction`: below 1.0 the aspect changes part
way through and the crop takes real picture from the rest.

| Feature                              | Unit / meaning                                                                                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera.widestGapRatio`              | homography ÷ fundamental inliers, ORB+RANSAC, at the widest gap (0.33/1/3 s) holding ≥4 pairs; ~1.0 = the camera only turned. Same test, threshold and width as `worker/stages/parallax_probe.py`. |
| `people.presenceFraction`            | share of detector samples with ≥1 **subject** = COCO person, score > 0.7, box ≥ 0.2 of frame height; same detector and cuts as `worker/wander_worker/masks.py`. `subjectCountMedian`/`Max` are subjects per sample and `countStability` the share of samples at the median. |
| `people.subjectHeightFractionMedian` | tallest subject's box height ÷ active picture height; `fullBodyFraction` = box clear of both frame edges; `personPixelFractionMedian` = union of subject masks ÷ active area, i.e. how much of the frame the cleaner has to inpaint.                                        |
| `image.sharpness`                    | variance of the Laplacian on 0–255 grey at 720 px. `darkFraction` / `brightFraction` = share of pixels at luma ≤ 16 / ≥ 246.                                                                        |
| `overlay.overlayFraction`            | bars plus pixels whose whole-window peak-to-peak is ≤ 6/255, but only while the picture moves: at ≥ 60 % static it reports `staticCameraSuspected` and falls back to the bars alone.                 |
| `continuity.minCarry`                | lowest `shot_cuts.carry` (normalised thumbnail correlation) between consecutive samples, 1/6 s apart.                                                                                                |
| `place.endpointOverlap`              | ORB matches between the window's first and last sample surviving a RANSAC fundamental matrix, ÷ the smaller keypoint count: does it end where it began?                                              |
| `seconds`                            | window length.                                                                                                                                                                                      |

## Score

A weighted sum in [0, 1], every term in [0, 1]. `--weights w.json` overrides, is normalised and is
recorded. Ties break to the earlier start, then end, then id; top-K is greedy, rejecting a window
overlapping a pick by more than `--max-overlap` (default 0) of the shorter of the two.

| Term            | Weight | Rationale                                                                                                                                                                                                       |
| --------------- | -----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `translation`   |   0.40 | Dominant by design: no camera travel, no reconstruction at all. 1.0 at ratio ≤ 0.80, 0.0 at ≥ 0.92 (also a veto), 0.3 unmeasured.                                                                                |
| `people`        |   0.22 | `presence × count × crowd × stability`. Count is 0 with nobody, **full from 1 subject up to `--people-cap` (4)**, decaying to 0.3 at twice it; crowd falls from 1.0 at 35 % person pixels to 0.15 at 60 %, because a close-up the cleaner must inpaint comes back invented. |
| `continuity`    |   0.12 | A break inside the window fuses two places into one world — the failure `shot_cuts.py` exists to prevent. Graded 0.10 → 0.60.                                                                                    |
| `image`         |   0.08 | `sharpness × exposure × (1 − overlay penalty)`: degrades the solve and the world, but is not fatal alone.                                                                                                        |
| `fullBody`      |   0.07 | A subject cropped at the waist gives the avatar chain no legs.                                                                                                                                                   |
| `place`         |   0.00 | Measured and reported on every candidate, but **not ranked on**: the measurement failed its own test (see Limits). `--weights` can switch it on.                                                                 |
| `duration`      |   0.06 | 2 s floor, full at 12 s. Mild: a clean 8 s beats a messy 12 s.                                                                                                                                                   |
| `subjectHeight` |   0.05 | 0.25 of frame height is about the smallest an avatar has been fitted from, 0.70 a full figure (inherited from `score_shot`).                                                                                     |

**Vetoes** score a window 0 so it is never chosen; its terms are still reported and
`--force-window` still overrides it. They are `too-short` (under `--min-window`, 6 s), `no-samples`
(fewer than two frames), `too-dark` (median `darkFraction` > 0.60 — `known-limits.md`: dark clips
fail), `internal-cut` (`minCarry` ≤ 0.10) and `rotation-only` (ratio ≥ 0.92, which `parallax_probe`
calls a collapse of the solve rather than a penalty). A traverse is **not** vetoed: the measure
cannot carry one. With everything vetoed the stage exits 1 and names `blockedBest`, not nothing.

## Windows, cuts and suspected breaks

Cuts come from `--truth`, else a `--cuts` report, else `shot_cuts.cut_report(score=False)` in
process — and if that import failed, the stage says so and asks for `--cuts` or `--truth`. A
consecutive sample pair carrying ≤ 0.10 is a **suspected break** and splits its shot, the 1/6 s
between the two samples being dropped: that is all the precision this measure has. Windows
are laid inside one segment, so they never cross a cut or a break, and each segment is first trimmed
by `shot_cuts.trim`'s own margin (½ frame off the head, 1½ off the tail), so a chosen window is
directly trimmable by that function. A segment ≤ `--max-window` (20 s) is one whole window if it is
≥ `--min-usable` (2.5 s); a longer one becomes `--target-window` (12 s) windows every `--stride`
(3 s), plus one aligned to its end.

## Override, judge and output

```sh
uv run --locked --group inference python scripts/select_segment.py --video clip.mp4 --out sel.json \
  --top 5 --contact-sheet sheets/ --force-window 72 84 --reason "the producer wants the entrance"
```

`--force-window` is refused without `--reason`; the output records it, its reason, the score it
measured and the windows it displaced. `--contact-sheet DIR` writes one JPG sheet per chosen window
plus `judge_request.json` (sheets, their SHA-256s, the question, the answer schema); use `--top 5`
for the judge hook. **Nothing here calls a model.** The orchestrator may buy one receipted answer
with `scripts/vlm_once.py` and feed it back through `--judge-opinion`; it is stored with
`applied: false` and changes nothing, because `docs/quality-rubric.md` makes an automated judge
triage and not proof. This stage is upstream of every criterion in that rubric — it chooses the
input they are measured on — and its numbers describe the **source**, never a reconstruction.
`selection.json` (`wander.segment-selection/1`) carries the source path and SHA-256, the cut source,
every parameter and weight, **every** candidate with features, terms, score, vetoes and a one-line
`why`, the chosen windows with exact source seconds, frame ordinals and crop, the override, the
opinion, and `coverageNotSelected` — the time thrown away, stated plainly.

## Limits

- **`place` does not work yet, so it is weighted 0.** Over 12 s windows the known-bad traverse
  measures 0.013–0.033, a clip that never leaves one boxing ring 0.021–0.131, a locked-off shot 0.92
  — already overlapping. On the clip the idea came from it is inverted: the 12 s traverse through
  three flooded chambers measures 0.027 and the single-platform boss fight after the cut, by eye the
  best thing in that clip, measures 0.012. A camera orbiting one place renews its view as thoroughly
  as one leaving it. `--max-window` is what actually keeps a 22 s traverse out of one solve.
- **`continuity` cannot find every cut.** At 1/6 s spacing, 11 reviewed cuts of the movie scene carry
  0.050–0.277 while 1638 pairs of a verified single take bottom out at 0.166: no threshold separates
  them. The gate sits at 0.10 — below everything real motion reached, above both independently
  verified cuts in the corpus. A cut carrying 0.10–0.28 is the detector's job, and after a split
  `minCarry` is measuring motion, not cuts.
- Exposure counts near-black and near-white pixels, so murky low-contrast footage (fog, underwater)
  passes while still being poor input; compare `image.meanLuma` there. `image.sharpness` saturates
  above 200 while ordinary HD footage measures 300–850, so it is a floor guard against a
  motion-blurred window, not a way to rank good footage against better.
- People features are only as good as Mask R-CNN on CPU at 0.7 — distant, occluded or blurred
  subjects are missed, and a missed subject reads as "nobody here". Without torchvision every people
  term degrades to 0.5 and `featuresAvailable.people` is false; never a silent pass.
- `overlayFraction` only sees pixels that never change, so a **game HUD misses it**: the minimap and
  health bars of the game capture animate, and every window there reports 0–1 % overlay with a HUD
  in every frame. It also cannot separate a scoreboard from a wall under a locked-off camera, and
  says so with `staticCameraSuspected`.
- Cost is dominated by the detector (~5 s a frame on a 4-core CPU box) and is capped by frame count
  (`--people-max-frames`, 48), not rate: a 30 s and a 3-minute clip pay the same detector bill. The
  long clip pays for that in resolution — 48 frames over 273 s leaves a 12 s window 2–3 people
  samples, so `presenceFraction` and `countStability` are coarse there.

## Worked example: a 3-minute sports clip

A 180 s single-take basketball clip. `cut_report(score=False)` finds no cut, so the old ranking
would score all 180 s as one number and mark its median of five players down to 0.2. Here the usable
span exceeds 20 s, so it becomes ~56 sliding 12 s windows. Walking the ball up court measures ratio
0.93 — `rotation-only`, the camera panning on the spot — and is vetoed; a sideline tracking shot
measures 0.74 and takes full marks on the dominant term, and five subjects against a cap of 4 cost
0.83 rather than 0.2. Where the broadcast cut to a replay and the detector missed it, a sample pair
carries 0.05 and the shot splits, so no window spans it. Top-3 returns three distinct plays,
`coverageNotSelected` names the other ~144 s, and the operator reads the contact sheets and takes
rank 1 or runs again with `--force-window 88 100 --reason "the dunk"`.
