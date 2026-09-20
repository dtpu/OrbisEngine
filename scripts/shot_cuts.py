#!/usr/bin/env python3
"""Is this clip one continuous shot? Answer before the pipeline spends anything on it.

Every stage after this one assumes continuous motion. A cut breaks that assumption silently: the
camera solve fits one trajectory through both shots, so the cut comes back as a teleport (Diagon
Alley, share/MOVIE-CLIP-STATUS.md: 0.95-4.16 units in 83 ms, every jump on a batch seam), person
tracking swaps identity across it, and two places get fused into one world. Nothing errors.

Two checks live here, one before the spend and one after the solve:

  --video FILE   cut detection and shot choice. ffmpeg's scene score and a collapse of the
                 frame-to-frame correlation each pick frames worth looking at, and neither decides
                 anything; each candidate is then settled on how much of the frame carries across
                 it, whether the view comes back, and a feature match on the exact frame pair. A
                 fade through black or white is found separately, as a region, because no
                 frame-pair test can see one. All of that rides on two decode passes and no GPU.
                 Multi-shot clips are then scored on the criteria this project already selects
                 shots by -- camera translation (parallax_probe), one clear person, full body,
                 person pixel height, duration -- and the best one is named.

  --poses DIR    the teleport guard, run on a pi3x output directory after the solve. Catches a cut
                 too soft for the scene score, and solve failure generally.

  --evaluate T   score a report against a hand-labelled truth file (see evaluate_truth).

  uv run --locked --group inference python scripts/shot_cuts.py --video public/clips/x.mp4 --json cuts.json
  uv run --locked --group inference python scripts/shot_cuts.py --poses .context/run/x/pi3x            # 3 = teleport, 4 = unchecked
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ffmpeg's scene score selects the frames worth looking at, and nothing more. It is
# min(mafd, |mafd - previous mafd|)/100, where mafd is the mean absolute difference from the frame
# before: a frame scores high only when it differs a lot from its predecessor AND differs by much
# more than the frame before it did. That makes it a change-of-motion detector, not a cut detector,
# and it fails in both directions:
#
#   false high   a whip pan that snaps to speed in one frame, or a subject crossing close to the
#                lens, steps mafd from 2 to 25 and scores like a cut. A one-frame exposure flash
#                scores 0.43-0.70 -- above any "certain" line one could draw -- with nothing cut.
#   false low    a cut INSIDE a fast pan is hidden, because mafd was already high: the same splice
#                that scores 0.28 in a still shot scores 0.05 in a whip and is never even a
#                candidate at the 0.10 gate.
#
# So no score, however high, is called a cut here on its own; and a candidate is allowed to move to
# the neighbouring frame that the evidence below actually points at.
#
# It also fails a third way, which is why it is no longer the only way in: on dark or low-contrast
# footage a real cut can score below any usable gate. Four of the 117 hand-labelled cuts in
# .context/evidence/new-clips/cuts-v2 scored 0.014-0.098 -- inside the noise of their own clips --
# so `carry` below CARRY_CANDIDATE makes a frame a candidate as well. That series is computed for
# every frame of the clip in the same pass, so the second gate is free to compute.
#
# CARRY_CANDIDATE sits BELOW CARRY_CARRIES, which looks inconsistent -- a frame whose carry lies in
# [0.62, 0.72) would not be cleared by the carry test, yet only the scene score can let it in. That
# gap is deliberate and measured. Closing it (CARRY_CANDIDATE = CARRY_CARRIES = 0.72) was run over
# the whole labelled corpus: it found not one further hand-labelled cut, added four false
# positives, and took 30-100% longer per clip, because on these six clips carry < 0.72 admits
# roughly twice as many frames as carry < 0.62. The four dark cuts this gate exists for sit at
# carry 0.363-0.667 and three of them are under 0.62; the fourth is reached from its neighbour by
# REFINE_FRAMES. So the gate is set where it pays.
SCENE_CANDIDATE = 0.10
CARRY_CANDIDATE = 0.62

# The evidence. All of it except the ORB match comes out of the single decode pass that scores the
# clip, so a candidate is usually settled without decoding anything again.
#
#   carry        how much of the previous frame is still in this one: the best normalised
#                correlation of a THUMB_WIDTH x THUMB_HEIGHT grey thumbnail against the previous
#                one over +/- CARRY_SEARCH_X, CARRY_SEARCH_Y cells. Coarse on purpose. Blur and
#                grain live in the detail it throws away, the shift search absorbs the pan, and
#                normalised correlation subtracts the mean and divides by the norm, so an exposure
#                jump moves it barely at all. Across a real cut there is nothing to correlate.
#   returns      the best carry between the frames either side of the candidate and the clip up to
#                CARRY_RETURN_FRAMES frames away, in both directions. An occlusion is temporary --
#                the lorry, the player, the arm passes and the view comes back. A cut never does.
#   isolation    the candidate's change (1 - carry) against the median change of its neighbours,
#                floored by ISOLATION_FLOOR so a dead-still shot cannot divide by nothing. A cut is
#                one frame of disagreement between two agreeing runs of frames; sustained motion is
#                a plateau, and a plateau is not an edit however high it sits.
#   match        the fraction of ORB keypoints that survive across the ONE frame pair the cut
#                would be between, pulled by frame index so the pair is the pair. Across a real cut
#                nothing survives; across blur and grain enough does.
#
# A candidate is cleared by the first of these that will have it, and is a cut only if none will.
# `isolation` no longer clears a candidate that any of the others could judge -- doing that cost 17
# of the 117 labelled cuts. It survives only as the last resort for a pair too blurred for ORB to
# have an opinion about at all; see the note beside its use in detect().
#
# CALIBRATION, re-derived on 2026-09-20 from the hand labels in
# .context/evidence/new-clips/cuts-v2 -- 103 high-confidence hard cuts and 124 adversarially
# chosen reviewed non-cuts across six real clips -- and scored in
# .context/evidence/new-clips/cuts-v3. Each row is measured on the frame the reviewer NAMED, not
# on every candidate inside the +/-2 frame match tolerance: the frame after a cut is inside that
# tolerance, but its own pair lies wholly within the incoming shot, so it correlates and matches
# like any other pair and would smear the two populations into each other. All 103 cut frames
# became candidates; 52 of the 124 non-cut frames did, and the rest never reached a test.
#
#   evidence    true cuts (n)     reviewed non-cuts    threshold, and what it costs
#   carry       -0.112-0.671 103  0.086-0.963  52      >=CARRY_CARRIES 0.72: 0 cuts, 14 non-cuts
#   returns      0.094-0.845 103  0.086-0.891  38      >=CARRY_RETURNS 0.90: 0 cuts, 0 non-cuts
#   isolation    3.40 -45.87 103  1.54 -35.56  38      no threshold separates these; see below
#   match        0.000-0.003  98  0.000-0.455  33      >=CUT_MATCH_MAX 0.02: 0 cuts, 26 non-cuts
#   scene score  0.098-0.809 103  0.003-0.661  52      see SCENE_CANDIDATE and CARRY_CANDIDATE
#
# So carry clears with 0.049 of margin over the highest true cut, returns with 0.055, and match
# with 0.017 -- 6.7x the highest true cut, on a quantity whose non-cut median is 0.056. `returns`
# clears no real non-cut in this corpus at all; it is kept because it is the only thing that
# separates a one- or two-frame occlusion from an edit (fixture L), and 0.90 is where it can do
# that without touching the 0.845 cut.
#
# The synthetic-only numbers this file used to quote (carry 0.932-0.984 continuous, isolation
# 1.04-2.80 continuous) do not survive contact with real footage, which is why the two populations
# above overlap where they do.
THUMB_WIDTH, THUMB_HEIGHT = 128, 72
CARRY_SEARCH_X, CARRY_SEARCH_Y = 28, 14
CARRY_CARRIES = 0.72
CARRY_RETURNS = 0.90
CARRY_RETURN_FRAMES = 6
ISOLATION_WINDOW = 12
ISOLATION_FLOOR = 0.02
# The level at which a candidate reads as "one frame of disagreement between two agreeing runs".
# It used to clear every candidate below it, and that cost 17 of the 117 labelled cuts: over the
# whole labelled corpus real cuts run down to 3.40 and reviewed non-cuts up to 35.56, so on the
# population the detector actually sees this ratio separates nothing. It now clears only a
# candidate no other test could judge, and the value is left where the synthetic fixtures put it
# (continuous 1.04-2.80, cut 32.8-43.1) because the real distribution offers nowhere better.
ISOLATION_MIN = 6.0
# A candidate may move this far to the frame the carry series says the change is really on. ffmpeg
# scores the frame where mafd CHANGED most, which inside fast motion can be a frame or two after
# the splice; the trimmer needs the splice itself.
REFINE_FRAMES = 2
# Two cuts cannot be three frames apart: that would be a shot no reconstruction could use, and on
# real footage a run of neighbouring accusations is one violent camera move reported many times.
# The closest pair of hand-labelled cuts in the corpus is 0.300 s apart, so this is 3x clear of it.
MIN_CUT_SECONDS = 0.10
# ORB descriptors are 256 bits. Under 48 of them a blurred frame pair matches on noise -- the
# fraction is normalised by the keypoint count, and blur leaves so few keypoints that a handful of
# accidental matches reads as 0.1-0.36, the same range a real cut's sharp frames reach. At 32 the
# accidental matches disappear: over the labelled corpus every high-confidence real cut scores
# 0.000-0.003 while non-cuts reach 0.455.
ORB_MATCH_DISTANCE = 32
CUT_MATCH_MAX = 0.02
# A fade through black or white is invisible to both gates above: ffmpeg's scene score sits at
# 0.000-0.013 through one (the luma moves ~3 levels a frame) and `carry` is a NORMALISED
# correlation, which is exactly what makes it survive an exposure change. So a fade is found on its
# own terms -- a monotone luma ramp of at least FADE_MIN_FRAMES frames, worth at least
# FADE_MIN_LEVELS of luma, that runs into a frame which is flat black or flat white -- and reported
# as a region, so that neither the shot before nor the shot after contains the faded frames.
FADE_MIN_FRAMES = 6
FADE_MIN_LEVELS = 12.0
FADE_BLACK = 8.0
FADE_WHITE = 247.0
FADE_FLAT = 4.0
# A title or outro card carries no footage: its frames are identical to within the encoder's noise.
# Measured over the shots of the six labelled clips, at the 90th percentile of a shot's per-frame
# change a frozen card runs 0.003-0.006 grey levels and the quietest real footage shot runs 0.734.
# It does NOT catch a card whose text animates (the An0x title card runs 0.242), so a shot without
# the flag is not thereby footage -- the flag is advisory, and only keeps the shot chooser from
# spending a Mask R-CNN pass on a frozen card and then picking it because it is the longest.
STATIC_CHANGE = 0.05
# Below this a segment is not worth reconstructing (the pipeline wants seconds of parallax), so it
# is not offered as a choice -- but the cut that made it still counts as a cut.
MIN_SHOT_SECONDS = 1.5
# Scoring a shot costs a Mask R-CNN pass; a 50-shot film would be minutes of it. Score the longest
# few and say so. Pointing run_clip.py at a whole film is what rank_film_shots.py is for.
MAX_SCORED_SHOTS = 12

# ---- teleport guard thresholds -------------------------------------------------------------
# Calibrated on every pi3x solve in .context/run (steps are per sampled frame, depth is the median
# distance from camera 0 to its own anchor cloud):
#   shipped single-shot clips   0.08-1.53 scene depths per second, worst step 1.5-3.3x the median
#   hp33 crane window (clean)   0.24 depths/s, 17.7x median (batch-seam noise on a slow camera)
#   hp33 whole clip, cut in it  29 depths/s, 134x median (reconstructed from its camera-batches)
# BOTH tests must hold. The speed test alone would flag a genuinely fast camera; the spike test
# alone flags the batch-seam noise that tos172 (70x median) and hp33-crane (17.7x) carry with no
# cut in them, because a near-static camera has a near-zero median to divide by.
# The spike is measured against the MEDIAN step, not a high percentile: the whole-clip hp33 solve
# teleports at twelve batch seams, and twelve outliers in 305 steps drag p90 up until the worst
# jump is only 2.2x it. The median does not move.
TELEPORT_DEPTHS_PER_SEC = 3.0
TELEPORT_SPIKE = 8.0


def ffprobe(video: Path) -> dict:
    s = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    )
    st = s["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    dur = float(st.get("duration") or s.get("format", {}).get("duration") or 0)
    return dict(
        width=int(st["width"]),
        height=int(st["height"]),
        fps=float(num) / float(den),
        frames=int(st.get("nb_frames") or 0),
        duration=dur,
    )


# ---------------------------------------------------------------- cut detection
def scene_pass(video: Path, width: int = 320) -> tuple[list[float], list[float], list, list[int]]:
    """One decode: (pts_time, scene score, grey thumbnail, pts) for every frame of the clip.

    The score is read at `width` -- the same 320 px the detector has always scored at, so the
    numbers in old reports still mean what they meant -- and the thumbnail falls out of the same
    pass. ffmpeg prints its metadata with the frame number on it, so the two streams are joined on
    that rather than on arrival order.

    The integer `pts` is returned as well as the float `pts_time`, because it is the only exact
    name a frame has: frames_at() re-decodes the clip to pull the frames a candidate needs and
    checks them against these, so a pair is never off by one.

    Memory is the clip's frame count times THUMB_WIDTH*THUMB_HEIGHT bytes: about 80 MB for five
    minutes at 30 fps.
    """
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="shotcuts-scene-") as work:
        meta = Path(work) / "scene.txt"
        graph = (
            f"scale={width}:-2,select='gte(scene,0)',metadata=print:file={meta},"
            f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}"
        )
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-an",
                "-sn",
                "-vf",
                graph,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        text = meta.read_text() if meta.exists() else ""
    thumbs = np.frombuffer(raw, np.uint8).reshape(-1, THUMB_HEIGHT, THUMB_WIDTH)
    times = [0.0] * len(thumbs)
    scores = [0.0] * len(thumbs)
    stamps = [-1] * len(thumbs)
    index = None
    for line in text.splitlines():
        m = re.match(r"frame:(\d+)\s+pts:(\S+)\s+pts_time:([0-9.]+)", line)
        if m:
            index = int(m.group(1))
            if index < len(times):
                times[index] = float(m.group(3))
                stamps[index] = int(m.group(2)) if m.group(2).lstrip("-").isdigit() else -1
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and index is not None and index < len(scores):
            scores[index] = float(m.group(1))
    return times, scores, thumbs, stamps


def frames_at(video: Path, wanted: set[int], stamps: list[int], width: int = 320) -> dict:
    """The greyscale frames with these indices, pulled by index out of one decode of the clip.

    This is the only way to get the frame pair a cut would be between. Seeking by time cannot do
    it: the seek lands on a frame boundary that a printed timestamp does not name exactly, and one
    frame of slip puts a frame of the OLD shot in the "after" slot, where it matches the old shot
    perfectly and clears the cut. That bug cleared 45 of the 113 labelled cuts that reached the ORB
    stage, and because the old code took the best of several pairs, the slip could only ever clear
    a cut and never accuse one.

    So the clip is decoded once more in the same order as scene_pass, the wanted frames are kept as
    they go past, and each one is checked against the pts scene_pass recorded for it. Decoding
    twice costs about a second per thirty seconds of clip and is the price of the pair being the
    pair.
    """
    import numpy as np

    if not wanted:
        return {}
    height = int(
        len(
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-sn"]
                + ["-vf", f"scale={width}:-2", "-frames:v", "1"]
                + ["-f", "rawvideo", "-pix_fmt", "gray", "-"],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        )
        // width
    )
    if height <= 0:
        return {}
    out, size, index = {}, width * height, 0
    with tempfile.TemporaryDirectory(prefix="shotcuts-frames-") as work:
        meta = Path(work) / "frames.txt"
        graph = f"scale={width}:-2,select='gte(scene,0)',metadata=print:file={meta}"
        decode = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-sn", "-vf", graph]
            + ["-f", "rawvideo", "-pix_fmt", "gray", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        last = max(wanted)
        while index <= last:
            buf = decode.stdout.read(size)
            if len(buf) < size:
                break
            if index in wanted:
                out[index] = np.frombuffer(buf, np.uint8).reshape(height, width).copy()
            index += 1
        decode.stdout.close()
        decode.wait()
        text = meta.read_text() if meta.exists() else ""
    seen = {}
    for line in text.splitlines():
        m = re.match(r"frame:(\d+)\s+pts:(-?\d+)\s", line)
        if m:
            seen[int(m.group(1))] = int(m.group(2))
    for i in list(out):
        if stamps[i] >= 0 and i in seen and seen[i] != stamps[i]:
            raise RuntimeError(
                f"{video}: frame {i} came back with pts {seen[i]}, not the {stamps[i]} the scoring "
                f"pass recorded. The two decodes disagree; the frame pair cannot be trusted."
            )
    return out


def previous_distinct(thumbs, i: int, same: float = 0.2) -> int:
    """The last frame before `i` that is not a duplicate of it.

    A 30 fps clip in a 60 fps container decodes as [A, A, B, B, C, C]. The frame before a candidate
    can then be the candidate itself, which matches itself perfectly and clears the cut. Duplicates
    are bit-identical out of the decoder, so a mean difference of a fraction of a grey level is
    enough to tell them apart.
    """
    import numpy as np

    a = thumbs[i].astype(np.int16)
    j = i - 1
    while j > 0 and float(np.abs(thumbs[j].astype(np.int16) - a).mean()) < same:
        j -= 1
    return j


def carry(previous, current) -> float:
    """How much of `previous` is still visible in `current`, allowing for a shift.

    The centre of `current` is slid over the whole of `previous` and the best normalised
    correlation wins, so a pan of up to CARRY_SEARCH_X cells a frame costs nothing. Correlation is
    taken after subtracting the mean and dividing by the norm, which is what makes it survive an
    exposure change.
    """
    import cv2

    patch = current[
        CARRY_SEARCH_Y : THUMB_HEIGHT - CARRY_SEARCH_Y,
        CARRY_SEARCH_X : THUMB_WIDTH - CARRY_SEARCH_X,
    ]
    return float(cv2.matchTemplate(previous, patch, cv2.TM_CCOEFF_NORMED).max())


def carry_series(thumbs) -> list[float]:
    return [1.0] + [carry(thumbs[i - 1], thumbs[i]) for i in range(1, len(thumbs))]


def isolation(change: list[float], i: int, window: int = ISOLATION_WINDOW) -> float:
    """How far frame `i` stands above the frames around it, on a series where a cut is a spike.

    The immediate neighbours are left out: a cut disturbs them too, and a flash disturbs the frame
    two away. The median of what is left is the level this stretch of clip normally runs at, so the
    ratio asks "is this one frame unlike its own surroundings", which is what an edit is and what
    sustained motion is not. Near the start or the end the window is simply shorter; with nothing
    left to compare against, a candidate counts as isolated rather than being silently cleared.
    """
    import numpy as np

    near = [
        change[k]
        for k in range(max(i - window, 0), min(i + window + 1, len(change)))
        if abs(k - i) > 1
    ]
    if not near:
        return float("inf")
    return change[i] / max(float(np.median(near)), ISOLATION_FLOOR)


def returns_nearby(thumbs, i: int, span: int = CARRY_RETURN_FRAMES) -> float:
    """Best carry between the frames either side of `i` and the clip `span` frames away.

    An occlusion ends and the view comes back; a cut does not. Without this a lorry crossing the
    lens is indistinguishable from an edit on the frame pair alone.

    Both directions are asked, because an occlusion has two edges and the candidate can land on
    either: the frame where the lens goes dark looks forward to the view returning, and the frame
    where it clears looks back to the view it is returning to. Over the labelled corpus the
    backward half clears two more reviewed non-cuts and the highest it reaches on a real cut is
    0.686, well under CARRY_RETURNS.
    """
    later = [carry(thumbs[i - 1], thumbs[i + j]) for j in range(2, span + 1) if i + j < len(thumbs)]
    earlier = [carry(thumbs[i - j], thumbs[i]) for j in range(2, span + 1) if i - j >= 0]
    return max(later + earlier) if (later or earlier) else -1.0


def orb_fraction(a, b, distance: int = ORB_MATCH_DISTANCE) -> float | None:
    """Fraction of ORB keypoints that survive from `a` to `b`, or None if there are too few.

    `distance` is a Hamming distance over the 256-bit descriptor; see ORB_MATCH_DISTANCE for why it
    is as tight as it is.
    """
    import cv2

    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None  # too featureless to judge
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    return len([x for x in m if x.distance < distance]) / min(len(ka), len(kb))


def fade_regions(thumbs, times: list[float]) -> list[dict]:
    """Every fade through black or white in the clip, as a region of frames.

    A fade is a ramp, not an event, so there is no frame pair to test and nothing above can see it:
    the scene score reads the per-frame luma step (about three levels) as nothing happening, and
    `carry` is normalised, so a frame that is a dimmer copy of the one before still correlates.
    What a fade does have is a flat frame at the end of it -- black or white, with the spatial
    variance of the picture gone -- with a monotone luma ramp running into it, out of it, or both.

    The region returned runs from the last frame of the outgoing shot's own exposure to the first
    frame of the incoming shot's, so the boundary can be placed with neither shot holding faded
    frames. A hard cut to a black card is not this: it takes one frame, not FADE_MIN_FRAMES.
    """
    import numpy as np

    flat = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    mean, spread = flat.mean(1), flat.std(1)
    is_flat = ((mean <= FADE_BLACK) & (spread <= FADE_FLAT)) | (
        (mean >= FADE_WHITE) & (spread <= 3 * FADE_FLAT)
    )

    def ramp(start: int, step: int, sign: int) -> int:
        at = start
        while 0 <= at + step < len(mean) and sign * (mean[at + step] - mean[at]) > 0.05:
            at += step
        return at

    out, i = [], 0
    while i < len(mean):
        if not is_flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mean) and is_flat[j + 1]:
            j += 1
        sign = 1 if mean[i] < 128 else -1  # away from the flat level is up out of black
        into, away = ramp(i, -1, sign), ramp(j, 1, sign)
        before = i - into >= FADE_MIN_FRAMES and abs(mean[into] - mean[i]) >= FADE_MIN_LEVELS
        after = away - j >= FADE_MIN_FRAMES and abs(mean[away] - mean[j]) >= FADE_MIN_LEVELS
        if before or after:
            first, last = (into if before else i), (away if after else j)
            out.append(
                dict(
                    startFrame=int(first),
                    endFrame=int(last),
                    start=round(times[first], 3),
                    end=round(times[last], 3),
                    through="black" if mean[i] < 128 else "white",
                    frames=int(last - first + 1),
                    fadesOut=bool(before),
                    fadesIn=bool(after),
                    luma=[
                        round(float(mean[first]), 1),
                        round(float(mean[i]), 1),
                        round(float(mean[last]), 1),
                    ],
                )
            )
        i = j + 1
    return out


def frame_changes(thumbs) -> list[float]:
    """Mean absolute grey-level change from each frame to the one before. 0 is a frozen card."""
    import numpy as np

    if len(thumbs) < 2:
        return [0.0] * len(thumbs)
    d = np.abs(thumbs[1:].astype(np.int16) - thumbs[:-1].astype(np.int16))
    return [0.0] + [float(v) for v in d.reshape(len(thumbs) - 1, -1).mean(1)]


def suppress_neighbours(cuts: list[dict], gap: float = MIN_CUT_SECONDS) -> None:
    """Within `gap` seconds, keep the frame that carries least and clear the rest, in place.

    A violent camera move accuses several frames in a row; a real edit is one of them. Which one
    matters, because the trimmer cuts on it, so the survivor is the frame the carry series puts the
    change on rather than the first one reported.
    """
    run: list[dict] = []
    for rec in sorted(cuts, key=lambda c: c["time"]) + [None]:
        if run and (rec is None or rec["time"] - run[-1]["time"] > gap):
            keep = min(run, key=lambda c: c["carry"])
            for other in run:
                if other is not keep:
                    other["cut"] = False
                    other["why"] = (
                        f"the same change as the cut at {keep['time']} s, {gap} s of accusations "
                        f"from one camera move; that frame carries less"
                    )
            run = []
        if rec is not None:
            run.append(rec)


def detect_cuts(
    video: Path, threshold: float = SCENE_CANDIDATE, fps: float | None = None
) -> list[dict]:
    """Every candidate frame, each marked cut or not, with the evidence that decided it.

    The cleared candidates are returned too, each with a concrete reason: a frame that scored 0.30
    and was cleared is where the operator should look if the run later trips the teleport guard.

    Order is deliberate. Everything except the ORB match comes free out of the scoring pass, so the
    candidates that fast motion produced -- the overwhelming majority on an action clip -- are
    settled without decoding a frame again, and the frames the survivors need are then pulled in
    one more pass rather than one seek each.
    """
    return detect(video, threshold, fps)["candidates"]


def detect(video: Path, threshold: float = SCENE_CANDIDATE, fps: float | None = None) -> dict:
    """detect_cuts, plus the fades and the per-frame series the shot stage wants from the pass."""
    fps = fps or ffprobe(video)["fps"]
    times, scores, thumbs, stamps = scene_pass(video)
    empty = dict(candidates=[], fades=[], times=times, changes=[], fps=fps)
    if len(thumbs) < 2:
        return empty
    carried = carry_series(thumbs)
    change = [1.0 - c for c in carried]
    fades = fade_regions(thumbs, times)
    seen, candidates = set(), []
    for i, score in enumerate(scores):
        if times[i] <= 0.05:
            continue
        # Either gate lets a frame in; neither decides anything. See SCENE_CANDIDATE.
        if not (score > threshold or carried[i] < CARRY_CANDIDATE):
            continue
        # Move to the frame the carry series says the change is really on; see REFINE_FRAMES.
        # `time` and `frame` then name that frame, while `score` stays the highest scene score that
        # made this a candidate -- the two can sit a frame apart, which is the point of moving.
        lo, hi = max(i - REFINE_FRAMES, 1), min(i + REFINE_FRAMES + 1, len(carried))
        at = min(range(lo, hi), key=lambda k: carried[k]) if lo < hi else i
        if at not in seen:
            seen.add(at)
            candidates.append((at, max(score, scores[at])))
    out, pending = [], []
    for at, score in sorted(candidates):
        rec = dict(
            time=round(times[at], 3),
            frame=at,
            kind="hard",
            score=round(score, 4),
            carry=round(carried[at], 3),
            returns=None,
            isolation=None,
            match=None,
            cut=False,
            why="",
        )
        out.append(rec)
        fade = next((f for f in fades if f["startFrame"] <= at <= f["endFrame"]), None)
        if fade:
            rec["why"] = (
                f"inside the fade through {fade['through']} at {fade['start']}-{fade['end']} s, "
                f"which is reported as that region rather than as this frame"
            )
            continue
        if carried[at] >= CARRY_CARRIES:
            rec["why"] = f"content carries across it ({rec['carry']} of the frame correlates)"
            continue
        rec["returns"] = round(returns_nearby(thumbs, at), 3)
        if rec["returns"] >= CARRY_RETURNS:
            rec["why"] = (
                f"the view comes back {rec['returns']} within {CARRY_RETURN_FRAMES} frames, so "
                f"something crossed the lens; a cut does not come back"
            )
            continue
        # Evidence, not a verdict: on real footage cuts and camera moves share this whole range.
        rec["isolation"] = round(isolation(change, at), 2)
        pending.append(rec)
    partner = {rec["frame"]: previous_distinct(thumbs, rec["frame"]) for rec in pending}
    frames, fetch_failed = {}, ""
    if partner:
        try:
            frames = frames_at(video, set(partner) | set(partner.values()), stamps)
        except Exception as err:
            # Without the frames there is no ORB verdict on anything, and every candidate would
            # otherwise read as "too blurred to judge" -- a sentence about the footage, when what
            # happened is that the second decode did not come back. Say which it was, on the
            # record, rather than letting a broken decode look like a hard clip.
            frames, fetch_failed = {}, f"{type(err).__name__}: {err}"
    for rec in pending:
        at, before = rec["frame"], partner[rec["frame"]]
        if fetch_failed:
            rec["matchFailed"] = fetch_failed
        try:
            got = orb_fraction(frames[before], frames[at]) if at in frames else None
        except Exception:
            got = None
        rec["match"] = None if got is None else round(got, 3)
        rec["matchedAgainstFrame"] = before
        unjudged = (
            f"the pair (frames {before} and {at}) could not be fetched ({fetch_failed}), so "
            f"there is no feature evidence either way"
            if fetch_failed
            else f"both sides of the pair (frames {before} and {at}) are too blurred for "
            f"features to judge"
        )
        if rec["match"] is not None and rec["match"] >= CUT_MATCH_MAX:
            rec["why"] = f"blurred, but {rec['match']} of its features still match across it"
            continue
        if rec["match"] is None and rec["isolation"] < ISOLATION_MIN:
            # Neither side has enough corners to key on, so there is no frame-pair evidence at all
            # and the only thing left is the shape of the change over time. A cut is one frame of
            # disagreement between two agreeing runs; a pan too fast for the correlation search is
            # a plateau of disagreement, and fixture B2_whip_beyond_search is nothing but that
            # plateau: six frames with no ORB verdict, isolation 1.83-2.94, and no cut in the clip.
            #
            # This is the ONLY place isolation decides anything, and it is the weakest decision the
            # detector makes. Over the twelve unjudgeable labelled frames in the corpus the two
            # populations interleave -- non-cuts at 4.16, 5.37, 8.13, 16.24, 27.63 and cuts at
            # 2.34, 4.65, 6.63, 9.10, 25.27, 39.62, 40.01 -- so clearing below 6.0 clears two of
            # the five non-cuts and costs two of the seven cuts, one of them high-confidence
            # (1A6z7R-aaDw at 111.000 s, isolation 4.65). No value in that range has a margin, so
            # the threshold is left where the synthetic fixtures put it rather than moved to the
            # gap of the day; see ISOLATION_MIN. The real fix is not a threshold: `carry` reports a
            # collapse here because the displacement runs off the end of the
            # CARRY_SEARCH_X/CARRY_SEARCH_Y shift search, and a carry that knew its best shift had
            # pinned to the edge of its own search window could say so instead of guessing.
            rec["why"] = (
                f"{unjudged}, and the frames around it disagree as much as it does "
                f"({rec['isolation']}x the local median, under {ISOLATION_MIN}): sustained "
                f"motion, not one edit"
            )
            continue
        rec["cut"] = True
        rec["why"] = (
            f"a cut: {rec['carry']} of the frame correlates with the one before, the view does not "
            f"come back ({rec['returns']}), and "
            + (
                f"only {rec['match']} of the features match across the pair "
                f"(frames {before} and {at})"
                if rec["match"] is not None
                else unjudged
            )
            + f"; it stands {rec['isolation']}x above its neighbourhood"
        )
    suppress_neighbours([rec for rec in out if rec["cut"]])
    for fade in fades:
        first, last = fade["startFrame"], fade["endFrame"]
        out.append(
            dict(
                time=round(times[last], 3),
                frame=last,
                kind="fade",
                fadeStart=round(times[first], 3),
                fadeStartFrame=first,
                fade=fade,
                score=round(max(scores[first : last + 1]), 4),
                carry=round(min(carried[first : last + 1]), 3),
                returns=None,
                isolation=None,
                match=None,
                cut=True,
                why=(
                    f"a fade through {fade['through']}: {fade['frames']} frames from "
                    f"{fade['start']} to {fade['end']} s, luma {fade['luma'][0]} -> "
                    f"{fade['luma'][1]} -> {fade['luma'][2]}. The boundary is the region, so "
                    f"neither shot holds the faded frames"
                ),
            )
        )
    out.sort(key=lambda rec: (rec["time"], rec["frame"]))
    return dict(candidates=out, fades=fades, times=times, changes=frame_changes(thumbs), fps=fps)


def shots_from_cuts(
    cuts: list[dict],
    duration: float,
    min_seconds: float,
    times: list[float] | None = None,
    changes: list[float] | None = None,
) -> list[dict]:
    """The segments between the cuts. A fade belongs to neither side, so it is left out of both.

    With `times` and `changes` from the same pass, a shot whose frames do not change at all is
    flagged `static`: a title or outro card is not footage and nothing downstream should spend on
    it. See STATIC_CHANGE for what that flag does and does not catch.
    """
    import numpy as np

    edges = []
    for c in cuts:
        edges.append((c.get("fadeStart", c["time"]), c["time"]))
    bounds = [(0.0, 0.0)] + edges + [(duration, duration)]
    shots = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i][1], bounds[i + 1][0]
        if b - a <= 1e-6:
            continue  # a fade that opens or closes the clip leaves no shot beside it
        shot = dict(
            index=len(shots),
            start=round(a, 3),
            end=round(b, 3),
            seconds=round(b - a, 3),
            tooShort=(b - a) < min_seconds,
        )
        if times and changes:
            lo = int(np.searchsorted(times, a, side="left"))
            hi = int(np.searchsorted(times, b, side="right"))
            span = changes[lo + 1 : hi]
            if len(span) >= 3:
                # The 90th percentile, not the median: in a 30 fps clip carried in a 60 fps
                # container every other frame is a duplicate, so half the differences are zero and
                # the median of real footage can be zero too. A card is quiet at every percentile.
                shot["change"] = round(float(np.percentile(span, 90)), 3)
                shot["static"] = bool(shot["change"] < STATIC_CHANGE)
        shots.append(shot)
    return shots


# ---------------------------------------------------------------- shot scoring
def extract(video: Path, shot: dict, out: Path, n: int, width: int) -> list[Path]:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    fps = n / max(shot["seconds"], 1e-3)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{shot['start']:.3f}",
            "-t",
            f"{shot['seconds']:.3f}",
            "-i",
            str(video),
            "-vf",
            f"fps={fps:.4f},scale={width}:-2",
            "-q:v",
            "3",
            str(out / "f_%04d.jpg"),
        ],
        check=True,
    )
    return sorted(out.glob("*.jpg"))


def person_stats(frames: list[Path], n: int = 3) -> dict:
    """People in this shot, from the same Mask R-CNN the person stages use.

    Returns the median person count over `n` sampled frames, the tallest person's height as a
    fraction of frame height, and whether that person's box clears the frame edges (full body).
    """
    import numpy as np

    try:
        import torch
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2,
            MaskRCNN_ResNet50_FPN_V2_Weights,
        )
    except Exception as e:  # scoring degrades, it does not fail
        return dict(available=False, reason=f"{type(e).__name__}: {e}")
    from PIL import Image

    net = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    pick = [
        frames[round(k * (len(frames) - 1) / max(n - 1, 1))] for k in range(min(n, len(frames)))
    ]
    counts, heights, full = [], [], []
    for f in pick:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        h = rgb.shape[0]
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            r = net([x])[0]
        boxes = [
            [float(v) for v in r["boxes"][i].numpy()]
            for i in range(len(r["labels"]))
            if int(r["labels"][i]) == 1 and float(r["scores"][i]) > 0.7
        ]
        counts.append(len(boxes))
        if boxes:
            x0, y0, x1, y1 = max(boxes, key=lambda b: b[3] - b[1])
            heights.append((y1 - y0) / h)
            # Full body = the box does not run off the bottom or the top of the frame. A subject
            # cropped at the waist gives LHM nothing to build legs from.
            full.append(y0 > 0.01 * h and y1 < 0.99 * h)
    counts.sort()
    heights.sort()
    return dict(
        available=True,
        frames=len(pick),
        peopleMedian=counts[len(counts) // 2] if counts else 0,
        peopleMax=max(counts) if counts else 0,
        personHeightFraction=round(heights[len(heights) // 2], 3) if heights else 0.0,
        fullBody=bool(full and sum(full) > len(full) / 2),
    )


def camera_stats(frames_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "worker/stages"))
    from parallax_probe import probe as parallax_probe

    try:
        return parallax_probe(frames_dir, gaps=(2, 8, 16), pairs_per_gap=6)
    except Exception as e:
        return dict(verdict="inconclusive", reason=f"{type(e).__name__}: {e}")


def score_shot(shot: dict) -> tuple[float, list[str]]:
    """One number in [0, 1] from the criteria this project has always picked shots by.

    Weights say what has actually decided it: a shot with no camera translation cannot be
    reconstructed at all, so parallax dominates; a second person or a half-body subject is the
    next most common reason a shot was put back.
    """
    par, per, why = shot.get("camera") or {}, shot.get("person") or {}, []
    translation = {"has-parallax": 1.0, "inconclusive": 0.3, "rotation-only": 0.0}.get(
        par.get("verdict"), 0.3
    )
    why.append(f"camera {par.get('verdict', '?')}")
    if per.get("available"):
        n = per["peopleMedian"]
        one = 1.0 if n == 1 else (0.45 if n == 2 else 0.0 if n == 0 else 0.2)
        body = 1.0 if per["fullBody"] else 0.0
        # Pixel height: 0.25 of the frame is about the smallest the avatar chain has worked from,
        # 0.7 is a comfortable full figure. Taller than that is a close-up and scores no higher.
        px = min(max((per["personHeightFraction"] - 0.25) / 0.45, 0.0), 1.0)
        why.append(
            f"{n} person(s), {'full body' if per['fullBody'] else 'cropped'}, "
            f"{per['personHeightFraction']:.2f} of frame height"
        )
    else:
        one = body = px = 0.5
        why.append("no person detector available")
    # Duration: 4 s is about the least the solve has held together on, 12 s is plenty.
    dur = min(max((shot["seconds"] - 2.0) / 10.0, 0.0), 1.0)
    why.append(f"{shot['seconds']:.1f} s")
    score = 0.35 * translation + 0.25 * one + 0.20 * body + 0.10 * px + 0.10 * dur
    return round(score, 4), why


def score_shots(
    video: Path,
    shots: list[dict],
    work: Path,
    frames: int = 16,
    width: int = 640,
    people: bool = True,
) -> None:
    """Fill in camera/person/score on each shot that is long enough to be a candidate."""
    for s in shots:
        if s.get("static"):
            s["skipped"] = (
                f"not scored: its frames do not change ({s.get('change')} grey levels a frame), "
                f"so it is a card and not footage"
            )
    candidates = [s for s in shots if not s["tooShort"] and not s.get("static")]
    candidates.sort(key=lambda s: -s["seconds"])
    for s in candidates[MAX_SCORED_SHOTS:]:
        s["skipped"] = f"not scored: only the {MAX_SCORED_SHOTS} longest shots are"
    for s in candidates[:MAX_SCORED_SHOTS]:
        d = work / f"shot{s['index']:03d}"
        fs = extract(video, s, d / "frames", frames, width)
        s["camera"] = camera_stats(d / "frames")
        s["person"] = person_stats(fs) if people else dict(available=False, reason="--no-people")
        s["score"], s["why"] = score_shot(s)


def choose(shots: list[dict]) -> dict | None:
    scored = [s for s in shots if "score" in s and not s.get("static")]
    return max(scored, key=lambda s: s["score"]) if scored else None


def cut_report(
    video: Path,
    threshold: float = SCENE_CANDIDATE,
    min_seconds: float = MIN_SHOT_SECONDS,
    work: Path | None = None,
    score: bool = True,
    people: bool = True,
) -> dict:
    info = ffprobe(video)
    found = detect(video, threshold, info["fps"])
    candidates = found["candidates"]
    cuts = [c for c in candidates if c["cut"]]
    shots = shots_from_cuts(cuts, info["duration"], min_seconds, found["times"], found["changes"])
    doc = dict(
        video=str(video),
        source=info,
        threshold=threshold,
        minSeconds=min_seconds,
        cuts=cuts,
        cutCount=len(cuts),
        cleared=[c for c in candidates if not c["cut"]],
        fades=found["fades"],
        shots=shots,
        shotCount=len(shots),
        continuous=not cuts,
    )
    if cuts and score:
        tmp = None
        if work is None:
            tmp = tempfile.TemporaryDirectory(prefix="shotcuts-")
            work = Path(tmp.name)
        try:
            score_shots(video, shots, Path(work), people=people)
        finally:
            if tmp:
                tmp.cleanup()
        best = choose(shots)
        if best:
            doc["best"] = best["index"]
            doc["bestWhy"] = "; ".join(best["why"])
    return doc


def trim(video: Path, shot: dict, out: Path, fps: float) -> Path:
    """Re-encode one shot to its own file, a frame clear of each boundary.

    The margin is not decoration. A cut time is the timestamp of the FIRST frame of the new shot,
    ffmpeg's seek and duration both round to whole frames, and half a frame of slack was measured
    to leave one frame of the next shot at the end of the segment -- which is exactly the failure
    this whole stage exists to prevent. A lost 30 ms at each end costs nothing.
    """
    f = 1.0 / max(fps, 1e-3)
    start, dur = shot["start"] + 0.5 * f, max(shot["seconds"] - 2.0 * f, 0.1)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.4f}",
            "-t",
            f"{dur:.4f}",
            "-i",
            str(video),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    return out


# ---------------------------------------------------------------- evaluation
def evaluate_truth(
    report: dict,
    truth: dict,
    tolerance_frames: float = 2.0,
    windows: list | None = None,
    verified_only: bool = False,
) -> dict:
    """Score a cut report against a hand-labelled truth file.

    The truth file is `{video, fps, cuts: [{time, kind, confidence}], reviewedNonCuts, unclear}` --
    the format of .context/evidence/new-clips/cuts-v2/<name>-truth.json. A detection matches a
    label within `tolerance_frames` frames; a detected FADE matches a label anywhere inside its
    region, because a fade is a region and a label is a point in it.

    What counts:
      required   a `high`-confidence label that is not a fade. Recall is over these.
      optional   a `medium` label, any fade label, and anything the reviewer left `unclear`.
                 Matching one is neither a hit nor a false positive -- precision counts it in the
                 numerator, recall does not ask for it.
      false      every other detection.
    With `windows` ([[start, end], ...]) only detections and labels inside them are scored, which
    is how a clip with only two densely-labelled stretches is measured honestly. With
    `verified_only`, only detections that land on a frame the reviewer actually adjudicated are
    scored at all: on a clip whose labels are not exhaustive, an unlabelled detection is unknown,
    not wrong, and counting it either way would be a guess.
    """
    fps = float(truth.get("fps") or report.get("source", {}).get("fps") or 25.0)
    tol = tolerance_frames / max(fps, 1e-6) + 1e-6

    def inside(t: float) -> bool:
        return not windows or any(a <= t <= b for a, b in windows)

    labels = [c for c in truth.get("cuts", []) if inside(c["time"])]
    required = [
        c
        for c in labels
        if c.get("confidence") == "high" and not str(c.get("kind", "")).startswith("fade")
    ]
    wanted = {id(c) for c in required}
    optional = [c for c in labels if id(c) not in wanted]
    optional += [
        dict(time=u["time"], kind="unclear") for u in truth.get("unclear", []) if inside(u["time"])
    ]
    found = [c for c in report.get("cuts", []) if inside(c["time"])]

    def covers(det: dict, t: float) -> bool:
        lo = det.get("fadeStart", det["time"]) - tol
        return lo <= t <= det["time"] + tol

    if verified_only:
        adjudicated = (
            [c["time"] for c in truth.get("cuts", [])]
            + [c["time"] for c in truth.get("reviewedNonCuts", [])]
            + [c["time"] for c in truth.get("unclear", [])]
        )
        found = [det for det in found if any(covers(det, t) for t in adjudicated)]

    used, hits, extra, spurious = set(), [], [], []
    for det in found:
        take = next(
            (i for i, c in enumerate(required) if i not in used and covers(det, c["time"])), None
        )
        if take is not None:
            used.add(take)
            hits.append(det)
        elif any(covers(det, c["time"]) for c in optional):
            extra.append(det)
        else:
            spurious.append(det)
    missed = [c for i, c in enumerate(required) if i not in used]
    return dict(
        clip=Path(truth.get("video", report.get("video", "?"))).name,
        detections=len(found),
        required=len(required),
        hits=len(hits),
        optional=len(extra),
        falsePositives=[
            dict(
                time=c["time"],
                frame=c["frame"],
                kind=c.get("kind"),
                carry=c["carry"],
                score=c["score"],
                returns=c.get("returns"),
                isolation=c.get("isolation"),
                match=c.get("match"),
            )
            for c in spurious
        ],
        missed=missed,
        precision=round((len(hits) + len(extra)) / max(len(found), 1), 3),
        recall=round(len(hits) / max(len(required), 1), 3),
        windows=windows,
        toleranceFrames=tolerance_frames,
    )


def print_evaluation(rows: list[dict]) -> None:
    print(f"{'clip':28s} {'det':>4} {'req':>4} {'hit':>4} {'opt':>4} {'fp':>4} {'P':>6} {'R':>6}")
    total = dict(detections=0, required=0, hits=0, optional=0, fp=0)
    for r in rows:
        print(
            f"{r['clip'][:28]:28s} {r['detections']:4d} {r['required']:4d} {r['hits']:4d} "
            f"{r['optional']:4d} {len(r['falsePositives']):4d} {r['precision']:6.3f} "
            f"{r['recall']:6.3f}"
        )
        for key in ("detections", "required", "hits", "optional"):
            total[key] += r[key]
        total["fp"] += len(r["falsePositives"])
    if len(rows) > 1:
        p = (total["hits"] + total["optional"]) / max(total["detections"], 1)
        print(
            f"{'TOTAL':28s} {total['detections']:4d} {total['required']:4d} {total['hits']:4d} "
            f"{total['optional']:4d} {total['fp']:4d} {p:6.3f} "
            f"{total['hits'] / max(total['required'], 1):6.3f}"
        )
    for r in rows:
        for m in r["missed"]:
            print(f"  missed {r['clip']} {m['time']:8.3f} s ({m.get('kind', '?')})")
        for f in r["falsePositives"]:
            print(
                f"  false  {r['clip']} {f['time']:8.3f} s  scene {f['score']} carry {f['carry']} "
                f"returns {f['returns']} isolation {f['isolation']} match {f['match']}"
            )


# ---------------------------------------------------------------- teleport guard
def scene_depth(pi3x: Path) -> float | None:
    """Median distance from the first anchor camera to its own points: the scene's own scale."""
    import numpy as np

    f = pi3x / "anchors.npz"
    if not f.exists():
        return None
    z = np.load(f)
    pts, valid, cam = z["points"][0], z["valid"][0].astype(bool), z["poses"][0][:3, 3]
    P = pts[valid].reshape(-1, 3)
    return float(np.median(np.linalg.norm(P - cam, axis=1))) if len(P) else None


def teleport_check(pi3x: Path) -> dict:
    """Does the camera path contain a step no camera could have taken?

    A cut looks to the solver like the camera crossing the scene between two frames, and so does a
    registration failure. Both are reported here, in the scene's own units, before anything
    downstream places a person against these poses.
    """
    import numpy as np

    cj = pi3x / "cameras.json"
    if not cj.exists():
        return dict(ok=None, reason=f"{cj} does not exist; the solve wrote no cameras")
    cams = sorted(json.loads(cj.read_text())["cameras"], key=lambda c: c.get("time", 0.0))
    if len(cams) < 4:
        return dict(ok=None, reason=f"only {len(cams)} cameras; nothing to test")
    P = np.array([c["position"] for c in cams], float)
    T = np.array([c.get("time", i) for i, c in enumerate(cams)], float)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dt = np.diff(T)
    dt[dt <= 0] = float(np.median(dt[dt > 0])) if (dt > 0).any() else 1.0
    depth = scene_depth(pi3x)
    if not depth:
        return dict(ok=None, reason="no anchors.npz, so the scene has no scale to measure against")
    speed = steps / dt / depth  # scene depths per second
    typical = max(float(np.median(steps)), 1e-9)
    spike = steps / typical
    i = int(np.argmax(speed))
    worst = dict(
        afterCamera=i,
        frame=cams[i + 1].get("sourceIndex", i + 1),
        timeSeconds=round(float(T[i + 1]), 3),
        stepUnits=round(float(steps[i]), 4),
        sceneDepthUnits=round(depth, 3),
        depthsPerSecond=round(float(speed[i]), 2),
        spikeOverMedian=round(float(spike[i]), 2),
    )
    over = [
        int(k) for k in np.where((speed >= TELEPORT_DEPTHS_PER_SEC) & (spike >= TELEPORT_SPIKE))[0]
    ]
    bad = bool(over)
    return dict(
        ok=not bad,
        cameras=len(cams),
        medianStepUnits=round(typical, 4),
        worst=worst,
        teleports=len(over),
        teleportFrames=[cams[k + 1].get("sourceIndex", k + 1) for k in over[:12]],
        limits=dict(depthsPerSecond=TELEPORT_DEPTHS_PER_SEC, spikeOverMedian=TELEPORT_SPIKE),
        message=None
        if not bad
        else (
            f"camera teleport at frame {worst['frame']} (t={worst['timeSeconds']:.2f}s): it "
            f"moves {worst['stepUnits']:.3f} units in {round(float(dt[i]), 3)} s through a "
            f"scene {depth:.2f} units deep -- {worst['depthsPerSecond']:.1f} scene depths per "
            f"second, {worst['spikeOverMedian']:.0f}x the median step. No camera did "
            f"that ({len(over)} step(s) like it). Either there is a cut the scene-score detector missed at that frame, or "
            f"the solve failed to register across it. Trim to one continuous shot "
            f"(scripts/shot_cuts.py --video <clip>) or re-solve; do not place people "
            f"against these poses."
        ),
    )


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", type=Path)
    ap.add_argument(
        "--poses", type=Path, help="a pi3x output directory (cameras.json + anchors.npz)"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=SCENE_CANDIDATE,
        help="scene score above which a frame is a candidate cut",
    )
    ap.add_argument("--min-seconds", type=float, default=MIN_SHOT_SECONDS)
    ap.add_argument("--work", type=Path, help="keep the per-shot frames here instead of a temp dir")
    ap.add_argument(
        "--no-score", action="store_true", help="detect cuts only, do not rank the shots"
    )
    ap.add_argument("--no-people", action="store_true", help="skip the Mask R-CNN pass")
    ap.add_argument("--json", type=Path)
    ap.add_argument(
        "--report", type=Path, help="with --trim: an earlier --json report to trim from"
    )
    ap.add_argument(
        "--trim", type=int, metavar="INDEX", help="write shot INDEX of --report to --out"
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--evaluate",
        type=Path,
        nargs="+",
        metavar="TRUTH",
        help="score reports against hand-labelled truth files; with --report, score that report, "
        "otherwise detect on each truth file's own video",
    )
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="START:END",
        help="with --evaluate: score only this stretch of seconds (repeatable)",
    )
    ap.add_argument(
        "--verified-only",
        action="store_true",
        help="with --evaluate: score only detections the reviewer adjudicated, for a truth file "
        "whose labels are not exhaustive",
    )
    a = ap.parse_args()
    if a.evaluate:
        windows = [[float(v) for v in w.split(":")] for w in a.window] or None
        if a.report and len(a.evaluate) > 1:
            sys.exit("--report scores one truth file; pass one, or drop it and let each detect")
        rows = []
        for path in a.evaluate:
            truth = json.loads(path.read_text())
            if a.report:
                doc = json.loads(a.report.read_text())
            else:
                video = Path(a.video or truth["video"])
                if not video.is_absolute() and not video.exists():
                    video = ROOT / video
                doc = cut_report(video, a.threshold, a.min_seconds, a.work, False, False)
                if a.json:
                    out = a.json if len(a.evaluate) == 1 else a.json.parent / f"{video.stem}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(doc, indent=1))
            rows.append(evaluate_truth(doc, truth, windows=windows, verified_only=a.verified_only))
        print_evaluation(rows)
        return 0
    if a.trim is not None:
        if not (a.report and a.out):
            sys.exit("--trim needs --report and --out")
        doc = json.loads(a.report.read_text())
        shot = next(s for s in doc["shots"] if s["index"] == a.trim)
        print(
            f"shot {a.trim}: {shot['start']:.3f}-{shot['end']:.3f} s -> "
            f"{trim(Path(doc['video']), shot, a.out, doc['source']['fps'])}"
        )
        return 0
    if not (a.video or a.poses):
        sys.exit("--video or --poses")
    if a.poses:
        doc = teleport_check(a.poses)
        if a.json:
            a.json.write_text(json.dumps(doc, indent=1))
        if doc.get("ok") is None:
            # An unchecked solve is not a passed solve. hp33 -- the clip this guard was written for
            # -- never wrote cameras.json, so returning 0 here passed the exact case it was built to
            # catch. "I could not look" exits 4, distinct from 3 = "I looked and it teleported".
            print(
                f"POSES NOT CHECKED: {doc['reason']}. This is a FAILURE, not a pass: the solve "
                f"cannot be shown to be a continuous camera path, so nothing may be placed "
                f"against it. Re-solve, or point --poses at a directory that has cameras.json "
                f"and anchors.npz."
            )
            return 4
        w = doc["worst"]
        print(
            f"{doc['cameras']} cameras, median step {doc['medianStepUnits']} u, worst "
            f"{w['stepUnits']} u at frame {w['frame']} = {w['depthsPerSecond']} scene depths/s, "
            f"{w['spikeOverMedian']}x median"
        )
        if not doc["ok"]:
            print("TELEPORT: " + doc["message"])
            return 3
        print("no teleport: the camera path is continuous")
        return 0
    doc = cut_report(a.video, a.threshold, a.min_seconds, a.work, not a.no_score, not a.no_people)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(doc, indent=1))
    for c in doc["cleared"]:
        print(f"  cleared {c['time']:7.2f} s  scene score {c['score']}: {c['why']}")
    if doc["continuous"]:
        print(f"one continuous shot, {doc['source']['duration']:.2f} s")
        return 0
    print(
        f"{doc['cutCount']} cut(s), {doc['shotCount']} shots in {doc['source']['duration']:.2f} s"
    )
    for c in doc["cuts"]:
        where = (
            f"{c['fadeStart']:7.2f}-{c['time']:7.2f} s"
            if c.get("kind") == "fade"
            else f"{c['time']:7.2f} s        "
        )
        print(f"  {c.get('kind', 'hard'):4s} {where}  {c['why']}")
    for s in doc["shots"]:
        tag = "too short" if s["tooShort"] else s.get("skipped", f"score {s.get('score', '-')}")
        print(
            f"  shot {s['index']:2d}  {s['start']:7.2f}-{s['end']:7.2f} s  {s['seconds']:6.2f} s  "
            f"{tag}" + (f"  ({'; '.join(s['why'])})" if s.get("why") else "")
        )
    if "best" in doc:
        print(f"best: shot {doc['best']} -- {doc['bestWhy']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
