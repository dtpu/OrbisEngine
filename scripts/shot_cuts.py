#!/usr/bin/env python3
"""Is this clip one continuous shot? Answer before the pipeline spends anything on it.

Every stage after this one assumes continuous motion. A cut breaks that assumption silently: the
camera solve fits one trajectory through both shots, so the cut comes back as a teleport (Diagon
Alley, share/MOVIE-CLIP-STATUS.md: 0.95-4.16 units in 83 ms, every jump on a batch seam), person
tracking swaps identity across it, and two places get fused into one world. Nothing errors.

Two checks live here, one before the spend and one after the solve:

  --video FILE   cut detection and shot choice. ffmpeg's scene score picks the frames worth
                 looking at and decides none of them; each candidate is then settled on how much
                 of the frame carries across it, whether the view comes back, whether it stands
                 out from its neighbours, and a feature match. All of that rides on one decode
                 pass and no GPU. Multi-shot clips are then scored on the criteria this project
                 already selects shots by -- camera translation (parallax_probe), one clear
                 person, full body, person pixel height, duration -- and the best one is named.

  --poses DIR    the teleport guard, run on a pi3x output directory after the solve. Catches a cut
                 too soft for the scene score, and solve failure generally.

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
SCENE_CANDIDATE = 0.10

# The evidence. All of it except the ORB match comes out of the single decode pass that scores the
# clip, so a candidate is usually settled without decoding anything again.
#
#   carry        how much of the previous frame is still in this one: the best normalised
#                correlation of a THUMB_WIDTH x THUMB_HEIGHT grey thumbnail against the previous
#                one over +/- CARRY_SEARCH_X, CARRY_SEARCH_Y cells. Coarse on purpose. Blur and
#                grain live in the detail it throws away, the shift search absorbs the pan, and
#                normalised correlation subtracts the mean and divides by the norm, so an exposure
#                jump moves it barely at all. Across a real cut there is nothing to correlate.
#   returns      the best carry from the frame before the candidate to any of the next
#                CARRY_RETURN_FRAMES frames. An occlusion is temporary -- the lorry, the player,
#                the arm passes and the view comes back. A cut never comes back.
#   isolation    the candidate's change (1 - carry) against the median change of its neighbours,
#                floored by ISOLATION_FLOOR so a dead-still shot cannot divide by nothing. A cut is
#                one frame of disagreement between two agreeing runs of frames; sustained motion is
#                a plateau, and a plateau is not an edit however high it sits.
#   match        the surviving ORB fraction, from the sharpest bracketing frames within
#                BRACKET_FRAMES on each side as well as the immediate pair, taking the best. Blur
#                is what breaks feature matching, so the bracket is allowed to step away from the
#                blurred frames; and any one pair that matches is proof the content carried.
#
# A candidate is cleared by the first of these that will have it, and is a cut only if none will.
# Measured over the rendered fixtures in scripts/test_shot_cuts.py, continuous | cut, for the
# candidates that actually reached each test:
#   carry       0.932-0.984 | 0.137-0.344      cleared at or above CARRY_CARRIES
#   returns     0.916       | 0.321-0.530      cleared at or above CARRY_RETURNS
#   isolation   1.04-2.80   | 32.8-43.1        cleared below ISOLATION_MIN
#   match       (see below) | none, 0.016, 0.102
# PROVISIONAL: every constant here except SCENE_CANDIDATE and CUT_MATCH_MAX was set on synthetic
# clips only, at the midpoint of a measured gap rather than at the edge of a real distribution. Two
# of them are weakly evidenced and want real footage most: ISOLATION_MIN, whose only continuous
# examples come from one deliberately extreme fixture, and CUT_MATCH_MAX, which no continuous
# fixture candidate even reached -- the three tests above it got there first -- so on this evidence
# the ORB stage only ever confirmed cuts and never cleared one. The report carries carry, returns,
# isolation and match on every candidate; re-measure where the two populations sit on a real
# multi-shot clip before trusting any of these four numbers.
THUMB_WIDTH, THUMB_HEIGHT = 128, 72
CARRY_SEARCH_X, CARRY_SEARCH_Y = 28, 14
CARRY_CARRIES = 0.62
CARRY_RETURNS = 0.70
CARRY_RETURN_FRAMES = 6
ISOLATION_WINDOW = 12
ISOLATION_FLOOR = 0.02
ISOLATION_MIN = 6.0
# A candidate may move this far to the frame the carry series says the change is really on. ffmpeg
# scores the frame where mafd CHANGED most, which inside fast motion can be a frame or two after
# the splice; the trimmer needs the splice itself.
REFINE_FRAMES = 2
BRACKET_FRAMES = 3
# ORB features across a real cut match almost nothing (0.00-0.34 of the weaker frame's keypoints,
# measured on spliced clips and 58 candidates in a Diagon Alley reel); across grain or a fast pan
# they still match 0.52-0.74.
CUT_MATCH_MAX = 0.45
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
def scene_pass(video: Path, width: int = 320) -> tuple[list[float], list[float], list]:
    """One decode: (pts_time, scene score, grey thumbnail) for every frame of the clip.

    The score is read at `width` -- the same 320 px the detector has always scored at, so the
    numbers in old reports still mean what they meant -- and the thumbnail falls out of the same
    pass. ffmpeg prints its metadata with the frame number on it, so the two streams are joined on
    that rather than on arrival order.

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
    index = None
    for line in text.splitlines():
        m = re.match(r"frame:(\d+)\s+pts:\S+\s+pts_time:([0-9.]+)", line)
        if m:
            index = int(m.group(1))
            if index < len(times):
                times[index] = float(m.group(2))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and index is not None and index < len(scores):
            scores[index] = float(m.group(1))
    return times, scores, thumbs


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


def returns_after(thumbs, i: int, span: int = CARRY_RETURN_FRAMES) -> float:
    """Best carry from the frame before `i` to any of the `span` frames after it.

    An occlusion ends and the view comes back; a cut does not. Without this a lorry crossing the
    lens is indistinguishable from an edit on the frame pair alone.
    """
    later = [carry(thumbs[i - 1], thumbs[i + j]) for j in range(2, span + 1) if i + j < len(thumbs)]
    return max(later) if later else -1.0


def bracket(video: Path, t: float, fps: float, k: int = BRACKET_FRAMES, width: int = 320) -> list:
    """The `k` frames before `t` and the `k` from `t` on, in one decode of that stretch.

    Half a frame of slack puts the seek on the frame before `t`, the same rounding the trimmer
    relies on, so the returned list splits exactly at the candidate.
    """
    import cv2, numpy as np

    step = 1.0 / max(fps, 1e-3)
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(t - (k + 0.5) * step, 0):.4f}",
            "-i",
            str(video),
            "-frames:v",
            str(2 * k),
            "-vf",
            f"scale={width}:-2",
            "-f",
            "image2pipe",
            "-vcodec",
            "bmp",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    out, at = [], 0
    while at + 6 <= len(raw):
        size = int.from_bytes(raw[at + 2 : at + 6], "little")
        if size <= 0 or at + size > len(raw):
            break
        out.append(cv2.imdecode(np.frombuffer(raw[at : at + size], np.uint8), cv2.IMREAD_GRAYSCALE))
        at += size
    return [f for f in out if f is not None]


def orb_fraction(a, b) -> float | None:
    """Fraction of ORB keypoints that survive from `a` to `b`, or None if there are too few."""
    import cv2

    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None  # too featureless to judge
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    return len([x for x in m if x.distance < 48]) / min(len(ka), len(kb))


def match_across(video: Path, t: float, fps: float, k: int = BRACKET_FRAMES) -> float | None:
    """Best ORB fraction across `t` over a few bracketing pairs, or None if all were featureless.

    Motion blur removes the corners ORB keys on, so the pair straddling the candidate can be
    unmatchable while the shot either side is perfectly sharp. The sharpest frame on each side is
    tried as well as the immediate pair, and the best result stands: one pair that matches is
    proof the content carried across, while a cut leaves every pair in the window with nothing to
    pair against.
    """
    import cv2

    frames = bracket(video, t, fps, k)
    if len(frames) < 2:
        return None
    before, after = frames[: len(frames) // 2], frames[len(frames) // 2 :]
    if not before or not after:
        return None
    sharp = [float(cv2.Laplacian(f, cv2.CV_32F).var()) for f in frames]
    sb = max(range(len(before)), key=lambda i: sharp[i])
    sa = max(range(len(after)), key=lambda i: sharp[len(before) + i])
    pairs = {(len(before) - 1, 0), (sb, sa), (sb, 0), (len(before) - 1, sa)}
    got = [orb_fraction(before[i], after[j]) for i, j in sorted(pairs)]
    got = [v for v in got if v is not None]
    return max(got) if got else None


def detect_cuts(
    video: Path, threshold: float = SCENE_CANDIDATE, fps: float | None = None
) -> list[dict]:
    """Every candidate frame, each marked cut or not, with the evidence that decided it.

    The cleared candidates are returned too, each with a concrete reason: a frame that scored 0.30
    and was cleared is where the operator should look if the run later trips the teleport guard.

    Order is deliberate. Everything except the ORB match comes free out of the scoring pass, so the
    candidates that fast motion produced -- the overwhelming majority on an action clip -- are
    settled without decoding a single frame again.
    """
    fps = fps or ffprobe(video)["fps"]
    times, scores, thumbs = scene_pass(video)
    if len(thumbs) < 2:
        return []
    carried = carry_series(thumbs)
    change = [1.0 - c for c in carried]
    seen, candidates = set(), []
    for i, score in enumerate(scores):
        if score <= threshold or times[i] <= 0.05:
            continue
        # Move to the frame the carry series says the change is really on; see REFINE_FRAMES.
        # `time` and `frame` then name that frame, while `score` stays the highest scene score that
        # made this a candidate -- the two can sit a frame apart, which is the point of moving.
        lo, hi = max(i - REFINE_FRAMES, 1), min(i + REFINE_FRAMES + 1, len(carried))
        at = min(range(lo, hi), key=lambda k: carried[k]) if lo < hi else i
        if at not in seen:
            seen.add(at)
            candidates.append((at, max(score, scores[at])))
    out = []
    for at, score in sorted(candidates):
        rec = dict(
            time=round(times[at], 3),
            frame=at,
            score=round(score, 4),
            carry=round(carried[at], 3),
            returns=None,
            isolation=None,
            match=None,
            cut=False,
            why="",
        )
        if carried[at] >= CARRY_CARRIES:
            rec["why"] = f"content carries across it ({rec['carry']} of the frame correlates)"
            out.append(rec)
            continue
        rec["returns"] = round(returns_after(thumbs, at), 3)
        if rec["returns"] >= CARRY_RETURNS:
            rec["why"] = (
                f"the view comes back {rec['returns']} within {CARRY_RETURN_FRAMES} frames, so "
                f"something crossed the lens; a cut does not come back"
            )
            out.append(rec)
            continue
        rec["isolation"] = round(isolation(change, at), 2)
        if rec["isolation"] < ISOLATION_MIN:
            rec["why"] = (
                f"the frames around it disagree as much as it does ({rec['isolation']}x the "
                f"local median, under {ISOLATION_MIN}): sustained motion, not one edit"
            )
            out.append(rec)
            continue
        try:
            rec["match"] = match_across(video, times[at], fps)
        except Exception:
            rec["match"] = None
        if rec["match"] is not None:
            rec["match"] = round(rec["match"], 3)
            if rec["match"] >= CUT_MATCH_MAX:
                rec["why"] = f"blurred, but {rec['match']} of its features still match across it"
                out.append(rec)
                continue
        rec["cut"] = True
        rec["why"] = (
            f"a cut: {rec['carry']} of the frame correlates with the one before, the view does not "
            f"come back ({rec['returns']}), the disagreement is {rec['isolation']}x its "
            f"neighbourhood, and "
            + (
                f"only {rec['match']} of the features match"
                if rec["match"] is not None
                else "both sides are too blurred for features to judge"
            )
        )
        out.append(rec)
    return out


def shots_from_cuts(cuts: list[dict], duration: float, min_seconds: float) -> list[dict]:
    bounds = [0.0] + [c["time"] for c in cuts] + [duration]
    shots = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        shots.append(
            dict(
                index=i,
                start=round(a, 3),
                end=round(b, 3),
                seconds=round(b - a, 3),
                tooShort=(b - a) < min_seconds,
            )
        )
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
    candidates = [s for s in shots if not s["tooShort"]]
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
    scored = [s for s in shots if "score" in s]
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
    candidates = detect_cuts(video, threshold, info["fps"])
    cuts = [c for c in candidates if c["cut"]]
    shots = shots_from_cuts(cuts, info["duration"], min_seconds)
    doc = dict(
        video=str(video),
        source=info,
        threshold=threshold,
        minSeconds=min_seconds,
        cuts=cuts,
        cutCount=len(cuts),
        cleared=[c for c in candidates if not c["cut"]],
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
    a = ap.parse_args()
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
