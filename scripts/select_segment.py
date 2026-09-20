#!/usr/bin/env python3
"""WHICH part of an arbitrary upload should be reconstructed? Measured, not judged by eye.

`scripts/shot_cuts.py` answers "where are the cuts" and then ranks WHOLE shots, only the twelve
longest, with a score that treats "exactly one person" as ideal. That is the wrong question for a
three-minute basketball clip (one take, many plays, five subjects), for a movie scene (fifty shots,
most of them useless), and for a thirty-second game capture (several locations, some cuts missed).
This stage answers the product's actual question -- give me the best ~12 seconds -- and writes down
every candidate it considered so the choice is reviewable instead of remembered.

What it does:

  * takes the cuts from a shot_cuts report, from `--truth` boundaries a human verified, or by
    calling `shot_cuts.cut_report(..., score=False)` itself;
  * splits a shot further at any SUSPECTED BREAK -- a pair of samples that does not correlate at
    all -- because the detector under-reports on dark and fast footage;
  * lays candidate WINDOWS inside each segment (the whole segment when it is short enough, sliding
    windows when it is longer than `--max-window`), never crossing a cut or a break, a frame clear
    of each boundary, exactly as `shot_cuts.trim` cuts;
  * measures each window from sparsely sampled, downscaled frames -- camera translation
    (worker/stages/parallax_probe.py's homography/fundamental degeneracy ratio, computed on the
    cached frames instead of a JPEG directory), people (torchvision Mask R-CNN, the same detector
    and the same 0.7 score and 0.2 box-height cut as worker/wander_worker/masks.py), sharpness,
    exposure, static overlay/letterbox, within-window discontinuity, whether the window ends in
    the place it started, and duration;
  * scores them with one documented weighted sum whose constants are named and general, applies
    hard vetoes, and greedily picks the top K non-overlapping windows;
  * writes `selection.json` (schema `wander.segment-selection/1`) with EVERY candidate, its
    features, its score, its vetoes and a one-line `why`.

It cannot know intent. It cannot know which play is famous, which line of dialogue matters, or
which of two equally reconstructable windows the operator wanted. `--force-window START END
--reason "..."` records a human override with its reason and the score it overrode; an optional
VLM tie-break is prepared as `judge_request.json` and its answer is recorded as an OPINION only.

  uv run --locked --group inference python scripts/select_segment.py --video clip.mp4 \
      --out .context/evidence/new-clips/selection/clip/selection.json --contact-sheet sheets/

No network, no GPU, no paid stage. See docs/segment-selection.md.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "worker/stages"))

SCHEMA = "wander.segment-selection/1"
JUDGE_SCHEMA = "wander.segment-judge-request/1"

# scripts/shot_cuts.py is the sibling stage this one builds on: its `carry` is the continuity
# measure below, its `cut_report` is the fallback cut source, and its thumbnail size fixes the
# units of both. It is also a file other people edit, and a half-saved edit there must not take
# this stage (or its tests) down with a traceback from someone else's module. So the import is
# attempted, the four names actually used are checked, and a failure degrades to the local
# fallbacks below with the reason recorded in the report -- the same contract the missing person
# detector gets. The fallbacks are deliberately the only duplicated logic in this file, and
# `document.featuresAvailable.shotCuts` is false whenever they are in use, because a local copy of
# `carry` cannot follow a change made to the original.
THUMB_WIDTH, THUMB_HEIGHT = 128, 72
CARRY_SEARCH_X, CARRY_SEARCH_Y = 28, 14


def _fallback_carry(previous, current) -> float:
    """shot_cuts.carry's contract: best normalised correlation of `current`'s centre over `previous`."""
    import cv2

    patch = current[
        CARRY_SEARCH_Y : THUMB_HEIGHT - CARRY_SEARCH_Y,
        CARRY_SEARCH_X : THUMB_WIDTH - CARRY_SEARCH_X,
    ]
    return float(cv2.matchTemplate(previous, patch, cv2.TM_CCOEFF_NORMED).max())


def _load_shot_cuts():
    try:
        import shot_cuts as module

        for name in ("carry", "cut_report", "THUMB_WIDTH", "THUMB_HEIGHT"):
            getattr(module, name)
    except Exception as exc:  # a broken sibling degrades this stage; it does not fail it
        return None, f"{type(exc).__name__}: {exc}"
    return module, None


shot_cuts, SHOT_CUTS_ERROR = _load_shot_cuts()
if shot_cuts is not None:
    THUMB_WIDTH, THUMB_HEIGHT = shot_cuts.THUMB_WIDTH, shot_cuts.THUMB_HEIGHT


def carry(previous, current) -> float:
    return shot_cuts.carry(previous, current) if shot_cuts else _fallback_carry(previous, current)


class BadRequest(ValueError):
    """An argument this stage refuses, as opposed to anything that went wrong while measuring.

    Only this is turned into a one-line CLI message. Everything else keeps its traceback, because
    an unexpected failure inside a dependency once came back as the bare words "too many values to
    unpack" with nothing saying where.
    """


# ---------------------------------------------------------------- window geometry
# A window is what the pipeline will actually reconstruct, so its length is bounded by what the
# rest of the pipeline has been shown to survive rather than by anything about a particular clip.
#   MIN_WINDOW    below this the camera solve has too little baseline and the world has too little
#                 coverage; shot_cuts.score_shot already scores 2 s as the floor and 12 s as full
#                 marks, and no shipped preset is shorter than 3.5 s.
#   TARGET_WINDOW the length sliding windows are cut at. Every shipped preset is 4-12 s, Marble is
#                 submitted the cleaned clip at 12 fps, and the person stages cost time per frame:
#                 12 s is the longest span that has gone end-to-end here without the cost or the
#                 drift becoming the problem.
#   MAX_WINDOW    a shot longer than this is not scored whole, because its best 12 s and its worst
#                 12 s average into one meaningless number -- that averaging is the specific defect
#                 this stage exists to remove.
#   MIN_USABLE    a shot shorter than this yields no window at all. shot_cuts.MIN_SHOT_SECONDS is
#                 1.5 s for "was this a real cut"; 2.5 s is the floor for "is this worth
#                 reconstructing", which is a stricter question.
MIN_WINDOW_SECONDS = 6.0
TARGET_WINDOW_SECONDS = 12.0
MAX_WINDOW_SECONDS = 20.0
MIN_USABLE_SECONDS = 2.5
# How far one sliding window starts after the last. 3 s over a 12 s window is a 75% overlap, which
# is dense enough that a 12 s span of good action cannot fall between two candidates.
WINDOW_STRIDE_SECONDS = 3.0
# Boundary margin, in frames, copied from shot_cuts.trim: a cut time is the timestamp of the FIRST
# frame of the new shot, and ffmpeg's seek and duration both round to whole frames, so half a frame
# in at the head and a frame and a half off the tail is what keeps a foreign frame out of the
# segment. A chosen window is therefore directly trimmable by the same convention.
MARGIN_HEAD_FRAMES = 0.5
MARGIN_TAIL_FRAMES = 1.5

# ---------------------------------------------------------------- sampling
# Everything below is measured from these samples, so they set both the cost and the resolution of
# every feature.
# MEASURED, and the one constant here that was wrong on first contact with real footage. At 3
# samples per second the within-window discontinuity check was useless on the 29 s game capture in
# .context/evidence/new-clips/selection/game-LvpSn: five frames fell under the threshold and only
# one of them was a real discontinuity, while the clip's most violent transition (a dark underwater
# shot to a bright sea surface) smeared across two samples and scored 0.41, above the threshold
# entirely. At 6 samples per second that clip's two genuine transitions measure 0.23 and 0.05 and
# every other pair in it stays at or above 0.38, because 1/6 s is short enough that fast motion
# still correlates and an edit still does not. That is one clip: what the same measure does over
# the whole corpus, and how little of a cut detector it therefore is, is under INTERNAL_CUT_CARRY.
# The cost of doubling the rate is one more decode pass and a Laplacian per frame; it does not
# touch the detector or the ORB pairs, which are what the runtime is actually made of.
SAMPLE_FPS = 6.0
# One fixed analysis width for every clip, so "variance of Laplacian" and "parallax ratio" mean the
# same number on a 4K phone clip and a 720p download. 720 px is what parallax_probe already probes
# at, so the ratios below are directly comparable with its output.
ANALYSIS_WIDTH = 720
# Mask R-CNN resizes its input to its own 800/1333 range, so the width here only controls decode
# cost and the box fractions, not detector resolution. 640 is what shot_cuts.person_stats feeds it.
PEOPLE_WIDTH = 640
# A Mask R-CNN forward pass is about 5 s per frame on this 4-core CPU box, so the detector is
# capped by frame COUNT rather than by rate: a 30 s clip and a 3 minute clip cost the same. 48
# frames spread over the source is ~4 minutes of CPU and leaves a 12 s window 3-6 detector samples.
PEOPLE_MAX_FRAMES = 48
PEOPLE_MIN_STRIDE_SECONDS = 0.5

# ---------------------------------------------------------------- camera translation
# Gaps, in seconds, at which the degeneracy ratio is measured. parallax_probe takes its verdict
# from the WIDEST separation the clip supports, because a slow dolly looks rotational between
# adjacent frames; 3 s is the widest gap a 6 s window can still hold several pairs of.
CAMERA_GAP_SECONDS = (0.33, 1.0, 3.0)
# One pair per second per gap. parallax_probe uses 12 pairs per gap, which is what a 12 s window
# gets here.
PAIR_STRIDE_SECONDS = 1.0
# Below this many successful pairs a gap is not evidence, matching parallax_probe's refusal to
# report a gap it could not match.
CAMERA_MIN_PAIRS = 4
# parallax_probe.ROTATION_ONLY_RATIO: at or above this the widest baseline still produces nothing a
# single homography cannot explain, and SfM will collapse. Imported rather than copied.
ROTATION_ONLY_RATIO = 0.92
# Below this the parallax is unambiguous. The gap between the two is graded rather than stepped, so
# "more translation" always scores higher instead of jumping at one threshold. 0.80 is the top of
# the has-parallax band measured on this repo's eight clips (docs/experiments/any-clip-pipeline.md
# reports the split as clean and wide); it is a shape constant, not a per-clip tuning.
PARALLAX_RATIO_CLEAR = 0.80
TRANSLATION_UNKNOWN = 0.3  # same degraded value shot_cuts.score_shot gives an inconclusive probe
# At or above this ratio the widest baseline in the window produced nothing a single flat warp
# cannot explain. parallax_probe states the consequence as a failure, not a penalty: "incremental
# SfM will fail to fix its gauge and collapse into three-frame fragments". A broadcast camera
# panning on a tripod is the common case and it is worth refusing to spend on, so this is a hard
# veto as well as a zero on the dominant term -- reported as such, with the ratio, so the operator
# can see it was measured rather than assumed, and can still take the window with --force-window.
ROTATION_ONLY_VETO = True

# ---------------------------------------------------------------- one place, or a traverse
# Tonight's measured lesson: a 22 s run through a game world spanned ten times the scene depth and
# failed anchor overlap -- the solve had no single map to be consistent in. The cheap question that
# speaks to it is whether the window ENDS where it began, measured on the features already cached
# for the parallax pairs: crossCheck ORB matches between the first and last sample that survive a
# RANSAC fundamental matrix, over the smaller keypoint count -- see overlap_lookup.
#
# MEASURED, and the numbers are why it is REPORTED BUT NOT SCORED. Over 12 s windows:
#   game capture, the known-bad traverse   0.013 - 0.033
#   creed, one boxing ring throughout      0.021 - 0.131  (the operator's 72-84 s is the 0.131)
#   movie, its locked-off closing shot     0.92
# The two populations that matter already overlap. Worse, on the clip the idea came from it points
# the wrong way: the game capture's 12 s traverse through three flooded chambers measures 0.027,
# and the window after its cut -- one character on one circular platform, which is by eye the best
# thing in the clip to reconstruct -- measures 0.012. A camera orbiting one place changes its view
# as thoroughly as a camera leaving it, and ORB cannot tell those apart at a 12 s baseline.
# So the feature is measured, reported on every candidate, and given a DEFAULT WEIGHT OF ZERO: it
# is evidence for a future calibration, not a number this stage is willing to rank on. `--weights`
# can turn it on for footage where it has been shown to work. What actually keeps a 22 s traverse
# out of one solve is MAX_WINDOW_SECONDS, which does not depend on this measure at all.
PLACE_OVERLAP_FLOOR = 0.015  # the measured noise floor: RANSAC's accidental fit on ~700 matches
PLACE_OVERLAP_CLEAR = 0.100
PLACE_UNKNOWN = 0.5  # too featureless to ask; degraded, not assumed good
# A window needs at least this many samples before its ends are worth comparing.
PLACE_MIN_SAMPLES = 3
# One pair of frames is a noisy sample of "do these ends share a place": a single motion-blurred or
# featureless end frame is enough to halve it, and on a first pass with one pair the measure jumped
# between 0.000 and 0.039 on adjacent windows of the same shot. The best of the pairs within one
# sample of each end is taken instead, so one bad frame cannot decide it.
PLACE_ENDPOINT_NEIGHBOURS = 1

# ---------------------------------------------------------------- people
# Same detector, same confidence and same box-height cut as worker/wander_worker/masks.py, so
# "reconstructable subject" means here exactly what it means to the cleaner and the person stages.
PERSON_SCORE = 0.7
SUBJECT_MIN_HEIGHT_FRAC = 0.2
# The pipeline's current cap on reconstructed people. Subject counts from 1 to the cap score the
# same: a fight, a dance or a three-on-three is not worse input than one person walking, which is
# the assumption shot_cuts.score_shot bakes in and this stage removes.
PEOPLE_CAP = 4
# Above the cap the extra subjects will not be reconstructed, so the window is partly
# misrepresented; it is not worthless. Full marks at the cap, falling to this floor at twice it.
PEOPLE_OVER_CAP_FLOOR = 0.3
# A window wants a subject present most of the time, not in one frame of it. At or above this
# fraction of sampled frames the presence term is full.
PRESENCE_TARGET = 0.6
# Foreground person pixels as a fraction of the frame. Below the first number the world generator
# still sees the room; above the second the people ARE the picture, the cleaner has to inpaint most
# of the frame, and what comes back is invented. Measured reference points: this repo's phone and
# game clips run 0.05-0.25, the stadium celebration frame in masks.py runs about 0.3.
PERSON_PIXELS_OK = 0.35
PERSON_PIXELS_BLOCKING = 0.60
PERSON_PIXELS_BLOCKING_FACTOR = 0.15
# A subject count that changes every sample means people entering and leaving, which the person
# stages track badly. Instability costs at most this much of the people term.
STABILITY_FLOOR = 0.6
PEOPLE_UNKNOWN = 0.5  # same degraded value shot_cuts.score_shot uses with no detector
# Inherited verbatim from shot_cuts.score_shot: 0.25 of frame height is about the smallest the
# avatar chain has worked from, 0.7 a comfortable full figure, and taller is a close-up that is no
# better.
SUBJECT_HEIGHT_FLOOR = 0.25
SUBJECT_HEIGHT_FULL = 0.70

# ---------------------------------------------------------------- image quality
# Variance of the Laplacian of the grey frame, 0-255, measured at ANALYSIS_WIDTH so the number does
# not depend on the source resolution. Soft footage and heavy motion blur sit under the floor;
# ordinary handheld video sits well above the ceiling. MEASURED on the three real clips under
# .context/evidence/new-clips/selection: windows run 64-848, so in practice this term is a FLOOR
# GUARD against a window that is entirely motion-blurred, not a way to rank good footage against
# better. Both numbers are re-measured per clip in the report (`image.sharpness`), so an operator
# can see where a clip actually fell.
SHARPNESS_FLOOR = 20.0
SHARPNESS_GOOD = 200.0
# Fraction of pixels at the ends of the range. docs/known-limits.md: clips too dark fail
# reconstruction, so darkness is both a graded penalty and, past DARK_VETO, a hard veto.
DARK_LUMA = 16  # 0-255
BRIGHT_LUMA = 246
DARK_OK = 0.25
DARK_BAD = 0.60
DARK_VETO = 0.60
BRIGHT_OK = 0.10
BRIGHT_BAD = 0.35
# Static overlay / letterbox as a fraction of the frame. A scoreboard or a black bar never moves,
# so SfM matches it perfectly and the solve is pulled towards "the camera did not move"; the world
# generator then paints the overlay into the room. Reported always, penalised above OVERLAY_OK.
OVERLAY_OK = 0.05
OVERLAY_BAD = 0.30
OVERLAY_MAX_PENALTY = 0.7
# A pixel counts as static when its whole-window peak-to-peak deviation is at or under this, on a
# 0-255 grey thumbnail. Small enough to exclude compression noise, large enough to include a
# semi-transparent HUD.
STATIC_DEVIATION = 6
# When this much of the frame never changes, the camera is probably locked off and the static-pixel
# measure cannot tell a HUD from a wall. The window is then scored on the letterbox bars alone and
# the report says the overlay measure was suppressed.
STATIC_CAMERA_SUSPECT = 0.60
LETTERBOX_LUMA = 20  # a bar row is at least LETTERBOX_ROW_DARK of it this dark
LETTERBOX_ROW_DARK = 0.98
LETTERBOX_MAX_BAND = 0.25  # bars are searched only this far in from each edge
# In what fraction of sampled frames a row must be dark before it counts as a bar. NOT 1.0, and the
# measurement that settled it: on the 99 s movie scene the bar rows are 98% dark in 0.84-0.86 of
# sampled frames and the picture rows in 0.00-0.02, so the two populations are separated by the
# whole interval. Requiring every frame found NO bars at all, because the last ~15 s of that clip is
# not letterboxed -- which is exactly the 16% of frames that were failing the test. The measured
# quorum is reported as `barFrameFraction`, so a clip whose aspect changes part way through says so
# instead of being silently cropped as if it did not.
LETTERBOX_FRAME_QUORUM = 0.75

# ---------------------------------------------------------------- within-window continuity
# The cut detector under-detects on dark and fast real footage, so a window can straddle a break it
# never reported. shot_cuts.carry (normalised correlation of a 128x72 grey thumbnail with a shift
# search) is reused here between CONSECUTIVE SAMPLES, which are 1/SAMPLE_FPS apart rather than one
# frame apart, so the thresholds are lower than shot_cuts' own carry gate by construction: real
# motion decorrelates much more over 1/6 s than over 1/24 s.
#
# This is NOT a second cut detector and is not trying to be. It answers the narrower question the
# selector actually needs: does anything inside this window break continuity badly enough that the
# camera solve will not register across it? An edit does. So does a camera crossing a water
# surface, a light being switched off, or a body passing through the whole lens -- and all of those
# ruin the same solve, so all of them are worth refusing.
#
# MEASURED over the three real clips, and the measurement is mostly a limit on what this gate can
# be asked to do. At 1/6 s spacing the two populations OVERLAP:
#   11 cuts of the movie scene, each one in a reviewed truth file   carry 0.050 - 0.277
#   1638 sample pairs of creed-long-take, a verified SINGLE TAKE    carry 0.166 - 0.999
# There is no threshold on this number that finds every cut without calling violent handheld
# motion a cut: at 0.30 it split that single take into 47 pieces and no 12 s window survived.
# So the gate is set where the evidence is unambiguous -- BELOW everything 1638 pairs of real
# continuous motion reached, and above both independently verified cuts in the corpus (the game
# capture's cut at 22.375 s carries 0.047, the movie's at 48.674 s carries 0.050). What it will
# NOT catch is a real cut carrying 0.10-0.28, and nothing here pretends otherwise: those are the
# cut detector's job, and the movie run shows the report and the truth file supplying them.
# Between the two numbers the term is graded, so motion that nearly broke continuity still costs
# score without refusing the window, and every window carries its own minCarry in the report.
INTERNAL_CUT_CARRY = 0.10  # at or below this the samples either side do not belong in one window
CONTINUITY_CLEAR = 0.60  # at or above this the continuity term is full

# ---------------------------------------------------------------- score
# One weighted sum, in [0, 1]. Every weight is a claim about what has actually decided whether a
# clip reconstructed here, not about this or that clip:
#   translation  dominant, and deliberately still dominant after every other term was added. A
#                window with no camera travel cannot be reconstructed AT ALL (known-limits, and
#                shot_cuts.score_shot's own rationale): no second viewpoint, no depth, nothing for
#                the other six terms to be good about.
#   people       the subjects are the product. Zero subjects means nothing to replay; more than the
#                cap means the replay will be wrong about who was there; a frame full of people
#                means the world behind them is invented.
#   continuity   a missed cut inside the window is the single failure this whole area exists to
#                prevent (shot_cuts' opening paragraph): it fuses two places into one world.
#                Weighted above image quality because its consequence is worse.
#   place        zero, and the only weight here set by what the measure could NOT do rather than
#                by what the failure costs. The failure is real and expensive; the measurement of
#                it did not survive its own test, so it is reported and not ranked on. See the
#                place constants above for the numbers that decided that.
#   fullBody     a subject cropped at the waist gives the avatar chain no legs to build.
#   subjectHeight  pixel height decides whether an avatar can be fitted at all.
#   image        sharpness, exposure and overlays degrade the solve and the generated world, but
#                none of them is fatal on its own the way the first three are.
#   duration     longer is better up to TARGET_WINDOW, and only mildly: a clean 8 s beats a messy
#                12 s.
DEFAULT_WEIGHTS = {
    "translation": 0.40,
    "people": 0.22,
    "continuity": 0.12,
    "place": 0.00,
    "fullBody": 0.07,
    "subjectHeight": 0.05,
    "image": 0.08,
    "duration": 0.06,
}
# Inherited from shot_cuts.score_shot: 2 s is the least a solve has held together on, 12 s plenty.
DURATION_FLOOR_SECONDS = 2.0
DURATION_FULL_SECONDS = 12.0
# Greedy top-K rejects a window overlapping an already chosen one by more than this fraction of the
# shorter of the two. 0 asks for genuinely distinct segments, which is the point of returning K.
MAX_OVERLAP_FRACTION = 0.0
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------- small helpers
def ramp(value: float, low: float, high: float) -> float:
    """0 at `low`, 1 at `high`, linear between, clamped. `low` may be above `high` (a falling ramp)."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- source
def probe_video(video: Path) -> dict:
    """ffprobe's stream facts plus the colour transfer, which decides the decode path."""
    raw = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,pix_fmt,"
                "color_transfer,color_primaries",
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
    st = raw["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(st.get("duration") or raw.get("format", {}).get("duration") or 0.0)
    return dict(
        width=int(st["width"]),
        height=int(st["height"]),
        fps=fps,
        frames=int(st.get("nb_frames") or 0),
        durationSeconds=round(duration, 3),
        codec=st.get("codec_name"),
        pixelFormat=st.get("pix_fmt"),
        colorTransfer=st.get("color_transfer"),
        colorPrimaries=st.get("color_primaries"),
    )


# HLG (arib-std-b67) and PQ (smpte2084) footage decoded as if it were BT.709 comes out flat and
# washed out: every exposure and sharpness number would then describe the decode, not the clip, and
# a phone HLG clip would read as "no near-black pixels" however dark it really is. Tone-map it
# instead, and record in the report that this happened.
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}


def tonemap_chain(info: dict) -> str:
    if (info.get("colorTransfer") or "").lower() not in HDR_TRANSFERS:
        return ""
    return "zscale=t=linear:npl=100,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"


def decode_grey(video: Path, info: dict, sample_fps: float, width: int):
    """One decode pass -> (times, frames uint8 (N, H, W)) of grey samples at `sample_fps`.

    The sample grid is the unit every feature below is measured on, and it is decoded ONCE: the
    sliding windows all share it, so a 75%-overlapping window costs no extra decoding.
    """
    import numpy as np

    height = max(2, 2 * round(info["height"] * width / info["width"] / 2))
    chain = f"fps={sample_fps:.6f}," + tonemap_chain(info) + f"scale={width}:{height},format=gray"
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
            chain,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    frames = np.frombuffer(raw, np.uint8)
    count = len(frames) // (width * height)
    frames = frames[: count * width * height].reshape(count, height, width)
    times = [i / sample_fps for i in range(count)]
    return times, frames


def decode_people_frames(video: Path, info: dict, times: list[float], width: int):
    """RGB samples at the requested times, one decode pass, for the detector only."""
    import numpy as np

    if not times:
        return np.zeros((0, 2, 2, 3), np.uint8)
    height = max(2, 2 * round(info["height"] * width / info["width"] / 2))
    picks = "+".join(f"eq(n\\,{round(t * info['fps'])})" for t in times)
    chain = f"select='{picks}'," + tonemap_chain(info) + f"scale={width}:{height},format=rgb24"
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
            chain,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    frames = np.frombuffer(raw, np.uint8)
    stride = width * height * 3
    count = len(frames) // stride
    return frames[: count * stride].reshape(count, height, width, 3)


# ---------------------------------------------------------------- per-frame measurements
def letterbox_bars(frames, max_band: float = LETTERBOX_MAX_BAND, sample_every: int = 4) -> dict:
    """The clip's black bars, as pixel counts off each edge of the analysis frame.

    MEASURED, and the reason this function exists: the 99 s letterboxed movie scene in
    .context/evidence/new-clips/selection/movie-JhvaJa47FHY is 2.39:1 in a 16:9 file, so 26% of
    every frame (53 rows off the top, 52 off the bottom at 720 px wide) is an identical black bar.
    Left in, those bars do two concrete kinds of damage.
    They are 26 points of `darkFraction` that have nothing to do with the exposure of the picture,
    which pushes a merely dim scene towards the too-dark veto. And because they correlate
    perfectly with themselves, they inflate `carry` ACROSS a cut: the missed cut at 76.5 s in that
    clip -- two shots of the same green courtyard -- still carried 0.43 with the bars in.

    So the bars are found once for the whole clip and cropped off before anything is measured.
    A row or column is a bar when at least LETTERBOX_ROW_DARK of it sits at or under LETTERBOX_LUMA
    in at least LETTERBOX_FRAME_QUORUM of the sampled frames, and only within `max_band` of its own
    edge -- so a dark sky, a shadowed floor or one dark shot cannot be mistaken for one.
    """
    import numpy as np

    if len(frames) == 0:
        return dict(topPx=0, bottomPx=0, leftPx=0, rightPx=0, fractionOfFrame=0.0)
    sub = frames[:: max(1, sample_every)] <= LETTERBOX_LUMA
    height, width = sub.shape[1], sub.shape[2]
    row_quorum = (sub.mean(axis=2) >= LETTERBOX_ROW_DARK).mean(axis=0)
    col_quorum = (sub.mean(axis=1) >= LETTERBOX_ROW_DARK).mean(axis=0)

    def run(values, limit: int) -> int:
        n = 0
        while n < limit and values[n] >= LETTERBOX_FRAME_QUORUM:
            n += 1
        return n

    top = run(row_quorum, int(height * max_band))
    bottom = run(row_quorum[::-1], int(height * max_band))
    left = run(col_quorum, int(width * max_band))
    right = run(col_quorum[::-1], int(width * max_band))
    edges = (
        list(row_quorum[:top])
        + list(row_quorum[height - bottom :])
        + list(col_quorum[:left])
        + list(col_quorum[width - right :])
    )
    kept = max(height - top - bottom, 1) * max(width - left - right, 1)
    return dict(
        topPx=int(top),
        bottomPx=int(bottom),
        leftPx=int(left),
        rightPx=int(right),
        fractionOfFrame=round(1.0 - kept / (height * width), 4),
        topFraction=round(top / height, 4),
        bottomFraction=round(bottom / height, 4),
        leftFraction=round(left / width, 4),
        rightFraction=round(right / width, 4),
        # The weakest bar row's quorum: 1.0 means the whole clip is letterboxed, and anything less
        # means the aspect ratio changes somewhere and this crop takes real picture from that part.
        barFrameFraction=round(float(np.min(edges)), 4) if edges else 0.0,
        frameQuorum=LETTERBOX_FRAME_QUORUM,
    )


def suggested_crop(bars: dict, info: dict) -> dict:
    """The measured bars as a crop of the SOURCE frame, ready to hand to ffmpeg.

    The bars are found on the 720 px analysis frame, so they come back as fractions and are scaled
    here to source pixels and rounded to even numbers, because yuv420p cannot encode an odd
    dimension. `barFrameFraction` travels with it: below 1.0 the clip is not letterboxed all the
    way through and this crop takes real picture from the rest of it.
    """
    width, height = info["width"], info["height"]

    def even(value: float) -> int:
        return 2 * int(round(value / 2))

    top = even(bars.get("topFraction", 0.0) * height)
    bottom = even(bars.get("bottomFraction", 0.0) * height)
    left = even(bars.get("leftFraction", 0.0) * width)
    right = even(bars.get("rightFraction", 0.0) * width)
    keep_w, keep_h = width - left - right, height - top - bottom
    if keep_w < 16 or keep_h < 16 or (top, bottom, left, right) == (0, 0, 0, 0):
        return dict(needed=False, ffmpeg=None, reason="no black bars measured on any edge")
    return dict(
        needed=True,
        x=left,
        y=top,
        width=keep_w,
        height=keep_h,
        ffmpeg=f"crop={keep_w}:{keep_h}:{left}:{top}",
        barFrameFraction=bars.get("barFrameFraction", 0.0),
        measuredAtWidth=ANALYSIS_WIDTH,
    )


def crop_bars(frames, bars: dict):
    """Drop the measured bars. Works for grey (N,H,W) and RGB (N,H,W,3) stacks alike."""
    height, width = frames.shape[1], frames.shape[2]
    top = int(round(bars.get("topFraction", 0.0) * height))
    bottom = height - int(round(bars.get("bottomFraction", 0.0) * height))
    left = int(round(bars.get("leftFraction", 0.0) * width))
    right = width - int(round(bars.get("rightFraction", 0.0) * width))
    if bottom - top < 8 or right - left < 8:
        return frames  # a crop that leaves nothing is not a crop; keep the frame and say so
    return frames[:, top:bottom, left:right]


def frame_measurements(frames, times: list[float]) -> list[dict]:
    """Per sampled frame: sharpness, exposure, and the carry from the previous sample.

    Cached once for the whole video; a window is an aggregate over a slice of this list, which is
    what makes 75%-overlapping candidates nearly free.
    """
    import cv2
    import numpy as np

    out: list[dict] = []
    thumbs = [
        cv2.resize(f, (THUMB_WIDTH, THUMB_HEIGHT), interpolation=cv2.INTER_AREA) for f in frames
    ]
    for i, f in enumerate(frames):
        out.append(
            dict(
                index=i,
                time=round(times[i], 4),
                sharpness=float(cv2.Laplacian(f, cv2.CV_32F).var()),
                darkFraction=float((f <= DARK_LUMA).mean()),
                brightFraction=float((f >= BRIGHT_LUMA).mean()),
                meanLuma=float(np.mean(f)),
                carryFromPrevious=None if i == 0 else carry(thumbs[i - 1], thumbs[i]),
            )
        )
    return out


def orb_cache(frames):
    """Lazy per-frame ORB keypoints/descriptors, so a frame in three pairs is detected once."""
    import cv2
    import numpy as np

    orb = cv2.ORB_create(3000)
    store: dict[int, tuple] = {}

    def get(i: int):
        if i not in store:
            kp, des = orb.detectAndCompute(frames[i], None)
            pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), "float32")
            store[i] = (pts, des)
        return store[i]

    return get


def overlap_lookup(feature, bf):
    """(i, j) -> how much of sample i is still in sample j: matched, geometrically checked, memoised.

    The fraction of crossCheck ORB matches that survive a RANSAC fundamental matrix, over the
    smaller keypoint count. Two views of one room from different angles still share an epipolar
    geometry; two different places only fit one by accident.

    The obvious alternative was shot_cuts.orb_fraction itself -- crossCheck matches under
    ORB_MATCH_DISTANCE Hamming, the gate that separates cuts from non-cuts on ADJACENT frames.
    MEASURED at the 6-12 s gaps a window actually spans, it reads 0.000-0.005 whether the footage
    stayed in one room or left it: it answers "is this the same moment", not "is this the same
    place", so it is not used here.

    Memoised, because neighbouring windows share endpoints and detection is already cached.
    """
    import cv2
    import numpy as np

    store: dict[tuple[int, int], float | None] = {}

    def get(i: int, j: int) -> float | None:
        key = (min(i, j), max(i, j))
        if key in store:
            return store[key]
        pts_a, des_a = feature(key[0])
        pts_b, des_b = feature(key[1])
        if des_a is None or des_b is None or len(pts_a) < 30 or len(pts_b) < 30:
            store[key] = None  # too featureless to judge, exactly as orb_fraction refuses
            return None
        matches = bf.match(des_a, des_b)
        overlap = 0.0
        if len(matches) >= 8:
            src = np.float32([pts_a[m.queryIdx] for m in matches])
            dst = np.float32([pts_b[m.trainIdx] for m in matches])
            F, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.999)
            if F is not None and mask is not None:
                overlap = int(mask.sum()) / min(len(pts_a), len(pts_b))
        store[key] = overlap
        return overlap

    return get


def pair_translation(pts_a, des_a, pts_b, des_b, bf) -> dict | None:
    """parallax_probe.pair_ratio, fed from cached features instead of two image files.

    Same test, same thresholds, same units: inliers to a homography over inliers to a fundamental
    matrix is ~1.0 for a camera that only turned, and falls as real parallax appears.
    """
    import cv2
    import numpy as np

    if des_a is None or des_b is None or len(pts_a) < 60 or len(pts_b) < 60:
        return None
    matches = bf.match(des_a, des_b)
    if len(matches) < 60:
        return None
    src = np.float32([pts_a[m.queryIdx] for m in matches])
    dst = np.float32([pts_b[m.trainIdx] for m in matches])
    H, hm = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=4000)
    F, fm = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.999)
    if H is None or F is None or hm is None or fm is None:
        return None
    hi, fi = int(hm.sum()), int(fm.sum())
    if fi < 40:
        return None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return dict(
        ratio=hi / fi,
        homographyResidualPx=float(np.median(np.linalg.norm(proj - dst, axis=1))),
    )


def orb_workbench(frames):
    """One ORB cache and one matcher, shared by the parallax pairs and the place overlap.

    Both measures want descriptors for the same sampled frames, and detection is the expensive
    half, so a frame is detected once however many pairs it appears in.
    """
    import cv2

    cv2.setRNGSeed(0)  # RANSAC draws; fixed so two runs of this script agree
    return orb_cache(frames), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def translation_pairs(
    frames,
    times: list[float],
    sample_fps: float,
    gaps,
    stride_seconds: float,
    feature=None,
    bf=None,
):
    """Every measured pair, as (gapSeconds, startTime, endTime, ratio, residualPx).

    Computed once over the whole sample grid; a window then takes the pairs that fall entirely
    inside it. This is what makes a per-window parallax probe affordable: parallax_probe on its own
    re-detects features for every directory it is given, and 30+ overlapping windows would redetect
    the same frames 30 times.
    """
    if feature is None or bf is None:
        feature, bf = orb_workbench(frames)
    step = max(1, round(stride_seconds * sample_fps))
    pairs = []
    for gap_seconds in gaps:
        gap = max(1, round(gap_seconds * sample_fps))
        if len(frames) - gap < CAMERA_MIN_PAIRS:
            continue
        for i in range(0, len(frames) - gap, step):
            pa, da = feature(i)
            pb, db = feature(i + gap)
            r = pair_translation(pa, da, pb, db, bf)
            if r is None:
                continue
            pairs.append(
                dict(
                    gapSeconds=round(gap / sample_fps, 3),
                    start=times[i],
                    end=times[i + gap],
                    ratio=r["ratio"],
                    homographyResidualPx=r["homographyResidualPx"],
                )
            )
    return pairs


def person_measurements(rgb_frames, times: list[float], min_height_frac: float) -> dict:
    """Per detector frame: reconstructable subjects, their heights, full-body, person pixels.

    Same model, same 0.7 confidence and same box-height rule as worker/wander_worker/masks.py, so a
    "subject" here is exactly an instance the cleaner would remove and the person stages would
    rebuild. Degrades loudly: with no torchvision the frames list is empty and `available` is False,
    and every score that depended on people says so.
    """
    import numpy as np

    try:
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
    except Exception as exc:  # the stage degrades; it does not pretend
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}", frames=[])
    try:
        net = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    except Exception as exc:
        return dict(available=False, reason=f"weights unavailable: {exc}", frames=[])
    rows = []
    with torch.no_grad():
        for i, im in enumerate(rgb_frames):
            h, w = im.shape[:2]
            x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255)
            pred = net([x])[0]
            keep = (pred["labels"] == 1) & (pred["scores"] > PERSON_SCORE)
            boxes = pred["boxes"][keep].cpu().numpy()
            masks = (pred["masks"][keep, 0] > 0.5).cpu().numpy()
            tall = [
                k for k in range(len(boxes)) if (boxes[k][3] - boxes[k][1]) >= min_height_frac * h
            ]
            heights = [float((boxes[k][3] - boxes[k][1]) / h) for k in tall]
            full = [bool(boxes[k][1] > 0.01 * h and boxes[k][3] < 0.99 * h) for k in tall]
            union = np.zeros((h, w), bool)
            for k in tall:
                union |= masks[k]
            rows.append(
                dict(
                    time=round(times[i], 4),
                    detections=int(len(boxes)),
                    subjects=len(tall),
                    tallestHeightFraction=round(max(heights), 4) if heights else 0.0,
                    fullBody=bool(full[int(np.argmax(heights))]) if heights else False,
                    personPixelFraction=round(float(union.mean()), 4),
                )
            )
    return dict(available=True, reason=None, frames=rows)


# ---------------------------------------------------------------- window features
def overlay_features(window_frames, letterbox_fraction: float = 0.0) -> dict:
    """Static overlay, over one window's samples of the ALREADY-CROPPED active picture.

    staticPixelFraction  active-area pixels whose peak-to-peak deviation across the window is at or
                         under STATIC_DEVIATION.
    letterboxFraction    the clip's black bars, measured once by letterbox_bars and passed in,
                         because bars belong to the file and not to a window.
    overlayFraction      bars plus the static pixels inside the picture, as a fraction of the WHOLE
                         frame -- but only while the picture is moving. A locked-off camera makes
                         the whole picture static and this measure cannot then separate a scoreboard
                         from a wall, so above STATIC_CAMERA_SUSPECT it falls back to the bars alone
                         and says it did.
    """
    if len(window_frames) < 2:
        return dict(
            available=False,
            staticPixelFraction=0.0,
            letterboxFraction=round(letterbox_fraction, 4),
            overlayFraction=round(letterbox_fraction, 4),
            staticCameraSuspected=False,
        )
    small = window_frames[:, ::4, ::4].astype("int16")
    static = (small.max(axis=0) - small.min(axis=0)) <= STATIC_DEVIATION
    static_fraction = float(static.mean())
    suspected = static_fraction >= STATIC_CAMERA_SUSPECT
    active = 1.0 - letterbox_fraction
    overlay = letterbox_fraction + (0.0 if suspected else static_fraction * active)
    return dict(
        available=True,
        staticPixelFraction=round(static_fraction, 4),
        letterboxFraction=round(letterbox_fraction, 4),
        overlayFraction=round(min(overlay, 1.0), 4),
        staticCameraSuspected=bool(suspected),
    )


def camera_features(pairs: list[dict], start: float, end: float) -> dict:
    """The widest gap with enough pairs entirely inside [start, end] decides, as parallax_probe."""
    by_gap: dict[float, list[dict]] = {}
    for p in pairs:
        if p["start"] >= start and p["end"] <= end:
            by_gap.setdefault(p["gapSeconds"], []).append(p)
    usable = {g: v for g, v in by_gap.items() if len(v) >= CAMERA_MIN_PAIRS}
    summary = {
        f"gap{g:g}s": dict(
            pairs=len(v),
            medianRatio=round(median([x["ratio"] for x in v]), 4),
            medianHomographyResidualPx=round(median([x["homographyResidualPx"] for x in v]), 3),
        )
        for g, v in sorted(by_gap.items())
    }
    if not usable:
        return dict(
            available=False,
            verdict="inconclusive",
            reason=f"no frame gap had {CAMERA_MIN_PAIRS} matchable pairs inside the window",
            byGap=summary,
        )
    widest = max(usable)
    rows = usable[widest]
    ratio = median([x["ratio"] for x in rows])
    return dict(
        available=True,
        verdict="rotation-only" if ratio >= ROTATION_ONLY_RATIO else "has-parallax",
        widestGapSeconds=widest,
        pairs=len(rows),
        widestGapRatio=round(ratio, 4),
        widestGapHomographyResidualPx=round(median([x["homographyResidualPx"] for x in rows]), 3),
        threshold=ROTATION_ONLY_RATIO,
        byGap=summary,
    )


def place_features(overlap, indices: list[int], times: list[float]) -> dict:
    """Does the window end where it began, or has it travelled somewhere else?

    `endpointOverlap` is the headline: of the keypoints the window opened on, the fraction still
    matched, and geometrically consistent, at its end. `halfOverlap` is the weaker of the two
    halves, which distinguishes a camera that leaves and comes back (high ends, low halves) from
    one that simply stayed put (both high) -- reported, not scored.
    """
    if overlap is None or len(indices) < PLACE_MIN_SAMPLES:
        return dict(available=False, reason=f"fewer than {PLACE_MIN_SAMPLES} samples in the window")
    first, last, mid = indices[0], indices[-1], indices[len(indices) // 2]
    near = PLACE_ENDPOINT_NEIGHBOURS
    ends = [
        overlap(a, b)
        for a in indices[: near + 1]
        for b in indices[-(near + 1) :]
        if b - a >= max(1, len(indices) // 2)
    ]
    ends = [e for e in ends if e is not None]
    halves = [h for h in [overlap(first, mid), overlap(mid, last)] if h is not None]
    if not ends:
        return dict(available=False, reason="too few ORB keypoints at the window's ends to compare")
    return dict(
        available=True,
        endpointOverlap=round(max(ends), 4),
        endpointPairs=len(ends),
        halfOverlap=round(min(halves), 4) if halves else None,
        spanSeconds=round(times[last] - times[first], 3),
    )


def people_features(rows: list[dict], available: bool, reason: str | None, cap: int) -> dict:
    """Aggregate the detector rows inside one window into the terms the score reads."""
    if not available:
        return dict(available=False, reason=reason, frames=0)
    if not rows:
        return dict(
            available=False,
            reason="no detector sample fell inside this window",
            frames=0,
        )
    counts = [r["subjects"] for r in rows]
    present = [r for r in rows if r["subjects"] > 0]
    count_median = int(median([float(c) for c in counts]))
    return dict(
        available=True,
        reason=None,
        frames=len(rows),
        presenceFraction=round(len(present) / len(rows), 4),
        subjectCountMedian=count_median,
        subjectCountMax=max(counts),
        # Fraction of samples whose subject count equals the median: 1.0 is a fixed cast, 0.3 is
        # people walking in and out of frame.
        countStability=round(sum(1 for c in counts if c == count_median) / len(counts), 4),
        subjectHeightFractionMedian=round(
            median([r["tallestHeightFraction"] for r in present]) if present else 0.0, 4
        ),
        fullBodyFraction=round(sum(1 for r in present if r["fullBody"]) / len(rows), 4)
        if present
        else 0.0,
        personPixelFractionMedian=round(median([r["personPixelFraction"] for r in rows]), 4),
        personPixelFractionMax=round(max(r["personPixelFraction"] for r in rows), 4),
        peopleCap=cap,
    )


def continuity_features(measurements: list[dict], start: float, end: float) -> dict:
    """Lowest thumbnail correlation between consecutive samples inside the window.

    A cut the detector missed shows up here as one sample pair that does not correlate at all. The
    measure is cheap because the correlations were computed once for the whole clip.
    """
    inside = [
        m
        for m in measurements
        if m["carryFromPrevious"] is not None and start <= m["time"] <= end and m["index"] > 0
    ]
    inside = [m for m in inside if measurements[m["index"] - 1]["time"] >= start]
    if not inside:
        return dict(available=False, samplePairs=0)
    worst = min(inside, key=lambda m: m["carryFromPrevious"])
    return dict(
        available=True,
        samplePairs=len(inside),
        minCarry=round(float(worst["carryFromPrevious"]), 4),
        medianCarry=round(median([float(m["carryFromPrevious"]) for m in inside]), 4),
        minCarryAtSeconds=worst["time"],
    )


def image_features(rows: list[dict]) -> dict:
    if not rows:
        return dict(available=False, samples=0)
    return dict(
        available=True,
        samples=len(rows),
        sharpness=round(median([r["sharpness"] for r in rows]), 2),
        darkFraction=round(median([r["darkFraction"] for r in rows]), 4),
        brightFraction=round(median([r["brightFraction"] for r in rows]), 4),
        meanLuma=round(median([r["meanLuma"] for r in rows]), 2),
    )


# ---------------------------------------------------------------- scoring (pure)
def translation_term(camera: dict) -> float:
    """1.0 at PARALLAX_RATIO_CLEAR or below, 0.0 at ROTATION_ONLY_RATIO or above, graded between."""
    if not camera.get("available"):
        return TRANSLATION_UNKNOWN
    return ramp(camera["widestGapRatio"], ROTATION_ONLY_RATIO, PARALLAX_RATIO_CLEAR)


def place_term(place: dict) -> float:
    """1.0 while the window's ends still see the same place, 0.0 once they share nothing."""
    if not place.get("available"):
        return PLACE_UNKNOWN
    return ramp(place["endpointOverlap"], PLACE_OVERLAP_FLOOR, PLACE_OVERLAP_CLEAR)


def count_term(count: int, cap: int) -> float:
    """0 with nobody, full from 1 subject up to the cap, decaying above it. Never a bonus for one."""
    if count <= 0:
        return 0.0
    if count <= cap:
        return 1.0
    over = (count - cap) / max(cap, 1)
    return max(PEOPLE_OVER_CAP_FLOOR, 1.0 - over * (1.0 - PEOPLE_OVER_CAP_FLOOR))


def crowd_term(person_pixel_fraction: float) -> float:
    """Full below PERSON_PIXELS_OK, falling to PERSON_PIXELS_BLOCKING_FACTOR when people fill the frame."""
    fall = ramp(person_pixel_fraction, PERSON_PIXELS_BLOCKING, PERSON_PIXELS_OK)
    return PERSON_PIXELS_BLOCKING_FACTOR + (1.0 - PERSON_PIXELS_BLOCKING_FACTOR) * fall


def people_term(people: dict) -> float:
    if not people.get("available"):
        return PEOPLE_UNKNOWN
    presence = ramp(people["presenceFraction"], 0.0, PRESENCE_TARGET)
    counts = count_term(people["subjectCountMedian"], people.get("peopleCap", PEOPLE_CAP))
    crowd = crowd_term(people["personPixelFractionMedian"])
    stability = STABILITY_FLOOR + (1.0 - STABILITY_FLOOR) * people["countStability"]
    return presence * counts * crowd * stability


def image_term(image: dict, overlay: dict) -> float:
    if not image.get("available"):
        return PEOPLE_UNKNOWN
    sharp = ramp(image["sharpness"], SHARPNESS_FLOOR, SHARPNESS_GOOD)
    dark = ramp(image["darkFraction"], DARK_BAD, DARK_OK)
    bright = ramp(image["brightFraction"], BRIGHT_BAD, BRIGHT_OK)
    penalty = OVERLAY_MAX_PENALTY * ramp(
        overlay.get("overlayFraction", 0.0), OVERLAY_OK, OVERLAY_BAD
    )
    return sharp * min(dark, bright) * (1.0 - penalty)


def continuity_term(continuity: dict) -> float:
    if not continuity.get("available"):
        return 0.5
    return ramp(continuity["minCarry"], INTERNAL_CUT_CARRY, CONTINUITY_CLEAR)


def duration_term(seconds: float) -> float:
    return ramp(seconds, DURATION_FLOOR_SECONDS, DURATION_FULL_SECONDS)


def vetoes_for(features: dict, params: dict) -> list[str]:
    """Hard refusals. A vetoed window scores 0 and is never chosen, whatever else it measures."""
    out = []
    seconds = features["seconds"]
    image = features["image"]
    if seconds < params["minWindowSeconds"] - 1e-6:
        out.append(f"too-short: {seconds:.2f} s under the {params['minWindowSeconds']:.2f} s floor")
    if not image.get("available") or image.get("samples", 0) < 2:
        out.append("no-samples: fewer than two frames decoded inside this window")
    elif image["darkFraction"] > DARK_VETO:
        out.append(
            f"too-dark: {image['darkFraction']:.2f} of the frame is near-black in the median "
            f"sample, over {DARK_VETO:.2f}; docs/known-limits.md says dark clips fail"
        )
    continuity = features["continuity"]
    if continuity.get("available") and continuity["minCarry"] <= INTERNAL_CUT_CARRY:
        out.append(
            f"internal-cut: only {continuity['minCarry']:.2f} of the frame correlates across "
            f"{continuity['minCarryAtSeconds']:.2f} s, at or under {INTERNAL_CUT_CARRY:.2f}; the "
            f"solve will not register across a break like that, whether or not it is an edit"
        )
    camera = features["camera"]
    if ROTATION_ONLY_VETO and camera.get("available") and camera["verdict"] == "rotation-only":
        out.append(
            f"rotation-only: every pair at the {camera['widestGapSeconds']:g} s baseline maps by "
            f"one homography (ratio {camera['widestGapRatio']:.2f}, at or over "
            f"{ROTATION_ONLY_RATIO:.2f}); the camera turned but did not travel, so there is no "
            f"second viewpoint and the solve has no depth to recover"
        )
    # A traverse is deliberately NOT vetoed here: see the place constants for the measurement that
    # refused it. MAX_WINDOW_SECONDS is what keeps a 22 s run out of one solve.
    return out


def score_window(features: dict, weights: dict, params: dict) -> dict:
    """features -> {score, terms, vetoes, why}. Pure: no video, no model, no filesystem."""
    terms = dict(
        translation=translation_term(features["camera"]),
        people=people_term(features["people"]),
        continuity=continuity_term(features["continuity"]),
        place=place_term(features["place"]),
        fullBody=features["people"].get("fullBodyFraction", PEOPLE_UNKNOWN)
        if features["people"].get("available")
        else PEOPLE_UNKNOWN,
        subjectHeight=ramp(
            features["people"].get("subjectHeightFractionMedian", 0.0),
            SUBJECT_HEIGHT_FLOOR,
            SUBJECT_HEIGHT_FULL,
        )
        if features["people"].get("available")
        else PEOPLE_UNKNOWN,
        image=image_term(features["image"], features["overlay"]),
        duration=duration_term(features["seconds"]),
    )
    vetoes = vetoes_for(features, params)
    total = sum(weights[k] * terms[k] for k in weights)
    score = 0.0 if vetoes else max(0.0, min(1.0, total))
    return dict(
        score=round(score, 4),
        scoreBeforeVetoes=round(max(0.0, min(1.0, total)), 4),
        terms={k: round(v, 4) for k, v in terms.items()},
        vetoes=vetoes,
        why=explain(features, terms, vetoes),
    )


def explain(features: dict, terms: dict, vetoes: list[str]) -> str:
    """One short machine-generated line: what this window is, in the units above."""
    cam, people = features["camera"], features["people"]
    bits = []
    if cam.get("available"):
        bits.append(
            f"camera {cam['verdict']} (ratio {cam['widestGapRatio']:.2f} at "
            f"{cam['widestGapSeconds']:g} s baseline)"
        )
    else:
        bits.append("camera translation not measurable here")
    if people.get("available"):
        bits.append(
            f"{people['subjectCountMedian']} subject(s) >= "
            f"{SUBJECT_MIN_HEIGHT_FRAC:g} frame height in {people['presenceFraction']:.0%} of "
            f"samples, {people['subjectHeightFractionMedian']:.2f} tall, "
            f"{people['fullBodyFraction']:.0%} full body, "
            f"{people['personPixelFractionMedian']:.0%} of pixels are people"
        )
    else:
        bits.append(f"no person detector ({people.get('reason')})")
    image, overlay = features["image"], features["overlay"]
    if image.get("available"):
        bits.append(
            f"sharpness {image['sharpness']:.0f}, {image['darkFraction']:.0%} near-black, "
            f"{overlay.get('overlayFraction', 0.0):.0%} static overlay"
        )
    cont = features["continuity"]
    if cont.get("available"):
        bits.append(f"worst sample-to-sample carry {cont['minCarry']:.2f}")
    place = features["place"]
    if place.get("available"):
        bits.append(f"ends share {place['endpointOverlap']:.1%} of their keypoints")
    bits.append(f"{features['seconds']:.1f} s")
    line = "; ".join(bits)
    return (
        ("VETOED (" + "; ".join(v.split(":")[0] for v in vetoes) + ") -- " + line)
        if vetoes
        else line
    )


# ---------------------------------------------------------------- windows (pure)
def shots_from_cut_times(cut_times: list[float], duration: float) -> list[dict]:
    bounds = [0.0] + sorted(float(t) for t in cut_times if 0.0 < float(t) < duration) + [duration]
    return [
        dict(
            index=i,
            start=round(bounds[i], 4),
            end=round(bounds[i + 1], 4),
            startSource="clip-start" if i == 0 else "detector",
            endSource="clip-end" if i == len(bounds) - 2 else "detector",
        )
        for i in range(len(bounds) - 1)
    ]


def suspected_breaks(measurements: list[dict], threshold: float = INTERNAL_CUT_CARRY) -> list[dict]:
    """Sample pairs that do not correlate at all: a break the cut detector did not report.

    The time is only known to one sample interval, because all this says is "the frame at t does
    not carry over from the frame at t - 1/SAMPLE_FPS". The whole interval between the two is
    therefore treated as unusable below, rather than one of them being called the cut.
    """
    return [
        dict(
            fromSeconds=round(measurements[m["index"] - 1]["time"], 4),
            toSeconds=round(m["time"], 4),
            carry=round(float(m["carryFromPrevious"]), 4),
        )
        for m in measurements
        if m["index"] > 0
        and m["carryFromPrevious"] is not None
        and m["carryFromPrevious"] <= threshold
    ]


def split_shots_at(shots: list[dict], breaks: list[dict]) -> list[dict]:
    """Cut each shot at every suspected break inside it, dropping the interval the break is in.

    Without this a clip whose every 12 s window straddles a break -- which is what a montage or a
    game capture is -- produces nothing but vetoes and no usable answer. With it, the stage returns
    the clean stretches BETWEEN the breaks, which is what the operator actually wanted.
    """
    out: list[dict] = []
    for shot in shots:
        start, start_source = shot["start"], shot["startSource"]
        inside = [b for b in breaks if shot["start"] < b["toSeconds"] <= shot["end"]]
        for b in sorted(inside, key=lambda b: b["toSeconds"]):
            if b["fromSeconds"] > start:
                out.append(
                    dict(
                        start=start,
                        end=b["fromSeconds"],
                        startSource=start_source,
                        endSource="suspected-break",
                    )
                )
            start, start_source = b["toSeconds"], "suspected-break"
        if shot["end"] > start:
            out.append(
                dict(
                    start=start,
                    end=shot["end"],
                    startSource=start_source,
                    endSource=shot["endSource"],
                )
            )
    for i, s in enumerate(out):
        s["index"] = i
        s["seconds"] = round(s["end"] - s["start"], 4)
    return out


def usable_span(shot: dict, fps: float) -> tuple[float, float]:
    """The shot minus shot_cuts.trim's boundary margin, so a chosen window is directly trimmable."""
    frame = 1.0 / max(fps, 1e-3)
    return shot["start"] + MARGIN_HEAD_FRAMES * frame, shot["end"] - MARGIN_TAIL_FRAMES * frame


def windows_for_shot(shot: dict, fps: float, params: dict) -> list[dict]:
    """Candidate windows inside one shot. They never cross a cut, because they never leave it."""
    lo, hi = usable_span(shot, fps)
    span = hi - lo
    if span < params["minUsableSeconds"] - 1e-9:
        return []
    if span <= params["maxWindowSeconds"] + 1e-9:
        return [dict(shotIndex=shot["index"], start=lo, end=hi, whole=True)]
    length = params["targetWindowSeconds"]
    stride = max(params["strideSeconds"], 1e-3)
    starts = []
    t = lo
    while t + length <= hi + 1e-9:
        starts.append(t)
        t += stride
    tail = hi - length
    if tail > lo + 1e-9 and (not starts or tail - starts[-1] > 1e-6):
        starts.append(tail)
    return [
        dict(shotIndex=shot["index"], start=s, end=s + length, whole=False) for s in sorted(starts)
    ]


def candidate_windows(shots: list[dict], fps: float, params: dict) -> list[dict]:
    out = []
    for shot in shots:
        out.extend(windows_for_shot(shot, fps, params))
    out.sort(key=lambda w: (w["start"], w["end"]))
    for i, w in enumerate(out):
        w["id"] = f"w{i:03d}"
    return out


def overlap_seconds(a: dict, b: dict) -> float:
    return max(
        0.0, min(a["endSeconds"], b["endSeconds"]) - max(a["startSeconds"], b["startSeconds"])
    )


def choose_top(candidates: list[dict], k: int, max_overlap: float) -> list[dict]:
    """Greedy by score, ties to the earlier start, skipping anything that overlaps a pick too much.

    Overlap is measured against the SHORTER of the two windows, so a 6 s window inside a 20 s one
    counts as fully overlapping rather than as 30%.
    """
    ranked = sorted(
        [c for c in candidates if not c["vetoes"]],
        key=lambda c: (-c["score"], c["startSeconds"], c["endSeconds"], c["id"]),
    )
    chosen: list[dict] = []
    for c in ranked:
        if len(chosen) >= k:
            break
        clash = False
        for picked in chosen:
            shorter = min(c["seconds"], picked["seconds"])
            if shorter > 0 and overlap_seconds(c, picked) / shorter > max_overlap + 1e-9:
                clash = True
                break
        if not clash:
            chosen.append(c)
    return [dict(c, rank=i + 1) for i, c in enumerate(chosen)]


def coverage_not_selected(chosen: list[dict], duration: float) -> dict:
    """Source time no chosen window covers -- what this stage is throwing away, stated plainly."""
    spans = sorted((c["startSeconds"], c["endSeconds"]) for c in chosen)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    gaps, cursor = [], 0.0
    for s, e in merged:
        if s - cursor > 1e-6:
            gaps.append([round(cursor, 3), round(s, 3)])
        cursor = max(cursor, e)
    if duration - cursor > 1e-6:
        gaps.append([round(cursor, 3), round(duration, 3)])
    left = sum(b - a for a, b in gaps)
    return dict(
        seconds=round(left, 3),
        fraction=round(left / duration, 4) if duration > 0 else 0.0,
        gaps=gaps,
    )


# ---------------------------------------------------------------- weights
def resolve_weights(path: Path | None) -> tuple[dict, dict]:
    """Defaults, or a JSON override, normalised so the score stays in [0, 1]. Both are recorded."""
    if path is None:
        return dict(DEFAULT_WEIGHTS), dict(source="default")
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise BadRequest(f"{path}: expected an object of term -> weight")
    unknown = sorted(set(raw) - set(DEFAULT_WEIGHTS))
    if unknown:
        raise BadRequest(f"{path}: unknown term(s) {unknown}; known: {sorted(DEFAULT_WEIGHTS)}")
    merged = dict(DEFAULT_WEIGHTS)
    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise BadRequest(f"{path}: weight {key} must be a number >= 0")
        merged[key] = float(value)
    total = sum(merged.values())
    if total <= 0:
        raise BadRequest(f"{path}: the weights sum to {total}")
    return (
        {k: v / total for k, v in merged.items()},
        dict(source=str(path), requested=raw, sumBeforeNormalisation=round(total, 6)),
    )


# ---------------------------------------------------------------- contact sheets and judge
def contact_sheet(video: Path, info: dict, window: dict, out: Path, tiles: int = 6) -> Path:
    """A small JPG grid of `tiles` frames across one window, for a human (or a VLM) to look at."""
    out.parent.mkdir(parents=True, exist_ok=True)
    seconds = max(window["seconds"], 1e-3)
    rate = tiles / seconds
    columns = 3
    rows = math.ceil(tiles / columns)
    chain = (
        f"fps={rate:.6f},"
        + tonemap_chain(info)
        + f"scale=320:-2,tile={columns}x{rows}:margin=4:padding=4:color=black"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{window['startSeconds']:.4f}",
            "-t",
            f"{seconds:.4f}",
            "-i",
            str(video),
            "-vf",
            chain,
            "-frames:v",
            "1",
            "-q:v",
            "4",
            str(out),
        ],
        check=True,
    )
    return out


JUDGE_QUESTION = (
    "Each image is a contact sheet of one candidate window from the same source video, six frames "
    "in time order, left to right and top to bottom. Every window below has already been measured "
    "as technically reconstructable -- the camera travels, the people are big enough, nothing is "
    "cut across -- so do not judge image quality or camera motion. Judge only this: which window "
    "holds the main action, the moment a viewer would want to stand inside and watch from any "
    "angle? Answer with one of the window ids listed and one sentence saying what happens in it."
)


def judge_request(document: dict, sheets: dict[str, Path], out: Path) -> Path:
    """Describe the top-K windows for an OPTIONAL receipted VLM tie-break. Nothing here calls it.

    scripts/vlm_once.py is what would spend the money, and it is deliberately not imported: this
    stage must produce the same answer offline. The orchestrator decides whether an opinion is
    worth a paid request, and its answer comes back through --judge-opinion as an opinion.
    """
    windows = []
    for c in document["chosen"]:
        sheet = sheets.get(c["id"])
        windows.append(
            dict(
                id=c["id"],
                rank=c["rank"],
                startSeconds=c["startSeconds"],
                endSeconds=c["endSeconds"],
                score=c["score"],
                why=c["why"],
                contactSheet=str(sheet) if sheet else None,
                contactSheetSha256=sha256_file(sheet) if sheet else None,
            )
        )
    payload = dict(
        schema=JUDGE_SCHEMA,
        video=document["video"]["path"],
        videoSha256=document["video"]["sha256"],
        question=JUDGE_QUESTION,
        windows=windows,
        responseSchema=dict(
            type="object",
            required=["windowId", "reason"],
            additionalProperties=False,
            properties=dict(
                windowId=dict(type="string", enum=[w["id"] for w in windows]),
                reason=dict(type="string"),
            ),
        ),
        note=(
            "Advisory only. This selector ranks windows from measurements and never calls a model; "
            "an answer returned via --judge-opinion is recorded as an opinion and does not change "
            "the ranking. See docs/segment-selection.md."
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    return out


def load_judge_opinion(path: Path, candidate_ids: set[str]) -> dict:
    raw = json.loads(Path(path).read_text())
    window_id = raw.get("windowId") or raw.get("preferred")
    return dict(
        source=str(path),
        windowId=window_id,
        reason=raw.get("reason"),
        model=raw.get("model"),
        receipt=raw.get("receipt"),
        known=bool(window_id in candidate_ids),
        applied=False,
        note="recorded as an opinion; the ranking above is measured and was not changed by it",
    )


# ---------------------------------------------------------------- the stage
def _params(
    min_window: float,
    target_window: float,
    max_window: float,
    min_usable: float,
    stride: float,
    people_cap: int,
    sample_fps: float,
    people_max_frames: int,
    top_k: int,
    max_overlap: float,
) -> dict:
    if not (min_usable <= min_window <= target_window <= max_window):
        raise BadRequest(
            "window bounds must satisfy min-usable <= min-window <= target-window <= max-window "
            f"(got {min_usable}, {min_window}, {target_window}, {max_window})"
        )
    return dict(
        minWindowSeconds=float(min_window),
        targetWindowSeconds=float(target_window),
        maxWindowSeconds=float(max_window),
        minUsableSeconds=float(min_usable),
        strideSeconds=float(stride),
        peopleCap=int(people_cap),
        sampleFps=float(sample_fps),
        analysisWidth=ANALYSIS_WIDTH,
        peopleWidth=PEOPLE_WIDTH,
        peopleMaxFrames=int(people_max_frames),
        topK=int(top_k),
        maxOverlapFraction=float(max_overlap),
        cameraGapSeconds=list(CAMERA_GAP_SECONDS),
        pairStrideSeconds=PAIR_STRIDE_SECONDS,
        subjectMinHeightFraction=SUBJECT_MIN_HEIGHT_FRAC,
        personScore=PERSON_SCORE,
    )


def cut_source(
    video: Path, cuts: Path | None, truth: Path | None, duration: float
) -> tuple[list[float], dict]:
    """Cut times and where they came from. `--truth` beats a report; a report beats detecting."""
    if truth is not None:
        raw = json.loads(Path(truth).read_text())
        entries = raw.get("cuts", raw) if isinstance(raw, dict) else raw
        times = sorted(float(e["time"] if isinstance(e, dict) else e) for e in entries)
        return times, dict(
            source="truth",
            path=str(truth),
            sha256=sha256_file(truth),
            note="human-verified boundaries; the detector's cuts were not used",
        )
    if cuts is not None:
        doc = json.loads(Path(cuts).read_text())
        times = sorted(float(c["time"]) for c in doc.get("cuts", []))
        return times, dict(
            source="cut-report",
            path=str(cuts),
            # The report's own hash, so a selection can be tied back to the exact cut list it was
            # made from -- a report regenerated at another threshold is a different input.
            sha256=sha256_file(cuts),
            reportVideo=doc.get("video"),
            threshold=doc.get("threshold"),
        )
    if shot_cuts is None:
        raise BadRequest(
            f"scripts/shot_cuts.py could not be imported ({SHOT_CUTS_ERROR}), so this stage cannot "
            "detect cuts itself; pass --cuts <report.json> or --truth <boundaries.json>"
        )
    doc = shot_cuts.cut_report(video, score=False)
    times = sorted(float(c["time"]) for c in doc.get("cuts", []))
    return times, dict(
        source="shot_cuts.cut_report",
        detectorSha256=sha256_file(ROOT / "scripts" / "shot_cuts.py"),
        threshold=doc.get("threshold"),
        clearedCandidates=len(doc.get("cleared", [])),
    )


def select(
    video,
    *,
    cuts=None,
    truth=None,
    min_window: float = MIN_WINDOW_SECONDS,
    target_window: float = TARGET_WINDOW_SECONDS,
    max_window: float = MAX_WINDOW_SECONDS,
    min_usable: float = MIN_USABLE_SECONDS,
    stride: float = WINDOW_STRIDE_SECONDS,
    people_cap: int = PEOPLE_CAP,
    top_k: int = DEFAULT_TOP_K,
    max_overlap: float = MAX_OVERLAP_FRACTION,
    sample_fps: float = SAMPLE_FPS,
    people_max_frames: int = PEOPLE_MAX_FRAMES,
    people: bool = True,
    split_suspected: bool = True,
    weights: Path | None = None,
    contact_sheet_dir=None,
    force_window: tuple[float, float] | None = None,
    force_reason: str | None = None,
    judge_opinion: Path | None = None,
    log=lambda _m: None,
) -> dict:
    """Measure every candidate window in `video` and choose the best K. Returns the report dict.

    This is the entry point scripts/run_clip.py should call: it needs no network, no GPU and no
    paid stage, and the dict it returns names the exact source seconds to trim.
    """
    started = time.time()
    video = Path(video).resolve()
    if force_window is not None and not (force_reason or "").strip():
        raise BadRequest(
            "--force-window requires --reason: an override without one is not a record"
        )
    params = _params(
        min_window,
        target_window,
        max_window,
        min_usable,
        stride,
        people_cap,
        sample_fps,
        people_max_frames,
        top_k,
        max_overlap,
    )
    effective_weights, weight_source = resolve_weights(weights)
    info = probe_video(video)
    duration = info["durationSeconds"]
    log(f"source: {duration:.2f} s, {info['width']}x{info['height']}, {info['fps']:.3f} fps")
    if tonemap_chain(info):
        log(f"   {info['colorTransfer']} transfer: tone-mapping to BT.709 before measuring")

    cut_times, cut_meta = cut_source(video, cuts, truth, duration)
    detected_shots = shots_from_cut_times(cut_times, duration)
    log(f"cuts: {len(cut_times)} from {cut_meta['source']} -> {len(detected_shots)} shot(s)")

    times, decoded = decode_grey(video, info, params["sampleFps"], params["analysisWidth"])
    log(
        f"decoded {len(decoded)} grey samples at {params['sampleFps']:g}/s, "
        f"{decoded.shape[2] if len(decoded) else 0}px wide"
    )
    bars = letterbox_bars(decoded)
    crop = suggested_crop(bars, info)
    frames = crop_bars(decoded, bars)
    if bars["fractionOfFrame"] > 0:
        log(
            f"letterbox: {bars['fractionOfFrame']:.0%} of the frame is black bar in "
            f"{bars['barFrameFraction']:.0%} of samples; cropped off before measuring, "
            f"suggested source crop {crop['ffmpeg']}"
        )
    if shot_cuts is None:
        log(f"shot_cuts.py unavailable ({SHOT_CUTS_ERROR}); using the local carry fallback")
    measurements = frame_measurements(frames, times)
    breaks = suspected_breaks(measurements) if split_suspected else []
    shots = split_shots_at(detected_shots, breaks)
    if breaks:
        log(
            f"continuity: {len(breaks)} suspected break(s) the cut detector did not report, at "
            f"{[b['toSeconds'] for b in breaks]} s -> {len(shots)} segment(s)"
        )
    feature, bf = orb_workbench(frames)
    pairs = translation_pairs(
        frames, times, params["sampleFps"], CAMERA_GAP_SECONDS, PAIR_STRIDE_SECONDS, feature, bf
    )
    overlap = overlap_lookup(feature, bf)
    log(f"camera: {len(pairs)} matched frame pairs over {len(CAMERA_GAP_SECONDS)} baselines")

    windows = candidate_windows(shots, info["fps"], params)
    forced = None
    if force_window is not None:
        forced = dict(
            shotIndex=None,
            start=float(force_window[0]),
            end=float(force_window[1]),
            whole=False,
            id="forced",
            forced=True,
        )
        windows = windows + [forced]
    if not windows:
        log("no window survives the shot bounds; nothing to measure")

    person_times: list[float] = []
    person = dict(available=False, reason="--no-people", frames=[])
    if people and duration > 0:
        stride_s = max(PEOPLE_MIN_STRIDE_SECONDS, duration / max(params["peopleMaxFrames"], 1))
        n = max(1, min(params["peopleMaxFrames"], int(duration / stride_s)))
        person_times = [round((i + 0.5) * duration / n, 4) for i in range(n)]
        rgb = crop_bars(
            decode_people_frames(video, info, person_times, params["peopleWidth"]), bars
        )
        person_times = person_times[: len(rgb)]
        log(f"people: {len(rgb)} frames through Mask R-CNN (this is the slow part)")
        person = person_measurements(rgb, person_times, SUBJECT_MIN_HEIGHT_FRAC)
        if not person["available"]:
            log(f"   person detector unavailable: {person['reason']} -- people features marked so")

    candidates = []
    for w in windows:
        start, end = w["start"], w["end"]
        rows = [m for m in measurements if start <= m["time"] <= end]
        window_frames = frames[[m["index"] for m in rows]] if rows else frames[:0]
        person_rows = [r for r in person.get("frames", []) if start <= r["time"] <= end]
        features = dict(
            seconds=round(end - start, 4),
            samples=len(rows),
            camera=camera_features(pairs, start, end),
            people=people_features(
                person_rows,
                person.get("available", False),
                person.get("reason"),
                params["peopleCap"],
            ),
            image=image_features(rows),
            overlay=overlay_features(window_frames, bars["fractionOfFrame"]),
            continuity=continuity_features(measurements, start, end),
            place=place_features(overlap, [m["index"] for m in rows], times),
        )
        scored = score_window(features, effective_weights, params)
        candidates.append(
            dict(
                id=w["id"],
                shotIndex=w["shotIndex"],
                wholeShot=bool(w.get("whole")),
                forced=bool(w.get("forced")),
                startSeconds=round(start, 4),
                endSeconds=round(end, 4),
                seconds=features["seconds"],
                startFrame=int(round(start * info["fps"])),
                endFrame=int(round(end * info["fps"])),
                features=features,
                **scored,
            )
        )

    selectable = [c for c in candidates if not c["forced"]]
    chosen = choose_top(selectable, params["topK"], params["maxOverlapFraction"])
    override = None
    if forced is not None:
        fc = next(c for c in candidates if c["forced"])
        override = dict(
            startSeconds=fc["startSeconds"],
            endSeconds=fc["endSeconds"],
            reason=force_reason.strip(),
            score=fc["score"],
            scoreBeforeVetoes=fc["scoreBeforeVetoes"],
            vetoes=fc["vetoes"],
            why=fc["why"],
            wouldHaveChosen=[
                dict(id=c["id"], startSeconds=c["startSeconds"], score=c["score"]) for c in chosen
            ],
        )
        chosen = [dict(fc, rank=1)]
    # The crop belongs to the file, not to a window, but it travels on every chosen window so the
    # runner can trim and crop from one object instead of joining two.
    chosen = [dict(c, suggestedCrop=crop) for c in chosen]

    document = dict(
        schema=SCHEMA,
        video=dict(
            path=str(video),
            sha256=sha256_file(video),
            **info,
            tonemapped=bool(tonemap_chain(info)),
            letterbox=bars,
            suggestedCrop=crop,
        ),
        cuts=dict(
            **cut_meta,
            times=[round(t, 3) for t in cut_times],
            suspectedBreaks=breaks,
            splitAtSuspectedBreaks=bool(split_suspected),
            detectedShots=detected_shots,
            shots=shots,
        ),
        parameters=params,
        weights={k: round(v, 6) for k, v in effective_weights.items()},
        weightsSource=weight_source,
        featuresAvailable=dict(
            people=bool(person.get("available")),
            peopleReason=person.get("reason"),
            camera=bool(pairs),
            cameraPairs=len(pairs),
            shotCuts=shot_cuts is not None,
            shotCutsReason=SHOT_CUTS_ERROR,
        ),
        candidateCount=len(selectable),
        candidates=candidates,
        chosen=chosen,
        # Nothing chosen is an answer, not a crash, but an operator needs somewhere to look: the
        # window that would have won if its veto were lifted, and why it was not.
        blockedBest=None
        if chosen or not selectable
        else max(selectable, key=lambda c: (c["scoreBeforeVetoes"], -c["startSeconds"])),
        coverageNotSelected=coverage_not_selected(chosen, duration),
        override=override,
        judge=None,
        contactSheets={},
    )

    if contact_sheet_dir is not None and chosen:
        sheets = {}
        directory = Path(contact_sheet_dir)
        for c in chosen:
            path = directory / f"{c['id']}_{c['startSeconds']:.1f}-{c['endSeconds']:.1f}s.jpg"
            try:
                sheets[c["id"]] = contact_sheet(video, info, c, path)
            except subprocess.CalledProcessError as exc:
                log(f"   contact sheet for {c['id']} failed: {exc}")
        document["contactSheets"] = {k: str(v) for k, v in sheets.items()}
        if sheets:
            document["judgeRequest"] = str(
                judge_request(document, sheets, directory / "judge_request.json")
            )
    if judge_opinion is not None:
        document["judge"] = load_judge_opinion(judge_opinion, {c["id"] for c in candidates})

    document["runtime"] = dict(
        seconds=round(time.time() - started, 2),
        secondsPerSourceMinute=round((time.time() - started) / max(duration / 60.0, 1e-6), 2),
    )
    return document


# ---------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--cuts", type=Path, help="an existing scripts/shot_cuts.py --json report")
    ap.add_argument(
        "--truth",
        type=Path,
        help='human-verified boundaries, [{"time": 12.3}, ...]; overrides the detector entirely',
    )
    ap.add_argument("--min-window", type=float, default=MIN_WINDOW_SECONDS)
    ap.add_argument("--target-window", type=float, default=TARGET_WINDOW_SECONDS)
    ap.add_argument("--max-window", type=float, default=MAX_WINDOW_SECONDS)
    ap.add_argument("--min-usable", type=float, default=MIN_USABLE_SECONDS)
    ap.add_argument("--stride", type=float, default=WINDOW_STRIDE_SECONDS)
    ap.add_argument("--people-cap", type=int, default=PEOPLE_CAP)
    ap.add_argument("--top", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--max-overlap", type=float, default=MAX_OVERLAP_FRACTION)
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--people-max-frames", type=int, default=PEOPLE_MAX_FRAMES)
    ap.add_argument("--no-people", action="store_true", help="skip the Mask R-CNN pass")
    ap.add_argument(
        "--no-split-suspected",
        action="store_true",
        help="do not split a shot at a suspected break; windows containing one are vetoed instead",
    )
    ap.add_argument("--weights", type=Path, help="JSON term -> weight override, recorded in --out")
    ap.add_argument("--out", type=Path, help="write selection.json here")
    ap.add_argument(
        "--contact-sheet", type=Path, metavar="DIR", help="one JPG sheet per chosen window"
    )
    ap.add_argument(
        "--force-window", type=float, nargs=2, metavar=("START", "END"), help="operator override"
    )
    ap.add_argument("--reason", help="why the override; required with --force-window")
    ap.add_argument("--judge-opinion", type=Path, help="a VLM answer to record as an opinion")
    a = ap.parse_args()

    def say(message: str) -> None:
        print(message, flush=True)

    try:
        document = select(
            a.video,
            cuts=a.cuts,
            truth=a.truth,
            min_window=a.min_window,
            target_window=a.target_window,
            max_window=a.max_window,
            min_usable=a.min_usable,
            stride=a.stride,
            people_cap=a.people_cap,
            top_k=a.top,
            max_overlap=a.max_overlap,
            sample_fps=a.sample_fps,
            people_max_frames=a.people_max_frames,
            people=not a.no_people,
            split_suspected=not a.no_split_suspected,
            weights=a.weights,
            contact_sheet_dir=a.contact_sheet,
            force_window=tuple(a.force_window) if a.force_window else None,
            force_reason=a.reason,
            judge_opinion=a.judge_opinion,
            log=say,
        )
    except BadRequest as exc:
        sys.exit(str(exc))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(document, indent=1) + "\n")
        say(f"wrote {a.out}")
    say(f"{document['candidateCount']} candidate window(s):")
    for c in sorted(document["candidates"], key=lambda c: c["startSeconds"]):
        tag = "FORCED" if c["forced"] else f"score {c['score']:.3f}"
        say(f"  {c['id']} {c['startSeconds']:8.2f}-{c['endSeconds']:8.2f} s  {tag}  {c['why']}")
    if document["override"]:
        say(f"OVERRIDE: {document['override']['reason']}")
    if not document["chosen"]:
        say("nothing selectable: every candidate was vetoed or no window fitted inside a shot")
        blocked = document.get("blockedBest")
        if blocked:
            say(
                f"   closest was {blocked['id']} {blocked['startSeconds']:.2f}-"
                f"{blocked['endSeconds']:.2f} s, which would have scored "
                f"{blocked['scoreBeforeVetoes']:.3f}: {'; '.join(blocked['vetoes'])}"
            )
        say("   a shorter --target-window, or --force-window with a reason, is how to proceed")
        return 1
    for c in document["chosen"]:
        say(
            f"chosen #{c['rank']}: {c['startSeconds']:.3f}-{c['endSeconds']:.3f} s "
            f"(frames {c['startFrame']}-{c['endFrame']}, score {c['score']:.3f}) -- {c['why']}"
        )
    left = document["coverageNotSelected"]
    say(f"not selected: {left['seconds']:.2f} s ({left['fraction']:.0%}) of the source")
    say(f"runtime {document['runtime']['seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
