#!/usr/bin/env python3
"""At least 12 independently reconstructed person frames/second, shared static anchors.

This increases temporal density, not hidden-side reconstruction. The single-view
surface limitation remains explicit. All chunks share a measured similarity frame.
"""

from __future__ import annotations
import argparse, gc, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fewer masked points than this is speckle -- a bag, a reflection, a sliver of a limb
# at the frame edge -- not a body surface worth exporting. It is a property of the mask
# and of Pi3X's point density, not of any one clip.
MIN_PERSON_POINTS = 100


def person_present(point_count, minimum=MIN_PERSON_POINTS):
    """Does this frame's masked cloud hold a usable person surface?

    Person absence is not a solve failure: the camera for the frame is still valid, the
    people have simply walked out of the shot. Callers record a gap instead of aborting.
    """
    return int(point_count) >= int(minimum)


def person_gap(sample, source_index, seconds, point_count):
    """One entry of the solve metadata's `personAbsentFrames` list."""
    return dict(
        sample=int(sample),
        sourceIndex=int(source_index),
        time=float(seconds),
        points=int(point_count),
    )


def person_gap_summary(gaps, count, minimum=MIN_PERSON_POINTS):
    """Disclosure block naming every frame whose person cloud was too thin to export.

    `personFrames` lists only the per-frame plys that actually exist; `frames` keeps its
    original one-name-per-sample shape, so readers that need real files use this instead.
    """
    count = int(count)
    ordered = sorted((dict(g) for g in gaps), key=lambda g: int(g["sample"]))
    absent = {int(g["sample"]) for g in ordered}
    return dict(
        personAbsentFrames=ordered,
        personAbsentCount=len(absent),
        personPresentCount=count - len(absent),
        personFrames=[f"frame_{i:03d}.ply" for i in range(count) if i not in absent],
        minPersonPoints=int(minimum),
        cameraOnly=count > 0 and len(absent) == count,
        personGapNote=(
            f"Frames with fewer than {int(minimum)} masked person points export no "
            "frame_NNN.ply; their camera entry in cameras.json is still a measured pose. "
            "Absence means nobody was observed in that frame, not a failed solve."
        ),
    )


# A similarity needs three non-collinear points; below this the fit is describing sampling
# noise rather than two views of the same scene, and its rotation is unconstrained about the
# axis the points happen to lie on. It is a property of the estimator, not of any clip.
MIN_SIMILARITY_POINTS = 12
# Smallest principal spread of the source points, relative to their largest. A ratio under
# this means the correspondences lie on a line or a plane-edge, where the SVD below either
# fails to converge or returns a rotation the data never determined.
MIN_SIMILARITY_SPREAD = 1e-6


def similarity(src, dst, minimum=MIN_SIMILARITY_POINTS, spread=MIN_SIMILARITY_SPREAD):
    """Trimmed similarity src -> dst, refusing inputs an SVD would crash or lie about.

    The trimmed fit is unchanged: four rounds, each keeping the closest 75%. What is new is
    that empty, non-finite and degenerate inputs raise a RuntimeError naming the counts
    instead of reaching `np.linalg.svd`, which answers "SVD did not converge" to a matrix of
    NaN and says nothing about the anchor set that was actually empty.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    dst = np.asarray(dst, dtype=float).reshape(-1, 3)
    if len(src) != len(dst):
        raise RuntimeError(f"Similarity needs paired points, got {len(src)} and {len(dst)}")
    finite = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[finite], dst[finite]
    if len(src) < minimum:
        raise RuntimeError(
            f"Only {len(src)} finite paired points for the anchor alignment "
            f"({int(finite.size - finite.sum())} dropped as NaN or infinite); "
            f"a similarity fit needs at least {minimum}"
        )

    def fit(a, b):
        if len(a) < minimum:
            raise RuntimeError(f"Similarity fit left with {len(a)} points, needs {minimum}")
        ma, mb = a.mean(0), b.mean(0)
        aa, bb = a - ma, b - mb
        moments = np.linalg.eigvalsh(aa.T @ aa / len(a))
        if not np.isfinite(moments).all() or moments[-1] <= 0:
            raise RuntimeError("Anchor correspondences have no finite spread to align")
        if moments[0] <= spread * moments[-1]:
            raise RuntimeError(
                f"Anchor correspondences are degenerate: principal spreads "
                f"{moments[0]:.3e}/{moments[-1]:.3e} put them on a line or a point, "
                "which does not determine a rotation"
            )
        try:
            U, S, V = np.linalg.svd(bb.T @ aa / len(a))
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                f"Anchor alignment SVD failed on {len(a)} points: {error}"
            ) from error
        D = np.eye(3)
        D[2, 2] = np.linalg.det(U @ V)
        R = U @ D @ V
        s = np.sum(S * np.diag(D)) / np.mean(np.sum(aa * aa, 1))
        return s, R, mb - s * R @ ma

    keep = np.ones(len(src), bool)
    for _ in range(4):
        s, R, t = fit(src[keep], dst[keep])
        error = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
        trimmed = error <= np.percentile(error, 75)
        # Trimming past the estimator's own minimum would hand `fit` a set too small to
        # constrain it; stop trimming instead of failing on the clip's last few matches.
        if trimmed.sum() < minimum:
            break
        keep = trimmed
    return s, R, t, float(np.sqrt(np.mean(error[keep] ** 2)))


# --- confidence, people and anchor selection ------------------------------------------------
# Pi3X's per-pixel confidence is the sigmoid of a head that is not calibrated across content.
# Measured on this stage's own saved anchor predictions (8 views each, .context/run/*/pi3x):
#   solved cleanly: img5594 max 0.91 / 91% of pixels over 0.3, img5593 0.91, hp-fly-s63 0.63,
#                   movie-s26 0.57
#   failed:         game-s1 max 0.48, creed-v1 0.44, movie-s17 0.40, soccer-s1 0.32
# soccer-s1 never crosses 0.3 in seven of its eight anchors, so `conf > 0.3` produced an empty
# `valid`, and no clip in the failing set crosses the 0.5 the anchor matches asked for, so every
# anchor's match set was empty and the alignment SVD was handed a NaN matrix. The cut therefore
# has to be a RANK inside each view, with the old absolute cut kept wherever the view is
# confident enough to satisfy it, so nothing changes for the clips that already solve.
CONF_ABSOLUTE = 0.3
# Share of a view that must clear CONF_ABSOLUTE for the absolute cut to describe that view.
# Every anchor of the four solved clips that carried the fit is far above this; the starved
# views are at zero.
CONF_MIN_FRACTION = 0.2
# Fallback cut: keep the more confident 40% of the view. Enough surface for a person cloud and
# an intrinsic fit, while still dropping the half of the map the model is least sure of.
CONF_QUANTILE = 0.6
# Under this, the model is not reporting geometry at all, only the floor of its own sigmoid.
CONF_FLOOR = 0.02

CONF_MODE_ABSOLUTE = "absolute"
CONF_MODE_RANK = "per-view-rank"


def confidence_threshold(
    conf,
    absolute=CONF_ABSOLUTE,
    min_fraction=CONF_MIN_FRACTION,
    quantile=CONF_QUANTILE,
    floor=CONF_FLOOR,
):
    """(threshold, mode) for one view's confidence map.

    The absolute cut when this view actually has that much confident surface, otherwise the
    view's own quantile, never below `floor`. Returning the mode keeps the run's metadata
    honest about which of the two a frame was measured under.
    """
    conf = np.asarray(conf, dtype=float)
    finite = conf[np.isfinite(conf)]
    if finite.size == 0:
        return float(floor), CONF_MODE_RANK
    if float((finite > absolute).mean()) >= min_fraction:
        return float(absolute), CONF_MODE_ABSOLUTE
    return max(float(floor), float(np.quantile(finite, quantile))), CONF_MODE_RANK


# Semantic person share of the anchor views, averaged, above which "person" is describing a
# crowd rather than the subjects. Measured on the same saved anchors: soccer-s1 0.355 (a
# stadium: the terraces are class 12 wall to wall), movie-s17 0.408 and creed-v1 0.361 (ringside
# crowd), against game-s1 0.061, img5594 0.104, movie-s26 0.125 and hp-fly-s63 0.149. Excluding
# a crowd from the static scene erases the stadium -- on soccer-s1 anchor 3 the semantic mask
# takes 51.7% of the view and leaves 122707 static pixels where the foreground-instance mask
# leaves 176547 -- and exporting it as the person ply makes the terraces "the person".
CROWD_PERSON_FRACTION = 0.25

MASK_SEMANTIC = "semantic-person"
MASK_FOREGROUND = "foreground-instances"


def mask_mode(person_fractions, threshold=CROWD_PERSON_FRACTION):
    """(mode, mean fraction) -- which people mask this clip's solve should use throughout.

    One decision per clip rather than per view, so the pixels excluded from the static scene
    and the pixels exported as the person come from the same question all the way through the
    run; the mode and the measured fraction are recorded in the solve metadata.
    """
    fractions = np.asarray(list(person_fractions), dtype=float)
    mean = float(fractions.mean()) if fractions.size else 0.0
    return (MASK_FOREGROUND if mean > threshold else MASK_SEMANTIC), mean


# Points sampled from each anchor for the similarity fit (unchanged).
ANCHOR_MATCH_TARGET = 1500
# Fewer usable static pixels than this in a view and its share of the fit is noise, so the
# anchor is dropped and recorded rather than contributing a handful of points.
ANCHOR_MATCH_MIN = 200
# The target is drawn at random from this many times the target's most confident static pixels,
# not from the top of the ranking directly: the most confident pixels of a view cluster on one
# well-textured surface, and a similarity fitted to one surface is barely constrained.
ANCHOR_MATCH_POOL = 4
# When most of the anchor views clear the absolute cut on their own, the fit is made from
# those alone, so a clip that already solves keeps the anchor set it always had and the rank
# fallback only ever rescues a clip that had none. Below this share the absolutely-confident
# views are too few to align on -- creed-v1 has two of eight, and they are adjacent -- so the
# clip is aligned on everything usable instead.
ABSOLUTE_ANCHOR_MAJORITY = 0.5
# An anchor whose selected pixels are this much less confident than the best anchor's is a view
# the model could not relate to the others -- on game-s1 anchor 7 sits at 0.157 against 0.406,
# on soccer-s1 anchor 7 at 0.049 against 0.242 -- and aligning to it drags the whole batch.
ANCHOR_RELATIVE_CONF = 0.4
# What the fit needs left over once starved anchors are dropped.
MIN_USABLE_ANCHORS = 2
MIN_TOTAL_MATCHES = 600
# Alignment residual a batch may leave against the anchor prediction, as a fraction of the
# clip's median anchor depth. A batch that registers this badly is not the same reconstruction
# of the same moment, whatever its scale factor says.
MAX_ALIGNMENT_RESIDUAL = 0.25


def anchor_matches(
    conf,
    static,
    rng=None,
    preferred=None,
    target=ANCHOR_MATCH_TARGET,
    minimum=ANCHOR_MATCH_MIN,
    pool=ANCHOR_MATCH_POOL,
    relative=ANCHOR_RELATIVE_CONF,
    floor=CONF_FLOOR,
    min_anchors=MIN_USABLE_ANCHORS,
    min_total=MIN_TOTAL_MATCHES,
    majority=ABSOLUTE_ANCHOR_MAJORITY,
):
    """Which pixel of each anchor view the batch alignment matches on.

    `conf` (A,H,W) and `static` (A,H,W) bool -- the pixels of each anchor that are scene rather
    than person. `preferred` (A,) bool names the views that cleared the absolute confidence cut
    on their own. Returns (kept, records): `kept` is a list of (anchor index, flat pixel indices)
    for the views that can carry the fit, `records` one dict per view for the metadata. A view
    with too few static pixels, or one the model is far less sure of than the best view, is
    dropped and named; only a clip where too little survives raises, and the error carries the
    counts rather than reaching the SVD as an empty array.
    """
    conf = np.asarray(conf, dtype=float)
    static = np.asarray(static, dtype=bool)
    rng = np.random.default_rng(13) if rng is None else rng
    picks, records = [], []
    for j in range(len(conf)):
        flat_conf = conf[j].reshape(-1)
        candidates = np.flatnonzero(static[j].reshape(-1) & np.isfinite(flat_conf))
        record = dict(anchor=int(j), staticPixels=int(candidates.size), matches=0, strength=0.0)
        if candidates.size < minimum:
            record["dropped"] = f"only {candidates.size} static pixels, needs {minimum}"
            picks.append(None)
            records.append(record)
            continue
        ranked = candidates[np.argsort(flat_conf[candidates], kind="stable")][::-1]
        top = ranked[: max(minimum, min(len(ranked), pool * target))]
        chosen = rng.choice(top, min(target, len(top)), replace=False) if len(top) > target else top
        record["matches"] = int(len(chosen))
        record["strength"] = float(np.median(flat_conf[top[: min(len(top), target)]]))
        picks.append(np.asarray(chosen, dtype=np.int64))
        records.append(record)
    strengths = [r["strength"] for r, p in zip(records, picks) if p is not None]
    best = max(strengths) if strengths else 0.0
    cut = max(float(floor), float(relative) * best)
    kept = []
    for j, (pick, record) in enumerate(zip(picks, records)):
        if pick is None:
            continue
        if record["strength"] < cut:
            record["dropped"] = (
                f"confidence {record['strength']:.3f} is under {cut:.3f}, "
                f"{relative:.0%} of the best anchor's {best:.3f}: this view shares no "
                "confident content with the others"
            )
            continue
        kept.append((j, pick))
    if preferred is not None:
        confident = [(j, pick) for j, pick in kept if bool(np.asarray(preferred)[j])]
        enough = sum(len(pick) for _, pick in confident)
        if (
            len(confident) >= max(min_anchors, math.ceil(majority * len(kept)))
            and enough >= min_total
        ):
            for j, _ in kept:
                if not bool(np.asarray(preferred)[j]):
                    records[j]["dropped"] = (
                        f"{len(confident)} of {len(kept)} usable anchors clear the absolute "
                        "confidence cut, so the fit is made from those alone"
                    )
            kept = confident
    total = sum(len(pick) for _, pick in kept)
    if len(kept) < min_anchors or total < min_total:
        raise RuntimeError(
            f"Anchor alignment has {len(kept)} usable anchor view(s) of {len(conf)} and "
            f"{total} matched points; it needs {min_anchors} views and {min_total} points. "
            "Per view: "
            + "; ".join(
                f"{r['anchor']}: {r['staticPixels']} static, {r['matches']} matched, "
                f"confidence {r['strength']:.3f}"
                + (f" -- dropped ({r['dropped']})" if "dropped" in r else "")
                for r in records
            )
        )
    return kept, records


# Largest camera baseline between any two anchors, as a multiple of the clip's median anchor
# depth. Pi3X relates views by content, so anchors that never see the same surface produce a
# joint prediction whose frames are not in one another's coordinates at all, and every batch is
# then aligned to a fiction. Measured across the same saved anchors: every clip that solved is
# at or under 1.04 (hp-fly-s63 0.01, img5594 0.68, movie-s26 1.04) and the two crowded failures
# are at 1.37 and 1.50, while game-s1 -- 22 s of third-person traversal across several areas --
# is at 11.44, with 3.9x and 5.6x between consecutive anchors. 3.0 sits above every clip this
# stage has solved or could have solved and well below a traversal.
ANCHOR_SPREAD_LIMIT = 3.0


def measure_spread(poses, depths):
    """(spread, median depth) for an anchor set, with no limit applied.

    `depths` is each anchor's median depth in its own camera. Separated from the refusal below
    because the spread is now also a *decision*: under the limit one global anchor set can carry
    the clip, over it the clip is solved as a chain of shorter windows instead of being refused.
    """
    centres = np.asarray(poses, dtype=float)[:, :3, 3]
    finite = np.asarray(depths, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if len(centres) < 2 or finite.size == 0:
        raise RuntimeError(
            f"Anchor geometry is unusable: {len(centres)} anchor pose(s) and "
            f"{finite.size} anchor(s) with a positive median depth"
        )
    depth = float(np.median(finite))
    gaps = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    return float(np.max(gaps) / depth), depth


def anchor_spread(poses, depths, limit=ANCHOR_SPREAD_LIMIT):
    """(spread, median depth) for the anchor set, refusing a set whose anchors do not overlap.

    The error names the measurement and the remedy, because no confidence or mask change can
    recover views that share no content: that span has to be cut to a window a single anchor set
    can carry. The stage now cuts it automatically (`plan_windows`); this stays the guard each
    individual window is still held to.
    """
    spread, depth = measure_spread(poses, depths)
    if spread > limit:
        centres = np.asarray(poses, dtype=float)[:, :3, 3]
        gaps = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        raise RuntimeError(
            f"The anchor views span {spread:.1f} scene depths (largest baseline "
            f"{float(np.max(gaps)):.2f} against a median depth of {depth:.2f}); above "
            f"{limit:.1f} they share too little content for one shared reference frame. "
            "This clip traverses further than a single solve can anchor: select a shorter "
            "segment and solve it on its own."
        )
    return spread, depth


# --- chained alignment ----------------------------------------------------------------------
# One global anchor set requires every anchor to be co-visible, which a travelling camera -- a
# chase down a causeway, a runner up a long flight of steps -- never is. Over the spread limit the
# clip is split into overlapping windows that each pass `anchor_spread` on their own, solved like
# small global solves, and registered to one another on the frames they share.
ALIGNMENT_GLOBAL = "global"
ALIGNMENT_CHAINED = "chained"

# What `anchors.npz` holds, so a resumed run can tell the survey prediction (the global anchor set
# that decided the mode) from the base-frame prediction (window 0's anchors) that replaces it in a
# chained solve. A file written before this existed is a global anchor set by construction.
ANCHOR_ROLE_GLOBAL = "global"
ANCHOR_ROLE_WINDOW = "chain-window-0"

# Share of ANCHOR_SPREAD_LIMIT a window is *planned* to spend. The plan comes from the survey
# prediction, whose baselines relate views that are not co-visible and so are an estimate, not a
# measurement; and it is measured as travelled path length, which is at least the largest baseline
# and equals it only for a straight run. 0.6 leaves room for both errors, and every window's own
# spread is measured afterwards anyway, so the cost of being wrong here is a re-prediction and not
# a bad solve. On game-run's surveyed 3.68-depth path this cuts 3 windows; on game-s1's 18.8, 11.
CHAIN_TARGET_SPREAD = 0.6
# A window that measures under this share of the limit left most of its budget unused -- the
# survey over-estimated its travel -- so it is grown once and measured again. The product wants
# segments as long as possible, and an unused budget is a window boundary, a link and its drift
# that the clip did not need.
CHAIN_GROW_SPREAD = 0.5
# Most a single grow may multiply a window's length by. A window whose measured spread is a tenth
# of the limit is not evidence that six times its length is co-visible; it is evidence the camera
# was still for that second.
CHAIN_MAX_GROW = 2.0
# Anchor predictions one window may spend settling its length. One is the normal case; the rest
# pay for a shrink after an over-budget measurement and for the single grow.
CHAIN_WINDOW_ATTEMPTS = 3
# Frames two consecutive windows share. One shared frame is already enough geometry -- a dense
# depth map of one view spans the scene in all three axes -- so this is redundancy against a
# shared frame that is blurred, nearly all person, or looking at a blank wall, and it averages the
# link over a third of a second of camera motion at this stage's minimum 12 fps. Each shared frame
# is predicted twice, once per window, which is the whole extra inference cost of a link.
CHAIN_OVERLAP_SAMPLES = 4
# Shortest window, one second at the minimum rate. Below this a window's own anchors are nearly
# the same view, so it measures a tiny spread and contributes a link and its drift while adding
# almost no independent geometry. A camera that cannot hold co-visibility for one second is not
# something this stage can chain, and says so.
CHAIN_MIN_WINDOW = 12
# A link is a similarity fitted to two predictions of the same frames, so it is held to the same
# residual the batch alignments are held to, measured against the same thing: the median depth of
# the shared frames. Not separately calibrated -- there is no GPU run of a chained solve to
# calibrate it on -- and deliberately not tightened past a limit this stage has measured.
MAX_LINK_RESIDUAL = MAX_ALIGNMENT_RESIDUAL
# The shared frames' median depth, measured in each window's own prediction and brought to a
# common frame by the link's scale, is the same physical distance twice. Each Pi3X prediction
# chooses its own arbitrary scale, so the link's raw scale says nothing; this ratio is the
# scale-free version of it and has to be near 1. 1.25 allows the depth medians of the two
# predictions to disagree by a quarter, which is the sort of disagreement two honest predictions
# of the same frames produce; a factor-of-two ratio is a link that matched the wrong surface.
LINK_DEPTH_RATIO_BAND = 1.25
# The link is fitted to scene points, so how well it also carries the shared frames' CAMERA
# centres from one window to the other is an independent check on the same data. Held to the same
# measured residual limit for the same reason as MAX_LINK_RESIDUAL.
MAX_LINK_CAMERA_RESIDUAL = MAX_ALIGNMENT_RESIDUAL


def travel_profile(poses, samples, depth, count):
    """Cumulative camera path length, in scene depths, at every sample of the clip.

    Built from the survey prediction -- the global anchor set -- by walking its camera centres in
    order and interpolating between the samples they sit on. Those centres relate views that are
    not co-visible, so this is an estimate of where the camera went and not a measurement of it;
    it is used only to propose window lengths, each of which is then measured on its own.
    """
    centres = np.asarray(poses, dtype=float)[:, :3, 3]
    samples = np.asarray(samples, dtype=float)
    if len(centres) != len(samples) or len(centres) < 2:
        raise RuntimeError(
            f"The travel profile needs a pose per anchor sample, got {len(centres)} and "
            f"{len(samples)}"
        )
    depth = float(depth)
    if not np.isfinite(depth) or depth <= 0:
        raise RuntimeError(f"The travel profile needs a positive scene depth, got {depth}")
    walked = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1) / depth)]
    )
    return np.interp(np.arange(int(count), dtype=float), samples, walked)


def next_window(
    profile,
    first,
    last,
    budget,
    overlap=CHAIN_OVERLAP_SAMPLES,
    minimum=CHAIN_MIN_WINDOW,
):
    """The end sample of the window starting at `first`, on the surveyed travel budget."""
    profile = np.asarray(profile, dtype=float)
    first, last, minimum, overlap = int(first), int(last), int(minimum), int(overlap)
    if first >= last:
        return last
    over = np.flatnonzero(profile[first : last + 1] - profile[first] > float(budget))
    end = last if over.size == 0 else first + int(over[0]) - 1
    end = min(last, max(end, first + minimum - 1))
    # A tail too short to be a window of its own is absorbed into this one rather than left to
    # become a one-frame window whose anchors are one view.
    if end < last and last - (end - overlap + 1) + 1 < minimum:
        end = last
    return int(end)


def plan_windows(
    profile,
    count,
    limit=ANCHOR_SPREAD_LIMIT,
    fraction=CHAIN_TARGET_SPREAD,
    overlap=CHAIN_OVERLAP_SAMPLES,
    minimum=CHAIN_MIN_WINDOW,
):
    """Overlapping inclusive [first, last] sample windows for a chained solve.

    The plan the survey proposes, before any window has been predicted. The runtime walks it one
    window at a time and re-plans from the same function whenever a measured window turns out
    shorter or longer than its proposal, so this is also what the planner would choose.
    """
    count = int(count)
    if count < int(minimum):
        raise RuntimeError(
            f"A chained solve needs at least {int(minimum)} samples to fill one window, got {count}"
        )
    if int(overlap) < 2 or int(overlap) + 2 > int(minimum):
        raise RuntimeError(
            f"A {int(overlap)}-sample overlap does not fit inside a {int(minimum)}-sample window"
        )
    budget = float(fraction) * float(limit)
    windows, first = [], 0
    while True:
        end = next_window(profile, first, count - 1, budget, overlap, minimum)
        windows.append((int(first), int(end)))
        if end >= count - 1:
            return windows
        following = end - int(overlap) + 1
        if following <= first:
            raise RuntimeError(
                f"Window planning made no progress at sample {first}: a window of "
                f"{end - first + 1} sample(s) cannot carry a {int(overlap)}-sample overlap"
            )
        first = following


def adapt_window(
    first,
    end,
    spread,
    last,
    limit=ANCHOR_SPREAD_LIMIT,
    fraction=CHAIN_TARGET_SPREAD,
    grow=CHAIN_GROW_SPREAD,
    cap=CHAIN_MAX_GROW,
    minimum=CHAIN_MIN_WINDOW,
):
    """(action, end) once a proposed window's own anchor spread has been measured.

    The measurement outranks the survey that proposed the length: it is made from an anchor set
    that IS co-visible, and it is the spread every batch of this window will be aligned against.
    `shrink` scales the window down by how far over the limit it came in, `grow` doubles back on a
    survey that over-estimated the travel, `fail` is a window already at its floor length that
    still does not hold together.
    """
    first, end, last, minimum = int(first), int(end), int(last), int(minimum)
    spread, limit = float(spread), float(limit)
    length = end - first + 1
    floor = min(minimum, last - first + 1)
    if not np.isfinite(spread) or spread > limit:
        if not np.isfinite(spread):
            return "fail", end
        shrunk = min(end - 1, first + max(floor, int(length * fraction * limit / spread)) - 1)
        if shrunk < first + floor - 1:
            return "fail", end
        return "shrink", int(shrunk)
    if end < last and spread < grow * limit:
        factor = min(float(cap), fraction * limit / max(spread, 1e-6))
        grown = min(last, first + int(math.ceil(length * factor)) - 1)
        if grown > end:
            return "grow", int(grown)
    return "accept", end


def compose_similarity(outer, inner):
    """The single similarity that applies `inner` and then `outer`."""
    outer_scale, outer_rotation, outer_translation = outer
    inner_scale, inner_rotation, inner_translation = inner
    outer_rotation = np.asarray(outer_rotation, dtype=float)
    return (
        float(outer_scale) * float(inner_scale),
        outer_rotation @ np.asarray(inner_rotation, dtype=float),
        float(outer_scale) * outer_rotation @ np.asarray(inner_translation, dtype=float)
        + np.asarray(outer_translation, dtype=float),
    )


def identity_similarity():
    """The transform a window 0 needs: its own prediction already defines the base frame."""
    return (1.0, np.eye(3), np.zeros(3))


def apply_similarity(transform, points):
    """`points` moved by a (scale, rotation, translation) triple, in the caller's dtype."""
    scale, rotation, translation = transform
    return np.asarray(points) @ np.asarray(rotation, dtype=float).T * float(scale) + np.asarray(
        translation, dtype=float
    )


def chain_transforms(links):
    """Every window's transform into window 0's frame, composed from the per-link fits."""
    transforms = [identity_similarity()]
    for link in links:
        transforms.append(compose_similarity(transforms[-1], link))
    return transforms


def link_quality(
    index,
    samples,
    points,
    rms,
    scale,
    depth_previous,
    depth_current,
    camera_rms,
    remaining_path=0.0,
    residual=MAX_LINK_RESIDUAL,
    band=LINK_DEPTH_RATIO_BAND,
    camera=MAX_LINK_CAMERA_RESIDUAL,
):
    """One window-to-window registration, measured and judged.

    `rms` and `camera_rms` are in the previous window's units, so both are reported against
    `depth_previous`. `depth_previous` and `depth_current` are the median depth of the SAME shared
    frames as each window's own prediction measured it, which makes `scale * depth_current /
    depth_previous` one physical distance compared across two predictions: it has to be near 1
    whatever arbitrary scale either prediction chose, while the raw `scale` on its own says
    nothing at all.
    """
    depth_previous, depth_current = float(depth_previous), float(depth_current)
    rms, camera_rms, scale = float(rms), float(camera_rms), float(scale)
    usable = np.isfinite(depth_previous) and depth_previous > 0
    ratio = scale * depth_current / depth_previous if usable else float("nan")
    record = dict(
        link=int(index),
        sharedSamples=[int(i) for i in samples],
        points=int(points),
        rms=rms,
        rmsDepths=rms / depth_previous if usable else float("nan"),
        scale=scale,
        depthRatio=ratio,
        depthPrevious=depth_previous,
        depthCurrent=depth_current,
        cameraRms=camera_rms,
        cameraRmsDepths=camera_rms / depth_previous if usable else float("nan"),
        remainingPathDepths=float(remaining_path),
    )
    failures = []
    if not usable:
        failures.append(f"the shared frames have no positive median depth ({depth_previous})")
    if not np.isfinite(record["rmsDepths"]) or record["rmsDepths"] > float(residual):
        failures.append(
            f"alignment residual {record['rmsDepths']:.0%} of the {depth_previous:.2f} shared "
            f"depth, over the {float(residual):.0%} this stage accepts"
        )
    if not np.isfinite(ratio) or not (1 / float(band)) <= ratio <= float(band):
        failures.append(
            f"the shared frames measure {depth_current:.2f} deep in this window and "
            f"{depth_previous:.2f} in the previous one, a ratio of {ratio:.2f} after the link's "
            f"scale, outside 1/{float(band):g}..{float(band):g}"
        )
    if not np.isfinite(record["cameraRmsDepths"]) or record["cameraRmsDepths"] > float(camera):
        failures.append(
            f"the shared frames' camera centres land {record['cameraRmsDepths']:.0%} of a scene "
            f"depth apart, over the {float(camera):.0%} this stage accepts"
        )
    record["ok"] = not failures
    record["failures"] = failures
    return record


def drift_bound(links):
    """How far the far end of the chain may have walked away from window 0's frame.

    Each link leaves a registration residual, and it leaves it as both a position error and a
    rotation error of roughly the same size in radians -- the points it was fitted to sit about
    one scene depth away -- which the rest of the chain then levers over however far the camera
    still travels. Both terms are in scene depths. This is a first-order bound from the residuals
    the links actually measured, not an observation of drift: nothing here sees the far end of the
    clip and window 0's scene at the same time, so a chained solve cannot close its own loop.
    """
    residual = 0.0
    bound = 0.0
    for link in links:
        value = float(link.get("rmsDepths", float("nan")))
        if not np.isfinite(value):
            continue
        residual += value
        bound += value * (1.0 + float(link.get("remainingPathDepths", 0.0)))
    return dict(
        links=len(links),
        residualDepths=residual,
        boundDepths=bound,
        note=(
            "Accumulated per-link registration residual, in scene depths, plus each link's "
            "rotation error levered over the path still to come. A first-order bound computed "
            "from the measured link residuals; the chain never observes both ends of the clip at "
            "once and so cannot measure its own drift."
        ),
    )


def localise_break(windows, links):
    """Which link broke the chain, and the longest prefix and suffix of windows that registered.

    `links[k]` registers window k+1 onto window k and carries `ok`: True, False, or None for a
    link the run never reached. A trim suggestion is only honest over links that were measured, so
    an unattempted link ends a run exactly as a failed one does and is reported separately.
    """
    ranges = [(int(first), int(last)) for first, last in windows]
    status = [link.get("ok") for link in links]
    if len(status) != len(ranges) - 1:
        raise RuntimeError(
            f"{len(ranges)} window(s) need {len(ranges) - 1} links, got {len(status)}"
        )

    def span(first_window, last_window):
        return dict(
            windows=[int(first_window), int(last_window)],
            firstSample=ranges[first_window][0],
            lastSample=ranges[last_window][1],
            samples=ranges[last_window][1] - ranges[first_window][0] + 1,
        )

    prefix = 0
    while prefix < len(status) and status[prefix] is True:
        prefix += 1
    suffix = len(ranges) - 1
    while suffix > 0 and status[suffix - 1] is True:
        suffix -= 1
    judged = [link for link in links if link.get("ok") is False]
    if not judged:
        judged = [link for link in links if link.get("ok") is True]
    weakest = None
    if judged:
        weakest = max(
            judged,
            key=lambda link: (
                float(link["rmsDepths"]) if np.isfinite(link.get("rmsDepths", np.nan)) else np.inf
            ),
        )
    return dict(
        ok=all(state is True for state in status),
        windows=len(ranges),
        weakestLink=weakest,
        failedLinks=[k for k, state in enumerate(status) if state is False],
        unattemptedLinks=[k for k, state in enumerate(status) if state is None],
        registeredPrefix=span(0, prefix),
        registeredSuffix=span(suffix, len(ranges) - 1),
    )


def viewer_camera(
    pose,
    batch_scale,
    batch_rotation,
    batch_translation,
    reference_rotation,
    reference_translation,
    world_scale,
):
    flip = np.diag([1.0, -1.0, -1.0])
    center = batch_scale * batch_rotation @ pose[:3, 3] + batch_translation
    matrix = np.eye(4)
    matrix[:3, :3] = flip @ reference_rotation @ batch_rotation @ pose[:3, :3] @ flip
    matrix[:3, 3] = flip @ (reference_rotation @ center + reference_translation) * world_scale
    return matrix


def fit_ray_intrinsics(rays, valid, source_size):
    """Fit a pinhole K to learned rays; retain residual because rays need not be pinhole."""
    h, w = rays.shape[:2]
    y, x = np.mgrid[:h, :w]
    good = valid & np.isfinite(rays).all(-1) & (rays[..., 2] > 1e-6)
    good[1::2] = False
    good[:, 1::2] = False
    xy = rays[good, :2] / rays[good, 2:3]
    pixels = np.stack([x[good] + 0.5, y[good] + 0.5], axis=1)
    if len(xy) < 100:
        raise ValueError("Insufficient valid rays for intrinsic fit")
    K = np.eye(3)
    errors = []
    for axis in range(2):
        design = np.column_stack([xy[:, axis], np.ones(len(xy))])
        focal, principal = np.linalg.lstsq(design, pixels[:, axis], rcond=None)[0]
        if focal <= 0:
            raise ValueError("Nonpositive fitted focal length")
        K[axis, axis], K[axis, 2] = focal, principal
        errors.append(design @ [focal, principal] - pixels[:, axis])
    source_K = np.diag([source_size[0] / w, source_size[1] / h, 1.0]) @ K
    return dict(
        intrinsics=K.tolist(),
        source_intrinsics=source_K.tolist(),
        image_size=[w, h],
        source_image_size=list(source_size),
        intrinsic_fit_pixel_rmse=float(np.sqrt(np.mean(np.sum(np.square(errors), axis=0)))),
        intrinsics_method="Least-squares zero-skew pinhole fit to predicted rays; pixel centers at x+0.5,y+0.5. Source K undoes the two resize operations. This is estimated, not supplied calibration.",
    )


# Frames whose own rays fitted a pinhole. Every other frame is filled from this clip's median
# K, so the median has to come from most of the clip: below half of it the intrinsics are no
# longer a measurement of this camera and no frame's projection can be trusted.
MIN_INTRINSIC_FIT_FRACTION = 0.5


def require_intrinsic_fits(fitted, total, fraction=MIN_INTRINSIC_FIT_FRACTION):
    """Refuse a solve whose intrinsics would mostly be borrowed from other frames."""
    fitted, total = int(fitted), int(total)
    if total < 1:
        raise ValueError("No solved camera frames to fit intrinsics for")
    if fitted < 1 or fitted < fraction * total:
        raise ValueError(
            f"Only {fitted} of {total} frames ({fitted / total:.1%}) fitted a pinhole to their "
            f"Pi3X rays; below {fraction:.0%} the clip median is not a measurement of this "
            "camera, so no frame's intrinsics are usable"
        )
    return fitted / total


def median_intrinsics(entries):
    """This clip's own median K, for a frame whose rays would not fit one.

    A dark, blurred or transition frame can leave too few confident rays for a pinhole fit,
    yet the camera did not change lens for that frame: the frames that did fit describe the
    same optics. The result is flagged per camera entry and never reported as a fit, and it
    carries no per-frame residual because none was measured.
    """
    if not entries:
        raise ValueError("No fitted intrinsics to take a median of")
    sizes = {tuple(entry["image_size"]) for entry in entries}
    sources = {tuple(entry["source_image_size"]) for entry in entries}
    if len(sizes) != 1 or len(sources) != 1:
        raise ValueError("Fitted intrinsics disagree about the image size")
    (w, h), (sw, sh) = sizes.pop(), sources.pop()
    stack = np.array([entry["intrinsics"] for entry in entries], float)
    K = np.eye(3)
    for row, column in ((0, 0), (1, 1), (0, 2), (1, 2)):
        K[row, column] = float(np.median(stack[:, row, column]))
    residuals = [
        float(entry["intrinsic_fit_pixel_rmse"])
        for entry in entries
        if entry.get("intrinsic_fit_pixel_rmse") is not None
    ]
    return dict(
        intrinsics=K.tolist(),
        source_intrinsics=(np.diag([sw / w, sh / h, 1.0]) @ K).tolist(),
        image_size=[int(w), int(h)],
        source_image_size=[int(sw), int(sh)],
        intrinsic_fit_pixel_rmse=None,
        intrinsics_method=(
            f"Per-parameter median of the {len(entries)} frames in this clip whose rays did fit "
            "a zero-skew pinhole; this frame's own rays did not. Same camera, estimated, not a "
            "fit of this frame and not supplied calibration."
        ),
        intrinsicsMedianFrames=len(entries),
        intrinsicsMedianPixelRmse=float(np.median(residuals)) if residuals else None,
    )


DECODE_OPENCV = "opencv-sequential"
DECODE_FFMPEG = "ffmpeg-rawvideo"

TIMES_PTS = "ffprobe-pts"
TIMES_POS_MSEC = "opencv-pos-msec"
TIMES_AVERAGE_FPS = "index-over-average-fps"

SELECT_PTS_SLOTS = "decoded-pts-slots"
SELECT_CONTAINER_INDEX = "container-index-over-average-fps"

# One second of samples at this stage's minimum 12 fps. Fewer than this is not a camera path
# anyone can walk along, so a source that decodes to less has failed rather than come up short.
MIN_OUTPUT_SAMPLES = 12

# A file that stops decoding before half the frames its container advertises is truncated or
# corrupt. Reconstructing the prefix and presenting it as the clip would hide that.
MIN_DECODED_SOURCE_FRACTION = 0.5

# Output slots with no decoded source frame of their own. A few are a stalled variable-rate
# source; many mean the source cannot supply the requested independent frame rate at all.
# FFmpeg's fps filter fills such a gap by duplicating a neighbour, and a duplicate is not an
# independent reconstruction, so the slot is skipped and counted instead.
MIN_OUTPUT_SLOT_FILL = 0.95


def opencv_first_frame(cv2, video):
    """Can OpenCV's bundled libavcodec open this file and hand back its first frame?"""
    cap = cv2.VideoCapture(str(video))
    try:
        opened = bool(cap.isOpened())
        first = bool(cap.read()[0]) if opened else False
    finally:
        cap.release()
    return opened, first


def choose_decode_backend(opened, first_frame):
    """OpenCV unless it cannot produce a frame.

    AV1 and some 10-bit HEVC files open cleanly through OpenCV -- frame count, size and rate
    all read back -- and then fail on every single read, because the bundled libavcodec has no
    software decoder for them. That is a property of the decoder, not of the clip, so the
    fallback is chosen from observed behaviour rather than from the codec name.
    """
    if opened and first_frame:
        return DECODE_OPENCV, "OpenCV decoded the first frame"
    if not opened:
        return DECODE_FFMPEG, "OpenCV could not open the source"
    return DECODE_FFMPEG, "OpenCV opened the source but could not decode its first frame"


def usable_timestamps(values):
    """Are these per-frame seconds a real, ordered decoder timeline?"""
    try:
        stamps = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    if len(stamps) < 2 or not all(math.isfinite(stamp) for stamp in stamps):
        return False
    return all(b > a for a, b in zip(stamps, stamps[1:]))


def decoded_timeline(frames, pts_seconds=None, pos_msec_seconds=None, average_fps=None):
    """Seconds per decoded frame, relative to the first decoded frame, plus their provenance.

    The decoder's own timestamps describe a variable-rate source; index divided by the
    container's average rate only describes a constant-rate one. Whichever is used is named
    in the output so a reader never has to guess which it got.
    """
    frames = int(frames)
    for values, label in ((pts_seconds, TIMES_PTS), (pos_msec_seconds, TIMES_POS_MSEC)):
        if values is None or len(values) < frames or frames < 2:
            continue
        stamps = list(values)[:frames]
        if usable_timestamps(stamps):
            return [float(stamp) - float(stamps[0]) for stamp in stamps], label
    average_fps = float(average_fps or 0)
    if not math.isfinite(average_fps) or average_fps <= 0 or frames < 2:
        raise ValueError(
            f"No usable decoder timestamps for {frames} frames and no positive average rate"
        )
    return [index / average_fps for index in range(frames)], TIMES_AVERAGE_FPS


def output_slots(times, fps):
    """The frame FFmpeg's fps filter would retain in each output slot, from the same timestamps.

    The filter rounds every decoded frame's relative timestamp into an output slot and writes
    the last input that landed there. Reproducing that rule from the decoded timestamps keeps
    the solver's samples on the frames the cleaner's `resample_source` keeps, instead of on
    whatever `round(time * average_fps)` happens to address in a variable-rate file.
    """
    fps = float(fps)
    latest = {}
    for index, seconds in enumerate(times):
        latest[int(math.floor(float(seconds) * fps + 0.5))] = index
    slots = sorted(latest)
    if not slots:
        return [], 0
    return [latest[slot] for slot in slots], slots[-1] - slots[0] + 1 - len(slots)


def container_samples(frames, average_fps, fps):
    """Index-addressed samples from container metadata alone, when no timestamps survive."""
    frames, average_fps, fps = int(frames), float(average_fps), float(fps)
    requested = np.arange(int(np.ceil(frames / average_fps * fps))) / fps
    indices = np.clip(np.round(requested * average_fps).astype(int), 0, frames - 1)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Source frame rate below requested independent output density")
    return [int(index) for index in indices]


def sample_plan(fps, frames, average_fps, pts_seconds=None, pos_msec_seconds=None):
    """Which source frames this solve will reconstruct, and where their times came from."""
    times, label = decoded_timeline(frames, pts_seconds, pos_msec_seconds, average_fps)
    if label == TIMES_AVERAGE_FPS:
        samples = container_samples(frames, average_fps, fps)
        empty, selection = 0, SELECT_CONTAINER_INDEX
    else:
        samples, empty = output_slots(times, fps)
        selection = SELECT_PTS_SLOTS
    return dict(
        samples=samples,
        times=[times[index] for index in samples],
        timestampSource=label,
        sampleSelection=selection,
        emptyOutputSlots=empty,
        plannedSamples=len(samples),
    )


def clamp_plan(
    plan,
    decodable,
    container_frames,
    fps,
    minimum=MIN_OUTPUT_SAMPLES,
    source_fraction=MIN_DECODED_SOURCE_FRACTION,
    slot_fill=MIN_OUTPUT_SLOT_FILL,
):
    """Cut the plan back to the frames that actually decoded, and refuse a broken source.

    A container can advertise more frames than a decoder will ever hand back. Dropping the
    samples behind that tail is a disclosure, not a failure, so the counts are returned for
    the metadata rather than raised; only an unreadable or truncated file raises.
    """
    fps, decodable, container_frames = float(fps), int(decodable), int(container_frames)
    if decodable < 2:
        raise RuntimeError(f"Only {decodable} source frames decode; the source is unreadable")
    if container_frames > 0 and decodable < source_fraction * container_frames:
        raise RuntimeError(
            f"Only {decodable} of the {container_frames} frames the container declares decode "
            f"({decodable / container_frames:.1%}); below {source_fraction:.0%} the file is "
            "truncated or corrupt and its prefix is not the clip"
        )
    kept = [int(index) for index in plan["samples"] if int(index) < decodable]
    dropped = len(plan["samples"]) - len(kept)
    if len(kept) < minimum:
        raise RuntimeError(
            f"Only {len(kept)} of {len(plan['samples'])} planned samples decode from "
            f"{decodable} decodable source frames; this stage needs at least {minimum}"
        )
    times = [float(second) for second in plan["times"][: len(kept)]]
    slots = int(math.floor(times[-1] * fps + 0.5)) + 1
    if len(kept) < slot_fill * slots:
        raise RuntimeError(
            f"Only {len(kept)} of {slots} output slots at {fps:g} fps have a decoded source "
            f"frame of their own ({len(kept) / slots:.1%}); below {slot_fill:.0%} the source "
            "cannot supply the requested independent frame rate"
        )
    return dict(
        plan,
        samples=kept,
        times=times,
        containerFrames=container_frames,
        decodableFrames=decodable,
        droppedTailSamples=dropped,
        emptyOutputSlots=slots - len(kept),
        outputSlotFill=len(kept) / slots,
    )


def probe_source(video):
    """FFprobe's view of the stream, including every decoded presentation timestamp.

    FFprobe decodes, so its frame list is the decodable truth the container may contradict,
    and its timestamps are the same ones `wander_worker.source_timing` binds the cleaner to.
    A missing or unreadable ffprobe is not fatal; the caller falls back and says so.
    """
    if not shutil.which("ffprobe"):
        return {}
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,nb_frames"
        ":stream_side_data=rotation:frame=pts_time",
        "-of",
        "json",
        str(video),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=True)
        document = json.loads(done.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    stream = (document.get("streams") or [{}])[0]
    stamps = []
    for frame in document.get("frames") or []:
        try:
            stamps.append(float(frame["pts_time"]))
        except (KeyError, TypeError, ValueError):
            stamps = []
            break
    try:
        average = float(Fraction(stream.get("avg_frame_rate") or "0/1"))
    except (ValueError, ZeroDivisionError):
        average = 0.0
    rotation = 0
    for side in stream.get("side_data_list") or []:
        try:
            rotation = int(float(side["rotation"])) % 360
        except (KeyError, TypeError, ValueError):
            continue
    return dict(
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        codec=stream.get("codec_name") or "",
        averageFps=average,
        containerFrames=int(stream.get("nb_frames") or 0),
        rotation=rotation,
        pts=stamps,
    )


def decode_opencv(cv2, video, wanted, sink):
    """Decode forward with grab/retrieve, converting only the frames the solve wants.

    Nothing seeks. `CAP_PROP_POS_FRAMES` cannot reach the last frames of a variable-rate file
    -- on the soccer clip every index from 706 on fails while all 712 decode in order -- and a
    sequential pass also counts what really decodes instead of trusting the container.
    """
    cap = cv2.VideoCapture(str(video))
    ordinal, stamps = 0, []
    try:
        while cap.grab():
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if ordinal in wanted:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                sink(ordinal, bgr)
            ordinal += 1
    finally:
        cap.release()
    return ordinal, stamps[:ordinal]


def decode_ffmpeg(video, width, height, wanted, sink):
    """Raw frames straight out of ffmpeg, for codecs OpenCV's libavcodec cannot decode.

    Full resolution and no re-encode, so the frames are the source's own pixels and the single
    resize the intrinsics undo stays the one OpenCV performs for both backends. Display
    rotation is left unapplied on purpose: OpenCV does not apply it either, and the recorded
    source size has to describe the same pixels whichever backend produced them.
    """
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise RuntimeError("The ffmpeg fallback needs the source frame size")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-noautorotate",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        # Without this ffmpeg makes the rawvideo output constant-rate by duplicating frames,
        # which would shift every ordinal the plan was built from.
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_bytes = width * height * 3
    ordinal = 0
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                buffer = proc.stdout.read(frame_bytes)
                if len(buffer) < frame_bytes:
                    break
                if ordinal in wanted:
                    frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3).copy()
                    sink(ordinal, frame)
                ordinal += 1
        finally:
            proc.stdout.close()
            proc.wait()
        errors.seek(0)
        message = errors.read().decode("utf-8", errors="replace").strip()[-2000:]
    if ordinal < 1:
        raise RuntimeError(
            f"ffmpeg decoded no frames from {video}; the image needs a decoder for this "
            f"codec. ffmpeg said: {message or '(nothing)'}"
        )
    return ordinal, []


def main():
    # Kept out of module scope so the pure gap helpers above import under plain numpy.
    import cv2
    from PIL import Image
    from wander_worker.masks import foreground_people_masks, people_masks
    from wander_worker.ply import write_point_ply, home_distance

    def solve_people_masks(images, mode, dilate_px):
        """The people mask this run decided on, for the static exclusion and the person export."""
        if mode == MASK_FOREGROUND:
            return foreground_people_masks(images, dilate_px=dilate_px)
        return people_masks(images, dilate_px=dilate_px)

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repo", default=os.path.expanduser("~/pi3"))
    a = ap.parse_args()
    if a.fps < 12:
        raise ValueError("Dense quality experiment requires >=12 fps")
    if a.anchors < 2 or a.batch < 1:
        raise ValueError("Need >=2 anchors and a positive batch size")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    framesdir = out / "input"
    framesdir.mkdir(exist_ok=True)
    temp = out / "batch-input"
    temp.mkdir(exist_ok=True)
    t0 = time.time()
    probe = probe_source(a.video)
    opened, first_frame = opencv_first_frame(cv2, a.video)
    backend, backend_reason = choose_decode_backend(opened, first_frame)
    with_cap = cv2.VideoCapture(a.video)
    source_size = (
        int(probe.get("width") or with_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        int(probe.get("height") or with_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    srcfps = float(probe.get("averageFps") or with_cap.get(cv2.CAP_PROP_FPS) or 0)
    container = int(probe.get("containerFrames") or with_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    with_cap.release()
    if min(source_size) < 1 or not np.isfinite(srcfps) or srcfps <= 0:
        raise ValueError("Invalid source video metadata")
    pts = probe.get("pts") or []
    horizon = len(pts) if len(pts) >= 2 else container
    stamps = []
    if len(pts) < 2 and backend == DECODE_OPENCV:
        # No ffprobe timestamps: decode once without keeping anything, for the decoder's own
        # clock and for the frame count the container may be lying about.
        horizon, stamps = decode_opencv(cv2, a.video, set(), lambda *_: None)
    if horizon < 2:
        raise ValueError("Invalid source video metadata")
    plan = sample_plan(a.fps, horizon, srcfps, pts_seconds=pts, pos_msec_seconds=stamps)
    wanted = {source: sample for sample, source in enumerate(plan["samples"])}
    # A resumed run reuses f_NNNNNN.jpg by sample number. If this run samples different source
    # frames than the last one did, those files are pictures of other moments, and the cached
    # anchor prediction was made from them, so both go.
    fingerprint = json.dumps(dict(fps=a.fps, backend=backend, samples=plan["samples"]))
    planfile = framesdir / "plan.json"
    if planfile.exists() and planfile.read_text() != fingerprint:
        for stale in framesdir.glob("f_*.jpg"):
            stale.unlink()
        (out / "anchors.npz").unlink(missing_ok=True)
        print("sample selection changed; discarded the cached frames and anchors", flush=True)
    planfile.write_text(fingerprint)
    print(
        f"decode backend {backend} ({backend_reason}); {len(plan['samples'])} samples over "
        f"{horizon} frames, times from {plan['timestampSource']}",
        flush=True,
    )

    def keep(source, bgr):
        file = framesdir / f"f_{wanted[source]:06d}.jpg"
        if file.exists():
            return
        if bgr.shape[1] > 1024:
            bgr = cv2.resize(bgr, (1024, round(bgr.shape[0] * 1024 / bgr.shape[1])))
        cv2.imwrite(str(file), bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])

    if backend == DECODE_OPENCV:
        decodable, _ = decode_opencv(cv2, a.video, set(wanted), keep)
    else:
        decodable, _ = decode_ffmpeg(a.video, *source_size, set(wanted), keep)
    plan = clamp_plan(plan, decodable, container, a.fps)
    duration = (container or decodable) / srcfps
    indices = np.array(plan["samples"], int)
    times = np.array(plan["times"], float)
    N = len(indices)
    files = [framesdir / f"f_{i:06d}.jpg" for i in range(N)]
    for stray in sorted(framesdir.glob("f_*.jpg")):
        if stray not in set(files):
            stray.unlink()
    missing = [file.name for file in files if not file.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} sampled frames were never written: {missing[:5]}")
    if plan["droppedTailSamples"]:
        print(
            f"container declares {container} frames, {decodable} decode; dropped "
            f"{plan['droppedTailSamples']} sample(s) past the decodable tail",
            flush=True,
        )
    import torch

    sys.path.insert(0, a.repo)
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_images_as_tensor
    from pi3.utils.geometry import depth_normal_edge

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().cuda()
    model.disable_multimodal()

    def predict(ids):
        for file in temp.glob("*.jpg"):
            file.unlink()
        for idx in ids:
            os.symlink(files[idx].resolve(), temp / files[idx].name)
        imgs = load_images_as_tensor(str(temp), verbose=False)[None].cuda()
        print("inference tensor", tuple(imgs.shape), flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            r = model(imgs=imgs)
        confidence = torch.sigmoid(r["conf"][0, ..., 0])
        conf_np = confidence.float().cpu().numpy()
        # One cut per view, absolute where the view can carry it. A single fixed cut selects
        # every pixel of a bright handheld clip and not one pixel of a broadcast crowd.
        cuts = [confidence_threshold(view) for view in conf_np]
        thresholds = torch.as_tensor(
            [cut for cut, _ in cuts], device=confidence.device, dtype=confidence.dtype
        )
        valid = confidence > thresholds[:, None, None]
        edge = depth_normal_edge(r["local_points"], rtol=0.03, mask=valid[None])[0]
        result = dict(
            points=r["points"][0].float().cpu().numpy(),
            poses=r["camera_poses"][0].float().cpu().numpy(),
            rays=r["rays"][0].float().cpu().numpy(),
            conf=conf_np,
            conf_threshold=np.array([cut for cut, _ in cuts], float),
            conf_mode=np.array([mode for _, mode in cuts]),
            valid=(valid & ~edge).cpu().numpy(),
            rgb=(imgs[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype("uint8"),
        )
        del imgs, r
        torch.cuda.empty_cache()
        return result

    anchors = np.unique(np.linspace(0, N - 1, min(a.anchors, N)).round().astype(int)).tolist()
    basefile = out / "anchors.npz"
    surveyfile = out / "anchor-survey.json"
    # A cached prediction from before this stage measured its own confidence cuts and chose a
    # people mask does not carry what the alignment now reads, and re-deriving it would mean
    # guessing which cut produced its `valid`. Predict again instead.
    needed = ("conf_threshold", "conf_mode", "static_people", "people_mask_mode")
    base = dict(np.load(basefile)) if basefile.exists() else None
    if base is not None and any(key not in base for key in needed):
        print("cached anchors predate the adaptive anchor selection; predicting again", flush=True)
        base = None
    if base is not None and str(base.get("anchor_role", ANCHOR_ROLE_GLOBAL)) != ANCHOR_ROLE_GLOBAL:
        # A chained run leaves its base window in anchors.npz, because that is the prediction the
        # world scale below is measured in. That file is not the survey, so the survey is made
        # again; a file written before this key existed is a survey by construction.
        print("cached anchors are a chained base window; predicting the survey again", flush=True)
        base = None
    if base is None:
        base = predict(anchors)
        semantic = people_masks(base["rgb"], dilate_px=2)
        people_mode, person_fraction = mask_mode(view.mean() for view in semantic)
        base["people"] = semantic
        base["static_people"] = (
            foreground_people_masks(base["rgb"], dilate_px=2)
            if people_mode == MASK_FOREGROUND
            else semantic
        )
        base["people_mask_mode"] = np.array(people_mode)
        base["semantic_person_fraction"] = np.array(person_fraction)
        base["anchor_role"] = np.array(ANCHOR_ROLE_GLOBAL)
        base["anchor_samples"] = np.array(anchors, dtype=np.int64)
        np.savez_compressed(basefile, **base)
    people_mode = str(base["people_mask_mode"])
    person_fraction = float(base["semantic_person_fraction"])
    print(
        f"people mask for this solve: {people_mode} (semantic person fraction "
        f"{person_fraction:.3f} over the anchors, crowd above {CROWD_PERSON_FRACTION})",
        flush=True,
    )
    flip = np.array([1.0, -1.0, -1.0])

    def static_pixels(prediction):
        """The pixels of each view that are scene rather than person, and really predicted."""
        return (
            prediction["valid"]
            & ~prediction["static_people"]
            & np.isfinite(prediction["points"]).all(-1)
        )

    def anchor_depths(prediction, static):
        """Each view's median depth in its own camera, over the static pixels in front of it."""
        measured = []
        for j in range(len(prediction["poses"])):
            inv_j = np.linalg.inv(prediction["poses"][j])
            away = (prediction["points"][j] @ inv_j[:3, :3].T + inv_j[:3, 3])[..., 2]
            ahead = away[static[j] & (away > 0)]
            measured.append(float(np.median(ahead)) if ahead.size else np.nan)
        return measured

    survey_static = static_pixels(base)
    survey_depths = anchor_depths(base, survey_static)
    spread, anchor_depth = measure_spread(base["poses"], survey_depths)
    print(f"anchor spread {spread:.2f} scene depths (median depth {anchor_depth:.2f})", flush=True)
    # The spread is now a decision rather than a refusal. Under the limit one global anchor set
    # spans the clip and the solve below is exactly the solve this stage has always run. Over it,
    # the clip travels further than any single anchor set can relate, and it is solved as a chain
    # of overlapping windows that each pass the same limit on their own.
    mode = ALIGNMENT_GLOBAL if spread <= ANCHOR_SPREAD_LIMIT else ALIGNMENT_CHAINED
    profile = travel_profile(base["poses"], anchors, anchor_depth, N)
    planned = [(0, N - 1)] if mode == ALIGNMENT_GLOBAL else plan_windows(profile, N)
    surveyfile.write_text(
        json.dumps(
            dict(
                alignmentMode=mode,
                surveyAnchors=anchors,
                surveySpreadDepths=spread,
                surveyMedianDepth=anchor_depth,
                surveyAnchorDepths=survey_depths,
                surveyedPathDepths=float(profile[-1]),
                maxAnchorSpreadDepths=ANCHOR_SPREAD_LIMIT,
                chainTargetSpreadFraction=CHAIN_TARGET_SPREAD,
                chainOverlapSamples=CHAIN_OVERLAP_SAMPLES,
                chainMinWindowSamples=CHAIN_MIN_WINDOW,
                plannedWindows=[[int(first), int(last)] for first, last in planned],
                travelProfileDepths=[float(walked) for walked in profile],
                surveyNote=(
                    "The survey is the one global anchor set, predicted once. Its camera centres "
                    "relate views that need not be co-visible, so `travelProfileDepths` is an "
                    "estimate of where the camera went, used only to propose window lengths; "
                    "every window's own anchor spread is measured afterwards and overrides it."
                ),
            ),
            indent=2,
        )
    )
    if mode == ALIGNMENT_CHAINED:
        if a.batch < CHAIN_OVERLAP_SAMPLES:
            raise RuntimeError(
                f"A chained solve joins windows on {CHAIN_OVERLAP_SAMPLES} shared frames, which "
                f"have to reach the next window's first batch; --batch {a.batch} is too small"
            )
        print(
            f"anchor spread is over the {ANCHOR_SPREAD_LIMIT:.1f} limit, so this clip is solved "
            f"as a chain: surveyed camera path {float(profile[-1]):.2f} scene depths over {N} "
            f"samples, planned as {len(planned)} window(s) sharing {CHAIN_OVERLAP_SAMPLES} "
            "frame(s) at each join -- " + ", ".join(f"{first}-{last}" for first, last in planned),
            flush=True,
        )
        # The survey has done its work -- it chose the mode, the people mask and the plan -- and
        # every window predicts its own anchors, so its maps do not stay in memory beside them.
        del base, survey_static, survey_depths
        gc.collect()

    def window_anchors(first, last):
        """One window's own anchor set: predicted, masked and measured like a global one."""
        ids = np.unique(
            np.linspace(first, last, min(a.anchors, last - first + 1)).round().astype(int)
        ).tolist()
        prediction = predict(ids)
        prediction["static_people"] = solve_people_masks(prediction["rgb"], people_mode, 2)
        static = static_pixels(prediction)
        depths = anchor_depths(prediction, static)
        window_spread, depth = measure_spread(prediction["poses"], depths)
        return dict(
            first=int(first),
            last=int(last),
            samples=ids,
            prediction=prediction,
            static=static,
            depths=depths,
            spread=window_spread,
            depth=depth,
        )

    def settle_window(first, budget):
        """Grow or shrink the proposed window until its own anchors are co-visible.

        The survey proposes a length; this measures it. A window that comes in over the limit is
        shrunk by how far over it came, one that leaves most of its budget unused is grown once,
        and whichever acceptable window reached furthest is the one that gets solved.
        """
        end = next_window(profile, first, N - 1, budget)
        best, tried = None, []
        for _ in range(CHAIN_WINDOW_ATTEMPTS):
            got = window_anchors(first, end)
            tried.append(dict(first=int(first), last=int(end), spread=float(got["spread"])))
            print(
                f"window {first}-{end}: anchor spread {got['spread']:.2f} scene depths "
                f"(median depth {got['depth']:.2f})",
                flush=True,
            )
            action, following = adapt_window(first, end, got["spread"], N - 1)
            if got["spread"] <= ANCHOR_SPREAD_LIMIT and (best is None or end > best["last"]):
                best = got
            elif best is not None:
                # A length that already worked is worth more than another attempt at one that
                # did not, and the batches only ever align against an accepted window.
                break
            if action in ("accept", "fail"):
                break
            print(f"window {first}-{end}: {action} to {first}-{following}", flush=True)
            end = following
        if best is None:
            raise RuntimeError(
                f"No window starting at sample {first} (t={times[first]:.3f}s) holds together: "
                + "; ".join(
                    f"{attempt['first']}-{attempt['last']} spans {attempt['spread']:.1f} scene "
                    "depths"
                    for attempt in tried
                )
                + f". Above {ANCHOR_SPREAD_LIMIT:.1f} the views of one window share too little "
                f"content to align on, and this stage will not go below {CHAIN_MIN_WINDOW} "
                "samples. The camera crosses the scene faster than one second of footage can "
                "span, which no chaining can fix: this segment cannot be solved at this rate."
            )
        best["attempts"] = tried
        return best

    def link_source(p, ids, pm, transform, samples):
        """The shared frames' confident static pixels and points, in this window's own frame.

        `anchor_matches` picks the pixels, because the shared frames ARE an anchor set: they
        simply happen to be frames the next window predicts again rather than frames this window
        predicts twice. Only the picked pixels are carried over, so a link costs a few thousand
        points per frame instead of a depth map.
        """
        rows = [ids.index(i) for i in samples]
        static = p["valid"][rows] & ~pm[rows] & np.isfinite(p["points"][rows]).all(-1)
        kept, _ = anchor_matches(p["conf"][rows], static)
        carried = []
        for k, pick in kept:
            row = rows[k]
            points = np.asarray(
                apply_similarity(transform, p["points"][row].reshape(-1, 3)[pick]), dtype=float
            )
            centre = np.asarray(apply_similarity(transform, p["poses"][row][:3, 3]), dtype=float)
            carried.append(
                dict(
                    sample=int(samples[k]),
                    pixels=np.asarray(pick, dtype=np.int64),
                    points=points,
                    centre=centre,
                    depth=float(np.median(np.linalg.norm(points - centre, axis=1))),
                )
            )
        return carried

    def register_link(incoming, p, ids, pm, transform, index, remaining):
        """Fit this window onto the previous one from the frames they both predicted.

        One physical frame predicted twice gives exact pixel-for-pixel 3D correspondences on the
        pixels the previous window already judged confident and static, so a link needs no
        matching and no descriptors; `anchor_matches` and `similarity` carry the same guards they
        carry for an anchor set. The shared frames' camera centres are deliberately left out of
        the fit, which leaves them free to check it afterwards.
        """
        src, dst, here, there, depth_here, depth_there, shared, dropped = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for frame in incoming["frames"]:
            i = frame["sample"]
            if i not in ids:
                dropped.append(f"{i} was not predicted again")
                continue
            j = ids.index(i)
            alive = (p["valid"][j] & ~pm[j] & np.isfinite(p["points"][j]).all(-1)).reshape(-1)[
                frame["pixels"]
            ]
            if int(alive.sum()) < ANCHOR_MATCH_MIN:
                dropped.append(f"{i} kept {int(alive.sum())} of {alive.size} shared pixels")
                continue
            mine = np.asarray(
                apply_similarity(transform, p["points"][j].reshape(-1, 3)[frame["pixels"][alive]]),
                dtype=float,
            )
            centre = np.asarray(apply_similarity(transform, p["poses"][j][:3, 3]), dtype=float)
            src.append(mine)
            dst.append(frame["points"][alive])
            here.append(centre)
            there.append(frame["centre"])
            depth_here.append(float(np.median(np.linalg.norm(mine - centre, axis=1))))
            depth_there.append(frame["depth"])
            shared.append(i)
        if len(src) < MIN_USABLE_ANCHORS:
            raise RuntimeError(
                f"only {len(src)} of the {len(incoming['frames'])} shared frame(s) survived in "
                f"both windows, and a link needs {MIN_USABLE_ANCHORS}"
                + (f" ({'; '.join(dropped)})" if dropped else "")
            )
        scale_link, rotation_link, offset_link, rms = similarity(
            np.concatenate(src), np.concatenate(dst)
        )
        moved = apply_similarity(
            (scale_link, rotation_link, offset_link), np.asarray(here, dtype=float)
        )
        camera_rms = float(
            np.sqrt(np.mean(np.sum((moved - np.asarray(there, dtype=float)) ** 2, axis=1)))
        )
        record = link_quality(
            index,
            shared,
            sum(len(points) for points in src),
            rms,
            scale_link,
            float(np.median(depth_there)),
            float(np.median(depth_here)),
            camera_rms,
            remaining_path=remaining,
        )
        record["droppedSharedFrames"] = dropped
        return (scale_link, rotation_link, offset_link), record

    def chain_break(solved, fitted_links):
        """Why the chain broke, and the sample ranges an orchestrator can trim to."""
        found = localise_break(solved, fitted_links)
        weakest = found["weakestLink"] or {}
        prefix = found["registeredPrefix"]
        tail = solved[-1][0]
        return (
            f"Chained alignment broke at link {weakest.get('link', len(fitted_links) - 1)}, which "
            f"joins window {len(fitted_links) - 1} to window {len(fitted_links)} on shared "
            f"samples {weakest.get('sharedSamples', [])}: "
            + "; ".join(weakest.get("failures", ["the link could not be fitted at all"]))
            + f". The chain did register samples {prefix['firstSample']}-{prefix['lastSample']} "
            f"(t={times[prefix['firstSample']]:.3f}-{times[prefix['lastSample']]:.3f}s, "
            f"{prefix['samples']} of {N} samples over window(s) {prefix['windows'][0]}-"
            f"{prefix['windows'][1]}), so re-running on that range keeps the part that solved. "
            f"Samples {tail}-{N - 1} (t={times[tail]:.3f}-{times[N - 1]:.3f}s) were never "
            "registered to it and would have to be solved as a separate segment."
        )

    cams = [None] * N
    log = []
    conf_modes = []
    point_counts = []
    gaps = []
    intrinsic_gaps = []
    home, home_frame = None, None
    camera_dir = out / "camera-batches"
    camera_dir.mkdir(exist_ok=True)
    window_records, links, solved = [], [], []
    budget = CHAIN_TARGET_SPREAD * ANCHOR_SPREAD_LIMIT
    to_base = identity_similarity()
    handover = None
    R0, t00, scale = None, None, np.nan
    first, index = 0, 0
    while True:
        if mode == ALIGNMENT_GLOBAL:
            window = dict(
                first=0,
                last=N - 1,
                samples=anchors,
                prediction=base,
                static=survey_static,
                depths=survey_depths,
                spread=spread,
                depth=anchor_depth,
                attempts=[],
            )
        else:
            window = settle_window(first, budget)
        wfirst, wlast = window["first"], window["last"]
        wanchors, wpred, wdepth = window["samples"], window["prediction"], window["depth"]
        owned = wfirst if index == 0 else wfirst + CHAIN_OVERLAP_SAMPLES
        solved.append((wfirst, wlast))
        if index == 0:
            # The base frame: camera 0 at the origin, world scale and gravity levelling from this
            # window's view 0, exactly as the global solve takes them from its anchor view 0.
            inv = np.linalg.inv(wpred["poses"][0])
            R0, t00 = inv[:3, :3], inv[:3, 3]
            zz = (wpred["points"][0] @ R0.T + t00)[..., 2]
            good = window["static"][0] & (zz > 0)
            scale = 3 / np.median(zz[good]) if int(good.sum()) >= ANCHOR_MATCH_MIN else np.nan
            if not np.isfinite(scale) or scale <= 0:
                # A genuine solve failure: the anchor view has no confident scene geometry in
                # front of the camera at all. Distinct from a frame with nobody in it.
                raise RuntimeError(
                    f"No valid static scene points in anchor view 0 ({int(good.sum())} of "
                    f"{int(window['static'][0].sum())} static pixels lie in front of the camera; "
                    f"this stage needs {ANCHOR_MATCH_MIN}), under the "
                    f"{str(wpred['conf_mode'][0])} confidence cut "
                    f"{float(wpred['conf_threshold'][0]):.3f} and the {people_mode} people mask"
                )
            if mode == ALIGNMENT_CHAINED:
                # anchors.npz is whichever prediction the base frame is measured in, so a chained
                # solve stores window 0's there: `scripts/shot_cuts.py --poses` reads its view 0
                # for the scene's own depth and has to read the prediction the world scale above
                # came from. The survey stays in anchor-survey.json.
                stored = dict(wpred)
                stored["people"] = people_masks(wpred["rgb"], dilate_px=2)
                stored["people_mask_mode"] = np.array(people_mode)
                stored["semantic_person_fraction"] = np.array(person_fraction)
                stored["anchor_role"] = np.array(ANCHOR_ROLE_WINDOW)
                stored["anchor_samples"] = np.array(wanchors, dtype=np.int64)
                np.savez_compressed(basefile, **stored)
                del stored
        usable, anchor_records = anchor_matches(
            wpred["conf"],
            window["static"],
            preferred=np.array([str(cut) == CONF_MODE_ABSOLUTE for cut in wpred["conf_mode"]]),
        )
        for record in anchor_records:
            if "dropped" in record:
                print(f"anchor view {record['anchor']} dropped: {record['dropped']}", flush=True)
        print(
            f"{len(usable)} of {len(wanchors)} anchor views carry the alignment, "
            f"{sum(len(pick) for _, pick in usable)} matched points",
            flush=True,
        )
        share = [] if wlast >= N - 1 else list(range(wlast - CHAIN_OVERLAP_SAMPLES + 1, wlast + 1))
        incoming, handover = handover, None
        window_to_base = to_base
        for lo in range(wfirst, wlast + 1, a.batch):
            chunk = list(range(lo, min(wlast + 1, lo + a.batch)))
            ids = sorted(set(wanchors + chunk))
            print("Pi3X batch", lo, "/", N, "views", len(ids), flush=True)
            p = predict(ids)
            src = np.concatenate(
                [p["points"][ids.index(wanchors[j])].reshape(-1, 3)[pick] for j, pick in usable]
            )
            dst = np.concatenate([wpred["points"][j].reshape(-1, 3)[pick] for j, pick in usable])
            s, R, t, rms = similarity(src, dst)
            print("alignment", float(s), rms, f"({len(src)} matched points)", flush=True)
            if not np.isfinite(rms) or not 0.1 < s < 10:
                raise RuntimeError(
                    f"Invalid anchor alignment at batch {lo}: scale {float(s):.4g}, residual {rms} "
                    f"over {len(src)} matched points from {len(usable)} anchor view(s)"
                )
            if rms > MAX_ALIGNMENT_RESIDUAL * wdepth:
                # The anchors registered, but not onto each other: at this residual the batch's
                # copy of an anchor view is a different reconstruction of that moment, which is
                # what happens when the anchor set spans content no single prediction can relate.
                raise RuntimeError(
                    f"Anchor alignment residual {rms:.3f} at batch {lo} is "
                    f"{rms / wdepth:.0%} of the {wdepth:.2f} median anchor depth, over "
                    f"the {MAX_ALIGNMENT_RESIDUAL:.0%} this stage accepts; the batch and the "
                    "anchor prediction do not agree on the same scene. Select a shorter segment."
                )
            # Full-resolution masks preserve limbs; resize to the actual Pi3X pixel grid.
            full = np.stack([np.asarray(Image.open(files[i])) for i in ids])
            pm = solve_people_masks(full, people_mode, 1)
            H, W = p["rgb"].shape[1:3]
            pm = np.stack(
                [
                    cv2.resize(m.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    for m in pm
                ]
            )
            if incoming is not None:
                try:
                    joint, record = register_link(
                        incoming,
                        p,
                        ids,
                        pm,
                        (s, R, t),
                        len(links),
                        float(profile[-1] - profile[wfirst]),
                    )
                except RuntimeError as error:
                    joint, record = (
                        None,
                        dict(
                            link=len(links),
                            sharedSamples=[frame["sample"] for frame in incoming["frames"]],
                            ok=False,
                            failures=[str(error)],
                        ),
                    )
                links.append(record)
                incoming = None
                if not record["ok"]:
                    raise RuntimeError(chain_break(solved, links))
                window_to_base = compose_similarity(to_base, joint)
                print(
                    f"link {record['link']} joins window {index - 1} to {index} on samples "
                    f"{record['sharedSamples']}: residual {record['rmsDepths']:.1%} of a scene "
                    f"depth, camera centres {record['cameraRmsDepths']:.1%}, depth ratio "
                    f"{record['depthRatio']:.3f} ({record['points']} points)",
                    flush=True,
                )
            composed = compose_similarity(window_to_base, (s, R, t))
            outgoing = [i for i in share if i in chunk]
            if outgoing:
                try:
                    carried = link_source(p, ids, pm, (s, R, t), outgoing)
                except RuntimeError as error:
                    # `anchor_matches` refusing these frames is the same refusal it makes of an
                    # anchor set, and it means the same thing: nothing confident and static to
                    # register on. The window still finishes; the chain stops after it, named.
                    print(f"shared frames {outgoing} carry no link: {error}", flush=True)
                    carried = []
                handover = handover or dict(frames=[], window=index)
                handover["frames"].extend(carried)
            conf_modes.extend(str(p["conf_mode"][ids.index(i)]) for i in chunk if i >= owned)
            np.savez_compressed(
                camera_dir
                / (
                    f"batch_{lo:03d}.npz"
                    if mode == ALIGNMENT_GLOBAL
                    else f"batch_w{index:02d}_{lo:03d}.npz"
                ),
                sample_ids=ids,
                source_indices=indices[ids],
                raw_opencv_camera_to_world=p["poses"],
                batch_rotation=composed[1],
                batch_translation=composed[2],
                batch_scale=composed[0],
                reference_rotation=R0,
                reference_translation=t00,
                world_scale=scale,
                alignment_mode=mode,
                window=index,
                window_first_sample=wfirst,
                window_last_sample=wlast,
                window_scale=window_to_base[0],
                window_rotation=window_to_base[1],
                window_translation=window_to_base[2],
                local_scale=s,
                local_rotation=R,
                local_translation=t,
            )
            for i in chunk:
                if i < owned:
                    # A shared frame, already exported by the window this one registers onto.
                    # It is predicted again only to carry the link.
                    continue
                j = ids.index(i)
                valid = p["valid"][j] & pm[j] & np.isfinite(p["points"][j]).all(-1)
                xyz = apply_similarity(composed, p["points"][j][valid])
                col = p["rgb"][j][valid]
                xyz = (xyz @ R0.T + t00) * scale * flip
                present = person_present(len(xyz))
                if present:
                    write_point_ply(out / f"frame_{i:03d}.ply", xyz.astype("float32"), col)
                else:
                    # Nobody in this frame. The camera below is still a measured pose, so the
                    # solve continues; only the person ply is withheld. A header-only ply would
                    # read as "person depth present" to the readers that check for these files,
                    # so the gap is a missing file plus an explicit record.
                    (out / f"frame_{i:03d}.ply").unlink(missing_ok=True)
                    gaps.append(person_gap(i, indices[i], times[i], len(xyz)))
                    print(f"no person at {times[i]:.3f}s ({len(xyz)} points)", flush=True)
                point_counts.append(len(xyz))
                camera = viewer_camera(p["poses"][j], *composed, R0, t00, scale)
                try:
                    optics = fit_ray_intrinsics(p["rays"][j], p["valid"][j], source_size)
                    optics["intrinsicsSource"] = "fit"
                except ValueError as error:
                    # One frame's rays are not the clip's camera. A dark, blurred or transition
                    # frame can starve the fit; the pose is still measured, so the frame is
                    # recorded here and filled from the clip median once every batch has run.
                    optics = dict(intrinsicsSource="clip-median")
                    intrinsic_gaps.append(
                        dict(
                            sample=int(i),
                            sourceIndex=int(indices[i]),
                            time=float(times[i]),
                            reason=str(error),
                        )
                    )
                    print(f"no intrinsic fit at {times[i]:.3f}s ({error})", flush=True)
                cams[i] = dict(
                    position=camera[:3, 3].tolist(),
                    time=float(times[i]),
                    sourceIndex=int(indices[i]),
                    camera_to_world=camera.tolist(),
                    world_to_camera=np.linalg.inv(camera).tolist(),
                    **optics,
                )
                if i == 0:
                    np.savez_compressed(
                        out / "source-camera-0.npz",
                        rays=p["rays"][j],
                        rgb=p["rgb"][j],
                        valid=p["valid"][j],
                        camera_to_world=camera,
                    )
                if present and home is None:
                    # Frame 0 when it has a person, as before; the first frame that does
                    # otherwise, because a clip may open before anyone walks in.
                    home, home_frame = home_distance(xyz), i
            log.append(
                dict(
                    first=lo,
                    frames=len(chunk),
                    scale=float(s),
                    rotation=R.tolist(),
                    translation=t.tolist(),
                    rms=rms,
                    window=index,
                    windowFirstSample=wfirst,
                    windowLastSample=wlast,
                    exported=sum(1 for i in chunk if i >= owned),
                    windowScale=float(window_to_base[0]),
                    windowRotation=np.asarray(window_to_base[1]).tolist(),
                    windowTranslation=np.asarray(window_to_base[2]).tolist(),
                )
            )
            (out / "alignment-batches.json").write_text(json.dumps(log, indent=2))
            del p, pm, full
            gc.collect()
            torch.cuda.empty_cache()
        window_records.append(
            dict(
                window=index,
                firstSample=int(wfirst),
                lastSample=int(wlast),
                ownedFirstSample=int(owned),
                ownedLastSample=int(wlast),
                firstTime=float(times[wfirst]),
                lastTime=float(times[wlast]),
                anchors=[int(i) for i in wanchors],
                anchorSpreadDepths=float(window["spread"]),
                anchorMedianDepth=float(wdepth),
                anchorsUsed=[int(wanchors[j]) for j, _ in usable],
                anchorMatchPoints=int(sum(len(pick) for _, pick in usable)),
                anchorSelection=anchor_records,
                anchorConfidenceThresholds=[float(cut) for cut in wpred["conf_threshold"]],
                anchorConfidenceModes=[str(cut) for cut in wpred["conf_mode"]],
                lengthAttempts=window["attempts"],
                scale=float(window_to_base[0]),
                rotation=np.asarray(window_to_base[1]).tolist(),
                translation=np.asarray(window_to_base[2]).tolist(),
            )
        )
        to_base = window_to_base
        del window, wpred
        gc.collect()
        if wlast >= N - 1:
            break
        if not (handover and len(handover["frames"]) >= MIN_USABLE_ANCHORS):
            raise RuntimeError(
                f"Window {index} (samples {wfirst}-{wlast}) left "
                f"{len(handover['frames']) if handover else 0} of its "
                f"{CHAIN_OVERLAP_SAMPLES} shared frame(s) usable, and a link needs "
                f"{MIN_USABLE_ANCHORS}; a chained solve cannot continue past it. Samples "
                f"0-{wlast} (t={times[0]:.3f}-{times[wlast]:.3f}s) did solve and can be re-run "
                "as their own segment."
            )
        following = wlast - CHAIN_OVERLAP_SAMPLES + 1
        if following <= wfirst:
            raise RuntimeError(
                f"Window {index} is only {wlast - wfirst + 1} sample(s) long and cannot carry a "
                f"{CHAIN_OVERLAP_SAMPLES}-sample overlap into the next window"
            )
        first, index = following, index + 1
    unsolved = [i for i in range(N) if cams[i] is None]
    if unsolved:
        raise RuntimeError(
            f"{len(unsolved)} sample(s) were never exported by any window: {unsolved[:8]}"
        )
    chain = localise_break([(w["firstSample"], w["lastSample"]) for w in window_records], links)
    drift = drift_bound(links)
    if mode == ALIGNMENT_CHAINED:
        print(
            f"chain closed: {len(window_records)} windows, {len(links)} link(s), worst link "
            f"residual {max((link['rmsDepths'] for link in links), default=0.0):.1%} of a scene "
            f"depth, accumulated drift bound {drift['boundDepths']:.2f} scene depths",
            flush=True,
        )
    fitted = [c for c in cams if c and c["intrinsicsSource"] == "fit"]
    fit_fraction = require_intrinsic_fits(len(fitted), N)
    if intrinsic_gaps:
        fallback = median_intrinsics(fitted)
        for record in intrinsic_gaps:
            cams[record["sample"]].update(fallback)
        print(f"filled {len(intrinsic_gaps)} frames from the clip median K", flush=True)
    meta = dict(
        backend="pi3x-dense-anchors",
        frames=[f"frame_{i:03d}.ply" for i in range(N)],
        count=N,
        fps=a.fps,
        timestamps=times.tolist(),
        duration=duration,
        sourceFps=srcfps,
        sourceIndices=indices.tolist(),
        people_only=True,
        home_distance=home,
        homeDistanceFrame=home_frame,
        decodeBackend=backend,
        decodeBackendReason=backend_reason,
        sourceCodec=probe.get("codec", ""),
        sourceRotationDegrees=int(probe.get("rotation") or 0),
        containerFrames=plan["containerFrames"],
        decodableFrames=plan["decodableFrames"],
        droppedTailSamples=plan["droppedTailSamples"],
        emptyOutputSlots=plan["emptyOutputSlots"],
        outputSlotFill=plan["outputSlotFill"],
        sampleSelection=plan["sampleSelection"],
        timestampSource=plan["timestampSource"],
        decodedSpanSeconds=float(times[-1] - times[0]),
        decodeNote=(
            "Frames were decoded in order, never seeked to. `decodableFrames` is what actually "
            "decoded; `containerFrames` is what the container declares, and any difference is "
            "the tail dropped in `droppedTailSamples`. `timestamps` are relative to the first "
            f"decoded frame and come from {plan['timestampSource']}. Any container display "
            "rotation is left unapplied, as it always has been; each camera's "
            "`source_image_size` describes the coded pixels the intrinsics were fitted to."
        ),
        intrinsicsFallbackFrames=intrinsic_gaps,
        intrinsicsFallbackCount=len(intrinsic_gaps),
        intrinsicsFitCount=len(fitted),
        intrinsicsFitFraction=fit_fraction,
        minIntrinsicFitFraction=MIN_INTRINSIC_FIT_FRACTION,
        intrinsicsNote=(
            "Cameras carry `intrinsicsSource`: `fit` is this frame's own rays, `clip-median` is "
            "the per-parameter median of the frames that did fit, used where a frame had too "
            "few valid rays. Poses are measured in both cases."
        ),
        **person_gap_summary(gaps, N),
        sourceSha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),
        seconds=time.time() - t0,
        peakVRAMGB=torch.cuda.max_memory_allocated() / 1e9,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        anchors=window_records[0]["anchors"],
        anchorsUsed=window_records[0]["anchorsUsed"],
        anchorSelection=window_records[0]["anchorSelection"],
        anchorMatchPoints=window_records[0]["anchorMatchPoints"],
        anchorSpreadDepths=window_records[0]["anchorSpreadDepths"],
        anchorMedianDepth=window_records[0]["anchorMedianDepth"],
        maxAnchorSpreadDepths=ANCHOR_SPREAD_LIMIT,
        alignmentMode=mode,
        alignmentWindowCount=len(window_records),
        alignmentWindows=window_records,
        alignmentLinks=links,
        chainDrift=drift,
        chainBreakCheck=chain,
        surveyAnchors=anchors,
        surveySpreadDepths=spread,
        surveyMedianDepth=anchor_depth,
        surveyedPathDepths=float(profile[-1]),
        chainOverlapSamples=CHAIN_OVERLAP_SAMPLES,
        chainTargetSpreadFraction=CHAIN_TARGET_SPREAD,
        chainMinWindowSamples=CHAIN_MIN_WINDOW,
        maxLinkResidual=MAX_LINK_RESIDUAL,
        maxLinkCameraResidual=MAX_LINK_CAMERA_RESIDUAL,
        linkDepthRatioBand=LINK_DEPTH_RATIO_BAND,
        alignmentNote=(
            f"`{ALIGNMENT_GLOBAL}` is one anchor set shared by every batch, which needs all of "
            f"its views to be co-visible. `{ALIGNMENT_CHAINED}` is used instead when those views "
            f"span more than `maxAnchorSpreadDepths`: the clip is split into the overlapping "
            "`alignmentWindows`, each solved exactly like a global solve against its own anchors, "
            "and `alignmentLinks` registers each window onto the previous one with a similarity "
            "fitted to the frames they both predicted. Every pose and point here is in window 0's "
            "frame; the top-level `anchors*` fields describe that window, and `survey*` describes "
            "the single global anchor set that chose the mode. A chained solve never sees both "
            "ends of the clip at once, so `chainDrift` is a bound computed from the link "
            "residuals and not an observation of drift, and each window's geometry is only as "
            "related to window 0's as the links between them."
        ),
        peopleMaskMode=people_mode,
        semanticPersonFraction=person_fraction,
        crowdPersonFraction=CROWD_PERSON_FRACTION,
        peopleMaskNote=(
            "`peopleMaskMode` is the mask this solve excluded from the static scene AND "
            f"exported as the person. `{MASK_SEMANTIC}` is every person-coloured pixel; "
            f"`{MASK_FOREGROUND}` is Mask R-CNN's reconstructable subjects, chosen when the "
            "semantic person share of the anchor views exceeds `crowdPersonFraction`, because "
            "a stadium crowd is neither scene to exclude nor a person to export."
        ),
        anchorConfidenceThresholds=window_records[0]["anchorConfidenceThresholds"],
        anchorConfidenceModes=window_records[0]["anchorConfidenceModes"],
        frameConfidenceModes=conf_modes,
        frameRankConfidenceCount=sum(1 for mode in conf_modes if mode == CONF_MODE_RANK),
        confidenceNote=(
            f"Pi3X confidence is not calibrated across content, so each view's cut is "
            f"`{CONF_MODE_ABSOLUTE}` ({CONF_ABSOLUTE}) where at least "
            f"{CONF_MIN_FRACTION:.0%} of the view clears it and `{CONF_MODE_RANK}` (the view's "
            f"own {CONF_QUANTILE:.0%} quantile, floored at {CONF_FLOOR}) where it does not. "
            "A rank-mode frame's geometry is the most confident part of a map the model was "
            "unsure of throughout, not confident geometry."
        ),
        batch=a.batch,
        pointCounts=point_counts,
        note="Each source frame reconstructed with shared static anchors. >=12 independent frames/sec; single observed-facing surface, not complete human volume.",
    )
    (out / "sequence.json").write_text(json.dumps(meta, indent=2))
    (out / "cameras.json").write_text(
        json.dumps(
            dict(
                coordinates="Both world and camera bases are OpenGL: x right, y up, -z forward. World matches exported PLY. Target camera zero is not assumed identity after batch registration.",
                reference=dict(
                    rotation=R0.tolist(),
                    translation=t00.tolist(),
                    scale=float(scale),
                    world_axis_flip=flip.tolist(),
                ),
                cameras=cams,
            ),
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in meta.items()
                if k not in ("frames", "personFrames", "timestamps", "sourceIndices")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
