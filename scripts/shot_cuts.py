#!/usr/bin/env python3
"""Is this clip one continuous shot? Answer before the pipeline spends anything on it.

Every stage after this one assumes continuous motion. A cut breaks that assumption silently: the
camera solve fits one trajectory through both shots, so the cut comes back as a teleport (Diagon
Alley, share/MOVIE-CLIP-STATUS.md: 0.95-4.16 units in 83 ms, every jump on a batch seam), person
tracking swaps identity across it, and two places get fused into one world. Nothing errors.

Two checks live here, one before the spend and one after the solve:

  --video FILE   cut detection and shot choice. ffmpeg's own scene score plus a feature match
                 across each candidate, so it costs one decode pass and no GPU. Multi-shot clips are then scored on the criteria this project
                 already selects shots by -- camera translation (parallax_probe), one clear
                 person, full body, person pixel height, duration -- and the best one is named.

  --poses DIR    the teleport guard, run on a pi3x output directory after the solve. Catches a cut
                 too soft for the scene score, and solve failure generally.

  python3 scripts/shot_cuts.py --video public/clips/x.mp4 --json cuts.json
  python3 scripts/shot_cuts.py --poses .context/run/x/pi3x            # 3 = teleport, 4 = unchecked
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ffmpeg's scene score is how much of the frame changed against the one before it. Measured over 16
# clips this project has reconstructed (6 phone clips, 10 film shots): inside a continuous shot it
# never exceeds 0.123, and that worst case is 1960s film grain. A hard cut between two locations
# reads 0.4-0.9 -- but a cut between two shots of the SAME place reads 0.15-0.32, which overlaps
# grain, so the score alone cannot decide those. Hence two stages:
#
#   score > SCENE_CERTAIN            a cut, no further work
#   score > SCENE_CANDIDATE          a candidate; confirmed by matching the frames either side
#
# The confirmation is the separation that matters. ORB features across a real cut match almost
# nothing (0.00-0.34 of the weaker frame's keypoints, measured on spliced clips and 58 candidates
# in a Diagon Alley reel); across grain or a fast pan they still match 0.52-0.74. On that reel the
# two stages find 42 of PySceneDetect's 51 boundaries with no false positive, where a bare 0.30
# threshold finds 13.
SCENE_CANDIDATE = 0.10
SCENE_CERTAIN = 0.35
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
    s = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames,duration", "-show_entries",
         "format=duration", "-of", "json", str(video)],
        check=True, stdout=subprocess.PIPE).stdout)
    st = s["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    dur = float(st.get("duration") or s.get("format", {}).get("duration") or 0)
    return dict(width=int(st["width"]), height=int(st["height"]), fps=float(num) / float(den),
                frames=int(st.get("nb_frames") or 0), duration=dur)


# ---------------------------------------------------------------- cut detection
def scene_scores(video: Path, threshold: float, width: int = 320) -> list[tuple[float, float]]:
    """(time, scene score) for every frame scoring above `threshold`.

    Decoded at `width`: the score is a mean absolute difference and is scale-stable, and a 320 px
    pass is several times cheaper than a full-resolution one.
    """
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-sn",
         "-vf", f"scale={width}:-2,select='gt(scene,{threshold})',metadata=print:file=-",
         "-f", "null", "-"], check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out, t = [], None
    for line in p.stdout.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and t is not None:
            # Frame 0 has no predecessor and ffmpeg scores it 0 or 1 depending on the decoder; it
            # is the start of the clip, never a cut inside it.
            if t > 0.05:
                out.append((t, float(m.group(1))))
            t = None
    return out


def _gray(video: Path, t: float, width: int = 320):
    import cv2, numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.4f}", "-i", str(video),
                          "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe",
                          "-vcodec", "png", "-"], check=True, stdout=subprocess.PIPE).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)


def match_across(video: Path, t: float, fps: float) -> float | None:
    """Fraction of ORB keypoints that survive from the frame before `t` to the frame at `t`.

    Content continues across a pan, a whip or a grainy print, so most features find their pair.
    Across a cut there is nothing to pair with, whatever the two shots look like as a whole.
    """
    import cv2
    a, b = _gray(video, t - 1.5 / max(fps, 1e-3)), _gray(video, t)
    if a is None or b is None:
        return None
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None                              # too featureless to judge; falls back to the score
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    return len([x for x in m if x.distance < 48]) / min(len(ka), len(kb))


def detect_cuts(video: Path, threshold: float = SCENE_CANDIDATE, fps: float | None = None,
                certain: float = SCENE_CERTAIN) -> list[dict]:
    """Every candidate frame, each marked cut or not. See the threshold comment at the top.

    The rejected candidates are returned too: a frame that scored 0.30 and was cleared by the
    feature match is worth printing, because it is where the operator should look if the run
    later trips the teleport guard.
    """
    fps = fps or ffprobe(video)["fps"]
    out = []
    for t, score in scene_scores(video, threshold):
        rec = dict(time=round(t, 3), score=round(score, 4), match=None, cut=True, why="scene score")
        if score <= certain:
            try:
                rec["match"] = match_across(video, t, fps)
            except Exception:
                rec["match"] = None
            m = rec["match"]
            rec["match"] = None if m is None else round(m, 3)
            rec["cut"] = m is not None and m < CUT_MATCH_MAX
            rec["why"] = ("scene score confirmed by feature match" if rec["cut"] else
                          f"content carries across it ({rec['match']} of features match)"
                          if m is not None else "too featureless to confirm; not called a cut")
        out.append(rec)
    return out


def shots_from_cuts(cuts: list[dict], duration: float, min_seconds: float) -> list[dict]:
    bounds = [0.0] + [c["time"] for c in cuts] + [duration]
    shots = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        shots.append(dict(index=i, start=round(a, 3), end=round(b, 3), seconds=round(b - a, 3),
                          tooShort=(b - a) < min_seconds))
    return shots


# ---------------------------------------------------------------- shot scoring
def extract(video: Path, shot: dict, out: Path, n: int, width: int) -> list[Path]:
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    fps = n / max(shot["seconds"], 1e-3)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{shot['start']:.3f}",
                    "-t", f"{shot['seconds']:.3f}", "-i", str(video),
                    "-vf", f"fps={fps:.4f},scale={width}:-2", "-q:v", "3", str(out / "f_%04d.jpg")],
                   check=True)
    return sorted(out.glob("*.jpg"))


def person_stats(frames: list[Path], n: int = 3) -> dict:
    """People in this shot, from the same Mask R-CNN the person stages use.

    Returns the median person count over `n` sampled frames, the tallest person's height as a
    fraction of frame height, and whether that person's box clears the frame edges (full body).
    """
    import numpy as np
    try:
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    except Exception as e:                      # scoring degrades, it does not fail
        return dict(available=False, reason=f"{type(e).__name__}: {e}")
    from PIL import Image
    net = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    pick = [frames[round(k * (len(frames) - 1) / max(n - 1, 1))] for k in range(min(n, len(frames)))]
    counts, heights, full = [], [], []
    for f in pick:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        h = rgb.shape[0]
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            r = net([x])[0]
        boxes = [[float(v) for v in r["boxes"][i].numpy()] for i in range(len(r["labels"]))
                 if int(r["labels"][i]) == 1 and float(r["scores"][i]) > 0.7]
        counts.append(len(boxes))
        if boxes:
            x0, y0, x1, y1 = max(boxes, key=lambda b: b[3] - b[1])
            heights.append((y1 - y0) / h)
            # Full body = the box does not run off the bottom or the top of the frame. A subject
            # cropped at the waist gives LHM nothing to build legs from.
            full.append(y0 > 0.01 * h and y1 < 0.99 * h)
    counts.sort(); heights.sort()
    return dict(available=True, frames=len(pick), peopleMedian=counts[len(counts) // 2] if counts else 0,
                peopleMax=max(counts) if counts else 0,
                personHeightFraction=round(heights[len(heights) // 2], 3) if heights else 0.0,
                fullBody=bool(full and sum(full) > len(full) / 2))


def camera_stats(frames_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "worker/experiments"))
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
        par.get("verdict"), 0.3)
    why.append(f"camera {par.get('verdict', '?')}")
    if per.get("available"):
        n = per["peopleMedian"]
        one = 1.0 if n == 1 else (0.45 if n == 2 else 0.0 if n == 0 else 0.2)
        body = 1.0 if per["fullBody"] else 0.0
        # Pixel height: 0.25 of the frame is about the smallest the avatar chain has worked from,
        # 0.7 is a comfortable full figure. Taller than that is a close-up and scores no higher.
        px = min(max((per["personHeightFraction"] - 0.25) / 0.45, 0.0), 1.0)
        why.append(f"{n} person(s), {'full body' if per['fullBody'] else 'cropped'}, "
                   f"{per['personHeightFraction']:.2f} of frame height")
    else:
        one = body = px = 0.5
        why.append("no person detector available")
    # Duration: 4 s is about the least the solve has held together on, 12 s is plenty.
    dur = min(max((shot["seconds"] - 2.0) / 10.0, 0.0), 1.0)
    why.append(f"{shot['seconds']:.1f} s")
    score = 0.35 * translation + 0.25 * one + 0.20 * body + 0.10 * px + 0.10 * dur
    return round(score, 4), why


def score_shots(video: Path, shots: list[dict], work: Path, frames: int = 16,
                width: int = 640, people: bool = True) -> None:
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


def cut_report(video: Path, threshold: float = SCENE_CANDIDATE,
               min_seconds: float = MIN_SHOT_SECONDS, work: Path | None = None,
               score: bool = True, people: bool = True) -> dict:
    info = ffprobe(video)
    candidates = detect_cuts(video, threshold, info["fps"])
    cuts = [c for c in candidates if c["cut"]]
    shots = shots_from_cuts(cuts, info["duration"], min_seconds)
    doc = dict(video=str(video), source=info, threshold=threshold, minSeconds=min_seconds,
               cuts=cuts, cutCount=len(cuts), cleared=[c for c in candidates if not c["cut"]],
               shots=shots, shotCount=len(shots), continuous=not cuts)
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
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.4f}", "-t", f"{dur:.4f}",
                    "-i", str(video), "-an", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
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
    speed = steps / dt / depth                       # scene depths per second
    typical = max(float(np.median(steps)), 1e-9)
    spike = steps / typical
    i = int(np.argmax(speed))
    worst = dict(afterCamera=i, frame=cams[i + 1].get("sourceIndex", i + 1),
                 timeSeconds=round(float(T[i + 1]), 3), stepUnits=round(float(steps[i]), 4),
                 sceneDepthUnits=round(depth, 3), depthsPerSecond=round(float(speed[i]), 2),
                 spikeOverMedian=round(float(spike[i]), 2))
    over = [int(k) for k in np.where((speed >= TELEPORT_DEPTHS_PER_SEC) & (spike >= TELEPORT_SPIKE))[0]]
    bad = bool(over)
    return dict(ok=not bad, cameras=len(cams), medianStepUnits=round(typical, 4),
                worst=worst, teleports=len(over),
                teleportFrames=[cams[k + 1].get("sourceIndex", k + 1) for k in over[:12]],
                limits=dict(depthsPerSecond=TELEPORT_DEPTHS_PER_SEC, spikeOverMedian=TELEPORT_SPIKE),
                message=None if not bad else (
                    f"camera teleport at frame {worst['frame']} (t={worst['timeSeconds']:.2f}s): it "
                    f"moves {worst['stepUnits']:.3f} units in {round(float(dt[i]), 3)} s through a "
                    f"scene {depth:.2f} units deep -- {worst['depthsPerSecond']:.1f} scene depths per "
                    f"second, {worst['spikeOverMedian']:.0f}x the median step. No camera did "
                    f"that ({len(over)} step(s) like it). Either there is a cut the scene-score detector missed at that frame, or "
                    f"the solve failed to register across it. Trim to one continuous shot "
                    f"(scripts/shot_cuts.py --video <clip>) or re-solve; do not place people "
                    f"against these poses."))


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--poses", type=Path, help="a pi3x output directory (cameras.json + anchors.npz)")
    ap.add_argument("--threshold", type=float, default=SCENE_CANDIDATE,
                    help="scene score above which a frame is a candidate cut")
    ap.add_argument("--min-seconds", type=float, default=MIN_SHOT_SECONDS)
    ap.add_argument("--work", type=Path, help="keep the per-shot frames here instead of a temp dir")
    ap.add_argument("--no-score", action="store_true", help="detect cuts only, do not rank the shots")
    ap.add_argument("--no-people", action="store_true", help="skip the Mask R-CNN pass")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--report", type=Path, help="with --trim: an earlier --json report to trim from")
    ap.add_argument("--trim", type=int, metavar="INDEX", help="write shot INDEX of --report to --out")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.trim is not None:
        if not (a.report and a.out):
            sys.exit("--trim needs --report and --out")
        doc = json.loads(a.report.read_text())
        shot = next(s for s in doc["shots"] if s["index"] == a.trim)
        print(f"shot {a.trim}: {shot['start']:.3f}-{shot['end']:.3f} s -> "
              f"{trim(Path(doc['video']), shot, a.out, doc['source']['fps'])}")
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
            print(f"POSES NOT CHECKED: {doc['reason']}. This is a FAILURE, not a pass: the solve "
                  f"cannot be shown to be a continuous camera path, so nothing may be placed "
                  f"against it. Re-solve, or point --poses at a directory that has cameras.json "
                  f"and anchors.npz.")
            return 4
        w = doc["worst"]
        print(f"{doc['cameras']} cameras, median step {doc['medianStepUnits']} u, worst "
              f"{w['stepUnits']} u at frame {w['frame']} = {w['depthsPerSecond']} scene depths/s, "
              f"{w['spikeOverMedian']}x median")
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
    print(f"{doc['cutCount']} cut(s), {doc['shotCount']} shots in {doc['source']['duration']:.2f} s")
    for s in doc["shots"]:
        tag = "too short" if s["tooShort"] else s.get("skipped", f"score {s.get('score', '-')}")
        print(f"  shot {s['index']:2d}  {s['start']:7.2f}-{s['end']:7.2f} s  {s['seconds']:6.2f} s  "
              f"{tag}" + (f"  ({'; '.join(s['why'])})" if s.get("why") else ""))
    if "best" in doc:
        print(f"best: shot {doc['best']} -- {doc['bestWhy']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
