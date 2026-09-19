#!/usr/bin/env python3
"""Recommend video first; measure sparse still alternatives only when explicitly requested.

Camera angles and image sharpness do not establish scene coverage, successful cleaning,
fixture consistency or generated geometry quality. They cannot justify discarding the
video's temporal evidence automatically. World Labs documents video input, not a general
caption-only limitation: https://docs.worldlabs.ai/api.

Use --still-images to evaluate the historical image/multi-image angular heuristic as an
explicit alternative. Its thresholds come from a few project clips, not a quality benchmark.
Every selected input still needs cleaning/coverage review before generation; no decision here
certifies reduced hallucination. --allow-video remains a compatibility alias for the default.
"""

import argparse, json
from pathlib import Path

import numpy as np

MULTI_MIN = 9.0  # deg: minimum pairwise angle among the frames we would send
ORBIT_MIN = 60.0  # deg: total angular sweep, so a jittery handheld pan is not an orbit
SWEEP_MIN = 72.0  # deg: the same threshold expressed on the pre-solve heading sweep


def poses(cameras_json):
    cams = json.loads(Path(cameras_json).read_text())["cameras"]
    M = np.array([c["camera_to_world"] for c in cams], float)
    return cams, M[:, :3, 3], -M[:, :3, 2]  # positions, forward (OpenGL looks -z)


def pick_spread(fwd, k):
    """Greedy farthest-point set on the sphere of viewing directions. Returns (indices, min angle)."""
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
    i, j = np.unravel_index(np.argmax(ang), ang.shape)
    sel = [int(i), int(j)]
    while len(sel) < min(k, len(fwd)):
        d = ang[:, sel].min(1)
        d[sel] = -1
        sel.append(int(np.argmax(d)))
    sub = ang[np.ix_(sel, sel)]
    sub = sub + np.eye(len(sel)) * 1e9
    return sorted(sel), float(sub.min())


def structure(clip, frames, top=0.55):
    """How much room, as opposed to bare floor, each frame shows. Marble's reconstruction mode has
    nothing to triangulate in a frame that is 90 % rubber matting, and this clip's camera tilts down
    at a supine subject often enough that the widest-angle frame is sometimes exactly that. Scored
    as the gradient energy of the top `top` of the frame, where the room is."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            out[f] = 0.0
            continue
        g = cv2.cvtColor(img[: int(img.shape[0] * top)], cv2.COLOR_BGR2GRAY)
        out[f] = float(
            cv2.Laplacian(
                cv2.resize(g, (480, max(1, g.shape[0] * 480 // g.shape[1]))), cv2.CV_64F
            ).var()
        )
    cap.release()
    return out


def refine(cams, fwd, sel, clip, window=10, min_sep=None):
    """Nudge each chosen camera to the most structured frame within +-`window` of it, as long as the
    set stays at least `min_sep` degrees apart. Angle picks WHERE to look; this picks WHICH frame."""
    min_sep = MULTI_MIN if min_sep is None else min_sep
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
    cand = sorted(
        {j for i in sel for j in range(max(0, i - window), min(len(cams), i + window + 1))}
    )
    sc = structure(clip, [cams[j]["sourceIndex"] for j in cand])
    score = {j: sc[cams[j]["sourceIndex"]] for j in cand}
    out = list(sel)
    for n, i in enumerate(sel):
        best, bs = i, score.get(i, 0.0)
        for j in range(max(0, i - window), min(len(cams), i + window + 1)):
            if score.get(j, 0.0) <= bs:
                continue
            others = [out[m] for m in range(len(out)) if m != n]
            if ang[j, others].min() < min_sep:
                continue
            best, bs = j, score[j]
        out[n] = best
    return sorted(out), {cams[j]["sourceIndex"]: round(score.get(j, 0.0), 1) for j in out}


def measure(cameras_json, k=8, clip=None):
    cams, T, F = poses(cameras_json)
    ang = np.degrees(np.arccos(np.clip(F @ F.T, -1, 1)))
    sel, spread8 = pick_spread(F, k)
    struct = None
    if clip:
        sel, struct = refine(cams, F, sel, clip)
        sub = ang[np.ix_(sel, sel)] + np.eye(len(sel)) * 1e9
        spread8 = round(float(sub.min()), 1)
    d = np.linalg.norm(T[:, None] - T[None], axis=-1)
    return dict(
        cameras=str(cameras_json),
        n=len(cams),
        structure=struct,
        spreadMax=round(float(ang.max()), 1),
        spread8=round(spread8, 1),
        baseline=round(float(d.max()), 3),
        pathLength=round(float(np.linalg.norm(np.diff(T, axis=0), axis=1).sum()), 3),
        pick=[int(cams[i]["sourceIndex"]) for i in sel],
        pickAzimuth=[
            round(float((np.degrees(np.arctan2(F[i][0], -F[i][2])) + 360) % 360), 1) for i in sel
        ],
    )


def video_decision():
    return dict(
        mode="video",
        quality_verified=False,
        why="video-first: retain temporal coverage instead of automatically selecting sparse stills; "
        "review cleaning, coverage and moving fixtures before generation. "
        "Reduced hallucination has not been verified by a controlled comparison",
    )


def decide(m, *, still_images=False):
    if not still_images:
        return video_decision()
    if m["spread8"] >= MULTI_MIN and m["spreadMax"] >= ORBIT_MIN:
        return dict(
            mode="multi-image",
            reconstruct_images=True,
            quality_verified=False,
            frames=m["pick"],
            azimuth=m["pickAzimuth"],
            why=f"explicit still-image alternative: separation {m['spread8']} deg over "
            f"{m['spreadMax']} deg meets the angular heuristic ({MULTI_MIN}/{ORBIT_MIN}); "
            "this does not verify structural coverage, cleaning or output quality",
        )
    return dict(
        mode="image",
        quality_verified=False,
        frames=m["pick"][:1],
        why=f"explicit still-image alternative: separation {m['spread8']} deg over "
        f"{m['spreadMax']} deg does not meet the multi-image angular heuristic "
        f"({MULTI_MIN}/{ORBIT_MIN}); a single image still invents unobserved structure",
    )


def decide_from_prediction(pred, *, still_images=False):
    """A coarse heading estimate is diagnostic, never evidence of generated quality."""
    sweep = float((pred.get("motion") or {}).get("headingSweepDeg", 0.0))
    if not still_images:
        return {**video_decision(), "headingSweepDeg": round(sweep, 1)}
    if pred.get("status") != "registered":
        return dict(
            mode="image",
            quality_verified=False,
            headingSweepDeg=sweep,
            why=f"explicit still-image alternative; coarse solve status {pred.get('status')} "
            "does not support selecting multiple views. Input/output review remains required",
        )
    mode = "multi-image" if sweep >= SWEEP_MIN else "image"
    return dict(
        mode=mode,
        quality_verified=False,
        headingSweepDeg=round(sweep, 1),
        reconstruct_images=(mode == "multi-image"),
        why=f"explicit still-image alternative: heading sweep {sweep:.1f} deg vs heuristic "
        f"{SWEEP_MIN} deg; coverage, cleaning and output quality are unverified",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="+")
    ap.add_argument(
        "--predict-json",
        help="Saved admission JSON containing motion.headingSweepDeg, for the pre-spend call",
    )
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument(
        "--clip", help="source clip: nudges each pick to the most structured nearby frame"
    )
    policy = ap.add_mutually_exclusive_group()
    policy.add_argument(
        "--still-images",
        action="store_true",
        help="explicitly evaluate image/multi-image alternatives; does not certify quality",
    )
    policy.add_argument(
        "--allow-video",
        action="store_true",
        help="compatibility alias; video is already the default",
    )
    ap.add_argument("--out", help="write the decision for the first --cameras here as JSON")
    a = ap.parse_args()
    if a.predict_json and not a.cameras:
        rec = decide_from_prediction(
            json.loads(Path(a.predict_json).read_text()), still_images=a.still_images
        )
        print(json.dumps(rec))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rec, indent=1))
        return
    if not a.cameras:
        raise SystemExit("need --cameras or --predict-json")
    for c in a.cameras:
        m = measure(c, a.max_images, a.clip)
        d = decide(m, still_images=a.still_images)
        rec = {**m, **d}
        print(json.dumps(rec))
        if a.out and c == a.cameras[0]:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
