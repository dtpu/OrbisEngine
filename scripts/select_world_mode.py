#!/usr/bin/env python3
"""Pick the Marble generation mode for a clip by MEASURING its camera motion, not by guessing.

Marble has three input modes and they do not degrade gracefully into one another:

  image        one frame -> one pano -> splats. Real pixels in one direction, invented everywhere
               else. Won on the forward-dolly corridor clip.
  multi-image  up to 8 frames with `reconstruct_images: true`. Real pixels in every direction the
               frames face -- but only if the frames genuinely face different directions. Won on
               the orbiting bedroom clip; FAILED on the dolly, where eight frames from a forward
               walk are eight views of the same wall and the reconstruction has nothing to triangulate.
  video        the clip is auto-captioned and the caption is generated from. This is why a Diagon
               Alley clip became a generic wizard street (share/HP-MARBLE-VERDICT.md). Never chosen
               here; it is available only behind --allow-video.

So the decision is one question with a number behind it: do the frames we would send face
genuinely different directions? The measurement is `spread8` -- greedily pick the 8 frames whose
forward vectors are as far apart as possible, then report the SMALLEST angle between any two of
them. That is exactly "how different are the eight images", not "does one outlier frame look away".

Measured on this project's own clips (Pi3X poses, public/worlds/*/cameras.json):

  clip       spreadMax  spread8   outcome on record
  corridor       ~1.5     ~0.2    dolly; image mode won, multi-image FAILED
  elevator       14.7      1.9
  tos-hall       20.7      2.5    walk down a hall
  stairs2        24.9      3.3    tracking alongside, camera never turns
  atrium2        32.5      3.9
  selfie         36.9      4.7
  atrium         46.9      5.5    walking away across a floor
  living         64.1      7.7
  bedroom        73.2      9.0    orbit; multi-image WON
  lobby          91.1     11.1    wide look-around but only 0.56 u of travel
  gym           131.8     20.1    a real 180 deg arc -- the widest in the project

The threshold sits between the two decided cases: the dolly that failed multi-image (spread8 ~0.2)
and the orbit that won it (9.0). MULTI_MIN is set at 9 deg, at bedroom, so bedroom and everything
wider is routed to multi-image and every dolly on record is routed to image mode.
Raise it if a multi-image world comes back stitched at the wrong scale.

Before the world is bought there are no Pi3X poses yet, so the same decision can be taken off the
coarse pre-solve this project already runs (worker/experiments/clip_admission.py, 32 frames,
exhaustive matching, the ranker behind scripts/rank_film_shots.py). Its `motion.headingSweepDeg` is
a coarse spreadMax; across the clips above spread8 is spreadMax/6.6..8.3, so the MULTI_MIN of 9 deg
lands at a heading sweep of about 72 deg, which is SWEEP_MIN below.

  select_world_mode.py --cameras public/worlds/gym-4d/cameras.json [--clip c.mp4] [--max-images 8]
  select_world_mode.py --predict-json .context/run/gym/admission.json
"""
import argparse, json
from pathlib import Path

import numpy as np

MULTI_MIN = 9.0        # deg: minimum pairwise angle among the frames we would send
ORBIT_MIN = 60.0       # deg: total angular sweep, so a jittery handheld pan is not an orbit
SWEEP_MIN = 72.0       # deg: the same threshold expressed on the pre-solve heading sweep


def poses(cameras_json):
    cams = json.loads(Path(cameras_json).read_text())['cameras']
    M = np.array([c['camera_to_world'] for c in cams], float)
    return cams, M[:, :3, 3], -M[:, :3, 2]          # positions, forward (OpenGL looks -z)


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
            out[f] = 0.0; continue
        g = cv2.cvtColor(img[:int(img.shape[0] * top)], cv2.COLOR_BGR2GRAY)
        out[f] = float(cv2.Laplacian(cv2.resize(g, (480, max(1, g.shape[0] * 480 // g.shape[1]))), cv2.CV_64F).var())
    cap.release()
    return out


def refine(cams, fwd, sel, clip, window=10, min_sep=None):
    """Nudge each chosen camera to the most structured frame within +-`window` of it, as long as the
    set stays at least `min_sep` degrees apart. Angle picks WHERE to look; this picks WHICH frame."""
    min_sep = MULTI_MIN if min_sep is None else min_sep
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
    cand = sorted({j for i in sel for j in range(max(0, i - window), min(len(cams), i + window + 1))})
    sc = structure(clip, [cams[j]['sourceIndex'] for j in cand])
    score = {j: sc[cams[j]['sourceIndex']] for j in cand}
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
    return sorted(out), {cams[j]['sourceIndex']: round(score.get(j, 0.0), 1) for j in out}


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
    return dict(cameras=str(cameras_json), n=len(cams), structure=struct,
                spreadMax=round(float(ang.max()), 1),
                spread8=round(spread8, 1),
                baseline=round(float(d.max()), 3),
                pathLength=round(float(np.linalg.norm(np.diff(T, axis=0), axis=1).sum()), 3),
                pick=[int(cams[i]['sourceIndex']) for i in sel],
                pickAzimuth=[round(float((np.degrees(np.arctan2(F[i][0], -F[i][2])) + 360) % 360), 1) for i in sel])


def decide(m, allow_video=False):
    if m['spread8'] >= MULTI_MIN and m['spreadMax'] >= ORBIT_MIN:
        return dict(mode='multi-image', reconstruct_images=True, frames=m['pick'], azimuth=m['pickAzimuth'],
                    why=f"8 frames separated by at least {m['spread8']} deg over a {m['spreadMax']} deg sweep: "
                        f"genuinely different directions, which is what reconstruction mode needs "
                        f"(threshold {MULTI_MIN}/{ORBIT_MIN})")
    if allow_video:
        return dict(mode='video', why='forced by --allow-video; the clip is auto-captioned and the '
                                      'caption is generated from, so content fidelity is not expected')
    return dict(mode='image', frames=m['pick'][:1],
                why=f"only {m['spread8']} deg between the most separated frames over a {m['spreadMax']} deg "
                    f"sweep (threshold {MULTI_MIN}/{ORBIT_MIN}): eight frames would face the same way, "
                    f"which is the case multi-image failed on")


def decide_from_prediction(pred):
    """The same call before any credits are spent, from the coarse solve's heading sweep."""
    sweep = float((pred.get('motion') or {}).get('headingSweepDeg', 0.0))
    if pred.get('status') != 'registered':
        return dict(mode='image', headingSweepDeg=sweep, why=f"coarse solve status {pred.get('status')}: "
                    "no usable camera motion, so one image is all that can be trusted")
    mode = 'multi-image' if sweep >= SWEEP_MIN else 'image'
    return dict(mode=mode, headingSweepDeg=round(sweep, 1), reconstruct_images=(mode == 'multi-image'),
                why=f"pre-solve heading sweep {sweep:.1f} deg vs threshold {SWEEP_MIN} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', nargs='+')
    ap.add_argument('--predict-json', help='clip_admission output, for the pre-spend call')
    ap.add_argument('--max-images', type=int, default=8)
    ap.add_argument('--clip', help='source clip: nudges each pick to the most structured nearby frame')
    ap.add_argument('--allow-video', action='store_true')
    ap.add_argument('--out', help='write the decision for the first --cameras here as JSON')
    a = ap.parse_args()
    if a.predict_json and not a.cameras:
        rec = decide_from_prediction(json.loads(Path(a.predict_json).read_text()))
        print(json.dumps(rec))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rec, indent=1))
        return
    if not a.cameras:
        raise SystemExit('need --cameras or --predict-json')
    for c in a.cameras:
        m = measure(c, a.max_images, a.clip)
        d = decide(m, a.allow_video)
        rec = {**m, **d}
        print(json.dumps(rec))
        if a.out and c == a.cameras[0]:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
