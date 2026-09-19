#!/usr/bin/env python3
"""Did the camera MOVE, or did it only turn? Answered from the video, before paying for SfM.

The sibling track recorded, as a product fact, that there was no way to tell a user what a clip
would give them without first running the reconstruction, because a pose-free HEADING estimate
drifts without bound. That is true of heading. It is not true of degeneracy, which is what
actually decides whether a clip reconstructs at all.

A camera that rotates on the spot maps every frame onto every other frame by a homography,
exactly and everywhere, however much the picture moves. A camera that translates does not: points
at different depths move by different amounts, and a single homography cannot hold them all. So:

    inliers to a homography  /  inliers to a fundamental matrix

is about 1.0 when the motion is a pure rotation (or the scene is one flat plane), and falls away
as real parallax appears. This is the standard degeneracy test, not a heading estimate, and it
needs no poses.

Measured at several frame separations, because a handheld camera drifts: adjacent frames of a slow
dolly look rotational and forty frames apart do not. The verdict uses the WIDEST separation the
clip can support, since that is the baseline SfM will actually get to use.

On this repo's eight clips the split is clean and wide - see docs/experiments/any-clip-pipeline.md.

  python3 worker/experiments/parallax_probe.py --images <dir> [--json out.json]
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np

# Below this the widest baseline still has not produced parallax a homography cannot explain, and
# incremental SfM will fail to fix its gauge and collapse into three-frame fragments.
ROTATION_ONLY_RATIO = 0.92


def pair_ratio(ga, gb, orb, bf) -> dict | None:
    import cv2

    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 60 or len(kb) < 60:
        return None
    m = bf.match(da, db)
    if len(m) < 60:
        return None
    src = np.float32([ka[x.queryIdx].pt for x in m])
    dst = np.float32([kb[x.trainIdx].pt for x in m])
    H, hm = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=4000)
    F, fm = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.999)
    if H is None or F is None or hm is None or fm is None:
        return None
    hi, fi = int(hm.sum()), int(fm.sum())
    if fi < 40:
        return None
    # Median symmetric transfer error under the homography, in pixels: how badly the flat-world
    # explanation actually fails, in a unit a person can picture.
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = float(np.median(np.linalg.norm(proj - dst, axis=1)))
    return dict(matches=len(m), homographyInliers=hi, fundamentalInliers=fi,
                ratio=hi / fi, homographyResidualPx=resid)


def probe(images: Path, gaps=(2, 8, 24), pairs_per_gap: int = 12, width: int = 720) -> dict:
    import cv2

    names = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n = len(names)
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cache: dict[int, np.ndarray] = {}

    def gray(i: int):
        if i not in cache:
            a = cv2.imread(str(names[i]), cv2.IMREAD_GRAYSCALE)
            h, w = a.shape[:2]
            cache[i] = cv2.resize(a, (width, round(h * width / w))) if w > width else a
        return cache[i]

    out = {}
    for g in gaps:
        if n - g < 4:
            continue
        starts = np.linspace(0, n - g - 1, min(pairs_per_gap, n - g)).astype(int)
        rs, ds = [], []
        for s in starts:
            r = pair_ratio(gray(int(s)), gray(int(s + g)), orb, bf)
            if r:
                rs.append(r["ratio"])
                ds.append(r["homographyResidualPx"])
        if rs:
            out[f"gap{g}"] = dict(pairs=len(rs), medianRatio=round(float(np.median(rs)), 4),
                                  medianHomographyResidualPx=round(float(np.median(ds)), 3))
    if not out:
        return dict(frames=n, verdict="inconclusive",
                    reason="not enough matchable frame pairs to test")

    widest = out[sorted(out, key=lambda k: int(k[3:]))[-1]]
    ratio = widest["medianRatio"]
    rec = dict(frames=n, byGap=out, widestGapRatio=ratio,
               widestGapHomographyResidualPx=widest["medianHomographyResidualPx"],
               threshold=ROTATION_ONLY_RATIO,
               verdict="rotation-only" if ratio >= ROTATION_ONLY_RATIO else "has-parallax")
    rec["headline"] = (
        "The camera turned but did not travel. Every frame maps onto every other by a single flat "
        "warp, so there is no second viewpoint and no depth to recover."
        if rec["verdict"] == "rotation-only" else
        "The camera travelled: near and far move by different amounts, which is the parallax a "
        "reconstruction needs.")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--gaps", type=int, nargs="*", default=[2, 8, 24])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rec = probe(a.images, gaps=tuple(a.gaps))
    print(json.dumps(rec, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
