#!/usr/bin/env python3
"""Silhouette agreement between a packaged person sequence and the real person in the clip.

  python scripts/motion_silhouette.py public/worlds/gym-4d --masks .context/run/gym/masks.npz

Projects the posed splats into every sampled source frame with that frame's Pi3X camera and
reports IoU against the person mask. Two numbers, because two different agents own the two
failure modes:
  iou          as packaged -- includes any world placement error
  iouAligned   after the 2D shift that maximises overlap -- pose/shape only
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

FLIP = np.diag([1.0, -1.0, -1.0])


def load_masks(path):
    d = np.load(path)
    H, W = int(d["shape"][1]), int(d["shape"][2])
    m = np.unpackbits(d["masks"], axis=-1)[:, :, :W].astype(bool)
    idx = d["indices"] if "indices" in d.files else np.arange(len(m))
    return {int(i): m[k] for k, i in enumerate(idx)}, (H, W)


def splat_silhouette(xyz, opa, scales, cam, hw, subsample=None):
    """Binary silhouette of the projected splats at the source resolution."""
    H, W = hw
    keep = 1 / (1 + np.exp(-opa)) > 0.5
    xyz, scales = xyz[keep], scales[keep]
    if subsample and len(xyz) > subsample:
        sel = np.random.default_rng(0).choice(len(xyz), subsample, replace=False)
        xyz, scales = xyz[sel], scales[sel]
    m = np.array(cam["camera_to_world"]); u, _, vt = np.linalg.svd(m[:3, :3]); Rw = u @ vt
    K = np.array(cam["source_intrinsics"])
    loc = (xyz - m[:3, 3]) @ Rw @ FLIP
    z = loc[:, 2]
    front = z > 1e-6
    loc, scales, z = loc[front], scales[front], z[front]
    if not len(loc):
        return np.zeros((H, W), bool)
    p = (loc @ K.T)[:, :2] / z[:, None]
    # projected splat radius, median over the body: one dilation for the whole cloud
    r_world = np.exp(scales).mean(1)
    r_px = float(np.median(K[0, 0] * r_world / z))
    r = int(np.clip(round(r_px * 1.5), 1, 24))
    img = np.zeros((H, W), np.uint8)
    xs = np.round(p[:, 0]).astype(int); ys = np.round(p[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    img[ys[ok], xs[ok]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    img = cv2.dilate(img, k)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return img.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def best_shift_iou(pred, gt, radius=80, step=4):
    """IoU after the translation that maximises it: pose quality with placement divided out."""
    if not pred.any() or not gt.any():
        return 0.0, (0, 0)
    ys, xs = np.nonzero(pred); py, px = ys.mean(), xs.mean()
    ys, xs = np.nonzero(gt); gy, gx = ys.mean(), xs.mean()
    dy0, dx0 = int(round(gy - py)), int(round(gx - px))
    best, arg = -1.0, (0, 0)
    for s in (step * 4, step):
        rng = radius if s == step * 4 else step * 4
        for dy in range(dy0 - rng, dy0 + rng + 1, s):
            for dx in range(dx0 - rng, dx0 + rng + 1, s):
                v = iou(np.roll(np.roll(pred, dy, 0), dx, 1), gt)
                if v > best:
                    best, arg = v, (dy, dx)
        dy0, dx0 = arg
    return best, arg


def audit(world, masks_path, limit=None, subsample=30000):
    world = Path(world)
    cams = json.loads((world / "cameras.json").read_text())["cameras"]
    seq = json.loads((world / "person" / "sequence.json").read_text())
    src = list(seq["sourceIndices"])
    gt, hw = load_masks(masks_path)
    rows = []
    cand = [c for c in cams if c["sourceIndex"] in gt]
    if limit and len(cand) > limit:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, limit).round().astype(int)]
    for c in cand:
        fi = c["sourceIndex"]
        k = int(np.argmin(np.abs(np.array(src) - fi)))
        v = PlyData.read(world / "person" / seq["frames"][k])["vertex"].data
        xyz = np.column_stack([v["x"], v["y"], v["z"]])
        sc = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])
        pred = splat_silhouette(xyz, np.asarray(v["opacity"]), sc, c, hw, subsample)
        g = gt[fi]
        a, sh = best_shift_iou(pred, g)
        rows.append(dict(frame=fi, iou=iou(pred, g), iouAligned=a, shift=sh,
                         predPx=int(pred.sum()), gtPx=int(g.sum())))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world"); ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=24); ap.add_argument("--json-out")
    a = ap.parse_args()
    rows = audit(a.world, a.masks, a.limit)
    iou_m = np.mean([r["iou"] for r in rows]); al = np.mean([r["iouAligned"] for r in rows])
    area = np.mean([r["predPx"] / max(r["gtPx"], 1) for r in rows])
    print(f"{a.world}: n={len(rows)} IoU {iou_m:.3f}  IoU-aligned {al:.3f}  pred/gt area {area:.2f}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(dict(world=a.world, rows=rows,
            iou=iou_m, iouAligned=al, areaRatio=area), indent=1))
