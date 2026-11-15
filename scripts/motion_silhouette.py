#!/usr/bin/env python3
"""Silhouette agreement between a packaged person sequence and the real person in the clip.

  uv run --locked --group inference python scripts/motion_silhouette.py public/worlds/gym-4d --masks .context/run/gym/masks.npz

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
    m = np.array(cam["camera_to_world"])
    u, _, vt = np.linalg.svd(m[:3, :3])
    Rw = u @ vt
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
    xs = np.round(p[:, 0]).astype(int)
    ys = np.round(p[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    img[ys[ok], xs[ok]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    img = cv2.dilate(img, k)
    img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    return img.astype(bool)


# The viewer's Spark settings (src/fourd-runtime.js): 0.3 px² of uncompensated pre-blur added to
# every projected covariance, and a footprint cut off at sqrt(5) standard deviations. Spark blends
# in display-encoded colour, so coverage is alpha itself.
VIEWER_PRE_BLUR, VIEWER_MAX_STD2 = 0.3, 5.0
MAX_FOOTPRINT_PX = 64


def quaternion_matrices(q):
    """Rotation matrices for (w, x, y, z) quaternions, the 3DGS PLY `rot_0..rot_3` order."""
    q = np.asarray(q, float)
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )


def splat_coverage(
    xyz, opa, scales, quats, cam, hw, pre_blur=VIEWER_PRE_BLUR, max_std2=VIEWER_MAX_STD2
):
    """Accumulated alpha of the projected Gaussians at the source resolution.

    Each splat is the viewer's footprint: its 3D covariance projected through the pinhole
    Jacobian, plus `pre_blur`, cut at `max_std2` squared standard deviations, alpha capped at
    0.99 and dropped below 1/255. Coverage is 1 - prod(1 - alpha), which does not depend on draw
    order, so thresholding it at 0.5 is the silhouette the viewer draws over any background.
    `scales` are log scales and `opa` opacity logits, as stored in the PLY. Checked against the
    live viewer (fourd.html, contact shadow hidden) on the forest walk: IoU 0.987-0.988, area
    within 0.1%, on three source frames.
    """
    H, W = hw
    m = np.array(cam["camera_to_world"], float)
    u, _, vt = np.linalg.svd(m[:3, :3])
    Rw = u @ vt
    K = np.array(cam["source_intrinsics"], float)
    loc = (np.asarray(xyz, float) - m[:3, 3]) @ Rw @ FLIP
    z = loc[:, 2]
    front = z > 1e-6
    loc, z = loc[front], z[front]
    alpha = 1 / (1 + np.exp(-np.asarray(opa, float)[front]))
    basis = (
        quaternion_matrices(np.asarray(quats, float)[front])
        * np.exp(np.asarray(scales, float)[front])[:, None, :]
    )
    view = FLIP @ Rw.T
    cov = view @ (basis @ basis.transpose(0, 2, 1)) @ view.T
    jac = np.zeros((len(z), 2, 3))
    jac[:, 0, 0] = K[0, 0] / z
    jac[:, 0, 2] = -K[0, 0] * loc[:, 0] / z**2
    jac[:, 1, 1] = K[1, 1] / z
    jac[:, 1, 2] = -K[1, 1] * loc[:, 1] / z**2
    cov2 = jac @ cov @ jac.transpose(0, 2, 1)
    a, b, c = cov2[:, 0, 0] + pre_blur, cov2[:, 0, 1], cov2[:, 1, 1] + pre_blur
    det = a * c - b * b
    inv_a, inv_b, inv_c = c / det, -b / det, a / det
    centre = (loc @ K.T)[:, :2] / z[:, None]
    largest = 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)
    radius = np.clip(np.ceil(np.sqrt(max_std2 * largest)).astype(int), 1, MAX_FOOTPRINT_PX)
    log_clear = np.zeros(H * W)
    for r in np.unique(radius):
        pick = np.nonzero(radius == r)[0]
        dy, dx = (offset.ravel() for offset in np.mgrid[-r : r + 1, -r : r + 1])
        px = np.round(centre[pick, 0]).astype(int)[:, None] + dx
        py = np.round(centre[pick, 1]).astype(int)[:, None] + dy
        ex, ey = px - centre[pick, 0][:, None], py - centre[pick, 1][:, None]
        d2 = inv_a[pick, None] * ex**2 + 2 * inv_b[pick, None] * ex * ey + inv_c[pick, None] * ey**2
        value = np.minimum(0.99, alpha[pick, None] * np.exp(-0.5 * d2))
        keep = (d2 <= max_std2) & (value >= 1 / 255) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        log_clear += np.bincount(
            (py * W + px)[keep], weights=np.log1p(-value[keep]), minlength=H * W
        )
    return 1 - np.exp(log_clear.reshape(H, W))


def drawn_silhouette(vertex, xyz, log_scales, cam, hw):
    """Where the viewer draws a PLY person at least half opaque: coverage > 0.5."""
    quats = np.column_stack([vertex[f"rot_{i}"] for i in range(4)])
    return splat_coverage(xyz, vertex["opacity"], log_scales, quats, cam, hw) > 0.5


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def translate_mask(mask, dy, dx):
    """Translate within the image bounds, discarding pixels that move outside them."""
    h, w = mask.shape
    shifted = np.zeros_like(mask)
    if abs(dy) >= h or abs(dx) >= w:
        return shifted
    shifted[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = mask[
        max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
    ]
    return shifted


def best_shift_iou(pred, gt, radius=80, step=4):
    """IoU after the translation that maximises it: pose quality with placement divided out."""
    if not pred.any() or not gt.any():
        return 0.0, (0, 0)
    ys, xs = np.nonzero(pred)
    py, px = ys.mean(), xs.mean()
    ys, xs = np.nonzero(gt)
    gy, gx = ys.mean(), xs.mean()
    dy0, dx0 = int(round(gy - py)), int(round(gx - px))
    best, arg = -1.0, (0, 0)
    for s in (step * 4, step):
        rng = radius if s == step * 4 else step * 4
        for dy in range(dy0 - rng, dy0 + rng + 1, s):
            for dx in range(dx0 - rng, dx0 + rng + 1, s):
                v = iou(translate_mask(pred, dy, dx), gt)
                if v > best:
                    best, arg = v, (dy, dx)
        dy0, dx0 = arg
    return best, arg


def source_frame_paths(seq, selected_indices):
    """Resolve exact source samples, rejecting ambiguous or unavailable sequence evidence."""
    src = list(seq["sourceIndices"])
    frames = seq["frames"]
    if len(src) != len(frames):
        raise ValueError(f"person sequence has {len(src)} sourceIndices but {len(frames)} frames")
    if len(set(src)) != len(src):
        raise ValueError("person sequence has duplicate sourceIndices")
    frame_by_source = dict(zip(src, frames, strict=True))
    missing = sorted(set(selected_indices) - frame_by_source.keys())
    if missing:
        raise ValueError(f"person sequence has no pose for selected sourceIndices: {missing}")
    return frame_by_source


def audit(world, masks_path, limit=None):
    world = Path(world)
    cams = json.loads((world / "cameras.json").read_text())["cameras"]
    seq = json.loads((world / "person" / "sequence.json").read_text())
    gt, hw = load_masks(masks_path)
    rows = []
    cand = [c for c in cams if c["sourceIndex"] in gt]
    if limit and len(cand) > limit:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, limit).round().astype(int)]
    frame_by_source = source_frame_paths(seq, (c["sourceIndex"] for c in cand))
    for c in cand:
        fi = c["sourceIndex"]
        v = PlyData.read(world / "person" / frame_by_source[fi])["vertex"].data
        xyz = np.column_stack([v["x"], v["y"], v["z"]])
        sc = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])
        pred = drawn_silhouette(v, xyz, sc, c, hw)
        g = gt[fi]
        a, sh = best_shift_iou(pred, g)
        rows.append(
            dict(
                frame=fi,
                iou=iou(pred, g),
                iouAligned=a,
                shift=sh,
                predPx=int(pred.sum()),
                gtPx=int(g.sum()),
            )
        )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    rows = audit(a.world, a.masks, a.limit)
    iou_m = np.mean([r["iou"] for r in rows])
    al = np.mean([r["iouAligned"] for r in rows])
    area = np.mean([r["predPx"] / max(r["gtPx"], 1) for r in rows])
    print(
        f"{a.world}: n={len(rows)} IoU {iou_m:.3f}  IoU-aligned {al:.3f}  pred/gt area {area:.2f}"
    )
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps(
                dict(world=a.world, rows=rows, iou=iou_m, iouAligned=al, areaRatio=area), indent=1
            )
        )
