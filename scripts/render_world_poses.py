#!/usr/bin/env python3
"""Render a Marble .spz from the clip's OWN recorded camera poses, on the CPU, no browser.

This is the render half of scripts/verify_world.py, kept separate because it is also the cheapest
way to put a world next to the footage for any other reason. It reproduces what fourd.html shows:
the spz is loaded y-down and flipped by `world.rotation.x = Math.PI`, and cameras.json translations
(Pi3X normalised units) are multiplied by the preset's fitted `scale` to reach world units. Same
convention as scripts/world-vs-source.mjs, which does this in a real browser -- the two agree.

  render_world_poses.py --world public/marble-gym-clean2.spz \
      --cameras public/worlds/gym-4d/cameras.json --scale0 1.0020 \
      --frames 0,108,216 --out .context/verify/gym/render [--width 960]
"""
import argparse, gzip, json, struct
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
SPZ_COLOR_SCALE = 0.15
FLIP = np.diag([1.0, -1.0, -1.0])       # GL camera -> CV camera, and spz y-down -> y-up


def read_spz(path: Path):
    """spz v2, shDegree 0 (every Marble world). Copied from scripts/bake_video_colours.py, which
    owns the canonical reader; duplicated rather than imported because that module needs torch."""
    raw = gzip.open(path).read()
    magic, ver, n, sh, fb, _flags, _ = struct.unpack('<IIIBBBB', raw[:16])
    assert magic == 0x5053474E and ver == 2 and sh == 0, (hex(magic), ver, sh)
    o = 16
    pos_b = np.frombuffer(raw, np.uint8, n * 9, o).reshape(n, 3, 3); o += n * 9
    alpha_b = np.frombuffer(raw, np.uint8, n, o); o += n
    col_b = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    scale_b = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    rot_b = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3); o += n * 3
    assert o == len(raw), (o, len(raw))
    fixed = pos_b[:, :, 0].astype(np.int32) | (pos_b[:, :, 1].astype(np.int32) << 8) | (pos_b[:, :, 2].astype(np.int32) << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    xyz = fixed.astype(np.float32) / float(1 << fb)
    q = (rot_b.astype(np.float32) - 127.5) / 127.5
    w = np.sqrt(np.clip(1.0 - (q * q).sum(1), 0, 1))
    return dict(n=n,
                xyz=xyz,
                alpha=alpha_b.astype(np.float32) / 255.0,
                rgb=np.clip(0.5 + SH_C0 * (col_b.astype(np.float32) / 255.0 - 0.5) / SPZ_COLOR_SCALE, 0, 1),
                scale=np.exp(scale_b.astype(np.float32) / 16.0 - 10.0),
                quat=np.concatenate([q, w[:, None]], 1).astype(np.float32))


def quat_to_rot(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _composite(order, u, v, rad, a00, a01, a11, op, rgb, W, H, img, T):
        for k in range(order.shape[0]):
            i = order[k]
            r = int(rad[i])
            x0 = max(int(u[i]) - r, 0); x1 = min(int(u[i]) + r + 1, W)
            y0 = max(int(v[i]) - r, 0); y1 = min(int(v[i]) + r + 1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            a = a00[i]; b = a01[i]; c = a11[i]; o = op[i]
            cr = rgb[i, 0]; cg = rgb[i, 1]; cb = rgb[i, 2]
            for y in range(y0, y1):
                dy = y + 0.5 - v[i]
                for x in range(x0, x1):
                    t = T[y, x]
                    if t < 1e-3:
                        continue
                    dx = x + 0.5 - u[i]
                    power = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy)
                    if power < -8.0:
                        continue
                    alpha = o * np.exp(power)
                    if alpha > 0.99:
                        alpha = 0.99
                    if alpha < 0.004:
                        continue
                    wgt = alpha * t
                    img[y, x, 0] += wgt * cr; img[y, x, 1] += wgt * cg; img[y, x, 2] += wgt * cb
                    T[y, x] = t * (1.0 - alpha)
except ImportError:                                                        # pragma: no cover
    _composite = None


def render(g, c2w, K, size, scale0, bg=0.0):
    """One frame. c2w is the cameras.json OpenGL camera-to-world; its translation is scaled by
    scale0. Returns (rgb float32 HxWx3, alpha HxW) -- alpha is the accumulated coverage, which is
    the honest "is there any world here at all" signal."""
    W, H = size
    Rc = np.asarray(c2w, np.float64)[:3, :3]
    u_, _, vt = np.linalg.svd(Rc); Rc = u_ @ vt                            # re-orthonormalise
    t = np.asarray(c2w, np.float64)[:3, 3] * scale0
    Wm = (FLIP @ Rc.T).astype(np.float32)                                  # world -> CV camera

    X = (g['xyz'] * np.array([1, -1, -1], np.float32))                     # world.rotation.x = PI
    P = (X - t.astype(np.float32)) @ Wm.T
    z = P[:, 2]
    keep = (z > 0.05) & (g['alpha'] > 0.02)
    if not keep.any():
        return np.full((H, W, 3), bg, np.float32), np.zeros((H, W), np.float32)
    idx = np.nonzero(keep)[0]
    P = P[idx]; z = z[idx]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * P[:, 0] / z + cx
    v = fy * P[:, 1] / z + cy
    on = (u > -64) & (u < W + 64) & (v > -64) & (v < H + 64)
    idx, P, z, u, v = idx[on], P[on], z[on], u[on], v[on]

    Rg = quat_to_rot(g['quat'][idx])
    Rg = np.einsum('ij,njk->nik', FLIP.astype(np.float32), Rg)             # spz -> world
    S = g['scale'][idx]
    M = Rg * S[:, None, :]                                                 # R diag(s)
    Sigma = np.einsum('nij,nkj->nik', M, M)
    Sigma_c = np.einsum('ij,njk,lk->nil', Wm, Sigma, Wm)
    J = np.zeros((len(z), 2, 3), np.float32)
    J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * P[:, 0] / (z * z)
    J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * P[:, 1] / (z * z)
    C = np.einsum('nij,njk,nlk->nil', J, Sigma_c, J)
    C[:, 0, 0] += 0.3; C[:, 1, 1] += 0.3
    det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2
    ok = det > 1e-8
    idx, u, v, z, C, det = idx[ok], u[ok], v[ok], z[ok], C[ok], det[ok]
    inv = 1.0 / det
    a00 = (C[:, 1, 1] * inv).astype(np.float32)
    a01 = (-C[:, 0, 1] * inv).astype(np.float32)
    a11 = (C[:, 0, 0] * inv).astype(np.float32)
    rad = np.clip(3.0 * np.sqrt(np.maximum(C[:, 0, 0], C[:, 1, 1])), 1, 256).astype(np.float32)

    order = np.argsort(z).astype(np.int64)                                 # front to back
    img = np.zeros((H, W, 3), np.float32)
    T = np.ones((H, W), np.float32)
    if _composite is None:
        raise SystemExit('numba is required for the rasteriser')
    _composite(order, u.astype(np.float32), v.astype(np.float32), rad, a00, a01, a11,
               g['alpha'][idx].astype(np.float32), g['rgb'][idx].astype(np.float32), W, H, img, T)
    alpha = 1.0 - T
    return img + bg * T[:, :, None], alpha


def cameras_for(cameras_json, frames):
    cams = json.loads(Path(cameras_json).read_text())['cameras']
    by_idx = {c['sourceIndex']: c for c in cams}
    out = []
    for f in frames:
        c = by_idx.get(f) or min(cams, key=lambda c: abs(c['sourceIndex'] - f))
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True)
    ap.add_argument('--cameras', required=True)
    ap.add_argument('--scale0', type=float, required=True)
    ap.add_argument('--frames', required=True, help='comma-separated sourceIndex values')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--out', required=True, help='output stem; writes <stem>-f<idx>.png')
    a = ap.parse_args()

    import cv2
    g = read_spz(Path(a.world))
    frames = [int(x) for x in a.frames.split(',')]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for c in cameras_for(a.cameras, frames):
        sw, sh = c['source_image_size']
        s = a.width / sw
        K = np.array(c['source_intrinsics'], np.float64) * s
        K[2, 2] = 1.0
        img, alpha = render(g, np.array(c['camera_to_world']), K, (a.width, int(round(sh * s))), a.scale0)
        p = f'{a.out}-f{c["sourceIndex"]}.png'
        cv2.imwrite(p, cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f'{p}  coverage {float((alpha > 0.5).mean()):.3f}')


if __name__ == '__main__':
    main()
