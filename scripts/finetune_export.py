#!/usr/bin/env python
"""Export a Marble world + the corridor wvp cameras/frames/masks into a gsplat training set.

Marble splats are moved into the SfM native frame so the wvp cameras can be used directly:
    raw = Rx(pi) * scale0 * inv(c2w[origin]) * native   =>   native = c2w[origin] * Rx(pi) * raw / scale0
(same mapping as bake_video_colours.py; origin frame 0 has the identity pose, scale0 = 1.2702).
Cameras are converted from OpenGL (wvp) to OpenCV world-to-camera for gsplat.

Output dir:  splats.npz (means, quats wxyz, log_scales, logit_opacities, f_dc, plus the mapping),
             cameras.npz (viewmats (F,4,4) OpenCV w2c, K (3,3) at the export size, c2w_gl (F,4,4)),
             frames/%04d.jpg (960x540), masks/%04d.png (255 = person, ignore in the loss),
             depth.npz (wvp z-depth layers 480x270 native units, supported flags, per-frame layer index).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_video_colours import ROOT, read_spz, read_wvp, body_height, video_frames  # noqa: E402


def quat_mul(a, b):
    """xyzw Hamilton product, batched."""
    ax, ay, az, aw = a.T; bx, by, bz, bw = b.T
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], 1)


def rotmat_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default=ROOT / 'public/marble-corridor-recon-f0-v2.spz', type=Path)
    ap.add_argument('--wvp', default=ROOT / 'public/worlds/overnight-corridor-video-projection/video-projection.wvp', type=Path)
    ap.add_argument('--clip', default=ROOT / 'public/clips/corridor-walk-elder.mp4', type=Path)
    ap.add_argument('--person', default=ROOT / 'public/worlds/overnight-corridor-video-projection/person/frame_000.ply', type=Path)
    ap.add_argument('--height', type=float, default=0.828)
    ap.add_argument('--origin-frame', type=int, default=0)
    ap.add_argument('--size', type=int, nargs=2, default=[960, 540])
    ap.add_argument('--mask-dilate', type=int, default=4, help='extra dilation of the person mask in export pixels')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'frames').mkdir(exist_ok=True); (a.out / 'masks').mkdir(exist_ok=True)

    spz = read_spz(a.world)
    bodyH = body_height(a.person); scale0 = a.height / bodyH
    H, depth, supported, masks, c2w = read_wvp(a.wvp)
    origin = c2w[a.origin_frame]
    flip = np.diag([1.0, -1.0, -1.0])
    R = origin[:3, :3] @ flip                                   # proper rotation (det +1)
    A = R / scale0; b = origin[:3, 3]
    means = (spz['xyz'] @ A.T + b).astype(np.float32)
    q_r = rotmat_to_quat_xyzw(R)[None].repeat(spz['n'], 0)
    quat_xyzw = quat_mul(q_r, spz['quat'])
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], 1).astype(np.float32)
    log_scales = (spz['log_scale'] - np.log(scale0)).astype(np.float32)
    al = np.clip(spz['alpha'], 1e-4, 1 - 1e-4)
    logit_op = np.log(al / (1 - al)).astype(np.float32)
    np.savez(a.out / 'splats.npz', means=means, quats=quat_wxyz, log_scales=log_scales, logit_opacities=logit_op,
             f_dc=spz['f_dc'].astype(np.float32), scale0=scale0, origin_c2w=origin, R_raw_to_native=R, fb=spz['fb'])
    print(f'splats: {spz["n"]} -> native bbox {means.min(0).round(2)} .. {means.max(0).round(2)}, scale0 {scale0:.4f}', flush=True)

    W, Hh = a.size
    K = np.array([[H['intrinsics']['fx'] * W / H['source']['width'], 0, H['intrinsics']['cx'] * W / H['source']['width']],
                  [0, H['intrinsics']['fy'] * Hh / H['source']['height'], H['intrinsics']['cy'] * Hh / H['source']['height']],
                  [0, 0, 1]], np.float64)
    gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_cv = c2w @ gl2cv
    viewmats = np.linalg.inv(c2w_cv)
    np.savez(a.out / 'cameras.npz', viewmats=viewmats.astype(np.float32), K=K.astype(np.float32), c2w_gl=c2w.astype(np.float32),
             width=W, height=Hh, fps=H['fps'])
    print(f'cameras: {len(c2w)} frames, K fx {K[0,0]:.2f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}', flush=True)
    # wvp depth: z-depth in native units rendered from the reconstruction, one layer per registered frame
    np.savez_compressed(a.out / 'depth.npz', depth=depth.astype(np.float32), supported=supported, depth_index=np.array(H['depthIndex'], np.int32))
    print(f'depth: {depth.shape} layers, supported {100 * supported.mean():.0f}%', flush=True)

    from PIL import Image
    from scipy import ndimage
    n_masked = []
    for i, img in enumerate(video_frames(a.clip, W, Hh)):
        if i >= len(c2w):
            break
        Image.fromarray(img).save(a.out / 'frames' / f'{i:04d}.jpg', quality=95)
        m = np.asarray(Image.fromarray(masks[i]).resize((W, Hh), Image.NEAREST)) > 127
        if a.mask_dilate > 0:
            m = ndimage.binary_dilation(m, iterations=a.mask_dilate)
        Image.fromarray((m * 255).astype(np.uint8)).save(a.out / 'masks' / f'{i:04d}.png')
        n_masked.append(m.mean())
        if i % 100 == 0:
            print(f'frame {i}: person covers {100 * m.mean():.1f}%', flush=True)
    print(f'wrote {len(n_masked)} frames/masks to {a.out}; mean person coverage {100 * np.mean(n_masked):.1f}%', flush=True)


if __name__ == '__main__':
    main()
