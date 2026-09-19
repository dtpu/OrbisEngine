#!/usr/bin/env python
"""Export the Marble bedroom world + the Pi3X posed frames into a gsplat training set.

Same mapping as finetune_export.py / bake_video_colours.py, but the cameras come from a plain
cameras.json (Pi3X, OpenGL camera_to_world, camera 0 ~ identity) instead of a wvp container, the
person masks are recomputed with SegFormer, and the depth layers are z-buffered from the Pi3X
anchor point cloud instead of read from the wvp.

    native = origin_c2w * (Rx(pi) raw / scale0)      scale0 fitted with bake_video_colours.py --diag

Output: splats.npz, cameras.npz, frames/%04d.jpg, masks/%04d.png (255 = person), depth.npz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_video_colours import ROOT, read_spz, static_cloud_from_anchors, zbuffer_depth, video_frames  # noqa: E402
from finetune_export import quat_mul, rotmat_to_quat_xyzw  # noqa: E402
from sfm_frame import origin_c2w as levelled_origin, describe as describe_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default=ROOT / 'public/marble-bedroom-clean.spz', type=Path)
    ap.add_argument('--cameras', default=ROOT / '.context/own/bedroom/pi3x/cameras.json', type=Path)
    ap.add_argument('--anchors', default=ROOT / '.context/own/bedroom/pi3x/anchors.npz', type=Path)
    ap.add_argument('--anchor-samples', default='0,15,31,46,62,77,93,108')
    ap.add_argument('--clip', default=ROOT / 'public/clips/bedroom.mp4', type=Path)
    ap.add_argument('--scale0', type=float, required=True, help='native -> raw scale (fit with --diag)')
    ap.add_argument('--origin-frame', type=int, default=0, help='index into cameras.json')
    ap.add_argument('--size', type=int, nargs=2, default=[960, 540])
    ap.add_argument('--depth-res', type=int, nargs=2, default=[480, 270])
    ap.add_argument('--mask-dilate', type=int, default=16, help='dilation of the person mask in full-res pixels')
    ap.add_argument('--masks-npz', type=Path, default=None, help='packed per-sample person masks (tracks_to_clean_masks.py) instead of SegFormer')
    ap.add_argument('--moved-mask', action='store_true', help='ignore everything that MOVED, not just the person (cast shadow, carried object, passer-by). Off by default.')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    (a.out / 'frames').mkdir(parents=True, exist_ok=True); (a.out / 'masks').mkdir(parents=True, exist_ok=True)

    import json
    cams = json.load(open(a.cameras))['cameras']
    c2w = np.array([c['camera_to_world'] for c in cams], np.float64)
    src_idx = [int(c['sourceIndex']) for c in cams]
    Ks = np.array([c['source_intrinsics'] for c in cams], np.float64)
    SW, SH_ = cams[0]['source_image_size']
    print(f'{len(cams)} cameras, K spread fx {Ks[:, 0, 0].std():.3f} cx {Ks[:, 0, 2].std():.3f}', flush=True)

    spz = read_spz(a.world)
    # the packagers' origin: camera 0 at the origin, gravity on +y when framealign.json exists
    origin, fa = levelled_origin(a.cameras, a.origin_frame)
    print(f'frame: {describe_frame(fa)}', flush=True)
    flip = np.diag([1.0, -1.0, -1.0])
    R = origin[:3, :3] @ flip
    A = R / a.scale0; b = origin[:3, 3]
    means = (spz['xyz'] @ A.T + b).astype(np.float32)
    q_r = rotmat_to_quat_xyzw(R)[None].repeat(spz['n'], 0)
    quat_xyzw = quat_mul(q_r, spz['quat'])
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], 1).astype(np.float32)
    log_scales = (spz['log_scale'] - np.log(a.scale0)).astype(np.float32)
    al = np.clip(spz['alpha'], 1e-4, 1 - 1e-4)
    logit_op = np.log(al / (1 - al)).astype(np.float32)
    np.savez(a.out / 'splats.npz', means=means, quats=quat_wxyz, log_scales=log_scales, logit_opacities=logit_op,
             f_dc=spz['f_dc'].astype(np.float32), scale0=a.scale0, origin_c2w=origin, R_raw_to_native=R, fb=spz['fb'])
    print(f'splats: {spz["n"]} -> native bbox {means.min(0).round(2)} .. {means.max(0).round(2)}, scale0 {a.scale0:.4f}', flush=True)

    W, H = a.size
    K = Ks[a.origin_frame].copy()
    K[0] *= W / SW; K[1] *= H / SH_
    gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])
    viewmats = np.linalg.inv(c2w @ gl2cv)
    np.savez(a.out / 'cameras.npz', viewmats=viewmats.astype(np.float32), K=K.astype(np.float32), c2w_gl=c2w.astype(np.float32),
             width=W, height=H, source_index=np.array(src_idx, np.int32))
    print(f'cameras: K fx {K[0,0]:.2f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}; positions bbox '
          f'{c2w[:, :3, 3].min(0).round(2)} .. {c2w[:, :3, 3].max(0).round(2)}', flush=True)

    # depth: one z-buffered layer per camera from the Pi3X anchor cloud, native units
    cache = a.anchors.with_name('static-scene.npy')
    scene = np.load(cache) if cache.exists() else static_cloud_from_anchors(a.anchors, a.cameras, [int(x) for x in a.anchor_samples.split(',')])
    if not cache.exists():
        np.save(cache, scene)
    DW, DH = a.depth_res
    depth = np.zeros((len(cams), DH, DW), np.float32); sup = np.zeros((len(cams), DH, DW), bool)
    for i in range(len(cams)):
        depth[i], sup[i] = zbuffer_depth(scene, c2w[i], Ks[i], (SW, SH_), (DW, DH))
    np.savez_compressed(a.out / 'depth.npz', depth=depth, supported=sup, depth_index=np.arange(len(cams), dtype=np.int32))
    print(f'depth: {depth.shape} layers, supported {100 * sup.mean():.0f}%, median depth {np.median(depth[sup]):.2f} native', flush=True)

    # floor from the anchor cloud -> metres per native unit (camera height assumption, see status doc)
    floor = float(np.percentile(scene[:, 1], 1.0))
    print(f'anchor-cloud floor y {floor:.2f}, camera mean y {c2w[:, 1, 3].mean():.2f} native '
          f'(=> 1 native unit = {1.4 / (c2w[:, 1, 3].mean() - floor):.3f} m if the phone was 1.4 m up)', flush=True)

    from PIL import Image
    from scipy import ndimage
    want = {s: i for i, s in enumerate(src_idx)}
    imgs = np.zeros((len(cams), H, W, 3), np.uint8)
    for j, img in enumerate(video_frames(a.clip, W, H)):
        if j in want:
            imgs[want[j]] = img
    if a.masks_npz:   # already dilated by tracks_to_clean_masks.py; Mask R-CNN instances, not SegFormer
        z = np.load(a.masks_npz); full = np.unpackbits(z['masks'], axis=2)[:, :, :int(z['shape'][2])] > 0
        assert len(full) == len(cams), f'{len(full)} masks vs {len(cams)} cameras'
        masks = np.stack([np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127 for m in full])
    elif a.moved_mask:
        # the loss ignores what MOVED, not only the person: a cast shadow or a carried object is
        # in the frames but not in the static world, so the trainer is graded against pixels the
        # model can never reproduce and paints them onto the geometry to close the gap
        sys.path.insert(0, str(ROOT / 'worker'))
        from wander_worker.masks import moved_content_masks
        res = moved_content_masks(imgs, dilate_px=max(1, a.mask_dilate * W // SW), batch_size=8)
        masks = res['moved']
        print(f"moved-content mask: person {100 * res['person'].mean():.1f}% -> moved "
              f"{100 * masks.mean():.1f}% of pixels, "
              f"{sum(1 for st in res['stats'] if st['rejected'])}/{len(imgs)} frames fell back to person only",
              flush=True)
    else:
        sys.path.insert(0, str(ROOT / 'worker'))
        from wander_worker.masks import people_masks
        masks = np.concatenate([people_masks(imgs[i:i + 8], dilate_px=max(1, a.mask_dilate * W // SW)) for i in range(0, len(imgs), 8)])
    for i in range(len(cams)):
        Image.fromarray(imgs[i]).save(a.out / 'frames' / f'{i:04d}.jpg', quality=95)
        m = masks[i]
        m = ndimage.binary_dilation(m, iterations=2)
        Image.fromarray((m * 255).astype(np.uint8)).save(a.out / 'masks' / f'{i:04d}.png')
    print(f'wrote {len(cams)} frames/masks to {a.out}; mean person coverage {100 * masks.mean():.1f}%', flush=True)


if __name__ == '__main__':
    main()
