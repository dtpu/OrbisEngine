#!/usr/bin/env python
"""Bake the recorded video onto a Marble world's splats, offline.

For every world splat, gather the real pixel colour from every source frame whose SfM camera sees
it (frustum, person mask, depth-map occlusion with a tolerance), weight it, average, and write a
new world where observed splats carry the averaged real colour (SH0 only) and the rest keep
Marble's colour. Also writes a per-splat float32 "observed" weight for the viewer to blend with.

Mapping is the one fourd.html uses for corridor-proj / corridor-exact:
    world = Rx(pi) * (scale0 * native_sfm),  scale0 = height / bodyH
where bodyH is the y extent of person/frame_000.ply and height the ?height= of the preset
(0.828 for corridor-free on recon-f0-v2). Raw Marble coordinates are what the spz stores; the
viewer applies Rx(pi) itself, so outputs are written in raw Marble coordinates.

World placement, general form:  raw = Rx(pi) * (scale0 * inv(c2w[origin]) * native)
(--origin-frame is the source frame whose camera is the Marble origin, e.g. 549 for recon-f131).

Sources: the corridor wvp container (cameras + depth + person masks), or a plain cameras.json
(Pi3X / SfM, OpenGL camera_to_world + source_intrinsics) with depth z-buffered per camera from a
static point cloud (--scene-points, e.g. built from Pi3X anchors with --anchors) or read from
--depth-dir (depth_{sourceIndex:04d}.npy, native units), and imagery from --clip (by sourceIndex)
or --frames-dir/--frames-pattern (by sample id, e.g. LaMa clean frames, then no mask is needed).

Usage:
  python scripts/bake_video_colours.py --diag            # depth-ratio check on a few frames
  python scripts/bake_video_colours.py                   # full corridor bake (recon-f0-v2)
  python scripts/bake_video_colours.py --world public/marble-corridor-recon-f131.spz --origin-frame 549 --height 0.7965 --out public/marble-corridor-f131-baked
  python scripts/bake_video_colours.py --cameras public/worlds/tos-hall-walk-4d/cameras.json --anchors .context/tos/pi3x/anchors.npz \
      --frames-dir .context/tos/clean-frames --frames-pattern 'f_{sample:04d}.png' --world public/marble-tos-hall-walk-clean.spz \
      --person public/worlds/tos-hall-walk-4d/person/frame_000.ply --height 2.0007 --out public/marble-tos-baked
"""
import argparse
import gzip
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sfm_frame import origin_c2w as levelled_origin, describe as describe_frame

ROOT = Path(__file__).resolve().parents[1]
SH_C0 = 0.28209479177387814
SPZ_COLOR_SCALE = 0.15


# ---------------------------------------------------------------- spz v2 (shDegree 0) ----
def read_spz(path: Path):
    raw = gzip.open(path).read()
    magic, ver, n, sh, fb, flags, _ = struct.unpack('<IIIBBBB', raw[:16])
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
    alpha = alpha_b.astype(np.float32) / 255.0                       # sigmoid(opacity)
    f_dc = (col_b.astype(np.float32) / 255.0 - 0.5) / SPZ_COLOR_SCALE
    log_scale = scale_b.astype(np.float32) / 16.0 - 10.0
    q = (rot_b.astype(np.float32) - 127.5) / 127.5                     # xyz, w >= 0
    w = np.sqrt(np.clip(1.0 - (q * q).sum(1), 0, 1))
    quat = np.concatenate([q, w[:, None]], 1)                          # xyzw
    return dict(raw=raw, n=n, fb=fb, xyz=xyz, alpha=alpha, f_dc=f_dc, log_scale=log_scale, quat=quat,
                col_b=col_b, header=raw[:16], pos_b=pos_b, alpha_b=alpha_b, scale_b=scale_b, rot_b=rot_b)


def write_spz_recolour(spz, rgb01: np.ndarray, out: Path):
    """Same packed bytes as the input, only the colour stream replaced (SH0, quantised)."""
    f_dc = (rgb01 - 0.5) / SH_C0
    cb = np.clip(np.round(f_dc * (SPZ_COLOR_SCALE * 255.0) + 127.5), 0, 255).astype(np.uint8)
    body = spz['header'] + spz['pos_b'].tobytes() + spz['alpha_b'].tobytes() + cb.tobytes() + spz['scale_b'].tobytes() + spz['rot_b'].tobytes()
    with gzip.open(out, 'wb', compresslevel=6) as f:
        f.write(body)


def write_gaussian_ply(spz, rgb01: np.ndarray, out: Path):
    n = spz['n']
    f_dc = ((rgb01 - 0.5) / SH_C0).astype(np.float32)
    a = np.clip(spz['alpha'], 1e-4, 1 - 1e-4)
    opacity = np.log(a / (1 - a)).astype(np.float32)
    q = spz['quat']
    rows = np.zeros((n, 17), np.float32)
    rows[:, 0:3] = spz['xyz']
    rows[:, 6:9] = f_dc
    rows[:, 9] = opacity
    rows[:, 10:13] = spz['log_scale']
    rows[:, 13] = q[:, 3]; rows[:, 14:17] = q[:, 0:3]                 # rot_0 = w, rot_1..3 = xyz
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n + ''.join('property float %s\n' % s for s in names) + 'end_header\n'
    with open(out, 'wb') as f:
        f.write(header.encode()); f.write(rows.tobytes())


# ---------------------------------------------------------------- wvp ----
def read_wvp(path: Path):
    raw = gzip.open(path).read()
    assert raw[:4] == b'WVP1'
    hl = struct.unpack('<I', raw[4:8])[0]
    H = json.loads(raw[8:8 + hl])
    W, Hh, L, F = H['width'], H['height'], H['depthFrames'], H['frames']
    o = 8 + hl
    depth_rgba = np.frombuffer(raw, np.uint8, L * Hh * W * 4, o).reshape(L, Hh, W, 4); o += L * Hh * W * 4
    masks = np.frombuffer(raw, np.uint8, F * Hh * W, o).reshape(F, Hh, W); o += F * Hh * W
    assert o == len(raw)
    q = (depth_rgba[..., 0].astype(np.uint32) | (depth_rgba[..., 1].astype(np.uint32) << 8)).astype(np.float32) / 65535.0
    near, far = H['depthRange']['near'], H['depthRange']['far']
    depth = 1.0 / (1.0 / far + (1.0 / near - 1.0 / far) * q)           # native units, row 0 = image top
    supported = depth_rgba[..., 3] > 64
    c2w = np.array(H['cameras'], np.float64).reshape(F, 4, 4)
    return H, depth.astype(np.float32), supported, masks, c2w


def body_height(ply: Path) -> float:
    b = ply.read_bytes()
    hi = b.index(b'end_header\n') + 11
    head = b[:hi].decode()
    props = [l.split()[2] for l in head.split('\n') if l.startswith('property')]
    n = int([l for l in head.split('\n') if l.startswith('element vertex')][0].split()[2])
    d = np.frombuffer(b, np.float32, n * len(props), hi).reshape(n, len(props))
    y = d[:, props.index('y')]
    return float(y.max() - y.min())


def video_frames(clip: Path, w: int, h: int):
    cmd = ['ffmpeg', '-v', 'error', '-i', str(clip), '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    nbytes = w * h * 3
    while True:
        buf = p.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def smoothstep(e0, e1, x):
    t = torch.clamp((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def splat_normals(quat: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
    """Axis of the smallest scale, in raw Marble coordinates."""
    x, y, z, w = quat.T
    cols = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1),
        np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1),
        np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)], 1)   # (n, axis, xyz)
    k = np.argmin(log_scale, 1)
    return cols[np.arange(len(k)), k].astype(np.float32)



def clip_frame_reader(clip: Path, w: int, h: int):
    """Random access by source index over a sequential decode (frames are requested in order)."""
    gen = video_frames(clip, w, h); state = {'i': -1, 'img': None}
    def get(i):
        while state['i'] < i:
            state['img'] = next(gen); state['i'] += 1
        return state['img']
    return get


class WvpSource:
    def __init__(self, wvp: Path, clip: Path, size):
        self.H, self.depth, self.supported, self.masks, self.c2w = read_wvp(wvp)
        H = self.H; K = H['intrinsics']
        self.K = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1]], np.float32)
        self.size = (H['source']['width'], H['source']['height'])
        self.frames = list(range(H['frames']))
        self.near = H['depthRange']['near']
        self.read = clip_frame_reader(clip, *size)
    def cam(self, f): return self.c2w[f], self.K, self.size
    def depth_map(self, f):
        dl = self.H['depthIndex'][f]; return self.depth[dl], self.supported[dl]
    def mask(self, f): return self.masks[f].astype(np.float32) / 255.0
    def image(self, f): return self.read(f)
    def sample_of(self, f): return f


def zbuffer_depth(points: np.ndarray, c2w: np.ndarray, K: np.ndarray, size, res, fill=True):
    """Nearest-point depth per texel for an OpenGL camera; holes nearest-filled and flagged unsupported."""
    from scipy import ndimage
    w2c = np.linalg.inv(c2w).astype(np.float32)
    C = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = -C[:, 2]; ok = z > 1e-3
    u = (K[0, 0] * C[:, 0] / np.where(ok, z, 1) + K[0, 2]) * res[0] / size[0]
    v = (-K[1, 1] * C[:, 1] / np.where(ok, z, 1) + K[1, 2]) * res[1] / size[1]
    ui = np.floor(u).astype(np.int64); vi = np.floor(v).astype(np.int64)
    ok &= (ui >= 0) & (ui < res[0]) & (vi >= 0) & (vi < res[1])
    d = np.full(res[1] * res[0], np.inf, np.float32)
    np.minimum.at(d, vi[ok] * res[0] + ui[ok], z[ok])
    d = d.reshape(res[1], res[0]); sup = np.isfinite(d)
    big = np.where(sup, d, np.inf)
    closed = ndimage.minimum_filter(big, size=3)           # close 1-texel gaps toward the nearer surface
    d = np.where(sup, d, closed); sup2 = np.isfinite(d)
    if fill and sup2.any() and not sup2.all():
        idx = ndimage.distance_transform_edt(~sup2, return_distances=False, return_indices=True)
        d = d[tuple(idx)]
    d = np.where(np.isfinite(d), d, 1.0)
    return d.astype(np.float32), sup


class JsonSource:
    def __init__(self, cameras: Path, size, depth_res, clip=None, frames_dir=None, frames_pattern=None, scene_points=None, depth_dir=None, mask_dir=None):
        cj = json.load(open(cameras)); self.cams = cj['cameras']
        self.frames = [c['sourceIndex'] for c in self.cams]
        self.by_idx = {c['sourceIndex']: (i, c) for i, c in enumerate(self.cams)}
        self.depth_res = depth_res; self.scene = scene_points; self.depth_dir = depth_dir; self.mask_dir = mask_dir
        self.frames_dir, self.pattern = frames_dir, frames_pattern
        self.read = clip_frame_reader(clip, *size) if clip else None
        self.vsize = size; self.near = 0.05
        self._cache = {}
    def sample_of(self, f): return self.by_idx[f][0]
    def cam(self, f):
        c = self.by_idx[f][1]
        return np.array(c['camera_to_world'], np.float64), np.array(c['source_intrinsics'], np.float32), tuple(c['source_image_size'])
    def depth_map(self, f):
        if self.depth_dir:
            d = np.load(self.depth_dir / f'depth_{f:04d}.npy').astype(np.float32); return d, np.isfinite(d) & (d > 0)
        c2w, K, size = self.cam(f)
        return zbuffer_depth(self.scene, c2w, K, size, self.depth_res)
    def mask(self, f):
        if not self.mask_dir: return None
        from PIL import Image
        m = np.asarray(Image.open(self.mask_dir / self.pattern.format(sample=self.sample_of(f), idx=f)).convert('L'), np.float32) / 255.0
        return m
    def image(self, f):
        if self.read: return self.read(f)
        from PIL import Image
        im = Image.open(self.frames_dir / self.pattern.format(sample=self.sample_of(f), idx=f)).convert('RGB').resize(self.vsize, Image.BILINEAR)
        return np.asarray(im)


def static_cloud_from_anchors(anchors: Path, cameras: Path, anchor_samples, conf_pct=20.0):
    """Pi3X dense anchor frames -> one static point cloud in the cameras.json (OpenGL, cam0 = identity) frame.
    Anchor poses are OpenCV; each anchor's points go through its own camera (local depth is what Pi3X
    measured), so only the world scale needs fitting (anchor positions are near-collinear on a dolly)."""
    a = np.load(anchors); cams = json.load(open(cameras))['cameras']
    poses = a['poses'].astype(np.float64); pub = np.array([cams[i]['camera_to_world'] for i in anchor_samples], np.float64)
    ap, pp = poses[:, :3, 3], pub[:, :3, 3]
    S, D = ap - ap.mean(0), pp - pp.mean(0)
    scale = float(np.sqrt((D ** 2).sum() / (S ** 2).sum()))
    keep = a['valid'] & ~a['people'] & (a['conf'] > np.percentile(a['conf'], conf_pct))
    F = np.diag([1.0, -1.0, -1.0])                                    # OpenCV camera -> OpenGL camera
    out = []
    for i in range(len(poses)):
        P = a['points'][i][keep[i]].astype(np.float64)
        C = (P - poses[i][:3, 3]) @ poses[i][:3, :3]                  # into the anchor's OpenCV camera
        C = C[C[:, 2] > 0] * scale
        W = (C @ F) @ pub[i][:3, :3].T + pub[i][:3, 3]                # OpenGL camera -> public world
        out.append(W.astype(np.float32))
    pts = np.concatenate(out)
    print(f'static cloud: {len(pts)} points from {len(poses)} anchors, scale {scale:.4f}, bbox {pts.min(0).round(2)} .. {pts.max(0).round(2)}', flush=True)
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default=ROOT / 'public/marble-corridor-recon-f0-v2.spz', type=Path)
    ap.add_argument('--wvp', default=None, type=Path, help='corridor wvp container (default when --cameras is absent)')
    ap.add_argument('--cameras', default=None, type=Path, help='cameras.json (OpenGL camera_to_world, source_intrinsics, source_image_size)')
    ap.add_argument('--clip', default=None, type=Path, help='source clip, frames by sourceIndex')
    ap.add_argument('--frames-dir', default=None, type=Path, help='image per registered camera instead of --clip')
    ap.add_argument('--frames-pattern', default='f_{sample:04d}.png', help='{sample} = camera index in cameras.json, {idx} = sourceIndex')
    ap.add_argument('--mask-dir', default=None, type=Path, help='optional person masks, same pattern (white = person)')
    ap.add_argument('--scene-points', default=None, type=Path, help='static point cloud .npy (N,3) in the cameras.json frame for per-camera depth')
    ap.add_argument('--anchors', default=None, type=Path, help='Pi3X anchors.npz -> static point cloud (cached next to it as static-scene.npy)')
    ap.add_argument('--anchor-samples', default='0,11,21,32,43,54,64,75')
    ap.add_argument('--depth-dir', default=None, type=Path, help='per-frame depth_{sourceIndex:04d}.npy instead of a point cloud')
    ap.add_argument('--depth-res', type=int, nargs=2, default=[480, 200], help='z-buffer resolution for point-cloud depth')
    ap.add_argument('--person', default=ROOT / 'public/worlds/overnight-corridor-video-projection/person/frame_000.ply', type=Path)
    ap.add_argument('--height', type=float, default=0.828, help='person height in world units (the preset\'s ?height=)')
    ap.add_argument('--scale0', type=float, default=None, help='override scale0 (native -> world)')
    ap.add_argument('--origin-frame', type=int, default=0, help='source frame whose camera is the Marble origin')
    ap.add_argument('--stride', type=int, default=1, help='use every Nth registered frame')
    ap.add_argument('--size', type=int, nargs=2, default=[960, 540], help='decode size for the imagery')
    ap.add_argument('--tol', type=float, default=0.25, help='occlusion tolerance: reject when splat/map depth > 1+tol (soft to 1+2tol)')
    ap.add_argument('--front', type=float, default=0.35, help='down-weight splats nearer than the map by more than this ratio (0 = accept all, like the viewer)')
    ap.add_argument('--feather', type=float, default=0.04)
    ap.add_argument('--unsupported', type=float, default=0.3, help='weight for texels the depth map does not support')
    ap.add_argument('--min-alpha', type=float, default=0.05, help='skip near-transparent splats')
    ap.add_argument('--count-sat', type=int, default=20, help='frame count at which coverage saturates')
    ap.add_argument('--std-max', type=float, default=0.25, help='colour std across frames at which consistency reaches 0')
    ap.add_argument('--min-count', type=int, default=3)
    ap.add_argument('--out', default=ROOT / 'public/marble-corridor-baked', type=Path, help='output stem')
    ap.add_argument('--share', default=Path(__file__).resolve().parents[1] / '.context/share', type=Path)
    ap.add_argument('--tag', default='bake', help='figure name prefix under --share')
    ap.add_argument('--fig-frames', default=None, help='two source frames for the figures, e.g. 0,275 (default first and middle)')
    ap.add_argument('--diag', action='store_true', help='depth-ratio diagnostic only')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.device)
    a.share.mkdir(parents=True, exist_ok=True)

    spz = read_spz(a.world)
    n = spz['n']
    print(f'world: {a.world.name}, {n} splats, bbox {spz["xyz"].min(0)} .. {spz["xyz"].max(0)}', flush=True)
    VW, VH = a.size
    if a.cameras:
        scene = None
        if a.depth_dir is None:
            if a.scene_points:
                scene = np.load(a.scene_points)
            elif a.anchors:
                cache = a.anchors.with_name('static-scene.npy')
                if cache.exists():
                    scene = np.load(cache)
                else:
                    scene = static_cloud_from_anchors(a.anchors, a.cameras, [int(x) for x in a.anchor_samples.split(',')]); np.save(cache, scene)
            else:
                sys.exit('need --scene-points, --anchors or --depth-dir with --cameras')
        src = JsonSource(a.cameras, (VW, VH), tuple(a.depth_res), clip=a.clip, frames_dir=a.frames_dir, frames_pattern=a.frames_pattern, scene_points=scene, depth_dir=a.depth_dir, mask_dir=a.mask_dir)
    else:
        src = WvpSource(a.wvp or ROOT / 'public/worlds/overnight-corridor-video-projection/video-projection.wvp', a.clip or ROOT / 'public/clips/corridor-walk-elder.mp4', (VW, VH))
    bodyH = body_height(a.person)
    scale0 = a.scale0 or a.height / bodyH
    if a.cameras:
        # the same origin the packagers use: camera 0 at the origin, gravity on +y when the run's
        # framealign.json exists beside cameras.json (scripts/sfm_frame.py). Re-anchoring on camera
        # 0's own axes here would undo the levelling and compare depths along tilted rays.
        origin_c2w, fa = levelled_origin(a.cameras, a.origin_frame)
        print(f'frame: {describe_frame(fa)}', flush=True)
    else:
        origin_c2w, _, _ = src.cam(a.origin_frame)
    # raw = Rx(pi) * scale0 * inv(origin) * native   ->  native = origin * (Rx(pi) raw / scale0)
    flip = np.diag([1.0, -1.0, -1.0])
    A = origin_c2w[:3, :3] @ flip / scale0; b = origin_c2w[:3, 3]
    print(f'source: {len(src.frames)} registered frames; bodyH {bodyH:.4f} native -> scale0 {scale0:.4f}; origin frame {a.origin_frame} at {b.round(3)}', flush=True)

    P = torch.from_numpy((spz['xyz'] @ A.T + b).astype(np.float32)).to(dev)
    Nrm = torch.from_numpy((splat_normals(spz['quat'], spz['log_scale']) @ (origin_c2w[:3, :3] @ flip).T).astype(np.float32)).to(dev)
    alpha = torch.from_numpy(spz['alpha']).to(dev)
    live = alpha > a.min_alpha

    sum_rgb = torch.zeros(n, 3, device=dev); sum_rgb2 = torch.zeros(n, 3, device=dev)
    sum_w = torch.zeros(n, device=dev)
    count = torch.zeros(n, device=dev, dtype=torch.int32)
    ratio_log = []
    frames = src.frames[::a.stride]
    fig_frames = [int(x) for x in a.fig_frames.split(',')] if a.fig_frames else [src.frames[0], src.frames[len(src.frames) // 2]]
    diag_frames = [src.frames[int(k * (len(src.frames) - 1))] for k in (0, 0.2, 0.5, 0.75, 1.0)] if a.diag else None
    src_imgs = {}
    for f in (f for f in src.frames if f in frames or f in fig_frames):
        img = src.image(f)
        if f in fig_frames:
            src_imgs[f] = img
        if f not in frames or (a.diag and f not in diag_frames):
            continue
        c2w, K, (SW, SH_) = src.cam(f)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        w2c = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(dev)
        cam = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = -cam[:, 2]
        u = cx + fx * cam[:, 0] / z
        v = cy - fy * cam[:, 1] / z
        vis = live & (z > src.near * 0.5) & (u >= 0) & (u < SW) & (v >= 0) & (v < SH_)
        idx = torch.nonzero(vis).squeeze(1)
        if idx.numel() == 0:
            continue
        uu, vv, zz = u[idx], v[idx], z[idx]
        dmap, dsup = src.depth_map(f); DH, DW = dmap.shape
        depth_t = torch.from_numpy(np.ascontiguousarray(dmap)).to(dev).reshape(-1)
        supp_t = torch.from_numpy(np.ascontiguousarray(dsup)).to(dev).reshape(-1)
        ti = torch.clamp((uu * (DW / SW)).long(), 0, DW - 1)
        tj = torch.clamp((vv * (DH / SH_)).long(), 0, DH - 1)
        flat = tj * DW + ti
        md = depth_t[flat]; sp = supp_t[flat]
        m = src.mask(f)
        if m is None:
            mk = torch.zeros_like(uu)
        else:
            MH, MW = m.shape; mt = torch.from_numpy(np.ascontiguousarray(m)).to(dev).reshape(-1)
            mk = mt[torch.clamp((vv * (MH / SH_)).long(), 0, MH - 1) * MW + torch.clamp((uu * (MW / SW)).long(), 0, MW - 1)]
        r = zz / torch.clamp(md, 1e-4)
        if a.diag:
            rr = r[sp & (mk < 0.5)]
            lr = torch.log(rr).cpu().numpy()
            ratio_log.append((f, lr))
            print(f'frame {f}: {idx.numel()} splats in frustum, {sp.float().mean().item() * 100:.0f}% on supported depth, log(splat/map depth) median {np.median(lr):+.3f} (ratio {np.exp(np.median(lr)):.3f}), p10 {np.exp(np.percentile(lr, 10)):.3f} p90 {np.exp(np.percentile(lr, 90)):.3f}', flush=True)
            continue
        un = uu / SW; vn = vv / SH_
        e = smoothstep(0, a.feather, un) * smoothstep(0, a.feather, 1 - un) * smoothstep(0, a.feather, vn) * smoothstep(0, a.feather, 1 - vn)
        w = e * (1 - smoothstep(0.25, 0.75, mk))
        occ = 1 - smoothstep(1 + a.tol, 1 + 2 * a.tol, r)
        if a.front > 0:
            occ = occ * smoothstep(1 - 2 * a.front, 1 - a.front, r)
        w = w * torch.where(sp, occ, torch.full_like(occ, a.unsupported))
        cen = 1 - 0.5 * torch.sqrt((un - 0.5) ** 2 * 4 + (vn - 0.5) ** 2 * 4).clamp(0, 1)
        camPos = torch.from_numpy(c2w[:3, 3].astype(np.float32)).to(dev)
        view = P[idx] - camPos; view = view / torch.clamp(view.norm(dim=1, keepdim=True), 1e-6)
        ang = torch.clamp((Nrm[idx] * view).sum(1).abs(), 0.2, 1)
        w = w * cen * ang / torch.clamp(zz, 0.5)
        pi = torch.clamp((uu * (VW / SW)).long(), 0, VW - 1)
        pj = torch.clamp((vv * (VH / SH_)).long(), 0, VH - 1)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).to(dev).reshape(-1, 3).float() / 255.0
        pix = img_t[pj * VW + pi]
        sum_rgb.index_add_(0, idx, pix * w[:, None])
        sum_rgb2.index_add_(0, idx, pix * pix * w[:, None])
        sum_w.index_add_(0, idx, w)
        count.index_add_(0, idx, (w > 0.02).int())
        if frames.index(f) % 50 == 0:
            print(f'frame {f}: {idx.numel()} in frustum, {(w > 0.02).sum().item()} weighted, {time.time() - t0:.0f}s', flush=True)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if a.diag:
        fig, ax = plt.subplots(figsize=(8, 4))
        for f, lr in ratio_log:
            ax.hist(lr, bins=120, range=(-1.5, 1.5), histtype='step', label=f'frame {f}')
        ax.axvline(0, color='k', lw=0.8); ax.set_xlabel('log(Marble splat depth / source depth)  (0 = agree)'); ax.legend()
        ax.set_title(f'{a.world.name}, scale0 {scale0:.3f}: Marble geometry vs source depth along the same rays')
        fig.tight_layout(); fig.savefig(a.share / f'{a.tag}-depth-ratio.png', dpi=110)
        print(f'wrote {a.share / (a.tag + "-depth-ratio.png")}'); return

    sum_rgb = sum_rgb.cpu().numpy(); sum_rgb2 = sum_rgb2.cpu().numpy(); sum_w = sum_w.cpu().numpy(); count = count.cpu().numpy()
    observed = (count >= a.min_count) & (sum_w > 1e-3)
    marble_rgb = np.clip(0.5 + SH_C0 * spz['f_dc'], 0, 1)
    baked = marble_rgb.copy()
    baked[observed] = np.clip(sum_rgb[observed] / sum_w[observed, None], 0, 1)
    # per-splat colour spread across the frames that saw it: a splat on the true surface sees the
    # same colour from every frame; a misprojected one sees whatever passes behind it and averages to mud
    mean = sum_rgb / np.maximum(sum_w, 1e-6)[:, None]
    var = np.clip(sum_rgb2 / np.maximum(sum_w, 1e-6)[:, None] - mean * mean, 0, None).mean(1)
    std = np.sqrt(var)
    conf = np.where(observed, np.clip(count / a.count_sat, 0, 1) * np.clip(1 - std / a.std_max, 0, 1), 0).astype(np.float32)
    frac = observed.mean()
    print(f'colour std across frames, observed splats: median {np.median(std[observed]):.3f}, p90 {np.percentile(std[observed], 90):.3f}; {100 * (std[observed] < 0.1).mean():.0f}% under 0.1', flush=True)
    print(f'observed: {observed.sum()} / {n} splats ({100 * frac:.1f}%), of live {100 * observed[spz["alpha"] > a.min_alpha].mean():.1f}%; median count among observed {np.median(count[observed]):.0f}; runtime {time.time() - t0:.0f}s', flush=True)

    out = a.out
    write_gaussian_ply(spz, baked, out.with_suffix('.ply'))
    write_spz_recolour(spz, baked, out.with_suffix('.spz'))
    conf.tofile(str(out) + '-weights.bin')
    np.savez_compressed(str(out) + '-bake.npz', sum_rgb=sum_rgb.astype(np.float32), sum_w=sum_w.astype(np.float32), count=count.astype(np.int16), scale0=scale0, origin_frame=a.origin_frame)
    print(f'wrote {out.with_suffix(".ply")} ({out.with_suffix(".ply").stat().st_size / 1e6:.0f} MB), {out.with_suffix(".spz")} ({out.with_suffix(".spz").stat().st_size / 1e6:.0f} MB), {str(out) + "-weights.bin"}', flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(count[count > 0], bins=60, range=(0, count.max() + 1), color='#4a7ebb')
    ax[0].set_xlabel('source frames observing the splat (count > 0 only)'); ax[0].set_ylabel('splats'); ax[0].set_yscale('log')
    ax[0].set_title(f'{100 * (count > 0).mean():.1f}% of {n} splats seen by >= 1 frame, {100 * frac:.1f}% observed (>= {a.min_count} frames)')
    ax[1].hist(std[observed], bins=50, range=(0, 0.4), color='#c46a2a')
    ax[1].set_xlabel('colour std across observing frames (0 = every frame agrees)'); ax[1].set_ylabel('splats'); ax[1].set_title(f'weights.bin = min(1, count/{a.count_sat}) * max(0, 1 - std/{a.std_max})')
    fig.tight_layout(); fig.savefig(a.share / f'{a.tag}-coverage.png', dpi=110)

    # crude point renders: nearest splat per pixel (Marble vs baked) at two recorded poses and off-path
    Pn = spz['xyz'] @ A.T + b
    def render(rgb, f, offset=(0, 0, 0), res=(480, 270)):
        c2w, K, (SW, SH_) = src.cam(f); c = c2w.copy(); c[:3, 3] += c[:3, :3] @ np.array(offset)
        w2c = np.linalg.inv(c).astype(np.float32)
        cam = Pn @ w2c[:3, :3].T + w2c[:3, 3]
        z = -cam[:, 2]; ok = (z > 0.1) & (spz['alpha'] > 0.3)
        zs = np.where(ok, z, 1)
        uu = (K[0, 2] + K[0, 0] * cam[:, 0] / zs) * res[0] / SW; vv = (K[1, 2] - K[1, 1] * cam[:, 1] / zs) * res[1] / SH_
        ok &= (uu >= 0) & (uu < res[0]) & (vv >= 0) & (vv < res[1])
        ii = np.nonzero(ok)[0]; pix = vv[ii].astype(int) * res[0] + uu[ii].astype(int)
        order = np.argsort(-z[ii]); pix = pix[order]; ii = ii[order]
        img = np.zeros((res[1] * res[0], 3), np.float32); img[pix] = rgb[ii]
        return img.reshape(res[1], res[0], 3)
    from PIL import Image
    for f in fig_frames:
        s_img = np.asarray(Image.fromarray(src_imgs[f]).resize((480, 270), Image.BILINEAR)).astype(np.float32) / 255
        errs = []
        for rgb in (marble_rgb, baked):
            r = render(rgb, f); cov = r.sum(2) > 0
            errs.append(np.abs(r - s_img)[cov].mean())
        print(f'frame {f}: mean |render - source| over covered pixels: Marble {errs[0]:.3f}, baked {errs[1]:.3f}', flush=True)
    f0, f1 = fig_frames
    _, _, (SW, SH_) = src.cam(f0)
    side = 0.5 / (bodyH / 1.7)              # 0.5 m in native units, the person being 1.7 m
    rows = [(f0, (0, 0, 0), f'frame {f0} pose'), (f1, (0, 0, 0), f'frame {f1} pose'), (f1, (side, 0, 0), f'frame {f1} + 0.5 m right (off path)')]
    fig, ax = plt.subplots(len(rows), 3, figsize=(15, 3 * len(rows) * SH_ / SW * 16 / 9))
    for i, (f, off, label) in enumerate(rows):
        ax[i, 0].imshow(src_imgs[f]); ax[i, 0].set_title(f'source frame {f}' + ('' if off == (0, 0, 0) else ' (for reference)'))
        ax[i, 1].imshow(render(marble_rgb, f, off)); ax[i, 1].set_title(f'Marble colour, {label}')
        ax[i, 2].imshow(render(baked, f, off)); ax[i, 2].set_title(f'baked colour, {label}')
        for j in range(3): ax[i, j].axis('off')
    fig.tight_layout(); fig.savefig(a.share / f'{a.tag}-before-after.png', dpi=100)
    print(f'wrote {a.share / (a.tag + "-coverage.png")}, {a.share / (a.tag + "-before-after.png")}', flush=True)


if __name__ == '__main__':
    main()
