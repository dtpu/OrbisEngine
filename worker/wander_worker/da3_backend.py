"""Track B backend: Depth Anything 3 (ByteDance-Seed, Apache-2.0 up to LARGE).

Multi-view depth + camera pose from unposed frames -> world-space colored .ply
(+ untrained isotropic-gaussian splat.ply + cameras.json). ~12 s for 20 frames on an
M3 Pro (MPS); much faster on CUDA. Install: see worker/README.md (xformers is NOT needed).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DESCRIPTION = "Depth Anything 3 LARGE-1.1: multi-view depth+pose -> world-space .ply (static scenes)"
MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import depth_anything_3  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_model = None


def _load(device):
    global _model
    if _model is None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # pycolmap + torch libomp clash on macOS
        from depth_anything_3.api import DepthAnything3

        _model = DepthAnything3.from_pretrained(MODEL_ID).to(device=device)
    return _model


def unproject_world(depth: np.ndarray, extr: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """depth (N,H,W); extr (N,3,4) world->cam OpenCV; intr (N,3,3) -> world points (N,H,W,3)."""
    N, H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ones = np.ones_like(xs)
    pix = np.stack([xs, ys, ones], -1).reshape(-1, 3)  # (HW,3)
    out = np.empty((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        Kinv = np.linalg.inv(intr[i])
        rays = pix @ Kinv.T  # (HW,3) camera-space directions with z=1
        cam = rays * depth[i].reshape(-1, 1)
        R, t = extr[i, :, :3], extr[i, :, 3]
        world = (cam - t) @ R  # R^T (x - t)
        out[i] = world.reshape(H, W, 3)
    return out


def run(frames: list[Path], out_dir: Path, hfov_deg: float = 60.0, progress=None,
        conf_percentile: float = 30.0, max_points: int = 600_000, use_ray_pose: bool = False,
        mask_people: bool = False, **_) -> dict:
    from .ply import home_distance, write_gaussian_ply, write_point_ply

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda s: None)
    device = _device()
    log(f"loading {MODEL_ID} on {device}")
    model = _load(device)
    log(f"inference on {len(frames)} frames")
    p = model.inference([str(f) for f in frames], use_ray_pose=use_ray_pose)
    depth = np.asarray(p.depth, dtype=np.float32)          # (N,H,W)
    conf = np.asarray(p.conf, dtype=np.float32)            # (N,H,W)
    extr = np.asarray(p.extrinsics, dtype=np.float32)      # (N,3,4) world->cam
    intr = np.asarray(p.intrinsics, dtype=np.float32)      # (N,3,3)
    rgb = np.asarray(p.processed_images)                   # (N,H,W,3) uint8
    N, H, W = depth.shape

    # Re-express everything in frame 0's camera so the viewer's home view is the first frame.
    R0, t0 = extr[0, :, :3], extr[0, :, 3]
    world = unproject_world(depth, extr, intr)
    valid = (depth > 0) & np.isfinite(depth)
    thr = np.percentile(conf[valid], conf_percentile)
    mask = valid & (conf >= thr)
    people_frac = 0.0
    if mask_people:
        from .masks import people_masks

        log("segmenting people (SegFormer-b0)")
        pm = people_masks(rgb)
        people_frac = float(pm.mean())
        mask &= ~pm
        log(f"masked {people_frac*100:.1f}% of pixels as people")
    xyz = world[mask].reshape(-1, 3)
    col = rgb[mask].reshape(-1, 3)
    xyz = xyz @ R0.T + t0  # into cam0 coordinates (OpenCV)
    if xyz.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(xyz.shape[0], max_points, replace=False)
        xyz, col = xyz[idx], col[idx]
    # Normalise scale so the median depth of frame 0 sits ~3 units away (matches the Track A viewer).
    med = float(np.median(depth[0][mask[0]])) if mask[0].any() else 1.0
    s = 3.0 / max(med, 1e-6)
    flip = np.array([1, -1, -1], dtype=np.float32)  # OpenCV -> viewer (+y up, looks down -z)
    xyz_v = (xyz * s * flip).astype(np.float32)
    log(f"{xyz_v.shape[0]} points kept ({mask.mean()*100:.0f}% above conf threshold), scale x{s:.2f}")

    cams = []
    for i in range(N):
        R, t = extr[i, :, :3], extr[i, :, 3]
        c_world = -R.T @ t
        c0 = (R0 @ c_world + t0) * s * flip
        cams.append({"position": c0.tolist(), "K": intr[i].tolist()})

    # Raw DA3 cameras (world->cam OpenCV) and the viewer transform, so other models can be
    # conditioned on these poses and land in the same viewer frame by construction.
    c2w = np.zeros((N, 4, 4), dtype=np.float32); c2w[:, 3, 3] = 1
    for i in range(N):
        R, t = extr[i, :, :3], extr[i, :, 3]
        c2w[i, :3, :3] = R.T; c2w[i, :3, 3] = -R.T @ t
    np.savez(out_dir / "poses.npz", c2w=c2w, w2c=extr, intrinsics=intr, image_size=np.array([W, H]),
             R0=R0, t0=t0, scale=np.float32(s), flip=flip)
    write_point_ply(out_dir / "points.ply", xyz_v, col)
    write_gaussian_ply(out_dir / "splat.ply", xyz_v, col, scale_mult=0.7)
    meta = {
        "backend": "da3", "home_distance": home_distance(xyz_v), "model": MODEL_ID, "device": str(device), "frames": N, "points": int(xyz_v.shape[0]),
        "files": {"points": "points.ply", "splat": "splat.ply", "cameras": "cameras.json"},
        "mask_people": mask_people, "people_fraction": people_frac,
        "note": "world frame = first camera (viewer convention); scale normalised so frame-0 median depth = 3; splat.ply is untrained isotropic gaussians",
    }
    (out_dir / "cameras.json").write_text(json.dumps({"cameras": cams, "image_size": [W, H]}, indent=1))
    (out_dir / "result.json").write_text(json.dumps(meta, indent=2))
    return meta
