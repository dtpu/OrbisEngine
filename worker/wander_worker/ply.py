"""PLY writers: plain colored point clouds and 3D Gaussian Splatting layout.

The 3DGS layout (x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3) is what Spark,
gaussian-splats-3d, antimatter15/splat and SuperSplat all read. Opacity is stored as a
logit and scales as log, matching the reference implementation's activations.
"""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814


def write_point_ply(path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """xyz float (N,3); rgb uint8 or float 0..1 (N,3). Binary little-endian."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def knn_mean_distance(xyz: np.ndarray, k: int = 3, sample: int = 200_000) -> np.ndarray:
    """Mean distance to k nearest neighbours; used to size isotropic gaussians.

    Uses scipy's cKDTree when available, otherwise a chunked brute force on a subsample.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(xyz)
        d, _ = tree.query(xyz, k=k + 1)
        return d[:, 1:].mean(axis=1)
    except Exception:
        n = xyz.shape[0]
        idx = np.random.default_rng(0).choice(n, size=min(sample, n), replace=False)
        ref = xyz[idx]
        out = np.empty(n, dtype=np.float32)
        for s in range(0, n, 4096):
            q = xyz[s : s + 4096]
            d2 = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
            d2.sort(axis=1)
            out[s : s + 4096] = np.sqrt(d2[:, 1 : k + 1]).mean(axis=1)
        return out


def write_gaussian_ply(
    path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    scale: np.ndarray | None = None,
    opacity: float | np.ndarray = 0.95,
    scale_mult: float = 1.0,
) -> None:
    """Point cloud -> isotropic gaussians in 3DGS .ply layout (no training).

    scale: per-point radius in world units; defaults to mean 3-NN distance.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    n = xyz.shape[0]
    if scale is None:
        scale = knn_mean_distance(xyz)
    scale = np.clip(np.asarray(scale, dtype=np.float32) * scale_mult, 1e-5, None)
    op = np.full(n, opacity, dtype=np.float32) if np.isscalar(opacity) else np.asarray(opacity, dtype=np.float32)
    op = np.clip(op, 1e-4, 1 - 1e-4)

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rec = np.zeros(n, dtype=[(f, "<f4") for f in fields])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    dc = (rgb - 0.5) / SH_C0
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    rec["opacity"] = np.log(op / (1 - op))
    ls = np.log(scale)
    rec["scale_0"] = rec["scale_1"] = rec["scale_2"] = ls
    rec["rot_0"] = 1.0  # identity quaternion (w,x,y,z)
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + "".join(
        f"property float {f}\n" for f in fields
    ) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def home_distance(xyz_v: np.ndarray) -> float:
    """Median distance along -z of points inside a narrow cone around camera 0's axis."""
    z = -xyz_v[:, 2]
    r = np.hypot(xyz_v[:, 0], xyz_v[:, 1])
    sel = (z > 0) & (r < 0.35 * z)
    return float(np.median(z[sel])) if sel.sum() > 100 else float(np.median(z[z > 0])) if (z > 0).any() else 3.0
