"""CPU endpoint RGB exposure over Gaussian footprints in recorded source views.

This reproduces the retained CPU audit approximation, not gsplat CUDA gradients or
physical visibility: center depth > .05 / scale, opacity > .02, a 64-pixel center
margin, .3 pixel covariance variance, integer-truncated 3-sigma radii clamped to
[1, 256], power >= -8, alpha >= .004 capped at .99, and transmittance >= 1e-3.
Every retained scene splat participates in front-to-back occlusion. Exposure is
summed alpha-composited weight, not a count of centers, pixels, or depth support.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from numba import njit

from render_world_poses import FLIP, quat_to_rot


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _poses(value, shape, name):
    array = _numeric(value, shape, name).astype(np.float64)
    if not np.allclose(array[..., 3, :], [0, 0, 0, 1], rtol=0, atol=1e-7):
        raise ValueError(f"{name} must contain homogeneous poses")
    try:
        inverse = np.linalg.inv(array)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be invertible") from exc
    if not np.isfinite(inverse).all():
        raise ValueError(f"{name} must have finite inverses")
    with np.errstate(over="ignore"):
        if not np.isfinite(inverse.astype(np.float32)).all():
            raise ValueError(f"{name} inverse exceeds finite float32 projection range")
    return array


def _validate(world, c2w_gl, intrinsics, masks, origin_c2w, scale, frame_indices, progress):
    if not isinstance(world, Mapping):
        raise ValueError("world must be a mapping from read_spz")
    n = world.get("n")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
        raise ValueError("world n must be a positive integer")
    arrays = {}
    for key, shape in (
        ("xyz", (n, 3)),
        ("alpha", (n,)),
        ("scale", (n, 3)),
        ("quat", (n, 4)),
        ("rgb", (n, 3)),
    ):
        if key not in world:
            raise ValueError(f"world is missing {key}")
        arrays[key] = _numeric(world[key], shape, f"world {key}")
    for key in ("alpha", "rgb"):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            raise ValueError(f"world {key} must lie in [0, 1]")
    if np.any(arrays["scale"] <= 0):
        raise ValueError("world scale must be positive")
    # SPZ quantization can make the stored quaternion slightly longer than one.
    norms = np.linalg.norm(arrays["quat"], axis=1)
    if np.any(abs(norms - 1) > 0.02):
        raise ValueError("world quat must be near unit length (SPZ xyzw order)")
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real):
        raise ValueError("scale must be finite and positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    poses = np.asarray(c2w_gl)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("c2w_gl must have shape (F, 4, 4) with F > 0")
    poses = _poses(poses, poses.shape, "c2w_gl")
    origin = _poses(origin_c2w, (4, 4), "origin_c2w")
    ks = _numeric(intrinsics, (len(poses), 3, 3), "intrinsics")
    if np.any(ks[:, 0, 0] <= 0) or np.any(ks[:, 1, 1] <= 0):
        raise ValueError("intrinsics focal lengths must be positive")
    if (
        not np.allclose(ks[:, 2, :], [0, 0, 1], rtol=0, atol=1e-7)
        or np.any(ks[:, 0, 1] != 0)
        or np.any(ks[:, 1, 0] != 0)
    ):
        raise ValueError("intrinsics must be standard zero-skew pinhole matrices")
    try:
        frames = list(frame_indices)
    except TypeError as exc:
        raise ValueError("frame_indices must be an explicit integer sequence") from exc
    if not frames or any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in frames
    ):
        raise ValueError("frame_indices must be nonempty nonboolean integers")
    if len(set(frames)) != len(frames) or any(i < 0 or i >= len(poses) for i in frames):
        raise ValueError("frame_indices must be unique and in camera range")
    if not isinstance(masks, Mapping):
        raise ValueError("masks must map selected frame indices to bool images")
    selected_masks = {}
    shape = None
    for frame in frames:
        if frame not in masks:
            raise ValueError(f"missing mask for selected frame {frame}")
        mask = np.asarray(masks[frame])
        if mask.dtype != np.bool_ or mask.ndim != 2 or min(mask.shape) == 0:
            raise ValueError(f"mask for frame {frame} must be a nonempty 2D bool image")
        if shape is not None and mask.shape != shape:
            raise ValueError("selected masks must have identical dimensions")
        shape = mask.shape
        selected_masks[frame] = np.ascontiguousarray(mask)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    return arrays, poses, ks, selected_masks, origin, np.asarray(frames, dtype=np.int64)


def _project(native, covariance, alpha, pose, intrinsics, width, height, scale):
    w2c = np.linalg.inv(pose).astype(np.float32)
    cam = native @ w2c[:3, :3].T + w2c[:3, 3]
    p = cam @ FLIP.astype(np.float32)
    z = p[:, 2]
    ids = np.flatnonzero((z > 0.05 / scale) & (alpha > 0.02))
    p, z = p[ids], z[ids]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u, v = fx * p[:, 0] / z + cx, fy * p[:, 1] / z + cy
    on = (u > -64) & (u < width + 64) & (v > -64) & (v < height + 64)
    ids, p, z, u, v = ids[on], p[on], z[on], u[on], v[on]
    wm = FLIP.astype(np.float32) @ w2c[:3, :3]
    sigma = np.einsum("ij,njk,lk->nil", wm, covariance[ids], wm)
    j = np.zeros((len(ids), 2, 3), np.float32)
    j[:, 0, 0], j[:, 0, 2] = fx / z, -fx * p[:, 0] / z**2
    j[:, 1, 1], j[:, 1, 2] = fy / z, -fy * p[:, 1] / z**2
    c = np.einsum("nij,njk,nlk->nil", j, sigma, j)
    c[:, 0, 0] += 0.3
    c[:, 1, 1] += 0.3
    determinant = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2
    ok = determinant > 1e-8
    ids, u, v, z, c, determinant = (ids[ok], u[ok], v[ok], z[ok], c[ok], determinant[ok])
    radius = np.clip(3 * np.sqrt(np.maximum(c[:, 0, 0], c[:, 1, 1])), 1, 256)
    return (
        ids,
        u.astype(np.float32),
        v.astype(np.float32),
        z,
        radius.astype(np.float32),
        (c[:, 1, 1] / determinant).astype(np.float32),
        (-c[:, 0, 1] / determinant).astype(np.float32),
        (c[:, 0, 0] / determinant).astype(np.float32),
    )


@njit(cache=True, fastmath=False)
def _composite(
    order, ids, u, v, radius, a00, a01, a11, opacity, mask, rendered, masked, clear, clear_views
):
    height, width = mask.shape
    transmittance = np.ones((height, width), np.float32)
    for k in range(len(order)):
        i = order[k]
        sid = ids[i]
        r = int(radius[i])
        x0, x1 = max(int(u[i]) - r, 0), min(int(u[i]) + r + 1, width)
        y0, y1 = max(int(v[i]) - r, 0), min(int(v[i]) + r + 1, height)
        has_clear = False
        for y in range(y0, y1):
            dy = y + 0.5 - v[i]
            for x in range(x0, x1):
                t = transmittance[y, x]
                if t < 1e-3:
                    continue
                dx = x + 0.5 - u[i]
                power = -0.5 * (a00[i] * dx * dx + 2 * a01[i] * dx * dy + a11[i] * dy * dy)
                if power < -8:
                    continue
                alpha = min(0.99, opacity[i] * np.exp(power))
                if alpha < 0.004:
                    continue
                weight = alpha * t
                rendered[sid] += weight
                if mask[y, x]:
                    masked[sid] += weight
                else:
                    clear[sid] += weight
                    has_clear = True
                transmittance[y, x] = t * (1 - alpha)
        if has_clear:
            clear_views[sid] += 1


def measure_source_exposure(
    world, c2w_gl, intrinsics, masks, *, origin_c2w, scale, frame_indices, progress=None
):
    """Return per-SPZ-ID rendered/masked/clear RGB weights and clear-view counts.

    Camera arrays are indexed on the source frame grid, with shapes F×4×4 and
    F×3×3. Selected frames must be unique integers with same-sized boolean masks
    (True excludes RGB supervision). Their order is preserved. The native mapping
    is ``origin_R @ FLIP / scale`` plus ``origin_translation``; input arrays are
    never changed. Float64 mass vectors and int32 clear-view counts have length N.
    A view counts once iff any clear pixel receives positive rendered weight.
    RGB values do not affect visibility. No depth input is needed.

    All inputs are validated before rendering. ``progress(frame, completed, total)``
    runs after each view; its exceptions propagate immediately, allowing cancellation.
    """
    g, poses, ks, selected_masks, origin, frames = _validate(
        world, c2w_gl, intrinsics, masks, origin_c2w, scale, frame_indices, progress
    )
    transform = origin[:3, :3] @ FLIP / scale
    with np.errstate(over="ignore", invalid="ignore"):
        native = (g["xyz"] @ transform.T + origin[:3, 3]).astype(np.float32)
        rotation = quat_to_rot(g["quat"])
        m = np.einsum("ij,njk->nik", transform, rotation) * g["scale"][:, None, :]
        covariance = (m @ m.transpose(0, 2, 1)).astype(np.float32)
    if not np.isfinite(native).all() or not np.isfinite(covariance).all():
        raise ValueError("world transform exceeds finite float32 projection range")
    del rotation, m
    n = len(native)
    result = {
        name: np.zeros(n, np.float64) for name in ("rendered_mass", "masked_mass", "clear_mass")
    }
    result["clear_views"] = np.zeros(n, np.int32)
    result["frame_indices"] = frames
    for completed, frame in enumerate(frames, 1):
        mask = selected_masks[frame]
        height, width = mask.shape
        ids, u, v, z, radius, a00, a01, a11 = _project(
            native, covariance, g["alpha"], poses[frame], ks[frame], width, height, scale
        )
        _composite(
            np.argsort(z).astype(np.int64),
            ids,
            u,
            v,
            radius,
            a00,
            a01,
            a11,
            g["alpha"][ids],
            mask,
            result["rendered_mass"],
            result["masked_mass"],
            result["clear_mass"],
            result["clear_views"],
        )
        if progress is not None:
            progress(int(frame), completed, len(frames))
    return result
