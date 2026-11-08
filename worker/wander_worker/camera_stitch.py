"""Correct independently inferred camera batches against shared anchor poses.

This is a coordinate correction, not a reconstruction or image-consistency gate.
All inputs must already use one world/camera convention and one length unit.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _times(values, name):
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError(f"{name} must contain at least two finite timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be strictly increasing and distinct")
    return times


def _rigid_poses(values, count, name):
    poses = np.asarray(values, dtype=np.float64)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain {count} finite 4x4 poses")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f"{name} must have homogeneous bottom rows [0, 0, 0, 1]")
    rotations = poses[:, :3, :3]
    if np.any(np.linalg.det(rotations) <= 0):
        raise ValueError(f"{name} must have proper rotations, without reflections")
    u, singular, vt = np.linalg.svd(rotations)
    # Low-precision inference can slightly disturb SO(3), even when stored as
    # float32. This tolerance admits numerical repair, not geometry quality.
    # Never absorb meaningful scale/shear: point-cloud scale belongs to the caller.
    if np.max(np.abs(singular - 1)) > 1.5e-3:
        raise ValueError(f"{name} contains scale or shear beyond rotation roundoff")
    result = poses.copy()
    roundoff = np.max(np.abs(rotations.swapaxes(1, 2) @ rotations - np.eye(3)), axis=(1, 2))
    repair = roundoff > 1e-12
    result[repair, :3, :3] = (u @ vt)[repair]
    result[:, 3] = [0, 0, 0, 1]
    return result


def stitch_camera_batch(batch_c2w, batch_times, anchor_times, canonical_anchor_c2w):
    """Return corrected poses and transforms for existing per-frame world points.

    Each anchor timestamp must occur exactly in ``batch_times``. The canonical
    poses and batch poses must already share a coordinate convention and scale;
    timestamps are recorded source times, not nominal frame ordinals. The input
    arrays are never modified, and no extrapolation outside the anchors is allowed.

    At each anchor, the left correction is ``canonical @ inverse(batch)``.
    Rotation corrections use shortest-path SLERP, and translation corrections
    interpolate linearly in source time. Apply the returned point map to homogeneous
    world points from that same frame: ``X_new = point_map @ X_old``. This preserves
    their original camera-space geometry and scale, including small numerical
    non-rigidity in the original stored camera. Consequently the point maps can
    contain the inverse of that numerical error; they are rigid for rigid inputs.

    A shared static anchor cloud must use one canonical reference similarity,
    rather than these per-frame maps. Recompute derived gravity, placement, depth
    and export artifacts after changing cameras. This function does not establish
    correct geometry, smooth batch seams, or image agreement; validate those
    independently before using the result.
    """
    times = _times(batch_times, "batch_times")
    anchors = _times(anchor_times, "anchor_times")
    if times[0] < anchors[0] or times[-1] > anchors[-1]:
        raise ValueError("batch_times must lie within the anchor timestamp range")
    positions = np.searchsorted(times, anchors)
    if np.any(positions >= len(times)) or not np.array_equal(times[positions], anchors):
        raise ValueError("Every anchor timestamp must occur exactly in batch_times")
    original = np.asarray(batch_c2w, dtype=np.float64)
    batch = _rigid_poses(original, len(times), "batch_c2w")
    canonical = _rigid_poses(canonical_anchor_c2w, len(anchors), "canonical_anchor_c2w")
    if np.array_equal(batch[positions], canonical):
        corrected = batch.copy()
    else:
        corrections = canonical @ np.linalg.inv(batch[positions])
        rotations = Slerp(anchors, Rotation.from_matrix(corrections[:, :3, :3]))(times)
        interpolated = np.broadcast_to(np.eye(4), batch.shape).copy()
        interpolated[:, :3, :3] = rotations.as_matrix()
        for axis in range(3):
            interpolated[:, axis, 3] = np.interp(times, anchors, corrections[:, axis, 3])
        corrected = interpolated @ batch
        # Remove interpolation roundoff at known anchors, without replacing any
        # non-anchor observation or altering its recorded time.
        corrected[positions] = canonical
    point_maps = corrected @ np.linalg.inv(original)
    unchanged = np.all(corrected == original, axis=(1, 2))
    point_maps[unchanged] = np.eye(4)
    return corrected, point_maps
