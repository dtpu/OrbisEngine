"""Source-bound selection and one-time depth registration; no model or provider calls."""

import json
from pathlib import Path

import numpy as np


def validate_registration_option(sample, fixed_world_scale=None):
    if sample is not None:
        if type(sample) is not int or sample < 0:
            raise ValueError("Registration sample must be a nonnegative integer")
        if fixed_world_scale is not None:
            raise ValueError("Registration sample contradicts explicit fixed world scale")


def animation_plan(poses, records, cameras, indices, registration_sample=None):
    """Separate the scale-reference pose from chronological export slots."""
    validate_registration_option(registration_sample)
    if len(poses) != len(indices) or len(records) != len(indices):
        raise ValueError("Pose records must match sampled source indices")
    exports = [i for i, pose in enumerate(poses) if pose is not None]
    if not exports:
        raise ValueError("No usable source pose")
    sample = exports[0] if registration_sample is None else registration_sample
    if sample not in exports or records[sample] is None:
        raise ValueError("Registration sample has no retained pose")
    if cameras is not None:
        if len(cameras) != len(indices) or any(
            c.get("sourceIndex") != int(index) for c, index in zip(cameras, indices)
        ):
            raise ValueError("Registration camera source indices mismatch")
        for i in exports:
            if records[i].get("sample") != i or records[i].get("sourceIndex") != int(indices[i]):
                raise ValueError("Registration pose source indices mismatch")
        camera = np.asarray(cameras[sample]["camera_to_world"], dtype=float)
        if (
            camera.shape != (4, 4)
            or not np.isfinite(camera).all()
            or np.linalg.det(camera[:3, :3]) <= 0
        ):
            raise ValueError("Invalid registration camera")
    elif registration_sample is not None:
        raise ValueError("Explicit registration requires source cameras")
    return sample, exports


def select_depth_reference(folder, track_motion, tracks, cameras, selected=None):
    """Earliest joint pose/depth sample, validated against saved source correspondence."""
    from plyfile import PlyData

    validate_registration_option(selected)
    folder = Path(folder)
    sequence = json.loads((folder / "sequence.json").read_text())
    indices = sequence.get("sourceIndices")
    frames = sequence.get("frames")
    if (
        not isinstance(indices, list)
        or not isinstance(frames, list)
        or len(indices) != len(frames)
        or len(indices) != len(cameras)
        or indices != tracks.get("sourceIndices")
        or not sequence.get("sourceSha256")
        or sequence["sourceSha256"] != tracks.get("sourceSha256")
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
        or any(c.get("sourceIndex") != i for c, i in zip(cameras, indices))
    ):
        raise ValueError("Pi3X/track/camera source indices or source hash mismatch")
    records = track_motion.get("frames", [])
    samples = [r.get("sample") for r in records]
    if (
        not records
        or any(type(i) is not int or not 0 <= i < len(indices) for i in samples)
        or len(set(samples)) != len(samples)
    ):
        raise ValueError("Invalid track registration samples")
    for record in records:
        if record.get("sourceIndex") != indices[record["sample"]]:
            raise ValueError("Track registration source index mismatch")
    for record in sorted(records, key=lambda r: r["sample"]):
        sample = record["sample"]
        if selected is not None and sample != selected:
            continue
        name = frames[sample]
        if name is None:
            continue
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".ply"):
            raise ValueError("Invalid declared person depth reference")
        path = folder / name
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or len(PlyData.read(path)["vertex"].data) == 0
        ):
            raise ValueError("Declared person depth reference is missing or empty")
        roi = np.asarray(record.get("maskBox"), dtype=float)
        if (
            roi.shape != (4,)
            or not np.isfinite(roi).all()
            or not (roi[2] > roi[0] and roi[3] > roi[1])
        ):
            raise ValueError("Invalid registration track ROI")
        return sample, path, ",".join(str(float(value)) for value in roi)
    raise ValueError("No track pose has declared supported Pi3X person depth")


def depth_registration(xyz, camera, intrinsics, native_depth, roi=None):
    """Original median-depth estimator, held fixed for all animation frames."""
    xyz, camera = np.asarray(xyz, dtype=float), np.asarray(camera, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("Invalid registration reference points")
    if (
        camera.shape != (4, 4)
        or not np.isfinite(camera).all()
        or np.linalg.det(camera[:3, :3]) <= 0
    ):
        raise ValueError("Invalid registration camera")
    local = (xyz - camera[:3, 3]) @ np.linalg.inv(camera[:3, :3]).T
    kept = None
    if roi:
        box = np.asarray([float(v) for v in roi.split(",")])
        K = np.asarray(intrinsics, dtype=float)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or not (box[2] > box[0] and box[3] > box[1])
            or K.shape != (3, 3)
            or not np.isfinite(K).all()
        ):
            raise ValueError("Invalid registration ROI/intrinsics")
        uvw = (local * [1, -1, -1]) @ K.T
        front = uvw[:, 2] > 1e-6
        uv = np.full((len(uvw), 2), -1e9)
        uv[front] = uvw[front, :2] / uvw[front, 2:3]
        inside = (
            front
            & (uv[:, 0] >= box[0])
            & (uv[:, 0] <= box[2])
            & (uv[:, 1] >= box[1])
            & (uv[:, 1] <= box[3])
        )
        if inside.sum() < 50:
            raise RuntimeError(
                f"Only {int(inside.sum())} depth-reference points inside the track ROI"
            )
        local = local[inside]
        kept = int(inside.sum())
    observed = -local[:, 2]
    observed = observed[np.isfinite(observed) & (observed > 0)]
    native = np.asarray(native_depth, dtype=float)
    native = native[np.isfinite(native) & (native > 0)]
    if not len(observed) or not len(native):
        raise ValueError("Registration requires positive observed and native depth")
    observed_median = float(np.median(observed))
    # torch.median uses the lower middle element for an even population.
    native_median = float(np.partition(native, (len(native) - 1) // 2)[(len(native) - 1) // 2])
    scale = observed_median / native_median
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fixed depth registration")
    return {
        "uniformScale": scale,
        "observedPersonMedianDepth": observed_median,
        "nativeGaussianMedianDepth": native_median,
        "depthRoi": roi,
        "depthRoiPoints": kept,
    }
