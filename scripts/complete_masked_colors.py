#!/usr/bin/env python3
"""Offline, inferred color completion from mask-clear donor splats.

Requires explicitly selected training views. This does not recover observations behind
people, change geometry, or establish scene acceptance. Never used by pipeline defaults.
"""

import argparse
import gzip
import hashlib
import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

try:
    from . import render_world_poses
    from .splat_observations import measure_source_exposure
except ImportError:
    import render_world_poses
    from splat_observations import measure_source_exposure


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def positive_int(value):
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0 or str(result) != str(value):
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument(
        "--training-frames", type=Path, required=True, help="JSON list of export indices"
    )
    p.add_argument("--body-height-native", type=positive_float, required=True)
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--radius-body-heights", type=positive_float, default=0.25)
    p.add_argument("--neighbors", type=positive_int, default=16)
    p.add_argument("--min-neighbors", type=positive_int, default=3)
    p.add_argument("--min-clear-views", type=positive_int, default=3)
    p.add_argument("--max-seconds", type=positive_float, default=900.0)
    return p


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inputs(paths):
    return {str(path.resolve()): sha256(path) for path in paths}


def code_inputs():
    """Bind project-local imported modules; dependency versions are bound by uv.lock."""
    root = Path(__file__).resolve().parents[1]
    paths = {Path(__file__).resolve(), root / "uv.lock"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(root / "scripts") and path.suffix == ".py":
                paths.add(path)
    return sorted(paths)


def read_raw_spz(path):
    try:
        raw = gzip.decompress(Path(path).read_bytes())
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"invalid gzip SPZ: {exc}") from exc
    if len(raw) < 16:
        raise ValueError("SPZ header is truncated")
    magic, version, n, sh, fractional_bits, flags, reserved = struct.unpack("<IIIBBBB", raw[:16])
    if magic != 0x5053474E or version != 2 or sh != 0:
        raise ValueError("requires SPZ v2 with SH degree 0")
    if not n or fractional_bits > 24 or flags != 0 or reserved != 0:
        raise ValueError("unsupported SPZ point count, fractional bits, flags or reserved byte")
    if len(raw) != 16 + n * 19:
        raise ValueError("SPZ payload length does not match its header")
    return raw, n


def patch_colors(raw, inferred, rgb):
    """Patch accepted RGB records only; preserve even noncanonical unselected bytes."""
    n = struct.unpack_from("<I", raw, 8)[0]
    offset = 16 + n * 10
    output = bytearray(raw)
    colors = np.frombuffer(output, dtype=np.uint8, count=n * 3, offset=offset).reshape(n, 3)
    packed = np.rint(
        (
            0.5
            + (rgb[inferred] - 0.5) * render_world_poses.SPZ_COLOR_SCALE / render_world_poses.SH_C0
        )
        * 255.0
    )
    colors[inferred] = np.clip(packed, 0, 255).astype(np.uint8)
    return bytes(output)


def weighted_median(values, weights):
    """Independent channel medians of retained donor RGB, with stable tie handling."""
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = weights[order]
    positions = np.argmax(np.cumsum(sorted_weights, axis=0) >= weights.sum() * 0.5, axis=0)
    return sorted_values[positions, np.arange(3)]


def infer_colors(world, exposure, *, radius, neighbors, min_neighbors, min_clear_views, check=None):
    """Select donors before KNN; never use newly inferred colors as donors."""
    target = (exposure["rendered_mass"] > 0) & (exposure["clear_mass"] == 0)
    donor_ids = np.flatnonzero(exposure["clear_views"] >= min_clear_views)
    inferred = np.zeros(world["n"], dtype=bool)
    rgb = world["rgb"].copy()
    counts = {
        "points": int(world["n"]),
        "targets": int(target.sum()),
        "eligibleDonors": int(len(donor_ids)),
        "inferred": 0,
        "rejectedInsufficientDonors": int(target.sum()),
    }
    if len(donor_ids) < min_neighbors:
        return rgb, inferred, counts
    tree = cKDTree(world["xyz"][donor_ids].astype(np.float64))
    k = min(neighbors, len(donor_ids))
    target_ids = np.flatnonzero(target)
    # Bound query memory and allow cancellation between batches and target records.
    for start in range(0, len(target_ids), 4096):
        if check:
            check()
        ids = target_ids[start : start + 4096]
        distances, local_ids = tree.query(
            world["xyz"][ids], k=k, distance_upper_bound=np.nextafter(radius, math.inf), workers=1
        )
        distances = np.asarray(distances).reshape(len(ids), k)
        local_ids = np.asarray(local_ids).reshape(len(ids), k)
        for row, point_id in enumerate(ids):
            if check and row % 256 == 0:
                check()
            valid = np.isfinite(distances[row]) & (distances[row] <= radius)
            if int(valid.sum()) < min_neighbors:
                continue
            donor_rgb = world["rgb"][donor_ids[local_ids[row, valid]]]
            weights = np.exp(-0.5 * (distances[row, valid] / (radius * 0.5)) ** 2)
            rgb[point_id] = weighted_median(donor_rgb, weights)
            inferred[point_id] = True
    counts["inferred"] = int(inferred.sum())
    counts["rejectedInsufficientDonors"] -= counts["inferred"]
    return rgb, inferred, counts


def load_frames(path):
    frames = json.loads(path.read_text())
    if (
        not isinstance(frames, list)
        or not frames
        or any(type(index) is not int or index < 0 for index in frames)
        or len(set(frames)) != len(frames)
    ):
        raise ValueError("training frames must be a nonempty list of unique nonnegative integers")
    return frames


def scalar_positive_int(value, name):
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in "iu" or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer scalar")
    return int(value)


def finite_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}")
    return value.astype(np.float64)


def load_export(directory, frames):
    with np.load(directory / "cameras.npz", allow_pickle=False) as cameras:
        poses = cameras["c2w_gl"]
        if poses.ndim != 3 or not len(poses):
            raise ValueError("c2w_gl must have shape (frames,4,4)")
        f = len(poses)
        poses = finite_array(poses, (f, 4, 4), "c2w_gl")
        ks = finite_array(cameras["Ks"], (f, 3, 3), "Ks")
        width = scalar_positive_int(cameras["width"], "width")
        height = scalar_positive_int(cameras["height"], "height")
        source = cameras["source_index"]
        if (
            source.shape != (f,)
            or source.dtype.kind not in "iu"
            or np.any(source < 0)
            or np.any(source[1:] <= source[:-1])
        ):
            raise ValueError(
                "source_index must be strictly increasing nonnegative integers, one per frame"
            )
    if max(frames) >= f:
        raise ValueError("training frame index outside camera archive")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("c2w_gl must contain homogeneous camera poses")
    if np.any(np.abs(np.linalg.det(poses[:, :3, :3])) < 1e-8):
        raise ValueError("c2w_gl contains singular camera rotation")
    if (
        np.any(ks[:, 0, 0] <= 0)
        or np.any(ks[:, 1, 1] <= 0)
        or not np.allclose(ks[:, 2], [0, 0, 1], atol=1e-6)
    ):
        raise ValueError("Ks must contain positive focal lengths and homogeneous intrinsics")
    with np.load(directory / "splats.npz", allow_pickle=False) as splats:
        # Large means/colors/covariances are intentionally not materialized.
        origin = finite_array(splats["origin_c2w"], (4, 4), "origin_c2w")
        scale_value = np.asarray(splats["scale0"])
        if scale_value.shape != () or scale_value.dtype.kind not in "fiu":
            raise ValueError("scale0 must be a positive finite scalar")
        scale = float(scale_value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale0 must be a positive finite scalar")
    rotation = origin[:3, :3]
    if (
        not np.allclose(origin[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-4)
    ):
        raise ValueError("origin_c2w must be a proper rigid homogeneous pose for radius conversion")
    receipt = json.loads((directory / "export-receipt.json").read_text())
    if not isinstance(receipt, dict):
        raise ValueError("export receipt must be a JSON object")
    indices = receipt.get("sourceIndices")
    size = receipt.get("size")
    receipt_scale = receipt.get("scale0")
    if (
        not isinstance(indices, list)
        or any(type(v) is not int for v in indices)
        or indices != source.tolist()
        or not isinstance(size, list)
        or any(type(v) is not int for v in size)
        or size != [width, height]
        or type(receipt_scale) not in (float, int)
        or receipt_scale != scale
    ):
        raise ValueError("export receipt sourceIndices, size or scale0 does not match archives")
    masks = {}
    for index in frames:
        with Image.open(directory / "masks" / f"{index:04d}.png") as im:
            values = np.asarray(im)
            if (
                im.format != "PNG"
                or im.mode != "L"
                or values.shape != (height, width)
                or not np.all((values == 0) | (values == 255))
            ):
                raise ValueError(f"mask {index} must be binary 0/255 grayscale PNG of export size")
            masks[index] = values == 255
    return poses, ks, origin, scale, masks, receipt


def validate_exposure(exposure, n, frames):
    for key in ("rendered_mass", "masked_mass", "clear_mass"):
        values = finite_array(exposure[key], (n,), key)
        if np.any(values < 0):
            raise ValueError(f"{key} must be nonnegative")
    views = exposure["clear_views"]
    if (
        views.shape != (n,)
        or views.dtype.kind not in "iu"
        or np.any((views < 0) | (views > len(frames)))
    ):
        raise ValueError("clear_views must contain valid per-point frame counts")
    if not np.array_equal(exposure["frame_indices"], frames):
        raise ValueError("exposure frame indices differ from requested training frames")
    if not np.allclose(
        exposure["rendered_mass"],
        exposure["masked_mass"] + exposure["clear_mass"],
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError("exposure rendered mass does not equal masked plus clear mass")
    if np.any((views > 0) != (exposure["clear_mass"] > 0)):
        raise ValueError("exposure clear mass and view counts disagree")


def run(args):
    start = time.monotonic()
    if args.out.exists():
        raise ValueError("output directory already exists; choose a new directory")
    # mkdir(exist_ok=False) also closes a concurrent creator's overwrite window.
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "wander.masked-color-completion/1",
        "status": "running",
        "acceptedScene": False,
        "parameters": {
            key: getattr(args, key)
            for key in (
                "body_height_native",
                "radius_body_heights",
                "neighbors",
                "min_neighbors",
                "min_clear_views",
                "max_seconds",
            )
        },
        "limitations": [
            "Colors are inferred appearance, not observed texture behind a person.",
            "Geometry, opacity and unresolved scene/placement failures are unchanged.",
            "Exposure is an endpoint CPU replay, not historical CUDA gradient evidence.",
            "Input hashes bind this execution; full source lineage is not independently proven.",
            "Source footage is not loaded and no heldout frames are selected implicitly.",
            "The time limit is cooperative at frame and inference-batch boundaries.",
        ],
    }

    def save_report():
        report["elapsedSeconds"] = time.monotonic() - start
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def check(*_):
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError(
                "color completion exceeded --max-seconds at a cooperative checkpoint"
            )

    try:
        for key in ("body_height_native", "radius_body_heights", "max_seconds"):
            value = getattr(args, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("neighbors", "min_neighbors", "min_clear_views"):
            if type(getattr(args, key)) is not int or getattr(args, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if args.min_neighbors > args.neighbors:
            raise ValueError("min-neighbors must not exceed neighbors")
        frames = load_frames(args.training_frames)
        if args.min_clear_views > len(frames):
            raise ValueError("min-clear-views must not exceed selected training frame count")
        input_paths = [
            args.world,
            args.training_frames,
            args.export / "cameras.npz",
            args.export / "splats.npz",
            args.export / "export-receipt.json",
            *(args.export / "masks" / f"{index:04d}.png" for index in frames),
            *code_inputs(),
        ]
        report["inputHashesBefore"] = hash_inputs(input_paths)
        report["trainingFrameIndices"] = frames
        save_report()
        poses, ks, origin, scale, masks, receipt = load_export(args.export, frames)
        report["exportSourceProvenance"] = receipt
        radius = args.radius_body_heights * args.body_height_native * scale
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius in raw world units must be finite and positive")
        report["radiusRawWorldUnits"] = radius
        raw, n = read_raw_spz(args.world)
        world = render_world_poses.read_spz(args.world)
        check()
        exposure = measure_source_exposure(
            world,
            poses,
            ks,
            masks,
            origin_c2w=origin,
            scale=scale,
            frame_indices=frames,
            progress=check,
        )
        check()
        validate_exposure(exposure, n, frames)
        np.savez_compressed(args.out / "exposure.npz", **exposure)
        rgb, inferred, counts = infer_colors(
            world,
            exposure,
            radius=radius,
            neighbors=args.neighbors,
            min_neighbors=args.min_neighbors,
            min_clear_views=args.min_clear_views,
            check=check,
        )
        patched = patch_colors(raw, inferred, rgb)
        counts["changedColorRecords"] = int(
            np.any(
                np.frombuffer(raw, np.uint8, n * 3, 16 + n * 10).reshape(n, 3)
                != np.frombuffer(patched, np.uint8, n * 3, 16 + n * 10).reshape(n, 3),
                axis=1,
            ).sum()
        )
        report["selectionCounts"] = counts
        np.save(args.out / "inferred-mask.npy", inferred, allow_pickle=False)
        compressed = gzip.compress(patched, mtime=0)
        check()
        report["inputHashesAfter"] = hash_inputs(input_paths)
        if report["inputHashesAfter"] != report["inputHashesBefore"]:
            raise ValueError("inputs changed during completion; final world was not written")
        report["inputsUnchanged"] = True
        check()
        (args.out / "world.spz").write_bytes(compressed)
        report["outputHashes"] = hash_inputs(
            [args.out / "world.spz", args.out / "inferred-mask.npy", args.out / "exposure.npz"]
        )
        report["status"] = "completed"
        save_report()
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report()
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except (ValueError, OSError, KeyError, TimeoutError) as exc:
        print(f"color completion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "selectionCounts": report["selectionCounts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
