#!/usr/bin/env python3
"""Package observed Pi3X anchor pixels in a reviewed people/camera coordinate frame.

This is an untrained source-coloured point surface, not a complete room or dynamic
reconstruction. Unknown surfaces and masked people are left empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from wander_worker.ply import write_gaussian_ply


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def camera_map(document: dict) -> dict:
    cameras = document["cameras"]
    result = {camera["sourceIndex"]: camera for camera in cameras}
    if len(result) != len(cameras):
        raise ValueError("Camera source indices must be unique")
    return result


def matrix(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("Expected a finite 4x4 camera pose")
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Expected an affine camera pose")
    return result


def package_arrays(
    anchors,
    sequence,
    raw_cameras,
    packaged_cameras,
    *,
    confidence,
    min_radius,
    max_radius,
    pixel_radius,
    selected_samples=None,
    extra_masks=None,
):
    """Return native packaged-world positions, unchanged RGB, radii, and diagnostics."""
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold must be in [0, 1]")
    if not 0 < min_radius <= max_radius or not np.isfinite([min_radius, max_radius]).all():
        raise ValueError("Radius bounds must be finite and positive")
    if not np.isfinite(pixel_radius) or pixel_radius <= 0:
        raise ValueError("Pixel radius must be finite and positive")
    points = np.asarray(anchors["points"])
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("Expected anchor points shaped (anchors, height, width, 3)")
    n, height, width, _ = points.shape
    rgb = np.asarray(anchors["rgb"])
    if rgb.shape != points.shape or rgb.dtype != np.uint8:
        raise ValueError("Expected original uint8 RGB matching anchor points")
    for key in ("valid", "people", "conf"):
        if anchors[key].shape != points.shape[:-1]:
            raise ValueError(f"Anchor {key} shape mismatch")
    if anchors["valid"].dtype != np.bool_ or anchors["people"].dtype != np.bool_:
        raise ValueError("Validity and person masks must be boolean")
    samples = sequence["anchors"]
    if len(samples) != n or anchors["poses"].shape != (n, 4, 4):
        raise ValueError("Anchor sample/pose count mismatch")
    if selected_samples is not None and not set(selected_samples).issubset(samples):
        raise ValueError("Selected sample is not a recorded anchor")
    if extra_masks is None:
        extra_masks = np.zeros(points.shape[:-1], dtype=bool)
    if extra_masks.shape != points.shape[:-1] or extra_masks.dtype != np.bool_:
        raise ValueError("Extra masks must be boolean and match the anchor image grid")
    scale = float(raw_cameras["reference"]["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Missing positive Pi3X normalization scale")
    raw, packaged = camera_map(raw_cameras), camera_map(packaged_cameras)
    transforms = []
    for index in raw.keys() & packaged.keys():
        transforms.append(
            matrix(packaged[index]["camera_to_world"])
            @ np.linalg.inv(matrix(raw[index]["camera_to_world"]))
        )
    if not transforms:
        raise ValueError("No source-index camera matches")
    transform = transforms[0]
    reframe_error = float(np.max(np.abs(np.asarray(transforms) - transform)))
    if (
        reframe_error > 0.01
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=0.01)
        or np.linalg.det(transform[:3, :3]) <= 0
    ):
        raise ValueError("Packaged cameras are not one rigid reframe of raw cameras")
    xyz_parts, rgb_parts, radii_parts, counts = [], [], [], []
    flip = np.array([1.0, -1.0, -1.0])
    for i, sample in enumerate(samples):
        if selected_samples is not None and sample not in selected_samples:
            continue
        source = sequence["sourceIndices"][sample]
        camera = packaged[source]
        if abs(camera["time"] - sequence["timestamps"][sample]) > 1e-5:
            raise ValueError("Anchor camera time differs from sequence time")
        pose = matrix(anchors["poses"][i])
        public = matrix(camera["camera_to_world"])
        inverse = np.linalg.inv(pose)
        local = points[i].astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
        valid = anchors["valid"][i]
        static = valid & ~anchors["people"][i] & ~extra_masks[i]
        finite = np.isfinite(local).all(-1) & np.isfinite(anchors["conf"][i])
        keep = static & finite & (anchors["conf"][i] >= confidence) & (local[..., 2] > 0)
        local = local[keep] * scale
        world = (local * flip) @ public[:3, :3].T + public[:3, 3]
        intrinsic = np.asarray(camera["source_intrinsics"], dtype=float)
        source_width, source_height = camera["source_image_size"]
        fx = intrinsic[0, 0] * width / source_width
        fy = intrinsic[1, 1] * height / source_height
        if not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
            raise ValueError("Invalid camera focal length")
        footprint = local[:, 2] * pixel_radius / np.sqrt(fx * fy)
        radius = np.clip(footprint, min_radius, max_radius)
        xyz_parts.append(world.astype(np.float32))
        rgb_parts.append(rgb[i][keep])
        radii_parts.append(radius.astype(np.float32))
        counts.append(
            dict(
                anchor=i,
                sample=int(sample),
                sourceIndex=int(source),
                time=float(sequence["timestamps"][sample]),
                pixels=int(valid.size),
                valid=int(valid.sum()),
                personExcluded=int((valid & anchors["people"][i]).sum()),
                extraMaskExcluded=int((valid & ~anchors["people"][i] & extra_masks[i]).sum()),
                nonfiniteExcluded=int((static & ~finite).sum()),
                confidenceExcluded=int((static & finite & (anchors["conf"][i] < confidence)).sum()),
                kept=int(keep.sum()),
                radiusClampedLow=int((footprint < min_radius).sum()),
                radiusClampedHigh=int((footprint > max_radius).sum()),
            )
        )
    xyz = np.concatenate(xyz_parts)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("No finite static anchor points survived")
    return (
        xyz,
        np.concatenate(rgb_parts),
        np.concatenate(radii_parts),
        dict(anchors=counts, normalizationScale=scale, cameraReframeMaxResidual=reframe_error),
    )


def track_box_masks(tracks, sequence, cameras, shape, margin):
    """Source-bound conservative rectangles from existing detector mask boxes."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Mask margin must be finite and nonnegative")
    height, width = shape[1:]
    masks = np.zeros(shape, dtype=bool)
    camera_by_source = camera_map(cameras)
    evidence = []
    for i, sample in enumerate(sequence["anchors"]):
        source = sequence["sourceIndices"][sample]
        camera = camera_by_source[source]
        source_width, source_height = camera["source_image_size"]
        for name, motion in tracks.items():
            frame = next((f for f in motion["frames"] if f["sourceIndex"] == source), None)
            if frame is None:
                continue
            box = np.asarray(frame.get("maskBox"), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Track lacks a measured maskBox for selected anchor")
            if abs(frame["time"] - sequence["timestamps"][sample]) > 1e-5:
                raise ValueError("Track mask time differs from anchor time")
            ratio = np.array([width / source_width, height / source_height])
            lo = np.maximum(0, np.floor((box[:2] - margin) * ratio).astype(int))
            hi = np.minimum([width, height], np.ceil((box[2:] + margin) * ratio).astype(int))
            masks[i, lo[1] : hi[1], lo[0] : hi[0]] = True
            evidence.append(
                dict(
                    anchor=i,
                    sample=sample,
                    sourceIndex=source,
                    track=name,
                    sourceMaskBox=box.tolist(),
                    expandedGridBox=[*lo.tolist(), *hi.tolist()],
                )
            )
    return masks, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("anchors", "sequence", "raw-cameras", "cameras", "people", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Optionally verify the actual source clip hash")
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--min-radius", type=float, required=True, help="Packaged native units")
    parser.add_argument("--max-radius", type=float, required=True, help="Packaged native units")
    parser.add_argument("--pixel-radius", type=float, default=0.6)
    parser.add_argument("--anchor-samples", help="Comma-separated recorded anchor sample IDs")
    parser.add_argument(
        "--track-masks",
        type=Path,
        help="Existing tracks directory for conservative maskBox exclusion",
    )
    parser.add_argument(
        "--mask-margin",
        type=float,
        default=20,
        help="Extra source-image pixels around measured maskBox",
    )
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".provenance.json").exists():
        parser.error("Output exists; choose a fresh candidate path")
    inputs = {
        name: getattr(args, name)
        for name in ("anchors", "sequence", "raw_cameras", "cameras", "people")
    }
    documents = {
        key: json.loads(value.read_text()) for key, value in inputs.items() if key != "anchors"
    }
    source_hash = documents["sequence"]["sourceSha256"]
    if documents["people"]["sourceSha256"] != source_hash:
        parser.error("People and anchor sequence source hashes differ")
    if args.source and sha256(args.source) != source_hash:
        parser.error("Source clip hash differs from anchor sequence")
    if args.source:
        inputs["source"] = args.source
    selected = (
        None if args.anchor_samples is None else [int(v) for v in args.anchor_samples.split(",")]
    )
    with np.load(args.anchors, allow_pickle=False) as anchors:
        extra_masks, mask_evidence = None, []
        if args.track_masks:
            metadata = args.track_masks / "tracks.json"
            if json.loads(metadata.read_text())["sourceSha256"] != source_hash:
                parser.error("Track masks and anchor sequence source hashes differ")
            inputs["tracks"] = metadata
            motions = {}
            for path in sorted(args.track_masks.glob("track_*/motion.json")):
                motions[path.parent.name] = json.loads(path.read_text())
                inputs["mask_" + path.parent.name] = path
            if not motions:
                parser.error("No recovered track motion files")
            extra_masks, mask_evidence = track_box_masks(
                motions,
                documents["sequence"],
                documents["raw_cameras"],
                anchors["valid"].shape,
                args.mask_margin,
            )
        xyz, rgb, radius, report = package_arrays(
            anchors,
            documents["sequence"],
            documents["raw_cameras"],
            documents["cameras"],
            confidence=args.confidence,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            pixel_radius=args.pixel_radius,
            selected_samples=selected,
            extra_masks=extra_masks,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_gaussian_ply(args.out, xyz, rgb, scale=radius, opacity=0.95)
    report.update(
        schema="wander.static-anchors/1",
        coordinates="Exact packaged cameras/people native OpenGL frame; no viewer scale or world-axis flip baked in",
        sourceSha256=source_hash,
        sourceHashVerified=args.source is not None,
        inputs={key: dict(path=str(value), sha256=sha256(value)) for key, value in inputs.items()},
        points=len(xyz),
        selectedAnchorSamples=selected,
        additionalMaskEvidence=mask_evidence,
        additionalMaskSemantics="Conservative expanded recorded detector rectangles, not new segmentation; removed background is left empty",
        bounds=[xyz.min(0).tolist(), xyz.max(0).tolist()],
        filtering=dict(
            confidence=args.confidence,
            minRadius=args.min_radius,
            maxRadius=args.max_radius,
            pixelRadius=args.pixel_radius,
        ),
        appearance="Original uint8 anchor RGB encoded as SH0, opacity 0.95; isotropic radius from depth and pixel focal length, clipped to explicit bounds. No training or colour synthesis.",
        limitations=[
            "Unobserved surfaces are not reconstructed; no invented floor, wall or mask fill.",
            "Depth and camera poses are estimated, not measured ground truth.",
            "Person exclusion uses the saved masks; missed people or moving fixtures can leave inconsistent surfaces.",
            "Opening refrigerator doors and other moving fixtures are unsupported by this static representation.",
            "Anchor observations are concatenated without temporal consistency repair or deduplication.",
            "Review candidate only; no visual acceptance, collider or walking-surface guarantee.",
        ],
        output=dict(path=str(args.out), sha256=sha256(args.out)),
    )
    args.out.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "points": len(xyz),
                "output": str(args.out),
                "sourceHashVerified": args.source is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
