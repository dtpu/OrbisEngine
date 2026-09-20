#!/usr/bin/env python3
"""Review-only camera-centred person scale against a source-labelled floor plane.

Writes placement metadata, never changes PLY or compact motion. A fixed positive
scale per track preserves every source-camera ray; floor residuals remain evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

CHANNELS = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"]


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def checked_file(root, name, size, sha):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input escapes package directory")
    if path.stat().st_size != size or digest(path) != sha:
        raise ValueError(f"Integrity mismatch: {path}")
    return path


def match_cameras(sequence, cameras):
    lookup = {c["sourceIndex"]: c for c in cameras["cameras"]}
    if len(lookup) != len(cameras["cameras"]):
        raise ValueError("Duplicate camera source indices")
    indices, times = sequence["sourceIndices"], sequence["timestamps"]
    if len(indices) != len(sequence["frames"]) or len(times) != len(indices):
        raise ValueError("Sequence timing length mismatch")
    if (
        len(set(indices)) != len(indices)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Sequence timing must be finite, unique and increasing")
    matched = []
    for source, time in zip(indices, times):
        camera = lookup.get(source)
        if camera is None or abs(camera["time"] - time) > 1e-6:
            raise ValueError("Sequence source index/time does not match camera")
        pose = np.asarray(camera["camera_to_world"], float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
        ):
            raise ValueError("Invalid camera pose")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Camera must have an orthonormal rotation")
        matched.append(camera)
    return matched


def solve_scale(foot_levels, centers, normal, offset, body_height, max_spread):
    levels, centers, normal = map(np.asarray, (foot_levels, centers, normal))
    if levels.ndim != 1 or centers.shape != (len(levels), 3) or normal.shape != (3,):
        raise ValueError("Floor sample dimensions differ")
    if (
        not np.isfinite([offset, body_height, max_spread]).all()
        or body_height <= 0
        or max_spread < 0
    ):
        raise ValueError("Invalid floor scale uncertainty bounds")
    if len(levels) < 3 or not all(np.isfinite(a).all() for a in (levels, centers, normal)):
        raise ValueError("Need at least three finite floor observations")
    camera_levels = centers @ normal
    denominator = levels - camera_levels
    if np.any(denominator >= -1e-4 * body_height) or np.any(offset >= camera_levels):
        raise ValueError("Degenerate floor geometry: camera must be above feet and floor")
    candidates = (offset - camera_levels) / denominator
    scale = float(np.median(candidates))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    spread = float((np.quantile(candidates, 0.9) - np.quantile(candidates, 0.1)) / scale)
    if spread > max_spread:
        raise ValueError(f"Uncertain scale: relative p10–p90 spread {spread:.4f} > {max_spread}")
    before = (levels - offset) / body_height
    after = (camera_levels + scale * denominator - offset) / body_height
    return scale, candidates, before, after, spread


def corrected_points(points, center, scale):
    return center + scale * (points - center)


def projection_error(points, corrected, camera):
    pose = np.asarray(camera["camera_to_world"], float)
    before = (points - pose[:3, 3]) @ pose[:3, :3]
    after = (corrected - pose[:3, 3]) @ pose[:3, :3]
    if np.any(np.abs(before[:, 2]) < 1e-8) or np.any(np.abs(after[:, 2]) < 1e-8):
        raise ValueError("Projection undefined at camera depth zero")
    focal = np.diag(np.asarray(camera["source_intrinsics"], float))[:2]
    error = np.abs((before[:, :2] / before[:, 2, None] - after[:, :2] / after[:, 2, None]) * focal)
    return float(error.max())


def summarize(values):
    return dict(
        median=float(np.median(values)),
        rms=float(np.sqrt(np.mean(values**2))),
        p10=float(np.quantile(values, 0.1)),
        p90=float(np.quantile(values, 0.9)),
        maximumAbsolute=float(np.max(np.abs(values))),
    )


def export(args):
    package = args.people.parent
    people = json.loads(args.people.read_text())
    cameras = json.loads(args.cameras.read_text())
    floor = json.loads(args.floor.read_text())
    if floor.get("schema") != "wander.observed-floor/1" or "native" not in floor.get(
        "coordinates", ""
    ):
        raise ValueError("Requires an observed floor in packaged native coordinates")
    source_hash = people["sourceSha256"]
    if cameras["sourceSha256"] != source_hash:
        raise ValueError("People and camera source hashes differ")
    inputs = {"people": args.people, "cameras": args.cameras, "floor": args.floor}
    correspondence = None
    if floor["sourceSha256"] != source_hash:
        reference = people.get("sourceCorrespondence", {})
        if reference.get("derivedSha256") != floor["sourceSha256"]:
            raise ValueError("Floor source differs without measured correspondence")
        path = package / reference["evidence"]
        correspondence = json.loads(path.read_text())
        if (
            correspondence["originalSha256"] != source_hash
            or correspondence["derivativeSha256"] != floor["sourceSha256"]
        ):
            raise ValueError("Correspondence hash binding mismatch")
        inputs["correspondence"] = path
    normal = np.asarray(floor["normal"], float)
    offset, body_height = float(floor["offset"]), float(floor["bodyHeightUnits"])
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or abs(np.linalg.norm(normal) - 1) > 1e-5
        or normal[1] < 0.95
    ):
        raise ValueError("Expected finite upward unit floor normal")
    if not np.isfinite([offset, body_height]).all() or body_height <= 0:
        raise ValueError("Invalid floor offset or body height")
    floor_rms = float(floor["residualRmsBodyHeights"])
    if (
        not np.isfinite([floor_rms, args.max_floor_rms]).all()
        or floor_rms < 0
        or args.max_floor_rms <= 0
    ):
        raise ValueError("Invalid floor uncertainty")
    if not floor.get("observations") or floor_rms > args.max_floor_rms:
        raise ValueError("Floor lacks source observations or is too uncertain")
    g = args.registration_scale
    if not np.isfinite(g) or g <= 0 or not 0 < args.foot_quantile < 0.5 or not 0 < args.opacity < 1:
        raise ValueError("Invalid scale/quantile/opacity")
    per_person, evidence = {}, {}
    for person in people["people"]:
        transform = person.get("transform", {})
        if (
            transform.get("scale", 1) != 1
            or not np.allclose(transform.get("translation", [0, 0, 0]), 0)
            or not np.allclose(transform.get("quaternionXYZW", [0, 0, 0, 1]), [0, 0, 0, 1])
        ):
            raise ValueError("Only identity manifest transforms are supported")
        sequence_path = package / person["sequence"]
        sequence = json.loads(sequence_path.read_text())
        if sequence["sourceSha256"] != source_hash:
            raise ValueError("Sequence source hash differs from people")
        matched = match_cameras(sequence, cameras)
        if correspondence is not None:
            rows = {row["derivedIndex"]: row for row in correspondence["rows"]}
            derived = sequence.get("derivedSourceIndices", [])
            if len(derived) != len(matched):
                raise ValueError("Missing per-frame derived correspondence")
            for i, derived_index in enumerate(derived):
                row = rows[derived_index]
                if (
                    row["originalIndex"] != sequence["sourceIndices"][i]
                    or abs(row["originalPtsSeconds"] - sequence["timestamps"][i]) > 1e-6
                ):
                    raise ValueError("Per-frame measured correspondence mismatch")
        motion = sequence["motion"]
        if (
            motion["schema"] != "wander.person-motion/1"
            or motion["dtype"] != "float32"
            or motion["channels"] != CHANNELS
            or motion["frameFiles"] != sequence["frames"]
            or motion["frames"] != len(matched)
        ):
            raise ValueError("Requires frame-bound lossless float32 motion")
        shape = (motion["frames"], motion["splats"], len(CHANNELS))
        if motion["bytes"] != int(np.prod(shape)) * 4:
            raise ValueError("Motion payload shape mismatch")
        payload = checked_file(
            sequence_path.parent, motion["file"], motion["bytes"], motion["sha256"]
        )
        base = motion["base"]
        if base["file"] != sequence["frames"][0] or base["sha256"] != sequence["frame_sha256"][0]:
            raise ValueError("Appearance is not bound to first original sequence frame")
        appearance = checked_file(sequence_path.parent, base["file"], base["bytes"], base["sha256"])
        vertices = PlyData.read(appearance)["vertex"].data
        if len(vertices) != motion["splats"]:
            raise ValueError("Appearance splat count mismatch")
        opacity_logit = np.log(args.opacity / (1 - args.opacity))
        if not np.isfinite(vertices["opacity"]).all():
            raise ValueError("Nonfinite appearance opacity")
        opaque = np.isfinite(vertices["opacity"]) & (vertices["opacity"] >= opacity_logit)
        if opaque.sum() < 100:
            raise ValueError("Too few opaque splats for foot quantile")
        data = np.memmap(payload, mode="r", dtype="<f4", shape=shape)
        levels = []
        for frame in data:
            if not np.isfinite(frame).all():
                raise ValueError("Nonfinite motion values")
            levels.append(float(np.quantile(frame[opaque, :3] @ normal, args.foot_quantile)))
        centers = np.array([np.array(c["camera_to_world"])[:3, 3] for c in matched])
        s, candidates, before, after, spread = solve_scale(
            levels, centers, normal, offset, body_height, args.max_scale_spread
        )
        max_projection = 0.0
        for i, frame in enumerate(data):
            points = np.asarray(frame[:, :3], float)
            max_projection = max(
                max_projection,
                projection_error(points, corrected_points(points, centers[i], s), matched[i]),
            )
        if max_projection > 1e-7:
            raise ValueError("Camera projection invariance failed")
        per_person[person["id"]] = dict(
            sizeScale=s,
            constantUnits=[0, 0, 0],
            offsetUnits=(g * (1 - s) * centers).tolist(),
            sourceIndices=sequence["sourceIndices"],
            timestamps=sequence["timestamps"],
            frameFiles=sequence["frames"],
        )
        evidence[person["id"]] = dict(
            sizeScale=s,
            candidateScale=summarize(candidates),
            relativeScaleSpreadP10P90=spread,
            opacityThreshold=args.opacity,
            opaqueSplats=int(opaque.sum()),
            footQuantile=args.foot_quantile,
            contactBeforeBodyHeights=summarize(before),
            contactAfterBodyHeights=summarize(after),
            contactReferenceBodyHeightUnits=body_height,
            maximumProjectionErrorPixels=max_projection,
            perFrame=dict(
                sourceIndices=sequence["sourceIndices"],
                timestamps=sequence["timestamps"],
                beforeBodyHeights=before.tolist(),
                afterBodyHeights=after.tolist(),
                candidateScale=candidates.tolist(),
            ),
        )
        inputs[person["id"] + "Sequence"] = sequence_path
        inputs[person["id"] + "Motion"] = payload
        inputs[person["id"] + "Appearance"] = appearance
    result = dict(
        schema="wander.placement/1",
        world=args.world,
        samples=people["samples"],
        registrationScale=g,
        pos0=[0, 0, 0],
        perPerson=per_person,
        sourceSha256=source_hash,
        floorY=g * offset / normal[1],
        floorPlane=dict(normal=normal.tolist(), offset=g * offset),
        accepted=False,
        status="offline calibration candidate; viewer review required",
        requiredViewerFlags=dict(
            place=1, rot="0,0,0", rotfix=0, feetmode="sfm", feetlock=0, stance=0, camdrift=0
        ),
        placementSemantics="offsetUnits are final viewer-space translations, so offsets=g*(1-s)*C. Keep feet processing enabled (feetmode=sfm): applyFeet installs per-person offsets. World and source camera must use matching global scale g.",
        limitations=[
            "Stable per-track scale cannot force every frame onto floor; residuals are retained.",
            "Opaque low quantile is a geometric foot proxy, not measured anatomical contact.",
            "Source-camera projection invariance holds at exact recorded sample times. Between samples, interpolated camera paths/rotations may not agree.",
            "Changes depth and apparent size from other viewpoints; does not repair articulation or appearance.",
            "floorY approximates tilted plane at native x=z=0; floorPlane preserves full measured normal.",
        ],
        evidence=evidence,
        inputs={key: dict(path=str(path), sha256=digest(path)) for key, path in inputs.items()},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("people", "cameras", "floor", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--world", required=True, help="Exact candidate world filename for viewer guard"
    )
    parser.add_argument("--registration-scale", type=float, required=True)
    parser.add_argument("--foot-quantile", type=float, default=0.01)
    parser.add_argument("--opacity", type=float, default=0.5)
    parser.add_argument("--max-scale-spread", type=float, default=0.2)
    parser.add_argument("--max-floor-rms", type=float, default=0.02, help="Body-heights")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new candidate path")
    result = export(args)
    print(
        json.dumps(
            {
                key: {
                    k: row[k]
                    for k in (
                        "sizeScale",
                        "contactBeforeBodyHeights",
                        "contactAfterBodyHeights",
                        "maximumProjectionErrorPixels",
                    )
                }
                for key, row in result["evidence"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
