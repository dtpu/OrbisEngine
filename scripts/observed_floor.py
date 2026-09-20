#!/usr/bin/env python3
"""Measure the floor a recorded person walks on, as a `wander.observed-floor/1` manifest.

`scripts/calibrate_person_floor.py` already turns an observed floor plane into a placement the
viewer reads: one fixed positive scale per track about each recorded camera centre, which moves the
feet onto that plane while preserving every source-camera ray. It needs the plane, and on the
kitchen that plane came from hand-labelled floor polygons. This module measures the same plane
automatically from data the run already has, so the calibrator is the single placement path for
clips nobody has labelled by hand.

Two providers, both recorded as observations with their own residuals:

  sfm    the Pi3X static anchor cloud, restricted to a column under the person's own walked
         footprint and to points BELOW the feet. Restricting to the walked column is what keeps a
         dominant high horizontal surface (docs/known-limits.md: a countertop, not a floor) out of
         the fit. This is the floor the FOOTAGE shows.
  world  the Marble world's own lowest opaque splats under that same footprint, divided back into
         native units by the registration scale. This is the floor the VIEWER DRAWS.

The two agree only when the world is registered to our solve. `crossCheck.disagreementBodyHeights`
reports the gap; a caller that finds it large has measured a registration failure, and no single
scalar placement can hide it.

  uv run --locked python scripts/observed_floor.py --people public/worlds/<clip>-4d/people.json \
      --anchors .context/run/<clip>/pi3x/anchors.npz --from sfm --out <dir>/floor-sfm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrate_person_floor import camera_source_binding

FLIP = np.array([1.0, -1.0, -1.0])  # raw spz -> viewer world (fourd.html's world.rotation.x = PI)


def digest(path: Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def gravity_normal(coordinates: str) -> np.ndarray:
    """The packaged frame is levelled, so up is exactly +y; refuse anything else.

    scripts/frame_align.py rotates the whole solve before packaging, and every packaged file says
    so in its `coordinates` string. Fitting a free normal here would re-absorb a tilt that has
    already been removed once.
    """
    if "gravity on +y" not in coordinates:
        raise ValueError("Requires a levelled packaged frame whose coordinates put gravity on +y")
    return np.array([0.0, 1.0, 0.0])


def track_mask(points: np.ndarray, track_xz: np.ndarray, radius: float) -> np.ndarray:
    """Points whose xz lies within `radius` of any sampled foot position.

    A grid keyed on `radius` keeps this linear in the cloud size: a point is kept when any of the
    nine cells around it holds a foot sample.
    """
    points, track_xz = np.asarray(points, float), np.asarray(track_xz, float)
    if points.ndim != 2 or points.shape[1] != 3 or track_xz.ndim != 2 or track_xz.shape[1] != 2:
        raise ValueError("Expected (n,3) points and (m,2) track positions")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be finite and positive")
    cells = set()
    for cx, cz in np.floor(track_xz / radius).astype(np.int64):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cells.add((int(cx) + dx, int(cz) + dz))
    keys = np.floor(points[:, [0, 2]] / radius).astype(np.int64)
    return np.fromiter((tuple(k) in cells for k in keys), dtype=bool, count=len(points))


def fit_offset(levels: np.ndarray, band: float, iters: int = 12) -> tuple[float, np.ndarray]:
    """Solve only the plane's offset along a known normal, robustly.

    Seeded by the MODE of a histogram rather than a low percentile: a percentile sits on whatever
    fuzz hangs below the surface, which is the same reason scripts/place_solve.py's `fit_plane`
    seeds from a mode. Then trimmed-mean the inliers inside `band` until they stop moving.
    """
    levels = np.asarray(levels, float)
    if levels.ndim != 1 or len(levels) < 32 or not np.isfinite(levels).all():
        raise ValueError("Need at least 32 finite floor candidates")
    if not np.isfinite(band) or band <= 0:
        raise ValueError("Band must be finite and positive")
    counts, edges = np.histogram(levels, bins=max(16, min(256, len(levels) // 32)))
    offset = float(0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1]))
    inliers = np.abs(levels - offset) < band
    for _ in range(iters):
        if inliers.sum() < 16:
            raise ValueError("Floor band holds too few points")
        moved = float(np.mean(levels[inliers]))
        nxt = np.abs(levels - moved) < band
        if moved == offset and np.array_equal(nxt, inliers):
            break
        offset, inliers = moved, nxt
    return offset, inliers


def measure_plane(
    points, track_xz, normal, foot_level, radius, band, body_height, grow=(1, 2, 4, 8)
):
    """The floor under a walked footprint: offset along `normal`, its inliers and its residual.

    Only points at or below the feet are candidates. Without that the fit can climb onto whatever
    the person walks past, and a plane above the feet cannot be a floor they stand on. A sparse
    cloud can leave the first column empty, so the column widens through `grow` before giving up;
    the multiple that was actually used is reported, because a wide column is a weaker measurement
    of the ground the person stands on.
    """
    normal = np.asarray(normal, float)
    if abs(np.linalg.norm(normal) - 1) > 1e-6:
        raise ValueError("Normal must be a unit vector")
    if not np.isfinite(body_height) or body_height <= 0:
        raise ValueError("Body height must be finite and positive")
    ceiling = foot_level + 0.25 * body_height
    for multiple in grow:
        near = track_mask(points, track_xz, radius * multiple)
        levels = np.asarray(points, float)[near] @ normal
        below = levels <= ceiling
        if below.sum() >= 32:
            break
    else:
        raise ValueError(
            f"Only {int(below.sum())} observed points under the walked footprint within "
            f"{radius * grow[-1]:.4f} units; no floor was measured"
        )
    levels = levels[below]
    offset, inliers = fit_offset(levels, band)
    if offset > ceiling:
        raise ValueError("Fitted surface sits above the feet; that is not a floor")
    rms = float(np.sqrt(np.mean((levels[inliers] - offset) ** 2)))
    return dict(
        offset=offset,
        inliers=int(inliers.sum()),
        candidates=int(len(levels)),
        nearTrack=int(near.sum()),
        residualRmsBodyHeights=rms / body_height,
        radiusUnits=float(radius * multiple),
        radiusGrowth=int(multiple),
        bandUnits=float(band),
    )


def static_cloud(anchors: Path, cameras: list[dict], samples, conf_pct=20.0):
    """Pi3X dense anchor frames as one cloud in the packaged frame.

    The same construction scripts/bake_video_colours.py uses for its depth diagnostic: each anchor's
    points go through its own camera, and one scale relates the anchor poses to the packaged ones.
    """
    a = np.load(anchors)
    poses = a["poses"].astype(np.float64)
    pub = np.array([cameras[i]["camera_to_world"] for i in samples], np.float64)
    src, dst = poses[:, :3, 3] - poses[:, :3, 3].mean(0), pub[:, :3, 3] - pub[:, :3, 3].mean(0)
    spread = float((src**2).sum())
    if spread <= 0:
        raise ValueError("Anchor poses coincide; no cloud scale can be fitted")
    scale = float(np.sqrt((dst**2).sum() / spread))
    keep = a["valid"] & ~a["people"] & (a["conf"] > np.percentile(a["conf"], conf_pct))
    flip = np.diag(FLIP)
    out = []
    for i in range(len(poses)):
        p = a["points"][i][keep[i]].astype(np.float64)
        c = (p - poses[i][:3, 3]) @ poses[i][:3, :3]
        c = c[c[:, 2] > 0] * scale
        out.append(((c @ flip) @ pub[i][:3, :3].T + pub[i][:3, 3]).astype(np.float32))
    return np.concatenate(out), scale


def world_cloud(spz: Path, registration_scale: float, min_alpha=0.5):
    """Opaque Marble splat centres in packaged native units."""
    from render_world_poses import read_spz

    if not np.isfinite(registration_scale) or registration_scale <= 0:
        raise ValueError("Registration scale must be finite and positive")
    g = read_spz(spz)
    keep = g["alpha"] > min_alpha
    return (g["xyz"][keep].astype(np.float64) * FLIP) / registration_scale


def foot_track(package: Path, person: dict, quantile: float, opacity: float):
    """Per-sample foot level and xz from the packaged per-frame PLYs.

    The PLYs are the sequence's own listed frames and every one is checked against its recorded
    sha256, so this reads the packaged geometry, not a re-derived copy of it.
    """
    from plyfile import PlyData

    sequence = json.loads((package / person["sequence"]).read_text())
    directory = (package / person["sequence"]).parent
    frames, hashes = sequence["frames"], sequence["frame_sha256"]
    if len(frames) != len(hashes):
        raise ValueError("Sequence frame/hash length mismatch")
    logit = float(np.log(opacity / (1 - opacity)))
    levels, xz = [], []
    for name, sha in zip(frames, hashes):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Frame escapes sequence directory")
        if digest(path) != sha:
            raise ValueError(f"Integrity mismatch: {path}")
        v = PlyData.read(path)["vertex"].data
        opaque = np.asarray(v["opacity"], float) >= logit
        if opaque.sum() < 100:
            raise ValueError(f"Too few opaque splats in {name}")
        xyz = np.stack([np.asarray(v[k], float) for k in ("x", "y", "z")], 1)[opaque]
        low = float(np.quantile(xyz[:, 1], quantile))
        near = xyz[:, 1] < low + 0.04 * float(np.ptp(xyz[:, 1]))
        levels.append(low)
        xz.append([float(xyz[near, 0].mean()), float(xyz[near, 2].mean())])
    return np.array(levels), np.array(xz), sequence


def build(args) -> dict:
    package = args.people.parent
    people = json.loads(args.people.read_text())
    cameras = json.loads((package / people["cameras"]).read_text())
    binding = camera_source_binding(cameras, people)
    normal = gravity_normal(people["coordinates"])
    primary = next(p for p in people["people"] if p["id"] == people["primary"])
    body_height = float(primary["bodyHeightUnits"])
    levels, xz, _ = foot_track(package, primary, args.foot_quantile, args.opacity)
    inputs = {"people": args.people, "cameras": package / people["cameras"]}
    radius = args.radius_body_heights * body_height
    band = args.band_body_heights * body_height
    foot = float(np.quantile(levels, 0.5))

    planes = {}
    if args.source in ("sfm", "both"):
        n = len(cameras["cameras"])
        samples = [int(round(k * (n - 1) / 7)) for k in range(8)]
        cloud, cloud_scale = static_cloud(args.anchors, cameras["cameras"], samples)
        planes["sfm"] = measure_plane(cloud, xz, normal, foot, radius, band, body_height) | dict(
            provider="pi3x-static-anchor-cloud",
            anchorCloudScale=cloud_scale,
            anchorSamples=samples,
            note="Floor the footage observed: Pi3X depth of the recorded frames, in the packaged frame.",
        )
        inputs["anchors"] = args.anchors
    if args.source in ("world", "both"):
        cloud = world_cloud(args.world, args.registration_scale)
        planes["world"] = measure_plane(cloud, xz, normal, foot, radius, band, body_height) | dict(
            provider="marble-world-ground",
            registrationScale=args.registration_scale,
            note="Floor the viewer draws: Marble's own opaque splats, divided back into native units.",
        )
        inputs["world"] = args.world

    chosen = planes[args.prefer if args.prefer in planes else next(iter(planes))]
    cross = None
    if len(planes) == 2:
        gap = (planes["world"]["offset"] - planes["sfm"]["offset"]) / body_height
        cross = dict(
            disagreementBodyHeights=gap,
            note=(
                "World minus observed floor. A registered world agrees here; a large gap is a "
                "registration failure that no single placement scalar can remove."
            ),
        )
    return dict(
        schema="wander.observed-floor/1",
        coordinates="packaged native units; " + people["coordinates"],
        clip=people.get("clip"),
        sourceSha256=people["sourceSha256"],
        cameraSourceBinding=binding,
        normal=normal.tolist(),
        offset=chosen["offset"],
        bodyHeightUnits=body_height,
        residualRmsBodyHeights=chosen["residualRmsBodyHeights"],
        floorSource=chosen["provider"],
        footLevelMedianUnits=foot,
        footAboveFloorBodyHeights=(foot - chosen["offset"]) / body_height,
        observations=[planes[k] | dict(key=k) for k in planes],
        crossCheck=cross,
        measurementOnly=(
            "Measured surfaces, not a physical survey. residualRmsBodyHeights is the consistency "
            "of the estimated depth in the fitted band."
        ),
        inputs={k: dict(path=str(v), sha256=digest(v)) for k, v in inputs.items()},
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--people", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="source", choices=("sfm", "world", "both"), default="sfm")
    ap.add_argument("--prefer", choices=("sfm", "world"), default="sfm")
    ap.add_argument("--anchors", type=Path, help="Pi3X anchors.npz, for the sfm provider")
    ap.add_argument("--world", type=Path, help="Marble .spz, for the world provider")
    ap.add_argument(
        "--registration-scale", type=float, help="native -> viewer world, world provider"
    )
    ap.add_argument("--radius-body-heights", type=float, default=1.5)
    ap.add_argument("--band-body-heights", type=float, default=0.06)
    ap.add_argument("--foot-quantile", type=float, default=0.01)
    ap.add_argument("--opacity", type=float, default=0.5)
    a = ap.parse_args()
    if a.source in ("sfm", "both") and not a.anchors:
        ap.error("--from sfm needs --anchors")
    if a.source in ("world", "both") and not (a.world and a.registration_scale):
        ap.error("--from world needs --world and --registration-scale")
    if a.out.exists():
        ap.error("Choose a new output path")
    doc = build(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as file:
        json.dump(doc, file, indent=2)
        file.write("\n")
    print(
        json.dumps(
            {
                k: doc[k]
                for k in (
                    "floorSource",
                    "offset",
                    "residualRmsBodyHeights",
                    "footAboveFloorBodyHeights",
                    "crossCheck",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
