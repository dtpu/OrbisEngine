#!/usr/bin/env python3
"""Where the source camera sits relative to the walker, measured from the footage itself.

place_solve.py's size check takes the person as 1.70 m and wants a handheld camera 1.15-1.85 m
above the floor. After the silhouette fit, the quantity it checks is the camera's height over the
walker's floor in body-heights, and angles alone fix that quantity. With the walker upright on the
floor, a camera H above it sees the feet at depression angle a_f and the top of the head at a_h, and

    H / stature = tan(a_f) / (tan(a_f) - tan(a_h)).

No registration scale, camera translation or avatar size enters it, only the camera's tilt against
the floor. So a camera outside the handheld band is either really there or that tilt is wrong. This
module measures the tilt twice, independently:

  registration  the camera track's rotation against the world's fitted floor normal;
  image         gravity from the source frames' own near-vertical lines (tree trunks, door frames,
                wall edges), with the walker excluded. Each sampled frame's segments are carried
                into the track frame by the track's rotations and pooled, so one vanishing
                direction is fitted to every frame's lines at once. Relative rotations are what a
                camera solve gets right; the absolute tilt against the floor is what is tested.

When the two agree, and the fitted placement shows the same height, the footage has measured where
the camera is, and the handheld prior can be set aside. When they disagree, or too few frames show
enough vertical lines, nothing is concluded and the handheld band stands.

Measured before adoption: on the lobby and elevator clips, whose camera tracks carry a 9.8 and 7.6
degree tilt that frame_align.py corrects, image gravity disagreed with the raw track by that tilt and
agreed with the frame-aligned track within 0.6 degrees (p90, per frame).

The answer is in body-heights. Metres still assume a 1.70 m walker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
# A candidate vertical is at least this long, as a fraction of the frame height, and within this
# many degrees of the image's vertical axis.
MIN_SEGMENT_FRACTION = 0.04
MAX_SEGMENT_TILT_DEG = 15.0
# Gravity must lie within this angle of the world's down axis. It rules out a cluster of parallel
# diagonals; a frame tilted further than this is not measured.
MAX_GRAVITY_TILT_DEG = 30.0
# A segment supports a direction when its back-projected plane passes within this angle of it.
INLIER_DEG = 0.6
# Enough support, from enough of the clip, that one frame's structure cannot decide it.
MIN_INLIERS = 30
MIN_FRAMES = 5
MIN_FRAME_INLIERS = 3
# Lines on the walker are not world-vertical; drop segments near the silhouette.
WALKER_MARGIN_FRACTION = 0.05
MAX_FRAMES = 12
# Same width as the scale gate's 0.9-1.1 band.
TOLERANCE = 0.10
RANSAC_ITERATIONS = 2000


def rotation(camera_to_world) -> np.ndarray:
    """The orthonormal part of an OpenGL camera_to_world."""
    u, _, vt = np.linalg.svd(np.asarray(camera_to_world, float)[:3, :3])
    return u @ vt


def camera_down(camera_to_world, world_down) -> np.ndarray:
    """A world direction in OpenCV camera axes (x right, y down the image, z forward)."""
    down = OPENGL_TO_OPENCV @ (rotation(camera_to_world).T @ np.asarray(world_down, float))
    return down / np.linalg.norm(down)


def tilt_degrees(down) -> tuple[float, float]:
    """(pitch, roll) of a camera whose world-down is `down`; positive pitch looks down."""
    down = np.asarray(down, float)
    pitch = float(np.degrees(np.arcsin(np.clip(down[2], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(-down[0], down[1])))
    return pitch, roll


def body_heights(K, down, top: float, bottom: float, column: float) -> float | None:
    """Camera height above the walker's floor, in walker heights, from one upright silhouette.

    None when the feet are not below the horizon or the head is not above the feet in angle:
    the formula needs a camera above the floor looking at a standing body.
    """
    K = np.asarray(K, float)
    down = np.asarray(down, float) / np.linalg.norm(down)

    def tan_depression(row: float) -> float:
        ray = np.linalg.solve(K, [column, row, 1.0])
        s = float(ray @ down / np.linalg.norm(ray))
        return s / np.sqrt(max(1e-12, 1.0 - s * s))

    feet, head = tan_depression(bottom), tan_depression(top)
    if feet <= 0 or feet - head <= 1e-9:
        return None
    return feet / (feet - head)


def vertical_segments(gray, walker_box=None) -> np.ndarray:
    """Long near-vertical line segments as rows (x1, y1, x2, y2, length), away from the walker."""
    import cv2

    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return np.zeros((0, 5))
    s = lines[:, 0].astype(float)
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    length = np.hypot(dx, dy)
    keep = (length >= MIN_SEGMENT_FRACTION * gray.shape[0]) & (
        np.abs(dx) <= np.tan(np.radians(MAX_SEGMENT_TILT_DEG)) * np.abs(dy)
    )
    if walker_box is not None:
        left, top, right, bottom = walker_box
        middle = ((s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2)
        for x, y in ((s[:, 0], s[:, 1]), (s[:, 2], s[:, 3]), middle):
            keep &= ~((x >= left) & (x <= right) & (y >= top) & (y <= bottom))
    return np.column_stack([s[keep], length[keep]])


def segment_planes(segments, K) -> np.ndarray:
    """Unit normals, in OpenCV camera axes, of the planes through the camera and each segment."""
    segments = np.asarray(segments, float).reshape(-1, 5)
    k_inv = np.linalg.inv(np.asarray(K, float))
    ones = np.ones(len(segments))
    p1 = np.c_[segments[:, 0], segments[:, 1], ones] @ k_inv.T
    p2 = np.c_[segments[:, 2], segments[:, 3], ones] @ k_inv.T
    planes = np.cross(p1, p2)
    return planes / np.linalg.norm(planes, axis=1, keepdims=True)


def common_direction(planes, weights, axis, *, seed: int = 0):
    """(direction, inliers): the direction most planes contain, within MAX_GRAVITY_TILT_DEG of axis.

    Parallel world lines share one direction, perpendicular to all their planes. RANSAC on pairs,
    then a weighted least-squares refit on the inliers. The direction points along `axis`.
    """
    planes = np.asarray(planes, float)
    weights = np.asarray(weights, float)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    if len(planes) < 2:
        return None
    limit = np.sin(np.radians(INLIER_DEG))
    upright = np.cos(np.radians(MAX_GRAVITY_TILT_DEG))
    rng = np.random.default_rng(seed)
    best_score, best = -1.0, None
    for _ in range(RANSAC_ITERATIONS):
        i, j = rng.choice(len(planes), 2, replace=False)
        direction = np.cross(planes[i], planes[j])
        norm = np.linalg.norm(direction)
        if norm < 1e-9 or abs(direction @ axis) < upright * norm:
            continue
        inliers = np.abs(planes @ (direction / norm)) < limit
        score = float(weights[inliers].sum())
        if score > best_score:
            best_score, best = score, inliers
    if best is None or best.sum() < 2:
        return None
    _, _, vt = np.linalg.svd(planes[best] * np.sqrt(weights[best])[:, None])
    direction = vt[-1] if vt[-1] @ axis >= 0 else -vt[-1]
    if direction @ axis < upright:
        return None
    return direction, best


def sample_sources(rows, limit: int = MAX_FRAMES) -> list[int]:
    """Evenly spaced source indices among the measured silhouettes."""
    sources = sorted(int(r["sourceIndex"]) for r in rows)
    if len(sources) <= limit:
        return sources
    picks = np.linspace(0, len(sources) - 1, limit).round().astype(int)
    return [sources[i] for i in dict.fromkeys(picks.tolist())]


def _summary(values):
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return dict(frames=0, bodyHeightsMedian=None)
    return dict(
        frames=int(len(values)),
        bodyHeightsMedian=float(np.median(values)),
        bodyHeightsP10=float(np.percentile(values, 10)),
        bodyHeightsP90=float(np.percentile(values, 90)),
    )


def measure(rows, cameras, normal, gray_by_source) -> dict:
    """Camera height in body-heights with the tilt from the registration and from the image.

    rows: uncropped upright silhouettes, {sourceIndex, top, bottom, left, right} in source px.
    gray_by_source: grayscale source frames for the sampled rows.
    """
    by_source = {int(c["sourceIndex"]): c for c in cameras}
    rows = [
        r for r in sorted(rows, key=lambda r: r["sourceIndex"]) if r["sourceIndex"] in by_source
    ]
    floor_down = -np.asarray(normal, float) / np.linalg.norm(normal)

    planes, weights, owners, segment_counts = [], [], [], {}
    for row in rows:
        gray = gray_by_source.get(int(row["sourceIndex"]))
        if gray is None:
            continue
        camera = by_source[int(row["sourceIndex"])]
        margin = WALKER_MARGIN_FRACTION * gray.shape[0]
        box = (
            row["left"] - margin,
            row["top"] - margin,
            row["right"] + margin,
            row["bottom"] + margin,
        )
        segments = vertical_segments(gray, box)
        segment_counts[int(row["sourceIndex"])] = int(len(segments))
        if not len(segments):
            continue
        # camera (OpenCV) -> camera (OpenGL) -> world, by the track's rotation alone
        to_world = rotation(camera["camera_to_world"]) @ OPENGL_TO_OPENCV
        planes.append(segment_planes(segments, camera["source_intrinsics"]) @ to_world.T)
        weights.append(segments[:, 4])
        owners.append(np.full(len(segments), int(row["sourceIndex"])))
    image_down, inliers, support = None, None, {}
    if planes:
        planes, weights, owners = (
            np.concatenate(planes),
            np.concatenate(weights),
            np.concatenate(owners),
        )
        # The guard axis is the world's own down (the viewer's -y), not the floor fit under test.
        found = common_direction(planes, weights, [0.0, -1.0, 0.0])
        if found is not None:
            image_down, mask = found
            inliers = int(mask.sum())
            sources, counts = np.unique(owners[mask], return_counts=True)
            support = {int(s): int(c) for s, c in zip(sources, counts)}

    registration, image, per_frame = [], [], []
    for row in rows:
        camera = by_source[int(row["sourceIndex"])]
        K = np.asarray(camera["source_intrinsics"], float)
        column = (row["left"] + row["right"]) / 2
        down = camera_down(camera["camera_to_world"], floor_down)
        value = body_heights(K, down, row["top"], row["bottom"], column)
        pitch, roll = tilt_degrees(down)
        registration.append((value, pitch, roll))
        frame = dict(sourceIndex=int(row["sourceIndex"]), registrationBodyHeights=value)
        if image_down is not None:
            down = camera_down(camera["camera_to_world"], image_down)
            image_value = body_heights(K, down, row["top"], row["bottom"], column)
            image.append((image_value, *tilt_degrees(down)))
            frame["imageBodyHeights"] = image_value
        if int(row["sourceIndex"]) in segment_counts:
            frame.update(
                segments=segment_counts[int(row["sourceIndex"])],
                inliers=support.get(int(row["sourceIndex"]), 0),
            )
            per_frame.append(frame)

    def tilt_medians(entries):
        if not entries:
            return {}
        return dict(
            pitchDegMedian=float(np.median([e[1] for e in entries])),
            rollDegMedian=float(np.median([e[2] for e in entries])),
        )

    supporting = sum(1 for c in support.values() if c >= MIN_FRAME_INLIERS)
    return dict(
        method="camera height over the walker's floor in body-heights, "
        "tan(a_f)/(tan(a_f)-tan(a_h)), with the camera tilt from the registration's floor "
        "normal and, independently, from vertical lines pooled across sampled frames",
        registration=dict(**_summary([r[0] for r in registration]), **tilt_medians(registration)),
        image=dict(
            **_summary([i[0] for i in image]),
            **tilt_medians(image),
            framesSampled=len(segment_counts),
            framesSupporting=supporting,
            segments=int(sum(segment_counts.values())),
            inliers=inliers,
            tiltFromFloorNormalDeg=float(
                np.degrees(np.arccos(np.clip(image_down @ floor_down, -1.0, 1.0)))
            )
            if image_down is not None
            else None,
        ),
        perFrame=per_frame,
        tolerance=TOLERANCE,
        minFrames=MIN_FRAMES,
        minInliers=MIN_INLIERS,
    )


def footage_verdict(geometry: dict, fitted_body_heights: float) -> dict:
    """Did the footage measure the camera height the fitted placement shows?"""
    why = []
    registration = geometry["registration"]["bodyHeightsMedian"]
    image = geometry["image"]
    tolerance = geometry["tolerance"]
    if registration is None:
        why.append("no upright, uncropped silhouette gave a camera height")
    elif image["bodyHeightsMedian"] is None or image["inliers"] is None:
        why.append(
            f"no common vertical direction in {image['segments']} near-vertical segments "
            f"over {image['framesSampled']} sampled frames"
        )
    elif (
        image["inliers"] < geometry["minInliers"]
        or image["framesSupporting"] < geometry["minFrames"]
    ):
        why.append(
            f"gravity rests on {image['inliers']} vertical segments from {image['framesSupporting']} "
            f"frames (need {geometry['minInliers']} from {geometry['minFrames']})"
        )
    else:
        off = image["bodyHeightsMedian"] / registration - 1
        if abs(off) > tolerance:
            why.append(
                f"gravity from the frames' vertical lines puts the camera "
                f"{image['bodyHeightsMedian']:.3f} body-heights above the floor, the registration "
                f"{registration:.3f}: {abs(off) * 100:.1f}% apart (limit {tolerance * 100:.0f}%)"
            )
    if registration is not None:
        off = fitted_body_heights / registration - 1
        if abs(off) > tolerance:
            why.append(
                f"the fitted placement puts the camera {fitted_body_heights:.3f} body-heights above "
                f"the floor, the footage {registration:.3f}: {abs(off) * 100:.1f}% apart "
                f"(limit {tolerance * 100:.0f}%)"
            )
    return dict(verified=not why, fittedBodyHeights=fitted_body_heights, why=why)


def read_gray_frames(video: Path, source_indices, size) -> dict[int, np.ndarray]:
    """Exact source frames, decoded the way the person masks were, as grayscale."""
    import cv2

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from worker.wander_worker.source_timing import select_source_frames

    indices = sorted(int(i) for i in source_indices)
    if not indices:
        return {}
    width, height = int(size[0]), int(size[1])
    rgb, provenance = select_source_frames(
        Path(video), source_indices=indices, width=width, height=height
    )
    if [f["sourceIndex"] for f in provenance["frames"]] != indices:
        raise ValueError("decoded frames do not match the requested source indices")
    frames = np.frombuffer(rgb, np.uint8).reshape(len(indices), height, width, 3)
    return {i: cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for i, f in zip(indices, frames)}


def measure_footage(video: Path, rows, cameras, normal) -> dict:
    """Decode the sampled frames and measure; the entry point place_solve.py uses."""
    picks = sample_sources(rows)
    gray = read_gray_frames(video, picks, cameras[0]["source_image_size"])
    return measure(rows, cameras, normal, gray)
