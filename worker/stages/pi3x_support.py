"""Finite camera normalization and explicit person support; no inference dependencies."""

import numpy as np


MIN_SUPPORT = 100


def static_scale_reference(base, anchors, source_indices, times, minimum=MIN_SUPPORT):
    """Keep camera zero's coordinates; normalize using the first supported static view."""
    counts = []
    for j, sample in enumerate(anchors):
        pose = base["poses"][j]
        if not np.isfinite(pose).all():
            raise ValueError(f"Nonfinite anchor pose {j}")
        inverse = np.linalg.inv(pose)
        points = base["points"][j]
        finite_points = np.isfinite(points).all(-1)
        depth = (
            np.where(finite_points[..., None], points, 0) @ inverse[:3, :3].T + inverse[:3, 3]
        )[..., 2]
        good = (
            base["valid"][j] & ~base["people"][j] & finite_points & np.isfinite(depth) & (depth > 0)
        )
        count = int(good.sum())
        counts.append(count)
        if count < minimum:
            continue
        median = float(np.median(depth[good]))
        scale = 3.0 / median
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid static normalization in anchor {j}")
        return scale, dict(
            anchorOrdinal=j,
            sample=int(sample),
            sourceIndex=int(source_indices[sample]),
            time=float(times[sample]),
            supportPoints=count,
            precedingSupportCounts=counts[:-1],
            medianOwnCameraDepth=median,
            scale=scale,
            minimumSupport=minimum,
            method="3 / median finite positive static depth in selected anchor camera coordinates; arbitrary normalization, not metric scale",
        )
    raise ValueError(f"No supported static anchor for normalization; support counts {counts}")


def fit_ray_intrinsics(rays, valid, source_size):
    """Fit estimated pinhole optics; depth-unfiltered fallback requires <=1px RMS."""
    h, w = rays.shape[:2]
    if min(h, w, *source_size) <= 0:
        raise ValueError("Invalid intrinsic image dimensions")
    y, x = np.mgrid[:h, :w]
    finite = np.isfinite(rays).all(-1) & (rays[..., 2] > 1e-6)
    grid = (y % 2 == 0) & (x % 2 == 0)
    good = valid & finite & grid
    depth_count = int(good.sum())
    fallback = depth_count < MIN_SUPPORT
    if fallback:
        good = finite & grid
    if int(good.sum()) < MIN_SUPPORT:
        raise ValueError("Insufficient finite positive predicted rays for intrinsic fit")
    xy = rays[good, :2] / rays[good, 2:3]
    pixels = np.stack([x[good] + 0.5, y[good] + 0.5], axis=1)
    K = np.eye(3)
    errors = []
    for axis in range(2):
        design = np.column_stack([xy[:, axis], np.ones(len(xy))])
        coefficients, _, rank, _ = np.linalg.lstsq(design, pixels[:, axis], rcond=None)
        focal, principal = coefficients
        if rank != 2 or not np.isfinite(coefficients).all() or focal <= 0:
            raise ValueError("Degenerate, nonfinite or nonpositive fitted intrinsic")
        K[axis, axis], K[axis, 2] = focal, principal
        errors.append(design @ coefficients - pixels[:, axis])
    rmse = float(np.sqrt(np.mean(np.sum(np.square(errors), axis=0))))
    source_K = np.diag([source_size[0] / w, source_size[1] / h, 1.0]) @ K
    if not np.isfinite(source_K).all() or not np.isfinite(rmse):
        raise ValueError("Nonfinite intrinsic matrix or residual")
    if fallback and rmse > 1.0:
        raise ValueError(f"Depth-unfiltered predicted-ray fit residual {rmse:.6g}px exceeds 1px")
    return dict(
        intrinsics=K.tolist(),
        source_intrinsics=source_K.tolist(),
        image_size=[w, h],
        source_image_size=list(source_size),
        intrinsic_fit_pixel_rmse=rmse,
        intrinsicsSource="predicted-rays-depth-unfiltered"
        if fallback
        else "predicted-rays-depth-valid",
        intrinsicFitRayCount=int(good.sum()),
        intrinsicDepthValidRayCount=depth_count,
        intrinsicFallbackMaxPixelRmse=1.0 if fallback else None,
        intrinsics_method=(
            "Least-squares zero-skew pinhole fit to predicted rays; pixel centers at x+0.5,y+0.5. "
            "Source K undoes the two resize operations. Estimated optics, not supplied calibration. "
            + (
                "Depth confidence/edge mask had insufficient support; fit uses all finite positive "
                "predicted rays on the same stride-two grid, with <=1 pixel RMS required."
                if fallback
                else "Fit uses depth-valid predicted rays on a stride-two grid."
            )
        ),
    )


def person_support(sample, source_index, seconds, point_count):
    return dict(
        sample=int(sample),
        sourceIndex=int(source_index),
        time=float(seconds),
        points=int(point_count),
        supported=int(point_count) >= MIN_SUPPORT,
        reason=None if point_count >= MIN_SUPPORT else "insufficient-person-support",
    )


def person_sequence_metadata(support):
    frames = [f"frame_{r['sample']:03d}.ply" if r["supported"] else None for r in support]
    gaps = [r for r in support if not r["supported"]]
    return dict(
        frames=frames,
        personFrames=[name for name in frames if name is not None],
        personSupport=support,
        personAbsentFrames=gaps,
        personAbsentCount=len(gaps),
        personPresentCount=len(support) - len(gaps),
        cameraOnly=bool(support) and len(gaps) == len(support),
        minPersonPoints=MIN_SUPPORT,
        personGapNote="Null person frames retain their camera/time slots. Insufficient mask/depth support is not proof of physical absence; no person geometry is invented.",
    )
