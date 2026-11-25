"""CPU geometry for experimental calibrated crops; no inference validation implied."""

import numpy as np


def _finite(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        array = raw.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    return array


def _scalar(value, name, positive=False, integer=False):
    array = _finite(value, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    number = float(array)
    if number < 0 or (positive and number == 0) or (integer and not number.is_integer()):
        raise ValueError(
            f"{name} must be {'positive' if positive else 'nonnegative'}"
            f"{' and integral' if integer else ''}"
        )
    return number


def _bounds(bounds):
    values = _finite(bounds, "bounds")
    if (
        values.shape != (4,)
        or (values < 0).any()
        or (values != np.floor(values)).any()
        or values[2] <= values[0]
        or values[3] <= values[1]
    ):
        raise ValueError("bounds must be nonnegative integer xyxy with positive extent")
    return values


def crop_bounds(box, width, height, padding=0.2):
    """Return half-open integer xyxy, padding each side by a fraction of box size.

    Round outward, then clamp to the source. The original box must intersect the
    image; padding cannot turn an entirely off-image detection into a valid crop.
    """
    width = _scalar(width, "width", positive=True, integer=True)
    height = _scalar(height, "height", positive=True, integer=True)
    padding = _scalar(padding, "padding")
    box = _finite(box, "box")
    if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box must be xyxy with positive extent")
    if box[0] >= width or box[1] >= height or box[2] <= 0 or box[3] <= 0:
        raise ValueError("box must intersect the source image")
    with np.errstate(over="ignore", invalid="ignore"):
        margin = (box[2:] - box[:2]) * padding
        expanded = np.concatenate((box[:2] - margin, box[2:] + margin))
    if not np.isfinite(expanded).all():
        raise ValueError("padded box must be finite")
    low = np.maximum(np.floor(expanded[:2]), 0)
    high = np.minimum(np.ceil(expanded[2:]), [width, height])
    return tuple(int(v) for v in np.concatenate((low, high)))


def crop_intrinsics(K, bounds):
    """Translate calibrated source intrinsics into crop pixels, preserving skew.

    K must be a normalized pinhole intrinsic matrix with positive focal lengths.
    Its principal point may lie outside the crop (or the source image).
    """
    matrix = _finite(K, "K")
    bounds = _bounds(bounds)
    if (
        matrix.shape != (3, 3)
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.array_equal(matrix[2], [0, 0, 1])
        or matrix[1, 0] != 0
    ):
        raise ValueError("K must be normalized upper-triangular pinhole intrinsics")
    translation = np.array([[1, 0, -bounds[0]], [0, 1, -bounds[1]], [0, 0, 1]])
    with np.errstate(over="ignore", invalid="ignore"):
        result = translation @ matrix
    if not np.isfinite(result).all():
        raise ValueError("crop intrinsics must be finite")
    return result


def source_joints(joints, pad_left, pad_top, factor, offset_w, offset_h, bounds):
    """Map an N x 2 estimator pixel array back to original source pixels.

    Undo preprocessing padding, resize, and center padding in that order, then
    add the crop origin. Predictions outside the crop remain unclipped.
    """
    joints = _finite(joints, "joints")
    if joints.ndim != 2 or joints.shape[1] != 2 or not len(joints):
        raise ValueError("joints must be a nonempty N x 2 array")
    pad = [_scalar(pad_left, "pad_left"), _scalar(pad_top, "pad_top")]
    scale = _scalar(factor, "factor", positive=True)
    offset = [_scalar(offset_w, "offset_w"), _scalar(offset_h, "offset_h")]
    origin = _bounds(bounds)[:2]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (joints - pad) / scale - offset + origin
    if not np.isfinite(result).all():
        raise ValueError("source joints must be finite")
    return result
