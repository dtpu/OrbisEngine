"""Resolve a constant source-frame clock without silently accepting conflicting metadata."""

from __future__ import annotations

import math
from statistics import median


def positive_fps(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} fps must be a finite positive number")
    return float(value)


def clock_tolerance(fps):
    # Allow small rounding differences in recorded frame timestamps, not another source rate.
    return max(0.01, fps * 1e-3)


def camera_source_fps(cameras):
    """Return a constant source FPS, or None only when camera timing is unavailable.

    These are source indices and source seconds, not the lower-rate reconstruction sample clock.
    A partial, reversed or variable clock is an error rather than permission to assume 30 FPS.
    """
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("camera clock needs nonempty camera rows")
    if any(not isinstance(row, dict) for row in cameras):
        raise ValueError("camera clock rows must be objects")
    times = [row.get("time") for row in cameras]
    if all(time is None for time in times):
        return None
    indices = [row.get("sourceIndex") for row in cameras]
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("camera source indices must be nonnegative integers")
    if any(
        isinstance(time, bool)
        or not isinstance(time, (int, float))
        or not math.isfinite(time)
        or time < 0
        for time in times
    ):
        raise ValueError("camera time must be complete, finite and nonnegative")
    if len(times) < 2:
        return None
    rates = []
    for i in range(1, len(times)):
        di, dt = indices[i] - indices[i - 1], times[i] - times[i - 1]
        if di <= 0 or dt <= 0:
            raise ValueError("camera source indices and times must increase strictly")
        rates.append(di / dt)
    fps = median(rates)
    if any(abs(rate - fps) > clock_tolerance(fps) for rate in rates):
        raise ValueError("variable camera timing is unsupported for ballistic fitting")
    return positive_fps(fps, "camera")


def resolve_source_fps(**sources):
    """Use the first available source in precedence order, checking all other known clocks."""
    known = [
        (label, positive_fps(value, label)) for label, value in sources.items() if value is not None
    ]
    if not known:
        return 30.0  # Legacy fits/camera manifests have no source-clock metadata.
    label, fps = known[0]
    for other, value in known[1:]:
        if abs(fps - value) > clock_tolerance(value):
            raise ValueError(f"{label}/{other} fps conflict; refusing an inconsistent source clock")
    return fps
