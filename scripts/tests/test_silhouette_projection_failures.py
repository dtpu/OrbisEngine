"""Invalid avatar projections must reject a fit with the original track/frame identified."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import place_solve
import silhouette_rows
from test_silhouette_cropping import fit_inputs, write_masks, mask


@pytest.fixture
def arguments(tmp_path):
    args, _ = fit_inputs(tmp_path)
    write_masks(args[7], {sample: mask() for sample in range(3)})
    return args


def test_behind_camera_geometry_has_clear_track_and_source_error(arguments):
    for opaque, all_points in arguments[0][0][1]:
        opaque[:, 2] = 2
        all_points[:, 2] = 2
    with pytest.raises(
        ValueError, match="track person, sourceIndex 10, base: insufficient in-front splats"
    ):
        place_solve.silhouette_fit(*arguments)


def test_nonfinite_geometry_has_clear_projection_error(arguments):
    arguments[0][0][1][0][0][:, 1] = np.nan
    with pytest.raises(
        ValueError,
        match="track person, sourceIndex 10, base: invalid silhouette projection.*nonfinite",
    ):
        place_solve.silhouette_fit(*arguments)


@pytest.mark.parametrize("key,value", [("top", np.nan), ("bottom", np.inf), ("zMedian", -1)])
def test_invalid_projected_row_measurement_is_rejected(arguments, monkeypatch, key, value):
    rows = dict(top=50, bottom=150, zMedian=2)
    rows[key] = value
    monkeypatch.setattr(silhouette_rows, "avatar_rows", lambda *args: rows)
    with pytest.raises(
        ValueError, match="track person, sourceIndex 10, base: invalid silhouette projection"
    ):
        place_solve.silhouette_fit(*arguments)


@pytest.mark.parametrize(
    "call,phase", [(2, "u perturbation"), (3, "w perturbation"), (4, "size perturbation")]
)
def test_failed_finite_difference_measurement_is_not_fabricated(
    arguments, monkeypatch, call, phase
):
    original = silhouette_rows.avatar_rows
    calls = 0

    def measure(*args):
        nonlocal calls
        calls += 1
        return None if calls == call else original(*args)

    monkeypatch.setattr(silhouette_rows, "avatar_rows", measure)
    with pytest.raises(
        ValueError, match=f"track person, sourceIndex 10, {phase}: insufficient in-front splats"
    ):
        place_solve.silhouette_fit(*arguments)
    assert calls == call


def test_invalid_final_projection_is_not_reported_as_a_successful_fit(arguments, monkeypatch):
    original = silhouette_rows.avatar_rows
    calls = 0

    def measure(*args):
        nonlocal calls
        calls += 1
        # Three frames * (base + two directions + size) in one converged iteration.
        return None if calls == 13 else original(*args)

    monkeypatch.setattr(silhouette_rows, "avatar_rows", measure)
    monkeypatch.setattr(
        np.linalg, "lstsq", lambda matrix, targets, **kwargs: (np.zeros(matrix.shape[1]),)
    )
    with pytest.raises(
        ValueError, match="track person, sourceIndex 10, final: insufficient in-front splats"
    ):
        place_solve.silhouette_fit(*arguments)
    assert calls == 13


def test_nonfinite_solver_step_is_rejected_before_application(arguments, monkeypatch):
    monkeypatch.setattr(
        np.linalg, "lstsq", lambda matrix, targets, **kwargs: (np.full(matrix.shape[1], np.nan),)
    )
    with pytest.raises(
        ValueError, match="track person, sourceIndices.*nonfinite silhouette fit step"
    ):
        place_solve.silhouette_fit(*arguments)


def test_nonfinite_equations_fail_before_linear_algebra(arguments, monkeypatch):
    arguments[3][0][0]["res"] = np.nan

    def solve(*args, **kwargs):
        pytest.fail("nonfinite measurements must not reach the linear solver")

    monkeypatch.setattr(np.linalg, "lstsq", solve)
    with pytest.raises(
        ValueError, match="track person, sourceIndices.*nonfinite silhouette fit equations"
    ):
        place_solve.silhouette_fit(*arguments)


def test_update_that_moves_valid_geometry_behind_camera_fails_explicitly(arguments, monkeypatch):
    original = silhouette_rows.avatar_rows
    calls = 0

    def measure(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    def update(matrix, targets, **kwargs):
        step = np.zeros(matrix.shape[1])
        step[:-1:2] = -1  # Repeated finite steps toward/through the camera along each feet ray.
        return (step,)

    monkeypatch.setattr(silhouette_rows, "avatar_rows", measure)
    monkeypatch.setattr(np.linalg, "lstsq", update)
    with pytest.raises(
        ValueError, match="track person, sourceIndex 10.*insufficient in-front splats"
    ):
        place_solve.silhouette_fit(*arguments)
    assert calls > 12  # Initial projections worked; this failure came from a later update.
