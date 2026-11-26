"""A contact plane needs finite, two-dimensional support; a seed is not a fit."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import place_solve


def plane_points(count=500):
    rng = np.random.default_rng(7)
    x, z = rng.uniform(-1, 1, (2, count))
    return np.column_stack([x, 0.01 * x - 0.02 * z + 0.3, z])


@pytest.mark.parametrize("count", [0, 333, 499])
def test_seed_is_never_returned_with_insufficient_support(count):
    points = plane_points(1000)
    points[:count, 1] = 0.3
    points[count:, 1] = 1.3
    with pytest.raises(place_solve.PlaneFitError, match="500.*inliers"):
        place_solve.fit_plane(points, seed=0.3)


def test_exactly_500_supported_points_fit_the_tilt_and_report_final_support():
    points = plane_points()
    outliers = plane_points(30) + [0, 1, 0]
    points = np.concatenate([points, outliers])
    coef, normal, distance, count = place_solve.fit_plane(points, seed=0.3)
    np.testing.assert_allclose(coef, [0.01, -0.02, 0.3], atol=1e-12)
    assert np.linalg.norm(normal) == pytest.approx(1)
    np.testing.assert_allclose(points[:500] @ normal, distance, atol=1e-12)
    residual = points[:, 1] - (points[:, 0] * coef[0] + points[:, 2] * coef[1] + coef[2])
    assert count == np.count_nonzero(np.abs(residual) < 0.055) == 500


@pytest.mark.parametrize("iters", [1, 2, 12])
def test_fit_that_loses_support_cannot_return_the_previous_inlier_count(iters):
    points = plane_points()
    points[:, 1] = 0.054
    points[:10, 1] = -0.054
    # All 500 are inside the seed band; the fitted plane excludes the ten low samples.
    with pytest.raises(place_solve.PlaneFitError, match="500.*inliers"):
        place_solve.fit_plane(points, seed=0, iters=iters)


@pytest.mark.parametrize("kind", ["coincident", "collinear"])
def test_many_points_do_not_rescue_a_rank_deficient_plane(kind):
    points = np.zeros((700, 3))
    if kind == "collinear":
        points[:, 0] = np.linspace(-1, 1, len(points))
        points[:, 2] = 2 * points[:, 0] + 1
    with pytest.raises(place_solve.PlaneFitError, match="degenerate"):
        place_solve.fit_plane(points, seed=0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_not_silently_dropped(value):
    points = plane_points(600)
    points[-1, 0] = value
    with pytest.raises(ValueError, match="finite") as error:
        place_solve.fit_plane(points, seed=0.3)
    assert type(error.value) is ValueError


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": np.nan},
        {"seed": np.inf},
        {"band": 0},
        {"band": -1},
        {"band": np.inf},
        {"iters": 0},
        {"iters": -1},
        {"iters": 1.5},
        {"y0_pct": np.nan},
    ],
)
def test_invalid_fit_controls_are_refused(kwargs):
    with pytest.raises(ValueError) as error:
        place_solve.fit_plane(plane_points(), **kwargs)
    assert type(error.value) is ValueError


@pytest.mark.parametrize("points", [np.zeros(500), np.zeros((500, 2))])
def test_malformed_point_sets_are_not_support_refusals(points):
    with pytest.raises(ValueError) as error:
        place_solve.fit_plane(points)
    assert type(error.value) is ValueError


@pytest.mark.parametrize("count", [0, 333, 499])
def test_too_few_finite_points_raise_typed_support_refusal(count):
    with pytest.raises(place_solve.PlaneFitError, match="500.*inliers") as error:
        place_solve.fit_plane(plane_points(count))
    assert isinstance(error.value, ValueError)


def test_nonfinite_solver_result_is_refused(monkeypatch):
    monkeypatch.setattr(
        np.linalg, "lstsq", lambda *args, **kwargs: (np.array([np.nan, 0, 0]), [], 3, np.ones(3))
    )
    with pytest.raises(ValueError, match="finite") as error:
        place_solve.fit_plane(plane_points(), seed=0.3)
    assert type(error.value) is ValueError


def test_placement_cli_does_not_write_a_seed_as_a_surface(tmp_path, monkeypatch):
    from test_placement_run_inputs import stub_placement_cli

    cli = stub_placement_cli(tmp_path, monkeypatch, -1.4, 1, None)
    points = plane_points(333)
    points[:, 1] = -1.4
    points[:, 2] -= 2
    monkeypatch.setattr(place_solve, "world_splats", lambda _: points)
    write_source_masks(cli.masks, usable=False)
    with pytest.raises(place_solve.PlaneFitError, match="500.*inliers"):
        place_solve.main()
    assert not cli.report.exists()
    assert cli.fits == []


def write_source_masks(path, *, usable=True):
    mask = np.zeros((480, 640), bool)
    if usable:
        mask[200:330, 310:330] = True
    np.savez_compressed(
        path,
        shape=[480, 640, 1],
        **{f"t0_s{sample:03d}": np.packbits(mask) for sample in range(8)},
    )
    motion = path.parent / "track_00/motion.json"
    motion.parent.mkdir(exist_ok=True)
    motion.write_text(
        json.dumps(
            {
                "frames": [
                    {"sample": i, "sourceIndex": i, "box": [310, 200, 330, 330]} for i in range(8)
                ]
            }
        )
    )


def source_support_cli(tmp_path, monkeypatch, count):
    from test_placement_run_inputs import stub_placement_cli

    cli = stub_placement_cli(tmp_path, monkeypatch, -0.7, 1, None)
    local = plane_points(333)
    local[:, 1] = -0.7
    # Keep this unsupported seed neighbourhood outside the source below-feet patch.
    local[:, 2] = -2 + 0.05 * local[:, 2]
    # Distinct pixels in the real helper's below-feet patch on its 384-pixel-wide grid.
    # Back-project those rays onto y=-1.4 with the fixture's camera intrinsics.
    u, v = np.meshgrid(np.arange(160, 190) + 0.5, np.arange(200, 220) + 0.5)
    depth = 300 * 1.4 / (v.ravel() - 144)
    ground = np.column_stack([(u.ravel() - 192) * depth / 300, np.full(len(depth), -1.4), -depth])[
        :count
    ]
    world = np.concatenate([local, ground, ground + [0, 2.8, 0]])
    monkeypatch.setattr(place_solve, "world_splats", lambda _: world)
    write_source_masks(cli.masks)
    return cli, ground


@pytest.mark.parametrize("count", [500, 600])
def test_cli_recovers_from_actual_source_visible_support(tmp_path, monkeypatch, count):
    cli, _ = source_support_cli(tmp_path, monkeypatch, count)
    place_solve.main()
    result = json.loads(cli.report.read_text())
    surface = result["surface"]
    assert surface["sourceFloorSupport"]["uniquePoints"] == count
    assert "333" in surface["sourceFloorSupport"]["initialFitRefusal"]
    assert surface["sourceFloorSupport"]["inferredSurface"] is True
    assert surface["floorPlane"]["inliers"] == count
    np.testing.assert_allclose(surface["floorPlane"]["coef"], [0, 0, -1.4], atol=1e-12)
    assert surface["sizeCheck"]["pass_"] is True
    assert len(cli.fits) == 1


@pytest.mark.parametrize("count", [0, 499])
def test_cli_refuses_insufficient_source_support_without_output(tmp_path, monkeypatch, count):
    cli, _ = source_support_cli(tmp_path, monkeypatch, count)
    with pytest.raises(place_solve.PlaneFitError, match=f"only {count} unique points"):
        place_solve.main()
    assert not cli.report.exists()
    assert cli.fits == []


def test_cli_refuses_missing_source_masks_without_output(tmp_path, monkeypatch):
    cli, _ = source_support_cli(tmp_path, monkeypatch, 600)
    cli.masks.unlink()
    with pytest.raises(FileNotFoundError, match="Tracked masks are missing"):
        place_solve.main()
    assert not cli.report.exists()
    assert cli.fits == []


def test_cli_rechecks_degeneracy_after_source_support_selection(tmp_path, monkeypatch):
    import floor_support

    cli, _ = source_support_cli(tmp_path, monkeypatch, 600)
    line = np.column_stack([np.zeros(600), np.full(600, -1.4), np.linspace(-7, -5, 600)])
    # Isolate the caller/fit boundary: a helper receipt cannot authorize a rank-deficient plane.
    monkeypatch.setattr(
        floor_support, "source_floor_points", lambda *args: (line, {"uniquePoints": 600})
    )
    with pytest.raises(place_solve.PlaneFitError, match="degenerate"):
        place_solve.main()
    assert not cli.report.exists()
    assert cli.fits == []


def test_valid_local_plane_does_not_invoke_source_support(tmp_path, monkeypatch):
    import floor_support
    from test_placement_run_inputs import stub_placement_cli

    cli = stub_placement_cli(tmp_path, monkeypatch, -1.4, 1, None)

    def unexpected(*args, **kwargs):
        pytest.fail("a supported local plane must not invoke source support")

    monkeypatch.setattr(floor_support, "source_floor_points", unexpected)
    place_solve.main()
    surface = json.loads(cli.report.read_text())["surface"]
    assert "sourceFloorSupport" not in surface
    assert surface["floorPlane"]["inliers"] >= 500
    assert len(cli.fits) == 1


def add_second_person(tmp_path):
    world = tmp_path / "external-world"
    sequence = world / "second/sequence.json"
    sequence.parent.mkdir()
    sequence.write_bytes((world / "person/sequence.json").read_bytes())
    people = world / "people.json"
    manifest = json.loads(people.read_text())
    manifest["people"].append({"id": "second", "track": 1})
    people.write_text(json.dumps(manifest))


def test_unsupported_multi_person_plane_refuses_before_source_fallback(tmp_path, monkeypatch):
    import floor_support

    cli, _ = source_support_cli(tmp_path, monkeypatch, 600)
    add_second_person(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("multi-person refusal must precede mask union and source support")

    monkeypatch.setattr(floor_support, "single_person_floor_masks", unexpected)
    monkeypatch.setattr(floor_support, "source_floor_points", unexpected)
    with pytest.raises(place_solve.PlaneFitError, match="one independently masked person"):
        place_solve.main()
    assert not cli.report.exists()
    assert cli.fits == []


def test_supported_multi_person_plane_still_uses_local_fit(tmp_path, monkeypatch):
    from copy import deepcopy

    import floor_support
    from test_placement_run_inputs import stub_placement_cli

    cli = stub_placement_cli(tmp_path, monkeypatch, -1.4, 1, None)
    add_second_person(tmp_path)
    single_fit = place_solve.silhouette_fit

    def two_person_fit(*args, footprints, **kwargs):
        assert len(footprints) == 2
        result = single_fit(*args, footprints=footprints[:1], **kwargs)
        result["second"] = deepcopy(result["person"])
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("a supported multi-person plane must not invoke source support")

    monkeypatch.setattr(place_solve, "silhouette_fit", two_person_fit)
    monkeypatch.setattr(floor_support, "source_floor_points", unexpected)
    place_solve.main()
    report = json.loads(cli.report.read_text())
    assert set(report["perPerson"]) == {"person", "second"}
    assert "sourceFloorSupport" not in report["surface"]
    assert report["surface"]["floorPlane"]["inliers"] >= 500
    assert report["surface"]["sizeCheck"]["pass_"] is True


@pytest.mark.parametrize("tilt", [0, 5, 14.9, 15.1, 45])
def test_source_floor_tilt_guard_refuses_steep_but_accepts_near_horizontal_support(
    tmp_path, monkeypatch, tilt
):
    import floor_support

    cli, _ = source_support_cli(tmp_path, monkeypatch, 600)
    points = plane_points(600)
    points[:, 0] *= 0.01
    points[:, 2] -= 6
    slope = np.tan(np.radians(tilt))
    points[:, 1] = slope * points[:, 0] - 1.4
    # Isolate the caller's tilt gate, retaining the real fitter. The narrow x footprint keeps
    # all 600 finite, independent observations inside its initial band even for a 45-degree face.
    histogram, edges = np.histogram(points[:, 1], 90)
    seed = float((edges[histogram.argmax()] + edges[histogram.argmax() + 1]) / 2)
    coef, _, _, count = place_solve.fit_plane(points, seed=seed)
    assert count == 600
    np.testing.assert_allclose(coef, [slope, 0, -1.4], atol=1e-12)
    monkeypatch.setattr(
        floor_support,
        "source_floor_points",
        lambda *args: (points, {"uniquePoints": 600, "inferredSurface": True}),
    )
    if tilt > 15:
        with pytest.raises(place_solve.PlaneFitError, match="at most 15 degrees"):
            place_solve.main()
        assert not cli.report.exists()
        assert cli.fits == []
    else:
        place_solve.main()
        surface = json.loads(cli.report.read_text())["surface"]
        assert surface["floorPlane"]["tiltDeg"] == pytest.approx(tilt, abs=1e-10)
        assert surface["floorPlane"]["inliers"] == 600
        assert surface["sourceFloorSupport"]["maxTiltDegrees"] == 15
        assert len(cli.fits) == 1
