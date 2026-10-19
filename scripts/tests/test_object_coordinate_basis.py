"""Fresh package_multiperson coordinates and legacy floor-fit compatibility."""

import contextlib
import copy
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from object_coordinate_basis import resolve_object_basis


def fresh_people():
    # Same coordinate fields emitted by package_multiperson, including its
    # camera-centred size correction and rounded 30 -> 12 fps sample grid.
    return {
        "schema": "wander.people/1",
        "fps": 12,
        "samples": 6,
        "sourceIndices": [0, 2, 5, 8, 10, 12],
        "timestamps": [f / 30 for f in [0, 2, 5, 8, 10, 12]],
        "sharedPlacement": {"bodyHeightUnits": 0.85, "feetY": -0.76},
        "people": [
            {
                "id": f"person{i}",
                "track": i,
                "registrationScale": 0.8,
                "sizeCorrection": {"multiplier": 0.625, "appliedAboutSourceCamera": True},
                "transform": {
                    "translation": [0, 0, 0],
                    "quaternionXYZW": [0, 0, 0, 1],
                    "scale": 1,
                },
            }
            for i in range(2)
        ],
    }


def legacy_people():
    people = fresh_people()
    people["floorFit"] = {
        "metresPerWorldUnit": 1.8,
        "sharedCameraDrift": {"samples": [0, 2, 5], "offsetYUnits": [0.1, 0.2, -0.1]},
        "levels": {
            "C_sharedDriftPlusConstant": {"translationY": {"person0": -0.03, "person1": 0.04}}
        },
    }
    people["people"][0]["transform"]["translation"][1] = -0.08  # old level B
    return people


def test_fresh_prior_has_zero_corrections_and_camera_centred_scales():
    basis = resolve_object_basis(fresh_people())
    assert basis.metres_per_world_unit == 2
    assert basis.scale == {0: 0.5, 1: 0.5}
    assert basis.constants == {0: 0, 1: 0}
    assert [basis.drift_at(f) for f in (-1, 0, 4, 12, 20)] == [0] * 5
    assert basis.provenance["mode"] == "avatarStaturePrior"
    assert basis.provenance["staturePriorMetres"] == 1.70
    assert "no measured metric anchor" in basis.provenance["caveat"]


def test_legacy_corrections_remain_identical():
    basis = resolve_object_basis(legacy_people())
    assert basis.metres_per_world_unit == 1.8
    assert basis.constants == {0: -0.03, 1: 0.04}
    np.testing.assert_allclose(
        [basis.drift_at(f) for f in (0, 2.5, 5, 8.5, 12)], [0.1, 0.15, 0.2, 0.05, -0.1]
    )
    basis.validate_fit({})  # Pre-provenance legacy fit remains supported.


@pytest.mark.parametrize("floor", [None, {}, {"metresPerWorldUnit": 2}])
def test_present_incomplete_floor_fit_never_falls_back(floor):
    people = fresh_people()
    people["floorFit"] = floor
    with pytest.raises(ValueError, match="incomplete"):
        resolve_object_basis(people)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "2"])
def test_invalid_scales_fail(value):
    for factory, path in [
        (fresh_people, ("sharedPlacement", "bodyHeightUnits")),
        (legacy_people, ("floorFit", "metresPerWorldUnit")),
    ]:
        people = factory()
        people[path[0]][path[1]] = value
        with pytest.raises(ValueError, match="finite positive"):
            resolve_object_basis(people)


@pytest.mark.parametrize(
    "drift",
    [
        {"samples": [], "offsetYUnits": []},
        {"samples": [0, 1], "offsetYUnits": [0]},
        {"samples": [0, 6], "offsetYUnits": [0, 1]},
        {"samples": [1, 0], "offsetYUnits": [0, 1]},
        {"samples": [0, 0], "offsetYUnits": [0, 1]},
        {"samples": [0, 0.5], "offsetYUnits": [0, 1]},
        {"samples": [0, 1], "offsetYUnits": [0, float("nan")]},
    ],
)
def test_invalid_legacy_drift_fails(drift):
    people = legacy_people()
    people["floorFit"]["sharedCameraDrift"] = drift
    with pytest.raises(ValueError, match="floorFit"):
        resolve_object_basis(people)


@pytest.mark.parametrize("translations", [{"person0": 0}, {"person0": 0, "person1": float("inf")}])
def test_invalid_legacy_constants_fail(translations):
    people = legacy_people()
    people["floorFit"]["levels"]["C_sharedDriftPlusConstant"]["translationY"] = translations
    with pytest.raises(ValueError):
        resolve_object_basis(people)


@pytest.mark.parametrize(
    "key,value",
    [
        ("translation", [0, 1, 0]),
        ("translation", [1, 0, 0]),
        ("quaternionXYZW", [1, 0, 0, 0]),
        ("scale", 2),
    ],
)
def test_fresh_nonidentity_transforms_fail(key, value):
    people = fresh_people()
    people["people"][0]["transform"][key] = value
    with pytest.raises(ValueError, match="unsupported nonidentity"):
        resolve_object_basis(people)


def test_fit_basis_cannot_cross_modes_or_change_corrections():
    people = fresh_people()
    basis = resolve_object_basis(people)
    fit = {"coordinateBasis": basis.provenance, "metresPerWorldUnit": basis.metres_per_world_unit}
    basis.validate_fit(fit)
    with pytest.raises(ValueError, match="no coordinateBasis"):
        basis.validate_fit({})
    with pytest.raises(ValueError, match="conflicts"):
        resolve_object_basis(legacy_people()).validate_fit(fit)
    for field in ("registrationScale", "id"):
        altered = copy.deepcopy(people)
        altered["people"][0][field] = 0.9 if field == "registrationScale" else "changed"
        with pytest.raises(ValueError, match="conflicts"):
            resolve_object_basis(altered).validate_fit(fit)
    legacy = legacy_people()
    fit["coordinateBasis"] = resolve_object_basis(legacy).provenance
    legacy["floorFit"]["sharedCameraDrift"]["offsetYUnits"][0] += 0.01
    with pytest.raises(ValueError, match="conflicts"):
        resolve_object_basis(legacy).validate_fit(fit)


def test_real_fresh_lift_and_package_preserve_basis_and_gates(tmp_path):
    torch = pytest.importorskip("torch", reason="needs inference group")
    import lift_object_3d
    import package_objects

    people = fresh_people()
    (tmp_path / "people.json").write_text(json.dumps(people))
    cameras = {
        "cameras": [
            {
                "sourceIndex": f,
                "time": f / 30,
                "camera_to_world": np.eye(4).tolist(),
                "source_intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            }
            for f in people["sourceIndices"]
        ]
    }
    (tmp_path / "cameras.json").write_text(json.dumps(cameras))
    frames = list(range(13))
    gravity = 9.80665 / 2
    positions = [
        package_objects.ballistic_position(
            [0, 0, -2], [1, gravity * 0.2, 0], [0, -gravity, 0], f, 0, 30
        )
        for f in frames
    ]
    for track in range(2):
        directory = tmp_path / "tracks" / f"track_{track:02d}"
        directory.mkdir(parents=True)
        # Wrists exactly meet the synthetic release/catch in the same registered frame.
        joints = np.tile([track * 0.8, 0, 4], (22, 1))
        torch.save(
            {
                "sourceIndices": people["sourceIndices"],
                "poses": [{"j3d": joints} for _ in people["sourceIndices"]],
            },
            directory / "source-poses.pt",
        )
    camera = lift_object_3d.load_cameras(tmp_path / "cameras.json")
    pixels = [lift_object_3d.project(camera, f, p)[0].tolist() for f, p in zip(frames, positions)]
    (tmp_path / "flights.json").write_text(
        json.dumps({"fps": 30, "flights": [{"frames": frames, "px": pixels}]})
    )
    args = [
        "lift_object_3d.py",
        "--people",
        str(tmp_path / "people.json"),
        "--cameras",
        str(tmp_path / "cameras.json"),
        "--tracks-dir",
        str(tmp_path / "tracks"),
        "--flights",
        str(tmp_path / "flights.json"),
        "--thrower",
        "0",
        "--catcher",
        "1",
        "--out",
        str(tmp_path / "fit.json"),
    ]
    with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
        lift_object_3d.main()
    fit = json.loads((tmp_path / "fit.json").read_text())
    assert len(fit["flights"]) == 1
    assert fit["droppedFlights"] == []
    assert fit["gates"] == {"minObs": 10, "maxHandMetres": 1.3, "maxReprojPx": 8.0}
    assert fit["constantY"] == 0
    assert fit["coordinateBasis"]["mode"] == "avatarStaturePrior"
    args = [
        "package_objects.py",
        "--world",
        str(tmp_path),
        "--fit",
        str(tmp_path / "fit.json"),
        "--cameras",
        str(tmp_path / "cameras.json"),
        "--tracks-dir",
        str(tmp_path / "tracks"),
    ]
    with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
        package_objects.main()
    packaged = json.loads((tmp_path / "objects.json").read_text())
    obj = packaged["objects"][0]
    assert (
        packaged["coordinateBasis"] == fit["coordinateBasis"] == obj["evidence"]["coordinateBasis"]
    )
    assert obj["appliesSharedCameraDrift"] is False
    assert obj["transform"]["translation"] == [0, 0, 0]
    np.testing.assert_allclose(obj["bakedTrack"]["positions"], positions, atol=1e-5)
    # A later floor correction cannot reuse this raw-frame fit or overwrite output.
    (tmp_path / "people.json").write_text(json.dumps(legacy_people()))
    previous = (tmp_path / "objects.json").read_text()
    with patch.object(sys, "argv", args), pytest.raises(SystemExit, match="conflicts"):
        package_objects.main()
    assert (tmp_path / "objects.json").read_text() == previous


def test_legacy_packager_keeps_nonzero_floor_corrections(tmp_path):
    pytest.importorskip("torch", reason="needs inference group")
    import package_objects
    from test_package_objects_timing import PackageObjectTimingTests

    args, output = PackageObjectTimingTests().package_fixture(tmp_path, 30)
    path = tmp_path / "world/people.json"
    people = json.loads(path.read_text())
    people["floorFit"]["sharedCameraDrift"]["offsetYUnits"] = [0.1, 0.2, -0.1]
    people["floorFit"]["levels"]["C_test"]["translationY"] = {"person0": -0.03, "person1": 0.04}
    path.write_text(json.dumps(people))
    with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
        package_objects.main()
    obj = json.loads(output.read_text())["objects"][0]
    assert obj["appliesSharedCameraDrift"] is True
    np.testing.assert_allclose(obj["transform"]["translation"], [0, 0.005, 0])
    np.testing.assert_allclose(obj["bakedTrack"]["positions"][30], [1, -1.108325, -2], atol=1e-5)
