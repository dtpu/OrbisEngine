"""The silhouette fit matches the edge the viewer draws, not the splat centres, to the mask."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)]

import place_solve
import silhouette_rows
from motion_silhouette import splat_coverage
from test_silhouette_cropping import write_masks

# An avatar of 3 px splats, 1.6 units tall at depth 4 under a 400 px focal: 160 px of centres.
K = [[400.0, 0.0, 50.0], [0.0, 400.0, 100.0], [0.0, 0.0, 1.0]]
SIZE = (100, 200)
SOURCES = (10, 40, 90)
xs, ys = np.meshgrid(np.arange(-0.15, 0.151, 0.02), np.arange(-0.8, 0.801, 0.02))
POINTS = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -4.0)])
SPLATS = dict(
    xyz=POINTS,
    logScales=np.full((len(POINTS), 3), np.log(0.03)),
    quats=np.tile([1.0, 0.0, 0.0, 0.0], (len(POINTS), 1)),
    opacity=np.full(len(POINTS), 5.0),
)
CAMERA = {"camera_to_world": np.eye(4).tolist(), "source_intrinsics": K}


def drawn(points, size=1.0):
    return (
        splat_coverage(
            points,
            SPLATS["opacity"],
            SPLATS["logScales"] + np.log(size),
            SPLATS["quats"],
            CAMERA,
            SIZE[::-1],
        )
        > 0.5
    )


def fit(tmp_path, footprints):
    # The walker is filmed exactly as the viewer draws this avatar where it already stands.
    masks = write_masks(tmp_path / "tracks/masks.npz", {i: drawn(POINTS) for i in range(3)})
    args = (
        [("person", [(POINTS, POINTS)] * 3, np.zeros(3), None, {"sourceIndices": list(SOURCES)})],
        [
            (
                None,
                None,
                None,
                {"sourceIndices": list(SOURCES), "people": [{"id": "person", "track": 0}]},
            )
        ],
        [
            dict(
                sourceIndex=s,
                camera_to_world=CAMERA["camera_to_world"],
                source_intrinsics=K,
                source_image_size=list(SIZE),
            )
            for s in SOURCES
        ],
        [[{"x": 0.0, "y": -0.8, "z": -4.0, "res": 0.0}] * 3],
        1.0,
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
        masks,
        1.0,
        0.03,
        0.05,
        1.6,
        0.15,
    )
    return place_solve.silhouette_fit(*args, footprints=footprints)["person"]


def drawn_rows_after(person, index=0):
    size = person["sizeScale"]
    placed = POINTS * size + np.asarray(person["offsetUnits"][index])
    return silhouette_rows.mask_rows(drawn(placed, size), 1.0, SIZE[1])


def test_fitting_drawn_rows_leaves_an_avatar_that_already_covers_the_walker_alone(tmp_path):
    target = silhouette_rows.mask_rows(drawn(POINTS), 1.0, SIZE[1])
    person = fit(tmp_path, [lambda _frame: SPLATS])
    after = drawn_rows_after(person)
    assert abs(after["top"] - target["top"]) <= 1 and abs(after["bottom"] - target["bottom"]) <= 1
    assert person["rowsFitted"] == "drawn" and person["heightRatioMedian"] == pytest.approx(
        1, abs=0.01
    )
    overhang = person["drawnOverhang"]
    assert overhang["topMedian"] > 0.01 and overhang["bottomMedian"] > 0.01
    assert person["drawnStatureRatio"] == pytest.approx(
        1 + overhang["topMedian"] + overhang["bottomMedian"], rel=1e-6
    )


def test_fitting_splat_centres_to_the_mask_draws_the_avatar_too_tall(tmp_path):
    target = silhouette_rows.mask_rows(drawn(POINTS), 1.0, SIZE[1])
    person = fit(tmp_path, None)
    after = drawn_rows_after(person)
    assert person["rowsFitted"] == "splat centres" and "drawnStatureRatio" not in person
    # Centre rows were stretched to the mask's edges, so the drawn footprint overhangs them.
    assert (after["bottom"] - after["top"]) - (target["bottom"] - target["top"]) >= 4


@pytest.mark.parametrize("veil", [False, True])
def test_drawn_rows_are_the_rows_of_the_full_render(veil):
    xyz, opacity = POINTS.copy(), SPLATS["opacity"].copy()
    if veil:
        # A faint column above the head: drawn, but under half covered for longer than the
        # head band, so the edge must come from the fallback over the whole canvas.
        faint = np.column_stack([np.zeros(60), np.linspace(0.85, 2.0, 60), np.full(60, -4.0)])
        xyz, opacity = np.vstack([xyz, faint]), np.r_[opacity, np.full(60, -3.0)]
    scales = np.full((len(xyz), 3), np.log(0.03))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (len(xyz), 1))
    tall = (400, SIZE[0])  # room above the body for the veil
    camera = dict(CAMERA, source_intrinsics=[[400.0, 0.0, 50.0], [0.0, 400.0, 300.0], [0, 0, 1]])
    full = silhouette_rows.mask_rows(
        splat_coverage(xyz, opacity, scales, quats, camera, tall) > 0.5, 1.0, tall[0]
    )
    rows = silhouette_rows.drawn_rows(
        xyz, opacity, scales, quats, np.eye(3), np.zeros(3), np.array(camera["source_intrinsics"])
    )
    assert (rows["top"], rows["bottom"]) == (full["top"], full["bottom"])


def test_the_drawn_fit_stops_once_its_edges_are_the_drawn_edges(tmp_path):
    overhang = fit(tmp_path, [lambda _frame: SPLATS])["drawnOverhang"]
    assert overhang["lastChangePx"] <= place_solve.OVERHANG_TOLERANCE_PX
    assert 0 <= overhang["refits"] <= place_solve.OVERHANG_REFITS
