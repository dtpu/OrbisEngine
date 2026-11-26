"""Source floor support must be observed, correctly mapped world points, never a seed plane."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor_support import single_person_floor_masks, source_floor_points


def camera(source=10, translation=(0, 0, 0), rotation=None):
    pose = np.eye(4)
    pose[:3, 3] = translation
    if rotation is not None:
        pose[:3, :3] = rotation
    return {
        "sourceIndex": source,
        "source_image_size": [100, 100],
        "source_intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        "camera_to_world": pose.tolist(),
    }


def person_mask():
    mask = np.zeros((100, 100), bool)
    mask[20:60, 45:55] = True
    return mask


def point_at(u, v, depth):
    """Independent pinhole fixture: an OpenGL world point for an identity source camera."""
    return [(u - 50) * depth / 100, -(v - 50) * depth / 100, -depth]


def test_masks_are_keyed_by_source_index_not_camera_position():
    points = np.array([point_at(50.5, 65.5, 4)])
    cameras = [camera(90), camera(10), camera(40)]
    masks = {0: person_mask(), 10: person_mask(), 90: np.zeros((100, 100), bool)}
    result, receipt = source_floor_points(points, cameras, 1, masks, width=100)
    np.testing.assert_array_equal(result, points)
    assert [frame["sourceIndex"] for frame in receipt["frames"]] == [10]
    assert receipt["uniquePoints"] == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_foreground_occlusion_is_resolved_before_selecting_floor_support(reverse):
    # Both centres project to the same patch pixel. The higher foreground point must hide
    # the lower, floor-like point even when an avatar-local height filter would drop it.
    near = point_at(50.5, 65.5, 2)
    hidden_floor = point_at(50.5, 65.5, 10)
    points = np.array([near, hidden_floor][:: -1 if reverse else 1])
    result, receipt = source_floor_points(points, [camera()], 1, {10: person_mask()}, width=100)
    np.testing.assert_array_equal(result, [near])
    assert receipt["frames"][0]["visiblePoints"] == 1


@pytest.mark.parametrize("scale", [0.5, 2.0])
@pytest.mark.parametrize("rotated", [False, True])
def test_native_camera_translation_is_scaled_once_and_output_stays_in_world_units(scale, rotated):
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) if rotated else np.eye(3)
    native_translation = np.array([12, -7, 9])
    local = np.array(point_at(50.5, 65.5, 4))
    expected = rotation @ local + native_translation * scale
    result, receipt = source_floor_points(
        np.array([expected]),
        [camera(10, native_translation, rotation)],
        scale,
        {10: person_mask()},
        width=100,
    )
    np.testing.assert_array_equal(result, [expected])
    assert receipt["uniquePoints"] == 1


@pytest.mark.parametrize("mask_size", [50, 100])
@pytest.mark.parametrize("grid_width", [50, 100, 200])
def test_patch_uses_mask_coordinates_and_requested_projection_resolution(mask_size, grid_width):
    mask = np.zeros((mask_size, mask_size), bool)
    mask[mask_size // 5 : 3 * mask_size // 5, 9 * mask_size // 20 : 11 * mask_size // 20] = True
    inside = point_at(50.5, 65.5, 4)
    outside = point_at(90.5, 65.5, 4)
    result, receipt = source_floor_points(
        np.array([inside, outside]), [camera()], 1, {10: mask}, width=grid_width
    )
    np.testing.assert_array_equal(result, [inside])
    assert receipt["frames"][0]["maskSize"] == [mask_size, mask_size]


@pytest.mark.parametrize("grid_width", [100, 200])
def test_non_square_source_uses_separate_horizontal_and_vertical_dimensions(grid_width):
    cam = camera()
    cam["source_image_size"] = [200, 100]
    cam["source_intrinsics"] = [[200, 0, 100], [0, 100, 50], [0, 0, 1]]
    mask = np.zeros((50, 100), bool)
    mask[10:30, 45:55] = True
    # At depth four these project to source pixels (101, 65.5) and (170, 65.5).
    # Only the first lies below the person, whose centre is at source x=99.
    points = np.array([[0.02, -0.62, -4], [1.4, -0.62, -4]])
    result, receipt = source_floor_points(points, [cam], 1, {10: mask}, width=grid_width)
    np.testing.assert_array_equal(result, points[:1])
    assert receipt["frames"][0]["maskSize"] == [100, 50]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_finite_binary_numeric_masks_remain_valid(dtype):
    points = np.array([point_at(50.5, 65.5, 4)])
    result, receipt = source_floor_points(
        points, [camera()], 1, {10: person_mask().astype(dtype)}, width=100
    )
    np.testing.assert_array_equal(result, points)
    assert receipt["uniquePoints"] == 1


@pytest.mark.parametrize(
    "kind",
    ["unknown", "empty", "all-unknown", "top", "bottom", "left", "right", "1d", "3d", "zero-size"],
)
def test_unknown_empty_malformed_and_edge_cropped_masks_supply_no_support(kind):
    mask = person_mask()
    if kind == "unknown":
        mask = None
    elif kind == "empty":
        mask[:] = False
    elif kind == "all-unknown":
        mask[:] = True
    elif kind == "top":
        mask[0, 50] = True
    elif kind == "bottom":
        mask[-1, 50] = True
    elif kind == "left":
        mask[40, 0] = True
    elif kind == "right":
        mask[40, -1] = True
    elif kind == "1d":
        mask = np.ones(100, bool)
    elif kind == "3d":
        mask = np.ones((100, 100, 1), bool)
    elif kind == "zero-size":
        mask = np.empty((0, 100), bool)
    result, receipt = source_floor_points(
        np.array([point_at(50.5, 65.5, 4)]), [camera()], 1, {10: mask}, width=100
    )
    assert result.shape == (0, 3)
    assert receipt["uniquePoints"] == 0
    assert receipt["frames"] == []


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_mask_values_cannot_invent_a_person_silhouette(invalid):
    # Boolean coercion turns all of these unknown measurements into foreground. Neither
    # a full refusal nor skipping the frame may expose world points as floor support.
    mask = np.zeros((100, 100), float)
    mask[20:60, 45:55] = invalid
    try:
        result, receipt = source_floor_points(
            np.array([point_at(50.5, 65.5, 4)]), [camera()], 1, {10: mask}, width=100
        )
    except ValueError:
        return
    assert result.shape == (0, 3), "invalid mask values must not create a below-feet patch"
    assert receipt["uniquePoints"] == 0


def test_repeated_views_do_not_inflate_unique_support_count():
    points = np.array([point_at(45.5, 65.5, 4), point_at(55.5, 66.5, 5)])
    sources = [90, 10, 40]
    result, receipt = source_floor_points(
        points, [camera(i) for i in sources], 1, {i: person_mask() for i in sources}, width=100
    )
    np.testing.assert_array_equal(result, points)
    assert receipt["uniquePoints"] == 2
    assert [f["visiblePoints"] for f in receipt["frames"]] == [2, 2, 2]
    assert [f["sourceIndex"] for f in receipt["frames"]] == sources
    assert receipt["inferredSurface"] is True


def test_sampling_limit_selects_bounded_source_frames_without_borrowing_masks():
    sources = [90, 10, 70, 30, 50]
    points = np.array([point_at(50.5, 65.5, 4)])
    result, receipt = source_floor_points(
        points,
        [camera(i) for i in sources],
        1,
        {i: person_mask() for i in sources},
        width=100,
        max_frames=3,
    )
    np.testing.assert_array_equal(result, points)
    assert [f["sourceIndex"] for f in receipt["frames"]] == [90, 70, 50]


@pytest.mark.parametrize("kind", ["empty-world", "behind", "offscreen", "outside-patch"])
def test_no_visible_support_never_fabricates_floor_points(kind):
    points = {
        "empty-world": np.empty((0, 3)),
        "behind": np.array([[0, -1, 4]]),
        "offscreen": np.array([point_at(150, 65, 4)]),
        "outside-patch": np.array([point_at(50, 90, 4)]),
    }[kind]
    result, receipt = source_floor_points(points, [camera()], 1, {10: person_mask()}, width=100)
    assert result.shape == (0, 3)
    assert receipt["uniquePoints"] == 0
    assert receipt["frames"][0]["visiblePoints"] == 0


@pytest.mark.parametrize(
    "points", [np.zeros(3), np.zeros((2, 2)), [[0, np.nan, -4]], [[0, 0, np.inf]]]
)
def test_invalid_world_points_are_refused(points):
    with pytest.raises(ValueError, match="finite.*world points"):
        source_floor_points(points, [camera()], 1, {10: person_mask()})


@pytest.mark.parametrize(
    "settings",
    [
        {"scale": 0},
        {"scale": -1},
        {"scale": np.nan},
        {"scale": np.inf},
        {"width": 0},
        {"max_frames": 0},
    ],
)
def test_invalid_scale_or_sampling_limits_are_refused(settings):
    kwargs = {"scale": 1, **settings}
    with pytest.raises(ValueError, match="positive scale and sampling limits"):
        source_floor_points(np.empty((0, 3)), [camera()], masks={10: person_mask()}, **kwargs)


def test_missing_cameras_are_refused():
    with pytest.raises(ValueError, match="requires cameras"):
        source_floor_points(np.empty((0, 3)), [], 1, {})


def tracked_floor_archive(tmp_path, archive_tracks=(7,), selected=7):
    root, world = tmp_path / "tracks", tmp_path / "world"
    root.mkdir()
    world.mkdir()
    payload = {"shape": [100, 100, 1]}
    masks = {90: person_mask(), 10: np.roll(person_mask(), 5, axis=1)}
    for track in archive_tracks:
        for sample, source in [(0, 90), (2, 10)]:
            payload[f"t{track}_s{sample:03d}"] = np.packbits(masks[source])
        directory = root / f"track_{track:02d}"
        directory.mkdir()
        (directory / "motion.json").write_text(
            json.dumps(
                {
                    "frames": [
                        {"sample": sample, "sourceIndex": source, "box": [45, 20, 60, 60]}
                        for sample, source in [(0, 90), (2, 10)]
                    ]
                }
            )
        )
    path = root / "masks.npz"
    np.savez(path, **payload)
    (world / "people.json").write_text(
        json.dumps({"people": [{"id": "person", "track": selected}], "sourceIndices": [10, 90]})
    )
    return path, world, masks


def files_snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_single_selected_track_preserves_source_mapping_pixels_and_original_files(tmp_path):
    path, world, expected = tracked_floor_archive(tmp_path)
    before = files_snapshot(tmp_path)
    result = single_person_floor_masks(path, "fixture", world)
    assert set(result) == {10, 90}, "sample ordinals must not replace source indices"
    for source, mask in expected.items():
        np.testing.assert_array_equal(result[source], mask)  # No dilation or identity union.
    assert files_snapshot(tmp_path) == before


@pytest.mark.parametrize("archive_tracks,selected", [((7, 9), 7), ((7,), 9)])
def test_mixed_or_wrong_tracks_are_refused_without_filtering_archive(
    tmp_path, monkeypatch, archive_tracks, selected
):
    import world_ruler

    path, world, _ = tracked_floor_archive(tmp_path, archive_tracks, selected)
    before = files_snapshot(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("ambiguous identities must be refused before loading or unioning masks")

    monkeypatch.setattr(world_ruler, "person_masks_for", unexpected)
    message = "exactly one mask track" if len(archive_tracks) > 1 else "does not match"
    with pytest.raises(ValueError, match=message):
        single_person_floor_masks(path, "fixture", world)
    assert files_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "people",
    [[], [{"id": "person"}], [{"track": True}], [{"track": "7"}], [{"track": 7}, {"track": 9}]],
)
def test_track_identity_requires_an_unambiguous_package_declaration(tmp_path, people):
    path, world, _ = tracked_floor_archive(tmp_path)
    (world / "people.json").write_text(json.dumps({"people": people, "sourceIndices": [10, 90]}))
    with pytest.raises(ValueError, match="one declared packaged person track"):
        single_person_floor_masks(path, "fixture", world)


def test_missing_archive_is_not_created_or_replaced_by_historical_masks(tmp_path):
    path = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        single_person_floor_masks(path, "gym", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_anonymous_merged_archive_cannot_establish_one_person(tmp_path):
    path = tmp_path / "merged.npz"
    np.savez(
        path, masks=np.packbits(person_mask()[None], axis=-1), shape=[1, 100, 100], indices=[10]
    )
    before = files_snapshot(tmp_path)
    with pytest.raises(ValueError, match="exact single-person or identified tracked masks"):
        single_person_floor_masks(path, "fixture", tmp_path)
    assert files_snapshot(tmp_path) == before


@pytest.mark.parametrize("wrong_source", [False, True])
def test_exact_single_person_archive_keeps_source_validation_and_unknown_status(
    tmp_path, wrong_source
):
    from test_single_person_masks import archive_fixture

    path, sequence, cameras, _, packed, _ = archive_fixture(
        tmp_path, indices=(90, 10), statuses=["known", "ambiguous"]
    )
    if wrong_source:
        sequence["sourceSha256"] = "f" * 64
    world = tmp_path / "exact-world"
    (world / "person").mkdir(parents=True)
    (world / "person/sequence.json").write_text(json.dumps(sequence))
    (world / "cameras.json").write_text(json.dumps({"cameras": cameras}))
    before = files_snapshot(tmp_path)
    if wrong_source:
        with pytest.raises(ValueError, match="source hash"):
            single_person_floor_masks(path, "fixture", world)
    else:
        result = single_person_floor_masks(path, "fixture", world)
        assert set(result) == {90}
        np.testing.assert_array_equal(result[90], np.unpackbits(packed[0], axis=-1)[:, :100])
    assert files_snapshot(tmp_path) == before
