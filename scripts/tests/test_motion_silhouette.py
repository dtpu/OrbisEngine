"""Silhouette alignment uses image translations, never periodic image boundaries."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_silhouette import best_shift_iou, iou


def test_edge_crossing_cannot_match_a_wrapped_person_as_one_silhouette():
    pred = np.zeros((8, 14), dtype=bool)
    pred[2:6, 10:14] = True
    gt = np.zeros_like(pred)
    gt[2:6, 12:14] = True
    gt[2:6, 0:2] = True
    # A periodic shift falsely makes this one person appear at both image edges.
    assert iou(np.roll(pred, 2, axis=1), gt) == 1.0
    score, _ = best_shift_iou(pred, gt, radius=20, step=1)
    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "dy, dx, expected",
    [
        (1, 2, [(1, 2), (2, 4)]),
        (-1, -2, [(0, 0), (1, 2)]),
        (1, -2, [(1, 2), (2, 0)]),
        (-1, 2, [(0, 4), (1, 2)]),
    ],
)
def test_zero_filled_translation_in_each_direction_on_a_non_square_mask(dy, dx, expected):
    from motion_silhouette import translate_mask

    mask = np.zeros((3, 5), dtype=bool)
    for y, x in [(0, 0), (0, 4), (1, 2), (2, 0), (2, 4)]:
        mask[y, x] = True
    shifted = translate_mask(mask, dy, dx)
    np.testing.assert_array_equal(np.argwhere(shifted), expected)
    assert shifted.dtype == mask.dtype
    assert mask.sum() == 5


@pytest.mark.parametrize("dy, dx", [(3, 0), (-3, 0), (0, 5), (0, -5), (20, -20)])
def test_translation_at_or_beyond_image_dimensions_leaves_no_wrapped_pixels(dy, dx):
    from motion_silhouette import translate_mask

    mask = np.ones((3, 5), dtype=bool)
    assert not translate_mask(mask, dy, dx).any()


@pytest.mark.parametrize("dy, dx", [(-6, -10), (-6, 10), (6, -10), (6, 10)])
def test_shift_search_recovers_a_real_translated_asymmetric_shape(dy, dx):
    pred = np.zeros((48, 68), dtype=bool)
    pred[12:16, 25:29] = True
    pred[16:27, 24:30] = True
    pred[18:21, 30:37] = True
    pred[27:32, 24:26] = True
    pred[27:34, 28:30] = True
    gt = np.zeros_like(pred)
    ys, xs = np.nonzero(pred)
    gt[ys + dy, xs + dx] = True
    score, shift = best_shift_iou(pred, gt)
    assert score == 1.0
    assert shift == (dy, dx)


@pytest.mark.parametrize("pred_empty, gt_empty", [(True, True), (True, False), (False, True)])
def test_empty_silhouettes_have_no_alignment_agreement(pred_empty, gt_empty):
    pred = np.full((4, 7), not pred_empty, dtype=bool)
    gt = np.full((4, 7), not gt_empty, dtype=bool)
    assert best_shift_iou(pred, gt) == (0.0, (0, 0))
    assert iou(pred, gt) == 0.0


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    import motion_silhouette

    world = tmp_path / "world"
    person = world / "person"
    person.mkdir(parents=True)
    mask_path = tmp_path / "masks.npz"
    masks = np.zeros((2, 5, 9), dtype=bool)
    masks[:, 1:4, 3:6] = True
    np.savez_compressed(
        mask_path,
        shape=np.array(masks.shape),
        masks=np.packbits(masks, axis=-1),
        indices=np.array([40, 42]),
    )
    vertices = np.zeros(
        1,
        dtype=[
            (name, np.float32)
            for name in ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"]
        ],
    )
    read = Mock(return_value={"vertex": SimpleNamespace(data=vertices)})
    raster = Mock(return_value=masks[0])
    monkeypatch.setattr(motion_silhouette.PlyData, "read", read)
    monkeypatch.setattr(motion_silhouette, "splat_silhouette", raster)

    def write(source_indices, frames, camera_indices=(40, 42)):
        (person / "sequence.json").write_text(
            json.dumps({"sourceIndices": source_indices, "frames": frames})
        )
        (world / "cameras.json").write_text(
            json.dumps({"cameras": [{"sourceIndex": index} for index in camera_indices]})
        )

    return SimpleNamespace(world=world, masks=mask_path, write=write, read=read, raster=raster)


def test_audit_uses_exact_source_indices_even_when_sequence_order_differs(audit_fixture):
    from motion_silhouette import audit

    fixture = audit_fixture
    fixture.write([42, 40], ["pose-42.ply", "pose-40.ply"])
    rows = audit(fixture.world, fixture.masks)
    assert [row["frame"] for row in rows] == [40, 42]
    assert [row["iou"] for row in rows] == [1.0, 1.0]
    assert [call.args[0].name for call in fixture.read.call_args_list] == [
        "pose-40.ply",
        "pose-42.ply",
    ]


def test_audit_rejects_missing_pose_instead_of_measuring_nearest_frame(audit_fixture):
    from motion_silhouette import audit

    fixture = audit_fixture
    fixture.write([40], ["pose-40.ply"])
    with pytest.raises(ValueError, match=r"no pose for selected sourceIndices: \[42\]"):
        audit(fixture.world, fixture.masks)
    fixture.read.assert_not_called()
    fixture.raster.assert_not_called()


@pytest.mark.parametrize(
    "indices, frames, error",
    [
        ([40, 42], ["pose.ply"], "2 sourceIndices but 1 frames"),
        ([40], ["pose-40.ply", "pose-42.ply"], "1 sourceIndices but 2 frames"),
        ([40, 40], ["pose-a.ply", "pose-b.ply"], "duplicate sourceIndices"),
    ],
)
def test_audit_rejects_ambiguous_or_misaligned_sequence_indices(
    audit_fixture, indices, frames, error
):
    from motion_silhouette import audit

    fixture = audit_fixture
    fixture.write(indices, frames)
    with pytest.raises(ValueError, match=error):
        audit(fixture.world, fixture.masks)
    fixture.read.assert_not_called()
    fixture.raster.assert_not_called()


@pytest.mark.parametrize(
    "indices, frames, error",
    [
        ([40], ["pose-40.ply"], r"no pose for selected sourceIndices: \[42\]"),
        ([40, 42], ["pose.ply"], "2 sourceIndices but 1 frames"),
        ([40, 40], ["pose-a.ply", "pose-b.ply"], "duplicate sourceIndices"),
    ],
)
def test_seed_validation_fails_before_any_ply_video_or_detection_work(
    audit_fixture, monkeypatch, indices, frames, error
):
    from unittest.mock import Mock

    import motion_audit

    fixture = audit_fixture
    fixture.write(indices, frames)
    decode = Mock(side_effect=AssertionError("must not decode video before validation"))
    detect = Mock(side_effect=AssertionError("must not run detection before validation"))
    monkeypatch.setattr(motion_audit.pmt, "read_frames", decode)
    monkeypatch.setattr(motion_audit.pmt, "masks_for", detect)
    with pytest.raises(ValueError, match=error):
        motion_audit.audit_world(fixture.world, "source.mp4")
    fixture.read.assert_not_called()
    fixture.raster.assert_not_called()
    decode.assert_not_called()
    detect.assert_not_called()


def test_seed_boxes_route_reordered_sequence_to_exact_source_pose(audit_fixture):
    from motion_audit import seed_boxes

    fixture = audit_fixture
    sequence = {"sourceIndices": [42, 40], "frames": ["pose-42.ply", "pose-40.ply"]}
    seeds = seed_boxes(
        fixture.world,
        [{"sourceIndex": 40}, {"sourceIndex": 42}],
        sequence,
        (5, 9),
    )
    assert list(seeds) == [40, 42]
    assert [call.args[0].name for call in fixture.read.call_args_list] == [
        "pose-40.ply",
        "pose-42.ply",
    ]
