"""Anchor-frame exclusions must survive projection into other camera views."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bake_video_colours import static_cloud_from_anchors, zbuffer_depth


@pytest.fixture
def anchor_scene(tmp_path):
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1, 0, 3] = 1
    shape = (2, 2, 3)
    points = np.zeros((*shape, 3), dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    people = np.zeros(shape, dtype=bool)
    # A moving foreground point missed by the original people mask.
    points[0, 0, 1], valid[0, 0, 1] = (0, 0, 1), True
    # Static background seen from a different anchor.
    points[1, 0, 1], valid[1, 0, 1] = (-1, 0, 2), True
    # This point must still be excluded by the original people mask.
    points[0, 1, 1], valid[0, 1, 1], people[0, 1, 1] = (0, 0, 0.5), True, True
    archive = tmp_path / "anchors.npz"
    np.savez(
        archive,
        poses=poses,
        points=points,
        valid=valid,
        people=people,
        conf=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
    )
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(
            {
                "cameras": [
                    {"camera_to_world": pose.tolist()} for pose in [poses[0], poses[1], poses[1]]
                ],
            }
        )
    )
    return archive, cameras, poses[1]


def test_anchor_mask_removes_foreground_depth_from_another_unmasked_view(anchor_scene):
    archive, cameras, target_pose = anchor_scene
    samples = [0, 2]
    legacy = static_cloud_from_anchors(archive, cameras, samples, conf_pct=0)
    np.testing.assert_array_equal(legacy, [[0, 0, -1], [-1, 0, -2]])
    exclusions = {sample: np.zeros((2, 3), dtype=bool) for sample in samples}
    exclusions[0][0, 1] = True
    counts = []
    filtered = static_cloud_from_anchors(
        archive, cameras, samples, conf_pct=0, exclusion_masks=exclusions, exclusion_report=counts
    )
    assert not exclusions[2].any()
    np.testing.assert_array_equal(filtered, [[-1, 0, -2]])
    assert counts == [
        {"sample": 0, "candidatePoints": 1, "excludedPoints": 1, "retainedPoints": 0},
        {"sample": 2, "candidatePoints": 1, "excludedPoints": 0, "retainedPoints": 1},
    ]
    K = np.array([[4, 0, 4], [0, 4, 4], [0, 0, 1]])
    old, old_support = zbuffer_depth(legacy, target_pose, K, (8, 8), (8, 8))
    new, new_support = zbuffer_depth(filtered, target_pose, K, (8, 8), (8, 8))
    assert old_support[4, 0] and new_support[4, 0]
    assert old[4, 0] == 1 and new[4, 0] == 2


@pytest.mark.parametrize("case", ["wrong-sample", "wrong-shape", "wrong-type", "empty-mask"])
def test_anchor_exclusion_validation(anchor_scene, case):
    archive, cameras, _ = anchor_scene
    masks = {0: np.zeros((2, 3), dtype=bool), 2: np.zeros((2, 3), dtype=bool)}
    if case == "wrong-sample":
        masks[1] = masks.pop(2)
    elif case == "wrong-shape":
        masks[2] = np.zeros((3, 2), dtype=bool)
    elif case == "wrong-type":
        masks[2] = masks[2].astype(np.uint8)
    else:
        masks[2] = np.zeros((0, 3), dtype=bool)
    with pytest.raises(ValueError, match="Anchor exclusion"):
        static_cloud_from_anchors(archive, cameras, [0, 2], exclusion_masks=masks)


def test_exclusions_cannot_leave_an_empty_cloud(anchor_scene):
    archive, cameras, _ = anchor_scene
    with pytest.raises(ValueError, match="no usable finite points"):
        static_cloud_from_anchors(
            archive,
            cameras,
            [0, 2],
            exclusion_masks={sample: np.ones((2, 3), dtype=bool) for sample in (0, 2)},
        )


def test_degenerate_anchor_baseline_rejected(anchor_scene):
    archive, cameras, _ = anchor_scene
    cameras.write_text(json.dumps({"cameras": [{"camera_to_world": np.eye(4).tolist()}] * 3}))
    with pytest.raises(ValueError, match="positive pose baselines"):
        static_cloud_from_anchors(archive, cameras, [0, 2])
