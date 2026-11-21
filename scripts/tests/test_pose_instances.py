"""Independent mask support must not create duplicate or unobserved people."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.stages.pose_instances import match_pose_instances, match_crop_instance


def test_lamp_pose_is_not_a_person_instance():
    assert match_pose_instances([[0, 0, 10, 50], [60, 0, 90, 80]], [[60, 0, 90, 80]]) == {
        1: (0, 1.0)
    }


def test_duplicate_poses_cannot_claim_same_person():
    pairs = match_pose_instances([[0, 0, 20, 80], [0, 0, 10, 80]], [[0, 0, 20, 80]])
    assert pairs == {0: (0, 1.0)}


def test_global_matching_preserves_second_person_over_greedy_choice():
    # First pose supports both masks; second supports only the first mask.
    poses = [[0, 0, 18, 10], [0, 0, 6, 10]]
    masks = [[0, 0, 10, 10], [10, 0, 20, 10]]
    result = match_pose_instances(poses, masks)
    assert {i: pair[0] for i, pair in result.items()} == {0: 1, 1: 0}


def test_separated_people_remain_separate_and_order_independent():
    boxes = [[0, 0, 10, 30], [40, 0, 50, 30], [80, 0, 90, 30]]
    assert match_pose_instances(boxes, boxes) == {i: (i, 1.0) for i in range(3)}
    assert match_pose_instances(boxes[::-1], boxes) == {i: (2 - i, 1.0) for i in range(3)}


@pytest.mark.parametrize("poses,masks", [([], []), ([], [[0, 0, 1, 1]]), ([[0, 0, 1, 1]], [])])
def test_absent_support_is_not_fabricated(poses, masks):
    assert match_pose_instances(poses, masks) == {}


def test_existing_strict_overlap_threshold_is_preserved():
    assert match_pose_instances([[0, 0, 2, 10]], [[0, 0, 10, 10]]) == {}
    assert match_pose_instances([[0, 0, 2.01, 10]], [[0, 0, 10, 10]])


def test_degenerate_boxes_cannot_supply_support():
    assert match_pose_instances([[0, 0, 0, 20], [10, 10, 0, 0]], [[0, 0, 10, 20]]) == {}


def test_nonfinite_boxes_fail_instead_of_becoming_a_match():
    with pytest.raises(ValueError, match="finite"):
        match_pose_instances([[0, 0, np.nan, 20]], [[0, 0, 10, 20]])


def test_crop_cannot_relabel_neighbor_as_missing_person():
    instances = [[0, 0, 20, 80], [10, 0, 30, 80]]
    assert match_pose_instances([instances[0]], [instances[1]])  # old target-only false positive
    assert match_crop_instance([instances[0]], instances, 1) == {}
    assert match_crop_instance([instances[0], instances[1]], instances, 1) == {1: (1, 1.0)}


def test_crop_refuses_ambiguous_support_and_duplicate_recovery():
    assert match_crop_instance([[0, 0, 20, 80]], [[0, 0, 20, 80]] * 2, 1) == {}
    assert len(match_crop_instance([[0, 0, 20, 80]] * 2, [[0, 0, 20, 80]], 0)) == 1
    assert match_crop_instance([], [[0, 0, 20, 80]], 0) == {}
