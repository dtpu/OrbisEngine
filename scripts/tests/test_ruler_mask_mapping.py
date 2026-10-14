"""Ruler and visibility masks stay attached to source frames after packaging subsets."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_observation_confidence as observation
import world_ruler


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pixel(row, col):
    mask = np.zeros((8, 8), bool)
    mask[row, col] = True
    return mask


def write_tracking(tmp_path, masks, records, selected):
    path = tmp_path / "tracks/masks.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        shape=[8, 8, 1],
        **{
            f"t{track:03d}_s{sample:03d}": np.packbits(mask)
            for (track, sample), mask in masks.items()
        },
    )
    for track, pairs in records.items():
        write_json(
            path.parent / f"track_{track:02d}/motion.json",
            {
                "frames": [
                    {"sourceIndex": source, "sample": sample, "box": [1, 1, 7, 7]}
                    for source, sample in pairs
                ]
            },
        )
    world = tmp_path / "packaged"
    write_json(world / "people.json", {"sourceIndices": selected})
    return path, world


@pytest.mark.parametrize("selected", [[90], [90, 10], [10, 40, 90]])
def test_packaged_subsets_and_reordering_keep_original_mask_samples(tmp_path, selected):
    inputs = {(0, 0): pixel(0, 0), (0, 2): pixel(2, 2), (0, 8): pixel(7, 7)}
    path, world = write_tracking(tmp_path, inputs, {0: [(10, 0), (40, 2), (90, 8)]}, selected)
    result = world_ruler.person_masks_for("", masks=path, world_dir=world, dilate=False)
    assert set(result) == {10, 40, 90}
    for source, sample in [(10, 0), (40, 2), (90, 8)]:
        np.testing.assert_array_equal(result[source], inputs[0, sample])
    # Exercise the actual visibility loader too: every selected camera receives its source mask.
    visible = observation.load_source_masks(path, world, [{"sourceIndex": s} for s in selected])
    for source in selected:
        np.testing.assert_array_equal(visible[source], result[source])


def test_tracks_map_independently_and_union_only_at_the_same_source_frame(tmp_path):
    inputs = {(0, 0): pixel(0, 0), (1, 0): pixel(3, 3), (1, 3): pixel(5, 5)}
    path, world = write_tracking(tmp_path, inputs, {0: [(10, 0)], 1: [(90, 0), (10, 3)]}, [90, 10])
    result = world_ruler.person_masks_for("", masks=path, world_dir=world, dilate=False)
    np.testing.assert_array_equal(result[10], inputs[0, 0] | inputs[1, 3])
    np.testing.assert_array_equal(result[90], inputs[1, 0])


def test_missing_motion_sample_is_not_reassigned_using_manifest_position(tmp_path):
    path, world = write_tracking(
        tmp_path,
        {(0, 0): pixel(0, 0), (0, 8): pixel(7, 7)},
        {0: [(90, 8)]},
        [10, 90],
    )
    with pytest.raises(ValueError, match="no masks for packaged source frames \\[10\\]"):
        world_ruler.person_masks_for("", masks=path, world_dir=world, dilate=False)
    with pytest.raises(ValueError, match="invalid --masks"):
        observation.load_source_masks(path, world, [{"sourceIndex": 10}])


def test_legacy_track_without_motion_uses_documented_manifest_order(tmp_path):
    inputs = {(0, 0): pixel(0, 0), (0, 1): pixel(7, 7)}
    path, world = write_tracking(tmp_path, inputs, {}, [90, 10])
    result = world_ruler.person_masks_for("", masks=path, world_dir=world, dilate=False)
    np.testing.assert_array_equal(result[90], inputs[0, 0])
    np.testing.assert_array_equal(result[10], inputs[0, 1])


def test_conflicting_motion_records_fail_instead_of_reusing_first_mapping(tmp_path):
    path, world = write_tracking(tmp_path, {(0, 0): pixel(0, 0)}, {0: [(10, 0), (90, 0)]}, [10])
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        world_ruler.person_masks_for("", masks=path, world_dir=world, dilate=False)


def test_dilation_is_applied_after_source_mapping(tmp_path):
    path, world = write_tracking(tmp_path, {(0, 8): pixel(4, 4)}, {0: [(90, 8)]}, [90])
    result = world_ruler.person_masks_for("", masks=path, world_dir=world)
    assert set(result) == {90}
    assert result[90].all()
