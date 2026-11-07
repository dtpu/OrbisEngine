"""Locked anchor clouds retain their shared world frame rather than moving per view."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
from bake_video_colours import static_cloud_from_anchors
from wander_worker.camera_stitch import _rigid_poses


@pytest.fixture
def locked_scene(tmp_path):
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec(
        [[0.1, 0.2, 0], [-0.1, 0.15, 0.2], [0.05, -0.2, 0.1]]
    ).as_matrix()
    poses[:, :3, 0] *= np.array([1.0004, 0.9996, 1.0002])[:, None]
    poses[:, :3, 3] = [[0, 0, 0], [0.4, 0.1, -0.1], [-0.2, 0.2, 0.1]]
    points = np.tile(
        np.array([[[0, 0, 4], [0.2, -0.3, 3]], [[0.4, 0.1, 3.5], [0.6, -0.2, 4.5]]], np.float32),
        (3, 1, 1, 1),
    )
    data = dict(
        poses=poses,
        points=points,
        valid=np.ones((3, 2, 2), bool),
        people=np.zeros((3, 2, 2), bool),
        conf=np.tile(np.arange(4, dtype=np.float32).reshape(1, 2, 2), (3, 1, 1)),
    )
    rotation = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    rotation[:, 1] *= 1.0001
    translation, scale, flip = np.array([1.1, -2.3, 0.6]), 1.7, np.array([1, -1, -1])
    expected = np.repeat(np.eye(4)[None], 3, axis=0)
    expected[:, :3, :3] = np.diag(flip) @ rotation @ poses[:, :3, :3] @ np.diag(flip)
    expected[:, :3, 3] = (poses[:, :3, 3] @ rotation.T + translation) * scale * flip
    expected = _rigid_poses(expected, 3, "fixture")
    indices, times = [0, 2, 5], [0.0, 0.2, 0.5]
    doc = dict(
        cameraStitch="anchor-locked",
        sourceSha256="a" * 64,
        reference=dict(
            rotation=rotation.tolist(),
            translation=translation.tolist(),
            scale=scale,
            world_axis_flip=flip.tolist(),
        ),
        cameras=[
            dict(camera_to_world=pose.tolist(), sourceIndex=index, time=time)
            for pose, index, time in zip(expected, indices, times)
        ],
    )
    sequence = dict(
        cameraStitch="anchor-locked",
        sourceSha256="a" * 64,
        anchors=[0, 1, 2],
        count=3,
        sourceIndices=indices,
        timestamps=times,
    )
    archive, cameras = tmp_path / "anchors.npz", tmp_path / "cameras.json"
    np.savez(archive, **data)
    cameras.write_text(json.dumps(doc))
    cameras.with_name("sequence.json").write_text(json.dumps(sequence))
    return archive, cameras, data, doc, sequence


def test_same_world_points_remain_identical_across_changing_near_rigid_poses(locked_scene):
    archive, cameras, data, doc, _ = locked_scene
    cloud = static_cloud_from_anchors(archive, cameras, [0, 1, 2], conf_pct=0)
    ref = doc["reference"]
    expected = (
        (
            data["points"][0].reshape(-1, 3)[1:].astype(float) @ np.array(ref["rotation"]).T
            + ref["translation"]
        )
        * ref["scale"]
        * ref["world_axis_flip"]
    ).astype(np.float32)
    np.testing.assert_array_equal(cloud, np.tile(expected, (3, 1)))


@pytest.mark.parametrize("mode", ["anchor-lock", "future", None, True, []])
def test_unknown_stitch_mode_cannot_silently_use_legacy_geometry(locked_scene, mode):
    archive, cameras, _, doc, _ = locked_scene
    doc["cameraStitch"] = mode
    cameras.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="cameraStitch"):
        static_cloud_from_anchors(archive, cameras, [0, 1, 2])


def test_locked_eligibility_and_exclusion_counts_remain_exact(locked_scene):
    archive, cameras, data, _, _ = locked_scene
    data["people"][0, 1, 0] = True
    data["valid"][1, 1, 1] = False
    # Behind the source camera: eligibility is the existing local-z test.
    data["points"][2, 1, 1] = [0, 0, -4]
    np.savez(archive, **data)
    masks = {sample: np.zeros((2, 2), bool) for sample in range(3)}
    masks[0][0, 1] = True
    masks[2][1, 1] = True  # Already ineligible, so never counted twice.
    counts = []
    cloud = static_cloud_from_anchors(
        archive, cameras, [0, 1, 2], conf_pct=0, exclusion_masks=masks, exclusion_report=counts
    )
    assert len(cloud) == 5
    assert counts == [
        dict(sample=0, candidatePoints=2, excludedPoints=1, retainedPoints=1),
        dict(sample=1, candidatePoints=2, excludedPoints=0, retainedPoints=2),
        dict(sample=2, candidatePoints=2, excludedPoints=0, retainedPoints=2),
    ]


def test_recorded_scale_does_not_require_a_camera_translation_baseline(locked_scene):
    archive, cameras, data, doc, _ = locked_scene
    data["poses"][:, :3, 3] = 0
    np.savez(archive, **data)
    ref = doc["reference"]
    center = np.array(ref["translation"]) * ref["scale"] * ref["world_axis_flip"]
    for camera in doc["cameras"]:
        matrix = np.array(camera["camera_to_world"])
        matrix[:3, 3] = center
        camera["camera_to_world"] = matrix.tolist()
    cameras.write_text(json.dumps(doc))
    cloud = static_cloud_from_anchors(archive, cameras, [0, 1, 2], conf_pct=0)
    np.testing.assert_array_equal(cloud[:3], cloud[3:6])
    np.testing.assert_array_equal(cloud[:3], cloud[6:])


@pytest.mark.parametrize(
    "field,value",
    [
        ("rotation", [[1, 0], [0, 1]]),
        ("rotation", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("rotation", (np.eye(3) * 1.01).tolist()),
        ("rotation", [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("translation", [1, 2]),
        ("translation", [1, float("inf"), 3]),
        ("world_axis_flip", [1, 1, -1]),
        ("world_axis_flip", [1, -1]),
        ("scale", 0),
        ("scale", -1),
        ("scale", float("nan")),
        ("scale", True),
        ("scale", "1"),
    ],
)
def test_rejects_invalid_reference(locked_scene, field, value):
    archive, cameras, _, doc, _ = locked_scene
    doc["reference"][field] = value
    cameras.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        static_cloud_from_anchors(archive, cameras, [0, 1, 2])


@pytest.mark.parametrize(
    "defect",
    [
        "reference",
        "reference-field",
        "sequence",
        "source-sha",
        "source-index",
        "source-time",
        "anchor-order",
        "sequence-marker",
        "sequence-count",
        "sequence-index",
        "sequence-time",
        "public-center",
        "public-rotation",
        "raw-reflection",
        "sequence-shape",
    ],
)
def test_rejects_missing_or_contradictory_locked_metadata(locked_scene, defect):
    archive, cameras, data, doc, sequence = locked_scene
    if defect == "reference":
        del doc["reference"]
    elif defect == "reference-field":
        del doc["reference"]["world_axis_flip"]
    elif defect == "source-sha":
        sequence["sourceSha256"] = "b" * 64
    elif defect == "source-index":
        doc["cameras"][1]["sourceIndex"] = 4
    elif defect == "source-time":
        doc["cameras"][1]["time"] = 0.25
    elif defect == "anchor-order":
        sequence["anchors"] = [0, 2, 1]
    elif defect == "sequence-marker":
        del sequence["cameraStitch"]
    elif defect == "sequence-count":
        sequence["count"] = 4
    elif defect == "sequence-index":
        sequence["sourceIndices"] = [0, 2, 4]
    elif defect == "sequence-time":
        sequence["timestamps"] = [0, 0.3, 0.5]
    elif defect == "public-center":
        doc["cameras"][1]["camera_to_world"][0][3] += 0.01
    elif defect == "public-rotation":
        doc["cameras"][1]["camera_to_world"][0][0] += 0.0001
    elif defect == "raw-reflection":
        data["poses"][1, :3, 0] *= -1
        np.savez(archive, **data)
    elif defect == "sequence-shape":
        sequence = []
    cameras.write_text(json.dumps(doc))
    cameras.with_name("sequence.json").write_text(json.dumps(sequence))
    if defect == "sequence":
        cameras.with_name("sequence.json").unlink()
    with pytest.raises(ValueError):
        static_cloud_from_anchors(archive, cameras, [0, 1, 2])
