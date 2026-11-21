"""Reviewed exclusions are exact-source instructions, never automatic semantic labels."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.stages.person_exclusions import load_exclusions, load_video_exclusions, match_exclusions


@pytest.fixture
def doc():
    return {
        "schema": "wander.person-exclusions/1",
        "sourceSha256": "ab" * 32,
        "width": 640,
        "height": 480,
        "reviewer": {"kind": "agent", "name": "frame reviewer"},
        "entries": [
            {
                "sourceIndex": 3,
                "box": [100, 50, 200, 250],
                "label": "depiction",
                "reason": "Reviewed figure on poster",
                "evidence": "frame-003.png",
            }
        ],
    }


def save(tmp_path, doc):
    path = tmp_path / "exclusions.json"
    path.write_text(json.dumps(doc))
    return path


@pytest.mark.parametrize("kind,label", [("agent", "depiction"), ("human", "statue")])
def test_valid_load_preserves_provenance_and_document(tmp_path, doc, kind, label):
    doc["reviewer"]["kind"] = kind
    doc["entries"][0]["label"] = label
    before = copy.deepcopy(doc)
    path = save(tmp_path, doc)
    original_bytes = path.read_bytes()
    loaded = load_exclusions(path, doc["sourceSha256"], 640, 480)
    assert loaded == before == doc
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("box", [[100, 50, 200, 250], [101, 51, 201, 251]])
def test_exact_and_near_match_preserve_unreviewed_and_inputs(doc, box):
    boxes = np.array([[300, 0, 400, 400], box], dtype=float)
    before_boxes, before_doc = boxes.copy(), copy.deepcopy(doc)
    result = match_exclusions(doc, 3, boxes)
    assert result == {1: doc["entries"][0]}
    assert result[1] is doc["entries"][0]
    assert 0 not in result
    np.testing.assert_array_equal(boxes, before_boxes)
    assert doc == before_doc
    assert match_exclusions(doc, 4, boxes) == {}
    assert match_exclusions(doc, 4, []) == {}


def test_empty_review_preserves_every_instance(tmp_path, doc):
    doc["entries"] = []
    loaded = load_exclusions(save(tmp_path, doc), doc["sourceSha256"], 640, 480)
    assert match_exclusions(loaded, 3, [[100, 50, 200, 250]]) == {}


@pytest.mark.parametrize("boxes", [[], [[300, 20, 400, 220]], [[100, 50, 190, 250]]])
def test_stale_or_missing_current_box_fails(doc, boxes):
    with pytest.raises(ValueError, match="stale review"):
        match_exclusions(doc, 3, boxes)


def test_strict_iou_threshold(doc):
    doc["width"] = doc["height"] = 100
    doc["entries"][0]["box"] = [0, 0, 100, 100]
    with pytest.raises(ValueError, match="stale review"):
        match_exclusions(doc, 3, [[0, 0, 95, 100]])
    assert match_exclusions(doc, 3, [[0, 0, 95.01, 100]])


def test_multiple_candidates_and_shared_instance_are_ambiguous(doc):
    box = doc["entries"][0]["box"]
    with pytest.raises(ValueError, match="ambiguous review"):
        match_exclusions(doc, 3, [box, box])
    second = copy.deepcopy(doc["entries"][0])
    second["box"] = [101, 50, 201, 250]
    doc["entries"].append(second)
    with pytest.raises(ValueError, match="ambiguous review"):
        match_exclusions(doc, 3, [box])


@pytest.mark.parametrize(
    "sha,width,height", [("cd" * 32, 640, 480), ("ab" * 32, 641, 480), ("ab" * 32, 640, 481)]
)
def test_stale_binding(tmp_path, doc, sha, width, height):
    with pytest.raises(ValueError, match="stale review"):
        load_exclusions(save(tmp_path, doc), sha, width, height)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "unknown"),
        ("sourceSha256", "x" * 64),
        ("sourceSha256", "ab" * 31),
        ("width", True),
        ("height", 0),
        ("width", 640.0),
        ("reviewer", None),
        ("reviewer", {"kind": "bot", "name": "x"}),
        ("reviewer", {"kind": "agent", "name": "  "}),
        ("entries", {}),
        ("entries", [None]),
    ],
)
def test_malformed_document(tmp_path, doc, key, value):
    doc[key] = value
    with pytest.raises(ValueError):
        load_exclusions(save(tmp_path, doc), "ab" * 32, 640, 480)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sourceIndex", True),
        ("sourceIndex", -1),
        ("sourceIndex", 3.0),
        ("sourceIndex", "3"),
        ("box", [0, 0, 0, 10]),
        ("box", [0, 0, 641, 10]),
        ("box", [-1, 0, 10, 10]),
        ("box", [0, 0, 10, 481]),
        ("box", [0, 0, float("nan"), 10]),
        ("box", [0, 0, float("inf"), 10]),
        ("box", [0, 0, True, 10]),
        ("box", [0, 0, "10", 10]),
        ("box", [0, 0, 10]),
        ("label", "person"),
        ("label", []),
        ("reason", " "),
        ("evidence", ""),
        ("evidence", ["frame.png"]),
        ("instance", 0),
    ],
)
def test_malformed_entry_and_unknown_keys(tmp_path, doc, key, value):
    doc["entries"][0][key] = value
    with pytest.raises(ValueError):
        load_exclusions(save(tmp_path, doc), "ab" * 32, 640, 480)


@pytest.mark.parametrize("key", ["sourceIndex", "box", "label", "reason", "evidence"])
def test_missing_entry_fields(tmp_path, doc, key):
    del doc["entries"][0][key]
    with pytest.raises(ValueError):
        load_exclusions(save(tmp_path, doc), "ab" * 32, 640, 480)


def test_duplicate_frame_box_rejected_but_other_frames_allowed(tmp_path, doc):
    doc["entries"].append(copy.deepcopy(doc["entries"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        load_exclusions(save(tmp_path, doc), "ab" * 32, 640, 480)
    doc["entries"][1]["sourceIndex"] = 4
    assert load_exclusions(save(tmp_path, doc), "ab" * 32, 640, 480) == doc


@pytest.mark.parametrize("raw", ['{"schema":', "[]", '{"schema":1,"schema":2}'])
def test_invalid_json(tmp_path, raw):
    path = tmp_path / "review.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_exclusions(path, "ab" * 32, 640, 480)


@pytest.fixture
def video_review(tmp_path, doc, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video fixture bytes; capture mocked")
    doc["sourceSha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    path = save(tmp_path, doc)
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = [640.0, 480.0, 10.0]
    monkeypatch.setattr(cv2, "VideoCapture", lambda filename: capture)
    return path, video, capture


def test_video_wrapper_binds_bytes_and_metadata(video_review, doc):
    path, video, capture = video_review
    assert load_video_exclusions(path, video) == doc
    capture.release.assert_called_once()
    video.write_bytes(b"changed source")
    capture.get.side_effect = [640.0, 480.0, 10.0]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_video_exclusions(path, video)
    assert capture.release.call_count == 2


@pytest.mark.parametrize(
    "metadata",
    [
        [0, 480, 10],
        [640, 0, 10],
        [640, 480, 0],
        [640.5, 480, 10],
        [640, 480, 10.5],
        [float("nan"), 480, 10],
        [640, 480, float("inf")],
        [640, 480, -1],
    ],
)
def test_video_invalid_metadata_releases_capture(video_review, metadata):
    path, video, capture = video_review
    capture.get.side_effect = metadata
    with pytest.raises(ValueError, match="positive integer"):
        load_video_exclusions(path, video)
    capture.release.assert_called_once()


def test_video_unreadable_and_out_of_range_review(video_review):
    path, video, capture = video_review
    capture.isOpened.return_value = False
    with pytest.raises(ValueError, match="unreadable"):
        load_video_exclusions(path, video)
    capture.release.assert_called_once()
    capture.isOpened.return_value = True
    capture.get.side_effect = [640, 480, 3]
    with pytest.raises(ValueError, match="frame count"):
        load_video_exclusions(path, video)
    assert capture.release.call_count == 2


def test_video_metadata_exception_still_releases(video_review):
    path, video, capture = video_review
    capture.get.side_effect = RuntimeError("decoder failed")
    with pytest.raises(RuntimeError, match="decoder failed"):
        load_video_exclusions(path, video)
    capture.release.assert_called_once()


def test_missing_files_rejected(tmp_path):
    with pytest.raises(ValueError, match="unreadable"):
        load_video_exclusions(tmp_path / "review.json", tmp_path / "absent.mp4")
    with pytest.raises(ValueError, match="unreadable"):
        load_exclusions(tmp_path / "review.json", "ab" * 32, 640, 480)


@pytest.mark.parametrize(
    "frame,accepted",
    [(0, True), (1, False), (2, True), (3, False), (5, True), (8, True), (9, False)],
)
def test_review_checked_against_exact_30_to_12_schedule(video_review, doc, frame, accepted):
    path, video, capture = video_review
    doc["entries"][0]["sourceIndex"] = frame
    path.write_text(json.dumps(doc))
    original = path.read_bytes()
    capture.get.side_effect = [640, 480, 10, 30.0]
    if accepted:
        assert load_video_exclusions(path, video, fps=12) == doc
    else:
        with pytest.raises(ValueError, match="tracking schedule"):
            load_video_exclusions(path, video, fps=12)
    capture.release.assert_called_once()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "source_fps,target_fps",
    [
        (0, 12),
        (-30, 12),
        (float("nan"), 12),
        (float("inf"), 12),
        (True, 12),
        ("30", 12),
        (None, 12),
        (30, 0),
        (30, -12),
        (30, float("nan")),
        (30, float("inf")),
        (30, True),
        (30, "12"),
        (30, [12]),
    ],
)
def test_malformed_fps_refused_and_capture_released(video_review, source_fps, target_fps):
    path, video, capture = video_review
    capture.get.side_effect = [640, 480, 10, source_fps]
    with pytest.raises(ValueError, match="FPS must be finite and positive"):
        load_video_exclusions(path, video, fps=target_fps)
    capture.release.assert_called_once()


def test_sampling_matches_worker_function_without_importing_models(video_review, doc):
    import ast

    path, video, capture = video_review
    source = (Path(__file__).resolve().parents[2] / "worker/stages/track_people.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_indices"
    )
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "sample_indices", "exec"), namespace)
    for source_fps, count, target_fps in [(30, 10, 12), (24000 / 1001, 186, 12), (12, 5, 30)]:
        capture.get.side_effect = [source_fps, count]
        indices, *_ = namespace["sample_indices"](video, target_fps)
        doc["entries"] = [dict(doc["entries"][0], sourceIndex=int(i)) for i in sorted(set(indices))]
        path.write_text(json.dumps(doc))
        capture.get.side_effect = [640, 480, count, source_fps]
        assert load_video_exclusions(path, video, fps=target_fps) == doc
    assert capture.release.call_count == 6
