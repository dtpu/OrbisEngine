"""Metric-depth cache provenance, with no model loading or actual inference."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world_ruler import predict_depths

MODEL = "local/DepthPro-stub"


@pytest.fixture
def inference(monkeypatch):
    calls = SimpleNamespace(loads=[], predictions=0, decoded=[])

    class VideoCapture:
        def __init__(self, path):
            self.path = Path(path)
            self.value = sum(self.path.read_bytes()) % 100
            self.frame = 0

        def set(self, property_id, frame):
            self.frame = frame

        def read(self):
            calls.decoded.append((self.path, self.frame))
            return True, np.full((2, 3, 3), self.value + self.frame, dtype=np.uint8)

        def release(self):
            pass

    class Depth:
        def __init__(self, values):
            self.values = values

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def __call__(self, images, return_tensors):
            self.rgb = images
            return {}

        def post_process_depth_estimation(self, output, target_sizes):
            return [
                {
                    "predicted_depth": Depth(self.rgb[:, :, 0].astype(np.float32)),
                    "field_of_view": 65.0,
                }
            ]

    class Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.loads.append(model_id)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            calls.predictions += 1
            return None

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=VideoCapture,
            CAP_PROP_POS_FRAMES=1,
            COLOR_BGR2RGB=2,
            cvtColor=lambda image, code: image,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(no_grad=nullcontext, float16="float16", float32="float32"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(DepthProImageProcessor=Processor, DepthProForDepthEstimation=Model),
    )
    return calls


def camera(frame=7, translation=0):
    pose = np.eye(4)
    pose[0, 3] = translation
    return {
        "sourceIndex": frame,
        "camera_to_world": pose.tolist(),
        "source_intrinsics": np.eye(3).tolist(),
        "source_image_size": [3, 2],
    }


def predict(source, cache, cams=None, model=MODEL):
    return predict_depths(cams or [camera()], str(source), model, "cpu", cache)


def test_replacing_same_named_source_recomputes_and_preserves_previous_cache(tmp_path, inference):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first source bytes")
    cache = tmp_path / "cache"
    before = predict(source, cache)[0][1]
    old_files = {path: path.read_bytes() for path in cache.rglob("*.npz")}
    source.write_bytes(b"other source bytes")
    after = predict(source, cache)[0][1]
    assert inference.predictions == 2
    assert not np.array_equal(before, after)
    assert len(list(cache.rglob("*.npz"))) == 2
    assert all(path.read_bytes() == contents for path, contents in old_files.items())


def test_same_upload_filename_in_different_runs_cannot_reuse_other_video(tmp_path, inference):
    cache = tmp_path / "cache"
    for directory, data in [("run-a", b"original"), ("run-b", b"unrelated")]:
        source = tmp_path / directory / "source.mp4"
        source.parent.mkdir()
        source.write_bytes(data)
        predict(source, cache)
    assert inference.predictions == 2
    assert len(list(cache.rglob("*.npz"))) == 2


def test_identical_bytes_reuse_prediction_with_new_camera_pose_and_source_path(tmp_path, inference):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"same pixels")
    cache = tmp_path / "cache"
    original = predict(source, cache)[0]
    renamed = tmp_path / "renamed.mp4"
    renamed.write_bytes(source.read_bytes())
    corrected_camera = camera(translation=42)
    corrected_camera["source_intrinsics"][0][0] = 20
    repeated = predict(renamed, cache, [corrected_camera])[0]
    assert inference.predictions == 1
    assert len(inference.decoded) == 1
    assert inference.loads == [MODEL]
    assert repeated[0] is corrected_camera
    np.testing.assert_array_equal(repeated[1], original[1])
    assert repeated[2] == original[2] == 65.0


def test_models_and_frame_indices_are_independent_even_when_legacy_tags_collide(
    tmp_path, inference
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"same source")
    cache = tmp_path / "cache"
    first_model = "local/DepthPro_stubs"
    second_model = "local_DepthPro/stubs"
    assert first_model.replace("/", "_") == second_model.replace("/", "_")
    initial = predict(source, cache, [camera(7), camera(2)], first_model)
    other_model = predict(source, cache, [camera(7)], second_model)
    assert inference.predictions == 3
    assert inference.loads == [first_model, second_model]
    assert len(list(cache.rglob("*.npz"))) == 3
    assert [row[0]["sourceIndex"] for row in initial] == [7, 2]
    assert not np.array_equal(initial[0][1], initial[1][1])
    repeated = predict(source, cache, [camera(2), camera(7)], first_model)
    predict(source, cache, [camera(7)], second_model)
    assert inference.predictions == 3
    np.testing.assert_array_equal(repeated[0][1], initial[1][1])
    np.testing.assert_array_equal(other_model[0][1], initial[0][1])


def test_legacy_unbound_cache_is_preserved_and_not_used_as_measurement(tmp_path, inference):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"new source")
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy = cache / f"source-{MODEL.replace('/', '_')}-7.npz"
    np.savez_compressed(legacy, depth=np.full((2, 3), 999), fov=np.array(10.0))
    retained = legacy.read_bytes()
    result = predict(source, cache)[0]
    assert inference.predictions == 1
    assert not np.any(result[1] == 999)
    assert result[2] == 65.0
    assert legacy.read_bytes() == retained
    predict(source, cache)
    assert inference.predictions == 1
