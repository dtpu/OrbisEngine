"""Prepared masks retain source segmentation evidence separately from padded model inputs."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_lhm_person as prepare


@pytest.fixture
def segmentation(monkeypatch):
    rgb = np.arange(14 * 12 * 3, dtype=np.uint8).reshape(14, 12, 3)
    selected = np.zeros(rgb.shape[:2], np.uint8)
    selected[2:12, 4:7] = 1
    selected[6, 5] = 0  # Keep the component's nonrectangular detail, too.
    raw = selected.copy()
    raw[0, 11] = raw[13, 0] = 1  # Disconnected blobs are not observation evidence.
    padded = np.zeros_like(raw)
    padded[:, 2:9] = 255
    segformer = Mock(side_effect=lambda *a, **k: raw[None].copy())
    maskrcnn = Mock(side_effect=lambda *a, **k: raw.copy())
    masks_module = ModuleType("wander_worker.masks")
    masks_module.people_masks = segformer
    monkeypatch.setitem(sys.modules, "wander_worker.masks", masks_module)
    monkeypatch.setattr(prepare, "maskrcnn_person", maskrcnn)
    return SimpleNamespace(
        rgb=rgb, selected=selected * 255, padded=padded, segformer=segformer, maskrcnn=maskrcnn
    )


def assert_segmented_once(segmentation, method, bbox=None):
    if method == "segformer":
        segmentation.segformer.assert_called_once()
        args, kwargs = segmentation.segformer.call_args
        np.testing.assert_array_equal(args[0], segmentation.rgb[None])
        assert kwargs == {"dilate_px": 0}
        segmentation.maskrcnn.assert_not_called()
    else:
        segmentation.maskrcnn.assert_called_once()
        args, kwargs = segmentation.maskrcnn.call_args
        np.testing.assert_array_equal(args[0], segmentation.rgb)
        assert args[1] == bbox
        assert kwargs == {}
        segmentation.segformer.assert_not_called()


@pytest.mark.parametrize("method", ["segformer", "maskrcnn"])
@pytest.mark.parametrize("dilate", [0, 2])
def test_masks_preserve_component_and_existing_padding(segmentation, method, dilate):
    observed, model = prepare.person_masks(segmentation.rgb, dilate, method)

    np.testing.assert_array_equal(observed, segmentation.selected)
    np.testing.assert_array_equal(model, segmentation.padded if dilate else segmentation.selected)
    assert observed.dtype == model.dtype == np.uint8
    assert_segmented_once(segmentation, method)


@pytest.mark.parametrize("method", ["segformer", "maskrcnn"])
def test_person_mask_api_still_returns_padded_model_input(segmentation, method):
    model = prepare.person_mask(segmentation.rgb, 2, method)

    np.testing.assert_array_equal(model, segmentation.padded)
    assert_segmented_once(segmentation, method)


def test_segformer_bbox_selects_component_in_full_source_coordinates(segmentation):
    bbox = [4, 4, 6, 8]
    observed, model = prepare.person_masks(segmentation.rgb, 0, "segformer", bbox)
    expected = np.zeros_like(segmentation.selected)
    expected[4:9, 4:7] = segmentation.selected[4:9, 4:7]

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(model, expected)
    assert prepare.box_report(observed, 12, 14)["box"] == bbox
    assert_segmented_once(segmentation, "segformer", bbox)


def test_maskrcnn_track_box_is_forwarded_to_instance_selection(segmentation):
    bbox = [4.0, 2.0, 6.0, 11.0]
    observed, _ = prepare.person_masks(segmentation.rgb, 2, "maskrcnn", bbox)

    np.testing.assert_array_equal(observed, segmentation.selected)
    assert_segmented_once(segmentation, "maskrcnn", bbox)


@pytest.fixture
def prepare_cli(tmp_path, monkeypatch, segmentation):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source video identity fixture; decoder is stubbed")
    output = tmp_path / "prepared"
    decoder = Mock(return_value=(segmentation.rgb, 25.0, 100))
    monkeypatch.setattr(prepare, "decode", decoder)

    def invoke(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prepare_lhm_person.py",
                str(video),
                "--frame",
                "37",
                "--out",
                str(output),
                *extra,
            ],
        )
        prepare.main()

    return SimpleNamespace(video=video, output=output, decoder=decoder, invoke=invoke)


@pytest.mark.parametrize("method", ["segformer", "maskrcnn"])
@pytest.mark.parametrize("dilate", [0, 2])
def test_prepared_artifacts_bind_source_and_distinguish_observation(
    prepare_cli, segmentation, method, dilate
):
    prepare_cli.invoke("--method", method, "--dilate", str(dilate), "--bbox", "0,0,11,13")
    output = prepare_cli.output
    metadata = json.loads((output / "prepared.json").read_text())
    expected_model = segmentation.padded if dilate else segmentation.selected
    for name, expected in (
        ("source.png", segmentation.rgb),
        ("observed-mask.png", segmentation.selected),
        ("mask.png", expected_model),
    ):
        with Image.open(output / name) as image:
            np.testing.assert_array_equal(np.asarray(image), expected)
            assert image.size == (12, 14)

    assert metadata["video"] == str(prepare_cli.video.resolve())
    assert metadata["sourceSha256"] == hashlib.sha256(prepare_cli.video.read_bytes()).hexdigest()
    assert metadata["frameIndex"] == 37
    assert metadata["time"] == 37 / 25.0
    assert (metadata["sourceWidth"], metadata["sourceHeight"]) == (12, 14)
    assert metadata["fps"] == 25.0
    assert metadata["duration"] == 4.0
    assert metadata["trackBox"] == [0.0, 0.0, 11.0, 13.0]
    assert metadata["sourceImage"] == {
        "path": "source.png",
        "sha256": hashlib.sha256((output / "source.png").read_bytes()).hexdigest(),
    }
    assert "full-resolution source pixels" in metadata["maskCoordinates"]
    assert "inclusive" in metadata["maskCoordinates"]
    for key, expected, padding in (
        ("observedMask", segmentation.selected, 0),
        ("modelMask", expected_model, dilate),
    ):
        artifact = metadata[key]
        assert artifact["path"] == ("observed-mask.png" if key == "observedMask" else "mask.png")
        assert (
            artifact["sha256"]
            == hashlib.sha256((output / artifact["path"]).read_bytes()).hexdigest()
        )
        ys, xs = np.nonzero(expected)
        assert artifact["bounds"] == [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        assert artifact["areaPixels"] == int(np.count_nonzero(expected))
        assert artifact["areaFraction"] == pytest.approx(np.count_nonzero(expected) / expected.size)
        assert artifact["dilationPx"] == padding
    assert "estimate, not ground truth" in metadata["observedMask"]["purpose"]
    assert "not observation evidence" in metadata["modelMask"]["purpose"]
    assert metadata["modelMask"]["derivedFrom"] == "observed-mask.png"
    # The legacy bounds continue to describe the model input, including its padding.
    assert metadata["personBounds"] == metadata["modelMask"]["bounds"]
    assert metadata["touchesFrame"] is (dilate > 0)
    assert metadata["maskMethod"].startswith(f"{method}: largest component, dilate {dilate}px")
    prepare_cli.decoder.assert_called_once_with(str(prepare_cli.video), 37)
    assert_segmented_once(segmentation, method, metadata["trackBox"])


@pytest.mark.parametrize("method", ["segformer", "maskrcnn"])
def test_empty_masks_retain_existing_cli_rejection(prepare_cli, segmentation, method):
    segmentation.segformer.side_effect = lambda *a, **k: np.zeros((1, 14, 12), np.uint8)
    segmentation.maskrcnn.side_effect = lambda *a, **k: np.zeros((14, 12), np.uint8)

    with pytest.raises(SystemExit, match="unusable person mask:.*empty"):
        prepare_cli.invoke("--method", method)

    assert not list(prepare_cli.output.glob("*.png"))
    assert not (prepare_cli.output / "prepared.json").exists()
    assert_segmented_once(segmentation, method)


def test_negative_dilation_cli_rejected_before_decoding(prepare_cli, segmentation, capsys):
    with pytest.raises(SystemExit) as error:
        prepare_cli.invoke("--dilate", "-1")

    assert error.value.code == 2
    assert "--dilate must be nonnegative" in capsys.readouterr().err
    prepare_cli.decoder.assert_not_called()
    segmentation.segformer.assert_not_called()
    segmentation.maskrcnn.assert_not_called()
    assert not prepare_cli.output.exists()


@pytest.mark.parametrize("method", ["segformer", "maskrcnn"])
def test_negative_dilation_api_rejected_before_segmentation(segmentation, method):
    with pytest.raises(ValueError, match="dilation must be nonnegative"):
        prepare.person_mask(segmentation.rgb, -1, method)

    segmentation.segformer.assert_not_called()
    segmentation.maskrcnn.assert_not_called()
