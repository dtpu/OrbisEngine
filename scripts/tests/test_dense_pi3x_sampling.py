"""Verify which source pixels and timestamps reach the Pi3X camera producer."""

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.stages import dense_pi3x


def encode_frames(path, frame_count, rate, *, timestamps=None):
    # Each frame has a distinct solid RGB color. FFV1 preserves it before JPEG preprocessing.
    colors = [(i * 18, 200 - i * 10, 30 + i * 14) for i in range(frame_count)]
    raw = b"".join(bytes(color) * (16 * 16) for color in colors)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "16x16",
        "-framerate",
        str(rate),
        "-i",
        "-",
    ]
    if timestamps:
        expression = str(timestamps[-1])
        for i, pts in reversed(list(enumerate(timestamps[:-1]))):
            expression = f"if(eq(N,{i}),{pts},{expression})"
        command.extend(["-vf", f"settb=1/1000,setpts='{expression}'", "-enc_time_base", "1:1000"])
    subprocess.run(
        [*command, "-fps_mode", "passthrough", "-c:v", "ffv1", str(path)],
        input=raw,
        check=True,
    )
    return colors


def check_decoded_pixels(files, colors, indices):
    assert [p.name for p in files] == [f"f_{i:06d}.jpg" for i in range(len(indices))]
    for path, ordinal in zip(files, indices, strict=True):
        with Image.open(path) as im:
            np.testing.assert_allclose(np.asarray(im)[8, 8], colors[ordinal], atol=3, rtol=0)


def test_variable_rate_nonzero_start_binds_camera_time_to_decoded_frame(tmp_path):
    source = tmp_path / "variable.mkv"
    pts = [1000, 1030, 1060, 1100, 1140, 1170, 1210, 1240]
    colors = encode_frames(source, len(pts), 25, timestamps=pts)
    files, provenance, size, duration, _ = dense_pi3x.decode_camera_samples(
        source, tmp_path / "input", 12
    )
    assert provenance["cameraSampling"] == "nearest-source-pts"
    assert provenance["sourceFirstPts"] == 1000
    assert provenance["sourceTimeBase"] == "1/1000"
    assert size == (16, 16)
    assert duration == pytest.approx(0.28)
    indices = [0, 3, 5, 7]
    assert [dense_pi3x.camera_source_fields(frame) for frame in provenance["frames"]] == [
        {"sourceIndex": ordinal, "sourcePts": pts[ordinal], "time": (pts[ordinal] - 1000) / 1000}
        for ordinal in indices
    ]
    check_decoded_pixels(files, colors, indices)
    assert [frame["cameraInputImageSha256"] for frame in provenance["frames"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    ]


def test_constant_rate_keeps_legacy_indices_and_records_decoded_pts(tmp_path):
    source = tmp_path / "constant.mkv"
    colors = encode_frames(source, 12, 30)
    files, provenance, size, duration, fps = dense_pi3x.decode_camera_samples(
        source, tmp_path / "input", 12
    )
    assert provenance["cameraSampling"] == "nominal-cfr-ordinal"
    assert (size, fps, duration) == ((16, 16), 30, 0.4)
    indices = [0, 2, 5, 8, 10]
    assert [f["sourceIndex"] for f in provenance["frames"]] == indices
    # Matroska rounds 30 fps onto milliseconds. Time uses actual PTS rather than index / 30.
    assert [f["sourcePts"] for f in provenance["frames"]] == [0, 67, 167, 267, 333]
    assert [f["sourceRelativeTimeSeconds"] for f in provenance["frames"]] == [
        0,
        0.067,
        0.167,
        0.267,
        0.333,
    ]
    check_decoded_pixels(files, colors, indices)


def test_sampling_fails_closed_on_stale_images_or_missing_view(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    encode_frames(source, 12, 30)
    stale = tmp_path / "stale"
    stale.mkdir()
    old = stale / "f_000000.jpg"
    old.write_bytes(b"historical image")
    with pytest.raises(FileExistsError, match="input directory already exists"):
        dense_pi3x.decode_camera_samples(source, stale, 12)
    assert old.read_bytes() == b"historical image"

    decode = dense_pi3x.select_source_frames

    def wrong_frame(*args, **kwargs):
        rgb, provenance = decode(*args, **kwargs)
        provenance["frames"][1]["sourceIndex"] += 1
        return rgb, provenance

    monkeypatch.setattr(dense_pi3x, "select_source_frames", wrong_frame)
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="do not match probed"):
        dense_pi3x.decode_camera_samples(source, fresh, 12)
    assert not fresh.exists()


def test_insufficient_frame_density_is_not_synthesized(tmp_path):
    source = tmp_path / "sparse.mkv"
    encode_frames(source, 6, 6)
    frames = tmp_path / "input"
    with pytest.raises(ValueError, match="below requested independent output density"):
        dense_pi3x.decode_camera_samples(source, frames, 12)
    assert not frames.exists()
