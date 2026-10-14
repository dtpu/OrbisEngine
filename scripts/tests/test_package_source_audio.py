"""Both package layouts declare real embedded audio without altering source media."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_multiperson
import package_person_sequence
import package_source_audio
import run_clip


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required for real stream fixtures")
    root = tmp_path_factory.mktemp("source-audio")
    outputs = {}
    for has_audio in (False, True):
        clip = root / f"audio-{has_audio}.mp4"
        command = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=96x64:d=1",
        ]  # fmt: skip
        if has_audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        command += ["-c:a", "aac", "-shortest"] if has_audio else ["-an"]
        subprocess.run([*command, str(clip)], check=True, capture_output=True, timeout=30)
        outputs[has_audio] = clip
    return outputs


def package_inputs(root, multiple, duration=0.7):
    cameras = root / "cameras.json"
    cameras.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "sourceIndex": index,
                        "time": time,
                        "camera_to_world": np.eye(4).tolist(),
                        "source_intrinsics": np.eye(3).tolist(),
                        "source_image_size": [96, 64],
                    }
                    for index, time in [(0, 0), (2, 0.1)]
                ]
            }
        )
    )
    motions = []
    fields = (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )  # fmt: skip
    for tid in range(2 if multiple else 1):
        motion = root / f"motion-{tid}"
        motion.mkdir()
        rows = np.zeros(8, dtype=[(field, "<f4") for field in fields])
        rows["y"] = np.linspace(-1, 1, 8)
        rows["z"], rows["opacity"], rows["rot_0"] = -2, 2, 1
        names = ["frame_000.ply", "frame_001.ply"]
        for index, name in enumerate(names):
            rows["x"] = index * 0.05 + tid
            PlyData([PlyElement.describe(rows, "vertex")], text=False, byte_order="<").write(
                motion / name
            )
        sequence = dict(frames=names, fps=12, sourceIndices=[0, 2], timestamps=[0, 0.1])
        if duration is not None:
            sequence["duration"] = duration
        (motion / "sequence.json").write_text(json.dumps(sequence))
        motions.append(motion)
    if not multiple:
        return package_person_sequence, [str(motions[0]), "--cameras", str(cameras)]
    tracks = root / "tracks"
    tracks.mkdir()
    (tracks / "tracks.json").write_text(
        json.dumps(
            dict(
                tracks=[dict(track=tid, quality={"score": 1}) for tid in range(2)],
                samples=2,
                duration=duration,
                sourceIndices=[0, 2],
                timestamps=[0, 0.1],
                method="owned synthetic fixture",
                trackCount=2,
                discardedShortTracks=[],
            )
        )
    )
    args = ["--tracks", str(tracks), "--cameras", str(cameras), "--shared-scale", "none"]
    for tid, motion in enumerate(motions):
        args += ["--motion", f"{tid}={motion}"]
    return package_multiperson, args


def run_package(monkeypatch, module, args, clip, out):
    monkeypatch.setattr(
        sys, "argv", [module.__name__, *args, "--clip", str(clip), "--out", str(out)]
    )
    module.main()


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("has_audio", [False, True])
def test_real_streams_and_package_timeline(tmp_path, monkeypatch, clips, multiple, has_audio):
    module, args = package_inputs(tmp_path, multiple)
    out = tmp_path / "world"
    clip = clips[has_audio]
    before = clip.read_bytes()
    run_package(monkeypatch, module, args, clip, out)
    audio = json.loads((out / "audio.json").read_text())
    timeline_file = out / ("people.json" if multiple else "person/sequence.json")
    timeline = json.loads(timeline_file.read_text())
    assert audio == {
        "schema": "wander.audio/1",
        "source": {"hasAudio": has_audio, "sha256": hashlib.sha256(before).hexdigest()},
        "timeline": {"durationSeconds": timeline["duration"]},
        "defaultMode": "original",
    }
    assert timeline["duration"] == 0.7  # Deliberately different from the one-second source.
    assert clip.read_bytes() == before  # Embedded AAC remains byte-for-byte intact.
    run_package(
        monkeypatch, module, args, clip, out
    )  # Identical generated declaration is reusable.
    assert json.loads((out / "audio.json").read_text()) == audio


def test_legacy_sequence_duration_matches_viewer(tmp_path, monkeypatch, clips):
    module, args = package_inputs(tmp_path, False, duration=None)
    out = tmp_path / "world"
    run_package(monkeypatch, module, args, clips[False], out)
    assert json.loads((out / "audio.json").read_text())["timeline"]["durationSeconds"] == 2 / 12


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize(
    "failure", ["corrupt-source", "authored-manifest", "malformed-manifest", "changed-source"]
)
def test_refusal_preserves_existing_package(tmp_path, monkeypatch, clips, multiple, failure):
    module, args = package_inputs(tmp_path, multiple)
    out = tmp_path / "world"
    out.mkdir()
    sentinel = out / "person" / "sequence.json"
    sentinel.parent.mkdir()
    sentinel.write_text("existing person bytes")
    audio = package_source_audio.source_audio_manifest(clips[False], 0.7, out)
    clip = clips[False]
    if failure == "authored-manifest":
        audio.update(original={"url": "audio/reviewed.wav"}, provenance={"attribution": "Owner"})
    elif failure == "changed-source":
        # Same stream types and duration, but a distinct source identity.
        clip = tmp_path / "changed.mp4"
        clip.write_bytes(clips[False].read_bytes() + b"changed source identity")
    elif failure == "corrupt-source":
        clip = tmp_path / "corrupt.mp4"
        clip.write_bytes(b"not a video")
    audio_file = out / "audio.json"
    audio_file.write_text("not JSON" if failure == "malformed-manifest" else json.dumps(audio))
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="ffprobe failed|fresh destination"):
        run_package(monkeypatch, module, args, clip, out)
    after = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize(
    "stdout",
    [
        "not JSON",
        "[]",
        "{}",
        '{"streams":[]}',
        '{"streams":[{}]}',
        '{"streams":[{"codec_type":"audio"}]}',
    ],
)
def test_invalid_probe_cannot_declare_silence(tmp_path, monkeypatch, stdout):
    monkeypatch.setattr(
        package_source_audio.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=stdout),
    )
    with pytest.raises(ValueError, match="Cannot determine source audio streams"):
        package_source_audio.source_audio_manifest(tmp_path / "clip.mp4", 1, tmp_path)
    assert not (tmp_path / "audio.json").exists()


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_package_duration_fails_before_probe(tmp_path, monkeypatch, duration):
    def unexpected_probe(*args, **kwargs):
        pytest.fail("Invalid package duration must fail before ffprobe")

    monkeypatch.setattr(package_source_audio.subprocess, "run", unexpected_probe)
    with pytest.raises(ValueError, match="finite positive package duration"):
        package_source_audio.source_audio_manifest(tmp_path / "clip.mp4", duration, tmp_path)


def runner_inputs(root, monkeypatch, clip, multiple, duration=0.7):
    ctx = root / "run"
    folder = ctx / ("lhm-motion-00" if multiple else "lhm-motion")
    folder.mkdir(parents=True)
    seq = dict(frames=["frame_000.ply", "frame_001.ply"], fps=12)
    if duration is not None:
        seq["duration"] = duration
    (folder / "sequence.json").write_text(json.dumps(seq))
    if multiple:
        (ctx / "tracks").mkdir()
        (ctx / "tracks" / "tracks.json").write_text(json.dumps({"duration": duration}))
    public = root / "public"
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.name, pipeline.ctx, pipeline.clip, pipeline.people_limit = "fixture", ctx, clip, 2
    out = public / "worlds" / "fixture-4d"
    out.mkdir(parents=True)
    child = Mock()
    monkeypatch.setattr(run_clip, "run", child)
    return pipeline, out, child


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("failure", ["authored", "changed-source", "malformed", "failed-probe"])
def test_runner_refusal_preserves_all_destination_bytes(
    tmp_path, monkeypatch, clips, multiple, failure
):
    pipeline, out, child = runner_inputs(tmp_path, monkeypatch, clips[False], multiple)
    audio = package_source_audio.source_audio_manifest(pipeline.clip, 0.7, out)
    if failure == "authored":
        audio.update(original={"url": "audio/reviewed.wav"}, provenance={"attribution": "Owner"})
    elif failure in {"changed-source", "failed-probe"}:
        pipeline.clip = tmp_path / "other.mp4"
        pipeline.clip.write_bytes(
            clips[False].read_bytes() + b"different identity"
            if failure == "changed-source"
            else b"invalid video"
        )
    (out / "audio.json").write_text("not JSON" if failure == "malformed" else json.dumps(audio))
    for name in ["person/sequence.json", "audio/reviewed.wav", "placement.json", "people.json"]:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"existing {name}".encode())
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="fresh destination|ffprobe failed"):
        (pipeline.package_people if multiple else pipeline.package)()
    child.assert_not_called()
    assert before == {
        path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("multiple,duration", [(False, None), (False, 0.7), (True, 0.7)])
def test_runner_matching_manifest_allows_repackage(
    tmp_path, monkeypatch, clips, multiple, duration
):
    pipeline, out, child = runner_inputs(tmp_path, monkeypatch, clips[True], multiple, duration)
    expected_duration = duration if duration is not None else 2 / 12
    audio = package_source_audio.source_audio_manifest(pipeline.clip, expected_duration, out)
    (out / "audio.json").write_text(json.dumps(audio))

    def launched(command, log):
        assert not out.exists()  # Preflight completed before the authorized rebuild removes it.
        assert Path(command[command.index("--out") + 1]) == out
        assert Path(command[command.index("--clip") + 1]) == pipeline.clip

    child.side_effect = launched
    (pipeline.package_people if multiple else pipeline.package)()
    child.assert_called_once()
