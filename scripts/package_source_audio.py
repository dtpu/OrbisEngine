"""Preserve source media and prepare lossless browser delivery for H.264/PCM clips."""

import hashlib
import json
import math
import tempfile
from pathlib import Path
import subprocess


def source_audio_manifest(clip, duration, world):
    """Probe before packaging; an unreadable stream list must never mean silence."""
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("Audio timeline requires a finite positive package duration")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,start_time,time_base,sample_rate,channels",
                "-of",
                "json",
                str(clip),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        probe = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ValueError(
            f"Cannot determine source audio streams for {clip}: ffprobe failed"
        ) from error
    streams = probe.get("streams") if isinstance(probe, dict) else None
    if (
        not isinstance(streams, list)
        or not streams
        or any(
            not isinstance(stream, dict)
            or not isinstance(stream.get("codec_type"), str)
            or not isinstance(stream.get("codec_name", ""), str)
            for stream in streams
        )
        or not any(stream["codec_type"] == "video" for stream in streams)
    ):
        raise ValueError(
            f"Cannot determine source audio streams for {clip}: invalid video stream list"
        )

    with Path(clip).open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = dict(
        schema="wander.audio/1",
        source={
            "hasAudio": any(stream["codec_type"] == "audio" for stream in streams),
            "sha256": source_hash,
        },
        timeline={"durationSeconds": duration},
        defaultMode="original",
    )
    audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]
    if any(stream.get("codec_name", "").startswith("pcm_") for stream in audio_streams):
        video_streams = [stream for stream in streams if stream["codec_type"] == "video"]
        if (
            len(video_streams) != 1
            or video_streams[0].get("codec_name") != "h264"
            or len(audio_streams) != 1
            or audio_streams[0].get("codec_name") not in {"pcm_s16le", "pcm_s24le", "pcm_s32le"}
            or any(stream.get("start_time") != "0.000000" for stream in streams)
            or not isinstance(video_streams[0].get("time_base"), str)
            or not video_streams[0]["time_base"].startswith("1/")
        ):
            raise ValueError(
                "Lossless PCM delivery requires one zero-origin H.264 video and one PCM audio stream"
            )
        manifest["source"].update(hasAudio=False, videoUrl="media/source.mp4")
        manifest["original"] = {"url": "media/original.wav", "offsetSeconds": 0}
        manifest["provenance"] = {"delivery": "media/provenance.json"}
    # Old authored audio may belong to another source, even if the clip's stream types match.
    path = Path(world) / "audio.json"
    if path.exists():
        message = (
            f"Existing audio manifest differs from the supplied source: {path}. "
            "Use a fresh destination or explicitly preserve and review authored audio."
        )
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(message) from error
        if previous != manifest:
            raise ValueError(message)
        # Runner rebuilds remove the destination after this preflight. Equal metadata
        # alone must never authorize deleting a missing or edited delivery asset.
        if "videoUrl" in manifest["source"]:
            try:
                validate_source_media(world, manifest, source_hash)
            except (OSError, ValueError) as error:
                raise ValueError(
                    "Existing source-media delivery is missing or changed; use a fresh destination"
                ) from error
    return manifest


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120).stdout


def _video_packets(path):
    return json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-show_entries",
                "packet=pts,dts,duration,data_hash:stream=time_base",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def _audio_timing(path):
    probe = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_packets",
                "-show_entries",
                "packet=pts,duration:stream=time_base,sample_rate,channels,codec_name",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    stream = probe["streams"][0]
    if stream["time_base"] != "1/" + stream["sample_rate"]:
        raise ValueError("PCM delivery requires sample-aligned timestamps")
    sample_count = 0
    for packet in probe["packets"]:
        if packet.get("pts") != sample_count or packet.get("duration", 0) <= 0:
            raise ValueError("PCM delivery cannot remove audio timestamp gaps or overlaps")
        sample_count += packet["duration"]
    if not sample_count:
        raise ValueError("PCM delivery requires recorded audio samples")
    return {**stream, "sampleCount": sample_count}


def _pcm_hash(path):
    return (
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ]
        )
        .strip()
        .removeprefix("SHA256=")
    )


def prepare_source_media(clip, duration, world):
    """Materialize only a declared lossless delivery; never overwrite authored audio.

    A staging directory keeps failed stream checks from changing package assets. The
    immutable submitted source remains authoritative; the MP4 only supplies the clock.
    """
    world = Path(world)
    manifest = source_audio_manifest(clip, duration, world)
    if "videoUrl" not in manifest["source"]:
        return manifest
    world.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-media-", dir=world.parent) as temporary:
        staged = Path(temporary)
        video, audio = staged / "source.mp4", staged / "original.wav"
        original_audio = _audio_timing(clip)
        original_packets = _video_packets(clip)
        time_base = original_packets["streams"][0]["time_base"]
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                "-video_track_timescale",
                time_base.split("/")[1],
                "-movflags",
                "+faststart",
                str(video),
            ]
        )
        _run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                str(audio),
            ]
        )
        delivered_packets = _video_packets(video)
        # Container metadata differs, but every encoded packet and its timestamp must match.
        if (
            original_packets["packets"] != delivered_packets["packets"]
            or time_base != delivered_packets["streams"][0]["time_base"]
        ):
            raise ValueError("Lossless delivery changed video packets or timing")
        pcm_hash = _pcm_hash(clip)
        if pcm_hash != _pcm_hash(audio) or original_audio != _audio_timing(audio):
            raise ValueError("Lossless delivery changed original PCM samples")
        receipt = {
            "schema": "wander.source-media/1",
            "sourceSha256": manifest["source"]["sha256"],
            "video": {"url": "source.mp4", "sha256": file_sha256(video), "mode": "stream-copy"},
            "audio": {"url": "original.wav", "sha256": file_sha256(audio), "mode": "stream-copy"},
            "verification": {
                "videoPacketsAndTimestampsIdentical": True,
                "videoPacketCount": len(original_packets["packets"]),
                "videoTimeBase": time_base,
                "pcmSha256": pcm_hash,
                "audio": original_audio,
            },
        }
        (staged / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
        destination = world / "media"
        if destination.exists():
            if (
                not destination.is_dir()
                or destination.is_symlink()
                or any(
                    not (destination / name).is_file()
                    or (destination / name).is_symlink()
                    or file_sha256(destination / name) != file_sha256(staged / name)
                    for name in ("source.mp4", "original.wav", "provenance.json")
                )
            ):
                raise ValueError("Existing source-media delivery differs; use a fresh destination")
        else:
            world.mkdir(parents=True, exist_ok=True)
            staged.rename(destination)
    return manifest


def validate_source_media(world, manifest, source_sha256):
    """Validate declared package-local delivery files without running a media tool."""
    world = Path(world)
    if not isinstance(manifest, dict):
        raise ValueError("Invalid source-media manifest")
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("Invalid source-media source declaration")
    if "videoUrl" not in source:
        return None
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Invalid source-media provenance declaration")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
        or manifest["source"].get("videoUrl") != "media/source.mp4"
        or manifest["source"].get("hasAudio") is not False
        or manifest["source"].get("sha256") != source_sha256
        or manifest.get("original") != {"url": "media/original.wav", "offsetSeconds": 0}
        or provenance.get("delivery") != "media/provenance.json"
    ):
        raise ValueError("Invalid lossless source-media declaration")
    for name in ("source.mp4", "original.wav", "provenance.json"):
        path = world / "media" / name
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(world.resolve())
        ):
            raise ValueError("Missing or unsafe source-media delivery")
    if (world / "media").is_symlink():
        raise ValueError("Unsafe source-media directory")
    receipt = json.loads((world / "media/provenance.json").read_text())
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "wander.source-media/1"
        or receipt.get("sourceSha256") != source_sha256
    ):
        raise ValueError("Source-media provenance names a different source")
    for kind, name in (("video", "source.mp4"), ("audio", "original.wav")):
        entry = receipt.get(kind)
        if not isinstance(entry, dict) or entry != {
            "url": name,
            "sha256": file_sha256(world / "media" / name),
            "mode": "stream-copy",
        }:
            raise ValueError("Source-media delivery bytes do not match provenance")
    return "media/source.mp4"
