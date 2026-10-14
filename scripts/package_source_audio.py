"""Declare the supplied video's embedded soundtrack without modifying media."""

import hashlib
import json
import math
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
                "stream=codec_type",
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
            not isinstance(stream, dict) or not isinstance(stream.get("codec_type"), str)
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
    return manifest
