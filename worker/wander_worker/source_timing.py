"""Observe actual FFmpeg fps selections without model/GPU dependencies.

The fps filter reports the source timestamp it writes, in its rounded output
timebase. Several inputs can share that timestamp; its last input at that time
is retained. We observe those events and verify the unchanged frame checksums
on both sides of the filter. Missing/changed diagnostics fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        # The decoder stdout is raw RGB; retain only bounded text diagnostics.
        stderr = error.stderr or b""
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8")
        tail = stderr[-8192:].decode("utf-8", errors="replace")
        truncated = " (tail; truncated)" if len(stderr) > 8192 else ""
        raise RuntimeError(
            f"{Path(command[0]).name} exited with status {error.returncode}; "
            f"stderr{truncated}:\n{tail}"
        ) from None


def _observed_frames(log: str, name: str) -> tuple[Fraction, list[dict]]:
    prefix = rf"\[showinfo@{name} @ [^\]]+\] "
    bases = re.findall(prefix + r"config in time_base: (\d+/\d+)", log)
    if len(bases) != 1:
        raise ValueError(f"Missing or changing {name} filter timebase")
    frames = []
    pending = None
    for event in re.finditer(rf"({prefix})([^\n]*)", log):
        instance, message = event.groups()
        header = re.match(r"n:\s*(\d+) pts:\s*(-?\d+) ", message)
        checksum = re.search(r"(?:^| )checksum:([0-9A-F]+)(?: |$)", message)
        if header:
            if pending is not None:
                raise ValueError(f"Incomplete {name} frame diagnostics")
            pending = (instance, int(header[1]), int(header[2]))
        if checksum:
            if pending is None or instance != pending[0]:
                raise ValueError(f"Orphan or wrong-instance {name} checksum diagnostics")
            if not header and not message.startswith("checksum:"):
                raise ValueError(f"Invalid {name} checksum continuation")
            frames.append({"index": pending[1], "pts": pending[2], "checksum": checksum[1]})
            pending = None
        elif pending is not None and not header:
            raise ValueError(f"Incomplete {name} frame diagnostics")
        # FFmpeg can interleave another thread's message before a showinfo
        # checksum. Only its same-instance continuation can finish that frame.
    if pending is not None or [f["index"] for f in frames] != list(range(len(frames))):
        raise ValueError(f"Incomplete {name} frame diagnostics")
    return Fraction(bases[0]), frames


def _parse_selection(log: str, source_pts: list[int], time_base: Fraction) -> dict:
    input_base, inputs = _observed_frames(log, "source")
    output_base, outputs = _observed_frames(log, "sampled")
    if len(inputs) != len(source_pts) or not outputs:
        raise ValueError("Decoded source/output counts do not match provenance")
    offsets = {
        pts * time_base - frame["pts"] * input_base
        for pts, frame in zip(source_pts, inputs, strict=True)
    }
    if len(offsets) != 1:
        raise ValueError("FFmpeg and FFprobe decoded source timestamps disagree")
    reads = []
    writes = []
    latest = {}
    for line in log.splitlines():
        if not re.match(r"\[fps@sample @ [^\]]+\] ", line):
            continue
        read = re.search(r"Read frame with in pts (-?\d+), out pts (-?\d+)$", line)
        write = re.search(r"Writing frame with pts (-?\d+) to pts (-?\d+)$", line)
        if read:
            pts, rounded = map(int, read.groups())
            ordinal = len(reads)
            if ordinal >= len(inputs) or inputs[ordinal]["pts"] != pts:
                raise ValueError("FPS filter input does not match decoded source")
            reads.append(pts)
            latest[rounded] = ordinal
        elif write:
            rounded, output_pts = map(int, write.groups())
            if rounded not in latest:
                raise ValueError("FPS filter wrote an unidentified source frame")
            writes.append((latest[rounded], output_pts))
    if len(reads) != len(inputs) or len(writes) != len(outputs):
        raise ValueError("Incomplete FPS selection diagnostics")
    frames = []
    for output, (source_index, output_pts) in zip(outputs, writes, strict=True):
        if output["pts"] != output_pts or output["checksum"] != inputs[source_index]["checksum"]:
            raise ValueError("FPS selection does not match the emitted frame")
        frames.append(
            {
                "frameIndex": output["index"],
                "sourceIndex": source_index,
                "sourcePts": source_pts[source_index],
                "sourceTimeSeconds": float(source_pts[source_index] * time_base),
                "sourceRelativeTimeSeconds": float(
                    (source_pts[source_index] - source_pts[0]) * time_base
                ),
                "resampledPts": output_pts,
            }
        )
    return {
        "sourceTimestampOffset": str(offsets.pop()),
        "resampledTimeBase": str(output_base),
        "frames": frames,
    }


def _decode_source(
    source: str | Path,
    fps: float = 12.0,
    *,
    source_indices: list[int] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[bytes, dict]:
    """Return RGB24 bytes and exact provenance; omit dimensions for mapping only.

    Mapping-only mode decodes to a null sink, without inference or retaining
    images. Reconstruction of an old run is observational evidence for this
    FFmpeg version, not independent proof of that run's historical inputs.
    Absolute source PTS are integers in sourceTimeBase; relative seconds use
    the first decoded source frame, not container start_time or average FPS.
    """
    source = Path(source)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be positive and finite")
    if (width is None) != (height is None) or (
        width is not None
        and (type(width) is not int or type(height) is not int or width <= 0 or height <= 0)
    ):
        raise ValueError("Specify both positive output dimensions or neither")
    source_sha = _sha256(source)
    probe = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=time_base,avg_frame_rate:frame=pts",
                "-of",
                "json",
                str(source),
            ]
        ).stdout
    )
    time_base = Fraction(probe["streams"][0]["time_base"])
    try:
        source_pts = [int(frame["pts"]) for frame in probe["frames"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Source has missing decoded PTS") from error
    if not source_pts or any(b <= a for a, b in pairwise(source_pts)):
        raise ValueError("Source PTS must be present and strictly increasing")
    if source_indices is not None:
        if (
            not isinstance(source_indices, list)
            or not source_indices
            or any(type(i) is not int or not 0 <= i < len(source_pts) for i in source_indices)
            or any(b <= a for a, b in pairwise(source_indices))
        ):
            raise ValueError(
                "Explicit source indices must be strictly increasing integers in range"
            )
        expression = "+".join(f"eq(n\\,{i})" for i in source_indices)
        filters = f"showinfo@source,select={expression},showinfo@sampled"
    else:
        filters = f"showinfo@source,fps@sample={fps},showinfo@sampled"
    output = ["-f", "null", "-"]
    if width is not None:
        filters += f",scale={width}:{height}"
        output = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "debug",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        filters,
        "-fps_mode",
        "passthrough",
        *output,
    ]
    decoded = _run(command)
    log = decoded.stderr.decode("utf-8", errors="replace")
    provenance = (
        _parse_explicit_selection(log, source_pts, time_base, source_indices)
        if source_indices is not None
        else _parse_selection(log, source_pts, time_base)
    )
    if width is not None and len(decoded.stdout) != len(provenance["frames"]) * width * height * 3:
        raise ValueError("RGB frame count does not match provenance")
    if _sha256(source) != source_sha:
        raise ValueError("Source changed during provenance capture")
    version = _run(["ffmpeg", "-version"]).stdout.decode().splitlines()[0]
    provenance.update(
        schema="wander.source-frame-provenance/1",
        method="ffmpeg-explicit-observed-selection"
        if source_indices is not None
        else "ffmpeg-fps-observed-selection",
        sourceSha256=source_sha,
        sourceTimeBase=str(time_base),
        sourceFirstPts=source_pts[0],
        sourceFrameCount=len(source_pts),
        sourceAverageFrameRate=probe["streams"][0].get("avg_frame_rate", "0/1"),
        requestedFps=fps,
        ffmpegVersion=version,
        filter=filters,
        stream="v:0",
        timestampPolicy="ffmpeg-default-input-offset; output-passthrough",
    )
    return decoded.stdout, provenance


def _parse_explicit_selection(log, source_pts, time_base, selection):
    input_base, inputs = _observed_frames(log, "source")
    output_base, outputs = _observed_frames(log, "sampled")
    if len(inputs) != len(source_pts) or len(outputs) != len(selection):
        raise ValueError("Explicit selection decoded counts disagree")
    offsets = {
        pts * time_base - frame["pts"] * input_base
        for pts, frame in zip(source_pts, inputs, strict=True)
    }
    if len(offsets) != 1:
        raise ValueError("FFmpeg and FFprobe decoded source timestamps disagree")
    frames = []
    for output, index in zip(outputs, selection, strict=True):
        if (
            output["pts"] * output_base != inputs[index]["pts"] * input_base
            or output["checksum"] != inputs[index]["checksum"]
        ):
            raise ValueError("Explicit selection does not match the emitted frame")
        frames.append(
            {
                "frameIndex": output["index"],
                "sourceIndex": index,
                "sourcePts": source_pts[index],
                "sourceTimeSeconds": float(source_pts[index] * time_base),
                "sourceRelativeTimeSeconds": float((source_pts[index] - source_pts[0]) * time_base),
                "resampledPts": output["pts"],
            }
        )
    return {
        "sourceTimestampOffset": str(offsets.pop()),
        "resampledTimeBase": str(output_base),
        "frames": frames,
    }


def resample_source(source, fps=12.0, *, width=None, height=None):
    """Return RGB24 bytes and observed FPS-filter provenance (or mapping only)."""
    return _decode_source(source, fps, width=width, height=height)


def select_source_frames(source, fps=12.0, *, source_indices, width=None, height=None):
    """Decode exact source ordinals; fps labels the output, never changes its selection."""
    if source_indices is None:
        raise ValueError("Explicit source indices are required")
    return _decode_source(source, fps, source_indices=source_indices, width=width, height=height)


def select_provenance(provenance: dict, selection: list[int]) -> list[dict]:
    """Retain exact source bindings in the caller's requested output order."""
    frames = provenance["frames"]
    if not selection or any(
        type(index) is not int or not 0 <= index < len(frames) for index in selection
    ):
        raise ValueError("Selected frame indices must be nonempty and in range")
    if len(selection) != len(set(selection)):
        raise ValueError("Selected frame indices must be unique")
    return [dict(frames[index]) for index in selection]


def match_camera_selection(
    provenance: dict, cameras: list[dict], source_indices: list[int]
) -> list[dict]:
    """Require exact retained source frames and consistent camera timestamp labels.

    This validates correspondence labels, not the estimated camera geometry.
    The caller separately binds the camera document and source file hashes.
    """
    if not source_indices or len(set(source_indices)) != len(source_indices):
        raise ValueError("Selected source indices must be nonempty and unique")
    rows = {}
    for camera in cameras:
        index = camera.get("sourceIndex")
        if type(index) is not int or index < 0 or index in rows:
            raise ValueError("Camera source indices must be nonnegative and unique")
        rows[index] = camera
    retained = {}
    for frame in provenance["frames"]:
        retained.setdefault(frame["sourceIndex"], frame)
    selected = []
    for index in source_indices:
        if type(index) is not int or index not in retained or index not in rows:
            raise ValueError(
                f"Selected camera source frame {index} is not retained by the FPS filter"
            )
        frame = retained[index]
        time = rows[index].get("time")
        if (
            isinstance(time, bool)
            or not isinstance(time, (int, float))
            or not math.isfinite(time)
            or abs(time - frame["sourceRelativeTimeSeconds"]) > 1e-6
        ):
            raise ValueError(f"Camera time does not match actual source PTS for frame {index}")
        selected.append(dict(frame))
    return selected
