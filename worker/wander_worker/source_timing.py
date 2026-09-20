"""Observe actual FFmpeg fps selections without model/GPU dependencies.

For each output timestamp the fps filter retains the last decoded frame whose
rounded timestamp still lies at or before it. We replay that rule over the
decoded source timestamps and prove every retained frame against the per-frame
CRC32 that FFmpeg writes for both sides of the filter, so no diagnostic is read
back out of its interleaved debug log. Missing/changed diagnostics fail closed.

FFmpeg composes one `showinfo` line from several unterminated `av_log` calls,
so a decoder worker thread logging at `-loglevel debug` splices its own message
into the middle of that line and the frame's `checksum:` field moves to a line
of its own. That race scales with frame count and CPU load, which is why long
clips failed while short ones passed. Machine-readable muxer output has no such
shared line state; keep it that way.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

# "0,      150000,      150000,      150000,   608016, 0xa112184c"
_FRAME_ROW = re.compile(
    r"^0,\s*(-?\d+|N/A),\s*(-?\d+|N/A),\s*(-?\d+|N/A),\s*(\d+),\s*(0x[0-9a-f]+)$"
)
_TIME_BASE_ROW = re.compile(r"^#tb 0: (\d+/\d+)$")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True)


def _observed_frames(report: Path, name: str) -> tuple[Fraction, list[dict]]:
    """Read one framecrc report: its stream timebase and every muxed frame."""
    bases = []
    frames = []
    for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
        base = _TIME_BASE_ROW.match(line)
        if base:
            bases.append(Fraction(base.group(1)))
        elif not line.startswith("#"):
            row = _FRAME_ROW.match(line)
            if not row or row.group(2) == "N/A":
                raise ValueError(f"Incomplete {name} frame diagnostics")
            frames.append(
                {
                    "index": len(frames),
                    "pts": int(row.group(2)),
                    "size": int(row.group(4)),
                    "checksum": row.group(5),
                }
            )
    if len(bases) != 1:
        raise ValueError(f"Missing or changing {name} filter timebase")
    return bases[0], frames


def _rescale_near_inf(pts: int, ratio: Fraction) -> int:
    """Reproduce av_rescale_q's default rounding: nearest, halves away from zero."""
    numerator, denominator = ratio.numerator, ratio.denominator
    if pts < 0:
        return -((-pts * numerator + denominator // 2) // denominator)
    return (pts * numerator + denominator // 2) // denominator


def _parse_selection(reports: dict[str, Path], source_pts: list[int], time_base: Fraction) -> dict:
    input_base, inputs = _observed_frames(reports["source"], "source")
    output_base, outputs = _observed_frames(reports["sampled"], "sampled")
    if len(inputs) != len(source_pts) or not outputs:
        raise ValueError("Decoded source/output counts do not match provenance")
    offsets = {
        pts * time_base - frame["pts"] * input_base
        for pts, frame in zip(source_pts, inputs, strict=True)
    }
    if len(offsets) != 1:
        raise ValueError("FFmpeg and FFprobe decoded source timestamps disagree")
    ratio = input_base / output_base
    rounded = [_rescale_near_inf(frame["pts"], ratio) for frame in inputs]
    # The fps filter numbers its output timestamps consecutively, so a report
    # that skips one lost a frame. How many it writes past the last decoded
    # frame depends on that frame's duration, so the count is not predictable.
    if any(b < a for a, b in pairwise(rounded)) or [f["pts"] for f in outputs] != list(
        range(outputs[0]["pts"], outputs[0]["pts"] + len(outputs))
    ):
        raise ValueError("Incomplete FPS selection diagnostics")
    frames = []
    source_index = 0
    for output in outputs:
        # Take the last decoded frame that still rounds to at or before this
        # output timestamp, then hold that choice to the emitted bytes. Where
        # neighbouring source frames are pixel-identical their checksums cannot
        # separate them, so scripts/test_source_timing.py also holds this rule
        # to the filter's own account of what it wrote.
        while source_index + 1 < len(rounded) and rounded[source_index + 1] <= output["pts"]:
            source_index += 1
        retained = inputs[source_index]
        if (
            rounded[source_index] > output["pts"]
            or retained["checksum"] != output["checksum"]
            or retained["size"] != output["size"]
        ):
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
                "resampledPts": output["pts"],
            }
        )
    return {
        "sourceTimestampOffset": str(offsets.pop()),
        "resampledTimeBase": str(output_base),
        "frames": frames,
    }


def resample_source(
    source: str | Path,
    fps: float = 12.0,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[bytes, dict]:
    """Return RGB24 bytes and exact provenance; omit dimensions for mapping only.

    Mapping-only mode decodes to CRC reports, without inference or retaining
    images. Reconstruction of an old run is observational evidence for this
    FFmpeg version, not independent proof of that run's historical inputs.
    Absolute source PTS are integers in sourceTimeBase; relative seconds use
    the first decoded source frame, not container start_time or average FPS.
    """
    source = Path(source)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if (width is None) != (height is None) or (width is not None and (width <= 0 or height <= 0)):
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
    # One decode feeds every branch, so the CRCs compare the same decoded frames
    # and the sampled branch shares a single fps instance with the RGB output.
    filters = f"[0:v:0]split=2[source][input];[input]fps@sample={fps}[sampled]"
    pixels = []
    if width is not None:
        filters += f";[sampled]split=2[crc][raw];[raw]scale={width}:{height}[pixels]"
        pixels = ["-map", "[pixels]", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    # Two scratch files, not a scratch directory: this runs inside a caller's own
    # temporary tree (the worker decodes from one), and removing a directory we
    # merely named would take the caller's inputs with it.
    with (
        tempfile.NamedTemporaryFile(suffix="-source.framecrc") as source_report,
        tempfile.NamedTemporaryFile(suffix="-sampled.framecrc") as sampled_report,
    ):
        reports = {"source": Path(source_report.name), "sampled": Path(sampled_report.name)}
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-nostdin",
            "-loglevel",
            "repeat+error",
            "-i",
            str(source),
            "-filter_complex",
            filters,
            # Without -enc_time_base the muxer stamps this branch in 1/framerate
            # (measured: 1/60 for a 1/9000000 source) and rescales every pts, so
            # a variable-rate source would lose the very timestamps the fps
            # filter read. -1 keeps the demuxer timebase; it needs FFmpeg 5.1.
            "-map",
            "[source]",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "-1",
            "-f",
            "framecrc",
            "-y",
            str(reports["source"]),
            "-map",
            "[crc]" if width is not None else "[sampled]",
            "-fps_mode",
            "passthrough",
            "-f",
            "framecrc",
            "-y",
            str(reports["sampled"]),
            *pixels,
        ]
        try:
            decoded = _run(command)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode("utf-8", errors="replace")
            if "enc_time_base" in detail:
                raise ValueError("FFmpeg is too old for -enc_time_base; 5.1 or newer") from error
            raise
        provenance = _parse_selection(reports, source_pts, time_base)
    if width is not None and len(decoded.stdout) != len(provenance["frames"]) * width * height * 3:
        raise ValueError("RGB frame count does not match provenance")
    if _sha256(source) != source_sha:
        raise ValueError("Source changed during provenance capture")
    version = _run(["ffmpeg", "-version"]).stdout.decode().splitlines()[0]
    provenance.update(
        schema="wander.source-frame-provenance/1",
        method="ffmpeg-fps-observed-selection",
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
