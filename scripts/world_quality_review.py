"""Read-only manual static-world import, adapted from Austin Jian's 52f211b.

Hashes bind the reviewed assertions to current files; they do not authenticate a human or
prove that camera registration is geometrically correct. No API, repair, publishing or ledger
operation is available here. ffprobe is a bounded local source-timestamp inspection only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import pairwise
from pathlib import Path

from quality_gate import (
    CRITERIA,
    EVIDENCE_SCHEMA,
    PASS_FRACTION,
    PLAN_SCHEMA,
    SCOPE,
    QualityGateError,
    evaluate_manual,
    evidence_file,
    file_sha256,
    finite,
    read_document,
    sha,
    text,
)

TIMING_METHOD = "ffprobe-pts/ffmpeg-select-v1"


def evidence_code_sha256() -> str:
    """Invalidate reports when capture/validation/render code changes."""
    root = Path(__file__).resolve().parent
    files = (
        "quality_gate.py",
        "world_quality_review.py",
        "verify_world.py",
        "render_world_poses.py",
    )
    return hashlib.sha256(
        json.dumps({name: file_sha256(root / name) for name in files}, sort_keys=True).encode()
    ).hexdigest()


def source_timestamps(clip: Path) -> list[float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "json",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    document = json.loads(completed.stdout)
    rows = document.get("frames") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise QualityGateError("source decoded PTS are unavailable")
    times = []
    for row in rows:
        if not isinstance(row, dict):
            raise QualityGateError("invalid source decoded PTS")
        raw = row.get("pts_time")
        if isinstance(raw, bool):
            raise QualityGateError("invalid source decoded PTS")
        times.append(finite(float(raw), "source PTS"))
    if any(right <= left for left, right in pairwise(times)):
        raise QualityGateError("source decoded PTS must be strictly increasing")
    return times


def source_frame_png(clip: Path, index: int) -> bytes:
    """Select the decoded frame ordinal; never approximate frame index from average FPS."""
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-map",
            "0:v:0",
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-vsync",
            "0",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    if not completed.stdout:
        raise QualityGateError(f"cannot decode source frame {index}")
    return completed.stdout


def current_inputs(clip: Path, world: Path, cameras: Path, scale0: float) -> dict:
    scale = finite(scale0, "registrationScale")
    if scale <= 0:
        raise QualityGateError("registrationScale must be positive")
    return {
        "sourceSha256": file_sha256(clip),
        "worldSha256": file_sha256(world),
        "camerasSha256": file_sha256(cameras),
        "registrationScale": scale,
    }


def camera_rows(cameras: Path) -> dict[int, dict]:
    rows = read_document(cameras).get("cameras")
    if not isinstance(rows, list) or len(rows) < 3:
        raise QualityGateError("at least three recorded camera samples are required")
    result = {}
    previous_index, previous_time = -1, -1.0
    for row in rows:
        if not isinstance(row, dict):
            raise QualityGateError("invalid camera correspondence row")
        index = row.get("sourceIndex")
        time = finite(row.get("time"), "camera.time")
        if type(index) is not int or index <= previous_index or time < 0 or time <= previous_time:
            raise QualityGateError(
                "camera indices and times must be unique, nonnegative and ordered"
            )
        for key, size in (("camera_to_world", 4), ("source_intrinsics", 3)):
            matrix = row.get(key)
            if not isinstance(matrix, list) or len(matrix) != size:
                raise QualityGateError(f"camera {key} must be {size}x{size}")
            for line in matrix:
                if not isinstance(line, list) or len(line) != size:
                    raise QualityGateError(f"camera {key} must be {size}x{size}")
                for number in line:
                    finite(number, key)
        if row["camera_to_world"][3] != [0, 0, 0, 1]:
            raise QualityGateError("camera transform must have a homogeneous last row")
        intrinsics = row["source_intrinsics"]
        if intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0 or intrinsics[2] != [0, 0, 1]:
            raise QualityGateError("camera intrinsics are invalid")
        size = row.get("source_image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
        ):
            raise QualityGateError("camera image size must contain positive integer dimensions")
        result[index] = row
        previous_index, previous_time = index, time
    return result


def validate_samples(plan: dict, cameras: dict[int, dict]) -> list[dict]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("criteria") != list(CRITERIA):
        raise QualityGateError("plan requires the fixed static-world schema and three criteria")
    if finite(plan.get("passFraction"), "passFraction") != PASS_FRACTION:
        raise QualityGateError("the static-world pass fraction is fixed at 0.75")
    samples = plan.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise QualityGateError("plan requires at least three samples")
    ids, frames = set(), []
    for sample in samples:
        if not isinstance(sample, dict):
            raise QualityGateError("plan sample must be an object")
        sample_id = text(sample.get("id"), "sample.id")
        index = sample.get("sourceFrame")
        if sample_id in ids or type(index) is not int or index not in cameras:
            raise QualityGateError("duplicate sample id or missing recorded camera")
        if type(sample.get("critical")) is not bool or sample.get("view") != "source-camera":
            raise QualityGateError("sample requires a boolean critical flag and source-camera view")
        if (
            abs(finite(sample.get("timeSeconds"), "sample.timeSeconds") - cameras[index]["time"])
            > 1e-6
        ):
            raise QualityGateError("planned sample time does not match camera correspondence")
        ids.add(sample_id)
        frames.append(index)
    all_frames = list(cameras)
    required = {all_frames[0], all_frames[len(all_frames) // 2], all_frames[-1]}
    if frames != sorted(set(frames)) or not required.issubset(frames):
        raise QualityGateError("samples must be ordered, unique and cover beginning/middle/end")
    return samples


def assess(
    *,
    plan_path: Path,
    result_path: Path,
    evidence_root: Path,
    clip: Path,
    world: Path,
    cameras: Path,
    scale0: float,
) -> dict:
    """Reassess current bytes on every call, including resumes. Never mutate any input."""
    try:
        inspected = {plan_path: file_sha256(plan_path), result_path: file_sha256(result_path)}
        inputs = current_inputs(clip, world, cameras, scale0)
        plan, result = read_document(plan_path), read_document(result_path)
        if not isinstance(plan.get("inputs"), dict):
            raise QualityGateError("plan inputs must be an object")
        finite(plan["inputs"].get("registrationScale"), "plan registrationScale")
        if plan.get("inputs") != inputs:
            raise QualityGateError(
                "plan source/world/camera hashes or registration do not match current inputs"
            )
        samples = validate_samples(plan, camera_rows(cameras))
        if sha(result.get("planSha256"), "review.planSha256") != file_sha256(plan_path):
            raise QualityGateError("review plan bytes changed")
        report_path = evidence_file(evidence_root, plan.get("evidenceReport"), "evidence report")
        inspected[report_path] = plan["evidenceReport"]["sha256"]
        report = read_document(report_path)
        if report.get("schema") != EVIDENCE_SCHEMA or report.get("inputs") != inputs:
            raise QualityGateError("diagnostic report is not bound to current inputs")
        finite(report["inputs"].get("registrationScale"), "report registrationScale")
        code_hash = evidence_code_sha256()
        if report.get("evidenceCodeSha256") != code_hash:
            raise QualityGateError(
                "diagnostic capture/validation code changed; create fresh evidence"
            )
        if report.get("timingMethod") != TIMING_METHOD:
            raise QualityGateError("diagnostic report lacks exact decoded source PTS")
        rows = report.get("frames")
        if not isinstance(rows, list) or len(rows) != len(samples):
            raise QualityGateError("report must contain exactly the planned samples")
        source_times = source_timestamps(clip)
        origin = finite(report.get("sourceTimeOriginSeconds"), "source time origin")
        if abs(origin - source_times[0]) > 1e-6:
            raise QualityGateError("source time origin changed")
        pairs = {}
        planned_frames = [sample["sourceFrame"] for sample in samples]
        for row, sample in zip(rows, samples):
            if not isinstance(row, dict) or type(row.get("frame")) is not int:
                raise QualityGateError("invalid report sample")
            index = row["frame"]
            if index != sample["sourceFrame"] or index >= len(source_times):
                raise QualityGateError("report frame order does not match planned source frames")
            pts = finite(row.get("sourcePtsSeconds"), "source PTS")
            time = finite(row.get("timeSeconds"), "report time")
            if abs(pts - source_times[index]) > 1e-6 or abs(time - (pts - origin)) > 1e-6:
                raise QualityGateError("report time does not match actual source decoded PTS")
            if abs(time - sample["timeSeconds"]) > 1e-6:
                raise QualityGateError(
                    "camera/sample time does not match actual source decoded PTS"
                )
            for name in ("pair", "sourceImage", "renderImage"):
                path = evidence_file(evidence_root, row.get(name), f"frame {index} {name}")
                inspected[path] = row[name]["sha256"]
            if (
                hashlib.sha256(source_frame_png(clip, index)).hexdigest()
                != row["sourceImage"]["sha256"]
            ):
                raise QualityGateError(
                    "source evidence pixels do not match the decoded source frame"
                )
            pairs[index] = row
        requested = report.get("requestedFrames")
        if requested != planned_frames or any(type(frame) is not int for frame in requested):
            raise QualityGateError("diagnostic requested frames differ from the frozen plan")
        summary = evaluate_manual(plan, result, pairs)
        summary.update(
            inputs=inputs,
            planSha256=inspected[plan_path],
            resultSha256=inspected[result_path],
            evidenceReportSha256=inspected[report_path],
            evidenceCodeSha256=code_hash,
        )
        # Detect input changes during local inspection instead of issuing a stale acceptance.
        if current_inputs(clip, world, cameras, scale0) != inputs:
            raise QualityGateError("inputs changed during review")
        if any(file_sha256(path) != digest for path, digest in inspected.items()):
            raise QualityGateError("review or evidence changed during inspection")
        if evidence_code_sha256() != code_hash:
            raise QualityGateError("capture/validation code changed during inspection")
        return summary
    except (QualityGateError, OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        return {"status": "blocked", "errors": [str(exc)], "scope": SCOPE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "evidence-root", "clip", "world", "cameras"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--scale0", required=True, type=float)
    args = parser.parse_args()
    summary = assess(
        plan_path=args.plan,
        result_path=args.result,
        evidence_root=args.evidence_root,
        clip=args.clip,
        world=args.world,
        cameras=args.cameras,
        scale0=args.scale0,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return {"passed": 0, "failed": 3, "blocked": 4}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
