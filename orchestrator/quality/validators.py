"""Automatic no-model validators executed before an attempt can be promoted."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from orchestrator.workspace import is_output_file


def validate_stage_outputs(
    definition: dict[str, Any],
    attempt_root: Path,
    output_roles: dict[str, str],
) -> tuple[str, ...]:
    files_by_role: dict[str, list[Path]] = {}
    for pattern, role in output_roles.items():
        for path in attempt_root.glob(pattern):
            if is_output_file(path):
                files_by_role.setdefault(role, []).append(path)
    issues: list[str] = []
    for name, contract in definition.get("outputs", {}).items():
        role = contract["role"]
        files = files_by_role.get(role, [])
        if contract.get("required", True) and not files:
            issues.append(f"required output {name!r} ({role}) is missing")
            continue
        if not contract.get("multiple", False) and len(files) > 1:
            issues.append(f"output {name!r} produced {len(files)} files, expected one")
        media_type = contract.get("media_type", "")
        for path in files:
            if not path.stat().st_size:
                issues.append(f"{role}: {path.name} is empty")
                continue
            try:
                if media_type == "application/json":
                    _json(path)
                elif media_type.startswith("image/"):
                    with Image.open(path) as image:
                        image.verify()
                elif media_type.startswith(("video/", "audio/")):
                    _probe_media(path)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                issues.append(f"{role}: {path.name} is unreadable: {error}")
    for check in definition.get("quality", {}).get("automatic_checks", []):
        validator = CHECKS.get(check)
        if validator:
            try:
                validator(files_by_role)
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"{check}: {error}")
    return tuple(issues)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _one_json(files: dict[str, list[Path]], role: str) -> Any:
    values = files.get(role, [])
    if len(values) != 1:
        raise ValueError(f"expected one {role} artifact, found {len(values)}")
    return _json(values[0])


def _probe_media(path: Path) -> None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise ValueError(completed.stderr.decode(errors="replace")[-300:])
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("media duration is not positive and finite")


def _provider_operation(files: dict[str, list[Path]]) -> None:
    receipt = _one_json(files, "provider_operation")
    operation_id = receipt.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError("receipt has no provider operation ID")


def _scale(files: dict[str, list[Path]]) -> None:
    report = _one_json(files, "scale_fit")
    value = report.get("scale0")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("scale0 is not positive and finite")


def _placement(files: dict[str, list[Path]]) -> None:
    report = _one_json(files, "placement")
    required = ("metresPerWorldUnit", "samples", "residualCm")
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"placement is missing {', '.join(missing)}")


def _manifest(files: dict[str, list[Path]], role: str, schema: str) -> None:
    report = _one_json(files, role)
    if report.get("schema") != schema:
        raise ValueError(f"{role} schema is not {schema}")


CHECKS = {
    "provider_operation": _provider_operation,
    "scale_ratio": _scale,
    "scale_spread": _scale,
    "room_size": _placement,
    "contact_residual": _placement,
    "audio_timeline": lambda files: _manifest(files, "audio_manifest", "wander.audio/1"),
    "objects_manifest": lambda files: _manifest(files, "objects_manifest", "wander.objects/2"),
}
