"""Manual-only static-world review contract; no provider or execution-ledger access.

Adapted from Austin Jian's 52f211b and Daniel Pu's f95daa0/60e61f7 contracts.
Their incompatible quality-attempt ledgers and uncalibrated LLM paths are not imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

PLAN_SCHEMA = "wander.static-world-plan/1"
RESULT_SCHEMA = "wander.static-world-review/1"
EVIDENCE_SCHEMA = "wander.static-world-evidence/1"
CRITERIA = ("room_layout", "scene_coverage", "appearance")
PASS_FRACTION = 0.75
SCOPE = "sampled static-world appearance only; no geometric, people, object or walking acceptance"
ACKNOWLEDGEMENTS = {
    "estimatedCameraCorrespondence": True,
    "unobservedGeometryNotVerified": True,
    "staticWorldOnly": True,
}


class QualityGateError(ValueError):
    """Evidence is missing, changed or outside the supported acceptance contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_document(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise QualityGateError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise QualityGateError(f"document must be an object: {path.name}")
    return value


def text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualityGateError(f"{name} must be nonempty text")
    return value.strip()


def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise QualityGateError(f"{name} must be finite")
    return float(value)


def sha(value, name: str) -> str:
    value = text(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise QualityGateError(f"{name} must be a lowercase SHA-256")
    return value


def evidence_file(root: Path, reference: dict, name: str) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise QualityGateError(f"{name} requires path and sha256")
    value = text(reference["path"], f"{name}.path")
    relative = PurePosixPath(value)
    if "\\" in value or "\x00" in value or relative.is_absolute() or ".." in relative.parts:
        raise QualityGateError(f"{name} must be a safe relative path")
    root = root.resolve()
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root) or not path.is_file() or not path.stat().st_size:
        raise QualityGateError(f"{name} is missing, empty or outside evidence root")
    if file_sha256(path) != sha(reference["sha256"], f"{name}.sha256"):
        raise QualityGateError(f"{name} evidence bytes changed")
    return path


def evaluate_manual(plan: dict, result: dict, pairs: dict[int, dict]) -> dict:
    """Require every planned judgment; unknown blocks and critical failures veto averages."""
    if result.get("schema") != RESULT_SCHEMA:
        raise QualityGateError("unsupported static-world review schema")
    if result.get("judgeKind") != "manual":
        raise QualityGateError(
            "LLM acceptance is disabled: frozen calibration, independent holdouts and request "
            "accounting are not reconciled"
        )
    reviewer = text(result.get("reviewer"), "reviewer")
    if result.get("acknowledgements") != ACKNOWLEDGEMENTS or any(
        type(result["acknowledgements"].get(key)) is not bool for key in ACKNOWLEDGEMENTS
    ):
        raise QualityGateError(
            "review must explicitly acknowledge the static-world evidence bounds"
        )
    judgments = result.get("samples")
    if not isinstance(judgments, list) or len(judgments) != len(plan["samples"]):
        raise QualityGateError("review must contain exactly the planned samples")
    expected = {sample["id"]: sample for sample in plan["samples"]}
    seen = set()
    criteria = {key: {"passed": 0, "total": 0, "criticalFailures": []} for key in CRITERIA}
    failures, unknown = [], []
    for sample in judgments:
        if not isinstance(sample, dict):
            raise QualityGateError("review sample must be an object")
        sample_id = text(sample.get("id"), "review.sample.id")
        if sample_id in seen or sample_id not in expected:
            raise QualityGateError("duplicate or unplanned review sample")
        seen.add(sample_id)
        planned = expected[sample_id]
        pair = pairs[planned["sourceFrame"]]["pair"]
        if sample.get("pairSha256") != pair["sha256"]:
            raise QualityGateError("review sample does not match the bound source/render pair")
        decisions = sample.get("criteria")
        if not isinstance(decisions, dict) or set(decisions) != set(CRITERIA):
            raise QualityGateError("each sample requires all three static-world criteria")
        for key, judgment in decisions.items():
            if not isinstance(judgment, dict):
                raise QualityGateError("criterion judgment must be an object")
            status = judgment.get("status")
            if not isinstance(status, str) or status not in {"pass", "fail", "unknown"}:
                raise QualityGateError("criterion status must be pass, fail or unknown")
            if type(judgment.get("criticalFailure")) is not bool:
                raise QualityGateError("criticalFailure must be boolean")
            reason = text(judgment.get("reason"), "criterion.reason")
            item = criteria[key]
            item["total"] += 1
            item["passed"] += status == "pass"
            if status == "unknown":
                unknown.append(f"{sample_id}/{key}: {reason}")
            if judgment["criticalFailure"] or (planned["critical"] and status == "fail"):
                failure = f"{sample_id}/{key}: {reason}"
                item["criticalFailures"].append(failure)
                failures.append(failure)
    for key, item in criteria.items():
        item["passingFraction"] = item["passed"] / item["total"]
        if item["passingFraction"] < PASS_FRACTION:
            failures.append(f"{key}: fewer than 75% of planned samples pass")
    return {
        "status": "blocked" if unknown else "failed" if failures else "passed",
        "errors": unknown or failures,
        "reviewer": reviewer,
        "criteria": criteria,
        "scope": SCOPE,
    }
