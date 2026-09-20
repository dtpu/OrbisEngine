"""Persistent allowance for runner-controlled paid Modal executions.

The three-execution ceiling is per original source SHA and logical stage, across candidates.
Inpainting operations share one allowance; individual LHM track suffixes have separate allowances.
This does not count direct experiment scripts, SSH fine-tuning, OpenAI, or Marble generation.
The orchestrator must account for those separately. Costs here are evidence, not dollar budgets.

This guard cannot discover executions made before it was installed or launches made directly from
worker scripts. Before future paid work on a source/stage with such history, review its receipts and
represent those attempts in the default ledger; an empty new ledger is not evidence of zero history.
Verified byte-distinct encodings of the same source may be recorded in the optional top-level
`sourceAliases` object as `{aliasSha256: canonicalSha256}`. Aliases resolve before claims and caps.

Reconcile an interrupted claim without submitting any work:
  uv run --locked python scripts/stage_attempts.py --ledger PATH --reconcile ID \\
      --status failed --evidence PATH --reason 'Inspected provider result and saved artifacts'
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "wander.pipeline-attempts/1"
MAX_EXECUTIONS = 3
INITIAL_HYPOTHESIS = "Initial paid execution using the recorded source, parameters, and code."
STATUSES = {"pending", "unknown", "completed", "failed"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_hypothesis(value):
    return " ".join(value.split()).casefold()


def logical_stage(operation):
    return "inpainting" if operation in {"clean", "clean_first", "clean_multi"} else operation


CODE_FILES = {
    "worker/modal_clean_video.py": ("worker/wander_worker/masks.py",),
    "worker/modal_motion.py": (
        "worker/stages/dense_pi3x.py",
        "worker/wander_worker/masks.py",
        "worker/wander_worker/ply.py",
    ),
    "worker/modal_lhm.py": (
        "worker/stages/lhm_person.py",
        "worker/stages/lhm_animate.py",
        "worker/stages/lhm_capacity.py",
    ),
    "worker/modal_multiperson.py::main": ("worker/stages/track_people.py",),
    "worker/modal_multiperson.py::animate": (
        "worker/stages/lhm_execution.py",
        "worker/stages/lhm_recovery.py",
        "worker/stages/lhm_person.py",
        "worker/stages/lhm_animate.py",
    ),
    "worker/modal_image_to_3d.py": (),
}
OUTPUT_FLAGS = {"--out", "--out-dir", "--frames-out", "--frame0", "--masks", "--report"}
BOOL_FLAGS = {
    "--allow-person-gaps",
    "--allow-missing-person-reference",
    "--moved-mask",
    "--extend-mask-to-bottom",
    "--refine-first",
    "--larger-model",
}
NUMBER_FLAGS = {
    "--fps",
    "--width",
    "--height",
    "--dilate",
    "--bottom-extra",
    "--lama-px",
    "--det-thresh",
    "--overlap",
    "--batch",
    "--track-id",
    "--fixed-world-scale",
    "--execution-timeout",
    "--refine-strength",
}
TEXT_FLAGS = {"--experiment", "--mask-backend", "--alignment", "--depth-roi", "--refine-prompt"}
INPUT_FLAGS = {
    "--image",
    "--cameras",
    "--depth-reference",
    "--prepared",
    "--canonical",
    "--track-dir",
    "--fixed-inputs",
    "--seed",
}
IGNORED_JSON_KEYS = {
    "path",
    "sourcePath",
    "derivedPath",
    "clip",
    "video",
    "out",
    "output",
    "destination",
    "candidate",
    "name",
    "log",
    "seconds",
    "elapsed",
    "estimatedComputeUSD",
    "startedAtEpoch",
    "finishedAtEpoch",
}


def _semantic_json(value):
    if isinstance(value, dict):
        return {
            key: _semantic_json(item) for key, item in value.items() if key not in IGNORED_JSON_KEYS
        }
    if isinstance(value, list):
        return [_semantic_json(item) for item in value]
    return value


def input_identity(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Paid-stage input is missing: {path}")
    if path.suffix == ".json":
        return {"semanticSha256": digest(_semantic_json(json.loads(path.read_text())))}
    return {"sha256": sha256_file(path)}


def command_identity(command, root, source_selection=None):
    """Allowlisted direct worker commands: paths never establish a changed hypothesis."""
    root = Path(root)
    if len(command) < 3 or command[1] != "run" or command[2] not in CODE_FILES:
        raise ValueError("Unregistered paid worker command; add its identity policy before launch")
    entry = command[2]
    code = {name: sha256_file(root / name) for name in (entry.split("::")[0], *CODE_FILES[entry])}
    parameters = {"worker": entry, "options": {}, "inputs": {}, "sourceSelection": source_selection}
    outputs = []
    index = 3
    while index < len(command):
        flag = command[index]
        if flag in BOOL_FLAGS:
            parameters["options"][flag] = True
            index += 1
            continue
        if index + 1 >= len(command):
            raise ValueError(f"Missing paid command option value: {flag}")
        value = command[index + 1]
        index += 2
        if flag in OUTPUT_FLAGS:
            outputs.append(Path(value))
        elif flag in {"--clip", "--video", "--note"}:
            continue
        elif flag in NUMBER_FLAGS:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Paid option must be finite")
            if flag == "--fixed-world-scale" and not 0 < number < 10:
                raise ValueError("Fixed world scale must be strictly between 0 and 10")
            if flag == "--execution-timeout" and (
                not number.is_integer() or not 0 <= number <= 3600
            ):
                raise ValueError("Execution timeout must be an integer from 0 to 3600")
            parameters["options"][flag] = number
        elif flag in TEXT_FLAGS:
            parameters["options"][flag] = value
        elif flag == "--only":
            parameters["options"][flag] = [int(part.strip()) for part in value.split(",")]
        elif flag in INPUT_FLAGS:
            path = Path(value)
            if flag == "--prepared":
                parameters["inputs"][flag] = {
                    name: input_identity(path / name)
                    for name in ("source.png", "mask.png", "prepared.json")
                }
            elif flag == "--canonical":
                reference = json.loads((path / "result.json").read_text())["prepared"]
                parameters["inputs"][flag] = {
                    "state": input_identity(path / "canonical-state.pt"),
                    "reference": {
                        key: reference.get(key)
                        for key in ("sourceSha256", "sourceWidth", "sourceHeight", "time")
                    },
                }
            elif flag in {"--track-dir", "--seed", "--fixed-inputs"}:
                names = (
                    ("source-pose.pt", "source-pose.json", "head-input.png")
                    if flag == "--fixed-inputs"
                    else ("source-poses.pt", "motion.json")
                )
                parameters["inputs"][flag] = {name: input_identity(path / name) for name in names}
            else:
                parameters["inputs"][flag] = input_identity(path)
        else:
            raise ValueError(f"Unregistered paid option {flag}; define its identity before launch")
    # Single-person animation implicitly reads this companion file.
    if entry == "worker/modal_lhm.py" and "--cameras" in command:
        cameras = Path(command[command.index("--cameras") + 1])
        parameters["inputs"]["depthReference"] = input_identity(cameras.parent / "frame_000.ply")
    return parameters, code, outputs


class StageAttempts:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text())
                except (OSError, ValueError):
                    raise ValueError(
                        "Paid-stage ledger is unreadable; no execution authorized"
                    ) from None
            else:
                data = {"schema": SCHEMA, "attempts": []}
            self._validate(data)
            yield data
            self._validate(data)
            temporary = self.path.with_name(self.path.name + f".{uuid.uuid4().hex}.tmp")
            with temporary.open("x") as stream:
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _validate(data):
        if (
            not isinstance(data, dict)
            or data.get("schema") != SCHEMA
            or not isinstance(data.get("attempts"), list)
        ):
            raise ValueError("Invalid paid-stage ledger")
        aliases = data.get("sourceAliases", {})
        if not isinstance(aliases, dict) or any(
            not isinstance(alias, str)
            or not isinstance(canonical, str)
            or not re.fullmatch(r"[0-9a-f]{64}", alias)
            or not re.fullmatch(r"[0-9a-f]{64}", canonical)
            or alias == canonical
            or canonical in aliases
            for alias, canonical in aliases.items()
        ):
            raise ValueError("Invalid paid-stage source aliases")
        identifiers, counts, direct_claim_sources = set(), {}, set()
        for attempt in data["attempts"]:
            if not isinstance(attempt, dict):
                raise ValueError("Malformed saved paid-stage attempt")  # noqa: TRY004 - malformed persisted data
            claim, events = attempt.get("claim"), attempt.get("events")
            if (
                not isinstance(claim, dict)
                or not isinstance(events, list)
                or not events
                or not isinstance(attempt.get("id"), str)
                or attempt["id"] in identifiers
                or not re.fullmatch(r"[0-9a-f]{64}", str(claim.get("sourceSha256", "")))
                or (
                    claim.get("submittedSourceSha256") is not None
                    and not re.fullmatch(
                        r"[0-9a-f]{64}", str(claim.get("submittedSourceSha256", ""))
                    )
                )
                or not isinstance(claim.get("stage"), str)
                or not claim["stage"]
                or not isinstance(claim.get("hypothesis"), str)
                or not claim["hypothesis"].strip()
                or claim.get("fingerprint")
                != digest({"parameters": claim.get("parameters"), "code": claim.get("codeVersion")})
                or any(
                    not isinstance(event, dict) or event.get("status") not in STATUSES
                    for event in events
                )
                or events[0].get("status") != "pending"
            ):
                raise ValueError("Malformed saved paid-stage accounting")
            submitted = claim.get("submittedSourceSha256", claim["sourceSha256"])
            if submitted != claim["sourceSha256"]:
                if aliases.get(submitted) != claim["sourceSha256"]:
                    raise ValueError("Saved paid-stage claim has an invalid source alias")
            else:
                direct_claim_sources.add(submitted)
            identifiers.add(attempt["id"])
            key = claim["sourceSha256"], claim["stage"]
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > MAX_EXECUTIONS or claim.get("number") != counts[key]:
                raise ValueError("Invalid saved paid-stage execution count")
        if direct_claim_sources.intersection(aliases):
            raise ValueError("Cannot alias a source hash that already owns paid-stage claims")

    def begin(
        self,
        source_sha256,
        operation,
        *,
        parameters,
        code_version,
        hypothesis,
        candidate,
        log,
        results,
    ):
        if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError("Original source SHA-256 is required for paid execution")
        if not isinstance(operation, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", operation):
            raise ValueError("Concrete paid stage is required")
        if (
            not isinstance(parameters, dict)
            or not isinstance(code_version, dict)
            or not code_version
        ):
            raise ValueError("Paid execution requires parameter and code identity")
        submitted_source = source_sha256
        stage = logical_stage(operation)
        explicit = isinstance(hypothesis, str) and bool(hypothesis.strip())
        hypothesis = (
            hypothesis.strip() if explicit else f"{INITIAL_HYPOTHESIS} Workflow: {operation}."
        )
        fingerprint = digest({"parameters": parameters, "code": code_version})
        with self._locked() as data:
            source_sha256 = data.get("sourceAliases", {}).get(submitted_source, submitted_source)
            prior = [
                item
                for item in data["attempts"]
                if item["claim"]["sourceSha256"] == source_sha256
                and item["claim"]["stage"] == stage
            ]
            if len(prior) >= MAX_EXECUTIONS:
                raise ValueError(
                    "Paid-stage execution allowance exhausted (three total per original source/stage)"
                )
            if any(item["events"][-1]["status"] in {"pending", "unknown"} for item in prior):
                raise ValueError(
                    "Pending/unknown paid execution requires evidence reconciliation before retry"
                )
            same_operation = [item for item in prior if item["claim"]["operation"] == operation]
            if same_operation and not explicit:
                raise ValueError("Paid retry requires an explicit new --stage-hypothesis")
            if same_operation and any(
                normalized_hypothesis(item["claim"]["hypothesis"])
                == normalized_hypothesis(hypothesis)
                for item in prior
            ):
                raise ValueError("Paid retry requires a new hypothesis, not a renamed candidate")
            if any(item["claim"]["fingerprint"] == fingerprint for item in prior):
                raise ValueError(
                    "Paid retry requires changed relevant parameters or code; output paths do not qualify"
                )
            attempt = {
                "id": uuid.uuid4().hex,
                "claim": {
                    "sourceSha256": source_sha256,
                    "submittedSourceSha256": submitted_source,
                    "stage": stage,
                    "operation": operation,
                    "number": len(prior) + 1,
                    "candidate": str(candidate),
                    "parameters": parameters,
                    "codeVersion": code_version,
                    "fingerprint": fingerprint,
                    "hypothesis": hypothesis,
                    "hypothesisExplicit": explicit,
                    "log": str(Path(log).resolve()),
                    "logStartByte": Path(log).stat().st_size if Path(log).is_file() else 0,
                    "results": [str(Path(path).resolve()) for path in results],
                    "startedAtEpoch": time.time(),
                },
                "events": [
                    {
                        "status": "pending",
                        "atEpoch": time.time(),
                        "reason": "Durable claim before paid launch",
                    }
                ],
            }
            data["attempts"].append(attempt)
        return attempt

    def finish(self, attempt_id, *, status, evidence, costs=None, reason=""):
        if status not in {"completed", "failed", "unknown"}:
            raise ValueError("Invalid terminal/unknown execution status")
        costs = costs or {}
        if not isinstance(costs, dict) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in costs.values()
        ):
            raise ValueError("Execution costs must be finite nonnegative estimates")
        with self._locked() as data:
            attempt = next((item for item in data["attempts"] if item["id"] == attempt_id), None)
            if attempt is None or attempt["events"][-1]["status"] != "pending":
                raise ValueError("Paid execution is missing or already recorded")
            log = Path(attempt["claim"]["log"])
            attempt["events"].append(
                {
                    "status": status,
                    "atEpoch": time.time(),
                    "reason": reason,
                    "evidence": [str(Path(path).resolve()) for path in evidence],
                    "costs": costs,
                    "logEndByte": log.stat().st_size if log.is_file() else None,
                }
            )

    def reconcile(self, attempt_id, *, status, evidence, reason):
        if (
            status not in {"completed", "failed"}
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("Reconciliation requires an observed terminal status and reason")
        evidence = Path(evidence)
        if not evidence.is_file() or evidence.stat().st_size == 0:
            raise ValueError("Reconciliation requires readable, nonempty saved evidence")
        evidence_sha = sha256_file(evidence)
        with self._locked() as data:
            attempt = next((item for item in data["attempts"] if item["id"] == attempt_id), None)
            if attempt is None or attempt["events"][-1]["status"] not in {"pending", "unknown"}:
                raise ValueError("Only a pending/unknown execution can be reconciled")
            attempt["events"].append(
                {
                    "status": status,
                    "atEpoch": time.time(),
                    "reason": reason.strip(),
                    "reconciliation": True,
                    "evidence": str(evidence.resolve()),
                    "evidenceSha256": evidence_sha,
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--reconcile", required=True, metavar="ATTEMPT_ID")
    parser.add_argument("--status", required=True, choices=("completed", "failed"))
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    StageAttempts(args.ledger).reconcile(
        args.reconcile, status=args.status, evidence=args.evidence, reason=args.reason
    )
    print("Reconciliation recorded; execution count unchanged. No provider work submitted.")


if __name__ == "__main__":
    main()
