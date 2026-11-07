"""Inspect and import manual static-world evidence from a settled run directory."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from orchestrator.api.viewer import asset_url, scene_manifest
from orchestrator.cli import RunContext, completion_problem, plan_for
from orchestrator.journal import ADMISSION_LOCK_FILE, MANUAL_WORLD_REVIEW, Journal
from orchestrator.run_policy import delivery_metadata
from orchestrator.steps import RunOptions
from orchestrator.workspace import atomic_json

# The existing standalone runner and importer use sibling script imports. Load only
# from this checkout, never from the repository field in a run manifest.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import run_clip  # noqa: E402
import world_quality_review as quality  # noqa: E402
from quality_gate import (  # noqa: E402
    ACKNOWLEDGEMENTS,
    CRITERIA,
    EVIDENCE_SCHEMA,
    RESULT_SCHEMA,
    SCOPE as SCOPE,
    file_sha256,
    read_document,
)


class ReviewUnavailable(ValueError):
    """The run does not have a retained diagnostic bundle."""


class ReviewConflict(ValueError):
    """A safe operator-facing reason the submission cannot proceed."""


def run_path(root: Path, path: Path) -> Path:
    """Reject escaping and symlinked paths, including symlinked parents."""
    relative = path.relative_to(root)
    if root.is_symlink() or ".." in relative.parts:
        raise ValueError("Review paths must stay inside the run")
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("Review paths cannot contain symlinks")
    if candidate.exists() and not (candidate.is_dir() or candidate.is_file()):
        raise ValueError("Review paths must be regular files or directories")
    return candidate


def read_run_context(root: Path, run_id: str) -> RunContext:
    described = read_document(run_path(root, root / "run.json"))
    if described.get("runId") != run_id or described.get("name") != root.name:
        raise ValueError("Run identity changed")
    options = RunOptions.model_validate(described["options"])
    if options.marble == "none" or described.get("retainedDelivery") is not None:
        raise ReviewUnavailable("This run does not have a static-world verification stage")
    source = run_path(root, Path(described["source"]))
    if source.parent != root or not source.name.startswith("source."):
        raise ValueError("The retained source does not belong to this run")
    return RunContext(
        run_id,
        root,
        root.name,
        source,
        SCRIPTS.parent,
        options.model_dump(),
        quality.sha(described.get("sourceSha256"), "submitted source"),
    )


class RetainedReviewPipeline(run_clip.Pipeline):
    """Use the runner's current-receipt checks without constructing a runnable pipeline."""

    def __init__(self, context: RunContext, state: dict):
        self.ctx = context.run_dir
        self.name = context.name
        self.marble = context.options["marble"]
        run_path(self.ctx, self.ctx / "tracks" / "tracks.json")
        self.stages = plan_for(context)
        self.state = run_clip.State(self.ctx / "state.json")
        self.state.data = state
        # API uploads have fixed runner defaults and do not set these stage flags.
        self.a = SimpleNamespace(
            allow_bad_size=False,
            ruler_tol=run_clip.DEFAULT_RULER_TOLERANCE,
            ruler_vlm=False,
            allow_weak_anchor=False,
        )
        self.clip = self.review_path(
            self.review_public_dir() / "clips" / self.name / context.source.name
        )
        if (
            self.input_hash(context.source) != context.source_sha256
            or self.input_hash(self.clip) != context.source_sha256
        ):
            raise ValueError("Submitted source or its exported copy changed")

    def review_public_dir(self) -> Path:
        return self.ctx / "public"

    def review_path(self, path: Path) -> Path:
        return run_path(self.ctx, path)

    def placement(self) -> dict:
        scale = self.state.data["stages"]["_scale"]["scale0"]
        ply = self.review_path(
            self.review_public_dir() / "worlds" / f"{self.name}-4d" / "person" / "frame_000.ply"
        )
        _, feet = run_clip.body_stats(ply)
        return {"floor": feet * scale}


@dataclass(frozen=True)
class Diagnostic:
    directory: Path
    plan: Path
    plan_sha: str
    samples: list[dict]
    inputs: dict

    @property
    def identifier(self) -> str:
        return self.directory.name

    def pair_path(self, source_frame: int) -> Path:
        return self.directory / f"pair-f{source_frame}.png"

    def template(self) -> dict:
        return {
            "schema": RESULT_SCHEMA,
            "planSha256": self.plan_sha,
            "judgeKind": "manual",
            "reviewer": "",
            "acknowledgements": dict.fromkeys(ACKNOWLEDGEMENTS, False),
            "samples": [
                {
                    "id": sample["id"],
                    "pairSha256": sample["pairSha256"],
                    "criteria": {
                        key: {"status": "unknown", "criticalFailure": False, "reason": ""}
                        for key in CRITERIA
                    },
                }
                for sample in self.samples
            ],
        }


def read_state(root: Path) -> dict:
    path = run_path(root, root / "state.json")
    if not path.is_file():
        raise ReviewUnavailable("This run does not have a retained diagnostic")
    state = read_document(path)
    if not isinstance(state.get("stages"), dict) or any(
        not isinstance(record, dict) for record in state["stages"].values()
    ):
        raise ValueError("Invalid run stage state")
    return state


def read_diagnostic(root: Path, state: dict, *, check_images=True) -> Diagnostic:
    verification = state["stages"].get("_verify") or {}
    pointer = verification.get("diagnosticReport")
    if not pointer:
        raise ReviewUnavailable("This run does not have a retained diagnostic")
    report_path = run_path(root, Path(pointer))
    directory = report_path.parent
    if (
        report_path.name != "report.json"
        or directory.parent != root / "verify"
        or not re.fullmatch(r"diagnostic-[A-Za-z0-9_-]{1,64}", directory.name)
    ):
        raise ValueError("Invalid retained diagnostic directory")
    plan_path = run_path(root, directory / "plan.json")
    plan_bytes = plan_path.read_bytes()
    plan = read_document(plan_path)
    report = read_document(report_path)
    if (
        plan.get("evidenceReport") != {"path": report_path.name, "sha256": file_sha256(report_path)}
        or report.get("schema") != EVIDENCE_SCHEMA
        or report.get("evidenceCodeSha256") != quality.evidence_code_sha256()
        or plan.get("inputs") != report.get("inputs")
        or not isinstance(plan.get("samples"), list)
        or not isinstance(report.get("frames"), list)
        or len(plan["samples"]) != len(report["frames"])
    ):
        raise ValueError("Retained diagnostic evidence is incomplete or changed")
    # validate_samples also checks ordered/unique camera correspondence during import.
    samples = []
    for sample, frame in zip(plan["samples"], report["frames"], strict=True):
        if not isinstance(sample, dict) or not isinstance(frame, dict):
            raise ValueError("Invalid retained diagnostic sample")
        index = sample.get("sourceFrame")
        if type(index) is not int or index < 0 or frame.get("frame") != index:
            raise ValueError("Invalid retained diagnostic frame")
        pair_sha = None
        for field, prefix in (
            ("pair", "pair"),
            ("sourceImage", "source"),
            ("renderImage", "render"),
        ):
            path = run_path(root, directory / f"{prefix}-f{index}.png")
            reference = frame.get(field)
            if not isinstance(reference, dict) or reference.get("path") != path.name:
                raise ValueError("Diagnostic images must be captured run files")
            digest = quality.sha(reference.get("sha256"), field)
            if check_images and file_sha256(path) != digest:
                raise ValueError("Retained diagnostic image bytes changed")
            if field == "pair":
                pair_sha = digest
        samples.append(
            {
                "id": quality.text(sample.get("id"), "sample.id"),
                "sourceFrame": index,
                "timeSeconds": quality.finite(sample.get("timeSeconds"), "sample time"),
                "critical": sample.get("critical"),
                "pairSha256": pair_sha,
            }
        )
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    if file_sha256(plan_path) != plan_sha:
        raise ValueError("Diagnostic plan changed while reading")
    return Diagnostic(directory, plan_path, plan_sha, samples, plan["inputs"])


def check_current(context: RunContext, state: dict, diagnostic: Diagnostic) -> dict:
    runner = RetainedReviewPipeline(context, state)
    try:
        world, cameras, scale = runner.verification_inputs()
    except run_clip.QualityStop as error:
        raise ReviewConflict(str(error)) from error
    inputs = {"clip": runner.clip, "world": world, "cameras": cameras, "scale0": scale}
    if diagnostic.inputs != quality.current_inputs(**inputs):
        raise ValueError("Diagnostic source, world, camera or scale changed")
    quality.validate_samples(read_document(diagnostic.plan), quality.camera_rows(cameras))
    return inputs


def review_hold(journal: Journal, state: dict) -> str | None:
    if control := journal.control_status():
        return f"The run is {control}"
    if journal.active_attempts() or any(
        isinstance(record, dict) and record.get("status") == "running"
        for record in state["stages"].values()
    ):
        return "Wait for active stage work to settle"
    if journal.unanswered():
        return "Resolve the active operator question before submitting a review"
    if journal.finished() != "failed":
        return "A review can be submitted after the run finishes with a blocked or failed review"
    return None


def viewer_binding_problem(context: RunContext, inputs: dict) -> str | None:
    """Check the delivered scene, including presentation overrides, against the review."""
    public = context.run_dir / "public"
    package = public / "worlds" / f"{context.name}-4d"
    run_path(context.run_dir, package / "scene.json")
    manifest = scene_manifest(context.run_dir)
    if manifest is None:
        return "The packaged viewer scene is incomplete"
    params = manifest["params"]
    world = asset_url(context.run_id, inputs["world"].relative_to(public).as_posix())
    people = {
        asset_url(context.run_id, (package / name).relative_to(public).as_posix())
        for name in ("people.json", "person/sequence.json")
    }
    if (
        params.get("world") != world
        or params.get("person") not in people
        or inputs["cameras"] != package / "cameras.json"
    ):
        return "The viewer world or packaged camera source differs from the reviewed inputs"
    if "height" in params or "scale" not in params or float(params["scale"]) != inputs["scale0"]:
        return "The viewer registration scale differs from the reviewed scale"
    return None


def finish_reviewed_run(context, journal, state, diagnostic, inputs, receipt, assessment, attempt):
    """Complete only while the package and the saved assessment still describe the same bytes."""
    root = context.run_dir
    with run_path(root, root / "state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        problem = completion_problem(context, journal) or viewer_binding_problem(context, inputs)
        observed = read_state(root)
        try:
            current = read_diagnostic(root, observed)
            if (
                observed != state
                or current != diagnostic
                or read_run_context(root, context.run_id) != context
                or file_sha256(run_path(root, receipt)) != assessment["resultSha256"]
                or check_current(context, observed, current) != inputs
            ):
                raise ValueError("Review bindings changed")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            reason = "Review inputs changed before run completion"
            if observed == state:
                state["stages"]["_verify"].update(status="blocked", passed=False, errors=[reason])
                state["stages"]["verify"] = {"status": "blocked", "error": reason}
                atomic_json(root / "state.json", state)
            journal.append(
                "step.finished", step="verify", attempt=attempt, status="blocked", error=reason
            )
            return reason, True
        if problem is None:
            journal.append(
                "run.finished",
                status="succeeded",
                summary="Manual static-world review and existing completion checks passed.",
            )
        return problem, False


def import_review(
    context: RunContext, journal: Journal, diagnostic_id: str, plan_sha: str, result: dict
) -> dict:
    root = context.run_dir
    for name in (ADMISSION_LOCK_FILE, "paused", "state.lock"):
        run_path(root, root / name)
    with journal.admission_lock():
        if read_run_context(root, context.run_id) != context:
            raise ReviewConflict("The run changed before review submission")
        state = read_state(root)
        if reason := review_hold(journal, state):
            raise ReviewConflict(reason)
        diagnostic = read_diagnostic(root, state)
        if (
            diagnostic.identifier != diagnostic_id
            or diagnostic.plan_sha != plan_sha
            or result.get("planSha256") != plan_sha
        ):
            raise ReviewConflict("The submitted review does not match the current diagnostic plan")
        inputs = check_current(context, state, diagnostic)
        identifier = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix=".review-", dir=diagnostic.directory) as temporary:
            candidate = Path(temporary) / "review.json"
            contents = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
            if len(contents) > 256_000:
                raise ReviewConflict("The manual review exceeds 256,000 bytes")
            candidate.write_bytes(contents + b"\n")
            assessment = quality.assess(
                plan_path=diagnostic.plan,
                result_path=candidate,
                evidence_root=diagnostic.directory,
                **inputs,
            )
            outcome = assessment["status"]
            response = {
                "reviewStatus": outcome,
                "runStatus": journal.finished(),
                "errors": assessment.get("errors", []),
                "completionReason": None,
            }
            if assessment.get("planSha256") != diagnostic.plan_sha:
                return response  # Malformed or stale evidence does not replace a recorded verdict.
            # Recheck shared placement/anchor bindings and the selected plan after source decoding.
            current_state = read_state(root)
            current = read_diagnostic(root, current_state)
            if (
                current_state != state
                or current != diagnostic
                or read_run_context(root, context.run_id) != context
            ):
                raise ValueError("Review inputs changed during validation")
            check_current(context, current_state, current)
            receipt = diagnostic.directory / f"manual-{identifier}.json"
            passed = outcome == "passed"
            stage_status = "ok" if passed else outcome
            error = "; ".join(response["errors"]) or None
            state["stages"]["_verify"] = {
                **assessment,
                "passed": passed,
                "diagnosticReport": str(diagnostic.directory / "report.json"),
                "manualResult": str(receipt),
            }
            state["stages"]["verify"] = {"status": stage_status, "error": error}
            # Direct operator commands also serialize their state writes on this lock.
            with run_path(root, root / "state.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if read_state(root) != current_state:
                    raise ReviewConflict("The run changed during review validation")
                candidate.rename(receipt)
                atomic_json(root / "state.json", state)

        number = journal.attempt_number("verify")
        attempt = f"{context.run_id}:verify:{number}"
        journal.append(
            "step.finished",
            origin=MANUAL_WORLD_REVIEW,
            step="verify",
            attempt=attempt,
            number=number,
            command=[],
            parameters={
                "manualReview": True,
                "diagnosticId": diagnostic_id,
                "planSha256": plan_sha,
                "resultSha256": assessment["resultSha256"],
            },
            paid=False,
            status="succeeded" if passed else outcome,
            error=error,
        )
        if passed:
            problem, stale = finish_reviewed_run(
                context, journal, state, diagnostic, inputs, receipt, assessment, attempt
            )
            response["completionReason"] = problem
            if stale:
                outcome = response["reviewStatus"] = "blocked"
                response["errors"] = [problem]
            if problem is None:
                response["runStatus"] = "succeeded"
                return response
        summary = (
            error or response["completionReason"] or "Manual static-world review remains blocked."
        )
        delivery = delivery_metadata(
            context, journal, summary, quality_status="failed" if outcome == "failed" else "blocked"
        )
        journal.append("run.finished", status="failed", summary=summary, delivery=delivery)
        return response
