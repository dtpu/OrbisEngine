"""Boundaries for an explicitly unattended run, independent of the agent's wording.

A preview is delivery evidence, never acceptance. The existing terminal failed status keeps
the supervisor and database settled while the journal retains why acceptance stopped.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from orchestrator.journal import Journal
from orchestrator.steps import STEPS, base_step, stage_graph


def hands_off(options: dict[str, Any]) -> bool:
    return options.get("hands_off") is True


def stage_records(run_dir) -> dict:
    try:
        state = json.loads((run_dir / "state.json").read_text())
        stages = state.get("stages") if isinstance(state, dict) else None
        return stages if isinstance(stages, dict) else {}
    except (OSError, ValueError):
        return {}


def admission_problem(context, step: str, extra: list[str], journal: Journal, attempt=None):
    """Limit hands-off commands to the submitted plan and a single paid launch per stage.

    A detached child may execute its already admitted attempt once. Recovery which could
    resubmit is deliberately blocked; an operator can inspect the retained provider receipt.
    """
    if not hands_off(context.options):
        return None
    if extra:
        # Upload runs use their submitted options and fixed stage defaults. Arbitrary runner
        # flags can change the source, stages, model, budget or quality tolerances, including
        # via argparse abbreviations, so do not forward them in this mode.
        return "Hands-off runs use submitted options and fixed stage defaults; extra stage flags are disabled."
    described = STEPS[base_step(step)]
    if described.paid and not attempt:
        prior = any(
            entry.kind == "step.started" and entry.data.get("step") == step
            for entry in journal.entries()
        )
        record = stage_records(context.run_dir).get(step)
        recorded = isinstance(record, dict) and record.get("status") not in {None, "pending"}
        if prior or recorded:
            return (
                f"Hands-off paid step {step} already has an attempt or retained result. "
                "Automatic paid retries are disabled; retain its outputs and provider receipt."
            )
        if lost := lost_package_prerequisite(context, journal):
            return (
                f"Hands-off paid step {step} was not started: required paid step {lost} cannot "
                "run again, so no packaged preview can follow and further paid work is wasted."
            )
    return None


def lost_package_prerequisite(context, journal: Journal) -> str | None:
    """A paid prerequisite of the package that failed or was blocked, if there is one.

    Hands-off runs never repeat a paid step, so such a failure is final: the package, and with
    it any preview, can no longer be produced. Unpaid steps can still be rerun and do not count.
    """
    stages, deps = stage_graph(context.options)
    package = "package_people" if "package_people" in stages else "package"
    required, pending = set(), [package]
    while pending:
        name = pending.pop()
        if name not in required:
            required.add(name)
            pending.extend(deps.get(name, []))
    outcomes = {
        entry.data.get("step"): entry.data.get("status")
        for entry in journal.entries()
        if entry.kind == "step.finished"
    }
    for name, record in stage_records(context.run_dir).items():
        if isinstance(record, dict) and record.get("status") == "failed":
            outcomes.setdefault(name, "failed")
    for name in stages:
        if (
            name in required
            and STEPS[base_step(name)].paid
            and outcomes.get(name) in {"failed", "blocked"}
        ):
            return f"{name} ({outcomes[name]})"
    return None


def delivery_metadata(context, journal: Journal, reason: str, quality_status="blocked") -> dict:
    """Describe a viewable candidate separately from unresolved or failed acceptance."""
    records = stage_records(context.run_dir)
    latest = {}
    for entry in journal.entries():
        if entry.kind == "step.finished":
            latest[entry.data.get("step")] = entry.data
    failures = {
        step: record.get("error") or "stage failed"
        for step, record in {**latest, **records}.items()
        if isinstance(step, str)
        and not step.startswith("_")
        and isinstance(record, dict)
        and record.get("status") == "failed"
    }
    verification = records.get("_verify")
    if failures or (isinstance(verification, dict) and verification.get("status") == "failed"):
        quality_status = "failed"
    preview = False
    try:
        with context.source.open("rb") as source:
            matching_source = (
                bool(context.source_sha256)
                and hashlib.file_digest(source, "sha256").hexdigest() == context.source_sha256
            )
        if matching_source:
            from orchestrator.api.viewer import asset_path, scene_manifest

            if scene_manifest(context.run_dir) is not None:
                exported = asset_path(
                    context.run_dir, f"clips/{context.name}/source{context.source.suffix.lower()}"
                )
                with exported.open("rb") as source:
                    preview = (
                        hashlib.file_digest(source, "sha256").hexdigest() == context.source_sha256
                    )
    except (OSError, ValueError, TypeError):
        pass
    return {
        "status": "preview_ready" if preview else "unavailable",
        "qualityStatus": quality_status,
        "reason": reason,
        "failedStages": failures,
    }


def finish_hands_off(context, journal: Journal, reason: str, *, quality_status="blocked"):
    """Write one terminal outcome while the caller holds the run admission lock.

    Already running jobs retain their receipts and may report their outcomes afterwards.
    A pause, cancellation or earlier finish wins over an agent's late response.
    """
    if journal.control_status() or journal.finished():
        return None
    return journal.append(
        "run.finished",
        status="failed",
        summary=reason,
        delivery=delivery_metadata(context, journal, reason, quality_status),
        handsOff=True,
    )
