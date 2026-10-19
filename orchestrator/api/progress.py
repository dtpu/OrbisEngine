"""A read-only explanation of the authoritative run directory for the dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse

from orchestrator.cli import RunContext, completed, graph_shape, in_flight, plan_for
from orchestrator.journal import Journal
from orchestrator.steps import STEPS, base_step, needs_of
from orchestrator.api.viewer import register_viewer_routes, scene_manifest


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def find_run_directory(root: Path, run_id: str) -> Path | None:
    """Find one published run, ignoring malformed or escaping neighbors."""
    root = root.resolve()
    for manifest in root.glob("*/run.json"):
        candidate = manifest.parent
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            continue
        if read_json(manifest).get("runId") == run_id:
            return candidate
    return None


def previews(root: Path) -> list[dict]:
    """Only known media/report outputs are downloadable; never arbitrary run files."""
    manifest = read_json(root / "run.json")
    source = Path(manifest.get("source") or root / "missing")
    candidates = [("Source clip", source, "video")]
    candidates += [("Cleaned clip", p, "video") for p in (root / "clips").glob("*-clean.mp4")]
    candidates += [
        (f"Cleaning candidate: {p.parent.name}", p, "video")
        for p in sorted(root.glob("clean-candidates/*/clean.mp4"))
    ]
    for pattern in (
        "first/*.png",
        "review/*.jpg",
        "review/*.png",
        "verify/*.png",
        "verify/diagnostic-*/*.png",
    ):
        candidates += [("Review frame", p, "image") for p in sorted(root.glob(pattern))[:12]]
    for pattern in (
        "public/worlds/*/person/sequence.json",
        "public/worlds/*/people.json",
        "public/worlds/*/cameras.json",
        "public/worlds/*/placement.json",
        "verify/report.json",
        "verify/diagnostic-*/report.json",
        "anchors.json",
        "scale-fit.json",
    ):
        candidates += [("Output report", p, "report") for p in sorted(root.glob(pattern))]
    result = []
    for label, path, kind in candidates:
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        result.append(
            {"label": label, "path": relative, "kind": kind, "bytes": path.stat().st_size}
        )
    return result


@dataclass(frozen=True)
class StageSnapshot:
    """The run directory's stage view, before preview and viewer discovery."""

    run_id: str
    journal: Journal
    stages: list[dict]
    required_dependencies: dict[str, list[str]]


def _stage_snapshot(root: Path) -> StageSnapshot | None:
    """Read stage state without changing the database or run directory.

    A summary can outlive its run directory, so a missing or incomplete manifest
    leaves the database summary available instead of failing the whole request.
    """
    described = read_json(root / "run.json")
    if not all(
        isinstance(described.get(key), str) and described[key]
        for key in ("runId", "name", "source", "repository")
    ) or not isinstance(described.get("options") or {}, dict):
        return None
    context = RunContext(
        described["runId"],
        root,
        described["name"],
        Path(described["source"]),
        Path(described["repository"]),
        described.get("options") or {},
        retained_delivery=described.get("retainedDelivery"),
    )
    journal = Journal(root)
    records = read_json(root / "state.json").get("stages", {})
    if not isinstance(records, dict):
        records = {}
    done = completed(context)
    people, many = graph_shape(context)
    rows = []
    running = in_flight(journal)
    pending = journal.pending_question()
    latest_results = {
        entry.data["step"]: entry.data
        for entry in journal.entries()
        if entry.kind == "step.finished" and entry.data.get("step")
    }
    latest_starts = {
        entry.data["step"]: entry.data
        for entry in journal.entries()
        if entry.kind == "step.started" and entry.data.get("step")
    }
    required_dependencies = {}
    for name in plan_for(context):
        record = records.get(name, {})
        if not isinstance(record, dict):
            record = {}
        status = (
            "running"
            if name in running
            else record.get("status") or journal.last_status(name) or "pending"
        )
        status = {"ok": "succeeded", "gate": "blocked"}.get(status, status)
        required = needs_of(name, people, many, context.options)
        if context.retained_delivery is not None:
            required = [dependency for dependency in required if dependency in plan_for(context)]
        required_dependencies[name] = required
        missing = [dep for dep in required if dep not in done]
        latest = latest_results.get(name, {})
        if latest.get("preflight") and name not in running:
            status = latest["status"]
        reason = (
            (record.get("error") or latest.get("error"))
            if status in {"failed", "blocked"}
            else None
        )
        if status == "pending" and missing:
            action = f"Waiting for {', '.join(missing)}."
        elif status in {"failed", "blocked"}:
            action = "Inspect the stage log and retained provider receipt before resuming."
            if name == "review":
                action = "Inspect the cleaned frames, then approve them before world generation."
        elif status == "pending":
            action = "Ready for the supervisor to start."
        elif status == "running":
            action = "Processing; follow the live transcript."
        else:
            action = "No action needed."
        if pending is not None and pending.data.get("step") == name:
            status = "waiting_human"
            reason = pending.data.get("question")
            action = "Answer the pending question to release this hold."
        row = {
            "name": name,
            "summary": STEPS[base_step(name)].summary,
            "status": status,
            "reason": reason,
            "nextAction": action,
            "dependencies": missing,
        }
        # run_clip merges state fields. An actual later attempt can leave the old
        # retained-input marker behind; only an explicit journal reuse claim may
        # carry that marker across a new start.
        started = latest_starts.get(name)
        reuse_claimed = (
            started
            and started.get("execution") == "retained-input-replay"
            and started.get("freshInference") is False
        )
        if not started or reuse_claimed:
            for field in ("execution", "freshInference", "sourceRun"):
                if field in record:
                    row[field] = record[field]
        rows.append(row)
    return StageSnapshot(context.run_id, journal, rows, required_dependencies)


def stage_snapshot(root: Path) -> StageSnapshot | None:
    """Return no snapshot when a run's persisted inputs cannot be interpreted."""
    try:
        return _stage_snapshot(root)
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def describe_progress(root: Path) -> dict:
    snapshot = stage_snapshot(root)
    if snapshot is None:
        raise ValueError(f"run manifest is unavailable: {root / 'run.json'}")
    journal = snapshot.journal
    rows = snapshot.stages
    inputs = previews(root)
    prefix = f"/api/pipeline/runs/{quote(snapshot.run_id, safe='')}/files/"
    for item in inputs:
        item["url"] = prefix + quote(item["path"], safe="/")
    stopped = [row for row in rows if row["status"] in {"failed", "blocked"}]
    current = [row["name"] for row in rows if row["status"] == "running"]
    viewer_available = scene_manifest(root) is not None
    result = None
    if journal.finished():
        finished = next(
            (entry for entry in reversed(journal.entries()) if entry.kind == "run.finished"),
            None,
        )
        if finished is not None:
            result = {"summary": finished.data.get("summary", "")}
            delivery = finished.data.get("delivery")
            if (
                isinstance(delivery, dict)
                and delivery.get("status") in {"preview_ready", "unavailable"}
                and delivery.get("qualityStatus") in {"blocked", "failed"}
            ):
                result["delivery"] = {
                    # A retained receipt cannot promise a scene whose files are now missing.
                    "status": delivery["status"] if viewer_available else "unavailable",
                    "qualityStatus": delivery["qualityStatus"],
                    "reason": delivery.get("reason", ""),
                }
    return {
        "runId": snapshot.run_id,
        "finished": journal.finished(),
        "paused": (root / "paused").exists(),
        "question": journal.unanswered(),
        "currentStages": current,
        "stopped": stopped,
        "stages": rows,
        "previews": inputs,
        "handsOff": (read_json(root / "run.json").get("options") or {}).get("hands_off") is True,
        "result": result,
        "viewerPath": (
            f"/fourd.html?scene={quote(f'/api/pipeline/runs/{snapshot.run_id}/viewer/scene.json', safe='')}"
            if viewer_available
            and (result or {}).get("delivery", {}).get("status") != "unavailable"
            else None
        ),
    }


def register_progress_routes(app, repository, actor, workspace_root: Path | None):
    def directory(run_id: str) -> Path:
        try:
            repository.run_summary(run_id)
        except KeyError as error:
            raise HTTPException(404, "unknown run") from error
        root = workspace_root.resolve() if workspace_root else None
        candidate = find_run_directory(root, run_id) if root else None
        if candidate is None or not candidate.resolve().is_relative_to(root):
            raise HTTPException(404, "run directory unavailable")
        return candidate

    register_viewer_routes(app, actor, directory)

    @app.get("/api/pipeline/runs/{run_id}/progress")
    def progress(run_id: str, user: str = Depends(actor)):
        try:
            return describe_progress(directory(run_id))
        except ValueError as error:
            raise HTTPException(503, "run progress unavailable") from error

    @app.get("/api/pipeline/runs/{run_id}/files/{path:path}")
    def preview(run_id: str, path: str, user: str = Depends(actor)):
        root = directory(run_id)
        allowed = {item["path"] for item in previews(root)}
        if path not in allowed:
            raise HTTPException(404, "unknown preview")
        return FileResponse(
            root / path, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        )
