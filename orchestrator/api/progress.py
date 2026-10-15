"""A read-only explanation of the authoritative run directory for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse

from orchestrator.api.reviews import run_directories
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


def previews(root: Path) -> list[dict]:
    """Only known media/report outputs are downloadable; never arbitrary run files."""
    manifest = read_json(root / "run.json")
    source = Path(manifest.get("source") or root / "missing")
    candidates = [("Source clip", source, "video")]
    candidates += [("Cleaned clip", p, "video") for p in (root / "clips").glob("*-clean.mp4")]
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


def describe_progress(root: Path) -> dict:
    described = read_json(root / "run.json")
    context = RunContext(
        described["runId"],
        root,
        described["name"],
        Path(described["source"]),
        Path(described["repository"]),
        described.get("options") or {},
    )
    journal = Journal(root)
    records = read_json(root / "state.json").get("stages", {})
    done = completed(context)
    people, many = graph_shape(context)
    rows = []
    running = in_flight(journal)
    latest_results = {
        entry.data["step"]: entry.data
        for entry in journal.entries()
        if entry.kind == "step.finished" and entry.data.get("step")
    }
    for name in plan_for(context):
        record = records.get(name, {})
        status = (
            "running"
            if name in running
            else record.get("status") or journal.last_status(name) or "pending"
        )
        status = {"ok": "succeeded", "gate": "blocked"}.get(status, status)
        missing = [dep for dep in needs_of(name, people, many, context.options) if dep not in done]
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
        rows.append(
            {
                "name": name,
                "summary": STEPS[base_step(name)].summary,
                "status": status,
                "reason": reason,
                "nextAction": action,
                "dependencies": missing,
            }
        )
    inputs = previews(root)
    prefix = f"/api/pipeline/runs/{quote(context.run_id, safe='')}/files/"
    for item in inputs:
        item["url"] = prefix + quote(item["path"], safe="/")
    stopped = [row for row in rows if row["status"] in {"failed", "blocked"}]
    current = [row["name"] for row in rows if row["status"] == "running"]
    return {
        "runId": context.run_id,
        "finished": journal.finished(),
        "paused": (root / "paused").exists(),
        "question": journal.unanswered(),
        "currentStages": current,
        "stopped": stopped,
        "stages": rows,
        "previews": inputs,
        "viewerPath": (
            f"/fourd.html?scene={quote(f'/api/pipeline/runs/{context.run_id}/viewer/scene.json', safe='')}"
            if scene_manifest(root) is not None
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
        candidate = run_directories(root).get(run_id) if root else None
        if candidate is None or not candidate.resolve().is_relative_to(root):
            raise HTTPException(404, "run directory unavailable")
        return candidate

    register_viewer_routes(app, actor, directory)

    @app.get("/api/pipeline/runs/{run_id}/progress")
    def progress(run_id: str, user: str = Depends(actor)):
        return describe_progress(directory(run_id))

    @app.get("/api/pipeline/runs/{run_id}/files/{path:path}")
    def preview(run_id: str, path: str, user: str = Depends(actor)):
        root = directory(run_id)
        allowed = {item["path"] for item in previews(root)}
        if path not in allowed:
            raise HTTPException(404, "unknown preview")
        return FileResponse(
            root / path, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        )
