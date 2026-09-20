"""Operator messages and the agent's live transcript, for the admin sidebar.

The harness streams ``codex exec --json`` (or the Claude equivalent) straight into
``<workspace root>/<run>/reviews/<node>/<attempt>/transcript.jsonl`` while the agent works, so
tailing that file is the only way to watch the agent think before its decision is recorded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status

from orchestrator.journal import Journal
from orchestrator.repository import PipelineRepository
from orchestrator.workspace import safe_id

TRANSCRIPT_FILE = "transcript.jsonl"
# One tail request returns at most this many bytes; the client keeps calling with `after`.
TAIL_LIMIT = 512_000


def _identifier(value: str) -> str:
    try:
        return safe_id(value)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


def _inside(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown transcript")
    return resolved


def _modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def run_directories(root: Path) -> dict[str, Path]:
    """Every run's directory, by run id.

    A run directory is named for the clip rather than the run, because that is the name the
    legacy pipeline writes under and a person reading the filesystem wants to see. The id the
    dashboard asks by is inside run.json.
    """
    found: dict[str, Path] = {}
    for described in root.glob("*/run.json"):
        try:
            run_id = json.loads(described.read_text()).get("runId")
        except (OSError, ValueError):
            continue
        if run_id:
            found[str(run_id)] = described.parent
    return found


def _described(run_dir: Path, run_id: str) -> dict[str, Any] | None:
    """One run's transcript, described the way the sidebar expects a review to be.

    There is one agent per run now rather than one per attempt, so the node and attempt this
    reports are the session itself. Keeping the shape means the admin panel did not have to
    learn about the change.
    """
    transcript = run_dir / TRANSCRIPT_FILE
    if transcript.is_symlink() or not transcript.is_file():
        return None
    journal = Journal(run_dir)
    return {
        "runId": run_id,
        "nodeId": "agent",
        "attemptId": run_dir.name,
        "bytes": transcript.stat().st_size,
        "updatedAt": _modified(transcript),
        "finished": journal.finished() is not None,
    }


def list_transcripts(root: Path, run_id: str) -> list[dict[str, Any]]:
    run_dir = run_directories(root).get(run_id)
    if run_dir is None:
        return []
    described = _described(_inside(root, run_dir), run_id)
    return [described] if described else []


def list_recent_transcripts(root: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Newest transcripts across every run, for the sidebar that follows whatever Codex is doing."""
    found = [
        described
        for run_id, run_dir in run_directories(root).items()
        if (described := _described(run_dir, run_id)) is not None
    ]
    found.sort(key=lambda item: item["updatedAt"], reverse=True)
    return found[:limit]


def tail_transcript(path: Path, after: int, limit: int = TAIL_LIMIT) -> dict[str, Any]:
    size = path.stat().st_size
    # A re-review reopens the file with "wb", so it shrinks; the reader must start over rather
    # than read from a stale offset.
    reset = after > size
    after = 0 if reset else max(0, after)
    with path.open("rb") as stream:
        stream.seek(after)
        chunk = stream.read(limit)
    # Serve whole lines only; a partial trailing line is re-read on the next call.
    cut = chunk.rfind(b"\n")
    chunk = chunk[: cut + 1] if cut != -1 else b""
    events: list[dict[str, Any]] = []
    for raw in chunk.decode("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            events.append({"type": "raw", "text": line})
            continue
        events.append(parsed if isinstance(parsed, dict) else {"type": "raw", "text": line})
    return {
        "offset": after + len(chunk),
        "size": size,
        "reset": reset,
        "updatedAt": _modified(path),
        "events": events,
    }


def register_review_routes(
    app: FastAPI,
    repository: PipelineRepository,
    actor: Callable[..., Coroutine[Any, Any, str]],
    workspace_root: Path | None,
) -> None:
    root = Path(workspace_root).resolve() if workspace_root else None

    @app.get("/api/pipeline/runs/{run_id}/messages")
    async def messages(run_id: str, user: str = Depends(actor)):
        del user
        return {"messages": repository.list_messages(run_id)}

    @app.get("/api/pipeline/reviews")
    async def recent_reviews(
        limit: int = Query(default=20, ge=1, le=100), user: str = Depends(actor)
    ):
        del user
        if root is None:
            return {"transcripts": [], "available": False}
        return {"transcripts": list_recent_transcripts(root, limit), "available": True}

    @app.get("/api/pipeline/runs/{run_id}/reviews")
    async def reviews(run_id: str, user: str = Depends(actor)):
        del user
        if root is None:
            return {"transcripts": [], "available": False}
        return {"transcripts": list_transcripts(root, _identifier(run_id)), "available": True}

    @app.get("/api/pipeline/runs/{run_id}/reviews/{node_id}/{attempt_id}/transcript")
    async def transcript(
        run_id: str,
        node_id: str,
        attempt_id: str,
        after: int = Query(default=0, ge=0),
        user: str = Depends(actor),
    ):
        del user
        if root is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="review transcripts are not configured for this API",
            )
        del node_id, attempt_id  # one agent per run; the session is the transcript
        run_dir = run_directories(root).get(run_id)
        if run_dir is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown run")
        path = _inside(root, run_dir / TRANSCRIPT_FILE)
        if path.is_symlink() or not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown transcript")
        payload = tail_transcript(path, after)
        payload["finished"] = (path.parent / "decision.json").is_file()
        return payload
