"""Authenticated operator import of retained, run-scoped static-world evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.api.progress import find_run_directory
from orchestrator.journal import JOURNAL_FILE, Journal, RunProjection
from orchestrator.steps import STEPS
from orchestrator.world_review import (
    ReviewConflict,
    ReviewUnavailable,
    SCOPE,
    check_current,
    import_review,
    read_diagnostic,
    read_run_context,
    read_state,
    review_hold,
    run_path,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class WorldReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosticId: str = Field(pattern=r"^diagnostic-[A-Za-z0-9_-]{1,64}$")
    planSha256: str = Field(pattern=SHA256_PATTERN)
    result: dict


def register_world_review_routes(app: FastAPI, repository, actor, workspace_root: Path | None):
    base = "/api/pipeline/runs/{run_id}/world-review"
    root = Path(workspace_root).resolve() if workspace_root else None

    def context_for(run_id):
        if root is None:
            raise HTTPException(404, "No retained review workspace is configured")
        directory = find_run_directory(root, run_id)
        if directory is None:
            raise HTTPException(404, "Unknown run")
        try:
            context = read_run_context(directory, run_id)
            for name in (JOURNAL_FILE, "paused"):
                run_path(directory, directory / name)
            return context
        except ReviewUnavailable as error:
            raise HTTPException(404, str(error)) from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(409, "The retained run manifest is invalid") from error

    @app.get(base)
    def world_review(run_id: str, user: str = Depends(actor)):
        context = context_for(run_id)
        try:
            state = read_state(context.run_dir)
            diagnostic = read_diagnostic(context.run_dir, state)
            check_current(context, state, diagnostic)
            hold = review_hold(Journal(context.run_dir), state)
        except ReviewUnavailable as error:
            raise HTTPException(404, str(error)) from error
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            raise HTTPException(
                409, "The retained diagnostic is stale, incomplete or invalid"
            ) from error

        path = base.format(run_id=quote(run_id, safe="")) + "/" + diagnostic.identifier
        return {
            "diagnosticId": diagnostic.identifier,
            "planSha256": diagnostic.plan_sha,
            "scope": SCOPE,
            "samples": [
                {
                    **sample,
                    "pairUrl": (
                        f"{path}/pairs/{sample['sourceFrame']}?planSha256={diagnostic.plan_sha}"
                    ),
                }
                for sample in diagnostic.samples
            ],
            "reviewTemplate": diagnostic.template(),
            "canSubmit": hold is None,
            "holdReason": hold,
        }

    @app.get(base + "/{diagnostic_id}/pairs/{source_frame}")
    def review_pair(
        run_id: str,
        diagnostic_id: str,
        source_frame: int,
        planSha256: str = Query(pattern=SHA256_PATTERN),
        user: str = Depends(actor),
    ):
        context = context_for(run_id)
        try:
            state = read_state(context.run_dir)
            diagnostic = read_diagnostic(context.run_dir, state, check_images=False)
            if diagnostic.identifier != diagnostic_id or diagnostic.plan_sha != planSha256:
                raise ValueError("The diagnostic plan changed")
            sample = next(
                (row for row in diagnostic.samples if row["sourceFrame"] == source_frame), None
            )
            if sample is None:
                raise HTTPException(404, "Unknown diagnostic pair")
            pair = run_path(context.run_dir, diagnostic.pair_path(source_frame)).read_bytes()
            if (
                not pair.startswith(b"\x89PNG\r\n\x1a\n")
                or hashlib.sha256(pair).hexdigest() != sample["pairSha256"]
            ):
                raise ValueError("The diagnostic pair bytes changed")
        except ReviewUnavailable as error:
            raise HTTPException(404, str(error)) from error
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            raise HTTPException(409, "The retained diagnostic pair is stale or invalid") from error
        return Response(
            pair,
            media_type="image/png",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post(base)
    def submit_world_review(run_id: str, request: WorldReviewRequest, user: str = Depends(actor)):
        context = context_for(run_id)
        journal = Journal(
            context.run_dir, projection=RunProjection(repository, run_id, catalogue=STEPS)
        )
        try:
            return import_review(
                context, journal, request.diagnosticId, request.planSha256, request.result
            )
        except ReviewUnavailable as error:
            raise HTTPException(404, str(error)) from error
        except ReviewConflict as error:
            raise HTTPException(409, str(error)) from error
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            raise HTTPException(
                409, "The retained diagnostic is stale, incomplete or invalid"
            ) from error
