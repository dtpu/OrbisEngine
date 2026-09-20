"""Read-only artifact egress: the admin viewer fetches bytes for a run's artifacts through here."""

from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from orchestrator.api.storage import ArtifactPayload
from orchestrator.repository import PipelineRepository


class ArtifactReader(Protocol):
    def read(self, storage_key: str, *, range_header: str | None = None) -> ArtifactPayload: ...


# Artifacts are untrusted bytes produced by tools and providers. Only types a browser renders
# harmlessly are served inline under the API origin; everything else (HTML, SVG with scripts,
# unknown binaries) is downgraded to an octet-stream download.
INLINE_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "video/mp4",
        "video/webm",
        "application/json",
        "text/plain",
        "text/csv",
        "text/markdown",
    }
)

_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


def base_media_type(media_type: str) -> str:
    return media_type.split(";", 1)[0].strip().lower()


def inline_allowed(media_type: str) -> bool:
    base = base_media_type(media_type)
    return base in INLINE_MEDIA_TYPES or base.startswith("audio/")


def safe_filename(name: str) -> str:
    candidate = _FILENAME_CHARACTERS.sub("_", name.rsplit("/", 1)[-1]).strip("._")
    return candidate[:120] or "artifact"


def register_artifact_routes(
    app: FastAPI,
    repository: PipelineRepository,
    actor: Callable[..., Coroutine[Any, Any, str]],
    reader: ArtifactReader | None,
) -> None:
    @app.get("/api/pipeline/runs/{run_id}/artifacts/{artifact_id}")
    async def read_artifact(
        run_id: str,
        artifact_id: str,
        request: Request,
        download: bool = False,
        user: str = Depends(actor),
    ):
        del user
        try:
            artifact = repository.get_artifact(run_id, artifact_id)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        if reader is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="artifact reads are not configured for this API",
            )

        inline = inline_allowed(artifact["mediaType"]) and not download
        served_type = (
            base_media_type(artifact["mediaType"]) if inline else "application/octet-stream"
        )
        if served_type == "application/json" or served_type.startswith("text/"):
            served_type += "; charset=utf-8"
        disposition = "inline" if inline else "attachment"
        headers = {
            "Content-Disposition": f'{disposition}; filename="{safe_filename(artifact["name"])}"',
            "ETag": f'"{artifact["sha256"]}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        }

        payload = reader.read(artifact["storageKey"], range_header=request.headers.get("range"))
        if payload.path is not None:
            # FileResponse negotiates Range itself and advertises Accept-Ranges: bytes.
            return FileResponse(payload.path, media_type=served_type, headers=headers)
        if payload.content_range:
            headers["Content-Range"] = payload.content_range
        headers["Accept-Ranges"] = "bytes"
        return Response(
            content=payload.body,
            status_code=payload.status_code,
            media_type=served_type,
            headers=headers,
        )
