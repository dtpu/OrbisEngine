"""Immutable source-video ingress and artifact-byte egress for the API."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from botocore.exceptions import ClientError
from fastapi import HTTPException, status


@dataclass(frozen=True)
class ArtifactPayload:
    """The result of reading one artifact's bytes.

    Exactly one of `path` or `body` is set. `path` (filesystem reads) is handed to
    Starlette's `FileResponse`, which negotiates `Range` requests itself. `body`
    (S3 reads) is already the requested slice, with `status_code`/`content_range`
    set when the backend served a partial response.
    """

    path: Path | None = None
    body: bytes | None = None
    status_code: int = 200
    content_range: str | None = None


class S3SourceStore:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def put(self, source: Path, storage_key: str) -> None:
        try:
            self.client.head_object(Bucket=self.bucket, Key=storage_key)
            return
        except Exception:
            pass
        self.client.upload_file(str(source), self.bucket, storage_key)


class FilesystemSourceStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def put(self, source: Path, storage_key: str) -> None:
        destination = (self.root / storage_key).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("artifact storage key escapes the local store")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)


class S3ArtifactReader:
    """Read artifact bytes out of S3, matching `DatabaseArtifactResolver`'s key handling."""

    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def read(self, storage_key: str, *, range_header: str | None = None) -> ArtifactPayload:
        arguments: dict[str, str] = {"Bucket": self.bucket, "Key": storage_key}
        if range_header:
            arguments["Range"] = range_header
        try:
            response = self.client.get_object(**arguments)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404"}:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="artifact bytes are unavailable",
                ) from error
            raise
        body = response["Body"].read()
        content_range = response.get("ContentRange")
        return ArtifactPayload(
            body=body,
            status_code=206 if content_range else 200,
            content_range=content_range,
        )


class FilesystemArtifactReader:
    """Read artifact bytes out of the local content-addressed store."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def read(self, storage_key: str, *, range_header: str | None = None) -> ArtifactPayload:
        del range_header  # FileResponse negotiates Range directly against the path.
        candidate = (self.root / storage_key).resolve()
        if not candidate.is_relative_to(self.root) or not candidate.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="artifact bytes are unavailable",
            )
        return ArtifactPayload(path=candidate)
