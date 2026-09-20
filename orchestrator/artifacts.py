"""Content-addressed artifacts, immutable attempt manifests, and resumable uploads."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.exceptions import ClientError
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.workspace import AttemptWorkspace, atomic_json, is_output_file, safe_id


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FrozenFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    relative_path: str
    role: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    media_type: str
    blob_key: str


class AttemptManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "wander.attempt-manifest/1"
    run_id: str
    node_id: str
    attempt_id: str
    status: str
    files: tuple[FrozenFile, ...]


class Archive(Protocol):
    def has_blob(self, sha256: str) -> bool: ...

    def put_blob(self, sha256: str, source: Path) -> None: ...

    def put_manifest(self, key: str, value: bytes) -> None: ...


class LocalCAS:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.blobs = self.root / "blobs"
        self.manifests = self.root / "manifests"

    def initialize(self) -> None:
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)

    def blob_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("invalid SHA-256")
        return self.blobs / sha256[:2] / sha256

    def add_file(self, source: Path) -> tuple[str, Path]:
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"artifacts must be regular files: {source}")
        digest = sha256_file(source)
        destination = self.blob_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise ValueError(f"content-addressed blob size mismatch: {digest}")
            return digest, destination
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{digest}-", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as writer, source.open("rb") as reader:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
            if sha256_file(temporary) != digest:
                raise ValueError("artifact changed while it was being frozen")
            os.replace(temporary, destination)
            destination.chmod(0o400)
        finally:
            temporary.unlink(missing_ok=True)
        return digest, destination

    def has_blob(self, sha256: str) -> bool:
        path = self.blob_path(sha256)
        return path.is_file() and sha256_file(path) == sha256

    def put_blob(self, sha256: str, source: Path) -> None:
        actual, _ = self.add_file(source)
        if actual != sha256:
            raise ValueError(f"blob identity mismatch: expected {sha256}, got {actual}")

    def put_manifest(self, key: str, value: bytes) -> None:
        destination = self.manifests / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != value:
                raise ValueError(f"immutable manifest differs: {key}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}-", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def hydrate(self, sha256: str, destination: Path) -> None:
        source = self.blob_path(sha256)
        if not self.has_blob(sha256):
            raise FileNotFoundError(f"verified blob is unavailable: {sha256}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256:
                raise FileExistsError(f"destination has different bytes: {destination}")
            return
        shutil.copyfile(source, destination)
        if sha256_file(destination) != sha256:
            destination.unlink(missing_ok=True)
            raise ValueError("hydrated artifact failed verification")


class S3Archive:
    """Immutable S3-compatible archive used with AWS S3 or local MinIO."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "pipeline",
        endpoint_url: str | None = None,
        client: Any | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3", endpoint_url=endpoint_url)

    def blob_key(self, sha256: str) -> str:
        return f"{self.prefix}/blobs/{sha256[:2]}/{sha256}"

    def has_blob(self, sha256: str) -> bool:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=self.blob_key(sha256))
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise
        metadata = response.get("Metadata", {})
        return metadata.get("sha256") == sha256

    def put_blob(self, sha256: str, source: Path) -> None:
        if sha256_file(source) != sha256:
            raise ValueError("refusing to upload a blob with mismatched bytes")
        try:
            with Path(source).open("rb") as stream:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self.blob_key(sha256),
                    Body=stream,
                    ContentLength=Path(source).stat().st_size,
                    Metadata={"sha256": sha256},
                    IfNoneMatch="*",
                )
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            if not self.has_blob(sha256):
                raise ValueError("existing S3 blob does not match its content identity") from error

    def put_manifest(self, key: str, value: bytes) -> None:
        manifest_key = f"{self.prefix}/manifests/{key}"
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=manifest_key,
                Body=value,
                ContentLength=len(value),
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            existing = self.client.get_object(Bucket=self.bucket, Key=manifest_key)["Body"].read()
            if existing != value:
                raise ValueError(f"immutable S3 manifest differs: {key}") from error


def assign_roles(root: Path, roles: dict[str, str]) -> dict[str, str]:
    """Which role each file under ``root`` takes, by the same globbing the QA validator uses.

    ``PurePath.match`` treats ``**`` as one segment and anchors from the right, so a bundle
    pattern like ``outputs/prepared-person/**/*`` matched nothing here while
    ``Path.glob`` matched every file in the bundle. A stage's outputs therefore passed QA and
    were then frozen as ``attempt_file``, so the next stage asked for the role and got nothing.
    Globbing in both places is what keeps the two answers the same.

    An exact path wins over a pattern, so a bundle's catch-all does not swallow the one file
    inside it that has a role of its own; between patterns, the first one to name a file keeps
    it.
    """
    wildcards = set("*?[")
    exact = {pattern: role for pattern, role in roles.items() if not (wildcards & set(pattern))}
    assigned: dict[str, str] = {}
    for pattern, role in roles.items():
        if pattern in exact:
            continue
        for path in root.glob(pattern):
            if is_output_file(path):
                assigned.setdefault(path.relative_to(root).as_posix(), role)
    assigned.update(exact)
    return assigned


def freeze_attempt(
    attempt: AttemptWorkspace,
    store: LocalCAS,
    *,
    status: str,
    roles: dict[str, str] | None = None,
) -> AttemptManifest:
    roles = roles or {}
    assigned = assign_roles(attempt.root, roles)
    files = []
    for path in sorted(attempt.root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"attempt artifacts cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(attempt.root).as_posix()
        digest, _ = store.add_file(path)
        role = assigned.get(relative, "attempt_file")
        safe_id(role)
        identity = hashlib.sha256(
            f"{attempt.run_id}\0{attempt.attempt_id}\0{relative}\0{digest}".encode()
        ).hexdigest()
        files.append(
            FrozenFile(
                artifact_id=f"artifact:{identity}",
                relative_path=relative,
                role=role,
                sha256=digest,
                size=path.stat().st_size,
                media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                blob_key=f"blobs/{digest[:2]}/{digest}",
            )
        )
    manifest = AttemptManifest(
        run_id=attempt.run_id,
        node_id=attempt.node_id,
        attempt_id=attempt.attempt_id,
        status=status,
        files=tuple(files),
    )
    key = f"{safe_id(attempt.run_id)}/{safe_id(attempt.node_id)}/{safe_id(attempt.attempt_id)}.json"
    store.put_manifest(key, manifest.model_dump_json(indent=2).encode())
    return manifest


class UploadOutbox:
    def __init__(self, root: Path, source: LocalCAS, archive: Archive):
        self.root = Path(root)
        self.source = source
        self.archive = archive
        self.root.mkdir(parents=True, exist_ok=True)

    def enqueue(self, manifest: AttemptManifest) -> Path:
        path = self.root / f"{safe_id(manifest.attempt_id)}.json"
        value = {
            "schema": "wander.artifact-outbox/1",
            "status": "pending",
            "manifest": manifest.model_dump(mode="json"),
        }
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["manifest"] != value["manifest"]:
                raise ValueError(f"outbox entry differs for {manifest.attempt_id}")
            return path
        atomic_json(path, value)
        return path

    def flush(self) -> int:
        completed = 0
        for path in sorted(self.root.glob("*.json")):
            entry = json.loads(path.read_text())
            if entry.get("status") == "uploaded":
                continue
            manifest = AttemptManifest.model_validate(entry["manifest"])
            for artifact in manifest.files:
                source = self.source.blob_path(artifact.sha256)
                if not self.archive.has_blob(artifact.sha256):
                    self.archive.put_blob(artifact.sha256, source)
            key = f"{manifest.run_id}/{manifest.node_id}/{manifest.attempt_id}.json"
            self.archive.put_manifest(key, manifest.model_dump_json(indent=2).encode())
            entry["status"] = "uploaded"
            atomic_json(path, entry)
            completed += 1
        return completed


class ManifestResolver:
    """Resolve artifact IDs from immutable manifests while preserving source basenames."""

    def __init__(self, store: LocalCAS, manifests: list[AttemptManifest]):
        self.store = store
        self.files = {
            artifact.artifact_id: artifact for manifest in manifests for artifact in manifest.files
        }

    def hydrate(self, artifact_id: str, destination_directory: Path) -> Path:
        artifact = self.files.get(artifact_id)
        if artifact is None:
            raise KeyError(f"unknown artifact ID: {artifact_id}")
        destination_directory = Path(destination_directory)
        destination_directory.mkdir(parents=True, exist_ok=True)
        relative = Path(artifact.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact relative path: {artifact.relative_path}")
        destination = destination_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != artifact.sha256:
            destination = destination.with_name(f"{artifact.sha256[:12]}-{destination.name}")
        self.store.hydrate(artifact.sha256, destination)
        return destination


class DatabaseArtifactResolver:
    """Hydrate an artifact selected in PostgreSQL from its immutable S3 object."""

    def __init__(self, sessions, client, bucket: str):
        self.sessions = sessions
        self.client = client
        self.bucket = bucket

    def hydrate(self, artifact_id: str, destination_directory: Path) -> Path:
        from orchestrator.database import ArtifactRecord

        with self.sessions() as session:
            artifact = session.get(ArtifactRecord, artifact_id)
            if artifact is None:
                raise KeyError(f"unknown artifact ID: {artifact_id}")
            metadata = dict(artifact.artifact_metadata or {})
            name = Path(metadata.get("relative_path", artifact.id)).name
            destination = Path(destination_directory) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            self.client.download_file(self.bucket, artifact.storage_key, str(temporary))
            if sha256_file(temporary) != artifact.sha256:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"artifact hash mismatch: {artifact_id}")
            os.replace(temporary, destination)
            return destination


class DatabaseFilesystemArtifactResolver:
    def __init__(self, sessions, root: Path):
        self.sessions = sessions
        self.root = Path(root).resolve()

    def hydrate(self, artifact_id: str, destination_directory: Path) -> Path:
        from orchestrator.database import ArtifactRecord

        with self.sessions() as session:
            artifact = session.get(ArtifactRecord, artifact_id)
            if artifact is None:
                raise KeyError(f"unknown artifact ID: {artifact_id}")
            source = (self.root / artifact.storage_key).resolve()
            if not source.is_relative_to(self.root) or not source.is_file():
                raise ValueError(f"artifact storage object is unavailable: {artifact_id}")
            metadata = dict(artifact.artifact_metadata or {})
            destination = (
                Path(destination_directory) / Path(metadata.get("relative_path", artifact.id)).name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != artifact.sha256:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"artifact hash mismatch: {artifact_id}")
            os.replace(temporary, destination)
            return destination


def generate_preview(source: Path, destination: Path, media_type: str) -> Path | None:
    """Create a bounded dashboard preview without modifying the source artifact."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if media_type.startswith("image/"):
        with Image.open(source) as image:
            image.thumbnail((1280, 720))
            image.convert("RGB").save(destination.with_suffix(".jpg"), "JPEG", quality=85)
        return destination.with_suffix(".jpg")
    if media_type.startswith("video/"):
        preview = destination.with_suffix(".jpg")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                "0",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(1280,iw)':-2",
                str(preview),
            ],
            check=True,
        )
        return preview
    if media_type in {"application/json", "text/plain"}:
        preview = destination.with_suffix(".txt")
        with source.open("rb") as stream:
            content = stream.read(64 * 1024)
        preview.write_bytes(content)
        return preview
    return None


def flush_pending_outboxes(workspace_root: Path, store: LocalCAS, archive) -> int:
    """Push attempt manifests that were queued but never reached the archive.

    A run enqueues its files and flushes them immediately, but a flush can fail -- a full disk
    is the usual reason -- and the work still finishes, because the bytes are already safe in
    the local store. Nothing retried those entries afterwards, so the database recorded
    artifacts whose blobs the archive never received, and the next thing to want one failed
    with "artifact storage object is unavailable". Recovering at start-up costs nothing when
    there is nothing to do.
    """
    recovered = 0
    for outbox in sorted(Path(workspace_root).glob("*/outbox")):
        try:
            recovered += UploadOutbox(outbox, store, archive).flush()
        except Exception as error:  # noqa: BLE001 - one bad run must not stop the rest
            print(f"could not flush {outbox}: {error}", flush=True)
    return recovered
