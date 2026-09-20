"""Read-only legacy-run discovery and deterministic graph/artifact import planning.

This module never executes legacy commands.  It turns saved filesystem evidence into a
content-addressed plan; applying a plan only copies verified bytes into a ``LocalCAS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import stat
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from orchestrator.artifacts import AttemptManifest, FrozenFile, LocalCAS, sha256_file
from orchestrator.contracts import (
    Artifact,
    AttemptStatus,
    NodeStatus,
    Run,
    RunStatus,
    StageAttempt,
    StageDefinition,
    StageKind,
)
from orchestrator.graph import GraphNode, RunGraph, instantiate_graph
from orchestrator.stages import GraphOptions
from orchestrator.workspace import safe_id

PLAN_SCHEMA = "wander.legacy-import-plan/1"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
ENVIRONMENT_FINGERPRINT = hashlib.sha256(b"wander legacy backfill environment unknown").hexdigest()

_STATUS = {
    "ok": AttemptStatus.SUCCEEDED,
    "completed": AttemptStatus.SUCCEEDED,
    "complete": AttemptStatus.SUCCEEDED,
    "succeeded": AttemptStatus.SUCCEEDED,
    "failed": AttemptStatus.FAILED,
    "blocked": AttemptStatus.FAILED,
    "canceled": AttemptStatus.CANCELED,
    "cancelled": AttemptStatus.CANCELED,
    "pending": AttemptStatus.UNKNOWN,
    "running": AttemptStatus.UNKNOWN,
    "unknown": AttemptStatus.UNKNOWN,
    "gate": AttemptStatus.UNKNOWN,
}

_NODE_STATUS = {
    "ok": NodeStatus.SUCCEEDED,
    "completed": NodeStatus.SUCCEEDED,
    "complete": NodeStatus.SUCCEEDED,
    "succeeded": NodeStatus.SUCCEEDED,
    "failed": NodeStatus.FAILED,
    "blocked": NodeStatus.BLOCKED,
    "canceled": NodeStatus.CANCELED,
    "cancelled": NodeStatus.CANCELED,
    "skipped": NodeStatus.SKIPPED,
    "gate": NodeStatus.WAITING_HUMAN,
}

_LEGACY_NODE_ALIASES = {
    "package": "package_people",
    "review": "clean_review",
    "objects": "object_detect",
}


class ImportPlan(BaseModel):
    """A stable, DB-shaped description with no database side effects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["wander.legacy-import-plan/1"] = PLAN_SCHEMA
    run: Run
    graph: RunGraph
    attempts: tuple[StageAttempt, ...]
    artifacts: tuple[Artifact, ...]
    manifests: tuple[AttemptManifest, ...]
    warnings: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        value = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(value.encode()).hexdigest()


def discover_legacy_runs(root: Path) -> tuple[Path, ...]:
    """Find directories containing ``state.json`` without following any symlink."""
    root = _validated_directory(root)
    _regular_files(root)
    if (root / "state.json").is_file():
        return (root,)
    runs = {path.parent for path in root.rglob("state.json") if path.is_file()}
    return tuple(sorted(runs, key=lambda path: path.relative_to(root).as_posix()))


def plan_legacy_run(
    run_directory: Path,
    *,
    source: Path | None = None,
    run_id: str | None = None,
    options: GraphOptions | None = None,
    code_revision: str = "legacy0",
) -> ImportPlan:
    """Hash a legacy run and return an immutable import plan.

    ``source`` is an explicit trusted input and may be outside the run directory.  Paths found
    inside legacy metadata must remain below the run directory.
    """
    plan, _ = _build_plan(
        run_directory,
        source=source,
        run_id=run_id,
        options=options,
        code_revision=code_revision,
    )
    return plan


def import_legacy_run(
    run_directory: Path,
    *,
    store: LocalCAS | None = None,
    dry_run: bool = True,
    source: Path | None = None,
    run_id: str | None = None,
    options: GraphOptions | None = None,
    code_revision: str = "legacy0",
) -> ImportPlan:
    """Plan a run and optionally copy every artifact into an initialized local CAS."""
    plan, sources = _build_plan(
        run_directory,
        source=source,
        run_id=run_id,
        options=options,
        code_revision=code_revision,
    )
    if dry_run:
        return plan
    if store is None:
        raise ValueError("a LocalCAS is required when dry_run is false")
    store.initialize()
    for artifact in plan.artifacts:
        actual, _ = store.add_file(sources[artifact.id])
        if actual != artifact.sha256:
            raise ValueError(
                f"artifact changed after planning: {artifact.metadata['relative_path']}"
            )
    return plan


def _build_plan(
    run_directory: Path,
    *,
    source: Path | None,
    run_id: str | None,
    options: GraphOptions | None,
    code_revision: str,
) -> tuple[ImportPlan, dict[str, Path]]:
    root = _validated_directory(run_directory)
    files = _regular_files(root)
    state_path = root / "state.json"
    if state_path not in files:
        raise ValueError(f"legacy run has no regular state.json: {root}")
    state = _json_object(state_path, "legacy state")
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise TypeError("legacy state must contain an object named stages")
    if len(code_revision) < 7:
        raise ValueError("code_revision must contain at least seven characters")

    source_path, source_digest = _source_evidence(root, state, source)
    identity_seed = f"{root.name}\0{source_digest}".encode()
    identifier = (
        run_id
        or f"legacy:{_safe_fragment(root.name)}:{hashlib.sha256(identity_seed).hexdigest()[:12]}"
    )
    safe_id(identifier)

    graph = instantiate_graph(options or GraphOptions(marble="none", objects=False))
    synthetic_nodes = []
    for legacy_node in sorted(stages):
        if legacy_node.startswith("_"):
            continue
        node_id = _canonical_node(legacy_node)
        safe_id(node_id)
        if node_id in graph.nodes:
            continue
        graph.nodes[node_id] = GraphNode(
            id=node_id,
            stage_type=f"legacy:{node_id}",
            definition=StageDefinition(
                id=node_id,
                title=f"Legacy stage {legacy_node}",
                kind=StageKind.JOIN,
            ),
            dependencies=(),
        )
        synthetic_nodes.append(node_id)
    attempt_specs = _attempt_specs(root, stages, graph, identifier)
    owners: dict[Path, str] = {}
    for spec in attempt_specs:
        for path in spec["outputs"]:
            for item in _expand_output(root, path):
                previous = owners.setdefault(item, spec["id"])
                if previous != spec["id"]:
                    raise ValueError(f"legacy artifact is claimed by multiple attempts: {item}")

    # Old state generally retained one snapshot rather than an output list. Attribute only names
    # that identify their stage; ambiguous files remain run-level evidence.
    latest_by_node = {spec["node_id"]: spec["id"] for spec in attempt_specs}
    for path in files:
        if path in owners or path == state_path:
            continue
        relative = path.relative_to(root)
        matches = [node_id for node_id in latest_by_node if _path_names_stage(relative, node_id)]
        if len(matches) == 1:
            owners[path] = latest_by_node[matches[0]]

    source_key: Path | None = None
    if source_path is not None:
        source_key = source_path
        if source_path.is_relative_to(root):
            owners.pop(source_path, None)

    artifacts: list[Artifact] = []
    sources: dict[str, Path] = {}
    artifact_by_path: dict[Path, Artifact] = {}
    all_sources = list(files)
    if source_path is not None and source_path not in files:
        all_sources.append(source_path)
    display_paths: dict[str, Path] = {}
    for path in all_sources:
        relative = _display_path(root, path)
        previous = display_paths.setdefault(relative, path)
        if previous != path:
            raise ValueError(f"artifacts have the same import path: {previous} and {path}")
    for path in sorted(all_sources, key=lambda item: _display_path(root, item)):
        relative = _display_path(root, path)
        digest = sha256_file(path)
        if path == source_key:
            role = "source_video"
            producer = None
        else:
            role = f"legacy_file:{hashlib.sha256(relative.encode()).hexdigest()[:24]}"
            producer = owners.get(path)
        artifact_id = (
            "artifact:" + hashlib.sha256(f"{identifier}\0{relative}\0{digest}".encode()).hexdigest()
        )
        artifact = Artifact(
            id=artifact_id,
            run_id=identifier,
            role=role,
            sha256=digest,
            size=path.stat().st_size,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            storage_key=f"blobs/{digest[:2]}/{digest}",
            producer_attempt_id=producer,
            metadata={"relative_path": relative, "legacy": True},
            created_at=EPOCH,
        )
        artifacts.append(artifact)
        sources[artifact.id] = path
        artifact_by_path[path] = artifact

    source_artifact = next(
        (artifact for artifact in artifacts if artifact.role == "source_video"),
        None,
    )
    if source_artifact is None:
        raise ValueError(
            "source bytes are required; pass source=PATH or record an in-run stages._run.clip"
        )

    attempts: list[StageAttempt] = []
    manifests: list[AttemptManifest] = []
    for spec in attempt_specs:
        outputs = tuple(
            artifact.id for artifact in artifacts if artifact.producer_attempt_id == spec["id"]
        )
        status = spec["status"]
        terminal = status in {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELED,
        }
        attempts.append(
            StageAttempt(
                id=spec["id"],
                run_id=identifier,
                node_id=spec["node_id"],
                number=spec["number"],
                retry_of=spec["retry_of"],
                status=status,
                parameters=spec["parameters"],
                code_revision=code_revision,
                environment_fingerprint=ENVIRONMENT_FINGERPRINT,
                output_artifact_ids=outputs,
                started_at=EPOCH if terminal else None,
                finished_at=EPOCH if terminal else None,
                error=spec["error"],
            )
        )
        frozen = []
        for path, artifact in artifact_by_path.items():
            if artifact.producer_attempt_id != spec["id"]:
                continue
            relative = _display_path(root, path)
            frozen.append(
                FrozenFile(
                    artifact_id=artifact.id,
                    relative_path=relative,
                    role=artifact.role,
                    sha256=artifact.sha256,
                    size=artifact.size,
                    media_type=artifact.media_type,
                    blob_key=artifact.storage_key,
                )
            )
        manifests.append(
            AttemptManifest(
                run_id=identifier,
                node_id=spec["node_id"],
                attempt_id=spec["id"],
                status=status,
                files=tuple(sorted(frozen, key=lambda item: item.relative_path)),
            )
        )

    for node_id, entry in stages.items():
        canonical = _canonical_node(node_id)
        if canonical in graph.nodes and isinstance(entry, dict):
            records = entry.get("attempts")
            latest = records[-1] if isinstance(records, list) and records else entry
            status = str(latest.get("status", entry.get("status", ""))).lower()
            if status in _NODE_STATUS:
                graph.nodes[canonical].status = _NODE_STATUS[status]

    run = Run(
        id=identifier,
        graph_version=graph.fingerprint,
        code_revision=code_revision,
        source_sha256=source_digest,
        source_artifact_id=source_artifact.id,
        configuration={
            "backfill": PLAN_SCHEMA,
            "graph_options": (options or GraphOptions(marble="none", objects=False)).model_dump(),
        },
        status=RunStatus.PAUSED,
        created_by="legacy-backfill",
        created_at=EPOCH,
        updated_at=EPOCH,
    )
    warnings = tuple(
        f"legacy stage has no current graph definition: {node_id}" for node_id in synthetic_nodes
    )
    plan = ImportPlan(
        run=run,
        graph=graph,
        attempts=tuple(attempts),
        artifacts=tuple(artifacts),
        manifests=tuple(manifests),
        warnings=warnings,
    )
    return plan, sources


def _attempt_specs(
    root: Path,
    stages: dict[str, Any],
    graph: RunGraph,
    run_id: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for legacy_node in sorted(stages):
        if legacy_node.startswith("_"):
            continue
        node_id = _canonical_node(legacy_node)
        if node_id not in graph.nodes:
            continue
        entry = stages[legacy_node]
        if not isinstance(entry, dict):
            raise TypeError(f"legacy stage record must be an object: {legacy_node}")
        saved_attempts = entry.get("attempts")
        records = saved_attempts if isinstance(saved_attempts, list) else [entry]
        previous = None
        for number, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise TypeError(f"legacy attempt record must be an object: {legacy_node}")
            raw_status = str(record.get("status", entry.get("status", "unknown"))).lower()
            status = _STATUS.get(raw_status, AttemptStatus.UNKNOWN)
            token = hashlib.sha256(f"{run_id}\0{node_id}\0{number}".encode()).hexdigest()[:16]
            attempt_id = f"attempt:{token}"
            output_values = _output_values(record)
            outputs = tuple(_confined_metadata_path(root, value) for value in output_values)
            parameters = record.get("parameters", {})
            if not isinstance(parameters, dict):
                raise TypeError(f"legacy attempt parameters must be an object: {legacy_node}")
            error = record.get("error")
            specs.append(
                {
                    "id": attempt_id,
                    "node_id": node_id,
                    "number": number,
                    "retry_of": previous,
                    "status": status,
                    "outputs": outputs,
                    "parameters": parameters,
                    "error": str(error) if error is not None else None,
                }
            )
            previous = attempt_id
    return specs


def _output_values(record: dict[str, Any]) -> tuple[str, ...]:
    for key in ("outputs", "results", "files"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise ValueError(f"legacy attempt {key} must be a path or list of paths")
    return ()


def _source_evidence(
    root: Path,
    state: dict[str, Any],
    explicit: Path | None,
) -> tuple[Path | None, str]:
    if explicit is not None:
        source = _validated_regular_file(explicit, "source")
        return source, sha256_file(source)
    run = state.get("stages", {}).get("_run", {})
    if isinstance(run, dict) and isinstance(run.get("clip"), str):
        candidate = _confined_metadata_path(root, run["clip"])
        source = _validated_regular_file(candidate, "source")
        return source, sha256_file(source)
    raise ValueError("source bytes are required; pass source=PATH")


def _regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted((*directories, *names)):
            path = current_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"legacy runs cannot contain symlinks: {path}")
            if name in directories and not stat.S_ISDIR(mode):
                raise ValueError(f"unexpected legacy directory entry: {path}")
            if name in names:
                if not stat.S_ISREG(mode):
                    raise ValueError(f"artifacts must be regular files: {path}")
                _ensure_below(root, path)
                files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _validated_directory(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"legacy run root cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"legacy run root is not a directory: {path}")
    return resolved


def _validated_regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return resolved


def _confined_metadata_path(root: Path, value: str) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError(f"legacy metadata refers to a symlink: {value}")
    _ensure_below(root, candidate.resolve(strict=False))
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"legacy metadata refers to a missing path: {value}") from error
    _ensure_below(root, resolved)
    return resolved


def _ensure_below(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"legacy path escapes run directory: {path}") from error


def _expand_output(root: Path, path: Path) -> Iterable[Path]:
    _ensure_below(root, path)
    if path.is_file():
        return (path,)
    if path.is_dir():
        return _regular_files(path)
    raise ValueError(f"legacy output is not a regular file or directory: {path}")


def _path_names_stage(relative: Path, node_id: str) -> bool:
    names = {node_id, node_id.replace(":", "_")}
    first = relative.parts[0].lower()
    stem = relative.stem.lower()
    return any(
        first == name
        or first.startswith((name + "-", name + "_"))
        or stem == name
        or stem.startswith((name + "-", name + "_"))
        for name in names
    )


def _canonical_node(value: str) -> str:
    value = _LEGACY_NODE_ALIASES.get(value, value)
    match = re.fullmatch(r"(person_prep|lhm_frozen|lhm_motion)_(\d+)", value)
    return f"{match.group(1)}:{match.group(2)}" if match else value


def _safe_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned[:48] or "run"


def _display_path(root: Path, path: Path) -> str:
    if path.is_relative_to(root):
        return path.relative_to(root).as_posix()
    return f"source/{path.name}"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--code-revision", default="legacy0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cas", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.dry_run and args.cas is None:
        parser.error("--cas is required unless --dry-run is used")
    store = LocalCAS(args.cas) if args.cas else None
    plan = import_legacy_run(
        args.run_directory,
        store=store,
        dry_run=args.dry_run,
        source=args.source,
        run_id=args.run_id,
        code_revision=args.code_revision,
    )
    rendered = plan.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
