"""Hash-pinned local room reuse, without generation or people/placement adoption."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marble_world import submission_history
from orchestrator.journal import Journal
from render_world_poses import read_spz

SCHEMA = "wander.local-world-import/1"
RESULT_SCHEMA = "wander.local-world-import-result/1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def no_symlinks(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Local world paths must be absolute and contain no symlinks")
    return path


def regular_file(path):
    path = no_symlinks(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("Local world input must be a nonempty regular file")
    return path


def checked_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Local world requires lowercase SHA-256 pins")
    if digest(regular_file(path)) != expected:
        raise ValueError("Local world input hash mismatch")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def validate_receipt(receipt_path, clip):
    receipt_path = regular_file(Path(receipt_path).absolute())
    receipt_hash = digest(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    fields = {
        "schema",
        "sourceSha256",
        "world",
        "associationEvidence",
        "generationLineage",
        "worldId",
        "operationId",
        "execution",
        "freshGeneration",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != fields
        or receipt["schema"] != SCHEMA
        or receipt["generationLineage"] != "unknown"
        or receipt["worldId"] is not None
        or receipt["operationId"] is not None
        or receipt["execution"] != "imported_local"
        or receipt["freshGeneration"] is not False
    ):
        raise ValueError("Local world receipt must explicitly retain unknown generation lineage")
    checked_hash(Path(clip), receipt["sourceSha256"])
    world = receipt["world"]
    if (
        not isinstance(world, dict)
        or set(world) != {"path", "bytes", "sha256"}
        or not isinstance(world["path"], str)
        or type(world["bytes"]) is not int
        or world["bytes"] <= 0
    ):
        raise ValueError("Local world requires an absolute path, bytes and hash")
    source = regular_file(world["path"])
    if source.suffix.lower() != ".spz" or source.stat().st_size != world["bytes"]:
        raise ValueError("Local world SPZ size or extension mismatch")
    checked_hash(source, world["sha256"])
    evidence = receipt["associationEvidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Local world requires pinned source association evidence")
    paired = False
    seen = set()
    for entry in evidence:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
        ):
            raise ValueError("Invalid local world association evidence")
        path = regular_file(entry["path"])
        if str(path) in seen:
            raise ValueError("Duplicate local world association evidence")
        seen.add(str(path))
        checked_hash(path, entry["sha256"])
        values = set(strings(json.loads(path.read_text())))
        paired = paired or {receipt["sourceSha256"], world["sha256"]} <= values
    if not paired:
        raise ValueError(
            "Association evidence must pair both original source and retained world hashes"
        )
    try:
        # The shared decoder uses assertions for its layout checks. Keep admission
        # checks active when a caller starts Python with optimization enabled.
        with gzip.open(source, "rb") as stream:
            magic, version, count, degree, _, _, _ = struct.unpack("<IIIBBBB", stream.read(16))
            if magic != 0x5053474E or version != 2 or degree != 0 or count <= 0:
                raise ValueError("Unsupported SPZ header")
            payload_bytes = sum(len(chunk) for chunk in iter(lambda: stream.read(1024 * 1024), b""))
            if payload_bytes != count * 19:
                raise ValueError("SPZ payload size mismatch")
        decoded = read_spz(source)
        if decoded["n"] <= 0 or any(
            not np.isfinite(decoded[k]).all() for k in ("xyz", "alpha", "rgb", "scale", "quat")
        ):
            raise ValueError("Empty/nonfinite decoded SPZ")
    except (AssertionError, OSError, ValueError, OverflowError, struct.error) as error:
        raise ValueError("Local world SPZ structure is invalid or unsupported") from error
    # Detect input changes during decoding without retaining decompressed geometry.
    checked_hash(source, world["sha256"])
    if digest(receipt_path) != receipt_hash:
        raise ValueError("Local world receipt changed during validation")
    return {
        "receipt": receipt,
        "receiptSha256": receipt_hash,
        "receiptPath": str(receipt_path),
        "splatCount": decoded["n"],
    }


def import_result(plan, target, status):
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "receipt": plan["receipt"],
        "receiptSha256": plan["receiptSha256"],
        "target": str(target),
        "splatCount": plan["splatCount"],
        "provider": "local",
        "execution": "imported_local",
        "freshGeneration": False,
        "generationLineage": "unknown",
        "worldId": None,
        "operationId": None,
        "qualityStatus": "unreviewed",
    }


def prepare(receipt_path, clip, ctx, public, marble_dir, name):
    """Read-only preflight; callers hold the existing selection lock when executing."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
        raise ValueError("Local world import requires a safe run name")
    plan = validate_receipt(receipt_path, clip)
    ctx, public = no_symlinks(Path(ctx).absolute()), no_symlinks(Path(public).absolute())
    target = no_symlinks(public / f"marble-{name}-clean.spz")
    retained = no_symlinks(ctx / "local-world-import.json")
    prior = None
    if retained.exists():
        prior = json.loads(regular_file(retained).read_text())
        if not isinstance(prior, dict) or prior.get("status") not in {"prepared", "complete"}:
            raise ValueError("Retained local world import does not match this receipt")
        expected = import_result(plan, target, prior["status"])
        if json.dumps(prior, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False
        ):
            raise ValueError("Retained local world import does not match this receipt")
    state_path = no_symlinks(ctx / "state.json")
    stages = (
        json.loads(regular_file(state_path).read_text()).get("stages", {})
        if state_path.exists()
        else {}
    )
    if stages.get("_marble"):
        raise ValueError("Existing Marble operation blocks local world import")
    no_symlinks(ctx / "journal.jsonl")
    no_symlinks(Path(marble_dir).absolute())
    active = Journal(ctx).active_attempts()
    for suffix, stage in (
        ("clean", "marble_video"),
        ("image", "marble_image"),
        ("multi", "marble_multi"),
    ):
        kind, _ = submission_history(marble_dir, f"{name}-{suffix}")
        if (
            kind != "new"
            or active.get(stage)
            or (ctx / f"marble_{suffix}.log").exists()
            or (ctx / f"{stage}.log").exists()
            or stages.get(stage, {}).get("status") == "running"
            or stages.get(stage, {}).get("status") == "ok"
            and not (stage == "marble_video" and prior)
            or suffix != "clean"
            and (public / f"marble-{name}-{suffix}.spz").exists()
        ):
            raise ValueError(
                "Existing/pending/uncertain Marble submission blocks local world import"
            )
    if (public / f"marble-{name}-finetuned.spz").exists():
        raise ValueError("Retained fine-tuned world blocks local world import")
    if target.exists():
        if prior is None:
            raise ValueError(
                "Existing world without retained local import receipt cannot be adopted"
            )
        checked_hash(target, plan["receipt"]["world"]["sha256"])
    plan.update(target=str(target), retained=str(retained), prior=prior)
    return plan


def write_json(path, document):
    fd, temporary = tempfile.mkstemp(prefix=".local-world-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def import_world(receipt_path, clip, ctx, public, marble_dir, name):
    plan = prepare(receipt_path, clip, ctx, public, marble_dir, name)
    target, retained = Path(plan["target"]), Path(plan["retained"])
    result = import_result(plan, target, "prepared")
    target.parent.mkdir(parents=True, exist_ok=True)
    if plan["prior"] is None:
        write_json(retained, result)
    if not target.exists():
        fd, temporary = tempfile.mkstemp(prefix=".local-world-", dir=target.parent)
        try:
            with (
                os.fdopen(fd, "wb") as output,
                Path(plan["receipt"]["world"]["path"]).open("rb") as source,
            ):
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            checked_hash(Path(temporary), plan["receipt"]["world"]["sha256"])
            os.link(temporary, target)  # Atomic publication without replacing an existing asset.
        finally:
            Path(temporary).unlink(missing_ok=True)
    final = validate_receipt(receipt_path, clip)
    if final["receiptSha256"] != plan["receiptSha256"]:
        raise ValueError("Local world receipt changed during import")
    checked_hash(target, plan["receipt"]["world"]["sha256"])
    result["status"] = "complete"
    write_json(retained, result)
    return result
