"""Durable generated LHM outputs and read-only, inference-free recovery.

Recover with: uv run --locked python worker/stages/lhm_recovery.py --receipt PATH
Only Modal Volume reads occur in this CLI; it never constructs an App or spawns inference.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

VOLUME_NAME = "wander-overnight-lhm-cache"
SCHEMA = "wander.lhm-recovery/1"
RECEIPT_NAME = "recovery-receipt.json"
FORBIDDEN_OUTPUTS = {
    "source.mp4",
    "source.mov",
    "source.png",
    "mask.png",
    "prepared.json",
    "recovery-receipt.json",
    "recovery-manifest.json",
    "artifacts.tar.gz",
}


# What lhm_animate.py writes as the reason when a track simply is not in a sample.
TRACK_ABSENT = "Track absent from this sample"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_root(recovery_id):
    if not isinstance(recovery_id, str) or not re.fullmatch(r"[0-9a-f]{32}", recovery_id):
        raise ValueError("Recovery id must be a lowercase UUID hex string")
    return f"recovery/{recovery_id}"


def safe_relative(value):
    if (
        not isinstance(value, str)
        or not value
        or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value)
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).name in FORBIDDEN_OUTPUTS
        or PurePosixPath(value).suffix.lower() in (".mp4", ".mov")
    ):
        raise ValueError("Unsafe or original-input output path")
    return value


def new_receipt(destination, mode):
    if mode not in ("frozen", "motion"):
        raise ValueError("Invalid LHM recovery mode")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    recovery_id = uuid.uuid4().hex
    receipt = {
        "schema": SCHEMA,
        "recoveryId": recovery_id,
        "volume": VOLUME_NAME,
        "remotePath": remote_root(recovery_id),
        "destination": str(destination.resolve()),
        "mode": mode,
        "status": "prepared",
        "functionCallId": None,
        "createdAtEpoch": time.time(),
    }
    path = destination / RECEIPT_NAME
    atomic_json(path, receipt)
    return path, receipt


def load_receipt(path):
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise ValueError(
            "Recovery receipt is missing or unreadable; inference was not submitted"
        ) from None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != SCHEMA
        or receipt.get("volume") != VOLUME_NAME
        or receipt.get("remotePath") != remote_root(receipt.get("recoveryId"))
        or receipt.get("mode") not in ("frozen", "motion")
        or not isinstance(receipt.get("destination"), str)
        or not Path(receipt["destination"]).is_absolute()
    ):
        raise ValueError("Invalid recovery receipt; inference was not submitted")
    return receipt


@contextmanager
def receipt_lock(receipt_path):
    """One local owner across threads/processes, including while a provider outcome is unknown.

    The lock file must stay in place: deleting it could let two callers lock different inodes.
    This coordinates callers sharing this filesystem, not copies on separate hosts.
    """
    receipt_path = Path(receipt_path)
    if receipt_path.is_symlink():
        raise ValueError("Recovery receipt must not be a symlink")
    lock_path = receipt_path.with_name(receipt_path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "LHM receipt is busy; another caller owns submission or recovery"
            ) from None
        yield


def submit_once(function, receipt_path, inputs, **kwargs):
    """Persist identity before waiting. An uncertain or interrupted receipt never resubmits."""
    with receipt_lock(receipt_path):
        return _submit_once(function, receipt_path, inputs, **kwargs)


def _submit_once(function, receipt_path, inputs, **kwargs):
    receipt = load_receipt(receipt_path)
    if receipt.get("status") != "prepared" or receipt.get("functionCallId") is not None:
        raise ValueError("Existing LHM receipt cannot resubmit inference; use recovery")
    receipt["status"] = "submitting"
    atomic_json(receipt_path, receipt)
    try:
        call = function.spawn(inputs, recovery_id=receipt["recoveryId"], **kwargs)
    except BaseException:
        receipt["status"] = "submission_uncertain"
        atomic_json(receipt_path, receipt)
        raise
    receipt["functionCallId"] = call.object_id
    receipt["status"] = "waiting"
    atomic_json(receipt_path, receipt)
    try:
        result = call.get()
    except BaseException:
        receipt["status"] = "wait_interrupted"
        atomic_json(receipt_path, receipt)
        raise
    if (
        not isinstance(result, dict)
        or result.get("recoveryId") != receipt["recoveryId"]
        or result.get("remotePath") != receipt["remotePath"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(result.get("manifestSha256", "")))
    ):
        raise ValueError("LHM returned an invalid checkpoint receipt; use recovery")
    receipt["totalWorkerSeconds"] = result.get("totalWorkerSeconds")
    receipt["manifestSha256"] = result["manifestSha256"]
    receipt["status"] = "checkpoint_available"
    atomic_json(receipt_path, receipt)
    return result


def start_checkpoint(cache_root, recovery_id, mode):
    cache_root = Path(cache_root).resolve()
    directory = cache_root / remote_root(recovery_id)
    if directory.parent.is_symlink():
        raise ValueError("Recovery cache parent must not be a symlink")
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "output"
    output.mkdir()
    atomic_json(
        directory / "manifest.json",
        {
            "schema": SCHEMA,
            "recoveryId": recovery_id,
            "mode": mode,
            "status": "running",
            "finalized": False,
            "files": [],
        },
    )
    return directory, output


def output_problems(output, mode, files, require_registration=False):
    """Check export completeness independently of the subprocess exit code."""
    problems = []
    required = {"modal-run.json", "inference.log"}
    required |= (
        {"sequence.json", "motion.json", "source-poses.pt", "missing-poses.json"}
        if mode == "motion"
        else {"person-posed.ply", "canonical-state.pt", "result.json"}
    )
    if require_registration:
        required.add("registration.json")
    for name in sorted(required):
        if name not in files or files[name]["bytes"] == 0:
            problems.append(f"missing or empty required output: {name}")
    for name in ("modal-run.json", "result.json" if mode == "frozen" else "motion.json"):
        if name in files:
            try:
                metadata = json.loads((output / name).read_text())
                if not isinstance(metadata, dict):
                    raise TypeError("Expected metadata object")
                if name == "modal-run.json" and metadata.get("error"):
                    problems.append("worker report contains an inference error")
            except (OSError, ValueError, TypeError):
                problems.append(f"malformed required metadata: {name}")
    if mode == "motion" and "sequence.json" in files:
        try:
            sequence = json.loads((output / "sequence.json").read_text())
            motion = json.loads((output / "motion.json").read_text())
            if not isinstance(sequence, dict) or not isinstance(motion, dict):
                raise TypeError("Expected motion and sequence objects")
            missing = json.loads((output / "missing-poses.json").read_text())
            if (
                not isinstance(missing, list)
                or missing != motion.get("missing")
                or missing != sequence.get("missingPoseSamples")
            ):
                raise ValueError("Inconsistent missing pose coverage")
            frames, hashes = sequence["frames"], sequence["frame_sha256"]
            if (
                not isinstance(frames, list)
                or not frames
                or sequence.get("count") != len(frames)
                or not isinstance(hashes, list)
                or len(hashes) != len(frames)
                or any(not isinstance(name, str) for name in frames)
                or len(set(frames)) != len(frames)
                or len(sequence.get("timestamps", [])) != len(frames)
                or len(sequence.get("sourceIndices", [])) != len(frames)
                or not isinstance(motion.get("frames"), list)
                or len(motion["frames"]) != len(frames)
            ):
                raise ValueError("inconsistent frame coverage")
            for name, digest in zip(frames, hashes):
                safe_relative(name)
                if (
                    not name.endswith(".ply")
                    or name not in files
                    or files[name]["sha256"] != digest
                ):
                    problems.append(f"sequence frame missing or hash mismatch: {name}")
            if set(frames) != {name for name in files if name.endswith(".ply")}:
                problems.append("exported PLY files differ from declared sequence frames")
            # A sample can be missing because the tracked person is not in it yet, which is
            # not a failure to reconstruct anything: a runner entering the shot has no pose in
            # the first frames, and the stage says so per sample. What must not be missing is a
            # sample nothing accounts for. Requiring every requested sample to have a frame
            # failed a clip whose actor arrives 83 ms in, after the GPU work had been paid for.
            explained = [
                sample
                for sample in missing
                if TRACK_ABSENT in str((sample or {}).get("reason", ""))
            ]
            if len(explained) != len(missing) or sequence.get("requestedSamples") != len(
                frames
            ) + len(missing):
                problems.append("source motion coverage is partial or unverified")
        except (OSError, ValueError, KeyError, TypeError):
            problems.append("motion/sequence metadata is missing, malformed or inconsistent")
    return problems


def finish_checkpoint(directory, *, mode, error=None, require_registration=False):
    directory = Path(directory)
    output = directory / "output"
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Generated output symlinks cannot be checkpointed")
        if not path.is_file():
            continue
        name = safe_relative(path.relative_to(output).as_posix())
        files[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    problems = output_problems(output, mode, files, require_registration)
    manifest = {
        "schema": SCHEMA,
        "recoveryId": directory.name,
        "mode": mode,
        "status": "failed" if error else "partial" if problems else "complete",
        "finalized": True,
        "files": files,
        "problems": problems,
        "inferenceError": bool(error),
        "requireRegistration": bool(require_registration),
        "finishedAtEpoch": time.time(),
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest, receipt):
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("recoveryId") != receipt["recoveryId"]
        or manifest.get("mode") != receipt["mode"]
        or manifest.get("status") not in ("complete", "partial", "failed")
        or manifest.get("finalized") is not True
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
        or not isinstance(manifest.get("problems"), list)
        or any(not isinstance(reason, str) for reason in manifest.get("problems", []))
        or type(manifest.get("inferenceError")) is not bool
        or type(manifest.get("requireRegistration")) is not bool
        or (
            manifest.get("status") == "complete"
            and (manifest.get("problems") or manifest.get("inferenceError"))
        )
    ):
        raise ValueError(
            "Checkpoint is missing, running or incomplete; retained files are not certified. Do not resubmit inference"
        )
    for name, item in manifest["files"].items():
        safe_relative(name)
        if (
            not isinstance(item, dict)
            or type(item.get("bytes")) is not int
            or item["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
        ):
            raise ValueError("Malformed output size or hash in recovery manifest")


def local_target(destination, relative):
    safe_relative(relative)
    target = destination / relative
    current = destination
    if destination.is_symlink():
        raise ValueError("Recovery destination must not be a symlink")
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Recovery output path must not contain symlinks")
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Recovery output escapes destination")
    return target


def recover_outputs(receipt_path, volume, destination=None):
    """Download verified output files only; no FunctionCall or inference API is used."""
    with receipt_lock(receipt_path):
        return _recover_outputs(receipt_path, volume, destination)


def _recover_outputs(receipt_path, volume, destination=None):
    receipt_path = Path(receipt_path)
    receipt = load_receipt(receipt_path)
    data = bytearray()
    try:
        for chunk in volume.read_file(receipt["remotePath"] + "/manifest.json"):
            data.extend(chunk)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("Recovery manifest exceeds 8 MiB")
    except Exception as error:
        raise ValueError(
            "Checkpoint manifest unavailable; preserve the receipt and do not resubmit inference"
        ) from error
    digest = hashlib.sha256(data).hexdigest()
    if receipt.get("manifestSha256") and receipt["manifestSha256"] != digest:
        raise ValueError("Recovery manifest differs from completed call receipt")
    try:
        manifest = json.loads(data)
    except ValueError:
        raise ValueError("Recovery manifest is not valid JSON") from None
    validate_manifest(manifest, receipt)
    destination = Path(destination or receipt["destination"])
    if destination.is_symlink():
        raise ValueError("Recovery destination must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    for name, item in manifest["files"].items():
        target = local_target(destination, name)
        if target.exists():
            if (
                not target.is_file()
                or target.stat().st_size != item["bytes"]
                or file_hash(target) != item["sha256"]
            ):
                raise ValueError(f"Existing output conflicts with recovery manifest: {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.part")
        try:
            received, computed = 0, hashlib.sha256()
            with temporary.open("xb") as stream:
                for chunk in volume.read_file(receipt["remotePath"] + "/output/" + name):
                    received += len(chunk)
                    if received > item["bytes"]:
                        raise ValueError(f"Recovery output exceeds manifest size: {name}")
                    computed.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if received != item["bytes"] or computed.hexdigest() != item["sha256"]:
                raise ValueError(f"Recovery output size/hash mismatch: {name}")
            # Link atomically without replacing a file created during transfer.
            os.link(temporary, target)
        finally:
            # Delete only this invocation's partial; other receipts may target the same output.
            temporary.unlink(missing_ok=True)
    problems = output_problems(
        destination, receipt["mode"], manifest["files"], manifest["requireRegistration"]
    )
    if manifest["status"] == "complete" and problems:
        raise ValueError(
            "Checkpoint claimed complete but exported output contract failed: "
            + "; ".join(problems)
        )
    snapshot = destination / "recovery-manifest.json"
    if snapshot.is_symlink():
        raise ValueError("Recovered manifest must not be a symlink")
    if snapshot.exists() and snapshot.read_bytes() != data:
        raise ValueError("Existing recovered manifest conflicts with this checkpoint")
    if not snapshot.exists():
        with snapshot.open("xb") as stream:
            stream.write(data)
    receipt.update(
        status="recovered",
        resultStatus=manifest["status"],
        manifestSha256=digest,
        recoveredDestination=str(destination.resolve()),
        recoveredAtEpoch=time.time(),
    )
    atomic_json(receipt_path, receipt)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument(
        "--out", help="Optional recovery destination; matching files resume, conflicting files fail"
    )
    args = parser.parse_args()
    # Validate before creating even a read-only provider handle.
    load_receipt(args.receipt)
    import modal

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    manifest = recover_outputs(args.receipt, volume, args.out)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "files": len(manifest["files"]),
                "problems": manifest["problems"],
            },
            indent=2,
        )
    )
    if manifest["status"] != "complete":
        raise SystemExit("Partial/failed outputs recovered; reconstruction is not complete")


if __name__ == "__main__":
    main()
