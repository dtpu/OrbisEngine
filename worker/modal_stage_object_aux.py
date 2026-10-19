"""Pinned CPU-only DINO source/U2Net setup, separate from primary model staging."""

from pathlib import Path
import hashlib
import json
import re
import tarfile
import time
import urllib.request

import modal

try:
    from worker.modal_stage_object_weights import (
        deadline_guard,
        digest_file,
        file_record,
        manifest_hash,
        validate_deadline,
        write_receipt,
    )
except ModuleNotFoundError:
    from modal_stage_object_weights import (
        deadline_guard,
        digest_file,
        file_record,
        manifest_hash,
        validate_deadline,
        write_receipt,
    )

SCHEMA = "wander.object-aux-manifest/1"
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"  # rembg v2.0.63 sessions/u2net.py
# Modal mounts this entrypoint alone; make its CPU-only helper an explicit image input.
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    Path(__file__).with_name("modal_stage_object_weights.py"),
    "/root/modal_stage_object_weights.py",
    copy=True,
)
app = modal.App("wander-object-aux-cache-setup")
cache = modal.Volume.from_name("wander-image-to-3d-cache")


def validate_manifest(doc):
    if (
        not isinstance(doc, dict)
        or set(doc) != {"schema", "dinoCode", "u2net"}
        or doc["schema"] != SCHEMA
    ):
        raise ValueError("Invalid object auxiliary manifest")
    code, model = doc["dinoCode"], doc["u2net"]
    if (
        not isinstance(code, dict)
        or set(code) != {"revision", "archiveSha256", "bytes"}
        or not re.fullmatch(r"[0-9a-f]{40}", str(code["revision"]))
        or not re.fullmatch(r"[0-9a-f]{64}", str(code["archiveSha256"]))
        or type(code["bytes"]) is not int
        or not 0 < code["bytes"] <= 20 * 1024**2
    ):
        raise ValueError("DINO source requires a pinned commit, archive SHA256 and size")
    if (
        not isinstance(model, dict)
        or set(model) != {"url", "md5", "bytes"}
        or model["url"] != U2NET_URL
        or model["md5"] != U2NET_MD5
        or type(model["bytes"]) is not int
        or not 0 < model["bytes"] <= 512 * 1024**2
    ):
        raise ValueError("U2Net requires the official rembg URL/checksum and bounded size")
    return doc


def download(url, path, expected_bytes, algorithm, expected_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.new(algorithm)
    total = 0
    with path.open("xb") as output, urllib.request.urlopen(url, timeout=30) as response:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > expected_bytes:
                raise ValueError("Auxiliary download exceeds pinned size")
            digest.update(chunk)
            output.write(chunk)
    if total != expected_bytes or digest.hexdigest() != expected_hash:
        raise ValueError("Auxiliary download checksum/size mismatch")


def extract_code(archive_path, destination, revision):
    destination.mkdir(parents=True, exist_ok=False)
    prefix = "dinov2-" + revision
    total = count = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] != prefix
                or ".." in parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unexpected DINO archive member")
            total += member.size
            count += 1
            if total > 100 * 1024**2 or count > 2000:
                raise ValueError("DINO source archive exceeds extraction limit")
            target = destination.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    if not (destination / "hubconf.py").is_file():
        raise ValueError("DINO source archive has no hub entrypoint")


def stage_aux(manifest, deadline_epoch, *, root, commit, call_id=None):
    validate_manifest(manifest)
    validate_deadline(deadline_epoch)
    root = Path(root).resolve()
    receipt = root / "setup-receipts/object-aux-attempt.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("status") != "completed" or old.get("manifestSha256") != manifest_hash(manifest):
            raise RuntimeError("Auxiliary setup requires reconciliation; no automatic repeat")
        for record in old["files"]:
            if file_record(root, root / record["path"]) != record:
                raise ValueError("Retained auxiliary cache changed")
        return {key: value for key, value in old.items() if key != "files"}
    revision = manifest["dinoCode"]["revision"]
    code = root / "torch/hub" / ("dinov2-" + revision)
    report = {
        "schema": "wander.object-aux-attempt/1",
        "status": "started",
        "manifestSha256": manifest_hash(manifest),
        "dinoRevision": revision,
        "dinoCodePath": str(code.relative_to(root)),
        "startedEpoch": time.time(),
        "deadlineEpoch": deadline_epoch,
        "functionCallId": call_id,
        "files": [],
    }
    write_receipt(receipt, report)
    commit()
    try:
        with deadline_guard(deadline_epoch):
            if code.exists() or code.is_symlink():
                raise ValueError("DINO source destination already exists; reconcile retained files")
            archive_path = root / "setup-receipts/dinov2-source.tar.gz"
            pin = manifest["dinoCode"]
            download(
                "https://codeload.github.com/facebookresearch/dinov2/tar.gz/" + revision,
                archive_path,
                pin["bytes"],
                "sha256",
                pin["archiveSha256"],
            )
            extract_code(archive_path, code, revision)
            model = root / "u2net/u2net.onnx"
            pin = manifest["u2net"]
            if model.exists() or model.is_symlink():
                raise ValueError("U2Net destination already exists; reconcile retained files")
            pending = model.with_suffix(".download")
            download(pin["url"], pending, pin["bytes"], "md5", pin["md5"])
            pending.replace(model)
            report["files"] = [
                file_record(root, path) for path in sorted(code.rglob("*")) if path.is_file()
            ]
            report["files"].append(file_record(root, model))
            report.update(
                status="completed",
                finishedEpoch=time.time(),
                fileCount=len(report["files"]),
                inventorySha256=manifest_hash(report["files"]),
                u2netSha256=digest_file(model),
            )
        write_receipt(receipt, report)
        commit()
    except BaseException as error:
        report.update(status="failed", finishedEpoch=time.time(), errorType=type(error).__name__)
        write_receipt(receipt, report)
        commit()
        raise RuntimeError(
            f"Auxiliary setup failed ({type(error).__name__}); reconcile receipt"
        ) from None
    return {key: value for key, value in report.items() if key != "files"}


@app.function(
    image=image,
    cpu=(1, 2),
    memory=(1024, 4096),
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": cache},
)
def stage(manifest: dict, deadline_epoch: float):
    cache.reload()
    return stage_aux(
        manifest,
        deadline_epoch,
        root="/cache",
        commit=cache.commit,
        call_id=modal.current_function_call_id(),
    )


@app.local_entrypoint()
def main(manifest: str, receipt: str, deadline_epoch: float):
    import fcntl

    manifest, receipt = Path(manifest), Path(receipt)
    if not manifest.is_absolute() or not receipt.is_absolute():
        raise ValueError("Manifest and receipt must be absolute paths")
    doc = validate_manifest(json.loads(manifest.read_text()))
    validate_deadline(deadline_epoch)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(receipt) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists() or receipt.is_symlink():
            raise RuntimeError("Existing submission receipt requires reconciliation")
        record = {
            "status": "started",
            "manifestSha256": manifest_hash(doc),
            "deadlineEpoch": deadline_epoch,
            "startedEpoch": time.time(),
        }
        write_receipt(receipt, record)
        try:
            result = stage.remote(doc, deadline_epoch)
            record.update(status="completed", result=result)
        except BaseException as error:
            record.update(status="failed", errorType=type(error).__name__)
            write_receipt(receipt, record)
            raise RuntimeError(
                "Auxiliary submission failed or ambiguous; reconcile receipt"
            ) from None
        write_receipt(receipt, record)
        print(json.dumps(result, indent=2))
