"""Explicit pinned, CPU-only object cache staging. No GPU image imports or inference.

modal run worker/modal_stage_object_weights.py --manifest /absolute/manifest.json \
  --receipt /absolute/submission.json --deadline-epoch UNIX_SECONDS

A failed or started receipt requires operator reconciliation; changing the manifest never
resets the remote attempt. Completed matching attempts only verify retained file hashes.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import tempfile
import time
import urllib.request

import modal

SCHEMA = "wander.object-cache-manifest/1"
RECEIPT_SCHEMA = "wander.object-cache-attempt/1"
VOLUME = "wander-image-to-3d-cache"
TRELLIS = "microsoft/TRELLIS-image-large"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
DINO_PATH = "torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
SDXL_REQUIRED = {
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
}

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.28.1")
app = modal.App("wander-object-cache-setup")
cache = modal.Volume.from_name(VOLUME)


def manifest_hash(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_manifest(manifest):
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "snapshots", "dino"}
        or manifest["schema"] != SCHEMA
    ):
        raise ValueError("Invalid object cache manifest")
    snapshots = manifest["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != 2:
        raise ValueError("Exactly TRELLIS and SDXL snapshots are required")
    seen = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or set(snapshot) != {"repoId", "revision", "files"}:
            raise ValueError("Invalid pinned snapshot fields")
        repo, revision, files = snapshot["repoId"], snapshot["revision"], snapshot["files"]
        if (
            repo not in (TRELLIS, SDXL)
            or repo in seen
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
        ):
            raise ValueError("Snapshot must name an allowed repository and exact commit")
        seen.add(repo)
        if not isinstance(files, list) or not 1 <= len(files) <= 64:
            raise ValueError("Snapshot requires a bounded explicit file list")
        for name in files:
            if (
                not isinstance(name, str)
                or len(name) > 200
                or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or str(PurePosixPath(name)) != name
            ):
                raise ValueError("Snapshot files must be exact relative paths without patterns")
        if len(set(files)) != len(files):
            raise ValueError("Duplicate snapshot file")
        if repo == SDXL and set(files) != SDXL_REQUIRED:
            raise ValueError("SDXL requires exactly the fp16 pipeline files")
        if repo == TRELLIS:
            if "pipeline.json" not in files or any(
                name != "pipeline.json"
                and not re.fullmatch(r"ckpts/[A-Za-z0-9_-]+\.(json|safetensors)", name)
                for name in files
            ):
                raise ValueError("TRELLIS requires pipeline.json and explicit checkpoint files")
            stems = {
                name.removesuffix(".json")
                for name in files
                if name.startswith("ckpts/") and name.endswith(".json")
            }
            if not stems or set(files) != {"pipeline.json"} | {
                stem + suffix for stem in stems for suffix in (".json", ".safetensors")
            }:
                raise ValueError("TRELLIS checkpoint config and weight pairs are required")
    dino = manifest["dino"]
    if (
        not isinstance(dino, dict)
        or set(dino) != {"url", "sha256", "bytes"}
        or dino["url"] != DINO_URL
        or not isinstance(dino["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", dino["sha256"])
        or type(dino["bytes"]) is not int
        or not 0 < dino["bytes"] <= 2 * 1024**3
    ):
        raise ValueError("DINO requires the official URL, exact SHA256 and bounded byte count")
    return manifest


def validate_deadline(deadline_epoch):
    if (
        type(deadline_epoch) not in (int, float)
        or not math.isfinite(deadline_epoch)
        or not 0 < deadline_epoch - time.time() <= 900
    ):
        raise ValueError("An absolute future deadline within 900 seconds is required")


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_receipt(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Receipt symlinks are not allowed")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".object-cache-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def deadline_guard(deadline_epoch):
    validate_deadline(deadline_epoch)

    def expired(*_):
        raise TimeoutError("Object cache staging deadline exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline_epoch - time.time() - 10))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def file_record(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing or escaping cache file")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest_file(path),
    }


def download_dino(root, pin):
    target = Path(root) / DINO_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if (
            target.is_symlink()
            or target.stat().st_size != pin["bytes"]
            or digest_file(target) != pin["sha256"]
        ):
            raise ValueError("Existing DINO checkpoint conflicts with the pinned manifest")
        return target
    pending = target.with_suffix(".download")
    with pending.open("xb") as output:
        with urllib.request.urlopen(pin["url"], timeout=30) as response:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > pin["bytes"]:
                    raise ValueError("DINO download exceeds pinned byte count")
                output.write(chunk)
    if total != pin["bytes"] or digest_file(pending) != pin["sha256"]:
        raise ValueError("DINO download does not match pinned size and SHA256")
    pending.replace(target)
    return target


def stage_cache(manifest, deadline_epoch, *, root, commit, downloader, call_id=None):
    """Local-testable staging core. Caller serializes invocations for this volume."""
    validate_manifest(manifest)
    validate_deadline(deadline_epoch)
    root = Path(root).resolve()
    receipt = root / "setup-receipts/object-weights-attempt.json"
    pin = manifest_hash(manifest)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get("status") != "completed" or saved.get("manifestSha256") != pin:
            raise RuntimeError("Prior cache attempt requires reconciliation; no automatic repeat")
        for record in saved["files"]:
            if file_record(root, root / record["path"]) != record:
                raise ValueError("Completed object cache changed; reconciliation required")
        return {key: value for key, value in saved.items() if key != "files"}
    report = {
        "schema": RECEIPT_SCHEMA,
        "status": "started",
        "manifestSha256": pin,
        "volume": VOLUME,
        "startedEpoch": time.time(),
        "deadlineEpoch": deadline_epoch,
        "functionCallId": call_id,
        "cpuLimit": 2,
        "memoryLimitMiB": 4096,
        "gpu": None,
        "files": [],
    }
    write_receipt(receipt, report)
    commit()  # Durable reservation before any download.
    try:
        with deadline_guard(deadline_epoch):
            hub = root / "huggingface/hub"
            for snapshot in manifest["snapshots"]:
                repo, revision = snapshot["repoId"], snapshot["revision"]
                model = hub / ("models--" + repo.replace("/", "--"))
                ref = model / "refs/main"
                if ref.exists() and ref.read_text().strip() != revision:
                    raise ValueError("Existing model main reference conflicts with pinned revision")
                for name in snapshot["files"]:
                    path = Path(
                        downloader(
                            repo_id=repo,
                            revision=revision,
                            filename=name,
                            cache_dir=str(hub),
                            token=False,
                        )
                    )
                    if path != model / "snapshots" / revision / name:
                        raise ValueError("Downloader returned an unexpected cache path")
                    report["files"].append(file_record(root, path))
                if repo == TRELLIS:
                    pipeline = json.loads(
                        (model / "snapshots" / revision / "pipeline.json").read_text()
                    )
                    models = pipeline.get("args", {}).get("models")
                    if (
                        not isinstance(models, dict)
                        or not models
                        or any(
                            not isinstance(value, str)
                            or {value + ".json", value + ".safetensors"} - set(snapshot["files"])
                            for value in models.values()
                        )
                    ):
                        raise ValueError(
                            "TRELLIS pipeline references checkpoints absent from manifest"
                        )
                ref.parent.mkdir(parents=True, exist_ok=True)
                ref.write_text(revision)
                report["files"].append(file_record(root, ref))
            report["files"].append(file_record(root, download_dino(root, manifest["dino"])))
            report.update(
                status="completed",
                finishedEpoch=time.time(),
                fileCount=len(report["files"]),
                bytes=sum(item["bytes"] for item in report["files"]),
                inventorySha256=manifest_hash(report["files"]),
                snapshots=[
                    {
                        "repoId": item["repoId"],
                        "revision": item["revision"],
                        "fileCount": len(item["files"]),
                    }
                    for item in manifest["snapshots"]
                ],
            )
        write_receipt(receipt, report)
        commit()
    except BaseException as error:
        report.update(status="failed", finishedEpoch=time.time(), errorType=type(error).__name__)
        write_receipt(receipt, report)
        commit()
        raise RuntimeError(
            f"Object cache staging failed ({type(error).__name__}); inspect retained receipt"
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
    from huggingface_hub import hf_hub_download

    cache.reload()
    return stage_cache(
        manifest,
        deadline_epoch,
        root="/cache",
        commit=cache.commit,
        downloader=hf_hub_download,
        call_id=modal.current_function_call_id(),
    )


@app.local_entrypoint()
def main(manifest: str, receipt: str, deadline_epoch: float):
    import fcntl

    manifest_path, receipt_path = Path(manifest), Path(receipt)
    if not manifest_path.is_absolute() or not receipt_path.is_absolute():
        raise ValueError("Manifest and submission receipt paths must be absolute")
    document = validate_manifest(json.loads(manifest_path.read_text()))
    validate_deadline(deadline_epoch)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(receipt_path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt_path.exists() or receipt_path.is_symlink():
            raise RuntimeError("Submission receipt exists; inspect/reconcile without resubmission")
        record = {
            "schema": RECEIPT_SCHEMA,
            "status": "started",
            "manifestSha256": manifest_hash(document),
            "startedEpoch": time.time(),
            "deadlineEpoch": deadline_epoch,
        }
        write_receipt(receipt_path, record)
        try:
            result = stage.remote(document, deadline_epoch)
        except BaseException as error:
            record.update(
                status="failed", errorType=type(error).__name__, finishedEpoch=time.time()
            )
            write_receipt(receipt_path, record)
            raise RuntimeError(
                "Object cache submission failed or is ambiguous; reconcile receipt before any retry"
            ) from None
        record.update(status="completed", result=result, finishedEpoch=time.time())
        write_receipt(receipt_path, record)
        print(json.dumps(result, indent=2))
