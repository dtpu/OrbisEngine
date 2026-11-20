"""Explicit CPU-only model cache staging; never performs inference or starts GPU workers.

Run the local entrypoint with the intended Modal profile and an absolute deadline. Each phase
runs once, preserves a failure receipt, and refuses to restart an incomplete copy/extraction
without operator reconciliation.
Public upstream URLs/revisions are recorded in receipts; source video is never uploaded.
"""

from pathlib import Path

import modal

LAMA_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)
PRIOR_URL = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar"
PRIOR_BYTES = 18_818_365_440
SEGFORMER_REV = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
LHM_REV = "dd6392905187a91fd67b3f6962aa74481e943764"
MASKRCNN_URL = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
RATE_CEILING = 2 * 0.0000131 + 4 * 0.00000222

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==0.28.1")
app = modal.App("wander-daniel-overnight-cache-setup")
VOLUMES = {
    "/source": modal.Volume.from_name("wander-weights"),
    "/clean": modal.Volume.from_name("wander-clean-video-cache", create_if_missing=True),
    "/motion": modal.Volume.from_name("wander-overnight-motion-cache", create_if_missing=True),
    "/lhm": modal.Volume.from_name("wander-overnight-lhm-cache", create_if_missing=True),
}
CLEAN_MOTION_PHASES = ("clean-motion", "clean-motion-repair", "clean-motion-repair-v2")
PHASE_MOUNTS = {
    **dict.fromkeys(CLEAN_MOTION_PHASES, ("/source", "/clean", "/motion")),
    "clean-only": ("/clean",),
    "clean-instance": ("/clean", "/lhm"),
    **dict.fromkeys(("lhm", "lhm-resume", "dino", "gfpgan"), ("/lhm",)),
}


def digest_file(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root, *, skip_git_metadata=False):
    """Hash real files and preserve/validate links without duplicating blob bytes in totals."""
    records = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if skip_git_metadata and ".git" in relative.parts:
            continue
        name = str(relative)
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_relative_to(Path(root).resolve()):
                raise ValueError("cache has a broken or escaping symlink")
            records.append({"path": name, "symlink": str(path.readlink())})
        elif path.is_file():
            records.append(
                {"path": name, "bytes": path.stat().st_size, "sha256": digest_file(path)}
            )
    return {
        "files": records,
        "fileCount": len(records),
        "bytes": sum(r.get("bytes", 0) for r in records),
    }


def write_receipt(path, document):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2))
    temporary.replace(path)


def extract_priors(response, destination, expected_bytes):
    """Hash and extract one streamed public tar with Python3.12's safe data filter."""
    import hashlib
    import tarfile

    class Reader:
        def __init__(self):
            self.bytes = 0
            self.digest = hashlib.sha256()

        def read(self, size=-1):
            chunk = response.read(size)
            self.bytes += len(chunk)
            self.digest.update(chunk)
            if self.bytes > expected_bytes:
                raise ValueError("prior archive exceeds recorded content length")
            return chunk

    reader = Reader()
    count = 0
    skipped_git_entries = skipped_git_bytes = 0
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            normalized = Path(member.name)
            parts = normalized.parts
            if not parts and member.isdir():
                continue
            if not parts or parts[0] not in ("pretrained_models", "gfpgan") or ".." in parts:
                raise ValueError("unexpected prior archive root")
            if ".git" in parts:
                skipped_git_entries += 1
                skipped_git_bytes += member.size
                continue
            archive.extract(member, path=destination, filter="data")
            count += 1
            if count % 25 == 0:
                print(
                    f"extracted {count} archive entries; received {reader.bytes} bytes", flush=True
                )
    while reader.read(4 * 1024 * 1024):
        pass
    if reader.bytes != expected_bytes:
        raise ValueError("prior archive content length mismatch")
    return {
        "archiveBytes": reader.bytes,
        "archiveSha256": reader.digest.hexdigest(),
        "archiveEntries": count,
        "skippedGitEntries": skipped_git_entries,
        "skippedGitBytes": skipped_git_bytes,
    }


def extract_missing_priors(url, destination, expected_bytes):
    """Continue an interrupted uncompressed tar using pinned ranges for missing runtime files.

    The resumed receipt has no whole-archive digest. Inventory hashes final runtime files instead.
    """
    import tarfile
    import urllib.request

    range_chunk_bytes = 16 * 1024 * 1024
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as head:
        etag = head.headers.get("ETag")
        if not etag or int(head.headers["Content-Length"]) != expected_bytes:
            raise ValueError("range recovery requires a known-size object and ETag")

    class RangeReader:
        def __init__(self):
            self.position = 0
            self.bytes = 0
            self.requests = 0

        def tell(self):
            return self.position

        def seek(self, offset, whence=0):
            origin = {0: 0, 1: self.position, 2: expected_bytes}[whence]
            position = origin + offset
            if not 0 <= position <= expected_bytes:
                raise ValueError("archive seek outside recorded object")
            self.position = position
            return position

        def read(self, size):
            if size < 0:
                raise ValueError("unbounded range read refused")
            chunks = []
            remaining = min(size, expected_bytes - self.position)
            while remaining:
                # A header read must stop before the member payload, which may be skipped.
                count = min(remaining, range_chunk_bytes)
                end = self.position + count - 1
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={self.position}-{end}", "If-Match": etag}
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected_range = f"bytes {self.position}-{end}/{expected_bytes}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                        or response.headers.get("ETag") != etag
                    ):
                        raise ValueError("upstream did not honor pinned byte range")
                    chunk = response.read(count + 1)
                if len(chunk) != count:
                    raise ValueError("range response length mismatch")
                chunks.append(chunk)
                self.position += count
                self.bytes += count
                self.requests += 1
                remaining -= count
            return b"".join(chunks)

    reader = RangeReader()
    reused = extracted = 0
    skipped_git_entries = skipped_git_bytes = 0
    with tarfile.open(fileobj=reader, mode="r:") as archive:
        archive.copybufsize = range_chunk_bytes
        for member in archive:
            parts = Path(member.name).parts
            if not parts and member.isdir():
                continue
            if not parts or parts[0] not in ("pretrained_models", "gfpgan") or ".." in parts:
                raise ValueError("unexpected prior archive root during recovery")
            if ".git" in parts:
                skipped_git_entries += 1
                skipped_git_bytes += member.size
                # TarFile.next otherwise probes the last byte of the skipped member's block.
                reader.seek(archive.offset)
                continue
            path = Path(destination) / member.name
            if (
                member.isfile()
                and path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(Path(destination).resolve())
                and path.stat().st_size == member.size
            ):
                reused += 1
            else:
                archive.extract(member, path=destination, filter="data")
                extracted += 1
            if (reused + extracted) % 25 == 0:
                print(
                    f"range recovery: reused {reused}, extracted {extracted}, received {reader.bytes} bytes",
                    flush=True,
                )
    return {
        "archiveBytes": expected_bytes,
        "archiveSha256": None,
        "sourceETag": etag,
        "rangeBytesReceived": reader.bytes,
        "rangeRequests": reader.requests,
        "reusedMembers": reused,
        "extractedMembers": extracted,
        "skippedGitEntries": skipped_git_entries,
        "skippedGitBytes": skipped_git_bytes,
    }


# Modal hydrates static mounts before main executes. Select mounts only on the invoked variant.
@app.function(
    image=image,
    cpu=(1, 2),
    memory=(1024, 4096),
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def stage(phase: str, deadline_epoch: float):
    import json
    import shutil
    import signal
    import time
    import urllib.request

    from huggingface_hub import snapshot_download

    if phase not in PHASE_MOUNTS:
        raise ValueError("unknown staging phase")
    if not all(Path(mount).is_dir() for mount in PHASE_MOUNTS[phase]):
        raise RuntimeError("staging phase mounts missing; invoke the main local entrypoint")
    remaining = min(1750, int(deadline_epoch - time.time()))
    if remaining <= 0:
        raise ValueError("setup deadline already exhausted")

    def deadline(*_):
        raise TimeoutError("cache staging deadline exhausted")

    started = time.time()
    report = {
        "phase": phase,
        "startedEpoch": started,
        "status": "started",
        "gpu": None,
        "cpuLimit": 2,
        "memoryLimitMiB": 4096,
        "functionCallId": modal.current_function_call_id(),
        "steps": [],
    }
    receipt_mount = "/clean" if phase.startswith("clean-") else "/lhm"
    receipt_root = Path(receipt_mount) / "setup-receipts"
    failure = receipt_root / f"{phase}-attempt.json"
    if failure.exists():
        previous = json.loads(failure.read_text())
        if previous.get("status") != "complete":
            raise RuntimeError(
                "prior incomplete attempt requires reconciliation; no automatic retry"
            )
        return previous

    def commit_changes():
        VOLUMES[receipt_mount].commit()
        if phase in CLEAN_MOTION_PHASES:
            VOLUMES["/motion"].commit()

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(remaining)
    write_receipt(failure, report)
    commit_changes()

    def step_done(name, document):
        report["steps"].append({"name": name, **document})
        receipt_name = f"clean-only-{name}" if phase == "clean-only" else name
        write_receipt(receipt_root / f"{receipt_name}.json", document)
        write_receipt(failure, report)
        commit_changes()
        print(f"cache step complete: {name}", flush=True)

    try:
        if phase in CLEAN_MOTION_PHASES:
            pi_source = Path("/source/hub/models--yyfz233--Pi3X")
            pi_dest = Path("/motion/huggingface/hub/models--yyfz233--Pi3X")
            pending = pi_dest.with_name(pi_dest.name + ".incomplete")
            if phase.startswith("clean-motion-repair"):
                previous = json.loads((receipt_root / "clean-motion-attempt.json").read_text())
                if previous.get(
                    "error"
                ) != "cache has a broken or escaping symlink" or previous.get("steps"):
                    raise RuntimeError("repair does not match the diagnosed first attempt")
                if not pending.is_dir() or pi_dest.exists():
                    raise RuntimeError("repair cache state differs from recorded failure")
                repairs = []
                for blob in (pending / "blobs").iterdir():
                    if blob.is_symlink():
                        original = pi_source / "blobs" / blob.name
                        backing = original.resolve()
                        if not backing.is_file() or not backing.is_relative_to(
                            Path("/source/hub").resolve()
                        ):
                            raise ValueError(
                                f"source blob cannot be resolved safely: {original.readlink()}"
                            )
                        materialized = blob.with_name(blob.name + ".copying")
                        shutil.copyfile(backing, materialized)
                        checksum = digest_file(materialized)
                        if len(blob.name) == 64 and checksum != blob.name:
                            raise ValueError("source blob hash differs from HF content hash")
                        materialized.replace(blob)
                        repairs.append(
                            {
                                "blob": blob.name,
                                "sourceLink": str(original.readlink()),
                                "sourceBacking": str(backing),
                                "sha256": checksum,
                            }
                        )
                report["repairs"] = repairs
            elif pending.exists() or pi_dest.exists():
                raise RuntimeError("Pi3X destination exists without this attempt's receipt")
            else:
                pending.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(pi_source, pending, symlinks=True)
            pi_inventory = inventory(pending)
            if not pi_inventory["bytes"]:
                raise ValueError("copied Pi3X cache is empty")
            pending.rename(pi_dest)
            step_done(
                "pi3x",
                {
                    "sourceVolume": "wander-weights",
                    "source": str(pi_source),
                    "destination": str(pi_dest),
                    **pi_inventory,
                },
            )

        if phase == "clean-only" or phase in CLEAN_MOTION_PHASES:
            snapshot = snapshot_download(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                revision=SEGFORMER_REV,
                cache_dir="/clean/huggingface/hub",
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
                max_workers=1,
            )
            model_dir = Path(snapshot).parents[1]
            # The active workers resolve main; pin its local ref to the staged immutable revision.
            (model_dir / "refs").mkdir(exist_ok=True)
            (model_dir / "refs/main").write_text(SEGFORMER_REV)
            if phase in CLEAN_MOTION_PHASES:
                shutil.copytree(
                    model_dir, Path("/motion/huggingface/hub") / model_dir.name, symlinks=True
                )
            step_done(
                "segformer",
                {
                    "model": "nvidia/segformer-b0-finetuned-ade-512-512",
                    "revision": SEGFORMER_REV,
                    "snapshot": snapshot,
                    **inventory(model_dir),
                },
            )

            lama = Path("/clean/lama/big-lama.pt")
            lama.parent.mkdir(parents=True, exist_ok=True)
            partial = lama.with_suffix(".pt.incomplete")
            if lama.exists() or partial.exists():
                raise RuntimeError("LaMa destination exists without this attempt's receipt")
            with (
                urllib.request.urlopen(LAMA_URL, timeout=45) as response,
                partial.open("wb") as output,
            ):
                expected = int(response.headers["Content-Length"])
                shutil.copyfileobj(response, output, 4 * 1024 * 1024)
            if partial.stat().st_size != expected:
                raise ValueError("LaMa content length mismatch")
            checksum = digest_file(partial)
            partial.rename(lama)
            step_done(
                "lama",
                {"url": LAMA_URL, "destination": str(lama), "bytes": expected, "sha256": checksum},
            )
        elif phase == "clean-instance":
            checkpoint = Path("/lhm/torch/hub/checkpoints") / Path(MASKRCNN_URL).name
            destination = Path("/clean/torch/hub/checkpoints") / checkpoint.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected_sha = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"
            if not checkpoint.is_file() or digest_file(checkpoint) != expected_sha:
                raise ValueError("Staged Mask R-CNN source checksum mismatch")
            if destination.exists():
                if digest_file(destination) != expected_sha:
                    raise ValueError("Existing clean Mask R-CNN checksum mismatch")
            else:
                partial = destination.with_suffix(".pth.incomplete")
                if partial.exists():
                    raise RuntimeError("Incomplete clean instance copy requires reconciliation")
                shutil.copyfile(checkpoint, partial)
                if digest_file(partial) != expected_sha:
                    raise ValueError("Copied Mask R-CNN checksum mismatch")
                partial.rename(destination)
            step_done(
                "clean-instance",
                {
                    "sourceVolume": "wander-overnight-lhm-cache",
                    "source": str(checkpoint),
                    "destination": str(destination),
                    "bytes": destination.stat().st_size,
                    "sha256": expected_sha,
                },
            )
        elif phase in ("lhm", "lhm-resume"):
            destination = Path("/lhm/data")
            pending = Path("/lhm/data.incomplete")
            if phase == "lhm-resume":
                previous = json.loads((receipt_root / "lhm-attempt.json").read_text())
                if (
                    (previous.get("errorType"), previous.get("error"))
                    not in (
                        ("TimeoutError", "cache staging deadline exhausted"),
                        ("ValueError", "unexpected prior archive root"),
                    )
                    or previous.get("steps")
                    or not pending.is_dir()
                    or destination.exists()
                ):
                    raise RuntimeError("range recovery requires a diagnosed incomplete extraction")
                receipt = extract_missing_priors(PRIOR_URL, pending, PRIOR_BYTES)
            else:
                if destination.exists() or pending.exists():
                    raise RuntimeError("prior destination exists without this attempt's receipt")
                pending.mkdir()
                with urllib.request.urlopen(PRIOR_URL, timeout=45) as response:
                    if int(response.headers["Content-Length"]) != PRIOR_BYTES:
                        raise ValueError("upstream prior archive size changed")
                    receipt = extract_priors(response, pending, PRIOR_BYTES)
            for name in ("human_model_files", "gagatracker", "dense_sample_points", "voxel_grid"):
                if not (pending / "pretrained_models" / name).exists():
                    raise ValueError("required prior subtree missing")
            prior_inventory = inventory(pending, skip_git_metadata=True)
            pending.rename(destination)
            step_done(
                "lhm-priors",
                {"url": PRIOR_URL, "destination": str(destination), **receipt, **prior_inventory},
            )

            snapshot = snapshot_download(
                "3DAIGC/LHM-500M-HF",
                revision=LHM_REV,
                cache_dir="/lhm/huggingface/hub",
                max_workers=1,
            )
            step_done(
                "lhm-model",
                {
                    "model": "3DAIGC/LHM-500M-HF",
                    "revision": LHM_REV,
                    "snapshot": snapshot,
                    **inventory(Path(snapshot).parents[1]),
                },
            )
            checkpoint = Path("/lhm/torch/hub/checkpoints") / Path(MASKRCNN_URL).name
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            partial = checkpoint.with_suffix(".pth.incomplete")
            if checkpoint.exists() or partial.exists():
                raise RuntimeError("Mask R-CNN destination exists without this attempt's receipt")
            with (
                urllib.request.urlopen(MASKRCNN_URL, timeout=45) as response,
                partial.open("wb") as output,
            ):
                expected = int(response.headers["Content-Length"])
                shutil.copyfileobj(response, output, 4 * 1024 * 1024)
            checksum = digest_file(partial)
            if partial.stat().st_size != expected or not checksum.startswith("73cbd019"):
                raise ValueError("Mask R-CNN content length or upstream hash mismatch")
            partial.rename(checkpoint)
            step_done(
                "maskrcnn",
                {
                    "url": MASKRCNN_URL,
                    "destination": str(checkpoint),
                    "bytes": expected,
                    "sha256": checksum,
                },
            )
        else:
            url = DINO_URL if phase == "dino" else GFPGAN_URL
            expected_bytes = 1_217_607_321 if phase == "dino" else 348_632_874
            directory = Path(
                "/lhm/torch/hub/checkpoints" if phase == "dino" else "/lhm/gfpgan/weights"
            )
            checkpoint = directory / Path(url).name
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            partial = checkpoint.with_suffix(".pth.incomplete")
            if checkpoint.exists() or partial.exists():
                raise RuntimeError(f"{phase} destination exists without this attempt's receipt")
            with (
                urllib.request.urlopen(url, timeout=45) as response,
                partial.open("wb") as output,
            ):
                expected = int(response.headers["Content-Length"])
                if expected != expected_bytes:
                    raise ValueError(f"upstream {phase} model length changed")
                shutil.copyfileobj(response, output, 4 * 1024 * 1024)
            if partial.stat().st_size != expected:
                raise ValueError(f"{phase} content length mismatch")
            checksum = digest_file(partial)
            partial.rename(checkpoint)
            step_done(
                phase,
                {
                    "url": url,
                    "destination": str(checkpoint),
                    "bytes": expected,
                    "sha256": checksum,
                },
            )
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["errorType"] = type(error).__name__
        report["error"] = str(error).split("?")[0][:300]
    finally:
        signal.alarm(0)
        report["elapsedSeconds"] = time.time() - started
        report["computeUpperEstimateUSD"] = report["elapsedSeconds"] * RATE_CEILING
        write_receipt(failure, report)
        commit_changes()
    return report


@app.local_entrypoint()
def main(phase: str, deadline_epoch: float, out: str):
    import json

    if phase not in PHASE_MOUNTS:
        raise ValueError("unknown staging phase")
    volumes = {mount: VOLUMES[mount] for mount in PHASE_MOUNTS[phase]}
    report = stage.with_options(volumes=volumes).remote(phase, deadline_epoch)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))
    if report["status"] != "complete":
        raise RuntimeError(f"cache setup failed; inspect {path}")
