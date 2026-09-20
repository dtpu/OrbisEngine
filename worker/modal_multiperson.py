"""Multi-person tracking stage on the existing LHM CUDA image.

Reuses `worker/modal_lhm.py`'s image object verbatim, so no layer is rebuilt: the same
MultiHMR pose estimator that `lhm_animate.py` already calls per frame is run once here with
every detection kept, and `worker/stages/track_people.py` links them into tracks.

  uv run --locked modal run worker/modal_multiperson.py --video public/clips/elevator.mp4 \
      --cameras .context/mp/elevator/pi3x/cameras.json --out .context/mp/elevator/tracks
"""

from pathlib import Path

import modal

# Copied verbatim from worker/modal_lhm.py: an identical image spec hashes to the same image id,
# so nothing rebuilds. Kept inline because a Modal container only mounts the entrypoint module.
HERE = Path(__file__).resolve().parent
LHM_REV = "4f88aaeb3629249fbbddb4d0784a06962d9e1338"
MODEL_REV = "dd6392905187a91fd67b3f6962aa74481e943764"
LARGER_MODEL_REV = "92372582f660066b9f1b9513860744357265b3d5"
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "build-essential", "ninja-build", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1"
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    .uv_pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "xformers==0.0.26.post1",
        "numpy==1.23.5",
        "setuptools==74.0.0",
        "wheel",
        "ninja",
    )
    .uv_pip_install(
        "numpy==1.23.5",
        "scipy==1.14.1",
        "pillow==10.4.0",
        "opencv-python-headless==4.11.0.86",
        "einops",
        "roma",
        "accelerate",
        "smplx",
        "decord==0.6.0",
        "diffusers==0.32.0",
        "gsplat==1.4.0",
        "huggingface_hub==0.28.1",
        "safetensors",
        "imageio==2.34.1",
        "imageio-ffmpeg",
        "jaxtyping==0.2.38",
        "kornia==0.7.2",
        "loguru",
        "lpips",
        "matplotlib==3.8.4",
        "omegaconf",
        "plyfile==1.0.3",
        "pygltflib",
        "pyrender",
        "pyyaml",
        "requests",
        "timm==1.0.15",
        "trimesh==4.4.9",
        "transformers==4.41.2",
        "tqdm",
        "typeguard==2.13.3",
        "iopath",
        "fvcore",
        "rembg==2.0.63",
        "onnxruntime",
        "addict",
        "future",
        "lmdb",
        "scikit-image==0.22.0",
        "tb-nightly",
        "yapf",
    )
    .uv_pip_install("pip==24.3.1")
    .uv_pip_install("chumpy==0.70", extra_options="--no-build-isolation --no-deps")
    .run_commands(
        "git clone https://github.com/XPixelGroup/BasicSR.git /opt/basicsr && git -C /opt/basicsr checkout 8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a",
    )
    .uv_pip_install("/opt/basicsr", extra_options="--no-deps")
    .uv_pip_install(
        "gfpgan==1.3.8", "facexlib==0.3.0", "filterpy==1.4.5", extra_options="--no-deps"
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d && git -C /opt/pytorch3d checkout 89653419d0973396f3eff1a381ba09a07fffc2ed",
    )
    .uv_pip_install(
        "/opt/pytorch3d",
        extra_options="--no-build-isolation --no-deps",
        env={"CC": "gcc", "CXX": "g++", "CUB_HOME": "/usr/local/cuda/include"},
    )
    .run_commands(
        "git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git /opt/diff-gaussian-rasterization && git -C /opt/diff-gaussian-rasterization checkout 8829d14f814fccdaf840b7b0f3021a616583c0a1",
    )
    .uv_pip_install(
        "/opt/diff-gaussian-rasterization",
        extra_options="--no-build-isolation --no-deps",
        env={"CC": "gcc", "CXX": "g++"},
    )
    .run_commands(
        "git clone https://github.com/camenduru/simple-knn.git /opt/simple-knn && git -C /opt/simple-knn checkout 60f461f4a56b7967e5d8045bf92f8c33f36976d0",
    )
    .uv_pip_install(
        "/opt/simple-knn",
        extra_options="--no-build-isolation --no-deps",
        env={"CC": "gcc", "CXX": "g++"},
    )
    .run_commands(
        f"git clone https://github.com/aigc3d/LHM.git /opt/lhm && git -C /opt/lhm checkout {LHM_REV}",
    )
    .add_local_dir(HERE / "stages", "/root/stages", ignore=["__pycache__"])
)
app = modal.App("wander-multiperson")
cache = modal.Volume.from_name("wander-overnight-lhm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=65536,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def track(
    video: bytes,
    cameras: bytes | None,
    fps: float,
    det_thresh: float,
    gate: float,
    max_gap: int,
    min_track_samples: int,
    overlay_every: int,
):
    import io
    import os
    import subprocess
    import tarfile
    import tempfile
    import time
    import traceback

    root = Path(tempfile.mkdtemp(prefix="tracks-"))
    (root / "source.mp4").write_bytes(video)
    if cameras:
        (root / "cameras.json").write_bytes(cameras)
    out = root / "out"
    out.mkdir()
    prior = Path("/cache/data/pretrained_models")
    link = Path("/opt/lhm/pretrained_models")
    if not prior.exists():
        raise RuntimeError("Stage public LHM prior assets before GPU inference")
    if not link.exists():
        link.symlink_to(prior, target_is_directory=True)
    os.chdir("/opt/lhm")
    started = time.time()
    error = None
    cmd = [
        "python",
        "/root/stages/track_people.py",
        "--video",
        str(root / "source.mp4"),
        "--out",
        str(out),
        "--fps",
        str(fps),
        "--det-thresh",
        str(det_thresh),
        "--gate",
        str(gate),
        "--max-gap",
        str(max_gap),
        "--min-track-samples",
        str(min_track_samples),
        "--overlay-every",
        str(overlay_every),
    ]
    if cameras:
        cmd += ["--cameras", str(root / "cameras.json")]
    try:
        with (out / "track.log").open("w") as log:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait() != 0:
                raise RuntimeError(f"track_people exited {proc.returncode}")
    except Exception:  # noqa: BLE001 - retain tracking diagnostics for any inference failure
        error = traceback.format_exc()
        print(error, flush=True)
    cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(out)))
    seconds = time.time() - started
    report = {
        "seconds": seconds,
        "error": error,
        "gpu": "L4",
        "codeRevision": LHM_REV,
        "estimatedComputeUSD": seconds * (0.000222 + 4 * 0.0000131 + 64 * 0.00000222),
    }
    return {"report": report, "archive": buf.getvalue()}


@app.local_entrypoint()
def main(
    video: str,
    out: str,
    cameras: str = "",
    fps: float = 12,
    det_thresh: float = 0.15,
    gate: float = 0.22,
    max_gap: int = 8,
    min_track_samples: int = 8,
    overlay_every: int = 0,
):
    import io
    import json
    import tarfile

    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    result = track.remote(
        Path(video).read_bytes(),
        Path(cameras).read_bytes() if cameras else None,
        fps,
        det_thresh,
        gate,
        max_gap,
        min_track_samples,
        overlay_every,
    )
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        archive.extractall(dest, filter="data")
    (dest / "modal-run.json").write_text(json.dumps(result["report"], indent=2))
    print(json.dumps(result["report"], indent=2))
    if result["report"]["error"]:
        raise RuntimeError("tracking failed; see the downloaded track.log")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=65536,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def animate_track(
    inputs: dict,
    flags: list,
    recovery_id: str = "",
    execution_timeout: int = 0,
    recover_missing_poses: bool = False,
):
    """Drive one saved canonical with retained track poses and durable generated outputs."""
    import hashlib
    import tempfile
    import time
    import traceback

    from stages.lhm_execution import animation_options, link_model_caches, run_logged_inference
    from stages.lhm_recovery import (
        atomic_json,
        file_hash,
        finish_checkpoint,
        remote_root,
        start_checkpoint,
    )

    animation_options(execution_timeout=execution_timeout)
    start = time.monotonic()
    checkpoint, out = start_checkpoint("/cache", recovery_id, "motion")
    cache.commit()
    error = None
    model_id = "3DAIGC/LHM-500M-HF"
    try:
        from huggingface_hub import snapshot_download

        root = Path(tempfile.mkdtemp(prefix="lhm-track-"))
        prepared = root / "prepared"
        prepared.mkdir()
        allowed = {
            "source.mp4",
            "canonical-state.pt",
            "reference.json",
            "cameras.json",
            "depth-reference.ply",
            "seed-poses.pt",
            "seed-motion.json",
        }
        if set(inputs) != allowed:
            raise ValueError("Animation requires exactly the declared saved inputs")
        for name, data in inputs.items():
            (prepared / name).write_bytes(data)
        links = link_model_caches()
        atomic_json(out / "model-cache-links.json", links)
        model = snapshot_download(model_id, revision=MODEL_REV, local_files_only=True)
        cmd = [
            "python",
            "/root/stages/lhm_animate.py",
            "--video",
            str(prepared / "source.mp4"),
            "--canonical",
            str(prepared / "canonical-state.pt"),
            "--reference",
            str(prepared / "reference.json"),
            "--out",
            str(out),
            "--model",
            model,
            "--cameras",
            str(prepared / "cameras.json"),
            "--depth-reference",
            str(prepared / "depth-reference.ply"),
            "--seed-poses",
            str(prepared / "seed-poses.pt"),
            "--seed-motion",
            str(prepared / "seed-motion.json"),
            *([] if recover_missing_poses else ["--track-only"]),
            *flags,
        ]
        run_logged_inference(
            cmd,
            out / "inference.log",
            started=start,
            execution_timeout=execution_timeout,
            cwd="/opt/lhm",
        )
    except Exception:  # noqa: BLE001 - preserve generated outputs even after timeout/setup failure
        error = traceback.format_exc()
        print(error, flush=True)
        with (out / "inference.log").open("a") as log:
            log.write(error)
    elapsed = time.monotonic() - start
    report = {
        "seconds": elapsed,
        "error": error,
        "experiment": "track-canonical-source-motion",
        "gpu": "L4",
        "cpu": 4,
        "memoryGiB": 64,
        "timeoutSeconds": execution_timeout or 3600,
        "executionFlags": flags,
        "recoverMissingPoses": recover_missing_poses,
        "inputSha256": {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
        "codeRevision": LHM_REV,
        "model": model_id,
        "modelRevision": MODEL_REV,
        "estimatedComputeUSD": elapsed * (0.000222 + 4 * 0.0000131 + 64 * 0.00000222),
        "timingScope": "Worker setup and inference; excludes final output hashing and volume commit",
        "recoveryId": recovery_id,
        "remotePath": remote_root(recovery_id),
    }
    atomic_json(out / "modal-run.json", report)
    manifest = finish_checkpoint(checkpoint, mode="motion", error=error, require_registration=True)
    cache.commit()
    return {
        "report": report,
        "recoveryId": recovery_id,
        "remotePath": remote_root(recovery_id),
        "manifestSha256": file_hash(checkpoint / "manifest.json"),
        "status": manifest["status"],
        "totalWorkerSeconds": time.monotonic() - start,
    }


@app.local_entrypoint()
def animate(
    video: str,
    canonical: str,
    cameras: str,
    track_dir: str,
    out: str,
    depth_reference: str = "",
    track_id: int = 0,
    depth_roi: str = "",
    fixed_world_scale: float | None = None,
    execution_timeout: int = 0,
    recover_missing_poses: bool = False,
):
    import hashlib
    import json
    import tarfile

    from worker.stages.lhm_execution import animation_options
    from worker.stages.lhm_recovery import atomic_json, new_receipt, recover_outputs, submit_once

    scale_flags, options = animation_options(fixed_world_scale, execution_timeout)
    if recover_missing_poses and fixed_world_scale is None:
        raise ValueError("Missing-pose recovery requires the measured fixed world scale")
    dest = Path(out)
    # Refuse an existing destination before any call, including after an ambiguous submission.
    receipt_path, receipt = new_receipt(dest, "motion")
    print(f"Recovery receipt: {receipt_path}; volume path: {receipt['remotePath']}", flush=True)
    print(
        "To recover without inference: uv run --locked python worker/stages/lhm_recovery.py "
        f"--receipt {receipt_path}",
        flush=True,
    )
    folder = Path(canonical)
    track = Path(track_dir)
    inputs = {
        "source.mp4": Path(video).read_bytes(),
        "canonical-state.pt": (folder / "canonical-state.pt").read_bytes(),
        "reference.json": (folder / "result.json").read_bytes(),
        "cameras.json": Path(cameras).read_bytes(),
        "depth-reference.ply": Path(
            depth_reference or (Path(cameras).parent / "frame_000.ply")
        ).read_bytes(),
        "seed-poses.pt": (track / "source-poses.pt").read_bytes(),
        "seed-motion.json": (track / "motion.json").read_bytes(),
    }
    flags = ["--track-id", str(track_id), *scale_flags]
    if depth_roi:
        flags += ["--depth-roi", depth_roi]
    receipt["execution"] = {
        "trackId": track_id,
        "recoverMissingPoses": recover_missing_poses,
        "flags": flags,
        "timeoutSeconds": execution_timeout or 3600,
        "inputSha256": {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()},
    }
    atomic_json(receipt_path, receipt)
    function = animate_track.with_options(**options) if options else animate_track
    result = submit_once(
        function,
        receipt_path,
        inputs,
        flags=flags,
        execution_timeout=execution_timeout,
        **({"recover_missing_poses": True} if recover_missing_poses else {}),
    )
    manifest = recover_outputs(receipt_path, cache)
    with tarfile.open(dest / "artifacts.tar.gz", mode="w:gz", compresslevel=1) as archive:
        for name in sorted(manifest["files"]):
            archive.add(dest / name, arcname=name)
    print(json.dumps(result["report"], indent=2))
    if manifest["status"] != "complete":
        raise RuntimeError(
            "LHM track outputs are partial/failed; see recovery-manifest.json and inference.log"
        )
