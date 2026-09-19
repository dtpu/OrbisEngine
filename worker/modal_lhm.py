"""Original LHM frozen-human experiment on bounded Modal CUDA workers.

Generated outputs are checkpointed in the LHM volume before a small return receipt.
Recover an interrupted call without inference using:
    uv run --locked python worker/stages/lhm_recovery.py --receipt OUT/recovery-receipt.json
Use ``modal run --detach`` when the remote job must survive loss of the local CLI.
"""

from pathlib import Path

import modal

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
app = modal.App("wander-overnight-lhm")
cache = modal.Volume.from_name("wander-overnight-lhm-cache", create_if_missing=True)


def link_model_caches(repo=Path("/opt/lhm"), cache_root=Path("/cache"), gfpgan_package=None):
    """Expose only the staged public model trees; refuse conflicting runtime directories."""
    repo, cache_root = Path(repo), Path(cache_root)
    links = []
    for relative, name in (
        ("data/pretrained_models", "pretrained_models"),
        ("data/gfpgan", "gfpgan"),
    ):
        source, target = cache_root / relative, repo / name
        if not source.is_dir() or not source.resolve().is_relative_to(cache_root.resolve()):
            raise RuntimeError(f"Stage the public model cache before inference: {relative}")
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Runtime model path conflicts with the staged cache: {name}")
        else:
            target.symlink_to(source, target_is_directory=True)
        links.append({"runtimePath": str(target), "cachePath": str(source)})
    if gfpgan_package is None:
        import importlib.util

        package = importlib.util.find_spec("gfpgan")
        if package is None or package.origin is None:
            raise RuntimeError("GFPGAN must be installed in the native worker image")
        gfpgan_package = Path(package.origin).parent
    source = cache_root / "gfpgan/weights/GFPGANv1.3.pth"
    target = Path(gfpgan_package) / "weights/GFPGANv1.3.pth"
    if not source.is_file() or not source.resolve().is_relative_to(cache_root.resolve()):
        raise RuntimeError("Stage the public GFPGANv1.3 checkpoint before inference")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError("GFPGAN package weight conflicts with the staged cache")
    else:
        target.symlink_to(source)
    links.append({"runtimePath": str(target), "cachePath": str(source)})
    return links


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=1800,
    retries=0,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def stage():
    import os
    import subprocess
    import time

    os.chdir("/opt/lhm")
    started = time.time()
    subprocess.run(
        [
            "python",
            "-c",
            "import torch, pytorch3d, diff_gaussian_rasterization, gsplat; from LHM.models import model_dict; print(torch.__version__, list(model_dict))",
        ],
        check=True,
    )
    return {"seconds": time.time() - started, "codeRevision": LHM_REV, "modelRevision": MODEL_REV}


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=65536,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def frozen(inputs: dict, animate: bool = False, larger_model: bool = False, recovery_id: str = ""):
    import hashlib
    import json
    import os
    import subprocess
    import tempfile
    import time
    import traceback

    from stages.lhm_recovery import (
        atomic_json,
        file_hash,
        finish_checkpoint,
        remote_root,
        start_checkpoint,
    )

    start = time.time()
    mode = "motion" if animate else "frozen"
    checkpoint, out = start_checkpoint("/cache", recovery_id, mode)
    cache.commit()
    model_id = "3DAIGC/LHM-1B-HF" if larger_model else "3DAIGC/LHM-500M-HF"
    model_revision = LARGER_MODEL_REV if larger_model else MODEL_REV
    error = None
    proc = None
    try:
        from huggingface_hub import snapshot_download

        root = Path(tempfile.mkdtemp(prefix="lhm-"))
        prepared = root / "prepared"
        prepared.mkdir()
        for name, data in inputs.items():
            if name not in (
                "source.png",
                "mask.png",
                "prepared.json",
                "source.mp4",
                "canonical-state.pt",
                "reference.json",
                "cameras.json",
                "depth-reference.ply",
                "seed-poses.pt",
                "seed-motion.json",
                "source-pose.pt",
                "source-pose.json",
                "head-input.png",
            ):
                raise ValueError("Unexpected prepared input")
            (prepared / name).write_bytes(data)
        links = link_model_caches()
        (out / "cache-links.json").write_text(json.dumps(links, indent=2))
        os.chdir("/opt/lhm")
        model = snapshot_download(model_id, revision=model_revision, local_files_only=True)
        if (prepared / "source-pose.pt").exists():
            if animate:
                raise ValueError("Fixed input comparison is frozen")
            command = [
                "python",
                "/root/stages/lhm_capacity.py",
                "--prepared",
                str(prepared),
                "--out",
                str(out),
                "--model",
                model,
                "--model-id",
                model_id,
                "--model-revision",
                model_revision,
            ]
        elif animate:
            command = [
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
            ]
            if (prepared / "cameras.json").exists():
                command += [
                    "--cameras",
                    str(prepared / "cameras.json"),
                    "--depth-reference",
                    str(prepared / "depth-reference.ply"),
                ]
            if (prepared / "seed-poses.pt").exists():
                command += [
                    "--seed-poses",
                    str(prepared / "seed-poses.pt"),
                    "--seed-motion",
                    str(prepared / "seed-motion.json"),
                ]
        else:
            command = [
                "python",
                "/root/stages/lhm_person.py",
                "--prepared",
                str(prepared),
                "--out",
                str(out),
                "--model",
                model,
            ]
        with (out / "inference.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait() != 0:
                raise RuntimeError(f"LHM inference exited {proc.returncode}")
    except Exception:  # noqa: BLE001 - preserve artifacts for any model/dependency failure
        error = traceback.format_exc()
        if proc is not None and proc.poll() is None:
            # Only our child process may write these outputs. Stop it before hashing.
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        print(error, flush=True)
        with (out / "inference.log").open("a") as log:
            log.write(error)
    auxiliary = {
        "/cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
        "/opt/lhm/gfpgan/weights/detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "/opt/lhm/gfpgan/weights/parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "/usr/local/lib/python3.10/site-packages/gfpgan/weights/GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    }
    try:
        provenance = []
        for path, url in auxiliary.items():
            file = Path(path)
            if file.exists():
                h = hashlib.sha256()
                with file.open("rb") as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                provenance.append(
                    {"url": url, "bytes": file.stat().st_size, "sha256": h.hexdigest()}
                )
        (out / "auxiliary-weight-provenance.json").write_text(json.dumps(provenance, indent=2))
    except Exception:  # noqa: BLE001 - provenance failure must not discard generated outputs
        error = (error or "") + traceback.format_exc()
        with (out / "inference.log").open("a") as log:
            log.write(error)
    elapsed = time.time() - start
    report = {
        "seconds": elapsed,
        "error": error,
        "experiment": "canonical-source-motion" if animate else "frozen-source-pose",
        "gpu": "L4",
        "cpu": 4,
        "memoryGiB": 64,
        "timeoutSeconds": 1800,
        "estimatedComputeUSD": elapsed * (0.000222 + 4 * 0.0000131 + 64 * 0.00000222),
        "timingScope": "worker setup and inference; excludes final output hashing and volume commit",
        "codeRevision": LHM_REV,
        "model": model_id,
        "modelRevision": model_revision,
        "recoveryId": recovery_id,
        "remotePath": remote_root(recovery_id),
    }
    atomic_json(out / "modal-run.json", report)
    manifest = finish_checkpoint(
        checkpoint,
        mode=mode,
        error=error,
        require_registration=bool(animate and "cameras.json" in inputs),
    )
    # Outputs are durable before returning this small receipt. Original video and
    # prepared inputs stay ephemeral; transfer and packaging run on the local caller.
    cache.commit()
    return {
        "report": report,
        "recoveryId": recovery_id,
        "remotePath": remote_root(recovery_id),
        "manifestSha256": file_hash(checkpoint / "manifest.json"),
        "status": manifest["status"],
        "totalWorkerSeconds": time.time() - start,
    }


@app.local_entrypoint()
def main(
    prepared: str = "",
    out: str = "",
    video: str = "",
    canonical: str = "",
    cameras: str = "",
    seed: str = "",
    fixed_inputs: str = "",
    larger_model: bool = False,
):
    if not prepared and not canonical:
        print(stage.remote())
        return
    import json
    import tarfile

    from worker.stages.lhm_recovery import new_receipt, recover_outputs, submit_once

    if not out:
        raise ValueError("LHM output destination is required")
    dest = Path(out)
    receipt_path, receipt = new_receipt(dest, "motion" if canonical else "frozen")
    print(f"Recovery receipt: {receipt_path}; volume path: {receipt['remotePath']}", flush=True)
    print(
        "To recover without inference: uv run --locked python worker/stages/lhm_recovery.py "
        f"--receipt {receipt_path}",
        flush=True,
    )
    if canonical:
        if not video:
            raise ValueError("Canonical animation requires its source video")
        folder = Path(canonical)
        inputs = {
            "source.mp4": Path(video).read_bytes(),
            "canonical-state.pt": (folder / "canonical-state.pt").read_bytes(),
            "reference.json": (folder / "result.json").read_bytes(),
        }
        if cameras:
            inputs["cameras.json"] = Path(cameras).read_bytes()
            inputs["depth-reference.ply"] = (Path(cameras).parent / "frame_000.ply").read_bytes()
        if seed:
            inputs["seed-poses.pt"] = (Path(seed) / "source-poses.pt").read_bytes()
            inputs["seed-motion.json"] = (Path(seed) / "motion.json").read_bytes()
    else:
        inputs = {
            name: (Path(prepared) / name).read_bytes()
            for name in ("source.png", "mask.png", "prepared.json")
        }
        if fixed_inputs:
            inputs.update(
                {
                    name: (Path(fixed_inputs) / name).read_bytes()
                    for name in ("source-pose.pt", "source-pose.json", "head-input.png")
                }
            )
    if larger_model and not fixed_inputs:
        raise ValueError("Larger checkpoint comparison requires saved fixed inputs")
    result = submit_once(
        frozen, receipt_path, inputs, animate=bool(canonical), larger_model=larger_model
    )
    manifest = recover_outputs(receipt_path, cache)
    # Preserve the normal archive artifact, without GPU compression or a giant return.
    with tarfile.open(dest / "artifacts.tar.gz", mode="w:gz", compresslevel=1) as archive:
        for name in sorted(manifest["files"]):
            archive.add(dest / name, arcname=name)
    print(json.dumps(result["report"], indent=2))
    print(f"Durable LHM output status: {manifest['status']}", flush=True)
    if manifest["status"] != "complete":
        raise RuntimeError(
            "LHM outputs are partial/failed; see recovery-manifest.json and inference.log"
        )
