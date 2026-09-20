"""Bounded CUDA motion experiments, with one hypothesis per invocation.

uv run --locked modal run worker/modal_motion.py --experiment vdpm --video clip.mp4 --out .context/vdpm
uv run --locked modal run worker/modal_motion.py --experiment pi3x --video clip.mp4 --out .context/dense

Source clips stay in the invocation's temporary directory. Only model caches
persist remotely; downloaded evidence is local and must remain untracked.
"""

from __future__ import annotations

from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
VDPM_REV = "5e2a57cf6007dfb0511a8b396a0805089b9edcc4"
PI3_REV = "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"
VGGT_REV = "44b3afb"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy==1.26.4",
        "pillow",
        "opencv-python-headless==4.11.0.86",
        "plyfile",
        "huggingface_hub",
        "safetensors",
        "scipy",
        "einops",
        "transformers==4.46.3",
        "omegaconf",
        "roma",
        "timm",
    )
    .run_commands(
        f"git clone https://github.com/eldar/vdpm.git /opt/vdpm && git -C /opt/vdpm checkout {VDPM_REV}",
        f"git clone https://github.com/yyfz/Pi3.git /opt/pi3 && git -C /opt/pi3 checkout {PI3_REV}",
        f"git clone https://github.com/facebookresearch/vggt.git /opt/vggt && git -C /opt/vggt checkout {VGGT_REV}",
    )
    .env(
        {
            "PYTHONPATH": "/root/worker:/opt/vggt",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
        }
    )
    .add_local_dir(HERE / "wander_worker", "/root/worker/wander_worker", ignore=["__pycache__"])
    .add_local_dir(HERE / "stages", "/root/worker/stages", ignore=["__pycache__"])
)
app = modal.App("wander-overnight-motion")
cache = modal.Volume.from_name("wander-overnight-motion-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=1500,
    scaledown_window=2,
    max_containers=1,
    retries=0,
    volumes={"/cache": cache},
)
def reconstruct(video_bytes: bytes, experiment: str, batch: int = 16, anchors: int = 8) -> dict:
    import io
    import json
    import subprocess
    import sys
    import tarfile
    import tempfile
    import time
    import traceback

    started = time.time()
    root = Path(tempfile.mkdtemp(prefix="motion-"))
    source, out = root / "input.mp4", root / "output"
    source.write_bytes(video_bytes)
    out.mkdir()
    error = None
    try:
        if experiment == "vdpm":
            from huggingface_hub import hf_hub_download

            model_file = Path(hf_hub_download("edgarsucar/vdpm", "model.pt"))
            checkpoint = Path("/cache/torch/hub/checkpoints/vdpm_model.pt")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.exists():
                checkpoint.symlink_to(model_file)
            command = [
                sys.executable,
                "/root/worker/stages/vdpm_person.py",
                "--video",
                str(source),
                "--out",
                str(out),
                "--repo",
                "/opt/vdpm",
                "--views",
                "16",
                "--targets",
                "0",
            ]
        elif experiment == "pi3x":
            command = [
                sys.executable,
                "/root/worker/stages/dense_pi3x.py",
                "--video",
                str(source),
                "--out",
                str(out),
                "--repo",
                "/opt/pi3",
                "--fps",
                "12",
                "--anchors",
                str(anchors),
                "--batch",
                str(batch),
            ]
        else:
            raise ValueError("Unknown experiment")
        cache.commit()
        with (out / "inference.log").open("w") as log:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait() != 0:
                raise RuntimeError(f"Inference process exited {proc.returncode}")
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        cache.commit()
    inference_wall = time.time() - started
    report = dict(
        experiment=experiment,
        anchors=anchors if experiment == "pi3x" else None,
        batch=batch if experiment == "pi3x" else None,
        inferenceWallSeconds=inference_wall,
        error=error,
        gpu="L4",
        cpu=4,
        memoryGiB=32,
        timeoutSeconds=1500,
        codeRevisions=dict(vdpm=VDPM_REV, pi3=PI3_REV, vggt=VGGT_REV),
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for item in sorted(out.iterdir()):
            if item.is_file() or item.name == "camera-batches":
                archive.add(item, arcname=item.name)
        elapsed = time.time() - started
        # Includes checkpoint loading, inference, cache commit, and compression.
        # Output transfer, image builds, and storage are additional billing items.
        report.update(
            seconds=elapsed,
            estimatedComputeUSD=elapsed * (0.000222 + 4 * 0.0000131 + 32 * 0.00000222),
        )
        (out / "modal-run.json").write_text(json.dumps(report, indent=2))
        archive.add(out / "modal-run.json", arcname="modal-run.json")
    return dict(report=report, archive=buf.getvalue())


@app.local_entrypoint()
def main(video: str, out: str, experiment: str = "vdpm", batch: int = 16, anchors: int = 8):
    import io
    import json
    import tarfile

    if batch < 1 or anchors < 2:
        raise ValueError("Need a positive batch size and at least two global anchors")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    result = reconstruct.remote(Path(video).read_bytes(), experiment, batch, anchors)
    archive_path = destination / "artifacts.tar.gz"
    archive_path.write_bytes(result["archive"])
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    print(json.dumps(result["report"], indent=2))
    print("Artifacts downloaded to", destination)
    if result["report"]["error"]:
        raise RuntimeError("CUDA experiment failed; see inference.log and modal-run.json")
