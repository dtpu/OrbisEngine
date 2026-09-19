"""Multi-person tracking stage on the existing LHM CUDA image.

Reuses `worker/modal_lhm.py`'s image object verbatim, so no layer is rebuilt: the same
MultiHMR pose estimator that `lhm_animate.py` already calls per frame is run once here with
every detection kept, and `worker/experiments/track_people.py` links them into tracks.

  modal run worker/modal_multiperson.py --video public/clips/elevator.mp4 \
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
    .apt_install("git", "build-essential", "ninja-build", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1")
    .env({"TORCH_CUDA_ARCH_LIST":"8.9", "FORCE_CUDA":"1", "MAX_JOBS":"4",
          "PYTHONUNBUFFERED":"1", "HF_HOME":"/cache/huggingface", "TORCH_HOME":"/cache/torch",
          "PYOPENGL_PLATFORM":"egl"})
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "xformers==0.0.26.post1", "numpy==1.23.5", "setuptools==74.0.0", "wheel", "ninja")
    .pip_install("numpy==1.23.5", "scipy==1.14.1", "pillow==10.4.0", "opencv-python-headless==4.11.0.86",
                 "einops", "roma", "accelerate", "smplx", "chumpy", "decord==0.6.0", "diffusers==0.32.0",
                 "gsplat==1.4.0", "huggingface_hub==0.28.1", "safetensors", "imageio==2.34.1", "imageio-ffmpeg",
                 "jaxtyping==0.2.38", "kornia==0.7.2", "loguru", "lpips", "matplotlib==3.8.4", "omegaconf",
                 "plyfile==1.0.3", "pygltflib", "pyrender", "pyyaml", "requests", "timm==1.0.15", "trimesh==4.4.9",
                 "transformers==4.41.2", "tqdm", "typeguard==2.13.3", "iopath", "fvcore", "rembg==2.0.63", "onnxruntime", "addict", "future", "lmdb", "scikit-image==0.22.0", "tb-nightly", "yapf")
    .run_commands(
        "git clone https://github.com/XPixelGroup/BasicSR.git /opt/basicsr && git -C /opt/basicsr checkout 8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a && pip install --no-deps /opt/basicsr",
        "pip install --no-deps gfpgan==1.3.8 facexlib==0.3.0 filterpy==1.4.5",
        "git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d && git -C /opt/pytorch3d checkout 89653419d0973396f3eff1a381ba09a07fffc2ed && CC=gcc CXX=g++ CUB_HOME=/usr/local/cuda/include pip install --no-build-isolation --no-deps /opt/pytorch3d",
        "git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git /opt/diff-gaussian-rasterization && git -C /opt/diff-gaussian-rasterization checkout 8829d14f814fccdaf840b7b0f3021a616583c0a1 && CC=gcc CXX=g++ pip install --no-build-isolation --no-deps /opt/diff-gaussian-rasterization",
        "git clone https://github.com/camenduru/simple-knn.git /opt/simple-knn && git -C /opt/simple-knn checkout 60f461f4a56b7967e5d8045bf92f8c33f36976d0 && CC=gcc CXX=g++ pip install --no-build-isolation --no-deps /opt/simple-knn",
        f"git clone https://github.com/aigc3d/LHM.git /opt/lhm && git -C /opt/lhm checkout {LHM_REV}",
    )
    .add_local_dir(HERE / "experiments", "/root/experiments", ignore=["__pycache__"])
)
app = modal.App("wander-multiperson")
cache = modal.Volume.from_name("wander-overnight-lhm-cache", create_if_missing=True)


@app.function(image=image, gpu="L4", cpu=4, memory=65536, timeout=3600, retries=0,
              max_containers=1, volumes={"/cache": cache}, scaledown_window=2)
def track(video: bytes, cameras: bytes | None, fps: float, det_thresh: float, gate: float,
          max_gap: int, min_track_samples: int, overlay_every: int):
    import io, json, os, subprocess, tarfile, tempfile, time, traceback
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
    cmd = ["python", "/root/experiments/track_people.py", "--video", str(root / "source.mp4"),
           "--out", str(out), "--fps", str(fps), "--det-thresh", str(det_thresh),
           "--gate", str(gate), "--max-gap", str(max_gap),
           "--min-track-samples", str(min_track_samples), "--overlay-every", str(overlay_every)]
    if cameras:
        cmd += ["--cameras", str(root / "cameras.json")]
    try:
        with (out / "track.log").open("w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait() != 0:
                raise RuntimeError(f"track_people exited {proc.returncode}")
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(out)))
    seconds = time.time() - started
    report = dict(seconds=seconds, error=error, gpu="L4", codeRevision=LHM_REV,
                  estimatedComputeUSD=seconds * (0.000222 + 4 * 0.0000131 + 64 * 0.00000222))
    return dict(report=report, archive=buf.getvalue())


@app.local_entrypoint()
def main(video: str, out: str, cameras: str = "", fps: float = 12, det_thresh: float = 0.15,
         gate: float = 0.22, max_gap: int = 8, min_track_samples: int = 8, overlay_every: int = 0):
    import io, json, tarfile
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    result = track.remote(Path(video).read_bytes(), Path(cameras).read_bytes() if cameras else None,
                          fps, det_thresh, gate, max_gap, min_track_samples, overlay_every)
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        archive.extractall(dest, filter="data")
    (dest / "modal-run.json").write_text(json.dumps(result["report"], indent=2))
    print(json.dumps(result["report"], indent=2))
    if result["report"]["error"]:
        raise RuntimeError("tracking failed; see the downloaded track.log")


@app.function(image=image, gpu="L4", cpu=4, memory=65536, timeout=3600, retries=0,
              max_containers=1, volumes={"/cache": cache}, scaledown_window=2)
def animate_track(inputs: dict, flags: list):
    """lhm_animate.py --track-only: one canonical avatar driven by ONE track's poses.

    Same code path as the single-person animate stage in worker/modal_lhm.py; the difference is
    that MultiHMR is never called here, so the pose for every sample is the one the tracker
    already assigned to this identity. Re-detecting would pick whichever person is nearest and
    is exactly how identities swap when two people cross.
    """
    import io, json, os, subprocess, tarfile, tempfile, time, traceback
    from huggingface_hub import snapshot_download
    start = time.time()
    root = Path(tempfile.mkdtemp(prefix="lhm-track-"))
    prepared = root / "prepared"
    prepared.mkdir()
    out = root / "output"
    out.mkdir()
    allowed = {"source.mp4", "canonical-state.pt", "reference.json", "cameras.json",
               "depth-reference.ply", "seed-poses.pt", "seed-motion.json"}
    for name, data in inputs.items():
        if name not in allowed:
            raise ValueError(f"Unexpected prepared input {name}")
        (prepared / name).write_bytes(data)
    prior = Path("/cache/data/pretrained_models")
    link = Path("/opt/lhm/pretrained_models")
    if not prior.exists():
        raise RuntimeError("Stage public LHM prior assets before GPU inference")
    if not link.exists():
        link.symlink_to(prior, target_is_directory=True)
    os.chdir("/opt/lhm")
    error = None
    model_id = "3DAIGC/LHM-500M-HF"
    try:
        model = snapshot_download(model_id, revision=MODEL_REV, local_files_only=True)
        cmd = ["python", "/root/experiments/lhm_animate.py", "--video", str(prepared / "source.mp4"),
               "--canonical", str(prepared / "canonical-state.pt"),
               "--reference", str(prepared / "reference.json"), "--out", str(out), "--model", model,
               "--cameras", str(prepared / "cameras.json"),
               "--depth-reference", str(prepared / "depth-reference.ply"),
               "--seed-poses", str(prepared / "seed-poses.pt"),
               "--seed-motion", str(prepared / "seed-motion.json"), "--track-only", *flags]
        with (out / "inference.log").open("w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if proc.wait() != 0:
                raise RuntimeError(f"LHM inference exited {proc.returncode}")
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for f in sorted(out.iterdir()):
            if f.is_file():
                archive.add(f, arcname=f.name)
        elapsed = time.time() - start
        report = dict(seconds=elapsed, error=error, experiment="track-canonical-source-motion",
                      gpu="L4", codeRevision=LHM_REV, model=model_id, modelRevision=MODEL_REV,
                      estimatedComputeUSD=elapsed * (0.000222 + 4 * 0.0000131 + 64 * 0.00000222))
        (out / "modal-run.json").write_text(json.dumps(report, indent=2))
        archive.add(out / "modal-run.json", arcname="modal-run.json")
    return dict(report=report, archive=buf.getvalue())


@app.local_entrypoint()
def animate(video: str, canonical: str, cameras: str, track_dir: str, out: str,
            depth_reference: str = "", track_id: int = 0, depth_roi: str = ""):
    import io, json, tarfile
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    folder = Path(canonical)
    track = Path(track_dir)
    inputs = {
        "source.mp4": Path(video).read_bytes(),
        "canonical-state.pt": (folder / "canonical-state.pt").read_bytes(),
        "reference.json": (folder / "result.json").read_bytes(),
        "cameras.json": Path(cameras).read_bytes(),
        "depth-reference.ply": Path(depth_reference or (Path(cameras).parent / "frame_000.ply")).read_bytes(),
        "seed-poses.pt": (track / "source-poses.pt").read_bytes(),
        "seed-motion.json": (track / "motion.json").read_bytes(),
    }
    flags = ["--track-id", str(track_id)]
    if depth_roi:
        flags += ["--depth-roi", depth_roi]
    result = animate_track.remote(inputs, flags)
    (dest / "artifacts.tar.gz").write_bytes(result["archive"])
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        archive.extractall(dest, filter="data")
    print(json.dumps(result["report"], indent=2))
    if result["report"]["error"]:
        raise RuntimeError("LHM track animation failed; see the downloaded inference.log")
