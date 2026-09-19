"""Does MultiHMR actually read this reference frame? -- the gate scripts/score_reference_frames.py lacked.

The scorer ranks frames on how much identity they carry. It never asked the question that actually
sinks a pick: can `lhm_person.py` build from this frame at all? Two separate things can say no, and
both were seen on bedroom:

  1. MultiHMR runs on the person CUT OUT ONTO WHITE, not on the raw frame. On a seated subject
     behind a desk it returns no detection and the build dies -- bedroom's top-scored frame 198
     (share/POSE-STATUS.md section 5). A local COCO detector does not reproduce this: it reads
     f198 at 0.999. Only MultiHMR itself answers it.
  2. The LHM crop itself can fail. `preprocess()` demands a portrait mask crop at least 5:3 tall,
     and pads to reach it; a crop that is already wider than 5:3 pads by a negative amount and
     raises. bedroom f222 -- the frame the probe picked once gate 1 was in -- dies there, 1 pixel
     short of the aspect it needs.

So this runs both, on the real GPU image: MultiHMR at lhm_person.py's threshold on lhm_person.py's
whitened cut-out, and lhm_person.py's own `preprocess`. A frame is readable only if both pass.

  MODAL run worker/modal_pose_probe.py --requests .context/pose/framefix/probe-in.json \
      --out .context/pose/framefix/probe-bedroom.json

`--requests` is [{clip, index, source, mask}, ...] with paths to the PNGs prepare_lhm_person.py
writes. The result is {"<clip>": {"<index>": {"detected": bool, "score": float, ...}}}, which
scripts/score_reference_frames.py consumes through --estimator-probe.

The image spec is copied verbatim from worker/modal_lhm.py -- an identical spec hashes to the same
image id, so no layer is rebuilt. Kept inline because a Modal container only mounts its entrypoint.
"""
from pathlib import Path

import modal

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
app = modal.App("wander-overnight-lhm")
app = modal.App("wander-pose-probe")
cache = modal.Volume.from_name("wander-overnight-lhm-cache", create_if_missing=True)


@app.function(image=image, gpu="L4", cpu=4, memory=65536, timeout=1800, retries=0,
              max_containers=1, volumes={"/cache": cache}, scaledown_window=2)
def probe(items: list, det_thresh: float = 0.3):
    """One MultiHMR forward pass per candidate, on lhm_person.py's exact whitened cut-out."""
    import json, os, sys, tempfile, time, traceback
    import numpy as np
    from PIL import Image
    prior = Path("/cache/data/pretrained_models")
    link = Path("/opt/lhm/pretrained_models")
    if not prior.exists():
        raise RuntimeError("Stage public LHM prior assets before GPU inference")
    if not link.exists():
        link.symlink_to(prior, target_is_directory=True)
    os.chdir("/opt/lhm")
    sys.path.insert(0, "/opt/lhm")
    started = time.time()
    import torch
    from accelerate import Accelerator
    torch._dynamo.config.disable = True
    torch.set_num_threads(4)
    Accelerator()
    from engine.pose_estimation.pose_estimator import PoseEstimator
    sys.path.insert(0, "/root/experiments")
    from lhm_person import preprocess as lhm_preprocess
    estimator = PoseEstimator("./pretrained_models/human_model_files", device="cuda")
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="probe-"))
    for it in items:
        rec = dict(clip=it["clip"], index=it["index"])
        try:
            (tmp / "s.png").write_bytes(it["source"])
            (tmp / "m.png").write_bytes(it["mask"])
            raw = np.asarray(Image.open(tmp / "s.png").convert("RGB"))
            mask = np.asarray(Image.open(tmp / "m.png").convert("L"))
            # lhm_person.py, verbatim: the person on white, full frame, then MultiHMR at det_thresh.
            pose_rgb = raw.copy()
            pose_rgb[mask < 128] = 255
            padded, ow, oh = estimator.img_center_padding(pose_rgb)
            tensor, annotation = estimator._preprocess(padded)
            K = estimator.get_camera_parameters()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                people = estimator.mhmr_model(tensor, is_training=False, nms_kernel_size=3,
                                              det_thresh=det_thresh, K=K, idx=None, max_dist=None)
            rec["poseDetected"] = bool(people)
            rec["people"] = len(people)
            rec["score"] = float(max(float(p["scores"]) for p in people)) if people else 0.0
            if people:
                best = max(people, key=lambda p: float(p["scores"]))
                j = best["j3d"][:22].detach().float().cpu().numpy()
                rec["bodyExtentM"] = float(j[:, 1].max() - j[:, 1].min())
            else:
                rec["reason"] = "MultiHMR found no person in the whitened cut-out"
            # The crop LHM itself would take, which has its own ways of refusing a frame.
            try:
                lhm_preprocess(Path("/opt/lhm"), tmp / "s.png", mask)
                rec["cropOk"] = True
            except Exception as exc:
                rec["cropOk"] = False
                rec["reason"] = f"LHM preprocess refused the crop: {exc}"
            rec["detected"] = bool(rec["poseDetected"] and rec["cropOk"])
        except Exception:
            rec["detected"] = False
            rec["score"] = 0.0
            rec["reason"] = "probe raised"
            rec["error"] = traceback.format_exc()[-800:]
        print(json.dumps({k: v for k, v in rec.items() if k != "error"}), flush=True)
        results.append(rec)
    return dict(results=results, seconds=time.time() - started, detThreshold=det_thresh,
                gpu="L4", codeRevision=LHM_REV,
                method="MultiHMR on the person cut out onto white at full source resolution, plus "
                       "lhm_person.py's own preprocess crop -- both exactly as the build runs them. "
                       "A frame that fails either cannot build an avatar however well it scores.")


@app.local_entrypoint()
def main(requests: str, out: str, det_thresh: float = 0.3):
    import json
    reqs = json.loads(Path(requests).read_text())
    items = [dict(clip=r["clip"], index=int(r["index"]),
                  source=Path(r["source"]).read_bytes(), mask=Path(r["mask"]).read_bytes())
             for r in reqs]
    res = probe.remote(items, det_thresh)
    by = {}
    for r in res["results"]:
        by.setdefault(r["clip"], {})[str(r["index"])] = {k: v for k, v in r.items() if k not in ("clip", "index")}
    doc = dict(method=res["method"], detThreshold=res["detThreshold"], seconds=res["seconds"],
               gpu=res["gpu"], codeRevision=res["codeRevision"], clips=by)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1))
    for clip, rows in by.items():
        ok = [i for i, v in rows.items() if v["detected"]]
        bad = [i for i, v in rows.items() if not v["detected"]]
        print(f"{clip}: readable {sorted(ok, key=int)}  UNREADABLE {sorted(bad, key=int)}")
    print("wrote", out)
