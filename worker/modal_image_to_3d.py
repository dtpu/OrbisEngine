"""One PNG crop in -> a coloured 3D Gaussian PLY (and a textured GLB) out, via TRELLIS on an L4.

TRELLIS predicts 3D Gaussians natively, which is the representation this project ships, so the PLY
is the real output and the mesh is the by-product (not the other way round as with LRM/mesh models).

  uv run --locked modal run worker/modal_image_to_3d.py --image crop.png --out-dir /tmp/obj
  uv run --locked modal run worker/modal_image_to_3d.py --image crop.png --out-dir /tmp/obj \
      --seed 1 --ss-steps 12 --slat-steps 12 --no-remove-bg

Writes object.ply, object.glb (when mesh extraction succeeds), an 8-view orbit contact.png, the
exact 518px image the pipeline consumed as cond.png, and meta.json into --out-dir. When a result
has the wrong number of objects in it, cond.png is the frame that says why: rembg will happily keep
a soft cast shadow, and TRELLIS then builds the shadow as a second object.

--no-remove-bg skips rembg for crops that are already tight or already alpha-matted. It describes
the INPUT crop; anything the refiner produces is matted regardless, since a generated product shot
arrives opaque and with a shadow whatever the crop was. --note "..." lands verbatim in meta.json,
for recording a provenance chain the file cannot work out for itself.

--refine-first adds an SDXL img2img pass before the 3D step, for crops too small or too blurry to
reconstruct directly (a 26px bottle comes back as a blob otherwise). It is conditioned on the crop,
not on text alone, but it still INVENTS detail that was never recorded -- refined.png and the
refiner block in meta.json are there so that can be declared rather than hidden.

  uv run --locked modal run worker/modal_image_to_3d.py --image tiny-crop.png --out-dir /tmp/obj \
      --refine-first --refine-prompt "a crushed clear PET water bottle with a dark blue label"

--refine-text-only goes further and drops the crop altogether, generating the conditioning image
from the prompt alone. Nothing in the result then comes from the footage; meta.json says so in
`refiner.mode`. Use --refine-candidates N to get N seeds as candidates.png and --refine-pick K to
choose one (the seeds are deterministic, so re-running with a different pick is free of surprises).

Licences of everything this pulls:
  microsoft/TRELLIS            code          MIT
  microsoft/TRELLIS-image-large weights      MIT
  facebookresearch/dinov2      image cond    Apache-2.0  (dinov2_vitl14_reg, via torch.hub)
  danielgatis/rembg            bg removal    MIT
  xuebinqin/U-2-Net  u2net.onnx weights      Apache-2.0  (rembg's default session)
  EasternJournalist/utils3d                  MIT
  stabilityai/stable-diffusion-xl-base-1.0   CreativeML Open RAIL++-M  (refine passes only)
    NOT an OSI licence: open weights with use-based restrictions. Only the optional refine passes
    touch it; without --refine-first or --refine-text-only nothing SDXL is downloaded or run.
  NVlabs/nvdiffrast            GLB texture bake   NVIDIA Source Code Licence (non-commercial)
  autonomousvision/mip-splatting diff-gaussian-rasterization  Inria/MPII non-commercial research
The last two are only reached by the GLB texture bake. The PLY path -- geometry and colour -- is
MIT/Apache the whole way down.
"""

from __future__ import annotations

from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
TRELLIS_REV = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
UTILS3D_REV = "9a4eb15e4021b67b12c460c7057d642626897ec8"  # the commit TRELLIS's setup.sh pins
MODEL_ID = "microsoft/TRELLIS-image-large"
REFINER_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINE_PROMPT = (
    "studio product photograph of a single object on a plain white background, "
    "sharp focus, even lighting, whole object in frame, high detail"
)
REFINE_NEGATIVE = "blurry, low resolution, jpeg artifacts, multiple objects, cropped, text, watermark, shadow, reflection"
CXX_ENV = {"CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -shared", "LDCXXSHARED": "g++ -shared"}

BASIC = [
    "pillow",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "scipy",
    "ninja",
    "rembg==2.0.63",
    "onnxruntime",
    "trimesh",
    "open3d",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
    "transformers==4.46.3",
    "safetensors",
    "huggingface_hub",
    "plyfile",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "cmake",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libglvnd-dev",
        "libgl1-mesa-dev",
        "libegl1-mesa-dev",
        "libgles2-mesa-dev",
        "libx11-dev",
    )
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "8.6;8.9",
            "FORCE_CUDA": "1",
            "MAX_JOBS": "4",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/opt/trellis",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
            "U2NET_HOME": "/cache/u2net",
            "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions",
            "ATTN_BACKEND": "xformers",
            "SPCONV_ALGO": "native",
        }
    )
    .uv_pip_install(
        "torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124"
    )
    .uv_pip_install("xformers==0.0.28.post3", index_url="https://download.pytorch.org/whl/cu124")
    .uv_pip_install(
        "setuptools==75.6.0", "wheel", "packaging", "ninja"
    )  # --no-build-isolation needs these present
    .uv_pip_install(*BASIC)
    .uv_pip_install("numpy==1.26.4")  # after rembg/open3d, which will happily drag in numpy 2
    .uv_pip_install(f"git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_REV}")
    .uv_pip_install("spconv-cu124")
    .run_commands(
        f"git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /opt/trellis"
        f" && git -C /opt/trellis checkout {TRELLIS_REV}",
        # Modal's add_python interpreter reports clang as its linker, which the CUDA devel image
        # does not carry; point every distutils compiler hook at gcc/g++ instead.
        "git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast",
    )
    .uv_pip_install("/opt/nvdiffrast", extra_options="--no-build-isolation", env=CXX_ENV)
    .run_commands(
        "git clone --recurse-submodules https://github.com/autonomousvision/mip-splatting.git /opt/mip-splatting",
    )
    .uv_pip_install(
        "/opt/mip-splatting/submodules/diff-gaussian-rasterization/",
        extra_options="--no-build-isolation",
        env=CXX_ENV,
    )
    # TRELLIS's FlexiCubes submodule imports kaolin; last layer so the CUDA compiles above stay cached.
    .uv_pip_install(
        "kaolin==0.17.0",
        find_links="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html",
    )
    .uv_pip_install("numpy==1.26.4")
    .uv_pip_install("diffusers==0.31.0", "accelerate==1.1.1")  # optional --refine-first pass only
)
app = modal.App("wander-image-to-3d")
cache = modal.Volume.from_name("wander-image-to-3d-cache", create_if_missing=True)


@app.function(
    image=image, cpu=4, memory=16384, timeout=900, volumes={"/cache": cache}, scaledown_window=2
)
def stage():
    """Import-only smoke test: proves the image builds and the pipeline module loads."""
    import subprocess

    subprocess.run(
        [
            "python",
            "-c",
            (
                "import os, torch, xformers, spconv, utils3d, nvdiffrast.torch, "
                "diff_gaussian_rasterization, xatlas, pyvista, pymeshfix, igraph, rembg;"
                "from trellis.pipelines import TrellisImageTo3DPipeline;"
                "from trellis.utils import postprocessing_utils;"
                "print('torch', torch.__version__, 'attn', os.environ.get('ATTN_BACKEND'))"
            ),
        ],
        check=True,
        cwd="/opt/trellis",
    )
    return {"ok": True, "trellisRevision": TRELLIS_REV}


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=1800,
    retries=0,
    max_containers=2,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def refine_crop(
    png_bytes: bytes,
    prompt: str = REFINE_PROMPT,
    negative_prompt: str = REFINE_NEGATIVE,
    strength: float = 0.55,
    steps: int = 40,
    guidance: float = 6.5,
    seed: int = 1,
    size: int = 1024,
    text_only: bool = False,
    candidates: int = 1,
    pick: int = 0,
) -> dict:
    """SDXL on the crop, so the 3D stage gets something with edges in it.

    img2img is conditioned on the crop, letterboxed onto white before upscaling and never stretched:
    proportions are the one thing a 26px source still tells us reliably, and the 3D model reads them
    straight off. text_only drops the crop entirely and generates from the prompt, which invents the
    whole object -- it exists because a crop that is only a label seen end-on has no bottle in it to
    recover, and meta records which of the two actually ran.
    """
    import hashlib, io, time
    import torch
    from PIL import Image

    t0 = time.time()
    src = Image.open(io.BytesIO(png_bytes))
    common = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )

    if text_only:
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            REFINER_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        common.update(height=size, width=size)
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            REFINER_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        flat = Image.new("RGB", src.size, (255, 255, 255))
        flat.paste(src, (0, 0), src if src.mode == "RGBA" else None)
        side = max(flat.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(flat, ((side - flat.width) // 2, (side - flat.height) // 2))
        common.update(
            image=square.resize((size, size), Image.Resampling.LANCZOS), strength=strength
        )

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    shots = [
        pipe(generator=torch.Generator("cuda").manual_seed(seed + i), **common).images[0]
        for i in range(max(1, candidates))
    ]
    del pipe
    torch.cuda.empty_cache()
    cache.commit()

    out = shots[min(pick, len(shots) - 1)]
    sheet_bytes = b""
    if len(shots) > 1:
        cell = 384
        sheet = Image.new("RGB", (cell * len(shots), cell))
        for i, im in enumerate(shots):
            sheet.paste(im.resize((cell, cell), Image.Resampling.LANCZOS), (i * cell, 0))
        sb = io.BytesIO()
        sheet.save(sb, format="PNG")
        sheet_bytes = sb.getvalue()

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(REFINER_ID).sha
    except Exception:
        revision = None
    meta = {
        "mode": "text-to-image (NO image conditioning)"
        if text_only
        else "img2img (conditioned on the crop)",
        "imageConditioned": not text_only,
        "model": REFINER_ID,
        "modelRevision": revision,
        "licence": "CreativeML Open RAIL++-M (open weights, use-based restrictions; not OSI)",
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "strength": None if text_only else strength,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "seedsTried": [seed + i for i in range(max(1, candidates))],
        "seedPicked": seed + min(pick, max(1, candidates) - 1),
        "size": size,
        "inputSha256": hashlib.sha256(png_bytes).hexdigest(),
        "inputSize": list(src.size),
        "seconds": round(time.time() - t0, 1),
        "note": (
            "generated from the prompt alone; the crop was NOT used, so nothing in this object "
            "comes from the footage"
        )
        if text_only
        else "img2img conditioned on the crop; invents detail the footage never recorded",
    }
    return {"png": buf.getvalue(), "candidates": sheet_bytes, "meta": meta}


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=1800,
    retries=0,
    max_containers=2,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def image_to_3d(
    png_bytes: bytes,
    seed: int = 1,
    ss_steps: int = 12,
    slat_steps: int = 12,
    ss_cfg: float = 7.5,
    slat_cfg: float = 3.0,
    remove_bg: bool = True,
    want_glb: bool = True,
    texture_size: int = 1024,
    simplify: float = 0.95,
    preview: bool = True,
    preview_video: bool = False,
    preview_views: int = 8,
    contact_px: int = 256,
    refiner: dict | None = None,
    note: str = "",
) -> dict:
    import hashlib, io, json, os, time, traceback
    import numpy as np
    from PIL import Image

    os.environ.setdefault("SPCONV_ALGO", "native")
    os.chdir("/opt/trellis")
    from trellis.pipelines import TrellisImageTo3DPipeline

    t0 = time.time()
    src = Image.open(io.BytesIO(png_bytes))
    sha = hashlib.sha256(png_bytes).hexdigest()

    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
    pipeline.cuda()
    cache.commit()

    if remove_bg:
        cond = pipeline.preprocess_image(src)
    else:
        # TRELLIS's own preprocess_image, minus the rembg call: alpha-crop (or whole-frame when the
        # crop is opaque), 1.2x margin, 518px, premultiply. Keep in step with upstream if it moves.
        im = src.convert("RGBA") if src.mode != "RGBA" else src
        arr = np.array(im)
        alpha = arr[:, :, 3]
        mask = np.argwhere(alpha > 0.8 * 255)
        if mask.size == 0:
            mask = np.argwhere(np.ones_like(alpha, dtype=bool))
        bbox = np.array([mask[:, 1].min(), mask[:, 0].min(), mask[:, 1].max(), mask[:, 0].max()])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
        crop = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2),
        )
        out = im.crop(crop).resize((518, 518), Image.Resampling.LANCZOS)
        f = np.array(out).astype(np.float32) / 255
        cond = Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

    # Always hand the pipeline a finished 518px conditioning image and ship a copy: when a result
    # has the wrong number of objects in it, this is the only frame that says why.
    outputs = pipeline.run(
        cond,
        seed=seed,
        preprocess_image=False,
        formats=["gaussian", "mesh"] if want_glb else ["gaussian"],
        sparse_structure_sampler_params={"steps": ss_steps, "cfg_strength": ss_cfg},
        slat_sampler_params={"steps": slat_steps, "cfg_strength": slat_cfg},
    )
    gen_seconds = time.time() - t0

    work = Path("/tmp/i23d")
    work.mkdir(parents=True, exist_ok=True)
    cond_buf = io.BytesIO()
    cond.save(cond_buf, format="PNG")
    ply_path = work / "object.ply"
    outputs["gaussian"][0].save_ply(str(ply_path))
    stats = _ply_stats(ply_path)

    glb_bytes, glb_error = b"", None
    if want_glb:
        try:
            from trellis.utils import postprocessing_utils

            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_size,
            )
            glb.export(str(work / "object.glb"))
            glb_bytes = (work / "object.glb").read_bytes()
        except Exception:
            glb_error = traceback.format_exc()
            print(glb_error, flush=True)

    contact_bytes, preview_bytes, preview_error = b"", b"", None
    if preview:
        try:
            from trellis.utils import render_utils

            frames = render_utils.render_video(
                outputs["gaussian"][0], resolution=384, num_frames=preview_views * 8
            )["color"]
            picks = [frames[round(i * len(frames) / preview_views)] for i in range(preview_views)]
            sheet = Image.new("RGB", (contact_px * preview_views, contact_px))
            for i, fr in enumerate(picks):
                sheet.paste(
                    Image.fromarray(fr).resize((contact_px, contact_px), Image.Resampling.LANCZOS),
                    (i * contact_px, 0),
                )
            sheet.save(work / "contact.png")
            contact_bytes = (work / "contact.png").read_bytes()
            if preview_video:
                import imageio

                imageio.mimsave(str(work / "preview.mp4"), frames, fps=24)
                preview_bytes = (work / "preview.mp4").read_bytes()
        except Exception:
            preview_error = traceback.format_exc()
            print(preview_error, flush=True)

    try:
        from huggingface_hub import HfApi

        model_revision = HfApi().model_info(MODEL_ID).sha
    except Exception:
        model_revision = None

    meta = {
        "model": MODEL_ID,
        "modelRevision": model_revision,
        "codeRepo": "https://github.com/microsoft/TRELLIS",
        "codeRevision": TRELLIS_REV,
        "licences": {
            "trellis-code": "MIT",
            "trellis-image-large-weights": "MIT",
            "dinov2-image-cond": "Apache-2.0",
            "rembg": "MIT",
            "u2net-weights": "Apache-2.0",
            "utils3d": "MIT",
            "nvdiffrast": "NVIDIA Source Code Licence (non-commercial) - GLB texture bake only",
            "diff-gaussian-rasterization": "Inria/MPII non-commercial research - GLB bake and preview only",
        },
        "seed": seed,
        "sampler": {
            "sparseStructureSteps": ss_steps,
            "sparseStructureCfg": ss_cfg,
            "slatSteps": slat_steps,
            "slatCfg": slat_cfg,
        },
        "removeBg": remove_bg,
        "refiner": refiner,
        "note": note or None,
        "inputSha256": sha,
        "inputSize": list(src.size),
        "gpu": "L4",
        "seconds": round(time.time() - t0, 1),
        "generateSeconds": round(gen_seconds, 1),
        "ply": stats,
        "glbError": glb_error,
        "previewError": preview_error,
    }
    print(
        json.dumps({k: meta[k] for k in ("model", "seed", "ply", "seconds")}, indent=2), flush=True
    )
    return {
        "ply": ply_path.read_bytes(),
        "glb": glb_bytes,
        "contact": contact_bytes,
        "cond": cond_buf.getvalue(),
        "preview": preview_bytes,
        "meta": meta,
    }


def _ply_stats(ply_path: "Path") -> dict:
    """Read back the file we are about to ship, so the numbers describe the PLY's own axes.

    save_ply writes opacity as an inverse-sigmoid logit, so a fully opaque splat lands as +inf and
    trips strict PLY loaders. Clamping to +-30 is the same value after the sigmoid in float32 and
    leaves position, colour and scale untouched.
    """
    import numpy as np

    with ply_path.open("rb") as f:
        header = b""
        while b"end_header\n" not in header:
            header += f.read(1)
        text = header.decode()
        count = int(
            next(l for l in text.splitlines() if l.startswith("element vertex")).split()[-1]
        )
        fields = [l.split()[-1] for l in text.splitlines() if l.startswith("property float")]
        body = np.frombuffer(f.read(count * len(fields) * 4), dtype="<f4").reshape(
            count, len(fields)
        )

    op = body[:, fields.index("opacity")]
    if not np.isfinite(op).all():
        body = body.copy()
        body[:, fields.index("opacity")] = np.nan_to_num(op, nan=0.0, posinf=30.0, neginf=-30.0)
        with ply_path.open("wb") as f:
            f.write(header)
            f.write(body.astype("<f4").tobytes())

    xyz = body[:, :3]
    rgb = np.clip(
        body[:, [fields.index(f"f_dc_{i}") for i in range(3)]] * 0.28209479177387814 + 0.5, 0, 1
    )
    scale = np.exp(body[:, [fields.index(f"scale_{i}") for i in range(3)]])
    alpha = 1 / (1 + np.exp(-np.clip(body[:, fields.index("opacity")], -30, 30)))
    grid = ((xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0) + 1e-9) * 31.999).astype(np.int32)
    return {
        "gaussians": int(count),
        "bboxMin": [round(float(v), 4) for v in xyz.min(0)],
        "bboxMax": [round(float(v), 4) for v in xyz.max(0)],
        "extents": [round(float(v), 4) for v in (xyz.max(0) - xyz.min(0))],
        "meanRGB": [round(float(v), 4) for v in rgb.mean(0)],
        "stdRGB": [round(float(v), 4) for v in rgb.std(0)],
        "meanScale": [round(float(v), 5) for v in scale.mean(0)],
        "meanOpacity": round(float(alpha.mean()), 4),
        "occupiedCellsOf32Cubed": int(len(np.unique(grid, axis=0))),
    }


@app.local_entrypoint()
def main(
    image: str = "",
    out_dir: str = "",
    seed: int = 1,
    ss_steps: int = 12,
    slat_steps: int = 12,
    ss_cfg: float = 7.5,
    slat_cfg: float = 3.0,
    remove_bg: bool = True,
    glb: bool = True,
    texture_size: int = 1024,
    preview: bool = True,
    preview_video: bool = False,
    refine_first: bool = False,
    refine_prompt: str = REFINE_PROMPT,
    refine_negative: str = REFINE_NEGATIVE,
    refine_strength: float = 0.55,
    refine_steps: int = 40,
    refine_guidance: float = 6.5,
    refine_seed: int = 1,
    refine_text_only: bool = False,
    refine_candidates: int = 1,
    refine_pick: int = 0,
    note: str = "",
):
    import json

    if not image:
        print(json.dumps(stage.remote(), indent=2))
        return
    if not out_dir:
        raise ValueError("--out-dir is required")

    png = Path(image).read_bytes()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    refiner = None
    if refine_first or refine_text_only:
        r = refine_crop.remote(
            png,
            prompt=refine_prompt,
            negative_prompt=refine_negative,
            strength=refine_strength,
            steps=refine_steps,
            guidance=refine_guidance,
            seed=refine_seed,
            text_only=refine_text_only,
            candidates=refine_candidates,
            pick=refine_pick,
        )
        png, refiner = r["png"], r["meta"]
        (dest / "refined.png").write_bytes(png)
        if r["candidates"]:
            (dest / "candidates.png").write_bytes(r["candidates"])
        print(
            f"wrote {dest}/refined.png  {len(png) // 1000} KB  "
            f"({refiner['model']}, {refiner['mode']}, seed {refiner['seedPicked']})"
        )
        # --remove-bg describes the input crop; the refined image is an opaque product shot
        # whatever the crop was, so the 3D stage always mattes it itself.
        remove_bg = True

    res = image_to_3d.remote(
        png,
        seed=seed,
        ss_steps=ss_steps,
        slat_steps=slat_steps,
        ss_cfg=ss_cfg,
        slat_cfg=slat_cfg,
        remove_bg=remove_bg,
        want_glb=glb,
        texture_size=texture_size,
        preview=preview,
        preview_video=preview_video,
        refiner=refiner,
        note=note,
    )
    (dest / "object.ply").write_bytes(res["ply"])
    if res["glb"]:
        (dest / "object.glb").write_bytes(res["glb"])
    if res["contact"]:
        (dest / "contact.png").write_bytes(res["contact"])
    (dest / "cond.png").write_bytes(res["cond"])
    if res["preview"]:
        (dest / "preview.mp4").write_bytes(res["preview"])
    (dest / "meta.json").write_text(json.dumps(res["meta"], indent=2))
    m = res["meta"]["ply"]
    print(
        f"wrote {dest}/object.ply  {len(res['ply']) // 1000} KB  "
        f"{m['gaussians']} gaussians  extents {m['extents']}  mean rgb {m['meanRGB']}"
    )
    if res["glb"]:
        print(f"wrote {dest}/object.glb  {len(res['glb']) // 1000} KB")
    elif res["meta"]["glbError"]:
        print("no GLB; see meta.json glbError")
