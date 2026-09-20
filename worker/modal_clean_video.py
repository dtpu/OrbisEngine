"""GPU person-removal pass: the scripts/clean_person_video.py path on an L4.

SegFormer people masks are dilated and extended downward for shadow/contact, then LaMa fills
the removal mask with feathering only outside its boundary before libx264 encoding. Masks and
LaMa run on CUDA. The clip is uploaded and decoded remotely; first-frame diagnostic PNGs return
with source bindings, model identity and hashes so blending and model failures can be separated.

  uv run --locked modal run worker/modal_clean_video.py --clip public/clips/bedroom.mp4 \
      --out .context/clips/bedroom-clean.mp4 \
      --frame0 .context/own/bedroom/clean-f0.png --dilate 40 --bottom-extra 80

Weights live in the wander-clean-video-cache volume (big-lama.pt staged once with `uv run --locked modal volume put`,
SegFormer downloaded into the HF cache on the first run), so a run never re-downloads them.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LAMA_PT = "/cache/lama/big-lama.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.5.1",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "opencv-python-headless==4.11.0.86",
        "transformers==4.46.3",
        "safetensors",
    )
    .uv_pip_install("simple-lama-inpainting==0.1.2", extra_options="--no-deps")
    .env(
        {
            "PYTHONPATH": "/root/worker",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
            "LAMA_MODEL": LAMA_PT,
        }
    )
    .add_local_dir(HERE / "wander_worker", "/root/worker/wander_worker", ignore=["__pycache__"])
)
app = modal.App("wander-clean-video")
cache = modal.Volume.from_name("wander-clean-video-cache", create_if_missing=True)


def composite_fill(image, fill, mask):
    """Replace every removal-mask pixel; feather only outside its boundary."""
    import cv2
    import numpy as np

    image, fill, mask = np.asarray(image), np.asarray(fill), np.asarray(mask)
    if (
        image.ndim != 3
        or image.shape[-1] != 3
        or fill.shape != image.shape
        or mask.shape != image.shape[:2]
        or not mask.size
    ):
        raise ValueError("Expected matching nonempty HWC RGB images and HW mask")
    if not all(np.isfinite(value).all() for value in (image, fill, mask)):
        raise ValueError("Composite inputs must be finite")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Removal mask must be binary")
    if any((value < 0).any() or (value > 255).any() for value in (image, fill)):
        raise ValueError("Composite image values must be in [0, 255]")
    mask = mask.astype(bool)
    alpha = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3) * 1.5, 0, 1)
    alpha = np.where(mask, 1.0, alpha)[..., None]
    return (image * (1 - alpha) + fill * alpha).astype(np.uint8)


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(32768, 32768),
    timeout=3600,
    retries=0,
    max_containers=2,
    volumes={"/cache": cache},
    scaledown_window=2,
)
def clean(
    video_bytes: bytes,
    fps: float = 12.0,
    width: int = 1920,
    height: int = 1080,
    dilate: int = 20,
    bottom_extra: int = 40,
    passes: int = 1,
    lama_px: int = 960,
    crf: int = 18,
    only: str | None = None,
    mask_batch: int = 8,
    return_frames: bool = False,
    encode: bool = True,
    masks_npz: bytes = b"",
    moved: bool = False,
    max_added_frac: float = 0.05,
) -> dict:
    """Returns the clean mp4, frame 0, a masks archive and the source indices of the kept frames.

    only: comma-separated indices into the decoded (post-fps) sequence, for a quick look or a
    numerical comparison against the CPU reference. return_frames: also hand back those frames as
    lossless PNGs, so a caller can diff pixels without an encoder in the way.

    moved: inpaint everything that MOVED, not just the person - the plate is what the static
    trainer sees, and a cast shadow, a carried object or a passer-by is as wrong in it as the
    person is. Off by default: every existing caller keeps the person mask it has always had.
    """
    import hashlib
    import importlib.metadata
    import io
    import subprocess
    import tarfile
    import tempfile
    import time
    import traceback

    import cv2
    import numpy as np
    from PIL import Image
    from wander_worker.source_timing import resample_source, select_provenance

    t0 = time.time()
    root = Path(tempfile.mkdtemp(prefix="cleanvid-"))
    src, work = root / "input.mp4", root / "frames"
    work.mkdir()
    src.write_bytes(video_bytes)
    W, H = width, height
    error, timings, result = None, {}, {}
    moved_stats = None
    out_mp4 = out_png = masks_out = b""
    frames_tar = b""
    diagnostic_images = {}
    diagnostics = {"schema": "wander.clean-first-frame-diagnostics/1", "images": {}}

    def capture(name, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        diagnostic_images[name] = data
        diagnostics["images"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "width": image.width,
            "height": image.height,
        }

    try:
        t = time.time()
        raw, provenance = resample_source(src, fps, width=W, height=H)
        imgs = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
        del raw
        timings["decode"] = time.time() - t
        n = len(imgs)
        indices = [frame["sourceIndex"] for frame in provenance["frames"]]
        num, den = provenance["sourceAverageFrameRate"].split("/")
        src_fps = float(num) / float(den) if float(den) else 0.0
        print(
            f"{n} frames decoded from {src_fps:.3f} fps source ({timings['decode']:.0f}s)",
            flush=True,
        )

        t = time.time()
        from wander_worker.masks import moved_content_masks, people_masks

        sel = list(range(n)) if not only else [int(x) for x in only.split(",")]
        kept_provenance = select_provenance(provenance, sel)
        # a partial run (the orchestrator's single first frame, or a quality gate) only needs its own masks
        need = np.array(sel) if only else np.arange(n)
        masks = np.zeros((n, H, W), bool)
        if moved and not masks_npz:
            # the residual needs neighbouring frames, so this one runs over the whole sequence
            # even when only a few frames are inpainted
            mres = moved_content_masks(
                imgs, dilate_px=dilate, batch_size=mask_batch, max_added_frac=max_added_frac
            )
            masks = mres["moved"]
            moved_stats = mres["stats"]
            timings["masks"] = time.time() - t
        elif masks_npz:
            # supplied masks (the quality gate feeds in the CPU reference's own) skip segmentation
            # and the extension below, so the comparison isolates LaMa
            masks = np.unpackbits(np.load(io.BytesIO(masks_npz))["masks"], axis=-1)[
                :, :H, :W
            ].astype(bool)
            timings["masks"] = time.time() - t
        else:
            masks[need] = people_masks(imgs[need], dilate_px=dilate, batch_size=mask_batch)
        for j in need if not masks_npz else []:
            m = masks[j]  # scalar index: a view, so the extension below writes back
            cols = np.where(m[int(0.6 * H) :].any(0))[0]
            if len(cols):
                low = np.argmax(m[::-1], axis=0)
                for c in cols:
                    m[H - 1 - low[c] :, c] = True
        if bottom_extra > 0 and not masks_npz:
            k = np.ones((bottom_extra + 1, 2 * dilate + 1), np.uint8)
            masks[need] = np.stack(
                [
                    cv2.dilate(m.astype(np.uint8), k, anchor=(dilate, bottom_extra)).astype(bool)
                    for m in masks[need]
                ]
            )
        timings["masks"] = time.time() - t
        covered = int(masks[need].reshape(len(need), -1).any(1).sum())
        print(
            f"masks: people {masks[need].mean() * 100:.2f}% of pixels, {covered}/{len(need)} frames "
            f"({timings['masks']:.0f}s)",
            flush=True,
        )

        t = time.time()
        import torch
        from simple_lama_inpainting import SimpleLama

        assert torch.cuda.is_available(), "no CUDA device"
        lama = SimpleLama(torch.device("cuda"))
        timings["lamaLoad"] = time.time() - t
        diagnostics["packageVersion"] = importlib.metadata.version("simple-lama-inpainting")
        with Path(LAMA_PT).open("rb") as weights:
            diagnostics["modelSha256"] = hashlib.file_digest(weights, "sha256").hexdigest()
        diagnostics["modelPath"] = LAMA_PT

        t = time.time()
        sw, sh = lama_px // 8 * 8, int(lama_px * H / W) // 8 * 8
        kept = []
        for i in sel:
            im, m = imgs[i].copy(), masks[i]
            diagnostic_frame = not kept
            if diagnostic_frame:
                diagnostics["source"] = dict(kept_provenance[0])
                diagnostics["sourceSha256"] = provenance["sourceSha256"]
                diagnostics["sourceTimeBase"] = provenance["sourceTimeBase"]
                diagnostics["inpaintingApplied"] = bool(m.any() and passes > 0)
                capture("source.png", Image.fromarray(im))
                capture("removal-mask.png", Image.fromarray(m.astype(np.uint8) * 255))
            if m.any():
                for pass_index in range(passes):
                    small = Image.fromarray(im).resize((sw, sh), Image.LANCZOS)
                    sm = Image.fromarray(m.astype(np.uint8) * 255).resize((sw, sh), Image.NEAREST)
                    if diagnostic_frame:
                        capture(f"pass-{pass_index:02d}-model-rgb.png", small)
                        capture(f"pass-{pass_index:02d}-model-mask.png", sm)
                    raw_fill = lama(small, sm)
                    if diagnostic_frame:
                        capture(f"pass-{pass_index:02d}-raw-fill.png", raw_fill)
                    fill = np.asarray(raw_fill.resize((W, H), Image.BICUBIC))
                    im = composite_fill(im, fill, m)
                    if diagnostic_frame:
                        capture(f"pass-{pass_index:02d}-composite.png", Image.fromarray(im))
            if diagnostic_frame:
                capture("composite.png", Image.fromarray(im))
            Image.fromarray(im).save(work / f"f_{i:04d}.png")
            kept_provenance[len(kept)]["cleanedImageSha256"] = hashlib.sha256(
                (work / f"f_{i:04d}.png").read_bytes()
            ).hexdigest()
            kept.append(i)
            if len(kept) % 20 == 0:
                print(f"  {len(kept)}/{len(sel)} ({time.time() - t:.0f}s)", flush=True)
        timings["inpaint"] = time.time() - t
        print(f"inpainted {len(kept)} frames ({timings['inpaint']:.0f}s)", flush=True)

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            masks=np.packbits(masks, axis=-1),
            shape=np.array([n, H, W]),
            indices=np.array(indices),
            source_pts=np.array([frame["sourcePts"] for frame in provenance["frames"]]),
            source_time_base=np.array(provenance["sourceTimeBase"]),
            source_sha256=np.array(provenance["sourceSha256"]),
        )
        masks_out = buf.getvalue()

        if encode and not only:
            t = time.time()
            enc = root / "clean.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(work / "f_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(enc),
                ],
                check=True,
            )
            out_mp4 = enc.read_bytes()
            out_png = (work / "f_0000.png").read_bytes()
            timings["encode"] = time.time() - t
        if return_frames:
            fb = io.BytesIO()
            with tarfile.open(fileobj=fb, mode="w") as tar:
                for i in kept:
                    tar.add(work / f"f_{i:04d}.png", arcname=f"f_{i:04d}.png")
            frames_tar = fb.getvalue()
        result = dict(
            frames=n,
            inpainted=len(kept),
            indices=indices,
            sourceFps=src_fps,
            sourceSha256=provenance["sourceSha256"],
            sourceProvenance=provenance,
            keptIndices=kept,
            keptFrameProvenance=kept_provenance,
            outputVideoSha256=hashlib.sha256(out_mp4).hexdigest() if out_mp4 else None,
            moved=moved,
            movedStats=moved_stats,
            fps=fps,
            width=W,
            height=H,
            dilate=dilate,
            bottomExtra=bottom_extra,
            passes=passes,
            lamaPx=lama_px,
            crf=crf,
            maskFraction=float(masks[need].mean()),
            framesWithPeople=covered,
        )
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        cache.commit()
    elapsed = time.time() - t0
    report = dict(
        seconds=elapsed,
        error=error,
        gpu="L4",
        cpu=4,
        timings=timings,
        estimatedComputeUSD=elapsed * (0.000222 + 4 * 0.0000131 + 32 * 0.00000222),
        diagnostics=diagnostics,
        **result,
    )
    return dict(
        report=report,
        mp4=out_mp4,
        frame0=out_png,
        masks=masks_out,
        frames=frames_tar,
        diagnostics=diagnostic_images,
    )


@app.function(image=image, cpu=2, memory=8192, timeout=900, volumes={"/cache": cache})
def stage_weights() -> dict:
    """Pull SegFormer into the volume's HF cache and check big-lama.pt is staged."""
    import os

    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    from wander_worker.masks import MODEL_ID

    AutoImageProcessor.from_pretrained(MODEL_ID)
    SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    cache.commit()
    return dict(
        lama=os.path.exists(LAMA_PT),
        lamaBytes=os.path.getsize(LAMA_PT) if os.path.exists(LAMA_PT) else 0,
        segformer=MODEL_ID,
    )


@app.local_entrypoint()
def main(
    clip: str,
    out: str = "",
    frame0: str = "",
    masks: str = "",
    report: str = "",
    fps: float = 12.0,
    width: int = 1920,
    height: int = 1080,
    dilate: int = 20,
    bottom_extra: int = 40,
    passes: int = 1,
    lama_px: int = 960,
    crf: int = 18,
    only: str = "",
    frames_out: str = "",
    masks_in: str = "",
    moved_mask: bool = False,
    max_added_frac: float = 0.05,
):
    import io
    import json
    import tarfile

    r = clean.remote(
        Path(clip).read_bytes(),
        fps=fps,
        width=width,
        height=height,
        dilate=dilate,
        bottom_extra=bottom_extra,
        passes=passes,
        lama_px=lama_px,
        crf=crf,
        only=only or None,
        return_frames=bool(frames_out),
        encode=not only,
        masks_npz=Path(masks_in).read_bytes() if masks_in else b"",
        moved=moved_mask,
        max_added_frac=max_added_frac,
    )
    rep = r["report"]
    if out and r["mp4"]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(r["mp4"])
    if frame0 and r["frame0"]:
        Path(frame0).parent.mkdir(parents=True, exist_ok=True)
        Path(frame0).write_bytes(r["frame0"])
    if masks and r["masks"]:
        Path(masks).parent.mkdir(parents=True, exist_ok=True)
        Path(masks).write_bytes(r["masks"])
    if frames_out and r["frames"]:
        d = Path(frames_out)
        d.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(r["frames"]), mode="r") as tar:
            tar.extractall(d, filter="data")
    diagnostic_base = report or frame0 or out or masks or frames_out
    if diagnostic_base and r.get("diagnostics"):
        diagnostic_path = Path(diagnostic_base).with_suffix("")
        diagnostic_path = diagnostic_path.with_name(diagnostic_path.name + "-diagnostics")
        diagnostic_path.mkdir(parents=True, exist_ok=True)
        for name, data in r["diagnostics"].items():
            if Path(name).name != name:
                raise ValueError("Unexpected diagnostic image filename")
            (diagnostic_path / name).write_bytes(data)
        (diagnostic_path / "manifest.json").write_text(json.dumps(rep["diagnostics"], indent=2))
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(rep, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in rep.items()
                if k not in ("indices", "movedStats", "sourceProvenance", "keptFrameProvenance")
            },
            indent=2,
        )
    )
    if rep["error"]:
        raise SystemExit("GPU clean pass failed")
