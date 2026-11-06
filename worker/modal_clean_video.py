"""GPU person-removal pass: the scripts/clean_person_video.py path on an L4.

Same three stages and the same numbers as the CPU reference (SegFormer people masks, dilated and
extended downward for the shadow/contact, LaMa at --lama-px with a feathered alpha blend, libx264),
but the masks run batched on CUDA and LaMa runs on CUDA, which is ~40x faster per frame. The clip
itself is uploaded and decoded remotely, so no frame dumps cross the wire.

  uv run --locked modal run worker/modal_clean_video.py --clip public/clips/bedroom.mp4 \
      --out .context/clips/bedroom-clean.mp4 \
      --frame0 .context/own/bedroom/clean-f0.png --dilate 40 --bottom-extra 80

Weights live in the wander-clean-video-cache volume (big-lama.pt staged once with `uv run --locked modal volume put`,
SegFormer downloaded into the HF cache on the first run), so a run never re-downloads them.
"""

from __future__ import annotations

from pathlib import Path

import modal

if modal.is_local():
    from worker.stages.worker_resources import inference_resources
else:
    from worker_resources import inference_resources

HERE = Path(__file__).resolve().parent
INFERENCE_RESOURCES = inference_resources(timeout=3600, memory=32768, max_containers=2)
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
image = image.add_local_file(HERE / "stages/worker_resources.py", "/root/worker_resources.py")
app = modal.App("wander-clean-video")
cache = modal.Volume.from_name("wander-clean-video-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/cache": cache},
    **INFERENCE_RESOURCES,
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
    inpaint_mode: str = "legacy",
) -> dict:
    """Returns the clean mp4, frame 0, a masks archive and the source indices of the kept frames.

    only: comma-separated indices into the decoded (post-fps) sequence, for a quick look or a
    numerical comparison against the CPU reference. return_frames: also hand back those frames as
    lossless PNGs, so a caller can diff pixels without an encoder in the way.

    moved: inpaint everything that MOVED, not just the person - the plate is what the static
    trainer sees, and a cast shadow, a carried object or a passer-by is as wrong in it as the
    person is. Off by default: every existing caller keeps the person mask it has always had.

    inpaint_mode: isolated is an experimental context-isolation and opaque-compositing path.
    It preserves pixels outside the reviewed mask before encoding; it does not establish quality.
    """
    import hashlib
    import io
    import subprocess
    import tarfile
    import tempfile
    import time
    import traceback

    import cv2
    import numpy as np
    from PIL import Image
    from wander_worker.source_timing import resample_source, select_provenance, select_source_frames

    t0 = time.time()
    root = Path(tempfile.mkdtemp(prefix="cleanvid-"))
    src, work = root / "input.mp4", root / "frames"
    work.mkdir()
    src.write_bytes(video_bytes)
    W, H = width, height
    error, timings = None, {}
    result = dict(
        sourceSha256=hashlib.sha256(video_bytes).hexdigest(),
        suppliedMasksSha256=hashlib.sha256(masks_npz).hexdigest() if masks_npz else None,
        inpaintMode=inpaint_mode,
        fps=fps,
        width=W,
        height=H,
        dilate=dilate,
        bottomExtra=bottom_extra,
        passes=passes,
        lamaPx=lama_px,
        crf=crf,
    )
    moved_stats = None
    out_mp4 = out_png = masks_out = b""
    frames_tar = b""
    composite_checks = []
    try:
        if inpaint_mode not in {"legacy", "isolated"}:
            raise ValueError("inpaint_mode must be legacy or isolated")
        t = time.time()
        supplied = None
        explicit = False
        if masks_npz:
            with np.load(io.BytesIO(masks_npz), allow_pickle=False) as archive:
                supplied = {key: archive[key] for key in archive.files}
            fields = {"source_indices", "source_pts", "source_time_base", "source_sha256"}
            present = fields.intersection(supplied)
            if present and present != fields:
                raise ValueError("Supplied masks have incomplete explicit source metadata")
            explicit = bool(present)
        if explicit:
            source_indices = supplied["source_indices"]
            source_pts = supplied["source_pts"]
            if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (source_indices, source_pts)):
                raise ValueError("Supplied source indices and PTS must be integer vectors")
            if any(
                supplied[key].shape != () or supplied[key].dtype.kind != "U"
                for key in ("source_time_base", "source_sha256")
            ):
                raise ValueError("Supplied source timebase and hash must be scalar strings")
            if str(supplied["source_sha256"]) != hashlib.sha256(video_bytes).hexdigest():
                raise ValueError("Supplied masks source hash mismatch")
            raw, provenance = select_source_frames(
                src, fps, source_indices=source_indices.tolist(), width=W, height=H
            )
            if (
                source_pts.tolist() != [f["sourcePts"] for f in provenance["frames"]]
                or str(supplied["source_time_base"]) != provenance["sourceTimeBase"]
                or str(supplied["source_sha256"]) != provenance["sourceSha256"]
            ):
                raise ValueError("Supplied masks source PTS/timebase/hash mismatch")
            expected = [len(source_indices), H, W]
            shape = supplied.get("shape")
            packed = supplied.get("masks")
            if (
                shape is None
                or shape.dtype.kind not in "iu"
                or shape.tolist() != expected
                or packed is None
                or packed.dtype != np.uint8
                or packed.shape != (len(source_indices), H, (W + 7) // 8)
            ):
                raise ValueError("Supplied masks shape does not match exact decoded grid")
        else:
            raw, provenance = resample_source(src, fps, width=W, height=H)
        imgs = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
        del raw
        timings["decode"] = time.time() - t
        n = len(imgs)
        indices = [frame["sourceIndex"] for frame in provenance["frames"]]
        num, den = provenance["sourceAverageFrameRate"].split("/")
        src_fps = float(num) / float(den) if float(den) else 0.0
        result.update(
            frames=n,
            indices=indices,
            sourceFps=src_fps,
            sourceProvenance=provenance,
        )
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
            masks = np.unpackbits(supplied["masks"], axis=-1)[:, :H, :W].astype(bool)
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
        result.update(
            moved=moved,
            movedStats=moved_stats,
            maskSource="supplied" if masks_npz else "moved" if moved else "segformer",
            appliedDilate=0 if masks_npz else dilate,
            appliedBottomExtra=0 if masks_npz else bottom_extra,
            extendedToBottom=not bool(masks_npz),
            maskFraction=float(masks[need].mean()),
            framesWithPeople=covered,
        )
        print(
            f"masks: people {masks[need].mean() * 100:.2f}% of pixels, {covered}/{len(need)} frames "
            f"({timings['masks']:.0f}s)",
            flush=True,
        )

        t = time.time()
        import torch
        from simple_lama_inpainting import SimpleLama
        from wander_worker.inpaint import inpaint_frame

        assert torch.cuda.is_available(), "no CUDA device"
        lama = SimpleLama(torch.device("cuda"))
        timings["lamaLoad"] = time.time() - t

        t = time.time()
        kept = []
        for i in sel:
            decoded_rgb_sha256 = hashlib.sha256(imgs[i].tobytes()).hexdigest()
            # Capture the approved write boundary before calling the inpainter. Comparing
            # against a separately decoded local video cannot prove byte preservation:
            # different FFmpeg versions can produce slightly different RGB values.
            outside = ~masks[i] if inpaint_mode == "isolated" else None
            original_outside = imgs[i][outside] if outside is not None else None
            mask_sha256 = hashlib.sha256(masks[i].tobytes()).hexdigest()
            im = inpaint_frame(
                imgs[i], masks[i], lama, lama_px=lama_px, passes=passes, mode=inpaint_mode
            )
            if outside is not None:
                if im.shape != imgs[i].shape or im.dtype != np.uint8:
                    raise ValueError("Isolated inpainting returned invalid RGB pixels")
                cleaned_outside = im[outside]
                changed = int(np.count_nonzero(np.any(cleaned_outside != original_outside, axis=1)))
                composite_checks.append(
                    {
                        "frameIndex": i,
                        "sourceIndex": indices[i],
                        "sourcePts": provenance["frames"][i]["sourcePts"],
                        "decodedRgbSha256": decoded_rgb_sha256,
                        "removalMaskSha256": mask_sha256,
                        "outsideMaskPixels": len(original_outside),
                        "outsideMaskChangedPixels": changed,
                        "sourceOutsideRgbSha256": hashlib.sha256(
                            original_outside.tobytes()
                        ).hexdigest(),
                        "cleanOutsideRgbSha256": hashlib.sha256(
                            cleaned_outside.tobytes()
                        ).hexdigest(),
                    }
                )
                if changed:
                    raise ValueError(
                        f"Isolated inpainting changed {changed} pixels outside the mask at frame {i}"
                    )
            Image.fromarray(im).save(work / f"f_{i:04d}.png")
            kept_provenance[len(kept)]["decodedRgbSha256"] = decoded_rgb_sha256
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
            source_indices=np.array(indices),
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
        result.update(
            inpainted=len(kept),
            keptIndices=kept,
            keptFrameProvenance=kept_provenance,
            outputVideoSha256=hashlib.sha256(out_mp4).hexdigest() if out_mp4 else None,
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
        timeoutSeconds=INFERENCE_RESOURCES["timeout"],
        timings=timings,
        compositeChecks=composite_checks,
        estimatedComputeUSD=elapsed * (0.000222 + 4 * 0.0000131 + 32 * 0.00000222),
        **result,
    )
    return dict(report=report, mp4=out_mp4, frame0=out_png, masks=masks_out, frames=frames_tar)


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
    inpaint_mode: str = "legacy",
):
    import io
    import json
    import tarfile

    if inpaint_mode not in {"legacy", "isolated"}:
        raise ValueError("--inpaint-mode must be legacy or isolated")
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
        inpaint_mode=inpaint_mode,
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
