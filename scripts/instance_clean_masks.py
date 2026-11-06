#!/usr/bin/env python3
"""Opt-in, local CPU person-instance masks; no downloads, inpainting, or paid submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
from fractions import Fraction
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from worker.wander_worker.source_timing import resample_source  # noqa: E402

CHECKPOINT = "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
MODEL = "torchvision.maskrcnn_resnet50_fpn_v2"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_dimensions(clip):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(clip),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    width, height = stream["width"], stream["height"]
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Source has invalid video dimensions")
    return width, height


def cached_predictor(weights=None):
    """Construct the existing architecture without any pretrained/backbone download path."""
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    path = (
        Path(weights)
        if weights is not None
        else Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT
    )
    if not path.is_file():
        raise ValueError(f"Cached Mask R-CNN checkpoint is required; no download attempted: {path}")
    digest = file_hash(path)
    if not digest.startswith("73cbd019"):
        raise ValueError(
            "Cached Mask R-CNN checkpoint checksum does not match the expected weights"
        )
    torch.set_num_threads(4)
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None).cpu().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def predict(rgb):
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255)
        with torch.inference_mode():
            prediction = model([tensor])[0]
        return {
            key: prediction[key].detach().cpu().numpy()
            for key in ("labels", "scores", "boxes", "masks")
        }

    return predict, {
        "model": MODEL,
        "checkpoint": str(path.resolve()),
        "checkpointSha256": digest,
        "checkpointBytes": path.stat().st_size,
        "device": "cpu",
        "torchThreads": 4,
        "torchVersion": torch.__version__,
        "torchvisionVersion": torchvision.__version__,
    }


def person_instances(prediction, height, width, *, score_threshold=0.5):
    """Validate model output before unioning every supported person, never one largest instance."""
    if (
        not isinstance(prediction, dict)
        or not {"labels", "scores", "boxes", "masks"} <= prediction.keys()
    ):
        raise ValueError("Malformed person-instance prediction")
    labels, scores, boxes, masks = (
        np.asarray(prediction[key]) for key in ("labels", "scores", "boxes", "masks")
    )
    count = len(labels) if labels.ndim == 1 else -1
    if (
        labels.dtype.kind not in "iu"
        or scores.shape != (count,)
        or boxes.shape != (count, 4)
        or masks.shape != (count, 1, height, width)
        or not all(np.issubdtype(array.dtype, np.number) for array in (scores, boxes, masks))
        or not all(np.isfinite(array).all() for array in (scores, boxes, masks))
        or np.any((scores < 0) | (scores > 1))
        or np.any((masks < 0) | (masks > 1))
    ):
        raise ValueError("Malformed person-instance prediction geometry or values")
    keep = (labels == 1) & (scores > score_threshold)
    instances = masks[keep, 0] > 0.5
    if not len(instances) or np.any(~instances.reshape(len(instances), -1).any(axis=1)):
        raise ValueError("Missing or empty supported person instance; frame requires review")
    return instances, scores[keep], boxes[keep]


def _save_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


def _contact_tile(rgb, mask, label):
    from PIL import Image, ImageDraw

    overlay = rgb.copy()
    overlay[mask] = (rgb[mask] * 0.45 + np.array([255, 0, 0]) * 0.55).astype(np.uint8)
    tile = Image.new("RGB", (640, 206), "#181818")
    tile.paste(Image.fromarray(rgb).resize((320, 180)), (0, 26))
    tile.paste(Image.fromarray(overlay).resize((320, 180)), (320, 26))
    ImageDraw.Draw(tile).text((8, 6), label, fill="white")
    return tile


def _save_contacts(directory, tiles, index):
    from PIL import Image

    sheet = Image.new("RGB", (2560, 206 * math.ceil(len(tiles) / 4)), "#181818")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (640 * (i % 4), 206 * (i // 4)))
    sheet.save(directory / f"contact-{index:03d}.jpg", quality=93)


def build_instance_clean_masks(
    clip, out, *, fps=12.0, dilate=2, weights=None, contact_sheets=False
):
    """Create a new directory. A usable masks.npz exists only after every frame succeeds.

    Empty detections fail closed, including genuine no-person frames: this intentionally
    conservative helper needs explicit review rather than silently inventing coverage.
    """
    clip, out = Path(clip).resolve(), Path(out)
    if not clip.is_file():
        raise ValueError("Source clip must be an existing local file")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    if type(dilate) is not int or not 0 <= dilate <= 128:
        raise ValueError("Dilation must be an integer from 0 to 128 pixels")
    if out.exists():
        raise ValueError("Output directory already exists; preserve it and choose a new output")
    started = time.monotonic()
    report = {
        "schema": "wander.instance-clean-masks/1",
        "status": "running",
        "acceptance": "none",
        "source": str(clip),
        "parameters": {
            "fps": fps,
            "dilatePixels": dilate,
            "scoreThreshold": 0.5,
            "maskThreshold": 0.5,
            "personLabel": 1,
            "emptyFramePolicy": "reject",
            "columnExtension": False,
            "movedContentUnion": False,
        },
        "limitations": [
            "Mask coverage is not inpainting-quality acceptance.",
            "Reflections, motion blur and occlusion may remain incomplete.",
            "Only detected COCO person instances are included; carried objects and shadows are not guaranteed.",
            "No person-instance identity or temporal-consistency claim.",
        ],
        "generatorCodeSha256": file_hash(__file__),
        "frames": [],
    }
    # No-network guard is process-local; the only subprocess decodes the existing local clip.
    with (
        patch(
            "socket.create_connection",
            side_effect=RuntimeError("Network disabled for cached mask generation"),
        ),
        patch(
            "socket.socket.connect",
            side_effect=RuntimeError("Network disabled for cached mask generation"),
        ),
    ):
        predict, model_metadata = cached_predictor(weights)
        report["model"] = model_metadata
        width, height = source_dimensions(clip)
        raw, provenance = resample_source(clip, fps, width=width, height=height)
        frames = provenance.get("frames", [])
        if not frames or file_hash(clip) != provenance.get("sourceSha256"):
            raise ValueError("Source sampling is empty or its content hash changed")
        indices = [f.get("sourceIndex") for f in frames]
        pts = [f.get("sourcePts") for f in frames]
        if (
            any(type(value) is not int or value < 0 for value in indices)
            or any(type(value) is not int for value in pts)
            or any(b <= a for a, b in zip(indices, indices[1:]))
            or any(b <= a for a, b in zip(pts, pts[1:]))
            or Fraction(provenance["sourceTimeBase"]) <= 0
        ):
            raise ValueError("Malformed source frame indices, PTS or timebase")
        images = np.frombuffer(raw, np.uint8)
        if width <= 0 or height <= 0 or images.size != len(frames) * height * width * 3:
            raise ValueError("Decoded source pixels disagree with sampling metadata")
        images = images.reshape(len(frames), height, width, 3)
        out.mkdir(parents=True, exist_ok=False)
        (out / "frames").mkdir()
        report["sourceProvenance"] = provenance
        report["shape"] = [len(frames), height, width]
        _save_report(out / "report.json", report)
        packed = np.empty((len(frames), height, (width + 7) // 8), np.uint8)
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        tiles, contact_index = [], 0
        try:
            for i, (rgb, source_frame) in enumerate(zip(images, frames, strict=True)):
                tick = time.monotonic()
                instances, scores, boxes = person_instances(predict(rgb), height, width)
                union = instances.any(axis=0)
                mask = cv2.dilate(union.astype(np.uint8), kernel).astype(bool) if dilate else union
                packed[i] = np.packbits(mask, axis=-1)
                np.savez_compressed(
                    out / "frames" / f"{i:06d}.npz",
                    masks=np.packbits(instances, axis=-1),
                    shape=np.array(instances.shape),
                    scores=scores,
                    boxes=boxes,
                    source_index=np.array(source_frame["sourceIndex"]),
                    source_pts=np.array(source_frame["sourcePts"]),
                )
                report["frames"].append(
                    {
                        **source_frame,
                        "instances": len(instances),
                        "scores": scores.tolist(),
                        "boxesXYXY": boxes.tolist(),
                        "rawUnionPx": int(union.sum()),
                        "maskPx": int(mask.sum()),
                        "seconds": time.monotonic() - tick,
                    }
                )
                _save_report(out / "report.json", report)
                if contact_sheets:
                    tiles.append(
                        _contact_tile(
                            rgb,
                            mask,
                            f"output {i} | source {source_frame['sourceIndex']} | {source_frame['sourceTimeSeconds']:.3f}s | {len(instances)} people",
                        )
                    )
                    if len(tiles) == 24:
                        _save_contacts(out, tiles, contact_index)
                        tiles, contact_index = [], contact_index + 1
                print(f"Masks {i + 1}/{len(frames)}: {len(instances)} person instances", flush=True)
            if tiles:
                _save_contacts(out, tiles, contact_index)
            if file_hash(clip) != provenance["sourceSha256"]:
                raise ValueError("Source content changed during mask generation")
            temporary = out / "masks.partial.npz"
            np.savez_compressed(
                temporary,
                masks=packed,
                shape=np.array([len(frames), height, width]),
                source_indices=np.array([f["sourceIndex"] for f in frames]),
                source_pts=np.array([f["sourcePts"] for f in frames]),
                source_sha256=np.array(provenance["sourceSha256"]),
                source_time_base=np.array(provenance["sourceTimeBase"]),
                fps=np.array(fps),
                checkpoint_sha256=np.array(model_metadata["checkpointSha256"]),
                generator_sha256=np.array(report["generatorCodeSha256"]),
            )
            temporary.replace(out / "masks.npz")
            report.update(status="complete", maskArchiveSha256=file_hash(out / "masks.npz"))
        except Exception as error:
            report.update(
                status="failed", errorType=type(error).__name__, failedFrame=len(report["frames"])
            )
            raise
        finally:
            report["seconds"] = time.monotonic() - started
            _save_report(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument(
        "--out",
        required=True,
        help="New directory for masks.npz, report.json and per-frame instances",
    )
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument(
        "--dilate",
        type=int,
        default=2,
        help="Small symmetric padding in pixels; no column extension",
    )
    parser.add_argument(
        "--weights", help="Existing checkpoint path; default is the local Torch hub cache"
    )
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args()
    report = build_instance_clean_masks(
        args.clip,
        args.out,
        fps=args.fps,
        dilate=args.dilate,
        weights=args.weights,
        contact_sheets=args.contact_sheets,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "shape": report["shape"],
                "seconds": report["seconds"],
                "acceptance": report["acceptance"],
            }
        )
    )


if __name__ == "__main__":
    main()
