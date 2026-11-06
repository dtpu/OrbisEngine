#!/usr/bin/env python3
"""Offline CPU LaMa diagnostics from exact-source masks and explicitly cached weights.

No downloads, provider clients, candidate adoption, or quality acceptance. ``progress.json``
is retained on failure; ``clean.json`` is published only after all selected outputs succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.wander_worker.inpaint import inpaint_frame  # noqa: E402
from worker.wander_worker.source_timing import select_source_frames  # noqa: E402


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixel_hash(pixels):
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_threads(threads):
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be an integer from 1 to 32")


def load_cpu_model(weights, *, threads=4):
    """Load only the supplied TorchScript file; never import a downloading wrapper."""
    import torch

    validate_threads(threads)
    torch.set_num_threads(threads)
    model = torch.jit.load(str(weights), map_location="cpu")
    model.eval()

    def predict(image, mask):
        rgb = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
        binary = (np.array(mask) > 0).astype(np.int64)[None]
        if image.mode != "RGB" or mask.size != image.size or any(n % 8 for n in image.size):
            raise ValueError("Model input must be matching RGB/mask dimensions divisible by eight")
        with torch.inference_mode():
            output = model(torch.from_numpy(rgb[None]), torch.from_numpy(binary[None]))
        if tuple(output.shape) != (1, 3, image.height, image.width):
            raise ValueError("Model returned an unexpected tensor shape")
        values = output[0].permute(1, 2, 0).detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Model returned nonfinite pixels")
        return Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8))

    return predict


def validate_masks(clip, masks_path):
    """Validate the entire archive against observed source metadata before model loading."""
    with np.load(masks_path, allow_pickle=False) as archive:
        required = {
            "shape",
            "masks",
            "source_indices",
            "source_pts",
            "source_time_base",
            "source_sha256",
        }
        if not required.issubset(archive.files):
            raise ValueError("Masks require complete exact-source metadata")
        data = {key: archive[key] for key in required}
    shape = data["shape"]
    indices, pts = data["source_indices"], data["source_pts"]
    if shape.shape != (3,) or shape.dtype.kind not in "iu" or np.any(shape <= 0):
        raise ValueError("Masks shape must contain positive integer count, height, width")
    count, height, width = map(int, shape)
    if any(a.shape != (count,) or a.dtype.kind not in "iu" for a in (indices, pts)):
        raise ValueError("Source indices and PTS must be complete integer vectors")
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("Source indices must be strictly increasing")
    if any(
        data[k].shape != () or data[k].dtype.kind != "U"
        for k in ("source_time_base", "source_sha256")
    ):
        raise ValueError("Source hash and timebase must be scalar strings")
    if str(data["source_sha256"]) != sha256(clip):
        raise ValueError("Masks source hash mismatch")
    packed = data["masks"]
    if packed.dtype != np.uint8 or packed.shape != (count, height, (width + 7) // 8):
        raise ValueError("Packed masks must be uint8 with exact source-grid shape")
    # Reject nonzero padding rather than silently discarding archive data.
    if width % 8 and np.any(packed[:, :, -1] & ((1 << (8 - width % 8)) - 1)):
        raise ValueError("Packed masks have nonzero padding bits")
    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,time_base:frame=pts",
                "-of",
                "json",
                str(clip),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        ).stdout
    )
    stream = probe["streams"][0]
    source_pts = [int(frame["pts"]) for frame in probe["frames"]]
    if not source_pts or any(b <= a for a, b in zip(source_pts, source_pts[1:])):
        raise ValueError("Source PTS must be complete and strictly increasing")
    if (stream["height"], stream["width"]) != (height, width):
        raise ValueError("Mask dimensions do not match source dimensions")
    if str(data["source_time_base"]) != stream["time_base"]:
        raise ValueError("Masks source timebase mismatch")
    if any(not 0 <= int(i) < len(source_pts) for i in indices):
        raise ValueError("Masks source index out of range")
    if pts.tolist() != [source_pts[int(i)] for i in indices]:
        raise ValueError("Masks source PTS mismatch")
    return data


def source_patch_hash(recipe):
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_source_patches(recipe, data, fps):
    """Check a complete measured recipe against the archive, without loading a model."""
    import cv2

    fields = {
        "schema",
        "sourceSha256",
        "sourceTimeBase",
        "sourceIndices",
        "sourcePts",
        "fps",
        "width",
        "height",
        "patches",
    }
    if not isinstance(recipe, dict) or set(recipe) != fields:
        raise ValueError("Source patch recipe has invalid fields")
    count, height, width = map(int, data["shape"])
    if (
        recipe["schema"] != "wander.source-patches/1"
        or recipe["sourceSha256"] != str(data["source_sha256"])
        or recipe["sourceTimeBase"] != str(data["source_time_base"])
        or type(recipe["fps"]) not in (int, float)
        or not math.isfinite(recipe["fps"])
        or recipe["fps"] != fps
        or type(recipe["width"]) is not int
        or recipe["width"] != width
        or type(recipe["height"]) is not int
        or recipe["height"] != height
    ):
        raise ValueError("Source patch recipe source/grid/FPS mismatch")
    for key, expected in (
        ("sourceIndices", data["source_indices"]),
        ("sourcePts", data["source_pts"]),
    ):
        values = recipe[key]
        if (
            not isinstance(values, list)
            or any(type(i) is not int for i in values)
            or values != expected.tolist()
        ):
            raise ValueError("Source patch recipe exact source grid mismatch")
    patches = recipe["patches"]
    if not isinstance(patches, list) or not patches:
        raise ValueError("Source patch recipe needs nonempty patches")
    targets = set()
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {
            "donorSample",
            "targetSample",
            "homography",
            "donorRoiXYXY",
            "featherPx",
            "maskMarginPx",
        }:
            raise ValueError("Source patch has invalid fields")
        for key in ("donorSample", "targetSample"):
            if type(patch[key]) is not int or not 0 <= patch[key] < count:
                raise ValueError("Source patch sample index out of range")
        if patch["targetSample"] in targets:
            raise ValueError("Only one source patch per target sample is supported")
        targets.add(patch["targetSample"])
        for key in ("featherPx", "maskMarginPx"):
            if type(patch[key]) is not int or patch[key] != 8:
                raise ValueError("Source patches require eight-pixel feather and mask margin")
        matrix = patch["homography"]
        if (
            not isinstance(matrix, list)
            or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row
            )
        ):
            raise ValueError("Source patch homography must be finite numeric 3x3")
        H = np.array(matrix, dtype=np.float64)
        if not math.isclose(H[2, 2], 1, rel_tol=0, abs_tol=1e-12) or abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("Source patch homography must be normalized and nonsingular")
        roi = patch["donorRoiXYXY"]
        if (
            not isinstance(roi, list)
            or len(roi) != 4
            or any(type(v) is not int for v in roi)
            or not 0 <= roi[0] < roi[2] <= width
            or not 0 <= roi[1] < roi[3] <= height
        ):
            raise ValueError("Source patch donor ROI must be bounded integer XYXY")
        x0, y0, x1, y1 = roi
        corners = np.float64([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]])
        projected = corners @ H.T
        # Reject projective poles through the ROI, not just singular matrices.
        if (
            not np.isfinite(projected).all()
            or np.any(np.abs(projected[:, 2]) < 1e-12)
            or np.min(projected[:, 2]) < 0 < np.max(projected[:, 2])
        ):
            raise ValueError("Source patch homography crosses a projective pole")
        donor_mask = np.unpackbits(data["masks"][patch["donorSample"]], axis=1, count=width)
        excluded = cv2.dilate(donor_mask, np.ones((17, 17), np.uint8)) > 0
        if excluded[y0:y1, x0:x1].any():
            raise ValueError("Source patch donor ROI overlaps the dilated removal mask")
    source_patch_hash(recipe)  # Also rejects nonfinite JSON outside numerical matrix fields.
    return {patch["targetSample"]: patch for patch in patches}


def apply_source_patch(lama_rgb, target_mask, donor_rgb, donor_mask, patch):
    """Replay only the measured source-guided composition, never LaMa inference."""
    import cv2

    height, width = target_mask.shape
    H = np.array(patch["homography"], dtype=np.float64)
    x0, y0, x1, y1 = patch["donorRoiXYXY"]
    roi = np.zeros((height, width), np.float32)
    roi[y0:y1, x0:x1] = 1
    valid = 1 - cv2.dilate(donor_mask.astype(np.uint8), np.ones((17, 17), np.uint8))
    warped = cv2.warpPerspective(donor_rgb, H, (width, height), flags=cv2.INTER_LINEAR)
    warped_roi = cv2.warpPerspective(roi, H, (width, height), flags=cv2.INTER_LINEAR) >= 0.999
    warped_valid = (
        cv2.warpPerspective(valid.astype(np.float32), H, (width, height), flags=cv2.INTER_LINEAR)
        >= 0.999
    )
    support = target_mask & warped_roi & warped_valid
    alpha = np.clip(cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5) / 8, 0, 1)
    mixed = np.clip(
        warped.astype(np.float32) * alpha[..., None]
        + lama_rgb.astype(np.float32) * (1 - alpha[..., None]),
        0,
        255,
    ).astype(np.uint8)
    result = lama_rgb.copy()
    result[support] = mixed[support]
    if not np.array_equal(result[~target_mask], lama_rgb[~target_mask]):
        raise ValueError("Source patch changed outside-mask pixels")
    evidence = {
        "donorSample": patch["donorSample"],
        "targetSample": patch["targetSample"],
        "donorDecodedRgbSha256": pixel_hash(donor_rgb),
        "donorRemovalMaskSha256": pixel_hash(donor_mask),
        "supportSha256": pixel_hash(support),
        "sourceSupportPixels": int(support.sum()),
        "fullySourcedPixels": int((alpha == 1).sum()),
        "featherPixels": int(((alpha > 0) & (alpha < 1)).sum()),
        "sourceAlphaSum": float(alpha.sum()),
        "sourcePatchChangedPixels": int(np.any(result != lama_rgb, axis=2).sum()),
        "outsideMaskChangedPixels": 0,
        "outsideSupportChangedPixels": 0,
    }
    return result, evidence


def clean_video(
    *,
    clip,
    masks,
    weights,
    out,
    only=None,
    lama_px=960,
    fps=12.0,
    encode=False,
    threads=4,
    source_patches=None,
    model_loader=load_cpu_model,
):
    started = time.monotonic()
    validate_threads(threads)
    if Path(out).expanduser().is_symlink():
        raise FileExistsError("Output directory must be new, not a symlink")
    clip, masks, weights, out = (
        Path(p).expanduser().resolve() for p in (clip, masks, weights, out)
    )
    if not math.isfinite(fps) or fps <= 0 or type(lama_px) is not int or lama_px < 8:
        raise ValueError("FPS must be finite and positive; lama-px must be at least eight")
    for path in (clip, masks, weights):
        if not path.is_file():
            raise ValueError(f"Required local input file is missing: {path}")
    # Reserve before any work, with no exist_ok and no output reuse on failure.
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "wander.local-clean/1",
        "status": "running",
        "qualityStatus": "unreviewed",
        "sourceSha256": sha256(clip),
        "suppliedMasksSha256": sha256(masks),
        "modelSha256": sha256(weights),
        "device": "cpu",
        "inpaintMode": "isolated",
        "parameters": {
            "fps": fps,
            "lamaPx": lama_px,
            "encode": encode,
            "passes": 1,
            "threads": threads,
        },
        "codeHashes": {
            name: sha256(ROOT / name)
            for name in (
                "scripts/local_clean_video.py",
                "worker/wander_worker/inpaint.py",
                "worker/wander_worker/source_timing.py",
            )
        },
        "compositeChecks": [],
        "timings": {},
    }
    progress = out / "progress.json"
    write_report(progress, report)
    try:
        data = validate_masks(clip, masks)
        count, height, width = map(int, data["shape"])
        if int(lama_px * height / width) // 8 * 8 <= 0:
            raise ValueError("LaMa target height rounds to zero")
        selected = list(range(count)) if only is None else list(only)
        if (
            not selected
            or any(type(i) is not int or not 0 <= i < count for i in selected)
            or selected != sorted(set(selected))
        ):
            raise ValueError("--only must contain unique increasing sample indices in range")
        if encode and selected != list(range(count)):
            raise ValueError("H264 encoding requires the full sample selection")
        patch_path = (
            Path(source_patches).expanduser().resolve() if source_patches is not None else None
        )
        patches = {}
        if patch_path is not None:
            recipe_bytes = patch_path.read_bytes()
            recipe_file_hash = hashlib.sha256(recipe_bytes).hexdigest()
            recipe = json.loads(recipe_bytes)
            patches = validate_source_patches(recipe, data, fps)
            report.update(
                compositionMethod="LaMa plus measured source-guided composite",
                sourcePatches=recipe,
                sourcePatchesSha256=source_patch_hash(recipe),
                sourcePatchesFileSha256=recipe_file_hash,
                sourcePatchesPath="source-patches.json",
                sourcePatchAlgorithm="wander.source-patch-composite/1",
            )
            (out / "source-patches.json").write_bytes(recipe_bytes)
            write_report(progress, report)
        decoded_samples = sorted(
            set(selected) | {patches[i]["donorSample"] for i in selected if i in patches}
        )
        indices = data["source_indices"][decoded_samples].tolist()
        raw, provenance = select_source_frames(
            clip, fps, source_indices=indices, width=width, height=height
        )
        expected_pts = data["source_pts"][decoded_samples].tolist()
        if (
            provenance["sourceSha256"] != report["sourceSha256"]
            or provenance["sourceTimeBase"] != str(data["source_time_base"])
            or [f["sourcePts"] for f in provenance["frames"]] != expected_pts
            or sha256(masks) != report["suppliedMasksSha256"]
        ):
            raise ValueError("Inputs changed during validation/decode")
        decoded_frames = dict(zip(decoded_samples, provenance["frames"], strict=True))
        for index, frame in decoded_frames.items():
            frame["frameIndex"] = index
        provenance["frames"] = [decoded_frames[i] for i in selected]
        report.update(
            sourceProvenance=provenance,
            selectedSampleIndices=selected,
            frames=count,
            width=width,
            height=height,
        )
        report["timings"]["validationDecodeSeconds"] = time.monotonic() - started
        write_report(progress, report)
        loaded = time.monotonic()
        model = model_loader(weights, threads=threads)
        if sha256(weights) != report["modelSha256"]:
            raise ValueError("Weights changed while loading")
        report["timings"]["modelLoadSeconds"] = time.monotonic() - loaded
        images = dict(
            zip(
                decoded_samples,
                np.frombuffer(raw, np.uint8).reshape(len(decoded_samples), height, width, 3),
                strict=True,
            )
        )
        for index, frame in zip(selected, provenance["frames"], strict=True):
            image = images[index]
            t = time.monotonic()
            mask = np.unpackbits(data["masks"][index], axis=1, count=width).astype(bool)
            outside = ~mask
            original = image[outside].copy()
            check = {
                **frame,
                "decodedRgbSha256": pixel_hash(image),
                "removalMaskSha256": pixel_hash(mask),
                "maskPixels": int(mask.sum()),
                "outsideMaskPixels": int(outside.sum()),
                "sourceOutsideRgbSha256": pixel_hash(original),
                "status": "running",
            }
            report["compositeChecks"].append(check)
            write_report(progress, report)
            cleaned = inpaint_frame(
                image.copy(), mask.copy(), model, lama_px=lama_px, mode="isolated"
            )
            if cleaned.shape != image.shape or cleaned.dtype != np.uint8:
                raise ValueError("Inpainting returned invalid RGB dimensions or dtype")
            lama_rgb = cleaned
            if not np.array_equal(lama_rgb[outside], original):
                check.update(
                    outsideMaskChangedPixels=int(
                        np.any(lama_rgb[outside] != original, axis=1).sum()
                    ),
                    cleanOutsideRgbSha256=pixel_hash(lama_rgb[outside]),
                )
                raise ValueError(f"Outside-mask pixels changed at sample {index}")
            if index in patches:
                patch = patches[index]
                donor_index = patch["donorSample"]
                donor_mask = np.unpackbits(data["masks"][donor_index], axis=1, count=width).astype(
                    bool
                )
                cleaned, patch_evidence = apply_source_patch(
                    lama_rgb, mask, images[donor_index], donor_mask, patch
                )
                patch_evidence["donorProvenance"] = decoded_frames[donor_index]
                check.update(
                    sourcePatch=patch_evidence,
                    lamaRgbSha256=pixel_hash(lama_rgb),
                    lamaPath=f"lama_{index:04d}.png",
                )
            changes = int(np.count_nonzero(np.any(cleaned[outside] != original, axis=1)))
            check.update(
                outsideMaskChangedPixels=changes, cleanOutsideRgbSha256=pixel_hash(cleaned[outside])
            )
            if changes:
                raise ValueError(f"Outside-mask pixels changed at sample {index}")
            if index in patches:
                lama_path = out / check["lamaPath"]
                Image.fromarray(lama_rgb).save(lama_path)
                check["lamaPngSha256"] = sha256(lama_path)
            path = out / f"f_{index:04d}.png"
            Image.fromarray(cleaned).save(path)
            with Image.open(path) as saved:
                if saved.mode != "RGB" or not np.array_equal(np.asarray(saved), cleaned):
                    raise ValueError("PNG round-trip changed RGB pixels")
            check.update(
                status="complete",
                cleanedRgbSha256=pixel_hash(cleaned),
                pngSha256=sha256(path),
                path=path.name,
                seconds=time.monotonic() - t,
            )
            write_report(progress, report)
        if encode:
            destination = out / "clean.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-n",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(out / "f_%04d.png"),
                    "-frames:v",
                    str(count),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(destination),
                ],
                check=True,
                capture_output=True,
            )
            report["video"] = {
                "path": destination.name,
                "sha256": sha256(destination),
                "audio": "absent",
            }
        for path, key in (
            (clip, "sourceSha256"),
            (masks, "suppliedMasksSha256"),
            (weights, "modelSha256"),
        ):
            if sha256(path) != report[key]:
                raise ValueError("Inputs changed during cleaning; final report withheld")
        if patch_path is not None and sha256(patch_path) != recipe_file_hash:
            raise ValueError("Source patch recipe changed during cleaning")
        report["status"] = "complete"
        report["timings"]["totalSeconds"] = time.monotonic() - started
        write_report(progress, report)
        write_report(out / "clean.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", errorType=type(error).__name__, error=str(error))
        report["timings"]["totalSeconds"] = time.monotonic() - started
        write_report(progress, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("clip", "masks", "weights", "out"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--source-patches", help="data-only measured source patch recipe JSON")
    parser.add_argument("--only", help="comma-separated increasing SAMPLE indices")
    parser.add_argument("--lama-px", type=int, default=960)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--threads", type=int, default=4, help="CPU Torch threads (1..32; default 4)"
    )
    parser.add_argument(
        "--encode", action="store_true", help="full-selection silent H264 diagnostic"
    )
    args = parser.parse_args()
    options = vars(args)
    options["only"] = (
        [int(value) for value in args.only.split(",")] if args.only is not None else None
    )
    clean_video(**options)


if __name__ == "__main__":
    main()
