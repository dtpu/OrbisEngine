#!/usr/bin/env python3
"""Build a plate-cleaning archive from complete, source-aligned tracking instance masks.

  uv run --locked python scripts/tracks_to_clean_masks.py RUN/tracks \
      --clip source.mp4 --fps 12 --width 1920 --height 1080 --out RUN/track-clean-masks.npz

The semantic clean pass can miss low-contrast lower bodies. Tracking has already produced
per-person masks. Their exact recorded source frames are retained in the archive so cleaning
can select those same pixels, rather than applying masks to a nearby frame on the FPS grid.
Missing, empty, malformed, or mismatched masks stop before an archive is written.

`--masks-in` skips the clean worker's own dilation and downward extension, so this helper
applies them with the same kernel anchoring as worker/modal_clean_video.py.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from worker.wander_worker.source_timing import resample_source, select_source_frames  # noqa: E402


def build_clean_masks(tracks, *, clip, out, width, height, fps, dilate=20, bottom_extra=40):
    """Validate all retained masks against observed clean inputs, then union and expand them."""
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Clean-mask dimensions must be positive integers")
    if any(type(value) is not int or value < 0 for value in (dilate, bottom_extra)):
        raise ValueError("Clean-mask margins must be nonnegative integers")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Clean-mask FPS must be positive and finite")
    root, out = Path(tracks), Path(out)
    document_path, masks_path = root / "tracks.json", root / "masks.npz"
    try:
        document_bytes = document_path.read_bytes()
        doc = json.loads(document_bytes)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Tracking metadata is unavailable or malformed: {document_path}"
        ) from error
    if not isinstance(doc, dict):
        raise ValueError("Tracking metadata must be an object")
    indices = doc.get("sourceIndices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or index < 0 for index in indices)
    ):
        raise ValueError("Tracking masks require an explicit source frame grid")
    _, sampled = resample_source(clip, fps)
    if len(indices) != len(sampled["frames"]):
        raise ValueError("Tracking mask frame count does not match observed clean source sampling")
    _, provenance = select_source_frames(clip, fps, source_indices=indices)
    if provenance["sourceSha256"] != sampled["sourceSha256"]:
        raise ValueError("Source changed while validating tracking mask sampling")
    frames = provenance["frames"]
    n = len(frames)
    if doc.get("sourceSha256") != provenance["sourceSha256"]:
        raise ValueError("Tracking masks do not belong to this source SHA-256")
    if (doc.get("width"), doc.get("height")) != (width, height):
        raise ValueError("Tracking masks do not match the source dimensions")
    if (
        type(doc.get("samples")) is not int
        or doc["samples"] != n
        or not n
        or type(doc.get("fps")) not in (int, float)
        or not math.isfinite(doc["fps"])
        or not math.isclose(doc["fps"], fps, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("Tracking mask frame grid does not match observed clean source sampling")
    timestamps = doc.get("timestamps")
    if (
        not isinstance(timestamps, list)
        or len(timestamps) != n
        or any(
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or abs(timestamp - frame["sourceRelativeTimeSeconds"]) > 1e-6
            for timestamp, frame in zip(timestamps, frames, strict=True)
        )
    ):
        raise ValueError("Tracking mask timestamps do not match observed clean source sampling")
    tracks = doc.get("tracks")
    count = doc.get("trackCount")
    if type(count) is not int or count <= 0 or not isinstance(tracks, list) or len(tracks) != count:
        raise ValueError("Tracking masks require a nonempty complete track inventory")
    expected, identifiers = set(), set()
    for track in tracks:
        if not isinstance(track, dict) or type(track.get("track")) is not int:
            raise ValueError("Malformed track identity for cleaning masks")
        track_id, samples = track["track"], track.get("samples")
        if track_id in identifiers or not 0 <= track_id < count:
            raise ValueError("Duplicate or out-of-range track identity for cleaning masks")
        identifiers.add(track_id)
        if (
            not isinstance(samples, list)
            or not samples
            or any(type(sample) is not int or not 0 <= sample < n for sample in samples)
            or samples != sorted(set(samples))
        ):
            raise ValueError("Malformed track sample inventory for cleaning masks")
        expected.update((track_id, sample) for sample in samples)
    if {sample for _, sample in expected} != set(range(n)):
        raise ValueError("Tracking masks have incomplete frame coverage")
    try:
        store = np.load(masks_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Tracking mask archive is unavailable or malformed: {masks_path}"
        ) from error
    if not isinstance(store, np.lib.npyio.NpzFile):
        raise ValueError("Tracking masks must be an NPZ archive")
    masks = np.zeros((n, height, width), bool)
    with store:
        shape = store["shape"] if "shape" in store.files else None
        if (
            shape is None
            or shape.shape != (3,)
            or not np.issubdtype(shape.dtype, np.number)
            or not np.isfinite(shape).all()
            or tuple(shape[:2]) != (height, width)
            or not 0 < shape[2] <= 1
        ):
            raise ValueError("Tracking mask archive has invalid source shape or mask scale")
        mh, mw = round(height * float(shape[2])), round(width * float(shape[2]))
        if min(mh, mw) <= 0:
            raise ValueError("Tracking mask scale produces empty dimensions")
        per_sample, actual = {}, set()
        for key in store.files:
            if key == "shape":
                continue
            match = re.fullmatch(r"t(\d+)_s(\d+)", key)
            if match is None:
                raise ValueError(f"Unrecognized tracking mask entry: {key}")
            identity = tuple(map(int, match.groups()))
            if identity in actual or identity not in expected:
                raise ValueError("Tracking mask archive disagrees with the track sample inventory")
            actual.add(identity)
            packed = store[key]
            if packed.dtype != np.uint8 or packed.shape != (math.ceil(mh * mw / 8),):
                raise ValueError(f"Tracking mask has invalid packed shape or dtype: {key}")
            mask = np.unpackbits(packed)[: mh * mw].reshape(mh, mw).astype(bool)
            if not mask.any():
                raise ValueError(f"Tracking mask is empty: {key}")
            per_sample.setdefault(identity[1], np.zeros((mh, mw), bool))[:] |= mask
        if actual != expected:
            raise ValueError("Tracking mask archive is missing declared track samples")
        for sample, small in per_sample.items():
            masks[sample] = cv2.resize(
                small.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        for mask in masks:
            mask[:] = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    for mask in masks:
        columns = np.where(mask[int(0.6 * height) :].any(0))[0]
        low = np.argmax(mask[::-1], axis=0)
        for column in columns:
            mask[height - 1 - low[column] :, column] = True
    if bottom_extra > 0:
        kernel = np.ones((bottom_extra + 1, 2 * dilate + 1), np.uint8)
        for mask in masks:
            mask[:] = cv2.dilate(
                mask.astype(np.uint8), kernel, anchor=(dilate, bottom_extra)
            ).astype(bool)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        masks=np.packbits(masks, axis=-1),
        shape=np.array([n, height, width]),
        source_sha256=np.array(provenance["sourceSha256"]),
        source_indices=np.array(indices),
        source_pts=np.array([frame["sourcePts"] for frame in frames]),
        source_time_base=np.array(provenance["sourceTimeBase"]),
        fps=np.array(fps),
        tracksSha256=np.array(hashlib.sha256(document_bytes).hexdigest()),
        source=np.array([str(root.resolve())]),
    )
    return dict(
        out=str(out),
        samples=n,
        samplesWithoutMask=0,
        tracks=count,
        maskFraction=float(masks.mean()),
        framesCovered=n,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tracks")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--dilate", type=int, default=20)
    ap.add_argument("--bottom-extra", type=int, default=40)
    args = ap.parse_args()
    print(json.dumps(build_clean_masks(**vars(args)), indent=1))


if __name__ == "__main__":
    main()
