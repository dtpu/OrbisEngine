#!/usr/bin/env python3
"""Add a compact motion track beside a packaged person's frame_NNN.ply files.

  uv run --locked python scripts/package_person_motion.py public/worlds/<clip>-4d/person
  uv run --locked python scripts/package_person_motion.py public/worlds/<clip>-4d/person_01 --quantize

The viewer's interpolated person (the default) reads every frame's PLY only to pull out the seven
values that change between frames: the splat centre (x, y, z) and its rotation (rot_0..3).
All static attributes must be bit-identical in every frame of the sequence.
This writes those seven channels for every frame into one little-endian binary file and records
it in sequence.json as `motion`, so a viewer that understands it fetches frame_000.ply plus one
file instead of every PLY. The PLYs stay where they are for the discrete mode and older viewers.

float32 (default) is bit-identical to the PLYs. Explicit --quantize opts into lossy uint16;
maxAbsError records the worst error after the viewer's float32 rounding.
The static attributes are checked against frame 0 for every frame first: a sequence whose colour,
opacity or scale drift would render wrongly from a motion track, so it is refused.
"""

import argparse
import hashlib
import json
from io import BytesIO
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

CHANNELS = ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3")
MOTION_FILES = {"uint16": "motion.u16", "float32": "motion.f32"}


def load_frames(person: Path, seq: dict, frame_hashes: list | None = None):
    """(F, N, 7) float32 motion channels, after checking every static attribute against frame 0."""
    if not isinstance(seq.get("frames"), list) or not seq["frames"]:
        raise ValueError("sequence has no frames")
    first = None
    motion = []
    for name in seq["frames"]:
        raw = (person / name).read_bytes()
        ply = PlyData.read(BytesIO(raw))
        if ply.text or ply.byte_order != "<" or len(ply.elements) != 1:
            raise ValueError(f"{name}: expected one binary little-endian vertex element")
        v = ply["vertex"].data
        if not len(v) or any(v.dtype[field].str != "<f4" for field in v.dtype.names):
            raise ValueError(f"{name}: expected nonempty float32 vertices")
        if frame_hashes is not None:
            frame_hashes.append(hashlib.sha256(raw).hexdigest())
        if first is None:
            first = v
            missing = [
                c
                for c in (
                    *CHANNELS,
                    "f_dc_0",
                    "f_dc_1",
                    "f_dc_2",
                    "opacity",
                    "scale_0",
                    "scale_1",
                    "scale_2",
                )
                if c not in v.dtype.names
            ]
            if missing:
                raise ValueError(f"{name} lacks {missing}; not a gaussian PLY")
        elif len(v) != len(first):
            raise ValueError(f"{name} has {len(v)} splats, frame 0 has {len(first)}")
        if v.dtype.names != first.dtype.names:
            raise ValueError(f"{name}: property layout differs from frame 0")
        for field in v.dtype.names:
            if not np.isfinite(v[field]).all():
                raise ValueError(f"{name}: {field} contains nonfinite values")
            if field in CHANNELS:
                continue
            if v[field].tobytes() != first[field].tobytes():
                raise ValueError(
                    f"{name}: {field} differs from frame 0; this sequence has per-frame "
                    "appearance and cannot use a motion track"
                )
        motion.append(np.column_stack([v[c] for c in CHANNELS]).astype(np.float32))
    return np.stack(motion)


def quantise(motion: np.ndarray):
    """uint16 per channel over the sequence range; returns (codes, lo, scale, max abs error)."""
    flat = motion.reshape(-1, motion.shape[-1]).astype(np.float64)
    lo = flat.min(0)
    hi = flat.max(0)
    scale = (hi - lo) / 65535.0
    safe = np.where(scale == 0, 1.0, scale)
    codes = np.round((flat - lo) / safe).clip(0, 65535).astype(np.uint16)
    back = (codes.astype(np.float64) * scale + lo).astype(np.float32).astype(np.float64)
    err = np.abs(back - flat).max(0)
    return codes.reshape(motion.shape), lo, scale, err


def package(person: Path, lossless: bool = True):
    seq_path = person / "sequence.json"
    seq = json.loads(seq_path.read_text())
    # A failed repack must not leave an older, now stale motion descriptor active.
    if seq.pop("motion", None) is not None:
        seq_path.write_text(json.dumps(seq, indent=2))
    frame_hashes = []
    motion = load_frames(person, seq, frame_hashes)
    frames, splats, _ = motion.shape
    dtype = "float32" if lossless else "uint16"
    out = person / MOTION_FILES[dtype]
    if lossless:
        payload = motion.astype("<f4").tobytes()
        lo = np.zeros(7)
        scale = np.ones(7)
        err = np.zeros(7)
    else:
        codes, lo, scale, err = quantise(motion)
        payload = codes.astype("<u2").tobytes()
    out.write_bytes(payload)
    ply_bytes = sum((person / f).stat().st_size for f in seq["frames"])
    record = {
        "schema": "wander.person-motion/1",
        "file": out.name,
        "frameFiles": list(seq["frames"]),
        "base": {
            "file": seq["frames"][0],
            "bytes": (person / seq["frames"][0]).stat().st_size,
            "sha256": frame_hashes[0],
        },
        "dtype": dtype,
        "frames": frames,
        "splats": splats,
        "channels": list(CHANNELS),
        "layout": "frame-major, then splat, then channel; little-endian; value = code * scale + min",
        "min": lo.tolist(),
        "scale": scale.tolist(),
        "maxAbsError": err.tolist(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "plyBytes": ply_bytes,
        "note": "The first listed PLY is the appearance keyframe; every static attribute was "
        "verified bit-identical across every frame before this track was written",
    }
    seq["frame_sha256"] = frame_hashes
    seq["motion"] = record
    seq_path.write_text(json.dumps(seq, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("person", help="a packaged person directory holding sequence.json")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quantize", action="store_true", help="opt into lossy uint16 motion")
    mode.add_argument("--lossless", action="store_true", help="float32 motion (the default)")
    a = ap.parse_args()
    person = Path(a.person)
    if not (person / "sequence.json").exists():
        sys.exit(f"{person} has no sequence.json")
    record = package(person, lossless=not a.quantize)
    summary = {k: record[k] for k in ("file", "dtype", "frames", "splats", "bytes", "plyBytes")}
    summary["ratio"] = round(record["plyBytes"] / record["bytes"], 2)
    summary["maxAbsError"] = max(record["maxAbsError"])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
