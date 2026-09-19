#!/usr/bin/env python3
"""Turn the tracking stage's per-track Mask R-CNN masks into a plate-cleaning mask archive.

  worker/.venv-da3/bin/python scripts/tracks_to_clean_masks.py .context/mp/<clip>/tracks \
      --width 1920 --height 1080 --dilate 30 --bottom-extra 60 --out .context/mp/<clip>/clean-masks.npz
  <modal> run worker/modal_clean_video.py --clip ... --masks-in .context/mp/<clip>/clean-masks.npz ...

The clean pass segments people with SegFormer, which is what the single-person pipeline has always
used and which is fine most of the time -- but on the elevator clip it leaves one man's light grey
trousers standing on the light floor in the last second, because SegFormer loses a low-contrast
lower body. The tracking stage has already run Mask R-CNN on every sampled frame for every track,
so the union of those instance masks is a better plate mask that costs nothing extra.

`--masks-in` skips the clean pass's own dilation and downward extension, so both are applied here
with the same numbers and the same kernel anchoring as worker/modal_clean_video.py.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--dilate", type=int, default=20)
    ap.add_argument("--bottom-extra", type=int, default=40)
    a = ap.parse_args()

    root = Path(a.tracks)
    doc = json.loads((root / "tracks.json").read_text())
    store = np.load(root / "masks.npz")
    sh = store["shape"]
    mh, mw = int(round(a.height * float(sh[2]))), int(round(a.width * float(sh[2])))
    n, H, W = doc["samples"], a.height, a.width

    masks = np.zeros((n, H, W), bool)
    per_sample = {}
    for key in store.files:
        if not key.startswith("t"):
            continue
        tid, _, srest = key[1:].partition("_s")
        per_sample.setdefault(int(srest), []).append(key)
    missing = 0
    for s in range(n):
        keys = per_sample.get(s, [])
        if not keys:
            missing += 1
            continue
        small = np.zeros((mh, mw), bool)
        for key in keys:
            m = np.unpackbits(store[key])[: mh * mw].reshape(mh, mw).astype(bool)
            small |= m
        masks[s] = cv2.resize(small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    if a.dilate > 0:
        k = np.ones((2 * a.dilate + 1, 2 * a.dilate + 1), np.uint8)
        masks = np.stack([cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks])
    # same downward extension as worker/modal_clean_video.py: carry every occupied column to the
    # bottom of the frame so the contact shadow goes with the person
    for m in masks:
        cols = np.where(m[int(0.6 * H):].any(0))[0]
        if len(cols):
            low = np.argmax(m[::-1], axis=0)
            for c in cols:
                m[H - 1 - low[c]:, c] = True
    if a.bottom_extra > 0:
        k = np.ones((a.bottom_extra + 1, 2 * a.dilate + 1), np.uint8)
        masks = np.stack([cv2.dilate(m.astype(np.uint8), k, anchor=(a.dilate, a.bottom_extra)).astype(bool)
                          for m in masks])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, masks=np.packbits(masks, axis=-1), shape=np.array([n, H, W]),
                        source=np.array([str(root)]))
    print(json.dumps(dict(out=a.out, samples=n, samplesWithoutMask=missing,
                          tracks=doc["trackCount"], maskFraction=float(masks.mean()),
                          framesCovered=int(masks.reshape(n, -1).any(1).sum())), indent=1))


if __name__ == "__main__":
    main()
