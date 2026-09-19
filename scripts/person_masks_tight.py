#!/usr/bin/env python3
"""Tight (undilated) per-instance person masks for silhouette scoring.

The cached masks under .context/*/masks.npz are grown for inpainting (dilate 40, extra at the
bottom), so they are 3x the area of the real body and useless as an IoU target. This makes
Mask R-CNN instance masks instead, and when a seed box is given picks the instance that
overlaps it -- which is how the right person is chosen in a multi-person clip.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

_rcnn = None


def rcnn():
    global _rcnn
    if _rcnn is None:
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
        m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
        dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        _rcnn = (m.to(dev), dev, torch)
    return _rcnn


def masks_for(frames, seeds=None, score=0.7):
    """frames uint8 (N,H,W,3); seeds optional list of (x0,y0,x1,y1) or None."""
    model, dev, torch = rcnn()
    out = np.zeros(frames.shape[:3], bool)
    with torch.no_grad():
        for i, im in enumerate(frames):
            t = torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(dev)
            p = model([t])[0]
            keep = (p["labels"] == 1) & (p["scores"] > score)
            if not keep.any():
                continue
            ms = (p["masks"][keep, 0] > 0.5).cpu().numpy()
            if seeds is not None and seeds[i] is not None:
                x0, y0, x1, y1 = seeds[i]
                box = np.zeros(im.shape[:2], bool); box[max(0, y0):y1, max(0, x0):x1] = True
                ov = [(m & box).sum() / max(m.sum(), 1) for m in ms]
                out[i] = ms[int(np.argmax(ov))] if max(ov) > 0.15 else ms[0]
            else:
                out[i] = ms.any(0)
    return out


def read_frames(clip, idx):
    cap = cv2.VideoCapture(clip); want = sorted(set(int(i) for i in idx)); got = {}
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n in want:
            got[n] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        n += 1
        if len(got) == len(want):
            break
    cap.release()
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", help="json: {frameIndex: [x0,y0,x1,y1]}")
    a = ap.parse_args()
    idx = [int(x) for x in a.frames.split(",")]
    fr = read_frames(a.clip, idx)
    idx = [i for i in idx if i in fr]
    arr = np.stack([fr[i] for i in idx])
    seeds = None
    if a.seeds:
        s = json.loads(Path(a.seeds).read_text()); seeds = [s.get(str(i)) for i in idx]
    m = masks_for(arr, seeds)
    np.savez_compressed(a.out, masks=np.packbits(m, axis=-1), shape=np.array(m.shape),
                        indices=np.array(idx))
    print(f"{a.out}: {len(idx)} frames, mean {m.mean()*100:.2f}% of pixels, "
          f"{int(m.reshape(len(m),-1).any(1).sum())}/{len(m)} non-empty")
