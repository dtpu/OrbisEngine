#!/usr/bin/env python
"""Compose the share figures for a finetune run dir (renders from finetune_gsplat.py):
finetune-vs-source-{0,275,549}.png = source | Marble | fine-tuned at the recorded pose,
finetune-offpath-{0.5m,1.5m}.png = Marble | fine-tuned 0.5 / 1.5 m right of the frame-275 pose.

    uv run --locked --group inference python scripts/finetune_figures.py --run <dir with before-/after-*.png> --data <export dir> --share <dir> [--tag finetune]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def label(im: Image.Image, texts):
    d = ImageDraw.Draw(im)
    w = im.width // len(texts)
    for i, t in enumerate(texts):
        d.rectangle([i * w, 0, i * w + 8 * len(t) + 12, 18], fill=(0, 0, 0))
        d.text((i * w + 6, 3), t, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--share", type=Path, default=Path(__file__).resolve().parents[1] / ".context/share"
    )
    ap.add_argument("--tag", default="finetune")
    ap.add_argument("--poses", default="0,275,549")
    ap.add_argument("--offpath-frame", type=int, default=275)
    ap.add_argument("--world-label", default="Marble recon-f0-v2 (as is)")
    ap.add_argument("--offpath", default="0.5,1.5")
    a = ap.parse_args()
    for f in [int(x) for x in a.poses.split(",")]:
        src = np.asarray(Image.open(a.data / "frames" / f"{f:04d}.jpg"))
        b = np.asarray(Image.open(a.run / f"before-pose-{f}.png"))
        c = np.asarray(Image.open(a.run / f"after-pose-{f}.png"))
        im = label(
            Image.fromarray(np.concatenate([src, b, c], 1)),
            [f"source frame {f}", a.world_label, "fine-tuned (gsplat)"],
        )
        im.save(a.share / f"{a.tag}-vs-source-{f}.png")
    for m in a.offpath.split(","):
        b = np.asarray(Image.open(a.run / f"before-offpath-{m}m.png"))
        c = np.asarray(Image.open(a.run / f"after-offpath-{m}m.png"))
        im = label(
            Image.fromarray(np.concatenate([b, c], 1)),
            [
                f"Marble, {m} m right of frame {a.offpath_frame}",
                f"fine-tuned, {m} m right of frame {a.offpath_frame}",
            ],
        )
        im.save(a.share / f"{a.tag}-offpath-{m}m.png")
    print("wrote", sorted(p.name for p in a.share.glob(f"{a.tag}-*.png")))


if __name__ == "__main__":
    main()
