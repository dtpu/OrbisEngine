#!/usr/bin/env python3
"""Per-clip audit of how well the posed avatar matches the person in the footage.

  python scripts/motion_audit.py --out .context/pose/audit.json

For every packaged world with a person: silhouette IoU against tight Mask R-CNN person masks
(as-packaged and after the best 2D shift, so placement error is separable from pose error) plus
the existing motion_jitter numbers. Masks are cached per clip under .context/pose/masks/.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_silhouette as ms
import motion_jitter as mj
import person_masks_tight as pmt

CACHE = Path(".context/pose/masks")


def seed_boxes(world, cams, seq, hw):
    src = list(seq["sourceIndices"]); out = {}
    for c in cams:
        fi = c["sourceIndex"]
        k = int(np.argmin(np.abs(np.array(src) - fi)))
        v = PlyData.read(Path(world) / "person" / seq["frames"][k])["vertex"].data
        xyz = np.column_stack([v["x"], v["y"], v["z"]])
        sc = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])
        sil = ms.splat_silhouette(xyz, np.asarray(v["opacity"]), sc, c, hw, 8000)
        if sil.any():
            ys, xs = np.nonzero(sil)
            pad = 0.45  # the avatar may be mis-placed; give the detector room to find him
            h, w = ys.ptp() + 1, xs.ptp() + 1
            out[fi] = [int(xs.min() - pad * w), int(ys.min() - pad * h),
                       int(xs.max() + pad * w), int(ys.max() + pad * h)]
    return out


def audit_world(world, clip, n=16, subsample=30000):
    world = Path(world)
    cams = json.loads((world / "cameras.json").read_text())["cameras"]
    seq = json.loads((world / "person" / "sequence.json").read_text())
    hw = tuple(cams[0]["source_image_size"][::-1]) if "source_image_size" in cams[0] else None
    if hw and hw[0] < hw[1] * 0.3:
        hw = tuple(cams[0]["source_image_size"])
    pick = [cams[i] for i in np.linspace(0, len(cams) - 1, min(n, len(cams))).round().astype(int)]
    seeds = seed_boxes(world, pick, seq, hw)
    idx = [c["sourceIndex"] for c in pick if c["sourceIndex"] in seeds]
    CACHE.mkdir(parents=True, exist_ok=True)
    mp = CACHE / f"{world.name}.npz"
    if not mp.exists():
        fr = pmt.read_frames(clip, idx)
        idx = [i for i in idx if i in fr]
        arr = np.stack([fr[i] for i in idx])
        m = pmt.masks_for(arr, [seeds[i] for i in idx])
        np.savez_compressed(mp, masks=np.packbits(m, axis=-1), shape=np.array(m.shape), indices=np.array(idx))
    rows = ms.audit(world, mp, limit=None, subsample=subsample)
    rows = [r for r in rows if r["gtPx"] > 500]
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True, help="comma list of world dir names")
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    res = {}
    for w in a.worlds.split(","):
        wd = Path("public/worlds") / w
        seq = json.loads((wd / "person" / "sequence.json").read_text())
        clip = seq.get("sourceClip")
        try:
            rows = audit_world(wd, clip, a.n)
            seq_dir = wd / "person"
            s, P, _ = mj.load(str(seq_dir))
            to_m = 1.0 / seq.get("alignment", {}).get("scaleApplied", 1.0)
            m = mj.metrics(P, s["fps"], to_m)
            res[w] = dict(clip=clip, n=len(rows),
                          iou=float(np.mean([r["iou"] for r in rows])),
                          iouAligned=float(np.mean([r["iouAligned"] for r in rows])),
                          areaRatio=float(np.mean([r["predPx"] / max(r["gtPx"], 1) for r in rows])),
                          accSplat=float(np.nanmean(m["acc_splat"])), hfMm=float(1000 * m["hf"].mean()),
                          rows=rows)
            r = res[w]
            print(f"{w:22s} n={r['n']:3d} IoU {r['iou']:.3f} aligned {r['iouAligned']:.3f} "
                  f"area {r['areaRatio']:.2f} accel {r['accSplat']:6.2f} HF {r['hfMm']:5.1f}mm", flush=True)
        except Exception as e:
            print(f"{w}: FAILED {type(e).__name__}: {e}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
