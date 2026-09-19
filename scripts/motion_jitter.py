#!/usr/bin/env python
"""Per-splat motion jitter for packaged person sequences.

Splat index is stable across frames (same canonical body posed per frame), so
splat i at frame t is the same body point: position differences are real motion.
"""
import argparse, json, sys
import numpy as np

HDR_END = b"end_header\n"

def read_xyz(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    i = buf.index(HDR_END) + len(HDR_END)
    hdr = buf[:i].decode("ascii", "replace")
    n = int([l for l in hdr.splitlines() if l.startswith("element vertex")][0].split()[-1])
    props = [l.split()[-1] for l in hdr.splitlines() if l.startswith("property float")]
    arr = np.frombuffer(buf, dtype="<f4", count=n * len(props), offset=i).reshape(n, len(props))
    return arr[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)

def load(seqdir, limit_idx=None):
    seq = json.load(open(f"{seqdir}/sequence.json"))
    frames = seq["frames"]
    src = seq.get("sourceIndices") or list(range(len(frames)))
    keep = range(len(frames))
    if limit_idx is not None:
        want = set(limit_idx)
        keep = [k for k in range(len(frames)) if src[k] in want]
    P = np.stack([read_xyz(f"{seqdir}/{frames[k]}") for k in keep])
    return seq, P, [src[k] for k in keep]

def smooth(P, win):
    """Moving average along time, reflect-padded. win odd."""
    h = win // 2
    Q = np.pad(P, ((h, h), (0, 0), (0, 0)), mode="reflect")
    k = np.ones(win) / win
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, Q)

def metrics(P, fps, to_m):
    """P (T,N,3) in world units. Returns dict of per-frame arrays in metres."""
    Pm = P * to_m
    dt = 1.0 / fps
    cen = Pm.mean(1)                                   # centroid track
    acc_c = np.full(len(Pm), np.nan)
    acc_c[1:-1] = np.linalg.norm(cen[2:] - 2 * cen[1:-1] + cen[:-2], axis=-1) / dt**2
    A = (Pm[2:] - 2 * Pm[1:-1] + Pm[:-2]) / dt**2      # per-splat accel
    acc_s = np.full(len(Pm), np.nan)
    acc_s[1:-1] = np.linalg.norm(A, axis=-1).mean(1)
    # high-frequency residual vs a ~0.25 s moving average (fps-independent window)
    win = max(3, int(round(0.25 * fps)) | 1)
    R = Pm - smooth(Pm, win)
    hf = np.sqrt((R**2).sum(-1).mean(1))               # RMS residual per frame, m
    return dict(centroid=cen, acc_centroid=acc_c, acc_splat=acc_s, hf=hf, win=win)

def bands(P0):
    y = P0[:, 1]
    q = np.quantile(y, [0, .2, .45, .7, .88, 1.0])
    lab = ["low 20% (feet/shins)", "20-45% (legs)", "45-70% (hips/torso)",
           "70-88% (chest/arms)", "top 12% (head)"]
    return [(lab[i], (y >= q[i]) & (y <= q[i + 1] + 1e-9)) for i in range(5)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqdir"); ap.add_argument("--to-m", type=float, default=1.0)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    seq, P, src = load(a.seqdir)
    m = metrics(P, seq["fps"], a.to_m)
    print(f"{a.seqdir}: T={len(P)} fps={seq['fps']} splats={P.shape[1]} win={m['win']}")
    print(f"  centroid accel  mean {np.nanmean(m['acc_centroid']):8.2f}  p95 {np.nanpercentile(m['acc_centroid'],95):8.2f} m/s^2")
    print(f"  per-splat accel mean {np.nanmean(m['acc_splat']):8.2f}  p95 {np.nanpercentile(m['acc_splat'],95):8.2f} m/s^2")
    print(f"  HF residual RMS mean {1000*m['hf'].mean():8.1f}  p95 {1000*np.percentile(m['hf'],95):8.1f} mm")
    rng = np.linalg.norm(m["centroid"] - m["centroid"][0], axis=-1)
    print(f"  root travel max {rng.max():.3f} m ; bbox y extent {(P[0][:,1].max()-P[0][:,1].min())*a.to_m:.3f} m")
    for lab, mask in bands(P[0]):
        mb = metrics(P[:, mask], seq["fps"], a.to_m)
        print(f"    {lab:24s} accel {np.nanmean(mb['acc_splat']):7.2f} m/s^2   HF {1000*mb['hf'].mean():6.1f} mm")
