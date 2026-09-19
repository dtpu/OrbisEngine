#!/usr/bin/env python3
"""Measure a rigid object's tumble, so it does not glide through the air axis-aligned.

An axis-aligned capsule sliding along a perfect parabola reads as fake however good the
trajectory under it is, because real thrown objects rotate. This recovers what the pixels
actually support and says plainly what they do not.

Two measurements, both from data that already exists:

1. **Rate**, from the in-flight blobs (`track_object_2d.py --patches`). A thrown elongated object
   is a motion smear, and the smear's extent *along* the image velocity is dominated by the blur,
   but its extent *perpendicular* to the velocity is not: that one rises and falls as the object
   presents its long axis and then its end. The projected extent of a rod is 180 deg symmetric, so
   one period of the perpendicular-extent signal is **half** a revolution.

2. **Axis**, from the throw itself. An underarm or overarm throw tumbles end over end in the plane
   the object travels in, so the axis is the normal of that plane, `v0 x g`. This is an assumption
   about throws, not a measurement of these pixels, and it is labelled as one in the output.

Phase (which way up it is at release) comes from the segmented in-hand mask's principal axis at the
last frame before release, when `--segment-dir` is given.

Writes an `orientation` block for `wander.objects/2` §5, including its own provenance string.

  object_orientation.py --track2d bottle2d.json --patches bottle-patches.npz \
      --v0 0.82,1.23,-0.06 --g 0,-5.32,0 --fps 30 --out orientation.json
"""

import argparse, json

import numpy as np


def perpendicular_extent(track2d, patches, min_area=200):
    """-> (frames, perpendicular extent in px, parallel extent in px)."""
    t = json.load(open(track2d))
    z = np.load(patches)
    obs = {o["sourceIndex"]: o for o in t["observations"]}
    fs = sorted(f for f in obs if obs[f].get("area", 0) > min_area)
    xs = np.array([obs[f]["x"] for f in fs])
    ys = np.array([obs[f]["y"] for f in fs])
    out = []
    for i, f in enumerate(fs):
        d = z.get(f"d{f:04d}")
        if d is None:
            continue
        d = d.astype(float)
        if d.ndim == 3:
            d = d.mean(2)
        m = d > 0.45 * np.percentile(d, 99.6)
        if m.sum() < 20:
            continue
        yy, xx = np.nonzero(m)
        C = np.cov(np.vstack([xx - xx.mean(), yy - yy.mean()]))
        j0, j1 = max(0, i - 1), min(len(fs) - 1, i + 1)
        v = np.array([xs[j1] - xs[j0], ys[j1] - ys[j0]])
        v = v / (np.linalg.norm(v) + 1e-9)
        p = np.array([-v[1], v[0]])
        out.append((f, float(np.sqrt(max(p @ C @ p, 0))), float(np.sqrt(max(v @ C @ v, 0)))))
    return (
        np.array([r[0] for r in out]),
        np.array([r[1] for r in out]),
        np.array([r[2] for r in out]),
    )


def dominant_half_period(frames, signal):
    """Half-period of the extent signal, in frames, by zero-crossings of the detrended series.

    Deliberately not an FFT: there are a dozen samples and barely two cycles, and an FFT on that
    invites a confident number where there is not one. Zero-crossings give a spacing and a spread,
    and the spread is the honest error bar.
    """
    d = signal - np.polyval(np.polyfit(frames, signal, 1), frames)
    sign = np.sign(d)
    cross = [
        (frames[i] + frames[i + 1]) / 2.0
        for i in range(len(d) - 1)
        if sign[i] != 0 and sign[i + 1] != 0 and sign[i] != sign[i + 1]
    ]
    if len(cross) < 2:
        return None, None, len(cross)
    gaps = np.diff(cross)  # one gap = half of the extent signal's period
    return (
        float(np.mean(gaps) * 2.0),
        float(np.std(gaps) * 2.0 / max(np.sqrt(len(gaps)), 1)),
        len(cross),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--track2d", required=True)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--v0", required=True, help='release velocity in world units/s, "x,y,z"')
    ap.add_argument("--g", required=True, help='gravity in world units/s^2, "x,y,z"')
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--flight-seconds", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames, perp, par = perpendicular_extent(a.track2d, a.patches)
    period_frames, period_err, ncross = dominant_half_period(frames, perp)

    v0 = np.array([float(v) for v in a.v0.split(",")])
    g = np.array([float(v) for v in a.g.split(",")])
    axis = np.cross(v0, g)
    n = np.linalg.norm(axis)
    axis = (axis / n) if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    res = dict(
        mode="measuredRate",
        spinAxisWorld=[round(float(v), 6) for v in axis],
        axisProvenance=(
            "normal of the plane the throw travels in, v0 x g. A throw tumbles end over "
            "end in that plane; this is an assumption about throws, not a measurement "
            "of these pixels."
        ),
        perpendicularExtentPx=dict(
            sourceFrames=[int(f) for f in frames],
            perpendicular=[round(float(v), 2) for v in perp],
            parallel=[round(float(v), 2) for v in par],
        ),
        crossings=int(ncross),
    )

    if period_frames:
        # the projected extent of a rod is 180 deg symmetric: one extent period is half a turn
        rev_per_sec = a.fps / (2.0 * period_frames)
        err = rev_per_sec * (period_err / period_frames) if period_frames else 0.0
        res.update(
            spinRevPerSec=round(float(rev_per_sec), 3),
            spinRevPerSecSigma=round(float(err), 3),
            extentPeriodFrames=round(period_frames, 2),
            provenance=(
                f"rate from the perpendicular extent of the in-flight difference blobs over "
                f"{len(frames)} frames: {ncross} zero crossings give an extent period of "
                f"{period_frames:.1f} frames, and a rod's projected extent is 180 deg "
                f"symmetric, so one revolution is twice that. Axis assumed from the throw "
                f"plane. Phase at release is not resolved by these pixels."
            ),
        )
        if a.flight_seconds:
            res["revolutionsOverFlight"] = round(float(rev_per_sec * a.flight_seconds), 2)
    else:
        res.update(
            mode="velocityAligned",
            spinRevPerSec=0.0,
            provenance="the perpendicular extent showed fewer than two crossings; no rate "
            "is claimed and the object is left velocity-aligned.",
        )

    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "perpendicularExtentPx"}, indent=1))


if __name__ == "__main__":
    main()
