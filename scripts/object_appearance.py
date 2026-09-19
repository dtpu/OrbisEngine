#!/usr/bin/env python3
"""Turn whatever produced an object's geometry into the appearance `wander.objects/2` wants.

Stage 4 of docs/objects.md. It is deliberately indifferent to where the geometry came from --
a multi-view fit, an image-to-3D model, a scan -- because appearance is a separate axis from
motion and should be solved once for all four motion cases.

What it does, in order:

1. **Drop the invisible.** Generators emit a long tail of near-transparent splats that cost
   bandwidth and contribute nothing at 30 px on screen.
1b. **Keep one object.** An image-to-3D model given a picture with a cast shadow, a reflection or a
   second copy of the thing will happily build all of them. `--largest-cluster` keeps the biggest
   connected blob on a coarse voxel grid and drops the rest, which is cheaper and more reliable
   than trying to get the input picture perfect.
2. **Find the object's own frame.** PCA on the surviving centres: `+y` becomes the long axis,
   `+x` the second. `wander.objects/2` says a model's local +y is its long axis, and a viewer
   that velocity-aligns or spins an object is rotating about axes that only mean anything if
   that holds.
3. **Orient the long axis.** For a bottle the narrow end is the neck. `--narrow-end` says which
   way along +y that end should point; the cross-sectional area profile decides which end is
   narrower, and the model is flipped if needed.
4. **Normalise.** Centre on the centroid and scale so the long axis spans exactly 1, so
   `package_objects.py` can size it from `sizeMetres` without knowing anything about the model.
5. **Decimate to a budget.** An object 6 cm long and 33 px on screen does not need 359 000
   Gaussians. Splats are kept by importance (opacity x footprint), with a random floor so a
   uniform surface does not lose whole regions to a deterministic top-k.

Gaussian rotations and scales are transformed properly, not just the centres: `rot_*` is composed
with the frame rotation and `scale_*` is shifted in log space by the uniform scale.

  object_appearance.py --in object.ply --out world/objects/bottle/object.ply --budget 6000
"""
import argparse, json

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=int, default=6000, help='Gaussian count after decimation; 0 keeps all')
    ap.add_argument('--min-opacity', type=float, default=0.05)
    ap.add_argument('--largest-cluster', action='store_true',
                    help='keep only the largest spatially connected blob (drops a generated shadow '
                         'twin or a duplicate object)')
    ap.add_argument('--cluster-voxels', type=int, default=48,
                    help='grid resolution on the long axis for --largest-cluster')
    ap.add_argument('--narrow-end', default='+y', choices=['+y', '-y', 'none'],
                    help='which way the narrow end (a bottle neck) points after alignment')
    ap.add_argument('--random-fraction', type=float, default=0.35,
                    help='share of the budget drawn at random rather than by importance')
    ap.add_argument('--report', default='')
    a = ap.parse_args()

    ply = PlyData.read(a.src)
    v = ply['vertex'].data
    names = list(v.dtype.names)
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-v['opacity'].astype(np.float64)))
    keep = op >= a.min_opacity
    before = len(xyz)

    idx = np.nonzero(keep)[0]
    xyz = xyz[idx]
    op = op[idx]

    if a.largest_cluster and len(xyz):
        from scipy.ndimage import label as cc3
        lo, hi = xyz.min(0), xyz.max(0)
        step = float((hi - lo).max()) / max(a.cluster_voxels, 4)
        g = np.floor((xyz - lo) / step).astype(int)
        dims = g.max(0) + 2
        occ = np.zeros(dims, bool)
        occ[g[:, 0], g[:, 1], g[:, 2]] = True
        lab, n = cc3(occ, structure=np.ones((3, 3, 3), bool))
        if n > 1:
            per = lab[g[:, 0], g[:, 1], g[:, 2]]
            counts = np.bincount(per, minlength=n + 1); counts[0] = 0
            best = int(counts.argmax())
            sel = per == best
            print(f'largest-cluster: {n} blobs, keeping {int(sel.sum())} of {len(xyz)} splats')
            idx = idx[sel]; xyz = xyz[sel]; op = op[sel]

    # --- object frame: PCA, long axis to +y
    c = xyz.mean(0)
    p = xyz - c
    _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    long_ax, mid_ax = vt[0], vt[1]
    third = np.cross(long_ax, mid_ax)
    R = np.stack([mid_ax, long_ax, third])          # world -> object: rows are the new x, y, z
    if np.linalg.det(R) < 0:
        R[2] = -R[2]
    q = p @ R.T

    if a.narrow_end != 'none':
        # which end is narrower: mean radial spread of the top and bottom deciles along +y
        hi = q[:, 1] > np.percentile(q[:, 1], 90)
        lo = q[:, 1] < np.percentile(q[:, 1], 10)
        r_hi = float(np.hypot(q[hi, 0], q[hi, 2]).mean())
        r_lo = float(np.hypot(q[lo, 0], q[lo, 2]).mean())
        want_hi_narrow = a.narrow_end == '+y'
        if (r_hi < r_lo) != want_hi_narrow:
            flip = np.diag([1.0, -1.0, -1.0])
            R = flip @ R
            q = q @ flip.T
            r_hi, r_lo = r_lo, r_hi
    else:
        r_hi = r_lo = float('nan')

    span = float(q[:, 1].max() - q[:, 1].min()) or 1.0
    q = q / span

    # --- decimate
    scale_cols = [n for n in names if n.startswith('scale_')]
    sc = np.exp(np.column_stack([v[n] for n in scale_cols]).astype(np.float64))[idx].mean(1) / span
    if a.budget and len(q) > a.budget:
        imp = op * sc ** 2
        n_rand = int(a.budget * a.random_fraction)
        top = np.argsort(-imp)[:a.budget - n_rand]
        rest = np.setdiff1d(np.arange(len(q)), top, assume_unique=False)
        rng = np.random.default_rng(0)
        pick = np.concatenate([top, rng.choice(rest, size=min(n_rand, len(rest)), replace=False)])
    else:
        pick = np.arange(len(q))

    out = np.empty(len(pick), dtype=v.dtype)
    for n in names:
        out[n] = v[n][idx][pick]
    out['x'], out['y'], out['z'] = q[pick, 0], q[pick, 1], q[pick, 2]
    for n in ('nx', 'ny', 'nz'):
        if n in names:
            out[n] = 0.0
    for n in scale_cols:                             # log-space shift by the uniform scale
        out[n] = (v[n][idx][pick].astype(np.float64) - np.log(span)).astype(out[n].dtype)
    rot_cols = [n for n in names if n.startswith('rot_')]
    if len(rot_cols) == 4:                           # PLY order is w, x, y, z
        wxyz = np.column_stack([v[n][idx][pick] for n in rot_cols]).astype(np.float64)
        nrm = np.linalg.norm(wxyz, axis=1, keepdims=True)
        wxyz = wxyz / np.where(nrm > 0, nrm, 1.0)
        rr = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
        rq = (Rotation.from_matrix(R) * rr).as_quat()[:, [3, 0, 1, 2]]
        for k, n in enumerate(rot_cols):
            out[n] = rq[:, k].astype(out[n].dtype)

    PlyData([PlyElement.describe(out, 'vertex')], text=False).write(a.out)

    rgb = np.clip(np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).astype(np.float64)[idx][pick], -2, 2)
    srgb = np.clip(0.5 + 0.2820948 * rgb, 0, 1)      # SH band 0 -> linear-ish colour
    rep = dict(source=a.src, out=a.out,
               gaussiansIn=int(before), afterOpacityCut=int(len(idx)), written=int(len(pick)),
               longAxisSpanBefore=round(span, 6),
               radiusPlusY=round(r_hi, 5), radiusMinusY=round(r_lo, 5),
               narrowEnd=a.narrow_end,
               extentAfter=[round(float(q[pick, k].max() - q[pick, k].min()), 4) for k in range(3)],
               meanColourSRGB=[round(float(x), 4) for x in srgb.mean(0)],
               frame='object-local; origin at the centroid; +y is the long axis; long axis spans 1')
    if a.report:
        json.dump(rep, open(a.report, 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
