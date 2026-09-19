#!/usr/bin/env python3
"""Per-sample head track for every tracked person, for fourd.html ?possess= (share/participant/PLAN.md §1).

Writes public/worlds/<world>/<person dir>/head.json, schema `wander.head/1`, in the bakedTrack
contract package_objects.py uses: raw SfM world, the frame the person PLYs are written in, BEFORE
transform.translation, the shared camera drift and the viewer's feet corrections. The viewer applies
the person's own group transform, so the head rides the body as drawn. Adds `head` to each person in
people.json.

Position: MultiHMR joint 15 (SMPL-X head) from the run's source-poses.pt, mapped through the clip's
SfM cameras exactly as lift_object_3d.py maps a wrist: (R @ CV) @ j * scale + t. Eyes: the mean of
the SMPL-X extra joints 56/57 the same way, kept only when they sit a plausible distance ahead of
the head joint. Orientation: the joint frame worn_object_fit.py builds for its `head` anchor - up =
HEAD - NECK, right = LSH -> RSH orthogonalised, forward = up x right - so body facing plus head
pitch; a sideways glance is not in the joints and is not invented. Quaternions map camera space to
world: the local frame is x right, y up, z back, i.e. a three.js camera's.

Smoothing, as the plan asks: the samples are independent MultiHMR fits (2.7 cm rms vertical), so a
Gaussian of sigma 2 samples on position and a weighted rotation mean over +-2 samples on
orientation, never across a gap in visibleSampleRuns. Raw joints otherwise: nothing is re-fitted.

  python scripts/package_head_track.py --world elevator-4d --world hpwide-4d
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV = np.diag([1.0, -1.0, -1.0])
NECK, HEAD, LSH, RSH = 12, 15, 16, 17
EYE_L, EYE_R = 56, 57
SIGMA = 2.0                      # samples


def load_cameras(path):
    cams = json.load(open(path))['cameras']
    idx = np.array([c['sourceIndex'] for c in cams], float)
    R = np.array([np.array(c['camera_to_world'])[:3, :3] for c in cams])
    for i in range(len(R)):
        u, _, vt = np.linalg.svd(R[i]); R[i] = u @ vt
    t = np.array([np.array(c['camera_to_world'])[:3, 3] for c in cams])
    return dict(idx=idx, t=t, slerp=Slerp(idx, Rotation.from_matrix(R)))


def cam_at(C, sf):
    sf = float(np.clip(sf, C['idx'][0], C['idx'][-1]))
    return C['slerp']([sf]).as_matrix()[0], np.array([np.interp(sf, C['idx'], C['t'][:, k]) for k in range(3)])


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v


def gauss_smooth(x, runs, sigma):
    """Gaussian window along axis 0, restarted at every run so nothing bleeds across a gap."""
    out = x.copy(); r = int(np.ceil(3 * sigma)); w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    for a, b in runs:
        for i in range(a, b + 1):
            lo, hi = max(a, i - r), min(b, i + r)
            ww = w[lo - i + r:hi - i + r + 1]
            out[i] = (x[lo:hi + 1] * ww[:, None]).sum(0) / ww.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', action='append', required=True, help='public/worlds/<world>, repeatable')
    ap.add_argument('--runs', default=os.path.join(ROOT, '.context', 'run'))
    ap.add_argument('--sigma', type=float, default=SIGMA)
    a = ap.parse_args()
    for world in a.world:
        wdir = os.path.join(ROOT, 'public', 'worlds', world)
        pj_path = os.path.join(wdir, 'people.json'); pj = json.load(open(pj_path))
        clip = os.path.splitext(os.path.basename(pj['clip']))[0]
        C = load_cameras(os.path.join(wdir, pj.get('cameras', 'cameras.json')))
        for p in pj['people']:
            tdir = os.path.join(a.runs, clip, 'tracks', f"track_{p['track']:02d}")
            pt = os.path.join(tdir, 'source-poses.pt')
            if not os.path.exists(pt):
                print(f'{world}/{p["id"]}: no {pt}, skipped', file=sys.stderr); continue
            ps = torch.load(pt, map_location='cpu', weights_only=False)
            ok = [k for k, q in enumerate(ps['poses']) if q is not None and q.get('j3d') is not None]   # a sample with no fit is skipped, not invented
            pidx = np.asarray(ps['sourceIndices'], float)[ok]
            J = np.array([np.asarray(ps['poses'][k]['j3d']) for k in ok])          # (n, 127, 3) camera space
            scale = p['registrationScale'] * p.get('sizeCorrection', {}).get('multiplier', 1.0)
            sf = np.array(p['sourceIndices'], float); n = len(sf)
            # every joint we need, interpolated over source index, then mapped per sample
            def world_at(joint):
                out = np.zeros((n, 3))
                for k, f in enumerate(sf):
                    jc = np.array([np.interp(f, pidx, J[:, joint, d]) for d in range(3)])
                    R, t = cam_at(C, f); out[k] = (R @ CV) @ jc * scale + t
                return out
            head, neck, lsh, rsh = world_at(HEAD), world_at(NECK), world_at(LSH), world_at(RSH)
            eyes = 0.5 * (world_at(EYE_L) + world_at(EYE_R)) if J.shape[1] > EYE_R else None
            # frame per sample: x right, y up, z back (a camera's), from the raw joints
            quats = np.zeros((n, 4)); fwd = np.zeros((n, 3))
            for k in range(n):
                up = unit(head[k] - neck[k]); right = rsh[k] - lsh[k]; right = unit(right - (right @ up) * up)
                back = unit(np.cross(right, up)); fwd[k] = -back
                quats[k] = Rotation.from_matrix(np.stack([right, up, back], 1)).as_quat()
            # visibleSampleRuns count global samples; this person's arrays start at firstSample
            f0 = int(p.get('firstSample', 0))
            runs = [(max(0, int(r[0]) - f0), min(n - 1, int(r[1]) - f0)) for r in p.get('visibleSampleRuns') or [[f0, f0 + n - 1]]]
            runs = [(a0, b0) for a0, b0 in runs if b0 >= a0] or [(0, n - 1)]
            head_s = gauss_smooth(head, runs, a.sigma)
            eyes_s = gauss_smooth(eyes, runs, a.sigma) if eyes is not None else None
            quats_s = quats.copy(); r = int(np.ceil(3 * a.sigma))
            for q0, q1 in runs:
                for i in range(q0, q1 + 1):
                    lo, hi = max(q0, i - r), min(q1, i + r)
                    w = np.exp(-0.5 * ((np.arange(lo, hi + 1) - i) / a.sigma) ** 2)
                    quats_s[i] = Rotation.from_quat(quats[lo:hi + 1]).mean(weights=w).as_quat()
            visible = np.zeros(n, bool)
            for q0, q1 in runs: visible[q0:q1 + 1] = True
            eye_ok = eyes is not None and 0.03 * scale <= np.median(np.linalg.norm(eyes - head, axis=1)) / 1.0 <= 0.25 * scale
            out = dict(schema='wander.head/1', person=p['id'], track=p['track'], fps=pj['fps'],
                       space='raw SfM world, identical to the person PLYs before transform.translation (the objects.json bakedTrack contract)',
                       sourceFrames=[int(v) for v in sf], sampleIndex=list(range(n)),
                       positions=np.round(head_s, 5).tolist(),
                       eyes=np.round(eyes_s, 5).tolist() if eye_ok else None,
                       quaternionsXYZW=np.round(quats_s, 5).tolist(),
                       forward=np.round(fwd, 4).tolist(),
                       visible=visible.tolist(),
                       joints='MultiHMR j3d: head = SMPL-X joint 15, eyes = mean of 56/57, frame from HEAD-NECK (up) and LSH->RSH (right); forward = up x right',
                       smoothing=f'gaussian sigma {a.sigma} samples on position, weighted rotation mean over +-{r} on orientation, per visibleSampleRun',
                       scale=scale, rawHeadRmsFromSmoothedUnits=float(np.sqrt(((head - head_s) ** 2).sum(1).mean())))
            outp = os.path.join(wdir, p['directory'], 'head.json')
            json.dump(out, open(outp, 'w'))
            p['head'] = f"{p['directory']}/head.json"
            yaw = np.degrees(np.arctan2(-fwd[:, 0], -fwd[:, 2]))
            print(f'{world}/{p["id"]} (track {p["track"]}): {n} samples from {len(ok)}/{len(ps["poses"])} fits, head y {head[:, 1].min():.3f}..{head[:, 1].max():.3f} u, '
                  f'raw-vs-smoothed rms {out["rawHeadRmsFromSmoothedUnits"] * 100:.1f} cm-units, eyes {"kept" if eye_ok else "dropped"}'
                  f'{f" ({np.median(np.linalg.norm(eyes - head, axis=1)) / scale * 100:.1f} cm ahead of the head joint)" if eyes is not None else ""}, '
                  f'yaw {yaw.min():.0f}..{yaw.max():.0f} deg -> {outp}')
        json.dump(pj, open(pj_path, 'w'), indent=1)
        print(f'updated {pj_path}')


if __name__ == '__main__':
    main()
