"""Lift 2D object observations to 3D world trajectories, one fit per free flight.

The object is small and its own depth is unmeasurable (a translucent bottle leaves
no Pi3X depth of its own), so nothing here trusts per-frame depth. Instead a free
flight is a 6-parameter ballistic curve p(t) = p0 + v0 t + g t^2 / 2 with |g| known
from the clip's metres-per-world-unit, fitted to the 2D observations through the
clip's own SfM cameras, optionally anchored on the throwing and catching hands
(3D joints from the per-track MultiHMR poses, mapped to world exactly as the
avatars are).

A clip may contain any number of free flights, so this takes a LIST of them and
fits each independently: every throw carries its own reprojection error, its own
hand anchors and its own thrower and catcher, rather than inheriting the first
throw's. Who threw and who caught is read off the footage (nearest tracked wrist
in the image at the first and last observation) unless overridden.

  --flights  the output of detect_object_flights.py   -> wander.object-fit/2 (N flights)
  --track2d --flight f0:f1                            -> the same schema with one flight
"""
import argparse, json, os
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch


def load_cameras(path):
    c = json.load(open(path))
    cams = c['cameras']
    idx = np.array([cm['sourceIndex'] for cm in cams], float)
    R = np.array([np.array(cm['camera_to_world'])[:3, :3] for cm in cams])
    for i in range(len(R)):
        u, _, vt = np.linalg.svd(R[i]); R[i] = u @ vt
    t = np.array([np.array(cm['camera_to_world'])[:3, 3] for cm in cams])
    K = np.array(cams[0]['source_intrinsics'])
    rots = Rotation.from_matrix(R)
    slerp = Slerp(idx, rots)
    return dict(idx=idx, R=R, t=t, K=K, slerp=slerp)


def cam_at(C, sf):
    sf = float(np.clip(sf, C['idx'][0], C['idx'][-1]))
    R = C['slerp']([sf]).as_matrix()[0]
    t = np.array([np.interp(sf, C['idx'], C['t'][:, k]) for k in range(3)])
    return R, t


CV = np.diag([1.0, -1.0, -1.0])


def project(C, sf, P):
    R, t = cam_at(C, sf)
    local = (P - t) @ R            # R^-1 (P-t) since R orthonormal -> (P-t) @ R
    cam = local @ CV               # OpenGL -> OpenCV
    z = cam[..., 2]
    uv = (cam @ C['K'].T)
    return uv[..., :2] / z[..., None], z


_POSE_CACHE = {}


def _poses(track_dir):
    if track_dir not in _POSE_CACHE:
        ps = torch.load(os.path.join(track_dir, 'source-poses.pt'), map_location='cpu', weights_only=False)
        _POSE_CACHE[track_dir] = (np.asarray(ps['sourceIndices'], float),
                                  np.array([np.asarray(p['j3d']) for p in ps['poses']]))
    return _POSE_CACHE[track_dir]


def wrist_world(track_dir, motion, C, scale, sf, joint):
    """3D joint of a tracked person in the SfM world at an arbitrary source frame."""
    idx, j3 = _poses(track_dir)
    j = j3[:, joint]
    out = []
    for f in np.atleast_1d(sf):
        jc = np.array([np.interp(f, idx, j[:, k]) for k in range(3)])
        R, t = cam_at(C, f)
        out.append((R @ CV) @ jc * scale + t)
    return np.array(out)


def joints2d(tracks_json, joint):
    """-> f(sourceFrame) giving {track: (u, v)} for one joint, interpolated between samples."""
    d = os.path.dirname(tracks_json)
    tj = json.load(open(tracks_json))
    per = {}
    for t in range(tj['trackCount']):
        m = json.load(open(os.path.join(d, f'track_{t:02d}', 'motion.json')))
        idx = np.array([fr['sourceIndex'] for fr in m['frames']], float)
        uv = np.array([np.asarray(fr['projectedBodyJoints'])[joint] for fr in m['frames']], float)
        per[t] = (idx, uv)

    def at(sf):
        return {t: np.array([np.interp(sf, idx, uv[:, k]) for k in range(2)]) for t, (idx, uv) in per.items()}
    return at


def fit_flight(frames, uv, C, mpu, g_units, drift_at, cconst, scale, tracks_dir,
               thrower, catcher, joint, hand_sigma_m, fps):
    fl = np.asarray(frames, float)
    uv = np.asarray(uv, float)
    tsec = (fl - fl[0]) / fps
    f0, f1 = int(fl[0]), int(fl[-1])

    hand_thr = wrist_world(os.path.join(tracks_dir, f'track_{thrower:02d}'), None, C,
                           scale[thrower], [f0 - 1, f0], joint)
    hand_cat = wrist_world(os.path.join(tracks_dir, f'track_{catcher:02d}'), None, C,
                           scale[catcher], [f1, f1 + 1], joint)
    hand_thr = hand_thr + np.array([[0, cconst[thrower] + drift_at(f), 0] for f in [f0 - 1, f0]])
    hand_cat = hand_cat + np.array([[0, cconst[catcher] + drift_at(f), 0] for f in [f1, f1 + 1]])
    anchor_release, anchor_catch = hand_thr[-1], hand_cat[0]

    gvec = np.array([0.0, -g_units, 0.0])
    sig_hand = hand_sigma_m / mpu
    obj_const = 0.5 * (cconst[thrower] + cconst[catcher])

    def traj(p, t):
        return p[:3] + np.outer(t, p[3:6]) + 0.5 * np.outer(t ** 2, gvec)

    def to_raw(P):
        # de-correct back to raw world for projection (observations live in the raw camera)
        return P - np.array([[0, drift_at(f) + obj_const, 0] for f in fl])

    def resid(p, use_hands):
        P = traj(p, tsec)
        Pr = to_raw(P)
        rows = [project(C, f, Pr[i])[0] - uv[i] for i, f in enumerate(fl)]
        r = np.array(rows).ravel()
        if use_hands:
            r = np.concatenate([r, (P[0] - anchor_release) / sig_hand, (P[-1] - anchor_catch) / sig_hand])
        return r

    p_init = np.concatenate([anchor_release,
                             (anchor_catch - anchor_release) / max(tsec[-1], 1e-6) - 0.5 * gvec * tsec[-1]])
    fits = {}
    for name, hands in (('imageOnly', False), ('handAnchored', True)):
        s = least_squares(resid, p_init, args=(hands,), method='lm', xtol=1e-12, ftol=1e-12)
        P = traj(s.x, tsec)
        px = np.array([project(C, f, to_raw(P)[i])[0] for i, f in enumerate(fl)])
        rp = np.linalg.norm(px - uv, axis=1)
        J = s.jac
        try:
            cov = np.linalg.inv(J.T @ J) * (s.fun @ s.fun) / max(1, len(s.fun) - 6)
            sd = np.sqrt(np.diag(cov))[:3]
        except np.linalg.LinAlgError:
            sd = np.full(3, np.nan)
        fits[name] = dict(p0=s.x[:3].tolist(), v0=s.x[3:6].tolist(),
                          reprojPxRms=float(np.sqrt((rp ** 2).mean())), reprojPxMax=float(rp.max()),
                          releaseToHandM=float(np.linalg.norm(P[0] - anchor_release) * mpu),
                          catchToHandM=float(np.linalg.norm(P[-1] - anchor_catch) * mpu),
                          p0SigmaM=(sd * mpu).tolist(),
                          speedMs=float(np.linalg.norm(s.x[3:6]) * mpu),
                          positions=P.tolist())
    best = fits['handAnchored']
    P = np.array(best['positions'])
    return dict(flight=[f0, f1], thrower=int(thrower), catcher=int(catcher), joint=int(joint),
                frames=[int(f) for f in fl], timesSec=tsec.tolist(), observations2d=uv.tolist(),
                fits=fits, anchorRelease=anchor_release.tolist(), anchorCatch=anchor_catch.tolist(),
                spanM=float(np.linalg.norm(P[-1] - P[0]) * mpu),
                apexYUnits=float(P[:, 1].max()), flightSec=float(tsec[-1]),
                constantY=float(obj_const), driftCorrected=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flights', help='detect_object_flights.py output; fits every free span in it')
    ap.add_argument('--track2d', help='track_object_2d.py output (single-flight legacy path)')
    ap.add_argument('--flight', help='firstFrame:lastFrame, with --track2d')
    ap.add_argument('--cameras', required=True)
    ap.add_argument('--people', required=True, help='people.json (scales, floor fit, drift)')
    ap.add_argument('--tracks-dir', required=True)
    ap.add_argument('--tracks', help='tracks.json, for reading 2D wrists when picking thrower/catcher')
    ap.add_argument('--thrower', type=int, default=None, help='override; default is the nearest wrist')
    ap.add_argument('--catcher', type=int, default=None)
    ap.add_argument('--joint', type=int, default=21, help='SMPL joint index of the throwing hand')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--hand-sigma-m', type=float, default=0.12)
    ap.add_argument('--min-obs', type=int, default=10,
                    help='a monocular ballistic fit over fewer observations than this does not '
                         'separate from noise: six points over 0.2 s fit a downward parabola by '
                         'accident. Spans shorter than this are reported as undetectable, not as '
                         'absent. Lower it only with frame-by-frame proof of the short toss.')
    ap.add_argument('--max-hand-metres', type=float, default=1.3,
                    help='drop a flight whose hand-anchored fit cannot reach within this of a wrist '
                         'at BOTH ends: a free span that leaves no hand and arrives at none is a '
                         'moving blob, not a throw')
    ap.add_argument('--max-reproj-px', type=float, default=8.0,
                    help='drop a flight whose hand-anchored fit does not reproject this well')
    ap.add_argument('--keep-all', action='store_true', help='report the gates but drop nothing')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    C = load_cameras(args.cameras)
    pj = json.load(open(args.people))
    mpu = pj['floorFit']['metresPerWorldUnit']
    g_units = 9.80665 / mpu
    drift = pj['floorFit']['sharedCameraDrift']
    doff = np.array(drift['offsetYUnits'], float)
    src_of_sample = np.array(pj['sourceIndices'], float)
    dsamp = src_of_sample[np.array(drift['samples'], int)]
    lvC = pj['floorFit']['levels'][[k for k in pj['floorFit']['levels'] if k.startswith('C_')][0]]
    ckey = [k for k in lvC if 'translation' in k.lower()][0]
    constC = lvC[ckey]

    by_track = {p['track']: p for p in pj['people']}
    scale = {t: by_track[t]['registrationScale'] * by_track[t].get('sizeCorrection', {}).get('multiplier', 1.0)
             for t in by_track}
    cconst = {t: constC[by_track[t]['id']] for t in by_track}

    def drift_at(sf):
        return float(np.interp(sf, dsamp, doff))

    # ---- assemble the list of flights to fit
    spans = []
    if args.flights:
        fl = json.load(open(args.flights))
        clip = fl.get('clip')
        for r in fl['flights']:
            spans.append((r['frames'], r['px']))
    else:
        tj = json.load(open(args.track2d))
        clip = tj['clip']
        obs = {o['sourceIndex']: [o['x'], o['y']] for o in tj['observations']}
        f0, f1 = [int(v) for v in args.flight.split(':')]
        fs = [f for f in sorted(obs) if f0 <= f <= f1]
        spans.append((fs, [obs[f] for f in fs]))

    j2d = joints2d(args.tracks, args.joint) if args.tracks else None

    def nearest_track(sf, p):
        if j2d is None:
            raise SystemExit('--tracks is required unless --thrower and --catcher are given')
        J = j2d(sf)
        return min(J, key=lambda t: float(np.hypot(*(J[t] - np.asarray(p)))))

    out_flights, dropped = [], []
    for frames, uv in spans:
        thr = args.thrower if args.thrower is not None else nearest_track(frames[0], uv[0])
        cat = args.catcher if args.catcher is not None else nearest_track(frames[-1], uv[-1])
        r = fit_flight(frames, uv, C, mpu, g_units, drift_at, cconst, scale, args.tracks_dir,
                       thr, cat, args.joint, args.hand_sigma_m, args.fps)
        b = r['fits']['handAnchored']
        print('f%-4d-%-4d  %2d obs  thrower %d -> catcher %d  reproj rms %.2f px max %.2f  '
              'release->hand %.3f m  catch->hand %.3f m  speed %.2f m/s  span %.3f m  flight %.2f s'
              % (r['flight'][0], r['flight'][1], len(r['frames']), thr, cat,
                 b['reprojPxRms'], b['reprojPxMax'], b['releaseToHandM'], b['catchToHandM'],
                 b['speedMs'], r['spanM'], r['flightSec']))
        why = []
        if len(r['frames']) < args.min_obs:
            why.append('%d observations (< %d): too short to separate from noise'
                       % (len(r['frames']), args.min_obs))
        if max(b['releaseToHandM'], b['catchToHandM']) > args.max_hand_metres:
            why.append('%.2f m from the nearest wrist at an end (> %.2f)'
                       % (max(b['releaseToHandM'], b['catchToHandM']), args.max_hand_metres))
        if b['reprojPxRms'] > args.max_reproj_px:
            why.append('hand-anchored reprojection %.2f px (> %.2f)' % (b['reprojPxRms'], args.max_reproj_px))
        r['rejected'] = '; '.join(why) or None
        if why and not args.keep_all:
            print('    DROPPED: ' + r['rejected'])
            dropped.append(r)
            continue
        out_flights.append(r)

    consts = sorted({f['constantY'] for f in out_flights})
    out = dict(schema='wander.object-fit/2', clip=clip,
               metresPerWorldUnit=mpu, gravityUnitsPerS2=g_units, fps=args.fps,
               driftCorrected=True,
               constantY=float(np.mean([f['constantY'] for f in out_flights])) if out_flights else 0.0,
               constantYPerFlight=consts, flights=out_flights, droppedFlights=dropped,
               gates=dict(minObs=args.min_obs, maxHandMetres=args.max_hand_metres,
                          maxReprojPx=args.max_reproj_px))
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out, '-', len(out_flights), 'flights')


if __name__ == '__main__':
    main()
