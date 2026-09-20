#!/usr/bin/env python3
"""Rotate each clip's SfM frame into the Marble world's frame.

THE DEFECT
----------
`package_person_sequence.py` re-bases every SfM reconstruction onto camera 0 (`R = R0.T`), so the
frame the person PLYs, `cameras.json` and `objects.json` live in is the PHONE'S BODY FRAME AT
FRAME 0: its y axis is whichever way "up" pointed out of the handset when the clip started. Marble
builds its world gravity-levelled -- every world floor in this repo fits to within 0.05-1.1 deg of
+y after the viewer's flip. Nothing in the pipeline rotates one frame into the other. `fourd.html`
composes the world as Rx(PI) * S(worldscale) and the SfM content as S(scale0) + T(pos0), i.e. it
models the SfM -> Marble map as a uniform scale and a translation. The rotation between the two
frames is simply dropped, and it is the phone's own pitch/roll at frame 0: 19 deg on the gym,
9 deg on the elevator.

Everything downstream that corrects a vertical error with a translation -- `sharedCameraDrift`'s
120-float table, `place_solve.py`'s per-sample `offsetUnits`, the hand-tuned `floor`/`pos` values --
is absorbing that one angle with the wrong kind of parameter, which is why each nearly works.

MEASURING GRAVITY IN THE SfM FRAME
----------------------------------
Three independent estimators, none of which is the thing being corrected:

  walls   Vertical surfaces are vertical in all of these scenes. Local normals of the Pi3X dense
          static scene concentrate on a great circle whose pole is gravity. Needs no floor, no
          camera motion, and does not care whether the dominant low surface is a floor or a ramp.
  floor   The dominant planar surface of the same point cloud, seeded from `walls` so it cannot
          latch onto a wall. Sharper than `walls` when the floor is large and flat -- and WRONG when
          the "floor" is a staircase, which is why it is only adopted if it agrees with `walls`.
  body    Principal axis of the person's own splats, averaged over the clip. The only estimator
          available for a clip whose Pi3X run is gone, and the sanity check on the other two.

The camera track's own plane is deliberately NOT used to fit: on a clip where the operator walked a
level line it is the same number, but it cannot tell a tilted frame from a real climb (stairs2) and
it is degenerate when the track is a straight line. It is reported as a cross-check only.

THE ROTATION
------------
The minimal rotation taking g_sfm to +y: axis g x y, angle acos(g.y). Minimal because any extra
component would be a yaw about gravity, and the yaw is already right -- Marble seeds its world on
source frame 0, so the two frames share an origin and a look direction.

Opt-in only. `measure` prints; `solve` writes `framealign.json` beside the clip's manifest;
fourd.html applies it under `?rotfix=1` and ignores it otherwise.
"""

import argparse, json, sys, zlib
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bake_video_colours import read_spz  # noqa: E402
from sfm_frame import orthonormal  # noqa: E402

RNG = np.random.default_rng(11)

# clip -> (world dir under public/worlds, Pi3X run dir or None, Marble spz, preset floor, preset scale)
CLIPS = {
    "gym": ("gym-4d", ".context/run/gym/pi3x", "marble-gym-clean2.spz", -0.7097, 1.0020),
    "elevator": (
        "elevator-4d",
        ".context/run/elevator/pi3x",
        "marble-elevator-clean.spz",
        -0.6145,
        1.0800,
    ),
    "lobby": ("lobby-4d", ".context/own/lobby/pi3x", "marble-lobby-clean.spz", -0.5464, 1.5300),
    "atrium": (
        "atrium-4dpp",
        ".context/run/atrium/pi3x",
        "marble-atrium-clean.spz",
        -0.9015,
        1.0221,
    ),
    "living": (
        "living-4dpp",
        ".context/own/living/pi3x",
        "marble-living-finetuned.spz",
        -0.7257,
        0.3750,
    ),
    "stairs2": (
        "stairs2-4d",
        ".context/run/stairs2/pi3x",
        "marble-stairs2-clean.spz",
        -0.9026,
        0.9760,
    ),
    "tos31": ("tos31-4d", None, "marble-tos31-image.spz", -0.5894, 1.2834),
    "selfie": (
        "selfie-4d",
        ".context/run/selfie/pi3x",
        "marble-selfie-finetuned.spz",
        -2.5498,
        1.4560,
    ),
    "bedroom": (
        "bedroom-4d",
        ".context/own/bedroom/pi3x",
        "marble-bedroom-clean.spz",
        -1.696,
        0.673,
    ),
}

FLOOR_GATE_DEG = 15.0  # a "floor" further than this from the wall pole is a ramp, not a floor

# Which person tracks each clip has, and what the primary stands on. Used by `residuals` and by the
# one global constant `solve --write` re-fits, because the preset's own `floor`/`pos` was fitted in
# the broken frame and is not reusable once the frame turns.
TRACKS = {"elevator": ["person", "person_01"]}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(unit(a), unit(b))), 0, 1))))


def tilt_from_y(g):
    return float(np.degrees(np.arccos(np.clip(unit(g)[1], -1, 1))))


def align_rotation(g):
    """Minimal rotation taking g to +y (no yaw about gravity)."""
    g = unit(g)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(g, y)
    s = np.linalg.norm(v)
    c = float(g @ y)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0, 0, 0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return [float(x) for x in q]


# ---------------------------------------------------------------- io
def cameras(world):
    p = ROOT / "public/worlds" / world / "cameras.json"
    if not p.exists():
        p = ROOT / "public/worlds" / (world.split("-")[0] + "-4d") / "cameras.json"
    d = json.load(open(p))
    return np.array([np.array(c["camera_to_world"])[:3, 3] for c in d["cameras"]]), p


def read_ply_xyz(p: Path):
    buf = p.read_bytes()
    hi = buf.index(b"end_header\n") + 11
    head = buf[:hi].decode()
    props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[2])
    d = np.frombuffer(buf[hi : hi + n * len(props) * 4], np.float32).reshape(n, len(props))
    return np.stack(
        [d[:, props.index("x")], d[:, props.index("y")], d[:, props.index("z")]], 1
    ).astype(np.float64)


# ---------------------------------------------------------------- estimators
def est_track(C):
    """Plane through the camera centres. Reported, never adopted -- see the module docstring."""
    A = np.stack([C[:, 0], C[:, 2], np.ones(len(C))], 1)
    k, *_ = np.linalg.lstsq(A, C[:, 1], rcond=None)
    r = C[:, 1] - A @ k
    ss = ((C[:, 1] - C[:, 1].mean()) ** 2).sum()
    P = C[:, [0, 2]] - C[:, [0, 2]].mean(0)
    s = np.linalg.svd(P, compute_uv=False)
    return (
        unit([-k[0], 1.0, -k[1]]),
        float(1 - (r**2).sum() / ss if ss > 0 else np.nan),
        float(s[1] / s[0]),
    )


def est_walls(P, seed, clip="", k=24, nsamp=120000):
    """Gravity from vertical surfaces: wall normals lie on a great circle whose pole is gravity.

    Run from four seeds (the track plane, +y, and +-30 deg off it) and the modal basin is taken.
    R3 measured this estimator returning the same answer to 0.3 deg from all four seeds on 31 of 32
    runs, the exception being a bedroom seed that fell into a basin 60 deg away -- so the vote is
    what makes it reproducible, and the subsample is seeded per clip so a rerun repeats exactly.
    """
    rng = np.random.default_rng(zlib.crc32(clip.encode()) if clip else 11)
    idx = rng.choice(len(P), min(nsamp, len(P)), replace=False)
    Q = P[idx]
    _, nb = cKDTree(P).query(Q, k=k, workers=-1)
    B = P[nb]
    B = B - B.mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("mki,mkj->mij", B, B) / k)
    N = v[:, :, 0][(1 - w[:, 0] / np.maximum(w[:, 1], 1e-12)) > 0.9]

    def basin(g0):
        g = unit(g0)
        nw = 0
        for _ in range(40):
            wall = np.abs(N @ g) < np.sin(np.radians(12))
            nw = int(wall.sum())
            if nw < 500:
                return None, 0
            _, v2 = np.linalg.eigh(N[wall].T @ N[wall])
            gn = v2[:, 0]
            if gn @ g < 0:
                gn = -gn
            if ang(gn, g) < 1e-3:
                g = gn
                break
            g = gn
        return unit(g), nw

    ax = unit(
        np.cross(unit(seed), [1.0, 0.0, 0.0])
        if abs(unit(seed)[0]) < 0.9
        else np.cross(unit(seed), [0.0, 0.0, 1.0])
    )
    seeds = [
        seed,
        np.array([0.0, 1.0, 0.0]),
        align_rotation(ax * np.sin(np.radians(30)) + unit(seed) * np.cos(np.radians(30))).T
        @ unit(seed),
        unit(unit(seed) * np.cos(np.radians(30)) - ax * np.sin(np.radians(30))),
    ]
    res = [basin(x) for x in seeds]
    # a phone is held roughly upright: a "gravity" more than 45 deg off the SfM y is a degenerate
    # basin, not a tilt. atrium has only ~5.6k wall normals and lands in one, so it gets no wall
    # estimate at all rather than a confident wrong one.
    res = [r for r in res if r[0] is not None and tilt_from_y(r[0]) < 45.0]
    if not res:
        return None, 0
    votes = [sum(1 for o in res if ang(r[0], o[0]) < 1.0) for r in res]
    best = res[int(np.argmax(votes))]
    return best[0], best[1]


def est_floor(P, C, seed):
    """Dominant plane below the cameras, along `seed`. Returns (normal, d, inliers, camera height)."""
    g = unit(seed)
    t = P @ g
    hc = C @ g
    sel = (t > hc.mean() - 3.0) & (t < hc.mean() - 0.15)
    if sel.sum() < 5000:
        sel = t < hc.mean()
    h, e = np.histogram(t[sel], 400)
    n, d = g.copy(), float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    for band in (0.30, 0.15, 0.08, 0.05, 0.035):
        for _ in range(8):
            inl = np.abs(P @ n - d) < band
            if inl.sum() < 800:
                break
            Q = P[inl]
            cm = Q.mean(0)
            _, _, vt = np.linalg.svd(Q - cm, full_matrices=False)
            nn = vt[2]
            if nn @ g < 0:
                nn = -nn
            n, d = nn, float(nn @ cm)
    inl = int((np.abs(P @ n - d) < 0.035).sum())
    return unit(n), d, inl, float(np.median(C @ n - d)), float(np.ptp(C @ n - d))


def est_body(world):
    sj = ROOT / "public/worlds" / world / "person/sequence.json"
    if not sj.exists():
        return None, 0
    seq = json.load(open(sj))
    base = sj.parent
    fr = seq["frames"][:: max(1, len(seq["frames"]) // 60)]
    A = []
    for fn in fr:
        X = read_ply_xyz(base / fn)
        X = X - X.mean(0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        a = vt[0]
        A.append(-a if a[1] < 0 else a)
    A = np.array(A)
    _, v = np.linalg.eigh(A.T @ A)
    g = v[:, -1]
    return unit(-g if g[1] < 0 else g), len(A)


def world_floor(spz, seed_y):
    """Marble floor in viewer coordinates (the viewer sets world.rotation.x = PI)."""
    s = read_spz(ROOT / "public" / spz)
    X = s["xyz"].copy()
    X[:, 1] *= -1
    X[:, 2] *= -1
    R = X[s["alpha"] > 0.5].astype(np.float64)
    if (
        seed_y is None
        or not np.isfinite(seed_y)
        or not ((R[:, 1] > seed_y - 0.2) & (R[:, 1] < seed_y + 0.2)).any()
    ):
        h, e = np.histogram(R[:, 1], 400)
        seed_y = float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    c = np.array([0.0, 0.0, seed_y])
    inl = np.zeros(len(R), bool)
    for band in (0.20, 0.12, 0.055, 0.055):
        for _ in range(10):
            d = R[:, 1] - (R[:, 0] * c[0] + R[:, 2] * c[1] + c[2])
            inl = np.abs(d) < band
            if inl.sum() < 500:
                break
            A = np.stack([R[inl, 0], R[inl, 2], np.ones(int(inl.sum()))], 1)
            c, *_ = np.linalg.lstsq(A, R[inl, 1], rcond=None)
    n = unit([-c[0], 1.0, -c[1]])
    return n, c, int(inl.sum())


# ---------------------------------------------------------------- per clip
# Clips whose subject keeps a continuous walking contact with the floor. Only these can be judged
# by the feet: the gym subject is supine on a bench and the bedroom one is seated, so their "lowest
# 1 % of splats" is whichever limb happens to hang lowest, and stairs2's contact is treads, not a
# plane. R3 asked for the estimator that zeroes the feet TREND to be adopted; that rule only has
# meaning where a trend exists.
CONTACT_CLIPS = {"elevator", "lobby", "atrium", "living", "tos31", "selfie"}


def clip_context(clip):
    """World splats, person frames, the ONE floor plane every candidate is scored against, and the
    avatar-metre.

    The plane is fitted once, from the unrotated probe, so that comparing candidate rotations is not
    also comparing four different planes. It is found place_solve's way -- the feet's own height
    says which surface is the floor -- because the preset `floor` is NOT the world's floor: for the
    round-17 clips it is the person's frame-0 feet height, which on selfie sits 1.25 u under the
    room. R3 caught the earlier version seeding it at `floor / scale`, a world-unit height divided by
    a registration scale, which returned zero inliers on living and bedroom.
    """
    from place_solve import world_splats, person_frames, fit_plane

    world, pi3x, spz, floor, scale = CLIPS[clip]
    wd = ROOT / "public/worlds" / world
    tracks = [t for t in TRACKS.get(clip, ["person"]) if (wd / t / "sequence.json").exists()]
    frames = {t: list(person_frames(wd / t / "sequence.json")) for t in tracks}
    W = world_splats(ROOT / "public" / spz)
    probe, foot = [], []
    for fr in frames.values():
        for op, _ in fr[:: max(1, len(fr) // 40)]:
            ys = np.sort(op[:, 1])
            lo = float(ys[int(len(ys) * 0.01)])
            f = op[:, 1] < lo + (op[:, 1].max() - op[:, 1].min()) * 0.04
            probe.append([op[f, 0].mean() * scale, op[f, 2].mean() * scale])
            foot.append(lo * scale)
    probe = np.array(probe)
    mx, mz, footY = (
        float(np.median(probe[:, 0])),
        float(np.median(probe[:, 1])),
        float(np.median(foot)),
    )
    coef, nin, band = None, 0, None
    for b in (0.45, 0.90, 1.60):
        reg = (
            (np.abs(W[:, 0] - mx) < 1.6)
            & (np.abs(W[:, 2] - mz) < 2.0)
            & (np.abs(W[:, 1] - footY) < b)
        )
        if reg.sum() < 5000:
            continue
        hy, ey = np.histogram(W[reg][:, 1], 90)
        c, _, _, n = fit_plane(W[reg], seed=float(0.5 * (ey[hy.argmax()] + ey[hy.argmax() + 1])))
        tilt = float(np.degrees(np.arctan(np.hypot(c[0], c[1]))))
        if n >= 5000 and tilt < 3.0:
            coef, nin, band = c, n, b
            break
        coef, nin, band = c, n, b  # keep the last attempt for the diagnosis
    tilt = (
        float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        if coef is not None
        else float("nan")
    )
    plane_ok = coef is not None and nin >= 5000 and tilt < 3.0

    # the avatar-metre: place_solve's own rule, 1.70 m over the body's longest principal extent.
    # It is a prior, not a ruler, and every centimetre printed anywhere in this file is in it.
    rng = np.random.default_rng(zlib.crc32(clip.encode()))
    st = []
    for op, _ in list(frames.values())[0][::10]:
        Q = op - op.mean(0)
        v = np.linalg.svd(
            Q[rng.choice(len(Q), min(5000, len(Q)), replace=False)], full_matrices=False
        )[2][0]
        pr = op @ v
        st.append(np.percentile(pr, 99.3) - np.percentile(pr, 0.7))
    stature = float(np.median(st))
    return dict(
        W=W,
        frames=frames,
        coef=coef,
        planeInliers=int(nin),
        planeTiltDeg=tilt,
        planeOk=bool(plane_ok),
        planeBand=band,
        stature=stature,
        footY=footY,
        mpu=1.70 / (stature * scale),
        scale=scale,
        floor=floor,
        walked=(mx, mz),
    )


def contact_series(frames, R, scale, coef, pos0=(0.0, 0.0, 0.0)):
    """Feet (1st percentile of opaque splats, measured along the WORLD's up) minus the plane."""
    out = []
    for op, _ in frames:
        P = op @ R.T
        y = P[:, 1]
        ys = np.sort(y)
        low = float(ys[int(len(ys) * 0.01)])
        f = y < low + (y.max() - y.min()) * 0.04
        wx = float(P[f, 0].mean()) * scale + pos0[0]
        wz = float(P[f, 2].mean()) * scale + pos0[2]
        out.append(low * scale + pos0[1] - (coef[0] * wx + coef[1] * wz + coef[2]))
    return np.array(out)


def trend_endpoints(res):
    """First and last value of a least-squares line through the residual, after its median is
    removed. This is the quantity nothing in the pipeline is fitted to, so it is the held-out check
    on a rotation: a frame tilted by theta makes it run away linearly with distance walked."""
    r = res - np.median(res)
    k = np.arange(len(r))
    a, b = np.polyfit(k, r, 1)
    return float(b), float(a * (len(r) - 1) + b)


def analyse(clip):
    world, pi3x, spz, floor, scale = CLIPS[clip]
    C, campath = cameras(world)
    g_track, r2, aniso = est_track(C)
    g_walls = g_floor = None
    n_walls = n_floor = 0
    camh = camspan = float("nan")
    dfloor = 0.0
    if pi3x and (ROOT / pi3x / "static-scene.npy").exists():
        P = np.load(ROOT / pi3x / "static-scene.npy").astype(np.float64)
        Cs = np.array(
            [
                np.array(c["camera_to_world"])[:3, 3]
                for c in json.load(open(ROOT / pi3x / "cameras.json"))["cameras"]
            ]
        )
        g_walls, n_walls = est_walls(P, g_track, clip)
        if g_walls is not None:
            g_floor, dfloor, n_floor, camh, camspan = est_floor(P, Cs, g_walls)
    g_body, n_body = est_body(world)

    # `walls` and `floor` are two reductions of the SAME Pi3X cloud (disjoint subsets, floor seeded
    # from walls), so they are not independent of each other; `body` is independent but only means
    # anything when the subject is upright - it is 54 deg off on the gym, where he is supine. They
    # disagree by 0.2-3.4 deg per clip and the feet move about 2.6 cm per degree per metre walked,
    # so the choice is worth more than the residual it is judged by.
    # `track` is measured and printed but never adopted: a track plane cannot tell a tilted frame
    # from a man walking up a staircase, which is the whole reason stairs2 reads 14.8 deg.
    cands = {}
    if g_walls is not None:
        cands["walls"] = g_walls
    if g_floor is not None:
        cands["floor"] = g_floor
    # the body axis is only gravity when the subject is upright. It is 54 deg off on the gym, where
    # he is supine, and 11 deg on stairs2; a candidate that far from the cloud is a different
    # quantity, not a noisy estimate of the same one, so it is dropped rather than averaged in.
    body_ok = g_body is not None and (g_walls is None or ang(g_body, g_walls) < 30.0)
    if body_ok:
        cands["body"] = g_body
    ref = dict(cands)
    ref["track"] = g_track
    if g_body is not None and not body_ok:
        ref["body"] = g_body

    ctx = clip_context(clip)
    cwf, niw = ctx["coef"], ctx["planeInliers"]
    nwf = unit([-cwf[0], 1.0, -cwf[1]])
    # score every candidate on the ONE thing nothing is fitted to: the trend of the feet against
    # the world's own floor over the clip. Identity is scored too, as the null.
    trials = {}
    for name, gc in list(ref.items()) + [("none", np.array([0.0, 1.0, 0.0]))]:
        Rc = align_rotation(gc)
        ends, rmss = [], []
        for fr in ctx["frames"].values():
            r = contact_series(fr, Rc, scale, ctx["coef"])
            ends.append(trend_endpoints(r))
            rmss.append(float(np.sqrt(np.mean((r - np.median(r)) ** 2))))
        trials[name] = dict(
            tiltDeg=tilt_from_y(gc),
            trendCm=[[e[0] * ctx["mpu"] * 100, e[1] * ctx["mpu"] * 100] for e in ends],
            driftCm=float(np.mean([abs(e[1] - e[0]) for e in ends]) * ctx["mpu"] * 100),
            rmsCm=float(np.mean(rmss) * ctx["mpu"] * 100),
        )

    if clip in CONTACT_CLIPS and ctx["planeOk"] and cands:
        src = min(cands, key=lambda k: trials[k]["driftCm"])
        if trials["none"]["driftCm"] <= trials[src]["driftCm"]:
            # the feet say the frame is already straight. That contradicts the cloud, and a
            # contradiction is the answer, not something to average away.
            src, g = "none", np.array([0.0, 1.0, 0.0])
            rule = (
                f"NONE: the feet drift {trials['none']['driftCm']:.1f} cm raw, less than under any "
                f"candidate rotation. Cloud and feet disagree on this clip; do not ship it."
            )
        else:
            g = cands[src]
            rule = f"{src}: least residual feet drift ({trials[src]['driftCm']:.1f} cm vs {trials['none']['driftCm']:.1f} raw)"
    elif g_walls is not None and g_floor is not None and ang(g_floor, g_walls) < FLOOR_GATE_DEG:
        src, g, rule = (
            "floor",
            g_floor,
            "floor (seeded by walls); no walking contact, so the trend rule cannot judge",
        )
    elif g_walls is not None:
        src, g, rule = "walls", g_walls, "walls (dominant low plane rejected as a ramp)"
    elif g_body is not None:
        src, g, rule = "body", g_body, "body axis (no Pi3X static scene for this clip)"
    else:
        src, g, rule = "track", g_track, "camera track plane (last resort)"
    spread = max((ang(a, b) for a in cands.values() for b in cands.values()), default=0.0)
    # R3's sensitivity: the feet move about 2.6 cm per degree of rotation per metre walked, so the
    # spread between estimators IS the error bar on every centimetre below.

    R = align_rotation(g)
    pos0, percon = refit_constant(ctx, R, scale)
    return dict(
        pos0Units=pos0,
        perPersonUnits=percon,
        clip=clip,
        world=world,
        cameras=str(campath.relative_to(ROOT)),
        spz=spz,
        gravitySfm=list(map(float, g)),
        source=rule,
        adopted=src,
        candidates=trials,
        estimatorSpreadDeg=spread,
        metresPerWorldUnitAvatar=ctx["mpu"],
        rotationRowMajor=[[float(x) for x in r] for r in R],
        quaternionXYZW=quat_xyzw(R),
        bodyAxisUsable=bool(body_ok),
        # R3's ship gate: only clips where a walking contact exists, the world plane is found, the
        # estimators agree to a few degrees, and the rotation actually flattens the held-out trend.
        recommended=bool(
            clip in CONTACT_CLIPS
            and ctx["planeOk"]
            and src not in ("none", "track")
            and spread < 6.0
            and trials[src]["driftCm"] < 0.25 * trials["none"]["driftCm"]
        ),
        angleDeg=ang(g, nwf),
        tiltFromYDeg=tilt_from_y(g),
        estimators=dict(
            track=dict(
                g=list(map(float, g_track)), tiltDeg=tilt_from_y(g_track), r2=r2, anisotropy=aniso
            ),
            walls=None
            if g_walls is None
            else dict(g=list(map(float, g_walls)), tiltDeg=tilt_from_y(g_walls), normals=n_walls),
            floor=None
            if g_floor is None
            else dict(
                g=list(map(float, g_floor)),
                tiltDeg=tilt_from_y(g_floor),
                inliers=n_floor,
                camHeightUnits=camh,
                camHeightSpanUnits=camspan,
                vsWallsDeg=ang(g_floor, g_walls),
            ),
            body=None
            if g_body is None
            else dict(g=list(map(float, g_body)), tiltDeg=tilt_from_y(g_body), frames=n_body),
        ),
        worldFloor=dict(
            normal=list(map(float, nwf)),
            coef=[float(x) for x in cwf],
            tiltDeg=ctx["planeTiltDeg"],
            inliers=niw,
            ok=ctx["planeOk"],
            bandUnits=ctx["planeBand"],
            footYUnits=ctx["footY"],
        ),
        note=(
            "Rotation to apply to everything in the SfM frame -- cameras.json, person PLYs, "
            "objects -- BEFORE the registration scale, to put it in the Marble world frame. "
            "fourd.html applies it only under ?rotfix=1."
        ),
    )


def refit_constant(ctx, R, scale):
    """The ONE constant the preset's `floor`/`pos` already is, re-fitted in the rotated frame.

    Not a third patch: `pos0` is a single global translation and the per-person term is one number
    per body -- the level that existed before `sharedCameraDrift` and before `place_solve`. Turning
    the frame invalidates the old values, so they have to be re-read; nothing per-sample is fitted.
    """
    med = {
        t: float(np.median(contact_series(fr, R, scale, ctx["coef"])))
        for t, fr in ctx["frames"].items()
    }
    shared = float(np.median(list(med.values())))
    return [0.0, -shared, 0.0], {t: [0.0, -(v - shared), 0.0] for t, v in med.items()}


def residuals(clip, verbose=True):
    """The same contact quantity under raw / rotation / each shipped patch, scored identically.

    Every condition gets ONE constant per person and nothing else, except the shipped rows, which
    get the full per-sample table they ship with. Two numbers are reported: the RMS, which the
    patches were fitted to minimise, and the TREND across the clip, which nothing is fitted to and
    which is therefore the only one that can adjudicate between them.

    Centimetres here are avatar-metres -- 1.70 m divided by the body's longest principal extent,
    place_solve's own rule. There is no external metre in this repo. When the world's floor plane
    cannot be found the row refuses to print centimetres rather than printing world units as cm,
    which is what R3 caught the first version doing on tos31 and stairs2.
    """
    world, pi3x, spz, floor, scale = CLIPS[clip]
    fa = json.load(open(ROOT / "public/worlds" / world / "framealign.json"))
    R = np.array(fa["rotationRowMajor"])
    ctx = clip_context(clip)
    mpu, coef = ctx["mpu"], ctx["coef"]
    wd = ROOT / "public/worlds" / world
    pj, ppl = wd / "placement.json", wd / "people.json"

    def score(RR, sc, p0, extra=None, pos_xz=(0.0, 0.0)):
        segs, ends = [], []
        for t, fr in ctx["frames"].items():
            r = contact_series(fr, RR, sc, coef, [p0[0] + pos_xz[0], p0[1], p0[2] + pos_xz[1]])
            if extra is not None:
                k = min(len(r), len(extra))
                r = r[:k] + extra[:k]
            segs.append(r - np.median(r))
            ends.append(trend_endpoints(r))
        v = np.concatenate(segs)
        return dict(
            rmsCm=float(np.sqrt(np.mean(v**2)) * mpu * 100),
            driftCm=float(np.mean([abs(e[1] - e[0]) for e in ends]) * mpu * 100),
            endsCm=[[e[0] * mpu * 100, e[1] * mpu * 100] for e in ends],
        )

    # a rotation about camera 0 also swings the person sideways; the rotated condition is given back
    # the same 2-dof xz recentre the viewer's `pos0` already is, and nothing per-sample
    cen = lambda RR: np.median(
        np.concatenate(
            [[(op @ RR.T).mean(0) * scale for op, _ in fr] for fr in ctx["frames"].values()]
        ),
        0,
    )
    d = cen(np.eye(3)) - cen(R)

    out = dict(
        clip=clip,
        tiltDeg=fa["tiltFromYDeg"],
        adopted=fa["adopted"],
        metresPerWorldUnitAvatar=mpu,
        planeOk=ctx["planeOk"],
        planeTiltDeg=ctx["planeTiltDeg"],
        planeInliers=ctx["planeInliers"],
        contact=clip in CONTACT_CLIPS,
        estimatorSpreadDeg=fa["estimatorSpreadDeg"],
    )
    out["raw"] = score(np.eye(3), scale, [0, 0, 0])
    out["rotation"] = score(R, scale, [0, 0, 0], pos_xz=(float(d[0]), float(d[2])))
    if pj.exists():
        P = json.load(open(pj))
        out["shipped_placesolve"] = score(
            np.eye(3), P["registrationScale"], P["pos0"], extra=np.array(P["offsetUnits"])[:, 1]
        )
    if ppl.exists():
        cd = (
            json.load(open(ppl))
            .get("floorFit", {})
            .get("sharedCameraDrift", {})
            .get("offsetYUnits")
        )
        if cd:
            out["shipped_camdrift"] = score(np.eye(3), scale, [0, 0, 0], extra=np.array(cd))
    if verbose:
        if not ctx["planeOk"]:
            print(
                f"{clip:9s} REFUSED: the world floor under his feet fits at {ctx['planeTiltDeg']:.1f} deg "
                f"with {ctx['planeInliers']} inliers, so no centimetre here would mean anything",
                flush=True,
            )
            return out
        f = lambda k: (
            "      --       "
            if k not in out
            else f"{out[k]['rmsCm']:5.1f} / {out[k]['driftCm']:5.1f}"
        )
        flag = "" if clip in CONTACT_CLIPS else "   (no walking contact: not a metric)"
        print(
            f"{clip:9s} {out['tiltDeg']:5.1f}d {out['adopted']:5s} +-{out['estimatorSpreadDeg']:4.1f}d | "
            f"raw {f('raw')} | rot {f('rotation')} | camdrift {f('shipped_camdrift')} | "
            f"place_solve {f('shipped_placesolve')}{flag}",
            flush=True,
        )
    return out


def scale_report(clip):
    """What scale the clip supports once the frame is straight.

    The world's metre is not observable on its own: the repo fixes it by taking the person as 1.70 m,
    so metres-per-world-unit m = 1.70 / (stature_native * s) and both rulers below are functions of
    the registration scale s alone.

      camera   a handheld phone sits 1.15-1.85 m above the floor. Before the rotation this ruler was
               unusable -- the gym's "camera height" swung over 2.2 m across one clip -- and the
               scale dispute (1.002 / 0.58 / 0.40) is what chasing it looked like.
      ceiling  floor to ceiling is 2.1-3.6 m in every room here. It goes as 1/s, so it pins the scale
               from the other side and the two have to meet.

    Stature is the body's longest principal extent, which a rotation does not change, so the scale
    moves only because the camera ruler moved.
    """
    from place_solve import world_splats, person_frames, fit_plane

    world, pi3x, spz, floor, preset_scale = CLIPS[clip]
    fa = json.load(open(ROOT / "public/worlds" / world / "framealign.json"))
    R = np.array(fa["rotationRowMajor"])
    coef = np.array(fa["worldFloor"]["coef"])
    C, _ = cameras(world)
    frames = list(person_frames(ROOT / "public/worlds" / world / "person/sequence.json"))
    st = []
    for op, _ in frames[::10]:
        Q = op - op.mean(0)
        v = np.linalg.svd(
            Q[RNG.choice(len(Q), min(5000, len(Q)), replace=False)], full_matrices=False
        )[2][0]
        pr = op @ v
        st.append(np.percentile(pr, 99.3) - np.percentile(pr, 0.7))
    stature = float(np.median(st))  # native units, rotation invariant

    W = world_splats(ROOT / "public" / spz)
    # centre the ceiling probe on the ground he actually walks, not on a hard-coded box
    pr = []
    for op, _ in frames[:: max(1, len(frames) // 40)]:
        P = op @ R.T
        ys = np.sort(P[:, 1])
        lo = ys[int(len(ys) * 0.01)]
        f = P[:, 1] < lo + (P[:, 1].max() - P[:, 1].min()) * 0.04
        pr.append([P[f, 0].mean() * preset_scale, P[f, 2].mean() * preset_scale])
    pr = np.array(pr)
    mx, mz = float(np.median(pr[:, 0])), float(np.median(pr[:, 1]))
    fy = float(coef[0] * mx + coef[1] * mz + coef[2])
    reg = (np.abs(W[:, 0] - mx) < 1.5) & (np.abs(W[:, 2] - mz) < 2.0)
    yy = W[reg][:, 1]
    yy = yy[(yy > fy - 0.6) & (yy < fy + 4.0)]
    hh, ee = np.histogram(yy, 160)
    cen = 0.5 * (ee[:-1] + ee[1:])
    abv = cen > fy + 0.9
    ceil = float(cen[abv][hh[abv].argmax()]) if abv.any() and len(yy) > 2000 else float("nan")
    f2c_u = ceil - fy  # world units, independent of s

    def cam_m(s, RR):
        P = (C @ RR.T) * s
        h = np.median(P[:, 1] - (coef[0] * P[:, 0] + coef[1] * P[:, 2] + coef[2]))
        return float(h * 1.70 / (stature * s)), float(
            np.ptp(P[:, 1] - (coef[0] * P[:, 0] + coef[1] * P[:, 2] + coef[2]))
            * 1.70
            / (stature * s)
        )

    # the s at which the camera ruler reads 1.45 m, and the s at which the ceiling reads 2.55 m
    from scipy.optimize import brentq

    def s_for_cam(target):
        try:
            return brentq(lambda s: cam_m(s, R)[0] - target, 1e-3, 80.0)
        except ValueError:
            return float("nan")

    s_cam = s_for_cam(1.45)
    s_cam_hi, s_cam_lo = s_for_cam(1.15), s_for_cam(1.85)  # a taller camera needs a SMALLER scale
    s_ceil = f2c_u * 1.70 / (stature * 2.55)
    s_ceil_lo, s_ceil_hi = f2c_u * 1.70 / (stature * 2.90), f2c_u * 1.70 / (stature * 2.30)
    return dict(
        clip=clip,
        statureNative=stature,
        presetScale=preset_scale,
        camMPreset=cam_m(preset_scale, np.eye(3))[0],
        camSpanPreset=cam_m(preset_scale, np.eye(3))[1],
        camMPresetRot=cam_m(preset_scale, R)[0],
        camSpanPresetRot=cam_m(preset_scale, R)[1],
        floorToCeilUnits=f2c_u,
        scaleFromCamera=s_cam,
        scaleFromCeiling=s_ceil,
        scaleCameraRange=[s_cam_lo, s_cam_hi],
        scaleCeilingRange=[s_ceil_lo, s_ceil_hi],
        mpuAtPreset=1.70 / (stature * preset_scale),
        heightAtPreset=1.70,
    )


# ---------------------------------------------------------------- the graph stage
def solve_raw(pi3x: Path, clip: str, anchor_samples: str | None = None):
    """Gravity in the RAW Pi3X frame, from the static cloud alone -- run_clip.py's `frame_align` stage.

    No world and no feet: this runs before the package and before any registration, so nothing it
    reads has been fitted to anything, and it needs no entry in CLIPS. Adoption is `analyse`'s
    non-contact rule -- the floor when it agrees with the walls to FLOOR_GATE_DEG, else the walls,
    else the floor seeded from the phone-upright prior. The output is what
    `sfm_frame.camera0_reframe` consumes: the packagers keep camera 0 at the origin and put this
    gravity on +y, so every consumer of the packaged frame (place_solve, the anchors, fourd.html)
    sees a levelled clip without a flag. It is NOT a `?rotfix` file: the packaged data is already
    rotated, and applying it again in the viewer would double the correction.
    """
    cams_p = pi3x / "cameras.json"
    cams = json.load(open(cams_p))["cameras"]
    # A frame the solve could not reconstruct records no camera. Gravity is estimated from the
    # positions that exist; a gap contributes nothing rather than breaking the estimate.
    recorded = [c for c in cams if c]
    if not recorded:
        raise ValueError(f"{cams_p} records no cameras")
    C = np.array([np.array(c["camera_to_world"])[:3, 3] for c in recorded], float)
    scene = pi3x / "static-scene.npy"
    if not scene.exists():
        from bake_video_colours import static_cloud_from_anchors

        n = len(cams)
        samples = (
            [int(x) for x in anchor_samples.split(",")]
            if anchor_samples
            else [round(k * (n - 1) / 7) for k in range(8)]
        )
        np.save(scene, static_cloud_from_anchors(pi3x / "anchors.npz", cams_p, samples))
    P = np.load(scene).astype(np.float64)
    g_track, r2, aniso = est_track(C)
    g_walls, n_walls = est_walls(P, g_track, clip)
    seed = g_walls if g_walls is not None else np.array([0.0, 1.0, 0.0])
    g_floor, _d, n_floor, camh, camspan = est_floor(P, C, seed)
    if g_walls is not None and ang(g_floor, g_walls) < FLOOR_GATE_DEG:
        src, g = "floor", g_floor
        rule = f"floor (seeded by walls, {ang(g_floor, g_walls):.1f} deg apart)"
    elif g_walls is not None:
        src, g = "walls", g_walls
        rule = f"walls (the dominant low plane is {ang(g_floor, g_walls):.1f} deg off them: a ramp, not a floor)"
    else:
        src, g = "floor", g_floor
        rule = "floor seeded from +y (too few vertical surfaces for a wall estimate)"
    c0 = np.array(cams[0]["camera_to_world"], float)
    R0 = orthonormal(c0[:3, :3])
    g_cam = unit(R0.T @ g)  # gravity as camera 0 sees it: what the packager rotates
    R = align_rotation(g_cam)
    cands = {k: v for k, v in (("walls", g_walls), ("floor", g_floor)) if v is not None}
    spread = max((ang(a, b) for a in cands.values() for b in cands.values()), default=0.0)
    return dict(
        schema="wander.framealign/2",
        clip=clip,
        cameras=str(cams_p),
        staticScene=str(scene),
        gravityRaw=[float(x) for x in unit(g)],
        gravityCam0=[float(x) for x in g_cam],
        tiltFromYDeg=tilt_from_y(g_cam),
        source=rule,
        adopted=src,
        estimatorSpreadDeg=spread,
        rotationRowMajor=[[float(x) for x in r] for r in R],
        quaternionXYZW=quat_xyzw(R),
        cameraHeightUnits=camh,
        cameraHeightSpanUnits=camspan,
        floorInliers=int(n_floor),
        estimators=dict(
            track=dict(
                g=[float(x) for x in g_track], tiltDeg=tilt_from_y(g_track), r2=r2, anisotropy=aniso
            ),
            walls=None
            if g_walls is None
            else dict(
                g=[float(x) for x in g_walls], tiltDeg=tilt_from_y(g_walls), normals=int(n_walls)
            ),
            floor=dict(
                g=[float(x) for x in g_floor],
                tiltDeg=tilt_from_y(g_floor),
                inliers=int(n_floor),
                vsWallsDeg=None if g_walls is None else ang(g_floor, g_walls),
            ),
        ),
        note="Read by scripts/sfm_frame.camera0_reframe: the packagers, bake_video_colours and "
        "finetune_export keep camera 0 at the origin and put gravityRaw on +y. "
        "tiltFromYDeg is the phone's pitch/roll at frame 0. Not a fourd.html ?rotfix file.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "solve", "residuals", "scale", "graph"])
    ap.add_argument("--clip", default="all")
    ap.add_argument(
        "--write",
        action="store_true",
        help="solve: write framealign.json beside the manifest; graph: beside the Pi3X cameras.json",
    )
    ap.add_argument(
        "--pi3x", type=Path, help="graph: the run's Pi3X directory (cameras.json, anchors.npz)"
    )
    ap.add_argument(
        "--anchor-samples",
        default=None,
        help="graph: anchor sample indices if static-scene.npy has to be built",
    )
    a = ap.parse_args()

    if a.cmd == "graph":
        if not a.pi3x or a.clip == "all":
            sys.exit("graph needs --pi3x DIR and --clip NAME")
        r = solve_raw(a.pi3x, a.clip, a.anchor_samples)
        e = r["estimators"]
        cel = lambda k: "  --  " if e.get(k) is None else f"{e[k]['tiltDeg']:5.1f}"
        print(
            f"{a.clip}: camera 0 is {r['tiltFromYDeg']:.1f} deg off gravity, adopted {r['adopted']} "
            f"({r['source']}); walls {cel('walls')} floor {cel('floor')} track {cel('track')} deg, "
            f"spread {r['estimatorSpreadDeg']:.1f} deg; camera {r['cameraHeightUnits']:.3f} native units "
            f"over the floor ({r['floorInliers']} inliers)",
            flush=True,
        )
        if a.write:
            out = a.pi3x / "framealign.json"
            out.write_text(json.dumps(r, indent=1) + "\n")
            print(f"wrote {out}", flush=True)
        return

    clips = list(CLIPS) if a.clip == "all" else a.clip.split(",")

    if a.cmd == "scale":
        print(
            f"{'clip':9s} {'preset s':>8s} | {'camera height, m, at the preset scale':^40s} | "
            f"{'s: camera 1.15-1.85 m':>21s} {'s: ceiling 2.3-2.9 m':>20s} | {'defensible s':>13s}"
        )
        for clip in clips:
            r = scale_report(clip)
            cl, ch = r["scaleCameraRange"]
            el, eh = r["scaleCeilingRange"]
            lo, hi = max(cl, el), min(ch, eh)
            band = "no overlap" if not (lo <= hi) else f"{lo:.2f}-{hi:.2f}"
            if not np.isfinite(el):
                band = f"{cl:.2f}-{ch:.2f} (cam only)"
            print(
                f"{r['clip']:9s} {r['presetScale']:8.4f} | raw {r['camMPreset']:5.2f} span {r['camSpanPreset']:5.2f}"
                f"  ->  rot {r['camMPresetRot']:5.2f} span {r['camSpanPresetRot']:4.2f} | "
                f"{cl:9.2f}-{ch:<10.2f} {el:8.2f}-{eh:<10.2f} | {band:>13s}",
                flush=True,
            )
        return

    if a.cmd == "residuals":
        print(
            "contact residual after ONE constant per person, RMS / trend across the clip, cm in the",
            flush=True,
        )
        print(
            "avatar-metre (1.70 m over the stature). The TREND is the held-out number: nothing fits it.",
            flush=True,
        )
        for clip in clips:
            residuals(clip)
        return

    print(
        f"{'clip':9s} | {'tilt deg + feet drift over the clip, cm (avatar-metre)':^62s} | "
        f"{'':^22s} | adopted",
        flush=True,
    )
    rows = []
    for clip in clips:
        r = analyse(clip)
        rows.append(r)
        c = r["candidates"]
        cel = lambda k: (
            "    --      " if k not in c else f"{c[k]['tiltDeg']:5.1f} {c[k]['driftCm']:5.1f}"
        )
        print(
            f"{clip:9s} | walls {cel('walls')} | floor {cel('floor')} | body {cel('body')} | "
            f"track {cel('track')} | raw {c['none']['driftCm']:5.1f} | spread {r['estimatorSpreadDeg']:4.1f}d | "
            f"ADOPT {r['adopted']:5s} {r['tiltFromYDeg']:5.1f}d | plane {r['worldFloor']['tiltDeg']:.2f}d "
            f"{r['worldFloor']['inliers']:7d} inl {'OK ' if r['worldFloor']['ok'] else 'BAD'}",
            flush=True,
        )
        if a.cmd == "solve" and a.write:
            out = ROOT / "public/worlds" / r["world"] / "framealign.json"
            out.write_text(json.dumps(dict(schema="wander.framealign/1", **r), indent=1) + "\n")
            print(f"           wrote {out.relative_to(ROOT)}", flush=True)
    if a.cmd == "solve" and not a.write:
        print(json.dumps(rows, indent=1))


# ---------------------------------------------------------------- residual comparison
# Appended: `uv run --locked --group inference python scripts/frame_align.py residuals` scores the SAME contact quantity under three
# conditions, so the rotation can be compared like for like against the patches that were built to
# stand in for it. Every condition is allowed one constant per person and nothing else, except
# "shipped", which gets the full per-sample table it ships with.


if __name__ == "__main__":
    main()
