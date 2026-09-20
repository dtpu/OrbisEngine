#!/usr/bin/env python3
"""Per-frame placement solve: put the person where the world's geometry says at every sample.

Every clip in this repo places its person with ONE similarity -- `scale` and `floor` in the fourd.html
preset -- fitted once (a geometric mean of a per-frame depth ratio) and then patched by hand per clip.
That single number is wrong at every moment except the average one, and the five known-bad cases are
all the same defect:

  gym       his pelvis height above the floor wanders over a 20 cm range while he lies still
  elevator  both men rise 50 cm together over 10 s (the Pi3X track sinking), plus +-6.4 cm between them
  lobby     the preset set `height` where it meant `scale`
  living    the person was 1.6x too large
  atrium    the per-frame depth ratio runs 1.115 .. 0.843 at the shipped scale

The shared cause is that the person is rigidly attached to the SfM camera track, and the SfM track
drifts relative to the Marble world. `scale`/`floor` can only absorb the constant part of that drift.
This solves the rest per sample.

WHAT IS SOLVED, AND WHY IT IS ONE SCALAR PER SAMPLE
---------------------------------------------------
The correction is `delta_f = d_f * n`, a translation along the contact surface's own normal `n`
(the world's floor plane, or the gym bench's pad plane), one scalar `d_f` per sample, shared by every
person in the clip, plus one constant `k_p` per person.

That dof set is not a simplification, it is what the evidence constrains (round 22's rule 3: fit only
the dofs the body constrains, or the fit absorbs registration drift into a tilt and scores better
while being wrong). Per sample the available evidence is:

  * contact       the person's lowest opaque splats against the world's own fitted floor plane, or --
                  on the gym -- his back against the bench pad. One number, along the surface normal.
  * reprojection  the avatar already lands on the filmed person (round 20: 1.00/0.92/1.00 in frame).
                  It does not pull the solve anywhere; it only CHARGES for moving. To first order a
                  world displacement `D` costs  |f/z * D_perp|  px from the silhouette edge and
                  |s_px/(2z) * D_par| px from the change in apparent size, so it is a quadratic ridge
                  with known anisotropic weights, not an independent measurement. It is measured and
                  reported in px, and it is why nothing lateral is solved.
  * depth ratio   dense, per frame, but it measures the WORLD against the depth map, not the person,
                  and round 20 measured Pearson r = -0.10 between it and the gym's actual contact
                  error. It is reported per clip as a diagnostic and given zero weight.

With tracked masks, the final fit adds measured silhouette height and centre for EACH person.
Ray depth, a vertical correction and one fixed body-size adjustment are jointly fitted against
those rows and contact. This removes the old equal-height assumption without changing articulation.
The shared solve remains the fallback when masks are unavailable.

WHY IT MUST BE REGULARISED
--------------------------
With one person there is one measurement and one free parameter per sample, so an unregularised
per-sample solve drives the residual to exactly zero and reports a meaningless victory while jittering
the body at the noise of a 1st-percentile splat height (2-3 cm). The objective is therefore

    min_d  sum_f,p  (m_fp - d_f - k_p)^2  +  lam * sum_f (d_f+1 - 2 d_f + d_f-1)^2

i.e. a cubic smoothing spline through the per-sample contact residual. The second difference is the
right penalty because a constant and a constant VELOCITY both cost nothing -- a camera track that
sinks linearly is exactly what the elevator turned out to have, and the prior should not fight it --
while curvature and jitter cost. `lam` is chosen by generalised cross-validation, i.e. by how well the
smoother predicts a held-out sample, which is precisely the question "how much of this is structure
and how much is foot-detection noise". A floor on lam keeps the residual jerk under --max-jerk-cm.

Outputs `placement.json` beside the person manifest. fourd.html reads it only under `?place=1`.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bake_video_colours import read_spz  # noqa: E402


# ---------------------------------------------------------------- io
def read_ply(p: Path):
    buf = p.read_bytes()
    hi = buf.index(b"end_header\n") + 11
    head = buf[:hi].decode()
    props = [l.split()[2] for l in head.splitlines() if l.startswith("property")]
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[2])
    d = np.frombuffer(buf[hi : hi + n * len(props) * 4], np.float32).reshape(n, len(props))
    return {k: d[:, i] for i, k in enumerate(props)}


def world_splats(spz_path: Path, min_alpha=0.5):
    """Marble spz -> fourd.html world coordinates (the viewer sets world.rotation.x = PI)."""
    s = read_spz(spz_path)
    W = s["xyz"].copy()
    W[:, 1] *= -1
    W[:, 2] *= -1
    return W[s["alpha"] > min_alpha]


def person_frames(seq_json: Path):
    """Every frame this person HAS, in their own order -- the list need not start at sample 0."""
    seq_json = Path(seq_json)
    seq = json.load(open(seq_json))
    base = seq_json.parent
    for fn in seq["frames"]:
        if not (base / fn).is_file():
            raise FileNotFoundError(
                f"{seq_json} lists {fn}, which is not on disk. The package stage left an incomplete "
                f"person; do not solve placement against a partial track."
            )
        P = read_ply(base / fn)
        m = 1.0 / (1.0 + np.exp(-P["opacity"])) > 0.502
        yield np.stack([P["x"][m], P["y"][m], P["z"][m]], 1), np.stack([P["x"], P["y"], P["z"]], 1)


# ---------------------------------------------------------------- surfaces
def fit_plane(R, y0_pct=12, band=0.055, iters=12, seed=None):
    """y = c0 x + c1 z + c2 through the dominant low surface (share/elevator-status.md section 3).

    `seed` is the height to start from. A percentile seed is what share/elevator-status.md caught
    reading 15-25 cm low, because Marble leaves fuzz under a polished floor and a percentile walks
    straight into it; on living it is 50 cm low, onto a layer below the flat entirely. The caller
    passes the MODE of the splat heights in a band around the feet instead, which is the surface.
    """
    coef = np.array([0.0, 0.0, np.percentile(R[:, 1], y0_pct) if seed is None else seed])
    inl = np.ones(len(R), bool)
    for _ in range(iters):
        d = R[:, 1] - (R[:, 0] * coef[0] + R[:, 2] * coef[1] + coef[2])
        inl = np.abs(d) < band
        if inl.sum() < 500:
            break
        A = np.stack([R[inl, 0], R[inl, 2], np.ones(int(inl.sum()))], 1)
        coef, *_ = np.linalg.lstsq(A, R[inl, 1], rcond=None)
    n = np.array([-coef[0], 1.0, -coef[1]])
    L = np.linalg.norm(n)
    return coef, n / L, coef[2] / L, int(inl.sum())  # plane: p.n = d


def bench_pad_plane(bench_spz: Path, base_spz: Path):
    """The gym bench's pad top, taken from the splats bench3d_gym.py added to the shipped world.

    The bench world is the clean world with 27,635 synthesised splats appended, so the tail of the
    file IS the bench and no re-fit of his back is needed. The pad is the bench's largest flat face;
    it is found by RANSAC over the added splats and its normal is checked against vertical.
    """
    A = read_spz(bench_spz)
    B = read_spz(base_spz)
    assert A["n"] > B["n"], (A["n"], B["n"])
    add = A["xyz"][B["n"] :].copy()
    add[:, 1] *= -1
    add[:, 2] *= -1
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        i = rng.choice(len(add), 3, replace=False)
        p0, p1, p2 = add[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        L = np.linalg.norm(nrm)
        if L < 1e-6:
            continue
        nrm = nrm / L
        if nrm[1] < 0:
            nrm = -nrm
        if nrm[1] < 0.7:  # the pad faces up; frame tubes and sides do not
            continue
        d = add @ nrm - p0 @ nrm
        k = int((np.abs(d) < 0.01).sum())
        if best is None or k > best[0]:
            best = (k, nrm, float(p0 @ nrm))
    k, nrm, d = best
    # the slab's underside is an equally large parallel plane; take the topmost dense layer instead
    proj = add @ nrm
    edges = np.arange(proj.min(), proj.max() + 0.006, 0.006)
    h, _ = np.histogram(proj, edges)
    cand = np.where(h >= 0.5 * h.max())[0]
    d = float(0.5 * (edges[cand[-1]] + edges[cand[-1] + 1]))
    sel = np.abs(add @ nrm - d) < 0.01
    A_ = np.stack([add[sel, 0], add[sel, 2], np.ones(int(sel.sum()))], 1)
    c, *_ = np.linalg.lstsq(A_, add[sel, 1], rcond=None)  # refit y = c0 x + c1 z + c2
    n = np.array([-c[0], 1.0, -c[1]])
    L = np.linalg.norm(n)
    n, d = n / L, c[2] / L
    pad = add[np.abs(add @ n - d) < 0.02]
    # in-plane frame: t1 up-slope (steepest ascent in the plane), t2 lateral
    up = np.array([0.0, 1.0, 0.0])
    t1 = up - (up @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return dict(
        n=n,
        d=float(d),
        t1=t1,
        t2=t2,
        pad=pad,
        nAdded=int(len(add)),
        nPad=int(len(pad)),
        inclineDeg=float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))),
    )


# ---------------------------------------------------------------- per-sample contact measurement
def measure_floor(frames, scale, pos0, plane_coef, foot_band=0.04):
    """Feet (1st percentile of opaque splats, fourd.html's own measureFeet rule) minus the plane."""
    rows = []
    for k, (op, _all) in enumerate(frames):
        y = op[:, 1]
        ys = np.sort(y)
        low = float(ys[int(len(ys) * 0.01)])
        f = y < low + (y.max() - y.min()) * foot_band
        cx, cz = float(op[f, 0].mean()), float(op[f, 2].mean())
        wx, wy, wz = cx * scale + pos0[0], low * scale + pos0[1], cz * scale + pos0[2]
        planeY = plane_coef[0] * wx + plane_coef[1] * wz + plane_coef[2]
        rows.append(dict(k=k, x=wx, y=wy, z=wz, res=wy - planeY, body=float(y.max() - y.min())))
    return rows


def measure_pad(frames, scale, pos0, pad, lat_cm=14.0, mpu=1.0, pct=6.0):
    """His back surface minus the pad plane, PERPENDICULAR to the pad (round 22's honest quantity).

    Sliced along the pad's up-slope axis in 6 cm bands and reduced by the median of the per-slice 6th
    percentile, so a raised arm or a foot on the floor cannot carry the frame.
    """
    n, d, t1, t2 = pad["n"], pad["d"], pad["t1"], pad["t2"]
    padA = pad["pad"] @ t1
    lo_pad, hi_pad = np.percentile(padA, 2), np.percentile(padA, 98)
    lat = lat_cm / 100.0 / mpu
    step = 0.06 / mpu
    rows = []
    for k, (op, _all) in enumerate(frames):
        P = op * scale + np.asarray(pos0)
        s = P @ n - d  # + floats above the pad
        al = P @ t1
        lt = P @ t2
        lt0 = (
            float(np.median(lt[np.abs(s) < 0.6 / mpu]))
            if np.isfinite(s).any()
            else float(np.median(lt))
        )
        keep = (
            (np.abs(lt - lt0) < lat)
            & (al > lo_pad)
            & (al < hi_pad)
            & (s > -0.8 / mpu)
            & (s < 1.2 / mpu)
        )
        per = []
        for a0 in np.arange(lo_pad, hi_pad - step, step):
            m = keep & (al >= a0) & (al < a0 + step)
            if m.sum() < 45:
                continue
            per.append(np.percentile(s[m], pct))
        rows.append(
            dict(k=k, res=float(np.median(per)) if len(per) >= 6 else float("nan"), n=len(per))
        )
    return rows


# ---------------------------------------------------------------- the solve
def build_system(M, A, B, Wa, Wb, sigma_c, lam):
    """Rows of the weighted least-squares system.

    Unknowns: a_f (slide along the source view ray), b_f (the orthogonal direction in the ray-normal
    plane), f = 0..nF-1, then the per-person constants k_p along the contact normal (nP-1 free, the
    last one is minus their sum so the shared part stays identifiable).

    Data rows   (m_fp + alpha_f a_f + beta_f b_f + k_p) / sigma_c
    Prior rows  Wa_f a_f  and  Wb_f b_f          -- the reprojection cost, in units of the pixel
                                                    tolerance, so a displacement is charged what it
                                                    actually costs the source silhouette
    Smooth rows sqrt(lam) * second difference of a and of b
    """
    nP, nF = M.shape
    nk = nP - 1
    N = 2 * nF + nk
    rows, rhs = [], []
    obs = np.isfinite(M)
    for p_ in range(nP):
        for f in range(nF):
            if not obs[p_, f]:
                continue
            r = np.zeros(N)
            r[f] = A[f] / sigma_c
            r[nF + f] = B[f] / sigma_c
            if p_ < nk:
                r[2 * nF + p_] = 1.0 / sigma_c
            elif nk:
                r[2 * nF :] = -1.0 / sigma_c
            rows.append(r)
            rhs.append(-M[p_, f] / sigma_c)
    nData = len(rows)
    for f in range(nF):
        r = np.zeros(N)
        r[f] = Wa[f]
        rows.append(r)
        rhs.append(0.0)
        r = np.zeros(N)
        r[nF + f] = Wb[f]
        rows.append(r)
        rhs.append(0.0)
    for i in range(nF - 2):
        for off in (0, nF):
            r = np.zeros(N)
            r[off + i], r[off + i + 1], r[off + i + 2] = (
                np.sqrt(lam),
                -2 * np.sqrt(lam),
                np.sqrt(lam),
            )
            rows.append(r)
            rhs.append(0.0)
    return np.array(rows), np.array(rhs), nData, nF, nk


def solve_at(M, A, B, Wa, Wb, sigma_c, lam):
    X, y, nData, nF, nk = build_system(M, A, B, Wa, Wb, sigma_c, lam)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = sol[:nF], sol[nF : 2 * nF]
    k = np.concatenate([sol[2 * nF :], [-sol[2 * nF :].sum()]]) if nk else np.zeros(1)
    # GCV: effective dof = trace of the hat matrix restricted to the data rows
    XtX = X.T @ X
    try:
        Hd = X[:nData] @ np.linalg.solve(XtX, X[:nData].T)
        edf = float(np.trace(Hd))
    except np.linalg.LinAlgError:
        return a, b, k, np.inf, np.nan
    fit = a[None, :] * A[None, :] + b[None, :] * B[None, :] + k[:, None]
    obs = np.isfinite(M)
    r = (M + fit)[obs]
    den = 1.0 - edf / nData
    score = float((r**2).mean() / den**2) if den > 1e-6 else np.inf
    return a, b, k, score, edf


def choose_lambda(M, A, B, Wa, Wb, sigma_c, max_jerk, grid=None):
    """lam by generalised cross-validation, then raised until the correction stops jittering."""
    grid = grid if grid is not None else np.logspace(-2, 7, 37)
    best = None
    for lam in grid:
        a, b, k, sc, edf = solve_at(M, A, B, Wa, Wb, sigma_c, lam)
        if not np.isfinite(sc):
            continue
        if best is None or sc < best[0]:
            best = (sc, lam, a, b, k, edf)
    sc, lam, a, b, k, edf = best
    for _ in range(24):
        j = max(np.abs(np.diff(a, 2)).max(), np.abs(np.diff(b, 2)).max()) if len(a) > 2 else 0.0
        if j <= max_jerk:
            break
        lam *= 1.8
        a, b, k, sc, edf = solve_at(M, A, B, Wa, Wb, sigma_c, lam)
    return lam, a, b, k, edf


# ---------------------------------------------------------------- reprojection cost
def reproj_cost(cameras_json, sample_src, P_world, D_world, scale):
    """Pixels the correction moves the avatar off its source-frame silhouette, per sample.

    D_world is the correction in world units; the person sits at P_world. Converted into the source
    camera at that sample: perpendicular displacement costs f/z px, along-ray displacement costs
    about (silhouette_px / 2z) px through the change in apparent size. Both in source pixels.
    """
    cams = json.load(open(cameras_json))["cameras"]
    by_src = {c["sourceIndex"]: c for c in cams}
    out = []
    for i, si in enumerate(sample_src):
        c = by_src.get(si)
        if c is None:
            out.append(float("nan"))
            continue
        c2w = np.array(c["camera_to_world"])
        K = np.array(c["source_intrinsics"])
        f = float((K[0, 0] + K[1, 1]) / 2)
        # camera position / forward in the SfM frame; the viewer's world is scale * SfM
        C = c2w[:3, 3] * scale
        ray = P_world[i] - C
        z = float(np.linalg.norm(ray))
        if z < 1e-6:
            out.append(float("nan"))
            continue
        u = ray / z
        D = D_world[i]
        par = float(D @ u)
        perp = float(np.linalg.norm(D - par * u))
        out.append(f * perp / z)  # the dominant term; the size term is ~4x smaller per metre
    return np.array(out)


# ---------------------------------------------------------------- clips
# Per clip: the world, the person, and the registration scale the solve starts from. `scale` is the
# SfM-to-Marble scale, a property of the two COORDINATE FRAMES, and it must never be re-derived from a
# swapped avatar's body height - doing that is how atrium came to be 25 % oversized. `sizeScale`, when
# present, is the correction the size check demands: the shipped value failed the room's own metre.
CLIPS = {
    # gym: the shipped 1.002 makes the room 11.5 x 12 m with a 1.5 m ceiling and 35 cm floor tiles, and
    # the man 2.94 m. 0.580 makes it 20 x 21 m, 2.65 m ceiling, 60 cm tiles, camera 1.30 m, man 1.70 m.
    # Contact is his FEET, which are planted on the floor beside the bench: the bench itself is a
    # separate defect (round 20's occlusion shadow) and marble-gym-clean2-bench.spz was built at the
    # old scale, so it is invalidated by this and has to be rebuilt before gym ships again.
    "gym": dict(
        world="public/marble-gym-clean2.spz",
        base="public/marble-gym-clean2.spz",
        seq=["public/worlds/gym-4d/person/sequence.json"],
        cameras="public/worlds/gym-4d/cameras.json",
        scale=1.0020,
        sizeScale=0.5800,
        floor=-0.7097,
        pos=None,
        contact="floor",
    ),
    "elevator": dict(
        world="public/marble-elevator-clean.spz",
        people="public/worlds/elevator-4d/people.json",
        cameras="public/worlds/elevator-4d/cameras.json",
        scale=1.0800,
        floor=-0.6145,
        pos=[0.0, 0.0, 0.0],
        contact="floor",
    ),
    "lobby": dict(
        world="public/marble-lobby-clean.spz",
        seq=["public/worlds/lobby-4d/person/sequence.json"],
        cameras="public/worlds/lobby-4d/cameras.json",
        scale=1.5300,
        floor=-0.5464,
        pos=None,
        contact="floor",
    ),
    "living": dict(
        world="public/marble-living-finetuned.spz",
        seq=["public/worlds/living-4dpp/person/sequence.json"],
        cameras="public/worlds/living-4dpp/cameras.json",
        scale=0.3750,
        floor=-0.7257,
        pos=None,
        contact="floor",
    ),
    # atrium: the -4dpp avatar swap re-derived `scale` from the new body's height (1.0221). The Pi3X
    # camera tracks of atrium-4d and atrium-4dpp are byte-identical, so the frames are the same frames
    # and the registration scale is still the fitted 0.816. At 1.0221 he walks a metre off the balcony.
    "atrium": dict(
        world="public/marble-atrium-clean.spz",
        seq=["public/worlds/atrium-4dpp/person/sequence.json"],
        cameras="public/worlds/atrium-4dpp/cameras.json",
        scale=1.0221,
        sizeScale=0.8160,
        floor=-0.9015,
        pos=None,
        contact="floor",
    ),
}


def load_tracks(cfg):
    """[(id, seq_path, transform_translation)] for the clip, in manifest order."""
    if "people" in cfg:
        man = json.load(open(ROOT / cfg["people"]))
        base = (ROOT / cfg["people"]).parent
        out = []
        for p in man["people"]:
            tr = (p.get("transform") or {}).get("translation") or [0, 0, 0]
            out.append((p["id"], base / p["id"] / "sequence.json", np.array(tr, float), man))
        return out
    return [(Path(s).parent.parent.name, ROOT / s, np.zeros(3), None) for s in cfg["seq"]]


def stats_cm(v, mpu):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)] * mpu * 100.0
    if not len(v):
        return dict(n=0)
    return dict(
        n=int(len(v)),
        median=float(np.median(v)),
        mean=float(v.mean()),
        p10=float(np.percentile(v, 10)),
        p90=float(np.percentile(v, 90)),
        rms=float(np.sqrt((v**2).mean())),
        absmed=float(np.median(np.abs(v))),
        span=float(np.percentile(v, 90) - np.percentile(v, 10)),
    )


def ray_geometry(cameras_json, sample_src, feet_world, n, scale, stature_world):
    """Per sample: the unit ray from the source camera to the feet, and the pixel cost of moving.

    alpha = u.n, beta = sqrt(1-alpha^2). A displacement along u leaves the silhouette's POSITION
    unchanged and only changes its apparent SIZE, which costs about stature_px/(2z) px per unit; a
    displacement along the orthogonal direction in the ray-normal plane moves the silhouette by
    f/z px per unit. Those two numbers are the whole reason the solve prefers one dof over the other,
    and they come from the source intrinsics rather than from a taste setting.
    """
    cams = json.load(open(cameras_json))["cameras"]
    by = {c["sourceIndex"]: c for c in cams}
    keys = sorted(by)
    out = []
    for i, si in enumerate(sample_src):
        c = by.get(si) or by[min(keys, key=lambda q: abs(q - si))]
        C = np.array(c["camera_to_world"])[:3, 3] * scale
        K = np.array(c["source_intrinsics"])
        f = float((K[0, 0] + K[1, 1]) / 2)
        d = feet_world[i] - C
        z = float(np.linalg.norm(d))
        u = d / max(z, 1e-9)
        al = float(u @ n)
        be = float(np.sqrt(max(0.0, 1.0 - al * al)))
        w = n - al * u
        w = w / max(np.linalg.norm(w), 1e-9)
        s_px = f * stature_world / max(z, 1e-9)
        out.append(
            dict(C=C, u=u, w=w, z=z, f=f, alpha=al, beta=be, costA=s_px / (2 * z), costB=f / z)
        )
    return out


def silhouette_fit(
    feats,
    tracks,
    cameras,
    rows_per_track,
    scale,
    pos0,
    normal,
    masks_path,
    sigma_px,
    sigma_contact,
    jerk,
    stature,
    size_prior,
):
    """Fit each person's ray depth from observed silhouette height, without changing articulation.

    Height and centre rows replace the old assumption that the input silhouette was correct.
    A normal-plane translation permits vertical registration; contact remains a measured residual,
    not an exact constraint. Missing/cropped measurements do not invent a complete body height.
    Tables follow each sequence's sourceIndices (including gaps), never its manifest offset.
    """
    from silhouette_rows import load_track_masks, mask_rows, cam_matrices, avatar_rows

    masks, _hw, ms = load_track_masks(masks_path)
    by_src = {c["sourceIndex"]: c for c in cameras}
    result = {}
    for pi, (tid, frames, tr, _low, seq) in enumerate(feats):
        man = tracks[pi][3]
        if not man:
            continue
        person = next(p for p in man["people"] if p["id"] == tid)
        samples = {s: i for i, s in enumerate(man["sourceIndices"])}
        data = []
        for fi, (fr, src) in enumerate(zip(frames, seq["sourceIndices"])):
            cam = by_src.get(src)
            mask = masks.get(person["track"], {}).get(samples.get(src))
            if cam is None or mask is None:
                data.append(None)
                continue
            mr = mask_rows(mask, ms, cam["source_image_size"][1])
            if mr is None or (mr["croppedTop"] and mr["croppedBottom"]):
                data.append(None)
                continue
            R, C, K = cam_matrices(cam, scale, [0, 0, 0])
            P = fr[0] * scale + pos0 + tr
            foot = rows_per_track[pi][fi]
            F = np.array([foot["x"], foot["y"], foot["z"]])
            u = F - C
            depth = np.linalg.norm(u)
            u /= max(depth, 1e-9)
            w = normal - normal.dot(u) * u
            w /= max(np.linalg.norm(w), 1e-9)
            data.append(
                dict(P=P, F=F, R=R, C=C, K=K, mask=mr, u=u, w=w, contact=foot["res"], depth=depth)
            )
        if sum(d is not None for d in data) < 3:
            continue
        nf = len(data)
        # Missing samples borrow a basis, but receive no fabricated silhouette observation.
        have = [i for i, d in enumerate(data) if d is not None]
        basis = [data[i] or data[min(have, key=lambda k: abs(k - i))] for i in range(nf)]
        x = np.zeros(2 * nf + 1)  # last: one log body-size correction shared over the track
        for _iteration in range(12):
            equations, targets = [], []

            def add(i, jac, residual, sigma):
                row = np.zeros(len(x))
                row[2 * i : 2 * i + 2] = np.asarray(jac[:2]) / sigma
                row[-1] = jac[2] / sigma
                equations.append(row)
                targets.append(-residual / sigma)

            for i, d in enumerate(data):
                if d is None:
                    continue
                D = x[2 * i] * d["u"] + x[2 * i + 1] * d["w"]
                body = (d["P"] - d["F"]) * np.exp(x[-1])
                ar = avatar_rows(d["F"] + body + D, d["R"], d["C"], d["K"])
                eps = max(1e-5, d["depth"] * 0.001)
                jac = []
                for direction in (d["u"], d["w"]):
                    other = avatar_rows(d["F"] + body + D + eps * direction, d["R"], d["C"], d["K"])
                    jac.append([(other[k] - ar[k]) / eps for k in ("top", "bottom")])
                other = avatar_rows(d["F"] + body * np.exp(0.001) + D, d["R"], d["C"], d["K"])
                jac.append([(other[k] - ar[k]) / 0.001 for k in ("top", "bottom")])
                jac = np.array(jac).T
                mr = d["mask"]
                if not mr["croppedTop"] and not mr["croppedBottom"]:
                    # Explicit height term constrains depth; centre constrains registration.
                    add(
                        i,
                        jac[1] - jac[0],
                        (ar["bottom"] - ar["top"]) - (mr["bottom"] - mr["top"]),
                        sigma_px * np.sqrt(2),
                    )
                    add(
                        i,
                        (jac[0] + jac[1]) / 2,
                        (ar["top"] + ar["bottom"] - mr["top"] - mr["bottom"]) / 2,
                        sigma_px / np.sqrt(2),
                    )
                else:
                    k = "bottom" if mr["croppedTop"] else "top"
                    add(i, jac[1 if k == "bottom" else 0], ar[k] - mr[k], sigma_px)
                add(
                    i,
                    [normal.dot(d["u"]), normal.dot(d["w"]), 0],
                    d["contact"] + normal.dot(D),
                    sigma_contact,
                )
            for i in range(nf):
                for axis in range(2):
                    row = np.zeros(len(x))
                    row[2 * i + axis] = 1 / (4 * stature)
                    equations.append(row)
                    targets.append(-row.dot(x))
            row = np.zeros(len(x))
            row[-1] = 1 / max(size_prior, 1e-6)
            equations.append(row)
            targets.append(-row.dot(x))
            # World-space acceleration, so a turning ray cannot itself introduce jitter.
            for i in range(1, nf - 1):
                for axis in range(3):
                    row = np.zeros(len(x))
                    for j, c in ((i - 1, 1), (i, -2), (i + 1, 1)):
                        row[2 * j : 2 * j + 2] = (
                            c * np.array([basis[j]["u"][axis], basis[j]["w"][axis]]) / jerk
                        )
                    equations.append(row)
                    targets.append(-row.dot(x))
            step, *_ = np.linalg.lstsq(np.array(equations), np.array(targets), rcond=None)
            # Trust region: never cross the camera in a single linearisation.
            cap = min(d["depth"] for d in basis) * 0.2
            step *= min(1.0, cap / max(np.abs(step).max(), 1e-9))
            x += step
            if np.abs(step).max() < 1e-5:
                break
        shifts = np.array([x[2 * i] * d["u"] + x[2 * i + 1] * d["w"] for i, d in enumerate(basis)])
        size = float(np.exp(x[-1]))
        # Group scaling is about the origin; compensate so this fit scales about each foot.
        feet = np.array([[r["x"], r["y"], r["z"]] for r in rows_per_track[pi]])
        offsets = shifts + (1 - size) * (feet - pos0 - tr)
        heights, contacts, edge_errors = [], [], []
        for i, d in enumerate(data):
            if d is None:
                continue
            ar = avatar_rows(d["F"] + (d["P"] - d["F"]) * size + shifts[i], d["R"], d["C"], d["K"])
            if not d["mask"]["croppedTop"] and not d["mask"]["croppedBottom"]:
                heights.append(
                    (ar["bottom"] - ar["top"]) / (d["mask"]["bottom"] - d["mask"]["top"])
                )
            for edge, cropped in [("top", "croppedTop"), ("bottom", "croppedBottom")]:
                if not d["mask"][cropped]:
                    edge_errors.append(abs(ar[edge] - d["mask"][edge]))
            contacts.append(d["contact"] + normal.dot(shifts[i]))
        result[tid] = dict(
            offsetUnits=offsets.tolist(),
            sizeScale=size,
            sourceIndices=seq["sourceIndices"],
            measuredFrames=len(contacts),
            heightRatioMedian=float(np.median(heights)) if heights else None,
            contactResidualUnits=contacts,
            edgeResidualsPx=edge_errors,
            sigmaPx=sigma_px,
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clip", required=True, help="a name in CLIPS, or any clip name with --world/--person"
    )
    # generic entry, for scripts/run_clip.py's `placement` stage on a clip CLIPS has never seen
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--person", type=Path, default=None, help="<clip>-4d/person/sequence.json")
    ap.add_argument("--people", type=Path, default=None, help="<clip>-4d/people.json, for a cast")
    ap.add_argument("--cameras", type=Path, default=None)
    ap.add_argument("--contact", default="floor", choices=("floor", "pad"))
    ap.add_argument(
        "--fail-on-size",
        action="store_true",
        help="exit non-zero if the size check fails, instead of writing the solve with a warning",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--max-jerk-cm", type=float, default=0.5)
    ap.add_argument(
        "--sigma-contact-cm", type=float, default=2.5, help="the foot/back measurement noise floor"
    )
    ap.add_argument(
        "--sigma-px",
        type=float,
        default=12.0,
        help="source pixels of reprojection drift that cost as much as sigma-contact",
    )
    ap.add_argument(
        "--silhouette-sigma-px",
        type=float,
        default=1.0,
        help="tracked silhouette edge noise in source pixels; 0 disables the term",
    )
    ap.add_argument(
        "--silhouette-size-prior",
        type=float,
        default=0.15,
        help="log-size prior width for one fixed body scale per track; 0 locks size",
    )
    ap.add_argument(
        "--masks",
        type=Path,
        help="tracked masks.npz; defaults to .context/run/<clip>/tracks/masks.npz",
    )
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--scale", type=float, default=None, help="override the registration scale")
    ap.add_argument("--floor", type=float, default=None)
    a = ap.parse_args()
    if a.world:
        rel = lambda q: (
            str(Path(q).resolve().relative_to(ROOT)) if Path(q).is_absolute() else str(q)
        )
        cfg = dict(
            world=rel(a.world),
            base=rel(a.world),
            cameras=rel(a.cameras),
            contact=a.contact,
            scale=a.scale,
            floor=a.floor,
            pos=None,
        )
        if a.people:
            cfg["people"] = rel(a.people)
        else:
            cfg["seq"] = [rel(a.person)]
        if cfg["scale"] is None or cfg["floor"] is None:
            sys.exit("--world needs --scale (the fitted scale0) and --floor")
    else:
        if a.clip not in CLIPS:
            sys.exit(
                f"unknown clip {a.clip}; pass --world/--person/--cameras/--scale/--floor instead"
            )
        cfg = dict(CLIPS[a.clip])
    if a.scale is not None:
        cfg["scale"] = a.scale
    elif cfg.get("sizeScale"):
        cfg["shippedScale"] = cfg["scale"]
        cfg["scale"] = cfg["sizeScale"]
    if a.floor is not None:
        cfg["floor"] = a.floor

    tracks = load_tracks(cfg)
    print(f"[{a.clip}] {len(tracks)} track(s), world {cfg['world']}", flush=True)

    feats, bodyH0, low0 = [], None, None
    rng = np.random.default_rng(0)
    for tid, seq, tr, _man in tracks:
        fr = list(person_frames(seq))
        op0 = fr[0][0]
        ys = np.sort(op0[:, 1])
        l0 = float(ys[int(len(ys) * 0.01)])
        if low0 is None:
            low0, bodyH0 = l0, float(fr[0][1][:, 1].max() - fr[0][1][:, 1].min())
        feats.append([tid, fr, tr, l0, json.load(open(seq))])
        print(f"  {tid}: {len(fr)} frames, frame-0 low {l0:.4f}", flush=True)

    scale = cfg["scale"]
    pos0 = (
        np.array(cfg["pos"], float)
        if cfg["pos"] is not None
        else np.array([0.0, cfg["floor"] - low0 * scale, 0.0])
    )
    # stature: the longest principal extent of the body, robust to a raised arm and valid for a
    # supine subject, where a bounding-box height is his thickness rather than his height
    st = []
    for op, _al in feats[0][1][::10]:
        Q = op - op.mean(0)
        v = np.linalg.svd(
            Q[rng.choice(len(Q), min(5000, len(Q)), replace=False)], full_matrices=False
        )[2][0]
        pr = op @ v
        st.append(np.percentile(pr, 99.3) - np.percentile(pr, 0.7))
    stature = float(np.median(st)) * scale
    mpu = 1.70 / stature  # metres per world unit, if the person is 1.70 m
    print(
        f"  scale {scale}  pos0 {pos0.round(4)}  stature {stature:.4f} u  ->  1 u = {mpu:.4f} m",
        flush=True,
    )

    W = world_splats(ROOT / cfg["world"])
    print(f"  world {len(W):,} opaque splats", flush=True)

    meta = {}
    if cfg["contact"] == "pad":
        pad = bench_pad_plane(ROOT / cfg["world"], ROOT / cfg["base"])
        print(
            f"  pad: {pad['nPad']:,}/{pad['nAdded']:,} added splats, incline {pad['inclineDeg']:.1f} deg",
            flush=True,
        )
        meta["pad"] = dict(
            normal=pad["n"].tolist(), d=pad["d"], inclineDeg=pad["inclineDeg"], nPad=pad["nPad"]
        )
        rows_per_track = [
            measure_pad(fr, scale, pos0 + tr, pad, mpu=mpu) for _t, fr, tr, _l, _s in feats
        ]
        normal = pad["n"]
        probe_xyz = None
    else:
        probe = []
        for _t, fr, tr, _l, _s in feats:
            for op, _al in fr[:: max(1, len(fr) // 40)]:
                ys = np.sort(op[:, 1])
                lo = float(ys[int(len(ys) * 0.01)])
                f = op[:, 1] < lo + (op[:, 1].max() - op[:, 1].min()) * 0.04
                probe.append(
                    [
                        op[f, 0].mean() * scale + pos0[0] + tr[0],
                        0,
                        op[f, 2].mean() * scale + pos0[2] + tr[2],
                    ]
                )
        probe = np.array(probe)
        mx, mz = float(np.median(probe[:, 0])), float(np.median(probe[:, 2]))
        # the feet's own height is what says which surface is the floor; the preset `floor` may be
        # 50 cm out (living) and a percentile of the band may be on sub-floor fuzz (elevator)
        footY = float(
            np.median(
                [
                    r["y"]
                    for r in measure_floor(
                        feats[0][1][:: max(1, len(feats[0][1]) // 25)],
                        scale,
                        pos0 + feats[0][2],
                        [0.0, 0.0, 0.0],
                    )
                ]
            )
        )
        reg = (
            (np.abs(W[:, 0] - mx) < 1.2)
            & (np.abs(W[:, 2] - mz) < 1.6)
            & (np.abs(W[:, 1] - footY) < 0.45)
        )
        hy, ey = np.histogram(W[reg][:, 1], 90)
        seed = float(0.5 * (ey[hy.argmax()] + ey[hy.argmax() + 1]))
        coef, nrm, dpl, nin = fit_plane(W[reg], seed=seed)
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        yc = float(coef[0] * mx + coef[1] * mz + coef[2])
        print(
            f"  floor plane y = {coef[0]:+.4f}x {coef[1]:+.4f}z {coef[2]:+.4f}, tilt {tilt:.2f} deg, "
            f"{nin:,} inliers; y at walked centre {yc:.4f} (preset floor {cfg['floor']})",
            flush=True,
        )
        meta["floorPlane"] = dict(
            coef=coef.tolist(), normal=nrm.tolist(), tiltDeg=tilt, inliers=nin, yAtWalkedCentre=yc
        )
        rows_per_track = [measure_floor(fr, scale, pos0 + tr, coef) for _t, fr, tr, _l, _s in feats]
        normal = nrm

    # ---- the size check, before anything is corrected
    cams = json.load(open(ROOT / cfg["cameras"]))["cameras"]
    if "floorPlane" in meta:
        cf = meta["floorPlane"]["coef"]
    else:
        cf = [0.0, 0.0, cfg["floor"]]
    hs = np.array(
        [
            (np.array(c["camera_to_world"])[:3, 3] * scale)[1]
            - (
                cf[0] * (np.array(c["camera_to_world"])[:3, 3] * scale)[0]
                + cf[1] * (np.array(c["camera_to_world"])[:3, 3] * scale)[2]
                + cf[2]
            )
            for c in cams
        ]
    )
    camM = float(np.median(hs) * mpu)
    # the trace, not just the median (review §4E): flat inside the band is a pass, flat outside it is
    # a scale error, and a slope of more than 20 cm is camera drift that must not be absorbed
    # silently by the per-sample table.
    trace = hs * mpu
    tt = np.arange(len(trace), dtype=float)
    slope_m = float(np.polyfit(tt, trace, 1)[0] * (len(trace) - 1)) if len(trace) > 2 else 0.0
    reg2 = (np.abs(W[:, 0] - pos0[0]) < 1.5) & (np.abs(W[:, 2] - pos0[2] - 1.5) < 3.0)
    yy = W[reg2][:, 1]
    yy = yy[(yy > cfg["floor"] - 0.6) & (yy < cfg["floor"] + 4.0)]
    hh, ee = np.histogram(yy, 160)
    cen = 0.5 * (ee[:-1] + ee[1:])
    abv = cen > cfg["floor"] + 0.55
    ceil = float(cen[abv][hh[abv].argmax()]) if abv.any() else float("nan")
    f2c = float((ceil - cf[2]) * mpu)
    # the camera-height ruler is the gate: a handheld phone is 1.15-1.85 m above the floor, it is
    # independent of the avatar entirely, and it catches every size error on record. Floor-to-ceiling
    # corroborates but does not gate, because the strongest y layer above the floor can be a soffit,
    # a mezzanine or the top of a mirror rather than the ceiling.
    size_ok = 1.15 <= camM <= 1.85
    ceil_ok = 2.10 <= f2c <= 3.60
    drifted = abs(slope_m) > 0.20
    meta["sizeCheck"] = dict(
        statureUnits=stature,
        metresPerWorldUnit=mpu,
        cameraHeightM=camM,
        floorToCeilingM=f2c,
        pass_=bool(size_ok),
        ceilingPlausible=bool(ceil_ok),
        rule="with the person taken as 1.70 m, a handheld source camera must sit 1.15-1.85 m "
        "above the fitted floor; floor-to-ceiling corroborates",
        cameraHeightTrace=dict(
            metres=[round(float(x), 4) for x in trace],
            medianM=camM,
            p10M=float(np.percentile(trace, 10)),
            p90M=float(np.percentile(trace, 90)),
            endToEndSlopeM=slope_m,
            drifted=bool(drifted),
            verdict=("drift" if drifted else "flat in band" if size_ok else "flat out of band"),
            note="in AVATAR metres unless scripts/world_ruler.py has adopted a "
            "world ruler for this clip; a flat trace outside the band is a "
            "scale error, a sloped one is solve drift",
        ),
    )
    print(
        f"  SIZE CHECK: source camera {camM:.2f} m above the floor (want 1.15-1.85), "
        f"room {f2c:.2f} m floor to ceiling ({'plausible' if ceil_ok else 'IMPLAUSIBLE'}) "
        f"-> {'PASS' if size_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"  CAMERA TRACE: {trace.min():.2f}-{trace.max():.2f} m over {len(trace)} cameras, "
        f"end to end {slope_m:+.2f} m -> {meta['sizeCheck']['cameraHeightTrace']['verdict']}",
        flush=True,
    )
    if not size_ok:
        # the camera's height over the world floor is fixed in world units; the metre comes from the
        # avatar, and the avatar shrinks as the scale drops, so the height in metres goes as 1/scale
        want = 1.45 / camM
        msg = (
            f"SIZE CHECK FAILED for {a.clip}. Taking the person as 1.70 m makes 1 world unit "
            f"{mpu:.3f} m, which puts the handheld source camera {camM:.2f} m above the floor and the "
            f"room {f2c:.2f} m floor to ceiling. One of those is not a real room. A registration "
            f"scale of about {cfg['scale'] / want:.4f} (x{1 / want:.2f}) would put the camera at 1.45 m. "
            f"The depth-ratio fit cannot catch this: it is a RATIO between the Marble world and the "
            f"depth map, so a global size error cancels in it. Check the world against something "
            f"real - floor tile pitch, ceiling height, a doorway - before shipping."
        )
        if a.fail_on_size:
            sys.exit(msg)
        print("  !! " + msg, flush=True)

    nF = max(len(r) for r in rows_per_track)
    M = np.full((len(feats), nF), np.nan)
    for i, rows in enumerate(rows_per_track):
        for r in rows:
            M[i, r["k"]] = r["res"]
    before = {feats[i][0]: stats_cm(M[i], mpu) for i in range(len(feats))}
    print("\n  BEFORE (cm, + floats above the surface):", flush=True)
    for kk, sv in before.items():
        print(
            f"    {kk:11s} n {sv['n']:4d}  median {sv['median']:+6.1f}  p10 {sv['p10']:+6.1f}  p90 {sv['p90']:+6.1f}  "
            f"rms {sv['rms']:5.1f}  |med| {sv['absmed']:4.1f}",
            flush=True,
        )

    if a.measure_only:
        if a.report:
            json.dump(
                dict(
                    clip=a.clip,
                    mpu=mpu,
                    scale=scale,
                    pos0=pos0.tolist(),
                    before=before,
                    meta=meta,
                    measured={feats[i][0]: M[i].tolist() for i in range(len(feats))},
                ),
                open(a.report, "w"),
                indent=1,
            )
        return

    # ---- ray geometry from the primary track
    src = feats[0][4].get("sourceIndices") or list(range(nF))
    feet_world = np.array(
        [
            [r["x"], r["y"], r["z"]] if "x" in r else [pos0[0], pos0[1], pos0[2]]
            for r in rows_per_track[0]
        ]
    )
    if cfg["contact"] == "pad":
        feet_world = np.array(
            [
                [
                    np.median(op[:, 0]) * scale + pos0[0],
                    np.median(op[:, 1]) * scale + pos0[1],
                    np.median(op[:, 2]) * scale + pos0[2],
                ]
                for op, _al in feats[0][1]
            ]
        )
    G = ray_geometry(
        ROOT / cfg["cameras"], src[: len(feet_world)], feet_world, normal, scale, stature
    )
    # The system is indexed by frame k over ALL tracks (nF = the longest one), but the ray geometry
    # comes from the primary's rows, which can be fewer (a frame it was not detected in) and are not
    # guaranteed contiguous. Key the geometry by the primary's frame index and fill the frames it
    # lacks from its nearest observed frame: the camera moves little between neighbours.
    ks = [r["k"] for r in rows_per_track[0]][: len(G)]
    byk = {k: g for k, g in zip(ks, G)}
    have = np.array(sorted(byk))

    def geo(k):
        return byk[k] if k in byk else byk[have[np.abs(have - k).argmin()]]

    G = [geo(k) for k in range(nF)]
    A = np.array([g["alpha"] for g in G])
    B = np.array([g["beta"] for g in G])
    Wa = np.array([g["costA"] for g in G]) / a.sigma_px
    Wb = np.array([g["costB"] for g in G]) / a.sigma_px
    sigma_c = a.sigma_contact_cm / 100.0 / mpu
    print(
        f"\n  ray geometry: alpha (ray.normal) {A.min():+.2f}..{A.max():+.2f}; "
        f"cost of 1 u along the ray {np.median([g['costA'] for g in G]):.0f} px, across it "
        f"{np.median([g['costB'] for g in G]):.0f} px",
        flush=True,
    )

    lam, av, bv, kv, edf = choose_lambda(M, A, B, Wa, Wb, sigma_c, a.max_jerk_cm / 100.0 / mpu)
    delta = av[:, None] * np.array([g["u"] for g in G]) + bv[:, None] * np.array(
        [g["w"] for g in G]
    )
    fit = av[None, :] * A[None, :] + bv[None, :] * B[None, :] + kv[:, None]
    after_M = M + fit
    aft = {feats[i][0]: stats_cm(after_M[i], mpu) for i in range(len(feats))}
    slide = np.abs(av) * mpu * 100
    across = np.abs(bv) * mpu * 100
    px = np.abs(av) * np.array([g["costA"] for g in G]) + np.abs(bv) * np.array(
        [g["costB"] for g in G]
    )
    print(f"  SOLVE: lam {lam:.4g}, effective dof {edf:.1f} of {nF} samples", flush=True)
    print(
        f"    slide along the ray  median {np.median(slide):5.1f} cm  max {slide.max():5.1f} cm",
        flush=True,
    )
    print(
        f"    move across the ray  median {np.median(across):5.1f} cm  max {across.max():5.1f} cm",
        flush=True,
    )
    print(
        f"    reprojection cost    median {np.median(px):5.1f} px  p90 {np.percentile(px, 90):5.1f} px  (1920 px source)",
        flush=True,
    )
    print(f"    per-person constants {[round(float(x) * mpu * 100, 1) for x in kv]} cm", flush=True)
    print("  AFTER (cm):", flush=True)
    for kk, sv in aft.items():
        print(
            f"    {kk:11s} n {sv['n']:4d}  median {sv['median']:+6.1f}  p10 {sv['p10']:+6.1f}  p90 {sv['p90']:+6.1f}  "
            f"rms {sv['rms']:5.1f}  |med| {sv['absmed']:4.1f}",
            flush=True,
        )

    # GUARDRAILS' rule in code (review §4J): a residual below the measurement's own noise floor is
    # not a better fit, it is a fit with nothing left to report. So is an "after" that equals its
    # "before". Recorded here; scripts/anchor_checks.py is what refuses on it.
    refusals = []
    for kk in aft:
        if aft[kk]["rms"] < a.sigma_contact_cm:
            refusals.append(
                f"{kk}: after RMS {aft[kk]['rms']:.2f} cm is below the "
                f"{a.sigma_contact_cm:.1f} cm contact noise floor, so it is not a "
                f"measurement of placement -- it is the smoother reproducing its input"
            )
        if all(abs(aft[kk][f] - before[kk][f]) < 1e-4 for f in ("median", "rms", "p10", "p90")):
            refusals.append(
                f"{kk}: after equals before to four decimals; this solve changed nothing"
            )
    if abs(float(np.median(px))) < 1e-9:
        refusals.append("the reprojection cost is 1e-9 or less: no information")
    for r in refusals:
        print(f"  REFUSAL: {r}", flush=True)

    out = (
        a.out
        or ((ROOT / cfg["people"]).parent if "people" in cfg else tracks[0][1].parent.parent)
        / "placement.json"
    )
    doc = dict(
        schema="wander.placement/1",
        clip=a.clip,
        note=(
            "Per-sample placement correction, solved by scripts/place_solve.py. The viewer applies it "
            "only under ?place=1: registrationScale and floorY replace the preset values, "
            "offsetUnits[sample] is added to every person, perPerson[id].constantUnits on top. "
            "A perPerson offsetUnits table overrides shared offsets, indexed by that person's sequence; "
            "sizeScale multiplies its body scale."
        ),
        world=cfg["world"],
        registrationScale=scale,
        floorY=float(cf[2] if "floorPlane" in meta else cfg["floor"]),
        pos0=pos0.tolist(),
        metresPerWorldUnit=mpu,
        samples=int(nF),
        contact=cfg["contact"],
        surfaceNormal=[float(x) for x in normal],
        sourceIndices=[int(x) for x in src[:nF]],
        lambdaGcv=float(lam),
        effectiveDof=float(edf),
        sigmaContactCm=a.sigma_contact_cm,
        sigmaPx=a.sigma_px,
        offsetUnits=[[float(x) for x in row] for row in delta],
        # TOTAL per-person offset from pos0, manifest transform included, so the viewer replaces
        # `transform.translation` rather than stacking on it and double-counting the level-B fix
        perPerson={
            feats[i][0]: dict(
                constantUnits=[float(x) for x in (feats[i][2] + kv[i] * normal)],
                solvedCm=float(kv[i] * mpu * 100),
                manifestTranslation=[float(x) for x in feats[i][2]],
            )
            for i in range(len(feats))
        },
        reprojectionCostPx=dict(
            median=float(np.median(px)), p90=float(np.percentile(px, 90)), max=float(px.max())
        ),
        residualCm=dict(before=before, after=aft),
        noiseFloor=dict(contactCm=a.sigma_contact_cm, refusals=refusals, ok=not refusals),
        surface=meta,
    )
    masks_path = a.masks or ROOT / ".context" / "run" / a.clip / "tracks" / "masks.npz"
    if a.silhouette_sigma_px > 0 and masks_path.exists() and cfg["contact"] == "floor":
        fits = silhouette_fit(
            feats,
            tracks,
            cams,
            rows_per_track,
            scale,
            pos0,
            normal,
            masks_path,
            a.silhouette_sigma_px,
            sigma_c,
            a.max_jerk_cm / 100 / mpu,
            stature,
            a.silhouette_size_prior,
        )
        for tid, fit_person in fits.items():
            entry = doc["perPerson"][tid]
            entry["constantUnits"] = entry["manifestTranslation"]
            entry.update(
                {
                    k: v
                    for k, v in fit_person.items()
                    if k not in ("contactResidualUnits", "edgeResidualsPx")
                }
            )
            entry["solvedCm"] = 0.0
            doc["residualCm"]["after"][tid] = stats_cm(fit_person["contactResidualUnits"], mpu)
            print(
                f"  SILHOUETTE {tid}: height {fit_person['heightRatioMedian']}, contact RMS "
                f"{doc['residualCm']['after'][tid]['rms']:.2f} cm (fit residual, not held-out)",
                flush=True,
            )
        if fits:
            # The old cost assumed the unfitted silhouette was correct; it is not the final fit's error.
            doc["legacyReprojectionCostPx"] = doc["reprojectionCostPx"]
            doc["reprojectionCostPx"] = None
            edges = np.array([e for f in fits.values() for e in f["edgeResidualsPx"]])
            doc["silhouetteFit"] = dict(
                masks=str(masks_path),
                sigmaPx=a.silhouette_sigma_px,
                edgeResidualPx=dict(
                    median=float(np.median(edges)), p90=float(np.percentile(edges, 90))
                ),
                note="Per-person offsetUnits replace the shared table; indexed by each sequence frame. In-sample mask fit, not independent validation.",
            )
            # Recompute the noise-floor refusal against the correction actually shipped.
            doc["noiseFloor"]["refusals"] = [
                f"{tid}: fitted contact is below the contact noise floor"
                for tid, res in doc["residualCm"]["after"].items()
                if res["rms"] < a.sigma_contact_cm
            ]
            doc["noiseFloor"]["ok"] = not doc["noiseFloor"]["refusals"]
            # Keep the stated ruler: the fitted primary body is assumed to be 1.70 m.
            primary_size = fits.get(feats[0][0], {}).get("sizeScale", 1.0)
            factor = 1 / primary_size
            doc["metresPerWorldUnit"] *= factor
            doc["sigmaContactCm"] *= factor
            doc["noiseFloor"]["contactCm"] *= factor
            for phase in doc["residualCm"].values():
                for stat in phase.values():
                    for key in stat:
                        if key != "n":
                            stat[key] *= factor
            sc = doc["surface"]["sizeCheck"]
            sc["statureUnits"] *= primary_size
            for key in ("metresPerWorldUnit", "cameraHeightM", "floorToCeilingM"):
                sc[key] *= factor
            ct = sc["cameraHeightTrace"]
            ct["metres"] = [v * factor for v in ct["metres"]]
            for key in ("medianM", "p10M", "p90M", "endToEndSlopeM"):
                ct[key] *= factor
            sc["pass_"] = 1.15 <= sc["cameraHeightM"] <= 1.85
            ct["drifted"] = abs(ct["endToEndSlopeM"]) > 0.20
            ct["verdict"] = (
                "drift" if ct["drifted"] else "flat in band" if sc["pass_"] else "flat out of band"
            )
            doc["silhouetteFit"]["rulerNote"] = (
                "Metres assume the fitted primary body is 1.70 m; quote body-heights without an independent ruler."
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out, "w"), indent=1)
    print(f"\n  wrote {out}", flush=True)
    if a.report:
        json.dump(doc, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
