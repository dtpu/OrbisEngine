#!/usr/bin/env python3
"""The filmed silhouette as a per-person, per-sample measurement of where the avatar is.

    uv run --locked --group inference python scripts/silhouette_rows.py --world public/worlds/hpwide-4d --scale 0.5725 --floor-coef a,b,c

place_solve.py's premise since round 20 was that the avatar already lands on the filmed person and the
solve only has to be CHARGED for moving it. hpwide broke that: both avatars stood 1.2x too large on
the boys with their feet 45 px below the filmed feet, and a solve that pays for every pixel it moves
could not fix what it was never allowed to see. This module reads the Mask R-CNN instance masks the
tracker saved (.context/run/<clip>/tracks/masks.npz, keyed t<track>_s<sample>) and returns, for each
person and each sample, the two image rows the footage gives -- the top of the head and the bottom of
the feet -- against the same two rows of the projected avatar, plus the numerical derivatives of those
rows with respect to the displacements the solve is allowed to make. Rows, not areas: a person cut off
at the frame's bottom edge (Harry, all 30 samples) still has a measurable head row, while his area is
a measurement of the frame edge.

Everything is projected with the packaged frame's own cameras.json (source intrinsics, packaged
camera-to-world). The avatar rows are percentile rows of the opaque splats, matching the stature rule
in place_solve; the mask rows come from the first and last rows with at least MIN_ROW_PX mask pixels
after upsampling from the tracker's half-resolution grid.

A silhouette row is one number, so this cannot measure lateral position or depth on its own. It
measures depth only together with the floor: the feet row and the floor plane fix where the feet are,
and the head row above that fixed point is then a measurement of the body's SIZE. That is what
place_solve solves with it.
"""

import argparse, json
from pathlib import Path

import numpy as np

MIN_ROW_PX = 3  # a mask row with fewer pixels is a stray blob, not the head or a foot
EDGE_PX = 2  # a mask that reaches within this of the frame edge is cropped there
TOP_PCT, BOT_PCT = 0.5, 99.5


def load_track_masks(npz_path: Path):
    """{track: {sample: bool (h, w)}}, (h, w), and the mask grid's scale against the source frame."""
    z = np.load(npz_path)
    full_h, full_w, ms = [float(x) for x in z["shape"]]
    h, w = int(round(full_h * ms)), int(round(full_w * ms))
    out = {}
    for k in z.files:
        if k == "shape":
            continue
        t, s = k.split("_")
        out.setdefault(int(t[1:]), {})[int(s[1:])] = (
            np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool)
        )
    return out, (h, w), ms


def mask_rows(m: np.ndarray, ms: float, full_h: int):
    """(top, bottom, cropped_top, cropped_bottom) in SOURCE rows, or None for an empty mask."""
    counts = m.sum(1)
    rows = np.nonzero(counts >= max(1, int(round(MIN_ROW_PX * ms))))[0]
    if not len(rows):
        return None
    top, bot = rows[0] / ms, (rows[-1] + 1) / ms
    return dict(
        top=float(top),
        bottom=float(bot),
        croppedTop=bool(rows[0] <= EDGE_PX * ms),
        croppedBottom=bool(rows[-1] >= m.shape[0] - 1 - EDGE_PX * ms),
        areaPx=float(m.sum() / (ms * ms)),
    )


def cam_matrices(cam, scale, pos0):
    """World-frame camera for the viewer's placement: the SfM frame is scale * native + pos0."""
    M = np.asarray(cam["camera_to_world"], float)
    u, _, vt = np.linalg.svd(M[:3, :3])
    R = u @ vt
    C = M[:3, 3] * scale + np.asarray(pos0, float)
    K = np.asarray(cam["source_intrinsics"], float)
    return R, C, K


def project(P, R, C, K):
    """Rows and columns of world points P in an OpenGL camera (looks down -z, y up)."""
    loc = (P - C) @ R
    z = -loc[:, 2]
    ok = z > 1e-6
    zs = np.where(ok, z, 1.0)
    u = K[0, 0] * loc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * (-loc[:, 1]) / zs + K[1, 2]
    return u, v, z, ok


def avatar_rows(P, R, C, K):
    _u, v, z, ok = project(P, R, C, K)
    if ok.sum() < 50:
        return None
    v = v[ok]
    return dict(
        top=float(np.percentile(v, TOP_PCT)),
        bottom=float(np.percentile(v, BOT_PCT)),
        zMedian=float(np.median(z[ok])),
    )


def measure(frames_world, cams_by_sample, masks, track, samples, hw_full, ms, feet_world, up):
    """Per sample: mask rows, avatar rows, and d(row)/d(displacement) for a 1 world-unit move along
    the feet ray (`u`), along the in-plane up direction (`w`) and for a 1 % size change about the feet.

    frames_world: list of (opaque splats in world units) aligned with `samples`
    feet_world:   the feet point per sample (place_solve's 1st-percentile foot), world units
    """
    H, W = hw_full
    out = []
    for i, (P, s) in enumerate(zip(frames_world, samples)):
        cam = cams_by_sample.get(s)
        m = masks.get(track, {}).get(s)
        if cam is None or m is None or P is None:
            out.append(None)
            continue
        R, C, K = cam_matrices(*cam)
        mr = mask_rows(m, ms, H)
        ar = avatar_rows(P, R, C, K)
        if mr is None or ar is None:
            out.append(None)
            continue
        F = feet_world[i]
        ray = F - C
        z = float(np.linalg.norm(ray))
        u = ray / z
        w = up - (up @ u) * u
        w /= max(np.linalg.norm(w), 1e-9)
        eps = 0.01 * z
        d = {}
        for name, D in (("u", u * eps), ("w", w * eps)):
            r2 = avatar_rows(P + D, R, C, K)
            d[name] = ((r2["top"] - ar["top"]) / eps, (r2["bottom"] - ar["bottom"]) / eps)
        # size about the feet point: rows move away from the feet row in proportion
        r3 = avatar_rows(F + (P - F) * 1.01, R, C, K)
        d["s"] = ((r3["top"] - ar["top"]) / 0.01, (r3["bottom"] - ar["bottom"]) / 0.01)
        out.append(
            dict(
                sample=int(s),
                mask=mr,
                avatar=ar,
                z=z,
                u=u.tolist(),
                w=w.tolist(),
                dTop=dict(u=d["u"][0], w=d["w"][0], s=d["s"][0]),
                dBottom=dict(u=d["u"][1], w=d["w"][1], s=d["s"][1]),
                headInFrame=bool(0 <= ar["top"] < H and not mr["croppedTop"]),
                feetInFrame=bool(ar["bottom"] < H - EDGE_PX and not mr["croppedBottom"]),
            )
        )
    return out


def summarize(rows):
    """Height ratio and row offsets over the samples where they are measurable."""
    hr, dtop, dbot = [], [], []
    for r in rows:
        if r is None:
            continue
        dtop.append(r["avatar"]["top"] - r["mask"]["top"])
        if r["feetInFrame"]:
            dbot.append(r["avatar"]["bottom"] - r["mask"]["bottom"])
            hr.append(
                (r["avatar"]["bottom"] - r["avatar"]["top"])
                / max(1.0, r["mask"]["bottom"] - r["mask"]["top"])
            )
    q = lambda v: float(np.median(v)) if len(v) else float("nan")
    return dict(
        n=sum(r is not None for r in rows),
        nFeet=len(dbot),
        headRowOffsetPx=q(dtop),
        feetRowOffsetPx=q(dbot),
        heightRatio=q(hr),
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from place_solve import person_frames, measure_floor

    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True, help="public/worlds/<clip>-4d")
    ap.add_argument("--masks", type=Path, default=None, help=".context/run/<clip>/tracks/masks.npz")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--pos0", default="0,0,0")
    ap.add_argument(
        "--floor-coef", default=None, help="a,b,c of y = a x + b z + c, for the floor-implied depth"
    )
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    man = json.loads((a.world / "people.json").read_text())
    clip = man["clip"]
    name = Path(clip).stem
    masks_path = a.masks or root / ".context" / "run" / name / "tracks" / "masks.npz"
    masks, hw, ms = load_track_masks(masks_path)
    cams = json.loads((a.world / "cameras.json").read_text())["cameras"]
    hw_full = tuple(cams[0]["source_image_size"][::-1])
    pos0 = np.array([float(x) for x in a.pos0.split(",")])
    by_src = {c["sourceIndex"]: c for c in cams}
    cams_by_sample = {
        s: (by_src[si], a.scale, pos0) for s, si in enumerate(man["sourceIndices"]) if si in by_src
    }
    coef = [float(x) for x in a.floor_coef.split(",")] if a.floor_coef else None
    for p in man["people"]:
        seq = a.world / p["sequence"]
        fr = list(person_frames(seq))
        samples = [man["sourceIndices"].index(si) for si in p["sourceIndices"]]
        frames_world = [op * a.scale + pos0 for op, _ in fr]
        feet = measure_floor(fr, a.scale, pos0, coef or [0, 0, 0])
        feet_world = np.array([[r["x"], r["y"], r["z"]] for r in feet])
        rows = measure(
            frames_world,
            cams_by_sample,
            masks,
            p["track"],
            samples,
            hw_full,
            ms,
            feet_world,
            np.array([0, 1.0, 0]),
        )
        print(f"\n{p['id']} (track {p['track']}): {summarize(rows)}")
        for r, f in zip(rows, feet):
            if r is None:
                continue
            zf = ""
            if coef is not None:
                C = (
                    np.array(cams_by_sample[r["sample"]][0]["camera_to_world"])[:3, 3] * a.scale
                    + pos0
                )
                u = np.array(r["u"])
                n = np.array([-coef[0], 1.0, -coef[1]])
                d = coef[2]
                t = (d - C @ n) / (u @ n) if abs(u @ n) > 1e-6 else float("nan")
                zf = f" zFloor {t:6.3f} ({t / r['z']:.3f})"
            print(
                f"  s{r['sample']:02d} mask {r['mask']['top']:6.1f}-{r['mask']['bottom']:6.1f}{'c' if r['mask']['croppedBottom'] else ' '} "
                f"avatar {r['avatar']['top']:6.1f}-{r['avatar']['bottom']:6.1f}  z {r['z']:.3f}{zf}  "
                f"dTop u {r['dTop']['u']:+7.1f} w {r['dTop']['w']:+7.1f} s {r['dTop']['s']:+7.1f} | dBot u {r['dBottom']['u']:+6.1f} w {r['dBottom']['w']:+7.1f} s {r['dBottom']['s']:+5.1f}  feet-plane {f['res'] * 100:+5.1f} cm/u"
            )
