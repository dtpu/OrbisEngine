#!/usr/bin/env python3
"""Derive collision geometry for a world from its own splat cloud. Format: `wander.collision/2`.

Nothing here is hand-authored. The floor is the dominant horizontal plane of the opaque splats, the
height field is the 10th-percentile height of those splats per cell around that plane (steps and
ramps survive; sub-floor fuzz does not), and the walls are the largest vertical planes RANSAC finds
above the floor. Every number carries the count of splats that voted for it, so a viewer or a
solver can tell a wall seen by 200 000 splats from one seen by 20 000.

Frame: the VIEWER's world frame - the Marble spz after fourd.html's `world.rotation.x = PI`
(y -> -y, z -> -z), in world units. That is the frame `floorY`, `pos0` and every placed person or
object live in, so a body simulated against this geometry needs no further transform to be drawn.
`metresPerWorldUnit` is copied from the world's people.json / placement.json so the solver can
turn 9.80665 m/s^2 into units.

  uv run --locked --group inference python scripts/world_collision.py --world elevator-4d --spz public/marble-elevator-clean.spz
"""

import argparse, gzip, json, os, struct, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_spz_xyz(path):
    """Positions and alpha from a v2 shDegree-0 spz (same decoder as scripts/world_ruler.py)."""
    raw = gzip.open(path).read()
    magic, ver, n, sh, fb, _flags, _r = struct.unpack("<IIIBBBB", raw[:16])
    assert magic == 0x5053474E and ver == 2 and sh == 0, (hex(magic), ver, sh)
    pos_b = np.frombuffer(raw, np.uint8, n * 9, 16).reshape(n, 3, 3)
    fixed = (
        pos_b[:, :, 0].astype(np.int32)
        | (pos_b[:, :, 1].astype(np.int32) << 8)
        | (pos_b[:, :, 2].astype(np.int32) << 16)
    )
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    xyz = fixed.astype(np.float32) / float(1 << fb)
    alpha = np.frombuffer(raw, np.uint8, n, 16 + n * 9).astype(np.float32) / 255.0
    return xyz, alpha


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_floor(P, seed_y=None):
    """Dominant near-horizontal plane. Same recipe as frame_align.world_floor: seed from the y
    histogram peak, then shrinking-band least squares. Returns (normal, d) with n.p = d, and the
    inlier count at the tightest band."""
    if seed_y is None:
        h, e = np.histogram(P[:, 1], 400)
        seed_y = float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
    c = np.array([0.0, 0.0, seed_y])
    inl = np.zeros(len(P), bool)
    for band in (0.20, 0.12, 0.055, 0.035):
        for _ in range(10):
            d = P[:, 1] - (P[:, 0] * c[0] + P[:, 2] * c[1] + c[2])
            inl = np.abs(d) < band
            if inl.sum() < 500:
                break
            A = np.stack([P[inl, 0], P[inl, 2], np.ones(int(inl.sum()))], 1)
            c, *_ = np.linalg.lstsq(A, P[inl, 1], rcond=None)
    n = unit([-c[0], 1.0, -c[1]])
    d = float(n @ np.array([0.0, c[2], 0.0]))
    return n, d, int(inl.sum())


def height_field(P, n, d, cell, band, min_count, pct):
    """Per-cell height of the floor surface along `n`, from the low percentile of splats within
    `band` of the plane. Cells with fewer than `min_count` votes are left empty (null): a solver must
    fall back to the plane there and say so, not invent a floor."""
    t = P @ n - d
    sel = np.abs(t) < band
    Q = P[sel]
    tq = t[sel]
    # plane-local axes for the grid
    u = unit(np.cross(n, [0, 0, 1]))
    w = unit(np.cross(u, n))
    gu = Q @ u
    gw = Q @ w
    # stray splats metres outside the room would otherwise set the grid extent
    lo_u, hi_u = np.percentile(gu, [0.2, 99.8])
    lo_w, hi_w = np.percentile(gw, [0.2, 99.8])
    keep = (gu >= lo_u) & (gu <= hi_u) & (gw >= lo_w) & (gw <= hi_w)
    gu, gw, tq = gu[keep], gw[keep], tq[keep]
    iu = np.floor(gu / cell).astype(np.int64)
    iw = np.floor(gw / cell).astype(np.int64)
    u0, u1, w0, w1 = int(iu.min()), int(iu.max()), int(iw.min()), int(iw.max())
    W, H = u1 - u0 + 1, w1 - w0 + 1
    key = (iw - w0) * W + (iu - u0)
    order = np.argsort(key, kind="stable")
    key = key[order]
    tq = tq[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    heights = np.full(W * H, np.nan, np.float32)
    counts = np.zeros(W * H, np.int32)
    for s, e in zip(starts, ends):
        if e - s >= min_count:
            heights[key[s]] = np.percentile(tq[s:e], pct)
            counts[key[s]] = e - s
    return dict(
        axisU=u.tolist(),
        axisW=w.tolist(),
        cell=cell,
        originU=u0 * cell,
        originW=w0 * cell,
        width=W,
        height=H,
        heights=heights.reshape(H, W),
        counts=counts.reshape(H, W),
    )


def occupancy_grid(
    P,
    seed_y,
    upm,
    cell_m,
    stature_m,
    min_count,
    block_count,
    window_m,
    pct,
    band_lo,
    band_hi,
    step_up,
    drop_max_m,
    hull_m,
    extent_m,
    close_m,
    below_m,
    bin_stature,
    bin_count,
    seed_xz=(0.0, 0.0),
):
    """Minecraft-style walk grid for fourd.html ?walk=1, axis-aligned in world x/z over the whole
    opaque cloud (extent clipped to the 0.2..99.8 percentiles so stray splats do not set it).
    Per column every dense surface is a candidate: each `window` tall slab holding `min_count`
    opaque splats, read at its `pct` percentile, taken greedily from the bottom. Which candidate is
    THE walkable floor is decided by connectivity, not height: a flood fill from the floor-known
    column nearest `seed_xz` at `seed_y` (the viewer's own floor under camera 0) carries a height
    into each neighbour and accepts the candidate nearest it within [-drop_max, +step_up x
    stature]; a column with none stays null. That is what keeps a ceiling from becoming the floor
    where the real floor is sparse - the band the people walked on is their own occlusion shadow
    and has almost no floor splats - and what lets a staircase be followed tread by tread.
    `dist` is every column's distance in cells to the nearest floor-known column: within `hull_m`
    of one a null column is walked over at held height, beyond it the viewer refuses to go further
    out (R9: without this the walker crossed 6-14 m of void at held height). A column is BLOCKED
    when `block_count` splats sit in the body band [band_lo, band_hi] x stature above its own
    floor, or above the nearest floor-known column's floor when it has none (R9: measured against
    a carried height instead, the escalator "balustrade" was 12-22 ceiling splats). People are not
    in the cloud; a person Marble baked in as static junk is, and blocks like furniture."""
    from collections import deque

    cell = cell_m * upm
    st = stature_m * upm
    win = window_m * upm
    drop = drop_max_m * upm
    step = step_up * st
    # a walk is a few tens of metres at most; an outdoor Marble world (hpwide) is 180 u across
    ext = extent_m * upm
    P = P[(np.abs(P[:, 0] - seed_xz[0]) <= ext) & (np.abs(P[:, 2] - seed_xz[1]) <= ext)]
    lo_x, hi_x = np.percentile(P[:, 0], [0.2, 99.8])
    lo_z, hi_z = np.percentile(P[:, 2], [0.2, 99.8])
    keep = (P[:, 0] >= lo_x) & (P[:, 0] <= hi_x) & (P[:, 2] >= lo_z) & (P[:, 2] <= hi_z)
    Q = P[keep]
    ix = np.floor(Q[:, 0] / cell).astype(np.int64)
    iz = np.floor(Q[:, 2] / cell).astype(np.int64)
    x0, x1, z0, z1 = int(ix.min()), int(ix.max()), int(iz.min()), int(iz.max())
    W, H = x1 - x0 + 1, z1 - z0 + 1
    key = (iz - z0) * W + (ix - x0)
    order = np.lexsort((Q[:, 1], key))
    key = key[order]
    ys = Q[order, 1]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    cols = {}
    cands = {}
    for s, e in zip(starts, ends):
        col = ys[s:e]
        cols[key[s]] = col
        out = []
        top = np.searchsorted(col, col + win, "right")
        i = 0
        while i < len(col):
            if top[i] - i >= min_count:
                out.append(float(col[i + int(pct / 100 * (top[i] - i - 1))]))
                i = top[i]
            else:
                i += 1
        cands[key[s]] = out
    # seed: the column nearest seed_xz (Chebyshev rings) holding a candidate within the step/drop
    # window of seed_y. stairs2 and atrium have no floor under camera 0 itself.
    i0 = min(W - 1, max(0, int(np.floor(seed_xz[0] / cell)) - x0))
    j0 = min(H - 1, max(0, int(np.floor(seed_xz[1] / cell)) - z0))
    seed_k, seed_h = j0 * W + i0, float(seed_y)
    for r in range(0, max(W, H)):
        found = []
        for i in range(i0 - r, i0 + r + 1):
            for j in (j0 - r, j0 + r) if r else (j0,):
                if 0 <= i < W and 0 <= j < H:
                    for h in cands.get(j * W + i, ()):
                        if -drop <= h - seed_y <= step:
                            found.append((abs(h - seed_y), j * W + i, h))
        for j in range(j0 - r + 1, j0 + r):
            for i in (i0 - r, i0 + r):
                if 0 <= i < W and 0 <= j < H:
                    for h in cands.get(j * W + i, ()):
                        if -drop <= h - seed_y <= step:
                            found.append((abs(h - seed_y), j * W + i, h))
        if found:
            _, seed_k, seed_h = min(found)
            break
    floor = np.full(W * H, np.nan, np.float32)
    seen = np.zeros(W * H, bool)
    q = deque([(seed_k, seed_h)])
    while q:
        k, carry = q.popleft()
        if seen[k]:
            continue
        seen[k] = True
        best = None
        for h in cands.get(k, ()):
            if -drop <= h - carry <= step and (best is None or abs(h - carry) < abs(best - carry)):
                best = h
        if best is not None:
            floor[k] = best
            carry = best
        i, j = k % W, k // W
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < W and 0 <= nj < H and not seen[nj * W + ni]:
                q.append((nj * W + ni, carry))
    # distance (cells, 4-connected) to the nearest floor-known column, and that column's floor
    dist = np.full(W * H, 65535, np.int64)
    ref = np.full(W * H, seed_h, np.float32)
    known = np.flatnonzero(np.isfinite(floor))
    q = deque(int(k) for k in known)
    dist[known] = 0
    ref[known] = floor[known]
    while q:
        k = q.popleft()
        i, j = k % W, k // W
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < W and 0 <= nj < H:
                nk = nj * W + ni
                if dist[nk] == 65535:
                    dist[nk] = dist[k] + 1
                    ref[nk] = ref[k]
                    q.append(nk)
    blocked = np.zeros(W * H, np.uint8)
    counts = np.zeros(W * H, np.int32)
    # the column's solid bins: bit b set when >= bin_count opaque splats sit in [y0 + b*binh, y0 + (b+1)*binh).
    # The viewer tests the body band at the WALKER's own height against these, so a ceiling only
    # blocks a walker whose floor is high enough for it to be in his band (an escalator's steps are
    # accepted as floor, and a single flag measured against that floor blocked the aisle beside it).
    binh = bin_stature * st
    y0 = seed_y - below_m * upm
    solid = np.zeros(W * H, np.int64)
    for k, col in cols.items():
        counts[k] = len(col)
        f = ref[k]
        blocked[k] = (
            int(((col >= f + band_lo * st) & (col <= f + band_hi * st)).sum()) >= block_count
        )
        b = np.floor((col - y0) / binh).astype(np.int64)
        b = b[(b >= 0) & (b < 32)]
        if len(b):
            bc = np.bincount(b, minlength=32)
            solid[k] = int(sum(1 << i for i in np.flatnonzero(bc >= bin_count)))
    # inside = the closing of the floor-known set by close_m (dilate, then erode): the space BETWEEN
    # known patches counts, the space beyond them does not. stairs2's start has floor 0.8 m behind
    # it and the flight 1.6 m ahead, and nothing under it.
    rc = int(round(close_m * upm / cell))
    D = dist <= rc
    far = np.full(W * H, 65535, np.int64)
    q = deque()
    for k in range(W * H):
        i, j = k % W, k // W
        if not D[k] or i == 0 or j == 0 or i == W - 1 or j == H - 1:
            far[k] = 0
            q.append(k)
    while q:
        k = q.popleft()
        i, j = k % W, k // W
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < W and 0 <= nj < H:
                nk = nj * W + ni
                if far[nk] == 65535:
                    far[nk] = far[k] + 1
                    q.append(nk)
    inside = ((far > rc) | (dist <= int(round(hull_m * upm / cell)))).astype(np.uint8)
    return dict(
        cell=cell,
        originX=x0 * cell,
        originZ=z0 * cell,
        width=W,
        height=H,
        floor=floor.reshape(H, W),
        blocked=blocked.reshape(H, W),
        counts=counts.reshape(H, W),
        dist=dist.reshape(H, W),
        hullCells=int(round(hull_m * upm / cell)),
        inside=inside.reshape(H, W),
        solid=solid.reshape(H, W),
        solidY0=float(y0),
        solidBin=float(binh),
        solidBins=32,
        upm=upm,
        stature=st,
        seedY=seed_y,
        seedCell=[seed_k % W, seed_k // W],
        seedFloor=seed_h,
        reached=int(seen.sum()),
    )


def ransac_walls(P, n, d, max_walls, tol, min_inliers, iters, rng, floor_pts):
    """Largest vertical planes above the floor. A wall is vertical when its normal is within
    ~6 deg of the floor plane (|normal . n| < 0.1).

    Two tests separate a wall from a person Marble baked into the world as static junk (a standing
    body is 20-30k coplanar splats at 3 cm tolerance, and RANSAC found four of them inside the
    elevator corridor before these existed): the inliers must form a SHEET, at least 1 u along and
    0.8 u tall when binned at 0.1 u, and the floor must lie on ONE side of the plane only, since a
    wall bounds the floor and a body stands on it."""
    above = P[(P @ n - d) > 0.25]
    walls = []
    pts = above
    rejected = []
    for _ in range(max_walls * 4):
        if len(pts) < min_inliers or len(walls) >= max_walls:
            break
        best = None
        for _ in range(iters):
            i = rng.choice(len(pts), 3, replace=False)
            a, b, c = pts[i]
            nn = np.cross(b - a, c - a)
            ln = np.linalg.norm(nn)
            if ln < 1e-9:
                continue
            nn /= ln
            if abs(nn @ n) > 0.1:
                continue
            dd = nn @ a
            cnt = int((np.abs(pts @ nn - dd) < tol).sum())
            if best is None or cnt > best[0]:
                best = (cnt, nn, dd)
        if best is None or best[0] < min_inliers:
            break
        cnt, nn, dd = best
        inl = np.abs(pts @ nn - dd) < tol
        Q = pts[inl]
        cm = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - cm, full_matrices=False)
        nn = vt[2]
        nn -= (nn @ n) * n
        nn = unit(nn)  # re-fit, forced vertical
        dd = float(nn @ cm)
        inl = np.abs(pts @ nn - dd) < tol
        Q = pts[inl]
        along = unit(np.cross(n, nn))
        s = Q @ along
        h = Q @ n - d
        # sheet test: occupied 0.1 u bins on the plane, and their extent
        bs = np.floor(s / 0.1).astype(np.int64)
        bh = np.floor(h / 0.1).astype(np.int64)
        occ = np.unique(bs * 100000 + bh)
        ubs = np.unique(bs)
        ubh = np.unique(bh)
        # a bin column counts only when it holds enough of the sheet, so a stray splat a metre
        # away does not stretch the extent
        cols = np.bincount(bs - bs.min())
        good_cols = np.flatnonzero(cols > 0.2 * cols.max()) + bs.min()
        rows = np.bincount(bh - bh.min())
        good_rows = np.flatnonzero(rows > 0.2 * rows.max()) + bh.min()
        span_along = 0.1 * (good_cols.max() - good_cols.min() + 1)
        span_up = 0.1 * (good_rows.max() - good_rows.min() + 1)
        # one-sided test: floor cells within 0.5 u of the plane, along the sheet's own span only
        # (a doorway in a wall leaves some floor behind it; a body has floor all round it)
        fd = floor_pts @ nn - dd
        fs = floor_pts @ along
        near = (
            (np.abs(fd) < 0.5) & (fs >= 0.1 * good_cols.min()) & (fs <= 0.1 * (good_cols.max() + 1))
        )
        pos = int((fd[near] > 0.05).sum())
        neg = int((fd[near] < -0.05).sum())
        one_sided = min(pos, neg) <= 0.35 * max(pos, neg, 1)
        rec = dict(
            normal=nn.tolist(),
            d=dd,
            inliers=int(inl.sum()),
            along=along.tolist(),
            spanAlong=float(span_along),
            spanUp=float(span_up),
            occupiedBins=int(len(occ)),
            floorCellsEachSide=[neg, pos],
            extentAlong=[float(0.1 * good_cols.min()), float(0.1 * (good_cols.max() + 1))],
            extentUp=[float(0.1 * good_rows.min()), float(0.1 * (good_rows.max() + 1))],
            centre=cm.tolist(),
        )
        if span_along >= 1.0 and span_up >= 0.8 and one_sided:
            # the floor lies on the -normal side: flip so the normal points INTO the room
            if neg < pos:
                rec["normal"] = (-nn).tolist()
                rec["d"] = -dd
                rec["along"] = (-along).tolist()
                rec["extentAlong"] = [-rec["extentAlong"][1], -rec["extentAlong"][0]]
            walls.append(rec)
        else:
            rec["rejected"] = (
                "not a sheet"
                if not (span_along >= 1.0 and span_up >= 0.8)
                else "floor on both sides (a body standing on the floor, not a wall)"
            )
            rejected.append(rec)
        pts = pts[~inl]
    return walls, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, help="public/worlds/<world>")
    ap.add_argument("--spz", required=True)
    ap.add_argument("--cell", type=float, default=0.10, help="height-field cell, world units")
    ap.add_argument("--band", type=float, default=0.25, help="splats within this of the plane vote")
    ap.add_argument("--min-count", type=int, default=25)
    # the median, not fourd.html's 10th percentile: that one was chosen for snapping feet and reads
    # 15-25 cm low wherever Marble leaves sub-floor fuzz (share/elevator-status.md). A body that
    # comes to rest wants the surface most splats agree on.
    ap.add_argument("--percentile", type=float, default=50.0)
    ap.add_argument("--min-alpha", type=float, default=0.5)
    ap.add_argument("--walls", type=int, default=6)
    ap.add_argument("--wall-tol", type=float, default=0.03)
    ap.add_argument("--wall-min-inliers", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    # occupancy grid (wander.collision/2, fourd.html ?walk=1). Sizes are in METRES and scaled by
    # --upm, the viewer's world units per metre (wander.upm: stature / 1.7), so the cell is 10 cm and
    # the body band is a fraction of the avatar's stature in every world regardless of its unit.
    ap.add_argument(
        "--upm",
        type=float,
        default=None,
        help="world units per metre; default 1/metresPerWorldUnit, else 1",
    )
    ap.add_argument("--grid-cell-m", type=float, default=0.10)
    ap.add_argument(
        "--grid-min-count",
        type=int,
        default=8,
        help="splats a 10 cm column needs in one slab to have a floor",
    )
    ap.add_argument(
        "--grid-block-count",
        type=int,
        default=8,
        help="splats in the body band that make a column a wall",
    )
    ap.add_argument(
        "--grid-window-m", type=float, default=0.15, help="slab height a floor surface is read from"
    )
    ap.add_argument("--grid-pct", type=float, default=10.0)
    ap.add_argument(
        "--grid-band",
        type=float,
        nargs=2,
        default=(0.2, 1.1),
        help="body band, fractions of stature above the floor",
    )
    ap.add_argument(
        "--grid-step-up",
        type=float,
        default=0.2,
        help="one stair, as a fraction of stature (fourd.html WALK_STEP_UP)",
    )
    ap.add_argument(
        "--grid-drop-m",
        type=float,
        default=0.6,
        help="deeper than this is a pit, not a step down (fourd.html WALK_DROP_MAX_M)",
    )
    ap.add_argument(
        "--grid-hull-m",
        type=float,
        default=1.0,
        help="how far from known floor an unknown column is still walked over (fourd.html WALK_HULL_M)",
    )
    ap.add_argument(
        "--grid-extent-m",
        type=float,
        default=20.0,
        help="half-extent of the grid about camera 0 (fourd.html WALK_EXTENT_M)",
    )
    ap.add_argument(
        "--grid-close-m",
        type=float,
        default=2.0,
        help="closing radius: gaps up to twice this between known floor count as inside (fourd.html WALK_CLOSE_M)",
    )
    ap.add_argument(
        "--grid-below-m",
        type=float,
        default=0.5,
        help="solid bins start this far under the seed floor",
    )
    ap.add_argument(
        "--grid-bin-stature", type=float, default=0.1, help="solid bin height, fraction of stature"
    )
    ap.add_argument(
        "--grid-bin-count",
        type=int,
        default=15,
        help="splats a bin needs to be solid (a real rail column holds hundreds per bin; ceiling fuzz and glass 6-22 in a whole band)",
    )
    ap.add_argument(
        "--seed-y",
        type=float,
        default=None,
        help="floor height under camera 0 to flood from; default the lowest strong y layer. "
        "Pass the preset floor (wander.floorY) so the walker starts on the surface the person is placed on",
    )
    a = ap.parse_args()

    wdir = os.path.join(ROOT, "public", "worlds", a.world)
    xyz, alpha = read_spz_xyz(a.spz)
    P = xyz[alpha > a.min_alpha].astype(np.float64)
    P[:, 1] *= -1
    P[:, 2] *= -1  # Marble OpenCV -> viewer (Rx(pi))

    mpu, mpu_src = None, None
    for cand, key in (("placement.json", "metresPerWorldUnit"), ("people.json", None)):
        p = os.path.join(wdir, cand)
        if os.path.exists(p):
            j = json.load(open(p))
            v = j.get(key) if key else j.get("floorFit", {}).get("metresPerWorldUnit")
            if v:
                mpu, mpu_src = float(v), cand
                break

    n, d, ninl = fit_floor(P)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n[1])))))
    hf = height_field(P, n, d, a.cell, a.band, a.min_count, a.percentile)
    rng = np.random.default_rng(a.seed)
    # floor cell centres in world space, for the one-sided test
    u_ax, w_ax = np.array(hf["axisU"]), np.array(hf["axisW"])
    jj, ii = np.nonzero(np.isfinite(hf["heights"]))
    floor_pts = (
        np.outer(hf["originU"] + (ii + 0.5) * a.cell, u_ax)
        + np.outer(hf["originW"] + (jj + 0.5) * a.cell, w_ax)
        + d * n
    )
    walls, rejected = ransac_walls(
        P, n, d, a.walls, a.wall_tol, a.wall_min_inliers, 400, rng, floor_pts
    )

    filled = int(np.isfinite(hf["heights"]).sum())
    upm = a.upm or (1.0 / mpu if mpu else 1.0)
    # The flood seed is the LOWEST strong horizontal layer of the clipped cloud, not the dominant
    # plane: on the lobby and the atrium the ceiling holds more opaque splats than the floor
    # (fit_floor lands on +0.73 / +0.48 there against viewer floors of -0.55 / -0.90). --seed-y
    # overrides it with the viewer's own floorY, which is what the person is placed on.
    lo_x, hi_x = np.percentile(P[:, 0], [0.2, 99.8])
    lo_z, hi_z = np.percentile(P[:, 2], [0.2, 99.8])
    Qy = P[(P[:, 0] >= lo_x) & (P[:, 0] <= hi_x) & (P[:, 2] >= lo_z) & (P[:, 2] <= hi_z), 1]
    binh = 0.05 * upm
    h, e = np.histogram(Qy, bins=np.arange(Qy.min(), Qy.max() + binh, binh))
    seed_y = float(d / n[1])
    for i in range(1, len(h) - 1):
        if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] >= 0.2 * h.max():
            seed_y = float(0.5 * (e[i] + e[i + 1]))
            break
    if a.seed_y is not None:
        seed_y = a.seed_y
    og = occupancy_grid(
        P,
        seed_y,
        upm,
        a.grid_cell_m,
        1.7,
        a.grid_min_count,
        a.grid_block_count,
        a.grid_window_m,
        a.grid_pct,
        a.grid_band[0],
        a.grid_band[1],
        a.grid_step_up,
        a.grid_drop_m,
        a.grid_hull_m,
        a.grid_extent_m,
        a.grid_close_m,
        a.grid_below_m,
        a.grid_bin_stature,
        a.grid_bin_count,
    )
    og_filled = int(np.isfinite(og["floor"]).sum())
    og_blocked = int(og["blocked"].sum())
    out = dict(
        schema="wander.collision/2",
        world=a.world,
        spz=os.path.relpath(a.spz, ROOT),
        frame="viewer world: Marble spz after Rx(pi) (y -> -y, z -> -z), world units. Same frame as "
        "floorY, pos0 and every placed person/object in fourd.html.",
        metresPerWorldUnit=mpu,
        metresPerWorldUnitSource=mpu_src,
        splats=int(len(xyz)),
        opaqueSplats=int(len(P)),
        minAlpha=a.min_alpha,
        floor=dict(
            normal=n.tolist(),
            d=d,
            yAtOrigin=float(d / n[1]) if abs(n[1]) > 1e-6 else None,
            tiltFromYDeg=tilt,
            inliers=ninl,
            provenance="dominant near-horizontal plane of the opaque splats: y-histogram peak "
            "seed, then shrinking-band (0.20 -> 0.035 u) least squares. Measured.",
        ),
        heightField=dict(
            axisU=hf["axisU"],
            axisW=hf["axisW"],
            cell=hf["cell"],
            originU=hf["originU"],
            originW=hf["originW"],
            width=hf["width"],
            height=hf["height"],
            # height ABOVE the floor plane along its normal; null = no vote, fall back to the plane
            heights=[
                [None if not np.isfinite(v) else round(float(v), 4) for v in row]
                for row in hf["heights"]
            ],
            counts=hf["counts"].tolist(),
            filledCells=filled,
            provenance=f"{a.percentile:.0f}th percentile of opaque splat height within +-{a.band} u of the "
            f"floor plane, per {a.cell} u cell, cells with < {a.min_count} splats left null. "
            "Measured; the percentile is a choice (fourd.html buildFloorMap uses the same one).",
        ),
        occupancy=dict(
            cell=og["cell"],
            originX=og["originX"],
            originZ=og["originZ"],
            width=og["width"],
            height=og["height"],
            upm=upm,
            upmSource="--upm" if a.upm else ("1/metresPerWorldUnit" if mpu else "assumed 1"),
            stature=og["stature"],
            seedY=og["seedY"],
            seedSource="--seed-y" if a.seed_y is not None else "lowest strong y layer",
            reachedCells=og["reached"],
            seedCell=og["seedCell"],
            seedFloor=round(float(og["seedFloor"]), 4),
            hullCells=og["hullCells"],
            # cells to the nearest floor-known column (65535 = none anywhere); inside = within hullCells of one
            # or in the closing of the known set; solid = 32-bit column occupancy from solidY0 in solidBin steps
            dist=[[int(v) for v in row] for row in og["dist"]],
            inside=[[int(v) for v in row] for row in og["inside"]],
            solid=[[int(v) for v in row] for row in og["solid"]],
            solidY0=og["solidY0"],
            solidBin=og["solidBin"],
            solidBins=og["solidBins"],
            closeCells=int(round(a.grid_close_m * upm / og["cell"])),
            # rows are z (index j from originZ), columns x (index i from originX); world y of the
            # walkable floor, null = no dense floor found; blocked = dense splats in the body band
            floor=[
                [None if not np.isfinite(v) else round(float(v), 4) for v in row]
                for row in og["floor"]
            ],
            blocked=[[int(v) for v in row] for row in og["blocked"]],
            filledCells=og_filled,
            blockedCells=og_blocked,
            provenance=f"axis-aligned {a.grid_cell_m} m x upm columns within {a.grid_extent_m} m of camera 0, over the 0.2..99.8 percentile extent of the "
            f"opaque splats. Candidate surfaces per column = {a.grid_pct:.0f}th percentile of each "
            f"{a.grid_window_m} m slab holding >= {a.grid_min_count} splats. floor = the candidate a flood "
            f"fill from the column under camera 0 at seedY reaches within [-{a.grid_drop_m} m, "
            f"+{a.grid_step_up} x stature] of the height carried from its neighbour, seeded at the "
            f"floor-known column nearest camera 0; null = none. dist = cells to the nearest floor-known "
            f"column; hullCells = {a.grid_hull_m} m in cells. blocked = >= {a.grid_block_count} splats within "
            f"[{a.grid_band[0]}, {a.grid_band[1]}] x 1.7 m x upm above the own floor, or the nearest known "
            f"floor where there is none. inside = within hullCells of known floor or in its closing by "
            f"{a.grid_close_m} m. solid = per-column bins of {a.grid_bin_stature} x stature from solidY0, set at >= "
            f"{a.grid_bin_count} splats; the viewer tests the body band at the walker's own height against it. "
            "Measured; the counts, band, step, hull and closing are choices.",
        ),
        walls=walls,
        rejectedPlanes=rejected,
        wallsProvenance=f"RANSAC vertical planes (|normal . floorNormal| < 0.1) on opaque splats more than "
        f"0.25 u above the floor, inlier tolerance {a.wall_tol} u, minimum {a.wall_min_inliers} "
        f"inliers, refit by SVD and forced vertical. Kept only when the inliers form a sheet "
        f"(>= 1.0 u along, >= 0.8 u tall at 0.1 u bins) AND the floor cells lie on one side "
        f"of the plane; the normal points into the room. Rejected planes are listed with "
        f"the reason. Measured; count says how much of the cloud agrees.",
        assumptions=[
            "walls are infinite planes clipped to their inlier extent; furniture and other "
            "clutter is not represented and a body will pass through it",
            "the height field is a 2.5D surface: nothing overhangs",
        ],
    )
    outp = a.out or os.path.join(wdir, "collision.json")
    json.dump(out, open(outp, "w"))
    print(
        f"floor y0 {out['floor']['yAtOrigin']:.4f} tilt {tilt:.2f} deg inliers {ninl}; "
        f"height field {hf['width']}x{hf['height']} cells, {filled} filled; {len(walls)} walls "
        f"{[w['inliers'] for w in walls]}; occupancy {og['width']}x{og['height']} @ {og['cell']:.4f} u, "
        f"{og_filled} floored ({100 * og_filled / (og['width'] * og['height']):.1f} %), {og_blocked} blocked, seed cell "
        f"{og['seedCell']} floor {og['seedFloor']:.3f} (seed y {og['seedY']:.3f}, upm {upm:.4f}); mpu {mpu} ({mpu_src}); wrote {outp} "
        f"({os.path.getsize(outp) / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
