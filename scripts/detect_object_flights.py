"""Find every free flight of a small carried/thrown object over a WHOLE clip.

`track_object_2d.py` answers "where did the object go between frames A and B, given a
seed on it". That question can only ever return one throw, and it silently returns the
first one it can chain. This answers the prior question: over the entire clip, when is
the object in the air at all?

Three stages, none of which know what the object is:

  detect   motion-compensated 3-frame differencing with the tracker's own person masks
           removed, every frame of the clip. Anything that moves against the scene and
           is not a person is a candidate.
  link     candidates are grown into tracklets by a constant-acceleration image-space
           predictor, seedless: every unconsumed candidate is tried as a start.
  classify a tracklet is a FREE span iff, after removing camera motion by chaining the
           frame-to-frame homographies across the tracklet, its stabilised image path
           fits a parabola whose curvature points DOWN, at the magnitude gravity
           predicts for the object's apparent scale, and it is not resting in a hand.

Everything the clip does not support falls out of that rule by itself: an object that is
only ever carried yields no accepted tracklet and therefore no free span, and an object
thrown five times yields five.
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_object_2d import homography, People


class SinglePerson:
    """The `People` interface over a single-person world (`person/motion.json`).

    The single-person pipeline keeps no tracks/ directory: its person masks are the run's packed
    `masks.npz` (sampled frames, bit-packed rows) when that still exists, and a dilated hull of the
    projected body joints when it does not. Either way the detector sees one track, index 0.
    """
    def __init__(self, motion_json, W, H, masks_npz=None):
        m = json.load(open(motion_json))
        self.mot = {0: {int(f['sourceIndex']): f for f in m['frames']}}
        self.src = sorted(self.mot[0])
        self.n = 1
        self.W, self.H = W, H
        self.packed = None
        if masks_npz and os.path.exists(masks_npz):
            z = np.load(masks_npz)
            self.packed, self.packed_idx = z['masks'], np.array([int(v) for v in z['indices']])
        self._cache = {}

    def bracket(self, sf):
        ks = self.src
        lo = max([i for i, k in enumerate(ks) if k <= sf], default=0)
        hi = min([i for i, k in enumerate(ks) if k >= sf], default=len(ks) - 1)
        return lo, hi

    def _mask_src(self, sf):
        # a person-sized hull dilation at 1080p costs ~1 s and is asked for four times a frame
        if sf not in self._cache:
            self._cache[sf] = np.packbits(self._mask_src_uncached(sf), axis=1)
        return np.unpackbits(self._cache[sf], axis=1)[:, :self.W]

    def _mask_src_uncached(self, sf):
        if self.packed is not None:
            i = int(np.argmin(np.abs(self.packed_idx - sf)))
            m = np.unpackbits(self.packed[i], axis=1)[:, :self.W]
            if m.shape[0] != self.H or m.shape[1] != self.W:
                m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            return m.astype(np.uint8)
        J = np.array(self.mot[0][sf]['projectedBodyJoints'], float)
        J = J[np.isfinite(J).all(1)]
        m = np.zeros((self.H, self.W), np.uint8)
        if len(J) < 3:
            return m
        hull = cv2.convexHull(J.astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
        # joints sit inside the body; pad by a fraction of the figure's height to cover clothing
        r = int(max(15, 0.12 * (J[:, 1].max() - J[:, 1].min())))
        q = 4  # dilate at quarter resolution; the hull is coarse anyway
        small = cv2.resize(m, (self.W // q, self.H // q), interpolation=cv2.INTER_NEAREST)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (r // q) + 1, 2 * (r // q) + 1))
        return cv2.resize(cv2.dilate(small, k), (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def mask(self, sf, dilate):
        lo, hi = self.bracket(sf)
        acc = self._mask_src(self.src[lo]) | self._mask_src(self.src[hi])
        acc = acc * 255
        if dilate:
            acc = cv2.dilate(acc, np.ones((dilate, dilate), np.uint8))
        return acc

    def joints(self, sf):
        lo, hi = self.bracket(sf)
        klo, khi = self.src[lo], self.src[hi]
        w = 0.0 if khi == klo else (sf - klo) / (khi - klo)
        a = np.array(self.mot[0][klo]['projectedBodyJoints'])
        b = np.array(self.mot[0][khi]['projectedBodyJoints'])
        return {0: a * (1 - w) + b * w}


def load_people(path, clip, masks=None):
    """tracks.json -> People; a single-person motion.json -> SinglePerson sized from the clip."""
    d = json.load(open(path))
    if 'trackCount' in d:
        return People(path)
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return SinglePerson(path, W, H, masks)


def load_mpu(path):
    """metres per world unit from whichever world file carries it: people.json (floorFit),
    placement.json / collision.json (metresPerWorldUnit) or framealign.json (…Avatar)."""
    d = json.load(open(path))
    for v in (d.get('floorFit', {}).get('metresPerWorldUnit'), d.get('metresPerWorldUnit'),
              d.get('metresPerWorldUnitAvatar')):
        if v:
            return float(v)
    h = d.get('sharedPlacement', {}).get('bodyHeightUnits')
    if h:
        # no floor fit yet: the stature prior the people package itself was placed with. Ten
        # per cent on |g| moves the ballistic reprojection by far less than the gate
        print('no floor fit in %s; |g| from the 1.70 m stature prior over bodyHeightUnits %.3f' % (path, h))
        return 1.70 / float(h)
    raise SystemExit('no metresPerWorldUnit in ' + path)


# ---------------------------------------------------------------- stage: detect
def detect(clip, P, first, last, dilate, min_area, max_area, progress=True):
    """-> (candidates per source frame, H[f] mapping frame f-1 into frame f)."""
    cap = cv2.VideoCapture(clip)
    W, H = P.W, P.H
    det, Hs = {}, {}

    # sequential read is far cheaper than seeking per frame, and a three-frame ring is all the
    # differencing needs: the whole clip is 1.8 GB at 1080p and does not fit beside everything else
    start = max(0, first - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ring = {}
    fi = start
    while fi <= last + 1:
        ok, fr = cap.read()
        if not ok:
            break
        ring[fi] = fr
        for k in [k for k in ring if k < fi - 2]:
            del ring[k]
        fi += 1
        sf = fi - 2
        if sf < first or sf > last:
            continue
        cur, prev, nxt = ring.get(sf), ring.get(sf - 1), ring.get(sf + 1)
        if cur is None or prev is None or nxt is None:
            continue
        pm = P.mask(sf, 40)
        bgm = np.where(pm > 0, 0, 255).astype(np.uint8)
        Hp = homography(prev, cur, bgm)
        Hn = homography(nxt, cur, bgm)
        Hs[sf] = Hp
        wp = cv2.warpPerspective(prev, Hp, (W, H))
        wn = cv2.warpPerspective(nxt, Hn, (W, H))
        d = np.minimum(cv2.absdiff(cur, wp).max(2), cv2.absdiff(cur, wn).max(2))
        d = cv2.GaussianBlur(d, (0, 0), 1.5)
        km = np.where(P.mask(sf, dilate) > 0, 0, 255).astype(np.uint8)
        km[:, :25] = 0; km[:, -25:] = 0; km[:25] = 0; km[-25:] = 0
        dm = np.where(km > 0, d, 0)
        nz = dm[dm > 0]
        thr = max(9.0, float(np.percentile(nz, 99.6)) * 0.45) if nz.size else 255.0
        _, bw = cv2.threshold(dm, thr, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        cands = []
        for c in range(1, n):
            a = float(stats[c, cv2.CC_STAT_AREA])
            if a < min_area or a > max_area:
                continue
            sel = lab == c
            wgt = dm[sel].astype(np.float64)
            ys, xs = np.nonzero(sel)
            cands.append(dict(x=float((xs * wgt).sum() / wgt.sum()), y=float((ys * wgt).sum() / wgt.sum()),
                              area=a, strength=float(wgt.mean()),
                              bx=[int(stats[c, cv2.CC_STAT_LEFT]), int(stats[c, cv2.CC_STAT_TOP]),
                                  int(stats[c, cv2.CC_STAT_WIDTH]), int(stats[c, cv2.CC_STAT_HEIGHT])]))
        det[sf] = cands
        if progress and sf % 20 == 0:
            print('  detect f%d  %d candidates' % (sf, len(cands)), flush=True)
    cap.release()
    return det, Hs


def top_k(det, k):
    """Keep only the k strongest movers per frame.

    A free-flying object is a motion smear: it is the fastest thing in the frame that the person
    masks do not cover, so `area x contrast` ranks it at or near the top in every frame it is
    detected at all. Incidental noise (mask edges, glass reflections, homography residue) is
    plentiful but weak. This is a generic strength ranking, not a size or colour prior, and it is
    what makes seedless linking tractable: 61 blobs a frame is 60 chances to link into garbage.
    """
    out = {}
    for f, cs in det.items():
        cs = sorted(cs, key=lambda c: -c['area'] * c['strength'])[:k]
        out[f] = cs
    return out


# ------------------------------------------------------------------ stage: link
def link(det, first, last, gate, init_gate, max_gap, min_len, max_span):
    """Seedless tracklet hypotheses. Every candidate is tried as a start and nothing is
    consumed, because a greedy first-come linker over 60 blobs a frame spends the real
    object's detections on whichever noise blob happened to be seeded first. Duplicates
    are collapsed at the end; physics decides which survive, not arrival order."""
    fs = sorted(det)
    XY = {f: np.array([[c['x'], c['y']] for c in det[f]], float).reshape(-1, 2) for f in fs}

    def nearest(f, px, py, rad):
        A = XY.get(f)
        if A is None or not len(A):
            return None, None
        d = np.hypot(A[:, 0] - px, A[:, 1] - py)
        i = int(np.argmin(d))
        return (i, float(d[i])) if d[i] < rad else (None, None)

    def grow(f0, i0, f1, i1):
        hist = [(f0, *XY[f0][i0]), (f1, *XY[f1][i1])]
        pts = {f0: i0, f1: i1}
        sf, gap = f1 + 1, 0
        stop = min(last, f0 + max_span)
        while sf <= stop and gap <= max_gap:
            if len(hist) == 2:
                dt = (sf - hist[-1][0]) / (hist[-1][0] - hist[-2][0])
                px = hist[-1][1] + (hist[-1][1] - hist[-2][1]) * dt
                py = hist[-1][2] + (hist[-1][2] - hist[-2][2]) * dt
            else:
                A = np.array(hist[-3:], float)
                t = A[:, 0] - sf
                px = float(np.polyval(np.polyfit(t, A[:, 1], 2), 0.0))
                py = float(np.polyval(np.polyfit(t, A[:, 2], 2), 0.0))
            i, _ = nearest(sf, px, py, gate * (1 + gap))
            if i is None:
                gap += 1; sf += 1; continue
            pts[sf] = i
            hist.append((sf, *XY[sf][i]))
            gap = 0; sf += 1
        return pts

    seen, out = set(), []
    for f0 in fs:
        for i0 in range(len(XY[f0])):
            for f1 in range(f0 + 1, min(f0 + 2 + max_gap, last + 1)):
                A = XY.get(f1)
                if A is None or not len(A):
                    continue
                d = np.hypot(A[:, 0] - XY[f0][i0, 0], A[:, 1] - XY[f0][i0, 1])
                for i1 in np.nonzero(d < init_gate * (f1 - f0))[0]:
                    pts = grow(f0, i0, f1, int(i1))
                    if len(pts) < min_len:
                        continue
                    key = tuple(sorted(pts.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({f: det[f][i] for f, i in pts.items()})
    return out


# -------------------------------------------------------------- stage: classify
def stabilise(pts, Hs):
    """Map every point of a tracklet into the image frame of the tracklet's first frame.

    Hs[f] warps frame f-1 into frame f, so the inverse chain takes frame f back to f0.
    Only the tracklet's own span is chained, so homography drift stays small.
    """
    fs = sorted(pts)
    f0 = fs[0]
    M = {f0: np.eye(3)}
    for f in range(f0 + 1, fs[-1] + 1):
        Hf = Hs.get(f)
        if Hf is None:
            Hf = np.eye(3)
        M[f] = M[f - 1] @ np.linalg.inv(Hf)
    out = []
    for f in fs:
        v = M[f] @ np.array([pts[f]['x'], pts[f]['y'], 1.0])
        out.append([f, v[0] / v[2], v[1] / v[2]])
    return np.array(out)


def _fit_resid(t, x, y, i, j):
    tt, xx, yy = t[i:j], x[i:j], y[i:j]
    cx = np.polyfit(tt, xx, 2); cy = np.polyfit(tt, yy, 2)
    r = float(np.sqrt(((xx - np.polyval(cx, tt)) ** 2 + (yy - np.polyval(cy, tt)) ** 2).mean()))
    return r, float(2 * cy[0])


def prefilter(tracklets, Hs, P, fps, joint, max_resid_px, min_travel_px, hand_px, min_len):
    """Cheap 2D screen before the expensive 3D fit.

    A tracklet is a chain of "something moved here"; only PART of it is usually a flight, because
    the linker happily carries on through whatever blob is nearest once the object lands. So this
    does not ask whether the whole chain is a parabola - that question fails on every real throw.
    It finds the LONGEST contiguous sub-window of the chain that is one, after chaining the
    frame-to-frame homographies to remove camera motion. Curvature must point DOWN, the window must
    actually travel, and it must be clear of every tracked hand.
    """
    seen, keep = set(), []
    for pts in tracklets:
        if len(pts) < min_len:
            continue
        S = stabilise(pts, Hs)
        f, x, y = S[:, 0], S[:, 1], S[:, 2]
        t = (f - f[0]) / fps
        n = len(f)
        wins = []
        for i0 in range(0, n - min_len + 1):
            i, j = i0, i0 + min_len
            r, g = _fit_resid(t, x, y, i, j)
            if r > max_resid_px or g <= 0:
                continue
            grew = True
            while grew:
                grew = False
                if j < n:
                    r2, g2 = _fit_resid(t, x, y, i, j + 1)
                    if r2 <= max_resid_px and g2 > 0:
                        j, grew = j + 1, True
                if i > 0:
                    r2, g2 = _fit_resid(t, x, y, i - 1, j)
                    if r2 <= max_resid_px and g2 > 0:
                        i, grew = i - 1, True
            wins.append((i, j))
        for i, j in set(wins):
            fs = [int(v) for v in f[i:j]]
            key = tuple(fs) + tuple(round(v, 1) for v in x[i:j])
            if key in seen:
                continue
            seen.add(key)
            r, g = _fit_resid(t, x, y, i, j)
            travel = float(np.hypot(x[j - 1] - x[i], y[j - 1] - y[i]))
            if travel < min_travel_px:
                continue
            sub = {k: pts[k] for k in fs}
            dh = []
            for k in sub:
                J = P.joints(k)
                dh.append(min((float(np.hypot(J[tr][joint][0] - sub[k]['x'], J[tr][joint][1] - sub[k]['y']))
                               for tr in J), default=float('inf')))
            if float(np.median(dh)) < hand_px:
                continue
            keep.append(dict(pts=sub, parabolaResidPx=r, imageGravityPxPerS2=g, travelPx=travel,
                             handDistPx=[float(v) for v in dh], medianHandDistPx=float(np.median(dh))))
    return keep


def dedupe(cands):
    """Many tracklets yield the same free span; keep the cleanest per (first, last) frame pair."""
    best = {}
    for c in cands:
        f = sorted(c['pts'])
        k = (f[0], f[-1])
        if k not in best or (len(c['pts']), -c['parabolaResidPx']) > (len(best[k]['pts']), -best[k]['parabolaResidPx']):
            best[k] = c
    return list(best.values())


def ballistic_screen(cands, cameras, people, fps, max_reproj_px, max_depth_m=30.0, max_speed_ms=25.0):
    """The real test: does ONE 6-parameter p0 + v0 t + g t^2/2, with the clip's own |g| and its
    own SfM cameras, reproject onto these observations? Nothing about a bottle, a person or a
    number of throws enters here; it is the definition of free flight."""
    import lift_object_3d as L
    C = L.load_cameras(cameras)
    mpu = load_mpu(people)
    gv = np.array([0.0, -9.80665 / mpu, 0.0])
    from scipy.optimize import least_squares
    out = []
    for c in cands:
        fl = np.array(sorted(c['pts']), float)
        uv = np.array([[c['pts'][int(f)]['x'], c['pts'][int(f)]['y']] for f in fl])
        ts = (fl - fl[0]) / fps

        # cameras are fixed per frame: resolve the Slerp ONCE, not once per residual evaluation
        Rs = np.array([L.cam_at(C, f)[0] for f in fl])
        Ts = np.array([L.cam_at(C, f)[1] for f in fl])
        K = C['K']

        def proj_all(P):
            loc = np.einsum('nij,ni->nj', Rs, P - Ts) @ L.CV
            uvz = loc @ K.T
            return uvz[:, :2] / uvz[:, 2:3], uvz[:, 2] / 1.0, loc[:, 2]

        def resid(p):
            P = p[:3] + np.outer(ts, p[3:6]) + 0.5 * np.outer(ts ** 2, gv)
            return (proj_all(P)[0] - uv).ravel()

        R0, t0 = Rs[0], Ts[0]
        ray = np.linalg.inv(K) @ np.array([uv[0, 0], uv[0, 1], 1.0])
        best = None
        for depth in (1.5, 3.5, 7.0):
            p_init = np.concatenate([t0 + (R0 @ L.CV) @ (ray * depth), np.zeros(3)])
            try:
                s = least_squares(resid, p_init, method='lm', xtol=1e-11, ftol=1e-11, max_nfev=400)
            except Exception:
                continue
            rp = np.linalg.norm(s.fun.reshape(-1, 2), axis=1)
            r = float(np.sqrt((rp ** 2).mean()))
            P0 = s.x[:3] + np.outer(ts, s.x[3:6]) + 0.5 * np.outer(ts ** 2, gv)
            dep = float(np.median(proj_all(P0)[2]) * mpu)
            if best is None or r < best[0]:
                best = (r, float(rp.max()), s.x, dep)
        if best is None:
            continue
        speed = float(np.linalg.norm(best[2][3:6]) * mpu)
        c = dict(c, ballisticReprojPxRms=best[0], ballisticReprojPxMax=best[1], speedMs=speed,
                 depthM=float(best[3]))
        # a fit that ran off to infinite depth reprojects perfectly and means nothing: a monocular
        # parabola is degenerate along the ray unless gravity actually bends it in the image
        if best[0] <= max_reproj_px and 0.2 <= best[3] <= max_depth_m and speed <= max_speed_ms:
            out.append(c)
    return out


def body_screen(cands, P, dilate, max_overlap, max_aspect, min_fill, min_obs):
    """Two things a parabola-with-gravity cannot tell apart from a throw, measured on the elevator
    clip where the detector as first written accepted them (share/objects-general/STATUS.md):

    a body part the person mask under-covers. A swinging shoe traces a downward arc at walking
    speed and reprojects as a ballistic curve to a few px. It is, however, a piece of the person:
    its difference blob straddles the dilated person mask (median 11-22 % of its bbox inside, against
    0 % for every real throw). A thrown object that overlaps the mask is masked out, so a blob that
    overlaps it over its span is a limb, not something in the air.

    a parallax edge. The homography aligns one plane; a wall edge off that plane slides slowly in
    the stabilised frame and a 7-frame slide fits a parabola. Its blob is the edge itself: bbox
    aspect 4.2 median, against 1.2-1.6 for a small object smeared along its velocity. A DIAGONAL
    edge (a chrome chair leg in the lobby clip) has a square bbox but fills 24 % of it, against
    52-66 % for a real throw's blob, so fill ratio is the second half of the same test.

    And a span too short to mean anything. Six observations over 0.2 s fit a downward parabola by
    accident (a wind-blown scrap of litter in the hpwide clip did, at 10 m/s); `lift_object_3d.py`
    already refuses fewer than ten for that reason, so the detector applies the same gate instead
    of passing on what the lift will drop. The linker still grows short tracklets so that an arc
    split by a detection gap can be merged back to ten or more before this is asked.
    """
    out = []
    for c in cands:
        ov, asp, fill = [], [], []
        for f, p in c['pts'].items():
            bx = p['bx']
            asp.append(max(bx[2], bx[3]) / max(1.0, min(bx[2], bx[3])))
            fill.append(p.get('area', 0.0) / max(1.0, bx[2] * bx[3]))
            m = P.mask(int(f), dilate)
            sub = m[bx[1]:bx[1] + bx[3], bx[0]:bx[0] + bx[2]]
            ov.append(float((sub > 0).mean()) if sub.size else 0.0)
        c = dict(c, medianMaskOverlap=float(np.median(ov)), medianBboxAspect=float(np.median(asp)),
                 medianFill=float(np.median(fill)))
        c['rejected'] = None
        if c['medianMaskOverlap'] > max_overlap:
            c['rejected'] = 'body part: %.0f%% of the blob bbox lies inside the person mask (> %.0f%%)' \
                            % (100 * c['medianMaskOverlap'], 100 * max_overlap)
        elif c['medianBboxAspect'] > max_aspect:
            c['rejected'] = 'edge, not a compact mover: bbox aspect %.1f (> %.1f)' % (c['medianBboxAspect'], max_aspect)
        elif c['medianFill'] < min_fill:
            c['rejected'] = 'edge, not a compact mover: blob fills %.0f%% of its bbox (< %.0f%%)' \
                            % (100 * c['medianFill'], 100 * min_fill)
        elif len(c['pts']) < min_obs:
            c['rejected'] = '%d observations (< %d): too short to separate a parabola from noise' \
                            % (len(c['pts']), min_obs)
        out.append(c)
    return out


def merge_spans(cands, cameras, people, fps, max_reproj_px, max_gap, max_depth_m, max_speed_ms):
    """Join adjacent accepted spans that are one flight the detector lost the middle of.

    A gap of a few frames - the object crossing a dark background, or passing in front of a
    person and being masked out - splits one arc into two. Two spans are the same flight iff
    their UNION still passes the same ballistic reprojection test, which is the same rule that
    accepted them, so nothing is merged on proximity alone.
    """
    cands = sorted(cands, key=lambda c: min(c['pts']))
    changed = True
    while changed and len(cands) > 1:
        changed = False
        for i in range(len(cands) - 1):
            a, b = cands[i], cands[i + 1]
            fa, fb = max(a['pts']), min(b['pts'])
            if not (0 < fb - fa <= max_gap):
                continue
            u = dict(a['pts']); u.update(b['pts'])
            cand = dict(a, pts=u,
                        parabolaResidPx=max(a['parabolaResidPx'], b['parabolaResidPx']),
                        imageGravityPxPerS2=0.5 * (a['imageGravityPxPerS2'] + b['imageGravityPxPerS2']),
                        travelPx=a['travelPx'] + b['travelPx'],
                        handDistPx=a['handDistPx'] + b['handDistPx'],
                        medianHandDistPx=float(np.median(a['handDistPx'] + b['handDistPx'])))
            ok = ballistic_screen([cand], cameras, people, fps, max_reproj_px, max_depth_m, max_speed_ms)
            if ok:
                cands[i:i + 2] = ok
                changed = True
                break
    return cands


def nms(cands, key):
    """One flight per span of time: keep the best-scoring tracklet, drop anything overlapping it."""
    cands = sorted(cands, key=key)
    taken, out = [], []
    for c in cands:
        f = sorted(c['pts'])
        if any(not (f[-1] < a or f[0] > b) for a, b in taken):
            continue
        taken.append((f[0], f[-1]))
        out.append(c)
    return out


def record(c, fps):
    pts = c['pts']
    fs = sorted(pts)
    return dict(frames=[int(f) for f in fs], nObs=len(fs),
                px=[[float(pts[f]['x']), float(pts[f]['y'])] for f in fs],
                areaPx=[float(pts[f].get('area', 0)) for f in fs],
                sizePx=[int(max(pts[f]['bx'][2], pts[f]['bx'][3])) for f in fs],
                parabolaResidPx=c['parabolaResidPx'], imageGravityPxPerS2=c['imageGravityPxPerS2'],
                travelPx=c['travelPx'], medianHandDistPx=c['medianHandDistPx'],
                handDistPx=c['handDistPx'],
                ballisticReprojPxRms=c.get('ballisticReprojPxRms'),
                ballisticReprojPxMax=c.get('ballisticReprojPxMax'),
                speedMs=c.get('speedMs'), depthM=c.get('depthM'),
                bboxPx=[[int(v) for v in pts[f]['bx']] for f in fs],
                medianMaskOverlap=c.get('medianMaskOverlap'), medianBboxAspect=c.get('medianBboxAspect'),
                medianFill=c.get('medianFill'),
                rejected=c.get('rejected'), free=c.get('rejected') is None)


def write_crops(clip, recs, out_dir, pad=1.5, min_side=48):
    """Cut every observation of every flight out of the source frames, bbox dilated by `pad`, and
    lay them out as one contact sheet per flight. This is what makes a detection inspectable
    without a viewer, and it is the raw material the appearance stage describes."""
    os.makedirs(out_dir, exist_ok=True)
    want = {}
    for i, r in enumerate(recs):
        for f, (x, y), s in zip(r['frames'], r['px'], r['sizePx']):
            want.setdefault(int(f), []).append((i, x, y, s))
    if not want:
        return
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tiles = {i: [] for i in range(len(recs))}
    fs = sorted(want)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fs[0])
    fi = fs[0]
    while fi <= fs[-1]:
        ok, fr = cap.read()
        if not ok:
            break
        for i, x, y, s in want.get(fi, []):
            h = max(min_side, int(s * pad))
            x0, y0 = int(max(0, x - h / 2)), int(max(0, y - h / 2))
            x1, y1 = int(min(W, x + h / 2)), int(min(H, y + h / 2))
            crop = fr[y0:y1, x0:x1]
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            cv2.imwrite(os.path.join(out_dir, 'flight%02d_f%04d.png' % (i, fi)), crop)
            recs[i].setdefault('crops', []).append(dict(frame=fi, box=[x0, y0, x1, y1],
                                                        laplacianVar=sharp,
                                                        file='flight%02d_f%04d.png' % (i, fi)))
            tiles[i].append((fi, crop))
        fi += 1
    cap.release()
    side = 96
    for i, ts in tiles.items():
        if not ts:
            continue
        row = []
        for f, c in ts:
            t = cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA)
            cv2.putText(t, 'f%d' % f, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            row.append(t)
        n = len(row)
        cols = min(n, 12)
        rows = (n + cols - 1) // cols
        sheet = np.zeros((rows * side, cols * side, 3), np.uint8)
        for k, t in enumerate(row):
            r, c = divmod(k, cols)
            sheet[r * side:(r + 1) * side, c * side:(c + 1) * side] = t
        cv2.imwrite(os.path.join(out_dir, 'flight%02d-sheet.png' % i), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--tracks', required=True,
                    help='tracks.json of a multi-person run, or person/motion.json of a single-person world')
    ap.add_argument('--masks', default=None,
                    help='single-person runs only: the run\'s masks.npz; a joint hull is used without it')
    ap.add_argument('--cameras', required=True)
    ap.add_argument('--people', required=True,
                    help='any world json carrying the metre scale: people.json, placement.json, ...')
    ap.add_argument('--first', type=int, default=None)
    ap.add_argument('--last', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--crops', default=None, help='directory for per-observation crops and contact sheets')
    ap.add_argument('--cache', default=None, help='npz of stage-A detections, reused if present')
    ap.add_argument('--joint', type=int, default=21)
    ap.add_argument('--fps', type=float, default=None, help='default: read from the clip')
    ap.add_argument('--dilate', type=int, default=9)
    ap.add_argument('--min-area', type=float, default=20.0)
    ap.add_argument('--max-area', type=float, default=6000.0)
    ap.add_argument('--gate', type=float, default=45.0)
    ap.add_argument('--init-gate', type=float, default=60.0)
    ap.add_argument('--max-gap', type=int, default=2)
    ap.add_argument('--min-len', type=int, default=6)
    ap.add_argument('--top-k', type=int, default=15,
                    help='strongest movers kept per frame before linking')
    ap.add_argument('--max-span', type=int, default=60,
                    help='longest tracklet a single free flight may occupy, in frames')
    ap.add_argument('--max-resid-px', type=float, default=6.0)
    ap.add_argument('--min-travel-px', type=float, default=150.0)
    ap.add_argument('--hand-px', type=float, default=45.0)
    ap.add_argument('--max-reproj-px', type=float, default=6.0)
    ap.add_argument('--max-depth-m', type=float, default=30.0,
                    help='reject fits that escaped to infinite depth, where the monocular arc is degenerate')
    ap.add_argument('--max-speed-ms', type=float, default=25.0)
    ap.add_argument('--merge-gap', type=int, default=6,
                    help='largest detection gap two accepted spans may be joined across')
    ap.add_argument('--max-mask-overlap', type=float, default=0.05,
                    help='median fraction of the blob bbox inside the dilated person mask above which '
                         'the mover is a limb the mask under-covered')
    ap.add_argument('--max-aspect', type=float, default=3.0,
                    help='median bbox aspect above which the mover is an edge, not a compact object')
    ap.add_argument('--min-fill', type=float, default=0.35,
                    help='median blob-area / bbox-area below which the mover is a thin diagonal edge')
    ap.add_argument('--min-obs', type=int, default=10,
                    help='observations a flight needs before its parabola means anything; the same '
                         'gate lift_object_3d.py applies')
    a = ap.parse_args()

    P = load_people(a.tracks, a.clip, a.masks)
    if a.fps is None:
        cap = cv2.VideoCapture(a.clip)
        a.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
    first = a.first if a.first is not None else 1
    last = a.last if a.last is not None else int(P.src[-1]) - 1

    det = Hs = None
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        det = {int(k): v for k, v in json.loads(str(z['det'])).items()}
        Hs = {int(k): v for k, v in zip(z['hf'], z['hm'])}
        print('reused cache', a.cache, len(det), 'frames')
    if det is None:
        det, Hs = detect(a.clip, P, first, last, a.dilate, a.min_area, a.max_area)
        if a.cache:
            np.savez_compressed(a.cache, det=json.dumps({str(k): v for k, v in det.items()}),
                                hf=np.array(sorted(Hs)), hm=np.array([Hs[k] for k in sorted(Hs)]))

    tot = sum(len(v) for v in det.values())
    print('candidates: %d over %d frames (median %.1f/frame)'
          % (tot, len(det), np.median([len(v) for v in det.values()])), flush=True)

    det = top_k(det, a.top_k)
    print('after keeping the %d strongest movers per frame: %d candidates'
          % (a.top_k, sum(len(v) for v in det.values())), flush=True)
    tl = link(det, first, last, a.gate, a.init_gate, a.max_gap, a.min_len, a.max_span)
    print('tracklet hypotheses >= %d obs: %d' % (a.min_len, len(tl)), flush=True)
    pf = prefilter(tl, Hs, P, a.fps, a.joint, a.max_resid_px, a.min_travel_px, a.hand_px, a.min_len)
    print('pass the stabilised-parabola screen: %d' % len(pf), flush=True)
    pf = dedupe(pf)
    print('distinct free-span candidates after dedupe: %d' % len(pf), flush=True)
    bs = ballistic_screen(pf, a.cameras, a.people, a.fps, a.max_reproj_px, a.max_depth_m, a.max_speed_ms)
    print('pass the 3D ballistic reprojection screen (<= %.1f px): %d' % (a.max_reproj_px, len(bs)), flush=True)
    kept = nms(bs, key=lambda c: (c['ballisticReprojPxRms'] / max(len(c['pts']), 1), -len(c['pts'])))
    kept = merge_spans(kept, a.cameras, a.people, a.fps, a.max_reproj_px,
                       a.merge_gap, a.max_depth_m, a.max_speed_ms)
    print('after merging split arcs (gap <= %d frames, union must still fit): %d'
          % (a.merge_gap, len(kept)), flush=True)
    scr = body_screen(kept, P, a.dilate, a.max_mask_overlap, a.max_aspect, a.min_fill, a.min_obs)
    kept = [c for c in scr if c['rejected'] is None]
    rejected = [c for c in scr if c['rejected'] is not None]
    print('clear of the person mask, compact and long enough (overlap <= %.0f%%, aspect <= %.1f, fill >= %.0f%%, '
          '>= %d obs): %d, rejected %d'
          % (100 * a.max_mask_overlap, a.max_aspect, 100 * a.min_fill, a.min_obs, len(kept), len(rejected)), flush=True)
    kept.sort(key=lambda c: min(c['pts']))
    rejected.sort(key=lambda c: min(c['pts']))
    recs = [record(c, a.fps) for c in kept]
    rej = [record(c, a.fps) for c in rejected]
    if a.crops:
        write_crops(a.clip, recs, a.crops)
        write_crops(a.clip, rej, os.path.join(a.crops, 'rejected'))

    out = dict(schema='wander.object-flights/1', clip=a.clip, tracks=a.tracks,
               first=first, last=last, fps=a.fps,
               rule=('a tracklet is a free span iff (1) after chaining the frame-to-frame homographies '
                     'across its own span to remove camera motion its stabilised image path fits a '
                     'parabola to within %.1f px with DOWNWARD curvature, (2) it travels at least %.0f px, '
                     '(3) its median distance to the nearest tracked wrist is at least %.0f px, and '
                     '(4) a single 6-parameter ballistic curve p0 + v0 t + g t^2/2, with |g| fixed at '
                     '9.80665 m/s^2 converted by the clip\'s own metres-per-world-unit, reprojects '
                     'through the clip\'s own SfM cameras onto its observations to within %.1f px RMS, '
                     '(5) its blob bbox lies outside the dilated person mask (median overlap <= %.0f%%, '
                     'else it is a limb the mask under-covered) and (6) the blob is compact (median bbox '
                     'aspect <= %.1f and median blob/bbox fill >= %.0f%%, else it is a parallax edge), and '
                     '(7) it has at least %d observations after merging split arcs. '
                     'Overlapping survivors are suppressed, best reprojection per observation first. '
                     'Nothing in the rule refers to the object, the clip or the number of people.'
                     % (a.max_resid_px, a.min_travel_px, a.hand_px, a.max_reproj_px,
                        100 * a.max_mask_overlap, a.max_aspect, 100 * a.min_fill, a.min_obs)),
               params={k: v for k, v in vars(a).items() if k not in ('out', 'cache')},
               candidateCount=tot, trackletCount=len(tl),
               parabolaSurvivors=len(pf), ballisticSurvivors=len(bs),
               flights=recs, rejectedCandidates=rej)
    json.dump(out, open(a.out, 'w'), indent=1)

    def line(r):
        return ('  f%-3d-%-3d  %.2f-%.2f s  %2d obs  parabola %.1f px  ballistic %.2f px rms  '
                '%.1f m/s  depth %.1f m  travel %.0f px  hand %.0f px  size<=%d px  overlap %.0f%%  aspect %.1f  fill %.0f%%'
                % (r['frames'][0], r['frames'][-1], r['frames'][0] / a.fps, r['frames'][-1] / a.fps,
                   r['nObs'], r['parabolaResidPx'], r['ballisticReprojPxRms'], r['speedMs'], r['depthM'],
                   r['travelPx'], r['medianHandDistPx'], max(r['sizePx']),
                   100 * r['medianMaskOverlap'], r['medianBboxAspect'], 100 * r['medianFill']))
    print('\n%d free flights:' % len(recs))
    for r in recs:
        print(line(r))
    if rej:
        print('%d rejected after the ballistic screen:' % len(rej))
        for r in rej:
            print(line(r)); print('      ' + r['rejected'])
    print('wrote', a.out)


if __name__ == '__main__':
    main()
