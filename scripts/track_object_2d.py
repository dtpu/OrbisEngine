"""Track a small rigid carried/thrown object in 2D.

Motion-compensated three-frame differencing (global homography from background
features) isolates anything that moves against the scene; the tracker's own
Mask R-CNN person masks remove the people; the remaining blobs are linked by a
constant-acceleration image-space predictor seeded from the strongest detections.

Generic in the object: nothing here knows it is a bottle.
"""

import argparse, json, os
import numpy as np
import cv2


def read_frames(path, f0, f1):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = []
    for _ in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def homography(a, b, mask):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(ga, 1500, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 20:
        return np.eye(3)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None, winSize=(21, 21), maxLevel=4)
    st = st.reshape(-1).astype(bool)
    if st.sum() < 20:
        return np.eye(3)
    H, _ = cv2.findHomography(p0[st], p1[st], cv2.RANSAC, 2.0)
    return H if H is not None else np.eye(3)


class People:
    def __init__(self, tracks_json):
        self.tj = json.load(open(tracks_json))
        d = os.path.dirname(tracks_json)
        self.W, self.H = self.tj["width"], self.tj["height"]
        self.src = self.tj["sourceIndices"]
        self.n = self.tj["trackCount"]
        self.mot = {}
        for t in range(self.n):
            m = json.load(open(os.path.join(d, f"track_{t:02d}", "motion.json")))
            self.mot[t] = {f["sourceIndex"]: f for f in m["frames"]}
        z = np.load(os.path.join(d, "masks.npz"))
        mh, mw = int(self.H * z["shape"][2]), int(self.W * z["shape"][2])
        self.masks = {}
        for t in range(self.n):
            for s in range(self.tj["samples"]):
                k = f"t{t}_s{s:03d}"
                if k in z:
                    self.masks[(t, s)] = np.unpackbits(z[k])[: mh * mw].reshape(mh, mw)
        self.mh, self.mw = mh, mw

    def bracket(self, sf):
        ks = self.src
        lo = max([i for i, k in enumerate(ks) if k <= sf], default=0)
        hi = min([i for i, k in enumerate(ks) if k >= sf], default=len(ks) - 1)
        return lo, hi

    def mask(self, sf, dilate):
        lo, hi = self.bracket(sf)
        acc = np.zeros((self.mh, self.mw), np.uint8)
        for t in range(self.n):
            for s in (lo, hi):
                m = self.masks.get((t, s))
                if m is not None:
                    acc |= m
        acc = cv2.resize(acc * 255, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        if dilate:
            acc = cv2.dilate(acc, np.ones((dilate, dilate), np.uint8))
        return acc

    def joints(self, sf, max_gap_samples=2):
        """{track: (J, 2)} at source frame sf. A track absent at sf (it left the frame, or was
        lost for a few samples) is interpolated from its nearest samples when they are within
        `max_gap_samples` of the bracket, and omitted otherwise."""
        step = max(1, (self.src[-1] - self.src[0]) // max(1, len(self.src) - 1))
        out = {}
        for t in range(self.n):
            ks = sorted(self.mot[t])
            below = [k for k in ks if k <= sf]
            above = [k for k in ks if k >= sf]
            if not below or not above:
                continue
            klo, khi = below[-1], above[0]
            if khi - klo > (2 * max_gap_samples + 1) * step:
                continue
            w = 0.0 if khi == klo else (sf - klo) / (khi - klo)
            a = np.array(self.mot[t][klo]["projectedBodyJoints"])
            b = np.array(self.mot[t][khi]["projectedBodyJoints"])
            out[t] = a * (1 - w) + b * w
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--seed", required=True, help="sf:x:y of a hand-picked frame the object is on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patches", default=None)
    ap.add_argument("--dilate", type=int, default=9)
    ap.add_argument("--min-area", type=float, default=20.0)
    ap.add_argument("--max-area", type=float, default=6000.0)
    ap.add_argument("--gate", type=float, default=45.0, help="px radius around the prediction")
    args = ap.parse_args()

    P = People(args.tracks)
    W, H = P.W, P.H
    pad = 2
    frames = read_frames(args.clip, args.first - pad, args.last + pad)
    base = args.first - pad

    det = {}
    dmaps = {}
    for sf in range(args.first, args.last + 1):
        i = sf - base
        cur, prev, nxt = frames[i], frames[i - 1], frames[i + 1]
        pm = P.mask(sf, 40)
        bgm = np.where(pm > 0, 0, 255).astype(np.uint8)
        wp = cv2.warpPerspective(prev, homography(prev, cur, bgm), (W, H))
        wn = cv2.warpPerspective(nxt, homography(nxt, cur, bgm), (W, H))
        d = np.minimum(cv2.absdiff(cur, wp).max(2), cv2.absdiff(cur, wn).max(2))
        d = cv2.GaussianBlur(d, (0, 0), 1.5)
        km = np.where(P.mask(sf, args.dilate) > 0, 0, 255).astype(np.uint8)
        km[:, :25] = 0
        km[:, -25:] = 0
        km[:25] = 0
        km[-25:] = 0
        dm = np.where(km > 0, d, 0)
        dmaps[sf] = dm
        nz = dm[dm > 0]
        thr = max(9.0, float(np.percentile(nz, 99.6)) * 0.45) if nz.size else 255.0
        _, bw = cv2.threshold(dm, thr, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
        cands = []
        for c in range(1, n):
            a = float(stats[c, cv2.CC_STAT_AREA])
            if a < args.min_area or a > args.max_area:
                continue
            sel = lab == c
            wgt = dm[sel].astype(np.float64)
            ys, xs = np.nonzero(sel)
            cands.append(
                dict(
                    x=float((xs * wgt).sum() / wgt.sum()),
                    y=float((ys * wgt).sum() / wgt.sum()),
                    area=a,
                    strength=float(wgt.mean()),
                    bx=[
                        int(stats[c, cv2.CC_STAT_LEFT]),
                        int(stats[c, cv2.CC_STAT_TOP]),
                        int(stats[c, cv2.CC_STAT_WIDTH]),
                        int(stats[c, cv2.CC_STAT_HEIGHT]),
                    ],
                )
            )
        det[sf] = cands

    # --- link, forward and backward, from the seed
    sfs, sx, sy = [float(v) for v in args.seed.split(":")]
    sfs = int(sfs)
    track = {sfs: dict(x=sx, y=sy, source="seed")}

    def nearest(sf, px, py):
        best, bd = None, args.gate
        for c in det.get(sf, []):
            dd = np.hypot(c["x"] - px, c["y"] - py)
            if dd < bd:
                best, bd = c, dd
        return best

    for direction in (1, -1):
        hist = [(sfs, sx, sy)]
        sf = sfs + direction
        while args.first <= sf <= args.last:
            if len(hist) == 1:
                px, py = hist[-1][1], hist[-1][2]
            elif len(hist) == 2:
                px = hist[-1][1] + (hist[-1][1] - hist[-2][1])
                py = hist[-1][2] + (hist[-1][2] - hist[-2][2])
            else:
                a = np.array(hist[-3:])
                t = a[:, 0] - sf
                px = np.polyval(np.polyfit(t, a[:, 1], 2), 0.0)
                py = np.polyval(np.polyfit(t, a[:, 2], 2), 0.0)
            c = nearest(sf, px, py)
            if c is None:
                break
            track[sf] = dict(
                x=c["x"],
                y=c["y"],
                area=c["area"],
                strength=c["strength"],
                bx=c["bx"],
                source="detected",
            )
            hist.append((sf, c["x"], c["y"]))
            sf += direction

    out = dict(
        clip=args.clip,
        first=args.first,
        last=args.last,
        seed=args.seed,
        method="motion-compensated 3-frame differencing + constant-acceleration linking",
        observations=[dict(sourceIndex=k, **v) for k, v in sorted(track.items())],
        allCandidates={str(k): v for k, v in det.items()},
    )
    json.dump(out, open(args.out, "w"))
    ks = sorted(track)
    print("tracked", len(ks), "frames", ks[0], "..", ks[-1])
    for k in ks:
        print(
            " ",
            k,
            round(track[k]["x"], 1),
            round(track[k]["y"], 1),
            track[k].get("area"),
            track[k]["source"],
        )

    if args.patches:
        pat = {}
        for k in ks:
            i = k - base
            x, y = int(round(track[k]["x"])), int(round(track[k]["y"]))
            r = 40
            x0, y0 = max(0, x - r), max(0, y - r)
            pat[f"f{k:04d}"] = frames[i][y0 : y + r, x0 : x + r]
            pat[f"d{k:04d}"] = dmaps[k][y0 : y + r, x0 : x + r]
            pat[f"o{k:04d}"] = np.array([x0, y0])
        np.savez_compressed(args.patches, **pat)


if __name__ == "__main__":
    main()
