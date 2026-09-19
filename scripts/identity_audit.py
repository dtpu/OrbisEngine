#!/usr/bin/env python3
"""Does each track still hold the person it started with?

  worker/.venv-da3/bin/python scripts/identity_audit.py --tracks .context/mp/<clip>/tracks \
      --clip public/clips/<clip>.mp4 --json-out share/identity-<clip>.json

The check this replaces (`reproject_multiperson.py`, hit-own-box vs hit-rival-box) could not answer
the question it printed. The boxes came from the same tracker that assigned the identities and the
avatar was posed from that track, so after a swap at frame f the track carries person B's poses AND
person B's box: the avatar lands inside its own box and scores ok. The test was tautological.

Identity here comes from something the tracker did not produce: the source pixels under each track's
mask, compared against a FIXED reference.

  own      = Bhattacharyya distance from this sample's appearance to this track's own reference
  rival    = the same distance to the best other track's reference
  margin   = rival - own; negative means this track now looks more like the other person

Two references per track, not one: frame 0 (forward) and frame N (backward). A tracker swap at
frame f is exactly the case where one half of the clip agrees with the forward reference and the
other half agrees with the backward one, so requiring BOTH to hold at every sample is a
forward/backward agreement test that needs no second tracker run. `--tracks-reverse` additionally
compares against a real reversed-clip tracker run when one exists.

--swap-test is the recall control: it relabels the tracks at the midpoint and re-runs the audit.
A detector with no power reports "ok" on that too, and then its "ok" on the real clip means nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BANDS = 3          # head / torso / legs, so grey trousers over black trousers is a large residual
H_BINS, S_BINS, V_BINS = 12, 8, 8
MIN_MASK_PX = 400  # below this the crop is too small for a stable histogram
MAX_BAD_RUN = 3    # review §4G: fail on any disagreement interval longer than 3 samples


def load_masks(tracks: Path):
    """{track: {sample: bool mask}}, plus (h, w) of the mask grid."""
    z = np.load(tracks / "masks.npz")
    full_h, full_w, ms = z["shape"]
    h, w = int(round(full_h * ms)), int(round(full_w * ms))
    out: dict[int, dict[int, np.ndarray]] = {}
    for k in z.files:
        if k == "shape":
            continue
        t, s = k.split("_")
        bits = np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool)
        out.setdefault(int(t[1:]), {})[int(s[1:])] = bits
    return out, (h, w)


def descriptor(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Band-wise HSV histogram of the masked pixels, L1-normalised per band.

    Bands are cut from the mask's own bounding box, so the descriptor follows the person up and
    down the frame and does not encode where in the image he is -- the thing the tracker already
    knows and the thing a swap preserves.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < MIN_MASK_PX:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y0, y1 = ys.min(), ys.max()
    edges = np.linspace(y0, y1 + 1, BANDS + 1)
    parts = []
    for b in range(BANDS):
        sel = (ys >= edges[b]) & (ys < edges[b + 1])
        if sel.sum() < 40:
            parts.append(np.zeros(H_BINS * S_BINS + V_BINS, np.float64))
            continue
        px = hsv[ys[sel], xs[sel]]
        hs, _, _ = np.histogram2d(px[:, 0], px[:, 1], bins=(H_BINS, S_BINS),
                                  range=((0, 180), (0, 256)))
        v, _ = np.histogram(px[:, 2], bins=V_BINS, range=(0, 256))
        hs = hs.ravel() / max(hs.sum(), 1.0)
        v = v / max(v.sum(), 1.0)
        parts.append(np.concatenate([hs, v]))
    d = np.concatenate(parts)
    return d / max(float(d.sum()), 1e-12)   # one distribution over the whole body, so BC <= 1


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1.0 - float(np.sqrt(p * q).sum()))))


def read_frames(clip: str, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    want = sorted(set(indices))
    out, i, k = {}, 0, 0
    while k < len(want):
        ok, frame = cap.read()
        if not ok:
            break
        if i == want[k]:
            out[i] = frame
            k += 1
        i += 1
    cap.release()
    return out


def audit(tracks: Path, clip: str, stride: int = 1, swap_at: int | None = None,
          swap_pair: tuple[int, int] | None = None) -> dict:
    masks, (h, w) = load_masks(tracks)
    tids = sorted(masks)
    if len(tids) < 2:
        return dict(ok=None, reason=f"{len(tids)} track(s); identity is not at risk with fewer than 2")

    motions = {t: json.loads((tracks / f"track_{t:02d}" / "motion.json").read_text())["frames"]
               for t in tids}
    src_of = {t: {f["sample"]: f["sourceIndex"] for f in motions[t]} for t in tids}

    # People enter and leave the frame, so there need not be one sample where every track is
    # present. Identity is only at risk where two or more tracks share a frame, so those are the
    # audited samples; each track's references still come from its own first and last frame.
    all_samples = sorted(set.union(*[set(masks[t]) for t in tids]))[::stride]
    present = {s: [t for t in tids if s in masks[t]] for s in all_samples}
    shared = [s for s in all_samples if len(present[s]) >= 2]
    if len(shared) < 8:
        return dict(ok=None, reason=f"only {len(shared)} samples where two or more tracks overlap")

    def source(s):
        return src_of[present[s][0]].get(s, s)

    # the swap control relabels the pair that overlaps most, from the midpoint of their overlap
    overlap = {(a, b): [s for s in shared if a in present[s] and b in present[s]]
               for i, a in enumerate(tids) for b in tids[i + 1:]}
    pair = max(overlap, key=lambda k: len(overlap[k]))
    pair_mid = overlap[pair][len(overlap[pair]) // 2]
    if swap_at is not None:
        a, b = swap_pair or pair
        for s in overlap[(a, b)]:
            if s >= swap_at:
                masks[a][s], masks[b][s] = masks[b][s], masks[a][s]

    frames = read_frames(clip, [source(s) for s in all_samples])
    desc: dict[int, dict[int, np.ndarray]] = {t: {} for t in tids}
    for s in all_samples:
        bgr = frames.get(source(s))
        if bgr is None:
            continue
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        for t in present[s]:
            d = descriptor(small, masks[t][s])
            if d is not None:
                desc[t][s] = d

    usable = [s for s in shared if sum(s in desc[t] for t in tids) >= 2]
    if len(usable) < 8:
        return dict(ok=None, reason=f"only {len(usable)} samples where two or more tracks have a usable mask")

    # forward reference = the track's first usable sample, backward = its last. Averaging the first
    # and last few would blur a swap that happened early or late, so the endpoints are taken as they are.
    ref = {}
    for t in tids:
        own = sorted(desc[t])
        if own:
            ref[t] = dict(fwd=desc[t][own[0]], bwd=desc[t][own[-1]])

    rows, bad = [], []
    for s in usable:
        row = dict(sample=s, sourceIndex=source(s), people=[])
        here = [t for t in tids if s in desc[t] and t in ref]
        for t in here:
            d = desc[t][s]
            entry = dict(track=t)
            worst = None
            for side in ("fwd", "bwd"):
                own = bhattacharyya(d, ref[t][side])
                rivals = {o: bhattacharyya(d, ref[o][side]) for o in here if o != t}
                ro, rv = min(rivals.items(), key=lambda kv: kv[1])
                entry[side] = dict(own=round(own, 4), rivalTrack=ro, rival=round(rv, 4),
                                   margin=round(rv - own, 4))
                worst = entry[side]["margin"] if worst is None else min(worst, entry[side]["margin"])
            entry["margin"] = worst
            entry["verdict"] = "ok" if worst > 0 else "SWAP"
            if worst <= 0:
                bad.append((s, t))
            row["people"].append(entry)
        rows.append(row)

    # longest run of consecutive audited samples with any track failing
    flagged = sorted({s for s, _ in bad})
    run = best = 0
    prev = None
    for s in flagged:
        run = run + 1 if prev is not None and usable.index(s) == usable.index(prev) + 1 else 1
        best = max(best, run)
        prev = s
    margins = [p["margin"] for r in rows for p in r["people"]]
    # the same run length per track, so a caller can drop the track that changed person and keep
    # the ones that did not, instead of refusing the whole clip over a background child
    per_track = {}
    for t in tids:
        mine = [s for s, tt in bad if tt == t]
        run_t = best_t = 0
        prev = None
        for s in mine:
            run_t = run_t + 1 if prev is not None and usable.index(s) == usable.index(prev) + 1 else 1
            best_t = max(best_t, run_t)
            prev = s
        per_track[t] = dict(longestBadRun=best_t, ok=bool(best_t <= MAX_BAD_RUN))
    failed = [t for t in tids if not per_track[t]["ok"]]
    return dict(ok=bool(best <= MAX_BAD_RUN), tracks=tids, samples=len(usable),
                perTrack=per_track, failedTracks=failed,
                swapPair=list(pair), swapMid=pair_mid,
                swapSamples=[[s, t] for s, t in bad], longestBadRun=best, maxBadRun=MAX_BAD_RUN,
                medianMargin=round(float(np.median(margins)), 4),
                minMargin=round(float(np.min(margins)), 4), frames=rows,
                method="appearance of the masked source pixels against each track's own first- and "
                       "last-frame references, audited wherever two or more tracks share a frame; "
                       "the tracker produced the masks but not the references, and a swap breaks "
                       "agreement with one of the two")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, type=Path)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--swap-test", action="store_true",
                    help="also run with the labels swapped at the midpoint and require a SWAP verdict")
    ap.add_argument("--tracks-reverse", type=Path,
                    help="a reversed-clip tracker run, if one exists, for a true forward/backward check")
    a = ap.parse_args()

    doc = audit(a.tracks, a.clip, a.stride)
    if doc.get("ok") is None:
        print(f"identity NOT CHECKED: {doc['reason']}")
        if a.json_out:
            a.json_out.write_text(json.dumps(doc, indent=1))
        return 0 if "fewer than 2" in doc["reason"] else 4

    if a.swap_test:
        mid = doc["swapMid"]
        ctrl = audit(a.tracks, a.clip, a.stride, swap_at=mid, swap_pair=tuple(doc["swapPair"]))
        doc["swapTest"] = dict(swapAt=mid, detected=not ctrl.get("ok", True),
                               longestBadRun=ctrl.get("longestBadRun"),
                               minMargin=ctrl.get("minMargin"))
        print(f"swap control: labels swapped at sample {mid} -> "
              f"{'DETECTED' if doc['swapTest']['detected'] else 'MISSED'} "
              f"(longest bad run {ctrl.get('longestBadRun')})")
        if not doc["swapTest"]["detected"]:
            doc["ok"] = False
            doc["reason"] = ("the swap control was not detected, so this audit has no power on this "
                            "clip and its 'ok' means nothing")

    if a.tracks_reverse:
        rev, _ = load_masks(a.tracks_reverse)
        fwd, _ = load_masks(a.tracks)
        shared = sorted(set(rev) & set(fwd))
        dis = []
        for s in sorted(set.intersection(*[set(fwd[t]) for t in shared],
                                         *[set(rev[t]) for t in shared])):
            for t in shared:
                iou = [(float((fwd[t][s] & rev[o][s]).sum()) / max((fwd[t][s] | rev[o][s]).sum(), 1), o)
                       for o in shared]
                if max(iou)[1] != t:
                    dis.append([s, t])
        doc["forwardBackward"] = dict(disagreements=dis, ok=len(dis) <= MAX_BAD_RUN)
        if not doc["forwardBackward"]["ok"]:
            doc["ok"] = False
    else:
        doc["forwardBackward"] = "no reversed-clip tracker run supplied; the frame-0/frame-N " \
                                 "reference pair is standing in for it"

    print(f"{doc['samples']} samples, {len(doc['tracks'])} tracks; median margin "
          f"{doc['medianMargin']:+.4f}, min {doc['minMargin']:+.4f}, "
          f"{len(doc['swapSamples'])} flagged sample-tracks, longest run {doc['longestBadRun']}")
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(doc, indent=1))
    if not doc["ok"]:
        print("IDENTITY: " + str(doc.get("reason") or
              f"a track matched the other person's reference better for "
              f"{doc['longestBadRun']} consecutive samples (limit {MAX_BAD_RUN}). The tracker may "
              f"have swapped the two people; do not ship avatars built from these tracks."))
        return 3
    print("no swap: every track matched its own frame-0 and frame-N reference at every sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
