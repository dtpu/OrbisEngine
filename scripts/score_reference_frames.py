#!/usr/bin/env python3
"""Score every frame of a clip as an avatar reference, and pick the best one (or best N views).

  KMP_DUPLICATE_LIB_OK=TRUE uv run --locked --group inference python scripts/score_reference_frames.py \
      public/clips/atrium.mp4 --stride 2 --n 8 --out .context/pose/frames/atrium.json --sheet share/frames-atrium.png

Avatar identity, build and clothing all come from the two-to-eight reference frames, so a bad pick
poisons everything downstream -- atrium's avatar read as a woman because the chosen frames showed his
back. Until now those frames were chosen by hand, or by the first frame that passed a heuristic.

Components, all 0..1, per frame (Mask R-CNN instance mask + COCO keypoints):
  height      person box height / frame height, saturating -- a 90 px person cannot carry identity
  sharp       variance of Laplacian inside the mask, relative to the best frame of this clip
  complete    visible keypoints, ankles and wrists weighted up (feet and hands are what LHM invents)
  unoccluded  mask solidity (area / convex hull) and no second person overlapping the box
  facing      1 facing the camera, 0.5 profile, 0 the back of the head -- from the image-x order of
              the left and right shoulder keypoints, not from whether a nose keypoint was emitted
  face        YuNet actually found a face, scaled by how big it is (identity detail)
  exposure    1 - fraction of body pixels crushed or blown
  framed      the box does not run off the frame edge

score = weighted sum, times hard gates (a person under 8 % of frame height, or an empty mask, is 0).
Weights are in WEIGHTS and were set against the known failures: atrium (back-view frames -> female
avatar) has to be ranked below its own frontal frames, and a full-body frame with feet has to beat a
sharper crop of a torso. --explain prints the per-component numbers so a ranking can be argued with.

ESTIMATOR-READABILITY GATE (--estimator-probe). Everything above measures how much identity a frame
carries. It does not ask whether the frame can be BUILT from, and those are different questions:
bedroom's top-scored frame 198 was rejected downstream because MultiHMR detects no person in the
whitened cut-out lhm_person.py feeds it (the subject is seated behind a desk). A frame that scores
well and cannot be used is worse than useless, so a probe file from
scripts/probe_pose_estimator.py -- real MultiHMR, same cut-out, same threshold -- hard-gates the
score to 0 for any candidate it could not read. Frames with no probe entry keep their score and are
reported as unverified; `estimatorReadable` on each row records which of the three it is. No local
detector substitutes for this: torchvision's keypoint R-CNN reads bedroom f198 at 0.999.
"""

import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

# COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
# 11/12 hips, 13/14 knees, 15/16 ankles
KP_W = np.array(
    [1.0, 0.6, 0.6, 0.5, 0.5, 1.0, 1.0, 0.7, 0.7, 1.4, 1.4, 1.0, 1.0, 0.9, 0.9, 1.6, 1.6]
)
WEIGHTS = dict(
    height=0.20,
    sharp=0.12,
    complete=0.18,
    unoccluded=0.10,
    facing=0.22,
    face=0.12,
    exposure=0.03,
    framed=0.03,
)
YUNET = ".context/pose/yunet.onnx"

_m = None
_yn = "unset"


def yunet(path=None):
    """YuNet face detector -- a real detection, unlike a hallucinated nose keypoint."""
    global _yn
    if _yn == "unset":
        p = Path(path or YUNET)
        _yn = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.6) if p.exists() else None
    return _yn


def models():
    global _m
    if _m is None:
        import torch
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2,
            MaskRCNN_ResNet50_FPN_V2_Weights,
            keypointrcnn_resnet50_fpn,
            KeypointRCNN_ResNet50_FPN_Weights,
        )

        dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        mk = (
            maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            .eval()
            .to(dev)
        )
        kp = (
            keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            .eval()
            .to(dev)
        )
        _m = (mk, kp, dev, torch)
    return _m


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score_frame(rgb, seed=None):
    mk, kp, dev, torch = models()
    H, W = rgb.shape[:2]
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(dev)
    with torch.no_grad():
        r = mk([t])[0]
        k = kp([t])[0]
    keep = [
        i
        for i in range(len(r["labels"]))
        if int(r["labels"][i]) == 1 and float(r["scores"][i]) > 0.7
    ]
    if not keep:
        return dict(empty=True, score=0.0)
    boxes = [[float(v) for v in r["boxes"][i].cpu().numpy()] for i in keep]
    if seed is not None:
        bi = int(np.argmax([_iou(seed, b) for b in boxes]))
    else:
        bi = int(np.argmax([(b[3] - b[1]) for b in boxes]))  # the tallest person is the subject
    box = boxes[bi]
    mask = r["masks"][keep[bi], 0].cpu().numpy() > 0.5
    if mask.sum() < 400:
        return dict(empty=True, score=0.0)

    x0, y0, x1, y1 = [int(v) for v in box]
    bh = max(1.0, y1 - y0)
    height = min(1.0, (bh / H) / 0.55)

    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    sharp_raw = float(lap[mask].var())

    px = rgb[mask]
    lum = px.mean(1)
    exposure = 1.0 - float(((lum < 8) | (lum > 248)).mean())

    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = 0.0
    if cnt:
        c = max(cnt, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        solid = float(cv2.contourArea(c) / ha) if ha > 0 else 0.0
    others = max([_iou(box, b) for j, b in enumerate(boxes) if j != bi] or [0.0])
    unocc = float(np.clip((solid - 0.35) / 0.45, 0, 1)) * (1.0 - min(1.0, others * 2.0))

    # keypoints of the instance whose box matches
    kkeep = [i for i in range(len(k["scores"])) if float(k["scores"][i]) > 0.5]
    kps = np.zeros((17, 3))
    if kkeep:
        kb = [[float(v) for v in k["boxes"][i].cpu().numpy()] for i in kkeep]
        ki = int(np.argmax([_iou(box, b) for b in kb]))
        kps = k["keypoints"][kkeep[ki]].cpu().numpy().copy()
        kps[:, 2] = k["keypoints_scores"][kkeep[ki]].cpu().numpy()
    vis = kps[:, 2] > 2.0
    inside = (kps[:, 0] > 2) & (kps[:, 0] < W - 3) & (kps[:, 1] > 2) & (kps[:, 1] < H - 3)
    complete = float((KP_W * (vis & inside)).sum() / KP_W.sum())

    # Body yaw, 0 deg = facing the camera, 180 = filmed from behind.
    # A COCO keypoint model emits a nose and eyes for the back of a head with high confidence --
    # measured on stairs2, which is filmed from behind throughout and still scored 1.00 on
    # "the nose keypoint exists". What it does get right is which shoulder is which: facing the
    # camera the subject's LEFT shoulder lands on the image right, so kps[5].x - kps[6].x > 0.
    # On stairs2 that difference is negative in every frame; on atrium it is negative through
    # f105 (the frame the avatar was actually built from, the one that came out female) and
    # turns positive at f141-161, exactly the frames FACEFIX found a usable face in.
    yaw, facing = None, 0.5
    if vis[5] and vis[6] and (vis[11] or vis[12]):
        dx = float(kps[5, 0] - kps[6, 0])
        hip = kps[[11, 12]][vis[[11, 12]]].mean(0)
        tl = abs((kps[5, 1] + kps[6, 1]) / 2 - hip[1])
        if tl > 5:
            off = float(np.degrees(np.arccos(np.clip((abs(dx) / tl) / 0.85, 0, 1))))
            yaw = off if dx > 0 else 180.0 - off
            facing = float((1 + np.cos(np.radians(yaw))) / 2)

    facepx = 0.0
    fd = yunet()
    if fd is not None:
        hh = max(32, int(0.35 * bh))
        fy0, fy1 = max(0, y0 - 10), min(H, y0 + hh)
        fx0, fx1 = max(0, x0 - 10), min(W, x1 + 10)
        crop = rgb[fy0:fy1, fx0:fx1]
        if crop.size and crop.shape[0] > 20 and crop.shape[1] > 20:
            fd.setInputSize((crop.shape[1], crop.shape[0]))
            _, det = fd.detect(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if det is not None and len(det):
                facepx = float(max(d[3] for d in det))
    face = float(np.clip(facepx / 70.0, 0, 1))

    framed = 1.0 - 0.5 * float(x0 <= 1 or x1 >= W - 2) - 0.5 * float(y1 >= H - 2)
    framed = max(0.0, framed)
    return dict(
        empty=False,
        box=[x0, y0, x1, y1],
        mask_px=int(mask.sum()),
        sharp_raw=sharp_raw,
        height=height,
        complete=complete,
        unoccluded=unocc,
        facing=facing,
        face=face,
        facePx=facepx,
        exposure=exposure,
        framed=framed,
        yawDeg=yaw,
        ankles=bool(vis[15] or vis[16]),
        wrists=bool(vis[9] or vis[10]),
    )


def load_probe(path, clip=None):
    """{index: {detected, score, ...}} for one clip, from scripts/probe_pose_estimator.py."""
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    clips = doc.get("clips", doc)
    if clip and clip in clips:
        rows = clips[clip]
    elif len(clips) == 1:
        rows = next(iter(clips.values()))
    else:
        raise SystemExit(f"probe file covers {sorted(clips)}; say which clip with --clip-name")
    return {int(k): v for k, v in rows.items()}


def finish(rows, probe=None):
    """sharp is relative to this clip's best frame, then the weighted sum and gates."""
    probe = probe or {}
    good = [r for r in rows if not r.get("empty")]
    smax = max([r["sharp_raw"] for r in good] or [1.0])
    for r in rows:
        if r.get("empty"):
            r["score"] = 0.0
            r["sharp"] = 0.0
            r["estimatorReadable"] = None
            continue
        r["sharp"] = float(np.clip(r["sharp_raw"] / (smax + 1e-9), 0, 1) ** 0.5)
        s = sum(WEIGHTS[k] * r[k] for k in WEIGHTS)
        gate = 1.0 if r["height"] > 0.15 else 0.0
        # The pose estimator has the last word: a frame it cannot read cannot build an avatar,
        # however much identity it carries. Unprobed frames are unverified, not assumed good.
        p = probe.get(r["index"])
        r["estimatorReadable"] = None if p is None else bool(p.get("detected"))
        if p is not None and not p.get("detected"):
            r["estimatorReason"] = p.get(
                "reason", "MultiHMR found no person in the whitened cut-out"
            )
            gate = 0.0
        r["score"] = float(s * gate)
    return rows


def pick_set(rows, n, min_yaw_gap=35.0):
    """Greedy: best frame first, then the best remaining frame that adds a new viewing angle.

    Without the spread term an 8-view set collapses onto 8 near-identical frames of the one moment
    the subject was best lit, which is what LHM++ was being fed.
    """
    cand = sorted([r for r in rows if r["score"] > 0], key=lambda r: -r["score"])
    if not cand:
        return []
    chosen = [cand[0]]
    while len(chosen) < n:
        best, bv = None, -1
        for r in cand:
            if r in chosen:
                continue
            gaps = []
            for c in chosen:
                if r.get("yawDeg") is not None and c.get("yawDeg") is not None:
                    gaps.append(min(1.0, abs(r["yawDeg"] - c["yawDeg"]) / min_yaw_gap))
                else:
                    gaps.append(min(1.0, abs(r["index"] - c["index"]) / 40.0))
            v = r["score"] * (0.35 + 0.65 * min(gaps))
            if v > bv:
                best, bv = r, v
        if best is None:
            break
        chosen.append(best)
    return sorted(chosen, key=lambda r: r["index"])


def track_seeds(motion):
    """{sourceIndex: box} from one track's motion.json, so a two-person clip scores ONE identity.

    Without this the scorer takes the tallest person in each frame, which on elevator swaps between
    the two men as they pass -- and the second avatar would then be built from the first man's face.
    """
    rows = json.loads(Path(motion).read_text())["frames"]
    return {
        int(r["sourceIndex"]): [float(v) for v in r["maskBox"]] for r in rows if r.get("maskBox")
    }


def score_video(
    video,
    stride=3,
    n=8,
    out=None,
    compare=None,
    max_frames=260,
    keep_images=False,
    probe=None,
    clip_name=None,
    seeds=None,
):
    """Score every sampled frame and choose a set. Returns the same dict the CLI writes."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.get(cv2.CAP_PROP_FPS)
    stride = max(stride, int(np.ceil(total / max_frames)))
    want = set(range(0, total, stride))
    comp = [int(x) for x in compare.split(",")] if compare else []
    want |= set(comp)
    if seeds:
        want &= set(seeds)  # a track is only present where it was tracked
    rows, keepimg, k = [], {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            r = score_frame(rgb, seed=seeds.get(k) if seeds else None)
            r["index"] = k
            r["time"] = k / fps
            rows.append(r)
            if keep_images:
                keepimg[k] = rgb
        k += 1
    cap.release()
    probe_rows = load_probe(probe, clip_name or Path(video).stem) if probe else {}
    rows = finish(rows, probe_rows)
    chosen = pick_set(rows, n)
    by = {r["index"]: r for r in rows}
    res = dict(
        video=str(video),
        frames=total,
        fps=fps,
        stride=stride,
        weights=WEIGHTS,
        scored=len(rows),
        rows=rows,
        chosen=[r["index"] for r in chosen],
        chosenBest=max(chosen, key=lambda r: r["score"])["index"] if chosen else None,
        current=comp,
        currentScores={i: by[i]["score"] for i in comp if i in by},
        estimatorProbe=str(probe) if probe else None,
        estimatorProbed=sorted(probe_rows),
        estimatorRejected=sorted([i for i, v in probe_rows.items() if not v.get("detected")]),
        chosenEstimatorReadable=(
            by[max(chosen, key=lambda r: r["score"])["index"]]["estimatorReadable"]
            if chosen
            else None
        ),
    )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1, default=float))
    return (res, keepimg) if keep_images else res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet")
    ap.add_argument("--compare", help="comma list of the frames chosen today")
    ap.add_argument("--max-frames", type=int, default=260)
    ap.add_argument(
        "--estimator-probe",
        help="scripts/probe_pose_estimator.py output: frames MultiHMR "
        "could not read are scored 0 however well they rank",
    )
    ap.add_argument(
        "--clip-name", help="which clip to read from the probe file (default: video stem)"
    )
    ap.add_argument(
        "--track-motion",
        help="one track's motion.json: score THAT identity, not the "
        "tallest person in each frame (multi-person clips)",
    )
    a = ap.parse_args()
    out, keepimg = score_video(
        a.video,
        stride=a.stride,
        n=a.n,
        out=a.out,
        compare=a.compare,
        max_frames=a.max_frames,
        keep_images=True,
        probe=a.estimator_probe,
        clip_name=a.clip_name,
        seeds=track_seeds(a.track_motion) if a.track_motion else None,
    )
    rows = out["rows"]
    chosen = [r for r in rows if r["index"] in out["chosen"]]
    comp = out["current"]
    by = {r["index"]: r for r in rows}
    print(f"{a.video}: {len(rows)} frames scored (stride {out['stride']})")
    print("  rank  frame   score  height sharp compl unocc facing face expos framed   yaw")
    top = sorted([r for r in rows if r["score"] > 0], key=lambda r: -r["score"])[:6]
    for r in top:
        print(
            f"        {r['index']:5d}  {r['score']:.3f}   {r['height']:.2f}  {r['sharp']:.2f}  "
            f"{r['complete']:.2f}  {r['unoccluded']:.2f}  {r['facing']:.2f}  {r['face']:.2f} "
            f"{r['exposure']:.2f}  {r['framed']:.2f}  {'' if r['yawDeg'] is None else round(r['yawDeg']):>5}"
        )
    if comp:
        print("  frames chosen today:")
        for i in comp:
            r = by.get(i)
            if r:
                rank = 1 + sorted([q["score"] for q in rows], reverse=True).index(r["score"])
                print(
                    f"        {i:5d}  {r['score']:.3f}  rank {rank}/{len(rows)}  facing {r['facing']:.2f} "
                    f"yaw {'' if r['yawDeg'] is None else round(r['yawDeg'])}  face {r['face']:.2f} "
                    f"height {r['height']:.2f} compl {r['complete']:.2f}"
                )
    if out["estimatorProbe"]:
        print(
            f"  estimator probe: {len(out['estimatorProbed'])} candidates, "
            f"rejected {out['estimatorRejected'] or 'none'}"
        )
    else:
        print(
            "  estimator readability UNVERIFIED (no --estimator-probe): the top frame may still "
            "be one MultiHMR cannot read"
        )
    print(f"  picked set: {[r['index'] for r in chosen]}")
    if a.sheet:
        tiles = []
        for lab, idxs in (("CURRENT", comp), ("SCORED", [r["index"] for r in chosen])):
            row = []
            for i in idxs:
                im = keepimg.get(i)
                if im is None:
                    continue
                r = by[i]
                c = im.copy()
                if not r.get("empty"):
                    x0, y0, x1, y1 = r["box"]
                    c = c[max(0, y0 - 20) : y1 + 20, max(0, x0 - 20) : x1 + 20]
                c = cv2.resize(c, (220, 380))
                cv2.rectangle(c, (0, 0), (219, 26), (0, 0, 0), -1)
                cv2.putText(
                    c,
                    f"{lab[0]}{i} s{r['score']:.2f} f{r['facing']:.1f}",
                    (4, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                )
                row.append(c)
            if row:
                tiles.append(np.concatenate(row, 1))
        if tiles:
            w = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
            Path(a.sheet).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(a.sheet, cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
            print("  sheet", a.sheet)


if __name__ == "__main__":
    main()
