#!/usr/bin/env python3
"""Prepared-person input for worker/modal_lhm.py --prepared: source.png, mask.png, prepared.json.

  KMP_DUPLICATE_LIB_OK=TRUE worker/.venv-da3/bin/python scripts/prepare_lhm_person.py clip.mp4 \
      --frame 120 --out .context/<clip>/prepared-person
  ... --candidates 90,105,120        # only report the person box per frame (does not write)

Mask: SegFormer-b0 person class (worker/wander_worker/masks.py), largest connected component,
optional small dilation. lhm_person.py needs a portrait person box, so the report flags boxes
that touch the frame edge or are wider than tall.

Omit --frame and the frame is CHOSEN, by scripts/score_reference_frames.py, instead of being
picked by hand. Everything about the avatar's identity comes from this one frame, and hand-picking
it is how atrium ended up built from a rear view and rendering as a woman. Passing --frame keeps
the old behaviour exactly, and also prints where that frame ranks.

Pass --estimator-probe as well (scripts/probe_pose_estimator.py) and frames MultiHMR cannot read
are struck out before the pick, so the chosen frame is one the build can actually use. Without it
the choice is unverified on that axis, which is how bedroom's top frame 198 was picked and then
refused by the GPU.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def decode(video, index):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, bgr = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fps, count


_rcnn = None


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def maskrcnn_person(rgb, bbox=None):
    """torchvision Mask R-CNN instance mask of one person (COCO label 1).

    Without --bbox: the highest-scoring person, which is the single-person behaviour. With a
    bbox (a track's box in this frame): the instance that best overlaps it, so a two-person
    frame yields the right identity rather than whichever person Mask R-CNN scored higher.
    """
    global _rcnn
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    if _rcnn is None:
        _rcnn = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        r = _rcnn([x])[0]
    keep = [i for i in range(len(r["labels"])) if int(r["labels"][i]) == 1 and float(r["scores"][i]) > 0.5]
    if not keep:
        return np.zeros(rgb.shape[:2], np.uint8)
    if bbox is None:
        best = max(keep, key=lambda i: float(r["scores"][i]))
    else:
        scored = [(_iou(bbox, [float(v) for v in r["boxes"][i].numpy()]), i) for i in keep]
        overlap, best = max(scored)
        if overlap < 0.3:
            raise SystemExit(f"no Mask R-CNN person overlaps the track box (best IoU {overlap:.2f})")
        print(json.dumps(dict(instances=len(keep), chosenIoU=round(overlap, 3),
                              chosenBox=[round(float(v)) for v in r["boxes"][best].numpy()])))
    return (r["masks"][best, 0].numpy() > 0.5).astype(np.uint8)


def person_mask(rgb, dilate, method="segformer", bbox=None):
    if method == "maskrcnn":
        m = maskrcnn_person(rgb, bbox)
    else:
        from wander_worker.masks import people_masks
        m = people_masks(rgb[None], dilate_px=0)[0].astype(np.uint8)
        if bbox is not None:
            keep = np.zeros_like(m)
            x0, y0, x1, y1 = [int(v) for v in bbox]
            keep[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 1
            m = m * keep
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (lab == largest).astype(np.uint8)
    if dilate > 0:
        m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return m * 255


def box_report(m, w, h):
    ys, xs = np.where(m > 127)
    if not len(xs):
        return dict(empty=True)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return dict(box=[x0, y0, x1, y1], width=x1 - x0, height=y1 - y0, portrait=(y1 - y0) > (x1 - x0),
                touchesFrame=x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1, areaFraction=float((m > 127).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--frame", type=int,
                    help="source frame index; omit to let score_reference_frames.py choose")
    ap.add_argument("--out")
    ap.add_argument("--score-stride", type=int, default=3)
    ap.add_argument("--scores-out", help="where to keep the full per-frame scores")
    ap.add_argument("--estimator-probe", help="scripts/probe_pose_estimator.py output: candidates "
                                              "MultiHMR could not read are struck out before the pick")
    ap.add_argument("--clip-name", help="which clip to read from the probe file (default: video stem)")
    ap.add_argument("--candidates", help="comma-separated frame indices to report and skip writing")
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--method", default="segformer", choices=["segformer", "maskrcnn"])
    ap.add_argument("--sheet", help="write a contact sheet of candidate masks here")
    ap.add_argument("--bbox", help="x0,y0,x1,y1 in source pixels: keep the person instance overlapping this "
                                   "box instead of the highest-scoring one (multi-person clips)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    if a.candidates:
        tiles = []
        for i in [int(x) for x in a.candidates.split(",")]:
            rgb, _, _ = decode(a.video, i); m = person_mask(rgb, a.dilate, a.method, bbox)
            print(i, json.dumps(box_report(m, rgb.shape[1], rgb.shape[0])))
            if a.sheet:
                over = rgb.copy(); over[m > 127] = (0.5 * over[m > 127] + [127, 0, 0]).astype(np.uint8)
                cv2.putText(over, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                tiles.append(cv2.resize(over, (rgb.shape[1] // 2, rgb.shape[0] // 2)))
        if a.sheet and tiles:
            rows = [np.concatenate(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])), 1) for i in range(0, len(tiles), 3)]
            Image.fromarray(np.concatenate(rows, 0)).save(a.sheet)
        return
    if not a.out:
        sys.exit("--out is required")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.frame is None or a.scores_out:
        import score_reference_frames as srf
        scores = srf.score_video(a.video, stride=a.score_stride, n=1,
                                 out=a.scores_out or (Path(a.out) / "frame-scores.json"),
                                 probe=a.estimator_probe, clip_name=a.clip_name)
        if a.frame is None:
            a.frame = scores["chosenBest"]
            print(f"frame chosen by score: {a.frame} (score {scores['currentScores'].get(a.frame, 0):.3f} "
                  f"of {scores['scored']} scored)")
            if scores["estimatorProbe"]:
                print(f"  estimator-readable: {scores['chosenEstimatorReadable']}; "
                      f"rejected by the probe: {scores['estimatorRejected'] or 'none'}")
            else:
                print("  WARNING: estimator readability unverified; run scripts/probe_pose_estimator.py "
                      "and pass --estimator-probe, or the build may refuse this frame")
        else:
            by = {r["index"]: r for r in scores["rows"]}
            if a.frame in by:
                rank = 1 + sorted([q["score"] for q in scores["rows"]], reverse=True).index(by[a.frame]["score"])
                print(f"frame {a.frame} was given: score {by[a.frame]['score']:.3f}, rank {rank}/{len(scores['rows'])}; "
                      f"the scorer would pick {scores['chosenBest']}")
    if a.frame is None:
        sys.exit("no scorable frame found in this clip")
    rgb, fps, count = decode(a.video, a.frame); h, w = rgb.shape[:2]
    m = person_mask(rgb, a.dilate, a.method, bbox); rep = box_report(m, w, h)
    if rep.get("empty") or not rep["portrait"]:
        sys.exit(f"unusable person mask: {rep}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / "source.png"); Image.fromarray(m).save(out / "mask.png")
    meta = dict(video=str(Path(a.video).resolve()), sourceSha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),
                frameIndex=a.frame, time=a.frame / fps, sourceWidth=w, sourceHeight=h, fps=fps, duration=count / fps,
                personBounds=rep["box"], touchesFrame=rep["touchesFrame"],
                maskMethod=f"{a.method}: largest component, dilate {a.dilate}px" + (f", instance selected by IoU with track box {a.bbox}" if bbox else ""),
                trackBox=bbox)
    (out / "prepared.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
