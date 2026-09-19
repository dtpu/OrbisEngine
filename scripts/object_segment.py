#!/usr/bin/env python3
"""Segment any non-body, non-static object from a clip, given a word for it.

ROI selection supports the motion labels documented in docs/objects.md:

  attached / handoff   the object is on or in a person -> ROI from a body joint
  free                 the object is in the air        -> ROI from a 2D track
  worldDynamic         no person involved at all       -> ROI from a fixed box, or whole frame

Two things had to change to make it general.

1. **Region of interest, not a person band.** The worn version scored CLIPSeg only inside a
   dilated person mask, which is exactly wrong for a thrown bottle or a swinging door. An ROI is
   now supplied per frame by `--roi-*` and the person mask is optional context.
2. **Crop and zoom.** CLIPSeg runs at 352x352. A backpack filling a third of the frame survives
   that; a 50 px bottle in a 1920x1080 frame becomes 9 px and is invisible. Each frame is now
   cropped to its ROI and the crop is upsampled before CLIPSeg and SAM see it, so a small object
   gets the network's full resolution. This is what makes held and thrown objects work.

3. **Proposal-and-score, not heatmap-and-threshold.** `--method clipseg` is the worn tool's
   original path: CLIPSeg heatmap -> box -> SAM. It fails on small objects for a reason worth
   writing down: on this clip's held bottle CLIPSeg confidently returned *the hand* at peak 0.96,
   because at 30 px the bottle and the fist are one blob to it. `--method sam-clip` (the default)
   inverts the order — SAM proposes every region in the ROI from a point grid, and each proposal
   is cut out and scored by full CLIP against the prompt and its negatives. Every candidate gets
   CLIP's whole 224 px input to itself, so a 30 px object is judged at 224 px, not at 9 px.

SegFormer-b0 ADE20K class 12 still supplies the person mask, and the temporal area gate still
rejects a frame rather than guessing it.

Outputs (per frame, at full frame resolution): object/, person/, dynamic/ (= person OR object)
mask PNGs, plus object-crop/ RGBA cutouts at native crop resolution for the appearance stage,
an overlay/ to look at, and segment.json.

`dynamic/` is the mask the static-scene trainer should subtract. `object/` is what the rigid
reconstruction consumes. Keeping them separate is the survey's one structural finding:
"what is the person" and "what moved" are different questions.
"""

import argparse, json, os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label as cc_label

PERSON = 12
SEGFORMER = "nvidia/segformer-b0-finetuned-ade-512-512"
CLIPSEG = "CIDAS/clipseg-rd64-refined"
SAM = "facebook/sam-vit-base"
CLIP = "openai/clip-vit-base-patch32"


def _as_tensor(out):
    """transformers changed get_*_features to return an output object in some versions."""
    if hasattr(out, "norm"):
        return out
    po = getattr(out, "pooler_output", None)
    return po if po is not None else out[0]


class Models:
    def __init__(self, threads=4, sam=SAM, need_clipseg=True, need_clip=True):
        import torch

        torch.set_num_threads(threads)
        from transformers import (
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
            SamModel,
            SamProcessor,
        )

        self.torch = torch
        self.seg_proc = AutoImageProcessor.from_pretrained(SEGFORMER)
        self.seg = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam)
        self.sam = SamModel.from_pretrained(sam).eval()
        self.clip_proc = self.clip = None
        if need_clipseg:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

            self.clip_proc = CLIPSegProcessor.from_pretrained(CLIPSEG)
            self.clip = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG).eval()
        self.cl_proc = self.cl = None
        if need_clip:
            from transformers import CLIPModel, CLIPProcessor

            self.cl_proc = CLIPProcessor.from_pretrained(CLIP)
            self.cl = CLIPModel.from_pretrained(CLIP).eval()
        self._txt_cache = {}

    # -- SAM with the image encoded once and many cheap point decodes -------------
    def sam_embed(self, im):
        inp = self.sam_proc(im, return_tensors="pt")
        with self.torch.no_grad():
            emb = self.sam.get_image_embeddings(inp["pixel_values"])
        return inp, emb

    def sam_points(self, im, inp, emb, points):
        """points: list of (x, y) in `im` pixels.

        One image, len(points) independent single-point prompts. The processor wants
        (batch, point_batch, nb_points, 2), so the nesting below is load-bearing: flattening it
        to (N, 1, 2) silently makes transformers read N as the *image* batch and the prompts
        stop corresponding to the image.
        """
        pi = self.sam_proc(
            im, input_points=[[[[float(x), float(y)]] for x, y in points]], return_tensors="pt"
        )
        with self.torch.no_grad():
            out = self.sam(
                input_points=pi["input_points"], image_embeddings=emb, multimask_output=True
            )
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), pi["original_sizes"].cpu(), pi["reshaped_input_sizes"].cpu()
        )[0]
        return masks.numpy().astype(bool), out.iou_scores[0].detach().numpy()

    def text_features(self, prompts):
        key = tuple(prompts)
        if key not in self._txt_cache:
            ti = self.cl_proc(text=list(prompts), return_tensors="pt", padding=True)
            with self.torch.no_grad():
                tf = _as_tensor(self.cl.get_text_features(**ti))
            self._txt_cache[key] = tf / tf.norm(dim=-1, keepdim=True)
        return self._txt_cache[key]

    def clip_score(self, crops, prompts):
        """-> (len(crops), len(prompts)) raw cosine similarity.

        Deliberately not CLIP's logit-scaled softmax: at scale 100 a cut-out that is 60 % hand and
        40 % bottle scores 0.00 for "bottle", so every candidate collapses to zero and the ranking
        becomes noise. The margin `sim(prompt) - max(sim(negatives))` ranks correctly instead.
        """
        tf = self.text_features(prompts)
        out = []
        for i in range(0, len(crops), 32):
            ii = self.cl_proc(images=crops[i : i + 32], return_tensors="pt")
            with self.torch.no_grad():
                imf = _as_tensor(self.cl.get_image_features(**ii))
                imf = imf / imf.norm(dim=-1, keepdim=True)
                out.append((imf @ tf.T).numpy())
        return np.concatenate(out, 0)

    def person(self, im):
        import torch.nn.functional as F

        a = np.asarray(im)
        with self.torch.no_grad():
            logits = self.seg(**self.seg_proc(images=[a], return_tensors="pt")).logits
            up = F.interpolate(
                logits.float(), size=a.shape[:2], mode="bilinear", align_corners=False
            )
        return up.argmax(1)[0].numpy() == PERSON

    def clipseg(self, im, prompts):
        """-> (len(prompts), H, W) probabilities at `im`'s resolution."""
        import torch.nn.functional as F

        with self.torch.no_grad():
            inp = self.clip_proc(
                text=prompts, images=[im] * len(prompts), padding=True, return_tensors="pt"
            )
            logits = self.clip(**inp).logits
        if logits.ndim == 2:
            logits = logits[None]
        up = F.interpolate(
            self.torch.sigmoid(logits)[:, None],
            size=(im.size[1], im.size[0]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return up.numpy()

    def sam_masks(self, im, box, pos, neg):
        points = [list(map(float, p)) for p in pos] + [list(map(float, p)) for p in neg]
        labels = [1] * len(pos) + [0] * len(neg)
        kw = dict(input_boxes=[[list(map(float, box))]])
        if points:
            kw["input_points"] = [[points]]
            kw["input_labels"] = [[labels]]
        inp = self.sam_proc(im, return_tensors="pt", **kw)
        with self.torch.no_grad():
            out = self.sam(**inp)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
        )[0][0].numpy()
        return masks, out.iou_scores[0][0].numpy()


def largest_component(m):
    if not m.any():
        return m
    lab, n = cc_label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def bbox_of(m, pad, shape):
    ys, xs = np.nonzero(m)
    H, W = shape
    return [
        int(max(0, xs.min() - pad)),
        int(max(0, ys.min() - pad)),
        int(min(W - 1, xs.max() + pad)),
        int(min(H - 1, ys.max() + pad)),
    ]


# ---------------------------------------------------------------- ROI sources


def roi_from_joint(tracks_dir, track, joint, radius, frames):
    """ROI centred on a projected body joint, interpolated to every source frame.

    This is the attached/handoff case: a held or worn object is near a joint by definition.
    """
    m = json.load(open(os.path.join(tracks_dir, f"track_{track:02d}", "motion.json")))
    si = np.array([f["sourceIndex"] for f in m["frames"]], float)
    xy = np.array([f["projectedBodyJoints"][joint] for f in m["frames"]], float)
    return {
        int(f): (
            float(np.interp(f, si, xy[:, 0])),
            float(np.interp(f, si, xy[:, 1])),
            float(radius),
        )
        for f in frames
    }


def roi_from_track2d(path, radius, frames):
    """ROI from an existing 2D object track (scripts/track_object_2d.py). The free case."""
    t = json.load(open(path))
    pts = (
        t["observations"]
        if isinstance(t, dict) and "observations" in t
        else (t["track"] if isinstance(t, dict) else t)
    )
    si = np.array([p["sourceIndex"] if isinstance(p, dict) else p[0] for p in pts], float)
    xs = np.array([p["x"] if isinstance(p, dict) else p[1] for p in pts], float)
    ys = np.array([p["y"] if isinstance(p, dict) else p[2] for p in pts], float)
    lo, hi = si.min(), si.max()
    return {
        int(f): (float(np.interp(f, si, xs)), float(np.interp(f, si, ys)), float(radius))
        for f in frames
        if lo <= f <= hi
    }


def roi_from_box(box, frames):
    """A fixed image region. The worldDynamic case: a doorway, a lift car, a corner of the room."""
    x0, y0, x1, y1 = box
    c = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(x1 - x0, y1 - y0) / 2.0)
    return {int(f): c for f in frames}


# ---------------------------------------------------------------- seed points


def propagate_seeds(get, seeds, frames, win=21, levels=4):
    """Carry a few known object points to every frame with Lucas-Kanade, outward from each seed.

    Why this exists: CLIP ranks a 400 px backpack cut-out confidently and a 30 px blurred
    translucent bottle not at all — on this clip every proposal scored between -0.08 and +0.01,
    which is noise. A word narrows the search; a point pins it. The points do not have to be hand
    placed: one detection anywhere in a run (from `track_object_2d.py`, or one click) propagates
    to the rest of the run by optical flow, which is cheap and needs no model.

    seeds: {sourceFrame: (x, y)}. Returns {sourceFrame: (x, y)} over `frames`.
    """
    import cv2

    out = dict(seeds)
    order = sorted(frames)
    gray = {}

    def g(f):
        if f not in gray:
            im = get(f)
            gray[f] = None if im is None else cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
            for k in [k for k in gray if abs(k - f) > 3]:
                gray.pop(k)
        return gray[f]

    for direction in (1, -1):
        seq = order if direction > 0 else order[::-1]
        cur = None
        for i, f in enumerate(seq):
            if f in seeds:
                cur = seeds[f]
                continue
            if cur is None:
                continue
            prev = seq[i - 1]
            a, b = g(prev), g(f)
            if a is None or b is None:
                cur = None
                continue
            p0 = np.array([[cur]], np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                a, b, p0, None, winSize=(win, win), maxLevel=levels
            )
            if st is None or not st.reshape(-1)[0]:
                cur = None
                continue
            cur = (float(p1[0, 0, 0]), float(p1[0, 0, 1]))
            if f in out:  # both directions reached it: average them
                out[f] = ((out[f][0] + cur[0]) / 2.0, (out[f][1] + cur[1]) / 2.0)
            else:
                out[f] = cur
    return out


# ---------------------------------------------------------------- proposal scoring


def point_grid(w, h, n):
    xs = (np.arange(n) + 0.5) / n * w
    ys = (np.arange(n) + 0.5) / n * h
    return [(float(x), float(y)) for y in ys for x in xs]


def dedupe(cands, iou_thresh=0.85):
    keep = []
    for m in cands:
        a = m.sum()
        if a == 0:
            continue
        dup = False
        for k in keep:
            inter = float((m & k).sum())
            if inter / max(float((m | k).sum()), 1.0) > iou_thresh:
                dup = True
                break
        if not dup:
            keep.append(m)
    return keep


def cutout_for_clip(rgb, m, pad=0.25, bg=(124, 116, 104)):
    """Tight cut-out on a neutral field, square, so CLIP judges the object and not its surroundings."""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad)) or 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[:] = bg
    sx0, sy0 = cx - side // 2, cy - side // 2
    ox0, oy0 = max(0, -sx0), max(0, -sy0)
    rx0, ry0 = max(0, sx0), max(0, sy0)
    rx1, ry1 = min(rgb.shape[1], sx0 + side), min(rgb.shape[0], sy0 + side)
    if rx1 <= rx0 or ry1 <= ry0:
        return Image.fromarray(canvas)
    sub = rgb[ry0:ry1, rx0:rx1].copy()
    sm = m[ry0:ry1, rx0:rx1]
    sub[~sm] = bg
    canvas[oy0 : oy0 + (ry1 - ry0), ox0 : ox0 + (rx1 - rx0)] = sub
    return Image.fromarray(canvas)


def propose_and_score(M, big, prompts, args, prior=None, prev_centroid=None, seed_xy=None):
    """SAM proposes, CLIP judges. Returns (mask, diagnostics) or (None, diagnostics).

    `seed_xy`, in `big` pixels, is a point known to be ON the object. When present it is added as a
    SAM prompt and every proposal that misses it is discarded, which turns CLIP's job from
    "find the bottle" into "of the three nested regions under this point, which one is the bottle".
    """
    bw, bh = big.size
    rgb = np.asarray(big)
    inp, emb = M.sam_embed(big)
    pts = point_grid(bw, bh, args.grid)
    if seed_xy is not None:
        sx, sy = seed_xy
        pts = [(sx, sy), (sx - 3, sy), (sx + 3, sy), (sx, sy - 3), (sx, sy + 3)] + pts
    cands = []
    for i in range(0, len(pts), 32):
        masks, _ = M.sam_points(big, inp, emb, pts[i : i + 32])
        for mm in masks:  # (3, H, W): SAM's part / sub-object / object
            for k in range(mm.shape[0]):
                m = binary_fill_holes(largest_component(mm[k]))
                a = m.sum()
                if args.min_area_px <= a <= args.max_area_frac * bw * bh:
                    cands.append(m)
    if seed_xy is not None:
        sx, sy = int(np.clip(seed_xy[0], 0, bw - 1)), int(np.clip(seed_xy[1], 0, bh - 1))
        cands = [m for m in cands if m[sy, sx]]
    cands = dedupe(cands, args.dedupe_iou)
    diag = dict(proposals=len(cands))
    if not cands:
        return None, diag
    crops = [cutout_for_clip(rgb, m) for m in cands]
    sims = M.clip_score(crops, prompts)
    probs = sims[:, 0] - (sims[:, 1:].max(axis=1) if sims.shape[1] > 1 else 0.0)
    score = probs.copy()
    if prior is not None:
        # CLIPSeg prior, an additive nudge rather than a gate
        ov = np.array([float((m & prior).sum()) / max(float(m.sum()), 1.0) for m in cands])
        score = score + args.prior_weight * ov
    if prev_centroid is not None:
        cs = np.array([[np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()] for m in cands])
        d = np.linalg.norm(cs - np.array(prev_centroid), axis=1) / max(bw, bh)
        score = score * np.exp(-((d / max(args.prev_sigma, 1e-3)) ** 2))
    order = np.argsort(-score)
    k = int(order[0])
    diag.update(
        clipProb=round(float(probs[k]), 4),
        combined=round(float(score[k]), 4),
        bestPx=int(cands[k].sum()),
        top=[
            dict(
                px=int(cands[j].sum()),
                clip=round(float(probs[j]), 3),
                combined=round(float(score[j]), 3),
            )
            for j in order[:5]
        ],
    )
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        n = max(1, len(order[:8]))
        cv = Image.new("RGB", (n * 160, 160), (10, 10, 10))
        for i, j in enumerate(order[:8]):
            cv.paste(crops[j].resize((160, 160)), (i * 160, 0))
        d = ImageDraw.Draw(cv)
        for i, j in enumerate(order[:8]):
            d.text((i * 160 + 4, 4), f"{probs[j]:+.3f}", fill=(255, 255, 0))
        cv.save(os.path.join(args.debug_dir, f"{args.debug_tag}.png"))
    if probs[k] < args.clip_floor:
        diag["reject"] = f"clip margin {probs[k]:.3f} below floor"
        return None, diag
    return cands[k], diag


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clip", help="source video; frames are decoded on demand")
    ap.add_argument("--images", help="directory of frames instead of --clip")
    ap.add_argument(
        "--frames", default="", help='"a:b" or "a,b,c" source frame indices (default: all)'
    )
    ap.add_argument(
        "--prompt",
        required=True,
        help='what the object is, in words: "a clear plastic water bottle"',
    )
    ap.add_argument("--negative-prompt", default="a person, a hand")
    ap.add_argument("--out", required=True)

    ap.add_argument(
        "--roi-joint", default="", help="TRACK:JOINT:RADIUS, e.g. 1:21:110 (right wrist)"
    )
    ap.add_argument("--roi-track2d", default="", help="PATH:RADIUS from scripts/track_object_2d.py")
    ap.add_argument("--roi-box", default="", help="x0,y0,x1,y1 fixed region")
    ap.add_argument("--tracks-dir", default="")
    ap.add_argument(
        "--roi-zoom",
        type=float,
        default=0.0,
        help="upsample the ROI crop to this many px on its long side before CLIPSeg/SAM "
        "(0 = auto: 352 px, which is CLIPSeg native)",
    )

    ap.add_argument(
        "--method",
        default="sam-clip",
        choices=["sam-clip", "clipseg"],
        help="sam-clip: SAM proposes, CLIP judges (default; right for small objects). "
        "clipseg: the worn tool's heatmap path, cheaper, right for large ones.",
    )
    ap.add_argument(
        "--seed-json",
        default="",
        help='{"<sourceFrame>": [x, y]} points known to be on the object, in full-frame '
        "pixels. Propagated to every frame by optical flow.",
    )
    ap.add_argument(
        "--seed-from-track2d",
        default="",
        help="PATH: take every observation in a track_object_2d.py file as a seed",
    )
    ap.add_argument("--seed-lk-win", type=int, default=21)
    ap.add_argument("--debug-dir", default="", help="dump the top proposals CLIP saw, per frame")
    ap.add_argument(
        "--grid", type=int, default=8, help="sam-clip: NxN point prompts across the ROI"
    )
    ap.add_argument(
        "--clip-floor",
        type=float,
        default=0.01,
        help="sam-clip: minimum CLIP margin, sim(prompt) - max sim(negatives)",
    )
    ap.add_argument(
        "--max-area-frac",
        type=float,
        default=0.35,
        help="sam-clip: largest proposal, as a fraction of the ROI",
    )
    ap.add_argument("--dedupe-iou", type=float, default=0.85)
    ap.add_argument(
        "--prior-weight",
        type=float,
        default=0.0,
        help="sam-clip: weight of the CLIPSeg prior, 0 disables it",
    )
    ap.add_argument(
        "--prev-sigma",
        type=float,
        default=0.35,
        help="sam-clip: centroid continuity, in ROI widths",
    )
    ap.add_argument(
        "--require-person",
        action="store_true",
        help="skip frames with no person; only meaningful for attached/handoff",
    )
    ap.add_argument("--clipseg-floor", type=float, default=0.25)
    ap.add_argument("--max-area-jump", type=float, default=2.5)
    ap.add_argument(
        "--min-area-px", type=int, default=60, help="in ROI-crop pixels, before rescaling back"
    )
    ap.add_argument("--sheet-every", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sam", default=SAM)
    a = ap.parse_args()

    if a.frames and ":" in a.frames:
        lo, hi = a.frames.split(":")
        frames = list(range(int(lo), int(hi) + 1))
    elif a.frames:
        frames = [int(v) for v in a.frames.split(",")]
    else:
        frames = None

    import cv2

    if a.clip:
        cap = cv2.VideoCapture(a.clip)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames is None:
            frames = list(range(total))

        def get(f):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, bgr = cap.read()
            return Image.fromarray(bgr[:, :, ::-1]) if ok else None
    else:
        names = sorted(
            n for n in os.listdir(a.images) if n.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if frames is None:
            frames = list(range(len(names)))

        def get(f):
            return (
                Image.open(os.path.join(a.images, names[f])).convert("RGB")
                if f < len(names)
                else None
            )

    if a.roi_joint:
        t, j, r = a.roi_joint.split(":")
        rois = roi_from_joint(a.tracks_dir, int(t), int(j), float(r), frames)
    elif a.roi_track2d:
        p, r = a.roi_track2d.rsplit(":", 1)
        rois = roi_from_track2d(p, float(r), frames)
    elif a.roi_box:
        rois = roi_from_box([float(v) for v in a.roi_box.split(",")], frames)
    else:
        rois = {}

    seeds = {}
    if a.seed_json:
        seeds.update({int(k): tuple(v) for k, v in json.load(open(a.seed_json)).items()})
    if a.seed_from_track2d:
        t = json.load(open(a.seed_from_track2d))
        for o in t.get("observations", t if isinstance(t, list) else []):
            seeds[int(o["sourceIndex"])] = (float(o["x"]), float(o["y"]))
    seed_pts = propagate_seeds(get, seeds, frames, a.seed_lk_win) if seeds else {}
    if seeds:
        print(f"{len(seeds)} seed(s) -> {len(seed_pts)} frames by optical flow", flush=True)

    for d in ("person", "object", "dynamic", "overlay", "object-crop"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    prompts = [a.prompt] + [p.strip() for p in a.negative_prompt.split(",") if p.strip()]
    M = Models(
        a.threads,
        a.sam,
        need_clipseg=(a.method == "clipseg" or a.prior_weight > 0),
        need_clip=(a.method == "sam-clip"),
    )
    rows, areas, sheet, prev_centroid = [], [], [], None
    zoom_to = a.roi_zoom or 352.0

    for f in frames:
        im = get(f)
        if im is None:
            continue
        W, H = im.size
        row = dict(sourceFrame=int(f))

        if f in rois:
            cx, cy, r = rois[f]
            x0 = int(np.clip(cx - r, 0, W - 1))
            x1 = int(np.clip(cx + r, 1, W))
            y0 = int(np.clip(cy - r, 0, H - 1))
            y1 = int(np.clip(cy + r, 1, H))
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        if x1 - x0 < 8 or y1 - y0 < 8:
            row["skipped"] = "roi degenerate"
            rows.append(row)
            continue
        row["roi"] = [x0, y0, x1, y1]

        crop = im.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        s = max(1.0, zoom_to / max(cw, ch))
        big = (
            crop.resize((int(round(cw * s)), int(round(ch * s))), Image.LANCZOS)
            if s > 1.0
            else crop
        )
        row["roiZoom"] = round(float(s), 2)

        person_full = M.person(im)
        row["person_px"] = int(person_full.sum())
        if a.require_person and person_full.sum() < 500:
            row["skipped"] = "no person"
            rows.append(row)
            continue

        prior = None
        if a.method == "clipseg" or a.prior_weight > 0:
            probs = M.clipseg(big, prompts[:2] if len(prompts) > 1 else prompts)
            score = np.clip(probs[0] - (0.5 * probs[1] if len(probs) > 1 else 0.0), 0, 1)
            peak = float(score.max())
            row["clipseg_peak"] = round(peak, 4)
            if peak >= a.clipseg_floor:
                prior = largest_component(score > max(0.5 * peak, a.clipseg_floor))
            elif a.method == "clipseg":
                row["skipped"] = f"clipseg peak {peak:.3f} below floor"
                rows.append(row)
                continue

        bw, bh = big.size
        if a.method == "sam-clip":
            a.debug_tag = f"{f:05d}"
            sxy = None
            if f in seed_pts:
                sxy = ((seed_pts[f][0] - x0) * s, (seed_pts[f][1] - y0) * s)
                if not (0 <= sxy[0] < big.size[0] and 0 <= sxy[1] < big.size[1]):
                    row["skipped"] = "seed outside roi"
                    rows.append(row)
                    continue
                row["seed"] = [round(seed_pts[f][0], 1), round(seed_pts[f][1], 1)]
            best, diag = propose_and_score(M, big, prompts, a, prior, prev_centroid, sxy)
            row.update(diag)
            if best is None:
                row["skipped"] = diag.get("reject", "no proposal")
                rows.append(row)
                continue
        else:
            seed = prior
            if seed is None or seed.sum() < 12:
                row["skipped"] = "clipseg seed too small"
                rows.append(row)
                continue
            box = bbox_of(seed, 6, (bh, bw))
            ys, xs = np.nonzero(seed)
            order = np.argsort(-score[ys, xs])[:3]
            pos = [(float(xs[k]), float(ys[k])) for k in order]
            neg = [(4.0, 4.0), (bw - 5.0, 4.0), (4.0, bh - 5.0), (bw - 5.0, bh - 5.0)]
            neg = [
                pnt
                for pnt in neg
                if not seed[int(np.clip(pnt[1], 0, bh - 1)), int(np.clip(pnt[0], 0, bw - 1))]
            ]
            masks, sam_scores = M.sam_masks(big, box, pos, neg)
            best, best_iou = None, -1.0
            for m, sc in zip(masks, sam_scores):
                m = binary_fill_holes(largest_component(m.astype(bool)))
                inter = float((m & seed).sum())
                union = float((m | seed).sum()) or 1.0
                iou = inter / union
                if m.sum() > 0.75 * bw * bh:
                    iou *= 0.1
                if iou > best_iou:
                    best, best_iou = m, iou
            row["sam_iou_with_clipseg"] = round(best_iou, 3)
        if best is None or best.sum() < a.min_area_px:
            row["skipped"] = f"object {0 if best is None else int(best.sum())} px"
            rows.append(row)
            continue

        med = float(np.median(areas)) if areas else float(best.sum() / (s * s))
        ratio = (best.sum() / (s * s)) / max(med, 1.0)
        row["area_ratio_to_median"] = round(float(ratio), 3)
        if areas and (ratio > a.max_area_jump or ratio < 1.0 / a.max_area_jump):
            row["skipped"] = f"area jumped {ratio:.2f}x against the running median"
            rows.append(row)
            continue
        areas.append(float(best.sum() / (s * s)))

        # RGBA cutout at crop-native resolution: the appearance stage's input
        crop_mask = (
            np.asarray(
                Image.fromarray((best * 255).astype(np.uint8)).resize((cw, ch), Image.BILINEAR)
            )
            > 127
        )
        cut = np.dstack([np.asarray(crop), (crop_mask * 255).astype(np.uint8)])
        ys2, xs2 = np.nonzero(crop_mask)
        if len(xs2) < 8:
            row["skipped"] = "cutout empty after rescale"
            rows.append(row)
            continue
        tb = [xs2.min(), ys2.min(), xs2.max() + 1, ys2.max() + 1]
        Image.fromarray(cut[tb[1] : tb[3], tb[0] : tb[2]]).save(
            os.path.join(a.out, "object-crop", f"{f:05d}.png")
        )

        obj_full = np.zeros((H, W), bool)
        obj_full[y0:y1, x0:x1] = crop_mask
        dynamic = person_full | obj_full
        person_only = person_full & ~obj_full
        stem = f"{f:05d}"
        for sub, m in (("person", person_only), ("object", obj_full), ("dynamic", dynamic)):
            Image.fromarray((m * 255).astype(np.uint8)).save(
                os.path.join(a.out, sub, stem + ".png")
            )

        bys, bxs = np.nonzero(best)
        prev_centroid = (float(bxs.mean()), float(bys.mean()))
        cys, cxs = np.nonzero(obj_full)
        row.update(
            object_px=int(obj_full.sum()),
            bbox=bbox_of(obj_full, 0, (H, W)),
            centroid=[round(float(cxs.mean()), 1), round(float(cys.mean()), 1)],
            overlap_with_person_px=int((obj_full & person_full).sum()),
            cropBboxInRoi=[int(v) for v in tb],
        )

        ov = np.asarray(crop).copy()
        edge = binary_dilation(crop_mask, iterations=2) & ~binary_erosion(crop_mask, iterations=2)
        ov[crop_mask] = (0.45 * ov[crop_mask] + 0.55 * np.array([220, 50, 47])).astype(np.uint8)
        ov[edge] = (255, 255, 0)
        ovim = Image.fromarray(ov)
        ovim.save(os.path.join(a.out, "overlay", stem + ".png"))
        if a.sheet_every and len(rows) % a.sheet_every == 0:
            sheet.append(ovim.resize((220, 220)))
        rows.append(row)
        print(
            f"f{f}: object {int(obj_full.sum())} px  {row.get('clipProb', row.get('sam_iou_with_clipseg'))}",
            flush=True,
        )

    if sheet:
        cols = min(8, len(sheet))
        rn = (len(sheet) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * 220, rn * 220), (20, 20, 20))
        for k, s2 in enumerate(sheet):
            canvas.paste(s2, ((k % cols) * 220, (k // cols) * 220))
        canvas.save(os.path.join(a.out, "overlay-sheet.jpg"), quality=92)

    kept = [r for r in rows if "object_px" in r]
    summary = dict(
        prompt=a.prompt,
        negativePrompt=a.negative_prompt,
        clip=a.clip or a.images,
        frames=len(rows),
        framesWithObject=len(kept),
        medianObjectPx=float(np.median([r["object_px"] for r in kept])) if kept else 0.0,
        medianOverlapWithPerson=float(np.median([r["overlap_with_person_px"] for r in kept]))
        if kept
        else 0.0,
        rejects={},
        perFrame=rows,
    )
    for r in rows:
        if "skipped" in r:
            k = r["skipped"].split(" ")[0]
            summary["rejects"][k] = summary["rejects"].get(k, 0) + 1
    with open(os.path.join(a.out, "segment.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "perFrame"}, indent=2))


if __name__ == "__main__":
    main()
