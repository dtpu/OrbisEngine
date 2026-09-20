"""People masks for still-scene fusion (Track B): a person who moved during the clip is not part
of the static scene, so drop their pixels before exporting points. SegFormer-b0 (ADE20K) on CPU
or MPS, ~0.1 s per frame. ADE20K class 12 = person.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
PERSON = 12
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).eval()
        dev = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        _bundle = (proc, model.to(dev), dev)
    return _bundle


def people_masks(
    images: np.ndarray, dilate_px: int = 6, classes=(PERSON,), batch_size: int = 8
) -> np.ndarray:
    """images uint8 (N,H,W,3) -> bool (N,H,W), True where a person is (dilated)."""
    import torch
    import torch.nn.functional as F

    proc, model, dev = _load()
    N, H, W, _ = images.shape
    out = np.zeros((N, H, W), dtype=bool)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = [images[j] for j in range(i, min(N, i + batch_size))]
            inputs = proc(images=batch, return_tensors="pt").to(dev)
            logits = model(**inputs).logits  # (b, C, h/4, w/4)
            # upsample one frame at a time: 150 classes at 1080p is ~1.2 GB per frame
            for b in range(len(batch)):
                up = F.interpolate(
                    logits[b : b + 1].float(), size=(H, W), mode="bilinear", align_corners=False
                )
                lab = up.argmax(1).cpu().numpy()
                out[i + b] = np.isin(lab, classes)[0]
    if dilate_px > 0:
        try:
            import cv2

            k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
            for i in range(N):
                out[i] = cv2.dilate(out[i].astype(np.uint8), k).astype(bool)
        except Exception:
            pass
    return out


# --- moved-content mask -------------------------------------------------------------------
# people_masks answers "what is the person". The static-scene trainer and the inpainter need
# the other question, "what moved": a pulled suitcase, an open umbrella, a cast shadow, a
# swinging door and a passer-by all move and none of them is ADE20K class 12, so today they are
# excluded from the person layer and left, moving, in the static plate. moved_content_masks
# answers the second question. Nothing calls it unless asked; people_masks is unchanged.

# ADE20K ids that CAN move. They are not dynamic by themselves - a parked car and a shut door
# are scene - so a component of one of these classes is only taken when the residual says it
# moved or it touches the person (something carried).
MOVABLE = (
    12,  # person
    20,  # car
    55,  # case  (suitcase, the survey's worst class)
    76,  # boat
    80,  # bus
    83,  # truck
    90,  # airplane
    102,  # van
    103,  # ship
    108,  # plaything
    114,  # tent  (SegFormer labels an open umbrella "tent")
    115,  # bag
    116,  # minibike
    119,  # ball
    126,  # animal
    127,  # bicycle
    14,  # door
    58,  # screen door
)

# the subset a person can plausibly hold, wear or lead. Only these may enter the mask on contact
# alone; everything else in MOVABLE has to be caught moving, or a doorway-sized wall of "door"
# pixels touching a passer-by is swallowed whole.
CARRIABLE = (55, 108, 114, 115, 119, 126)
# classes that are usually in motion when present: never seed the camera fit on them
UNSTABLE = (12, 20, 76, 80, 83, 90, 102, 103, 116, 127) + CARRIABLE


def segment_labels(images: np.ndarray, batch_size: int = 8) -> np.ndarray:
    """images uint8 (N,H,W,3) -> uint8 (N,H,W) ADE20K class ids, full resolution."""
    import torch
    import torch.nn.functional as F

    proc, model, dev = _load()
    N, H, W, _ = images.shape
    out = np.zeros((N, H, W), dtype=np.uint8)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = [images[j] for j in range(i, min(N, i + batch_size))]
            inputs = proc(images=batch, return_tensors="pt").to(dev)
            logits = model(**inputs).logits
            for b in range(len(batch)):
                up = F.interpolate(
                    logits[b : b + 1].float(), size=(H, W), mode="bilinear", align_corners=False
                )
                out[i + b] = up.argmax(1).cpu().numpy()[0].astype(np.uint8)
    return out


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    import cv2

    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _homography(gi: np.ndarray, gj: np.ndarray, ignore: np.ndarray):
    """j -> i, from LK tracks seeded outside `ignore`. None if it cannot be estimated."""
    import cv2

    ok = (~ignore).astype(np.uint8) * 255
    pts = cv2.goodFeaturesToTrack(gj, maxCorners=900, qualityLevel=0.01, minDistance=7, mask=ok)
    if pts is None or len(pts) < 30:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(gj, gi, pts, None, winSize=(21, 21), maxLevel=3)
    st = st.reshape(-1).astype(bool)
    if st.sum() < 30:
        return None, 0
    Hm, inl = cv2.findHomography(pts[st], nxt[st], cv2.RANSAC, 2.0, maxIters=2000)
    if Hm is None:
        return None, 0
    return Hm, int(inl.sum()) if inl is not None else 0


def moved_content_masks(
    images: np.ndarray,
    dilate_px: int = 6,
    classes=(PERSON,),
    batch_size: int = 8,
    work_px: int = 640,
    gaps=(2, 5),
    resid_k: float = 4.0,
    resid_floor: float = 6.0,
    resid_norm: float = 2.0,
    grad_floor: float = 8.0,
    open_px: int = 2,
    min_area_frac: float = 3e-4,
    grow_px: int | None = None,
    shadow: bool = True,
    shadow_radius_frac: float = 1.0,
    max_comp_frac: float = 0.12,
    max_added_frac: float = 0.05,
    extra_classes: bool = True,
    warp_fn=None,
    far_px: int = 60,
) -> dict:
    """images uint8 (N,H,W,3) -> {"person", "moved", "residual", "stats"}.

    "person" is exactly what people_masks returns today, so a caller can hand the person layer
    the same pixels it has always had. "moved" is person UNION what moved, from three signals:

      1. frame differencing compensated for camera motion. A neighbour frame at +-gap is warped
         into this frame by a homography fitted to tracks OUTSIDE the person, and the residual
         is the per-pixel MIN over the neighbours, which cancels disocclusion (background
         revealed on one side only). Two gaps are used and maxed, because a slow subject
         overlaps itself at a short gap. Pass warp_fn(i, j, img_j) -> (warped, valid) to
         substitute the principled version where camera poses and depth exist: a static-scene
         reprojection handles parallax, which a homography cannot.
      2. the segmenter's other movable classes, but only where signal 1 or contact with the
         person says that component actually moved.
      3. cast shadow: near the person, darker than the warped reference with its chromaticity
         preserved. The shadow is attached to the subject, moves with it, is never class 12,
         and is what poisons the static plate most visibly.

    Every added region is opened, area-filtered and grown by the same dilation the person gets,
    so a single noisy pixel cannot enter the mask. stats[i] reports how much each signal added
    and how much of it sits far from the person (a passer-by or a door - or over-segmentation).
    """
    import cv2

    N, H, W, _ = images.shape
    grow = max(2, dilate_px // 4) if grow_px is None else grow_px
    labels = segment_labels(images, batch_size=batch_size)
    person = np.stack([_dilate(np.isin(labels[i], classes), dilate_px) for i in range(N)])

    w = min(work_px, W)
    h = max(1, int(round(H * w / W)))
    sw, sh = w / W, h / H
    small = np.stack(
        [cv2.resize(images[i], (w, h), interpolation=cv2.INTER_AREA) for i in range(N)]
    )
    gray = np.stack([cv2.cvtColor(small[i], cv2.COLOR_RGB2GRAY) for i in range(N)])
    pers_s = np.stack(
        [
            cv2.resize(person[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(
                bool
            )
            for i in range(N)
        ]
    )
    lab_s = np.stack(
        [cv2.resize(labels[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in range(N)]
    )
    # never seed the camera-motion fit on something that can move: a tracked suitcase filling the
    # frame will otherwise win the RANSAC and the fit locks onto the object, not the scene
    seed_out = np.stack([pers_s[i] | np.isin(lab_s[i], UNSTABLE) for i in range(N)])
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
    grow_s = max(1, int(round(grow * sw)))
    grow_k = np.ones((2 * grow_s + 1, 2 * grow_s + 1), np.uint8)
    min_area = max(12, int(min_area_frac * w * h))
    far_s = max(2, int(round(far_px * sw)))
    contact_s = max(3, int(round(0.012 * max(h, w))))

    moved = np.zeros((N, H, W), bool)
    resid_out = np.zeros((N, h, w), bool)
    stats = []
    INF = np.float32(1e9)
    for i in range(N):
        gi = gray[i].astype(np.float32)
        best = np.full((h, w), -1.0, np.float32)  # -1 = never measured
        ref = small[i].astype(np.float32)  # brightest warped neighbour = unshadowed background
        ref_lum = np.full((h, w), -1.0, np.float32)
        fits = 0
        for gap in gaps:
            per_gap = np.full((h, w), INF, np.float32)
            got = np.zeros((h, w), bool)
            for j in (i - gap, i + gap):
                if j < 0 or j >= N:
                    continue
                if warp_fn is not None:
                    wj, valid_j = warp_fn(i, j, small[j])
                    wc = np.asarray(wj, np.float32)
                    wg = cv2.cvtColor(wc.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                else:
                    Hm, _ = _homography(gray[i], gray[j], seed_out[i] | seed_out[j])
                    if Hm is None:
                        continue
                    fits += 1
                    wg = cv2.warpPerspective(
                        gray[j].astype(np.float32), Hm, (w, h), flags=cv2.INTER_LINEAR
                    )
                    wc = cv2.warpPerspective(
                        small[j].astype(np.float32), Hm, (w, h), flags=cv2.INTER_LINEAR
                    )
                    valid_j = cv2.warpPerspective(np.ones((h, w), np.float32), Hm, (w, h)) > 0.99
                d = np.abs(gi - wg)
                # MIN over the two sides cancels one-sided disocclusion: background revealed at i
                # but visible in the other neighbour is not motion
                per_gap = np.where(valid_j & (d < per_gap), d, per_gap)
                got |= valid_j
                lum = wc.mean(2)
                brighter = valid_j & (lum > ref_lum)
                ref[brighter] = wc[brighter]
                ref_lum = np.where(brighter, lum, ref_lum)
            # MAX over gaps: a subject too slow to clear itself at the short gap does at the long one
            best = np.where(got, np.maximum(best, np.where(got, per_gap, 0.0)), best)
        valid = best >= 0
        ref_ok = ref_lum >= 0
        if not valid.any():
            moved[i] = person[i]
            stats.append(
                dict(
                    frame=i,
                    personPx=int(person[i].sum()),
                    movedPx=int(person[i].sum()),
                    residPx=0,
                    classPx=0,
                    shadowPx=0,
                    addedPx=0,
                    addedFarPx=0,
                    addedFrac=0.0,
                    rejected=False,
                    homographies=fits,
                    thresh=0.0,
                )
            )
            continue
        r = np.where(valid, np.maximum(best, 0), 0).astype(np.float32)
        noise = float(np.median(r[valid])) * 1.4826 + 1e-3
        thr = max(resid_floor, resid_k * noise)
        # a homography cannot model parallax, and what it leaves behind is proportional to the
        # local image gradient: wind in foliage and any near, high-contrast edge light up. Dividing
        # by the local gradient separates misalignment from real change - a cast shadow on asphalt
        # is a large residual over almost no gradient, moving leaves are the reverse.
        gm = cv2.GaussianBlur(
            np.abs(cv2.Sobel(gi, cv2.CV_32F, 1, 0, ksize=3))
            + np.abs(cv2.Sobel(gi, cv2.CV_32F, 0, 1, ksize=3)),
            (0, 0),
            2.0,
        )
        raw = (r > thr) & (r > resid_norm * (gm + grad_floor)) & valid
        raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, open_k)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, open_k)
        n_lab, lab, st_cc, _ = cv2.connectedComponentsWithStats(raw, 8)
        keep = np.zeros((h, w), bool)
        for c in range(1, n_lab):
            if st_cc[c, cv2.CC_STAT_AREA] >= min_area:
                keep |= lab == c
        resid = keep
        resid_out[i] = resid

        extra = np.zeros((h, w), bool)
        if extra_classes:
            cand = np.isin(lab_s[i], MOVABLE) & ~pers_s[i]
            if cand.any():
                contact = cv2.dilate(
                    pers_s[i].astype(np.uint8),
                    np.ones((2 * contact_s + 1, 2 * contact_s + 1), np.uint8),
                ).astype(bool)
                n2, l2, s2, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
                for c in range(1, n2):
                    comp = l2 == c
                    a = s2[c, cv2.CC_STAT_AREA]
                    if a < min_area:
                        continue
                    it_moved = (comp & resid).sum() > max(40, 0.08 * a)
                    # carried: a movable class in contact with the person. An umbrella is class
                    # "tent" and never overlaps the body by much, so an overlap test misses it and
                    # a contact test finds it. Size-capped, or a wall-sized "door" comes in too.
                    carried = (
                        lab_s[i][comp][0] in CARRIABLE
                        and (comp & contact).sum() >= 25
                        and a <= max_comp_frac * w * h
                        and a <= 1.5 * max(pers_s[i].sum(), 1)
                    )
                    if it_moved or carried:
                        extra |= comp

        shadow_m = np.zeros((h, w), bool)
        if shadow and ref_ok.any() and pers_s[i].any():
            # the band is per person and scaled to THAT person's height. A union-wide radius is
            # wrong both ways: a crowd of distant people shrinks it for the near subject, and the
            # near subject's own height would swallow the shot around the distant ones
            np_, lp, sp, _ = cv2.connectedComponentsWithStats(pers_s[i].astype(np.uint8), 8)
            dist, near_lab = cv2.distanceTransformWithLabels(
                (~pers_s[i]).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
            )
            # near_lab indexes components of the ZERO set (the people); map each to its own radius
            rad_of = np.zeros(near_lab.max() + 2, np.float32)
            for c in range(1, np_):
                if sp[c, cv2.CC_STAT_AREA] < min_area:
                    continue
                l = near_lab[lp == c]
                if len(l):
                    rad_of[np.bincount(l.ravel()).argmax()] = max(
                        8.0, shadow_radius_frac * sp[c, cv2.CC_STAT_HEIGHT]
                    )
            near = dist <= rad_of[np.clip(near_lab, 0, len(rad_of) - 1)]
            cur = small[i].astype(np.float32) + 4.0
            rf = ref + 4.0
            lum_c = cur.mean(2)
            lum_r = rf.mean(2)
            darker = (lum_c < 0.93 * lum_r) & (lum_c > 0.22 * lum_r)
            lr = np.log(cur / rf)
            chroma = np.abs(lr - lr.mean(2, keepdims=True)).max(2) < 0.13
            cand = darker & chroma & near & ref_ok
            cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, open_k)
            # a cast shadow is attached to the thing that casts it, so it must touch something
            # already known to have moved. Without this the term drifts onto static dark ground.
            attached = cv2.dilate((resid | pers_s[i]).astype(np.uint8), grow_k).astype(bool)
            n3, l3, s3, _ = cv2.connectedComponentsWithStats(cand, 8)
            for c in range(1, n3):
                comp = l3 == c
                if s3[c, cv2.CC_STAT_AREA] >= min_area and (comp & attached).any():
                    shadow_m |= comp

        # Self-check. A homography cannot model parallax, so on a clip with real camera
        # translation through a deep scene the residual fires along every depth edge and would
        # hand the trainer half the room. When the alignment-dependent signals claim more than
        # max_added_frac of the frame, the camera model is not good enough here: drop them and
        # fall back to person + carried, which is never worse than what the pipeline does today.
        # Measured on the park clip at 1080x1920: 0.05 refuses 34/93 frames and leaves the cast
        # shadow in the plate on some of them, 0.10 refuses 17/93 and cleans them with no visible
        # damage. 0.05 is the default because the clips this pipeline actually runs on are
        # handheld walkthroughs, where the guard is the only thing standing between a homography
        # and half the room (share/maskfix-apartment-diff.png).
        rejected = float((resid | shadow_m).mean()) > max_added_frac
        if rejected:
            resid = np.zeros_like(resid)
            shadow_m = np.zeros_like(shadow_m)
        add_s = cv2.dilate((resid | extra | shadow_m).astype(np.uint8), grow_k).astype(bool)
        add_full = cv2.resize(
            add_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        moved[i] = person[i] | add_full
        added = moved[i] & ~person[i]
        far = added & ~cv2.resize(
            cv2.dilate(
                pers_s[i].astype(np.uint8), np.ones((2 * far_s + 1, 2 * far_s + 1), np.uint8)
            ),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        stats.append(
            dict(
                frame=i,
                personPx=int(person[i].sum()),
                movedPx=int(moved[i].sum()),
                residPx=int(resid.sum() / (sw * sh)),
                classPx=int(extra.sum() / (sw * sh)),
                shadowPx=int(shadow_m.sum() / (sw * sh)),
                addedPx=int(added.sum()),
                addedFarPx=int(far.sum()),
                addedFrac=float(added.mean()),
                rejected=bool(rejected),
                homographies=fits,
                thresh=float(thr),
            )
        )
    return dict(person=person, moved=moved, residual=resid_out, stats=stats)


# --- foreground people --------------------------------------------------------------------
# people_masks labels every person PIXEL, which is the wrong question for the person-removal
# cleaner. In a broadcast stadium shot the whole crowd in the stands is class 12, so the cleaner
# inpaints the terraces into a grey smear; in an indoor phone clip the same map drops a limb or a
# partly occluded body and leaves half a person in the plate. What the cleaner wants removed is
# the FOREGROUND people, the ones that come back as avatars. Mask R-CNN answers that question:
# it returns instances with boxes, so "big enough to reconstruct" is a box-height test, and the
# semantic map is then used only to COMPLETE the instances that survived the test.
#
# Thresholds measured on the three fixture clips in public/clips (nine frames, boxes at score
# >0.7, detector and segmenter run on CPU):
#   - stadium celebration frame (848x478): the four foreground subjects measure 0.73, 0.60, 0.59
#     and 0.38 of frame height; the stewards and spectators at the pitch-side barrier top out at
#     0.19. Any cut in (0.19, 0.38] separates them on this frame.
#   - phone/game clips: their single or paired subjects measure 0.75-0.93, far above either end.
# 0.2 sits just above the measured crowd and well below the smallest real subject. It also agrees
# with reconstruction: 0.2 of a 1080-line frame is a 216 px body, whose head is ~27 px, already
# under the face height docs/known-limits.md calls unreadable - below this a person cannot become
# an avatar, so leaving them in the plate as background texture is the consistent choice. A person
# who is short only because the frame cuts them off is treated as background too; the growth below
# is what recovers a limb of a SELECTED person, not a new subject.
FOREGROUND_MIN_HEIGHT_FRAC = 0.2
# same detector confidence scripts/shot_cuts.py::person_stats and the person stages use
FOREGROUND_SCORE = 0.7
# Growth band, as a fraction of the selected instance's larger box side. An instance mask is tight
# and what it misses is thin - a hand, hair, a bag strap - so the band only has to be thin: 0.05 is
# 17 px around the stadium player (box 350 px) and 40 px around a full-height 1080p subject, which
# is the order of the dilation the cleaner applies anyway. 0.15 was measured to be far too much:
# behind the players the crowd is one connected sheet of "person" pixels welded to the subject, and
# a 52 px bite out of it took most of several spectators with it. On the stadium celebration frame
# 0.15 removed 22.4% of the pixels, 0.05 with the guard below removes 14.9%, against the 32.8% the
# semantic mask removes today - and what is left removed is the four foreground subjects.
FOREGROUND_GROWTH_BOX_FRAC = 0.05
# Self-check on that same failure, in the spirit of moved_content_masks' max_added_frac: what the
# instance mask misses is a fraction of a body, so when the connected semantic pixels would add
# more than half of the instance's own area, the semantic map is describing a crowd standing
# against the subject rather than a missed limb. Drop the growth for that instance and keep the
# instance alone, which is never worse than the instance mask the detector already committed to.
FOREGROUND_MAX_GROWTH_FRAC = 0.5
COCO_PERSON = 1
_detector = None


def _load_detector():
    global _detector
    if _detector is None:
        import torch
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2,
            MaskRCNN_ResNet50_FPN_V2_Weights,
        )

        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
        dev = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        _detector = (model.to(dev), dev)
    return _detector


def person_instances(images: np.ndarray, score: float = FOREGROUND_SCORE, batch_size: int = 2):
    """images uint8 (N,H,W,3) -> yields (boxes (K,4) float x0,y0,x1,y1, masks (K,H,W) bool).

    One tuple per frame, in order, so a caller never holds every frame's instances at once: at
    1080p a dozen kept instances are already 25 MB of bool. Mask R-CNN keeps full-resolution
    feature maps, so its forward batch is smaller than the segmenter's.
    """
    import torch

    model, dev = _load_detector()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [
                torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255).to(dev)
                for im in images[i : i + batch_size]
            ]
            for pred in model(batch):
                keep = (pred["labels"] == COCO_PERSON) & (pred["scores"] > score)
                boxes = pred["boxes"][keep].float().cpu().numpy()
                masks = (pred["masks"][keep, 0] > 0.5).cpu().numpy()
                yield boxes, masks


def select_foreground_mask(
    boxes,
    instances,
    semantic: np.ndarray,
    min_height_frac: float = FOREGROUND_MIN_HEIGHT_FRAC,
    growth_box_frac: float = FOREGROUND_GROWTH_BOX_FRAC,
    max_growth_frac: float = FOREGROUND_MAX_GROWTH_FRAC,
) -> tuple[np.ndarray, dict]:
    """One frame of detections + the semantic person map -> (bool (H,W), counts).

    boxes (K,4) x0,y0,x1,y1 in pixels, instances bool (K,H,W), semantic bool (H,W) UNDILATED.
    No model runs here: this is the whole selection rule, so it is testable on synthetic masks.

    An instance is kept when its box is at least min_height_frac of the frame height. Each kept
    instance is then grown with the semantic person pixels CONNECTED to it, so a limb, a bag or
    hair the instance mask cut off is still covered, but the growth is confined to a band around
    that instance (growth_box_frac of its larger box side) and is dropped entirely when it would
    add more than max_growth_frac of the instance's own area. Semantic person pixels that reach no
    kept instance - the crowd in the stands - stay out of the mask, and a crowd blob that touches
    a kept instance contributes at most a band-deep bite of itself, or nothing.
    """
    import cv2

    semantic = np.asarray(semantic, dtype=bool)
    H, W = semantic.shape
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    instances = np.asarray(instances, dtype=bool).reshape(-1, H, W)
    out = np.zeros((H, W), dtype=bool)
    selected, rejected, crowded, bands, grown_px = 0, 0, 0, [], 0
    for k in range(len(boxes)):
        x0, y0, x1, y1 = boxes[k]
        if (y1 - y0) < min_height_frac * H:
            rejected += 1
            continue
        selected += 1
        seed = instances[k]
        if not seed.any():
            continue
        band_px = max(1, int(round(growth_box_frac * max(y1 - y0, x1 - x0))))
        bands.append(band_px)
        region = (semantic | seed) & _dilate(seed, band_px)
        _, lab = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
        touched = np.unique(lab[seed])
        grown = np.isin(lab, touched[touched > 0]) & ~seed
        if grown.sum() > max_growth_frac * seed.sum():
            crowded += 1
            grown[:] = False
        grown_px += int(grown.sum())
        out |= seed | grown
    counts = dict(
        instances=int(len(boxes)),
        selected=int(selected),
        rejectedSmall=int(rejected),
        rejectedGrowth=int(crowded),
        bandPx=int(max(bands)) if bands else 0,
        grownPx=int(grown_px),
        maskPx=int(out.sum()),
        semanticPx=int(semantic.sum()),
    )
    return out, counts


def foreground_people_masks(
    images: np.ndarray,
    *,
    min_height_frac: float = FOREGROUND_MIN_HEIGHT_FRAC,
    score: float = FOREGROUND_SCORE,
    dilate_px: int = 6,
    batch_size: int = 8,
    growth_box_frac: float = FOREGROUND_GROWTH_BOX_FRAC,
    max_growth_frac: float = FOREGROUND_MAX_GROWTH_FRAC,
    stats: list | None = None,
) -> np.ndarray:
    """images uint8 (N,H,W,3) -> bool (N,H,W), True where a FOREGROUND person is (dilated).

    Same shape, dtype and dilation convention as people_masks, and a drop-in replacement for it
    wherever the caller wants the reconstructed subjects rather than every person-coloured pixel.
    Pass a list as `stats` to receive one select_foreground_mask count dict per frame.
    """
    N, H, W, _ = images.shape
    out = np.zeros((N, H, W), dtype=bool)
    det_batch = max(1, batch_size // 4)
    for i in range(0, N, batch_size):
        chunk = images[i : min(N, i + batch_size)]
        semantic = people_masks(chunk, dilate_px=0, batch_size=batch_size)
        for b, (boxes, instances) in enumerate(
            person_instances(chunk, score=score, batch_size=det_batch)
        ):
            mask, counts = select_foreground_mask(
                boxes,
                instances,
                semantic[b],
                min_height_frac=min_height_frac,
                growth_box_frac=growth_box_frac,
                max_growth_frac=max_growth_frac,
            )
            out[i + b] = mask
            if stats is not None:
                stats.append(dict(frame=i + b, **counts))
    if dilate_px > 0:
        for i in range(N):
            out[i] = _dilate(out[i], dilate_px)
    return out
