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
