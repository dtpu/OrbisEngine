#!/usr/bin/env python3
"""Multi-person detection and tracking over a clip's sampled frames.

MultiHMR is natively multi-person: `estimator.mhmr_model(...)` already returns every detection
in the frame, and lhm_animate.py throws all but one away (it keeps `choices[0]`). This script
keeps them all and links them across samples into identity-stable tracks, then writes, per track,
exactly the `--seed-poses` / `--seed-motion` pair lhm_animate.py already knows how to consume.

Detection: MultiHMR for the poses (one forward pass gives detection + SMPL-X + 3D joints, so a
second pose model is never needed) plus torchvision Mask R-CNN on the same frame for real
per-person instance masks, associated to the MultiHMR detections by joint-box IoU. Mask R-CNN is
also the fallback detector: a Mask R-CNN person with no MultiHMR partner is recorded (so the
quality score knows somebody was there) but cannot be animated.

Association: per-sample Hungarian assignment against each live track's constant-velocity
prediction, with a cost combining centre distance, box-height ratio, camera depth ratio and a
masked upper/lower-body colour histogram. The colour term is what survives two people in similar
dark tops crossing paths: their trousers differ, and the mask splits torso from legs.

  python track_people.py --video clip.mp4 --out tracks/ --fps 12 [--cameras cameras.json]
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HIST_BINS = 6  # per channel, so 216 bins per body half


# Share of sampled frames that must have a camera before the solve is judged to describe a
# different video rather than merely having gaps.
MINIMUM_CAMERA_COVERAGE = 0.5

# Share of sampled frames that must actually decode before the source is called unusable.
MINIMUM_DECODED_FRAMES = 0.5


def decoded_span_usable(decoded: int, requested: int, floor=MINIMUM_DECODED_FRAMES) -> bool:
    """Whether enough of the requested span decoded to track from."""
    if requested == 0:
        return False
    return decoded / requested >= floor


def sample_indices(video, fps_out, start=0.0, stop=None):
    """The decode schedule lhm_animate.py uses; every stage must agree on it exactly."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = count / fps
    stop = duration if stop is None else stop
    requested = start + np.arange(int(np.ceil((stop - start) * fps_out))) / fps_out
    requested = requested[requested < stop]
    indices = np.clip(np.rint(requested * fps).astype(int), 0, count - 1)
    return indices, indices / fps, fps, count, duration


def colour_histogram(rgb, mask, box):
    """Normalised RGB histograms of the masked pixels, upper and lower body halves."""
    x0, y0, x1, y1 = box
    out = []
    mid = (y0 + y1) / 2
    for lo, hi in ((y0, mid), (mid, y1)):
        lo, hi = int(max(0, lo)), int(min(rgb.shape[0], np.ceil(hi)))
        sub_m = mask[lo:hi]
        sub_rgb = rgb[lo:hi]
        px = sub_rgb[sub_m] if sub_m.any() else np.zeros((0, 3), np.uint8)
        if len(px) < 30:
            out.append(np.zeros(HIST_BINS**3, np.float32))
            continue
        q = (px.astype(np.int32) * HIST_BINS // 256).clip(0, HIST_BINS - 1)
        flat = q[:, 0] * HIST_BINS * HIST_BINS + q[:, 1] * HIST_BINS + q[:, 2]
        h = np.bincount(flat, minlength=HIST_BINS**3).astype(np.float32)
        out.append(h / h.sum())
    return np.concatenate(out)


def hist_distance(a, b):
    """Bhattacharyya distance on each half, averaged; 0 = identical, 1 = disjoint."""
    n = HIST_BINS**3
    d = []
    for s in (slice(0, n), slice(n, 2 * n)):
        pa, pb = a[s], b[s]
        if pa.sum() <= 0 or pb.sum() <= 0:
            continue
        d.append(1.0 - float(np.sqrt(pa * pb).sum()))
    return float(np.mean(d)) if d else 0.5


def joints_box(joints, w, h):
    x0, y0 = joints[:22].min(0)
    x1, y1 = joints[:22].max(0)
    return [float(x0), float(y0), float(x1), float(y1)]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (
        max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        + max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        - inter
    )
    return inter / ua if ua > 0 else 0.0


class Track:
    def __init__(self, tid, obs):
        self.id = tid
        self.obs = [obs]
        self.hist = obs["hist"].copy()
        self.missed = 0

    @property
    def last(self):
        return self.obs[-1]

    def predict(self, sample):
        """Constant velocity from the last two observations, damped, clamped to one box."""
        last = self.obs[-1]
        c = np.array(last["center"])
        if len(self.obs) < 2:
            return c
        prev = self.obs[-2]
        dt = last["sample"] - prev["sample"]
        if dt <= 0:
            return c
        v = (c - np.array(prev["center"])) / dt
        step = sample - last["sample"]
        return c + v * step * 0.8

    def update(self, obs):
        self.obs.append(obs)
        self.missed = 0
        # Slow appearance EMA: fast enough for lighting, slow enough to resist an identity swap.
        self.hist = 0.85 * self.hist + 0.15 * obs["hist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="/opt/lhm")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--cameras")
    ap.add_argument(
        "--det-thresh",
        type=float,
        default=0.15,
        help="MultiHMR detection threshold; lower than the single-person 0.3 because a "
        "partly occluded second person scores lower than a clear first one",
    )
    ap.add_argument("--new-track-thresh", type=float, default=0.30)
    ap.add_argument("--max-gap", type=int, default=8, help="samples a track may coast unmatched")
    ap.add_argument("--min-track-samples", type=int, default=8)
    ap.add_argument(
        "--gate", type=float, default=0.22, help="max normalised centre distance for a match"
    )
    ap.add_argument("--overlay-every", type=int, default=0, help="0 = only first/middle/last")
    ap.add_argument("--mask-scale", type=float, default=0.5, help="stored mask resolution")
    ap.add_argument("--no-maskrcnn", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(a.repo)
    sys.path.insert(0, a.repo)
    start = time.time()
    import torch
    from accelerate import Accelerator
    from engine.pose_estimation.pose_estimator import PoseEstimator
    from scipy.optimize import linear_sum_assignment

    torch._dynamo.config.disable = True
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    Accelerator()

    source_sha = hashlib.sha256(Path(a.video).read_bytes()).hexdigest()
    indices, times, src_fps, count, duration = sample_indices(a.video, a.fps)
    camera_doc = json.loads(Path(a.cameras).read_text()) if a.cameras else None
    cameras = camera_doc["cameras"] if camera_doc else None
    # Cameras are matched by the source frame they describe, not by position. The solve records
    # no camera for a frame whose subject it could not reconstruct, and it may cover a shorter
    # span than this schedule if the container overstated its frame count. Both leave gaps that
    # are absences, not disagreement; only a wholesale mismatch means the schedules differ.
    camera_by_source = {}
    if cameras:
        camera_by_source = {int(c["sourceIndex"]): c for c in cameras if c}
        covered = sum(1 for i in indices if int(i) in camera_by_source)
        if covered < len(indices) * MINIMUM_CAMERA_COVERAGE:
            raise ValueError(
                f"Camera records cover only {covered} of {len(indices)} sampled source frames; "
                "the solve and this schedule do not describe the same video"
            )
        if covered < len(indices):
            print(
                f"{len(indices) - covered} sampled frames have no camera and will be tracked "
                "without one",
                flush=True,
            )

    rcnn = None
    if not a.no_maskrcnn:
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2,
            MaskRCNN_ResNet50_FPN_V2_Weights,
        )

        rcnn = (
            maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().cuda()
        )

    estimator = PoseEstimator("./pretrained_models/human_model_files", device="cuda")
    cap = cv2.VideoCapture(a.video)
    tracks, closed, next_id = [], [], 0
    overlay_samples = {0, len(indices) // 2, len(indices) - 1}
    mask_store = {}
    unmatched_rcnn_total = 0
    w = h = None

    for sample, (index, timestamp) in enumerate(zip(indices, times)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if not ok:
            # Container metadata routinely overstates frame count, so the requested tail can be
            # undecodable even though the file is sound. Track the span that exists rather than
            # discarding every frame already tracked.
            if not decoded_span_usable(sample, len(indices)):
                raise RuntimeError(
                    f"Unable to decode source frame {index}: only {sample} of {len(indices)} "
                    "sampled frames are readable"
                )
            print(
                f"decode stopped at source frame {index} ({sample} of {len(indices)} sampled)",
                flush=True,
            )
            indices, times = indices[:sample], times[:sample]
            break
        raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = raw.shape[:2]
        diag = float(np.hypot(w, h))
        padded, ow, oh = estimator.img_center_padding(raw)
        tensor, annotation = estimator._preprocess(padded)
        pl, pt, factor, _, _ = annotation
        frame_camera = camera_by_source.get(int(index)) if cameras else None
        if frame_camera is not None:
            source_K = np.array(frame_camera["source_intrinsics"])
            K = torch.tensor(
                np.array([[factor, 0, factor * ow + pl], [0, factor, factor * oh + pt], [0, 0, 1]])
                @ source_K,
                dtype=torch.float32,
                device="cuda",
            )[None]
        else:
            K = estimator.get_camera_parameters()
            source_K = K[0].cpu().numpy().copy()
            source_K[:2] /= factor
            source_K[0, 2] -= pl / factor + ow
            source_K[1, 2] -= pt / factor + oh
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            detected = estimator.mhmr_model(
                tensor,
                is_training=False,
                nms_kernel_size=3,
                det_thresh=a.det_thresh,
                K=K,
                idx=None,
                max_dist=None,
            )

        # Mask R-CNN instance masks on the same raw frame.
        inst_masks, inst_boxes = [], []
        if rcnn is not None:
            x = torch.from_numpy(raw).permute(2, 0, 1).float().cuda() / 255
            with torch.inference_mode():
                r = rcnn([x])[0]
            for i in range(len(r["labels"])):
                if int(r["labels"][i]) == 1 and float(r["scores"][i]) > 0.5:
                    inst_masks.append((r["masks"][i, 0] > 0.5).cpu().numpy())
                    inst_boxes.append([float(v) for v in r["boxes"][i].cpu().numpy()])
            del x, r

        observations = []
        used_inst = set()
        for det_i, person in enumerate(detected):
            joints = (person["j2d"].float().cpu().numpy() - [pl, pt]) / factor - [ow, oh]
            box = joints_box(joints, w, h)
            centre = np.median(joints[:22], axis=0) / [w, h]
            j3d = person["j3d"].float().cpu().numpy()
            # Pick the Mask R-CNN instance that best covers this skeleton.
            best_inst, best_iou = -1, 0.0
            for i, ib in enumerate(inst_boxes):
                v = iou(box, ib)
                if v > best_iou:
                    best_inst, best_iou = i, v
            mask = inst_masks[best_inst] if (best_inst >= 0 and best_iou > 0.2) else None
            if mask is not None:
                used_inst.add(best_inst)
                mbox = inst_boxes[best_inst]
            else:
                mbox = box
                mask = np.zeros((h, w), bool)
                x0, y0, x1, y1 = [int(max(0, v)) for v in box]
                mask[y0 : min(h, y1 + 1), x0 : min(w, x1 + 1)] = True
            observations.append(
                dict(
                    sample=sample,
                    sourceIndex=int(index),
                    time=float(timestamp),
                    det=det_i,
                    score=float(person["scores"]),
                    box=box,
                    maskBox=mbox,
                    center=centre.tolist(),
                    depth=float(j3d[0, 2]),
                    joints=joints,
                    j3dRoot=j3d[0].tolist(),
                    heightFrac=float(box[3] - box[1]) / h,
                    maskArea=float(mask.mean()),
                    inFrame=(
                        (joints[:22, 0] >= 0)
                        & (joints[:22, 0] < w)
                        & (joints[:22, 1] >= 0)
                        & (joints[:22, 1] < h)
                    ),
                    hist=colour_histogram(raw, mask, mbox),
                    mask=mask,
                    pose={
                        k: v.detach().float().cpu()
                        for k, v in person.items()
                        if k in ("rotvec", "shape", "transl_pelvis", "j3d", "scores")
                    },
                    source_intrinsics=source_K.tolist(),
                    maskIou=best_iou,
                )
            )
        unmatched_rcnn_total += len(inst_boxes) - len(used_inst)

        # ---- association ----------------------------------------------------
        live = [t for t in tracks if t.missed <= a.max_gap]
        if live and observations:
            cost = np.full((len(live), len(observations)), 1e6)
            for i, t in enumerate(live):
                pred = t.predict(sample)
                lh = t.last["box"][3] - t.last["box"][1]
                ld = t.last["depth"]
                for j, o in enumerate(observations):
                    dist = float(np.linalg.norm(np.array(o["center"]) - pred))
                    if dist > a.gate * (1 + 0.15 * t.missed):
                        continue
                    oh_ = o["box"][3] - o["box"][1]
                    size = abs(np.log(max(oh_, 1e-3) / max(lh, 1e-3)))
                    depth = abs(np.log(max(o["depth"], 1e-3) / max(ld, 1e-3)))
                    app = hist_distance(t.hist, o["hist"])
                    cost[i, j] = 4.0 * dist + 0.6 * size + 0.8 * depth + 1.5 * app
            rows, cols = linear_sum_assignment(cost)
            matched_obs = set()
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e5:
                    continue
                live[i].update(observations[j])
                matched_obs.add(j)
            for t in live:
                if t.last["sample"] != sample:
                    t.missed += 1
        else:
            matched_obs = set()
            for t in live:
                t.missed += 1

        for j, o in enumerate(observations):
            if j in matched_obs:
                continue
            if o["score"] < a.new_track_thresh:
                continue
            tracks.append(Track(next_id, o))
            next_id += 1

        for t in tracks:
            for o in t.obs:
                if o["sample"] == sample and o.get("mask") is not None:
                    key = (t.id, sample)
                    small = cv2.resize(
                        o["mask"].astype(np.uint8),
                        None,
                        fx=a.mask_scale,
                        fy=a.mask_scale,
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    mask_store[key] = np.packbits(small)

        if sample in overlay_samples or (a.overlay_every and sample % a.overlay_every == 0):
            over = raw.copy()
            palette = [(255, 60, 60), (60, 160, 255), (80, 220, 80), (255, 200, 40), (220, 80, 255)]
            for t in tracks:
                if t.last["sample"] != sample:
                    continue
                c = palette[t.id % len(palette)]
                x0, y0, x1, y1 = [int(v) for v in t.last["box"]]
                cv2.rectangle(over, (x0, y0), (x1, y1), c, 4)
                cv2.putText(
                    over,
                    f"t{t.id} {t.last['score']:.2f}",
                    (x0, max(30, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    c,
                    3,
                )
                for px, py in t.last["joints"][:22]:
                    cv2.circle(over, (round(float(px)), round(float(py))), 5, c, -1)
            Image.fromarray(over).save(out / f"track-overlay-{sample:03d}.jpg", quality=88)

        for o in observations:
            o.pop("mask", None)
        print(
            f"sample {sample + 1}/{len(indices)} detections {len(detected)} "
            f"rcnn {len(inst_boxes)} live {sum(1 for t in tracks if t.last['sample'] == sample)}",
            flush=True,
        )

    cap.release()
    del estimator, rcnn
    torch.cuda.empty_cache()

    # ---- occlusion: who is in front of whom, per sample ---------------------
    by_sample = {}
    for t in tracks:
        for o in t.obs:
            by_sample.setdefault(o["sample"], []).append((t.id, o))
    for sample, entries in by_sample.items():
        for tid, o in entries:
            occ = 0.0
            for other_id, other in entries:
                if other_id == tid or other["depth"] >= o["depth"]:
                    continue
                occ = max(occ, iou(o["box"], other["box"]))
            o["occludedFraction"] = float(occ)

    # ---- keep the real tracks, write per-track seed pairs --------------------
    kept = [t for t in tracks if len(t.obs) >= a.min_track_samples]
    kept.sort(key=lambda t: -len(t.obs))
    report = []
    for rank, t in enumerate(kept):
        obs_by_sample = {o["sample"]: o for o in t.obs}
        poses = [
            obs_by_sample[s]["pose"] if s in obs_by_sample else None for s in range(len(indices))
        ]
        records = []
        for s in range(len(indices)):
            o = obs_by_sample.get(s)
            if o is None:
                continue
            joints = o["joints"]
            records.append(
                dict(
                    sample=s,
                    sourceIndex=o["sourceIndex"],
                    time=o["time"],
                    detectedPeople=len(by_sample.get(s, [])),
                    score=o["score"],
                    originalSourceTime=o["time"],
                    projectedBodyJoints=joints[:22].tolist(),
                    jointProjectionInImage=o["inFrame"].tolist(),
                    confidenceSemantics="Score is person detection confidence, not per-joint accuracy or visibility confidence.",
                    source_intrinsics=o["source_intrinsics"],
                    rootCamera=o["j3dRoot"],
                    smplTranslationCamera=o["pose"]["transl_pelvis"].reshape(3).tolist(),
                    feetCamera=o["pose"]["j3d"][[7, 8, 10, 11]].tolist(),
                    footJointOrder=["leftAnkle", "rightAnkle", "leftFoot", "rightFoot"],
                    rootRotationVector=o["pose"]["rotvec"][0].tolist(),
                    detectionThreshold=a.det_thresh,
                    trackId=t.id,
                    box=o["box"],
                    maskBox=o["maskBox"],
                    maskIou=o["maskIou"],
                    occludedFraction=o.get("occludedFraction", 0.0),
                    depth=o["depth"],
                    heightFraction=o["heightFrac"],
                    maskAreaFraction=o["maskArea"],
                )
            )
        samples = sorted(obs_by_sample)
        in_frame = np.array([obs_by_sample[s]["inFrame"].all() for s in samples])
        heights = np.array([obs_by_sample[s]["heightFrac"] for s in samples])
        occl = np.array([obs_by_sample[s].get("occludedFraction", 0.0) for s in samples])
        scores = np.array([obs_by_sample[s]["score"] for s in samples])
        coverage = len(samples) / len(indices)
        quality = dict(
            samples=len(samples),
            coverage=float(coverage),
            firstSample=int(samples[0]),
            lastSample=int(samples[-1]),
            gaps=[[int(x), int(y)] for x, y in _gaps(samples)],
            fullyInFrameFraction=float(in_frame.mean()),
            medianHeightFraction=float(np.median(heights)),
            meanOccludedFraction=float(occl.mean()),
            maxOccludedFraction=float(occl.max()),
            meanScore=float(scores.mean()),
            # One number for ranking: present, big, unoccluded, whole body in shot.
            score=float(
                coverage
                * (0.4 + 0.6 * in_frame.mean())
                * min(1.0, np.median(heights) / 0.5)
                * (1.0 - 0.5 * occl.mean())
            ),
        )
        # Avatar-input candidates: full body, tall, unoccluded, confident.
        cand = sorted(
            samples,
            key=lambda s: (
                -(
                    (1.0 if obs_by_sample[s]["inFrame"].all() else 0.3)
                    * obs_by_sample[s]["heightFrac"]
                    * (1.0 - obs_by_sample[s].get("occludedFraction", 0.0))
                    * obs_by_sample[s]["score"]
                )
            ),
        )[:12]
        folder = out / f"track_{rank:02d}"
        folder.mkdir(exist_ok=True)
        torch.save(
            dict(poses=poses, sourceIndices=indices, timestamps=times), folder / "source-poses.pt"
        )
        (folder / "motion.json").write_text(json.dumps(dict(frames=records), indent=1))
        report.append(
            dict(
                track=rank,
                rawId=t.id,
                quality=quality,
                candidateSamples=[
                    dict(
                        sample=int(s),
                        sourceIndex=int(indices[s]),
                        box=obs_by_sample[s]["box"],
                        maskBox=obs_by_sample[s]["maskBox"],
                        heightFraction=obs_by_sample[s]["heightFrac"],
                        fullyInFrame=bool(obs_by_sample[s]["inFrame"].all()),
                        occludedFraction=obs_by_sample[s].get("occludedFraction", 0.0),
                        score=obs_by_sample[s]["score"],
                    )
                    for s in cand
                ],
                samples=[int(s) for s in samples],
            )
        )
        print(f"track {rank} (raw {t.id}): {json.dumps(quality)}", flush=True)

    if mask_store:
        remap = {t.id: rank for rank, t in enumerate(kept)}
        np.savez_compressed(
            out / "masks.npz",
            shape=np.array([h, w, a.mask_scale], np.float64),
            **{
                f"t{remap[tid]}_s{s:03d}": arr
                for (tid, s), arr in mask_store.items()
                if tid in remap
            },
        )

    doc = dict(
        video=str(Path(a.video).resolve()),
        sourceSha256=source_sha,
        sourceFps=src_fps,
        sourceFrames=count,
        duration=duration,
        fps=a.fps,
        samples=len(indices),
        sourceIndices=[int(i) for i in indices],
        timestamps=[float(t) for t in times],
        width=w,
        height=h,
        detectionThreshold=a.det_thresh,
        gate=a.gate,
        maxGap=a.max_gap,
        minTrackSamples=a.min_track_samples,
        camerasSha256=hashlib.sha256(Path(a.cameras).read_bytes()).hexdigest()
        if a.cameras
        else None,
        trackCount=len(kept),
        discardedShortTracks=len(tracks) - len(kept),
        maskrcnnDetectionsWithoutPose=unmatched_rcnn_total,
        tracks=report,
        method=(
            "MultiHMR multi-person detections (one forward pass per sample, all persons kept) "
            "associated by Hungarian assignment on constant-velocity centre prediction, box-height "
            "ratio, camera-depth ratio and a Mask R-CNN-masked upper/lower-body RGB histogram. "
            "Mask R-CNN supplies the per-person instance masks; a Mask R-CNN person with no "
            "MultiHMR partner is counted but not tracked."
        ),
        seconds=time.time() - start,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
    )
    (out / "tracks.json").write_text(json.dumps(doc, indent=1))
    print(
        json.dumps(
            {k: v for k, v in doc.items() if k not in ("sourceIndices", "timestamps", "tracks")},
            indent=1,
        ),
        flush=True,
    )


def _gaps(samples):
    out = []
    for a_, b_ in zip(samples, samples[1:]):
        if b_ - a_ > 1:
            out.append((a_ + 1, b_ - 1))
    return out


if __name__ == "__main__":
    main()
