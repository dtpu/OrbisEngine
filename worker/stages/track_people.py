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

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# worker/modal_multiperson.py mounts all of worker/stages as /root/stages, so the solver's own
# module travels with this one: its backend decision and container probe are imported rather
# than guessed again here. Its decoders push frames into a sink; tracking has to pull them one
# at a time in order, so the two generators below are that same sequential walk, reshaped.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dense_pi3x
from dense_pi3x import DECODE_FFMPEG, DECODE_OPENCV, choose_decode_backend, opencv_first_frame

HIST_BINS = 6  # per channel, so 216 bins per body half
SAMPLE_AUTHORITY_CAMERAS = "cameras"
SAMPLE_AUTHORITY_FPS_RULE = "fps-rule"


def source_metadata(video):
    """Container frame rate, frame count and frame size."""
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    cap.release()
    if fps <= 0 or count < 1 or min(size) < 1:
        # A codec OpenCV cannot decode still has a container ffprobe can read.
        probe = dense_pi3x.probe_source(video)
        fps = fps if fps > 0 else float(probe.get("averageFps") or 0.0)
        count = count if count > 0 else int(probe.get("containerFrames") or 0)
        size = (size[0] or int(probe.get("width") or 0), size[1] or int(probe.get("height") or 0))
    if fps <= 0 or count < 1:
        raise ValueError(f"Unusable source metadata for {video}: {fps} fps, {count} frames")
    return fps, count, size


def fps_rule_indices(fps_out, src_fps, count, start=0.0, stop=None):
    """The container-rate schedule this stage used before the solver measured its own.

    It addresses frames by `round(time * average_fps)`, which is only the frame ffmpeg's `fps`
    filter keeps when the source really is constant-rate.
    """
    duration = count / src_fps
    stop = duration if stop is None else stop
    requested = start + np.arange(int(np.ceil((stop - start) * fps_out))) / fps_out
    requested = requested[requested < stop]
    indices = np.clip(np.rint(requested * src_fps).astype(int), 0, count - 1)
    return indices, indices / src_fps


def plan_samples(cameras, fps_out, src_fps, count):
    """Which source frames to track, at what times, and which rule chose them.

    When a solve is supplied its samples are the authority. dense_pi3x.py selects them from the
    decoded presentation timestamps the way ffmpeg's `fps` filter does, so on a variable-rate
    source they are not `round(time * average_fps)` at all: on a 59.49 fps phone clip the
    solver kept frames 2, 7, 12, 17, 22 ... while this stage's own rule asked for 0, 5, 10, and
    the two lists could never be reconciled. Adopting each camera's `sourceIndex` and `time` is
    what keeps every per-sample record paired with the camera that reconstructed that frame.
    """
    if cameras is not None:
        indices = np.array([int(c["sourceIndex"]) for c in cameras], int)
        times = np.array([float(c["time"]) for c in cameras], float)
        if len(indices) < 1:
            raise ValueError("Camera records are empty")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("Camera records must name increasing, unique source indices")
        return indices, times, SAMPLE_AUTHORITY_CAMERAS
    indices, times = fps_rule_indices(fps_out, src_fps, count)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Source frame rate below the requested independent sample density")
    return indices, times, SAMPLE_AUTHORITY_FPS_RULE


def decode_shortfall(indices, decoded):
    """The complaint when the decoder never handed back some of the planned frames."""
    missing = [int(index) for index in indices if int(index) not in decoded]
    if not missing:
        return None
    tail = f", and {len(missing) - 1} more up to {missing[-1]}" if len(missing) > 1 else ""
    return (
        f"Decoded {len(indices) - len(missing)} of {len(indices)} planned source frames; "
        f"source frame {missing[0]} never decoded{tail}. The source is shorter or more damaged "
        "than the sample plan it was given."
    )


def opencv_frames(video, wanted):
    """Yield `(source index, BGR frame)` for the wanted indices, decoding strictly forward.

    Nothing seeks. `CAP_PROP_POS_FRAMES` cannot reach the last frames of a variable-rate file
    -- on the soccer clip every index from 706 on fails while all 712 decode in order -- and
    `dense_pi3x.decode_opencv` counted the solver's ordinals with this same walk, so the
    indices in cameras.json mean exactly what this loop counts.
    """
    cap = cv2.VideoCapture(str(video))
    ordinal = 0
    try:
        while cap.grab():
            if ordinal in wanted:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                yield ordinal, bgr
            ordinal += 1
    finally:
        cap.release()


def ffmpeg_frames(video, size, wanted):
    """The same walk piped out of ffmpeg, for codecs OpenCV opens but cannot decode (AV1).

    Full resolution and no re-encode, so the pixels are the source's own. `-fps_mode
    passthrough` matters: without it ffmpeg pads a variable-rate stream up to a constant rate
    by duplicating frames, and every ordinal the plan was built from would shift.
    """
    width, height = int(size[0]), int(size[1])
    if width < 1 or height < 1:
        raise RuntimeError("The ffmpeg fallback needs the source frame size")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-noautorotate",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_bytes = width * height * 3
    ordinal = 0
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buffer = proc.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            if ordinal in wanted:
                yield ordinal, np.frombuffer(buffer, np.uint8).reshape(height, width, 3).copy()
            ordinal += 1
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        proc.wait()


def sample_stream(video, backend, size, wanted):
    """The decode generator this run uses, chosen from what OpenCV could actually do."""
    if backend == DECODE_OPENCV:
        return opencv_frames(video, wanted)
    return ffmpeg_frames(video, size, wanted)


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
    src_fps, count, source_size = source_metadata(a.video)
    duration = count / src_fps
    camera_doc = json.loads(Path(a.cameras).read_text()) if a.cameras else None
    cameras = camera_doc["cameras"] if camera_doc else None
    indices, times, authority = plan_samples(cameras, a.fps, src_fps, count)
    # Retained as a guard, not as a schedule: after adopting the cameras' own samples this can
    # only fire if a camera record is malformed.
    if cameras and (
        len(cameras) != len(indices)
        or any(c["sourceIndex"] != int(i) for c, i in zip(cameras, indices))
    ):
        raise ValueError("Camera records must correspond exactly to sampled source indices")
    backend, backend_reason = choose_decode_backend(*opencv_first_frame(cv2, a.video))
    print(
        f"{len(indices)} samples from {authority}; decode backend {backend} ({backend_reason})",
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
    wanted = {int(index): sample for sample, index in enumerate(indices)}
    decoded = set()
    tracks, closed, next_id = [], [], 0
    overlay_samples = {0, len(indices) // 2, len(indices) - 1}
    mask_store = {}
    unmatched_rcnn_total = 0
    w = h = None

    for index, bgr in sample_stream(a.video, backend, source_size, set(wanted)):
        sample = wanted[index]
        timestamp = float(times[sample])
        decoded.add(index)
        raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = raw.shape[:2]
        diag = float(np.hypot(w, h))
        padded, ow, oh = estimator.img_center_padding(raw)
        tensor, annotation = estimator._preprocess(padded)
        pl, pt, factor, _, _ = annotation
        if cameras:
            source_K = np.array(cameras[sample]["source_intrinsics"])
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

    shortfall = decode_shortfall(indices, decoded)
    if shortfall:
        raise RuntimeError(shortfall)
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
        sampleAuthority=authority,
        sampleAuthorityNote=(
            "cameras: the sampled source frames and their times were adopted from the solve's "
            "cameras.json, which selects them from decoded presentation timestamps. fps-rule: "
            "no cameras were supplied, so they came from round(time * container average fps), "
            "which only addresses the intended frames on a constant-rate source."
        ),
        decodeBackend=backend,
        decodeBackendReason=backend_reason,
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
