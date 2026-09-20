#!/usr/bin/env python3
"""One canonical LHM appearance animated by independently estimated source poses."""

import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# worker/modal_multiperson.py mounts all of worker/stages as /root/stages, so the solver's and
# the tracker's own modules travel with this one: their sample plan and their sequential decode
# are imported rather than recomputed here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_pi3x import choose_decode_backend, opencv_first_frame
from track_people import fps_rule_indices, sample_stream, source_metadata

SAMPLE_AUTHORITY_CAMERAS = "cameras"
SAMPLE_AUTHORITY_SEED = "seed-track"
SAMPLE_AUTHORITY_FPS_RULE = "fps-rule"
SAMPLE_AUTHORITY_NOTE = (
    "cameras: the sampled source frames and their times were adopted from the solve's "
    "cameras.json, which selects them from decoded presentation timestamps. seed-track: adopted "
    "from the tracked person's own samples, which came from that same solve. fps-rule: neither "
    "was supplied, so they came from round(time * container average fps), which only addresses "
    "the right frames on a constant-rate source."
)


def adopted_samples(indices, times, authority):
    """A supplied sample list, checked for the shape every per-sample record depends on."""
    indices = np.asarray(indices, dtype=int)
    times = np.asarray(times, dtype=float)
    if len(indices) < 1:
        raise ValueError(f"The supplied {authority} sample list is empty")
    if len(indices) != len(times):
        raise ValueError(f"The supplied {authority} indices and times differ in length")
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError(
            f"The supplied {authority} samples must name increasing, unique source indices"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError(f"The supplied {authority} sample times are not finite")
    return indices, times, authority


def plan_samples(cameras, seed_indices, seed_times, fps_out, src_fps, count, start=0.0, stop=None):
    """Which source frames to animate, at what times, and which rule chose them.

    A supplied solve is the authority. dense_pi3x.py selects its samples from the decoded
    presentation timestamps the way ffmpeg's `fps` filter does, so on a variable-rate source they
    are not `round(time * average_fps)` at all and this stage's own rule could only disagree with
    them -- it aborted on the mismatch before any GPU work, and where it did agree it then used a
    decode-order index as a seek target. Adopting each camera's `sourceIndex` and `time` (or, with
    no cameras, the tracked person's own samples, which came from that same solve) keeps every
    pose paired with the camera that reconstructed its frame. With neither supplied the legacy
    constant-rate rule stands, and says so.
    """
    if cameras is not None:
        indices, times, authority = adopted_samples(
            [int(c["sourceIndex"]) for c in cameras],
            [float(c["time"]) for c in cameras],
            SAMPLE_AUTHORITY_CAMERAS,
        )
    elif seed_indices is not None:
        supplied = [int(i) for i in seed_indices]
        indices, times, authority = adopted_samples(
            supplied,
            [float(t) for t in seed_times]
            if seed_times is not None
            else [index / float(src_fps) for index in supplied],
            SAMPLE_AUTHORITY_SEED,
        )
    else:
        duration = count / src_fps
        stop = duration if stop is None else stop
        indices, times = fps_rule_indices(fps_out, src_fps, count, start, stop)
        if np.any(times < start) or np.any(times >= stop):
            raise ValueError("Rounded samples leave the requested interval")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("Repeated source frames")
        authority = SAMPLE_AUTHORITY_FPS_RULE
    return indices, times, authority


def requested_samples(indices, times, seed_poses, track_only):
    """Which samples this run is asked to produce, and the source frame and time each one means.

    A `--track-only` run animates one tracked person, and that person exists in the samples the
    tracker found them in. The solve's other samples are the track's gaps -- it walked out of
    shot, or nobody could be detected there -- not work this run was asked for and did not do,
    and counting them as requested is what made a complete 90-of-96 track certify as partial
    coverage. Which source frame and time a sample number stands for still comes from the
    supplied cameras: the seed track decides which samples this person exists in, the cameras
    stay the authority for what those samples address.
    """
    if seed_poses is not None and len(seed_poses) != len(indices):
        raise ValueError("Seed poses and sampled source indices differ in length")
    kept = []
    for sample, (index, timestamp) in enumerate(zip(indices, times)):
        if track_only and (seed_poses is None or seed_poses[sample] is None):
            continue
        kept.append((int(sample), int(index), float(timestamp)))
    return kept


def main():
    ap = argparse.ArgumentParser()
    for key in ("video", "canonical", "reference", "out", "model"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--repo", default="/opt/lhm")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--cameras")
    ap.add_argument("--depth-reference")
    ap.add_argument("--seed-poses")
    ap.add_argument("--seed-motion")
    ap.add_argument(
        "--track-only",
        action="store_true",
        help="Seed poses are one tracked person: animate exactly those samples, never re-detect (a re-detection would pick whichever person is nearest and swap identity)",
    )
    ap.add_argument("--track-id", type=int, help="Track index recorded in sequence.json")
    ap.add_argument(
        "--depth-roi",
        help="x0,y0,x1,y1 source-pixel box; the depth reference holds every masked person in the frame, so restrict the observed median to this track",
    )
    ap.add_argument(
        "--fixed-world-scale",
        type=float,
        help="Separate camera-model control: retain an explicitly supplied native/world scale without depth or floor refitting",
    )
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--overlay-all", action="store_true")
    ap.add_argument(
        "--start",
        type=float,
        default=0,
        help="Bundled-video interval start; source bytes are not trimmed",
    )
    ap.add_argument("--stop", type=float, help="Exclusive bundled-video interval end")
    ap.add_argument(
        "--original-source-offset",
        type=float,
        default=0,
        help="Recorded original-recording time minus bundled-video time",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(a.repo)
    sys.path.insert(0, a.repo)
    start = time.time()
    import torch
    from accelerate import Accelerator
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
    from engine.pose_estimation.pose_estimator import PoseEstimator
    from lhm_person import pose_parameters

    torch._dynamo.config.disable = True
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    Accelerator()
    source_sha = hashlib.sha256(Path(a.video).read_bytes()).hexdigest()
    if not 0.1 <= a.detection_threshold <= 0.3:
        raise ValueError("Detection threshold outside bounded experiment range")
    if a.fixed_world_scale is not None and (not a.cameras or not 0 < a.fixed_world_scale < 10):
        raise ValueError("Fixed scale requires supplied cameras and a finite positive scale")
    reference = json.loads(Path(a.reference).read_text())
    if source_sha != reference["prepared"]["sourceSha256"]:
        raise ValueError("Canonical appearance and motion must come from the same source video")
    fps, count, source_size = source_metadata(a.video)
    duration = count / fps
    if a.fps < 12 or fps < a.fps:
        raise ValueError("Require at least 12 distinct source frames per second")
    stop = duration if a.stop is None else a.stop
    if (
        not np.isfinite([a.start, stop, a.original_source_offset]).all()
        or not 0 <= a.start < stop <= duration
    ):
        raise ValueError("Invalid source interval")
    camera_doc = json.loads(Path(a.cameras).read_text()) if a.cameras else None
    cameras = camera_doc["cameras"] if camera_doc else None
    seed = (
        torch.load(a.seed_poses, map_location="cpu", weights_only=False) if a.seed_poses else None
    )
    seed_records = (
        {r["sample"]: r for r in json.loads(Path(a.seed_motion).read_text())["frames"]}
        if seed
        else {}
    )
    # Match the dense camera export's divide-then-multiply order at half-frame ties when this
    # stage has to choose its own samples; adopt the solve's when it does not.
    indices, times, authority = plan_samples(
        cameras,
        seed["sourceIndices"] if seed else None,
        seed.get("timestamps") if seed else None,
        a.fps,
        fps,
        count,
        a.start,
        stop,
    )
    print(f"{len(indices)} samples from {authority}", flush=True)
    # Retained as guards, not as a schedule: after adoption these can only fire if the cameras
    # and the seed track disagree with each other or a record is malformed.
    if cameras and (
        len(cameras) != len(indices)
        or any(c["sourceIndex"] != int(i) for c, i in zip(cameras, indices))
    ):
        raise ValueError("Camera records must correspond exactly to sampled source indices")
    if seed and not np.array_equal(np.asarray(seed["sourceIndices"], dtype=int), indices):
        raise ValueError("Seed pose source samples differ")
    if seed and not a.track_only and all(p is not None for p in seed["poses"]):
        raise ValueError("Seed has no missing source poses to recover")
    if a.track_only and not seed:
        raise ValueError("Track mode requires the track's seed poses")
    requested = requested_samples(
        indices, times, seed["poses"] if seed else None, bool(a.track_only)
    )
    print(f"{len(requested)} of those samples are requested from this run", flush=True)
    estimator = (
        None
        if a.track_only
        else PoseEstimator("./pretrained_models/human_model_files", device="cuda")
    )
    # Only the samples this run has to re-estimate are decoded, and strictly forward. Nothing
    # seeks: `CAP_PROP_POS_FRAMES` cannot reach the tail of a variable-rate file and can land on
    # a neighbouring frame elsewhere in it, while these indices name positions in the decoder's
    # own sequence -- the ordinals dense_pi3x.py counted when it solved the cameras.
    wanted = (
        set()
        if a.track_only
        else {
            int(index)
            for sample, index in enumerate(indices)
            if not (seed and seed["poses"][sample] is not None)
        }
    )
    backend, backend_reason = choose_decode_backend(*opencv_first_frame(cv2, a.video))
    frames = sample_stream(a.video, backend, source_size, wanted) if wanted else iter(())
    if wanted:
        print(f"decode backend {backend} ({backend_reason})", flush=True)
    poses = []
    records = []
    missing = []
    gaps = []
    last_center = None
    for sample, (index, timestamp) in enumerate(zip(indices, times)):
        if seed and seed["poses"][sample] is not None:
            pose = seed["poses"][sample]
            record = dict(seed_records[sample])
            record["retainedOriginalPose"] = True
            poses.append(pose)
            records.append(record)
            K0 = np.array(record["source_intrinsics"])
            projected = pose["j3d"][:22].numpy() @ K0.T
            # Source dimensions are fixed; project the retained person to track only missing samples.
            wh = np.array(
                [reference["prepared"]["sourceWidth"], reference["prepared"]["sourceHeight"]]
            )
            last_center = np.median(projected[:, :2] / projected[:, 2:3], axis=0) / wh
            continue
        if a.track_only:
            # A gap in the track, not a sample this run was asked for: the tracker already
            # decided this person is not in this sample, so nothing here was left undone.
            gaps.append(
                dict(
                    sample=sample,
                    sourceIndex=int(index),
                    time=float(timestamp),
                    reason="Track absent from this sample (out of frame, undetected, or occluded)",
                )
            )
            poses.append(None)
            records.append(None)
            continue
        decoded, bgr = next(frames, (None, None))
        if decoded is None:
            raise RuntimeError(
                f"Unable to decode source frame {index} for sample {sample}: the source stopped "
                f"handing back frames before it ({backend}: {backend_reason})"
            )
        if int(decoded) != int(index):
            raise RuntimeError(
                f"Decoded source frame {int(decoded)} where sample {sample} asked for {int(index)}"
            )
        raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = raw.shape[:2]
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
                det_thresh=0.1 if seed else a.detection_threshold,
                K=K,
                idx=None,
                max_dist=None,
            )
        choices = []
        for person in detected:
            joints = (person["j2d"].float().cpu().numpy() - [pl, pt]) / factor - [ow, oh]
            center = np.median(joints[:22], axis=0) / [w, h]
            score = float(person["scores"])
            distance = float(np.linalg.norm(center - last_center)) if last_center is not None else 0
            choices.append((score - distance * 2, person, joints, center, distance))
        choices.sort(key=lambda x: x[0], reverse=True)
        if not choices or (last_center is not None and choices[0][4] > 0.35):
            missing.append(
                dict(
                    sample=sample,
                    sourceIndex=int(index),
                    time=float(timestamp),
                    reason="No consistent source person detected",
                )
            )
            print("missing pose", missing[-1], flush=True)
            poses.append(None)
            records.append(None)
            continue
        _, person, joints, last_center, _ = choices[0]
        pose = {
            k: v.detach().float().cpu()
            for k, v in person.items()
            if k in ("rotvec", "shape", "transl_pelvis", "j3d", "scores")
        }
        poses.append(pose)
        records.append(
            dict(
                sample=sample,
                sourceIndex=int(index),
                time=float(timestamp),
                detectedPeople=len(detected),
                score=float(pose["scores"]),
                originalSourceTime=float(timestamp + a.original_source_offset),
                sourceFrameRGBSha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                projectedBodyJoints=joints[:22].tolist(),
                jointProjectionInImage=(
                    (joints[:22, 0] >= 0)
                    & (joints[:22, 0] < w)
                    & (joints[:22, 1] >= 0)
                    & (joints[:22, 1] < h)
                ).tolist(),
                confidenceSemantics="Score is person detection confidence, not per-joint accuracy or visibility confidence.",
                source_intrinsics=source_K.tolist(),
                rootCamera=pose["j3d"][0].tolist(),
                smplTranslationCamera=pose["transl_pelvis"].reshape(3).tolist(),
                feetCamera=pose["j3d"][[7, 8, 10, 11]].tolist(),
                footJointOrder=["leftAnkle", "rightAnkle", "leftFoot", "rightFoot"],
                rootRotationVector=pose["rotvec"][0].tolist(),
                detectionThreshold=0.1 if seed else a.detection_threshold,
                recoveredMissingPose=bool(seed),
            )
        )
        if seed or a.overlay_all or sample in (0, len(indices) // 2, len(indices) - 1):
            overlay = raw.copy()
            for x, y in joints:
                cv2.circle(overlay, (round(float(x)), round(float(y))), 4, (255, 0, 0), -1)
            Image.fromarray(overlay).save(out / f"pose-overlay-{sample:03d}.jpg")
        print(
            "source pose", sample + 1, "/", len(indices), "score", float(pose["scores"]), flush=True
        )
    if hasattr(frames, "close"):
        frames.close()  # Stops the decoder as soon as the last wanted frame has been used.
    if not a.track_only:
        del tensor, detected, K, choices
    del estimator
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(dict(poses=poses, sourceIndices=indices, timestamps=times), out / "source-poses.pt")
    (out / "missing-poses.json").write_text(json.dumps(missing, indent=2))
    first = next((i for i, p in enumerate(poses) if p is not None), None)
    if first is None:
        raise RuntimeError("Track has no usable source pose")
    if first != 0 and not a.track_only:
        raise RuntimeError("Source pose zero missing; cannot fit a fixed depth scale")
    from LHM.models import model_dict
    from LHM.utils.hf_hub import wrap_model_hub

    print("Loading canonical appearance and native LHM skinning", flush=True)
    model = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"]).from_pretrained(a.model)
    model.eval()
    model.cuda()
    state = torch.load(a.canonical, map_location="cuda", weights_only=False)
    frames = []
    hashes = []
    export_times = []
    used_indices = []
    world_scale = a.fixed_world_scale
    registration = None
    if world_scale is not None:
        registration = dict(
            uniformScale=world_scale,
            method="Explicit native/world scale retained from the control. No first-frame depth normalization, floor fitting or per-frame grounding adjustment.",
        )
        (out / "registration.json").write_text(json.dumps(registration, indent=2))
    for sample, pose in enumerate(poses):
        if pose is None:
            continue
        params = pose_parameters(pose, betas=state["params"]["betas"])
        params["transform_mat_neutral_pose"] = state["neutral"]
        with torch.inference_mode():
            gs = model.animation_infer_gs(state["attrs"], state["query"], params)
        if not all(torch.isfinite(x).all() for x in (gs.xyz, gs.opacity, gs.scaling, gs.rotation)):
            raise RuntimeError("Non-finite animated Gaussians")
        record = records[sample]
        if cameras:
            camera = np.array(cameras[sample]["camera_to_world"])
            rotation = camera[:3, :3]
            u, _, vt = np.linalg.svd(rotation)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                raise RuntimeError("Reflected source camera")
            cv_to_world = rotation @ np.diag([1.0, -1.0, -1.0])
            if world_scale is None:
                from plyfile import PlyData

                p = PlyData.read(a.depth_reference)["vertex"].data
                xyz = np.column_stack([p["x"], p["y"], p["z"]])
                local = (xyz - camera[:3, 3]) @ np.linalg.inv(camera[:3, :3]).T
                roi_kept = None
                if a.depth_roi:
                    x0, y0, x1, y1 = [float(v) for v in a.depth_roi.split(",")]
                    K0 = np.array(record["source_intrinsics"])
                    cam_cv = local * np.array([1.0, -1.0, -1.0])
                    uvw = cam_cv @ K0.T
                    front = uvw[:, 2] > 1e-6
                    uv = np.full((len(uvw), 2), -1e9)
                    uv[front] = uvw[front, :2] / uvw[front, 2:3]
                    inside = (
                        front
                        & (uv[:, 0] >= x0)
                        & (uv[:, 0] <= x1)
                        & (uv[:, 1] >= y0)
                        & (uv[:, 1] <= y1)
                    )
                    if inside.sum() < 50:
                        raise RuntimeError(
                            f"Only {int(inside.sum())} depth-reference points inside the track ROI"
                        )
                    local = local[inside]
                    roi_kept = int(inside.sum())
                observed_depth = -local[:, 2]
                observed_depth = observed_depth[np.isfinite(observed_depth) & (observed_depth > 0)]
                opaque = gs.opacity[:, 0] > 0.5
                native_depth = gs.xyz[opaque, 2]
                native_depth = native_depth[torch.isfinite(native_depth) & (native_depth > 0)]
                observed_median = float(np.median(observed_depth))
                native_median = float(native_depth.median())
                world_scale = observed_median / native_median
                if not np.isfinite(world_scale) or world_scale <= 0:
                    raise RuntimeError("Invalid first-frame depth registration")
                registration = dict(
                    uniformScale=world_scale,
                    observedPersonMedianDepth=observed_median,
                    nativeGaussianMedianDepth=native_median,
                    registrationSample=sample,
                    registrationSourceIndex=int(indices[sample]),
                    depthRoi=a.depth_roi,
                    depthRoiPoints=roi_kept,
                    method="One positive scale from first-frame person camera-space median depths, held fixed throughout. Pi3X observed person points versus opaque LHM body Gaussians; body-depth prior, not pixel-correspondence or floor fitting.",
                )
                (out / "registration.json").write_text(json.dumps(registration, indent=2))
                print("registration", registration, flush=True)
            rotation_t = torch.tensor(cv_to_world, dtype=torch.float32, device="cuda")
            translation_t = torch.tensor(camera[:3, 3], dtype=torch.float32, device="cuda")
            gs.xyz = (gs.xyz @ rotation_t.T) * world_scale + translation_t
            gs.scaling = gs.scaling * world_scale
            gs.rotation = quaternion_multiply(matrix_to_quaternion(rotation_t)[None], gs.rotation)
            record["rootWorld"] = (
                cv_to_world @ np.array(record["rootCamera"]) * world_scale + camera[:3, 3]
            ).tolist()
            record["feetWorld"] = (
                np.array(record["feetCamera"]) @ cv_to_world.T * world_scale + camera[:3, 3]
            ).tolist()
            record["camera_to_world"] = camera.tolist()
        name = f"frame_{sample:03d}.ply"
        gs.save_ply(str(out / name))
        frames.append(name)
        hashes.append(hashlib.sha256((out / name).read_bytes()).hexdigest())
        export_times.append(float(times[sample]))
        used_indices.append(int(indices[sample]))
        print("animated Gaussian frame", sample + 1, "/", len(poses), flush=True)
    valid_records = [r for r in records if r is not None]
    for previous, current in zip(valid_records, valid_records[1:]):
        dt = current["time"] - previous["time"]
        for key in ("rootWorld", "feetWorld"):
            if key in current:
                current[key + "Velocity"] = ((np.array(current[key]) - previous[key]) / dt).tolist()
    motion = dict(
        coordinates="Raw roots/feet: OpenCV source camera. World roots/feet: exported Gaussian OpenGL world when camera metadata supplied.",
        footContact=dict(
            status="unmeasured",
            reason="No measured floor plane or contact constraint. Feet are source-pose joint estimates; no snapping or synthetic stance.",
            smplxJointIndices=[7, 8, 10, 11],
            semantics=[
                "left ankle joint",
                "right ankle joint",
                "left foot joint",
                "right foot joint",
            ],
            soleOrHeelVertices=False,
        ),
        canonicalShapeFixed=True,
        frames=valid_records,
        missing=missing,
        trackGaps=gaps,
    )
    (out / "motion.json").write_text(json.dumps(motion, indent=2))
    sequence = dict(
        frame_format="gaussian-ply",
        frames=frames,
        count=len(frames),
        fps=a.fps,
        timestamps=export_times,
        frame_sha256=hashes,
        sourceIndices=used_indices,
        sourceFps=fps,
        sampleAuthority=authority,
        sampleAuthorityNote=SAMPLE_AUTHORITY_NOTE,
        decodeBackend=backend,
        decodeBackendReason=backend_reason,
        sourceSha256=source_sha,
        duration=duration,
        solvedSamples=len(indices),
        requestedSamples=len(requested),
        requestedSampleSemantics=(
            "Samples this run was asked to animate. In track mode that is the tracked person's "
            "own samples, taken from the seed track and addressed by the supplied cameras; the "
            "solve's remaining samples are listed as trackGapSamples and were never requested, "
            "because the tracker had already found this person absent from them."
        ),
        missingPoseSamples=missing,
        trackGapSamples=gaps,
        allRequestedSamplesReconstructed=not missing,
        canonicalStateSha256=hashlib.sha256(Path(a.canonical).read_bytes()).hexdigest(),
        poseRecovery=dict(
            originalPosesRetained=len(seed_records),
            lowerDetectionThreshold=0.1,
            recoveredSamples=[r["sample"] for r in valid_records if r.get("recoveredMissingPose")],
        )
        if (seed and not a.track_only)
        else None,
        trackId=a.track_id,
        trackOnly=a.track_only,
        firstReconstructedSample=first,
        referenceTime=reference["prepared"]["time"],
        canonicalShapeFixed=True,
        people_only=True,
        sourceInterval=dict(
            start=a.start,
            stopExclusive=stop,
            originalSourceOffset=a.original_source_offset,
            originalStart=a.start + a.original_source_offset,
            originalStopExclusive=stop + a.original_source_offset,
        ),
        coordinates=(
            "OpenGL supplied fixed-SfM world; explicitly retained control scale"
            if a.fixed_world_scale is not None
            else "OpenGL Pi3X world; one fixed first-frame body-depth scale"
        )
        if cameras
        else "OpenCV source camera for each timestamp; world motion unregistered",
        fixedWorldScaleControl=a.fixed_world_scale,
        sourcePosesReestimatedWithSuppliedIntrinsics=bool(a.fixed_world_scale is not None),
        cameraMetadataSha256=hashlib.sha256(Path(a.cameras).read_bytes()).hexdigest()
        if cameras
        else None,
        seconds=time.time() - start,
        peakVRAMGB=torch.cuda.max_memory_allocated() / 1e9,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        note="One source-conditioned canonical appearance. Independent MultiHMR pose per decoded source sample; no stock motion, pose interpolation, or appearance regeneration. Hidden surfaces remain learned inference.",
    )
    (out / "sequence.json").write_text(json.dumps(sequence, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in sequence.items()
                if k not in ("frames", "timestamps", "frame_sha256", "sourceIndices")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
