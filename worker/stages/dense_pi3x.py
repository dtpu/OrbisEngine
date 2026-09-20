#!/usr/bin/env python3
"""At least 12 independently reconstructed person frames/second, shared static anchors.

This increases temporal density, not hidden-side reconstruction. The single-view
surface limitation remains explicit. All chunks share a measured similarity frame.
"""

from __future__ import annotations
import argparse, gc, hashlib, json, os, shutil, sys, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wander_worker.masks import people_masks
from wander_worker.ply import write_point_ply, home_distance


# A similarity transform in three dimensions is only determined by three independent
# correspondences. Trimming below that leaves an unsolvable system whose SVD does not converge.
MINIMUM_CORRESPONDENCES = 3

# Confidence a reconstructed point needs before it anchors the shared frame.
ANCHOR_CONFIDENCE = 0.5

# Absolute confidence a point needs to count as reconstructed at all.
VALID_CONFIDENCE = 0.3

# When no point clears that bar, keep this share of each view's most confident points instead.
VALID_FALLBACK_FRACTION = 0.2

# Points a frame needs before its person is worth exporting.
MINIMUM_PERSON_POINTS = 100

# Share of frames that must reconstruct a person for the stage to be worth continuing with.
MINIMUM_PERSON_FRAMES = 0.5

# Share of requested frames that must actually decode before the source is called unusable.
MINIMUM_DECODED_FRAMES = 0.5


def decoded_span_usable(decoded: int, requested: int, floor=MINIMUM_DECODED_FRAMES) -> bool:
    """Whether enough of the requested span decoded to reconstruct from.

    Container metadata routinely overstates frame count, so the last requested frames can be
    undecodable even though the file is fine. Losing a short tail is normal; losing most of the
    clip means the source is broken and should be reported as such.
    """
    if requested == 0:
        return False
    return decoded / requested >= floor


def person_frames_usable(reconstructed: int, skipped: int, floor=MINIMUM_PERSON_FRAMES) -> bool:
    """Whether enough frames carry a person to be worth continuing with.

    A single sparse frame is normal: a subject can be occluded, leave the shot, or sit too far
    from camera to resolve. Aborting the stage on the first one throws away every frame that did
    reconstruct. Judge the run as a whole instead.
    """
    total = reconstructed + skipped
    if total == 0:
        return False
    return reconstructed / total >= floor


def confidence_cutoff(confidence, threshold=VALID_CONFIDENCE, fraction=VALID_FALLBACK_FRACTION):
    """The confidence a point must reach to count, for one view.

    Low-texture, motion-blurred or wide broadcast footage can leave every point under the
    absolute bar, which produced a reconstruction of nothing and an unreadable failure much
    further down. Falling back to the view's own most confident points keeps a weak
    reconstruction, which the alignment guard downstream can still reject on its merits.
    """
    finite = confidence[np.isfinite(confidence)]
    if finite.size == 0:
        return threshold, False
    if (finite > threshold).any():
        return threshold, False
    return float(np.quantile(finite, 1.0 - fraction)), True


def anchor_indices(*, valid, people, conf):
    """Pick the points that tie one anchor view into the shared frame.

    Static, confident, non-person geometry is what should carry the alignment. Footage that is
    mostly people over low-texture ground can leave that set empty, and an empty set used to
    travel silently into the solver. Relax instead, in order, and say which basis was used: a
    weaker anchor that reports itself beats no reconstruction at all.
    """
    attempts = (
        ("static", valid & (~people) & (conf > ANCHOR_CONFIDENCE)),
        ("low-confidence static", valid & (~people)),
        ("confident including people", valid & (conf > ANCHOR_CONFIDENCE)),
        ("any valid", valid),
    )
    for basis, mask in attempts:
        idx = np.flatnonzero(mask)
        if len(idx) >= MINIMUM_CORRESPONDENCES:
            return idx, basis
    return np.flatnonzero(valid), "any valid"


def similarity(src, dst):
    """Fit scale, rotation and translation taking src onto dst, trimming outliers.

    Short shots and sparse matches can leave very few usable correspondences, and depth can
    produce non-finite points. Both used to reach ``np.linalg.svd`` as a degenerate or NaN
    matrix, which fails as "SVD did not converge" and hides the real cause. Non-finite pairs are
    dropped up front, and outlier trimming never shrinks the set below a solvable system: a
    caller gets either a fit it can judge, or a message naming what was missing.
    """

    def fit(a, b):
        ma, mb = a.mean(0), b.mean(0)
        aa, bb = a - ma, b - mb
        U, S, V = np.linalg.svd(bb.T @ aa / len(a))
        D = np.eye(3)
        D[2, 2] = np.linalg.det(U @ V)
        R = U @ D @ V
        s = np.sum(S * np.diag(D)) / np.mean(np.sum(aa * aa, 1))
        return s, R, mb - s * R @ ma

    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) != len(dst):
        raise ValueError(f"correspondences must pair up: {len(src)} source, {len(dst)} target")
    finite = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if int(finite.sum()) < MINIMUM_CORRESPONDENCES:
        raise ValueError(
            f"anchor alignment needs at least {MINIMUM_CORRESPONDENCES} finite correspondences, "
            f"got {int(finite.sum())} of {len(src)}"
        )
    keep = finite.copy()
    s, R, t = fit(src[keep], dst[keep])
    error = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
    for _ in range(4):
        candidate = finite & (error <= np.percentile(error[finite], 75))
        if int(candidate.sum()) < MINIMUM_CORRESPONDENCES:
            break
        keep = candidate
        s, R, t = fit(src[keep], dst[keep])
        error = np.linalg.norm(src @ R.T * s + t - dst, axis=1)
    return s, R, t, float(np.sqrt(np.mean(error[keep] ** 2)))


def viewer_camera(
    pose,
    batch_scale,
    batch_rotation,
    batch_translation,
    reference_rotation,
    reference_translation,
    world_scale,
):
    flip = np.diag([1.0, -1.0, -1.0])
    center = batch_scale * batch_rotation @ pose[:3, 3] + batch_translation
    matrix = np.eye(4)
    matrix[:3, :3] = flip @ reference_rotation @ batch_rotation @ pose[:3, :3] @ flip
    matrix[:3, 3] = flip @ (reference_rotation @ center + reference_translation) * world_scale
    return matrix


def fit_ray_intrinsics(rays, valid, source_size):
    """Fit a pinhole K to learned rays; retain residual because rays need not be pinhole."""
    h, w = rays.shape[:2]
    y, x = np.mgrid[:h, :w]
    good = valid & np.isfinite(rays).all(-1) & (rays[..., 2] > 1e-6)
    good[1::2] = False
    good[:, 1::2] = False
    xy = rays[good, :2] / rays[good, 2:3]
    pixels = np.stack([x[good] + 0.5, y[good] + 0.5], axis=1)
    if len(xy) < 100:
        raise ValueError("Insufficient valid rays for intrinsic fit")
    K = np.eye(3)
    errors = []
    for axis in range(2):
        design = np.column_stack([xy[:, axis], np.ones(len(xy))])
        focal, principal = np.linalg.lstsq(design, pixels[:, axis], rcond=None)[0]
        if focal <= 0:
            raise ValueError("Nonpositive fitted focal length")
        K[axis, axis], K[axis, 2] = focal, principal
        errors.append(design @ [focal, principal] - pixels[:, axis])
    source_K = np.diag([source_size[0] / w, source_size[1] / h, 1.0]) @ K
    return dict(
        intrinsics=K.tolist(),
        source_intrinsics=source_K.tolist(),
        image_size=[w, h],
        source_image_size=list(source_size),
        intrinsic_fit_pixel_rmse=float(np.sqrt(np.mean(np.sum(np.square(errors), axis=0)))),
        intrinsics_method="Least-squares zero-skew pinhole fit to predicted rays; pixel centers at x+0.5,y+0.5. Source K undoes the two resize operations. This is estimated, not supplied calibration.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repo", default=os.path.expanduser("~/pi3"))
    a = ap.parse_args()
    if a.fps < 12:
        raise ValueError("Dense quality experiment requires >=12 fps")
    if a.anchors < 2 or a.batch < 1:
        raise ValueError("Need >=2 anchors and a positive batch size")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    framesdir = out / "input"
    framesdir.mkdir(exist_ok=True)
    temp = out / "batch-input"
    temp.mkdir(exist_ok=True)
    t0 = time.time()
    cap = cv2.VideoCapture(a.video)
    srcfps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / srcfps
    if not np.isfinite(srcfps) or srcfps <= 0 or count < 2:
        raise ValueError("Invalid source video metadata")
    source_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    requested = np.arange(int(np.ceil(duration * a.fps))) / a.fps
    indices = np.clip(np.round(requested * srcfps).astype(int), 0, count - 1)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Source frame rate below requested independent output density")
    times = indices / srcfps
    N = len(indices)
    decoded = 0
    for i, idx in enumerate(indices):
        file = framesdir / f"f_{i:06d}.jpg"
        if file.exists():
            decoded = i + 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            # The container claimed more frames than it holds. Keep the span that exists.
            print(
                f"decode stopped at source frame {idx} ({i} of {len(indices)} requested)",
                flush=True,
            )
            break
        if bgr.shape[1] > 1024:
            bgr = cv2.resize(bgr, (1024, round(bgr.shape[0] * 1024 / bgr.shape[1])))
        cv2.imwrite(str(file), bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        decoded = i + 1
    cap.release()
    if decoded < len(indices):
        if not decoded_span_usable(decoded, len(indices)):
            raise RuntimeError(
                f"Decode failed: only {decoded} of {len(indices)} requested frames are readable"
            )
        indices, times = indices[:decoded], times[:decoded]
        N = decoded
    files = sorted(framesdir.glob("f_*.jpg"))
    import torch

    sys.path.insert(0, a.repo)
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_images_as_tensor
    from pi3.utils.geometry import depth_normal_edge

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().cuda()
    model.disable_multimodal()

    def predict(ids):
        for file in temp.glob("*.jpg"):
            file.unlink()
        for idx in ids:
            os.symlink(files[idx].resolve(), temp / files[idx].name)
        imgs = load_images_as_tensor(str(temp), verbose=False)[None].cuda()
        print("inference tensor", tuple(imgs.shape), flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            r = model(imgs=imgs)
        confidence = torch.sigmoid(r["conf"][0, ..., 0])
        valid = confidence > VALID_CONFIDENCE
        for j in range(valid.shape[0]):
            if bool(valid[j].any()):
                continue
            cutoff, relaxed = confidence_cutoff(confidence[j].float().cpu().numpy())
            if relaxed:
                valid[j] = confidence[j] >= cutoff
                print(
                    f"view {ids[j]}: no point reached confidence {VALID_CONFIDENCE}; "
                    f"keeping the top {VALID_FALLBACK_FRACTION:.0%} above {cutoff:.3f}",
                    flush=True,
                )
        edge = depth_normal_edge(r["local_points"], rtol=0.03, mask=valid[None])[0]
        result = dict(
            points=r["points"][0].float().cpu().numpy(),
            poses=r["camera_poses"][0].float().cpu().numpy(),
            rays=r["rays"][0].float().cpu().numpy(),
            conf=confidence.float().cpu().numpy(),
            valid=(valid & ~edge).cpu().numpy(),
            rgb=(imgs[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype("uint8"),
        )
        del imgs, r
        torch.cuda.empty_cache()
        return result

    anchors = np.unique(np.linspace(0, N - 1, min(a.anchors, N)).round().astype(int)).tolist()
    basefile = out / "anchors.npz"
    if basefile.exists():
        base = dict(np.load(basefile))
    else:
        base = predict(anchors)
        base["people"] = people_masks(base["rgb"], dilate_px=2)
        np.savez_compressed(basefile, **base)
    inv = np.linalg.inv(base["poses"][0])
    R0, t00 = inv[:3, :3], inv[:3, 3]
    flip = np.array([1.0, -1.0, -1.0])
    zz = (base["points"][0] @ R0.T + t00)[..., 2]
    good = base["valid"][0] & (~base["people"][0]) & (zz > 0)
    scale = 3 / np.median(zz[good])
    rng = np.random.default_rng(13)
    matches = []
    for j in range(len(anchors)):
        idx, basis = anchor_indices(
            valid=base["valid"][j], people=base["people"][j], conf=base["conf"][j]
        )
        if len(idx) < MINIMUM_CORRESPONDENCES:
            raise ValueError(
                f"anchor {j} has no usable points: valid={int(base['valid'][j].sum())}, "
                f"non-person={int((~base['people'][j]).sum())}, "
                f"conf>{ANCHOR_CONFIDENCE}={int((base['conf'][j] > ANCHOR_CONFIDENCE).sum())}"
            )
        if basis != "static":
            print(f"anchor {j}: falling back to {basis} points", flush=True)
        matches.append(rng.choice(idx, min(1500, len(idx)), replace=False))
    cams = [None] * N
    log = []
    point_counts = []
    sparse_frames = []
    camera_dir = out / "camera-batches"
    camera_dir.mkdir(exist_ok=True)
    for lo in range(0, N, a.batch):
        chunk = list(range(lo, min(N, lo + a.batch)))
        ids = sorted(set(anchors + chunk))
        print("Pi3X batch", lo, "/", N, "views", len(ids), flush=True)
        p = predict(ids)
        src = np.concatenate(
            [p["points"][ids.index(ai)].reshape(-1, 3)[matches[j]] for j, ai in enumerate(anchors)]
        )
        dst = np.concatenate(
            [base["points"][j].reshape(-1, 3)[matches[j]] for j in range(len(anchors))]
        )
        s, R, t, rms = similarity(src, dst)
        print("alignment", float(s), rms, flush=True)
        if not np.isfinite(rms) or not 0.1 < s < 10:
            raise RuntimeError("Invalid anchor alignment")
        np.savez_compressed(
            camera_dir / f"batch_{lo:03d}.npz",
            sample_ids=ids,
            source_indices=indices[ids],
            raw_opencv_camera_to_world=p["poses"],
            batch_rotation=R,
            batch_translation=t,
            batch_scale=s,
            reference_rotation=R0,
            reference_translation=t00,
            world_scale=scale,
        )
        # Full-resolution masks preserve limbs; resize to the actual Pi3X pixel grid.
        full = np.stack([np.asarray(Image.open(files[i])) for i in ids])
        pm = people_masks(full, dilate_px=1)
        H, W = p["rgb"].shape[1:3]
        pm = np.stack(
            [cv2.resize(m.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in pm]
        )
        for i in chunk:
            j = ids.index(i)
            valid = p["valid"][j] & pm[j] & np.isfinite(p["points"][j]).all(-1)
            xyz = p["points"][j][valid] @ R.T * s + t
            col = p["rgb"][j][valid]
            xyz = (xyz @ R0.T + t00) * scale * flip
            sparse = len(xyz) < MINIMUM_PERSON_POINTS
            if sparse:
                # The subject could not be reconstructed here, so write no person cloud. The
                # camera is solved from the view itself and does not depend on the subject, so
                # it is still recorded: dropping it would leave a hole that every consumer of
                # cameras.json has to reason about, including the anchors this solve is built on.
                sparse_frames.append((float(times[i]), int(len(xyz))))
                print(
                    f"frame {i} at {times[i]:.3f}s: only {len(xyz)} person points, "
                    "recording the camera without a person cloud",
                    flush=True,
                )
            else:
                write_point_ply(out / f"frame_{i:03d}.ply", xyz.astype("float32"), col)
                point_counts.append(len(xyz))
            camera = viewer_camera(p["poses"][j], s, R, t, R0, t00, scale)  # solved per view
            cams[i] = dict(
                position=camera[:3, 3].tolist(),
                time=float(times[i]),
                sourceIndex=int(indices[i]),
                camera_to_world=camera.tolist(),
                world_to_camera=np.linalg.inv(camera).tolist(),
                **fit_ray_intrinsics(p["rays"][j], p["valid"][j], source_size),
            )
            if i == 0:
                np.savez_compressed(
                    out / "source-camera-0.npz",
                    rays=p["rays"][j],
                    rgb=p["rgb"][j],
                    valid=p["valid"][j],
                    camera_to_world=camera,
                )
            if i == 0:
                home = home_distance(xyz)
        log.append(
            dict(
                first=lo,
                frames=len(chunk),
                scale=float(s),
                rotation=R.tolist(),
                translation=t.tolist(),
                rms=rms,
            )
        )
        (out / "alignment-batches.json").write_text(json.dumps(log, indent=2))
        del p, pm, full
        gc.collect()
        torch.cuda.empty_cache()
    meta = dict(
        backend="pi3x-dense-anchors",
        frames=[f"frame_{i:03d}.ply" for i in range(N)],
        count=N,
        fps=a.fps,
        timestamps=times.tolist(),
        duration=duration,
        sourceFps=srcfps,
        sourceIndices=indices.tolist(),
        people_only=True,
        home_distance=home,
        sourceSha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),
        seconds=time.time() - t0,
        peakVRAMGB=torch.cuda.max_memory_allocated() / 1e9,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        anchors=anchors,
        batch=a.batch,
        pointCounts=point_counts,
        note="Each source frame reconstructed with shared static anchors. >=12 independent frames/sec; single observed-facing surface, not complete human volume.",
    )
    (out / "sequence.json").write_text(json.dumps(meta, indent=2))
    if not person_frames_usable(len(point_counts), len(sparse_frames)):
        worst = ", ".join(f"{t:.2f}s={n}" for t, n in sparse_frames[:6])
        raise RuntimeError(
            f"Person reconstructed in only {len(point_counts)} of "
            f"{len(point_counts) + len(sparse_frames)} frames "
            f"(need {MINIMUM_PERSON_FRAMES:.0%}); sparse frames: {worst}"
        )
    if sparse_frames:
        print(
            f"{len(sparse_frames)} of {len(point_counts) + len(sparse_frames)} frames had too "
            f"few person points and were skipped",
            flush=True,
        )
    (out / "cameras.json").write_text(
        json.dumps(
            dict(
                coordinates="Both world and camera bases are OpenGL: x right, y up, -z forward. World matches exported PLY. Target camera zero is not assumed identity after batch registration.",
                reference=dict(
                    rotation=R0.tolist(),
                    translation=t00.tolist(),
                    scale=float(scale),
                    world_axis_flip=flip.tolist(),
                ),
                cameras=cams,
            ),
            indent=1,
        )
    )
    print(
        json.dumps(
            {k: v for k, v in meta.items() if k not in ("frames", "timestamps", "sourceIndices")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
