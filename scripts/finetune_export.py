#!/usr/bin/env python
"""Export a generated world and its solved camera frames into a gsplat training set.

Uses the mapping in bake_video_colours.py; the cameras come from a plain
cameras.json (Pi3X, OpenGL camera_to_world, camera 0 ~ identity) instead of a wvp container, the
person masks are recomputed with SegFormer, and the depth layers are z-buffered from the Pi3X
anchor point cloud instead of read from the wvp.

    native = origin_c2w * (Rx(pi) raw / scale0)      scale0 fitted with bake_video_colours.py --diag

Output: splats.npz, cameras.npz, frames/%04d.jpg, masks/%04d.png (255 = excluded),
depth.npz, static-scene.npy and export-receipt.json.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bake_video_colours import (
    ROOT,
    read_spz,
    static_cloud_from_anchors,
    zbuffer_depth,
)  # noqa: E402
from local_clean_video import sha256, validate_masks, write_report  # noqa: E402
from sfm_frame import framealign_path  # noqa: E402
from sfm_frame import origin_c2w as levelled_origin, describe as describe_frame  # noqa: E402
from worker.wander_worker.source_timing import match_camera_selection, select_source_frames  # noqa: E402


def quat_mul(a, b):
    """xyzw Hamilton product, batched."""
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        1,
    )


def rotmat_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def decode_frames(clip, indices, size, fps=12.0):
    width, height = size
    rgb, provenance = select_source_frames(
        clip, fps, source_indices=indices, width=width, height=height
    )
    if len(rgb) != len(indices) * height * width * 3:
        raise ValueError("Decoded RGB frame count does not match requested source frames")
    return np.frombuffer(rgb, dtype=np.uint8).reshape(len(indices), height, width, 3), provenance


def supplied_mask_inputs(clip, masks_path, cameras_path, camera_doc, size):
    """Bind masks, camera labels and decoded RGB to the same source samples."""
    cameras = camera_doc["cameras"]
    masks_sha = sha256(masks_path)
    archive = validate_masks(clip, masks_path)
    if sha256(masks_path) != masks_sha:
        raise ValueError("Masks changed during validation")
    _, source_height, source_width = map(int, archive["shape"])
    indices = archive["source_indices"].tolist()
    source_sha = str(archive["source_sha256"])
    camera_indices = [camera.get("sourceIndex") for camera in cameras]
    if any(type(index) is not int for index in camera_indices) or camera_indices != indices:
        raise ValueError("Masks source indices do not match camera source indices")
    for camera in cameras:
        dimensions = camera.get("source_image_size")
        if (
            not isinstance(dimensions, list)
            or any(type(value) is not int for value in dimensions)
            or dimensions != [source_width, source_height]
        ):
            raise ValueError("Camera dimensions do not match source mask dimensions")

    sequence_path = None
    sequence_sha = None
    fps = 12.0
    if "sourceSha256" in camera_doc:
        camera_source_sha = camera_doc["sourceSha256"]
    else:
        sequence_path = cameras_path.with_name("sequence.json")
        if not sequence_path.is_file():
            raise ValueError("Camera source binding requires sourceSha256 or sibling sequence.json")
        sequence_bytes = sequence_path.read_bytes()
        sequence_sha = hashlib.sha256(sequence_bytes).hexdigest()
        sequence = json.loads(sequence_bytes)
        camera_source_sha = sequence.get("sourceSha256")
        sequence_indices = sequence.get("sourceIndices")
        if (
            not isinstance(sequence_indices, list)
            or any(type(index) is not int for index in sequence_indices)
            or sequence_indices != indices
        ):
            raise ValueError("Camera sequence source indices do not match camera source indices")
        timestamps = sequence.get("timestamps")
        if (
            not isinstance(timestamps, list)
            or any(type(value) not in (float, int) for value in timestamps)
            or timestamps != [camera.get("time") for camera in cameras]
        ):
            raise ValueError("Camera sequence timestamps do not match camera timestamps")
        fps = sequence.get("fps", fps)
    if camera_source_sha != source_sha:
        raise ValueError("Camera source hash mismatch")

    images, provenance = decode_frames(clip, indices, size, fps)
    if provenance["sourceSha256"] != source_sha:
        raise ValueError("Decoded source hash does not match masks")
    if (
        provenance["sourceTimeBase"] != str(archive["source_time_base"])
        or [frame["sourceIndex"] for frame in provenance["frames"]] != indices
        or [frame["sourcePts"] for frame in provenance["frames"]] != archive["source_pts"].tolist()
    ):
        raise ValueError("Decoded source timing does not match masks")
    match_camera_selection(provenance, cameras, indices)
    return images, archive, provenance, masks_sha, sequence_path, sequence_sha


def resize_exclusion_overlap(mask, shape):
    """Exclude a destination cell if it overlaps any excluded source cell."""
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Exclusion resize requires a 2D boolean mask")
    if not all(mask.shape) or len(shape) != 2 or not all(n > 0 for n in shape):
        raise ValueError("Exclusion resize requires nonempty grids")
    if mask.shape == shape:
        return mask.copy()

    def bounds(source, target):
        # Integer edges avoid treating cells that merely touch as overlapping.
        edges = np.arange(target + 1, dtype=np.int64) * source
        return edges[:-1] // target, (edges[1:] + target - 1) // target

    y0, y1 = bounds(mask.shape[0], shape[0])
    x0, x1 = bounds(mask.shape[1], shape[1])
    sums = np.pad(mask, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (
        sums[np.ix_(y1, x1)] - sums[np.ix_(y0, x1)] - sums[np.ix_(y1, x0)] + sums[np.ix_(y0, x0)]
    ) > 0


def anchor_exclusions(anchors, anchor_samples, masks, archive):
    with np.load(anchors, allow_pickle=False) as data:
        grid = data["valid"].shape
    if len(grid) != 3 or grid[0] != len(anchor_samples) or not all(grid):
        raise ValueError("Anchor sample count or pixel grid does not match the anchor archive")
    exclusions = {}
    for sample in anchor_samples:
        mask = resize_exclusion_overlap(masks[sample], grid[1:])
        if archive is not None:
            full = np.unpackbits(archive["masks"][sample], axis=1, count=int(archive["shape"][2]))
            mask |= resize_exclusion_overlap(full.astype(bool), grid[1:])
        exclusions[sample] = mask
    return exclusions


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default=ROOT / "public/marble-bedroom-clean.spz", type=Path)
    ap.add_argument("--cameras", default=ROOT / ".context/own/bedroom/pi3x/cameras.json", type=Path)
    ap.add_argument("--anchors", default=ROOT / ".context/own/bedroom/pi3x/anchors.npz", type=Path)
    ap.add_argument("--anchor-samples", default="0,15,31,46,62,77,93,108")
    ap.add_argument("--clip", default=ROOT / "public/clips/bedroom.mp4", type=Path)
    ap.add_argument(
        "--scale0", type=float, required=True, help="native -> raw scale (fit with --diag)"
    )
    ap.add_argument("--origin-frame", type=int, default=0, help="index into cameras.json")
    ap.add_argument("--size", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--depth-res", type=int, nargs=2, default=[480, 270])
    ap.add_argument(
        "--mask-dilate", type=int, default=16, help="dilation of the person mask in full-res pixels"
    )
    ap.add_argument(
        "--masks-npz",
        type=Path,
        default=None,
        help="exact-source packed exclusion masks on the camera sample grid instead of SegFormer",
    )
    ap.add_argument(
        "--moved-mask",
        action="store_true",
        help="ignore everything that MOVED, not just the person (cast shadow, carried object, passer-by). Off by default.",
    )
    ap.add_argument("--out", type=Path, required=True, help="new export directory")
    a = ap.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():
        raise FileExistsError(f"Export directory already exists: {a.out}")
    if not math.isfinite(a.scale0) or a.scale0 <= 0:
        raise ValueError("World scale must be finite and positive")
    camera_bytes = a.cameras.read_bytes()
    camera_doc = json.loads(camera_bytes)
    cams = camera_doc["cameras"]
    camera_sha = hashlib.sha256(camera_bytes).hexdigest()
    archive = masks_sha = sequence_path = sequence_sha = None
    W, H = a.size
    if a.masks_npz:
        imgs, archive, provenance, masks_sha, sequence_path, sequence_sha = supplied_mask_inputs(
            a.clip, a.masks_npz, a.cameras, camera_doc, a.size
        )
        src_idx = archive["source_indices"].tolist()
    else:
        src_idx = [int(camera["sourceIndex"]) for camera in cams]
        indices = sorted(set(src_idx))
        imgs, provenance = decode_frames(a.clip, indices, a.size)
        if indices != src_idx:
            positions = {index: i for i, index in enumerate(indices)}
            imgs = imgs[[positions[index] for index in src_idx]]
    source_sha = provenance["sourceSha256"]

    c2w = np.array([c["camera_to_world"] for c in cams], np.float64)
    Ks = np.array([c["source_intrinsics"] for c in cams], np.float64)
    SW, SH_ = cams[0]["source_image_size"]
    if not 0 <= a.origin_frame < len(cams):
        raise ValueError("Origin frame must index an exported camera")
    if Ks.shape != (len(cams), 3, 3) or SW <= 0 or SH_ <= 0:
        raise ValueError("Expected one 3x3 intrinsic matrix per camera and positive source size")
    export_Ks = Ks.copy()
    export_Ks[:, 0] *= W / SW
    export_Ks[:, 1] *= H / SH_
    export_Ks = export_Ks.astype(np.float32)
    if not np.isfinite(export_Ks).all() or np.any(export_Ks[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("Exported camera intrinsics must be finite with positive focal lengths")
    anchor_samples = [int(sample) for sample in a.anchor_samples.split(",")]
    if anchor_samples != sorted(set(anchor_samples)) or any(
        not 0 <= sample < len(cams) for sample in anchor_samples
    ):
        raise ValueError("Anchor samples must be unique, ordered camera samples")
    sibling = a.cameras.with_name("sequence.json")
    if sibling.is_file():
        sequence_bytes = sibling.read_bytes()
        sibling_sha = hashlib.sha256(sequence_bytes).hexdigest()
        if sequence_sha is not None and sibling_sha != sequence_sha:
            raise ValueError("Camera sequence changed during validation")
        sequence = json.loads(sequence_bytes)
        if "anchors" in sequence:
            recorded = sequence["anchors"]
            indices = sequence.get("sourceIndices")
            if (
                not isinstance(recorded, list)
                or any(type(sample) is not int for sample in recorded)
                or recorded != anchor_samples
                or sequence.get("sourceSha256") != source_sha
                or not isinstance(indices, list)
                or any(type(index) is not int for index in indices)
                or indices != src_idx
            ):
                raise ValueError("Anchor samples or source binding do not match camera sequence")
            sequence_path, sequence_sha = sibling, sibling_sha
    input_hashes = {
        a.clip: source_sha,
        a.cameras: camera_sha,
        a.anchors: sha256(a.anchors),
        a.world: sha256(a.world),
    }
    if a.masks_npz:
        input_hashes[a.masks_npz] = masks_sha
    if sequence_path:
        input_hashes[sequence_path] = sequence_sha
    alignment_path = framealign_path(a.cameras)
    alignment_sha = sha256(alignment_path) if alignment_path.exists() else None
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Export input changed during preflight: {path}")

    from PIL import Image
    from scipy import ndimage

    if archive is not None:
        mask_method = "supplied"
        full = np.unpackbits(archive["masks"], axis=2, count=SW)
        masks = np.stack([resize_exclusion_overlap(m.astype(bool), (H, W)) for m in full])
    elif a.moved_mask:
        mask_method = "moved"
        # the loss ignores what MOVED, not only the person: a cast shadow or a carried object is
        # in the frames but not in the static world, so the trainer is graded against pixels the
        # model can never reproduce and paints them onto the geometry to close the gap
        sys.path.insert(0, str(ROOT / "worker"))
        from wander_worker.masks import moved_content_masks

        res = moved_content_masks(imgs, dilate_px=max(1, a.mask_dilate * W // SW), batch_size=8)
        masks = res["moved"]
        print(
            f"moved-content mask: person {100 * res['person'].mean():.1f}% -> moved "
            f"{100 * masks.mean():.1f}% of pixels, "
            f"{sum(1 for st in res['stats'] if st['rejected'])}/{len(imgs)} frames fell back to person only",
            flush=True,
        )
    else:
        mask_method = "person"
        sys.path.insert(0, str(ROOT / "worker"))
        from wander_worker.masks import people_masks

        masks = np.concatenate(
            [
                people_masks(imgs[i : i + 8], dilate_px=max(1, a.mask_dilate * W // SW))
                for i in range(0, len(imgs), 8)
            ]
        )
    if (
        not isinstance(masks, np.ndarray)
        or masks.dtype != np.bool_
        or masks.shape != (len(cams), H, W)
    ):
        raise ValueError("Expected boolean exclusion masks on the training frame grid")
    masks = np.stack([ndimage.binary_dilation(mask, iterations=2) for mask in masks])
    exclusions = anchor_exclusions(a.anchors, anchor_samples, masks, archive)
    anchor_counts = []
    # Use the same final exclusion as training, plus the original supplied mask
    # projected directly onto the anchor grid to preserve source-cell coverage.
    scene = static_cloud_from_anchors(
        a.anchors,
        a.cameras,
        anchor_samples,
        exclusion_masks=exclusions,
        exclusion_report=anchor_counts,
    )
    for record in anchor_counts:
        sample = record["sample"]
        mask = exclusions[sample]
        record.update(
            sourceIndex=src_idx[sample],
            excludedAnchorPixels=int(mask.sum()),
            anchorMaskSha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            finalTrainingMaskSha256=hashlib.sha256(masks[sample].tobytes()).hexdigest(),
        )

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "frames").mkdir()
    (a.out / "masks").mkdir()
    print(
        f"{len(cams)} cameras, K spread fx {Ks[:, 0, 0].std():.3f} cx {Ks[:, 0, 2].std():.3f}",
        flush=True,
    )

    spz = read_spz(a.world)
    # the packagers' origin: camera 0 at the origin, gravity on +y when framealign.json exists
    origin, fa = levelled_origin(a.cameras, a.origin_frame)
    print(f"frame: {describe_frame(fa)}", flush=True)
    flip = np.diag([1.0, -1.0, -1.0])
    R = origin[:3, :3] @ flip
    A = R / a.scale0
    b = origin[:3, 3]
    means = (spz["xyz"] @ A.T + b).astype(np.float32)
    q_r = rotmat_to_quat_xyzw(R)[None].repeat(spz["n"], 0)
    quat_xyzw = quat_mul(q_r, spz["quat"])
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], 1).astype(np.float32)
    log_scales = (spz["log_scale"] - np.log(a.scale0)).astype(np.float32)
    al = np.clip(spz["alpha"], 1e-4, 1 - 1e-4)
    logit_op = np.log(al / (1 - al)).astype(np.float32)
    np.savez(
        a.out / "splats.npz",
        means=means,
        quats=quat_wxyz,
        log_scales=log_scales,
        logit_opacities=logit_op,
        f_dc=spz["f_dc"].astype(np.float32),
        scale0=a.scale0,
        origin_c2w=origin,
        R_raw_to_native=R,
        fb=spz["fb"],
    )
    print(
        f"splats: {spz['n']} -> native bbox {means.min(0).round(2)} .. {means.max(0).round(2)}, scale0 {a.scale0:.4f}",
        flush=True,
    )

    K = export_Ks[a.origin_frame]
    gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])
    viewmats = np.linalg.inv(c2w @ gl2cv)
    np.savez(
        a.out / "cameras.npz",
        viewmats=viewmats.astype(np.float32),
        K=K,
        Ks=export_Ks,
        c2w_gl=c2w.astype(np.float32),
        width=W,
        height=H,
        source_index=np.array(src_idx, np.int32),
    )
    print(
        f"cameras: K fx {K[0, 0]:.2f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}; positions bbox "
        f"{c2w[:, :3, 3].min(0).round(2)} .. {c2w[:, :3, 3].max(0).round(2)}",
        flush=True,
    )

    # depth: one z-buffered layer per camera from the Pi3X anchor cloud, native units
    np.save(a.out / "static-scene.npy", scene)
    DW, DH = a.depth_res
    depth = np.zeros((len(cams), DH, DW), np.float32)
    sup = np.zeros((len(cams), DH, DW), bool)
    for i in range(len(cams)):
        depth[i], sup[i] = zbuffer_depth(scene, c2w[i], Ks[i], (SW, SH_), (DW, DH))
    np.savez_compressed(
        a.out / "depth.npz",
        depth=depth,
        supported=sup,
        depth_index=np.arange(len(cams), dtype=np.int32),
    )
    print(
        f"depth: {depth.shape} layers, supported {100 * sup.mean():.0f}%, median depth {np.median(depth[sup]):.2f} native",
        flush=True,
    )

    # floor from the anchor cloud -> metres per native unit (camera height assumption, see status doc)
    floor = float(np.percentile(scene[:, 1], 1.0))
    print(
        f"anchor-cloud floor y {floor:.2f}, camera mean y {c2w[:, 1, 3].mean():.2f} native "
        f"(=> 1 native unit = {1.4 / (c2w[:, 1, 3].mean() - floor):.3f} m if the phone was 1.4 m up)",
        flush=True,
    )

    for i in range(len(cams)):
        Image.fromarray(imgs[i]).save(a.out / "frames" / f"{i:04d}.jpg", quality=95)
        Image.fromarray((masks[i] * 255).astype(np.uint8)).save(a.out / "masks" / f"{i:04d}.png")
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Export input changed during export: {path}")
    if (sha256(alignment_path) if alignment_path.exists() else None) != alignment_sha:
        raise ValueError(f"Export input changed during export: {alignment_path}")
    write_report(
        a.out / "export-receipt.json",
        {
            "schema": "wander.finetune-export/1",
            "sourceSha256": source_sha,
            "sourceIndices": src_idx,
            "sourceProvenance": provenance,
            "camerasSha256": camera_sha,
            "cameraSequenceSha256": sequence_sha,
            "masksSha256": masks_sha,
            "framealignSha256": alignment_sha,
            "maskMethod": mask_method,
            "maskDilationFullRes": None if archive is not None else a.mask_dilate,
            "maskDilationExportPixels": 2,
            "anchorsSha256": input_hashes[a.anchors],
            "anchorSamples": anchor_samples,
            "depthExclusions": {
                "method": "source-pixel-cell-overlap",
                "includesFinalTrainingMasks": True,
                "includesOriginalSuppliedMasks": archive is not None,
                "anchorMaskShape": list(exclusions[anchor_samples[0]].shape),
                "maskHashEncoding": "row-major boolean bytes (0 or 1)",
                "anchors": anchor_counts,
            },
            "worldSha256": input_hashes[a.world],
            "size": a.size,
            "depthRes": a.depth_res,
            "scale0": a.scale0,
            "originFrame": a.origin_frame,
        },
    )
    print(
        f"wrote {len(cams)} frames/masks to {a.out}; mean person coverage {100 * masks.mean():.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
