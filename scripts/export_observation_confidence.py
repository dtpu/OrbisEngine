#!/usr/bin/env python
"""Per-splat observation confidence for a Marble world: did any source frame ever SEE this splat?

Same question the fine-tune's observed mask answers (share/FINETUNE-STATUS.md, `--mode sweep`), asked
of a shipping world instead of a training run, and written as a sidecar the viewer loads next to the
.spz - the same shape as bake_video_colours.py's `-weights.bin`, uint8 instead of float32.

For each source camera (public/worlds/<dir>/cameras.json, Pi3X/SfM native, OpenGL camera_to_world,
camera 0 = the origin) placed into viewer world units by the preset's own `scale`, every splat is
projected; a splat counts as observed in that frame when it lands inside the image AND is within
--tol of the nearest splat depth along that pixel (a z-buffer built from the cloud itself, so
"observed" means "was the front surface", not "was somewhere down the ray behind a wall")
AND the pixel it lands on was not covered by the walker in that frame.

That last clause is the fix for the defect share/METHODS-REVIEW.md §2.8 names: with no person mask,
the splats in the walker's occlusion shadow are the front surface of the cloud on the pixels he
covered, so the single worst artifact in every world was being marked "observed" and never faded.
Pass --masks to use source-index-mapped tracking or tight person masks. Every selected camera must
have a mask, including an explicitly empty mask when no person was present. Without --masks the
legacy mode projects and dilates the packaged avatar; --no-person-mask disables that fallback.
The sidecar measures visibility under these inputs; it does not establish reconstruction quality.

    confidence = min(1, seen_frames / --saturate)     quantised to uint8

Splat -> world is the viewer's own mapping: raw Marble coords are OpenCV and the viewer applies
Rx(pi), so world = (x, -y, -z). Cameras are already in that frame (scripts/ab-world-cliff.mjs uses
the same `camera_to_world[i][3] * scale`).

    uv run --locked --group inference python scripts/export_observation_confidence.py                  # all five shipping worlds
    uv run --locked --group inference python scripts/export_observation_confidence.py lobby stairs2
    uv run --locked --group inference python scripts/export_observation_confidence.py --world public/x.spz --cameras public/worlds/y/cameras.json --scale 1.0 --out public/x-obs.bin
"""

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# clip -> (world spz, cameras.json dir, the preset's `scale` in fourd.html)
WORLDS = {
    "lobby": ("public/marble-lobby-clean.spz", "public/worlds/lobby-4d", 1.5300),
    "atrium": ("public/marble-atrium-clean.spz", "public/worlds/atrium-4dpp", 1.0221),
    "elevator": ("public/marble-elevator-clean.spz", "public/worlds/elevator-4d", 1.0800),
    "stairs2": ("public/marble-stairs2-clean.spz", "public/worlds/stairs2-4d", 0.9760),
    "tos31": ("public/marble-tos31-image.spz", "public/worlds/tos31-4d", 1.2834),
}


def read_spz_xyz(path: Path):
    """Positions only, from a v2 shDegree-0 spz (same layout bake_video_colours.py reads)."""
    raw = gzip.open(path).read()
    magic, ver, n, sh, fb, _flags, _r = struct.unpack("<IIIBBBB", raw[:16])
    assert magic == 0x5053474E and ver == 2 and sh == 0, (hex(magic), ver, sh)
    pos_b = np.frombuffer(raw, np.uint8, n * 9, 16).reshape(n, 3, 3)
    fixed = (
        pos_b[:, :, 0].astype(np.int32)
        | (pos_b[:, :, 1].astype(np.int32) << 8)
        | (pos_b[:, :, 2].astype(np.int32) << 16)
    )
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    return fixed.astype(np.float32) / float(1 << fb)


def person_pixels(world_dir: Path, cam, gw, gh, ds, dilate=6):
    """The walker's own footprint in one source frame, as a boolean on the z-buffer grid.

    The packaged avatar (public/worlds/<clip>-4d/person/frame_NNN.ply) is already registered to
    these cameras in native SfM units, which is how scripts/reproject_multiperson.py projects it.
    Its silhouette, generously dilated, is where the source frame had a person and therefore had no
    view of the world.
    """
    seqs = sorted(world_dir.glob("*/sequence.json"))
    if not seqs:
        return None
    m = np.zeros((gh, gw), dtype=bool)
    fi = cam["sourceIndex"]
    M = np.array(cam["camera_to_world"], dtype=np.float64)
    R = M[:3, :3]
    u_, _, vt = np.linalg.svd(R)
    R = u_ @ vt
    t = M[:3, 3]  # native units: the person PLYs are not scaled
    K = np.array(cam["source_intrinsics"], dtype=np.float64)
    for sj in seqs:
        seq = json.loads(sj.read_text())
        src = np.array(seq.get("sourceIndices") or [])
        if not len(src):
            continue
        k = int(np.argmin(np.abs(src - fi)))
        if abs(int(src[k]) - fi) > 3:
            continue
        ply = sj.parent / seq["frames"][k]
        if not ply.exists():
            continue
        from plyfile import PlyData

        v = PlyData.read(str(ply))["vertex"].data
        P = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
        op = 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
        P = P[op > 0.5]
        pc = (P - t) @ R
        z = -pc[:, 2]
        ok = z > 0.05
        u = (K[0, 0] / ds * pc[:, 0] / np.where(z > 0, z, 1) + K[0, 2] / ds).astype(np.int64)
        vv = (K[1, 1] / ds * (-pc[:, 1]) / np.where(z > 0, z, 1) + K[1, 2] / ds).astype(np.int64)
        ok &= (u >= 0) & (u < gw) & (vv >= 0) & (vv < gh)
        m[vv[ok], u[ok]] = True
    if not m.any():
        return None
    if dilate:
        k = np.ones((dilate, dilate), bool)
        pad = dilate // 2
        big = np.zeros((gh + 2 * pad, gw + 2 * pad), bool)
        big[pad : pad + gh, pad : pad + gw] = m
        out = np.zeros_like(m)
        for dy in range(dilate):
            for dx in range(dilate):
                out |= big[dy : dy + gh, dx : dx + gw]
        m = out
    return m


def load_source_masks(path: Path, world_dir: Path, cams):
    """Load explicit source masks without falling back to historical caches or avatars."""
    from world_ruler import person_masks_for

    if not path.is_file():
        raise ValueError(f"--masks is not a file: {path}")
    try:
        with np.load(path) as archive:
            if "masks" in archive.files:
                indices = archive["indices"]
                shape = tuple(int(v) for v in archive["shape"])
                if (
                    len(shape) != 3
                    or min(shape) < 1
                    or indices.dtype.kind not in "iu"
                    or indices.shape != (shape[0],)
                    or len(np.unique(indices)) != len(indices)
                    or archive["masks"].dtype != np.uint8
                    or archive["masks"].shape != (shape[0], shape[1], (shape[2] + 7) // 8)
                ):
                    raise ValueError("invalid tight-mask shape or source indices")
        masks = person_masks_for("", masks=path, world_dir=world_dir, dilate=False)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise ValueError(f"invalid --masks {path}: {error}") from error
    require_mask_coverage(masks, cams)
    return masks


def require_mask_coverage(masks, cams):
    if not isinstance(masks, dict):
        raise ValueError("source masks must map source frame indices to masks")
    for camera in cams:
        index = camera.get("sourceIndex")
        mask = masks.get(index)
        if mask is None:
            raise ValueError(f"source masks have no coverage for camera sourceIndex {index}")
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size:
            raise ValueError(f"invalid source mask for sourceIndex {index}")


def observe(
    xyz_raw, cams, scale, ds=4, tol=0.03, near=0.05, device="cpu", world_dir=None, person_masks=None
):
    """Frames-seen count per splat. xyz_raw = raw Marble coords; cams = cameras.json['cameras']."""
    if person_masks is not None:
        require_mask_coverage(person_masks, cams)
    P = torch.as_tensor(xyz_raw, dtype=torch.float32, device=device)
    P = torch.stack([P[:, 0], -P[:, 1], -P[:, 2]], 1)  # the viewer's Rx(pi)
    n = P.shape[0]
    seen = torch.zeros(n, dtype=torch.int32, device=device)
    infront = torch.zeros(n, dtype=torch.int32, device=device)

    for c in cams:
        M = torch.tensor(c["camera_to_world"], dtype=torch.float32, device=device)
        R, t = M[:3, :3], M[:3, 3] * scale
        K = c["source_intrinsics"]
        W, H = c["source_image_size"]
        gw, gh = max(1, W // ds), max(1, H // ds)
        fx, fy = K[0][0] / ds, K[1][1] / ds
        cx, cy = K[0][2] / ds, K[1][2] / ds

        pc = (P - t) @ R  # R^T (p - t), OpenGL camera coords
        z = -pc[:, 2]  # OpenGL looks -z
        ok = z > near
        u = (fx * pc[:, 0] / z + cx).long()
        v = (fy * (-pc[:, 1]) / z + cy).long()
        ok &= (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
        if not bool(ok.any()):
            continue
        idx = (v.clamp(0, gh - 1) * gw + u.clamp(0, gw - 1))[ok]
        zi = z[ok]
        zbuf = torch.full((gw * gh,), float("inf"), device=device)
        zbuf.scatter_reduce_(0, idx, zi, reduce="amin", include_self=True)
        front = zi <= zbuf[idx] * (1.0 + tol) + 1e-3

        if person_masks is not None:
            import cv2

            pm = cv2.resize(
                person_masks[c["sourceIndex"]].astype(np.uint8),
                (gw, gh),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            covered = torch.as_tensor(pm.reshape(-1), device=device)[idx]
            front &= ~covered
        elif world_dir is not None:
            pm = person_pixels(Path(world_dir), c, gw, gh, ds)
            if pm is not None:
                covered = torch.as_tensor(pm.reshape(-1), device=device)[idx]
                front &= ~covered  # the walker stood there: this frame saw him, not the world

        hit = torch.zeros(n, dtype=torch.bool, device=device)
        hit[torch.nonzero(ok, as_tuple=True)[0][front]] = True
        seen += hit.int()
        infront += ok.int()
    return seen.cpu().numpy(), infront.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "clips", nargs="*", default=[], help=f"any of {', '.join(WORLDS)} (default: all)"
    )
    ap.add_argument("--world", type=Path)
    ap.add_argument("--cameras", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--world-dir", type=Path, help="packaged world directory; default: camera parent"
    )
    ap.add_argument(
        "--masks", type=Path, help="tracked/tight person masks with source-frame indices"
    )
    ap.add_argument("--scale", type=float)
    ap.add_argument("--saturate", type=int, default=8, help="frames at which confidence reaches 1")
    ap.add_argument(
        "--ds", type=int, default=4, help="z-buffer downsample from the source resolution"
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.03,
        help='relative depth tolerance for "is the front surface"',
    )
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument(
        "--no-person-mask",
        action="store_true",
        help="reproduce the old numbers, which counted the walker's occlusion shadow as observed",
    )
    ap.add_argument(
        "--suffix",
        default="-obs.bin",
        help="sidecar name; use another to avoid overwriting a shipped one",
    )
    a = ap.parse_args()

    if a.masks and a.no_person_mask:
        ap.error("--masks cannot be combined with --no-person-mask")
    if a.ds < 1 or a.saturate < 1:
        ap.error("--ds and --saturate must be positive")
    for label, path in (("world", a.world), ("cameras", a.cameras), ("masks", a.masks)):
        if path is not None and not path.is_file():
            ap.error(f"--{label} is not a file: {path}")
    if a.world_dir is not None and not a.world_dir.is_dir():
        ap.error(f"--world-dir is not a directory: {a.world_dir}")
    if a.world:
        if a.cameras is None or a.scale is None or a.out is None:
            ap.error("--world requires --cameras, --scale and --out")
        if not np.isfinite(a.scale) or a.scale <= 0:
            ap.error("--scale must be finite and positive")
        jobs = [("custom", a.world, a.cameras, a.scale, a.out)]
    else:
        if a.cameras is not None or a.out is not None:
            ap.error("--cameras and --out require --world")
        names = a.clips or list(WORLDS)
        if (a.masks is not None or a.world_dir is not None) and len(names) != 1:
            ap.error("--masks and --world-dir require --world or exactly one historical clip")
        jobs = []
        for k in names:
            if k not in WORLDS:
                ap.error(f"unknown historical clip: {k}")
            w, d, s = WORLDS[k]
            place = ROOT / d / "placement.json"
            if place.exists():
                fitted = json.loads(place.read_text()).get("registrationScale")
                if fitted and abs(fitted / s - 1) > 0.02:
                    sys.exit(
                        f"{k}: this table says scale {s}, {place} says the solve fitted "
                        f"{fitted}. The cameras would be placed {abs(fitted / s - 1) * 100:.0f}% "
                        f"wrong and every observed fraction with them (atrium shipped this way "
                        f"at 1.0221 against a fitted 0.816). Pass --scale explicitly to say "
                        f"which you mean."
                    )
            jobs.append((k, ROOT / w, ROOT / d / "cameras.json", s, ROOT / (w[:-4] + a.suffix)))

    rows = []
    for name, world, camf, scale, out in jobs:
        cams = json.load(open(camf))["cameras"]
        if not cams:
            ap.error(f"no source cameras in {camf}")
        mask_root = a.world_dir or Path(camf).parent
        try:
            masks = load_source_masks(a.masks, mask_root, cams) if a.masks else None
        except ValueError as error:
            ap.error(str(error))
        xyz = read_spz_xyz(world)
        wdir = None if a.no_person_mask else mask_root
        seen, infront = observe(
            xyz,
            cams,
            scale,
            ds=a.ds,
            tol=a.tol,
            device=a.device,
            world_dir=wdir,
            person_masks=masks,
        )
        conf = np.minimum(1.0, seen / a.saturate)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(np.round(conf * 255).astype(np.uint8).tobytes())
        obs = float((seen > 0).mean())
        rows.append(
            (
                name,
                len(xyz),
                len(cams),
                obs,
                float((infront > 0).mean()),
                float(np.median(seen[seen > 0])) if obs else 0.0,
                out.name,
            )
        )
        print(
            f"{name:9s} {len(xyz):>9,} splats  {len(cams):>4} frames  "
            f"observed {obs * 100:5.1f}%  in-frustum {(infront > 0).mean() * 100:5.1f}%  "
            f"median frames/observed splat {rows[-1][5]:.0f}  -> {out.name}",
            flush=True,
        )

    print("\n| world | splats | frames | observed | in frustum | frames/observed splat (med) |")
    print("|---|---|---|---|---|---|")
    for n, ns, nc, o, f, m, _ in rows:
        print(f"| {n} | {ns:,} | {nc} | {o * 100:.1f}% | {f * 100:.1f}% | {m:.0f} |")


if __name__ == "__main__":
    sys.exit(main())
