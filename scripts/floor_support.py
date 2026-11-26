"""Visible world samples immediately below complete source-person masks.

These samples locate a candidate supporting surface without using the avatar's uncertain
3D depth. They are not semantic floor labels or independent proof of physical geometry.
"""

import json
import re
from pathlib import Path

import numpy as np


def single_person_floor_masks(path, clip, world_dir):
    """Read undilated source masks only when their single-person identity is established.

    A one-person package can still reference an archive containing other tracked people.
    Never infer identity from that package's size or manufacture a filtered mask archive.
    Exact single-person archives retain the shared loader's source/provenance validation.
    """
    from world_ruler import person_masks_for

    path, world_dir = Path(path), Path(world_dir)
    with np.load(path, allow_pickle=False) as archive:
        if "packedMasks" not in archive.files:
            tracks = set()
            for key in archive.files:
                if key == "shape":
                    continue
                match = re.fullmatch(r"t(\d+)_s(\d+)", key)
                if match is None:
                    raise ValueError(
                        "Source floor support requires exact single-person or identified tracked masks"
                    )
                tracks.add(int(match[1]))
            if len(tracks) != 1:
                raise ValueError(
                    f"Source floor support requires exactly one mask track; found {sorted(tracks)}"
                )
            manifest = json.loads((world_dir / "people.json").read_text())
            people = manifest.get("people") if isinstance(manifest, dict) else None
            if (
                not isinstance(people, list)
                or len(people) != 1
                or not isinstance(people[0], dict)
                or type(people[0].get("track")) is not int
                or people[0]["track"] < 0
            ):
                raise ValueError("Source floor support requires one declared packaged person track")
            if tracks != {people[0]["track"]}:
                raise ValueError(
                    f"Source floor mask track {next(iter(tracks))} does not match "
                    f"packaged person track {people[0]['track']}"
                )
    return person_masks_for(clip, masks=path, world_dir=world_dir, dilate=False)


def source_floor_points(points, cameras, scale, masks, *, max_frames=12, width=384):
    """Return unique frontmost points in mask-derived below-feet patches and their receipt.

    Camera translations are native units; points and the output use viewer world units.
    Unknown, empty or edge-cropped masks contribute nothing. Occlusion is established before
    selecting the patch, so a hidden floor cannot become visible by filtering foreground points.
    """
    import cv2

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Source floor support requires finite (N, 3) world points")
    if not np.isfinite(scale) or scale <= 0 or max_frames < 1 or width < 1:
        raise ValueError("Source floor support requires positive scale and sampling limits")
    if not cameras:
        raise ValueError("Source floor support requires cameras")
    selected = np.unique(
        np.linspace(0, len(cameras) - 1, min(max_frames, len(cameras))).round().astype(int)
    )
    gathered, receipt = [], []
    for ci in selected:
        camera = cameras[int(ci)]
        source = int(camera["sourceIndex"])
        mask = masks.get(source)
        if mask is None:
            continue
        mask = np.asarray(mask)
        if mask.ndim != 2 or not np.isfinite(mask).all() or not mask.any():
            continue
        mask = mask.astype(bool)
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        mh, mw = mask.shape
        yy, xx = np.where(mask)
        body = float(yy.max() - yy.min() + 1)
        center = (xx.min() + xx.max()) / 2
        # A patch below the observed feet, scaled to the observed body instead of world units.
        box = [
            center - 0.75 * body,
            yy.max() + 0.02 * body,
            center + 0.75 * body,
            yy.max() + 0.30 * body,
        ]
        sw, sh = camera["source_image_size"]
        gw, gh = min(width, int(sw)), max(1, round(sh * min(width, int(sw)) / sw))
        x0, x1 = np.clip(np.array(box)[[0, 2]] * gw / mw, 0, gw)
        y0, y1 = np.clip(np.array(box)[[1, 3]] * gh / mh, 0, gh)
        matrix = np.asarray(camera["camera_to_world"], dtype=float)
        intrinsic = np.asarray(camera["source_intrinsics"], dtype=float)
        pc = (points - matrix[:3, 3] * scale) @ matrix[:3, :3]
        z = -pc[:, 2]
        safe = np.where(z > 0, z, 1)
        u = (intrinsic[0, 0] * pc[:, 0] / safe + intrinsic[0, 2]) * gw / sw
        v = (-intrinsic[1, 1] * pc[:, 1] / safe + intrinsic[1, 2]) * gh / sh
        live = (z > 0) & (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
        ids = np.flatnonzero(live)
        pix = np.floor(v[ids]).astype(int) * gw + np.floor(u[ids]).astype(int)
        order = np.argsort(z[ids], kind="stable")
        _, first = np.unique(pix[order], return_index=True)
        ids = ids[order[first]]
        px, py = np.floor(u[ids]).astype(int), np.floor(v[ids]).astype(int)
        clear = cv2.resize(mask.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) == 0
        keep = (px >= x0) & (px < x1) & (py >= y0) & (py < y1) & clear[py, px]
        ids = ids[keep]
        gathered.extend(ids.tolist())
        receipt.append(
            dict(sourceIndex=source, patchMaskPixels=box, maskSize=[mw, mh], visiblePoints=len(ids))
        )
    ids = np.unique(np.asarray(gathered, dtype=int))
    return points[ids], dict(
        method="visible world below source feet/1",
        uniquePoints=len(ids),
        frames=receipt,
        inferredSurface=True,
    )
