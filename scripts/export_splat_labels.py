#!/usr/bin/env python
"""Per-splat semantic class for a Marble world, lifted from 2D segmentation through the known cameras.

The world ships as one baked cloud with no objects in it. This pass gives every splat an ADE20K
class (SegFormer-b0, the same model worker/wander_worker/masks.py uses for the person mask) by
voting: every source frame the run sampled (public/worlds/<clip>-4d/cameras.json, one camera per
sampled frame at the run fps) is segmented, every splat is projected into that camera exactly as
scripts/export_observation_confidence.py does (viewer frame = Marble spz after Rx(pi), cameras placed
by placement.json's registrationScale, z-buffer from the cloud itself so only the front surface
votes), and the class under the pixel it lands on is one vote. Pixels the segmenter calls an
occluder class (default: person) vote for nobody, since the frame saw the walker there, not the wall.

Outputs, next to the .spz (public/*.bin is gitignored):
    <world>-labels.bin   uint8 per splat, ADE20K class id, 255 = fewer than --min-votes votes
    <world>-labels.json  class id -> name, per-class splat counts and share, observed fraction,
                         vote purity, the parameters used, and the cluster report when asked

    uv run --locked --group inference python scripts/export_splat_labels.py elevator lobby
    uv run --locked --group inference python scripts/export_splat_labels.py elevator --cluster-class door --render-dir share/labels
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_observation_confidence import read_spz_xyz  # noqa: E402

UNOBSERVED = 255
N_CLASSES = 150


def class_names():
    from transformers import AutoConfig
    from wander_worker.masks import MODEL_ID

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    return {int(k): v.strip() for k, v in cfg.id2label.items()}


def video_frames(clip: Path, w: int, h: int):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(clip),
        "-vf",
        f"scale={w}:{h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4
    )
    nbytes = w * h * 3
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def clip_frame_reader(clip: Path, w: int, h: int):
    """Random access by source index over a sequential decode (frames are requested in order)."""
    gen = video_frames(clip, w, h)
    state = {"i": -1, "img": None}

    def get(i):
        while state["i"] < i:
            state["img"] = next(gen)
            state["i"] += 1
        return state["img"]

    return get


def project(P, cam, scale, ds, near):
    """Splats -> (u, v, z, in_frustum) on the z-buffer grid. Same maths as observe()."""
    M = torch.tensor(cam["camera_to_world"], dtype=torch.float32, device=P.device)
    R, t = M[:3, :3], M[:3, 3] * scale
    K = cam["source_intrinsics"]
    W, H = cam["source_image_size"]
    gw, gh = max(1, W // ds), max(1, H // ds)
    fx, fy = K[0][0] / ds, K[1][1] / ds
    cx, cy = K[0][2] / ds, K[1][2] / ds
    pc = (P - t) @ R
    z = -pc[:, 2]
    ok = z > near
    zs = torch.where(ok, z, torch.ones_like(z))
    u = (fx * pc[:, 0] / zs + cx).long()
    v = (fy * (-pc[:, 1]) / zs + cy).long()
    ok &= (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
    return u, v, z, ok, gw, gh


def front_surface(u, v, z, ok, gw, gh, tol):
    """Indices of splats that are the front surface of the cloud at their pixel, and their flat pixel ids."""
    sel = torch.nonzero(ok, as_tuple=True)[0]
    idx = v[sel] * gw + u[sel]
    zi = z[sel]
    zbuf = torch.full((gw * gh,), float("inf"), device=z.device)
    zbuf.scatter_reduce_(0, idx, zi, reduce="amin", include_self=True)
    front = zi <= zbuf[idx] * (1.0 + tol) + 1e-3
    return sel[front], idx[front]


def dilate_mask(m: torch.Tensor, px: int) -> torch.Tensor:
    if px <= 0:
        return m
    k = 2 * px + 1
    return torch.nn.functional.max_pool2d(m[None, None].float(), k, stride=1, padding=px)[0, 0] > 0


def vote(xyz_raw, cams, scale, clip, a, seg_h, device):
    from wander_worker.masks import segment_labels

    P = torch.as_tensor(xyz_raw, dtype=torch.float32, device=device)
    P = torch.stack([P[:, 0], -P[:, 1], -P[:, 2]], 1)  # the viewer's Rx(pi)
    n = P.shape[0]
    hist = np.zeros((n, N_CLASSES), dtype=np.uint8)  # < 256 frames per run, one vote per frame
    pixel_counts = np.zeros(N_CLASSES, dtype=np.int64)
    occluded_votes = 0
    read = clip_frame_reader(clip, a.seg_width, seg_h)
    occl = torch.tensor(sorted(a.occluders), device=device)
    t0 = time.time()
    for b0 in range(0, len(cams), a.batch):
        batch = cams[b0 : b0 + a.batch]
        imgs = np.stack([read(c["sourceIndex"]) for c in batch])
        labs = segment_labels(imgs, batch_size=len(batch))  # (b, seg_h, seg_w) uint8
        for c, lab in zip(batch, labs):
            pixel_counts += np.bincount(lab.ravel(), minlength=N_CLASSES)[:N_CLASSES]
            lab_t = torch.as_tensor(lab, device=device)
            occ = dilate_mask(torch.isin(lab_t, occl), a.occluder_dilate)
            u, v, z, ok, gw, gh = project(P, c, scale, a.ds, a.near)
            if not bool(ok.any()):
                continue
            sel, idx = front_surface(u, v, z, ok, gw, gh, a.tol)
            us = (u[sel] * a.seg_width) // gw
            vs = (v[sel] * seg_h) // gh
            cls = lab_t[vs, us]
            keep = ~occ[vs, us]
            occluded_votes += int((~keep).sum())
            sel_np = sel[keep].cpu().numpy()
            hist[sel_np, cls[keep].cpu().numpy().astype(np.int64)] += 1
        print(
            f"  frames {min(b0 + a.batch, len(cams))}/{len(cams)}  {time.time() - t0:.0f}s",
            flush=True,
        )
    return hist, pixel_counts, occluded_votes


def cluster_class(P, sel, radius_units):
    """Connected components of the selected splats on a voxel grid of cell = radius (26-connectivity)."""
    from scipy import ndimage

    Q = P[sel]
    if len(Q) == 0:
        return dict(splats=0, clusters=[])
    lo = Q.min(0)
    ijk = np.floor((Q - lo) / radius_units).astype(np.int64)
    shape = tuple(int(x) + 1 for x in ijk.max(0))
    occ = np.zeros(shape, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, nlab = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=int))
    comp = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    sizes = np.bincount(comp, minlength=nlab + 1)[1:]
    order = np.argsort(-sizes)
    clusters = []
    for k in order[:10]:
        m = comp == k + 1
        clusters.append(
            dict(
                splats=int(sizes[k]),
                fraction=float(sizes[k] / len(Q)),
                bboxMin=Q[m].min(0).tolist(),
                bboxMax=Q[m].max(0).tolist(),
            )
        )
    return dict(
        splats=int(len(Q)),
        voxelUnits=float(radius_units),
        nClusters=int(nlab),
        bboxMin=Q.min(0).tolist(),
        bboxMax=Q.max(0).tolist(),
        clusters=clusters,
    )


# --- renders ------------------------------------------------------------------------------
# Eight categorical hues in fixed order (dataviz reference palette); the top eight classes by splat
# count take them, everything else folds into "other". Neutral greys for other / unobserved.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER, UNSEEN, SURFACE = "#8a8a86", "#d9d8d4", "#fcfcfb"


def colour_table(top_ids):
    tab = np.full((256, 3), 0, dtype=np.float32)
    tab[:] = np.array([int(OTHER[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    for k, cid in enumerate(top_ids):
        tab[cid] = np.array([int(PALETTE[k][i : i + 2], 16) for i in (1, 3, 5)]) / 255
    tab[UNOBSERVED] = np.array([int(UNSEEN[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    return tab


def add_legend(ax, top_ids, names, counts, extra):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            color=PALETTE[k],
            label=f"{names[cid]} ({counts[cid]:,})",
        )
        for k, cid in enumerate(top_ids)
    ]
    handles += [Line2D([], [], marker="s", ls="", ms=9, color=c, label=l) for l, c in extra]
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8
    )


def render_camera_view(
    out, xyz_w, labels, cams, scale, clip, a, seg_h, names, counts, top_ids, device
):
    """Three panels: source frame, its 2D segmentation, the labelled cloud's front surface from the same camera."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wander_worker.masks import segment_labels

    cam = cams[0]
    img = clip_frame_reader(clip, a.seg_width, seg_h)(cam["sourceIndex"])
    lab2d = segment_labels(img[None], batch_size=1)[0]
    tab = colour_table(top_ids)
    P = torch.as_tensor(xyz_w, dtype=torch.float32, device=device)
    u, v, z, ok, gw, gh = project(P, cam, scale, a.ds, a.near)
    sel, idx = front_surface(u, v, z, ok, gw, gh, a.tol)
    sel = sel.cpu().numpy()
    canvas = (
        np.ones((gh, gw, 3), dtype=np.float32)
        * np.array([int(SURFACE[i : i + 2], 16) for i in (1, 3, 5)])
        / 255
    )
    zz = z[sel].cpu().numpy()
    order = np.argsort(-zz)  # far first, near overwrites
    uu, vv = u[sel].cpu().numpy()[order], v[sel].cpu().numpy()[order]
    canvas[vv, uu] = tab[labels[sel][order]]

    fig, axs = plt.subplots(1, 3, figsize=(16, 3.6), facecolor=SURFACE)
    axs[0].imshow(img)
    axs[0].set_title(f"source frame {cam['sourceIndex']}", fontsize=9)
    axs[1].imshow(tab[lab2d])
    axs[1].set_title("SegFormer-b0 ADE20K, 2D", fontsize=9)
    axs[2].imshow(canvas, interpolation="nearest")
    axs[2].set_title(f"splat labels, front surface from camera 0 ({len(sel):,} splats)", fontsize=9)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    add_legend(
        axs[2], top_ids, names, counts, [("other classes", OTHER), ("unobserved (255)", UNSEEN)]
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def render_top_down(
    out, xyz_w, labels, cams, scale, a, names, counts, top_ids, ceiling_id, metres_per_unit
):
    """Plan view of the labelled cloud, viewer world y up so x/z is the floor plane. Ceiling-class
    splats are dropped (they would cover the room); everything else is drawn low to high."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    keep = labels != ceiling_id
    idx = np.flatnonzero(keep)
    if len(idx) > a.render_count:
        idx = rng.choice(idx, a.render_count, replace=False)
    idx = idx[np.argsort(xyz_w[idx, 1])]
    tab = colour_table(top_ids)
    Q = xyz_w[idx]
    lo, hi = np.percentile(Q[:, [0, 2]], [0.5, 99.5], axis=0)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    ax.scatter(
        Q[:, 0], Q[:, 2], c=tab[labels[idx]], s=0.4, marker=".", linewidths=0, rasterized=True
    )
    cp = np.array([c["camera_to_world"] for c in cams])[:, :3, 3] * scale
    ax.plot(cp[:, 0], cp[:, 2], color="#0b0b0b", lw=1.2, label="camera path")
    ax.plot(cp[0, 0], cp[0, 2], "o", color="#0b0b0b", ms=5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(f"x (world units, {metres_per_unit:.2f} m/unit)", fontsize=8)
    ax.set_ylabel("z (world units)", fontsize=8)
    ax.tick_params(labelsize=7, colors="#52514e")
    for s in ax.spines.values():
        s.set_color("#d9d8d4")
    ax.set_title(
        f"top-down, {len(idx):,} of {len(xyz_w):,} splats, ceiling class hidden; dot = camera 0",
        fontsize=9,
    )
    add_legend(
        ax,
        top_ids,
        names,
        counts,
        [("other classes", OTHER), ("unobserved (255)", UNSEEN), ("camera path", "#0b0b0b")],
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "clips",
        nargs="+",
        help="clip names; world, cameras, scale and clip resolve from public/worlds/<clip>-4d/",
    )
    ap.add_argument("--world-dir", help="override public/worlds/<clip>-4d (single clip only)")
    ap.add_argument("--spz", help='override the world (default: placement.json "world")')
    ap.add_argument("--scale", type=float, help="override placement.json registrationScale")
    ap.add_argument(
        "--clip",
        help="override the source clip (default: cameras.json sourceClip, else public/clips/<clip>.mp4)",
    )
    ap.add_argument(
        "--ds", type=int, default=4, help="z-buffer downsample from the source resolution"
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.03,
        help='relative depth tolerance for "is the front surface"',
    )
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument(
        "--seg-width",
        type=int,
        default=960,
        help="frame width fed to the segmenter (it resizes to 512 internally)",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--occluders",
        type=int,
        nargs="*",
        default=[12],
        help="ADE20K ids whose pixels vote for nobody (default person)",
    )
    ap.add_argument("--occluder-dilate", type=int, default=8, help="px at --seg-width")
    ap.add_argument(
        "--min-votes", type=int, default=2, help="splats with fewer votes are written as 255"
    )
    ap.add_argument("--limit-frames", type=int, help="use only the first N cameras (smoke test)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--suffix", default="-labels", help="sidecar stem suffix next to the .spz")
    ap.add_argument(
        "--cluster-class", help="ADE20K class name to cluster spatially and report (e.g. door)"
    )
    ap.add_argument(
        "--cluster-radius-m", type=float, default=0.10, help="voxel size for connectivity, metres"
    )
    ap.add_argument(
        "--render-dir",
        type=Path,
        help="write <clip>-labels-cam0.png and <clip>-labels-topdown.png here",
    )
    ap.add_argument(
        "--render-count", type=int, default=400000, help="max splats drawn in the top-down render"
    )
    a = ap.parse_args()
    if a.world_dir and len(a.clips) != 1:
        ap.error("--world-dir applies to a single clip")

    names = class_names()
    name_to_id = {v: k for k, v in names.items()}
    ceiling_id = name_to_id["ceiling"]
    if a.cluster_class and a.cluster_class not in name_to_id:
        ap.error(f"unknown class {a.cluster_class!r}")

    for clip_name in a.clips:
        wdir = Path(a.world_dir) if a.world_dir else ROOT / "public" / "worlds" / f"{clip_name}-4d"
        place = json.loads((wdir / "placement.json").read_text())
        camj = json.loads((wdir / "cameras.json").read_text())
        cams = sorted(camj["cameras"], key=lambda c: c["sourceIndex"])
        if a.limit_frames:
            cams = cams[: a.limit_frames]
        spz = Path(a.spz) if a.spz else ROOT / place["world"]
        scale = a.scale if a.scale is not None else float(place["registrationScale"])
        clip = (
            Path(a.clip)
            if a.clip
            else Path(camj.get("sourceClip") or ROOT / "public" / "clips" / f"{clip_name}.mp4")
        )
        if not clip.exists():
            clip = ROOT / "public" / "clips" / f"{clip_name}.mp4"
        W, H = cams[0]["source_image_size"]
        seg_h = int(round(H * a.seg_width / W / 2)) * 2
        out_bin = spz.with_name(spz.stem + a.suffix + ".bin")
        out_json = spz.with_name(spz.stem + a.suffix + ".json")
        print(
            f"{clip_name}: {spz.name} scale {scale} {len(cams)} frames from {clip.name} on {a.device}",
            flush=True,
        )

        xyz = read_spz_xyz(spz)
        hist, pixel_counts, occluded = vote(xyz, cams, scale, clip, a, seg_h, a.device)
        total = hist.sum(1, dtype=np.int32)
        top = hist.argmax(1).astype(np.uint8)
        top_votes = hist[np.arange(len(hist)), top].astype(np.int32)
        labels = np.where(total >= a.min_votes, top, UNOBSERVED).astype(np.uint8)
        out_bin.write_bytes(labels.tobytes())

        n = len(xyz)
        labelled = labels != UNOBSERVED
        counts = np.bincount(labels[labelled], minlength=256)[:N_CLASSES]
        purity = top_votes[labelled] / np.maximum(total[labelled], 1)
        order = np.argsort(-counts)
        classes = [
            dict(
                id=int(c),
                name=names[int(c)],
                splats=int(counts[c]),
                shareOfLabelled=float(counts[c] / max(labelled.sum(), 1)),
                pixelShare=float(pixel_counts[c] / pixel_counts.sum()),
            )
            for c in order
            if counts[c] > 0
        ]
        never_fired = [names[c] for c in range(N_CLASSES) if pixel_counts[c] == 0]
        fired_no_splats = [
            dict(name=names[c], pixelShare=float(pixel_counts[c] / pixel_counts.sum()))
            for c in range(N_CLASSES)
            if pixel_counts[c] > 0 and counts[c] == 0
        ]
        report = dict(
            schema="wander.labels/1",
            clip=clip_name,
            world=str(spz.relative_to(ROOT)),
            cameras=str((wdir / "cameras.json").relative_to(ROOT)),
            sourceClip=str(clip),
            scale=scale,
            metresPerWorldUnit=place.get("metresPerWorldUnit"),
            segmenter="nvidia/segformer-b0-finetuned-ade-512-512",
            frames=len(cams),
            splats=n,
            params=dict(
                ds=a.ds,
                tol=a.tol,
                near=a.near,
                segWidth=a.seg_width,
                segHeight=seg_h,
                occluders=a.occluders,
                occluderDilatePx=a.occluder_dilate,
                minVotes=a.min_votes,
            ),
            votesThreshold=a.min_votes,
            observed=dict(
                count=int((total > 0).sum()),
                fraction=float((total > 0).mean()),
                note="at least one front-surface vote",
            ),
            labelled=dict(
                count=int(labelled.sum()),
                fraction=float(labelled.mean()),
                note=f">= {a.min_votes} votes; the rest are 255",
            ),
            unobserved=dict(count=int((~labelled).sum()), fraction=float((~labelled).mean())),
            votesPerLabelledSplat=dict(
                median=float(np.median(total[labelled])) if labelled.any() else 0,
                max=int(total.max()),
            ),
            occludedVotesDropped=int(occluded),
            purity=dict(
                note="top-class votes / all votes, over labelled splats",
                mean=float(purity.mean()) if labelled.any() else 0,
                below050=float((purity < 0.5).mean()) if labelled.any() else 0,
                below075=float((purity < 0.75).mean()) if labelled.any() else 0,
                exactly100=float((purity >= 1.0).mean()) if labelled.any() else 0,
            ),
            classNames={int(k): v for k, v in names.items()},
            classes=classes,
            classesNeverFired=never_fired,
            classesFiredButNoSplats=fired_no_splats,
        )
        xyz_w = np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], 1)
        if a.cluster_class:
            cid = name_to_id[a.cluster_class]
            mpu = float(place.get("metresPerWorldUnit") or 1.0)
            rep = cluster_class(xyz_w, labels == cid, a.cluster_radius_m / mpu)
            rep.update(classId=cid, className=a.cluster_class, radiusM=a.cluster_radius_m)
            report["cluster"] = rep
        out_json.write_text(json.dumps(report, indent=1))

        print(
            f"  observed {report['observed']['fraction'] * 100:.1f}%  labelled {report['labelled']['fraction'] * 100:.1f}%  "
            f"purity mean {report['purity']['mean']:.3f}  -> {out_bin.name}, {out_json.name}"
        )
        print("  | class | splats | share of labelled |")
        for c in classes[:20]:
            print(f"  | {c['name']} | {c['splats']:,} | {c['shareOfLabelled'] * 100:.1f}% |")
        if a.cluster_class:
            rep = report["cluster"]
            print(
                f"  {a.cluster_class}: {rep['splats']:,} splats, {rep.get('nClusters', 0)} components at {a.cluster_radius_m} m, "
                f"largest {[c['splats'] for c in rep.get('clusters', [])[:5]]}"
            )

        if a.render_dir:
            a.render_dir.mkdir(parents=True, exist_ok=True)
            top_ids = [c["id"] for c in classes[: len(PALETTE)]]
            render_camera_view(
                a.render_dir / f"{clip_name}-labels-cam0.png",
                xyz_w,
                labels,
                cams,
                scale,
                clip,
                a,
                seg_h,
                names,
                counts,
                top_ids,
                a.device,
            )
            render_top_down(
                a.render_dir / f"{clip_name}-labels-topdown.png",
                xyz_w,
                labels,
                cams,
                scale,
                a,
                names,
                counts,
                top_ids,
                ceiling_id,
                float(place.get("metresPerWorldUnit") or 1.0),
            )
            print(f"  renders -> {a.render_dir}")


if __name__ == "__main__":
    sys.exit(main())
