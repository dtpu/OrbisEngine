#!/usr/bin/env python3
"""Every anchored check share/METHODS-REVIEW.md §4 asks for, per clip, as one gate.

  uv run --locked --group inference python scripts/anchor_checks.py --clip gym --json-out share/anchors-gym.json

Each check here is anchored to something outside the estimate it tests. They FAIL the run; none of
them warns. What they replace, in order:

  ruler       (§4A) the metre. Not `1.70 / stature` -- scripts/world_ruler.py measures it from a
              metric-depth model, a VLM-named standard object, or a falling object, and the avatar's
              1.70 m becomes a 1.5-1.9 m band that can be outvoted.
  silhouette  (§4D) IoU >= 0.75 and area ratio 0.85-1.15 against Mask R-CNN, on >= 8 frames. This is
              what "in-frame 1.00" could never be: a 1.83x avatar centred on the man scores 1.00
              in-frame and fails the area ratio outright.
  cameraTrace (§4E) the source camera's height above the fitted floor as a per-frame trace. Flat in
              1.15-1.85 m passes; flat outside it is a scale error; sloped by > 20 cm is solve drift
              and must be reported as drift, not absorbed by the placement table.
  noiseFloor  (§4J) GUARDRAILS' rule in code: a solve whose residual sits below its own measurement
              noise, or whose "after" equals its "before", refuses instead of reporting.
  obsMask     (§4F) the observation sidecar must be ~0 inside the walker's mask and ~1 on textured
              non-person pixels. The old sidecar was neither: with no person mask, the walker's
              occlusion shadow was the front surface of the cloud and scored fully observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PY = sys.executable

IOU_MIN, AREA_LO, AREA_HI, MIN_FRAMES = 0.75, 0.85, 1.15, 8
OBS_IN_MASK_MAX, OBS_TEXTURED_MIN = 0.20, 0.80

WORLD_DIR = {
    "gym": "gym-4d",
    "elevator": "elevator-4d",
    "lobby": "lobby-4d",
    "living": "living-4dpp",
    "atrium": "atrium-4dpp",
    "stairs2": "stairs2-4d",
    "tos31": "tos31-4d",
}


def check_ruler(clip: str, extra: list[str], tol: float, *, report=None, **inputs) -> dict:
    out = report or ROOT / ".context" / "anchors" / f"ruler-{clip}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(ROOT / "scripts" / "world_ruler.py"),
        "--clip",
        clip,
        "--tol",
        str(tol),
        *extra,
    ]
    for name, value in inputs.items():
        if value is not None:
            cmd += ["--" + name.replace("_", "-"), str(value)]
    # A failed child must never reuse the successful JSON from an earlier candidate.
    with tempfile.TemporaryDirectory(prefix="anchor-ruler-") as temporary:
        fresh = Path(temporary) / "ruler.json"
        p = subprocess.run(
            [*cmd, "--json-out", str(fresh)], cwd=ROOT, capture_output=True, text=True
        )
        if p.returncode != 0 or not fresh.is_file():
            return dict(ok=False, why=[f"world_ruler.py failed: {p.stderr.strip()[-400:]}"])
        doc = json.loads(fresh.read_text())
        out.write_text(json.dumps(doc, indent=1))
    v = doc["verdict"]
    why = []
    if v.get("weaklyAnchored"):
        why.append(
            f"only {len(v['pointAnchors'])} point anchor(s): this clip's metres rest on the "
            f"avatar assumption and nothing else"
        )
    for d in v.get("disagreements", []):
        why.append(f"{d['a']} and {d['b']} disagree by {d['offBy']}")
    for b in v.get("bandViolations", []):
        why.append(
            f"the adopted metre puts {b['band'].replace('Band', '')} at {b['implied']} "
            f"{b['unit']}, outside {b['want']}"
        )
    pv = doc.get("personVsWorld")
    if pv and not pv["agrees"]:
        why.append(
            f"the avatar metre and the world metre differ by {pv['offBy']} (limit {tol * 100:.0f}%)"
        )
    return dict(
        ok=bool(v.get("ok")) and (pv or {}).get("agrees", True),
        why=why,
        mpuPerson=doc.get("mpuPerson"),
        adopted=v.get("adopted"),
        anchors={
            k: (a.get("mpu") if a.get("available") else a.get("reason"))
            for k, a in doc["anchors"].items()
        },
        report=str(out),
    )


def single_silhouette_rows(wd, cameras, video, masks, n):
    """Use this run's camera/source pair, without the shared historical mask cache."""
    import motion_audit
    import motion_silhouette as ms
    import person_masks_tight as pmt
    from place_solve import read_ply
    from PIL import Image
    from world_ruler import person_masks_for

    cams = json.loads(cameras.read_text())["cameras"]
    seq = json.loads((wd / "person" / "sequence.json").read_text())
    pick = [cams[i] for i in np.linspace(0, len(cams) - 1, min(n, len(cams))).round().astype(int)]
    if masks is not None:
        ground = person_masks_for("", masks=masks, world_dir=wd, dilate=False)
    else:
        source = Path(video)
        stamp = (str(source.resolve()), source.stat().st_size, source.stat().st_mtime_ns)
        key = hashlib.sha256(
            (str(wd.resolve()) + json.dumps(stamp) + json.dumps(pick) + json.dumps(seq)).encode()
        ).hexdigest()
        cache = ROOT / ".context" / "anchors" / f"tight-masks-{key}.npz"
        if not cache.is_file():
            hw = tuple(pick[0]["source_image_size"][::-1])
            seeds = motion_audit.seed_boxes(wd, pick, seq, hw)
            frames = pmt.read_frames(str(source), list(seeds))
            indices = [i for i in seeds if i in frames]
            if not indices:
                return []
            tight = pmt.masks_for(
                np.stack([frames[i] for i in indices]), [seeds[i] for i in indices]
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, masks=np.packbits(tight, axis=-1), shape=tight.shape, indices=indices
            )
        ground, _ = ms.load_masks(cache)
    rows = []
    for cam in pick:
        index = cam["sourceIndex"]
        if index not in ground:
            continue
        k = int(np.argmin(np.abs(np.asarray(seq["sourceIndices"]) - index)))
        v = read_ply(wd / "person" / seq["frames"][k])
        xyz = np.column_stack([v["x"], v["y"], v["z"]])
        sizes = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])
        hw = tuple(cam["source_image_size"][::-1])
        gt = np.asarray(
            Image.fromarray(ground[index]).resize((hw[1], hw[0]), Image.Resampling.NEAREST)
        )
        pred = ms.splat_silhouette(xyz, v["opacity"], sizes, cam, hw, 30000)
        if gt.sum() > 500:
            rows.append(dict(iou=ms.iou(pred, gt), predPx=int(pred.sum()), gtPx=int(gt.sum())))
    return rows


def check_silhouette(
    clip: str, n: int = 12, *, world_dir=None, cameras=None, video=None, masks=None, placement=None
) -> dict:
    """§4D. motion_audit already computes both numbers; this is the floor under them."""
    import motion_audit

    wd = world_dir or ROOT / "public" / "worlds" / WORLD_DIR.get(clip, f"{clip}-4d")
    placement_path = placement or wd / "placement.json"
    place = json.loads(placement_path.read_text()) if placement_path.exists() else {}
    if place.get("silhouetteFit"):
        # Audit what the viewer now draws, including every track and its fitted rigid transform.
        # These are the fitting masks: an in-sample consistency check, not held-out validation.
        from silhouette_rows import load_track_masks
        from motion_silhouette import splat_silhouette, iou as mask_iou
        from place_solve import read_ply
        from PIL import Image

        masks, _, _ = load_track_masks(masks or Path(place["silhouetteFit"]["masks"]))
        man = json.loads((wd / "people.json").read_text())
        cams = {
            c["sourceIndex"]: c
            for c in json.loads((cameras or wd / "cameras.json").read_text())["cameras"]
        }
        seqs = {p["id"]: json.loads((wd / p["sequence"]).read_text()) for p in man["people"]}
        ratios, overlaps = [], []
        scale = place["registrationScale"]
        for sample in (
            np.linspace(0, len(man["sourceIndices"]) - 1, min(n, len(man["sourceIndices"])))
            .round()
            .astype(int)
        ):
            src = man["sourceIndices"][sample]
            if src not in cams:
                continue
            cam = cams[src]
            hw = tuple(cam["source_image_size"][::-1])
            pred, gt = np.zeros(hw, bool), np.zeros(hw, bool)
            for person in man["people"]:
                pid = person["id"]
                seq = seqs[pid]
                mask = masks.get(person["track"], {}).get(sample)
                if src not in seq["sourceIndices"] or mask is None:
                    continue
                k = seq["sourceIndices"].index(src)
                v = read_ply((wd / person["sequence"]).parent / seq["frames"][k])
                xyz = np.column_stack([v["x"], v["y"], v["z"]])
                sizes = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])
                pp = place["perPerson"][pid]
                size = pp.get("sizeScale", 1)
                offset = pp.get("offsetUnits", place["offsetUnits"])[k]
                xyz = (
                    xyz * size
                    + (np.array(place["pos0"]) + pp["constantUnits"] + np.array(offset)) / scale
                )
                pred |= splat_silhouette(xyz, v["opacity"], sizes + np.log(size), cam, hw, 30000)
                gt |= np.array(
                    Image.fromarray(mask).resize((hw[1], hw[0]), Image.Resampling.NEAREST)
                )
            if gt.any():
                ratios.append(float(pred.sum() / gt.sum()))
                overlaps.append(mask_iou(pred, gt))
        area, overlap = float(np.mean(ratios)), float(np.mean(overlaps))
        why = []
        if overlap < IOU_MIN:
            why.append(f"silhouette IoU {overlap:.3f} < {IOU_MIN}")
        if not AREA_LO <= area <= AREA_HI:
            why.append(f"area ratio {area:.3f} outside {AREA_LO}-{AREA_HI}")
        if len(ratios) < MIN_FRAMES:
            why.append(f"only {len(ratios)} measured frames")
        return dict(
            ok=not why,
            n=len(ratios),
            iou=overlap,
            areaRatio=area,
            why=why,
            inSample=True,
            method="all tracked people, applied placement; CPU splat approximation; verify in live viewer",
        )
    seq = json.loads((wd / "person" / "sequence.json").read_text())
    src = video or seq.get("sourceClip") or str(ROOT / "public" / "clips" / f"{clip}.mp4")
    if any(value is not None for value in (world_dir, cameras, video, masks)):
        rows = single_silhouette_rows(wd, cameras or wd / "cameras.json", src, masks, n)
    else:
        rows = motion_audit.audit_world(wd, src, n)
    if len(rows) < MIN_FRAMES:
        return dict(
            ok=False,
            n=len(rows),
            why=[f"only {len(rows)} frames had a usable person mask; {MIN_FRAMES} needed"],
        )
    iou = float(np.mean([r["iou"] for r in rows]))
    area = float(np.mean([r["predPx"] / max(r["gtPx"], 1) for r in rows]))
    why = []
    if iou < IOU_MIN:
        why.append(f"silhouette IoU {iou:.3f} < {IOU_MIN}")
    if not (AREA_LO <= area <= AREA_HI):
        why.append(
            f"area ratio {area:.3f} outside {AREA_LO}-{AREA_HI}: the avatar is the wrong "
            f"SIZE on him, which the in-frame fraction cannot see"
        )
    return dict(ok=not why, n=len(rows), iou=round(iou, 4), areaRatio=round(area, 4), why=why)


def check_camera_trace(place: dict) -> dict:
    """§4E."""
    t = ((place.get("surface") or {}).get("sizeCheck") or {}).get("cameraHeightTrace")
    if not t:
        return dict(
            ok=False, why=["placement.json has no camera-height trace; re-run place_solve.py"]
        )
    why = []
    if t["verdict"] == "drift":
        why.append(
            f"the source camera climbs {t['endToEndSlopeM']:+.2f} m end to end: that is solve "
            f"drift, and a per-sample placement table absorbs it silently"
        )
    elif t["verdict"] == "flat out of band":
        why.append(
            f"the trace is flat at {t['medianM']:.2f} m, outside 1.15-1.85 m: a scale error, "
            f"not drift"
        )
    return dict(
        ok=not why, verdict=t["verdict"], medianM=t["medianM"], slopeM=t["endToEndSlopeM"], why=why
    )


def check_noise_floor(place: dict) -> dict:
    """§4J."""
    nf = place.get("noiseFloor")
    if nf is None:
        return dict(
            ok=False, why=["placement.json predates the noise-floor rule; re-run place_solve.py"]
        )
    return dict(ok=bool(nf["ok"]), why=list(nf["refusals"]), contactCm=nf["contactCm"])


def check_obs_mask(
    clip: str,
    sidecar: Path | None,
    frames: int = 3,
    *,
    world_dir=None,
    world=None,
    cameras=None,
    masks=None,
    placement=None,
) -> dict:
    """§4F: the confidence must be ~0 where the walker stood and ~1 on real textured surface.

    Read the two numbers together. `inMask` is high for a good reason as well as a bad one: where the
    walker moves across a room, the wall he covered at frame f is in plain view at frame f+k, and
    that wall is genuinely observed. The threshold bites on the case the review is about -- a walker
    who covers the same patch for the whole clip, so nothing ever saw behind him. Measured on lobby,
    turning the person mask on moved 0.95 % of splats and dropped those by 0.27 on average, while
    `inMask` stayed at 0.99: on that clip the shadow really is re-observed. A clip where `inMask`
    stays near 1 AND the mask changes nothing is the one to distrust.
    """
    import export_observation_confidence as eo

    wd = world_dir or ROOT / "public" / "worlds" / WORLD_DIR.get(clip, f"{clip}-4d")
    place = placement or wd / "placement.json"
    if sidecar is None or not sidecar.exists():
        return dict(ok=False, why=[f"no observation sidecar at {sidecar}"])
    scale0 = json.loads(place.read_text())["registrationScale"] if place.exists() else None
    if scale0 is None:
        return dict(ok=False, why=["no fitted scale0 for this clip"])
    conf = np.frombuffer(sidecar.read_bytes(), np.uint8).astype(np.float32) / 255.0
    if world is None:
        world = Path(str(sidecar)[: -len(sidecar.name.split("-obs")[-1]) - 4] + ".spz")
        if not world.exists():
            world = next((ROOT / "public").glob(f"marble-{clip}*.spz"))
    xyz = eo.read_spz_xyz(world)
    alpha = np.ones(len(xyz), np.float32)
    if len(xyz) != len(conf):
        return dict(
            ok=False,
            why=[f"{sidecar.name} has {len(conf):,} entries, {world.name} has {len(xyz):,} splats"],
        )
    cams = json.loads((cameras or wd / "cameras.json").read_text())["cameras"]
    ground = None
    if masks is not None:
        from world_ruler import person_masks_for

        ground = person_masks_for(clip, masks=masks, world_dir=wd, dilate=False)
    pick = [cams[i] for i in np.linspace(0, len(cams) - 1, frames).round().astype(int)]
    ins, outs = [], []
    for c in pick:
        W, H = c["source_image_size"]
        ds = 4
        gw, gh = W // ds, H // ds
        if ground is not None:
            import cv2

            original = ground.get(c["sourceIndex"])
            pm = (
                None
                if original is None
                else cv2.resize(
                    original.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            )
        else:
            pm = eo.person_pixels(wd, c, gw, gh, ds, dilate=0)
        if pm is None:
            continue
        # per-pixel confidence of the front-most splat
        M = np.asarray(c["camera_to_world"], np.float64)
        R = M[:3, :3]
        u_, _, vt = np.linalg.svd(R)
        R = u_ @ vt
        t = M[:3, 3] * scale0
        K = np.asarray(c["source_intrinsics"], np.float64)
        P = np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], 1).astype(np.float64)
        pc = (P - t) @ R
        z = -pc[:, 2]
        ok = (z > 0.05) & (alpha > 0.2)
        uf = K[0, 0] / ds * pc[:, 0] / np.where(z > 0, z, 1) + K[0, 2] / ds
        vf = K[1, 1] / ds * (-pc[:, 1]) / np.where(z > 0, z, 1) + K[1, 2] / ds
        ok &= np.isfinite(uf) & np.isfinite(vf)
        u = np.where(ok, uf, 0).astype(np.int64)
        v = np.where(ok, vf, 0).astype(np.int64)
        ok &= (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
        flat = v[ok] * gw + u[ok]
        zb = np.full(gw * gh, np.inf)
        np.minimum.at(zb, flat, z[ok])
        img = np.full(gw * gh, np.nan, np.float32)
        front = z[ok] <= zb[flat] * 1.01
        img[flat[front]] = conf[np.nonzero(ok)[0][front]]
        img = img.reshape(gh, gw)
        good = np.isfinite(img)
        if pm.any() and (good & pm).any():
            ins.append(float(np.nanmean(img[good & pm])))
        bg = good & ~pm
        if bg.any():
            outs.append(float(np.nanmean(img[bg])))
    if not ins:
        return dict(ok=False, why=["no frame had both a walker mask and rendered confidence"])
    a, b = float(np.mean(ins)), float(np.mean(outs))
    why = []
    if a > OBS_IN_MASK_MAX:
        why.append(
            f"mean confidence inside the walker's mask is {a:.2f} (want < {OBS_IN_MASK_MAX}): "
            f"his occlusion shadow is being sold to the viewer as observed geometry"
        )
    if b < OBS_TEXTURED_MIN:
        why.append(
            f"mean confidence on non-person pixels is {b:.2f} (want > {OBS_TEXTURED_MIN}): "
            f"the sidecar is fading geometry the cameras did see"
        )
    return dict(ok=not why, inMask=round(a, 3), outMask=round(b, 3), frames=len(ins), why=why)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clip", required=True)
    ap.add_argument("--world-dir", type=Path, help="packaged world directory for this run")
    ap.add_argument("--world", type=Path, help="exact retained Marble SPZ for this run")
    ap.add_argument("--cameras", type=Path, help="packaged source cameras for this run")
    ap.add_argument("--video", type=Path, help="source video for this run")
    ap.add_argument("--masks", type=Path, help="tracking or tight masks with source-frame mapping")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--obs-sidecar", type=Path)
    ap.add_argument(
        "--placement",
        type=Path,
        help="a placement.json to read instead of the one beside the person; use this to "
        "check a fresh solve without overwriting what the viewer ships",
    )
    ap.add_argument(
        "--skip", default="", help="comma list of checks to skip (they then report 'skipped')"
    )
    ap.add_argument(
        "--ruler-arg",
        action="append",
        default=[],
        help="passed through to world_ruler.py, e.g. --ruler-arg=--vlm",
    )
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--no-gate", action="store_true", help="report without failing the run")
    a = ap.parse_args()

    inputs = {
        name: getattr(a, name)
        for name in ("world_dir", "world", "cameras", "video", "masks", "placement")
    }
    for name, path in inputs.items():
        if path is not None and not (path.is_dir() if name == "world_dir" else path.is_file()):
            ap.error(
                f"--{name.replace('_', '-')} is not a {'directory' if name == 'world_dir' else 'file'}: {path}"
            )
    wd = a.world_dir or ROOT / "public" / "worlds" / WORLD_DIR.get(a.clip, f"{a.clip}-4d")
    pj = a.placement or (wd / "placement.json")
    place = json.loads(pj.read_text()) if pj.exists() else {}
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}

    runners = {
        "ruler": lambda: check_ruler(
            a.clip,
            a.ruler_arg,
            a.tol,
            report=a.json_out.with_name("ruler-" + a.json_out.name) if a.json_out else None,
            **inputs,
        ),
        "silhouette": lambda: check_silhouette(
            a.clip, **{k: v for k, v in inputs.items() if k != "world"}
        ),
        "cameraTrace": lambda: check_camera_trace(place),
        "noiseFloor": lambda: check_noise_floor(place),
        "obsMask": lambda: check_obs_mask(
            a.clip, a.obs_sidecar, **{k: v for k, v in inputs.items() if k != "video"}
        ),
    }
    res = {}
    for name, fn in runners.items():
        if name in skip:
            res[name] = dict(ok=None, skipped=True)
            continue
        try:
            res[name] = fn()
        except Exception as e:
            res[name] = dict(ok=False, why=[f"the check itself failed: {type(e).__name__}: {e}"])

    print(f"\nanchor checks: {a.clip}")
    bad = []
    for name, r in res.items():
        if r.get("skipped"):
            print(f"  {name:12s} skipped")
            continue
        tag = "PASS" if r["ok"] else "FAIL"
        extra = {k: v for k, v in r.items() if k not in ("ok", "why", "report", "anchors")}
        print(f"  {name:12s} {tag}  {extra}")
        for w in r.get("why", []):
            print(f"               - {w}")
        if not r["ok"]:
            bad.append(name)

    doc = dict(
        clip=a.clip,
        inputs={k: str(v) for k, v in inputs.items() if v is not None},
        checks=res,
        failed=bad,
        ok=not bad,
    )
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(doc, indent=1))
        print(f"  wrote {a.json_out}")
    if bad and not a.no_gate:
        print(
            f"\nFAILED: {', '.join(bad)}. These are the checks that can see the error they are "
            f"named for; do not ship this clip until they pass or the failure is understood."
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
