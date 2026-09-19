#!/usr/bin/env python3
"""What is one world unit in metres, measured from something that is not the avatar?

  uv run --locked --group inference python scripts/world_ruler.py --clip gym
  uv run --locked --group inference python scripts/world_ruler.py --clip elevator --vlm --json-out share/ruler-elevator.json
  uv run --locked --group inference python scripts/world_ruler.py --clip gym --rotfix --vlm --vlm-cache share/vlm-gym.json

Every metre this project has ever printed came from `mpu = 1.70 / stature`: the avatar was DEFINED
to be 1.70 m and the room was measured against him. That is an assumption, not a measurement, and it
was wrong by 1.25x to 2.9x on four shipped clips at various times (share/METHODS-REVIEW.md §2.1).
Nothing checked it, because there was nothing to check it against.

This measures the ruler instead, unattended, from a stack of independent estimators. No estimator is
trusted alone; where two or more are available they must agree, and the run says which disagreed.

  metricDepth   a metric monocular depth model (Depth Anything V2 Metric Indoor by default) predicts
                absolute metres from a single frame. Ratio against the world's own rendered depth at
                the same camera is metres per world unit. Needs nothing in the scene, so it runs on
                every clip. This is the primary anchor.
  vlmObject     the VLM names a standard-size structure it can see (interior door 2.03-2.10 m, stair
                riser 17-19 cm, ceiling tile 600 mm, a power outlet, a light switch, a standard
                chair or table) and gives its pixel endpoints; both endpoints are ray-cast onto the
                world splats and the known metres are divided by the world-unit separation.
  gravity       for a clip with anything falling or thrown: fit the arc with g = 9.81 and solve for
                the metre, instead of fixing the metre and checking g. Exact when it is available.
  statureBand   adults are 1.5-1.9 m. An INTERVAL, not a point, and it can be outvoted. This is what
                the old ruler used as a single number.
  cameraBand    a handheld source camera sits 1.15-1.85 m above the floor. Also an interval; this is
                place_solve.py's existing size check, restated as a constraint rather than a gate.

Verdict: the adopted metre is the median of the point anchors. Two point anchors that differ by more
than --tol fail the run and name themselves. A clip with no point anchor at all is reported as
WEAKLY ANCHORED -- the bands alone cannot decide a metre, they can only refuse an absurd one.

CAVEAT, and it is not small: every number here is computed with the camera track as cameras.json
currently holds it. A track whose plane is tilted against the world floor moves the camera along the
clip, which changes the world-unit depth this measures and makes a body appear to change size along
its path; a rotation fix therefore changes these metres. The JSON records scale0, the model id and
the per-frame ratios, so re-running the same command after such a fix re-derives every number here.
Do not quote a metre from this script as settled while the frame it was measured in is being changed.

`--rotfix` is that re-run: it premultiplies every camera by scripts/frame_align.py's rotation -- the
same one fourd.html applies under ?rotfix=1 -- before anything is measured, and re-reads the camera
band out of the rotated track instead of dividing placement.json's value, which was fitted in the
frame being corrected. Without the flag the script behaves exactly as before.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The metric-depth model matters more than any other choice here, and the two families behave
# differently on this footage. Depth Anything V2 Metric takes NO intrinsics and has no field-of-view
# head, so its metres are quoted for whatever focal it was trained around: measured on these clips it
# put the elevator's men at 2.9-3.2 m (94 deg lens, f 888 px) and the lobby's at 1.3-1.7 m (59 deg,
# f 1697 px) -- a 2x swing that tracks the lens, not the room. DepthPro predicts its own field of
# view, so it is the default; its prediction is also checked against the solve's known focal here and
# reported, because a 25 % disagreement there is a reason to distrust its metres.
METRIC_MODEL = os.environ.get("WANDER_METRIC_DEPTH", "apple/DepthPro-hf")
DEFAULT_TOL = 0.15
# A named object only votes if the world agrees with itself about where it is. Both gates exist
# because the first VLM run measured tos31's 0.18 m stair riser as 1.16 world units: one endpoint
# ray missed the tread and landed somewhere down the hall, and nothing caught it.
VLM_RAY_TOL = 1.35  # two-ray vs fronto-parallel reading of the same dimension
VLM_MIN_SPAN_PX = 60.0  # below this, a few px of endpoint error is a double-digit % of the metre
STATURE_BAND = (1.50, 1.90)  # adult height, the loose version of the 1.70 m assumption
CAMERA_BAND = (1.15, 1.85)  # place_solve.py's existing handheld-camera prior

# Historical path overrides: clip -> (world spz, placement/camera directory, source video).
# New clips use run_clip.py's layout; they do not need an entry here.
CLIPS = {
    "gym": ("public/marble-gym-clean2.spz", "public/worlds/gym-4d", "public/clips/gym.mp4"),
    "elevator": (
        "public/marble-elevator-clean.spz",
        "public/worlds/elevator-4d",
        "public/clips/elevator.mp4",
    ),
    "lobby": ("public/marble-lobby-clean.spz", "public/worlds/lobby-4d", "public/clips/lobby.mp4"),
    "living": (
        "public/marble-living-finetuned.spz",
        "public/worlds/living-4dpp",
        "public/clips/living.mp4",
    ),
    "atrium": (
        "public/marble-atrium-clean.spz",
        "public/worlds/atrium-4dpp",
        "public/clips/atrium.mp4",
    ),
    "stairs2": (
        "public/marble-stairs2-clean.spz",
        "public/worlds/stairs2-4d",
        "public/clips/stairs2.mp4",
    ),
    "tos31": ("public/marble-tos31-image.spz", "public/worlds/tos31-4d", "public/clips/tos31.mp4"),
    "selfie": (
        "public/marble-selfie-finetuned.spz",
        "public/worlds/selfie-4d",
        "public/clips/selfie.mp4",
    ),
    "bedroom": (
        "public/marble-bedroom-clean.spz",
        "public/worlds/bedroom-4d",
        "public/clips/bedroom.mp4",
    ),
}

# The registration scale each preset ships with, for the clips place_solve.py has not fitted (no
# placement.json). Same values as scripts/frame_align.py's table, which reads them from fourd.html.
PRESET_SCALE0 = {
    "gym": 1.0020,
    "elevator": 1.0800,
    "lobby": 1.5300,
    "atrium": 1.0221,
    "living": 0.3750,
    "stairs2": 0.9760,
    "tos31": 1.2834,
    "selfie": 1.4560,
    "bedroom": 0.673,
}


def resolve_paths(clip_name, world=None, depth=None, cameras=None, video=None):
    """Resolve inputs without changing the historical clips' measurement frame.

    --depth names the Pi3X directory or an anchor/depth file within it. The ruler
    uses its companion cameras.json; it does not measure the anchor cloud itself.
    Placement remains in the published world directory, separate from raw Pi3X.
    """
    legacy = CLIPS.get(clip_name)
    wdir = ROOT / legacy[1] if legacy else ROOT / "public" / "worlds" / f"{clip_name}-4d"
    if world is None:
        if legacy:
            world = ROOT / legacy[0]
        else:
            # Match run_clip.py's world_spz(), then allow exported/other variants.
            candidates = [
                ROOT / "public" / f"marble-{clip_name}-{suffix}.spz"
                for suffix in ("clean", "image", "multi", "finetuned")
            ]
            world = next((p for p in candidates if p.is_file()), None)
            if world is None:
                matches = sorted((ROOT / "public").glob(f"marble-{clip_name}-*.spz"))
                if len(matches) != 1:
                    raise ValueError(
                        f"cannot choose a Marble world for {clip_name}: "
                        f"found {len(matches)} other variants; pass --world"
                    )
                world = matches[0]
    if cameras is None:
        if depth is not None:
            depth_dir = depth if depth.is_dir() else depth.parent
            cameras = depth_dir / "cameras.json"
        else:
            # a run's packaged cameras.json is the frame the viewer and placement.json use (levelled
            # by frame_align); the raw Pi3X one is the fallback for a run that never packaged
            packaged = wdir / "cameras.json"
            cameras = (
                wdir if legacy else ROOT / ".context" / "run" / clip_name / "pi3x"
            ) / "cameras.json"
            if not legacy and packaged.is_file():
                cameras = packaged
    elif (cameras.parent / "placement.json").is_file():
        # Preserve the existing explicit-camera mode for self-contained exports.
        wdir = cameras.parent
    if video is None:
        video = ROOT / legacy[2] if legacy else ROOT / "public" / "clips" / f"{clip_name}.mp4"
    return world, wdir, cameras, video


# ---------------------------------------------------------------- world geometry
def read_spz_xyz(path: Path):
    """Positions and alpha from a v2 shDegree-0 spz."""
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
    xyz = fixed.astype(np.float32) / float(1 << fb)
    alpha = np.frombuffer(raw, np.uint8, n, 16 + n * 9).astype(np.float32) / 255.0
    return xyz, alpha


def world_depth(xyz, alpha, cam, scale0, ds=4, min_alpha=0.2):
    """Front-most splat depth per pixel, in WORLD UNITS, at one source camera.

    The viewer's mapping: raw Marble coords are OpenCV, the viewer applies Rx(pi), and cameras.json
    is already in that frame with its translation scaled by scale0 (same convention as
    export_observation_confidence.py and scripts/ab-world-cliff.mjs).
    """
    M = np.asarray(cam["camera_to_world"], np.float64)
    R = M[:3, :3]
    u_, _, vt = np.linalg.svd(R)
    R = u_ @ vt
    t = M[:3, 3] * scale0
    K = np.asarray(cam["source_intrinsics"], np.float64)
    W, H = cam["source_image_size"]
    gw, gh = max(1, W // ds), max(1, H // ds)

    P = np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], 1).astype(np.float64)
    pc = (P - t) @ R  # R^T (p - t): OpenGL camera coords
    z = -pc[:, 2]  # OpenGL looks down -z
    ok = (z > 0.05) & (alpha > min_alpha)
    u = K[0, 0] / ds * pc[:, 0] / np.where(z > 0, z, 1) + K[0, 2] / ds
    v = K[1, 1] / ds * (-pc[:, 1]) / np.where(z > 0, z, 1) + K[1, 2] / ds
    ok &= np.isfinite(u) & np.isfinite(v)
    ui = np.floor(np.where(ok, u, 0)).astype(np.int64)
    vi = np.floor(np.where(ok, v, 0)).astype(np.int64)
    ok &= (ui >= 0) & (ui < gw) & (vi >= 0) & (vi < gh)
    buf = np.full(gw * gh, np.inf)
    np.minimum.at(buf, vi[ok] * gw + ui[ok], z[ok])
    return buf.reshape(gh, gw), (gw, gh)


def cast_ray(xyz, alpha, cam, scale0, px, py, radius_px=14.0, min_alpha=0.2):
    """Front-most splat within radius_px of (px, py): dict(p=world-unit point, z=camera depth)."""
    M = np.asarray(cam["camera_to_world"], np.float64)
    R = M[:3, :3]
    u_, _, vt = np.linalg.svd(R)
    R = u_ @ vt
    t = M[:3, 3] * scale0
    K = np.asarray(cam["source_intrinsics"], np.float64)
    P = np.stack([xyz[:, 0], -xyz[:, 1], -xyz[:, 2]], 1).astype(np.float64)
    pc = (P - t) @ R
    z = -pc[:, 2]
    ok = (z > 0.05) & (alpha > min_alpha)
    u = K[0, 0] * pc[:, 0] / np.where(z > 0, z, 1) + K[0, 2]
    v = K[1, 1] * (-pc[:, 1]) / np.where(z > 0, z, 1) + K[1, 2]
    near = ok & (np.abs(u - px) < radius_px) & (np.abs(v - py) < radius_px)
    if near.sum() < 8:
        return None
    zc = z[near]
    lo = np.percentile(zc, 12)  # the front surface, not one stray floater
    sel = np.nonzero(near)[0][zc <= lo * 1.05]
    return dict(p=P[sel].mean(0), z=float(z[sel].mean()))


# ---------------------------------------------------------------- anchor A: metric depth
def predict_depths(cams, clip: str, model_id=None, device=None, cache: Path | None = None):
    """[(cam, metric depth map in metres, predicted FOV or None)], one per decodable frame.

    Cached on disk by (model, video, frame). The prediction depends on the source pixels alone, so
    a run in a rotated camera frame reuses it -- which makes the before/after comparison exact
    rather than merely repeated, and halves the CPU bill for it.
    """
    import cv2
    import torch

    model_id = model_id or METRIC_MODEL
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    depthpro = "depthpro" in model_id.lower()
    tag = model_id.replace("/", "_")
    want = []
    out = {}
    cap = cv2.VideoCapture(clip)
    for cam in cams:
        fi = cam["sourceIndex"]
        cp = cache and Path(cache) / f"{Path(clip).stem}-{tag}-{fi}.npz"
        if cp and cp.exists():
            z = np.load(cp)
            out[fi] = (
                cam,
                z["depth"].astype(np.float32),
                (float(z["fov"]) if z["fov"].size and np.isfinite(z["fov"]) else None),
            )
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if ok:
            want.append((cam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cp))
    cap.release()

    if want:
        if depthpro:
            from transformers import DepthProForDepthEstimation, DepthProImageProcessor

            proc = DepthProImageProcessor.from_pretrained(model_id)
            model = (
                DepthProForDepthEstimation.from_pretrained(model_id, dtype=torch.float16)
                .eval()
                .to(device)
            )
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            proc = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(device)
        for cam, rgb, cp in want:
            inp = proc(images=rgb, return_tensors="pt")
            inp = {
                k: (v.to(device).half() if depthpro and v.dtype == torch.float32 else v.to(device))
                for k, v in inp.items()
            }
            with torch.no_grad():
                o = model(**inp)
            fov = None
            if depthpro:
                post = proc.post_process_depth_estimation(o, target_sizes=[rgb.shape[:2]])[0]
                dm = post["predicted_depth"].float().cpu().numpy()
                if post.get("field_of_view") is not None:
                    fov = float(post["field_of_view"])
            else:
                dm = (
                    torch.nn.functional.interpolate(
                        o.predicted_depth[None].float(),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
            out[cam["sourceIndex"]] = (cam, dm, fov)
            if cp:
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cp, depth=dm.astype(np.float16), fov=np.array(np.nan if fov is None else fov)
                )
    return [out[c["sourceIndex"]] for c in cams if c["sourceIndex"] in out]


def anchor_metric_depth(
    world_spz: Path,
    cams,
    clip: str,
    scale0: float,
    frames: list[int],
    person_masks=None,
    ds=4,
    device=None,
    model_id=None,
    depth_cache: Path | None = None,
) -> dict:
    import cv2

    model_id = model_id or METRIC_MODEL
    xyz, alpha = read_spz_xyz(world_spz)

    per_frame = []
    for cam, dm, fov in predict_depths(cams, clip, model_id, device, depth_cache):
        fi = cam["sourceIndex"]
        dw, (gw, gh) = world_depth(xyz, alpha, cam, scale0, ds=ds)
        dm_s = cv2.resize(dm, (gw, gh), interpolation=cv2.INTER_AREA)
        good = np.isfinite(dw) & (dw > 0.05) & (dm_s > 0.2) & (dm_s < 19.0)
        if person_masks is not None and fi in person_masks:
            pm = cv2.resize(
                person_masks[fi].astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            good &= ~pm
        if good.sum() < 500:
            continue
        r = np.log(dm_s[good] / dw[good])
        row = dict(
            sourceIndex=fi,
            n=int(good.sum()),
            mpu=float(np.exp(np.median(r))),
            p10=float(np.exp(np.percentile(r, 10))),
            p90=float(np.exp(np.percentile(r, 90))),
        )
        if fov is not None:
            K = np.asarray(cam["source_intrinsics"], float)
            f_true = float((K[0, 0] + K[1, 1]) / 2)
            f_pred = dm.shape[1] / (2 * np.tan(np.deg2rad(fov) / 2))
            row.update(
                fovDeg=round(fov, 2),
                focalPredictedPx=round(f_pred, 1),
                focalSolvedPx=round(f_true, 1),
                focalAgreement=round(f_pred / f_true, 3),
            )
        per_frame.append(row)
    if not per_frame:
        return dict(available=False, reason="no frame produced enough overlapping depth")
    vals = np.array([f["mpu"] for f in per_frame])
    fa = [f["focalAgreement"] for f in per_frame if "focalAgreement" in f]
    return dict(
        available=True,
        mpu=float(np.median(vals)),
        frames=len(vals),
        spreadPct=float(100 * (vals.max() - vals.min()) / np.median(vals)),
        model=model_id,
        perFrame=per_frame,
        focalAgreement=(round(float(np.median(fa)), 3) if fa else None),
        focalWarning=(
            None
            if not fa or abs(float(np.median(fa)) - 1) < 0.12
            else f"the model's own field of view implies a focal "
            f"{float(np.median(fa)):.2f}x the one the solve fitted; its metres "
            f"carry that error"
        ),
        note="median over frames of median(metric_depth_m / world_depth_units); the metre "
        "comes from the depth model, not from the world or the avatar",
    )


# ---------------------------------------------------------------- anchor B: VLM standard objects
VLM_PROMPT = """This is one frame from a video of a real indoor place. A 3D reconstruction of this
place needs a real-world size reference, and you are the only one who can name what is in the shot.

Find up to 4 things in THIS image whose real-world size is standardised and well known. Good
candidates, best first: an interior doorway (opening height 2.03-2.10 m), a single stair riser
(0.17-0.19 m), a ceiling tile (0.60 m pitch), a mains power outlet plate (0.115 m tall in North
America), a light switch plate (0.115 m tall), a standard interior door width (0.81-0.91 m), a
standard office/dining chair seat height (0.45 m), a desk or table surface height (0.73-0.76 m), a
handrail height (0.90-1.07 m), a lift/elevator door opening height (2.10 m).

For each, give the two pixel endpoints of the dimension you are naming, in this image's own pixel
coordinates with (0,0) at the TOP-LEFT. Be precise: the endpoints must sit on the two ends of the
dimension, not on the middle of the object. Only include something you can see clearly and whose
two endpoints are both visible and unoccluded. If you cannot see anything that qualifies, return an
empty list -- a wrong reference is much worse than none.

Answer with JSON and nothing else:
{"items":[{"object":"...","dimension":"height|width|pitch","metres":<number>,
           "p1":[x,y],"p2":[x,y],"confidence":<0-1>,"why":"what you are looking at"}]}"""


def anchor_vlm(
    cams,
    clip: str,
    world_spz: Path,
    scale0: float,
    model="gpt-4o",
    cache: Path | None = None,
    world=None,
) -> dict:
    """Standard-size objects the VLM names, ray-cast onto the world splats.

    What the VLM answers depends only on the source frame, never on the camera pose, so the answers
    are cached per (model, video, frame) and the ray-cast is redone from them. Re-casting the same
    objects in a rotated frame therefore costs nothing -- which is the only reason it is affordable
    to run this on both sides of a frame fix.
    """
    import cv2

    store = {}
    if cache and Path(cache).exists():
        store = json.loads(Path(cache).read_text())
    if "OPENAI_API_KEY" not in os.environ and not store:
        return dict(available=False, reason="OPENAI_API_KEY unset; source ~/.openai-env")
    import ssl
    import urllib.request

    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    xyz, alpha = world if world is not None else read_spz_xyz(world_spz)
    cap = None
    items = []
    for cam in cams:
        fi = cam["sourceIndex"]
        key = f"{model}|{Path(clip).name}|{fi}"
        if key in store:
            doc = store[key]
        else:
            # decoded only when the frame has to be sent; a cached answer never touches the video,
            # which is what makes the scale0 sweep below cheap enough to run at all
            if cap is None:
                cap = cv2.VideoCapture(clip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = cap.read()
            if not ok:
                continue
            b64 = base64.b64encode(
                cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
            ).decode()
            req_body = dict(
                model=model,
                response_format=dict(type="json_object"),
                messages=[
                    dict(
                        role="user",
                        content=[
                            dict(type="text", text=VLM_PROMPT),
                            dict(
                                type="image_url",
                                image_url=dict(url=f"data:image/jpeg;base64,{b64}"),
                            ),
                        ],
                    )
                ],
            )
            if not model.startswith("gpt-5"):  # the reasoning models take the default only
                req_body["temperature"] = 0
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(req_body).encode(),
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
                    raw = json.loads(r.read())
                doc = json.loads(raw["choices"][0]["message"]["content"] or "{}")
                doc["_usage"] = raw.get("usage")
            except Exception as e:  # one bad frame must not take the anchor down
                detail = ""
                body_txt = getattr(e, "read", None)
                if body_txt:
                    try:
                        detail = " " + body_txt().decode()[:300]
                    except Exception:
                        pass
                items.append(dict(sourceIndex=fi, error=f"{type(e).__name__}: {e}{detail}"))
                continue
            store[key] = doc
            if cache:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                Path(cache).write_text(json.dumps(store, indent=1))
        K = np.asarray(cam["source_intrinsics"], float)
        fpx = float((K[0, 0] + K[1, 1]) / 2)
        for it in doc.get("items") or []:
            try:
                p1, p2 = it["p1"], it["p2"]
                a = cast_ray(xyz, alpha, cam, scale0, p1[0], p1[1])
                b = cast_ray(xyz, alpha, cam, scale0, p2[0], p2[1])
                mid = cast_ray(xyz, alpha, cam, scale0, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            except Exception:
                continue
            span_px = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            row = dict(
                sourceIndex=fi,
                object=it.get("object"),
                dimension=it.get("dimension"),
                metres=it.get("metres"),
                confidence=it.get("confidence"),
                p1=it.get("p1"),
                p2=it.get("p2"),
                spanPx=round(span_px, 1),
                why=it.get("why"),
            )
            if a is None or b is None:
                row["skipped"] = "an endpoint hit no world splat"
                items.append(row)
                continue
            d = float(np.linalg.norm(a["p"] - b["p"]))
            row["worldUnits"] = round(d, 4)
            row["mpuTwoRay"] = round(float(it["metres"]) / d, 4) if d > 1e-6 else None
            # Second reading of the same object, from ONE depth instead of a difference of two. A
            # dimension across the view direction subtends span_px at the focal length, so its world
            # length is span_px * z / f. It costs one ray instead of two, so a single bad hit cannot
            # blow it up the way tos31's 0.18 m riser measured 1.16 units, and it is the reading the
            # two-ray one has to agree with before the object is allowed to vote.
            z = (mid or a)["z"] if (mid or a) else None
            d_fp = span_px * z / fpx if (z and fpx > 0) else None
            if d_fp and d_fp > 1e-6:
                row.update(
                    depthUnits=round(float(z), 4),
                    worldUnitsFrontoParallel=round(d_fp, 4),
                    mpu=round(float(it["metres"]) / d_fp, 4),
                )
            if row.get("mpu") and row.get("mpuTwoRay"):
                r = row["mpuTwoRay"] / row["mpu"]
                row["twoRayRatio"] = round(float(r), 3)
                if not (1 / VLM_RAY_TOL < r < VLM_RAY_TOL):
                    row["rejected"] = (
                        f"the two readings of this object differ {abs(r - 1) * 100:.0f}%"
                        f" -- an endpoint did not land on it"
                    )
            if span_px < VLM_MIN_SPAN_PX:
                row["rejected"] = (
                    f"{span_px:.0f} px across: too few pixels to place the endpoints"
                    f" to better than the tolerance"
                )
            items.append(row)
    if cap is not None:
        cap.release()
    good = [
        i["mpu"] for i in items if i.get("mpu") and not i.get("rejected") and 0.05 < i["mpu"] < 60
    ]
    if not good:
        return dict(
            available=False,
            items=items,
            reason=(
                "the VLM named nothing the world could measure"
                if items
                else "the VLM named nothing measurable in the world"
            ),
        )
    return dict(
        available=True,
        mpu=float(np.median(good)),
        n=len(good),
        items=items,
        model=model,
        spreadPct=float(100 * (max(good) - min(good)) / np.median(good)),
        note="known metres of a named standard object over its measured separation in the "
        "world splats; the size comes from the object, not from the avatar",
    )


# ---------------------------------------------------------------- anchor C: gravity
def anchor_gravity(fit_json: Path | None, mpu_person: float) -> dict:
    """mpu from a free-flight arc fitted with g free, rather than g fixed from the avatar metre.

    A free-g fit returns gravity in the CURRENT ruler's metres; the ruler that makes it 9.80665 is
    the one the arc measured. The uncertainty carries straight through, and on a short flight with
    no stereo baseline it is large enough to be worth printing rather than believing.
    """
    if fit_json is None or not Path(fit_json).exists():
        return dict(available=False, reason="no free-g object fit for this clip")
    d = json.loads(Path(fit_json).read_text())
    g = d.get("freeGravityMs2") or (d.get("freeG") or {}).get("g")
    sg = d.get("freeGravitySigma") or (d.get("freeG") or {}).get("sigma")
    if not g:
        return dict(available=False, reason=f"{fit_json} has no free-g control")
    mpu = mpu_person * 9.80665 / float(g)
    out = dict(available=True, mpu=float(mpu), freeGravityMs2=float(g))
    if sg:
        out.update(
            sigma=float(sg),
            lo=float(mpu_person * 9.80665 / (float(g) + 2 * float(sg))),
            hi=float(mpu_person * 9.80665 / max(float(g) - 2 * float(sg), 1e-6)),
        )
    out["note"] = (
        "the arc is fitted in metres with g = 9.81 and the metre is solved for; a wide "
        "band here means the arc cannot decide the ruler, not that the ruler is right"
    )
    return out


# ---------------------------------------------------------------- scale0 from the metric ruler
def fit_scale0(
    world_spz: Path,
    cams,
    clip: str,
    mpu_person_ref: float,
    scale0_ref: float,
    person_masks=None,
    ds=4,
    device=None,
    model_id=None,
    span=(0.25, 4.0),
    steps=41,
    depth_cache: Path | None = None,
) -> dict:
    """The registration scale that makes the world ruler and the avatar ruler agree.

    `scale0` places the SfM camera track into the Marble world, so it sets how far the camera stands
    from every wall. The avatar's metre moves as 1/scale0 (he is scaled by it); the metric-depth
    metre moves the other way, because moving the camera changes the world-unit depth it measures.
    They cross once, and the crossing is a scale0 fitted against an external metre rather than
    against a ratio that cancels it -- the exact blindness share/METHODS-REVIEW.md §2.3 names.

    Reported, never applied: a preset's scale is other people's decision, and this is one number to
    put beside theirs.
    """
    import cv2

    xyz, alpha = read_spz_xyz(world_spz)
    preds = [
        (cam, dm)
        for cam, dm, _ in predict_depths(cams, clip, model_id or METRIC_MODEL, device, depth_cache)
    ]
    if not preds:
        return dict(available=False, reason="no frame decoded")

    curve = []
    for s0 in np.geomspace(span[0], span[1], steps):
        vals = []
        for cam, dm in preds:
            dw, (gw, gh) = world_depth(xyz, alpha, cam, float(s0), ds=ds)
            dm_s = cv2.resize(dm, (gw, gh), interpolation=cv2.INTER_AREA)
            good = np.isfinite(dw) & (dw > 0.05) & (dm_s > 0.2) & (dm_s < 19.0)
            fi = cam["sourceIndex"]
            if person_masks is not None and fi in person_masks:
                pm = cv2.resize(
                    person_masks[fi].astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                good &= ~pm
            if good.sum() < 500:
                continue
            vals.append(float(np.exp(np.median(np.log(dm_s[good] / dw[good])))))
        if not vals:
            continue
        mw = float(np.median(vals))
        mp = mpu_person_ref * scale0_ref / float(s0)
        curve.append(
            dict(
                scale0=round(float(s0), 4),
                mpuWorld=round(mw, 4),
                mpuPerson=round(mp, 4),
                logRatio=round(float(np.log(mw / mp)), 4),
            )
        )
    if len(curve) < 3:
        return dict(available=False, reason="the sweep produced no usable points")
    lr = np.array([c["logRatio"] for c in curve])
    s = np.array([c["scale0"] for c in curve])
    sign = np.where(np.diff(np.sign(lr)) != 0)[0]
    best = None
    if len(sign):
        i = int(sign[0])
        w = lr[i] / (lr[i] - lr[i + 1])
        best = float(np.exp(np.log(s[i]) + w * (np.log(s[i + 1]) - np.log(s[i]))))
    return dict(
        available=True,
        fittedScale0=best,
        referenceScale0=scale0_ref,
        ratioToReference=(round(best / scale0_ref, 3) if best else None),
        curve=curve,
        note="scale0 where the metric-depth metre and the avatar metre cross; reported, not "
        "applied -- presets are not this script's to change",
    )


def fit_scale0_vlm(
    world_spz: Path,
    cams,
    clip: str,
    mpu_person_ref: float,
    scale0_ref: float,
    model: str,
    cache: Path | None,
    span=(0.25, 4.0),
    steps=41,
) -> dict:
    """The same crossing as fit_scale0, read off the named objects instead of the depth model.

    scale0 sets where the source camera stands, so it sets how far a doorway is and therefore how
    many world units its known 2.03 m covers. The avatar's metre moves the other way. They cross
    once. Nothing here calls the API -- the VLM's answers are pixels, and pixels do not move when
    the camera does -- so this is the cheap half of the adjudication and the only one that puts a
    standard object's opinion of scale0 beside the depth model's.
    """
    world = read_spz_xyz(world_spz)
    curve = []
    for s0 in np.geomspace(span[0], span[1], steps):
        a = anchor_vlm(cams, clip, world_spz, float(s0), model, cache=cache, world=world)
        if not a.get("available"):
            continue
        mw, mp = float(a["mpu"]), mpu_person_ref * scale0_ref / float(s0)
        curve.append(
            dict(
                scale0=round(float(s0), 4),
                mpuWorld=round(mw, 4),
                mpuPerson=round(mp, 4),
                n=a["n"],
                logRatio=round(float(np.log(mw / mp)), 4),
            )
        )
    if len(curve) < 3:
        return dict(available=False, reason="too few scale0 steps produced a measurable object")
    lr = np.array([c["logRatio"] for c in curve])
    sc = np.array([c["scale0"] for c in curve])
    sign = np.where(np.diff(np.sign(lr)) != 0)[0]
    best = None
    if len(sign):
        i = int(sign[0])
        w = lr[i] / (lr[i] - lr[i + 1])
        best = float(np.exp(np.log(sc[i]) + w * (np.log(sc[i + 1]) - np.log(sc[i]))))
    return dict(
        available=True,
        fittedScale0=best,
        referenceScale0=scale0_ref,
        ratioToReference=(round(best / scale0_ref, 3) if best else None),
        curve=curve,
        note="scale0 where the standard-object metre and the avatar metre cross; reported, "
        "not applied",
    )


# ---------------------------------------------------------------- the frame the metres live in
def apply_rotfix(cams, fa: dict):
    """Rotate every source camera out of the phone's frame-0 body frame into the world's.

    The docstring's CAVEAT, made runnable. scripts/frame_align.py measured the angle (6-19 deg on
    eight of nine clips) and fourd.html applies it under ?rotfix=1 by premultiplying each
    camera_to_world; this is the same premultiply, so the metres below are measured from where the
    viewer actually puts the camera. Everything else -- scale0, the model, the frame picks -- is
    unchanged, which is what makes the two runs comparable.
    """
    R = np.asarray(fa["rotationRowMajor"], float)
    for c in cams:
        M = np.asarray(c["camera_to_world"], float).copy()
        M[:3, :3] = R @ M[:3, :3]
        M[:3, 3] = R @ M[:3, 3]
        c["camera_to_world"] = M.tolist()
    return R


def camera_height_units(cams, fa: dict, scale0: float):
    """Median and span of the source camera's height above the world floor, in world units.

    placement.json's `cameraHeightM` was fitted in the unrotated frame, so once the frame turns it
    has to be re-read rather than divided out: the rotation is precisely what was making it swing.
    The span is the tell -- a handheld phone does not change height by two metres mid-clip.
    """
    coef = np.asarray(fa["worldFloor"]["coef"], float)
    C = np.array([np.asarray(c["camera_to_world"], float)[:3, 3] for c in cams]) * scale0
    h = C[:, 1] - (coef[0] * C[:, 0] + coef[1] * C[:, 2] + coef[2])
    return float(np.median(h)), float(np.ptp(h))


# ---------------------------------------------------------------- the stack
def combine(anchors: dict, bands: dict, tol: float) -> dict:
    pts = {k: v["mpu"] for k, v in anchors.items() if v.get("available") and v.get("mpu")}
    out = dict(pointAnchors=pts, tolerance=tol)
    if not pts:
        out.update(
            ok=False,
            weaklyAnchored=True,
            adopted=None,
            reason="no point anchor available; the bands alone cannot fix a metre",
        )
        return out
    adopted = float(np.median(list(pts.values())))
    disagree = []
    names = sorted(pts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = pts[a] / pts[b]
            if abs(r - 1.0) > tol:
                disagree.append(
                    dict(a=a, b=b, ratio=round(float(r), 4), offBy=f"{abs(r - 1) * 100:.1f}%")
                )
    out.update(adopted=adopted, weaklyAnchored=len(pts) < 2, disagreements=disagree)
    band_fail = []
    for k, (lo, hi, val, unit) in bands.items():
        implied = val * adopted
        if not (lo <= implied <= hi):
            band_fail.append(dict(band=k, implied=round(implied, 3), want=[lo, hi], unit=unit))
    out["bandViolations"] = band_fail
    out["ok"] = (not disagree) and (not band_fail) and not out["weaklyAnchored"]
    return out


def frames_for(cams, n):
    idx = np.linspace(0, len(cams) - 1, min(n, len(cams))).round().astype(int)
    return [cams[i] for i in dict.fromkeys(idx.tolist())]


def person_masks_for(clip_name: str, ds_shape=None):
    """Per-source-frame person masks, when the tracking stage left any on disk."""
    import cv2

    p = ROOT / ".context" / "mp" / clip_name / "tracks" / "masks.npz"
    if not p.exists():
        # the single-person pipeline caches Mask R-CNN masks per world instead (motion_audit.py)
        for wd in (ROOT / ".context" / "pose" / "masks").glob(f"{clip_name}-4d*.npz"):
            z = np.load(wd)
            shp = z["shape"]
            m = np.unpackbits(z["masks"], axis=-1)[..., : shp[2]].reshape(tuple(shp)).astype(bool)
            out = {
                int(i): cv2.dilate(m[k].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                for k, i in enumerate(z["indices"])
            }
            return out
        return None
    z = np.load(p)
    fh, fw, ms = z["shape"]
    h, w = int(round(fh * ms)), int(round(fw * ms))
    tdirs = sorted((p.parent).glob("track_*"))
    src = {}
    for td in tdirs:
        t = int(td.name.split("_")[1])
        for f in json.loads((td / "motion.json").read_text())["frames"]:
            src.setdefault(f["sample"], f["sourceIndex"])
    out = {}
    for k in z.files:
        if k == "shape":
            continue
        s = int(k.split("_")[1][1:])
        fi = src.get(s)
        if fi is None:
            continue
        m = np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool)
        out[fi] = out[fi] | m if fi in out else m
    # dilate a little: the mask is tight and the person's shadow edge is not world geometry
    return {
        k: cv2.dilate(v.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        for k, v in out.items()
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--clip", required=True, help="clip name in run_clip.py's layout, or a historical clip"
    )
    ap.add_argument("--world", type=Path, help="override the Marble world SPZ")
    ap.add_argument(
        "--depth",
        type=Path,
        help="override the Pi3X directory or anchor/depth file; use its companion cameras.json",
    )
    ap.add_argument(
        "--cameras", type=Path, help="override cameras.json (takes precedence over --depth)"
    )
    ap.add_argument("--video", type=Path, help="override the source video")
    ap.add_argument("--scale0", type=float, help="default: placement.json's registrationScale")
    ap.add_argument("--mpu-person", type=float, help="default: placement.json's metresPerWorldUnit")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--vlm", action="store_true", help="also run the standard-object anchor")
    ap.add_argument("--vlm-model", default="gpt-4o")
    ap.add_argument(
        "--vlm-cache",
        type=Path,
        help="reuse/record the VLM's answers here; they do not depend on the camera pose",
    )
    ap.add_argument(
        "--rotfix",
        action="store_true",
        help="rotate the SfM frame into the world frame first (framealign.json, the "
        "same correction fourd.html applies under ?rotfix=1)",
    )
    ap.add_argument(
        "--stature-units",
        type=float,
        help="statureBand value for a clip place_solve.py has not fitted",
    )
    ap.add_argument("--object-fit", type=Path, help="a lift_object_3d.py run with a free-g control")
    ap.add_argument("--no-metric-depth", action="store_true")
    ap.add_argument("--metric-model", default=METRIC_MODEL)
    ap.add_argument("--metric-device")
    ap.add_argument(
        "--depth-cache",
        type=Path,
        help="keep the metric-depth predictions here; they do not depend on the camera",
    )
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--gate", action="store_true", help="exit non-zero unless the stack agrees")
    ap.add_argument(
        "--fit-scale0",
        action="store_true",
        help="also sweep scale0 and report where the two rulers cross (never applies it)",
    )
    a = ap.parse_args()

    try:
        world, wdir, cams_json, clip = resolve_paths(a.clip, a.world, a.depth, a.cameras, a.video)
    except ValueError as e:
        ap.error(str(e))
    pj = wdir / "placement.json"
    place = json.loads(pj.read_text()) if pj.exists() else {}
    scale0 = a.scale0 if a.scale0 is not None else place.get("registrationScale")
    mpu_person = a.mpu_person if a.mpu_person is not None else place.get("metresPerWorldUnit")
    if scale0 is None and a.clip in PRESET_SCALE0:
        scale0 = PRESET_SCALE0[a.clip]
    if scale0 is None:
        sys.exit("no scale0: pass --scale0, or run place_solve.py so placement.json exists")

    cams = json.loads(Path(cams_json).read_text())["cameras"]
    fa = None
    if a.rotfix:
        fa_path = wdir / "framealign.json"
        if not fa_path.exists():
            sys.exit(f"--rotfix: no {fa_path}; run scripts/frame_align.py solve --write")
        fa = json.loads(fa_path.read_text())
        apply_rotfix(cams, fa)
    pick = frames_for(cams, a.frames)
    anchors = {}
    if not a.no_metric_depth:
        anchors["metricDepth"] = anchor_metric_depth(
            Path(world),
            pick,
            str(clip),
            scale0,
            [c["sourceIndex"] for c in pick],
            person_masks=person_masks_for(a.clip),
            ds=a.ds,
            device=a.metric_device,
            model_id=a.metric_model,
            depth_cache=a.depth_cache,
        )
    if a.vlm:
        anchors["vlmObject"] = anchor_vlm(
            pick, str(clip), Path(world), scale0, a.vlm_model, cache=a.vlm_cache
        )
    anchors["gravity"] = anchor_gravity(a.object_fit, mpu_person or 1.0)
    sweep = vsweep = None
    if a.fit_scale0 and mpu_person:
        if not a.no_metric_depth:
            sweep = fit_scale0(
                Path(world),
                pick,
                str(clip),
                mpu_person,
                scale0,
                person_masks=person_masks_for(a.clip),
                ds=a.ds,
                device=a.metric_device,
                model_id=a.metric_model,
                depth_cache=a.depth_cache,
            )
        if a.vlm:
            vsweep = fit_scale0_vlm(
                Path(world), pick, str(clip), mpu_person, scale0, a.vlm_model, a.vlm_cache
            )

    sc = (place.get("surface") or {}).get("sizeCheck") or {}
    stature = a.stature_units or sc.get("statureUnits")
    cam_units = (
        (sc.get("cameraHeightM") / mpu_person) if (sc.get("cameraHeightM") and mpu_person) else None
    )
    cam_span = None
    if fa is not None:
        cam_units, cam_span = camera_height_units(cams, fa, scale0)
    bands = {}
    if stature:
        bands["statureBand"] = (STATURE_BAND[0], STATURE_BAND[1], stature, "m tall")
    if cam_units:
        bands["cameraBand"] = (CAMERA_BAND[0], CAMERA_BAND[1], cam_units, "m above the floor")

    verdict = combine(anchors, bands, a.tol)
    doc = dict(
        clip=a.clip,
        world=str(world),
        scale0=scale0,
        mpuPerson=mpu_person,
        rotfix=(
            None
            if fa is None
            else dict(
                applied=True,
                tiltFromYDeg=fa["tiltFromYDeg"],
                source=fa["source"],
                cameraHeightUnits=cam_units,
                cameraHeightSpanUnits=cam_span,
            )
        ),
        anchors=anchors,
        bands={k: dict(lo=v[0], hi=v[1], valueUnits=v[2], unit=v[3]) for k, v in bands.items()},
        verdict=verdict,
        scale0Fit=sweep,
        scale0FitVlm=vsweep,
    )
    if mpu_person and verdict.get("adopted"):
        r = verdict["adopted"] / mpu_person
        doc["personVsWorld"] = dict(
            ratio=round(float(r), 4),
            offBy=f"{abs(r - 1) * 100:.1f}%",
            agrees=bool(abs(r - 1) <= a.tol),
        )

    print(
        f"\n{a.clip}: scale0 {scale0}, avatar metre {mpu_person}"
        + (f", ROTFIX {fa['tiltFromYDeg']:.1f} deg" if fa else "")
    )
    for k, v in anchors.items():
        if v.get("available"):
            print(
                f"  {k:13s} 1 u = {v['mpu']:.4f} m"
                + (
                    f"   (spread {v['spreadPct']:.0f}% over {v.get('frames', v.get('n'))} samples)"
                    if v.get("spreadPct") is not None
                    else ""
                )
            )
        else:
            print(f"  {k:13s} not available: {v.get('reason')}")
    for k, v in doc["bands"].items():
        print(
            f"  {k:13s} implies 1 u in [{v['lo'] / v['valueUnits']:.4f}, "
            f"{v['hi'] / v['valueUnits']:.4f}] m"
        )
    if verdict.get("adopted"):
        print(
            f"  ADOPTED       1 u = {verdict['adopted']:.4f} m"
            + (f"   (avatar metre is off by {doc['personVsWorld']['offBy']})" if mpu_person else "")
        )
    for d_ in verdict.get("disagreements", []):
        print(
            f"  DISAGREE      {d_['a']} vs {d_['b']}: {d_['offBy']} apart (limit {a.tol * 100:.0f}%)"
        )
    for b in verdict.get("bandViolations", []):
        print(
            f"  BAND VIOLATED {b['band']}: adopted metre implies {b['implied']} {b['unit']}, "
            f"want {b['want']}"
        )
    for nm, sw in (("depth", sweep), ("object", vsweep)):
        if sw and sw.get("available") and sw.get("fittedScale0"):
            print(
                f"  SCALE0 FIT    the avatar and the {nm} ruler cross at scale0 "
                f"{sw['fittedScale0']:.4f} ({sw['ratioToReference']:.2f}x the {scale0} this run "
                f"used). Reported only."
            )
    if verdict.get("weaklyAnchored"):
        print(
            f"  WEAKLY ANCHORED: {len(verdict['pointAnchors'])} point anchor(s). This clip's "
            f"metres are not corroborated."
        )

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(doc, indent=1))
        print(f"  wrote {a.json_out}")

    if a.gate and not verdict["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
