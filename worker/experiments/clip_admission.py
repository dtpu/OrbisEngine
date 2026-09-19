#!/usr/bin/env python3
"""Tell someone what a clip will let them do, before spending minutes reconstructing it.

Measured on this repo's clips (see docs/experiments/orbit-clip-generalization.md): how far you can
look around is set by how far the camera *turned*, not how far it travelled. A forward walk that
sweeps 1.5 degrees of heading and a lateral truck that sweeps 42.9 degrees while travelling 65%
further both top out near 41 degrees; the clip that sweeps 135 degrees reaches 120.

Two ways in, and they answer different questions:

  --sfm DIR     the look-around verdict. Needs camera poses, so it runs after SfM, not before.
  --video FILE  reconstruction-risk signals, which need no poses and take a couple of seconds.
  --frames DIR  a PREDICTION before the real reconstruction: a coarse solve of 32 evenly spaced
                frames at 640 px, exhaustively matched, which takes tens of seconds instead of
                twenty minutes. It answers both questions approximately - whether the frames
                register at all, and how far the heading sweeps - and is recorded so that the
                prediction can be held against the full solve afterwards.

The two answer different questions and neither substitutes for the other: heading sweep says whether
a clip *can* let you look around, never whether it will reconstruct at all.

A pose-free heading estimate was tried and dropped. Chaining frame-to-frame rotations from
essential matrices drifts without bound: it read 180 degrees of sweep on all five clips here,
including the corridor walk that actually turns 1.52. Separating rotation from translation without
depth is the problem SfM exists to solve, and a proxy that is confidently wrong is worse than an
honest wait.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_motion import camera_inliers, horizontal_headings, sweep_deg
from heading_baseline import BASELINE_FLOOR_DEG, BIN_DEG, MIN_BIN_FRAMES, heading_bins

# Bands are grounded in the measured clips. lookAroundDeg is the angle off the recorded heading at
# which the view is expected to stop being worth looking at: two thirds of the geometric reach,
# because on the one clip where both were measured the splat degraded to smear at about 60 degrees
# while structure carried on to 85.
USABLE_FRACTION = 0.66
BANDS = [
    (25.0, "none", "This clip will not let you look around. The camera barely turned, so there is "
                   "nothing recorded to either side of where it pointed."),
    (50.0, "narrow", "You will get a narrow window. You can look a little off to the side, but the "
                     "view falls apart well before a quarter turn."),
    (80.0, "moderate", "You will be able to look around comfortably, over roughly a third of a "
                       "turn, before the view runs out."),
    (1e9, "wide", "You will be able to look around freely, most of the way around from where you "
                  "stand."),
]


def verdict(look_around_deg: float) -> dict:
    for limit, tier, sentence in BANDS:
        if look_around_deg < limit:
            return dict(tier=tier, headline=sentence,
                        lookAroundDeg=round(look_around_deg, 1))
    raise AssertionError


def risk_signals(frames: list[np.ndarray]) -> dict:
    """Cheap per-frame descriptions of the footage. Not a failure predictor - see `discrimination`.

    An earlier version of this raised a "reconstruction risk" tier from these numbers. It was wrong:
    it ranked a cinema shot that reconstructed fine *below* the one clip that failed outright, and a
    threshold that fires on working footage is worse than no threshold.
    """
    import cv2

    texture, blown, moving = [], [], []
    for a in frames:
        g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        texture.append(float(((gx + gy) > 40).mean()))
        blown.append(float((g > 250).mean()))

    # Content that moves after the camera's own motion is taken out: people, traffic, screens,
    # and specular highlights sliding over a polished surface all land here.
    orb = cv2.ORB_create(1500)
    for a, b in zip(frames, frames[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        ka, da = orb.detectAndCompute(ga, None)
        kb, db = orb.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 20 or len(kb) < 20:
            continue
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        if len(m) < 20:
            continue
        src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            continue
        warp = cv2.warpPerspective(ga, H, (ga.shape[1], ga.shape[0]))
        valid = warp > 0
        if valid.sum() < 0.5 * valid.size:
            continue
        diff = np.abs(warp.astype(np.float32) - gb.astype(np.float32))
        moving.append(float((diff[valid] > 40).mean()))

    tex = float(np.median(texture))
    blo = float(np.median(blown))
    res = float(np.median(moving)) if moving else float("nan")
    # These do NOT separate the failures from the successes measured so far, and saying so is more
    # use than a threshold that fires on working footage. The clip that failed outright scored
    # texture 0.199; a cinema shot that reconstructed fine scored 0.195, and the sparsest success
    # here matches the failure's planar residual to three decimals. So they are reported as context
    # and flagged only at values below anything yet seen to work.
    notes = []
    if tex < 0.12:
        notes.append("Very little texture to match, below anything measured here that reconstructed.")
    if res == res and res < 0.03:
        notes.append("Frame-to-frame change is almost entirely explained by one flat plane, below "
                     "anything measured here that reconstructed: either the scene is flat or the "
                     "camera barely moved relative to how far away everything is.")
    if blo > 0.05:
        notes.append("Blown-out highlights over a noticeable part of the frame carry no detail to "
                     "match.")
    return dict(textureDensity=round(tex, 4), blownHighlightFraction=round(blo, 4),
                planarResidualFraction=None if res != res else round(res, 4),
                notes=notes,
                discrimination="none demonstrated: on the clips measured these do not separate the "
                               "one reconstruction failure from successes scoring the same. The "
                               "check that does work is the preflight itself - a 10 s reconstruction "
                               "at every tenth frame either registers or it does not.")


# ---------------------------------------------------------------- scope

# Out-of-scope classes, from the user's scope note (): volumetrics (fire, smoke, fog,
# water), rigid moving objects, deformable things, camera effects, and multi-shot edits. The
# thresholds were set against eight Pexels clips of exactly those things and nine clips this
# pipeline reconstructs; each is placed above everything measured in scope.
# Measured  on eight Pexels effects clips (fire, sparkler, fog, thin smoke, fountain,
# curtain in wind, rain on glass, passing car) and nine clips this pipeline solved or refused:
#   warm flicker   fireplace 0.18, sparkler 0.013 | in scope at most 0.007 (a red hammock in sun)
#   unstable       sparkler 0.67, fireplace 0.45-0.54, passing car 0.30-0.41 | in scope at most
#                  0.19 (plants a metre from a moving lens: parallax, not motion) - so the gate
#                  sits above parallax and catches only content that dominates the frame
#   lighting jump  car headlights 43-48, fire 30, sparkler 30-40 | in scope at most 22 (gallery)
#   haze (dark channel) fog 0.60 BUT a white kitchen 0.50, a white bedroom 0.62, a white curtain
#                  0.91: it cannot tell fog from a bright room, so it is reported and NOT gated.
# Not detectable with these signals: fog, thin smoke, a small fountain, a slow curtain, rain on
# glass. Those need a semantic detector (fire/smoke/water/vehicle segmentation), not photometry.
SCOPE_UNSTABLE_MEDIAN = 0.25   # pixels still changing after camera-motion alignment: fire, spray, vehicles, crowds
SCOPE_WARM_FLICKER = 0.01      # bright saturated warm pixels that flicker: flames, sparks
SCOPE_LUM_JUMP = 28.0          # grey levels between consecutive frames: flashes, strobes, headlights
SCOPE_CUT_RATIO = 6.0          # frame difference over the clip median: a shot cut (no cut in the calibration set; untested)
SCOPE_ZOOM_DRIFT = 0.08        # DA3 focal spread over its median: a zoom (no zoom in the calibration set; untested)


def scope_signals(frames_dir: Path, focal: dict | None = None, n: int = 32, width: int = 320) -> dict:
    """Is this clip the kind of thing the pipeline is FOR? Measured, before any compute is spent.

    Every signal is reported; the verdict names the first class that trips. None of these
    signals can tell a person from a fountain - both are content that moves on its own after the
    camera's motion is taken out - so the class names are what the numbers are consistent with,
    not a recognition result.
    """
    import cv2
    names = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def load(i):
        f = cv2.imread(str(names[i]))
        h, w = f.shape[:2]
        return cv2.resize(f, (width, round(h * width / w)))

    # Motion and lighting are judged between ADJACENT extracted frames, never between sparse
    # samples: on a 37 s phone clip sampled 1.2 s apart, a tracked walking person read as "36% of
    # the frame moves on its own" and a doorway into a darker room as a 46-level flash. Both are
    # ordinary footage. n adjacent pairs are spread evenly over the clip.
    starts = np.unique(np.linspace(0, max(len(names) - 2, 0), min(n, max(len(names) - 1, 1))).astype(int))
    pairs = [(load(i), load(i + 1)) for i in starts]
    frames = [a for a, _ in pairs]
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    haze = [float((cv2.erode(f.min(axis=2), np.ones((7, 7), np.uint8)) > 120).mean()) for f in frames]
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    unstable, warm = [], []
    for fa, fb in pairs:
        a, b = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        H = None
        if da is not None and db is not None and len(ka) >= 30 and len(kb) >= 30:
            m = bf.match(da, db)
            if len(m) >= 30:
                src = np.float32([ka[x.queryIdx].pt for x in m])
                dst = np.float32([kb[x.trainIdx].pt for x in m])
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            H = np.eye(3)
        wa = cv2.warpPerspective(fa, H, (fa.shape[1], fa.shape[0]))
        valid = wa.max(axis=2) > 0
        if valid.sum() < 0.5 * valid.size:
            continue
        diff = np.abs(wa.astype(np.float32) - fb.astype(np.float32)).max(axis=2)
        unstable.append(float((diff[valid] > 40).mean()))
        hsv = cv2.cvtColor(fb, cv2.COLOR_BGR2HSV)
        fire = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 180)
        warm.append(float((fire & (diff > 40) & valid).mean()))
    # Lighting jumps and cuts over EVERY adjacent pair of extracted frames (grey means are cheap).
    means = []
    for nm in names:
        gi = cv2.imread(str(nm), cv2.IMREAD_GRAYSCALE)
        means.append(cv2.resize(gi, (64, round(gi.shape[0] * 64 / gi.shape[1]))).astype(np.float32))
    lum = np.array([x.mean() for x in means])
    dl = np.abs(np.diff(lum))
    dd = np.array([float(np.mean(np.abs(a - b))) for a, b in zip(means, means[1:])])
    rec = dict(
        framesSampled=len(frames), adjacentPairs=len(pairs),
        hazeP90=round(float(np.percentile(haze, 90)), 3),
        unstableMedian=round(float(np.median(unstable)), 3) if unstable else None,
        warmFlickerMedian=round(float(np.median(warm)), 4) if warm else None,
        lumJumpMax=round(float(dl.max()), 1) if len(dl) else 0.0,
        cutRatioMax=round(float(dd.max() / max(np.median(dd), 1e-6)), 2) if len(dd) else 0.0,
        focalDrift=(round(float(focal.get("focalSpreadPx", 0) or 0) / max(float(focal.get("focalPx", 1)), 1e-6), 4)
                    if focal and focal.get("source") == "da3" else None),
    )
    reasons = []
    notes = []
    if rec["hazeP90"] > 0.35:
        notes.append(f"dark-channel haze fraction {rec['hazeP90']:.2f}: fog, or just a bright pale room; the "
                     f"signal cannot tell them apart, so this is a note, not a refusal")
    rec["notes"] = notes
    if rec["warmFlickerMedian"] is not None and rec["warmFlickerMedian"] > SCOPE_WARM_FLICKER:
        reasons.append(f"flames or sparks ({rec['warmFlickerMedian']:.4f} of the frame is bright, saturated and flickering); fire is out of scope")
    if rec["unstableMedian"] is not None and rec["unstableMedian"] > SCOPE_UNSTABLE_MEDIAN:
        # A walking person tracked by the camera trips this too, and people are IN scope. The
        # refusal is therefore deferred to the judge (scope_judge) when it is available: a person
        # reading downgrades this to a note, an effects reading confirms it.
        rec["unstableDominates"] = True
        reasons.append(f"{rec['unstableMedian']:.0%} of the frame keeps changing between adjacent frames after the camera's motion is removed: water, cloth, a moving object or a crowd dominates the shot")
    if rec["cutRatioMax"] > SCOPE_CUT_RATIO:
        reasons.append(f"a shot cut ({rec['cutRatioMax']:.1f}x the median frame change); one shot per clip")
    if rec["lumJumpMax"] > SCOPE_LUM_JUMP:
        reasons.append(f"a lighting jump of {rec['lumJumpMax']:.0f} grey levels between frames: a flash, strobe or screen; camera and lighting effects are out of scope")
    if rec["focalDrift"] is not None and rec["focalDrift"] > SCOPE_ZOOM_DRIFT:
        reasons.append(f"the focal length drifts {rec['focalDrift']:.0%} across the clip: a zoom; a fixed lens is assumed")
    rec["inScope"] = not reasons
    rec["reasons"] = reasons
    rec["headline"] = ("In scope: a single shot, fixed lens, no volumetrics or lighting effects detected."
                       if not reasons else "Out of scope: " + reasons[0])
    return rec


def scope_judge(frames_dir: Path, n: int = 3) -> dict | None:
    """The semantic half: `vlm-judge scope` on a few single frames, if the tool and its key exist.

    Photometry above catches flames, sparks and a vehicle sweeping the frame and misses fog, water
    and thin smoke. The judge (validated on this project's own renders) caught fog, a fountain and
    rain on glass that photometry did not, and missed the thin coloured smoke that photometry also
    missed. Both halves run; the receipt carries both. Verdict rule: out of scope when a majority
    of judged frames say in_scope=false AND effects=true. "Insufficient texture" alone is a note:
    whether it reconstructs is the solve's question, and the coarse prediction answers it.
    """
    import shutil, subprocess
    if not shutil.which("vlm-judge") or not os.environ.get("OPENAI_API_KEY"):
        return None
    names = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not names:
        return None
    picks = [names[i] for i in np.linspace(len(names) * 0.2, len(names) * 0.8, n).astype(int)]
    verdicts = []
    for f in picks:
        try:
            out = subprocess.run(["vlm-judge", "scope", str(f)], capture_output=True, text=True, timeout=120).stdout
            line = [l for l in out.splitlines() if l.strip().startswith("{")]
            v = json.loads(line[-1]) if line else dict(error="no json")
        except Exception as e:
            v = dict(error=str(e)[:120])
        v["frame"] = f.name
        verdicts.append(v)
    ok = [v for v in verdicts if "error" not in v]
    if not ok:
        return dict(frames=verdicts, verdict="unavailable")
    effects = [v for v in ok if v.get("effects") and not v.get("in_scope", True)]
    texture = [v for v in ok if not v.get("in_scope", True) and not v.get("effects")]
    people = sorted({str(v.get("people")) for v in ok})
    rec = dict(frames=verdicts, framesJudged=len(ok), effectsOut=len(effects), textureDoubt=len(texture),
               people=people, setting=sorted({str(v.get("setting")) for v in ok}),
               motionBlur=sorted({str(v.get("motion_blur")) for v in ok}))
    if len(effects) * 2 > len(ok):
        rec["verdict"] = "out-of-scope"
        rec["reason"] = effects[0].get("reason", "effects dominate")
    else:
        rec["verdict"] = "in-scope"
        if texture:
            rec["note"] = f"{len(texture)} of {len(ok)} frames judged short of texture: " + texture[0].get("reason", "")
    return rec


def sample_frames(video: Path, n: int, width: int = 640) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 2, 0), n).astype(int)
    out, want = [], set(int(i) for i in idx)
    for i in range(total):
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            h, w = fr.shape[:2]
            out.append(cv2.resize(fr, (width, round(h * width / w))))
    cap.release()
    return out


def from_sfm(sfm: Path, expected_frames: int | None = None,
             bin_deg: float = BIN_DEG, floor_deg: float = BASELINE_FLOOR_DEG,
             min_bin_frames: int = MIN_BIN_FRAMES) -> tuple[dict, dict]:
    """(motion, per-heading baseline). The second is the answer to "swept, but rebuildable?"."""
    from warp_coverage_common import load_model
    frames, cams, xyz, lookup = load_model(sfm)
    cam = cams[sorted(cams)[0]]
    f, cx, cy = cam["params"][:3]
    half_hfov = float(np.degrees(np.arctan(cx / f)))

    order = sorted(frames)
    down = np.array([frames[i]["R"].T @ np.array([0.0, 1.0, 0.0]) for i in order])
    up = -down.mean(0); up /= np.linalg.norm(up)
    F = np.array([frames[i]["R"].T @ np.array([0.0, 0.0, 1.0]) for i in order])
    C = np.array([-(frames[i]["R"].T @ frames[i]["t"]) for i in order])

    keep = camera_inliers(C)
    outliers = int((~keep).sum())
    Fh = horizontal_headings(F[keep], up)
    sweep, sweep_max = sweep_deg(Fh)
    Ck = C[keep]
    path = float(np.linalg.norm(np.diff(Ck, axis=0), axis=1).sum())
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    med = (float(np.median(np.linalg.norm(xyz[:, None, :] - Ck[None, ::max(1, len(Ck) // 40 or 1), :], axis=2)))
           if len(xyz) else float("nan"))
    registered = None if not expected_frames else len(frames) / expected_frames

    # Per heading bin: was the content under this heading ever seen from a second place? The sweep
    # above cannot answer that, and the phone clip's final pan is 180 degrees of sweep over content
    # with no baseline at all. See heading_baseline.py and docs/experiments/heading-baseline.md.
    obs = np.full(len(order), np.nan)
    for j, i in enumerate(order):
        idx = [lookup[int(p)] for p in frames[i]["pids"] if int(p) >= 0 and int(p) in lookup]
        if len(idx) >= 8:
            obs[j] = float(np.median(np.linalg.norm(xyz[idx] - C[j], axis=1)))
    baseline = heading_bins(Ck, Fh, up, med, frame_ids=np.array(order)[keep],
                            observed_distance=obs[keep], bin_deg=bin_deg, floor_deg=floor_deg,
                            min_frames=min_bin_frames, half_fov_deg=half_hfov)

    return dict(source="sfm", frames=len(frames), registeredFraction=registered,
                outlierCameras=outliers, inlierCameras=int(keep.sum()),
                headingSweepDeg=round(sweep, 2), headingSweepMaxDeg=round(sweep_max, 2),
                halfHorizontalFovDeg=round(half_hfov, 2),
                structureDistance=round(med, 3),
                pathOverStructureDistance=round(path / max(med, 1e-9), 3)), baseline


def predict_from_frames(frames: Path, focal_px: float, n: int = 32, max_size: int = 640,
                        work: Path | None = None, camera_model: str = "SIMPLE_PINHOLE",
                        refine_focal: bool = False) -> dict:
    """The cheap solve. Every n-th frame, small images, every pair matched.

    Exhaustive matching on 32 frames is 496 pairs - cheaper than one window pass over the clip -
    and it does not depend on the sequential chain holding, so a clip whose middle is blurred can
    still register its two ends. The heading sweep from 32 cameras is a coarse version of the one
    the full solve gives; the registered fraction is the earliest honest sign of a clip that will
    not solve at all.
    """
    import os, shutil, tempfile, time
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import pycolmap
    from PIL import Image

    names = sorted(p for p in frames.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(names) < 8:
        return dict(status="too-few-frames", frames=len(names))
    pick = [names[i] for i in np.linspace(0, len(names) - 1, min(n, len(names))).astype(int)]
    w, h = Image.open(pick[0]).size
    t0 = time.time()
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="predict-"))
    imgs = work / "images"
    shutil.rmtree(work, ignore_errors=True)
    imgs.mkdir(parents=True)
    for p in pick:
        os.symlink(p.resolve(), imgs / p.name)
    db = work / "db.db"
    params = f"{focal_px},{w / 2},{h / 2}" + (",0" if camera_model == "SIMPLE_RADIAL" else "")
    reader = pycolmap.ImageReaderOptions(camera_model=camera_model, camera_params=params)
    extract = pycolmap.FeatureExtractionOptions(num_threads=4, max_image_size=max_size)
    extract.sift.max_num_features = 2048
    pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extract,
                              device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, matching_options=pycolmap.FeatureMatchingOptions(num_threads=4),
                              device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions(num_threads=4, random_seed=42,
                                               max_runtime_seconds=300, max_num_models=4)
    opts.mapper.init_min_tri_angle = 4
    # A film shot has no DA3 focal and no EXIF; with a pinhole model and exhaustive matching the
    # focal can be let go in bundle adjustment for RANKING purposes (the sweep barely depends on
    # it). The pipeline proper keeps it fixed.
    opts.ba_refine_focal_length = bool(refine_focal)
    opts.mapper.abs_pose_refine_focal_length = bool(refine_focal)
    opts.ba_refine_extra_params = camera_model == "SIMPLE_RADIAL"
    opts.mapper.abs_pose_refine_extra_params = camera_model == "SIMPLE_RADIAL"
    models = work / "models"
    models.mkdir()
    recs = pycolmap.incremental_mapping(db, imgs, models, options=opts)
    rec = dict(source="coarse-solve", framesTried=len(pick), maxImageSize=max_size,
               focalPx=focal_px, elapsedSeconds=round(time.time() - t0, 1))
    if not recs:
        rec.update(status="no-reconstruction", registeredFraction=0.0,
                   headline="Prediction: the sample did not register at all. Expect the full solve "
                            "to fail too, unless the failure is the fixed focal.")
        return rec
    best_i, best = max(recs.items(), key=lambda kv: kv[1].num_images())
    frac = best.num_images() / len(pick)
    if best.num_images() < 3 or best.num_points3D() == 0:
        # A two-frame "model" with no structure is what a locked-off camera produces: the mapper
        # initialises a pair and cannot triangulate anything. It is a failure, not a solve.
        rec.update(status="no-structure", registeredFraction=round(frac, 3),
                   registered=best.num_images(), points=best.num_points3D(),
                   headline=f"Prediction: {best.num_images()} frames posed with "
                            f"{best.num_points3D()} 3D points - no structure was recovered. "
                            f"Expect the full solve to fail; the camera probably did not move.")
        return rec
    rec.update(status="registered", registeredFraction=round(frac, 3),
               registered=best.num_images(), models=len(recs),
               meanReprojectionError=round(float(best.compute_mean_reprojection_error()), 3))
    motion, _ = from_sfm(models / str(best_i), len(pick))   # (motion, heading baseline)
    rec["motion"] = motion
    cam = list(best.cameras.values())[0]
    rec["solvedFocalPx"] = round(float(cam.params[0]), 1)
    rec["solvedHfovDeg"] = round(float(np.degrees(2 * np.arctan(w / (2 * cam.params[0])))), 1)
    reach = motion["headingSweepDeg"] / 2.0 + motion["halfHorizontalFovDeg"]
    rec["geometricReachDeg"] = round(reach, 1)
    if frac < 0.7:
        rec["verdict"] = dict(tier="inconclusive", lookAroundDeg=None,
                              headline=f"Prediction: only {frac:.0%} of the sample registered in "
                                       "one model. Expect a fragmented solve.")
    else:
        rec["verdict"] = verdict(USABLE_FRACTION * reach)
        rec["verdict"]["headline"] = "Prediction: " + rec["verdict"]["headline"]
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sfm", type=Path, help="COLMAP sparse dir: exact, but costs an SfM run")
    ap.add_argument("--video", type=Path, help="clip, for the reconstruction-risk signals")
    ap.add_argument("--expected-frames", type=int, default=None,
                    help="frames the reconstruction was asked to register; below "
                         "--min-registered the verdict is withheld rather than guessed")
    ap.add_argument("--min-registered", type=float, default=0.8)
    ap.add_argument("--max-outlier-fraction", type=float, default=0.02,
                    help="above this share of runaway camera positions the solve is "
                         "treated as unsound rather than merely having a stray pose")
    ap.add_argument("--risk-frames", type=int, default=16)
    ap.add_argument("--heading-bin-deg", type=float, default=BIN_DEG,
                    help="width of a heading bin in the per-heading baseline table")
    ap.add_argument("--baseline-floor-deg", type=float, default=BASELINE_FLOOR_DEG,
                    help="triangulation angle below which a heading bin is reported as seen but "
                         "not reconstructable")
    ap.add_argument("--min-bin-frames", type=int, default=MIN_BIN_FRAMES)
    ap.add_argument("--frames", type=Path, help="extracted frames: run the coarse predictive solve")
    ap.add_argument("--focal", type=float, default=None, help="focal in frame pixels, for --frames")
    ap.add_argument("--camera-model", default="SIMPLE_PINHOLE", help="for --frames")
    ap.add_argument("--predict-n", type=int, default=32)
    ap.add_argument("--refine-focal", action="store_true", help="for --frames: let bundle adjustment move the focal (ranking use)")
    ap.add_argument("--work", type=Path, default=None, help="scratch dir for --frames")
    ap.add_argument("--scope", type=Path, help="extracted frames: report the out-of-scope signals only")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.scope:
        res = scope_signals(a.scope)
        print(json.dumps(res, indent=1))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1) + "\n")
        return
    if a.frames:
        if a.focal is None:
            raise SystemExit("--frames needs --focal (pixels at the extracted frame size)")
        res = predict_from_frames(a.frames, a.focal, a.predict_n, work=a.work,
                                  camera_model=a.camera_model, refine_focal=a.refine_focal)
        print(json.dumps(res, indent=1))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1) + "\n")
        return
    if not a.sfm:
        raise SystemExit("--sfm is required: the look-around verdict needs camera poses. See the "
                         "write-up for why the pose-free estimate was dropped.")

    # The registration gate is the thing that stops a 10-frame fragment being reported as a
    # verdict, so it must not depend on the caller remembering to pass a frame count. orbit_sfm.py
    # leaves one in run.json beside the model.
    expected = a.expected_frames
    if expected is None:
        for up in (a.sfm.parent, a.sfm.parent.parent):
            run = up / "run.json"
            if run.exists():
                expected = json.loads(run.read_text()).get("keptFrames")
                break

    res = dict(clip=str(a.video) if a.video else None)
    motion, baseline = from_sfm(a.sfm, expected, bin_deg=a.heading_bin_deg,
                                floor_deg=a.baseline_floor_deg, min_bin_frames=a.min_bin_frames)
    res["motion"] = motion
    res["headingBaseline"] = baseline

    # Heading spread from a station in the middle of the move is about half the whole sweep, which
    # is what the measured clips show; add half the field of view for what the frame itself covers.
    reach = motion["headingSweepDeg"] / 2.0 + motion["halfHorizontalFovDeg"]
    res["geometricReachDeg"] = round(reach, 1)
    frac = motion["registeredFraction"]
    if frac is None:
        res["verdict"] = dict(
            tier="inconclusive", lookAroundDeg=None,
            headline="Cannot tell. The number of frames this reconstruction was asked to place is "
                     "unknown, so there is no way to tell a whole clip from a surviving fragment.")
    elif motion["outlierCameras"] / max(motion["frames"], 1) > a.max_outlier_fraction:
        # A few stray cameras are dropped and noted. A large share of them means the solve never
        # settled into one consistent solution, and no sweep measured off it is worth reporting.
        res["verdict"] = dict(
            tier="inconclusive", lookAroundDeg=None,
            headline=f"Cannot tell. {motion['outlierCameras']} of {motion['frames']} camera "
                     "positions came out far away from all the others, so the reconstruction has "
                     "not settled into one consistent solution.")
    elif frac < a.min_registered:
        # Poses from a reconstruction that dropped most of its frames describe whichever fragment
        # survived, not the clip. Withhold rather than guess.
        res["verdict"] = dict(
            tier="inconclusive", lookAroundDeg=None,
            headline="Cannot tell yet. Only "
                     f"{frac * 100:.0f}% of the frames could be placed, so there is not enough of "
                     "this clip reconstructed to say what looking around will be like.")
    else:
        res["verdict"] = verdict(USABLE_FRACTION * reach)
        if motion["outlierCameras"]:
            res["verdict"]["note"] = (
                f"{motion['outlierCameras']} of {motion['frames']} camera positions were excluded "
                "as strays before measuring. A handful is normal; it is reported so the number is "
                "never quietly measured off a broken pose.")

    # The tier above says how far you can TURN and is left exactly as it was. What follows says
    # which of the directions you can turn to were ever seen from a second place, which is a
    # different question and the one the phone clip's tail failed.
    caveats = []
    if baseline.get("caveat"):
        caveats.append(baseline["caveat"])

    if a.video:
        res["footage"] = risk_signals(sample_frames(a.video, a.risk_frames))
        if res["footage"]["notes"]:
            caveats.append("Separately from how far you can look: " +
                           " ".join(res["footage"]["notes"]))
    if caveats:
        res["verdict"]["caveat"] = " ".join(caveats)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
