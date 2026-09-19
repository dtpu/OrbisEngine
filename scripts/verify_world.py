#!/usr/bin/env python3
"""Capture offline static-world diagnostics; completion is never visual acceptance.

Source frames use decoded ordinal and PTS, rather than average-FPS labels. Camera registration
remains an estimate: matching timestamps cannot prove geometric correspondence. Review beginning,
middle and end pairs using world_quality_review.py's explicit manual contract.

  verify_world.py --world public/marble-gym-clean2.spz --cameras public/worlds/gym-4d/cameras.json \
      --clip public/clips/gym.mp4 --scale0 1.0020 --tag gym-raw --out .context/verify/gym-raw \
      [--frames 0,108,216] [--prompt-file .context/prompt/gym.json] [--no-vlm]

A world that was not generated from this clip's own frames has no fitted scale0 yet, and rendering
it at the wrong one looks exactly like a wrong room. `--fit-scale` sweeps scale0 and keeps the best
mean gradient correlation, and the curve it prints is itself a registration diagnostic: on the
shipped gym world it is flat (0.06-0.11 everywhere), which says the world does not line up with the
footage at ANY scale.

Writes source/render/pair images, sheet.png and report.json into a NEW output directory.
No model/API call, corrected prompt, regeneration or quality acceptance is available here.
"""

import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "worker" / "stages"))
from render_world_poses import cameras_for, read_spz, render  # noqa: E402
from quality_gate import (  # noqa: E402
    ACKNOWLEDGEMENTS,
    CRITERIA,
    EVIDENCE_SCHEMA,
    PASS_FRACTION,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    file_sha256,
)
from world_quality_review import (  # noqa: E402
    TIMING_METHOD,
    camera_rows,
    current_inputs,
    evidence_code_sha256,
    source_frame_png,
    source_timestamps,
)


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(99.0 if m < 1e-9 else 10 * np.log10(255.0 * 255.0 / m))


def grad_corr(a, b):
    """Correlation of gradient magnitude. Insensitive to exposure, sensitive to structure being in
    the same place, which is the thing a wrong room gets wrong."""

    def g(x):
        x = cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.6)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
        return np.hypot(gx, gy)

    ga, gb = g(a).ravel(), g(b).ravel()
    ga -= ga.mean()
    gb -= gb.mean()
    d = np.linalg.norm(ga) * np.linalg.norm(gb)
    return float(0.0 if d < 1e-9 else np.dot(ga, gb) / d)


def label(img, text, colour=(255, 255, 255)):
    out = cv2.copyMakeBorder(img, 30, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument(
        "--scale0", type=float, default=1.0, help="SfM-to-world scale; --fit-scale finds it"
    )
    ap.add_argument(
        "--frames",
        default=None,
        help="comma-separated sourceIndex (default: 5 spread over the clip)",
    )
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="world")
    ap.add_argument("--share", default=None, help="also copy the sheet here as <tag>-verify.png")
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="deprecated; retained for CLI compatibility, no correction step is run",
    )
    ap.add_argument("--model", default=None, help="unsupported: model review is disabled")
    ap.add_argument(
        "--pass-score", type=int, default=None, help="unsupported: scores cannot accept a world"
    )
    ap.add_argument(
        "--no-vlm", action="store_true", help="compatibility flag; diagnostics are always offline"
    )
    ap.add_argument(
        "--fit-scale",
        action="store_true",
        help="sweep scale0 and keep the value with the best mean gradient correlation. A "
        "world generated from something other than the clip has no fitted scale yet, "
        "and rendering it at the wrong one is indistinguishable from a wrong room.",
    )
    ap.add_argument("--fit-range", default="0.2,2.4,12")
    a = ap.parse_args()
    if a.model is not None or a.pass_score is not None:
        ap.error(
            "model/score acceptance is disabled; import an explicit manual static-world review"
        )
    if a.n_frames < 3 or a.width < 1:
        ap.error("at least three frames and a positive image width are required")

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        ap.error("output directory is not empty; retain existing evidence and choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    cameras = camera_rows(Path(a.cameras))
    cams = list(cameras.values())
    if a.frames:
        frames = [int(x) for x in a.frames.split(",")]
    else:
        idx = np.linspace(0, len(cams) - 1, a.n_frames).round().astype(int)
        frames = sorted(
            {cams[i]["sourceIndex"] for i in idx}
            | {
                cams[0]["sourceIndex"],
                cams[len(cams) // 2]["sourceIndex"],
                cams[-1]["sourceIndex"],
            }
        )

    if frames != sorted(set(frames)) or any(frame not in cameras for frame in frames):
        ap.error("frames must be ordered, unique recorded camera indices")
    source_times = source_timestamps(Path(a.clip))
    for frame in frames:
        if (
            frame >= len(source_times)
            or abs(cameras[frame]["time"] - (source_times[frame] - source_times[0])) > 1e-6
        ):
            ap.error(
                "camera time does not match exact decoded source PTS; fix source/camera provenance"
            )
    inputs = current_inputs(Path(a.clip), Path(a.world), Path(a.cameras), a.scale0)
    source_pngs = {frame: source_frame_png(Path(a.clip), frame) for frame in frames}
    source_images = {
        frame: cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        for frame, data in source_pngs.items()
    }
    if any(image is None for image in source_images.values()):
        ap.error("a source frame could not be decoded")

    g = read_spz(Path(a.world))

    curve = None
    if a.fit_scale:
        lo, hi, n = a.fit_range.split(",")
        grid = np.exp(np.linspace(np.log(float(lo)), np.log(float(hi)), int(n)))
        reals, ks, c2ws = [], [], []
        for c in cameras_for(a.cameras, frames):
            sw, sh = c["source_image_size"]
            sc = 480 / sw
            K = np.array(c["source_intrinsics"], np.float64) * sc
            K[2, 2] = 1.0
            r = source_images[c["sourceIndex"]]
            reals.append(cv2.resize(r, (480, int(round(sh * sc)))))
            ks.append(K)
            c2ws.append(np.array(c["camera_to_world"]))
        curve = []
        for s0 in grid:
            gc = []
            for real, K, c2w in zip(reals, ks, c2ws):
                img, _ = render(g, c2w, K, (480, real.shape[0]), float(s0))
                gc.append(
                    grad_corr(
                        real,
                        cv2.cvtColor(
                            (np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                        ),
                    )
                )
            curve.append((round(float(s0), 4), round(float(np.mean(gc)), 4)))
            print(f"  scale0 {s0:.4f}  mean gradCorr {np.mean(gc):+.4f}", flush=True)
        a.scale0 = max(curve, key=lambda t: t[1])[0]
        inputs["registrationScale"] = a.scale0
        print(f"fitted scale0 {a.scale0}")

    rows, per_frame = [], []
    for c in cameras_for(a.cameras, frames):
        fi = c["sourceIndex"]
        sw, sh = c["source_image_size"]
        s = a.width / sw
        K = np.array(c["source_intrinsics"], np.float64) * s
        K[2, 2] = 1.0
        H = int(round(sh * s))
        img, alpha = render(g, np.array(c["camera_to_world"]), K, (a.width, H), a.scale0)
        ren = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        real = source_images[fi]
        if list(real.shape[1::-1]) != c["source_image_size"]:
            ap.error("source frame dimensions do not match recorded camera intrinsics")
        real = cv2.resize(real, (a.width, H), interpolation=cv2.INTER_AREA)
        pair = np.hstack(
            [
                label(real, f"REAL  clip frame {fi}", (80, 255, 255)),
                label(ren, f"WORLD  {Path(a.world).name}  estimated pose"),
            ]
        )
        p = out / f"pair-f{fi}.png"
        source_path, render_path = out / f"source-f{fi}.png", out / f"render-f{fi}.png"
        source_path.write_bytes(source_pngs[fi])
        if not cv2.imwrite(str(p), pair) or not cv2.imwrite(str(render_path), ren):
            raise OSError("could not save comparison evidence")
        per_frame.append(
            dict(
                frame=fi,
                timeSeconds=source_times[fi] - source_times[0],
                sourcePtsSeconds=source_times[fi],
                pair={"path": p.name, "sha256": file_sha256(p)},
                sourceImage={"path": source_path.name, "sha256": file_sha256(source_path)},
                renderImage={"path": render_path.name, "sha256": file_sha256(render_path)},
                psnr=round(psnr(real, ren), 2),
                gradCorr=round(grad_corr(real, ren), 3),
                coverage=round(float((alpha > 0.5).mean()), 3),
                renderMeanLuma=round(float(cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY).mean()), 1),
                realMeanLuma=round(float(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).mean()), 1),
            )
        )
        rows.append(pair)
    if not rows:
        raise SystemExit("nothing rendered")

    sheet = np.vstack([cv2.resize(r, (rows[0].shape[1], rows[0].shape[0])) for r in rows])
    sheet_p = out / "sheet.png"
    cv2.imwrite(
        str(sheet_p),
        cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA),
    )
    if a.share:
        shared_sheet = Path(a.share) / f"{a.tag}-verify.png"
        if shared_sheet.exists():
            print(f"retaining existing shared sheet: {shared_sheet}")
        elif not cv2.imwrite(str(shared_sheet), cv2.imread(str(sheet_p))):
            raise OSError("could not save shared comparison sheet")

    report = dict(
        schema=EVIDENCE_SCHEMA,
        status="diagnostic",
        inputs=inputs,
        evidenceCodeSha256=evidence_code_sha256(),
        requestedFrames=frames,
        timingMethod=TIMING_METHOD,
        sourceTimeOriginSeconds=source_times[0],
        cameraCorrespondence="estimated; geometric alignment remains unverified",
        world=a.world,
        clip=a.clip,
        cameras=a.cameras,
        scale0=a.scale0,
        scaleCurve=curve,
        frames=per_frame,
        sheet=str(sheet_p),
    )

    if current_inputs(Path(a.clip), Path(a.world), Path(a.cameras), a.scale0) != inputs:
        raise RuntimeError("source/world/cameras changed during diagnostic capture")
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=1, allow_nan=False))
    plan = {
        "schema": PLAN_SCHEMA,
        "inputs": inputs,
        "criteria": list(CRITERIA),
        "passFraction": PASS_FRACTION,
        "evidenceReport": {"path": report_path.name, "sha256": file_sha256(report_path)},
        "samples": [
            {
                "id": f"f{row['frame']}",
                "sourceFrame": row["frame"],
                "timeSeconds": row["timeSeconds"],
                "view": "source-camera",
                "critical": True,
            }
            for row in per_frame
        ],
    }
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False))
    template = {
        "schema": RESULT_SCHEMA,
        "planSha256": file_sha256(plan_path),
        "judgeKind": "manual",
        "reviewer": "",
        "acknowledgements": {key: False for key in ACKNOWLEDGEMENTS},
        "samples": [
            {
                "id": f"f{row['frame']}",
                "pairSha256": row["pair"]["sha256"],
                "criteria": {
                    key: {"status": "unknown", "criticalFailure": False, "reason": ""}
                    for key in CRITERIA
                },
            }
            for row in per_frame
        ],
    }
    (out / "review-template.json").write_text(json.dumps(template, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "verdicts"}, indent=1))
    print("sheet ->", sheet_p)


if __name__ == "__main__":
    main()
