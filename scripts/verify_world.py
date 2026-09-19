#!/usr/bin/env python3
"""Does the world we got back agree with the footage it was made from?

The world analogue of scripts/verify_avatar.py. That script caught an avatar rendered as the wrong
gender the morning it was written; nothing equivalent existed for worlds, which is why "Marble built
the wrong gym, with mirrors on two walls" was only ever noticed by a human looking at a figure.

Three steps, all unattended:
  1. render the world from the clip's OWN recorded camera poses (scripts/render_world_poses.py),
  2. stand each render next to the real frame and measure it (PSNR, gradient-correlation, coverage),
  3. ask the VLM (worker/stages/vlm_judge.py, OPENAI_API_KEY) what specifically differs, on a
     fixed rubric -- layout, objects, invented content, materials, lighting -- and score it.
  4. if the score is bad, ask for a CORRECTED text_prompt that names the errors, and say whether a
     regeneration is worth 1600 credits.

The photometric numbers cannot decide this on their own: a world that is the wrong room but the
right brightness scores better than a right room that is soft, which is exactly the trap
NEW-CLIPS-STATUS hit when a -0.22 dB PSNR change hid a large artefact win. They are reported as
supporting evidence; the verdict is the VLM's, on the pictures.

  verify_world.py --world public/marble-gym-clean2.spz --cameras public/worlds/gym-4d/cameras.json \
      --clip public/clips/gym.mp4 --scale0 1.0020 --tag gym-raw --out .context/verify/gym-raw \
      [--frames 0,108,216] [--prompt-file .context/prompt/gym.json] [--no-vlm]

A world that was not generated from this clip's own frames has no fitted scale0 yet, and rendering
it at the wrong one looks exactly like a wrong room. `--fit-scale` sweeps scale0 and keeps the best
mean gradient correlation, and the curve it prints is itself a registration diagnostic: on the
shipped gym world it is flat (0.06-0.11 everywhere), which says the world does not line up with the
footage at ANY scale.

Writes <out>/pair-f<idx>.png, <out>/sheet.png and <out>/report.json. Exit code 0 always; the
verdict is `regenerate` in report.json.
"""

import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "worker" / "stages"))
from render_world_poses import cameras_for, read_spz, render  # noqa: E402

RUBRIC = """You are the fidelity gate of a pipeline that turns one video clip of a real place into a
navigable 3D world. A generative world model was given that clip (with the person inpainted out) and
returned a world. Your job is to decide whether it returned THAT place or merely A place.

The image is one pair. LEFT is a real frame from the clip. RIGHT is the generated world rendered
from the very same recorded camera pose, same field of view, same instant. They should show the same
room from the same spot.

Ignore, and never report as an error:
 - the person (or people) on the left: they were deliberately removed before generation,
 - softness, blur, noise, missing fine detail, low resolution on the right,
 - small differences of exposure or white balance.

Report ONLY differences of content and geometry. Answer with JSON and nothing else:
{"same_place": true|false,
 "score": <0-100, how much of THIS room a viewer would recognise; 0 = a different room entirely,
           50 = the right kind of room with the wrong layout, 100 = the same room>,
 "layout": "<one sentence: are the walls, openings and the direction the camera faces the same?>",
 "wrong_objects": ["<object present on the left and absent or replaced on the right>"],
 "invented": ["<object, surface or feature present on the right that is NOT in the real frame>"],
 "materials": "<floor, wall and ceiling materials: same or different, and how>",
 "lighting": "<light sources, direction and time of day: same or different, and how>",
 "coverage": "empty|partial|full",
 "worst_error": "<the single biggest thing that is wrong, in under 15 words>"}
"coverage" describes the RIGHT image only: "empty" if it is mostly blank, smeared floor or void
rather than a room.
"""

CORRECT = """You are writing the text prompt for ONE regeneration of a 3D world from a video clip.
The previous attempt was verified against the real footage frame by frame and failed. Below are the
prompt that was used (if any) and the per-frame verdicts.

Write a prompt that fixes the specific, named failures. Rules that come from measured experiments on
this pipeline, and that you must respect:
 - The visible part of the scene is already pinned by the input image; the prompt only governs what
   gets INVENTED in regions the camera never filmed. Spend the words there.
 - Negative constraints work: naming what must not appear ("no signage, no text, no framed pictures,
   plain plaster walls") measurably removed invented banners and portraits on an earlier clip.
 - Material, lighting and time-of-day words work.
 - Spatial instructions ("the stairs are on the left", "the door is 3 m behind the camera") are
   followed poorly or not at all. Do not waste the prompt on them.
 - Mirrors are the hardest case: a generative model reconstructs the reflection as more room. If the
   verdicts mention mirrors, say explicitly that the mirrored wall is a flat mirror on a solid wall
   and that the space does not continue behind it.
 - Keep it under 90 words, factual, no style or mood words.

Answer with JSON and nothing else:
{"regenerate": true|false,
 "why": "<under 25 words: is a regeneration likely to fix these errors, or is this a limit of the
          model that a prompt cannot reach?>",
 "text_prompt": "<the corrected prompt>",
 "negative_constraints": ["<each must-not-invent clause you put in the prompt>"]}
"""


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
        help="JSON from scripts/world_prompt.py, for the correction step",
    )
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument(
        "--pass-score", type=int, default=55, help="median score at or above which the world passes"
    )
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument(
        "--fit-scale",
        action="store_true",
        help="sweep scale0 and keep the value with the best mean gradient correlation. A "
        "world generated from something other than the clip has no fitted scale yet, "
        "and rendering it at the wrong one is indistinguishable from a wrong room.",
    )
    ap.add_argument("--fit-range", default="0.2,2.4,12")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = json.loads(Path(a.cameras).read_text())["cameras"]
    if a.frames:
        frames = [int(x) for x in a.frames.split(",")]
    else:
        idx = np.linspace(0, len(cams) - 1, a.n_frames).round().astype(int)
        frames = [cams[i]["sourceIndex"] for i in idx]

    g = read_spz(Path(a.world))
    cap = cv2.VideoCapture(a.clip)

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
            cap.set(cv2.CAP_PROP_POS_FRAMES, c["sourceIndex"])
            ok, r = cap.read()
            if not ok:
                continue
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, real = cap.read()
        if not ok:
            print(f"cannot read frame {fi}", file=sys.stderr)
            continue
        real = cv2.resize(real, (a.width, H), interpolation=cv2.INTER_AREA)
        pair = np.hstack(
            [
                label(real, f"REAL  clip frame {fi}", (80, 255, 255)),
                label(ren, f"WORLD  {Path(a.world).name}  same pose"),
            ]
        )
        p = out / f"pair-f{fi}.png"
        cv2.imwrite(str(p), pair)
        per_frame.append(
            dict(
                frame=fi,
                pair=str(p),
                psnr=round(psnr(real, ren), 2),
                gradCorr=round(grad_corr(real, ren), 3),
                coverage=round(float((alpha > 0.5).mean()), 3),
                renderMeanLuma=round(float(cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY).mean()), 1),
                realMeanLuma=round(float(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).mean()), 1),
            )
        )
        rows.append(pair)
    cap.release()
    if not rows:
        raise SystemExit("nothing rendered")

    sheet = np.vstack([cv2.resize(r, (rows[0].shape[1], rows[0].shape[0])) for r in rows])
    sheet_p = out / "sheet.png"
    cv2.imwrite(
        str(sheet_p),
        cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA),
    )
    if a.share:
        cv2.imwrite(str(Path(a.share) / f"{a.tag}-verify.png"), cv2.imread(str(sheet_p)))

    report = dict(
        world=a.world,
        clip=a.clip,
        cameras=a.cameras,
        scale0=a.scale0,
        scaleCurve=curve,
        frames=per_frame,
        sheet=str(sheet_p),
    )

    if not a.no_vlm:
        if "OPENAI_API_KEY" not in os.environ:
            report["vlm"] = {"error": "OPENAI_API_KEY unset; source ~/.openai-env"}
        else:
            from vlm_judge import ask_images  # noqa: E402

            verdicts = []
            for f in per_frame:
                v = ask_images(a.model, [f["pair"]], RUBRIC, detail="high")
                v["frame"] = f["frame"]
                verdicts.append(v)
                print(json.dumps(v), flush=True)
            scores = [v["score"] for v in verdicts if isinstance(v.get("score"), (int, float))]
            median = float(np.median(scores)) if scores else None
            report["verdicts"] = verdicts
            report["medianScore"] = median
            report["pass"] = bool(median is not None and median >= a.pass_score)
            prior = ""
            if a.prompt_file and Path(a.prompt_file).exists():
                prior = json.loads(Path(a.prompt_file).read_text()).get("text_prompt", "")
            if median is not None and median < a.pass_score:
                msg = (
                    CORRECT
                    + "\n\nPROMPT USED LAST TIME:\n"
                    + (prior or "(none: the model captioned the clip itself)")
                    + "\n\nPER-FRAME VERDICTS:\n"
                    + json.dumps(verdicts, indent=1)
                )
                report["correction"] = ask_images(
                    a.model, [f["pair"] for f in per_frame], msg, detail="low"
                )
            report["regenerate"] = bool(report.get("correction", {}).get("regenerate", False))

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "verdicts"}, indent=1))
    print("sheet ->", sheet_p)


if __name__ == "__main__":
    main()
