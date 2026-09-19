#!/usr/bin/env python3
"""Look at the clip, then write the prompt the world model is generated from.

Nothing in this pipeline ever looked at a clip before asking Marble for a world. Video mode
auto-captions the clip and generates from its own caption -- that is how a Diagon Alley clip became
a generic wizard street (share/HP-MARBLE-VERDICT.md). `disable_recaption: true` with our own
`text_prompt` replaces that caption, and a controlled experiment showed prompting genuinely governs
what gets invented where the camera never looked: naming "plain plaster, no signage" removed
invented CJK-lettered banners and framed portraits (share/FINETUNE-STATUS.md, the bare-walls run).
Every such prompt so far was hand-written per clip. This writes it, from the frames, unattended.

What the prompt is FOR, and this is the whole design: the part of the scene the camera filmed is
already pinned by the input image(s). The prompt only governs the regions Marble has to invent --
behind the camera, past the far wall, the ceiling, the space beyond a doorway. So the brief asks
for materials, architecture, lighting and what the space opens onto, and for explicit negative
constraints, which is where this project's failures live.

  world_prompt.py --clip public/clips/gym.mp4 --n 6 --out .context/prompt/gym.json \
      [--frames 0,108,216,324,432,540] [--model gpt-6-astra]

Writes {"text_prompt", "structured": {...}, "frames": [...]} plus the sampled PNGs next to it.
"""

import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

BRIEF = """These images are frames from ONE continuous shot of ONE real place, in time order. A
generative 3D world model will be given one or more of these frames and must return a world a person
can walk around inside. You are writing the text prompt it generates from.

Understand what the prompt controls. The parts of the room these frames show are already pinned by
the input image itself. Your words govern only the regions the camera NEVER filmed: behind the
camera, beyond the far wall, above the top of frame, through any opening. Those regions are where
this pipeline's failures live -- invented signage, invented furniture, invented corridors, a ceiling
that does not exist. So describe the place factually and then constrain what must not be invented.

Rules, each from a measured result on this pipeline:
 - Material, surface, lighting and time-of-day words are followed.
 - Negative constraints are followed: "no signage, no readable text, no framed pictures, plain
   plaster walls" measurably removed exactly those things on an earlier clip.
 - Spatial instructions are NOT followed. Never write "the door is on the left" or "the stairs are
   behind the camera". Do not spend words on placement.
 - Mirrors are the known hard case: the model rebuilds a reflection as more room. If you see a
   mirrored wall, say it is a flat mirror on a solid wall and that the space does not continue
   behind it.
 - Ignore any people: they are removed before generation. Never describe or ask for people.
 - No style, mood or camera words ("cinematic", "8k", "beautiful"). This is a description, not art
   direction.

Answer with JSON and nothing else:
{"space": "<what kind of room or place, one clause>",
 "architecture": "<walls, ceiling, floor plan, columns, openings, their construction>",
 "materials": "<floor, wall, ceiling and fitting materials and colours, concretely>",
 "lighting": "<sources, colour, direction, and whether there is daylight; time of day>",
 "opens_onto": "<what is beyond the openings the camera can see into, or 'nothing visible'>",
 "contents": "<the furniture and equipment actually present, concretely>",
 "mirrors": "<mirrored or glazed surfaces present, or 'none'>",
 "negative_constraints": ["<each thing that must NOT be invented, as a short clause>"],
 "text_prompt": "<the final prompt, under 100 words, factual, ending with the negative constraints
                  as one sentence beginning 'Do not add'>"}
"""


def sample(clip, frames, n, out_dir):
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not frames:
        frames = np.linspace(0, max(total - 1, 0), n).round().astype(int).tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        s = 1024 / max(w, h)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        p = out_dir / f"frame_{int(f):04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    cap.release()
    return frames, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument(
        "--frames",
        default=None,
        help="comma-separated frame numbers (default: --n spread over the clip)",
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6-astra")
    a = ap.parse_args()

    out = Path(a.out)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else None
    frames, paths = sample(Path(a.clip), frames, a.n, out.parent / (out.stem + "-frames"))
    if not paths:
        raise SystemExit(f"no frames read from {a.clip}")
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY unset; source ~/.openai-env")
    from vlm_judge import ask_images  # noqa: E402

    rec = ask_images(a.model, paths, BRIEF, detail="high")
    rec["clip"] = a.clip
    rec["frames"] = frames
    rec["framePaths"] = paths
    rec["model"] = a.model
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
