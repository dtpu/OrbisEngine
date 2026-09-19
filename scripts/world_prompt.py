#!/usr/bin/env python3
"""Write a source-grounded description to accompany Marble video or image inputs.

Video remains the default visual input. A description can state observed materials and constrain
unsupported additions, but does not pin visible geometry or prove that hallucination is reduced.
Unknown, occluded and unobserved regions must remain labeled as such. Review the sampled evidence
and the full cleaned clip before generation; a few prompt frames do not prove temporal coverage.

  world_prompt.py --clip public/clips/gym.mp4 --n 6 --out .context/prompt/gym.json

Writes {"text_prompt", "structured": {...}, "frames": [...]} plus sampled PNGs next to it.
"""

import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

BRIEF = """These images sample ONE continuous shot of a real place in time order. A generative
3D world model will receive the cleaned video, or explicitly selected still images. Write a factual
supporting description. The description and visual input do not guarantee correct geometry.

Describe only structure, fixtures and materials supported by the supplied images. Do not complete a
floor plan from assumptions or invent what lies behind the camera, beyond an occlusion, or through an
opening. Label ambiguous or unseen details as unknown; they are not evidence that a surface is absent.
Preserve observed details. Negative constraints must not ask to remove real recorded features.

Rules:
 - Describe visible materials, surfaces, fixtures and lighting without guessing hidden structure.
 - Describe what is actually visible through openings; otherwise say 'nothing visible'.
 - Do not invent additional rooms, corridors, furniture, signage, windows or doors.
 - If a surface is visibly a mirror, describe it as a flat mirror on a solid wall, not another room.
   If reflection versus opening is ambiguous, say so rather than guessing.
 - People are removed before world generation; do not request people or their body parts.
 - Do not add style, mood or camera adjectives such as 'cinematic', '8k' or 'beautiful'.
 - If sampled views conflict because an object moved or a door opened, record that uncertainty;
   do not blend incompatible fixture states into a confident description.

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
