#!/usr/bin/env python3
"""Ask a vision model which shots of a cut-up clip show the same physical place, once.

scripts/shot_cuts.py splits a clip at its cuts and reports `shots` with `index`, `start`, `end`,
`seconds` and `tooShort`. It says nothing about whether shot 2 and shot 7 are the same room filmed
from two positions, and every later stage assumes one continuous place: fusing two places into one
world is exactly the failure shot_cuts.py exists to catch.

This judge tiles two frames from each long-enough shot, numbers them, and asks for a grouping by
physical place plus each shot's main subjects, in one request.

THE GROUPING IS A PROPOSAL, NOT A RESULT. A vision model saying two shots are the same pitch is a
hypothesis about two pictures; only geometric registration -- a solve that actually puts both
shots' cameras in one reconstruction -- can confirm that they are the same place, and only that
may be acted on. Nothing here measures a viewpoint, a direction, a distance or an overlap, and a
group must never be read as a claim that two shots can be reconstructed together.

Shots marked `tooShort` are never sent. At most 24 shots are judged in one request; if the report
has more, the 24 longest are judged and the rest are recorded as `unjudged`.

  uv run --locked python scripts/judge_shots.py \
      --report .context/run/<name>/cuts.json [--video <clip>] \
      --out .context/run/<name>/shot-groups.json [--model gpt-6-astra]

One paid request per `--out`. `<out>.receipt.json` and `<out>.response.raw` carry the receipt
lifecycle in scripts/vlm_once.py: an interrupted run finishes from the bytes already paid for, and
a failed or unknown receipt refuses instead of submitting a second request.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import vlm_once

MAX_JUDGED_SHOTS = 24
FRAME_FRACTIONS = (0.25, 0.75)
TILE_EDGE = 512

PROPOSAL = (
    "A vision model's proposal from two frames per shot. It is not a registration: only a solve "
    "that places both shots' cameras in one reconstruction may confirm that they share a place."
)

BRIEF = """Each image below is ONE shot from a single video. It shows two frames of that shot side
by side -- a quarter and three quarters of the way through it -- and the shot's number is printed
on it.

Group the shots by PHYSICAL PLACE. Two shots belong in the same group when they were filmed in the
same place on the ground: the same room, the same pitch, the same street. The camera may be
somewhere else in that place, pointing another way, nearer or further; that does not make it a
different place. Two similar-looking but different places are different groups. A shot that shares
its place with nothing else is a group of its own.

Judge only from what is visible. Do not infer a floor plan, a compass direction, a distance, a
viewpoint, or that two places adjoin one another. Name each place in one clause.

Every one of these shot numbers must appear in exactly one group:
  {ids}

Then name each shot's main subjects -- who or what the shot is of -- in a few words.

Answer with JSON and nothing else:
{{"groups": [{{"id": <0, 1, 2, ...>,
               "place": "<one clause naming the place, from what is visible>",
               "shots": [<the shot numbers filmed in that place>]}}],
  "shots": [{{"index": <shot number>, "subjects": "<a few words>"}}]}}
"""


def judged_shots(document, limit=MAX_JUDGED_SHOTS):
    """The long-enough shots this request will carry, and the ones it will not.

    `tooShort` shots are not reconstructable and are never sent. Past `limit`, the longest shots
    are kept because they are the ones a run would actually choose between.
    """
    shots = [shot for shot in document.get("shots") or [] if not shot.get("tooShort")]
    shots.sort(key=lambda shot: int(shot["index"]))
    if len(shots) <= limit:
        return shots, []
    ranked = sorted(shots, key=lambda shot: (-float(shot["seconds"]), int(shot["index"])))
    keep = {int(shot["index"]) for shot in ranked[:limit]}
    dropped = [
        {
            "index": int(shot["index"]),
            "seconds": float(shot["seconds"]),
            "reason": f"not judged: only the {limit} longest shots fit in one request",
        }
        for shot in shots
        if int(shot["index"]) not in keep
    ]
    return [shot for shot in shots if int(shot["index"]) in keep], dropped


def grab(video, seconds):
    """One frame at `seconds`, decoded from ffmpeg's PNG output.

    `-ss` before `-i` seeks by keyframe and then decodes forward, so the frame is close to the
    asked-for time rather than exactly on it. Two frames a quarter and three quarters into a shot
    only have to show the place; nothing here depends on the timestamp.
    """
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(seconds, 0.0):.4f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not raw:
        raise ValueError(f"No frame could be read from {video} at {seconds:.3f} s")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"The frame read from {video} at {seconds:.3f} s could not be decoded")
    return image


def fit(image, edge=TILE_EDGE):
    height, width = image.shape[:2]
    scale = min(1.0, edge / max(width, height))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )


def tile(images, index, path):
    """The shot's frames side by side, with its number drawn large enough to read."""
    images = [fit(image) for image in images]
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        pad = height - image.shape[0]
        if pad:
            image = cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(image)
    canvas = np.hstack(padded)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.0, max(canvas.shape[:2]) / 700)
    thickness = max(2, round(max(canvas.shape[:2]) / 320))
    text = f"shot {index}"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(canvas, (0, 0), (text_w + 16, text_h + baseline + 16), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, text_h + 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return str(path)


def build_tiles(video, shots, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for shot in shots:
        index = int(shot["index"])
        start, seconds = float(shot["start"]), float(shot["seconds"])
        frames = [grab(video, start + fraction * seconds) for fraction in FRAME_FRACTIONS]
        built.append(tile(frames, index, out_dir / f"shot_{index:03d}.png"))
    return built


def reuse_tiles(shots, out_dir):
    """The tiles already beside `out`, when every expected one is still there.

    A run that has a receipt must present the bytes it paid to have looked at, so the tiles are
    not redrawn under it; a missing file falls back to rebuilding them.
    """
    out_dir = Path(out_dir)
    built = []
    for shot in shots:
        path = out_dir / f"shot_{int(shot['index']):03d}.png"
        if not path.is_file() or path.is_symlink():
            return None
        built.append(str(path))
    return built


def check_groups(record, indices):
    """Every judged shot in exactly one group, each group named and identified."""
    groups = record.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("the answer proposes no groups")
    seen_ids, placed, out = set(), {}, []
    for group in groups:
        if not isinstance(group, dict):
            raise TypeError("a group is not an object")
        group_id = group.get("id")
        if not isinstance(group_id, int) or isinstance(group_id, bool):
            raise TypeError("a group has no integer id")
        if group_id in seen_ids:
            raise ValueError(f"group {group_id} appears twice")
        seen_ids.add(group_id)
        place = group.get("place")
        if not isinstance(place, str) or not place.strip():
            raise ValueError(f"group {group_id} names no place")
        shots = group.get("shots")
        if not isinstance(shots, list) or not shots:
            raise ValueError(f"group {group_id} holds no shots")
        members = []
        for shot in shots:
            if not isinstance(shot, int) or isinstance(shot, bool):
                raise TypeError(f"group {group_id} holds a non-integer shot")
            if shot not in indices:
                raise ValueError(f"shot {shot} was not judged but is in group {group_id}")
            if shot in placed:
                raise ValueError(f"shot {shot} is in groups {placed[shot]} and {group_id}")
            placed[shot] = group_id
            members.append(shot)
        out.append({"id": group_id, "place": place.strip(), "shots": sorted(members)})
    missing = sorted(index for index in indices if index not in placed)
    if missing:
        raise ValueError(f"shots {missing} were left out of every group")
    return sorted(out, key=lambda group: group["id"])


def check_subjects(record, indices):
    """Per-shot subjects, kept where they are usable and recorded as absent where they are not.

    The grouping is the contract and is checked strictly. Subjects are a description beside it, so
    a missing one is reported rather than turned into a failed receipt over a billed response.
    """
    entries = record.get("shots")
    subjects = dict.fromkeys(sorted(indices))
    if entries is None:
        return subjects, sorted(indices)
    if not isinstance(entries, list):
        raise TypeError("shots is not a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("a shot entry is not an object")
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in indices:
            raise ValueError(f"a shot entry names {index!r}, which was not judged")
        text = entry.get("subjects")
        if isinstance(text, str) and text.strip():
            subjects[index] = text.strip()
    return subjects, [index for index, text in subjects.items() if text is None]


def judge_shots(
    report_path,
    out,
    *,
    video=None,
    model=vlm_once.DEFAULT_MODEL,
    limit=MAX_JUDGED_SHOTS,
    urlopen=None,
    report=None,
):
    """Group the shots in one paid request, or reuse what an earlier run already paid for."""
    report_path = Path(report_path)
    out = Path(out)
    document = json.loads(report_path.read_text())
    video = Path(video) if video else Path(document["video"])
    shots, unjudged = judged_shots(document, limit)
    if not shots:
        raise ValueError(f"{report_path} has no shot long enough to judge")

    frames_dir = out.parent / (out.stem + "-frames")
    built = None
    if vlm_once.receipt_status(out) is not None:
        built = reuse_tiles(shots, frames_dir)
    if built is None:
        built = build_tiles(video, shots, frames_dir)

    indices = [int(shot["index"]) for shot in shots]
    brief = BRIEF.format(ids=", ".join(str(index) for index in indices))

    def validate(record):
        groups = check_groups(record, set(indices))
        subjects, missing = check_subjects(record, set(indices))
        return {
            "judge": "shots",
            "model": model,
            "proposal": PROPOSAL,
            "video": str(video),
            "report": str(report_path),
            "groups": groups,
            "subjects": [
                {"index": index, "subjects": subjects[index]} for index in sorted(subjects)
            ],
            "subjectsMissing": sorted(missing),
            "judged": indices,
            "unjudged": unjudged,
            "framePaths": list(built),
            "frameFractions": list(FRAME_FRACTIONS),
        }

    return vlm_once.request_once(
        out,
        brief=brief,
        images=built,
        model=model,
        validate=validate,
        identity_extra={
            "video": vlm_once.file_identity(video),
            "report": vlm_once.file_identity(report_path),
            "judged": indices,
            "shotBounds": [
                [float(shot["start"]), float(shot["end"])]
                for shot in sorted(shots, key=lambda s: int(s["index"]))
            ],
            "frameFractions": list(FRAME_FRACTIONS),
        },
        urlopen=urlopen,
        label="shot judgement",
        report=report,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", required=True, help="a scripts/shot_cuts.py --json cut report")
    ap.add_argument("--video", help="override the video named in the report")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=vlm_once.DEFAULT_MODEL)
    ap.add_argument("--max-shots", type=int, default=MAX_JUDGED_SHOTS)
    a = ap.parse_args()
    try:
        record = judge_shots(
            a.report,
            Path(a.out),
            video=a.video,
            model=a.model,
            limit=a.max_shots,
            report=lambda line: print(line, file=sys.stderr),
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
