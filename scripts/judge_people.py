#!/usr/bin/env python3
"""Ask a vision model WHO the tracked people in a clip are, once, and record it as an opinion.

The `tracks` stage (worker/stages/track_people.py) links every MultiHMR detection in a clip into
identity-stable tracks and ranks them by how many samples they survived. scripts/run_clip.py then
builds `person_prep_NN` slots for the first `--people N` of those ranks. Rank is presence, not
importance: a poster, a passer-by or half of a crowd can outrank the person the clip is about.

This judge looks at N annotated frames -- every track's box with its track id drawn large -- and
labels each track id `main`, `secondary`, `background`, `crowd` or `not_a_person`, with a clause of
reasoning, and lists anybody clearly visible who is obviously main or secondary and has no box at
all. That is the whole job: WHO. It states nothing about geometry, depth, height or placement, and
must never be read as measuring any of them.

The result is a MODEL OPINION about a handful of frames, recorded as such. It is not evidence that
a track is correct, that an unlabelled person is absent, or that the selection is right; a human
looks at the annotated frames before anything acts on it. `selected` orders the main/secondary
tracks by measured on-screen prominence (total box area over the track's life), which IS measured
from the tracker's own boxes -- the ordering is arithmetic, the labels are opinion.

  uv run --locked python scripts/judge_people.py \
      --tracks .context/run/<name>/tracks --clip <the clip tracking ran on> \
      --out .context/run/<name>/people-judge.json [--n 6] [--model gpt-6-astra]

One paid request per `--out`. `<out>.receipt.json` and `<out>.response.raw` carry the receipt
lifecycle in scripts/vlm_once.py: an interrupted run finishes from the bytes already paid for, and
a failed or unknown receipt refuses instead of submitting a second request.
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import vlm_once

LABELS = ("main", "secondary", "background", "crowd", "not_a_person")
SELECTED_LABELS = ("main", "secondary")
MAX_FRAME_EDGE = 1024
DEFAULT_FRAMES = 6

OPINION = (
    "A vision model's opinion about the annotated frames listed here, not a measurement and not "
    "evidence that a track is correct or that an unlabelled person is absent. Review the frames."
)
CAP_NOTE = (
    "scripts/run_clip.py's --people N takes track ranks 0..N-1, which are ordered by how many "
    "samples each track survived. `selected` is an alternative order for a human to apply; this "
    "file changes nothing on its own."
)

BRIEF = """These {count} images are frames from ONE video of a real place, in time order.

Every box drawn on them is an automatic tracker's box for one person, and the large number beside
a box is that person's track id. The same id means the same tracked person in every frame.

Decide WHO each tracked person is to somebody watching this clip. Judge only from what is visible.
Do not describe geometry, depth, distance, size or position in the world; do not guess what is
outside a frame or behind anybody.

Label every track id exactly once, with exactly one of these:
  main          the clip is about this person; they carry its action
  secondary     a person who matters to the action but is not its subject
  background    a real person who is incidental: passing through, standing about, unrelated
  crowd         one of an indistinct mass of people rather than a distinguishable individual
  not_a_person  the box is not on a person: a poster, a screen, a reflection, a statue, a
                mannequin, part of somebody else, or nothing at all

Then list any person who is clearly visible, who would be main or secondary, and who has NO box on
them in any of these images. Give the frame number printed on the image you saw them in and where
in that frame they are, in a few words. If everybody who matters already has a box, return [].

Track ids that appear in these images: {ids}
Frame numbers printed on these images: {frames}

Answer with JSON and nothing else:
{{"tracks": [{{"id": <track id>,
               "label": "<one of main, secondary, background, crowd, not_a_person>",
               "reason": "<one clause, from what is visible>"}}],
  "untracked": [{{"frame": <a frame number from the list above>,
                  "where": "<a few words: where in that frame>",
                  "why": "<one clause: why this person is main or secondary>"}}]}}
"""

PALETTE = [
    (60, 60, 255),
    (255, 160, 60),
    (80, 220, 80),
    (40, 200, 255),
    (255, 80, 255),
    (255, 255, 60),
    (150, 120, 255),
    (60, 255, 200),
]


def load_tracks(tracks_dir):
    """The `tracks` stage output: tracks.json plus each track's per-sample motion.json.

    Format from worker/stages/track_people.py: the document at lines 526-560 and the per-track
    `report` entries at lines 490-511; `track_<rank:02d>/motion.json` holds `frames`, each record
    built at lines 420-447 with `sample`, `sourceIndex`, `box` and `maskBox` in source pixels.
    """
    tracks_dir = Path(tracks_dir)
    document = json.loads((tracks_dir / "tracks.json").read_text())
    tracks = []
    for entry in document["tracks"]:
        index = int(entry["track"])
        motion = tracks_dir / f"track_{index:02d}" / "motion.json"
        if motion.is_file():
            records = json.loads(motion.read_text())["frames"]
            boxes = {int(record["sample"]): record for record in records}
        else:
            # A packaged run may keep tracks.json without the per-track folders. The ranked
            # candidate samples carry the same box fields and are enough to annotate frames.
            boxes = {int(c["sample"]): c for c in entry.get("candidateSamples") or []}
        tracks.append({"id": index, "entry": entry, "boxes": boxes})
    return document, tracks


def box_of(record, box_source):
    box = record.get(box_source) or record.get("box")
    return [float(value) for value in box] if box else None


def prominence(tracks, document, box_source):
    """Total on-screen box area over a track's life, as a fraction of one whole frame.

    Measured from the tracker's own boxes, so it is arithmetic rather than opinion. A track seen
    briefly but hugely can outrank one seen faintly throughout; that is the intent.
    """
    area = float(document["width"]) * float(document["height"])
    out = {}
    for track in tracks:
        total = 0.0
        for record in track["boxes"].values():
            box = box_of(record, box_source)
            if not box:
                continue
            total += max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        out[track["id"]] = round(total / area, 4) if area > 0 else 0.0
    return out


def choose_samples(tracks, sample_count, n):
    """`n` samples spread over the clip, each the busiest in its own stretch, plus any track's
    own best sample when `n` frames still show it nowhere.

    A track the model never sees cannot be labelled, and the answer is required to label every
    track, so coverage wins over keeping the frame count at exactly `n`.
    """
    if sample_count <= 0:
        raise ValueError("The tracking document records no samples")
    presence = {}
    for track in tracks:
        for sample in track["boxes"]:
            presence.setdefault(int(sample), set()).add(track["id"])

    edges = np.linspace(0, sample_count, min(n, sample_count) + 1).round().astype(int)
    chosen = []
    for lo, hi in pairwise(edges):
        span = range(int(lo), max(int(hi), int(lo) + 1))
        centre = (span.start + span.stop - 1) / 2
        pick = max(span, key=lambda s: (len(presence.get(s, ())), -abs(s - centre), -s))
        if pick not in chosen:
            chosen.append(pick)

    covered = set().union(*(presence.get(s, set()) for s in chosen)) if chosen else set()
    extra = []
    for track in tracks:
        if track["id"] in covered or not track["boxes"]:
            continue
        best = max(
            track["boxes"],
            key=lambda s, _t=track: _area(box_of(_t["boxes"][s], "maskBox")),
        )
        if int(best) not in chosen and int(best) not in extra:
            extra.append(int(best))
        covered.add(track["id"])
    return sorted(chosen + extra), sorted(extra)


def _area(box):
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def read_frames(clip, indices):
    """Decode the requested source-frame ordinals in stream order.

    Seeking with CAP_PROP_POS_FRAMES lands on keyframes in many containers, so frames are grabbed
    sequentially and only the wanted ordinals retrieved -- the same rule the other stages follow.
    """
    wanted = sorted({int(index) for index in indices})
    images = {}
    cap = cv2.VideoCapture(str(clip))
    try:
        remaining = iter(wanted)
        target = next(remaining, None)
        ordinal = 0
        while target is not None:
            if not cap.grab():
                break
            if ordinal == target:
                ok, image = cap.retrieve()
                if ok and image is not None:
                    images[ordinal] = image
                target = next(remaining, None)
            ordinal += 1
    finally:
        cap.release()
    missing = [index for index in wanted if index not in images]
    if missing:
        raise ValueError(f"Frames {missing} could not be decoded from {clip}")
    return images


def annotate(image, boxes, source_index, sample, path):
    """One frame with each present track's box and its id drawn large enough to read."""
    height, width = image.shape[:2]
    scale = min(1.0, MAX_FRAME_EDGE / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    edge = max(image.shape[:2])
    thickness = max(2, round(edge / 320))
    font_scale = max(0.9, edge / 700)
    for track_id, box in sorted(boxes.items()):
        colour = PALETTE[track_id % len(PALETTE)]
        x0, y0, x1, y1 = (round(value * scale) for value in box)
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, thickness)
        _label(image, f"{track_id}", (x0, y0), colour, font_scale * 1.4, thickness)
    _label(
        image,
        f"frame {source_index}",
        (0, 0),
        (255, 255, 255),
        font_scale,
        thickness,
        anchor="below",
    )
    cv2.imwrite(str(path), image)
    return {"path": str(path), "sourceIndex": int(source_index), "sample": int(sample)}


def _label(image, text, corner, colour, font_scale, thickness, anchor="above"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = min(max(corner[0], 0), max(image.shape[1] - text_w - 6, 0))
    if anchor == "above":
        y = corner[1] - 6 if corner[1] - text_h - 10 >= 0 else corner[1] + text_h + 10
    else:
        y = text_h + 10
    top = max(0, y - text_h - baseline)
    cv2.rectangle(image, (x, top), (x + text_w + 8, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 4, y), font, font_scale, colour, thickness, cv2.LINE_AA)


def build_frames(clip, document, tracks, samples, out_dir, box_source):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_indices = [int(index) for index in document["sourceIndices"]]
    wanted = {sample: source_indices[sample] for sample in samples}
    images = read_frames(clip, wanted.values())
    built = []
    for sample in samples:
        source_index = wanted[sample]
        boxes = {}
        for track in tracks:
            record = track["boxes"].get(sample)
            box = box_of(record, box_source) if record else None
            if box:
                boxes[track["id"]] = box
        path = out_dir / f"people_{source_index:06d}.png"
        built.append(annotate(images[source_index].copy(), boxes, source_index, sample, path))
    return built


def reuse_frames(document, samples, out_dir):
    """The frame files already beside `out`, when every expected one is still there.

    A run that has a receipt must present the bytes it paid to have looked at, so the frames are
    not redrawn under it; a missing file falls back to rebuilding them.
    """
    out_dir = Path(out_dir)
    source_indices = [int(index) for index in document["sourceIndices"]]
    built = []
    for sample in samples:
        source_index = source_indices[sample]
        path = out_dir / f"people_{source_index:06d}.png"
        if not path.is_file() or path.is_symlink():
            return None
        built.append({"path": str(path), "sourceIndex": source_index, "sample": int(sample)})
    return built


def check_labels(record, ids):
    """Every track id, exactly once, with a label from the enum and a reason."""
    entries = record.get("tracks")
    if not isinstance(entries, list) or not entries:
        raise ValueError("the answer lists no tracks")
    seen = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("a track entry is not an object")
        if not isinstance(entry.get("id"), int) or isinstance(entry["id"], bool):
            raise TypeError("a track entry has no integer id")
        track_id = entry["id"]
        if track_id not in ids:
            raise ValueError(f"track {track_id} was not in the images")
        if track_id in seen:
            raise ValueError(f"track {track_id} was labelled twice")
        if entry.get("label") not in LABELS:
            raise ValueError(f"track {track_id} has label {entry.get('label')!r}")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"track {track_id} has no reason")
        seen[track_id] = {"label": entry["label"], "reason": reason.strip()}
    missing = [track_id for track_id in ids if track_id not in seen]
    if missing:
        raise ValueError(f"tracks {missing} were not labelled")
    return seen


def check_untracked(record, frames):
    entries = record.get("untracked")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise TypeError("untracked is not a list")
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("an untracked entry is not an object")
        frame = entry.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise TypeError("an untracked entry has no integer frame")
        where = entry.get("where")
        if not isinstance(where, str) or not where.strip():
            raise ValueError("an untracked entry says nothing about where")
        why = entry.get("why")
        out.append(
            {
                "frame": frame,
                "frameShown": frame in frames,
                "where": where.strip(),
                "why": why.strip() if isinstance(why, str) and why.strip() else None,
            }
        )
    return out


def judge_people(
    tracks_dir,
    clip,
    out,
    *,
    n=DEFAULT_FRAMES,
    model=vlm_once.DEFAULT_MODEL,
    allow_clip_mismatch=False,
    box_source="maskBox",
    urlopen=None,
    report=None,
):
    """Label every track in one paid request, or reuse what an earlier run already paid for."""
    tracks_dir = Path(tracks_dir)
    clip = Path(clip)
    out = Path(out)
    document, tracks = load_tracks(tracks_dir)
    if not tracks:
        raise ValueError(f"{tracks_dir}/tracks.json records no tracks to judge")

    clip_identity = vlm_once.file_identity(clip)
    recorded = document.get("sourceSha256")
    if recorded and clip_identity["sha256"] != recorded and not allow_clip_mismatch:
        raise ValueError(
            "The boxes in tracks.json index the video tracking ran on, and this clip is a "
            "different file. Pass that video, or --allow-clip-mismatch to draw them anyway."
        )

    samples, extra = choose_samples(tracks, int(document["samples"]), n)
    frames_dir = out.parent / (out.stem + "-frames")
    built = None
    if vlm_once.receipt_status(out) is not None:
        built = reuse_frames(document, samples, frames_dir)
    if built is None:
        built = build_frames(clip, document, tracks, samples, frames_dir, box_source)

    ids = [track["id"] for track in tracks]
    shown = [frame["sourceIndex"] for frame in built]
    brief = BRIEF.format(
        count=len(built),
        ids=", ".join(str(track_id) for track_id in ids),
        frames=", ".join(str(index) for index in shown),
    )
    areas = prominence(tracks, document, box_source)

    def validate(record):
        labels = check_labels(record, set(ids))
        untracked = check_untracked(record, set(shown))
        selected = sorted(
            (track_id for track_id in ids if labels[track_id]["label"] in SELECTED_LABELS),
            key=lambda track_id: (-areas[track_id], track_id),
        )
        return {
            "judge": "people",
            "model": model,
            "opinion": OPINION,
            "clip": str(clip),
            "tracksDir": str(tracks_dir),
            "boxSource": box_source,
            "trackCount": len(ids),
            "labels": [
                {
                    "id": track_id,
                    "label": labels[track_id]["label"],
                    "reason": labels[track_id]["reason"],
                    "boxAreaOverTime": areas[track_id],
                    "samples": len(next(t for t in tracks if t["id"] == track_id)["boxes"]),
                }
                for track_id in ids
            ],
            "selected": selected,
            "unresolved": [
                {
                    "id": track_id,
                    "label": labels[track_id]["label"],
                    "reason": labels[track_id]["reason"],
                }
                for track_id in ids
                if track_id not in selected
            ],
            "untracked": untracked,
            "frames": shown,
            "samples": [frame["sample"] for frame in built],
            "coverageFrames": extra,
            "framePaths": [frame["path"] for frame in built],
            "peopleCapNote": CAP_NOTE,
        }

    return vlm_once.request_once(
        out,
        brief=brief,
        images=[frame["path"] for frame in built],
        model=model,
        validate=validate,
        identity_extra={
            "clip": clip_identity,
            "tracksJson": vlm_once.file_identity(tracks_dir / "tracks.json"),
            "trackIds": ids,
            "samples": [int(sample) for sample in samples],
            "frames": shown,
            "boxSource": box_source,
        },
        urlopen=urlopen,
        label="people judgement",
        report=report,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tracks", required=True, help="the tracks stage output directory")
    ap.add_argument("--clip", required=True, help="the video tracking ran on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_FRAMES, help="annotated frames to send")
    ap.add_argument("--model", default=vlm_once.DEFAULT_MODEL)
    ap.add_argument("--box", default="maskBox", choices=["maskBox", "box"])
    ap.add_argument(
        "--allow-clip-mismatch",
        action="store_true",
        help="draw the boxes even though this clip is not the file tracking ran on",
    )
    a = ap.parse_args()
    try:
        record = judge_people(
            a.tracks,
            a.clip,
            Path(a.out),
            n=a.n,
            model=a.model,
            allow_clip_mismatch=a.allow_clip_mismatch,
            box_source=a.box,
            report=lambda line: print(line, file=sys.stderr),
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
