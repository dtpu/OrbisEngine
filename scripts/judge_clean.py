#!/usr/bin/env python3
"""Ask a vision model whether a cleaned plate is safe to spend 1600 Marble credits on.

scripts/run_clip.py's `review` stage (its `review()` method) is a human stop: it pulls six frames
out of the cleaned video into `.context/run/<name>/review/clean_NNNN.png` and asks an operator to
check that the person is fully gone, that nothing of theirs is left hanging in mid-air, that there
are no limb smears and that nobody is left at the frame edge. This judge asks the same question of
the OpenAI API instead, so an unattended run has a recorded opinion rather than nothing.

A cleaned frame on its own cannot answer it. A grey slab where a crowd stood looks like a clean
wall until the original is beside it, and a person the detector never saw looks like part of the
scene. So every request carries pairs: the ORIGINAL frame on the left, the CLEANED frame on the
right, the same index drawn on both. The originals are decoded from the run's own source clip at
the source ordinals `clean.json` recorded for those cleaned frames, sequentially -- seeking lands
on keyframes and variable-frame-rate clips break it outright near the end of a file.

What comes back is a MODEL OPINION about a handful of frames. It is triage, not proof of visual
quality (docs/quality-rubric.md): it is `calibration: "uncalibrated"` until somebody records
agreement with human verdicts through `--human-verdict`, and it advances nothing on its own.

The verdict this file writes is NOT the model's overall answer. The per-frame answers are combined
here, deterministically, under the rubric's rules: a critical failure -- a subject not removed, a
non-human subject left standing, or major background destruction -- vetoes the whole judgement and
can never be averaged away, and otherwise at least 75% of frames must pass. The model's own overall
verdict can only make the result worse, never better. Alongside that, cheap measured signals are
recorded -- how much of the frame changed, how much changed outside the mask the clean stage saved,
and how much flatter the changed region became -- which are evidence for a human, never a silent
override.

  uv run --locked python scripts/judge_clean.py \\
      --run .context/run/<name> --out .context/run/<name>/clean-judge.json \\
      [--max-frames 8] [--model gpt-6-astra]

Exit codes: 0 pass, 10 retry, 11 fail, 2 error, so a runner can branch on it. `--human-verdict`
appends to `clean-judge-calibration.jsonl` beside the output and takes over the exit code: a human
who has looked at the frames always outranks this file.

One paid request per `--out`. `<out>.receipt.json` and `<out>.response.raw` carry the receipt
lifecycle in scripts/vlm_once.py: an interrupted run finishes from the bytes already paid for, and
a failed or unknown receipt refuses instead of submitting a second request.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import vlm_once

SCHEMA = "wander.clean-judge/1"
CALIBRATION_SCHEMA = "wander.clean-judge-calibration/1"
CALIBRATION_NAME = "clean-judge-calibration.jsonl"

MAX_FRAMES = 8
HALF_EDGE = 768
SEPARATOR_PX = 8
PASS_SHARE = 0.75
CHANGED_LEVEL = 8.0
# Thresholds for the advisory signal lines only. They were read off one inpainted clip, they are
# not calibrated against human verdicts, and nothing branches on them.
OUTSIDE_MASK_FLAG = 0.10
PLATE_DETAIL_FLAG = 0.35
CHANGED_FRAME_FLAG = 0.5

REMOVED = ("all", "partial", "none")
DAMAGE = ("none", "minor", "major")
SEVERITY = ("none", "minor", "major")
VERDICTS = ("pass", "retry", "fail")
ACTIONS = (
    "increase_dilation",
    "switch_to_semantic_mask",
    "switch_to_foreground_mask",
    "enable_moved_mask",
    "none",
)
WORSE = {"pass": 0, "retry": 1, "fail": 2}
EXIT_CODES = {"pass": 0, "retry": 10, "fail": 11}
ERROR_EXIT = 2

OPINION = (
    "An automated triage opinion about the frame pairs listed here. It is not proof of visual "
    "quality, not a measurement of the cleaned plate, and not evidence that anything absent from "
    "these frames is absent from the clip. A human looks at the pairs before credits are spent."
)
CALIBRATION_NOTE = (
    "No agreement with human verdicts has been recorded for this judge yet. Record one per run "
    f"with --human-verdict/--human-note; the entries land in {CALIBRATION_NAME} beside this file "
    "so false passes and false fails can be counted and reported, as docs/quality-rubric.md "
    "requires before an automated judge advances anything."
)
DECISION_RULES = (
    "Computed here from the per-frame answers, with no further model call. A frame passes when its "
    "own verdict is pass AND every subject was removed AND no remnant is left floating AND nobody "
    "is left at an edge AND background damage is none or minor AND smearing is not major AND no "
    "non-human subject was left. A critical failure -- a subject not fully removed, a non-human "
    "subject left, or major background destruction -- vetoes the judgement and can never be "
    "averaged away; it is a retry when one supported change was recommended and a fail otherwise. "
    "Without a critical failure, at least 75% of frames passing is a pass and anything less is a "
    "retry. The model's own overall verdict is then applied in the severe direction only: it can "
    "turn a pass into a retry or a fail, never the other way. A retry is automatic only when it "
    "names a supported change that is not already the current setting."
)
SIGNALS_NOTE = (
    "Measured from the frame pairs and the clean stage's saved masks, recorded beside the "
    "judgement as evidence for a human. Nothing here changes the verdict, and nothing here is a "
    "quality measurement: the cleaned frames come out of an H.264 encode, so even untouched "
    "pixels differ a little from the original, and both blur ratios are unvalidated proxies. The "
    "thresholds the signal lines use were read off one inpainted clip and are not calibrated."
)

BRIEF = """These {count} images each show ONE frame of a video twice. On the LEFT is the ORIGINAL
frame. On the RIGHT is the same frame after an automatic pass tried to erase every person from it.
The number printed on both halves is that frame's index.

Judge the RIGHT half. Use the LEFT half only to know who was there and what the background behind
them should still look like. The cleaned plate on the right is about to be turned into a 3D room,
so anything left behind or destroyed becomes permanent.

Look for, in this order:
  1. every person in the original is completely gone on the right: no half body, no limb, no face,
     no ghost, no person-shaped dark blob;
  2. nothing a person carried or wore is left hanging in mid-air: a bag, a dumbbell, a boxing
     glove, a racket, a hat, a shoe;
  3. nobody left at the edge of the frame, however small, blurred or partly cut off;
  4. a subject that is not an ordinary human -- an animated or costumed character, a mascot -- has
     been removed too;
  5. the background that was never a person is still there. A crowd in the stands, spectators,
     furniture, a table, gym equipment, a wall or a sign must NOT be flattened into a grey or
     blurred slab. Erased background is as bad as a person left behind;
  6. no heavy smearing: the repaired area should read as the room, not as a wash or a stain.

Judge only what is visible in these images. Do not guess what was behind anybody, and say nothing
about anything outside the frame.

How this plate was made, so your recommendation names a change that is actually available:
{settings}

Answer for EVERY frame index in this list, exactly once, and for no other index: {indices}

Per frame, each field with exactly one of the listed values:
  subjects_removed          "all" | "partial" | "none"
  subjects_remaining        "" when subjects_removed is "all"; otherwise a few words naming who or
                            what is still visible, and where in the frame
  floating_remnants         [] or short names of things left hanging with nobody holding them,
                            e.g. ["yellow glove", "shoulder bag"]
  edge_people               [] or short descriptions of people still visible at a frame edge,
                            e.g. ["head and shoulder, bottom right"]
  background_destroyed      "none" | "minor" | "major"   (major = something that should have
                            stayed, such as a crowd, the stands or furniture, has been erased)
  background_note           "" when "none"; otherwise a few words naming what was destroyed
  smear_severity            "none" | "minor" | "major"
  smear_fraction            a number from 0 to 1, roughly how much of the frame is smeared
  non_human_subject_left    "none", or a few words naming the non-human subject still present
  verdict                   "pass" | "retry" | "fail"
  reason                    one clause, from what is visible

Then one overall answer for the whole set:
  verdict                   "pass" | "retry" | "fail"
  recommended_action        exactly one of:
                              "increase_dilation"          masks too tight: outlines, remnants or
                                                           held objects survive around a person
                              "switch_to_semantic_mask"    somebody was never detected, so was
                                                           never removed at all
                              "switch_to_foreground_mask"  background people who should have
                                                           stayed (a crowd, the stands) were erased
                              "enable_moved_mask"          what is left behind is an object that
                                                           moved with the person, not a body
                              "none"                       nothing to change
                            Use "none" when, and only when, your overall verdict is "pass". Use
                            one of the other four when your overall verdict is "retry".
  reason                    one clause

Answer with JSON and nothing else:
{{"frames": [{{"index": <a frame index from the list>,
               "subjects_removed": "all",
               "subjects_remaining": "",
               "floating_remnants": [],
               "edge_people": [],
               "background_destroyed": "none",
               "background_note": "",
               "smear_severity": "none",
               "smear_fraction": 0,
               "non_human_subject_left": "none",
               "verdict": "pass",
               "reason": "<one clause>"}}],
 "overall": {{"verdict": "pass",
              "recommended_action": "none",
              "reason": "<one clause>"}}}}
"""


# ---- the run's own files ----------------------------------------------------------------


def load_clean_report(run_dir):
    """`clean.json`, written by worker/modal_clean_video.py: what the clean stage actually did."""
    path = Path(run_dir) / "clean.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{path} is missing; the clean stage has not run for this run directory")
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise TypeError(f"{path} is not a clean-stage report")
    if report.get("error"):
        raise ValueError(f"{path} records a failed clean stage; there is nothing to judge")
    return path, report


def recorded_clip(run_dir):
    """The source clip the run was launched on, from `state.json`'s `_run` record."""
    path = Path(run_dir) / "state.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{path} is missing; pass --clip to name the source clip")
    stages = json.loads(path.read_text()).get("stages")
    clip = (stages or {}).get("_run", {}).get("clip")
    if not isinstance(clip, str) or not clip:
        raise ValueError(f"{path} records no source clip; pass --clip")
    return Path(clip)


def settings_line(clean):
    """The clean stage's own parameters, for the brief and for the record."""
    return {
        "peopleMask": clean.get("peopleMask"),
        "moved": bool(clean.get("moved")),
        "dilate": clean.get("dilate"),
        "bottomExtra": clean.get("bottomExtra"),
        "maskFraction": clean.get("maskFraction"),
        "framesWithPeople": clean.get("framesWithPeople"),
        "frames": clean.get("frames"),
        "foregroundSelected": clean.get("foregroundSelected"),
        "foregroundRejectedSmall": clean.get("foregroundRejectedSmall"),
        "fps": clean.get("fps"),
        "width": clean.get("width"),
        "height": clean.get("height"),
    }


def settings_text(clean):
    mode = "moved-content masks (everything that moved)" if clean.get("moved") else None
    mode = mode or f"{clean.get('peopleMask')} person masks"
    growth = (
        f"  mask growth: dilate {clean.get('dilate')} px, "
        f"{clean.get('bottomExtra')} px extra downwards"
    )
    lines = [
        f"  mask mode: {mode}",
        growth,
        f"  masked share of all pixels: {100 * float(clean.get('maskFraction') or 0.0):.1f}%",
    ]
    if clean.get("foregroundSelected") is not None:
        lines.append(
            f"  instance detector: {clean.get('foregroundSelected')} people kept, "
            f"{clean.get('foregroundRejectedSmall')} rejected as too small to reconstruct "
            "(rejected people stay in the plate on purpose)"
        )
    return "\n".join(lines)


def spread(items, limit):
    """`limit` items spread evenly over `items`, keeping the first and the last."""
    if limit < 1:
        raise ValueError("--max-frames must be at least 1")
    if len(items) <= limit:
        return list(items)
    picks = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[index] for index in sorted(dict.fromkeys(int(p) for p in picks))]


def review_frames(run_dir, clean, limit=MAX_FRAMES):
    """The gate's own cleaned frames, paired with the source ordinal each one came from.

    run_clip.py writes `review/clean_<i>.png` where `i` is the frame's ordinal in the cleaned mp4.
    The clean stage encodes the frames it inpainted in order, so that ordinal indexes
    `keptIndices`, which indexes `indices` -- the source frame ordinal the resampler took. A full
    clean pass keeps every resampled frame, and this mapping still holds when it did not.
    """
    review = Path(run_dir) / "review"
    if not review.is_dir():
        raise ValueError(f"{review} is missing; the review gate has not built its frames")
    files = sorted(
        path for path in review.glob("clean_*.png") if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise ValueError(f"No clean_NNNN.png frames in {review}; there is nothing to judge")
    indices = [int(value) for value in (clean.get("indices") or [])]
    if not indices:
        raise ValueError("clean.json records no source `indices`; originals cannot be paired")
    kept = [int(value) for value in (clean.get("keptIndices") or [])] or list(range(len(indices)))
    frames = []
    for path in spread(files, limit):
        try:
            ordinal = int(path.stem.split("_")[-1])
        except ValueError:
            raise ValueError(f"{path.name} is not a clean_NNNN.png review frame") from None
        if ordinal >= len(kept):
            raise ValueError(f"{path.name} is beyond the {len(kept)} frames clean.json recorded")
        resampled = kept[ordinal]
        if resampled >= len(indices):
            raise ValueError(f"{path.name} maps past the source indices clean.json recorded")
        frames.append(
            {
                "index": ordinal,
                "resampledIndex": resampled,
                "sourceIndex": indices[resampled],
                "cleanPath": path,
            }
        )
    return frames


# ---- decoding the matching originals ----------------------------------------------------


def read_frames(clip, ordinals):
    """Decode the requested source-frame ordinals in stream order.

    Seeking with CAP_PROP_POS_FRAMES lands on keyframes in many containers and a variable-frame-
    rate clip loses the mapping entirely near the end of the file, so frames are grabbed
    sequentially and only the wanted ordinals retrieved -- the rule the stages already follow.
    ffmpeg decodes the same way for the frames cv2 could not hand back.
    """
    wanted = sorted({int(index) for index in ordinals})
    if not wanted:
        return {}
    images = {}
    capture = cv2.VideoCapture(str(clip))
    try:
        remaining = iter(wanted)
        target = next(remaining, None)
        ordinal = 0
        while target is not None:
            if not capture.grab():
                break
            if ordinal == target:
                ok, image = capture.retrieve()
                if ok and image is not None:
                    images[ordinal] = image
                target = next(remaining, None)
            ordinal += 1
    finally:
        capture.release()
    missing = [index for index in wanted if index not in images]
    if missing:
        images.update(ffmpeg_frames(clip, missing))
    missing = [index for index in wanted if index not in images]
    if missing:
        raise ValueError(f"Source frames {missing} could not be decoded from {clip}")
    return images


def ffmpeg_frames(clip, ordinals):
    """The same ordinals through ffmpeg's own sequential decode, as a fallback for cv2."""
    ordinals = sorted({int(index) for index in ordinals})
    selection = "+".join(f"eq(n\\,{index})" for index in ordinals)
    with tempfile.TemporaryDirectory(prefix="clean-judge-") as work:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip),
                "-vf",
                f"select={selection}",
                "-vsync",
                "0",
                "-frames:v",
                str(len(ordinals)),
                str(Path(work) / "f_%05d.png"),
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            return {}
        images = {}
        for position, index in enumerate(ordinals, start=1):
            path = Path(work) / f"f_{position:05d}.png"
            if path.is_file():
                image = cv2.imread(str(path))
                if image is not None:
                    images[index] = image
        return images


# ---- the composites the model is asked about --------------------------------------------


def fit(image, edge=HALF_EDGE):
    height, width = image.shape[:2]
    scale = min(1.0, edge / max(width, height))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )


def label(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, max(image.shape[:2]) / 900)
    thickness = max(1, round(max(image.shape[:2]) / 400))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (0, 0), (text_w + 16, text_h + baseline + 12), (0, 0, 0), -1)
    cv2.putText(image, text, (8, text_h + 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def composite(original, cleaned, index, path):
    """One pair: the original on the left, the cleaned frame on the right, the index on both."""
    left, right = fit(original.copy()), fit(cleaned.copy())
    if left.shape[:2] != right.shape[:2]:
        left = cv2.resize(left, (right.shape[1], right.shape[0]), interpolation=cv2.INTER_AREA)
    label(left, f"ORIGINAL frame {index}")
    label(right, f"CLEANED frame {index}")
    separator = np.full((right.shape[0], SEPARATOR_PX, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([left, separator, right]))
    return str(path)


def build_composites(clip, frames, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    originals = read_frames(clip, [frame["sourceIndex"] for frame in frames])
    built = []
    for frame in frames:
        cleaned = cv2.imread(str(frame["cleanPath"]))
        if cleaned is None:
            raise ValueError(f"{frame['cleanPath']} could not be decoded")
        original = originals[frame["sourceIndex"]]
        if original.shape[:2] != cleaned.shape[:2]:
            # The clean stage resampled the source to its own width/height; match it so the two
            # halves show the same field of view rather than a crop.
            original = cv2.resize(
                original,
                (cleaned.shape[1], cleaned.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        path = out_dir / f"pair_{frame['index']:04d}.png"
        built.append(
            {
                **frame,
                "original": original,
                "cleaned": cleaned,
                "compositePath": composite(original, cleaned, frame["index"], path),
            }
        )
    return built


def reuse_composites(clip, frames, out_dir):
    """The composites already beside `out`, when every expected one is still there.

    A run that has a receipt must present the bytes it paid to have looked at, so the pairs are
    not redrawn under it; a missing file falls back to building them again.
    """
    out_dir = Path(out_dir)
    paths = [out_dir / f"pair_{frame['index']:04d}.png" for frame in frames]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        return None
    originals = read_frames(clip, [frame["sourceIndex"] for frame in frames])
    built = []
    for frame, path in zip(frames, paths, strict=True):
        cleaned = cv2.imread(str(frame["cleanPath"]))
        if cleaned is None:
            return None
        original = originals[frame["sourceIndex"]]
        if original.shape[:2] != cleaned.shape[:2]:
            original = cv2.resize(
                original, (cleaned.shape[1], cleaned.shape[0]), interpolation=cv2.INTER_AREA
            )
        built.append(
            {**frame, "original": original, "cleaned": cleaned, "compositePath": str(path)}
        )
    return built


# ---- cheap measured signals (evidence, never an override) -------------------------------


def load_masks(run_dir, resampled_indices):
    """The person masks the clean stage saved, for the frames being judged.

    `masks.npz` holds them bit-packed along the last axis, so only the wanted rows are unpacked.
    A run without the file is judged without mask-relative signals rather than refused.
    """
    path = Path(run_dir) / "masks.npz"
    if not path.is_file() or path.is_symlink():
        return {}, f"{path.name} is not beside the run; mask-relative signals are absent"
    try:
        with np.load(path) as store:
            packed = store["masks"]
            shape = [int(value) for value in store["shape"]] if "shape" in store.files else None
            masks = {}
            for index in sorted({int(value) for value in resampled_indices}):
                if index >= len(packed):
                    continue
                bits = np.unpackbits(packed[index], axis=-1).astype(bool)
                masks[index] = bits[: shape[1], : shape[2]] if shape else bits
    except (OSError, ValueError, KeyError, IndexError):
        return {}, f"{path.name} could not be read; mask-relative signals are absent"
    return masks, None


def measure(original, cleaned, mask):
    """What changed between the two halves, in numbers, with no model involved."""
    left = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    difference = np.abs(left - right)
    changed = difference > CHANGED_LEVEL
    signals = {
        "changedFraction": round(float(changed.mean()), 4),
        "meanAbsDifference": round(float(difference.mean()), 3),
        "maskFraction": None,
        "meanAbsDifferenceInsideMask": None,
        "meanAbsDifferenceOutsideMask": None,
        "changedOutsideMaskFraction": None,
        "blurRatioInChangedRegion": None,
        "blurRatioAgainstPlate": None,
    }
    if mask is not None and mask.shape == changed.shape:
        outside = ~mask
        signals["maskFraction"] = round(float(mask.mean()), 4)
        if mask.any():
            signals["meanAbsDifferenceInsideMask"] = round(float(difference[mask].mean()), 3)
        if outside.any():
            signals["meanAbsDifferenceOutsideMask"] = round(float(difference[outside].mean()), 3)
            signals["changedOutsideMaskFraction"] = round(float(changed[outside].mean()), 4)
    masked = mask is not None and mask.shape == changed.shape and mask.any()
    region = mask if masked else changed
    if region.any():
        before = cv2.Laplacian(left, cv2.CV_32F)
        after = cv2.Laplacian(right, cv2.CV_32F)
        original_detail = float(before[region].var())
        repaired_detail = float(after[region].var())
        if original_detail > 1e-6:
            # Detail lost where somebody stood. Removing a person removes their detail too, so a
            # low ratio here is expected and is not on its own a smear.
            signals["blurRatioInChangedRegion"] = round(repaired_detail / original_detail, 3)
        plate = after[~mask] if masked and (~mask).any() else None
        if plate is not None and float(plate.var()) > 1e-6:
            # Detail inside the repair against the plate around it. A smear is flat where the
            # room it sits in is not, so this is the closer proxy of the two.
            signals["blurRatioAgainstPlate"] = round(repaired_detail / float(plate.var()), 3)
    return signals


def signal_flags(frames):
    """One line per signal a human should look at. Advisory: the verdict never reads these."""
    flags = []
    for frame in frames:
        signals, index = frame["signals"], frame["index"]
        outside = signals.get("changedOutsideMaskFraction")
        if outside is not None and outside > OUTSIDE_MASK_FLAG:
            flags.append(
                f"frame {index}: {100 * outside:.0f}% of the pixels outside the saved mask "
                "changed, more than an encode alone explains"
            )
        blur = signals.get("blurRatioAgainstPlate")
        if blur is not None and blur < PLATE_DETAIL_FLAG:
            flags.append(
                f"frame {index}: the repaired region came back {blur:.2f}x as detailed as the "
                "plate around it, which is what a smear measures like"
            )
        if signals["changedFraction"] > CHANGED_FRAME_FLAG:
            flags.append(
                f"frame {index}: {100 * signals['changedFraction']:.0f}% of the frame changed, so "
                "most of what the world is built from is invented"
            )
    return flags


def aggregate(frames):
    changed = [frame["signals"]["changedFraction"] for frame in frames]
    outside = [
        frame["signals"]["changedOutsideMaskFraction"]
        for frame in frames
        if frame["signals"]["changedOutsideMaskFraction"] is not None
    ]
    blur = [
        frame["signals"]["blurRatioInChangedRegion"]
        for frame in frames
        if frame["signals"]["blurRatioInChangedRegion"] is not None
    ]
    plate = [
        frame["signals"]["blurRatioAgainstPlate"]
        for frame in frames
        if frame["signals"]["blurRatioAgainstPlate"] is not None
    ]
    return {
        "meanChangedFraction": round(float(np.mean(changed)), 4) if changed else None,
        "maxChangedFraction": round(float(np.max(changed)), 4) if changed else None,
        "maxChangedOutsideMaskFraction": round(float(np.max(outside)), 4) if outside else None,
        "minBlurRatioInChangedRegion": round(float(np.min(blur)), 3) if blur else None,
        "minBlurRatioAgainstPlate": round(float(np.min(plate)), 3) if plate else None,
        "framesWithMask": len(outside),
    }


# ---- validating what came back ----------------------------------------------------------


def _clause(value, field, index):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"frame {index} has no {field}")
    return value.strip()


def _optional_clause(value, field, index):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"frame {index} has a non-text {field}")
    text = value.strip()
    return text or None


def _string_list(value, field, index):
    """`[]`, or short non-empty names. `null` and a lone "none" are the empty list spelled out."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"frame {index} has a non-list {field}")
    items = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(f"frame {index} has a non-text entry in {field}")
        text = entry.strip()
        if text:
            items.append(text)
    if len(items) == 1 and items[0].lower() in ("none", "no", "nothing"):
        return []
    return items


def _enum(entry, field, allowed, index):
    value = entry.get(field)
    if value not in allowed:
        raise ValueError(f"frame {index} has {field} {value!r}, not one of {list(allowed)}")
    return value


def check_frames(record, indices):
    """Every judged frame index, exactly once, every field from its own enumeration."""
    entries = record.get("frames")
    if not isinstance(entries, list) or not entries:
        raise ValueError("the answer judges no frames")
    wanted = set(indices)
    answers = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("a frame entry is not an object")
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("a frame entry has no integer index")
        if index not in wanted:
            raise ValueError(f"frame {index} was not in the images")
        if index in answers:
            raise ValueError(f"frame {index} was judged twice")
        removed = _enum(entry, "subjects_removed", REMOVED, index)
        remaining = _optional_clause(entry.get("subjects_remaining"), "subjects_remaining", index)
        if removed != "all" and not remaining:
            raise ValueError(f"frame {index} says {removed!r} removed but names nobody remaining")
        destroyed = _enum(entry, "background_destroyed", DAMAGE, index)
        note = _optional_clause(entry.get("background_note"), "background_note", index)
        if destroyed != "none" and not note:
            raise ValueError(f"frame {index} reports {destroyed} background damage without a note")
        severity = _enum(entry, "smear_severity", SEVERITY, index)
        fraction = entry.get("smear_fraction")
        if fraction is None and severity == "none":
            fraction = 0.0
        if type(fraction) not in (int, float) or isinstance(fraction, bool):
            raise TypeError(f"frame {index} has a non-numeric smear_fraction")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"frame {index} has smear_fraction {fraction} outside 0..1")
        non_human = entry.get("non_human_subject_left")
        if not isinstance(non_human, str):
            raise TypeError(f"frame {index} has no non_human_subject_left")
        non_human = non_human.strip()
        answers[index] = {
            "subjectsRemoved": removed,
            "subjectsRemaining": remaining,
            "floatingRemnants": _string_list(
                entry.get("floating_remnants"), "floating_remnants", index
            ),
            "edgePeople": _string_list(entry.get("edge_people"), "edge_people", index),
            "backgroundDestroyed": destroyed,
            "backgroundNote": note,
            "smearSeverity": severity,
            "smearFraction": round(float(fraction), 3),
            "nonHumanSubjectLeft": None if non_human.lower() in ("", "none") else non_human,
            "verdict": _enum(entry, "verdict", VERDICTS, index),
            "reason": _clause(entry.get("reason"), "reason", index),
        }
    missing = [index for index in indices if index not in answers]
    if missing:
        raise ValueError(f"frames {missing} were not judged")
    return answers


def check_overall(record):
    overall = record.get("overall")
    if not isinstance(overall, dict):
        raise TypeError("the answer has no overall object")
    verdict = overall.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"the overall verdict is {verdict!r}, not one of {list(VERDICTS)}")
    action = overall.get("recommended_action")
    if action not in ACTIONS:
        raise ValueError(f"the recommended action is {action!r}, not one of {list(ACTIONS)}")
    if verdict == "pass" and action != "none":
        raise ValueError("an overall pass must recommend no change")
    if verdict == "retry" and action == "none":
        raise ValueError("an overall retry must recommend one supported change")
    reason = overall.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("the overall answer has no reason")
    return {"verdict": verdict, "recommendedAction": action, "reason": reason.strip()}


# ---- the deterministic decision ---------------------------------------------------------


def critical_kinds(answer):
    """What the rubric's critical-failure veto covers here; never averaged away."""
    kinds = []
    if answer["subjectsRemoved"] != "all":
        kinds.append("subject_not_removed")
    if answer["nonHumanSubjectLeft"]:
        kinds.append("non_human_subject_left")
    if answer["backgroundDestroyed"] == "major":
        kinds.append("background_destroyed")
    return kinds


def frame_passes(answer):
    return (
        answer["verdict"] == "pass"
        and not critical_kinds(answer)
        and not answer["floatingRemnants"]
        and not answer["edgePeople"]
        and answer["smearSeverity"] != "major"
    )


def action_is_noop(action, clean):
    """True when the recommended change is already how this plate was made."""
    mode, moved = clean.get("peopleMask"), bool(clean.get("moved"))
    if action == "switch_to_semantic_mask":
        return mode == "semantic" and not moved
    if action == "switch_to_foreground_mask":
        return mode == "foreground" and not moved
    if action == "enable_moved_mask":
        return moved
    return False


def suggested_flags(action, clean):
    """The run_clip.py flags the recommendation means, so a retry is one changed command.

    The dilation step is a general doubling of whatever this run used, not a number measured from
    any clip; the operator can pick a different one, and the reason is what justifies the retry.
    """
    if action == "increase_dilation":
        dilate = max(2 * int(clean.get("dilate") or 0), 10)
        bottom = max(2 * int(clean.get("bottomExtra") or 0), 10)
        return ["--dilate", str(dilate), "--bottom-extra", str(bottom)]
    if action == "switch_to_semantic_mask":
        return ["--people-mask", "semantic"]
    if action == "switch_to_foreground_mask":
        return ["--people-mask", "foreground"]
    if action == "enable_moved_mask":
        # modal_clean_video.py refuses --moved-mask with a non-semantic people mask, and
        # run_clip.py's clean_flags() drops --people-mask entirely once --moved-mask is set.
        return ["--moved-mask"]
    return []


def decide(frames, overall, clean):
    """Combine the per-frame answers into one verdict. See DECISION_RULES."""
    critical = [
        {"index": frame["index"], "kinds": kinds}
        for frame in frames
        if (kinds := critical_kinds(frame["answers"]))
    ]
    passing = [frame["index"] for frame in frames if frame_passes(frame["answers"])]
    share = round(len(passing) / len(frames), 4)
    if critical:
        recoverable = overall["recommendedAction"] != "none" and overall["verdict"] != "fail"
        verdict = "retry" if recoverable else "fail"
    elif share >= PASS_SHARE:
        verdict = "pass"
    else:
        verdict = "retry"
    if WORSE[overall["verdict"]] > WORSE[verdict]:
        verdict = overall["verdict"]
    action = overall["recommendedAction"] if verdict == "retry" else "none"
    noop = action != "none" and action_is_noop(action, clean)
    return {
        "verdict": verdict,
        "recommendedAction": action,
        "actionReason": overall["reason"] if action != "none" else None,
        "actionFlags": [] if noop else suggested_flags(action, clean),
        "actionIsNoOp": noop,
        "autoRetryable": verdict == "retry" and action != "none" and not noop,
        "passingFrames": passing,
        "passShare": share,
        "passThreshold": PASS_SHARE,
        "criticalFailures": critical,
    }


# ---- the judge --------------------------------------------------------------------------


def judge_clean(
    run_dir,
    out,
    *,
    max_frames=MAX_FRAMES,
    model=vlm_once.DEFAULT_MODEL,
    clip=None,
    allow_clip_mismatch=False,
    urlopen=None,
    report=None,
):
    """Judge one run's cleaned frames in one paid request, or reuse what a run already paid for."""
    run_dir = Path(run_dir)
    out = Path(out)
    limit = min(int(max_frames), MAX_FRAMES)
    if limit < 1:
        raise ValueError("--max-frames must be at least 1")
    clean_path, clean = load_clean_report(run_dir)
    clip = Path(clip) if clip else recorded_clip(run_dir)
    if not clip.is_file():
        raise ValueError(f"The source clip {clip} is missing; pass --clip")
    clip_identity = vlm_once.file_identity(clip)
    recorded = clean.get("sourceSha256")
    if recorded and clip_identity["sha256"] != recorded and not allow_clip_mismatch:
        raise ValueError(
            "The cleaned frames were made from the video clean.json records, and this clip is a "
            "different file. Pass that video, or --allow-clip-mismatch to pair them anyway."
        )

    frames = review_frames(run_dir, clean, limit)
    pairs_dir = out.parent / (out.stem + "-pairs")
    built = None
    if vlm_once.receipt_status(out) is not None:
        built = reuse_composites(clip, frames, pairs_dir)
    if built is None:
        built = build_composites(clip, frames, pairs_dir)

    masks, mask_note = load_masks(run_dir, [frame["resampledIndex"] for frame in built])
    measured = []
    for frame in built:
        measured.append(
            {
                "index": frame["index"],
                "resampledIndex": frame["resampledIndex"],
                "sourceIndex": frame["sourceIndex"],
                "cleanPath": str(frame["cleanPath"]),
                "compositePath": frame["compositePath"],
                "signals": measure(
                    frame["original"], frame["cleaned"], masks.get(frame["resampledIndex"])
                ),
            }
        )
    indices = [frame["index"] for frame in measured]
    brief = BRIEF.format(
        count=len(measured),
        settings=settings_text(clean),
        indices=", ".join(str(index) for index in indices),
    )

    def validate(record):
        answers = check_frames(record, indices)
        overall = check_overall(record)
        judged = [{**frame, "answers": answers[frame["index"]]} for frame in measured]
        decision = decide(judged, overall, clean)
        return {
            "schema": SCHEMA,
            "judge": "clean",
            "model": model,
            "promptSha256": hashlib.sha256(brief.encode()).hexdigest(),
            "opinion": OPINION,
            "calibration": "uncalibrated",
            "calibrationNote": CALIBRATION_NOTE,
            "run": str(run_dir),
            "runName": run_dir.name,
            "clip": str(clip),
            "cleanReport": str(clean_path),
            "clean": settings_line(clean),
            "framesJudged": indices,
            "frames": judged,
            **decision,
            "modelOverall": overall,
            "decisionRules": DECISION_RULES,
            "signals": aggregate(judged),
            "signalFlags": signal_flags(judged),
            "signalsNote": SIGNALS_NOTE,
            "maskNote": mask_note,
        }

    return vlm_once.request_once(
        out,
        brief=brief,
        images=[frame["compositePath"] for frame in measured],
        model=model,
        validate=validate,
        identity_extra={
            "clip": clip_identity,
            "cleanJson": vlm_once.file_identity(clean_path),
            "cleanFrames": [vlm_once.file_identity(frame["cleanPath"]) for frame in built],
            "frames": indices,
            "sourceIndices": [frame["sourceIndex"] for frame in measured],
        },
        urlopen=urlopen,
        label="clean judgement",
        report=report,
    )


# ---- the calibration ledger -------------------------------------------------------------


def append_calibration(out, record, verdict, note):
    """Record one human verdict beside the judgement, so disagreements can be counted later."""
    out = Path(out)
    if verdict not in VERDICTS:
        raise ValueError(f"--human-verdict must be one of {list(VERDICTS)}")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("--human-note is required with --human-verdict")
    judged = record.get("verdict")
    entry = {
        "schema": CALIBRATION_SCHEMA,
        "recorded": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run": record.get("runName"),
        "judgement": str(out),
        "judgementSha256": vlm_once.file_identity(out)["sha256"] if out.is_file() else None,
        "model": record.get("model"),
        "judgeVerdict": judged,
        "judgeAction": record.get("recommendedAction"),
        "humanVerdict": verdict,
        "humanNote": note.strip(),
        "agreed": judged == verdict,
        "falsePass": judged == "pass" and verdict != "pass",
        "falseFail": judged in ("retry", "fail") and verdict == "pass",
    }
    path = out.parent / CALIBRATION_NAME
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked calibration ledger: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
    return entry


def summary(record):
    headline = (
        f"clean judge: {record['verdict']} "
        f"({len(record['passingFrames'])}/{len(record['framesJudged'])} frames pass, "
        f"threshold {int(100 * record['passThreshold'])}%)"
    )
    lines = [headline]
    for failure in record["criticalFailures"]:
        lines.append(f"  critical, frame {failure['index']}: {', '.join(failure['kinds'])}")
    if record["recommendedAction"] != "none":
        flags = " ".join(record["actionFlags"]) or "(already the current setting)"
        lines.append(f"  recommended: {record['recommendedAction']} -> {flags}")
        lines.append(f"  because: {record['actionReason']}")
    lines.extend(f"  signal: {flag}" for flag in record["signalFlags"])
    lines.append(f"  {record['calibration']}: a triage opinion, not proof of visual quality")
    return "\n".join(lines)


def main(argv=None, urlopen=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="the run directory, .context/run/<name>")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help=f"cleaned frames to pair and send, at most {MAX_FRAMES}",
    )
    ap.add_argument("--model", default=vlm_once.DEFAULT_MODEL)
    ap.add_argument("--clip", help="the source clip, when state.json does not name a readable one")
    ap.add_argument(
        "--allow-clip-mismatch",
        action="store_true",
        help="pair the originals even though this clip is not the file the clean stage read",
    )
    ap.add_argument(
        "--human-verdict",
        choices=VERDICTS,
        help="record what a human decided about the same frames; it overrides the exit code",
    )
    ap.add_argument("--human-note", help="one clause, required with --human-verdict")
    a = ap.parse_args(argv)
    try:
        record = judge_clean(
            a.run,
            Path(a.out),
            max_frames=a.max_frames,
            model=a.model,
            clip=a.clip,
            allow_clip_mismatch=a.allow_clip_mismatch,
            urlopen=urlopen,
            report=lambda line: print(line, file=sys.stderr),
        )
        verdict = record["verdict"]
        if a.human_verdict:
            entry = append_calibration(Path(a.out), record, a.human_verdict, a.human_note)
            verdict = entry["humanVerdict"]
            print(
                f"human verdict {verdict} recorded (judge said {entry['judgeVerdict']}); "
                "the human decision wins",
                file=sys.stderr,
            )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return ERROR_EXIT
    print(json.dumps(record, indent=1))
    print(summary(record), file=sys.stderr)
    return EXIT_CODES[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
