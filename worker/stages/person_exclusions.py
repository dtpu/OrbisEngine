"""Validate explicit review provenance and box correspondence, not semantic truth."""

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re


SCHEMA = "wander.person-exclusions/1"
ENTRY_KEYS = {"sourceIndex", "box", "label", "reason", "evidence"}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("sourceSha256 must be 64 hexadecimal characters")
    return value.lower()


def _box(value, width, height):
    try:
        if len(value) != 4:
            raise ValueError
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in value):
            raise ValueError
        box = tuple(float(v) for v in value)
        if not all(math.isfinite(v) for v in box):
            raise ValueError
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError
        return box
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "box must have four finite source pixels inside dimensions with positive extent"
        ) from exc


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_exclusions(path, source_sha, width, height):
    """Load a review bound to this exact source; return its unchanged JSON document."""
    expected_sha = _hash(source_sha)
    width = _integer(width, "width", 1)
    height = _integer(height, "height", 1)
    try:
        doc = json.loads(Path(path).read_text(), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("unreadable or malformed person exclusions") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("invalid person exclusions schema")
    if _hash(doc.get("sourceSha256")) != expected_sha:
        raise ValueError("stale review: source hash mismatch")
    review_width = _integer(doc.get("width"), "review width", 1)
    review_height = _integer(doc.get("height"), "review height", 1)
    if (review_width, review_height) != (width, height):
        raise ValueError("stale review: source dimensions mismatch")
    reviewer = doc.get("reviewer")
    if not isinstance(reviewer, dict) or reviewer.get("kind") not in ("human", "agent"):
        raise ValueError("reviewer kind must be human or agent")
    _text(reviewer.get("name"), "reviewer name")
    entries = doc.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise ValueError("entry must contain exactly sourceIndex, box, label, reason, evidence")
        index = _integer(entry["sourceIndex"], "sourceIndex")
        if not isinstance(entry["box"], list):
            raise ValueError("entry box must be a list")
        box = _box(entry["box"], width, height)
        if entry["label"] not in ("depiction", "statue"):
            raise ValueError("entry label must be depiction or statue")
        _text(entry["reason"], "reason")
        _text(entry["evidence"], "evidence")
        key = (index, box)
        if key in seen:
            raise ValueError("duplicate reviewed frame and box")
        seen.add(key)
    return doc


def load_video_exclusions(path, video, *, fps=None):
    """Bind a review to video bytes, dimensions and reported frame count.

    Container metadata is checked, not every frame decoded. This does not establish
    footage readability throughout the clip or validate the reviewer's labels.
    When fps is supplied, reviewed frames must belong to the tracker's exact
    source-index sampling schedule (including NumPy's round-to-even behavior).
    """
    import cv2

    digest = hashlib.sha256()
    try:
        with Path(video).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("unreadable source video") from exc
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("unreadable source video")
        metadata = [
            cap.get(key)
            for key in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FRAME_COUNT,
            )
        ]
        values = []
        for name, value in zip(("width", "height", "frame count"), metadata):
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"video {name} must be a positive integer")
            if not math.isfinite(value) or value <= 0 or int(value) != value:
                raise ValueError(f"video {name} must be a positive integer")
            values.append(int(value))
        if fps is not None:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            for name, value in (("source FPS", source_fps), ("target FPS", fps)):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{name} must be finite and positive")
    finally:
        cap.release()
    width, height, count = values
    doc = load_exclusions(path, digest.hexdigest(), width, height)
    if any(entry["sourceIndex"] >= count for entry in doc["entries"]):
        raise ValueError("stale review: sourceIndex exceeds video frame count")
    if fps is not None:
        import numpy as np

        duration = count / source_fps
        sample_count = duration * fps
        if not math.isfinite(duration) or not math.isfinite(sample_count):
            raise ValueError("tracking schedule must be finite")
        requested = np.arange(int(np.ceil(sample_count))) / fps
        requested = requested[requested < duration]
        indices = np.clip(np.rint(requested * source_fps).astype(int), 0, count - 1)
        if {entry["sourceIndex"] for entry in doc["entries"]} - set(map(int, indices)):
            raise ValueError("stale review: reviewed frames outside the tracking schedule")
    return doc


def match_exclusions(doc, source_index, instance_boxes):
    """Match a validated review at one frame, preserving every unreviewed instance.

    Every reviewed box must have exactly one strict IoU > 0.95 correspondence.
    Multiple reviewed boxes may not consume the same instance. Return original
    entries by instance index without changing the document or the input boxes.
    """
    source_index = _integer(source_index, "source_index")
    entries = [entry for entry in doc["entries"] if entry["sourceIndex"] == source_index]
    if not entries:
        return {}
    boxes = [_box(box, doc["width"], doc["height"]) for box in instance_boxes]
    result = {}
    for entry in entries:
        reviewed = entry["box"]
        matches = []
        for i, box in enumerate(boxes):
            # Normalize coordinates first to avoid overflowing area arithmetic.
            a = [v / scale for v, scale in zip(reviewed, (doc["width"], doc["height"]) * 2)]
            b = [v / scale for v, scale in zip(box, (doc["width"], doc["height"]) * 2)]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
            if union > 0 and intersection / union > 0.95:
                matches.append(i)
        if not matches:
            raise ValueError(f"stale review: no matching instance at source frame {source_index}")
        if len(matches) != 1 or matches[0] in result:
            raise ValueError(f"ambiguous review: nonunique instance at source frame {source_index}")
        result[matches[0]] = entry
    return result
