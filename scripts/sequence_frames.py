#!/usr/bin/env python3
"""The one place that answers "which frames does this person actually have?".

A packaged person is NOT dense from sample 0. `worker/stages/lhm_animate.py` animates only the
samples the tracker found this person in (`requestedSampleSemantics` in the manifest it writes), so
`frames` can start at `frame_022.ply`, skip samples in the middle, and stop before the last camera:

    creed-v2  frames frame_022.ply .. frame_144.ply, 106 of 145 samples, first source index 44

Every stage that reached for `person/frame_000.ply`, or indexed the frame list by a solver sample,
was reading a file that need never exist. `scale_fit` on creed-v2 stopped with

    scale_fit FAILED: public/worlds/creed-v2-4d/person/frame_000.ply is missing;
    the package stage has not run

while the package stage had in fact run and left 106 frames on disk. The helpers here take the
person's own `frames` list as the authority, so a reference frame is the first one that EXISTS and a
per-sample lookup goes through the sequence's own `sourceIndices`.

For a dense track starting at sample 0 every helper returns exactly what the old hard-coded paths
did: `first_frame()` is `frame_000.ply` and `index_of_source()` is the identity on the samples the
person covers.

json and pathlib only: this is imported by the pipeline runner, which must not pay for numpy to ask
where a file is.
"""

from __future__ import annotations

import json
from pathlib import Path


class MissingPersonFrames(RuntimeError):
    """The person genuinely has no frames on disk -- as opposed to none at sample 0."""


def sequence_path(world_dir, person: str = "person") -> Path:
    """`<world>/person/sequence.json`, the manifest a single-person package writes."""
    return Path(world_dir) / person / "sequence.json"


def read_sequence(seq_path) -> dict:
    """The manifest, with the precise error the caller would otherwise raise from `KeyError`."""
    seq_path = Path(seq_path)
    if not seq_path.is_file():
        raise MissingPersonFrames(f"{seq_path} is missing; the package stage has not run")
    try:
        seq = json.loads(seq_path.read_text())
    except ValueError as error:
        raise MissingPersonFrames(f"{seq_path} is not readable JSON: {error}") from error
    if not isinstance(seq, dict):
        raise MissingPersonFrames(f"{seq_path} is not a JSON object")
    return seq


def frame_names(seq: dict, seq_path) -> list[str]:
    """Every frame this person declares, in order. Empty is an error, not an empty person."""
    frames = seq.get("frames")
    if not isinstance(frames, list) or not frames:
        raise MissingPersonFrames(
            f"{seq_path} lists no `frames`: this person was never reconstructed, so there is no "
            f"frame to measure"
        )
    for name in frames:
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
        ):
            raise MissingPersonFrames(f"{seq_path}: {name!r} is not a frame name inside the person")
    return list(frames)


def frame_paths(seq_path) -> list[Path]:
    """Declared frame paths, existing or not, in the sequence's own order."""
    seq_path = Path(seq_path)
    seq = read_sequence(seq_path)
    return [seq_path.parent / name for name in frame_names(seq, seq_path)]


def existing_frames(seq_path) -> list[Path]:
    """The declared frames that are actually on disk, in order. May be shorter than `frames`."""
    return [p for p in frame_paths(seq_path) if p.is_file()]


def first_frame(seq_path) -> Path:
    """The first frame this person HAS -- the reference every stage used to take from sample 0.

    Raises `MissingPersonFrames` only when the person truly has nothing on disk, and says which
    frames were expected so the message can be acted on.
    """
    seq_path = Path(seq_path)
    declared = frame_paths(seq_path)
    for path in declared:
        if path.is_file():
            return path
    raise MissingPersonFrames(
        f"none of the {len(declared)} frames {seq_path} lists exist on disk "
        f"(first listed: {declared[0].name}, last: {declared[-1].name}); the package stage did not "
        f"finish"
    )


def first_world_frame(world_dir, person: str = "person") -> Path:
    """`first_frame()` for a packaged world directory."""
    return first_frame(sequence_path(world_dir, person))


def optional_first_world_frame(world_dir, person: str = "person") -> Path | None:
    """`first_world_frame()` for a summary that must print whatever exists, without raising."""
    try:
        return first_world_frame(world_dir, person)
    except MissingPersonFrames:
        return None


def source_indices(seq: dict, seq_path=None) -> list[int]:
    """This person's source frame index per declared frame; `range(n)` when the manifest omits it."""
    frames = seq.get("frames") or []
    raw = seq.get("sourceIndices")
    if raw is None:
        return list(range(len(frames)))
    if not isinstance(raw, list) or len(raw) != len(frames):
        raise MissingPersonFrames(
            f"{seq_path or 'sequence.json'}: sourceIndices has {len(raw) if isinstance(raw, list) else '?'} "
            f"entries for {len(frames)} frames; the two must describe the same samples"
        )
    return [int(x) for x in raw]


def index_of_source(seq: dict, seq_path=None):
    """`source frame index -> position in this person's frame list`, for exact per-sample lookups.

    Returns a plain dict, so a caller asks with `.get(src)` and gets `None` for a sample this person
    is absent from instead of the nearest frame in time. Nearest-match was the dense-track
    assumption: on creed-v2 it answered `frame_022.ply` for source frame 0, where the tracker had
    already recorded that nobody was there.
    """
    return {src: i for i, src in enumerate(source_indices(seq, seq_path))}
