#!/usr/bin/env python3
"""Merge per-shot 4D candidates into ONE package that plays the whole original clip.

A clip with cuts is reconstructed shot by shot: each shot becomes its own candidate world
(`public/worlds/<name>-shotNN-4d/` plus its own Marble `.spz`), on its own local clock that starts
at zero. This tool stitches those candidates back onto the ORIGINAL source clock, with the original
soundtrack, so the viewer plays the clip straight through and swaps world + cast at each cut.

--------------------------------------------------------------------------------------------------
VIEWER CONTRACT -- what fourd.html must do with the new `shots` block
--------------------------------------------------------------------------------------------------
The merged `people.json` is a normal `wander.people/1` manifest: shared `timestamps`/`duration` on
the source clock, `people[]` with per-person `visibleSampleRuns` and `transform`. Everything the
viewer already does with those keys keeps working unchanged, because the manifest is read as plain
JSON (fourd.html:1556) and today's viewer touches exactly eight top-level keys and nothing else:
`people` (fourd.html:1559), `sharedScale` (1574-1575), `floorFit` (1583), `sharedPlacement` (1610),
`peopleCount` (1627), `timestamps` and `duration` (2328-2329, 2513) and `primary` (2450). An
unknown key such as `shots` is therefore ignored by today's viewer, so an old viewer still renders
the merged cast, just over one fixed world.

What has to be added:

1. ACTIVE SHOT LOOKUP BY TIME. `shots` is ordered, non-overlapping and closed-open:
   shot k is active for `t` in `[shots[k].sourceStart, shots[k].sourceEnd)`. Outside every window
   (the `gaps` in sequence-report.json, and the same list is mirrored in `shotSequence.gaps`) NO
   shot is active: that source time was never reconstructed. Show the gap as such (hold the last
   world, or blank it) -- do not stretch a neighbouring shot over it. Lookup belongs in
   `applyTime()` (fourd.html:3100-3106), which already runs on every frame and every seek
   (`setTime`, fourd.html:3107-3112, and the `applyTime()` calls at fourd.html:3131, 4371, 4519,
   4638). People need no extra gating: `visibleSampleRuns` already confines each person to its own
   shot window, so a stale cast member cannot survive a cut.

2. WORLD SWAP. `worldUrl` is currently a startup constant (fourd.html:1065, consumed at
   fourd.html:1081-1086). It has to become per-shot: `shots[k].world` is the viewer URL of that
   shot's `.spz`. The cheapest correct shape is to load every shot's `SplatMesh` up front (or
   lazily on first activation) and toggle `.visible`, because `SplatMesh` construction is async and
   `applyTime()` is synchronous.

3. PER-WORLD DERIVED STATE. Everything below is derived FROM `worldUrl` or from the manifest's
   `primary`, so each of these becomes per-shot state that must be recomputed/kept per world and
   selected by the active shot:
     - `floorY` -- `?floor=` default (fourd.html:1230) and the splat-percentile estimate
       (fourd.html:1334-1342). Per shot: `shots[k].placement` (`shots/NN/placement.json`,
       `floorY`), else a per-world estimate.
     - the local floor map -- `buildFloorMap` (fourd.html:1368) and its localStorage cache key
       `floormap:${worldUrl}:${floorY.toFixed(3)}` (fourd.html:1351). The key already contains the
       world URL, so per-shot maps cache without collision; only the *selection* is new.
     - the walk grid -- `walkGrid` (fourd.html:3293), built from a probe of `worldUrl`
       (fourd.html:3706-3716, `buildWalkGrid` fourd.html:3428). One grid per world.
     - the shared cast scale and base position -- `scale0` (fourd.html:2457-2460) and `pos0`
       (fourd.html:2467-2470) are derived once from `primary.bodyH` and `floorY`. Each shot has its
       own `primary` (`shots[k].primary`) and its own floor, so these must be derived per shot and
       applied to that shot's people only. `shots[k].sharedPlacement` / `shots[k].sharedScale`
       carry the candidate manifest's own numbers for that shot.
     - `?place=1` placement -- discovered today beside people.json (fourd.html:1243-1259). In a
       merged package there is no top-level placement.json on purpose; use
       `shots[k].placement`.
     - per-shot cameras -- `shots[k].cameras` (`shots/NN/cameras.json`). There is no top-level
       cameras.json: the shots are separate SfM solves and share no frame.
   Per-person `transform` (fourd.html:2277-2290, 2471-2482) is already per person and is copied
   through unchanged, so it needs nothing new.

4. AUDIO AND THE VIDEO INSET are NOT per shot. `audio.json` beside the merged people.json is the
   whole original soundtrack at offset 0 on the merged clock, and `?video=` should be the original
   clip, so both run straight through the cuts.

--------------------------------------------------------------------------------------------------
TIME MAPPING
--------------------------------------------------------------------------------------------------
A candidate's local time `t` maps to source time `sourceStart + trimOffset + t`.

`scripts/shot_cuts.py::trim` (scripts/shot_cuts.py:650-686) seeks to `shot.start + 0.5/fps` and
keeps `shot.seconds - 2.0/fps`, so local zero is NOT `shot.start`. FFmpeg then rounds that seek to
a whole frame, so the true offset is a property of the actual trim, not of the arithmetic. This
tool therefore never guesses it. Each shot must resolve `trimOffset` from exactly one of:

  1. `trimOffsetSeconds` on the shot entry (an explicit measured number);
  2. a recorded `sourceMapping` -- `sourceMapping.trimOffsetSeconds`, or
     `sourceMapping.localZeroSourceSeconds - sourceStart`. `sourceMapping` is looked for on the
     shot entry, then in `stateJson` (`.context/run/<name>/state.json`, stages._run.shot, written
     by scripts/run_clip.py:365-373 and read back at scripts/run_clip.py:383-387), then in the
     candidate's own people.json;
  3. `trimOffsetFromShotCutsFps: <fps>` on the shot entry -- an explicit opt-in to the nominal
     `0.5/fps` seek of shot_cuts.py::trim (scripts/shot_cuts.py:659). This is DERIVED, not
     measured; the report labels it so.

Nothing else is accepted: a shot with no resolvable offset is an error.

The merged sample grid is the union of every shot's mapped sample times plus one terminator sample
at each `sourceEnd` (so a person's last visibility interval, which ends at `timestamps[run[1] + 1]`
-- src/person-visibility.ts:29 -- stops exactly at the cut instead of running into the next shot).
A shot whose `sourceEnd` is the end of the clip needs no terminator: `?? duration` supplies it.

--------------------------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------------------------
  uv run --locked python scripts/package_shot_sequence.py \
      --source public/clips/<clip>.mp4 \
      --shots .context/run/<clip>/shot-sequence.json \
      --out public/worlds/<clip>-sequence-4d [--link] [--force] [--no-audio]

`--shots` is a JSON list (or `{"shots": [...]}`) of, in source order:

  {
    "candidateDir": "public/worlds/clip-shot01-4d",   # required, holds people.json
    "world": "/marble-clip-shot01-clean.spz",         # required, the viewer URL of this shot's spz
    "sourceStart": 0.0, "sourceEnd": 4.233,           # required, seconds on the ORIGINAL clip
    "trimOffsetSeconds": 0.0167,                      # one of the three offset routes above
    "trimOffsetFromShotCutsFps": 30.0,
    "sourceMapping": {...}, "stateJson": "...",
    "worldFile": "public/marble-clip-shot01-clean.spz" # optional, checked for existence
  }

Outputs, all inside `--out`: `people.json` (merged, source clock), `audio.json` + `audio/`,
`<sNN-personid>/` per person (frames copied, or hardlinked with `--link`; `sequence.json`
rewritten onto the source clock), `shots/NN/{cameras,placement}.json`, `sequence-report.json`.
No network, no GPU, no paid service: ffmpeg/ffprobe on the source clip is the only subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from package_audio import wav_info  # noqa: E402

MANIFEST_SCHEMA = "wander.people/1"
SEQUENCE_SCHEMA = "wander.shot-sequence/1"
REPORT_SCHEMA = "wander.shot-sequence-report/1"
AUDIO_SCHEMA = "wander.audio/1"
# Merged sample times are canonicalised to this many decimals so a person's own sample time and the
# merged grid entry it produced compare equal, and every run index is exact rather than nearest.
TIME_DECIMALS = 9
TIME_EPS = 1e-6
# docs/audio.md: "The timeline duration must match the viewer's people/sequence duration within 0.1
# seconds."
AUDIO_TOLERANCE_SECONDS = 0.1


class SequenceError(ValueError):
    """Anything the operator has to fix before this package can be trusted."""


def fail(message: str):
    raise SequenceError(message)


def number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canon(seconds: float) -> float:
    return round(float(seconds), TIME_DECIMALS)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_inside(base: Path, name, what: str) -> Path:
    """A candidate-relative path that cannot escape its candidate directory."""
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        fail(f"{what}: expected a relative path inside the candidate, got {name!r}")
    resolved = (base / name).resolve()
    if base.resolve() != resolved and base.resolve() not in resolved.parents:
        fail(f"{what}: {name!r} escapes {base}")
    return resolved


# ---------------------------------------------------------------- source clip
def probe_source(video: Path) -> dict:
    """Duration and audio shape of the ORIGINAL clip, from ffprobe only."""
    if not video.is_file():
        fail(f"--source {video} does not exist")
    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,sample_rate,channels,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    doc = json.loads(raw)
    streams = doc.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not video_streams:
        fail(f"{video} has no video stream")
    candidates = [video_streams[0].get("duration"), doc.get("format", {}).get("duration")]
    duration = next((float(c) for c in candidates if c not in (None, "N/A")), 0.0)
    if not number(duration) or duration <= 0:
        fail(f"{video}: ffprobe reports no usable duration")
    return {
        "path": str(video),
        "sha256": sha256_file(video),
        "durationSeconds": duration,
        "videoCodec": video_streams[0].get("codec_name"),
        "hasAudio": bool(audio_streams),
        "audioCodec": audio_streams[0].get("codec_name") if audio_streams else None,
        "audioSampleRate": int(audio_streams[0]["sample_rate"])
        if audio_streams and audio_streams[0].get("sample_rate")
        else None,
        "audioChannels": audio_streams[0].get("channels") if audio_streams else None,
    }


def extract_original_audio(video: Path, dest: Path) -> dict:
    """The clip's whole soundtrack as PCM WAV at its NATIVE sample rate and channel count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        check=True,
    )
    if not dest.is_file():
        fail(f"ffmpeg wrote no audio to {dest}")
    return wav_info(dest)


# ---------------------------------------------------------------- shot list
def read_shot_list(path: Path) -> list[dict]:
    doc = json.loads(path.read_text())
    entries = doc.get("shots") if isinstance(doc, dict) else doc
    if not isinstance(entries, list) or not entries:
        fail(f"{path}: expected a non-empty JSON list of shots, or {{'shots': [...]}}")
    for entry in entries:
        if not isinstance(entry, dict):
            fail(f"{path}: every shot must be a JSON object")
    return entries


def find_source_mapping(entry: dict, manifest: dict, base: Path) -> tuple[dict | None, str]:
    """The recorded mapping for this shot, and where it was found."""
    mapping = entry.get("sourceMapping")
    if isinstance(mapping, dict):
        return mapping, "shot entry sourceMapping"
    state_path = entry.get("stateJson")
    if isinstance(state_path, str) and state_path:
        resolved = Path(state_path)
        if not resolved.is_absolute():
            resolved = (base / resolved).resolve()
        if not resolved.is_file():
            fail(f"stateJson {state_path} does not exist")
        state = json.loads(resolved.read_text())
        shot = ((state.get("stages") or {}).get("_run") or {}).get("shot") or {}
        mapping = shot.get("sourceMapping")
        if isinstance(mapping, dict):
            return mapping, f"{resolved} stages._run.shot.sourceMapping"
    mapping = manifest.get("sourceMapping")
    if isinstance(mapping, dict):
        return mapping, "candidate people.json sourceMapping"
    return None, "none recorded"


def resolve_window(
    entry: dict, mapping: dict | None, where: str, label: str
) -> tuple[float, float]:
    start, end = entry.get("sourceStart"), entry.get("sourceEnd")
    if not number(start) or not number(end):
        fail(f"{label}: sourceStart and sourceEnd must be finite numbers of seconds")
    if start < 0 or end <= start:
        fail(f"{label}: sourceStart {start} .. sourceEnd {end} is not a forward window")
    if mapping:
        for key, value, name in (
            ("sourceStartSeconds", start, "sourceStart"),
            ("sourceEndSeconds", end, "sourceEnd"),
        ):
            recorded = mapping.get(key)
            if recorded is None:
                continue
            if not number(recorded) or abs(float(recorded) - value) > TIME_EPS:
                fail(
                    f"{label}: {name} {value} disagrees with the recorded {key} {recorded!r} "
                    f"({where}). Fix the shot list or the recorded mapping; this tool will not "
                    f"pick one."
                )
    return float(start), float(end)


def resolve_trim_offset(entry: dict, mapping: dict | None, where: str, label: str, start: float):
    """The one measured/declared offset from `sourceStart` to the candidate's local zero."""
    explicit = entry.get("trimOffsetSeconds")
    if explicit is not None:
        if not number(explicit) or explicit < 0:
            fail(f"{label}: trimOffsetSeconds must be a finite, non-negative number of seconds")
        return float(explicit), "shot entry trimOffsetSeconds", True
    if mapping:
        recorded = mapping.get("trimOffsetSeconds")
        if recorded is not None:
            if not number(recorded) or recorded < 0:
                fail(f"{label}: recorded trimOffsetSeconds {recorded!r} is not usable ({where})")
            return float(recorded), f"{where} trimOffsetSeconds", True
        local_zero = mapping.get("localZeroSourceSeconds")
        if local_zero is not None:
            if not number(local_zero) or local_zero < start - TIME_EPS:
                fail(f"{label}: recorded localZeroSourceSeconds {local_zero!r} is before the shot")
            return float(local_zero) - start, f"{where} localZeroSourceSeconds", True
    fps = entry.get("trimOffsetFromShotCutsFps")
    if fps is not None:
        if not number(fps) or fps <= 0:
            fail(f"{label}: trimOffsetFromShotCutsFps must be a positive frame rate")
        return 0.5 / float(fps), f"derived 0.5/{fps} fps (scripts/shot_cuts.py:659)", False
    fail(
        f"{label}: no trim offset. scripts/shot_cuts.py::trim seeks to start + 0.5/fps and drops "
        f"two frames (scripts/shot_cuts.py:650-686), so the candidate's local zero is NOT "
        f"sourceStart and this tool will not guess it. Supply trimOffsetSeconds, a recorded "
        f"sourceMapping (shot entry, stateJson or the candidate people.json), or the explicit "
        f"opt-in trimOffsetFromShotCutsFps."
    )


def check_local_timeline(manifest: dict, label: str) -> list[float]:
    """The candidate's own shared grid, held to the rule src/person-visibility.ts:8-15 enforces."""
    timestamps = manifest.get("timestamps")
    duration = manifest.get("duration")
    if not isinstance(timestamps, list) or not timestamps:
        fail(f"{label}: people.json has no `timestamps` array")
    previous = None
    for i, value in enumerate(timestamps):
        if not number(value):
            fail(f"{label}: timestamps[{i}] = {value!r} is not a finite number")
        if previous is not None and value <= previous:
            fail(
                f"{label}: timestamps are not strictly increasing at index {i} "
                f"({previous} then {value})"
            )
        previous = float(value)
    if not number(duration) or duration <= timestamps[-1]:
        fail(
            f"{label}: duration {duration!r} must be greater than the last timestamp "
            f"{timestamps[-1]} (src/person-visibility.ts:12)"
        )
    return [float(t) for t in timestamps]


def load_shots(entries: list[dict], list_path: Path, source: dict) -> list[dict]:
    base = list_path.resolve().parent
    clip_duration = source["durationSeconds"]
    shots, previous = [], None
    for index, entry in enumerate(entries):
        label = f"shot {index}"
        candidate = entry.get("candidateDir")
        if not isinstance(candidate, str) or not candidate:
            fail(f"{label}: candidateDir is required")
        candidate_dir = Path(candidate)
        if not candidate_dir.is_absolute():
            candidate_dir = (base / candidate_dir).resolve()
        if not candidate_dir.is_dir():
            fail(f"{label}: candidateDir {candidate} is not a directory")
        people_path = candidate_dir / "people.json"
        if not people_path.is_file():
            fail(f"{label}: {people_path} does not exist")
        manifest = json.loads(people_path.read_text())
        if not isinstance(manifest, dict):
            fail(f"{label}: {people_path} is not a JSON object")
        world = entry.get("world")
        if not isinstance(world, str) or not world.strip():
            fail(f"{label}: world must be the viewer URL or id of this shot's spz")
        world_file = entry.get("worldFile")
        if isinstance(world_file, str) and world_file:
            resolved = Path(world_file)
            if not resolved.is_absolute():
                resolved = (base / resolved).resolve()
            if not resolved.is_file():
                fail(f"{label}: worldFile {world_file} does not exist")
            world_file = str(resolved)
        else:
            world_file = None

        mapping, where = find_source_mapping(entry, manifest, base)
        start, end = resolve_window(entry, mapping, where, label)
        offset, offset_source, offset_measured = resolve_trim_offset(
            entry, mapping, where, label, start
        )
        if end > clip_duration + TIME_EPS:
            fail(
                f"{label}: sourceEnd {end} is past the source clip's {clip_duration:.3f} s "
                f"({source['path']})"
            )
        if previous is not None and start < previous["sourceEnd"] - TIME_EPS:
            fail(
                f"{label}: starts at {start} s, inside shot {previous['index']} which runs to "
                f"{previous['sourceEnd']} s; shots must be ordered and must not overlap"
            )

        local = check_local_timeline(manifest, f"{label} ({people_path})")
        shift = start + offset
        mapped = [canon(shift + t) for t in local]
        if mapped[0] < canon(start) - TIME_EPS:
            fail(
                f"{label}: first sample maps to {mapped[0]} s, before sourceStart {start} s; the "
                f"trim offset ({offset_source}) is wrong"
            )
        if mapped[-1] >= canon(end) - TIME_EPS:
            fail(
                f"{label}: last sample maps to {mapped[-1]} s, which is not before sourceEnd "
                f"{end} s; the trim offset ({offset_source}) is wrong or the window is too short"
            )
        shot = dict(
            index=index,
            entry=entry,
            candidate=candidate_dir,
            candidateName=candidate,
            manifest=manifest,
            peoplePath=people_path,
            world=world,
            worldFile=world_file,
            sourceStart=start,
            sourceEnd=end,
            trimOffsetSeconds=offset,
            trimOffsetSource=offset_source,
            trimOffsetMeasured=offset_measured,
            sourceMappingFrom=where,
            localTimestamps=local,
            localDuration=float(manifest["duration"]),
            shift=shift,
            mapped=mapped,
            mappedEnd=canon(shift + float(manifest["duration"])),
        )
        shots.append(shot)
        previous = shot
    return shots


# ---------------------------------------------------------------- merged grid
def build_grid(shots: list[dict], duration: float) -> tuple[list[float], dict[float, int]]:
    """Every mapped sample time, plus a terminator at each cut so visibility stops there."""
    points: set[float] = set()
    for shot in shots:
        points.update(shot["mapped"])
    limit = canon(duration)
    for shot in shots:
        end = canon(shot["sourceEnd"])
        # `timestamps[run[1] + 1] ?? duration` (src/person-visibility.ts:29) ends the last interval.
        # At the end of the clip `duration` already supplies exactly the right value; anywhere else
        # the cut itself has to be on the grid, or the person bleeds into the next shot.
        if end < limit:
            points.add(end)
    grid = sorted(points)
    for i in range(1, len(grid)):
        if grid[i] - grid[i - 1] <= TIME_EPS:
            fail(
                f"merged samples {grid[i - 1]} and {grid[i]} are not separated; two shots map onto "
                f"the same source time"
            )
    if grid[-1] >= limit:
        fail(f"merged grid ends at {grid[-1]} s, not before the clip duration {duration} s")
    return grid, {t: i for i, t in enumerate(grid)}


def validate_runs(runs, samples: int, label: str) -> list[list[int]]:
    """The same shape src/person-visibility.ts:17-28 refuses to accept anything else in."""
    if not isinstance(runs, list) or not runs:
        fail(f"{label}: visibleSampleRuns must be a non-empty list")
    previous_end = -1
    out = []
    for run in runs:
        if (
            not isinstance(run, list)
            or len(run) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) for v in run)
            or run[0] < 0
            or run[1] < run[0]
            or run[1] >= samples
            or run[0] <= previous_end
        ):
            fail(f"{label}: {run!r} is not an ordered source sample range within {samples} samples")
        previous_end = run[1]
        out.append([int(run[0]), int(run[1])])
    return out


def person_runs(person: dict, shot: dict, label: str) -> tuple[list[list[int]], str]:
    samples = len(shot["localTimestamps"])
    runs = person.get("visibleSampleRuns")
    if runs is not None:
        return validate_runs(runs, samples, label), "candidate visibleSampleRuns"
    first, last = person.get("firstSample"), person.get("lastSample")
    if isinstance(first, int) and isinstance(last, int) and not isinstance(first, bool):
        return validate_runs([[first, last]], samples, label), "candidate firstSample/lastSample"
    # No declared visibility at all: the viewer would treat this person as visible for the WHOLE
    # merged clip (src/person-visibility.ts:7). Confine them to their own shot instead, and say so.
    return [[0, samples - 1]], "derived: whole shot (candidate declared no visibility)"


# ---------------------------------------------------------------- copying
def copy_tree(src: Path, dst: Path, link: bool, skip: set[str]) -> tuple[int, bool]:
    """Copy a person directory. `link` hardlinks the per-frame payload instead of duplicating it."""
    dst.mkdir(parents=True, exist_ok=True)
    count, linked = 0, link
    for item in sorted(src.rglob("*")):
        relative = item.relative_to(src)
        if relative.parts[0] in skip:
            continue
        target = dst / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        if link:
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
                linked = False
        else:
            shutil.copy2(item, target)
        count += 1
    return count, linked


# ---------------------------------------------------------------- packaging
def package(
    source: Path,
    shot_list: Path,
    out: Path,
    link: bool = False,
    force: bool = False,
    audio: bool = True,
) -> dict:
    source, shot_list, out = Path(source), Path(shot_list), Path(out)
    info = probe_source(source)
    shots = load_shots(read_shot_list(shot_list), shot_list, info)
    duration = info["durationSeconds"]
    grid, index_of = build_grid(shots, duration)

    if out.exists():
        if any(out.iterdir()):
            if not force:
                fail(f"{out} is not empty; pass --force to replace an earlier sequence package")
            if not (out / "sequence-report.json").is_file():
                fail(
                    f"{out} is not empty and holds no sequence-report.json, so it is not a package "
                    f"this tool wrote; remove it by hand or choose another --out"
                )
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    inputs = {str(source): info["sha256"]}
    people, id_map, shot_blocks = [], {}, []
    for shot in shots:
        prefix = f"s{shot['index']:02d}"
        candidate = shot["candidate"]
        inputs[str(shot["peoplePath"])] = sha256_file(shot["peoplePath"])
        shot_dir = out / "shots" / f"{shot['index']:02d}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        copied = {}
        for name in ("cameras.json", "placement.json"):
            src_file = candidate / name
            if src_file.is_file():
                shutil.copy2(src_file, shot_dir / name)
                inputs[str(src_file)] = sha256_file(src_file)
                copied[name] = f"shots/{shot['index']:02d}/{name}"
            else:
                copied[name] = None
        if copied["cameras.json"] is None:
            fail(f"shot {shot['index']}: {candidate / 'cameras.json'} does not exist")

        roster = shot["manifest"].get("people")
        if not isinstance(roster, list):
            fail(f"shot {shot['index']}: people.json has no `people` list")
        shot_people, linked_any = [], False
        for person in roster:
            if not isinstance(person, dict):
                fail(f"shot {shot['index']}: every entry of `people` must be an object")
            original_id = person.get("id")
            if not isinstance(original_id, str) or not original_id:
                fail(f"shot {shot['index']}: a person has no id")
            merged_id = f"{prefix}-{original_id}"
            if merged_id in id_map:
                fail(f"{merged_id} appears twice; candidate person ids must be unique per shot")
            label = f"shot {shot['index']} person {original_id}"
            sequence = person.get("sequence")
            directory = person.get("directory") or (
                sequence.rsplit("/", 1)[0]
                if isinstance(sequence, str) and "/" in sequence
                else None
            )
            if not isinstance(directory, str) or not directory:
                fail(f"{label}: neither `directory` nor a `sequence` path to take it from")
            src_dir = relative_inside(candidate, directory, label)
            if not src_dir.is_dir():
                fail(f"{label}: {src_dir} does not exist")
            seq_path = src_dir / "sequence.json"
            if not seq_path.is_file():
                fail(f"{label}: {seq_path} does not exist")
            seq = json.loads(seq_path.read_text())
            frames = seq.get("frames")
            if not isinstance(frames, list) or not frames:
                fail(f"{label}: sequence.json has no `frames`")

            runs, runs_from = person_runs(person, shot, label)
            merged_runs = []
            for run in runs:
                a, b = index_of[shot["mapped"][run[0]]], index_of[shot["mapped"][run[1]]]
                if b - a != run[1] - run[0]:
                    fail(
                        f"{label}: sample run {run} is not contiguous on the merged grid "
                        f"({a}..{b}); the shots overlap in source time"
                    )
                merged_runs.append([a, b])

            person_dir = out / merged_id
            count, linked = copy_tree(src_dir, person_dir, link, {"sequence.json"})
            linked_any = linked_any or (link and linked)
            for name in frames:
                if not (person_dir / name).is_file():
                    fail(f"{label}: frame {name} is missing from {src_dir}")

            local_ts = seq.get("timestamps")
            if isinstance(local_ts, list) and len(local_ts) == len(frames):
                if any(not number(t) for t in local_ts):
                    fail(f"{label}: sequence.json timestamps must all be finite")
                if any(local_ts[i] <= local_ts[i - 1] for i in range(1, len(local_ts))):
                    fail(f"{label}: sequence.json timestamps are not strictly increasing")
                mapped_ts = [canon(shot["shift"] + float(t)) for t in local_ts]
            else:
                fps = seq.get("fps") or shot["manifest"].get("fps") or 12
                if not number(fps) or fps <= 0:
                    fail(f"{label}: sequence.json has neither usable timestamps nor fps")
                mapped_ts = [canon(shot["shift"] + i / float(fps)) for i in range(len(frames))]
            local_duration = seq.get("duration")
            if not number(local_duration) or local_duration <= 0:
                local_duration = len(frames) / float(seq.get("fps") or 12)
            seq_out = dict(seq)
            seq_out.update(
                timestamps=mapped_ts,
                duration=canon(shot["shift"] + float(local_duration)),
                personId=merged_id,
                manifest="../people.json",
                sourceTimeline=dict(
                    schema=SEQUENCE_SCHEMA,
                    note="timestamps and duration are on the ORIGINAL clip's clock; the candidate's "
                    "own local values are this shift below them",
                    shotIndex=shot["index"],
                    shiftSeconds=canon(shot["shift"]),
                    sourceStartSeconds=shot["sourceStart"],
                    sourceEndSeconds=shot["sourceEnd"],
                    trimOffsetSeconds=shot["trimOffsetSeconds"],
                    trimOffsetSource=shot["trimOffsetSource"],
                    localDurationSeconds=float(local_duration),
                    candidate=shot["candidateName"],
                    originalPersonId=original_id,
                ),
            )
            (person_dir / "sequence.json").write_text(json.dumps(seq_out, indent=2))

            merged = dict(person)
            merged.update(
                id=merged_id,
                sequence=f"{merged_id}/sequence.json",
                directory=merged_id,
                label=f"{person.get('label') or original_id} (shot {shot['index']})",
                timestamps=mapped_ts,
                visibleSampleRuns=merged_runs,
                firstSample=merged_runs[0][0],
                lastSample=merged_runs[-1][1],
                shotIndex=shot["index"],
                originalId=original_id,
                candidate=shot["candidateName"],
                visibilitySource=runs_from,
                visibleSeconds=[
                    [grid[a], grid[b + 1] if b + 1 < len(grid) else duration]
                    for a, b in merged_runs
                ],
            )
            # `transform` is carried through untouched: it is the only per-person placement the
            # candidate solved, and a merged package must not invent a new one.
            people.append(merged)
            id_map[merged_id] = dict(
                id=original_id,
                shot=shot["index"],
                candidate=shot["candidateName"],
                frames=count,
            )
            shot_people.append(merged_id)

        primary = shot["manifest"].get("primary")
        shot_primary = f"{prefix}-{primary}" if isinstance(primary, str) and primary else None
        if shot_primary is not None and shot_primary not in id_map:
            shot_primary = shot_people[0] if shot_people else None
        first_index = index_of[shot["mapped"][0]]
        last_index = index_of[shot["mapped"][-1]]
        shot_blocks.append(
            dict(
                index=shot["index"],
                candidate=shot["candidateName"],
                world=shot["world"],
                sourceStart=shot["sourceStart"],
                sourceEnd=shot["sourceEnd"],
                placement=copied["placement.json"],
                cameras=copied["cameras.json"],
                primary=shot_primary,
                people=shot_people,
                sampleRange=[first_index, last_index],
                firstSampleSeconds=grid[first_index],
                lastSampleSeconds=grid[last_index],
                leadInSeconds=canon(grid[first_index] - shot["sourceStart"]),
                trimOffsetSeconds=shot["trimOffsetSeconds"],
                trimOffsetSource=shot["trimOffsetSource"],
                trimOffsetMeasured=shot["trimOffsetMeasured"],
                shiftSeconds=canon(shot["shift"]),
                localDurationSeconds=shot["localDuration"],
                sharedPlacement=shot["manifest"].get("sharedPlacement"),
                sharedScale=shot["manifest"].get("sharedScale"),
                floorFit=shot["manifest"].get("floorFit"),
                linkedFrames=linked_any,
            )
        )

    if not people:
        fail("no people in any shot; the merged package would have an empty cast")

    gaps = []
    cursor = 0.0
    for shot in shots:
        if shot["sourceStart"] - cursor > TIME_EPS:
            gaps.append(
                dict(
                    startSeconds=canon(cursor),
                    endSeconds=canon(shot["sourceStart"]),
                    seconds=canon(shot["sourceStart"] - cursor),
                    reason="not reconstructed: no shot covers this source time",
                )
            )
        cursor = shot["sourceEnd"]
    if duration - cursor > TIME_EPS:
        gaps.append(
            dict(
                startSeconds=canon(cursor),
                endSeconds=canon(duration),
                seconds=canon(duration - cursor),
                reason="not reconstructed: no shot covers this source time",
            )
        )

    manifest = dict(
        schema=MANIFEST_SCHEMA,
        clip=str(source),
        sourceSha256=info["sha256"],
        fps=shots[0]["manifest"].get("fps"),
        samples=len(grid),
        duration=duration,
        timestamps=grid,
        cameras=None,
        primary=next((p["id"] for p in people), None),
        peopleCount=len(people),
        people=people,
        shots=shot_blocks,
        personIdMap=id_map,
        shotSequence=dict(
            schema=SEQUENCE_SCHEMA,
            source=str(source),
            sourceDurationSeconds=duration,
            shotCount=len(shots),
            gaps=gaps,
            note="One package, several worlds. `shots` is ordered and non-overlapping; shot k is "
            "active for sourceStart <= t < sourceEnd. There is no top-level cameras.json or "
            "placement.json: the shots are separate solves and share no frame, so the viewer must "
            "take both from the active shot. `timestamps`/`duration` are the ORIGINAL clip's clock.",
            perShotState=[
                "worldUrl (fourd.html:1065)",
                "floorY (fourd.html:1230, 1334-1342)",
                "floorMap (fourd.html:1368)",
                "walkGrid (fourd.html:3293, 3428)",
                "scale0/pos0 (fourd.html:2457-2470)",
                "placement.json (fourd.html:1243-1259)",
                "cameras.json",
            ],
        ),
        coordinates="Per shot. Each shot keeps its candidate's own camera-0 frame; nothing here "
        "claims the shots share one world frame.",
        backwardCompatible="A viewer that ignores `shots` still loads the whole cast over one "
        "world, each person visible only inside their own shot's window (fourd.html:1556-1565 "
        "reads only the keys it knows).",
    )
    (out / "people.json").write_text(json.dumps(manifest, indent=1))

    audio_report = dict(written=False, reason="--no-audio" if not audio else None)
    if audio:
        if not info["hasAudio"]:
            audio_manifest = dict(
                schema=AUDIO_SCHEMA,
                source=dict(hasAudio=False),
                timeline=dict(durationSeconds=duration),
                defaultMode="original",
                provenance=dict(
                    note="The original clip carries no audio stream; this package declares no "
                    "soundtrack rather than inventing one.",
                    sourceClip=str(source),
                    sourceSha256=info["sha256"],
                ),
            )
            audio_report = dict(
                written=True, hasSoundtrack=False, reason="source clip has no audio stream"
            )
        else:
            wav = out / "audio" / "original.wav"
            details = extract_original_audio(source, wav)
            if abs(details["durationSeconds"] - duration) > AUDIO_TOLERANCE_SECONDS:
                fail(
                    f"extracted soundtrack is {details['durationSeconds']:.3f} s but the merged "
                    f"timeline is {duration:.3f} s; docs/audio.md requires these to agree within "
                    f"{AUDIO_TOLERANCE_SECONDS} s"
                )
            audio_manifest = dict(
                schema=AUDIO_SCHEMA,
                # docs/audio.md: hasAudio describes the PACKAGED video, and stays false when the
                # sound is recovered from the source file into an external original.
                source=dict(hasAudio=False),
                timeline=dict(durationSeconds=duration),
                original=dict(url="audio/original.wav", offsetSeconds=0),
                defaultMode="original",
                provenance=dict(
                    note="The whole original clip's soundtrack, extracted with ffmpeg at its native "
                    "sample rate and played straight through the cuts. Offset 0: the merged people "
                    "timeline IS the original clip's clock.",
                    sourceClip=str(source),
                    sourceSha256=info["sha256"],
                    sampleRate=details["sampleRate"],
                    channels=details["channels"],
                    sha256=details["sha256"],
                ),
            )
            audio_report = dict(
                written=True,
                hasSoundtrack=True,
                url="audio/original.wav",
                durationSeconds=details["durationSeconds"],
                sampleRate=details["sampleRate"],
                channels=details["channels"],
                sha256=details["sha256"],
            )
        (out / "audio.json").write_text(json.dumps(audio_manifest, indent=1))

    covered = sum(s["sourceEnd"] - s["sourceStart"] for s in shots)
    report = dict(
        schema=REPORT_SCHEMA,
        tool="scripts/package_shot_sequence.py",
        source=info,
        shotList=dict(path=str(shot_list), sha256=sha256_file(shot_list)),
        output=dict(
            path=str(out),
            samples=len(grid),
            durationSeconds=duration,
            peopleCount=len(people),
            requestedHardlinks=link,
        ),
        shots=[
            dict(
                index=block["index"],
                candidate=block["candidate"],
                world=block["world"],
                sourceStart=block["sourceStart"],
                sourceEnd=block["sourceEnd"],
                mappedInterval=[block["firstSampleSeconds"], block["lastSampleSeconds"]],
                sampleRange=block["sampleRange"],
                peopleCount=len(block["people"]),
                people=block["people"],
                trimOffsetSeconds=block["trimOffsetSeconds"],
                trimOffsetSource=block["trimOffsetSource"],
                trimOffsetMeasured=block["trimOffsetMeasured"],
                leadInSeconds=block["leadInSeconds"],
                placement=block["placement"],
                cameras=block["cameras"],
                linkedFrames=block["linkedFrames"],
            )
            for block in shot_blocks
        ],
        gaps=gaps,
        coverage=dict(
            coveredSeconds=canon(covered),
            gapSeconds=canon(duration - covered),
            fraction=canon(covered / duration) if duration else None,
            note="Gap seconds are source time no shot reconstructs. Disclose them as 'not "
            "reconstructed'; nothing in this package fills them.",
        ),
        audio=audio_report,
        inputs=inputs,
    )
    (out / "sequence-report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, type=Path, help="the ORIGINAL clip, with its cuts")
    ap.add_argument("--shots", required=True, type=Path, help="ordered shot list (JSON)")
    ap.add_argument("--out", required=True, type=Path, help="merged package directory")
    ap.add_argument(
        "--link",
        action="store_true",
        help="hardlink per-frame payload instead of copying it (same filesystem only)",
    )
    ap.add_argument("--force", action="store_true", help="replace an earlier sequence package")
    ap.add_argument("--no-audio", action="store_true", help="skip audio.json and the WAV extract")
    a = ap.parse_args()
    try:
        report = package(a.source, a.shots, a.out, a.link, a.force, not a.no_audio)
    except SequenceError as error:
        print(f"package_shot_sequence: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            dict(
                out=report["output"]["path"],
                samples=report["output"]["samples"],
                duration=report["output"]["durationSeconds"],
                people=report["output"]["peopleCount"],
                shots=[
                    f"{s['index']}: {s['sourceStart']:.3f}-{s['sourceEnd']:.3f} s "
                    f"{s['world']} ({s['peopleCount']} people, offset {s['trimOffsetSeconds']:.4f} s "
                    f"{'measured' if s['trimOffsetMeasured'] else 'DERIVED'})"
                    for s in report["shots"]
                ],
                gaps=[f"{g['startSeconds']:.3f}-{g['endSeconds']:.3f} s" for g in report["gaps"]],
                audio=report["audio"],
            ),
            indent=1,
        )
    )
    for gap in report["gaps"]:
        print(
            f"NOT RECONSTRUCTED: {gap['startSeconds']:.3f}-{gap['endSeconds']:.3f} s "
            f"({gap['seconds']:.3f} s of the source clip)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
