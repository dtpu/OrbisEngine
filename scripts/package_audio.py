#!/usr/bin/env python3
"""Package reviewed dialogue tracks. This tool never separates audio or infers who is speaking.

Inputs are either supplied isolated stems (--config) or a scripts/separate_audio.py output
directory (--separation), whose voice track is a *model estimate* and therefore needs the
operator's --voice-reviewed-by attestation, recorded in the manifest, before anything is written.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import wave


def fail(message):
    raise ValueError(message)


def wav_info(path, mono=False):
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE" or w.getsampwidth() not in (2, 3, 4):
            fail(f"{path.name}: expected uncompressed PCM WAV (16/24/32 bit)")
        channels, rate, samples = w.getnchannels(), w.getframerate(), w.getnframes()
    if channels not in (1, 2) or (mono and channels != 1):
        fail(f"{path.name}: dialogue must be mono; ambience may be mono or stereo")
    if not samples or not rate:
        fail(f"{path.name}: empty audio")
    return {
        "channels": channels,
        "sampleRate": rate,
        "samples": samples,
        "durationSeconds": samples / rate,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def separation_config(
    separation,
    person_id,
    anchor_file,
    sequence_file=None,
    offset_body_heights=None,
    attributed_by=None,
    voice_id="voice",
    bed_id="bed",
):
    """Build the usual track configuration from a scripts/separate_audio.py output directory.

    The separated voice is a model estimate, so its provenance says so. `reviewed` is set here
    only because the caller passed the operator attestation flag; nothing in the tool decides it.
    """
    separation = Path(separation).resolve()
    receipt_path = separation / "receipt.json"
    if not receipt_path.is_file():
        fail("Separation directory needs the receipt.json written beside voice.wav and bed.wav")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "wander.audio-separation/1":
        fail("Unrecognised separation receipt schema")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, str) or not provenance:
        fail("Separation receipt must carry a provenance string")
    if attributed_by:
        provenance = f"{provenance};attributed-by:{attributed_by}"
    voice = {
        "id": voice_id,
        "kind": "dialogue",
        "file": receipt["outputs"]["voice"]["file"],
        "personId": person_id,
        "reviewed": True,
        "provenance": provenance,
        "anchorFile": str(Path(anchor_file).resolve()),
        "offsetBodyHeights": offset_body_heights or [0, 0, 0],
    }
    if sequence_file:
        voice["sequenceFile"] = str(Path(sequence_file).resolve())
    bed = {"id": bed_id, "kind": "ambience", "file": receipt["outputs"]["bed"]["file"]}
    return {"personIds": [person_id], "tracks": [voice, bed]}, separation, receipt


def prepare(config_path, world, reviewed=False, check=False, config=None, reviewed_by=None):
    if not reviewed:
        fail(
            "Human review required: pass --reviewed only after checking identities, all words/overlap, head alignment, and the complete replacement mix"
        )
    config_path, world = Path(config_path).resolve(), Path(world).resolve()
    # A supplied dict resolves its relative files against config_path as a directory.
    base = config_path if config is not None else config_path.parent
    if config is None:
        config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        fail("Configuration must be a JSON object")
    manifest_path = world / "audio.json"
    original = json.loads(manifest_path.read_text())
    if (
        not isinstance(original, dict)
        or not isinstance(original.get("source"), dict)
        or not isinstance(original["source"].get("hasAudio"), bool)
    ):
        fail("Original manifest must declare source.hasAudio")
    duration = original.get("timeline", {}).get("durationSeconds")
    if original.get("schema") != "wander.audio/1" or not number(duration) or duration <= 0:
        fail("World requires an existing wander.audio/1 original-fallback manifest")
    fallback = original.get("original", {})
    fallback_url = fallback.get("url", "")
    if (
        not isinstance(fallback_url, str)
        or not fallback_url
        or "://" in fallback_url
        or fallback_url.startswith("/")
    ):
        fail("Original fallback must reference an existing relative local WAV")
    fallback_path = (world / fallback_url).resolve()
    if not fallback_path.is_file():
        fail("Original fallback file is missing")
    fallback_info = wav_info(fallback_path)
    if (
        abs(fallback_info["durationSeconds"] - duration) > 0.05
        or fallback.get("offsetSeconds", 0) != 0
    ):
        fail("Original fallback must match the full normalized clip, with offset zero")
    person_ids = config.get("personIds")
    if (
        not isinstance(person_ids, list)
        or not person_ids
        or not all(isinstance(x, str) and x for x in person_ids)
    ):
        fail(
            "personIds must list actual IDs from the target viewer; the tool does not infer identities"
        )
    tracks = config.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        fail("Supply dialogue tracks and any required dialogue-free ambience bed")
    files, output, identities, ids = [], [], set(), set()
    for track in tracks:
        if not isinstance(track, dict):
            fail("Each track must be a JSON object")
        ident = track.get("id", "")
        if (
            not isinstance(ident, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ident)
            or ident in ids
        ):
            fail("Track IDs must be unique safe names")
        ids.add(ident)
        kind = track.get("kind")
        if kind not in ("dialogue", "ambience"):
            fail("Track kind must be dialogue or ambience")
        path = (base / track["file"]).resolve()
        info = wav_info(path, mono=kind == "dialogue")
        if (
            info["sampleRate"] != fallback_info["sampleRate"]
            or abs(info["durationSeconds"] - duration) > 0.05
        ):
            fail(
                f"{ident}: use the original sample rate and full clip length, padding silence to preserve timing"
            )
        if path == fallback_path or info["sha256"] == fallback_info["sha256"]:
            fail("The original mix cannot also be a speaker stem or residual bed")
        dest = f"{ident}.wav"
        files.append((path, dest))
        item = {"id": ident, "kind": kind, "url": dest, "offsetSeconds": 0, **info}
        if kind == "dialogue":
            if not isinstance(
                track.get("provenance", "user-supplied-reviewed"), str
            ) or not track.get("provenance", "user-supplied-reviewed"):
                fail(f"{ident}: provenance must describe the supplied stem")
            person = track.get("personId")
            if (
                person not in person_ids
                or person in identities
                or track.get("reviewed") is not True
            ):
                fail(f"{ident}: needs a reviewed, distinct personId from personIds")
            identities.add(person)
            anchor = json.loads((base / track["anchorFile"]).read_text())
            positions = anchor.get("positions", anchor.get("eyes"))
            times = anchor.get("times")
            if times is None and track.get("sequenceFile"):
                sequence = json.loads((base / track["sequenceFile"]).read_text())
                times = sequence.get("timestamps")
            if (
                not isinstance(positions, list)
                or not positions
                or not isinstance(times, list)
                or len(positions) != len(times)
            ):
                fail(
                    f"{ident}: head positions need matching explicit times (or sequenceFile timestamps)"
                )
            if any(
                not isinstance(p, list) or len(p) != 3 or not all(number(v) for v in p)
                for p in positions
            ):
                fail(f"{ident}: head positions must be finite local XYZ triples")
            if any(
                not number(t) or t < 0 or t > duration or (i and t <= times[i - 1])
                for i, t in enumerate(times)
            ):
                fail(f"{ident}: head times must increase within the clip")
            if times[0] != 0 or (
                len(times) > 1 and duration - times[-1] > max(0.001, times[-1] - times[-2]) + 0.001
            ):
                fail(
                    f"{ident}: head timeline must cover the clip, including its final sampled frame"
                )
            if len(times) == 1:
                fail(f"{ident}: provide a full head timeline, not a guessed static source")
            offset = track.get("offsetBodyHeights", [0, 0, 0])
            if (
                not isinstance(offset, list)
                or len(offset) != 3
                or not all(number(v) for v in offset)
            ):
                fail(f"{ident}: offsetBodyHeights must be a finite XYZ triple")
            anchor_name = f"{ident}-head.json"
            anchor_bytes = (
                json.dumps({"positions": positions, "times": times}, sort_keys=True) + "\n"
            ).encode()
            files.append((anchor_bytes, anchor_name))
            item.update(
                personId=person,
                reviewed=True,
                provenance=track.get("provenance", "user-supplied-reviewed"),
                anchor={"url": anchor_name, "space": "person-local", "offsetBodyHeights": offset},
            )
        output.append(item)
    if not identities:
        fail("At least one reviewed dialogue speaker is required")
    # Content-addressed package prevents rewriting files used by an active viewer.
    digest = hashlib.sha256(json.dumps(output, sort_keys=True).encode())
    for source, name in files:
        digest.update(name.encode())
        if isinstance(source, bytes):
            digest.update(source)
    prefix = "audio-reviewed/" + digest.hexdigest()[:20]
    for item in output:
        item["url"] = prefix + "/" + item["url"]
        if "anchor" in item:
            item["anchor"]["url"] = prefix + "/" + item["anchor"]["url"]
    packaged = {
        **original,
        "tracks": output,
        "spatialMixComplete": True,
        "defaultMode": "original",
        "review": {
            "method": "explicit-human-review",
            "scope": "speaker identity, complete mix, overlaps, timing, head coordinates",
            **({"reviewedBy": reviewed_by} if reviewed_by else {}),
        },
    }
    if check:
        return packaged
    target = world / prefix
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".audio-package-", dir=world))
    try:
        for source, name in files:
            if isinstance(source, bytes):
                (temp / name).write_bytes(source)
            else:
                shutil.copyfile(source, temp / name)
        for item in output:
            copied = temp / (item["id"] + ".wav")
            if hashlib.sha256(copied.read_bytes()).hexdigest() != item["sha256"]:
                fail("Input audio changed during packaging; original manifest remains unchanged")
        if target.exists():
            for _, name in files:
                if (temp / name).read_bytes() != (target / name).read_bytes():
                    fail("Existing package content mismatch; refusing replacement")
        else:
            os.replace(temp, target)
        fd, name = tempfile.mkstemp(prefix=".audio-manifest-", dir=world)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(packaged, handle, indent=2)
                handle.write("\n")
            os.replace(name, manifest_path)
        finally:
            Path(name).unlink(missing_ok=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return packaged


def triple(text):
    values = [float(v) for v in text.split(",")]
    if len(values) != 3:
        return fail("--offset-body-heights needs three comma-separated numbers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Supplied-stem configuration file")
    source.add_argument(
        "--separation",
        type=Path,
        help="scripts/separate_audio.py output directory (voice.wav, bed.wav, receipt.json)",
    )
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--person-id", help="With --separation: the viewer person ID to anchor to")
    parser.add_argument(
        "--anchor-file", type=Path, help="With --separation: that person's head.json"
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        help="With --separation: that person's sequence.json, when head.json carries no times",
    )
    parser.add_argument("--offset-body-heights", type=triple, help="Local XYZ, e.g. 0,0.05,0.08")
    parser.add_argument(
        "--attributed-by",
        help="How the voice was attributed to this person, recorded in the track provenance",
    )
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="Explicitly attest human review of identity, timing, anchors and complete replacement mix",
    )
    parser.add_argument(
        "--voice-reviewed-by",
        metavar="NAME",
        help="Operator attesting, after listening, that a separated voice estimate is correctly "
        "attributed and complete. Required with --separation; the tool never asserts review itself",
    )
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    config, base = None, args.config
    try:
        if args.separation:
            if not args.person_id or not args.anchor_file:
                fail("--separation needs --person-id and --anchor-file")
            if not args.voice_reviewed_by:
                fail(
                    "A separated voice is a model estimate: pass --voice-reviewed-by NAME only "
                    "after listening to the voice, the bed and the complete replacement mix"
                )
            config, base, _ = separation_config(
                args.separation,
                args.person_id,
                args.anchor_file,
                args.sequence_file,
                args.offset_body_heights,
                args.attributed_by,
            )
        elif args.voice_reviewed_by:
            fail("--voice-reviewed-by applies to --separation; use --reviewed with --config")
        manifest = prepare(
            base,
            args.world,
            args.reviewed or bool(args.voice_reviewed_by),
            args.check,
            config=config,
            reviewed_by=args.voice_reviewed_by,
        )
    except (ValueError, KeyError, OSError, EOFError, wave.Error) as error:
        parser.exit(2, f"Audio package rejected: {error}\n")
    print(
        json.dumps(
            {
                "written": not args.check,
                "tracks": len(manifest["tracks"]),
                "manifest": str(args.world / "audio.json"),
                "defaultMode": "original",
                "reviewedBy": args.voice_reviewed_by,
                "provenance": [
                    t.get("provenance") for t in manifest["tracks"] if "provenance" in t
                ],
                "note": "Prepared for spatial audition; this tool performs no separation or identity inference",
            }
        )
    )


if __name__ == "__main__":
    main()
