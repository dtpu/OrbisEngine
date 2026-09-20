#!/usr/bin/env python3
"""Owned synthetic fixtures only: hand-written candidates and one ffmpeg lavfi clip.

No network, no model, no GPU, no real footage. The clip is generated locally from lavfi sources;
the "reconstructions" are a few bytes of fake PLY each, because this tool never reads splat data.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from package_shot_sequence import SequenceError, package

SOURCE_SECONDS = 6.0


def make_source(path: Path, seconds: float = SOURCE_SECONDS, audio: bool = True):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x48:rate=10:duration={seconds}",
    ]
    if audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=16000:duration={seconds}",
            "-c:a",
            "aac",
        ]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(command, check=True)


def make_candidate(root: Path, name: str, timestamps: list[float], duration: float, people: list):
    """A minimal `public/worlds/<name>-shotNN-4d/` as package_multiperson.py would leave it."""
    world = root / name
    world.mkdir(parents=True)
    (world / "cameras.json").write_text(json.dumps({"coordinates": "fixture", "cameras": []}))
    entries = []
    for person in people:
        directory = person["id"]
        person_dir = world / directory
        person_dir.mkdir()
        frames = [f"frame_{i:04d}.ply" for i in range(len(person["timestamps"]))]
        for frame in frames:
            (person_dir / frame).write_bytes(b"ply\nfixture\n")
        (person_dir / "sequence.json").write_text(
            json.dumps(
                {
                    "frames": frames,
                    "count": len(frames),
                    "fps": 4,
                    "duration": duration,
                    "timestamps": person["timestamps"],
                    "sourceIndices": list(range(len(frames))),
                }
            )
        )
        entry = {
            "id": person["id"],
            "sequence": f"{directory}/sequence.json",
            "directory": directory,
            "label": person["id"],
            "frames": len(frames),
            "fps": 4,
            "timestamps": person["timestamps"],
            "transform": person.get(
                "transform",
                {
                    "translation": [0.0, 0.0, 0.0],
                    "quaternionXYZW": [0.0, 0.0, 0.0, 1.0],
                    "scale": 1.0,
                },
            ),
        }
        if "runs" in person:
            entry["visibleSampleRuns"] = person["runs"]
            entry["firstSample"] = person["runs"][0][0]
            entry["lastSample"] = person["runs"][-1][1]
        entries.append(entry)
    (world / "people.json").write_text(
        json.dumps(
            {
                "schema": "wander.people/1",
                "fps": 4,
                "samples": len(timestamps),
                "duration": duration,
                "timestamps": timestamps,
                "cameras": "cameras.json",
                "primary": people[0]["id"],
                "peopleCount": len(people),
                "people": entries,
                "sharedPlacement": {"bodyHeightUnits": 0.7},
            }
        )
    )
    return world


def person_visibility(runs, timestamps, duration):
    """src/person-visibility.ts:2-33, re-implemented here so the merged runs are checked, not trusted.

    Lines 8-15: a null `runs` means always visible, and a grid that is empty, non-finite, not
    strictly increasing, or not shorter than `duration` is rejected outright.
    Lines 17-30: each run is an ordered, non-overlapping [first, last] index pair, and becomes the
    half-open interval [timestamps[run[0]], timestamps[run[1] + 1] ?? duration].
    Lines 31-32: visible when some interval contains the time.
    """
    if runs is None:
        return lambda time: True
    if (
        not timestamps
        or not math.isfinite(duration)
        or any(
            not math.isfinite(t) or (i > 0 and t <= timestamps[i - 1])
            for i, t in enumerate(timestamps)
        )
        or duration <= timestamps[-1]
    ):
        raise AssertionError("Person visibility requires a valid shared source timeline")
    previous_end = -1
    intervals = []
    for run in runs:
        if (
            len(run) != 2
            or any(not isinstance(index, int) or isinstance(index, bool) for index in run)
            or run[0] < 0
            or run[1] < run[0]
            or run[1] >= len(timestamps)
            or run[0] <= previous_end
        ):
            raise AssertionError("Person visibility runs must be ordered source sample ranges")
        previous_end = run[1]
        intervals.append(
            (
                timestamps[run[0]],
                timestamps[run[1] + 1] if run[1] + 1 < len(timestamps) else duration,
            )
        )
    return lambda time: any(start <= time < end for start, end in intervals)


class ShotSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.source = Path(cls.shared.name) / "source.mp4"
        make_source(cls.source)
        cls.silent = Path(cls.shared.name) / "silent.mp4"
        make_source(cls.silent, audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "merged-4d"

        # shot 0: two people, the second only in the middle of the shot
        self.shot0 = make_candidate(
            self.root,
            "clip-shot00-4d",
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            1.6,
            [
                {
                    "id": "person",
                    "timestamps": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                    "runs": [[0, 6]],
                },
                {"id": "person_01", "timestamps": [0.5, 0.75, 1.0], "runs": [[2, 4]]},
            ],
        )
        (self.shot0 / "placement.json").write_text(json.dumps({"floorY": -0.5, "samples": 7}))
        # shot 1: one person, offset derived from shot_cuts.py::trim at 20 fps
        self.shot1 = make_candidate(
            self.root,
            "clip-shot01-4d",
            [0.0, 0.25, 0.5, 0.75, 1.0],
            1.25,
            [{"id": "person", "timestamps": [0.0, 0.25, 0.5, 0.75, 1.0], "runs": [[0, 4]]}],
        )
        # shot 2: one person with NO declared visibility, ending exactly at the clip's end
        self.shot2 = make_candidate(
            self.root,
            "clip-shot02-4d",
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
            1.4,
            [{"id": "person", "timestamps": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]}],
        )
        self.entries = [
            {
                "candidateDir": str(self.shot0),
                "world": "/marble-clip-shot00-clean.spz",
                "sourceStart": 0.0,
                "sourceEnd": 2.0,
                "trimOffsetSeconds": 0.05,
            },
            {
                "candidateDir": str(self.shot1),
                "world": "/marble-clip-shot01-clean.spz",
                "sourceStart": 2.0,
                "sourceEnd": 4.0,
                "trimOffsetFromShotCutsFps": 20.0,
            },
            {
                "candidateDir": str(self.shot2),
                "world": "/marble-clip-shot02-clean.spz",
                "sourceStart": 4.5,
                "sourceEnd": 6.0,
                "sourceMapping": {
                    "sourceStartSeconds": 4.5,
                    "sourceEndSeconds": 6.0,
                    "trimOffsetSeconds": 0.05,
                },
            },
        ]

    def write_list(self, entries=None, name="shots.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps({"shots": entries if entries is not None else self.entries}))
        return path

    def run_package(self, entries=None, **kwargs) -> dict:
        return package(self.source, self.write_list(entries), self.out, **kwargs)

    def merged(self) -> dict:
        return json.loads((self.out / "people.json").read_text())

    # ---------------------------------------------------------------- timeline
    def test_merged_timeline_is_strictly_increasing_and_shorter_than_duration(self):
        self.run_package()
        manifest = self.merged()
        timestamps = manifest["timestamps"]
        self.assertEqual(manifest["schema"], "wander.people/1")
        self.assertTrue(
            all(b > a for a, b in zip(timestamps, timestamps[1:])),
            "merged timestamps must be strictly increasing (src/person-visibility.ts:11)",
        )
        self.assertGreater(
            manifest["duration"],
            timestamps[-1],
            "duration must exceed the last timestamp (src/person-visibility.ts:12)",
        )
        self.assertAlmostEqual(manifest["duration"], SOURCE_SECONDS, places=2)
        self.assertEqual(manifest["samples"], len(timestamps))
        # every mapped sample plus one terminator per cut that is not the end of the clip
        self.assertEqual(len(timestamps), 7 + 5 + 6 + 2)
        self.assertIn(2.0, timestamps)
        self.assertIn(4.0, timestamps)
        self.assertNotIn(6.0, timestamps)

    def test_time_mapping_uses_each_shot_s_own_trim_offset(self):
        report = self.run_package()
        by_index = {shot["index"]: shot for shot in report["shots"]}
        self.assertAlmostEqual(by_index[0]["mappedInterval"][0], 0.05)
        self.assertAlmostEqual(by_index[0]["mappedInterval"][1], 1.55)
        self.assertAlmostEqual(by_index[1]["trimOffsetSeconds"], 0.025)
        self.assertFalse(by_index[1]["trimOffsetMeasured"])
        self.assertAlmostEqual(by_index[1]["mappedInterval"][0], 2.025)
        self.assertTrue(by_index[2]["trimOffsetMeasured"])
        self.assertAlmostEqual(by_index[2]["mappedInterval"][0], 4.55)
        self.assertEqual(
            [shot["world"] for shot in report["shots"]],
            [entry["world"] for entry in self.entries],
        )

    # ---------------------------------------------------------------- visibility
    def test_every_person_is_visible_only_inside_their_own_shot_window(self):
        self.run_package()
        manifest = self.merged()
        windows = {
            shot["index"]: (shot["sourceStart"], shot["sourceEnd"]) for shot in manifest["shots"]
        }
        samples = [i * SOURCE_SECONDS / 600 for i in range(600)]
        for person in manifest["people"]:
            start, end = windows[person["shotIndex"]]
            visible = person_visibility(
                person["visibleSampleRuns"], manifest["timestamps"], manifest["duration"]
            )
            seen = [t for t in samples if visible(t)]
            self.assertTrue(seen, f"{person['id']} is never visible")
            self.assertTrue(
                all(start <= t < end for t in seen),
                f"{person['id']} is visible outside [{start}, {end})",
            )
            self.assertTrue(
                all(not visible(t) for t in (start - 0.01, end, end + 0.01) if t >= 0),
                f"{person['id']} leaks across its own cut",
            )

    def test_a_person_absent_for_part_of_a_shot_keeps_their_gap(self):
        self.run_package()
        manifest = self.merged()
        extra = next(p for p in manifest["people"] if p["originalId"] == "person_01")
        visible = person_visibility(
            extra["visibleSampleRuns"], manifest["timestamps"], manifest["duration"]
        )
        self.assertFalse(visible(0.1))
        self.assertTrue(visible(0.6))
        self.assertFalse(visible(1.4))

    def test_a_candidate_without_declared_visibility_is_confined_to_its_shot(self):
        self.run_package()
        manifest = self.merged()
        last = next(p for p in manifest["people"] if p["shotIndex"] == 2)
        self.assertIn("derived", last["visibilitySource"])
        visible = person_visibility(
            last["visibleSampleRuns"], manifest["timestamps"], manifest["duration"]
        )
        self.assertFalse(visible(3.0))
        self.assertTrue(visible(5.0))
        # the shot runs to the end of the clip, so `?? duration` closes the last interval
        self.assertTrue(visible(5.99))

    # ---------------------------------------------------------------- package shape
    def test_ids_are_unique_and_map_back_to_their_candidate(self):
        self.run_package()
        manifest = self.merged()
        ids = [person["id"] for person in manifest["people"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, ["s00-person", "s00-person_01", "s01-person", "s02-person"])
        self.assertEqual(manifest["personIdMap"]["s01-person"]["id"], "person")
        self.assertEqual(manifest["personIdMap"]["s01-person"]["shot"], 1)
        for person in manifest["people"]:
            sequence = self.out / person["sequence"]
            self.assertTrue(sequence.is_file())
            seq = json.loads(sequence.read_text())
            self.assertEqual(len(seq["timestamps"]), len(seq["frames"]))
            self.assertEqual(seq["personId"], person["id"])
            for frame in seq["frames"]:
                self.assertTrue((sequence.parent / frame).is_file())
            # the sequence clock is the source clock, so applyTime() steps every person on it
            window = manifest["shots"][person["shotIndex"]]
            self.assertGreaterEqual(seq["timestamps"][0], window["sourceStart"])
            self.assertLess(seq["timestamps"][-1], window["sourceEnd"])
            self.assertEqual(person["transform"]["scale"], 1.0)

    def test_shots_block_carries_per_shot_cameras_and_placement(self):
        self.run_package()
        manifest = self.merged()
        self.assertEqual(len(manifest["shots"]), 3)
        self.assertEqual(manifest["shots"][0]["placement"], "shots/00/placement.json")
        self.assertIsNone(manifest["shots"][1]["placement"])
        for shot in manifest["shots"]:
            self.assertEqual(shot["cameras"], f"shots/{shot['index']:02d}/cameras.json")
            self.assertTrue((self.out / shot["cameras"]).is_file())
        self.assertTrue((self.out / "shots/00/placement.json").is_file())
        # the merged package deliberately has no top-level cameras/placement to mistake for shared
        self.assertFalse((self.out / "cameras.json").exists())
        self.assertFalse((self.out / "placement.json").exists())
        self.assertIsNone(manifest["cameras"])

    def test_gaps_between_shots_are_reported_as_not_reconstructed(self):
        report = self.run_package()
        self.assertEqual(len(report["gaps"]), 1)
        gap = report["gaps"][0]
        self.assertAlmostEqual(gap["startSeconds"], 4.0)
        self.assertAlmostEqual(gap["endSeconds"], 4.5)
        self.assertIn("not reconstructed", gap["reason"])
        self.assertAlmostEqual(report["coverage"]["gapSeconds"], 0.5)
        self.assertEqual(report["gaps"], self.merged()["shotSequence"]["gaps"])
        self.assertIn(str(self.source), report["inputs"])
        self.assertIn(str(self.shot1 / "people.json"), report["inputs"])

    def test_hardlinks_share_the_candidate_frames(self):
        self.run_package(link=True)
        frame = self.out / "s00-person" / "frame_0000.ply"
        self.assertEqual(frame.stat().st_ino, (self.shot0 / "person/frame_0000.ply").stat().st_ino)
        # sequence.json is rewritten, so it must never share an inode with the candidate's
        self.assertNotEqual(
            (self.out / "s00-person/sequence.json").stat().st_ino,
            (self.shot0 / "person/sequence.json").stat().st_ino,
        )

    # ---------------------------------------------------------------- audio
    def test_audio_manifest_covers_the_whole_original_clip(self):
        report = self.run_package()
        manifest = json.loads((self.out / "audio.json").read_text())
        merged_duration = self.merged()["duration"]
        self.assertEqual(manifest["schema"], "wander.audio/1")
        self.assertIs(manifest["source"]["hasAudio"], False)
        self.assertEqual(manifest["original"], {"url": "audio/original.wav", "offsetSeconds": 0})
        self.assertLessEqual(
            abs(manifest["timeline"]["durationSeconds"] - merged_duration),
            0.1,
            "docs/audio.md requires the audio timeline to match the people duration within 0.1 s",
        )
        wav = self.out / "audio/original.wav"
        self.assertTrue(wav.is_file())
        self.assertLessEqual(abs(report["audio"]["durationSeconds"] - merged_duration), 0.1)
        self.assertEqual(report["audio"]["sampleRate"], 16000)

    def test_a_silent_source_declares_no_soundtrack_instead_of_inventing_one(self):
        report = package(self.silent, self.write_list(), self.out)
        manifest = json.loads((self.out / "audio.json").read_text())
        self.assertIs(manifest["source"]["hasAudio"], False)
        self.assertNotIn("original", manifest)
        self.assertFalse(report["audio"]["hasSoundtrack"])
        self.assertFalse((self.out / "audio").exists())

    def test_no_audio_skips_the_manifest(self):
        self.run_package(audio=False)
        self.assertFalse((self.out / "audio.json").exists())

    # ---------------------------------------------------------------- errors
    def test_overlapping_shots_are_refused(self):
        self.entries[1]["sourceStart"] = 1.5
        with self.assertRaisesRegex(SequenceError, "must not overlap"):
            self.run_package()

    def test_a_shot_past_the_end_of_the_source_is_refused(self):
        self.entries[2]["sourceEnd"] = 7.0
        self.entries[2]["sourceMapping"]["sourceEndSeconds"] = 7.0
        with self.assertRaisesRegex(SequenceError, "past the source clip"):
            self.run_package()

    def test_a_missing_candidate_file_is_refused(self):
        (self.shot1 / "cameras.json").unlink()
        with self.assertRaisesRegex(SequenceError, "cameras.json does not exist"):
            self.run_package()

    def test_a_missing_frame_is_refused(self):
        (self.shot0 / "person" / "frame_0003.ply").unlink()
        with self.assertRaisesRegex(SequenceError, "frame frame_0003.ply is missing"):
            self.run_package()

    def test_a_missing_candidate_directory_is_refused(self):
        self.entries[0]["candidateDir"] = str(self.root / "absent-4d")
        with self.assertRaisesRegex(SequenceError, "is not a directory"):
            self.run_package()

    def test_non_increasing_candidate_timestamps_are_refused(self):
        people = json.loads((self.shot1 / "people.json").read_text())
        people["timestamps"] = [0.0, 0.25, 0.25, 0.75, 1.0]
        (self.shot1 / "people.json").write_text(json.dumps(people))
        with self.assertRaisesRegex(SequenceError, "not strictly increasing"):
            self.run_package()

    def test_a_candidate_duration_inside_its_last_sample_is_refused(self):
        people = json.loads((self.shot1 / "people.json").read_text())
        people["duration"] = 1.0
        (self.shot1 / "people.json").write_text(json.dumps(people))
        with self.assertRaisesRegex(SequenceError, "must be greater than the last timestamp"):
            self.run_package()

    def test_an_unresolvable_trim_offset_is_refused_rather_than_guessed(self):
        del self.entries[0]["trimOffsetSeconds"]
        with self.assertRaisesRegex(SequenceError, "no trim offset"):
            self.run_package()

    def test_a_recorded_mapping_that_disagrees_with_the_shot_list_is_refused(self):
        self.entries[2]["sourceMapping"]["sourceStartSeconds"] = 4.25
        with self.assertRaisesRegex(SequenceError, "disagrees with the recorded"):
            self.run_package()

    def test_a_trim_offset_that_pushes_samples_past_the_cut_is_refused(self):
        self.entries[0]["trimOffsetSeconds"] = 1.0
        with self.assertRaisesRegex(SequenceError, "not before sourceEnd"):
            self.run_package()

    def test_a_state_json_supplies_the_recorded_mapping(self):
        state = self.root / "state.json"
        state.write_text(
            json.dumps(
                {
                    "stages": {
                        "_run": {
                            "shot": {
                                "sourceMapping": {
                                    "sourceStartSeconds": 2.0,
                                    "localZeroSourceSeconds": 2.04,
                                }
                            }
                        }
                    }
                }
            )
        )
        del self.entries[1]["trimOffsetFromShotCutsFps"]
        self.entries[1]["stateJson"] = str(state)
        report = self.run_package()
        shot = report["shots"][1]
        self.assertAlmostEqual(shot["trimOffsetSeconds"], 0.04)
        self.assertTrue(shot["trimOffsetMeasured"])
        self.assertIn("localZeroSourceSeconds", shot["trimOffsetSource"])

    def test_a_non_empty_output_directory_needs_force(self):
        self.run_package()
        with self.assertRaisesRegex(SequenceError, "pass --force"):
            self.run_package()
        first = self.merged()
        self.assertEqual(self.run_package(force=True)["output"]["samples"], first["samples"])

    def test_force_refuses_a_directory_this_tool_did_not_write(self):
        self.out.mkdir(parents=True)
        (self.out / "someone-elses-work.txt").write_text("keep me")
        with self.assertRaisesRegex(SequenceError, "not a package this tool wrote"):
            self.run_package(force=True)
        self.assertTrue((self.out / "someone-elses-work.txt").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
