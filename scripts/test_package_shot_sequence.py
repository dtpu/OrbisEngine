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


def make_world_only_cameras(path: Path, packaged: bool = False) -> Path:
    """A raw `.context/run/<name>/pi3x/cameras.json` for a shot that reconstructs no person.

    Camera 0 sits at (1, 2, 3) in the raw solve, so the packagers' reframe (camera 0 to the origin,
    scripts/sfm_frame.py) is visible in the output rather than a no-op. `packaged=True` writes the
    file scripts/package_multiperson.py would already have produced, which must be copied through.
    """

    def translation(x, y, z):
        return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]

    document = {
        "coordinates": "Pi3X raw export" if not packaged else "OpenGL, camera 0 = identity",
        "cameras": [
            {
                "sourceIndex": 0,
                "time": 0.0,
                "camera_to_world": translation(1, 2, 3) if not packaged else translation(0, 0, 0),
                "source_intrinsics": [[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]],
                "source_image_size": [64, 48],
            },
            {
                "sourceIndex": 4,
                "time": 0.4,
                "camera_to_world": translation(1, 2, 4) if not packaged else translation(0, 0, 1),
                "source_intrinsics": [[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]],
                "source_image_size": [64, 48],
            },
        ],
    }
    if packaged:
        document["frameAlign"] = None
        document["source"] = "already packaged by scripts/package_multiperson.py"
        document["sourceClip"] = "fixture.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


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
        # a close-up shot's own solve: a world and cameras, and no people.json anywhere
        self.raw_cameras = make_world_only_cameras(self.root / "run/clip-shot01/pi3x/cameras.json")
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

    def world_only_entries(self, **overrides):
        """people, world-only, people -- the shape a cut sequence with one close-up shot has."""
        entry = {
            "world": "/marble-clip-shot01-clean.spz",
            "sourceStart": 2.0,
            "sourceEnd": 4.0,
            "people": "none",
            "noPeopleReason": "at most 24% of the person is ever inside the frame",
            "camerasJson": str(self.raw_cameras),
        }
        entry.update(overrides)
        return [self.entries[0], entry, self.entries[2]]

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

    # ---------------------------------------------------------------- world-only shots
    def test_a_world_only_shot_contributes_its_window_world_and_cameras(self):
        report = self.run_package(self.world_only_entries())
        manifest = self.merged()
        self.assertEqual(len(manifest["shots"]), 3)
        middle = manifest["shots"][1]
        self.assertEqual(middle["sourceStart"], 2.0)
        self.assertEqual(middle["sourceEnd"], 4.0)
        self.assertEqual(middle["world"], "/marble-clip-shot01-clean.spz")
        self.assertEqual(middle["people"], [])
        self.assertIsNone(middle["primary"])
        self.assertIsNone(middle["sampleRange"])
        self.assertIsNone(middle["trimOffsetSeconds"])
        self.assertIn("world-only", middle["trimOffsetSource"])
        self.assertEqual(middle["cameras"], "shots/01/cameras.json")
        self.assertTrue((self.out / "shots/01/cameras.json").is_file())
        # the cast is only the two people shots'; nobody was invented for the middle one
        self.assertEqual(
            [p["id"] for p in manifest["people"]], ["s00-person", "s00-person_01", "s02-person"]
        )
        self.assertEqual(report["output"]["peopleCount"], 3)

    def test_a_world_only_shot_s_cameras_are_written_in_the_packaged_shape(self):
        self.run_package(self.world_only_entries())
        packaged = json.loads((self.out / "shots/01/cameras.json").read_text())
        # the shape scripts/package_multiperson.py:183-194 writes into a world directory
        self.assertEqual(
            sorted(packaged), ["cameras", "coordinates", "frameAlign", "source", "sourceClip"]
        )
        self.assertIn("camera 0 = identity", packaged["coordinates"])
        self.assertIsNone(packaged["frameAlign"])
        self.assertEqual(packaged["source"], str(self.raw_cameras))
        self.assertEqual(packaged["sourceClip"], str(self.source))
        for camera in packaged["cameras"]:
            self.assertEqual(
                sorted(camera),
                [
                    "camera_to_world",
                    "sourceIndex",
                    "source_image_size",
                    "source_intrinsics",
                    "time",
                ],
            )
            self.assertEqual(len(camera["camera_to_world"]), 4)
        # camera 0 goes to the origin, and every other camera moves with it: the frame fourd.html
        # measures a start pose in (src/start-view.ts firstSourceCamera)
        first, second = (c["camera_to_world"] for c in packaged["cameras"])
        self.assertEqual([row[3] for row in first], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual([row[3] for row in second], [0.0, 0.0, 1.0, 1.0])

    def test_an_already_packaged_cameras_file_is_copied_through_unchanged(self):
        already = make_world_only_cameras(self.root / "packaged/cameras.json", packaged=True)
        report = self.run_package(self.world_only_entries(camerasJson=str(already)))
        self.assertEqual(
            json.loads((self.out / "shots/01/cameras.json").read_text()),
            json.loads(already.read_text()),
        )
        self.assertIn("already packaged", report["shots"][1]["camerasFrom"])

    def test_a_world_only_shot_still_terminates_the_grid_at_its_cut(self):
        self.run_package(self.world_only_entries())
        manifest = self.merged()
        # its own cut is on the grid, so a neighbour's last visibility interval stops there
        self.assertIn(4.0, manifest["timestamps"])
        self.assertIn(2.0, manifest["timestamps"])
        # every mapped sample of the two people shots, plus a terminator at each of the three cuts
        # that is not the end of the clip (2.0 and 4.0; shot 2 ends at the clip's end)
        self.assertEqual(len(manifest["timestamps"]), 7 + 6 + 2)

    def test_nobody_is_visible_during_a_world_only_shot(self):
        self.run_package(self.world_only_entries())
        manifest = self.merged()
        for person in manifest["people"]:
            visible = person_visibility(
                person["visibleSampleRuns"], manifest["timestamps"], manifest["duration"]
            )
            for time in (2.0, 2.5, 3.0, 3.5, 3.99):
                self.assertFalse(visible(time), f"{person['id']} is visible in the world-only shot")
            start, end = (
                manifest["shots"][person["shotIndex"]]["sourceStart"],
                manifest["shots"][person["shotIndex"]]["sourceEnd"],
            )
            seen = [t for t in (i * SOURCE_SECONDS / 600 for i in range(600)) if visible(t)]
            self.assertTrue(seen and all(start <= t < end for t in seen))

    def test_the_report_lists_the_world_only_shot_with_its_reason(self):
        report = self.run_package(self.world_only_entries())
        self.assertEqual(
            report["shotsWithoutPeople"],
            [
                dict(
                    index=1,
                    world="/marble-clip-shot01-clean.spz",
                    sourceStart=2.0,
                    sourceEnd=4.0,
                    reason="at most 24% of the person is ever inside the frame",
                    cameras="shots/01/cameras.json",
                )
            ],
        )
        self.assertEqual(report["shots"][1]["peopleCount"], 0)
        self.assertIsNone(report["shots"][1]["mappedInterval"])
        self.assertEqual(
            self.merged()["shotSequence"]["shotsWithoutPeople"][0]["reason"],
            "at most 24% of the person is ever inside the frame",
        )

    def test_a_world_only_shot_may_still_carry_its_own_placement(self):
        placement = self.root / "run/clip-shot01/placement.json"
        placement.write_text(json.dumps({"floorY": -0.25}))
        self.run_package(self.world_only_entries(placementJson=str(placement)))
        self.assertEqual(self.merged()["shots"][1]["placement"], "shots/01/placement.json")
        self.assertEqual(
            json.loads((self.out / "shots/01/placement.json").read_text())["floorY"], -0.25
        )

    def test_a_missing_people_json_is_an_error_not_a_silent_world_only_shot(self):
        (self.shot1 / "people.json").unlink()
        with self.assertRaisesRegex(SequenceError, 'must say so: set "people": "none"'):
            self.run_package()

    def test_a_world_only_shot_without_a_reason_is_refused(self):
        with self.assertRaisesRegex(SequenceError, "requires a noPeopleReason"):
            self.run_package(self.world_only_entries(noPeopleReason=None))
        with self.assertRaisesRegex(SequenceError, "requires a noPeopleReason"):
            self.run_package(self.world_only_entries(noPeopleReason="   "))

    def test_only_the_string_none_declares_a_world_only_shot(self):
        with self.assertRaisesRegex(SequenceError, 'people must be absent or the string "none"'):
            self.run_package(self.world_only_entries(people=[]))
        with self.assertRaisesRegex(SequenceError, 'people must be absent or the string "none"'):
            self.run_package(self.world_only_entries(people=False))

    def test_a_world_only_shot_must_name_its_cameras(self):
        with self.assertRaisesRegex(SequenceError, "camerasJson must be a path"):
            self.run_package(self.world_only_entries(camerasJson=None))
        with self.assertRaisesRegex(SequenceError, "camerasJson .* does not exist"):
            self.run_package(self.world_only_entries(camerasJson=str(self.root / "absent.json")))

    def test_a_cameras_file_with_no_cameras_is_refused(self):
        empty = self.root / "empty-cameras.json"
        empty.write_text(json.dumps({"cameras": []}))
        with self.assertRaisesRegex(SequenceError, "no non-empty `cameras` list"):
            self.run_package(self.world_only_entries(camerasJson=str(empty)))

    def test_a_cameras_file_missing_a_camera_s_own_fields_is_refused(self):
        broken = self.root / "broken-cameras.json"
        broken.write_text(json.dumps({"cameras": [{"sourceIndex": 0}]}))
        with self.assertRaisesRegex(SequenceError, "camera 0 is missing time"):
            self.run_package(self.world_only_entries(camerasJson=str(broken)))

    def test_declaring_no_people_over_a_candidate_that_has_them_is_refused(self):
        with self.assertRaisesRegex(SequenceError, "will not discard"):
            self.run_package(self.world_only_entries(candidateDir=str(self.shot1)))

    def test_a_sequence_of_only_world_only_shots_is_refused(self):
        entries = [
            self.world_only_entries()[1],
            dict(self.world_only_entries()[1], sourceStart=4.5, sourceEnd=6.0),
        ]
        with self.assertRaisesRegex(SequenceError, "no shot contributes a sample"):
            self.run_package(entries)

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
