#!/usr/bin/env python3
"""A person whose track starts late and has gaps, against the helpers that used to assume sample 0.

The fixture is the real shape creed-v2 has -- `frames` from `frame_022.ply` to `frame_144.ply`,
missing samples in the middle, first source frame 44 -- written as a few bytes of fake PLY each,
because nothing here reads splat data. No network, no model, no real footage.

What is proved:
  * the reference frame is the first frame that EXISTS, not `frame_000.ply`;
  * per-sample lookups go through the sequence's own `sourceIndices` and answer `None` for a sample
    the tracker recorded this person absent from, instead of the nearest frame in time;
  * a dense track starting at 0 gets byte-identical answers to the old hard-coded paths;
  * the precise error fires only when the person TRULY has no frames.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_clip
import sequence_frames
from sequence_frames import (
    MissingPersonFrames,
    existing_frames,
    first_frame,
    first_world_frame,
    index_of_source,
    optional_first_world_frame,
    read_sequence,
    source_indices,
)

# creed-v2's own shape: starts at sample 22 (source frame 44), gaps at 105-106, 110, 115-116.
SPARSE_SAMPLES = [22, 23, 24, 25, 104, 107, 108, 109, 111, 144]


def write_person(
    world: Path,
    samples: list[int],
    person: str = "person",
    on_disk: list[int] | None = None,
    width: int = 3,
) -> Path:
    """A packaged person directory: sequence.json plus one tiny fake PLY per frame on disk."""
    directory = world / person
    directory.mkdir(parents=True, exist_ok=True)
    frames = [f"frame_{s:0{width}d}.ply" for s in samples]
    written = samples if on_disk is None else on_disk
    for sample, name in zip(samples, frames):
        if sample in written:
            (directory / name).write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    (directory / "sequence.json").write_text(
        json.dumps(
            {
                "frame_format": "gaussian-ply",
                "frames": frames,
                "count": len(frames),
                "fps": 12,
                "timestamps": [round(s / 12, 6) for s in samples],
                "sourceIndices": [s * 2 for s in samples],
                "requestedSampleSemantics": "Samples this run was asked to animate.",
                "trackGapSamples": [
                    s for s in range(samples[0], samples[-1] + 1) if s not in samples
                ],
            },
            indent=1,
        )
    )
    return directory


class SparseSequenceHelpers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.world = self.root / "sparse-4d"
        self.person = write_person(self.world, SPARSE_SAMPLES)
        self.seq = self.person / "sequence.json"

    # ---------------------------------------------------------------- reference frame
    def test_the_reference_frame_is_the_first_frame_that_exists(self):
        self.assertEqual(first_frame(self.seq).name, "frame_022.ply")
        self.assertEqual(first_world_frame(self.world).name, "frame_022.ply")
        self.assertFalse((self.person / "frame_000.ply").exists())

    def test_a_dense_track_from_zero_still_answers_frame_000(self):
        world = self.root / "dense-4d"
        write_person(world, list(range(6)))
        self.assertEqual(first_world_frame(world).name, "frame_000.ply")
        self.assertEqual(
            [p.name for p in existing_frames(world / "person" / "sequence.json")],
            [f"frame_{i:03d}.ply" for i in range(6)],
        )
        seq = read_sequence(world / "person" / "sequence.json")
        self.assertEqual(index_of_source(seq), {i * 2: i for i in range(6)})

    def test_a_leading_frame_missing_from_disk_is_skipped_not_fatal(self):
        world = self.root / "partial-4d"
        write_person(world, SPARSE_SAMPLES, on_disk=SPARSE_SAMPLES[2:])
        self.assertEqual(first_world_frame(world).name, "frame_024.ply")
        self.assertEqual(len(existing_frames(world / "person" / "sequence.json")), 8)

    # ---------------------------------------------------------------- per-sample lookup
    def test_lookups_use_the_sequence_s_own_samples_and_refuse_to_guess(self):
        seq = read_sequence(self.seq)
        at = index_of_source(seq)
        self.assertEqual(source_indices(seq, self.seq)[0], 44)
        self.assertEqual(at[44], 0)  # the first frame is source frame 44, not 0
        self.assertEqual(at[288], len(SPARSE_SAMPLES) - 1)
        self.assertIsNone(at.get(0), "source frame 0 has no frame; nearest-match invented one")
        self.assertIsNone(at.get(220), "sample 110 is a track gap, not the nearest frame")
        self.assertEqual(seq["trackGapSamples"][:2], [26, 27])

    def test_a_manifest_whose_sample_list_disagrees_with_its_frames_is_refused(self):
        doc = json.loads(self.seq.read_text())
        doc["sourceIndices"] = doc["sourceIndices"][:2]
        self.seq.write_text(json.dumps(doc))
        with self.assertRaisesRegex(MissingPersonFrames, "same samples"):
            index_of_source(read_sequence(self.seq), self.seq)

    # ---------------------------------------------------------------- precise errors
    def test_a_person_with_no_frames_at_all_is_the_only_error(self):
        world = self.root / "empty-4d"
        write_person(world, SPARSE_SAMPLES, on_disk=[])
        with self.assertRaisesRegex(MissingPersonFrames, r"none of the 10 frames"):
            first_world_frame(world)
        # and it names what it looked for, so the message can be acted on
        try:
            first_world_frame(world)
        except MissingPersonFrames as error:
            self.assertIn("frame_022.ply", str(error))
            self.assertIn("frame_144.ply", str(error))

    def test_an_unpackaged_world_still_says_the_package_stage_has_not_run(self):
        with self.assertRaisesRegex(MissingPersonFrames, "has not run"):
            first_world_frame(self.root / "absent-4d")
        self.assertIsNone(optional_first_world_frame(self.root / "absent-4d"))

    def test_an_empty_frame_list_is_never_reconstructed_not_missing(self):
        world = self.root / "noframes-4d"
        directory = world / "person"
        directory.mkdir(parents=True)
        (directory / "sequence.json").write_text(json.dumps({"frames": [], "count": 0}))
        with self.assertRaisesRegex(MissingPersonFrames, "never reconstructed"):
            first_world_frame(world)

    def test_a_frame_name_outside_the_person_directory_is_refused(self):
        doc = json.loads(self.seq.read_text())
        doc["frames"][0] = "../../../etc/passwd"
        self.seq.write_text(json.dumps(doc))
        with self.assertRaisesRegex(MissingPersonFrames, "not a frame name"):
            first_frame(self.seq)


class ScaleFitReadsTheTrack(unittest.TestCase):
    """scale_fit's own accessors, which stopped creed-v2 by asking for frame_000.ply."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch.object(run_clip, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pipeline = object.__new__(run_clip.Pipeline)
        self.pipeline.name = "sparse"
        self.pipeline.state = run_clip.State(self.root / "state.json")
        self.world = self.root / "public" / "worlds" / "sparse-4d"

    def test_person_ply_is_the_first_packaged_frame_of_a_late_track(self):
        write_person(self.world, SPARSE_SAMPLES)
        self.assertEqual(self.pipeline.person_ply().name, "frame_022.ply")

    def test_person_ply_is_frame_000_for_a_dense_track(self):
        write_person(self.world, list(range(4)))
        self.assertEqual(
            self.pipeline.person_ply(),
            self.world / "person" / "frame_000.ply",
        )

    def test_person_ply_fails_precisely_when_the_package_stage_has_not_run(self):
        with self.assertRaisesRegex(run_clip.MissingPersonFrames, "has not run"):
            self.pipeline.person_ply()

    def test_placement_measures_the_first_existing_frame_and_omits_it_otherwise(self):
        write_person(self.world, SPARSE_SAMPLES)
        measured = []

        def fake_stats(path):
            measured.append(Path(path).name)
            return 0.8, -0.4

        with patch.object(run_clip, "body_stats", fake_stats):
            out = self.pipeline.placement()
        self.assertEqual(measured, ["frame_022.ply"])
        self.assertAlmostEqual(out["bodyHeight"], 0.8)
        self.assertAlmostEqual(out["feetY"], -0.4)

    def test_placement_reports_no_body_instead_of_raising_when_nothing_is_packaged(self):
        out = self.pipeline.placement()
        self.assertNotIn("bodyHeight", out)
        self.assertNotIn("floor", out)


class SparseTrackAudits(unittest.TestCase):
    """motion_audit / motion_silhouette matched cameras to the NEAREST frame in time."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def sequence(self, samples):
        world = self.root / f"w{samples[0]}-4d"
        write_person(world, samples)
        return read_sequence(world / "person" / "sequence.json")

    def test_a_camera_before_the_track_starts_selects_no_frame(self):
        seq = self.sequence(SPARSE_SAMPLES)
        at = index_of_source(seq)
        cameras = [{"sourceIndex": i} for i in range(0, 290, 2)]
        chosen = [c["sourceIndex"] for c in cameras if c["sourceIndex"] in at]
        self.assertEqual(chosen[0], 44)
        self.assertEqual(len(chosen), len(SPARSE_SAMPLES))
        self.assertNotIn(0, chosen)
        self.assertNotIn(220, chosen)  # the 110 gap

    def test_every_camera_is_kept_for_a_dense_track(self):
        seq = self.sequence(list(range(8)))
        at = index_of_source(seq)
        cameras = [{"sourceIndex": i * 2} for i in range(8)]
        self.assertEqual([at[c["sourceIndex"]] for c in cameras], list(range(8)))


class ImportedByTheStagesThatNeedIt(unittest.TestCase):
    def test_no_stage_reaches_for_a_hard_coded_person_frame_zero(self):
        """Executable `frame_000.ply` literals, prose excluded, with the reason each survivor stays.

        The only ones left name a PI3X depth frame -- `.context/run/<clip>/pi3x/frame_000.ply`, the
        cloud the LHM stages register against. That is the solver's own sample 0, which always
        exists; it is not the packaged person's frame list.
        """
        scripts = Path(__file__).resolve().parent
        found = []
        for name in (
            "run_clip.py",
            "anchor_checks.py",
            "motion_audit.py",
            "motion_silhouette.py",
            "silhouette_rows.py",
            "place_solve.py",
            "world_ruler.py",
        ):
            source = (scripts / name).read_text()
            lines = source.splitlines()
            tree = ast.parse(source)
            documentation = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    first = node.body[0] if node.body else None
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                        documentation.add(id(first.value))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "frame_000.ply" in node.value
                    and id(node) not in documentation
                ):
                    found.append((f"{name}:{node.lineno}", lines[node.lineno - 1].strip()))
        for where, line in found:
            self.assertIn("pi3x", line, f"{where} hard-codes a person's frame 0: {line}")
        self.assertTrue(found, "the pi3x depth references were expected to survive unchanged")

    def test_the_helper_carries_no_heavy_dependency(self):
        # the pipeline runner imports this at start-up; it must not pull numpy in to find a file
        self.assertFalse(hasattr(sequence_frames, "np"))
        self.assertIsInstance(sequence_frames.MissingPersonFrames("x"), RuntimeError)
        self.assertIs(run_clip.MissingPersonFrames, sequence_frames.MissingPersonFrames)


if __name__ == "__main__":
    unittest.main(verbosity=2)
