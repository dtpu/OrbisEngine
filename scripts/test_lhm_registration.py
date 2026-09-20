"""No-spend tests for later supported depth without discarding earlier source poses."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_clip
from stage_attempts import command_identity
from stages.lhm_registration import (
    animation_plan,
    depth_registration,
    select_depth_reference,
    validate_registration_option,
)

from worker import modal_multiperson


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folder = self.root / "pi3x"
        self.folder.mkdir()
        self.track = self.root / "tracks/track_00"
        self.track.mkdir(parents=True)
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"source fixture")
        self.indices = list(range(0, 586, 2))
        self.cameras = [
            {"sourceIndex": i, "camera_to_world": np.eye(4).tolist()} for i in self.indices
        ]
        self.records = [
            {"sample": i, "sourceIndex": self.indices[i], "maskBox": [-1, -1, 1, 1]}
            for i in range(31, 293)
        ]
        self.motion = {"frames": self.records}
        self.tracks = {
            "sourceIndices": self.indices,
            "sourceSha256": hashlib.sha256(self.video.read_bytes()).hexdigest(),
        }
        self.sequence = dict(self.tracks, frames=[None] * 293)
        self.sequence["frames"][41] = "frame_041.ply"
        self.save()
        vertices = np.zeros(60, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertices["z"] = -2
        PlyData([PlyElement.describe(vertices, "vertex")]).write(self.folder / "frame_041.ply")

    def save(self):
        (self.folder / "sequence.json").write_text(json.dumps(self.sequence))
        (self.folder / "cameras.json").write_text(json.dumps({"cameras": self.cameras}))
        (self.track / "motion.json").write_text(json.dumps(self.motion))
        (self.track.parent / "tracks.json").write_text(json.dumps(self.tracks))

    def select(self, selected=None):
        return select_depth_reference(self.folder, self.motion, self.tracks, self.cameras, selected)

    def test_initial_missing_depth_selects_later_matching_reference(self):
        sample, reference, roi = self.select()
        self.assertEqual(sample, 41)
        self.assertEqual(reference, self.folder / "frame_041.ply")
        self.assertEqual([float(v) for v in roi.split(",")], self.records[10]["maskBox"])
        self.assertEqual(len(self.motion["frames"]), 262)
        self.assertEqual(self.motion["frames"][0]["sample"], 31)

    def test_chronological_full_export_plan_separate_from_registration(self):
        poses = [None] * 31 + [{} for _ in range(262)]
        records = [None] * 31 + self.records
        sample, exports = animation_plan(poses, records, self.cameras, self.indices, 41)
        self.assertEqual(sample, 41)
        self.assertEqual(exports, list(range(31, 293)))
        self.assertEqual(animation_plan(poses, records, self.cameras, self.indices)[0], 31)
        for invalid in (0, 30, 293, -1, 41.5, True):
            with self.subTest(sample=invalid), self.assertRaises(ValueError):
                animation_plan(poses, records, self.cameras, self.indices, invalid)

    def test_source_correspondence_and_declared_missing_depth_fail_closed(self):
        self.cameras[41]["sourceIndex"] += 1
        with self.assertRaisesRegex(ValueError, "source indices"):
            self.select()
        self.cameras[41]["sourceIndex"] -= 1
        self.records[10]["sourceIndex"] += 1
        with self.assertRaisesRegex(ValueError, "source index"):
            self.select()
        self.records[10]["sourceIndex"] -= 1
        self.sequence["frames"][31] = "frame_031.ply"
        self.save()
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            self.select()

    def test_invalid_or_missing_selected_depth_and_scale_conflict(self):
        for invalid in (-1, 1.5, True):
            with self.subTest(sample=invalid), self.assertRaises(ValueError):
                self.select(invalid)
        with self.assertRaisesRegex(ValueError, "No track pose"):
            self.select(31)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            validate_registration_option(41, 1.0)
        (self.folder / "frame_041.ply").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            self.select()

    def test_numeric_scale_uses_selected_camera_and_preserves_median_rule(self):
        xyz = np.tile([0, 0, -20.0], (60, 1))
        camera = np.eye(4)
        camera[2, 3] = -10
        result = depth_registration(xyz, camera, np.eye(3), np.array([10, 12]), "-1,-1,1,1")
        self.assertEqual(result["uniformScale"], 1)
        self.assertEqual(result["nativeGaussianMedianDepth"], 10)
        wrong_camera = depth_registration(
            xyz, np.eye(4), np.eye(3), np.array([10, 12]), "-1,-1,1,1"
        )
        self.assertEqual(wrong_camera["uniformScale"], 2)
        with self.assertRaisesRegex(RuntimeError, "Only 49"):
            depth_registration(xyz[:49], camera, np.eye(3), [10], "-1,-1,1,1")
        with self.assertRaisesRegex(ValueError, "positive"):
            depth_registration(xyz, camera, np.eye(3), [-1], None)

    def test_runner_selects_matching_sample_reference_and_roi_without_truncation(self):
        pipeline = object.__new__(run_clip.Pipeline)
        pipeline.ctx, pipeline.clip = self.root, self.video
        pipeline.track_or_skip = Mock(return_value={"quality": {"firstSample": 31}})
        pipeline.recover_lhm = Mock(return_value=False)
        pipeline.paid_run = Mock()
        pipeline.lhm_motion(0)
        command = pipeline.paid_run.call_args.args[1]
        self.assertEqual(command[command.index("--registration-sample") + 1], "41")
        self.assertEqual(
            command[command.index("--depth-reference") + 1], str(self.folder / "frame_041.ply")
        )
        self.assertNotIn("--start", command)
        self.assertNotIn("--fixed-world-scale", command)
        self.assertEqual(len(json.loads((self.track / "motion.json").read_text())["frames"]), 262)

    def test_cli_validates_and_forwards_before_submission(self):
        import torch

        from worker.stages import lhm_recovery

        canonical = self.root / "canonical"
        canonical.mkdir()
        (canonical / "canonical-state.pt").write_bytes(b"canonical fixture")
        (canonical / "result.json").write_text("{}")
        torch.save(
            {"poses": [None] * 31 + [{} for _ in range(262)], "sourceIndices": self.indices},
            self.track / "source-poses.pt",
        )
        arguments = {
            "video": str(self.video),
            "canonical": str(canonical),
            "cameras": str(self.folder / "cameras.json"),
            "track_dir": str(self.track),
            "out": str(self.root / "result"),
            "depth_reference": str(self.folder / "frame_041.ply"),
            "depth_roi": "-1,-1,1,1",
            "registration_sample": 41,
        }
        with patch.object(
            lhm_recovery, "submit_once", side_effect=RuntimeError("no provider call")
        ) as submit:
            with self.assertRaisesRegex(RuntimeError, "no provider call"):
                modal_multiperson.animate(**arguments)
            self.assertIn("--registration-sample", submit.call_args.kwargs["flags"])
            self.assertEqual(submit.call_args.kwargs["flags"][-2:], ["--depth-roi", "-1,-1,1,1"])
        arguments["out"] = str(self.root / "blocked")
        arguments["registration_sample"] = 31
        with patch.object(lhm_recovery, "submit_once") as submit, self.assertRaises(ValueError):
            modal_multiperson.animate(**arguments)
        submit.assert_not_called()
        self.assertFalse(Path(arguments["out"]).exists())

    def test_registration_sample_fingerprinted_and_invalid_values_rejected(self):
        base = ["modal", "run", "worker/modal_multiperson.py::animate", "--registration-sample"]
        root = Path(__file__).resolve().parents[1]
        first, code, _ = command_identity(base + ["41"], root)
        changed, _, _ = command_identity(base + ["42"], root)
        self.assertNotEqual(first, changed)
        self.assertIn("worker/stages/lhm_registration.py", code)
        for bad in ("-1", "1.5", "nan"):
            with self.subTest(sample=bad), self.assertRaises(ValueError):
                command_identity(base + [bad], root)


if __name__ == "__main__":
    unittest.main()
