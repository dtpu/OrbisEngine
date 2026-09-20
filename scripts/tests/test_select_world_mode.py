"""Offline input policy and pipeline wiring; no generation or quality verdict is simulated."""

import sys
from pathlib import Path

# The modules under test are this directory's parent; importing them by name is what
# running from scripts/ used to give for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_clip
import select_world_mode as selector


class VideoFirstPolicy(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prediction = self.root / "admission.json"
        self.prediction.write_text(
            json.dumps({"status": "registered", "motion": {"headingSweepDeg": 130}})
        )
        self.wide = {
            "spread8": 20,
            "spreadMax": 130,
            "pick": [0, 10, 20],
            "pickAzimuth": [0, 60, 130],
        }
        self.narrow = {**self.wide, "spread8": 0.2, "spreadMax": 1.5}

    def test_camera_spread_does_not_automatically_replace_video(self):
        for measurement in (self.wide, self.narrow):
            with self.subTest(measurement=measurement):
                decision = selector.decide(measurement)
                self.assertEqual(decision["mode"], "video")
                self.assertFalse(decision["quality_verified"])
                self.assertNotIn("frames", decision)

    def test_stills_are_an_explicit_unverified_alternative(self):
        for measurement, expected in ((self.wide, "multi-image"), (self.narrow, "image")):
            decision = selector.decide(measurement, still_images=True)
            self.assertEqual(decision["mode"], expected)
            self.assertFalse(decision["quality_verified"])
            self.assertTrue(decision["frames"])

    def test_missing_or_large_predicted_motion_never_certifies_quality(self):
        for prediction in ({}, {"status": "failed"}, json.loads(self.prediction.read_text())):
            decision = selector.decide_from_prediction(prediction)
            self.assertEqual(decision["mode"], "video")
            self.assertFalse(decision["quality_verified"])

    def test_cli_video_default_legacy_alias_and_explicit_stills(self):
        for flags, expected in (
            ([], "video"),
            (["--allow-video"], "video"),
            (["--still-images"], "multi-image"),
        ):
            output = self.root / "decision.json"
            argv = [
                "select_world_mode.py",
                "--predict-json",
                str(self.prediction),
                "--out",
                str(output),
                *flags,
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                selector.main()
            self.assertEqual(json.loads(output.read_text())["mode"], expected)

    def test_default_runner_graph_uses_one_reviewed_clean_video(self):
        for graph in (run_clip.single_graph, lambda mode: run_clip.multiperson_graph(2, mode)):
            stages, dependencies = graph("video")
            self.assertEqual([s for s in stages if s.startswith("marble_")], ["marble_video"])
            self.assertEqual(dependencies["marble_video"], ["clean", "review"])
            self.assertIn("clean", dependencies["review"])
            self.assertIn("world_prompt", dependencies["review"])

    def test_runner_cli_defaults_to_video_without_launching_work(self):
        with (
            patch.object(
                sys,
                "argv",
                ["run_clip.py", "--clip", "fixture.mp4", "--name", "fixture", "--no-publish"],
            ),
            patch.object(run_clip, "resolve_shots", return_value=[]) as resolve,
            patch.object(run_clip, "Pipeline") as pipeline,
            self.assertRaises(SystemExit) as stopped,
        ):
            run_clip.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(resolve.call_args.args[0].marble, "video")
        pipeline.assert_not_called()

    def test_pipeline_forwards_the_clean_video_and_generated_description(self):
        pipeline = object.__new__(run_clip.Pipeline)
        pipeline.ctx = self.root
        pipeline.clean_mp4 = self.root / "clean.mp4"
        pipeline.clean_mp4.write_bytes(b"complete cleaned clip")
        pipeline.prompt_json.write_text('{"text_prompt": "Source description"}')
        with patch.object(pipeline, "marble_world") as generate:
            pipeline.marble_video()
        generate.assert_called_once_with(
            "video", pipeline.clean_mp4, "clean", ["--prompt-file", str(pipeline.prompt_json)]
        )

    def test_explicit_multi_pipeline_requests_still_alternatives(self):
        pipeline = object.__new__(run_clip.Pipeline)
        pipeline.ctx = self.root
        pipeline.clip = self.root / "clip.mp4"
        for mode in ("video", "multi"):
            pipeline.marble = mode

            def decision(cmd, *args, **kwargs):
                pipeline.mode_json.write_text('{"mode": "video", "why": "test decision"}')

            with (
                patch.object(run_clip, "run", side_effect=decision) as command,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                pipeline.world_mode()
            self.assertEqual("--still-images" in command.call_args.args[0], mode == "multi")


if __name__ == "__main__":
    unittest.main()
