"""No-spend shared-scale selection through the runner's packaging subprocess."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_clip


class SharedScaleTests(unittest.TestCase):
    def test_packaging_forwards_default_none_and_explicit_reference_to_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = object.__new__(run_clip.Pipeline)
            pipeline.ctx = root / "run"
            pipeline.name = "fixture"
            pipeline.clip = root / "source.mp4"
            pipeline.people_limit = 2
            for tid in range(2):
                motion = pipeline.ctx / f"lhm-motion-{tid:02d}"
                motion.mkdir(parents=True)
                (motion / "sequence.json").write_text("{}")
            (pipeline.ctx / "tracks").mkdir()
            (pipeline.ctx / "tracks/tracks.json").write_text(json.dumps({"duration": 10}))
            for value in (None, "auto", "none", "1"):
                with (
                    self.subTest(value=value),
                    patch.object(run_clip, "public_dir", return_value=root / "public"),
                    patch.object(run_clip, "source_audio_manifest"),
                    patch.object(
                        run_clip.subprocess, "run", return_value=SimpleNamespace(returncode=0)
                    ) as process,
                ):
                    pipeline.a = SimpleNamespace(**({"shared_scale": value} if value else {}))
                    pipeline.package_people()
                process.assert_called_once()
                command = process.call_args.args[0]
                self.assertEqual(command[:2], [run_clip.PY, "scripts/package_multiperson.py"])
                self.assertEqual(command[command.index("--shared-scale") + 1], value or "auto")
                self.assertIn(f"1={pipeline.ctx / 'lhm-motion-01'}", command)

    def test_missing_reference_animation_is_rejected_before_output_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = object.__new__(run_clip.Pipeline)
            pipeline.ctx = root / "run"
            pipeline.name = "fixture"
            pipeline.people_limit = 2
            pipeline.a = SimpleNamespace(shared_scale="1")
            motion = pipeline.ctx / "lhm-motion-00"
            motion.mkdir(parents=True)
            (motion / "sequence.json").write_text("{}")
            with (
                patch.object(run_clip, "source_audio_manifest") as audio,
                patch.object(run_clip.shutil, "rmtree") as remove,
                patch.object(run_clip.subprocess, "run") as process,
                self.assertRaisesRegex(ValueError, "no completed animation"),
            ):
                pipeline.package_people()
            audio.assert_not_called()
            remove.assert_not_called()
            process.assert_not_called()

    def test_invalid_or_unsupported_values_fail_before_source_or_provider_work(self):
        cases = [
            ["--people", "2", "--shared-scale", value]
            for value in ("", "floor", "-1", "1.5", "two", "2", "+1", " 1")
        ] + [
            ["--all-people", "--shared-scale", "4"],
            ["--people", "1", "--shared-scale", "1"],
            ["--shared-scale", "none"],
            ["--shared-scale", "0"],
            ["--orchestrated", "--people", "2", "--shared-scale", "1"],
            ["--orchestrated", "--all-people", "--shared-scale", "none"],
        ]
        for flags in cases:
            with (
                self.subTest(flags=flags),
                patch.object(
                    sys,
                    "argv",
                    ["run_clip.py", "--clip", "unused.mp4", "--name", "fixture", *flags],
                ),
                patch.object(run_clip, "resolve_shots") as resolve,
                patch.object(run_clip, "Pipeline") as pipeline,
                patch.object(run_clip.subprocess, "run") as process,
                patch("sys.stderr"),
                self.assertRaises(SystemExit) as exited,
            ):
                run_clip.main()
            self.assertEqual(exited.exception.code, 2)
            resolve.assert_not_called()
            pipeline.assert_not_called()
            process.assert_not_called()

    def test_cli_accepts_default_and_configured_slot_boundaries(self):
        cases = (
            ([], "auto"),
            (["--shared-scale", "auto"], "auto"),
            (["--people", "2", "--shared-scale", "none"], "none"),
            (["--people", "2", "--shared-scale", "0"], "0"),
            (["--people", "2", "--shared-scale", "1"], "1"),
            (["--all-people", "--shared-scale", "3"], "3"),
            (["--people", "1", "--shared-scale", "0"], "0"),
        )
        for flags, expected in cases:
            with (
                self.subTest(flags=flags),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_clip.py",
                        "--clip",
                        "unused.mp4",
                        "--name",
                        "fixture",
                        "--no-publish",
                        *flags,
                    ],
                ),
                patch.object(run_clip, "resolve_shots", return_value=[]) as resolve,
                self.assertRaises(SystemExit) as exited,
            ):
                run_clip.main()
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(resolve.call_args.args[0].shared_scale, expected)

    def test_wander_stage_command_carries_configured_people_and_explicit_reference(self):
        from orchestrator.cli import run_clip_command

        context = SimpleNamespace(
            repository=Path(__file__).resolve().parents[2],
            source=Path("unused.mp4"),
            name="fixture",
            options={"people": 2},
        )
        command = run_clip_command(context, "package_people", ["--shared-scale", "1"])
        self.assertEqual(command[command.index("--only") + 1], "package_people")
        self.assertEqual(command[command.index("--people") + 1], "2")
        self.assertEqual(command[-2:], ["--shared-scale", "1"])


if __name__ == "__main__":
    unittest.main()
