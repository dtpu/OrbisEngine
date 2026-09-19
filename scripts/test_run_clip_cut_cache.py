"""Focused cache-admission tests for run_clip.cut_check."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_clip


class CutCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"AAAA")
        os.utime(self.clip, (1_700_000_000, 1_700_000_000))
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "shot_cuts.py").write_text("detector-v1")
        self.calls = []

        def fake(command, **_kwargs):
            self.calls.append(command)
            output = Path(command[command.index("--json") + 1])
            output.write_text(
                json.dumps(
                    {
                        "continuous": False,
                        "cutCount": len(self.calls),
                        "cuts": [{}] * len(self.calls),
                        "shots": [],
                    }
                )
            )
            return SimpleNamespace(returncode=1, stdout="")

        self.fake = fake
        self.old_root = run_clip.ROOT
        run_clip.ROOT = self.root
        self.addCleanup(self.restore)

    def restore(self):
        run_clip.ROOT = self.old_root
        self.temp.cleanup()

    def args(self, threshold=0.1, shot=None, all_shots=False):
        return SimpleNamespace(
            clip=str(self.clip),
            name="same-run",
            cut_threshold=threshold,
            shot=shot,
            all_shots=all_shots,
        )

    def test_hit_and_same_size_mtime_content_change_miss(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            first = run_clip.cut_check(self.args())
            second = run_clip.cut_check(self.args())
            self.clip.write_bytes(b"BBBB")
            os.utime(self.clip, (1_700_000_000, 1_700_000_000))
            third = run_clip.cut_check(self.args())
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(first["cutCount"], second["cutCount"])
        self.assertNotEqual(second["cutCount"], third["cutCount"])

    def test_threshold_change_same_score_mode_misses(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
            run_clip.cut_check(self.args(threshold=0.3))
        self.assertEqual(len(self.calls), 2)

    def test_detector_exit_status_must_agree_with_valid_report(self):
        def detector(continuous, returncode):
            def run(command, **_kwargs):
                Path(command[command.index("--json") + 1]).write_text(
                    json.dumps(
                        {
                            "continuous": continuous,
                            "cutCount": 0 if continuous else 1,
                            "cuts": [] if continuous else [{"time": 1.0}],
                            "shots": [],
                        }
                    )
                )
                return SimpleNamespace(returncode=returncode, stdout="")

            return run

        for continuous, returncode in ((True, 0), (False, 1)):
            with patch.object(
                run_clip.subprocess, "run", side_effect=detector(continuous, returncode)
            ):
                result = run_clip.cut_check(self.args(threshold=0.1 + returncode))
                self.assertEqual(result["continuous"], continuous)
        cache = self.root / ".context/run/same-run/cuts.json"
        original = cache.read_bytes()
        for continuous, returncode in ((True, 1), (False, 0), (False, 2)):
            with (
                self.subTest(continuous=continuous, returncode=returncode),
                patch.object(
                    run_clip.subprocess, "run", side_effect=detector(continuous, returncode)
                ),
                self.assertRaises(SystemExit),
            ):
                run_clip.cut_check(self.args(threshold=0.9))
            self.assertEqual(cache.read_bytes(), original)

    def test_score_mode_change_same_threshold_misses(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
            run_clip.cut_check(self.args(shot=1))
        self.assertEqual(len(self.calls), 2)
        self.assertIn("--no-score", self.calls[-1])

    def test_detector_key_change_misses(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
            (self.root / "scripts" / "shot_cuts.py").write_text("detector-v2")
            run_clip.cut_check(self.args())
        self.assertEqual(len(self.calls), 2)

    def test_malformed_cached_reports_recompute(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
            cache = self.root / ".context" / "run" / "same-run" / "cuts.json"
            cache.write_text("[]")
            run_clip.cut_check(self.args())
            cache.write_text(
                json.dumps(
                    {
                        "_cacheKey": {},
                        "continuous": "yes",
                        "cutCount": 0,
                        "cuts": [],
                        "shots": [],
                    }
                )
            )
            run_clip.cut_check(self.args())
        self.assertEqual(len(self.calls), 3)

    def test_failed_or_malformed_rerun_preserves_old_report(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
        cache = self.root / ".context" / "run" / "same-run" / "cuts.json"
        old = cache.read_bytes()
        self.clip.write_bytes(b"BBBB")
        with (
            patch.object(
                run_clip.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=1, stdout="failed"),
            ),
            self.assertRaises(SystemExit),
        ):
            run_clip.cut_check(self.args())
        self.assertEqual(cache.read_bytes(), old)

        def malformed(command, **_kwargs):
            Path(command[command.index("--json") + 1]).write_text("[]")
            return SimpleNamespace(returncode=0, stdout="")

        with (
            patch.object(run_clip.subprocess, "run", side_effect=malformed),
            self.assertRaises(SystemExit),
        ):
            run_clip.cut_check(self.args())
        self.assertEqual(cache.read_bytes(), old)

    def test_inconsistent_continuity_or_count_does_not_admit_cached_report(self):
        for change in ({"continuous": True}, {"cutCount": 999}):
            with (
                self.subTest(change=change),
                patch.object(run_clip.subprocess, "run", side_effect=self.fake),
            ):
                run_clip.cut_check(self.args())
                cache = self.root / ".context" / "run" / "same-run" / "cuts.json"
                doc = json.loads(cache.read_text())
                doc.update(change)
                cache.write_text(json.dumps(doc))
                calls = len(self.calls)
                run_clip.cut_check(self.args())
                self.assertEqual(len(self.calls), calls + 1)

    def test_subprocess_exception_cleans_temp_and_preserves_old_report(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
        cache = self.root / ".context" / "run" / "same-run" / "cuts.json"
        old = cache.read_bytes()
        self.clip.write_bytes(b"CCCC")
        with (
            patch.object(run_clip.subprocess, "run", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError),
        ):
            run_clip.cut_check(self.args())
        self.assertEqual(cache.read_bytes(), old)
        self.assertEqual(list(cache.parent.glob(".cuts.json.*.json")), [])

    def test_publish_failure_cleans_temp_and_preserves_old_report(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
        cache = self.root / ".context" / "run" / "same-run" / "cuts.json"
        old = cache.read_bytes()
        self.clip.write_bytes(b"DDDD")
        with (
            patch.object(run_clip.subprocess, "run", side_effect=self.fake),
            patch.object(run_clip.os, "replace", side_effect=OSError("collision")) as replace,
            self.assertRaises(SystemExit),
        ):
            run_clip.cut_check(self.args())
        replace.assert_called_once()
        self.assertEqual(cache.read_bytes(), old)
        self.assertEqual(list(cache.parent.glob(".cuts.json.*.json")), [])

    def test_publish_serialization_failure_cleans_temp_and_preserves_old_report(self):
        with patch.object(run_clip.subprocess, "run", side_effect=self.fake):
            run_clip.cut_check(self.args())
        cache = self.root / ".context" / "run" / "same-run" / "cuts.json"
        old = cache.read_bytes()
        self.clip.write_bytes(b"EEEE")
        with (
            patch.object(run_clip.subprocess, "run", side_effect=self.fake),
            patch.object(run_clip.json, "dump", side_effect=OSError("write failed")),
            self.assertRaises(SystemExit),
        ):
            run_clip.cut_check(self.args())
        self.assertEqual(cache.read_bytes(), old)
        self.assertEqual(list(cache.parent.glob(".cuts.json.*.json")), [])


if __name__ == "__main__":
    unittest.main()
