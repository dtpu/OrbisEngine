"""No-spend checks for the pipeline's S3 completion hook."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("run_clip", Path(__file__).with_name("run_clip.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublishCompletionTests(unittest.TestCase):
    def run_main(self, stage_status=0, publish_status=0, extra=()):
        with patch.object(sys, "argv", ["run_clip.py", "--clip", "unused.mp4", "--name", "fixture", *extra]), \
             patch.object(module, "resolve_shots", return_value=[("fixture", Path("unused.mp4"), None)]), \
             patch.object(module, "Pipeline") as pipeline, \
             patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=publish_status)) as publish:
            pipeline.return_value.go.return_value = stage_status
            with self.assertRaises(SystemExit) as result:
                module.main()
            return result.exception.code, publish.call_args_list

    def test_success_archives_automatically(self):
        status, calls = self.run_main()
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0][:2], ["node", "scripts/publish-runs.mjs"])

    def test_failed_and_gated_runs_are_saved(self):
        for code in (1, 2):
            status, calls = self.run_main(stage_status=code)
            self.assertEqual(status, code)
            self.assertEqual(len(calls), 1)

    def test_publish_failure_is_not_reported_as_success(self):
        status, _ = self.run_main(publish_status=1)
        self.assertEqual(status, 3)

    def test_offline_opt_out_does_not_touch_cloud(self):
        status, calls = self.run_main(extra=("--no-publish",))
        self.assertEqual(status, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
