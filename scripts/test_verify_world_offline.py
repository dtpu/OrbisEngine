"""Tiny local diagnostic captures; no model, browser, GPU or generated-world inference."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import verify_world
import world_quality_review
from quality_gate import ACKNOWLEDGEMENTS, file_sha256
from test_world_quality_review import create_fixture, write_json


class OfflineDiagnosticTests(unittest.TestCase):
    def test_default_capture_is_offline_and_generated_template_cannot_accept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs, _documents = create_fixture(root)
            clip = root / "actual.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=16x16:rate=1",
                    "-frames:v",
                    "3",
                    "-c:v",
                    "ffv1",
                    str(clip),
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
            out = root / "capture"
            argv = [
                "verify_world.py",
                "--world",
                str(kwargs["world"]),
                "--cameras",
                str(kwargs["cameras"]),
                "--clip",
                str(clip),
                "--width",
                "16",
                "--n-frames",
                "3",
                "--out",
                str(out),
            ]

            def forbidden(*args, **options):
                raise AssertionError("a diagnostic must not call a model")

            with (
                patch.object(sys, "argv", argv),
                patch.object(verify_world, "read_spz", return_value={}),
                patch.object(
                    verify_world, "render", return_value=(np.zeros((16, 16, 3)), np.ones((16, 16)))
                ),
                patch.dict(sys.modules, {"vlm_judge": SimpleNamespace(ask_images=forbidden)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                verify_world.main()
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["status"], "diagnostic")
            for forbidden_key in ("pass", "regenerate", "medianScore", "correction"):
                self.assertNotIn(forbidden_key, report)
            arguments = dict(
                kwargs,
                clip=clip,
                plan_path=out / "plan.json",
                result_path=out / "review-template.json",
                evidence_root=out,
            )
            self.assertEqual(world_quality_review.assess(**arguments)["status"], "blocked")
            result = json.loads(arguments["result_path"].read_text())
            result["reviewer"] = "Synthetic fixture reviewer, not a real visual judgment"
            result["acknowledgements"] = ACKNOWLEDGEMENTS.copy()
            for sample in result["samples"]:
                for judgment in sample["criteria"].values():
                    judgment.update(status="pass", reason="Synthetic contract assertion")
            write_json(arguments["result_path"], result)
            self.assertEqual(world_quality_review.assess(**arguments)["status"], "passed")
            before = {p.name: file_sha256(p) for p in out.iterdir()}
            with (
                patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                verify_world.main()
            self.assertEqual(before, {p.name: file_sha256(p) for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
