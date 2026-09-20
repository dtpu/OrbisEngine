"""Offline source-bound authored prompt admission and no-API regressions."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_clip


class SuppliedPromptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clip = self.root / "selected.mov"
        self.clip.write_bytes(b"selected shot bytes, not original full clip")
        self.input = self.root / "authored.json"
        self.doc = {
            "text_prompt": "  Stone steps between brick walls.  ",
            "sourceSha256": run_clip.file_sha256(self.clip),
            "author": "operator fixture",
            "provenance": "description of source frames; no quality verdict",
        }
        self.input.write_text(json.dumps(self.doc))
        self.pipeline = object.__new__(run_clip.Pipeline)
        p = self.pipeline
        p.clip = self.clip
        p.ctx = self.root / "run"
        p.ctx.mkdir()
        p.state = run_clip.State(p.ctx / "state.json")
        p.a = SimpleNamespace(
            world_prompt_file=self.input,
            only=None,
            force=None,
            skip_finetune=True,
            reuse_world=None,
        )
        p.stages = ["clean", "pi3x", "world_prompt"]
        p.deps = {stage: [] for stage in p.stages}
        p.marble = "video"
        p.multi = False
        p.summary = Mock()

    def test_valid_copy_has_explicit_authorship_and_no_api_call(self):
        with patch.object(run_clip, "run") as invoke:
            self.pipeline.world_prompt()
        invoke.assert_not_called()
        result = json.loads(self.pipeline.prompt_json.read_text())
        self.assertEqual(result["text_prompt"], self.doc["text_prompt"].strip())
        self.assertEqual(result["sourceSha256"], self.doc["sourceSha256"])
        self.assertEqual(result["origin"], "supplied")
        self.assertEqual(result["authorship"], "externally-authored")
        self.assertNotIn("passed", result)
        self.assertNotIn("reviewer", result)
        self.assertEqual(self.pipeline.state.data["stages"]["world_prompt"]["origin"], "supplied")

    def test_wrong_source_rejected_before_any_stage_launch(self):
        self.doc["sourceSha256"] = "0" * 64
        self.input.write_text(json.dumps(self.doc))
        with patch.object(run_clip.threading, "Thread") as thread:
            with self.assertRaisesRegex(ValueError, "selected clip bytes"):
                self.pipeline.go()
        thread.assert_not_called()
        self.assertFalse(self.pipeline.prompt_json.exists())

    def test_malformed_inputs_never_write_or_call_api(self):
        invalid = [
            "not json",
            "[]",
            json.dumps({}),
            json.dumps(dict(self.doc, text_prompt=" ")),
            json.dumps(dict(self.doc, text_prompt=float("nan"))),
            json.dumps(dict(self.doc, text_prompt="x" * 10001)),
            json.dumps(dict(self.doc, text_prompt="x\0y")),
            json.dumps(dict(self.doc, author="")),
            json.dumps(dict(self.doc, provenance=False)),
            json.dumps(dict(self.doc, sourceSha256="invalid")),
            json.dumps(dict(self.doc, reviewed=True)),
            " " * 65537,
        ]
        for raw in invalid:
            with self.subTest(raw=raw[:80]), patch.object(run_clip, "run") as invoke:
                self.input.write_text(raw)
                with self.assertRaises(ValueError):
                    self.pipeline.world_prompt()
                invoke.assert_not_called()
                self.assertFalse(self.pipeline.prompt_json.exists())

    def test_clean_pi3x_selection_does_not_read_supplied_file(self):
        self.input.unlink()
        self.pipeline.a.only = "clean,pi3x"
        for stage in ("clean", "pi3x"):
            self.pipeline.state.record(stage, status="ok")
        with patch.object(run_clip, "supplied_world_prompt") as read:
            self.assertEqual(self.pipeline.go(), 0)
        read.assert_not_called()

    def test_changed_cached_prompt_blocks_without_paid_retry(self):
        self.pipeline.world_prompt()
        self.pipeline.state.record("world_prompt", status="ok")
        self.doc["text_prompt"] = "A changed source description."
        self.input.write_text(json.dumps(self.doc))
        with patch.object(run_clip.threading, "Thread") as thread:
            with self.assertRaisesRegex(ValueError, "new run name"):
                self.pipeline.go()
        thread.assert_not_called()

    def test_default_generated_path_retains_existing_command(self):
        self.pipeline.a.world_prompt_file = None

        def generated(*args, **kwargs):
            self.pipeline.prompt_json.write_text(json.dumps({"text_prompt": "generated"}))

        with patch.object(run_clip, "run", side_effect=generated) as invoke:
            self.pipeline.world_prompt()
        self.assertEqual(invoke.call_args.args[0][1], "scripts/world_prompt.py")
        self.assertEqual(invoke.call_args.kwargs["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
