"""Legacy backfill discovery, safety, determinism, and artifact-preservation tests."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.artifacts import LocalCAS
from orchestrator.backfill import (
    discover_legacy_runs,
    import_legacy_run,
    plan_legacy_run,
)
from orchestrator.contracts import AttemptStatus


class BackfillTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "runs" / "legacy-a"
        self.run.mkdir(parents=True)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"original source bytes")

    def write_state(self, stages):
        (self.run / "state.json").write_text(json.dumps({"stages": stages}))

    def test_preserves_successful_and_failed_attempt_outputs_and_hashes_every_file(self):
        clean = self.run / "clean.mp4"
        clean.write_bytes(b"successful clean")
        failed = self.run / "pi3x" / "partial.json"
        failed.parent.mkdir()
        failed.write_bytes(b'{"partial":true}')
        log = self.run / "pi3x.log"
        log.write_text("worker failed after writing a partial result")
        note = self.run / "operator-note.txt"
        note.write_text("keep this too")
        self.write_state(
            {
                "clean": {
                    "attempts": [
                        {"status": "failed", "outputs": ["clean.mp4"], "error": "first try"},
                        {"status": "ok", "outputs": ["clean.mp4"]},
                    ]
                },
                "pi3x": {
                    "status": "failed",
                    "outputs": ["pi3x", "pi3x.log"],
                    "error": "provider failed",
                },
            }
        )

        # One immutable file cannot honestly belong to two attempts.
        with self.assertRaisesRegex(ValueError, "multiple attempts"):
            plan_legacy_run(self.run, source=self.source)

        first = self.run / "clean-first.mp4"
        clean.rename(first)
        clean.write_bytes(b"successful retry")
        state = json.loads((self.run / "state.json").read_text())
        state["stages"]["clean"]["attempts"][0]["outputs"] = ["clean-first.mp4"]
        (self.run / "state.json").write_text(json.dumps(state))
        plan = plan_legacy_run(self.run, source=self.source)

        self.assertEqual(
            [attempt.status for attempt in plan.attempts],
            [
                AttemptStatus.FAILED,
                AttemptStatus.SUCCEEDED,
                AttemptStatus.FAILED,
            ],
        )
        manifests = {manifest.attempt_id: manifest for manifest in plan.manifests}
        for attempt in plan.attempts:
            self.assertEqual(
                set(attempt.output_artifact_ids),
                {item.artifact_id for item in manifests[attempt.id].files},
            )
        self.assertTrue(
            any(
                item.relative_path == "pi3x/partial.json"
                for manifest in plan.manifests
                for item in manifest.files
            )
        )
        by_path = {artifact.metadata["relative_path"]: artifact for artifact in plan.artifacts}
        expected = {
            "clean-first.mp4",
            "clean.mp4",
            "operator-note.txt",
            "pi3x.log",
            "pi3x/partial.json",
            "state.json",
            "source/source.mp4",
        }
        self.assertEqual(set(by_path), expected)
        for relative, artifact in by_path.items():
            path = self.source if relative == "source/source.mp4" else self.run / relative
            self.assertEqual(artifact.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        unassigned = by_path["operator-note.txt"]
        self.assertIsNone(unassigned.producer_attempt_id)

    def test_plan_is_byte_deterministic_and_does_not_execute_subprocesses(self):
        output = self.run / "clean.json"
        output.write_text('{"ok":true}')
        self.write_state({"clean": {"status": "ok", "outputs": ["clean.json"]}})
        with mock.patch("subprocess.run", side_effect=AssertionError("inference launched")):
            first = plan_legacy_run(self.run, source=self.source)
            second = plan_legacy_run(self.run, source=self.source)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_unknown_legacy_stages_remain_insertable_in_the_graph(self):
        partial = self.run / "custom_gpu.partial"
        partial.write_bytes(b"partial provider output")
        self.write_state(
            {
                "custom_gpu": {
                    "status": "failed",
                    "outputs": ["custom_gpu.partial"],
                    "error": "legacy worker failed",
                }
            }
        )
        plan = plan_legacy_run(self.run, source=self.source)
        self.assertIn("custom_gpu", plan.graph.nodes)
        self.assertEqual(plan.graph.nodes["custom_gpu"].status, "failed")
        self.assertEqual(plan.attempts[0].status, AttemptStatus.FAILED)
        self.assertEqual(len(plan.attempts[0].output_artifact_ids), 1)
        self.assertEqual(
            plan.warnings,
            ("legacy stage has no current graph definition: custom_gpu",),
        )

    def test_dry_run_does_not_create_cas_and_apply_copies_verified_blobs(self):
        result = self.run / "pi3x.json"
        result.write_text('{"frames":[]}')
        self.write_state({"pi3x": {"status": "ok", "outputs": ["pi3x.json"]}})
        cas_root = self.root / "cas"
        store = LocalCAS(cas_root)

        dry_plan = import_legacy_run(
            self.run,
            source=self.source,
            store=store,
            dry_run=True,
        )
        self.assertFalse(cas_root.exists())
        applied = import_legacy_run(
            self.run,
            source=self.source,
            store=store,
            dry_run=False,
        )
        self.assertEqual(dry_plan, applied)
        self.assertTrue(all(store.has_blob(item.sha256) for item in applied.artifacts))

    def test_rejects_symlinks_and_metadata_path_escape(self):
        outside = self.root / "secret"
        outside.write_text("secret")
        self.write_state({"clean": {"status": "failed", "outputs": ["../secret"]}})
        with self.assertRaisesRegex(ValueError, "escapes run directory"):
            plan_legacy_run(self.run, source=self.source)

        self.write_state({"clean": {"status": "failed"}})
        (self.run / "leak").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            plan_legacy_run(self.run, source=self.source)

    def test_discovers_runs_in_stable_order_and_rejects_symlinked_trees(self):
        self.write_state({})
        earlier = self.root / "runs" / "a-run"
        earlier.mkdir()
        (earlier / "state.json").write_text('{"stages":{}}')
        self.assertEqual(discover_legacy_runs(self.root / "runs"), (earlier, self.run))

        (self.root / "runs" / "linked").symlink_to(earlier, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            discover_legacy_runs(self.root / "runs")


if __name__ == "__main__":
    unittest.main()
