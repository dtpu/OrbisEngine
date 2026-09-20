"""An attempt's files must reach the archive even if the flush failed when it ran.

A stage enqueues its outputs and flushes immediately, but a flush can fail — a full disk is the
usual reason — while the attempt still succeeds, because its outputs are already in the local
store. Nothing retried those entries, so the database recorded artifacts whose blobs the archive
never received, and the next stage to want one failed to hydrate its input.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.artifacts import LocalCAS, UploadOutbox, freeze_attempt
from orchestrator.artifacts import flush_pending_outboxes
from orchestrator.workspace import RunWorkspace

RUN_ID = "run-outbox"


class OutboxRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-outbox-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.store = LocalCAS(self.root / "cas")
        self.store.initialize()
        self.archive = LocalCAS(self.root / "archive")
        self.archive.initialize()
        self.workspaces = self.root / "runs"

    def enqueue_attempt(self, attempt_id="run-outbox:1:1", payload=b"cameras"):
        """An attempt that finished with its manifest queued but never flushed."""
        workspace = RunWorkspace(self.workspaces, RUN_ID)
        workspace.initialize()
        attempt = workspace.create_attempt("pi3x", attempt_id, {"runId": RUN_ID})
        (attempt.outputs / "cameras.json").write_bytes(payload)
        attempt.finalize({"status": "succeeded", "error": None, "command": []})
        manifest = freeze_attempt(
            attempt, self.store, status="succeeded", roles={"outputs/cameras.json": "cameras"}
        )
        UploadOutbox(workspace.outbox, self.store, self.archive).enqueue(manifest)
        return manifest

    def archived(self, manifest):
        return all(self.archive.has_blob(f.sha256) for f in manifest.files)

    def test_a_queued_manifest_reaches_the_archive_on_recovery(self):
        manifest = self.enqueue_attempt()
        self.assertFalse(self.archived(manifest), "precondition: nothing flushed yet")
        self.assertEqual(flush_pending_outboxes(self.workspaces, self.store, self.archive), 1)
        self.assertTrue(self.archived(manifest))

    def test_recovery_is_idempotent(self):
        self.enqueue_attempt()
        self.assertEqual(flush_pending_outboxes(self.workspaces, self.store, self.archive), 1)
        self.assertEqual(flush_pending_outboxes(self.workspaces, self.store, self.archive), 0)

    def test_nothing_to_do_costs_nothing(self):
        self.workspaces.mkdir(parents=True, exist_ok=True)
        self.assertEqual(flush_pending_outboxes(self.workspaces, self.store, self.archive), 0)

    def test_one_unreadable_run_does_not_stop_the_others(self):
        good = self.enqueue_attempt()
        broken = self.workspaces / "run-broken" / "outbox"
        broken.mkdir(parents=True)
        (broken / "corrupt.json").write_text("{not json")
        self.assertEqual(flush_pending_outboxes(self.workspaces, self.store, self.archive), 1)
        self.assertTrue(self.archived(good))


if __name__ == "__main__":
    unittest.main()
