"""Filesystem isolation, immutable artifacts, and resumable-outbox tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.activities.adapters import default_adapters
from orchestrator.artifacts import (
    LocalCAS,
    ManifestResolver,
    UploadOutbox,
    assign_roles,
    freeze_attempt,
    generate_preview,
    sha256_file,
)
from orchestrator.stages.registry import stage_registry
from orchestrator.workspace import RunWorkspace


class FailingArchive:
    def __init__(self, destination):
        self.destination = destination
        self.failed = False

    def has_blob(self, sha256):
        return self.destination.has_blob(sha256)

    def put_blob(self, sha256, source):
        if not self.failed:
            self.failed = True
            raise ConnectionError("simulated upload interruption")
        self.destination.put_blob(sha256, source)

    def put_manifest(self, key, value):
        self.destination.put_manifest(key, value)


class WorkspaceArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = RunWorkspace(self.root / "runs", "run-1")
        self.workspace.initialize()
        self.store = LocalCAS(self.root / "cas")
        self.store.initialize()

    def test_workspace_separates_inputs_agent_attempts_and_manifests(self):
        source = self.root / "source.mp4"
        source.write_bytes(b"source bytes")
        hydrated = self.workspace.hydrate_input("source_video", source)
        self.assertEqual(hydrated.read_bytes(), b"source bytes")
        self.assertEqual(hydrated.stat().st_mode & 0o777, 0o400)
        for directory in (
            self.workspace.agent,
            self.workspace.attempts,
            self.workspace.manifests,
            self.workspace.outbox,
        ):
            self.assertTrue(directory.is_dir())

    def test_attempts_are_exclusive_and_finalize_once(self):
        attempt = self.workspace.create_attempt("clean", "attempt-1", {"parameters": {}})
        self.assertTrue(attempt.outputs.is_dir())
        self.assertEqual(json.loads(attempt.request.read_text()), {"parameters": {}})
        attempt.finalize({"status": "failed"})
        with self.assertRaises(FileExistsError):
            attempt.finalize({"status": "succeeded"})
        with self.assertRaises(FileExistsError):
            self.workspace.create_attempt("clean", "attempt-1", {})

    def test_failed_attempt_freezes_logs_and_partial_outputs(self):
        attempt = self.workspace.create_attempt("clean", "attempt-1", {"parameters": {}})
        attempt.stdout.write_text("provider started\n")
        (attempt.outputs / "partial.png").write_bytes(b"partial image")
        attempt.finalize({"status": "failed", "error": "worker lost"})
        manifest = freeze_attempt(
            attempt,
            self.store,
            status="failed",
            roles={"outputs/partial.png": "clean_frame"},
        )
        paths = {item.relative_path for item in manifest.files}
        self.assertEqual(
            paths,
            {"outputs/partial.png", "request.json", "result.json", "stdout.log"},
        )
        partial = next(item for item in manifest.files if item.role == "clean_frame")
        self.assertTrue(self.store.has_blob(partial.sha256))
        restored = self.root / "restored.png"
        self.store.hydrate(partial.sha256, restored)
        self.assertEqual(restored.read_bytes(), b"partial image")

    def test_symlinks_are_rejected_from_inputs_and_attempts(self):
        source = self.root / "source"
        source.write_bytes(b"secret")
        link = self.root / "source-link"
        link.symlink_to(source)
        with self.assertRaises(ValueError):
            self.workspace.hydrate_input("source_video", link)
        attempt = self.workspace.create_attempt("clean", "attempt-1", {})
        (attempt.outputs / "leak").symlink_to(source)
        with self.assertRaises(ValueError):
            freeze_attempt(attempt, self.store, status="failed")

    def test_outbox_resumes_without_rerunning_or_losing_artifacts(self):
        attempt = self.workspace.create_attempt("pi3x", "attempt-1", {})
        (attempt.outputs / "cameras.json").write_text('{"frames":[]}')
        attempt.finalize({"status": "succeeded"})
        manifest = freeze_attempt(
            attempt,
            self.store,
            status="succeeded",
            roles={"outputs/cameras.json": "cameras"},
        )
        destination = LocalCAS(self.root / "archive")
        destination.initialize()
        archive = FailingArchive(destination)
        outbox = UploadOutbox(self.workspace.outbox, self.store, archive)
        entry = outbox.enqueue(manifest)
        with self.assertRaises(ConnectionError):
            outbox.flush()
        self.assertEqual(json.loads(entry.read_text())["status"], "pending")
        self.assertEqual(outbox.flush(), 1)
        self.assertEqual(json.loads(entry.read_text())["status"], "uploaded")
        for artifact in manifest.files:
            self.assertTrue(destination.has_blob(artifact.sha256))
            self.assertEqual(
                sha256_file(destination.blob_path(artifact.sha256)),
                artifact.sha256,
            )

    def test_dashboard_previews_are_bounded_derived_files(self):
        image = self.root / "large.png"
        Image.new("RGB", (2400, 1600), (12, 34, 56)).save(image)
        preview = generate_preview(image, self.root / "preview", "image/png")
        self.assertIsNotNone(preview)
        with Image.open(preview) as result:
            self.assertLessEqual(result.width, 1280)
            self.assertLessEqual(result.height, 720)
        text = self.root / "report.json"
        text.write_text("x" * (70 * 1024))
        text_preview = generate_preview(text, self.root / "report-preview", "application/json")
        self.assertEqual(text_preview.stat().st_size, 64 * 1024)

    def test_manifest_resolver_preserves_filenames_for_existing_scripts(self):
        attempt = self.workspace.create_attempt("pi3x", "attempt-1", {})
        cameras = attempt.outputs / "cameras.json"
        cameras.write_text('{"cameras":[]}')
        attempt.finalize({"status": "succeeded"})
        manifest = freeze_attempt(
            attempt,
            self.store,
            status="succeeded",
            roles={"outputs/cameras.json": "cameras"},
        )
        artifact = next(item for item in manifest.files if item.role == "cameras")
        resolver = ManifestResolver(self.store, [manifest])
        hydrated = resolver.hydrate(artifact.artifact_id, self.root / "inputs/cameras")
        self.assertEqual(hydrated.name, "cameras.json")
        self.assertEqual(hydrated.read_bytes(), cameras.read_bytes())


class BundleRoleTests(unittest.TestCase):
    """A directory of files is one role, and every file in it has to carry that role.

    `PurePath.match` treats `**` as one segment and anchors from the right, so the bundle
    patterns matched nothing while the QA validator's `Path.glob` matched everything. The
    outputs passed QA and were frozen as `attempt_file`, and the next stage asked for the
    role and got nothing.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, *relatives):
        for relative in relatives:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)

    def test_every_file_in_a_bundle_carries_the_bundle_role(self):
        self.write(
            "outputs/prepared-person/mask.png",
            "outputs/prepared-person/prepared.json",
            "outputs/prepared-person/scores/frame-scores.json",
            "outputs/unrelated.txt",
        )
        assigned = assign_roles(self.root, {"outputs/prepared-person/**/*": "prepared_person"})
        self.assertEqual(
            sorted(assigned),
            [
                "outputs/prepared-person/mask.png",
                "outputs/prepared-person/prepared.json",
                "outputs/prepared-person/scores/frame-scores.json",
            ],
        )
        self.assertEqual(set(assigned.values()), {"prepared_person"})

    def test_a_named_file_keeps_its_own_role_inside_a_bundle(self):
        self.write(
            "outputs/lhm-frozen/model.ply",
            "outputs/lhm-frozen/recovery-receipt.json",
        )
        assigned = assign_roles(
            self.root,
            {
                "outputs/lhm-frozen/**/*": "canonical_person",
                "outputs/lhm-frozen/recovery-receipt.json": "recovery_receipt",
            },
        )
        self.assertEqual(assigned["outputs/lhm-frozen/recovery-receipt.json"], "recovery_receipt")
        self.assertEqual(assigned["outputs/lhm-frozen/model.ply"], "canonical_person")

    def test_a_pattern_only_matches_from_the_top_of_the_attempt(self):
        """`path.match` matched from the right, so a stray copy took the role too."""
        self.write("outputs/crops/a.png", "inputs/outputs/crops/b.png")
        assigned = assign_roles(self.root, {"outputs/crops/*.png": "object_crops"})
        self.assertEqual(list(assigned), ["outputs/crops/a.png"])


class DeclaredBundleTests(unittest.TestCase):
    def test_a_role_globbed_as_a_bundle_is_declared_as_many(self):
        """Otherwise QA rejects the stage: "produced 4 files, expected one"."""
        for executor, adapter in default_adapters().items():
            patterns = getattr(adapter, "output_roles", None)
            if not patterns:
                continue
            many = {
                role for pattern, role in patterns.items() if "*" in pattern.replace("{name}", "")
            }
            for stage in stage_registry().values():
                if stage.executor != executor:
                    continue
                for name, contract in stage.outputs.items():
                    if contract.role in many:
                        with self.subTest(stage=stage.id, output=name):
                            self.assertTrue(
                                contract.multiple,
                                f"{stage.id}.{name} is globbed as many files but declared as one",
                            )


if __name__ == "__main__":
    unittest.main()
