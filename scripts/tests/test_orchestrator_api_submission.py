"""Submission integration tests use the production adapter, SQLite, and actual byte storage."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from orchestrator.api.app import ApiSettings
from orchestrator.api.service import create_service
from orchestrator.api.storage import FilesystemSourceStore
from orchestrator.config import REPOSITORY_ROOT, resolve_runs_root
from orchestrator.database import ArtifactRecord, Base, EventRecord, NodeRecord, RunRecord
from orchestrator.repository import PipelineRepository
from orchestrator.supervisor import runs_needing_work

TOKEN = "test-token-with-at-least-24-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
VIDEO = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"source-video-data" * 100


class ProductionSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-production-api-")).resolve()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.temporary = self.root / "temporary"
        self.temporary.mkdir()
        temporary_patch = patch.object(tempfile, "tempdir", str(self.temporary))
        temporary_patch.start()
        self.addCleanup(temporary_patch.stop)
        self.engine = create_engine(
            f"sqlite:///{self.root / 'pipeline.sqlite'}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.repository = PipelineRepository(self.sessions)
        self.settings = ApiSettings(bearer_token=TOKEN, code_revision="1234567")
        self.app = create_service(
            repository=self.repository,
            runs_root=self.root / "runs",
            settings=self.settings,
            artifact_root=self.root / "artifacts",
        )
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def upload(self, run_id="run-upload", filename="camera.MOV", content=VIDEO, **data):
        return self.client.post(
            "/api/pipeline/runs/upload",
            headers=AUTH,
            data={"requested_id": run_id, "options": '{"marble":"none"}', **data},
            files={"source": (filename, content, "video/quicktime")},
        )

    def replay(self, source, run_id="run-replay", **changes):
        return self.client.post(
            "/api/pipeline/runs",
            headers=AUTH,
            json={
                "id": run_id,
                "source_sha256": source["sourceSha256"],
                "source_artifact_id": source["sourceArtifactId"],
                "options": {"marble": "none"},
                **changes,
            },
        )

    def test_monetary_cap_is_rejected_before_source_storage(self):
        response = self.upload(budget=json.dumps({"maximum_cost": 0}))
        self.assertEqual(response.status_code, 422)
        self.assertIn("not enforced", response.json()["detail"])
        self.assert_no_submission()
        self.assertFalse(list((self.root / "artifacts").rglob("*")))

    def manifests(self):
        return list((self.root / "runs").glob("*/run.json"))

    def assert_no_submission(self, run_id="run-upload"):
        for record in (RunRecord, ArtifactRecord, NodeRecord, EventRecord):
            with self.sessions() as session:
                column = record.id if record is RunRecord else record.run_id
                self.assertEqual(session.scalars(select(record).where(column == run_id)).all(), [])
        self.assertFalse(
            any(json.loads(path.read_text())["runId"] == run_id for path in self.manifests())
        )
        self.assertEqual(list(self.temporary.iterdir()), [])
        self.assertEqual(list((self.root / "runs" / ".submissions").glob("*")), [])

    def test_upload_publishes_verified_source_and_resumable_plan(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["sourceSha256"], hashlib.sha256(VIDEO).hexdigest())
        manifest = self.manifests()[0]
        described = json.loads(manifest.read_text())
        source = Path(described["source"])
        self.assertEqual(source, manifest.parent / "source.mov")
        self.assertEqual(source.read_bytes(), VIDEO)
        self.assertEqual(described["sourceArtifactId"], body["sourceArtifactId"])
        artifact = self.repository.get_source_artifact(body["sourceArtifactId"])
        self.assertEqual((self.root / "artifacts" / artifact["storageKey"]).read_bytes(), VIDEO)
        self.assertTrue(self.repository.run_summary(body["id"])["nodes"])
        self.assertEqual(runs_needing_work(self.root / "runs"), [manifest.parent])
        self.assertEqual(list(self.temporary.iterdir()), [])
        # A fresh adapter after process restart discovers the same immutable local input.
        restarted = create_service(
            repository=self.repository,
            runs_root=self.root / "runs",
            settings=self.settings,
            artifact_root=self.root / "artifacts",
        )
        self.assertEqual(restarted.state.runs._dir(body["id"]), manifest.parent)
        self.assertEqual(source.read_bytes(), VIDEO)

    def test_json_replay_hydrates_and_verifies_existing_artifact(self):
        uploaded = self.upload().json()
        replay = self.replay(uploaded)
        self.assertEqual(replay.status_code, 201, replay.text)
        directory = self.app.state.runs._dir("run-replay")
        described = json.loads((directory / "run.json").read_text())
        self.assertEqual(Path(described["source"]).read_bytes(), VIDEO)
        self.assertEqual(Path(described["source"]).suffix, ".mov")
        self.assertEqual(described["sourceArtifactId"], uploaded["sourceArtifactId"])
        # Source storage and the original upload can disappear after hydration; replay is local.
        original = self.app.state.runs._dir("run-upload")
        shutil.rmtree(original)
        shutil.rmtree(self.root / "artifacts")
        self.assertEqual(Path(described["source"]).read_bytes(), VIDEO)
        self.assertEqual(list(self.temporary.iterdir()), [])

    def test_missing_corrupt_and_wrong_digest_replays_leave_no_run(self):
        source = self.upload().json()
        wrong = self.replay(source, source_sha256="0" * 64)
        self.assertEqual(wrong.status_code, 422, wrong.text)
        self.assert_no_submission("run-replay")
        artifact = self.repository.get_source_artifact(source["sourceArtifactId"])
        archived = self.root / "artifacts" / artifact["storageKey"]
        archived.write_bytes(b"tampered")
        corrupt = self.replay(source)
        self.assertEqual(corrupt.status_code, 422, corrupt.text)
        self.assert_no_submission("run-replay")
        archived.unlink()
        missing = self.replay(source)
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assert_no_submission("run-replay")
        unknown = self.replay(source, source_artifact_id="artifact:unknown")
        self.assertEqual(unknown.status_code, 404, unknown.text)
        self.assert_no_submission("run-replay")

    def test_generated_artifact_cannot_be_replayed_as_source(self):
        source = self.upload().json()
        with self.sessions.begin() as session:
            artifact = session.get(ArtifactRecord, source["sourceArtifactId"])
            artifact.role = "world_spz"
        response = self.replay(source)
        self.assertEqual(response.status_code, 422, response.text)
        self.assert_no_submission("run-replay")

    def test_duplicate_upload_and_replay_preserve_original_bytes(self):
        original = self.upload()
        self.assertEqual(original.status_code, 201, original.text)
        path = self.manifests()[0]
        before = path.read_bytes()
        duplicate = self.upload(content=b"replacement")
        self.assertEqual(duplicate.status_code, 409, duplicate.text)
        replay = self.replay(original.json(), run_id="run-upload")
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(Path(json.loads(before)["source"]).read_bytes(), VIDEO)
        self.assertEqual(len(self.manifests()), 1)
        self.assertEqual(list(self.temporary.iterdir()), [])

    def test_same_filename_gets_distinct_sources_and_directories(self):
        first = self.upload("run-first")
        second = self.upload("run-second", content=VIDEO + b"second")
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(second.status_code, 201, second.text)
        for run_id, expected in (("run-first", VIDEO), ("run-second", VIDEO + b"second")):
            directory = self.app.state.runs._dir(run_id)
            source = Path(json.loads((directory / "run.json").read_text())["source"])
            self.assertEqual(source.read_bytes(), expected)
        self.assertEqual(len(self.manifests()), 2)

    def test_size_limit_and_empty_upload_leave_no_files_or_rows(self):
        self.settings = self.settings.model_copy(update={"maximum_upload_bytes": 10})
        app = create_service(
            repository=self.repository,
            runs_root=self.root / "runs",
            settings=self.settings,
            artifact_root=self.root / "artifacts",
        )
        self.client.close()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        oversized = self.upload()
        self.assertEqual(oversized.status_code, 413, oversized.text)
        self.assert_no_submission()
        empty = self.upload(content=b"")
        self.assertEqual(empty.status_code, 422, empty.text)
        self.assert_no_submission()

    def test_archive_failure_removes_upload_temporary(self):
        with patch.object(FilesystemSourceStore, "put", side_effect=OSError("disk unavailable")):
            response = self.upload()
        self.assertEqual(response.status_code, 500, response.text)
        self.assert_no_submission()

    def test_interrupted_archive_copy_cleans_partial_bytes(self):
        original = shutil.copyfile

        def interrupted(source, destination, *args, **kwargs):
            if str(destination).endswith(".partial"):
                Path(destination).write_bytes(b"incomplete")
                raise OSError("archive copy interrupted")
            return original(source, destination, *args, **kwargs)

        with patch("orchestrator.api.storage.shutil.copyfile", side_effect=interrupted):
            response = self.upload()
        self.assertEqual(response.status_code, 500, response.text)
        self.assert_no_submission()
        self.assertFalse(any(path.is_file() for path in (self.root / "artifacts").rglob("*")))

    def test_long_filename_preserves_extension_and_long_id_fits_artifact_column(self):
        response = self.upload(run_id="r" * 128, filename="x" * 200 + ".MOV")
        self.assertEqual(response.status_code, 201, response.text)
        self.assertLessEqual(len(response.json()["sourceArtifactId"]), 128)
        directory = self.app.state.runs._dir("r" * 128)
        described = json.loads((directory / "run.json").read_text())
        self.assertEqual(Path(described["source"]).suffix, ".mov")
        self.assertEqual(Path(described["source"]).read_bytes(), VIDEO)

    def test_artifact_row_failure_compensates_run_row_and_initial_event(self):
        with patch.object(
            self.repository, "add_source_artifact", side_effect=RuntimeError("db failed")
        ):
            response = self.upload()
        self.assertEqual(response.status_code, 500, response.text)
        self.assert_no_submission()

    def test_partial_plan_failure_compensates_rows_and_staged_run(self):
        original = self.repository.ensure_node
        seeded = []

        def fail_after_first(**kwargs):
            if seeded:
                raise OSError("plan interrupted")
            seeded.append(kwargs["node_id"])
            return original(**kwargs)

        with patch.object(self.repository, "ensure_node", side_effect=fail_after_first):
            response = self.upload()
        self.assertEqual(response.status_code, 500, response.text)
        self.assertTrue(seeded)
        self.assert_no_submission()

    def test_json_replay_plan_failure_compensates_without_deleting_source(self):
        source = self.upload().json()
        with patch.object(self.repository, "ensure_node", side_effect=OSError("plan interrupted")):
            response = self.replay(source)
        self.assertEqual(response.status_code, 500, response.text)
        self.assert_no_submission("run-replay")
        self.assertEqual(
            self.repository.get_source_artifact(source["sourceArtifactId"])["sha256"],
            hashlib.sha256(VIDEO).hexdigest(),
        )
        self.assertEqual(len(self.manifests()), 1)

    def test_invalid_run_ids_and_untrusted_filename_cannot_escape_root(self):
        for run_id in ("../escaped", "/tmp/escaped", "..", "a/b", "a\\b"):
            response = self.upload(run_id=run_id)
            self.assertEqual(response.status_code, 422, (run_id, response.text))
            self.assert_no_submission(run_id)
            replay = self.client.post(
                "/api/pipeline/runs",
                headers=AUTH,
                json={"id": run_id, "source_sha256": "a" * 64, "source_artifact_id": "unknown"},
            )
            self.assertEqual(replay.status_code, 422, replay.text)
        response = self.upload(filename="../../escaped.MOV")
        self.assertEqual(response.status_code, 201, response.text)
        directory = self.app.state.runs._dir("run-upload")
        self.assertTrue(directory.is_relative_to(self.root / "runs"))
        self.assertEqual((directory / "source.mov").read_bytes(), VIDEO)

    def test_options_schema_is_authenticated_and_matches_validation(self):
        self.assertEqual(self.client.get("/api/pipeline/options").status_code, 401)
        response = self.client.get("/api/pipeline/options", headers=AUTH)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["additionalProperties"], False)
        self.assertIn("marble", response.json()["properties"])


class ServiceConfigurationTests(unittest.TestCase):
    def test_import_does_not_require_environment_or_contact_cloud(self):
        import orchestrator.api.service as service

        with patch.dict(os.environ, {}, clear=True), patch("boto3.client") as cloud:
            importlib.reload(service)
        cloud.assert_not_called()

    def test_runs_root_is_shared_and_independent_of_current_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_runs_root(), REPOSITORY_ROOT / ".context/runs")
        with patch.dict(os.environ, {"WANDER_RUNS_ROOT": "custom/runs"}):
            self.assertEqual(resolve_runs_root(), REPOSITORY_ROOT / "custom/runs")
            self.assertEqual(resolve_runs_root("override"), REPOSITORY_ROOT / "override")
            self.assertEqual(
                resolve_runs_root("/tmp/explicit-runs"), Path("/tmp/explicit-runs").resolve()
            )
