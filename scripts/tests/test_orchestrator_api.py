"""Authenticated API, durable events, and workflow-command tests."""

import hashlib
import sys
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from orchestrator.api.app import ApiSettings, create_app
from orchestrator.api.storage import FilesystemArtifactReader, FilesystemSourceStore
from orchestrator.database import ArtifactRecord, AttemptRecord, Base
from orchestrator.repository import PipelineRepository

TOKEN = "test-token-with-at-least-24-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Runs:
    """Runs are directories; opening one is making it. Nothing is scheduled."""

    def __init__(self, root, repository=None):
        self.root = Path(root)
        self.repository = repository
        self.started = []
        self.paused = {}

    def open(self, *, run_id, name, source, options, source_artifact_id=None):
        from orchestrator.supervisor import open_run

        self.started.append(run_id)
        return open_run(
            self.root,
            run_id=run_id,
            name=name,
            source=source,
            options=options,
            repository=self.repository,
            source_artifact_id=source_artifact_id,
        )

    def journal(self, run_id):
        from orchestrator.journal import Journal

        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return Journal(run_dir)

    def pause(self, run_id, paused):
        self.paused[run_id] = paused


class SourceStore:
    def __init__(self):
        self.values = {}

    def put(self, source, storage_key):
        self.values[storage_key] = source.read_bytes()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        self.root = Path(tempfile.mkdtemp(prefix="wander-api-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.repository = PipelineRepository(sessions)
        self.runs = Runs(self.root / "runs", self.repository)
        self.source_store = SourceStore()
        app = create_app(
            self.repository,
            self.runs,
            ApiSettings(bearer_token=TOKEN, code_revision="1234567"),
            self.source_store,
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def create_run(self):
        response = self.client.post(
            "/api/pipeline/runs",
            headers=AUTH,
            json={
                "id": "run-1",
                "source_sha256": "a" * 64,
                "source_artifact_id": "artifact:source",
                "options": {"marble": "none", "objects": False},
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response

    def test_authentication_and_run_creation(self):
        unauthorized = self.client.post("/api/pipeline/runs", json={})
        self.assertEqual(unauthorized.status_code, 401)
        response = self.create_run()
        self.assertEqual(response.json()["id"], "run-1")
        self.assertEqual(len(self.runs.started), 1)
        summary = self.client.get("/api/pipeline/runs/run-1", headers=AUTH)
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["live"]["source"], "journal")
        self.assertTrue(summary.json()["nodes"])
        listing = self.client.get("/api/pipeline", headers=AUTH)
        self.assertEqual([run["id"] for run in listing.json()["runs"]], ["run-1"])

    def test_messages_approvals_and_commands_are_durable_before_signal(self):
        self.create_run()
        message = self.client.post(
            "/api/pipeline/runs/run-1/messages",
            headers=AUTH,
            json={"message": "Inspect the left edge", "node_id": "clean"},
        )
        self.assertEqual(message.status_code, 202)
        command = self.client.post(
            "/api/pipeline/runs/run-1/pause",
            headers=AUTH,
            json={"node_id": "clean", "rationale": "Wait for operator"},
        )
        self.assertEqual(command.status_code, 202)
        # What an operator says is a line in the run's journal, which the agent reads at
        # the start of its next turn; pausing is a marker the supervisor honours.
        self.assertEqual(
            [e.data.get("text") for e in self.runs.journal("run-1").entries()],
            ["Inspect the left edge"],
        )
        self.assertIs(self.runs.paused["run-1"], True)
        events = self.repository.events_after("run-1", 0)
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        resumed = self.repository.events_after("run-1", events[0].sequence)
        self.assertEqual(resumed, events[1:])

    def test_duplicate_run_is_conflict(self):
        self.create_run()
        response = self.client.post(
            "/api/pipeline/runs",
            headers=AUTH,
            json={
                "id": "run-1",
                "source_sha256": "a" * 64,
                "source_artifact_id": "artifact:source",
                "options": {"marble": "none", "objects": False},
            },
        )
        self.assertEqual(response.status_code, 409)

    def test_source_upload_hashes_archives_and_starts_run(self):
        response = self.client.post(
            "/api/pipeline/runs/upload",
            headers=AUTH,
            data={
                "requested_id": "run-upload",
                "options": '{"marble":"none","objects":false}',
                "budget": "{}",
            },
            files={"source": ("clip.mp4", b"deterministic-video", "video/mp4")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["id"], "run-upload")
        self.assertEqual(
            self.source_store.values[f"sources/{body['sourceSha256'][:2]}/{body['sourceSha256']}"],
            b"deterministic-video",
        )
        summary = self.repository.run_summary("run-upload")
        self.assertEqual(summary["artifacts"][0]["role"], "source_video")


class ArtifactReadTests(unittest.TestCase):
    """The admin viewer reads artifact bytes through the API; the bytes are untrusted."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.repository = PipelineRepository(self.sessions)
        self.root = Path(tempfile.mkdtemp(prefix="wander-artifacts-"))
        app = create_app(
            self.repository,
            Runs(self.root / "runs"),
            ApiSettings(bearer_token=TOKEN, code_revision="1234567"),
            FilesystemSourceStore(self.root),
            artifact_reader=FilesystemArtifactReader(self.root),
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        for run_id in ("run-a", "run-b"):
            response = self.client.post(
                "/api/pipeline/runs",
                headers=AUTH,
                json={
                    "id": run_id,
                    "source_sha256": "a" * 64,
                    "source_artifact_id": f"artifact:{run_id}:source",
                    "options": {"marble": "none", "objects": False},
                },
            )
            self.assertEqual(response.status_code, 201, response.text)

    def store(self, run_id, artifact_id, content, media_type, *, name=None, attempt_id=None):
        digest = hashlib.sha256(content).hexdigest()
        storage_key = f"blobs/{digest[:2]}/{digest}"
        destination = self.root / storage_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        with self.sessions.begin() as session:
            session.add(
                ArtifactRecord(
                    id=artifact_id,
                    run_id=run_id,
                    role="clean_frame",
                    sha256=digest,
                    size=len(content),
                    media_type=media_type,
                    storage_key=storage_key,
                    producer_attempt_id=attempt_id,
                    artifact_metadata={"relative_path": name} if name else {},
                    created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                )
            )
        return digest

    def test_requires_authentication(self):
        self.store("run-a", "artifact:png", b"\x89PNG fake", "image/png")
        response = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:png")
        self.assertEqual(response.status_code, 401)

    def test_serves_inline_media_with_immutable_caching(self):
        digest = self.store(
            "run-a", "artifact:png", b"\x89PNG fake", "image/png", name="outputs/clean_frame.png"
        )
        response = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:png", headers=AUTH)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"\x89PNG fake")
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["etag"], f'"{digest}"')
        self.assertEqual(response.headers["cache-control"], "private, max-age=31536000, immutable")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(
            response.headers["content-disposition"], 'inline; filename="clean_frame.png"'
        )
        self.assertEqual(response.headers["accept-ranges"], "bytes")

    def test_range_requests_serve_a_slice(self):
        self.store("run-a", "artifact:mp4", b"0123456789", "video/mp4")
        response = self.client.get(
            "/api/pipeline/runs/run-a/artifacts/artifact:mp4",
            headers={**AUTH, "Range": "bytes=2-5"},
        )
        self.assertEqual(response.status_code, 206, response.text)
        self.assertEqual(response.content, b"2345")
        self.assertEqual(response.headers["content-range"], "bytes 2-5/10")

    def test_json_and_text_declare_utf8(self):
        self.store("run-a", "artifact:json", b"{}", "application/json", name="report.json")
        response = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:json", headers=AUTH)
        self.assertEqual(response.headers["content-type"], "application/json; charset=utf-8")

    def test_unsafe_types_and_downloads_become_attachments(self):
        self.store("run-a", "artifact:html", b"<script>", "text/html", name='evil "name"/../x.html')
        self.store("run-a", "artifact:png", b"\x89PNG", "image/png", name="frame.png")
        html = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:html", headers=AUTH)
        self.assertEqual(html.status_code, 200)
        self.assertEqual(html.headers["content-type"], "application/octet-stream")
        self.assertEqual(html.headers["content-disposition"], 'attachment; filename="x.html"')
        forced = self.client.get(
            "/api/pipeline/runs/run-a/artifacts/artifact:png?download=1", headers=AUTH
        )
        self.assertEqual(forced.headers["content-type"], "application/octet-stream")
        self.assertEqual(forced.headers["content-disposition"], 'attachment; filename="frame.png"')

    def test_unknown_or_foreign_artifacts_are_not_found(self):
        self.store("run-b", "artifact:other", b"bytes", "image/png")
        missing = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:nope", headers=AUTH)
        self.assertEqual(missing.status_code, 404)
        foreign = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:other", headers=AUTH)
        self.assertEqual(foreign.status_code, 404)
        owner = self.client.get("/api/pipeline/runs/run-b/artifacts/artifact:other", headers=AUTH)
        self.assertEqual(owner.status_code, 200)

    def test_missing_bytes_are_reported_not_guessed(self):
        digest = self.store("run-a", "artifact:gone", b"soon gone", "image/png")
        (self.root / "blobs" / digest[:2] / digest).unlink()
        response = self.client.get("/api/pipeline/runs/run-a/artifacts/artifact:gone", headers=AUTH)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "artifact bytes are unavailable")

    def test_summary_names_artifacts_and_times_attempts(self):
        started = datetime(2026, 1, 2, 3, 0, tzinfo=timezone.utc)
        with self.sessions.begin() as session:
            session.add(
                AttemptRecord(
                    id="run-a:1:1",
                    run_id="run-a",
                    node_id="admission",
                    number=1,
                    schema_version="wander.stage-attempt/1",
                    status="succeeded",
                    hypothesis="first pass",
                    code_revision="1234567",
                    environment_fingerprint="f" * 64,
                    started_at=started,
                    finished_at=datetime(2026, 1, 2, 3, 5, tzinfo=timezone.utc),
                )
            )
        self.store(
            "run-a",
            "artifact:named",
            b"{}",
            "application/json",
            name="outputs/shots.json",
            attempt_id="run-a:1:1",
        )
        summary = self.client.get("/api/pipeline/runs/run-a", headers=AUTH).json()
        artifact = next(item for item in summary["artifacts"] if item["id"] == "artifact:named")
        self.assertEqual(artifact["name"], "shots.json")
        # SQLite drops the offset; PostgreSQL keeps it. Either way the instant is the same.
        self.assertTrue(artifact["createdAt"].startswith("2026-01-02T00:00:00"))
        attempt = summary["attempts"][0]
        self.assertEqual(attempt["hypothesis"], "first pass")
        self.assertTrue(attempt["startedAt"].startswith("2026-01-02T03:00:00"))
        self.assertTrue(attempt["finishedAt"].startswith("2026-01-02T03:05:00"))
        self.assertIsNone(attempt["retryOf"])


class ReviewSidebarTests(unittest.TestCase):
    """Operator messages and the agent transcript tail that feed the admin sidebar."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        self.repository = PipelineRepository(sessionmaker(self.engine, expire_on_commit=False))
        self.root = Path(tempfile.mkdtemp(prefix="wander-workspace-"))
        app = create_app(
            self.repository,
            Runs(self.root / "runs"),
            ApiSettings(bearer_token=TOKEN, code_revision="1234567"),
            review_workspace_root=self.root / "runs",
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        response = self.client.post(
            "/api/pipeline/runs",
            headers=AUTH,
            json={
                "id": "run-1",
                "source_sha256": "a" * 64,
                "source_artifact_id": "artifact:source",
                "options": {"marble": "none", "objects": False},
            },
        )
        self.assertEqual(response.status_code, 201, response.text)

    def transcript(self, run_id="run-1", name=None):
        """The one transcript a run has, in the directory that run.json points at."""
        import json as _json

        run_dir = self.root / "runs" / (name or run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(_json.dumps({"runId": run_id, "name": name or run_id}))
        return run_dir / "transcript.jsonl"

    def test_messages_are_listed_in_order(self):
        for text in ("first", "second"):
            response = self.client.post(
                "/api/pipeline/runs/run-1/messages",
                headers=AUTH,
                json={"message": text, "node_id": "clean"},
            )
            self.assertEqual(response.status_code, 202)
        listing = self.client.get("/api/pipeline/runs/run-1/messages", headers=AUTH)
        self.assertEqual(listing.status_code, 200)
        messages = listing.json()["messages"]
        self.assertEqual([item["message"] for item in messages], ["first", "second"])
        self.assertEqual(messages[0]["author"], "operator")
        self.assertEqual(messages[0]["nodeId"], "clean")
        self.assertTrue(messages[0]["createdAt"])

    def test_transcripts_are_listed_and_tailed_by_offset(self):
        path = self.transcript()
        path.write_text('{"type":"thread.started","thread_id":"t"}\n{"type":"turn.started"}\n')
        listing = self.client.get("/api/pipeline/runs/run-1/reviews", headers=AUTH).json()
        self.assertTrue(listing["available"])
        self.assertEqual(
            [
                (item["nodeId"], item["attemptId"], item["finished"])
                for item in listing["transcripts"]
            ],
            [("agent", "run-1", False)],
        )
        first = self.client.get(
            "/api/pipeline/runs/run-1/reviews/agent/run-1/transcript", headers=AUTH
        ).json()
        self.assertEqual(
            [event["type"] for event in first["events"]], ["thread.started", "turn.started"]
        )
        self.assertEqual(first["offset"], path.stat().st_size)
        self.assertFalse(first["finished"])
        # A partial line is held back until the newline arrives.
        with path.open("a") as stream:
            stream.write('{"type":"item.completed","item":{"type":"agent_message","text":"hi"}')
        held = self.client.get(
            f"/api/pipeline/runs/run-1/reviews/agent/run-1/transcript?after={first['offset']}",
            headers=AUTH,
        ).json()
        self.assertEqual(held["events"], [])
        self.assertEqual(held["offset"], first["offset"])
        with path.open("a") as stream:
            stream.write("}\nnot json\n")
        (path.parent / "decision.json").write_text("{}")
        rest = self.client.get(
            f"/api/pipeline/runs/run-1/reviews/agent/run-1/transcript?after={first['offset']}",
            headers=AUTH,
        ).json()
        self.assertEqual(rest["events"][0]["item"]["text"], "hi")
        self.assertEqual(rest["events"][1], {"type": "raw", "text": "not json"})
        self.assertTrue(rest["finished"])
        self.assertEqual(rest["offset"], path.stat().st_size)
        self.assertFalse(rest["reset"])
        # A re-review truncates the file; a stale offset must restart from the top.
        path.write_text('{"type":"thread.started","thread_id":"again"}\n')
        again = self.client.get(
            f"/api/pipeline/runs/run-1/reviews/agent/run-1/transcript?after={rest['offset']}",
            headers=AUTH,
        ).json()
        self.assertTrue(again["reset"])
        self.assertEqual(again["events"][0]["thread_id"], "again")

    def test_recent_transcripts_span_runs_newest_first(self):
        older = self.transcript("run-1")
        older.write_text("{}\n")
        other = self.transcript("run-2")
        other.write_text("{}\n")
        import os

        os.utime(older, (1_700_000_000, 1_700_000_000))
        listing = self.client.get("/api/pipeline/reviews", headers=AUTH).json()
        self.assertEqual(
            [(item["runId"], item["nodeId"]) for item in listing["transcripts"]],
            [("run-2", "agent"), ("run-1", "agent")],
        )
        limited = self.client.get("/api/pipeline/reviews?limit=1", headers=AUTH).json()
        self.assertEqual(len(limited["transcripts"]), 1)

    def test_a_transcript_can_only_be_a_run_that_exists(self):
        """The run id is the lookup now, so it is the thing that has to be safe.

        The node and attempt in the path are vestigial -- there is one session per run -- and
        are ignored rather than joined onto a filesystem path, which is why a dotted segment
        there cannot reach anything.
        """
        self.transcript().write_text("{}\n")
        for node in ("../..", "..", "clean/../../etc"):
            response = self.client.get(
                f"/api/pipeline/runs/run-1/reviews/{node}/anything/transcript", headers=AUTH
            )
            # Starlette collapses dotted segments onto other routes (405) or rejects them;
            # what must never happen is reading a file outside the runs root.
            self.assertIn(response.status_code, {200, 404, 405, 422}, node)
        for run_id in ("run-nope", "../../etc", "."):
            missing = self.client.get(
                f"/api/pipeline/runs/{run_id}/reviews/agent/x/transcript", headers=AUTH
            )
            self.assertIn(missing.status_code, {404, 405, 422}, run_id)
        unauthenticated = self.client.get("/api/pipeline/runs/run-1/reviews")
        self.assertEqual(unauthenticated.status_code, 401)


if __name__ == "__main__":
    unittest.main()
