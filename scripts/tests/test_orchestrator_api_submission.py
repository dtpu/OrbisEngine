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

    def test_hands_off_upload_is_persisted_and_immediately_discoverable(self):
        response = self.upload(options=json.dumps({"marble": "none", "hands_off": True}))
        self.assertEqual(response.status_code, 201, response.text)
        manifest = self.manifests()[0]
        self.assertIs(json.loads(manifest.read_text())["options"]["hands_off"], True)
        self.assertIn(manifest.parent, runs_needing_work(self.root / "runs"))
        summary = self.repository.run_summary("run-upload")
        self.assertIs(summary["configuration"]["options"]["hands_off"], True)
        self.assertFalse((manifest.parent / "paused").exists())

    def test_clean_candidates_are_previewable_without_exposing_other_candidate_files(self):
        response = self.upload("clean-recovery")
        self.assertEqual(response.status_code, 201, response.text)
        run = self.manifests()[0].parent
        candidate = run / "clean-candidates" / "tracked-masks"
        candidate.mkdir(parents=True)
        (candidate / "clean.mp4").write_bytes(VIDEO)
        (candidate / "clean.log").write_text("private provider diagnostics")
        outside = self.root / "outside-video.mp4"
        outside.write_bytes(VIDEO)
        escaped = run / "clean-candidates" / "escape"
        escaped.mkdir()
        (escaped / "clean.mp4").symlink_to(outside)

        from orchestrator.api.progress import previews

        media = previews(run)
        preview = next(p for p in media if p["path"].startswith("clean-candidates/"))
        self.assertEqual(preview["label"], "Cleaning candidate: tracked-masks")
        self.assertEqual(preview["path"], "clean-candidates/tracked-masks/clean.mp4")
        self.assertEqual(len([p for p in media if p["path"].startswith("clean-candidates/")]), 1)
        video = self.client.get(
            "/api/pipeline/runs/clean-recovery/files/clean-candidates/tracked-masks/clean.mp4",
            headers=AUTH,
        )
        self.assertEqual(video.status_code, 200, video.text)
        self.assertEqual(video.content, VIDEO)
        for path in ("tracked-masks/clean.log", "escape/clean.mp4"):
            denied = self.client.get(
                f"/api/pipeline/runs/clean-recovery/files/clean-candidates/{path}", headers=AUTH
            )
            self.assertEqual(denied.status_code, 404, denied.text)

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

    def test_summary_uses_dynamic_directory_plan_without_inventing_attempts_or_reviews(self):
        response = self.upload(
            run_id="two-person-run",
            options='{"marble":"video","people":2,"all_people":true,"objects":true,"finetune":false}',
        )
        self.assertEqual(response.status_code, 201, response.text)
        run_dir = self.app.state.runs._dir("two-person-run")
        seeded = self.repository.run_summary("two-person-run")
        (run_dir / "tracks").mkdir()
        (run_dir / "tracks" / "tracks.json").write_text(json.dumps({"trackCount": 2}))
        retained = {
            name: {
                "status": "ok",
                "execution": "retained-input-replay",
                "freshInference": False,
                "sourceRun": "saved-input-run",
            }
            for name in ("pi3x", "tracks", "lhm_motion_00", "lhm_motion_01", "frame_align")
        }
        retained["frame_align"]["status"] = "skipped"
        (run_dir / "state.json").write_text(json.dumps({"stages": retained}))
        journal = self.app.state.runs.journal("two-person-run")
        journal.append(
            "step.started",
            step="package_people",
            attempt="two-person-run:package_people:1",
            number=1,
        )
        journal.append(
            "step.finished",
            step="package_people",
            attempt="two-person-run:package_people:1",
            status="succeeded",
        )
        # An attempted stage outside this run's plan is still real history.
        journal.append(
            "step.started", step="clean_first", attempt="two-person-run:clean_first:1", number=1
        )
        journal.append(
            "step.finished",
            step="clean_first",
            attempt="two-person-run:clean_first:1",
            status="failed",
        )

        progress = self.client.get(
            "/api/pipeline/runs/two-person-run/progress", headers=AUTH
        ).json()
        detail = self.client.get("/api/pipeline/runs/two-person-run", headers=AUTH).json()
        listed = next(
            run
            for run in self.client.get("/api/pipeline", headers=AUTH).json()["runs"]
            if run["id"] == "two-person-run"
        )
        stages = {stage["name"]: stage for stage in progress["stages"]}
        self.assertEqual(len(seeded["nodes"]), 16)
        self.assertEqual(len(stages), 19)
        self.assertIn("person_prep_01", stages)
        for summary in (detail, listed):
            nodes = {node["id"]: node for node in summary["nodes"]}
            self.assertEqual(set(stages) | {"clean_first"}, set(nodes))
            self.assertEqual(
                {name: nodes[name]["status"] for name in stages},
                {name: stage["status"] for name, stage in stages.items()},
            )
            self.assertEqual(nodes["person_prep_01"]["dependencies"], ["tracks"])
            self.assertEqual(stages["person_prep_01"]["dependencies"], [])
            self.assertEqual(nodes["package_people"]["selectedAttemptId"], None)
            self.assertEqual(len(summary["attempts"]), 2)
            self.assertEqual(
                {attempt["nodeId"] for attempt in summary["attempts"]},
                {"package_people", "clean_first"},
            )
            self.assertEqual(summary["reviews"], [])
            self.assertEqual(nodes["clean_first"]["status"], "failed")
            self.assertEqual(nodes["clean_first"]["definition"]["executor"], "wander")
            self.assertEqual(nodes["person_prep_01"]["definition"]["kind"], "compute")
            self.assertEqual(nodes["review"]["definition"]["kind"], "human")
            self.assertIsNone(nodes["review"]["definition"]["executor"])
            self.assertNotIn("freshInference", nodes["package_people"])
            for name in retained:
                self.assertEqual(nodes[name]["execution"], "retained-input-replay")
                self.assertIs(nodes[name]["freshInference"], False)
                self.assertEqual(nodes[name]["sourceRun"], "saved-input-run")
                self.assertIs(stages[name]["freshInference"], False)
        # State.record merges keys. A subsequent real attempt must not inherit a stale
        # retained-input badge from its earlier state record.
        journal.append("step.started", step="pi3x", attempt="two-person-run:pi3x:1", number=1)
        active = self.client.get("/api/pipeline/runs/two-person-run", headers=AUTH).json()
        active_node = next(node for node in active["nodes"] if node["id"] == "pi3x")
        self.assertEqual(active_node["status"], "running")
        self.assertNotIn("execution", active_node)
        self.assertNotIn("freshInference", active_node)
        active_stage = next(
            stage
            for stage in self.client.get(
                "/api/pipeline/runs/two-person-run/progress", headers=AUTH
            ).json()["stages"]
            if stage["name"] == "pi3x"
        )
        self.assertNotIn("execution", active_stage)
        self.repository.set_node_status("two-person-run", "verify", "waiting_agent")
        waiting_agent = self.client.get("/api/pipeline/runs/two-person-run", headers=AUTH).json()
        self.assertEqual(
            next(node for node in waiting_agent["nodes"] if node["id"] == "verify")["status"],
            "waiting_agent",
        )
        self.repository.set_node_status(
            "two-person-run", "package_people", "waiting_human", "operator review"
        )
        waiting = self.client.get("/api/pipeline/runs/two-person-run", headers=AUTH).json()
        self.assertEqual(waiting["status"], "waiting_human")
        self.assertEqual(
            next(node for node in waiting["nodes"] if node["id"] == "package_people")["status"],
            "waiting_human",
        )
        self.repository.finish_run("two-person-run", "canceled")
        terminal = self.client.get("/api/pipeline/runs/two-person-run", headers=AUTH).json()
        self.assertEqual(terminal["status"], "canceled")

    def test_summary_falls_back_to_database_when_run_manifest_is_missing(self):
        self.assertEqual(self.upload(run_id="missing-directory").status_code, 201)
        before = self.repository.run_summary("missing-directory")
        run_dir = self.app.state.runs._dir("missing-directory")
        (run_dir / "run.json").unlink()
        detail = self.client.get("/api/pipeline/runs/missing-directory", headers=AUTH).json()
        listed = next(
            run
            for run in self.client.get("/api/pipeline", headers=AUTH).json()["runs"]
            if run["id"] == "missing-directory"
        )
        for summary in (detail, listed):
            self.assertEqual(summary["nodes"], before["nodes"])
            self.assertEqual(summary["attempts"], before["attempts"])
            self.assertEqual(summary["status"], before["status"])

    def test_corrupt_run_cannot_break_listing_or_another_runs_progress(self):
        self.assertEqual(self.upload(run_id="healthy").status_code, 201)
        self.assertEqual(self.upload(run_id="corrupt").status_code, 201)
        corrupt = self.app.state.runs._dir("corrupt")
        manifest = corrupt / "run.json"
        original_manifest = json.loads(manifest.read_text())
        original_nodes = self.repository.run_summary("corrupt")["nodes"]

        for kind in ("state", "tracks", "manifest"):
            with self.subTest(kind=kind):
                manifest.write_text(json.dumps(original_manifest))
                (corrupt / "state.json").write_text(json.dumps({"stages": {}}))
                (corrupt / "tracks").mkdir(exist_ok=True)
                (corrupt / "tracks" / "tracks.json").write_text('{"trackCount":1}')
                if kind == "state":
                    (corrupt / "state.json").write_text('{"stages":["not a stage map"]}')
                elif kind == "tracks":
                    (corrupt / "tracks" / "tracks.json").write_text('{"trackCount":"many"}')
                else:
                    manifest.write_text(json.dumps({**original_manifest, "source": 3}))

                response = self.client.get("/api/pipeline", headers=AUTH)
                self.assertEqual(response.status_code, 200, response.text)
                listed = {run["id"]: run for run in response.json()["runs"]}
                self.assertEqual(listed["corrupt"]["nodes"], original_nodes)
                self.assertTrue(listed["healthy"]["nodes"])
                self.assertEqual(
                    self.client.get(
                        "/api/pipeline/runs/healthy/progress", headers=AUTH
                    ).status_code,
                    200,
                )
                unavailable = self.client.get("/api/pipeline/runs/corrupt/progress", headers=AUTH)
                self.assertEqual(unavailable.status_code, 503, unavailable.text)
                self.assertEqual(unavailable.json()["detail"], "run progress unavailable")

    def test_answer_releases_only_the_question_hold_in_list_detail_and_progress(self):
        for result in ("pending", "succeeded", "failed", "waiting_human"):
            with self.subTest(result=result):
                run_id = f"question-{result}"
                self.assertEqual(self.upload(run_id=run_id).status_code, 201)
                journal = self.app.state.runs.journal(run_id)
                if result in {"succeeded", "failed"}:
                    journal.append("step.started", step="clean", attempt=f"{run_id}:1", number=1)
                    journal.append(
                        "step.finished",
                        step="clean",
                        attempt=f"{run_id}:1",
                        status=result,
                        error="stage failed" if result == "failed" else None,
                    )
                elif result == "waiting_human":
                    self.repository.set_node_status(run_id, "clean", result, "quality approval")
                before = self.repository.run_summary(run_id)
                journal.append("question", question="Use this source clip?", step="clean")
                held = self.client.get(f"/api/pipeline/runs/{run_id}", headers=AUTH).json()
                self.assertEqual(held["status"], "waiting_human")
                self.assertEqual(
                    next(n for n in held["nodes"] if n["id"] == "clean")["status"],
                    "waiting_human",
                )
                progress = self.client.get(
                    f"/api/pipeline/runs/{run_id}/progress", headers=AUTH
                ).json()
                self.assertEqual(
                    next(s for s in progress["stages"] if s["name"] == "clean")["status"],
                    "waiting_human",
                )
                # An unscoped answer still answers the latest question. Each API call uses
                # a new Journal/Projection, so the handoff cannot rely on process memory.
                answered = self.client.post(
                    f"/api/pipeline/runs/{run_id}/messages",
                    headers=AUTH,
                    json={"message": "Use the retained source; no new inference."},
                )
                self.assertEqual(answered.status_code, 202, answered.text)
                detail = self.client.get(f"/api/pipeline/runs/{run_id}", headers=AUTH).json()
                listed = next(
                    r
                    for r in self.client.get("/api/pipeline", headers=AUTH).json()["runs"]
                    if r["id"] == run_id
                )
                for summary in (detail, listed):
                    node = next(n for n in summary["nodes"] if n["id"] == "clean")
                    self.assertEqual(node["status"], result)
                    self.assertEqual(summary["attempts"], before["attempts"])
                    self.assertEqual(summary["reviews"], before["reviews"])
                    if result == "waiting_human":
                        self.assertEqual(node["blockedReason"], "quality approval")
                    else:
                        self.assertNotEqual(summary["status"], "waiting_human")
                self.assertIsNone(self.app.state.runs.journal(run_id).unanswered())

    def test_question_answer_preserves_pause_cancel_and_other_stage_review(self):
        self.assertEqual(self.upload(run_id="held").status_code, 201)
        journal = self.app.state.runs.journal("held")
        self.repository.set_node_status("held", "pi3x", "waiting_human", "independent review")
        for control in ("paused", "canceled"):
            if control == "paused":
                self.app.state.runs.pause("held", True)
            else:
                journal.append("run.finished", status="canceled")
            journal.append("question", question="Source question", step="clean")
            response = self.client.post(
                "/api/pipeline/runs/held/messages",
                headers=AUTH,
                json={"message": "Answer only", "node_id": "clean"},
            )
            self.assertEqual(response.status_code, 202)
            summary = self.client.get("/api/pipeline/runs/held", headers=AUTH).json()
            self.assertEqual(summary["status"], control)
            review = next(n for n in summary["nodes"] if n["id"] == "pi3x")
            self.assertEqual(
                (review["status"], review["blockedReason"]), ("waiting_human", "independent review")
            )

    def test_question_on_unplanned_stage_stays_visible_until_answered(self):
        self.assertEqual(self.upload(run_id="unplanned-question").status_code, 201)
        journal = self.app.state.runs.journal("unplanned-question")
        journal.append("question", question="Include a world?", step="marble_video")
        detail = self.client.get("/api/pipeline/runs/unplanned-question", headers=AUTH).json()
        listed = next(
            run
            for run in self.client.get("/api/pipeline", headers=AUTH).json()["runs"]
            if run["id"] == "unplanned-question"
        )
        for summary in (detail, listed):
            node = next(n for n in summary["nodes"] if n["id"] == "marble_video")
            self.assertEqual(
                (node["status"], node["blockedReason"]), ("waiting_human", "Include a world?")
            )
            self.assertFalse(summary["attempts"])
        self.assertEqual(
            self.client.post(
                "/api/pipeline/runs/unplanned-question/messages",
                headers=AUTH,
                json={"message": "Keep the original plan."},
            ).status_code,
            202,
        )
        answered = self.client.get("/api/pipeline/runs/unplanned-question", headers=AUTH).json()
        self.assertNotIn("marble_video", [node["id"] for node in answered["nodes"]])
        self.assertNotEqual(answered["status"], "waiting_human")

    def test_legacy_question_projection_restores_history_without_clearing_independent_holds(self):
        from orchestrator.journal import Entry, Journal, RunProjection
        from datetime import UTC, datetime

        for prior, later in (
            ("queued", None),
            ("waiting_human", None),
            ("queued", "quality"),
            ("queued", "same-text"),
        ):
            with self.subTest(prior=prior, later=later):
                run_id = f"legacy-{prior}-{later}"
                self.assertEqual(self.upload(run_id=run_id).status_code, 201)
                root = self.app.state.runs._dir(run_id)
                journal = Journal(root)
                if prior == "waiting_human":
                    self.repository.set_node_status(run_id, "clean", prior, "independent review")
                entry = Entry(
                    "question",
                    datetime.now(UTC).isoformat(),
                    {"step": "clean", "question": "Source question"},
                )
                journal.path.write_text(
                    json.dumps({"kind": entry.kind, "at": entry.at, "data": entry.data}) + "\n"
                )
                # Reproduce the exact pre-fix projection, including its event/message pair.
                self.repository.set_node_status(run_id, "clean", "waiting_human", "Source question")
                self.repository.add_message(
                    run_id=run_id, author="agent", message="Source question", node_id="clean"
                )
                # Recovery replays and subsequent questions must not invalidate the old
                # status event's ownership by touching the underlying node timestamp.
                RunProjection(self.repository, run_id).replay(journal)
                if later:
                    self.repository.set_node_status(
                        run_id,
                        "clean",
                        "waiting_human",
                        "Source question" if later == "same-text" else "new quality approval",
                    )
                self.assertEqual(
                    self.client.post(
                        f"/api/pipeline/runs/{run_id}/messages",
                        headers=AUTH,
                        json={"message": "Continue", "node_id": "clean"},
                    ).status_code,
                    202,
                )
                summary = self.client.get(f"/api/pipeline/runs/{run_id}", headers=AUTH).json()
                node = next(n for n in summary["nodes"] if n["id"] == "clean")
                if later or prior == "waiting_human":
                    expected = (
                        "Source question"
                        if later == "same-text"
                        else "new quality approval"
                        if later
                        else "independent review"
                    )
                    self.assertEqual(
                        (node["status"], node["blockedReason"]), ("waiting_human", expected)
                    )
                else:
                    self.assertEqual(node["status"], "pending")
                    self.assertNotEqual(summary["status"], "waiting_human")
                # Compatibility is a read-only response repair, not a history rewrite.
                raw = self.repository.run_summary(run_id)
                self.assertEqual(
                    next(n for n in raw["nodes"] if n["id"] == "clean")["status"], "waiting_human"
                )
                if later is None and prior == "queued":
                    self.app.state.runs.journal(run_id).append(
                        "question", question="One more question", step="clean"
                    )
                    self.assertEqual(
                        self.client.post(
                            f"/api/pipeline/runs/{run_id}/messages",
                            headers=AUTH,
                            json={"message": "Answered again"},
                        ).status_code,
                        202,
                    )
                    after = self.client.get(f"/api/pipeline/runs/{run_id}", headers=AUTH).json()
                    self.assertEqual(
                        next(n for n in after["nodes"] if n["id"] == "clean")["status"], "pending"
                    )

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
