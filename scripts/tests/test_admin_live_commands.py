"""Operator controls against the production filesystem API, without a worker."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.api.app import ApiSettings
from orchestrator.api.service import create_service
from orchestrator.database import Base
from orchestrator.journal import Journal
from orchestrator.repository import PipelineRepository
from orchestrator.supervisor import runs_needing_work


@pytest.fixture
def service(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pipeline.db'}")
    Base.metadata.create_all(engine)
    repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
    settings = ApiSettings(
        bearer_token="local-test-token-at-least-24-characters", code_revision="1234567"
    )

    def client():
        return TestClient(
            create_service(
                repository=repository,
                runs_root=tmp_path / "runs",
                artifact_root=tmp_path / "artifacts",
                settings=settings,
            ),
            headers={"Authorization": f"Bearer {settings.bearer_token}"},
        )

    with client() as api:
        response = api.post(
            "/api/pipeline/runs/upload",
            data={"requested_id": "operator-test", "options": '{"marble":"none","objects":false}'},
            files={"source": ("clip.mp4", b"local-source-video", "video/mp4")},
        )
        assert response.status_code == 201, response.text
        yield api, client, tmp_path / "runs"
    engine.dispose()


def command(api, action, run_id="operator-test"):
    return api.post(
        f"/api/pipeline/runs/{run_id}/{action}",
        json={"node_id": "clean", "rationale": f"Operator requested {action}."},
    )


def test_queued_source_and_plan_survive_service_restart(service):
    api, restart, root = service
    before = api.get("/api/pipeline/runs/operator-test").json()
    assert before["status"] == "queued"
    manifest = json.loads((root / "clip" / "run.json").read_text())
    assert Path(manifest["source"]).read_bytes() == b"local-source-video"
    with restart() as recovered:
        after = recovered.get("/api/pipeline/runs/operator-test").json()
        assert after["status"] == "queued"
        assert after["nodes"] == before["nodes"]
        preview = recovered.get("/api/pipeline/runs/operator-test/progress").json()["previews"][0]
        assert recovered.get(preview["url"]).content == b"local-source-video"
    assert runs_needing_work(root) == [root / "clip"]


def test_pause_is_visible_and_resume_removes_persisted_marker(service):
    api, restart, root = service
    assert command(api, "pause").status_code == 202
    assert not runs_needing_work(root)
    with restart() as recovered:
        summary = recovered.get("/api/pipeline/runs/operator-test").json()
        assert summary["live"]["paused"] is True
        assert summary["status"] == "paused"
        assert recovered.get("/api/pipeline").json()["runs"][0]["status"] == "paused"
        assert command(recovered, "resume").status_code == 202
        assert recovered.get("/api/pipeline/runs/operator-test").json()["live"]["paused"] is False
    assert runs_needing_work(root) == [root / "clip"]


def test_cancel_is_terminal_in_journal_database_and_restarted_api(service):
    api, restart, root = service
    assert command(api, "pause").status_code == 202
    assert command(api, "cancel").status_code == 202
    assert Journal(root / "clip").finished() == "canceled"
    assert not runs_needing_work(root)
    with restart() as recovered:
        summary = recovered.get("/api/pipeline/runs/operator-test").json()
        assert summary["status"] == summary["recordedStatus"] == "canceled"
        assert summary["live"]["paused"] is False
        assert (
            recovered.get("/api/pipeline/runs/operator-test/progress").json()["finished"]
            == "canceled"
        )


@pytest.mark.parametrize("action", ["pause", "resume", "cancel", "retry", "reject"])
def test_unknown_run_commands_return_not_found(service, action):
    api, _, root = service
    response = command(api, action, "unknown")
    assert response.status_code == 404, response.text
    assert len(list(root.glob("*/run.json"))) == 1


@pytest.mark.parametrize("action", ["pause", "cancel"])
def test_stale_supervisor_selection_cannot_reopen_operator_stopped_run(
    service, monkeypatch, action
):
    from unittest.mock import Mock
    import orchestrator.supervisor as supervisor

    api, _, root = service
    selected = supervisor.runs_needing_work(root)
    assert selected == [root / "clip"]
    assert command(api, action).status_code == 202
    repository = Mock()
    session = Mock()
    monkeypatch.setattr(supervisor, "RunSession", session)
    result = supervisor.work_on(selected[0], repository, Mock(), turns=1)
    assert "operator" in result
    repository.reopen_run.assert_not_called()
    session.assert_not_called()
    summary = api.get("/api/pipeline/runs/operator-test").json()
    assert summary["status"] == {"pause": "paused", "cancel": "canceled"}[action]
