"""Operator controls stop new local work without canceling submitted provider operations."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import cli
from orchestrator.agent.harness import HarnessAgent, HarnessPolicy
from orchestrator.contracts import Run
from orchestrator.database import Base
from orchestrator.journal import Journal, RunProjection
from orchestrator.repository import PipelineRepository
from orchestrator.session import RunSession
from orchestrator.steps import STEPS


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture source; never sent to a provider")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    engine = create_engine(f"sqlite:///{tmp_path / 'run.db'}")
    Base.metadata.create_all(engine)
    repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
    repository.create_run_row(
        Run(
            id="run-1",
            graph_version="wander.agent-run/1",
            code_revision="test-revision",
            source_sha256=digest,
            source_artifact_id="artifact:source",
            created_by="tests",
        )
    )
    (tmp_path / "run.json").write_text(
        json.dumps({"runId": "run-1", "name": "clip", "options": {"marble": "none"}})
    )
    context = cli.RunContext("run-1", tmp_path, "clip", source, ROOT, {"marble": "none"}, digest)
    journal = Journal(tmp_path, projection=RunProjection(repository, "run-1", catalogue=STEPS))
    monkeypatch.setattr(cli.RunContext, "journal", lambda self: journal)
    monkeypatch.delenv("WANDER_STEP_CHILD", raising=False)
    result = SimpleNamespace(context=context, journal=journal, repository=repository)
    yield result
    engine.dispose()


def step_args(wait=False, *, attempt=None):
    tokens = ["step"]
    if wait:
        tokens.append("--wait")
    if attempt:
        tokens.extend(["--attempt", attempt, "--number", "1"])
    return cli.build_parser().parse_args([*tokens, "pi3x"])


def control(run, status):
    if status == "paused":
        (run.context.run_dir / "paused").write_text("paused by an operator\n")
    else:
        run.journal.append("run.finished", status="canceled", summary="operator canceled")


def stage_process():
    return SimpleNamespace(stdout=[], wait=lambda: 0)


def session(run):
    return RunSession(
        "run-1",
        run.context.run_dir,
        HarnessAgent(HarnessPolicy(kind="command", command=(sys.executable, "-c", "pass"))),
        repository=run.repository,
    )


@pytest.mark.parametrize("status", ["paused", "canceled"])
@pytest.mark.parametrize("wait", [False, True])
def test_operator_control_rejects_stage_before_preflight_or_subprocess(
    run, monkeypatch, status, wait
):
    control(run, status)
    entries = run.journal.entries()
    popen = Mock(side_effect=AssertionError("must not spawn"))
    preflight = Mock(side_effect=AssertionError("must not preflight a controlled run"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "source_problem", preflight)
    assert cli.command_step(run.context, step_args(wait)) == 2
    popen.assert_not_called()
    preflight.assert_not_called()
    assert run.journal.entries() == entries
    if status == "canceled":
        assert run.journal.finished() == "canceled"
        assert run.repository.run_summary("run-1")["status"] == "canceled"


@pytest.mark.parametrize("status", ["paused", "canceled"])
def test_detached_child_closes_pending_attempt_without_launch_or_receipt_change(
    run, monkeypatch, status
):
    popen = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.command_step(run.context, step_args()) == 0
    child = cli.build_parser().parse_args(popen.call_args.args[0][3:])
    assert child.attempt == "run-1:pi3x:1"
    receipt = run.context.run_dir / "provider-receipt.json"
    receipt.write_text('{"operation": "retained-existing-operation"}')
    control(run, status)
    popen.reset_mock()
    assert cli.command_step(run.context, child) == 2
    popen.assert_not_called()
    attempts = run.repository.run_summary("run-1")["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["status"] == "blocked"
    assert attempts[0]["finishedAt"]
    assert receipt.read_text() == '{"operation": "retained-existing-operation"}'
    assert cli.command_step(run.context, child) == 2
    assert len([e for e in run.journal.entries() if e.kind == "step.finished"]) == 1
    if status == "canceled":
        assert run.journal.finished() == "canceled"
        assert run.repository.run_summary("run-1")["status"] == "canceled"


@pytest.mark.parametrize("wait", [False, True])
def test_cancel_arriving_after_attempt_admission_prevents_launch_and_settles_attempt(
    run, monkeypatch, wait
):
    append = run.journal.append

    def cancel_after_start(kind, **data):
        entry = append(kind, **data)
        if kind == "step.started":
            append("run.finished", status="canceled", summary="canceled before subprocess")
        return entry

    monkeypatch.setattr(run.journal, "append", cancel_after_start)
    popen = Mock(side_effect=AssertionError("must not spawn after cancellation"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.command_step(run.context, step_args(wait)) == 2
    popen.assert_not_called()
    assert run.journal.last_status("pi3x") == "blocked"
    assert run.journal.finished() == "canceled"
    assert run.repository.run_summary("run-1")["status"] == "canceled"


@pytest.mark.parametrize("source_error", [None, "source changed during preflight"])
def test_cancel_arriving_during_preflight_does_not_record_a_new_start(
    run, monkeypatch, source_error
):
    def cancel_during_source_check(context):
        control(run, "canceled")
        return source_error

    monkeypatch.setattr(cli, "source_problem", cancel_during_source_check)
    popen = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.command_step(run.context, step_args(True)) == 2
    assert [e.kind for e in run.journal.entries()] == ["run.finished"]
    popen.assert_not_called()


@pytest.mark.parametrize("status", ["paused", "canceled"])
def test_session_turn_admission_never_launches_harness(run, monkeypatch, status):
    turn = session(run)
    control(run, status)
    harness = Mock(side_effect=AssertionError("must not launch harness"))
    monkeypatch.setattr(turn.harness, "run", harness)
    outcome = turn.work(turns=3)
    harness.assert_not_called()
    assert outcome.steps_run == 0
    assert not outcome.wants_another_turn
    assert not (run.context.run_dir / "BRIEF.md").exists()
    if status == "canceled":
        assert outcome.finished == "canceled"
        assert run.repository.run_summary("run-1")["status"] == "canceled"


@pytest.mark.parametrize("status", ["paused", "canceled"])
def test_control_arriving_while_brief_is_prepared_blocks_harness(run, monkeypatch, status):
    turn = session(run)
    monkeypatch.setattr(
        "orchestrator.session.install_skill", lambda directory: control(run, status)
    )
    harness = Mock(side_effect=AssertionError("must recheck control before harness"))
    monkeypatch.setattr(turn.harness, "run", harness)
    outcome = turn.turn()
    harness.assert_not_called()
    assert not outcome.wants_another_turn
    if status == "canceled":
        assert outcome.finished == "canceled"


@pytest.mark.parametrize("status", ["paused", "canceled"])
def test_explicit_resume_allows_stage_and_session_work(run, monkeypatch, status):
    control(run, status)
    (run.context.run_dir / "paused").unlink(missing_ok=True)
    run.journal.append("run.resumed", reason="operator explicitly resumed")
    popen = Mock(return_value=stage_process())
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.command_step(run.context, step_args(True)) == 0
    popen.assert_called_once()
    turn = session(run)
    harness = Mock(
        return_value=SimpleNamespace(session=None, result=SimpleNamespace(status="completed"))
    )
    monkeypatch.setattr(turn.harness, "run", harness)
    assert turn.turn().finished is None
    harness.assert_called_once()


@pytest.mark.parametrize("status", ["failed", "blocked", "succeeded"])
def test_existing_finished_run_can_still_explicitly_retry_a_stage(run, monkeypatch, status):
    run.journal.append("run.finished", status=status, summary="previous outcome")
    (run.context.run_dir / "state.json").write_text('{"stages":{"pi3x":{"status":"failed"}}}')
    popen = Mock(return_value=stage_process())
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.command_step(run.context, step_args(True)) == 1  # retained cache still says failed
    popen.assert_called_once()
    assert len([e for e in run.journal.entries() if e.kind == "step.started"]) == 1
    assert run.journal.finished() is None


@pytest.mark.parametrize("status", ["failed", "blocked", "succeeded"])
def test_stale_agent_cannot_replace_cancellation_with_its_finish(run, monkeypatch, status):
    control(run, "canceled")
    entries = run.journal.entries()
    source_check = Mock(side_effect=AssertionError("must not validate canceled finish"))
    monkeypatch.setattr(cli, "source_problem", source_check)
    args = SimpleNamespace(status=status, summary="stale agent outcome")
    assert cli.command_finish(run.context, args) == 2
    source_check.assert_not_called()
    assert run.journal.entries() == entries
    assert run.repository.run_summary("run-1")["status"] == "canceled"


def test_cancel_remains_effective_despite_a_stale_start_or_finish_until_resume(run):
    control(run, "canceled")
    # Legacy/stale writers may already have appended these. Neither is an operator resume.
    run.journal.append("step.started", step="pi3x", attempt="stale", number=1)
    run.journal.append("run.finished", status="failed", summary="stale agent")
    assert run.journal.control_status() == "canceled"
    assert run.journal.finished() == "canceled"
    assert run.repository.run_summary("run-1")["status"] == "canceled"
    RunProjection(run.repository, "run-1", catalogue=STEPS).replay(run.journal)
    assert run.repository.run_summary("run-1")["status"] == "canceled"
    run.journal.append("run.resumed", reason="explicit operator action")
    assert run.journal.control_status() is None
    assert run.repository.run_summary("run-1")["status"] == "running"


def test_cancellation_keeps_results_from_an_already_running_provider(run, monkeypatch):
    def already_running(*args, **kwargs):
        control(run, "canceled")
        return stage_process()

    popen = Mock(side_effect=already_running)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.command_step(run.context, step_args(True)) == 0
    popen.assert_called_once()
    assert run.journal.last_status("pi3x") == "succeeded"
    assert run.journal.finished() == "canceled"
    summary = run.repository.run_summary("run-1")
    assert summary["status"] == "canceled"
    assert summary["attempts"][0]["status"] == "succeeded"
