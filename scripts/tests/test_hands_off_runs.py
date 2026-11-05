"""Hands-off admission and delivery use local fixtures, never a provider or real agent."""

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import cli
from orchestrator.agent.contracts import AgentResult
from orchestrator.agent.harness import HarnessAgent, HarnessOutcome, HarnessPolicy
from orchestrator.journal import Journal
from orchestrator.run_policy import (
    admission_problem,
    delivery_metadata,
    unpaid_prerequisite_problem,
)
from orchestrator.session import RunSession
from orchestrator.steps import RunOptions
from orchestrator.supervisor import runs_needing_work, work_on


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "example"
    root.mkdir(parents=True)
    source = root / "source.mp4"
    source.write_bytes(b"source fixture")
    (root / "run.json").write_text(
        json.dumps(
            {
                "runId": "example",
                "name": "example",
                "source": str(source),
                "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "repository": str(Path(__file__).resolve().parents[2]),
                "options": {"hands_off": True, "marble": "none", "objects": False},
            }
        )
    )
    monkeypatch.setenv("WANDER_RUN_DIR", str(root))
    monkeypatch.delenv("WANDER_RUN_ID", raising=False)
    monkeypatch.delenv("WANDER_DATABASE_URL", raising=False)
    return cli.RunContext.load()


def package(context):
    world = context.run_dir / "public/worlds/example-4d"
    person = world / "person"
    person.mkdir(parents=True)
    frame = person / "frame.ply"
    frame.write_bytes(b"fixture frame")
    (person / "sequence.json").write_text(
        json.dumps(
            {
                "frames": [frame.name],
                "frame_sha256": [hashlib.sha256(frame.read_bytes()).hexdigest()],
                "sourceSha256": context.source_sha256,
            }
        )
    )
    (world / "cameras.json").write_text('{"cameras":[{}]}')
    (world / "audio.json").write_text(
        '{"schema":"wander.audio/1","source":{"hasAudio":false},"timeline":{"durationSeconds":1}}'
    )
    source = context.run_dir / "public/clips/example/source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(context.source.read_bytes())
    return world, source


def session_for(context, *, status="completed", body=None):
    session = RunSession(
        context.run_id,
        context.run_dir,
        HarnessAgent(HarnessPolicy(kind="command", command=(sys.executable, "-c", "pass"))),
    )
    outcome = HarnessOutcome(
        AgentResult(
            status=status,
            transcript_path="fixture",
            error=None if status == "completed" else "session unavailable",
        )
    )
    session.harness.run = Mock(side_effect=body) if body else Mock(return_value=outcome)
    return session


def test_legacy_default_and_persisted_run_control_only_clean_human_pause(run):
    assert RunOptions().hands_off is False
    assert RunOptions.model_validate({"hands_off": True}).model_dump()["hands_off"] is True
    command = cli.run_clip_command(run, "review", [])
    assert "--no-gate" in command
    assert "--gate-pass" not in command
    run.options.pop("hands_off")
    assert "--no-gate" not in cli.run_clip_command(run, "review", [])


def test_question_becomes_durable_terminal_failure_and_no_more_stages(run, monkeypatch):
    journal = run.journal()
    journal.append("step.started", step="pi3x", attempt="already-running", number=1)
    assert cli.main(["ask", "Which source should I use?"]) == 0
    assert journal.finished() == "failed"
    assert journal.unanswered() is None
    assert not any(entry.kind == "question" for entry in journal.entries())
    popen = Mock(side_effect=AssertionError("a terminal run launched work"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "clean"]) == 2
    journal.append("step.finished", step="pi3x", attempt="already-running", status="succeeded")
    assert Journal(run.run_dir).finished() == "failed"
    assert runs_needing_work(run.run_dir.parent) == []
    assert cli.main(["finish", "blocked", "late agent response"]) == 2
    popen.assert_not_called()


@pytest.mark.parametrize("status", ["succeeded", "failed", "blocked", "canceled"])
def test_finished_runs_require_explicit_resume_even_after_late_start(run, status):
    journal = run.journal()
    journal.append("run.finished", status=status)
    journal.append("step.started", step="clean", attempt="late")
    assert journal.finished() == status
    journal.append("run.resumed", author="operator")
    assert journal.finished() is None


def test_direct_supervisor_call_does_not_reopen_finished_database_row(run):
    run.journal().append("run.finished", status="failed", summary="Retained failure")
    repository = Mock()
    policy = HarnessPolicy(kind="command", command=(sys.executable, "-c", "pass"))
    outcome = work_on(run.run_dir, repository, policy, turns=1)
    assert outcome.message == "example: finished failed"
    repository.reopen_run.assert_not_called()


@pytest.mark.parametrize("flag", ["--force", "--for=clean", "--only=marble_image", "--model"])
def test_runner_overrides_never_reach_a_hands_off_subprocess(run, monkeypatch, flag):
    popen = Mock(side_effect=AssertionError("unsafe flag reached a subprocess"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "clean", flag]) == 2
    assert run.journal().finished() == "failed"
    popen.assert_not_called()


@pytest.mark.parametrize("recorded", ["journal", "state"])
def test_paid_retry_is_refused_after_restart_and_resume(run, monkeypatch, recorded):
    journal = run.journal()
    if recorded == "journal":
        journal.append("step.started", step="clean", attempt="one", number=1)
        journal.append("step.finished", step="clean", attempt="one", status="failed")
    else:
        (run.run_dir / "state.json").write_text('{"stages":{"clean":{"status":"ok"}}}')
    journal.append("run.finished", status="failed")
    journal.append("run.resumed", author="operator")
    popen = Mock(side_effect=AssertionError("paid stage was retried"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "clean"]) == 2
    assert Journal(run.run_dir).finished() == "failed"
    popen.assert_not_called()


@pytest.mark.parametrize("status", ["blocked", "failed"])
def test_lost_package_prerequisite_stops_further_paid_work(run, monkeypatch, status):
    journal = run.journal()
    journal.append("step.started", step="pi3x", attempt="one", number=1)
    journal.append("step.finished", step="pi3x", attempt="one", status=status)
    popen = Mock(
        side_effect=AssertionError("paid work launched after the package became impossible")
    )
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "lhm_frozen"]) == 2
    finished = Journal(run.run_dir).entries()[-1]
    assert finished.kind == "run.finished" and finished.data["status"] == "failed"
    assert "pi3x" in finished.data["summary"]
    popen.assert_not_called()


def test_lost_prerequisite_check_ignores_unpaid_and_unrelated_failures(run):
    journal = run.journal()
    # Unpaid steps can be rerun; clean does not feed the package of a run without a world.
    journal.append("step.finished", step="frame_align", attempt="one", status="failed")
    journal.append("step.finished", step="clean", attempt="one", status="failed")
    assert admission_problem(run, "lhm_frozen", [], journal) is None
    (run.run_dir / "state.json").write_text('{"stages":{"lhm_frozen":{"status":"failed"}}}')
    assert "lhm_frozen" in admission_problem(run, "pi3x", [], journal)


def test_failed_unpaid_prerequisite_holds_paid_work_until_it_succeeds(run, monkeypatch):
    journal = run.journal()
    journal.append("step.finished", step="person_prep", attempt="one", status="failed")
    popen = Mock(side_effect=AssertionError("paid work launched without its prerequisite"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "lhm_frozen"]) == 2
    # Held, not finished: the unpaid step may still be rerun.
    assert Journal(run.run_dir).finished() is None
    popen.assert_not_called()
    assert unpaid_prerequisite_problem(run, "person_prep", journal) is None
    journal.append("step.finished", step="person_prep", attempt="two", status="succeeded")
    assert unpaid_prerequisite_problem(run, "lhm_frozen", journal) is None


def test_existing_detached_attempt_can_claim_execution_once(run):
    journal = run.journal()
    journal.append(
        "step.started",
        step="clean",
        attempt="example:clean:1",
        number=1,
        detached=True,
        command=cli.run_clip_command(run, "clean", []),
    )
    args = cli.build_parser().parse_args(
        ["step", "--wait", "--number", "1", "--attempt", "example:clean:1", "clean"]
    )
    with journal.admission_lock():
        admitted = cli.admit_step(run, args, journal)
        assert isinstance(admitted, cli.AdmittedStep)
        assert cli.admit_step(run, args, journal) == 2


def test_ambiguous_paid_attempt_stops_without_clearing_its_receipt(run, monkeypatch):
    run.journal().append("step.started", step="clean", attempt="unknown", number=1)
    popen = Mock(side_effect=AssertionError("ambiguous paid stage was resubmitted"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "clean"]) == 2
    assert run.journal().finished() == "failed"
    assert "unknown" in run.journal().active_attempts()["clean"]
    popen.assert_not_called()


def test_unrequested_generation_mode_is_refused_before_credentials_or_launch(run, monkeypatch):
    popen = Mock(side_effect=AssertionError("unrequested world generation launched"))
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    assert cli.main(["step", "--wait", "marble_image"]) == 2
    assert run.journal().finished() == "failed"
    popen.assert_not_called()


def test_preview_delivery_preserves_blocked_acceptance_and_real_stage_failure(run):
    package(run)
    journal = run.journal()
    (run.run_dir / "state.json").write_text(
        '{"stages":{"verify":{"status":"blocked"},"_verify":{"status":"blocked","passed":false}}}'
    )
    assert cli.main(["finish", "preview_ready", "No manual review was supplied."]) == 0
    result = journal.entries()[-1].data
    assert result["status"] == "failed"
    assert result["delivery"]["status"] == "preview_ready"
    assert result["delivery"]["qualityStatus"] == "blocked"
    journal.append("run.resumed", author="operator")
    (run.run_dir / "state.json").write_text(
        '{"stages":{"scale_fit":{"status":"failed","error":"scale disagreement"}}}'
    )
    assert cli.main(["finish", "blocked", "Retained the available preview."]) == 0
    result = journal.entries()[-1].data
    assert result["delivery"]["qualityStatus"] == "failed"
    assert result["delivery"]["failedStages"] == {"scale_fit": "scale disagreement"}


@pytest.mark.parametrize("damaged", ["source", "export", "frame"])
def test_preview_requires_current_source_match_and_complete_viewer_assets(run, damaged):
    world, exported = package(run)
    path = {"source": run.source, "export": exported, "frame": world / "person/frame.ply"}[damaged]
    if damaged == "frame":
        path.unlink()
    else:
        path.write_bytes(b"changed source")
    assert delivery_metadata(run, run.journal(), "review missing")["status"] == "unavailable"
    assert cli.main(["finish", "preview_ready", "review missing"]) == 1
    assert run.journal().finished() is None


def test_success_still_requires_full_existing_completion_checks(run):
    package(run)
    assert cli.main(["finish", "succeeded", "Done"]) == 1
    (run.run_dir / "state.json").write_text(
        json.dumps({"stages": {name: {"status": "ok"} for name in cli.plan_for(run)}})
    )
    assert cli.main(["finish", "succeeded", "All requested stages passed"]) == 0
    assert run.journal().finished() == "succeeded"


def test_viewable_world_with_missing_manual_review_never_succeeds(run):
    package(run)
    described = json.loads((run.run_dir / "run.json").read_text())
    described["options"]["marble"] = "video"
    (run.run_dir / "run.json").write_text(json.dumps(described))
    run = cli.RunContext.load()
    (run.run_dir / "public/marble-example-clean.spz").write_bytes(b"world fixture")
    stages = {name: {"status": "ok"} for name in cli.plan_for(run)}
    stages["_verify"] = {"status": "blocked", "passed": False}
    (run.run_dir / "state.json").write_text(json.dumps({"stages": stages}))
    assert cli.main(["finish", "succeeded", "Readable files are not approval"]) == 1
    assert cli.main(["finish", "preview_ready", "Manual review is unavailable"]) == 0
    assert run.journal().finished() == "failed"
    assert run.journal().entries()[-1].data["delivery"]["qualityStatus"] == "blocked"
    assert (
        json.loads((run.run_dir / "state.json").read_text())["stages"]["_verify"]
        == stages["_verify"]
    )


@pytest.mark.parametrize("status", ["completed", "stalled", "failed", "timed_out"])
def test_idle_or_failed_agent_ends_hands_off_run_and_restart_does_not_spawn(run, status):
    session = session_for(run, status=status)
    outcome = session.work()
    assert outcome.finished == "failed" and outcome.waiting is None
    restarted = session_for(run)
    assert restarted.turn().finished == "failed"
    restarted.harness.run.assert_not_called()


def test_idle_turn_keeps_existing_work_in_flight_without_relaunching(run):
    run.journal().append("step.started", step="clean", attempt="running", number=1)
    session = session_for(run)
    outcome = session.work()
    assert outcome.finished is None
    assert outcome.running == 1
    session.harness.run.assert_called_once()


def test_session_converts_existing_question_without_spawning(run):
    run.journal().append("question", question="Human review required")
    assert runs_needing_work(run.run_dir.parent) == [run.run_dir]
    session = session_for(run)
    outcome = session.turn()
    assert outcome.finished == "failed" and outcome.waiting is None
    session.harness.run.assert_not_called()


@pytest.mark.parametrize("control", ["paused", "canceled"])
def test_controls_win_over_question_or_agent_completion(run, control):
    journal = run.journal()
    journal.append("question", question="Human review required")
    if control == "paused":
        (run.run_dir / "paused").touch()
    else:
        journal.append("run.finished", status="canceled")
    before = journal.entries()
    assert cli.main(["finish", "blocked", "late response"]) == 2
    session_for(run).turn()
    assert journal.entries() == before


def test_brief_records_no_questions_and_no_fake_approval(run):
    session_for(run).turn()
    brief = (run.run_dir / "BRIEF.md").read_text()
    assert "Do not ask questions" in brief
    assert "Do not fabricate" in brief
    assert "all quality checks still apply" in brief
