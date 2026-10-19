"""Journal events that arrive as a turn ends must wake the next supervisor pass."""

import time

import pytest

from orchestrator import supervisor
from orchestrator.agent.harness import HarnessPolicy
from orchestrator.journal import Journal


def test_answer_after_outcome_is_seen_on_next_pass_without_waiting_for_timer(tmp_path, monkeypatch):
    run = supervisor.open_run(tmp_path, run_id="reply-run", name="clip", source=None)
    supervisor.open_run(tmp_path, run_id="unrelated-run", name="another-clip", source=None)
    journal = Journal(run)
    calls = []

    def work(*args):
        calls.append(args[0])
        if len(calls) == 1:
            journal.append("question", question="Continue?")
            outcome = supervisor.WorkOutcome("waiting", len(journal.entries()))
            # The operator answers after the session's last read but before the main
            # loop records its cursor. The answer must still trigger another turn.
            journal.append("answer", text="Continue with retained inputs only")
            return outcome
        assert journal.unanswered() is None
        journal.append("question", question="Finished the bounded check; review outputs")
        return supervisor.WorkOutcome("waiting for review", len(journal.entries()))

    passes = 0

    class EndCheck(Exception):
        pass

    def tick(_):
        nonlocal passes
        passes += 1
        if passes == 3:
            raise EndCheck

    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        HarnessPolicy,
        "from_environment",
        lambda: HarnessPolicy(kind="command", command=("unused",)),
    )
    monkeypatch.setattr(supervisor, "work_on", work)
    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(EndCheck):
        supervisor.main(
            ["--runs", str(tmp_path), "--push-after", "3600", "--run-id", "reply-run", "--watch"]
        )
    assert calls == [run, run]
    assert supervisor.runs_needing_work(tmp_path) == [tmp_path / "another-clip"]


def test_observed_journal_stays_quiet_until_a_new_event_or_timer(tmp_path):
    journal = Journal(tmp_path)
    journal.append("note", text="No progress in this turn")
    pushed = {str(tmp_path): (time.monotonic(), len(journal.entries()))}
    assert not supervisor.due(tmp_path, pushed, 3600)
    journal.append("answer", text="Try the retained source")
    assert supervisor.due(tmp_path, pushed, 3600)


@pytest.mark.parametrize("flags", [[], ["--once"]])
def test_named_run_still_exits_after_one_pass_without_watch(tmp_path, monkeypatch, flags):
    run = supervisor.open_run(tmp_path, run_id="reply-run", name="clip", source=None)
    calls = []
    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        supervisor,
        "work_on",
        lambda path, *_: calls.append(path) or supervisor.WorkOutcome("quiet", 0),
    )
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: pytest.fail("one pass must not wait"))
    assert supervisor.main(["--runs", str(tmp_path), "--run-id", "reply-run", *flags]) == 0
    assert calls == [run]


def test_once_and_watch_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        supervisor.main(["--once", "--watch"])
    assert error.value.code == 2


@pytest.mark.parametrize("turns", [1, 3])
def test_fast_detached_finish_on_last_turn_resumes_on_next_pass(tmp_path, monkeypatch, turns):
    from orchestrator.agent.contracts import AgentResult
    from orchestrator.agent.harness import HarnessAgent, HarnessOutcome

    run = supervisor.open_run(tmp_path, run_id="fast-run", name="clip", source=None)
    journal = Journal(run)
    calls = []

    def agent_turn(*args, **kwargs):
        calls.append(len(journal.entries()))
        if len(calls) <= turns:
            attempt = f"attempt-{len(calls)}"
            journal.append("step.started", step="review", attempt=attempt, number=len(calls))
            # The detached child finishes while the harness still prepares its final
            # message, so RunSession.turn observes BOTH events before returning.
            journal.append("step.finished", step="review", attempt=attempt, status="succeeded")
        else:
            assert journal.active_attempts() == {}
            assert journal.last_status("review") == "succeeded"
            journal.append("question", question="Review the next bounded action")
        return HarnessOutcome(AgentResult(status="completed", transcript_path="inert"))

    passes = 0

    class EndCheck(Exception):
        pass

    def tick(_):
        nonlocal passes
        passes += 1
        if passes == 1:
            # The fix must not exceed this pass's budget to evade the race.
            assert len(calls) == turns
        if passes == 3:
            raise EndCheck

    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        HarnessPolicy,
        "from_environment",
        lambda: HarnessPolicy(kind="command", command=("unused",)),
    )
    monkeypatch.setattr(HarnessAgent, "run", agent_turn)
    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(EndCheck):
        supervisor.main(
            [
                "--runs",
                str(tmp_path),
                "--run-id",
                "fast-run",
                "--watch",
                "--turns",
                str(turns),
                "--push-after",
                "3600",
            ]
        )
    assert len(calls) == turns + 1
    assert journal.unanswered() == "Review the next bounded action"
    assert len([e for e in journal.entries() if e.kind == "step.started"]) == turns


@pytest.mark.parametrize("mode", ["active", "idle", "question", "finished", "paused", "error"])
def test_fast_work_followup_does_not_bypass_waiting_or_backoff(tmp_path, monkeypatch, mode):
    from orchestrator.agent.contracts import AgentResult
    from orchestrator.agent.harness import HarnessAgent, HarnessOutcome

    run = supervisor.open_run(tmp_path, run_id="held-run", name="clip", source=None)
    journal = Journal(run)
    calls = []

    def agent_turn(*args, **kwargs):
        calls.append(1)
        if mode != "idle":
            journal.append("step.started", step="review", attempt="one", number=1)
            if mode != "active":
                journal.append("step.finished", step="review", attempt="one", status="succeeded")
        if mode == "question":
            journal.append("question", question="Operator decision required")
        elif mode == "finished":
            journal.append("run.finished", status="succeeded")
        elif mode == "paused":
            (run / "paused").touch()
        result = AgentResult(
            status="stalled" if mode == "error" else "completed",
            transcript_path="inert",
            error="silent" if mode == "error" else None,
        )
        return HarnessOutcome(result)

    passes = 0

    class EndCheck(Exception):
        pass

    def tick(_):
        nonlocal passes
        passes += 1
        if passes == 3:
            raise EndCheck

    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        HarnessPolicy,
        "from_environment",
        lambda: HarnessPolicy(kind="command", command=("unused",)),
    )
    monkeypatch.setattr(HarnessAgent, "run", agent_turn)
    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(EndCheck):
        supervisor.main(
            [
                "--runs",
                str(tmp_path),
                "--run-id",
                "held-run",
                "--watch",
                "--turns",
                "1",
                "--push-after",
                "3600",
            ]
        )
    assert len(calls) == 1
    assert len(journal.active_attempts()) == (1 if mode == "active" else 0)
