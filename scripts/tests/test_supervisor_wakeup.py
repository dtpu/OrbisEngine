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
