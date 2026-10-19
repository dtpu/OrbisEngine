"""Finite upload admissions persist without invoking a model or provider."""

import json

import pytest

from orchestrator import supervisor
from orchestrator.journal import Journal
from orchestrator.upload_admission import (
    AdmissionError,
    DISPATCH_LOCK_FILE,
    LEDGER_FILE,
    OWNER_LOCK_FILE,
    UploadAdmission,
)
from orchestrator.worker_presence import worker_status


def upload(root, name="one"):
    return supervisor.open_run(
        root, run_id=name, name=name, source=None, options={"hands_off": True}
    )


def claim(policy, run):
    with policy.dispatch():
        return policy.claim(run, supervisor.run_manifest(run))


def ledger(root):
    return json.loads((root / LEDGER_FILE).read_text())


@pytest.fixture(autouse=True)
def no_providers(monkeypatch):
    monkeypatch.delenv("WANDER_STAGE_AUTHORIZATION", raising=False)
    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Admission tests must not invoke a real harness")

    monkeypatch.setattr(supervisor.HarnessAgent, "run", forbidden)


def test_restart_keeps_failed_admission_consumed(tmp_path):
    first, second = upload(tmp_path), upload(tmp_path, "two")
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
        Journal(first).append("run.finished", status="failed")
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert policy.remaining == 0
        assert policy.active is None
        assert not claim(policy, second)
    assert len(ledger(tmp_path)["admissions"]) == 1


@pytest.mark.parametrize("status", ["failed", "succeeded", "canceled"])
def test_terminal_slot_waits_for_every_detached_attempt(tmp_path, status):
    first, second = upload(tmp_path), upload(tmp_path, "two")
    with UploadAdmission(tmp_path, 2, uploads_only=True) as policy:
        assert claim(policy, first)
        journal = Journal(first)
        for attempt in ("old", "new"):
            journal.append("step.started", step="world", attempt=attempt)
        journal.append("run.finished", status=status)
        assert not claim(policy, second)
        journal.append("step.finished", step="world", attempt="new", status="failed")
        assert not claim(policy, second)
        journal.append("step.finished", step="world", attempt="old", status="succeeded")
        assert claim(policy, second)
        assert policy.active == "two"
        assert policy.remaining == 0


def test_paused_run_keeps_slot_and_restart_reclaims_it(tmp_path):
    first, second = upload(tmp_path), upload(tmp_path, "two")
    with UploadAdmission(tmp_path, 2, uploads_only=True) as policy:
        assert claim(policy, first)
        (first / "paused").touch()
    with UploadAdmission(tmp_path, 2, uploads_only=True) as policy:
        assert policy.active == first.name
        assert not claim(policy, second)
        (first / "paused").unlink()
        assert claim(policy, first)
        assert len(ledger(tmp_path)["admissions"]) == 1


def test_resume_reclaims_existing_admission_without_refund(tmp_path):
    first, second = upload(tmp_path), upload(tmp_path, "two")
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
        Journal(first).append("run.finished", status="failed")
        assert not claim(policy, second)
        Journal(first).append("run.resumed", reason="operator")
        assert claim(policy, first)
        assert policy.remaining == 0


@pytest.mark.parametrize("mutation", ["manifest", "directory", "runId", "journal", "ledger"])
def test_missing_or_modified_admitted_evidence_fails_closed(tmp_path, mutation):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
    if mutation == "manifest":
        (first / "run.json").unlink()
    elif mutation == "directory":
        first.rename(tmp_path / "renamed")
    elif mutation == "runId":
        manifest = json.loads((first / "run.json").read_text())
        manifest["runId"] = "replacement"
        (first / "run.json").write_text(json.dumps(manifest))
    elif mutation == "journal":
        (first / "journal.jsonl").unlink()
    else:
        (tmp_path / LEDGER_FILE).unlink()
    with pytest.raises((AdmissionError, OSError)):
        with UploadAdmission(tmp_path, 1, uploads_only=True):
            pytest.fail("Changed evidence must not authorize work")


def test_manifest_hash_is_canonical_but_option_change_is_rejected(tmp_path):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
        manifest = json.loads((first / "run.json").read_text())
        (first / "run.json").write_text(json.dumps(manifest, sort_keys=True))
        assert claim(policy, first)
        manifest["options"]["people"] = 2
        (first / "run.json").write_text(json.dumps(manifest))
        with pytest.raises(AdmissionError, match="identity changed"):
            claim(policy, first)


@pytest.mark.parametrize("contents", ["{}", "[]", "{bad", '{"x":1,"x":2}', "NaN"])
@pytest.mark.parametrize("target", [LEDGER_FILE, "journal.jsonl"])
def test_corrupt_ledger_or_journal_never_releases_slot(tmp_path, contents, target):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
    path = tmp_path / target if target == LEDGER_FILE else first / target
    path.write_text(contents)
    with pytest.raises(AdmissionError):
        with UploadAdmission(tmp_path, 1, uploads_only=True):
            pytest.fail("Corrupt evidence must fail closed")


@pytest.mark.parametrize(
    "evidence", ["started", "terminal_unresolved", "transcript", "finished_step"]
)
def test_new_policy_refuses_unadmitted_existing_work(tmp_path, evidence):
    first = upload(tmp_path)
    if evidence == "transcript":
        (first / "transcript.jsonl").touch()
    elif evidence == "finished_step":
        Journal(first).append("step.finished", step="world", attempt="one", status="succeeded")
    else:
        Journal(first).append("step.started", step="world", attempt="one")
        if evidence == "terminal_unresolved":
            Journal(first).append("run.finished", status="failed")
    with pytest.raises(AdmissionError, match="already started"):
        with UploadAdmission(tmp_path, 1, uploads_only=True):
            pytest.fail("Existing work cannot be made free")
    assert not (tmp_path / LEDGER_FILE).exists()


def test_owner_lock_excludes_second_worker_and_releases_on_exit(tmp_path):
    with UploadAdmission(tmp_path, 2, uploads_only=True):
        with pytest.raises(AdmissionError, match="already owns"):
            with UploadAdmission(tmp_path, 2, uploads_only=True):
                pytest.fail("Only one bounded owner is allowed")
    with UploadAdmission(tmp_path, 2, uploads_only=True):
        pass


def test_unrestricted_existing_worker_rechecks_new_policy(tmp_path):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, None) as unrestricted:
        assert claim(unrestricted, first)
        with UploadAdmission(tmp_path, 1, uploads_only=True):
            assert not claim(unrestricted, first)
        assert not claim(unrestricted, first)
        manual = supervisor.open_run(tmp_path, run_id="manual", name="manual", source=None)
        assert claim(unrestricted, manual)


@pytest.mark.parametrize("target", [LEDGER_FILE, DISPATCH_LOCK_FILE, OWNER_LOCK_FILE])
def test_state_and_lock_symlinks_are_never_followed(tmp_path, target):
    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    (root / target).symlink_to(outside)
    with pytest.raises((AdmissionError, OSError)):
        with UploadAdmission(root, 1, uploads_only=True):
            pytest.fail("Must not follow symlinks")
    assert outside.read_text() == "preserve"


@pytest.mark.parametrize("target", ["run.json", "journal.jsonl"])
def test_admitted_manifest_and_journal_symlinks_fail_closed(tmp_path, target):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
        path = first / target
        outside = tmp_path / "outside"
        path.rename(outside)
        path.symlink_to(outside)
        with pytest.raises((AdmissionError, OSError)):
            claim(policy, first)


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-uploads", "0"],
        ["--max-uploads", "-1"],
        ["--max-uploads", "1"],
        ["--uploads-only", "--max-uploads", "1.5"],
    ],
)
def test_invalid_flags_fail_before_presence_or_harness(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(supervisor, "WorkerPresence", lambda *_a, **_kw: pytest.fail("no presence"))
    with pytest.raises(SystemExit) as error:
        supervisor.main(["--runs", str(tmp_path), "--once", *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("flags", [["--uploads-only"], ["--uploads-only", "--max-uploads", "2"]])
def test_existing_policy_requires_same_limit_before_presence(tmp_path, monkeypatch, flags):
    with UploadAdmission(tmp_path, 1, uploads_only=True):
        pass
    monkeypatch.setattr(supervisor, "WorkerPresence", lambda *_a, **_kw: pytest.fail("no presence"))
    with pytest.raises(SystemExit) as error:
        supervisor.main(["--runs", str(tmp_path), "--once", *flags])
    assert error.value.code == 2


def test_watch_presence_scopes_active_pause_drain_and_exhaustion(tmp_path, monkeypatch):
    first = upload(tmp_path)
    second = upload(tmp_path, "two")
    calls = []
    passes = 0

    class Done(Exception):
        pass

    def work(path, *_args):
        calls.append(path)
        assert worker_status(tmp_path, hands_off=True)["status"] == "unavailable"
        assert worker_status(tmp_path, path.name, hands_off=True)["busyWorkers"] == 1
        Journal(path).append("step.started", step="world", attempt="one")
        (path / "paused").touch()
        return supervisor.WorkOutcome("paused", 1)

    def tick(_):
        nonlocal passes
        passes += 1
        assert calls == [first]
        assert worker_status(tmp_path, hands_off=True)["status"] == "unavailable"
        if passes == 1:
            assert worker_status(tmp_path, "one", hands_off=True)["status"] == "available"
            Journal(first).append("run.finished", status="canceled")
        elif passes == 2:
            assert worker_status(tmp_path, "one", hands_off=True)["status"] == "available"
            Journal(first).append("step.finished", step="world", attempt="one", status="failed")
        else:
            assert worker_status(tmp_path, "one", hands_off=True)["status"] == "unavailable"
            assert worker_status(tmp_path, "two", hands_off=True)["status"] == "unavailable"
            raise Done

    monkeypatch.setattr(supervisor, "work_on", work)
    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(Done):
        supervisor.main(
            ["--runs", str(tmp_path), "--uploads-only", "--max-uploads", "1", "--watch"]
        )
    assert not (second / "journal.jsonl").exists()


def test_idle_bounded_watcher_accepts_new_uploads(tmp_path, monkeypatch):
    class Done(Exception):
        pass

    def tick(_):
        assert worker_status(tmp_path, hands_off=True)["status"] == "available"
        raise Done

    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(Done):
        supervisor.main(
            ["--runs", str(tmp_path), "--uploads-only", "--max-uploads", "1", "--watch"]
        )


@pytest.mark.parametrize("status", ["blocked", "skipped"])
def test_normal_nonsuccess_step_outcomes_settle_admission(tmp_path, status):
    first, second = upload(tmp_path), upload(tmp_path, "two")
    with UploadAdmission(tmp_path, 2, uploads_only=True) as policy:
        assert claim(policy, first)
        journal = Journal(first)
        journal.append("step.started", step="world", attempt="one")
        journal.append("step.executing", step="world", attempt="one")
        journal.append("step.finished", step="world", attempt="one", status=status)
        journal.append("run.finished", status="failed")
        assert claim(policy, second)


@pytest.mark.parametrize(
    "admission,reason", [("occupied", "admission_busy"), ("exhausted", "admission_exhausted")]
)
def test_presence_admission_reason_requires_automatic_unmatched_query(tmp_path, admission, reason):
    from orchestrator.worker_presence import WorkerPresence

    with WorkerPresence(tmp_path, uploads_only=True) as presence:
        presence.set_scope(
            run_id="one" if admission == "occupied" else None,
            watcher=admission == "occupied",
            active_run_id="one" if admission == "occupied" else None,
            admission=admission,
        )
        status = worker_status(tmp_path, hands_off=True)
        assert status["status"] == "unavailable"
        assert status["reason"] == reason
        assert "queued" in status["detail"]
        assert "reason" not in worker_status(tmp_path)
        assert "reason" not in worker_status(tmp_path, "two")
        if admission == "occupied":
            assert "reason" not in worker_status(tmp_path, "one", hands_off=True)
        with WorkerPresence(tmp_path, uploads_only=True):
            assert worker_status(tmp_path, hands_off=True)["status"] == "available"
            assert "reason" not in worker_status(tmp_path, hands_off=True)


def test_dispatch_serializes_policy_installation_with_unrestricted_work(tmp_path):
    import threading

    first = upload(tmp_path)
    attempting, completed = threading.Event(), threading.Event()
    errors = []

    def install():
        attempting.set()
        try:
            with UploadAdmission(tmp_path, 1, uploads_only=True):
                errors.append("unexpected policy installation")
        except AdmissionError as error:
            errors.append(str(error))
        finally:
            completed.set()

    with UploadAdmission(tmp_path, None) as unrestricted:
        with unrestricted.dispatch():
            assert unrestricted.claim(first, supervisor.run_manifest(first))
            thread = threading.Thread(target=install)
            thread.start()
            assert attempting.wait(2)
            assert not completed.wait(0.05)
            Journal(first).append("step.started", step="world", attempt="existing")
        assert completed.wait(2)
        thread.join()
    assert len(errors) == 1 and "already started" in errors[0]
    assert not (tmp_path / LEDGER_FILE).exists()


@pytest.mark.parametrize("target", [DISPATCH_LOCK_FILE, OWNER_LOCK_FILE])
def test_replaced_lock_stops_live_owner(tmp_path, target):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
        path = tmp_path / target
        contents = path.read_bytes()
        path.unlink()
        path.write_bytes(contents)
        with pytest.raises(AdmissionError, match="lock was replaced"):
            claim(policy, first)


def test_conflicting_run_id_fails_before_presence(tmp_path, monkeypatch):
    first = upload(tmp_path)
    with UploadAdmission(tmp_path, 2, uploads_only=True) as policy:
        assert claim(policy, first)
    monkeypatch.setattr(supervisor, "WorkerPresence", lambda *_a, **_kw: pytest.fail("no presence"))
    with pytest.raises(SystemExit) as error:
        supervisor.main(
            ["--runs", str(tmp_path), "--uploads-only", "--max-uploads", "2", "--run-id", "other"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("maxUploads", True),
        ("root", "/different"),
        ("active", "unclaimed"),
        ("admissions", {}),
        ("extra", 1),
    ],
)
def test_ledger_schema_and_root_are_strict(tmp_path, field, value):
    with UploadAdmission(tmp_path, 1, uploads_only=True):
        pass
    state = ledger(tmp_path)
    state[field] = value
    (tmp_path / LEDGER_FILE).write_text(json.dumps(state))
    with pytest.raises(AdmissionError):
        with UploadAdmission(tmp_path, 1, uploads_only=True):
            pytest.fail("Invalid policy must not be used")


def test_unrestricted_selected_manual_run_still_dispatches_under_active_policy(
    tmp_path, monkeypatch
):
    first = upload(tmp_path)
    manual = supervisor.open_run(tmp_path, run_id="manual", name="manual", source=None)
    with UploadAdmission(tmp_path, 1, uploads_only=True) as policy:
        assert claim(policy, first)
    calls = []
    monkeypatch.setattr(
        supervisor,
        "work_on",
        lambda path, *_: calls.append(path) or supervisor.WorkOutcome("manual", 0),
    )
    assert supervisor.main(["--runs", str(tmp_path), "--run-id", "manual"]) == 0
    assert calls == [manual]


def test_unrestricted_watcher_on_policy_root_advertises_manual_only(tmp_path, monkeypatch):
    with UploadAdmission(tmp_path, 1, uploads_only=True):
        pass

    class Done(Exception):
        pass

    def tick(_):
        assert worker_status(tmp_path)["status"] == "available"
        assert worker_status(tmp_path, "manual")["status"] == "available"
        assert worker_status(tmp_path, hands_off=True)["status"] == "unavailable"
        assert worker_status(tmp_path, "automatic", hands_off=True)["status"] == "unavailable"
        raise Done

    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(Done):
        supervisor.main(["--runs", str(tmp_path), "--watch"])


def test_selected_watcher_stops_if_another_admitted_run_is_resumed(tmp_path, monkeypatch):
    first, second = upload(tmp_path), upload(tmp_path, "two")
    with UploadAdmission(tmp_path, 2, uploads_only=True) as policy:
        assert claim(policy, first)
        Journal(first).append("run.finished", status="failed")
        assert claim(policy, second)
        Journal(second).append("run.finished", status="failed")
    passes = 0

    def tick(_):
        nonlocal passes
        passes += 1
        assert passes == 1
        Journal(first).append("run.resumed", reason="operator")

    monkeypatch.setattr(supervisor.time, "sleep", tick)
    with pytest.raises(SystemExit) as error:
        supervisor.main(
            [
                "--runs",
                str(tmp_path),
                "--uploads-only",
                "--max-uploads",
                "2",
                "--run-id",
                "two",
                "--watch",
            ]
        )
    assert error.value.code == 2
    assert worker_status(tmp_path, "one", hands_off=True)["status"] == "unavailable"
