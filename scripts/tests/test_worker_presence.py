"""Supervisor liveness is fresh, scoped, and independent of blocking agent work."""

import json
import os
import threading
import time
import uuid

import pytest

from orchestrator import supervisor
from orchestrator.agent.harness import HarnessPolicy
from orchestrator.worker_presence import FRESHNESS_SECONDS, WorkerPresence, worker_status


def test_missing_root_and_empty_directory_are_unavailable(tmp_path):
    for root in (None, tmp_path, tmp_path / "missing"):
        assert worker_status(root) == {
            "status": "unavailable",
            "matchingWorkers": 0,
            "busyWorkers": 0,
            "detail": "No matching supervisor has a recent heartbeat.",
        }


def test_independent_workers_count_matching_and_busy_and_cleanup(tmp_path):
    with WorkerPresence(tmp_path) as broad:
        with WorkerPresence(tmp_path, run_id="selected") as selected:
            broad.set_active("other")
            selected.set_active("selected")
            assert worker_status(tmp_path)["matchingWorkers"] == 1
            assert worker_status(tmp_path)["busyWorkers"] == 1
            assert worker_status(tmp_path, "selected")["matchingWorkers"] == 2
            assert worker_status(tmp_path, "selected")["busyWorkers"] == 2
            selected.set_active(None)
            assert worker_status(tmp_path, "selected")["busyWorkers"] == 1
        assert worker_status(tmp_path, "selected")["matchingWorkers"] == 1
    assert worker_status(tmp_path)["status"] == "unavailable"
    assert list((tmp_path / ".workers").iterdir()) == []


@pytest.mark.parametrize("selected", [None, "selected"])
def test_one_pass_matches_only_its_active_run(tmp_path, selected):
    with WorkerPresence(tmp_path, run_id=selected, watcher=False) as presence:
        assert worker_status(tmp_path)["status"] == "unavailable"
        assert worker_status(tmp_path, "selected")["status"] == "unavailable"
        presence.set_active("selected")
        assert worker_status(tmp_path)["status"] == "unavailable"
        assert worker_status(tmp_path, "other")["status"] == "unavailable"
        assert worker_status(tmp_path, "selected")["matchingWorkers"] == 1
        presence.set_active(None)
        assert worker_status(tmp_path, "selected")["status"] == "unavailable"


def test_selected_watcher_never_advertises_new_uploads(tmp_path):
    with WorkerPresence(tmp_path, run_id="selected"):
        assert worker_status(tmp_path)["status"] == "unavailable"
        assert worker_status(tmp_path, "other")["status"] == "unavailable"
        assert worker_status(tmp_path, "selected")["status"] == "available"


@pytest.mark.parametrize(
    "mutation",
    [
        {"last_seen": time.time() - FRESHNESS_SECONDS - 5},
        {"last_seen": time.time() + 3600},
        {"last_seen": float("nan")},
        {"last_seen": float("inf")},
        {"last_seen": True},
        {"last_seen": "now"},
        {"version": 3},
        {"uploads_only": "true"},
        {"uploads_only": None},
        {"version": True},
        {"schema": "other"},
        {"watcher": "true"},
        {"runs_root": "/elsewhere"},
        {"run_id": []},
        {"active_run_id": {}},
        {"run_id": "selected", "active_run_id": "different"},
    ],
)
def test_invalid_records_are_ignored(tmp_path, mutation):
    with WorkerPresence(tmp_path) as presence:
        path = tmp_path / ".workers" / presence.name
        record = json.loads(path.read_text())
        record.update(mutation)
        path.write_text(json.dumps(record))
        assert worker_status(tmp_path)["status"] == "unavailable"


@pytest.mark.parametrize("contents", ["{bad", "[]", "null", "{}", '"text"', "[" * 1500])
def test_malformed_records_are_ignored(tmp_path, contents):
    with WorkerPresence(tmp_path) as presence:
        (tmp_path / ".workers" / presence.name).write_text(contents)
        assert worker_status(tmp_path)["status"] == "unavailable"


def test_escaping_and_symlink_records_are_ignored(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    with WorkerPresence(root) as presence:
        path = root / ".workers" / presence.name
        original = path.read_text()
        path.unlink()
        outside = tmp_path / "outside.json"
        outside.write_text(original)
        path.symlink_to(outside)
        assert worker_status(root)["status"] == "unavailable"
        path.unlink()
        record = json.loads(original)
        record["runs_root"] = str(root / ".." / "runs")
        path.write_text(json.dumps(record))
        assert worker_status(root)["status"] == "unavailable"
    (root / ".workers").rmdir()
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (outside_directory / f"{uuid.uuid4()}.json").write_text(original)
    (root / ".workers").symlink_to(outside_directory, target_is_directory=True)
    assert worker_status(root)["status"] == "unavailable"
    with pytest.raises(RuntimeError, match="Could not register"):
        with WorkerPresence(root):
            pytest.fail("registration through a symlink must fail")


def test_nonregular_oversized_and_temporary_records_are_ignored(tmp_path):
    with WorkerPresence(tmp_path) as presence:
        directory = tmp_path / ".workers"
        path = directory / presence.name
        original = path.read_text()
        path.unlink()
        (directory / "invalid-name.json").write_text(original)
        (directory / f".{uuid.uuid4()}.tmp").write_text(original)
        (directory / f"{uuid.uuid4()}.json").write_text(" " * 17000 + original)
        (directory / f"{uuid.uuid4()}.json").mkdir()
        os.mkfifo(directory / f"{uuid.uuid4()}.json")
        assert worker_status(tmp_path)["status"] == "unavailable"


def test_exception_removes_presence(tmp_path):
    with pytest.raises(ValueError, match="work failed"):
        with WorkerPresence(tmp_path):
            raise ValueError("work failed")
    assert list((tmp_path / ".workers").iterdir()) == []


def test_registration_failure_does_not_advertise_worker(tmp_path):
    with pytest.raises(RuntimeError, match="Could not register"):
        with WorkerPresence(tmp_path, heartbeat=tmp_path / "missing" / "heartbeat"):
            pytest.fail("failed heartbeat must not enter work")
    assert worker_status(tmp_path)["status"] == "unavailable"
    assert list((tmp_path / ".workers").iterdir()) == []


def test_background_failure_removes_presence_and_surfaces_error(tmp_path, monkeypatch):
    failed = threading.Event()
    presence = WorkerPresence(tmp_path, refresh_interval=0.01)
    with pytest.raises(RuntimeError, match="stopped refreshing"):
        with presence:

            def fail_write():
                failed.set()
                raise OSError("disk unavailable")

            monkeypatch.setattr(presence, "_write", fail_write)
            assert failed.wait(2)
            # Synchronize with the thread so cleanup has completed after the failing write.
            with presence._lock:
                assert worker_status(tmp_path)["status"] == "unavailable"
    assert list((tmp_path / ".workers").iterdir()) == []


@pytest.mark.parametrize(
    ("flags", "generic_available", "run_available"),
    [
        ([], True, True),
        (["--once"], False, True),
        (["--run-id", "selected"], False, True),
        (["--run-id", "selected", "--watch"], False, True),
    ],
)
@pytest.mark.parametrize("uploads_only", [False, True])
def test_supervisor_refreshes_during_blocking_work_and_cleans_up(
    tmp_path, monkeypatch, flags, generic_available, run_available, uploads_only
):
    supervisor.open_run(
        tmp_path, run_id="selected", name="clip", source=None, options={"hands_off": uploads_only}
    )
    monkeypatch.delenv("WANDER_STAGE_AUTHORIZATION", raising=False)
    if uploads_only:
        flags = [*flags, "--uploads-only"]
    heartbeat = tmp_path / "legacy-heartbeat"
    monkeypatch.setenv("WANDER_HEARTBEAT", str(heartbeat))
    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        HarnessPolicy,
        "from_environment",
        lambda: HarnessPolicy(kind="command", command=("unused",)),
    )
    refreshed = threading.Event()
    baseline = None
    instances = []

    class ObservedPresence(WorkerPresence):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, refresh_interval=0.01)
            instances.append(self)

        def _write(self):
            super()._write()
            if baseline is not None and threading.current_thread() is self._thread:
                refreshed.set()

    monkeypatch.setattr(supervisor, "WorkerPresence", ObservedPresence)

    def work(*_):
        nonlocal baseline
        assert (
            worker_status(tmp_path, hands_off=True)["status"] == "available"
        ) == generic_available
        if uploads_only:
            assert worker_status(tmp_path)["status"] == "unavailable"
            assert worker_status(tmp_path, "selected")["status"] == "unavailable"
        assert (
            worker_status(tmp_path, "selected", hands_off=True)["status"] == "available"
        ) == run_available
        assert worker_status(tmp_path, "selected", hands_off=True)["busyWorkers"] == 1
        path = next((tmp_path / ".workers").glob("*.json"))
        baseline = json.loads(path.read_text())["last_seen"]
        # The main supervisor is blocked here; only its background heartbeat can advance.
        assert refreshed.wait(2)
        with instances[0]._lock:
            current = json.loads(path.read_text())["last_seen"]
            assert current > baseline
            assert float(heartbeat.read_text()) == current
            assert heartbeat.read_text().endswith("\n")
        return supervisor.WorkOutcome("done", 0)

    class StopWatch(Exception):
        pass

    def stop_watch(_):
        assert worker_status(tmp_path, "selected")["busyWorkers"] == 0
        raise StopWatch

    monkeypatch.setattr(supervisor, "work_on", work)
    monkeypatch.setattr(supervisor.time, "sleep", stop_watch)
    once = "--once" in flags or ("--run-id" in flags and "--watch" not in flags)
    if once:
        assert supervisor.main(["--runs", str(tmp_path), *flags]) == 0
    else:
        with pytest.raises(StopWatch):
            supervisor.main(["--runs", str(tmp_path), *flags])
    assert list((tmp_path / ".workers").iterdir()) == []


def test_supervisor_work_failure_cleans_up(tmp_path, monkeypatch):
    supervisor.open_run(tmp_path, run_id="failed", name="clip", source=None)
    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        HarnessPolicy,
        "from_environment",
        lambda: HarnessPolicy(kind="command", command=("unused",)),
    )

    def fail_work(*_):
        assert worker_status(tmp_path, "failed")["busyWorkers"] == 1
        raise ValueError("work failed")

    monkeypatch.setattr(supervisor, "work_on", fail_work)
    with pytest.raises(ValueError, match="work failed"):
        supervisor.main(["--runs", str(tmp_path), "--once"])
    assert list((tmp_path / ".workers").iterdir()) == []


@pytest.mark.parametrize("selected", [None, "selected"])
def test_uploads_only_capability_matches_automatic_requests_only(tmp_path, selected):
    with WorkerPresence(tmp_path, run_id=selected, uploads_only=True) as presence:
        assert worker_status(tmp_path)["status"] == "unavailable"
        assert worker_status(tmp_path, "selected")["status"] == "unavailable"
        assert worker_status(tmp_path, hands_off=True)["matchingWorkers"] == (selected is None)
        assert worker_status(tmp_path, "selected", hands_off=True)["matchingWorkers"] == 1
        assert worker_status(tmp_path, "other", hands_off=True)["matchingWorkers"] == (
            selected is None
        )
        record = json.loads((tmp_path / ".workers" / presence.name).read_text())
        assert record["run_id"] == selected
        assert record["version"] == 2  # Version-1 readers reject instead of overstating capability.
        assert record["uploads_only"] is True


@pytest.mark.parametrize("hands_off", [False, True])
def test_historical_version_one_worker_is_unrestricted(tmp_path, hands_off):
    with WorkerPresence(tmp_path) as presence:
        path = tmp_path / ".workers" / presence.name
        record = json.loads(path.read_text())
        record["version"] = 1
        del record["uploads_only"]
        path.write_text(json.dumps(record))
        assert worker_status(tmp_path, hands_off=hands_off)["matchingWorkers"] == 1
        assert (
            worker_status(tmp_path, "manual-or-automatic", hands_off=hands_off)["matchingWorkers"]
            == 1
        )


def test_version_two_requires_explicit_capability(tmp_path):
    with WorkerPresence(tmp_path) as presence:
        path = tmp_path / ".workers" / presence.name
        record = json.loads(path.read_text())
        del record["uploads_only"]
        path.write_text(json.dumps(record))
        assert worker_status(tmp_path)["status"] == "unavailable"
        assert worker_status(tmp_path, hands_off=True)["status"] == "unavailable"


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_presence_rejects_non_boolean_capability(tmp_path, value):
    with pytest.raises(ValueError, match="scope"):
        WorkerPresence(tmp_path, uploads_only=value)
