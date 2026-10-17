"""Overlapping agents and child replay cannot admit the same stage twice."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import cli
from orchestrator.journal import Journal
from scripts.tests import test_run_controls as fixtures

run = fixtures.run
step_args = fixtures.step_args


def pending_child(run, monkeypatch):
    spawn = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, step_args()) == 0
    return cli.build_parser().parse_args(spawn.call_args.args[0][3:])


def test_simultaneous_detach_requests_create_one_attempt_and_child(run, monkeypatch):
    spawn = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    ready = Barrier(2)

    def start():
        ready.wait(timeout=5)
        return cli.command_step(run.context, step_args())

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(start), pool.submit(start)
        assert sorted([first.result(timeout=5), second.result(timeout=5)]) == [0, 2]
    spawn.assert_called_once()
    assert len([e for e in run.journal.entries() if e.kind == "step.started"]) == 1
    assert not run.journal.execution_claimed("pi3x", "run-1:pi3x:1")


@pytest.mark.parametrize("detached", [False, True])
def test_running_executor_keeps_log_and_blocks_duplicate_and_child_replay(
    run, monkeypatch, detached
):
    args = pending_child(run, monkeypatch) if detached else step_args(True)
    started, finish = Event(), Event()

    def lines():
        yield b"original executor log\n"
        started.set()
        assert finish.wait(5), "test must release the original executor"

    spawn = Mock(return_value=SimpleNamespace(stdout=lines(), wait=lambda: 0))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(cli.command_step, run.context, args)
        try:
            assert started.wait(5)
            assert cli.command_step(run.context, step_args(True)) == 2
            assert cli.command_step(run.context, step_args()) == 2
            if detached:
                assert cli.command_step(run.context, args) == 2
            assert (
                run.context.run_dir / "logs/pi3x.1.log"
            ).read_bytes() == b"original executor log\n"
        finally:
            finish.set()
        assert owner.result(timeout=5) == 0
    spawn.assert_called_once()
    assert not run.journal.active_attempts()
    if detached:
        assert cli.command_step(run.context, args) == 2
        spawn.assert_called_once()


def test_independent_stage_can_finish_while_first_executor_is_running(run, monkeypatch):
    started, finish = Event(), Event()

    def lines():
        started.set()
        assert finish.wait(5), "test must release the original executor"
        yield b"original executor finished\n"

    spawn = Mock(
        side_effect=[
            SimpleNamespace(stdout=lines(), wait=lambda: 0),
            fixtures.stage_process(),
        ]
    )
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cli.command_step, run.context, step_args(True))
        try:
            assert started.wait(5)
            independent = pool.submit(
                cli.command_step,
                run.context,
                cli.build_parser().parse_args(["step", "--wait", "tracks"]),
            )
            assert independent.result(timeout=3) == 0
            assert set(run.journal.active_attempts()) == {"pi3x"}
        finally:
            finish.set()
        assert owner.result(timeout=5) == 0
    assert spawn.call_count == 2
    assert not run.journal.active_attempts()


@pytest.mark.parametrize("change", ["parameters", "number", "unrecorded", "legacy"])
def test_child_must_match_the_unclaimed_parent_admission(run, monkeypatch, change):
    child = pending_child(run, monkeypatch)
    original = cli.build_parser().parse_args(
        ["step", "--wait", "--number", "1", "--attempt", child.attempt, "pi3x"]
    )
    if change == "parameters":
        child.rest += ["--fps", "6"]
    elif change == "number":
        child.number = 2
    elif change == "unrecorded":
        child.attempt = "unrecorded-attempt"
    else:
        # Pre-upgrade starts have no durable child-claim protocol; fail closed.
        run.journal.append("step.finished", step="pi3x", attempt=child.attempt, status="failed")
        run.journal.append(
            "step.started",
            step="pi3x",
            attempt=child.attempt,
            number=1,
            command=cli.run_clip_command(run.context, "pi3x", []),
        )
    spawn = Mock(side_effect=AssertionError("mismatched child must not execute"))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, child) == 2
    spawn.assert_not_called()
    assert len(run.journal.active_attempts()["pi3x"]) == 1
    if change != "legacy":
        spawn.side_effect = None
        spawn.return_value = fixtures.stage_process()
        assert cli.command_step(run.context, original) == 0
        spawn.assert_called_once()


def test_execution_claim_survives_reentry_after_local_executor_disappears(run, monkeypatch):
    child = pending_child(run, monkeypatch)
    run.journal.append("step.executing", step="pi3x", attempt=child.attempt, pid=123)
    monkeypatch.setattr(cli.RunContext, "journal", lambda self: Journal(self.run_dir))
    spawn = Mock(side_effect=AssertionError("uncertain execution must not be retried"))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, child) == 2
    assert cli.command_step(run.context, step_args(True)) == 2
    spawn.assert_not_called()
    assert not [e for e in run.journal.entries() if e.kind == "step.finished"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), OSError("log stream lost")])
def test_losing_executor_output_does_not_release_a_possibly_running_attempt(
    run, monkeypatch, interruption
):
    def lines():
        yield b"local child started\n"
        raise interruption

    spawn = Mock(return_value=SimpleNamespace(stdout=lines(), wait=lambda: 0))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, step_args(True)) == 1
    assert set(run.journal.active_attempts()["pi3x"]) == {"run-1:pi3x:1"}
    assert not [e for e in run.journal.entries() if e.kind == "step.finished"]
    assert "remains unresolved" in run.journal.entries()[-1].data["text"]
    assert cli.command_step(run.context, step_args(True)) == 2
    spawn.assert_called_once()


def test_spawn_failure_closes_admission_before_an_explicit_retry(run, monkeypatch):
    spawn = Mock(side_effect=OSError("executable unavailable"))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, step_args()) == 1
    assert run.journal.active_attempts() == {}
    failed = run.journal.entries()[-1]
    assert failed.kind == "step.finished" and failed.data["preflight"] is True
    assert not run.journal.execution_claimed("pi3x", "run-1:pi3x:1")
    spawn.side_effect = None
    spawn.return_value = SimpleNamespace()
    assert cli.command_step(run.context, step_args()) == 0
    assert set(run.journal.active_attempts()["pi3x"]) == {"run-1:pi3x:2"}


def test_child_missing_credentials_closes_only_its_unlaunched_attempt(run, monkeypatch):
    child = pending_child(run, monkeypatch)
    monkeypatch.setattr(cli, "missing_credentials", lambda step: ["EXAMPLE_REQUIRED_KEY"])
    spawn = Mock(side_effect=AssertionError("missing credentials must not launch"))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, child) == 2
    assert run.journal.active_attempts() == {}
    assert run.journal.last_status("pi3x") == "blocked"
    assert not run.journal.execution_claimed("pi3x", child.attempt)
    spawn.assert_not_called()


def test_stale_completion_cannot_hide_older_running_attempt(run):
    for attempt in ("older", "newer"):
        run.journal.append("step.started", step="pi3x", attempt=attempt, number=1)
    run.journal.append("step.finished", step="pi3x", attempt="newer", status="failed")
    assert cli.in_flight(run.journal) == {"pi3x": "older"}


@pytest.mark.parametrize("status", ["paused", "canceled", "question"])
def test_control_between_claim_and_launch_prevents_execution(run, monkeypatch, status):
    open_file = Path.open

    def control_before_log(path, *args, **kwargs):
        if path == run.context.run_dir / "logs/pi3x.1.log":
            with run.journal.admission_lock():
                fixtures.control(run, status)
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", control_before_log)
    spawn = Mock(side_effect=AssertionError("must not execute after an accepted control"))
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(run.context, step_args(True)) == 2
    assert run.journal.execution_claimed("pi3x", "run-1:pi3x:1")
    assert run.journal.last_status("pi3x") == "blocked"
    assert not run.journal.active_attempts()
    spawn.assert_not_called()


@pytest.mark.parametrize("status", ["paused", "question"])
def test_launch_and_operator_control_are_serialized_without_waiting_for_execution(
    run, monkeypatch, status
):
    at_launch, allow_launch, at_execution, finish = Event(), Event(), Event(), Event()
    control_requested, control_accepted = Event(), Event()

    def lines():
        at_execution.set()
        assert finish.wait(5)
        yield b"already launched work can finish\n"

    def spawn(*args, **kwargs):
        at_launch.set()
        assert allow_launch.wait(5)
        return SimpleNamespace(stdout=lines(), wait=lambda: 0)

    def pause():
        control_requested.set()
        if status == "question":
            cli.command_ask(run.context, SimpleNamespace(question="Review first", step="pi3x"))
        else:
            with run.journal.admission_lock():
                fixtures.control(run, status)
        control_accepted.set()

    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(cli.command_step, run.context, step_args(True))
        try:
            assert at_launch.wait(5)
            control = pool.submit(pause)
            assert control_requested.wait(5)
            assert not control_accepted.wait(0.1)
            allow_launch.set()
            assert at_execution.wait(5)
            assert control_accepted.wait(5)
            assert not owner.done()
            control.result(timeout=5)
        finally:
            allow_launch.set()
            finish.set()
        assert owner.result(timeout=5) == 0
    if status == "question":
        assert run.journal.unanswered() == "Review first"
    else:
        assert run.journal.control_status() == "paused"


def test_real_detached_processes_share_admission_and_reject_child_replay(run, tmp_path):
    """Run the actual CLI/child protocol with a harmless local executor, never a provider."""
    repository = tmp_path / "stub-repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (repository / "orchestrator").symlink_to(ROOT / "orchestrator", target_is_directory=True)
    (scripts / "run_clip.py").write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['WANDER_RUN_DIR'])\n"
        "with (root / 'executions.txt').open('a') as handle:\n"
        "    handle.write('executed once\\n')\n"
        "print('original executor log', flush=True)\n"
        "deadline = time.monotonic() + 15\n"
        "while not (root / 'release').exists():\n"
        "    if time.monotonic() > deadline:\n"
        "        raise SystemExit('test executor timed out')\n"
        "    time.sleep(0.02)\n"
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "runId": "run-1",
                "name": "clip",
                "source": str(run.context.source),
                "sourceSha256": run.context.source_sha256,
                "repository": str(repository),
                "options": {"marble": "none"},
            }
        )
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "WANDER_RUN_ID": "run-1",
        "WANDER_RUN_DIR": str(tmp_path),
        "WANDER_DATABASE_URL": "",
        "WANDER_STAGE_LEDGER": str(tmp_path / "unused-ledger.json"),
    }
    command = [sys.executable, "-m", "orchestrator.cli", "step"]

    def wait_for(condition):
        deadline = time.monotonic() + 10
        while not condition():
            assert time.monotonic() < deadline, "local child did not reach the expected state"
            time.sleep(0.02)

    processes = []
    try:
        for _ in range(2):
            processes.append(
                subprocess.Popen(
                    [*command, "pi3x"],
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        for process in processes:
            process.communicate(timeout=10)
        assert sorted(p.returncode for p in processes) == [0, 2]
        wait_for(lambda: (tmp_path / "logs/pi3x.1.log").exists())
        log = tmp_path / "logs/pi3x.1.log"
        wait_for(lambda: b"original executor log" in log.read_bytes())
        replay = [*command, "--wait", "--number", "1", "--attempt", "run-1:pi3x:1", "pi3x"]
        for duplicate in ([*command, "--wait", "pi3x"], replay):
            result = subprocess.run(
                duplicate, cwd=ROOT, env=environment, capture_output=True, timeout=10
            )
            assert result.returncode == 2, result.stderr.decode()
        assert log.read_bytes() == b"original executor log\n"
    finally:
        (tmp_path / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    wait_for(lambda: not Journal(tmp_path).active_attempts())
    assert (tmp_path / "executions.txt").read_text() == "executed once\n"
    assert Journal(tmp_path).last_status("pi3x") == "succeeded"
    result = subprocess.run(replay, cwd=ROOT, env=environment, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert not (tmp_path / "unused-ledger.json").exists()
