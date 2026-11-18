"""Result waits must not leave an owned queued call running or buy a second call."""

import io
import json
import runpy
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from worker.stages.motion_submission import bounded_motion_result


def test_identity_is_durable_before_wait_and_success_is_not_cancelled(tmp_path):
    call = Mock(object_id="fc-owned")
    function = Mock()
    function.spawn.return_value = call

    def result(*, timeout):
        receipt = json.loads((tmp_path / "motion-call.json").read_text())
        assert receipt["functionCallId"] == "fc-owned"
        assert receipt["status"] == "waiting"
        assert timeout == 1200
        return {"archive": b"result"}

    call.get.side_effect = result
    assert bounded_motion_result(function, tmp_path, b"video", timeout=1200, batch=16) == {
        "archive": b"result"
    }
    function.spawn.assert_called_once_with(b"video", batch=16)
    call.cancel.assert_not_called()
    assert json.loads((tmp_path / "motion-call.json").read_text())["status"] == "result_received"


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_queue_timeout_cancels_only_owned_call_and_never_retries(tmp_path, cancel_fails):
    function = Mock()
    call = function.spawn.return_value
    call.object_id = "fc-owned"
    call.get.side_effect = TimeoutError()
    if cancel_fails:
        call.cancel.side_effect = OSError("sensitive provider response")
    with pytest.raises(RuntimeError, match="do not resubmit"):
        bounded_motion_result(function, tmp_path, b"video", timeout=1200)
    function.spawn.assert_called_once()
    call.get.assert_called_once_with(timeout=1200)
    call.cancel.assert_called_once_with()
    receipt = json.loads((tmp_path / "motion-call.json").read_text())
    assert receipt["functionCallId"] == "fc-owned"
    assert receipt["status"] == "wait_timeout"
    assert receipt["cancellation"] == ("unknown" if cancel_fails else "requested")
    assert "sensitive" not in json.dumps(receipt)
    with pytest.raises(FileExistsError):
        bounded_motion_result(function, tmp_path, b"video", timeout=1200)
    function.spawn.assert_called_once()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("provider failure")])
def test_interruption_cancels_without_reclassifying_as_wait_timeout(tmp_path, error):
    function = Mock()
    call = function.spawn.return_value
    call.object_id = "fc-owned"
    call.get.side_effect = error
    with pytest.raises(type(error)):
        bounded_motion_result(function, tmp_path, b"video", timeout=1200)
    call.cancel.assert_called_once_with()
    assert json.loads((tmp_path / "motion-call.json").read_text())["status"] == "wait_interrupted"


def test_unknown_submission_is_preserved_and_cannot_be_repeated(tmp_path):
    function = Mock()
    function.spawn.side_effect = OSError("lost response")
    with pytest.raises(OSError):
        bounded_motion_result(function, tmp_path, b"video", timeout=1200)
    receipt = json.loads((tmp_path / "motion-call.json").read_text())
    assert receipt["status"] == "submission_uncertain"
    assert receipt["functionCallId"] is None
    with pytest.raises(FileExistsError):
        bounded_motion_result(function, tmp_path, b"video", timeout=1200)
    function.spawn.assert_called_once()


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5, "600"])
def test_invalid_wait_fails_before_submission(tmp_path, timeout):
    function = Mock()
    with pytest.raises(ValueError):
        bounded_motion_result(function, tmp_path, b"video", timeout=timeout)
    function.spawn.assert_not_called()


@pytest.mark.parametrize("bounded", [False, True])
def test_motion_entrypoint_wires_bounded_wait_without_changing_default(
    tmp_path, monkeypatch, bounded
):
    import modal

    monkeypatch.delenv("WANDER_WORKER_TIMEOUT_SECONDS", raising=False)
    if bounded:
        monkeypatch.setenv("WANDER_WORKER_TIMEOUT_SECONDS", "600")
    identity = lambda *args, **kwargs: lambda function: function
    with (
        patch.object(modal.App, "function", identity),
        patch.object(modal.App, "local_entrypoint", identity),
    ):
        namespace = runpy.run_path(str(ROOT / "worker/modal_motion.py"))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz"):
        pass
    result = {"archive": archive.getvalue(), "report": {"error": None}}
    function = Mock()
    function.remote.return_value = result
    call = function.spawn.return_value
    call.object_id = "fc-entrypoint"
    call.get.return_value = result
    main = namespace["main"]
    main.__globals__["reconstruct"] = function
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    destination = tmp_path / "output"
    main(str(source), str(destination), experiment="pi3x")
    if bounded:
        function.remote.assert_not_called()
        function.spawn.assert_called_once_with(
            b"original", experiment="pi3x", batch=16, anchor_lock=False
        )
        call.get.assert_called_once_with(timeout=1200)
        assert (destination / "motion-call.json").exists()
    else:
        function.remote.assert_called_once_with(b"original", "pi3x", 16, False)
        function.spawn.assert_not_called()
        assert not (destination / "motion-call.json").exists()


def test_pinned_sdk_empty_poll_raises_builtin_timeout():
    """Verify the timeout class caught above against the installed SDK, without a request."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from modal._functions import _Invocation
    from modal_proto import api_pb2

    invocation = SimpleNamespace(
        pop_function_call_outputs=AsyncMock(
            return_value=api_pb2.FunctionGetOutputsResponse(num_unfinished_inputs=1)
        )
    )
    with pytest.raises(TimeoutError):
        asyncio.run(_Invocation.poll_function(invocation, timeout=0))
    invocation.pop_function_call_outputs.assert_awaited_once_with(
        index=0, timeout=0, clear_on_success=False
    )


def test_receipt_failure_after_spawn_still_cancels_owned_call(tmp_path, monkeypatch):
    from worker.stages import motion_submission

    write = motion_submission.atomic_json

    def fail_waiting(path, receipt):
        if receipt["status"] == "waiting":
            raise OSError("disk error")
        write(path, receipt)

    monkeypatch.setattr(motion_submission, "atomic_json", fail_waiting)
    function = Mock()
    call = function.spawn.return_value
    call.object_id = "fc-owned"
    with pytest.raises(OSError, match="disk error"):
        bounded_motion_result(function, tmp_path, b"video", timeout=1200)
    call.get.assert_not_called()
    call.cancel.assert_called_once_with()
    assert json.loads((tmp_path / "motion-call.json").read_text())["cancellation"] == "requested"
