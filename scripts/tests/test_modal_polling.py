"""Offline Modal 1.5.5 polling-origin and finite-deadline regression checks."""

import asyncio
import builtins
import sys
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock

import pytest
from modal._functions import _Invocation
from modal._serialization import serialize
from modal.exception import FunctionTimeoutError, OutputExpiredError
from modal.exception import TimeoutError as ModalTimeoutError
from modal_proto import api_pb2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal_polling


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def pending():
    return api_pb2.FunctionGetOutputsResponse(num_unfinished_inputs=1)


def completed(value=None, error=None, status=None):
    if status is None:
        status = (
            api_pb2.GenericResult.GENERIC_STATUS_FAILURE
            if error is not None
            else api_pb2.GenericResult.GENERIC_STATUS_SUCCESS
        )
    response = api_pb2.FunctionGetOutputsResponse()
    response.outputs.add(
        result=api_pb2.GenericResult(
            status=status,
            data=serialize(error if error is not None else value),
            exception="remote failure" if error is not None else "",
        ),
        data_format=api_pb2.DATA_FORMAT_PICKLE,
    )
    return response


class OfflineSdkCall:
    """Run the actual SDK poll/deserializer; replace its network boundary only."""

    def __init__(self, clock, responses):
        self.clock = clock
        self.responses = iter(responses)
        self.timeouts = []
        self.spawn = Mock(side_effect=AssertionError("must not submit"))
        self.invocation = _Invocation(stub=None, function_call_id="offline", client=None)

        async def pop(*, index, timeout, clear_on_success):
            assert index == 0
            assert clear_on_success is False
            elapsed, response = next(self.responses)
            self.clock.now += timeout if elapsed is None else elapsed
            return response

        self.invocation.pop_function_call_outputs = pop

    def get(self, *, timeout):
        self.timeouts.append(timeout)
        return asyncio.run(self.invocation.poll_function(timeout=timeout))


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(modal_polling.time, "monotonic", clock)
    return clock


def test_pinned_sdk_timeout_classes_are_distinct():
    assert version("modal") == "1.5.5", "Re-audit private polling behavior after SDK upgrade"
    assert not issubclass(ModalTimeoutError, builtins.TimeoutError)
    assert not issubclass(FunctionTimeoutError, builtins.TimeoutError)


def test_actual_sdk_pending_timeout_then_result(clock):
    call = OfflineSdkCall(clock, [(None, pending()), (1, completed({"ok": True}))])
    cancel = Mock()
    assert modal_polling.wait_for_call(call, timeout_seconds=75, cancel=cancel) == {"ok": True}
    assert call.timeouts == [30, 30]
    cancel.assert_not_called()
    call.spawn.assert_not_called()


def test_actual_sdk_remote_builtin_timeout_propagates(clock):
    call = OfflineSdkCall(clock, [(0, completed(error=builtins.TimeoutError()))])
    cancel = Mock()
    with pytest.raises(builtins.TimeoutError) as caught:
        modal_polling.wait_for_call(call, cancel=cancel)
    assert type(caught.value) is builtins.TimeoutError
    assert not modal_polling._is_local_poll_timeout(caught.value)
    assert call.timeouts == [30]
    cancel.assert_not_called()
    call.spawn.assert_not_called()


def test_actual_sdk_provider_timeout_propagates(clock):
    response = completed(status=api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT)
    call = OfflineSdkCall(clock, [(0, response)])
    cancel = Mock()
    with pytest.raises(FunctionTimeoutError):
        modal_polling.wait_for_call(call, cancel=cancel)
    assert call.timeouts == [30]
    cancel.assert_not_called()
    call.spawn.assert_not_called()


def test_actual_sdk_expired_output_propagates(clock):
    call = OfflineSdkCall(clock, [(0, api_pb2.FunctionGetOutputsResponse())])
    cancel = Mock()
    with pytest.raises(OutputExpiredError):
        modal_polling.wait_for_call(call, cancel=cancel)
    cancel.assert_not_called()


@pytest.mark.parametrize(
    "error", [builtins.TimeoutError(), ModalTimeoutError(), RuntimeError("bad")]
)
def test_unrecognized_errors_propagate_without_retry_or_cancel(clock, error):
    call = Mock()
    call.get.side_effect = error
    cancel = Mock()
    with pytest.raises(type(error)) as caught:
        modal_polling.wait_for_call(call, cancel=cancel)
    assert caught.value is error
    call.get.assert_called_once_with(timeout=30)
    call.spawn.assert_not_called()
    cancel.assert_not_called()


def test_other_sdk_version_fails_closed(clock, monkeypatch):
    monkeypatch.setattr(modal_polling, "version", lambda name: "1.5.6")
    call = OfflineSdkCall(clock, [(None, pending())])
    cancel = Mock()
    with pytest.raises(builtins.TimeoutError) as caught:
        modal_polling.wait_for_call(call, cancel=cancel)
    assert type(caught.value) is builtins.TimeoutError
    assert call.timeouts == [30]
    cancel.assert_not_called()


def test_absolute_deadline_caps_last_poll_and_cancels_once(clock):
    call = OfflineSdkCall(clock, [(None, pending())] * 3)
    cancel = Mock()
    with pytest.raises(modal_polling.CallDeadlineExceeded):
        modal_polling.wait_for_call(call, timeout_seconds=65, poll_seconds=30, cancel=cancel)
    assert call.timeouts == [30, 30, 5]
    assert clock.now == 165
    cancel.assert_called_once_with()
    call.spawn.assert_not_called()


def test_short_deadline_bounds_first_poll(clock):
    call = OfflineSdkCall(clock, [(None, pending())])
    with pytest.raises(modal_polling.CallDeadlineExceeded):
        modal_polling.wait_for_call(call, timeout_seconds=2, poll_seconds=30)
    assert call.timeouts == [2]


def test_result_returned_after_deadline_cancels_once(clock):
    call = OfflineSdkCall(clock, [(6, completed("late"))])
    cancel = Mock()
    with pytest.raises(modal_polling.CallDeadlineExceeded):
        modal_polling.wait_for_call(call, timeout_seconds=5, cancel=cancel)
    assert call.timeouts == [5]
    cancel.assert_called_once_with()
    call.spawn.assert_not_called()


def test_cancellation_error_propagates_without_second_attempt(clock):
    call = OfflineSdkCall(clock, [(None, pending())])
    error = RuntimeError("cancellation unresolved")
    cancel = Mock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        modal_polling.wait_for_call(call, timeout_seconds=5, cancel=cancel)
    assert caught.value is error
    cancel.assert_called_once_with()
    assert call.timeouts == [5]
    call.spawn.assert_not_called()


@pytest.mark.parametrize("name", ["timeout_seconds", "poll_seconds"])
@pytest.mark.parametrize(
    "value", [0, -1, float("nan"), float("inf"), -float("inf"), True, None, "30"]
)
def test_invalid_bounds_do_not_poll_cancel_or_submit(clock, name, value):
    call, cancel = Mock(), Mock()
    with pytest.raises(ValueError, match="finite positive"):
        modal_polling.wait_for_call(call, cancel=cancel, **{name: value})
    call.get.assert_not_called()
    call.spawn.assert_not_called()
    cancel.assert_not_called()


def test_overflowing_absolute_deadline_is_rejected(clock):
    clock.now = 1e308
    call, cancel = Mock(), Mock()
    with pytest.raises(ValueError, match="finite future"):
        modal_polling.wait_for_call(call, timeout_seconds=1e308, cancel=cancel)
    call.get.assert_not_called()
    cancel.assert_not_called()


def test_noncallable_cancellation_is_rejected_before_poll(clock):
    call = Mock()
    with pytest.raises(TypeError, match="cancel"):
        modal_polling.wait_for_call(call, cancel=False)
    call.get.assert_not_called()
