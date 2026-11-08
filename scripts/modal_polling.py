"""Bounded waiting for an already submitted Modal call; never submits or relaunches.

Modal 1.5.5 raises builtin TimeoutError for a local get() poll expiry, but can also
raise that exact class after deserializing a remote failure. Recognize only the
known SDK poll frame; fail closed for other origins or SDK versions. Re-audit this
private SDK behavior when upgrading Modal.
"""

import builtins
import math
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Protocol, TypeVar

Result = TypeVar("Result", covariant=True)


class _Call(Protocol[Result]):
    def get(self, *, timeout: float) -> Result: ...


class CallDeadlineExceeded(builtins.TimeoutError):
    """The caller's overall waiting deadline expired."""


def _is_local_poll_timeout(error: builtins.TimeoutError) -> bool:
    if type(error) is not builtins.TimeoutError or error.args:
        return False
    try:
        if version("modal") != "1.5.5":
            return False
        from modal._functions import _Invocation
    except (ImportError, PackageNotFoundError):
        return False
    traceback = error.__traceback__
    if traceback is None:
        return False
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    frame = traceback.tb_frame
    return (
        frame.f_globals.get("__name__") == "modal._functions"
        and frame.f_code.co_name == "poll_function"
        and frame.f_code is _Invocation.poll_function.__code__
    )


def _positive_seconds(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def wait_for_call(
    call: _Call[Result],
    *,
    timeout_seconds: float = 1250,
    poll_seconds: float = 30,
    cancel: Callable[[], object] | None = None,
) -> Result:
    """Wait within one monotonic deadline, retrying only verified local poll misses.

    Each get() receives min(poll_seconds, remaining seconds). At the overall
    deadline, invoke the supplied owned-call cancellation callback exactly once,
    then raise CallDeadlineExceeded. Cancellation errors propagate without retry.
    Provider errors, remote errors (including builtin TimeoutError), and unknown
    timeout origins propagate immediately without cancellation by this helper.

    This is a cooperative synchronous bound: get() and cancel() must themselves
    honor their contracts. No thread, new submission, or automatic recovery is
    started. A get() result returned after the deadline is treated as expired.
    """
    timeout_seconds = _positive_seconds(timeout_seconds, "timeout_seconds")
    poll_seconds = _positive_seconds(poll_seconds, "poll_seconds")
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be callable or None")
    start = time.monotonic()
    deadline = start + timeout_seconds
    if not math.isfinite(start) or not math.isfinite(deadline) or deadline <= start:
        raise ValueError("timeout_seconds must produce a finite future monotonic deadline")

    def expired() -> None:
        if cancel is not None:
            cancel()
        raise CallDeadlineExceeded(f"Modal call wait exceeded {timeout_seconds:g} seconds")

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            expired()
        try:
            result = call.get(timeout=min(poll_seconds, remaining))
        except builtins.TimeoutError as error:
            if not _is_local_poll_timeout(error):
                raise
            continue
        if time.monotonic() >= deadline:
            expired()
        return result
