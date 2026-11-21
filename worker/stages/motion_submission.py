"""One bounded motion submission, with a durable call ID and no automatic retry."""

from pathlib import Path

from worker.stages.lhm_recovery import atomic_json


def bounded_motion_result(
    function, destination, inputs, *, timeout, app_id=None, modal_profile=None, **kwargs
):
    """Bound result waiting (including queueing), not upload, cancellation, or billing.

    The caller owns a new output directory. An existing receipt is never resubmitted.
    Cancellation targets only the call we just spawned, without terminating shared containers.
    """
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("Motion result timeout must be a positive integer")
    path = Path(destination) / "motion-call.json"
    # Reserve before submission, including when a previous outcome is uncertain.
    with path.open("x") as stream:
        stream.write("{}\n")
    receipt = {
        "schema": "wander.motion-call/1",
        "status": "submitting",
        "appId": app_id,
        "modalProfile": modal_profile,
        "functionCallId": None,
        "resultTimeoutSeconds": timeout,
    }
    atomic_json(path, receipt)
    try:
        call = function.spawn(inputs, **kwargs)
    except BaseException:
        receipt["status"] = "submission_uncertain"
        atomic_json(path, receipt)
        raise
    receipt.update(functionCallId=call.object_id, status="waiting")
    try:
        atomic_json(path, receipt)
        result = call.get(timeout=timeout)
    except BaseException as error:
        receipt["status"] = (
            "wait_timeout" if isinstance(error, TimeoutError) else "wait_interrupted"
        )
        try:
            call.cancel()
            receipt["cancellation"] = "requested"
        except BaseException as cancellation_error:
            receipt["cancellation"] = "unknown"
            receipt["cancellationErrorType"] = type(cancellation_error).__name__
        atomic_json(path, receipt)
        if isinstance(error, TimeoutError):
            raise RuntimeError(
                f"Motion result wait timed out after {timeout}s; cancellation "
                f"{receipt['cancellation']}. Inspect {path} before any recovery; do not resubmit."
            ) from error
        raise
    receipt["status"] = "result_received"
    atomic_json(path, receipt)
    return result
