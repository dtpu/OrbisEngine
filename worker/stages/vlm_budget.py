"""Opt-in, local, never-refunded request allowance for a bounded VLM run."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


class BudgetError(ValueError):
    pass


PROVIDER_ERROR_CODES = frozenset(
    (
        "insufficient_quota",
        "model_not_found",
        "invalid_api_key",
        "rate_limit_exceeded",
        "unsupported_parameter",
        "invalid_request_error",
        "invalid_value",
        "context_length_exceeded",
    )
)
PROVIDER_ERROR_PARAMS = frozenset(
    (
        "model",
        "messages",
        "max_completion_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "response_format",
        "reasoning_effort",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "n",
        "stream",
    )
)


def safe_error_details(details):
    """Keep only finite diagnostic identifiers, never provider messages or arbitrary strings."""
    if not isinstance(details, dict):
        return {}
    safe = {}
    status = details.get("httpStatus")
    if type(status) is int and 100 <= status <= 599:
        safe["httpStatus"] = status
    code = details.get("providerCode")
    if isinstance(code, str) and code in PROVIDER_ERROR_CODES:
        safe["providerCode"] = code
    param = details.get("providerParam")
    if isinstance(param, str) and param in PROVIDER_ERROR_PARAMS:
        safe["providerParam"] = param
    return safe


def _validate(data):
    if (
        not isinstance(data, dict)
        or data.get("schema") != "wander.vlm-budget/1"
        or type(data.get("maxCalls")) is not int
        or data["maxCalls"] <= 0
        or type(data.get("maxCompletionTokens")) is not int
        or data["maxCompletionTokens"] <= 0
        or not isinstance(data.get("allowedModels"), list)
        or not data["allowedModels"]
        or any(not isinstance(m, str) or not m.strip() for m in data["allowedModels"])
        or not isinstance(data.get("calls"), list)
    ):
        raise BudgetError("invalid VLM budget")
    hashes = set()
    for call in data["calls"]:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("bodyHash"), str)
            or len(call["bodyHash"]) != 64
            or any(c not in "0123456789abcdef" for c in call["bodyHash"])
            or call["bodyHash"] in hashes
            or call.get("status") not in ("claimed", "completed", "error")
        ):
            raise BudgetError("invalid VLM budget history")
        hashes.add(call["bodyHash"])
    return data


@contextmanager
def _locked(path):
    # Lock a stable sibling inode: replacing the JSON must not replace the lock.
    with open(str(path) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield _validate(json.loads(path.read_text()))


def _save(path, data):
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
            name = out.name
            json.dump(data, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def claim(path, body):
    """Persist a claim before any network action; also apply the completion token cap."""
    path = Path(path).resolve(strict=True)
    with _locked(path) as data:
        if body["model"] not in data["allowedModels"]:
            raise BudgetError("VLM model is not allowed")
        body["max_completion_tokens"] = data["maxCompletionTokens"]
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if any(c["bodyHash"] == digest for c in data["calls"]):
            raise BudgetError("duplicate VLM request denied")
        if len(data["calls"]) >= data["maxCalls"]:
            raise BudgetError("VLM request allowance exhausted")
        data["calls"].append(
            {
                "bodyHash": digest,
                "model": body["model"],
                "status": "claimed",
                "claimedAt": time.time(),
            }
        )
        _save(path, data)
    return path, digest


def finish(receipt, status, usage=None, error=None, error_details=None):
    path, digest = receipt
    with _locked(path) as data:
        call = next(c for c in data["calls"] if c["bodyHash"] == digest)
        call["status"] = status
        call["finishedAt"] = time.time()
        # Allowlist numeric counters only: provider response/error text can contain inputs.
        if isinstance(usage, dict):
            call["usage"] = {
                key: value
                for key, value in usage.items()
                if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                and type(value) is int
                and value >= 0
            }
        if error is not None:
            call["errorType"] = type(error).__name__
            call.update(safe_error_details(error_details))
        _save(path, data)
