#!/usr/bin/env python3
"""Exactly one paid vision request per `--out`, with a durable receipt that never pays twice.

This is the receipt lifecycle `scripts/world_prompt.py` already implements, lifted out so more
than one judge can share it. world_prompt.py predates this module and still carries its own copy;
it could be migrated onto `request_once` later without changing the files it writes, but it is
deliberately left alone here so a refactor cannot disturb a stage that already spends money.

The lifecycle, for one output path `out`:

  pending             written before the socket is opened, so an interrupted run is visible
  response_received   the raw bytes are on disk first, then the receipt that describes them
  completed           the validated record is on disk and its identity is recorded
  failed / unknown    terminal; a later run refuses instead of submitting a second request

`<out>.receipt.json` records the request identity, the raw response digest and the reported token
usage; `<out>.response.raw` keeps the exact bytes that were billed, so an interrupted run finishes
from them with no network call and no API key. A 4xx is a rejection (nothing was charged for
work), anything else leaves billing ambiguous and is recorded as `unknown`. Only enumerated,
identifier-shaped provider error codes reach the receipt, never provider free text, which can echo
the request back. Identities compare bytes and hashes, never absolute paths, so a receipt written
on another machine still reuses.

`max_completion_tokens` bounds the output tokens of one request. It is not an input-cost cap and
not an account budget: the brief and the images are billed too. Reconcile the usage recorded in
the receipt into the spend ledger by hand.
"""

import base64
import errno
import fcntl
import hashlib
import json
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import certifi

DEFAULT_MODEL = "gpt-6-astra"
MAX_COMPLETION_TOKENS = 4096
RECEIPT_SCHEMA = 1
REQUEST_TIMEOUT_SECONDS = 180
ENDPOINT = "https://api.openai.com/v1/chat/completions"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())

TERMINAL_STATES = ("completed", "response_received")
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def receipt_path(out):
    out = Path(out)
    return out.with_name(out.name + ".receipt.json")


def response_raw_path(out):
    out = Path(out)
    return out.with_name(out.name + ".response.raw")


def receipt_status(out):
    """The recorded status of `out`'s receipt, or None when no receipt exists.

    Callers use this to decide whether to rebuild their request images or reuse the ones already
    beside `out`: a run that has a receipt must present the same bytes it paid to have looked at.
    """
    path = receipt_path(out)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    return receipt.get("status") if isinstance(receipt, dict) else "unreadable"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _without_path(entry):
    """Drop the recorded absolute path, which is informational and machine specific."""
    if not isinstance(entry, dict):
        return entry
    return {key: value for key, value in entry.items() if key != "path"}


def _strip_paths(value):
    """Recursively drop `path` keys, so an identity compares bytes and hashes only."""
    if isinstance(value, dict):
        return {key: _strip_paths(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    return value


def _comparable_identity(identity):
    if not isinstance(identity, dict):
        return identity
    return _strip_paths(identity)


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked state file: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(document, handle, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_bytes(path, payload):
    """Persist response bytes before anything interprets them, owner readable only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked state file: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _request_lock(out):
    lock_path = receipt_path(out).with_name(receipt_path(out).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError(f"Refusing symlinked request lock: {lock_path}") from None
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _load_receipt(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} receipt is missing or unsafe; no request submitted")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} receipt is unreadable; preserve it and do not resubmit") from (
            error
        )
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"{label} receipt is incompatible; preserve it and do not resubmit")
    return receipt


def _media_type(path):
    suffix = Path(path).suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise ValueError(f"Unsupported image type for a vision request: {path}")
    return MEDIA_TYPES[suffix]


def _request_body(model, images, brief, max_completion_tokens):
    content = [{"type": "text", "text": brief}]
    for image in images:
        encoded = base64.b64encode(Path(image).read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_media_type(image)};base64,{encoded}",
                    "detail": "high",
                },
            }
        )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_completion_tokens,
    }


def _request_identity(images, model, brief, body, identity_extra, max_completion_tokens):
    request_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {
        "images": [file_identity(path) for path in images],
        "inputs": dict(identity_extra or {}),
        "request": {
            "sha256": hashlib.sha256(request_bytes).hexdigest(),
            "promptSha256": hashlib.sha256(brief.encode()).hexdigest(),
            "model": model,
            "maxCompletionTokens": max_completion_tokens,
        },
    }


def _validate_usage(usage):
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(usage, dict) or any(
        not isinstance(usage.get(key), int) or usage[key] < 0 for key in required
    ):
        raise ValueError("missing token usage")
    return usage


def _optional_string(value):
    return value if isinstance(value, str) else None


def _header_request_id(headers):
    getter = getattr(headers, "get", None)
    return _optional_string(getter("x-request-id")) if callable(getter) else None


def _response_summary(raw, status_code, request_id, raw_name):
    """Describe the billed response before any of its content is trusted.

    Everything past the digest is best effort: a malformed or truncated body still leaves the
    operator the status, the provider request id and the exact bytes that were charged for.
    """
    summary = {
        "httpStatus": int(status_code),
        "requestId": _optional_string(request_id),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rawFile": raw_name,
        "id": None,
        "model": None,
        "finishReason": None,
        "usage": None,
    }
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return summary
    if not isinstance(document, dict):
        return summary
    summary["id"] = _optional_string(document.get("id"))
    summary["model"] = _optional_string(document.get("model"))
    choices = document.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    if isinstance(first, dict):
        summary["finishReason"] = _optional_string(first.get("finish_reason"))
    try:
        summary["usage"] = _validate_usage(document.get("usage"))
    except ValueError:
        summary["usage"] = None
    return summary


def _parse_response(raw, validate):
    """The model's JSON object, put through the caller's validator before it is written."""
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise TypeError("invalid response object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty completion")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    record = json.loads(text)
    if not isinstance(record, dict) or not record:
        raise ValueError("invalid record")
    _validate_usage(response.get("usage"))
    checked = validate(record)
    if not isinstance(checked, dict) or not checked:
        raise TypeError("validator returned no record")
    return checked


def _fail_receipt(path, receipt, status, category, message):
    failed = {**receipt, "status": status, "errorCategory": category}
    atomic_json(path, failed)
    raise RuntimeError(message)


def usage_line(label, response):
    """One non-secret line naming what this request cost, or None when nothing was reported."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    return (
        f"{label} billed: prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')} total={usage.get('total_tokens')} "
        f"response={response.get('id')}"
    )


def _reuse_completed(out, receipt, identity, label):
    if _comparable_identity(receipt.get("identity")) != _comparable_identity(identity):
        raise ValueError(
            f"Completed {label} does not match this input, images, prompt or request; "
            "a changed input needs a new --out"
        )
    out = Path(out)
    if out.is_symlink() or not out.is_file():
        raise ValueError(f"Completed {label} output is missing or unsafe")
    if _without_path(receipt.get("output")) != _without_path(file_identity(out)):
        raise ValueError(f"Completed {label} output was modified; refusing reuse")
    try:
        record = json.loads(out.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Completed {label} output is unreadable") from error
    if not isinstance(record, dict) or not record:
        raise TypeError(f"Completed {label} output is malformed")
    return record


def _finish(out, attempt_path, received, raw, validate, label, report):
    """Turn a persisted response into the output, or a failed receipt that keeps it."""
    try:
        record = _parse_response(raw, validate)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return _fail_receipt(
            attempt_path,
            received,
            "failed",
            "invalid_response",
            f"OpenAI returned an unusable {label}; preserve the receipt and do not retry",
        )

    atomic_json(out, record)
    completed = {**received, "status": "completed", "output": file_identity(out)}
    atomic_json(attempt_path, completed)
    if report is not None:
        line = usage_line(label, received.get("response"))
        if line:
            report(line)
    return record


def _recover_response(out, attempt_path, receipt, identity, validate, label, report):
    """Finish a crashed run from the response already paid for, without any network call."""
    if _comparable_identity(receipt.get("identity")) != _comparable_identity(identity):
        raise ValueError(
            f"Recorded {label} does not match this input, images, prompt or request; "
            "a changed input needs a new --out"
        )
    recorded = receipt.get("response")
    raw_path = response_raw_path(out)
    if raw_path.is_symlink() or not raw_path.is_file() or not isinstance(recorded, dict):
        raise RuntimeError(
            f"{label} response bytes are missing; preserve the receipt and do not resubmit"
        )
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != recorded.get("sha256") or len(raw) != recorded.get(
        "bytes"
    ):
        raise RuntimeError(
            f"{label} response bytes do not match their receipt; preserve them and do not resubmit"
        )
    return _finish(out, attempt_path, receipt, raw, validate, label, report)


def _provider_error_codes(error):
    """The provider's enumerated error code and type, never its free-text message.

    `insufficient_quota` and `rate_limit_exceeded` are both HTTP 429 and need different operator
    action. Only short identifier-shaped values are kept, so nothing echoed from the request
    (a key fragment in a message, say) can reach the receipt.
    """
    try:
        document = json.loads(error.read(65536))
    except Exception:  # noqa: BLE001
        return {}
    detail = document.get("error") if isinstance(document, dict) else None
    if not isinstance(detail, dict):
        return {}
    codes = {}
    for key, name in (("code", "errorCode"), ("type", "errorType")):
        value = detail.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
            codes[name] = value
    return codes


def _fail_http(attempt_path, receipt, error, label):
    """A 4xx was rejected before work started; anything else leaves billing ambiguous."""
    status = int(error.code)
    carried = {
        **receipt,
        "response": {
            "httpStatus": status,
            "requestId": _header_request_id(error.headers),
            **_provider_error_codes(error),
        },
    }
    if 400 <= status < 500:
        return _fail_receipt(
            attempt_path,
            carried,
            "failed",
            "http_rejected",
            f"OpenAI rejected the {label} request with HTTP status {status}; no retry submitted",
        )
    return _fail_receipt(
        attempt_path,
        carried,
        "unknown",
        "http_server",
        f"OpenAI {label} request outcome is unknown after HTTP status {status}; "
        "preserve the receipt and do not retry",
    )


def _submit(out, attempt_path, body, identity, validate, label, urlopen, report):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY unset; the {label} request was not submitted")

    receipt = {"schema": RECEIPT_SCHEMA, "status": "pending", "identity": identity}
    atomic_json(attempt_path, receipt)
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        response_handle = urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS, context=TLS_CONTEXT)
        with response_handle as response_stream:
            status_code = getattr(response_stream, "status", 200)
            if not 200 <= status_code < 300:
                raise urllib.error.HTTPError(
                    request.full_url,
                    status_code,
                    "request rejected",
                    response_stream.headers,
                    None,
                )
            raw = response_stream.read()
            response_request_id = _header_request_id(response_stream.headers)
    except urllib.error.HTTPError as error:
        return _fail_http(attempt_path, receipt, error, label)
    except (TimeoutError, urllib.error.URLError):
        return _fail_receipt(
            attempt_path,
            receipt,
            "unknown",
            "transport",
            f"OpenAI {label} request outcome is unknown; preserve the receipt and do not retry",
        )
    # Provider transports can surface backend-specific exception classes. Keep the pending
    # receipt and replace every such detail with one stable, non-secret diagnostic.
    except Exception:  # noqa: BLE001
        return _fail_receipt(
            attempt_path,
            receipt,
            "unknown",
            "transport",
            f"OpenAI {label} request outcome is unknown; preserve the receipt and do not retry",
        )

    # The request is billed from here on, so persist the bytes and a receipt describing them
    # before any of their content is parsed or trusted.
    raw_path = response_raw_path(out)
    _atomic_bytes(raw_path, raw)
    summary = _response_summary(raw, status_code, response_request_id, raw_path.name)
    received = {**receipt, "status": "response_received", "response": summary}
    atomic_json(attempt_path, received)
    return _finish(out, attempt_path, received, raw, validate, label, report)


def request_once(
    out,
    *,
    brief,
    images,
    model=DEFAULT_MODEL,
    validate,
    identity_extra=None,
    max_completion_tokens=MAX_COMPLETION_TOKENS,
    urlopen=None,
    label="judgement",
    report=None,
):
    """Submit one request for `out`, or safely reuse what an earlier run already paid for.

    `validate(record) -> record` receives the JSON object the model answered with and returns the
    document to write; raising ValueError, TypeError or KeyError records a `failed` receipt that
    keeps the billed response block and writes no output. `identity_extra` is recorded and
    compared verbatim (minus any `path` keys), so a caller can bind the receipt to the hashes of
    its own inputs rather than to the request images alone.

    An existing receipt is never replaced by a second request. `completed` reuses the saved
    output, `response_received` finishes from the response bytes already paid for, and every
    other state refuses until an operator reconciles it by hand.
    """
    out = Path(out)
    images = [Path(image) for image in images]
    if not images:
        raise ValueError(f"No images were built for the {label} request")
    urlopen = urllib.request.urlopen if urlopen is None else urlopen
    attempt_path = receipt_path(out)

    with _request_lock(out):
        receipt = None
        if attempt_path.exists() or attempt_path.is_symlink():
            receipt = _load_receipt(attempt_path, label.capitalize())
            status = receipt.get("status")
            if status not in TERMINAL_STATES:
                raise RuntimeError(
                    f"{label.capitalize()} receipt is {status or 'invalid'}; "
                    "preserve it and do not resubmit"
                )

        body = _request_body(model, images, brief, max_completion_tokens)
        identity = _request_identity(
            images, model, brief, body, identity_extra, max_completion_tokens
        )

        if receipt is None:
            if out.exists() or out.is_symlink():
                raise ValueError(
                    f"{label.capitalize()} output exists without a receipt; refusing overwrite"
                )
            return _submit(out, attempt_path, body, identity, validate, label, urlopen, report)

        if receipt.get("status") == "completed":
            return _reuse_completed(out, receipt, identity, label)
        return _recover_response(out, attempt_path, receipt, identity, validate, label, report)
