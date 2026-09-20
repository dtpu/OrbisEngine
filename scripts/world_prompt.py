#!/usr/bin/env python3
"""Write a source-grounded description to accompany Marble video or image inputs.

Video remains the default visual input. A description can state observed materials and constrain
unsupported additions, but does not pin visible geometry or prove that hallucination is reduced.
Unknown, occluded and unobserved regions must remain labeled as such. Review the sampled evidence
and the full cleaned clip before generation; a few prompt frames do not prove temporal coverage.

  world_prompt.py --clip public/clips/gym.mp4 --n 6 --out .context/prompt/gym.json

Writes {"text_prompt", "structured": {...}, "frames": [...]} plus sampled PNGs next to it.
One paid request is made per --out. `<out>.receipt.json` records the request identity, the raw
response digest and the reported token usage, and `<out>.response.raw` keeps the response bytes so
an interrupted run can finish without paying twice.
"""

import argparse
import base64
import errno
import fcntl
import hashlib
import json
import os
import re
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import certifi
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

BRIEF = """These images sample ONE continuous shot of a real place in time order. A generative
3D world model will receive the cleaned video, or explicitly selected still images. Write a factual
supporting description. The description and visual input do not guarantee correct geometry.

Describe only structure, fixtures and materials supported by the supplied images. Do not complete a
floor plan from assumptions or invent what lies behind the camera, beyond an occlusion, or through an
opening. Label ambiguous or unseen details as unknown; they are not evidence that a surface is absent.
Preserve observed details. Negative constraints must not ask to remove real recorded features.

Rules:
 - Describe visible materials, surfaces, fixtures and lighting without guessing hidden structure.
 - Describe what is actually visible through openings; otherwise say 'nothing visible'.
 - Do not invent additional rooms, corridors, furniture, signage, windows or doors.
 - If a surface is visibly a mirror, describe it as a flat mirror on a solid wall, not another room.
   If reflection versus opening is ambiguous, say so rather than guessing.
 - People are removed before world generation; do not request people or their body parts.
 - Do not add style, mood or camera adjectives such as 'cinematic', '8k' or 'beautiful'.
 - If sampled views conflict because an object moved or a door opened, record that uncertainty;
   do not blend incompatible fixture states into a confident description.

Answer with JSON and nothing else:
{"space": "<what kind of room or place, one clause>",
 "architecture": "<walls, ceiling, floor plan, columns, openings, their construction>",
 "materials": "<floor, wall, ceiling and fitting materials and colours, concretely>",
 "lighting": "<sources, colour, direction, and whether there is daylight; time of day>",
 "opens_onto": "<what is beyond the openings the camera can see into, or 'nothing visible'>",
 "contents": "<the furniture and equipment actually present, concretely>",
 "mirrors": "<mirrored or glazed surfaces present, or 'none'>",
 "negative_constraints": ["<each thing that must NOT be invented, as a short clause>"],
 "text_prompt": "<the final prompt, under 100 words, factual, ending with the negative constraints
                  as one sentence beginning 'Do not add'>"}
"""

MAX_COMPLETION_TOKENS = 4096
RECEIPT_SCHEMA = 1
REQUEST_TIMEOUT_SECONDS = 180
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
MAX_FRAME_EDGE = 1024

# CAP_PROP_POS_MSEC is whatever the container reports for the decoded frame. It is an approximate
# label for the operator, not a presentation timestamp that binds the frame to the audio timeline.
FRAME_TIME_NOTE = "container-reported CAP_PROP_POS_MSEC, approximate; not a PTS binding"

IDENTITY_MISMATCH = (
    "Completed source description does not match this source, frames, prompt, or request; "
    "a changed source needs a new --out"
)


def receipt_path(out):
    out = Path(out)
    return out.with_name(out.name + ".receipt.json")


def response_raw_path(out):
    out = Path(out)
    return out.with_name(out.name + ".response.raw")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path):
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


def _comparable_identity(identity):
    """Identity as compared: bytes, hashes, frame ordinals and the request block, never paths."""
    if not isinstance(identity, dict):
        return identity
    comparable = dict(identity)
    comparable["source"] = _without_path(identity.get("source"))
    images = identity.get("frameImages")
    if isinstance(images, list):
        comparable["frameImages"] = [_without_path(image) for image in images]
    return comparable


def _atomic_json(path, document):
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
def _prompt_lock(out):
    lock_path = receipt_path(out).with_name(receipt_path(out).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise ValueError(f"Refusing symlinked prompt lock: {lock_path}") from None
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _load_receipt(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source-description receipt is missing or unsafe; no request submitted")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Source-description receipt is unreadable; preserve it and do not resubmit"
        ) from error
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(
            "Source-description receipt is incompatible; preserve it and do not resubmit"
        )
    return receipt


def _request_body(model, paths, brief):
    content = [{"type": "text", "text": brief}]
    for image in paths:
        encoded = base64.b64encode(Path(image).read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
            }
        )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }


def _request_identity(clip, sampling, model, brief, body):
    request_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source": _file_identity(clip),
        "requestedFrames": [int(frame) for frame in sampling["requested"]],
        "selectedFrames": [int(frame) for frame in sampling["selected"]],
        "frameImages": [_file_identity(path) for path in sampling["paths"]],
        "request": {
            "sha256": hashlib.sha256(request_bytes).hexdigest(),
            "promptSha256": hashlib.sha256(brief.encode()).hexdigest(),
            "model": model,
            "maxCompletionTokens": MAX_COMPLETION_TOKENS,
        },
    }


def _prepared_request(clip, sampling, model, brief):
    if not sampling["paths"]:
        raise ValueError(f"No frames could be read from {clip}")
    body = _request_body(model, sampling["paths"], brief)
    return body, _request_identity(clip, sampling, model, brief, body)


def _frames_block(sampling):
    """Frame coverage as recorded: what was asked for, what decoded, and what did not."""
    return {
        "requestedFrames": [int(frame) for frame in sampling["requested"]],
        "selectedFrames": [int(frame) for frame in sampling["selected"]],
        "undecodedFrames": [int(frame) for frame in sampling["undecoded"]],
        "framePaths": list(sampling["paths"]),
        "frameTimeMs": list(sampling["containerTimeMs"]),
        "frameTimeNote": FRAME_TIME_NOTE,
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


def _parse_response(raw):
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
    if (
        not isinstance(record, dict)
        or not record
        or not isinstance(record.get("text_prompt"), str)
        or not record["text_prompt"].strip()
    ):
        raise ValueError("invalid source description")
    _validate_usage(response.get("usage"))
    return record


def _fail_receipt(path, receipt, status, category, message):
    failed = {**receipt, "status": status, "errorCategory": category}
    _atomic_json(path, failed)
    raise RuntimeError(message)


def _report_usage(response):
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return
    print(
        "source description billed: "
        f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
        f"total={usage.get('total_tokens')} response={response.get('id')}",
        file=sys.stderr,
    )


def _reuse_completed(out, receipt, identity):
    if _comparable_identity(receipt.get("identity")) != _comparable_identity(identity):
        raise ValueError(IDENTITY_MISMATCH)
    out = Path(out)
    if out.is_symlink() or not out.is_file():
        raise ValueError("Completed source-description output is missing or unsafe")
    if _without_path(receipt.get("output")) != _without_path(_file_identity(out)):
        raise ValueError("Completed source-description output was modified; refusing reuse")
    try:
        record = json.loads(out.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Completed source-description output is unreadable") from error
    if not isinstance(record, dict) or not isinstance(record.get("text_prompt"), str):
        raise TypeError("Completed source-description output is malformed")
    return record


def _finish(clip, out, attempt_path, received, raw, model, block):
    """Turn a persisted response into the prompt output, or a failed receipt that keeps it."""
    try:
        record = _parse_response(raw)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return _fail_receipt(
            attempt_path,
            received,
            "failed",
            "invalid_response",
            "OpenAI returned an invalid source description; preserve the receipt and do not retry",
        )

    record["clip"] = str(clip)
    record["model"] = model
    record["frames"] = list(block["selectedFrames"])
    record["requestedFrames"] = list(block["requestedFrames"])
    record["undecodedFrames"] = list(block["undecodedFrames"])
    record["framePaths"] = list(block["framePaths"])
    record["frameTimeMs"] = list(block["frameTimeMs"])
    record["frameTimeNote"] = block["frameTimeNote"]
    _atomic_json(out, record)
    completed = {**received, "status": "completed", "output": _file_identity(out)}
    _atomic_json(attempt_path, completed)
    _report_usage(received.get("response"))
    return record


def _recover_response(clip, out, attempt_path, receipt, identity, model, sampling):
    """Finish a crashed run from the response already paid for, without any network call."""
    if _comparable_identity(receipt.get("identity")) != _comparable_identity(identity):
        raise ValueError(IDENTITY_MISMATCH)
    recorded = receipt.get("response")
    raw_path = response_raw_path(out)
    if raw_path.is_symlink() or not raw_path.is_file() or not isinstance(recorded, dict):
        raise RuntimeError(
            "Source-description response bytes are missing; "
            "preserve the receipt and do not resubmit"
        )
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != recorded.get("sha256") or len(raw) != recorded.get(
        "bytes"
    ):
        raise RuntimeError(
            "Source-description response bytes do not match their receipt; "
            "preserve them and do not resubmit"
        )
    block = receipt.get("frames")
    if not isinstance(block, dict):
        block = _frames_block(sampling)
    return _finish(clip, out, attempt_path, receipt, raw, model, block)


def _submit(clip, out, attempt_path, body, identity, sampling, model, urlopen):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY unset; source description was not submitted")

    block = _frames_block(sampling)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "pending",
        "identity": identity,
        "frames": block,
    }
    _atomic_json(attempt_path, receipt)
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
        return _fail_http(attempt_path, receipt, error)
    except (TimeoutError, urllib.error.URLError):
        return _fail_receipt(
            attempt_path,
            receipt,
            "unknown",
            "transport",
            "OpenAI request outcome is unknown; preserve the receipt and do not retry",
        )
    # Provider transports can surface backend-specific exception classes. Keep the pending
    # receipt and replace every such detail with one stable, non-secret diagnostic.
    except Exception:  # noqa: BLE001
        return _fail_receipt(
            attempt_path,
            receipt,
            "unknown",
            "transport",
            "OpenAI request outcome is unknown; preserve the receipt and do not retry",
        )

    # The request is billed from here on, so persist the bytes and a receipt describing them
    # before any of their content is parsed or trusted.
    raw_path = response_raw_path(out)
    _atomic_bytes(raw_path, raw)
    summary = _response_summary(raw, status_code, response_request_id, raw_path.name)
    received = {**receipt, "status": "response_received", "response": summary}
    _atomic_json(attempt_path, received)
    return _finish(clip, out, attempt_path, received, raw, model, block)


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


def _fail_http(attempt_path, receipt, error):
    """A 4xx was rejected before work started; anything else leaves billing ambiguous."""
    status = int(error.code)
    rejected = 400 <= status < 500
    carried = {
        **receipt,
        "response": {
            "httpStatus": status,
            "requestId": _header_request_id(error.headers),
            **_provider_error_codes(error),
        },
    }
    if rejected:
        return _fail_receipt(
            attempt_path,
            carried,
            "failed",
            "http_rejected",
            f"OpenAI rejected the request with HTTP status {status}; no retry submitted",
        )
    return _fail_receipt(
        attempt_path,
        carried,
        "unknown",
        "http_server",
        f"OpenAI request outcome is unknown after HTTP status {status}; "
        "preserve the receipt and do not retry",
    )


def generate_prompt(clip, frames, n, out, model, *, brief=None, urlopen=None):
    """Generate exactly once, or safely reuse a completed output with matching identities.

    MAX_COMPLETION_TOKENS bounds the output tokens of this one request only. It is not an
    input-cost cap and not an account budget: the brief and the sampled images are still billed,
    and nothing here limits spending across runs. The operator must reconcile the token usage
    recorded in `<out>.receipt.json` into the spend ledger.

    An existing receipt is never replaced by a second request. `completed` reuses the saved
    output, `response_received` finishes from the response bytes already paid for, and every
    other state refuses until an operator reconciles it by hand.
    """
    clip = Path(clip)
    out = Path(out)
    brief = BRIEF if brief is None else brief
    urlopen = urllib.request.urlopen if urlopen is None else urlopen
    attempt_path = receipt_path(out)
    with _prompt_lock(out):
        receipt = None
        if attempt_path.exists() or attempt_path.is_symlink():
            receipt = _load_receipt(attempt_path)
            status = receipt.get("status")
            if status not in ("completed", "response_received"):
                raise RuntimeError(
                    f"Source-description receipt is {status or 'invalid'}; "
                    "preserve it and do not resubmit"
                )

        if receipt is None:
            if out.exists() or out.is_symlink():
                raise ValueError(
                    "Source-description output exists without a receipt; refusing overwrite"
                )
            sampling = sample(clip, frames, n, out.parent / (out.stem + "-frames"))
            body, identity = _prepared_request(clip, sampling, model, brief)
            return _submit(clip, out, attempt_path, body, identity, sampling, model, urlopen)

        # The recorded frames are evidence of the request that was already paid for. Re-sample
        # into a scratch directory so checking identity cannot rewrite or renumber them.
        with tempfile.TemporaryDirectory(prefix="world-prompt-check-") as scratch:
            sampling = sample(clip, frames, n, Path(scratch))
            _, identity = _prepared_request(clip, sampling, model, brief)
            if receipt.get("status") == "completed":
                return _reuse_completed(out, receipt, identity)
            return _recover_response(clip, out, attempt_path, receipt, identity, model, sampling)


def _write_frame(image, out_dir, ordinal):
    h, w = image.shape[:2]
    scale = MAX_FRAME_EDGE / max(w, h)
    if scale < 1:
        image = cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    path = Path(out_dir) / f"frame_{ordinal:04d}.png"
    cv2.imwrite(str(path), image)
    return str(path)


def sample(clip, frames, n, out_dir):
    """Decode the requested frame ordinals in stream order and report only what decoded.

    Ordinals count decoded frames from the start of the stream. Seeking with CAP_PROP_POS_FRAMES
    lands on keyframes in many containers, so frames are grabbed sequentially and only the
    requested ordinals are retrieved. Explicitly requested frames that do not decode are an error;
    the default spread proceeds and records the shortfall instead of reporting frames it never read.
    """
    explicit = bool(frames)
    out_dir = Path(out_dir)
    cap = cv2.VideoCapture(str(clip))
    try:
        if not explicit:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = np.linspace(0, max(total - 1, 0), n).round().astype(int).tolist()
        requested = [int(frame) for frame in frames]
        wanted = sorted({frame for frame in requested if frame >= 0})

        out_dir.mkdir(parents=True, exist_ok=True)
        selected, paths, times = [], [], []
        remaining = iter(wanted)
        target = next(remaining, None)
        ordinal = 0
        while target is not None:
            if not cap.grab():
                break
            if ordinal == target:
                position = cap.get(cv2.CAP_PROP_POS_MSEC)
                ok, image = cap.retrieve()
                if ok and image is not None:
                    paths.append(_write_frame(image, out_dir, ordinal))
                    selected.append(ordinal)
                    readable = isinstance(position, (int, float)) and position >= 0
                    times.append(round(float(position), 3) if readable else None)
                target = next(remaining, None)
            ordinal += 1
    finally:
        cap.release()

    decoded = set(selected)
    undecoded = [frame for frame in dict.fromkeys(requested) if frame not in decoded]
    if explicit and undecoded:
        raise ValueError(f"Requested frames {undecoded} could not be decoded from {clip}")
    return {
        "requested": requested,
        "selected": selected,
        "undecoded": undecoded,
        "paths": paths,
        "containerTimeMs": times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument(
        "--frames",
        default=None,
        help="comma-separated frame numbers (default: --n spread over the clip)",
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6-astra")
    a = ap.parse_args()

    out = Path(a.out)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else None
    try:
        rec = generate_prompt(Path(a.clip), frames, a.n, out, a.model)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
