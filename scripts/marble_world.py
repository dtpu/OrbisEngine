#!/usr/bin/env python3
"""One Marble client for image, multi-image reconstruction, and video inputs.

  uv run --locked python scripts/marble_world.py image submit clean.png --name example --marble-dir .context/marble
  uv run --locked python scripts/marble_world.py multi submit --images a.png:0 b.png:90 --prompt-file prompt.json \
      --name example --marble-dir .context/marble
  uv run --locked python scripts/marble_world.py video submit clean.mp4 --name example --marble-dir .context/marble
  uv run --locked python scripts/marble_world.py image poll OPERATION_ID --name example --marble-dir .context/marble
  uv run --locked python scripts/marble_world.py video fetch WORLD_ID --name example --marble-dir .context/marble

Only submit uploads inputs and generates a world. Poll/fetch recover existing results without
spending generation credits. The input type preserves each mode's payload and ops-log format;
Video accepts an optional source description and does not send image-only recaptioning fields.
All submissions pin their model and request private world permissions.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.worldlabs.ai"
UPLOAD_LIMIT = 104857600
DOWNLOAD_TIMEOUT = 600


def safe_endpoint(url):
    """Return the non-secret portion of an HTTP destination for diagnostics."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "unknown-host"
    # Do not turn an unexpected URL (which may contain encoded credentials) into a log entry.
    if not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host):
        host = "unknown-host"
    path = parsed.path if re.fullmatch(r"/[A-Za-z0-9._~/-]{0,512}", parsed.path) else "/"
    return f"{host}{path}"


def write_json(path, value):
    """Persist recovery metadata atomically, before progressing to the next remote action."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def submission_history(directory, name, ops=None):
    """Inspect persisted receipts and legacy outputs without contacting the API."""
    directory = Path(directory)
    for filename in (f"{name}-world.json", f"{name}.spz"):
        if (directory / filename).exists():
            return "world", str(directory / filename)
    operation_ids = set()
    attempted = False
    world_id = None
    for suffix in ("generation", "op"):
        path = directory / f"{name}-{suffix}.json"
        if not path.exists():
            continue
        attempted = True
        try:
            doc = json.loads(path.read_text())
            operation_id = doc.get("operation_id")
            if isinstance(operation_id, str) and operation_id:
                operation_ids.add(operation_id)
            if suffix == "op" and doc.get("done") and not doc.get("error"):
                world_id = (doc.get("metadata") or {}).get("world_id")
        except (ValueError, AttributeError) as error:
            raise ValueError(
                f"Unreadable generation history {path}; inspect it before recovery"
            ) from error
    path = directory / f"{ops or name}-ops.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if ops and ops != name and f"[{name}] " not in line:
                continue
            attempted = True
            match = re.search(r"\bop (\S+) (?:submitted|recovery:)", line)
            if match:
                operation_ids.add(match[1])
    attempted |= (directory / f"{name}-request.json").exists()
    if len(operation_ids) > 1:
        raise ValueError(
            f"Multiple operations recorded for {name}; recover an explicit ID, never resubmit"
        )
    if operation_ids:
        return "operation", operation_ids.pop()
    if isinstance(world_id, str) and world_id:
        return "completed", world_id
    return ("uncertain", name) if attempted else ("new", None)


def recover_submission(a):
    kind, value = submission_history(a.marble_dir, a.name, getattr(a, "ops", None))
    if kind == "new":
        return False
    if kind == "world":
        raise ValueError(
            f"Existing world metadata {value}; refusing another generation. "
            "Use --reuse-world WORLD_ID in run_clip.py, or fetch WORLD_ID here."
        )
    if kind == "operation":
        log(a, f"op {value} recovery: polling the recorded attempt, never resubmitting")
        poll(a, value)
    elif kind == "completed":
        fetch(a, value)
    else:
        raise ValueError(
            f"A prior attempt for {a.name} has no recoverable operation ID; its charge status "
            "is unknown. Check the provider account and recover with poll/fetch. "
            "Do not delete the receipt or resubmit."
        )
    return True


def claim_submission(a):
    """An exclusive, durable receipt prevents competing processes from generating twice."""
    directory = Path(a.marble_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{a.name}-generation.json"
    receipt = {
        "schema": "wander.marble-attempt/1",
        "name": a.name,
        "input_type": a.input_type,
        "status": "preparing",
    }
    try:
        with path.open("x") as handle:
            json.dump(receipt, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(directory)
    except FileExistsError as error:
        raise ValueError(
            f"Another attempt already reserved {a.name}; rerun to inspect recovery, never resubmit"
        ) from error
    return path, receipt


def call(method, path, body=None, raw=None, headers=None, timeout=120):
    url = BASE + path if path.startswith("/") else path
    parsed = urllib.parse.urlsplit(url)
    is_api = parsed.scheme == "https" and parsed.netloc == "api.worldlabs.ai"
    h = {}
    if is_api:
        api_key = os.environ.get("WLT_API_KEY")
        if not api_key:
            sys.exit("set WLT_API_KEY")
        h["WLT-Api-Key"] = api_key
    if body is not None:
        raw = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    if not is_api:
        # Presigned media uploads authenticate through their URL, not the API key.
        h = {k: v for k, v in h.items() if k.lower() != "wlt-api-key"}
    # urllib omits unredirected headers when following redirects, including other origins.
    credential = next((v for k, v in h.items() if k.lower() == "wlt-api-key"), None)
    h = {k: v for k, v in h.items() if k.lower() != "wlt-api-key"}
    request = urllib.request.Request(url, data=raw, method=method, headers=h)
    if credential is not None:
        request.add_unredirected_header("WLT-Api-Key", credential)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            return response.status, (
                json.loads(data)
                if data and response.headers.get("Content-Type", "").startswith("application/json")
                else data
            )
    except urllib.error.HTTPError as error:
        # Provider responses and presigned URLs can contain credentials.  Keep failure receipts
        # useful without copying either into stdout, logs, or an outer pipeline log.
        sys.exit(f"{method} {safe_endpoint(url)} -> HTTP {error.code}")
    except urllib.error.URLError:
        sys.exit(f"{method} {safe_endpoint(url)} -> network error")


def log(a, line):
    directory = Path(a.marble_dir)
    directory.mkdir(parents=True, exist_ok=True)
    video = a.input_type == "video"
    stem = a.name if video else (a.ops or a.name)
    with (directory / f"{stem}-ops.txt").open("a") as handle:
        handle.write(
            time.strftime("%Y-%m-%d %H:%M:%S ") + ("" if video else f"[{a.name}] ") + line + "\n"
        )
    print(line, flush=True)


def download(url, dest, require_gzip=False):
    """Atomically download an asset, rejecting a short response before it becomes visible."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}-", suffix=".part", dir=dest.parent)
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response,
        ):
            length = response.headers.get("Content-Length")
            expected = int(length) if length is not None else None
            if expected is not None and expected < 0:
                raise ValueError("download supplied an invalid Content-Length")
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if expected is not None and size != expected:
            raise ValueError(f"download length mismatch (expected {expected}, received {size})")
        if not size:
            raise ValueError("download was empty")
        if require_gzip:
            with open(temporary, "rb") as handle:
                if handle.read(2) != b"\x1f\x8b":
                    raise ValueError("SPZ download does not have a gzip header")
        os.replace(temporary, dest)
        sync_directory(dest.parent)
        return size, digest.hexdigest()
    except urllib.error.HTTPError as error:
        raise ValueError(f"download {safe_endpoint(url)} -> HTTP {error.code}") from None
    except urllib.error.URLError:
        raise ValueError(f"download {safe_endpoint(url)} -> network error") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def checked_world(response):
    """Accept current wrapped and legacy flat responses before replacing local metadata."""
    world = (
        response.get("world") if isinstance(response, dict) and "world" in response else response
    )
    if not isinstance(world, dict):
        raise ValueError("world response is not an object")  # noqa: TRY004 - CLI validation error
    assets = world.get("assets")
    splats = assets.get("splats") if isinstance(assets, dict) else None
    urls = splats.get("spz_urls") if isinstance(splats, dict) else None
    full_res = urls.get("full_res") if isinstance(urls, dict) else None
    if not isinstance(full_res, str) or not full_res:
        raise ValueError("world response has no full-resolution SPZ URL")
    thumbnail = assets.get("thumbnail_url")
    if thumbnail is not None and not isinstance(thumbnail, str):
        raise ValueError("world response has an invalid thumbnail URL")
    return world


def write_spz_receipt(directory, name, path, size, sha256):
    write_json(
        Path(directory) / f"{name}-spz-receipt.json",
        {
            "schema": "wander.marble-spz/1",
            "path": str(Path(path).resolve()),
            "bytes": size,
            "sha256": sha256,
        },
    )


def matching_spz_receipt(directory, name, path):
    """A receipt verifies only the exact bytes downloaded by this client, never legacy files."""
    path = Path(path)
    receipt_path = Path(directory) / f"{name}-spz-receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("path") != str(path.resolve())
            or receipt.get("bytes") != path.stat().st_size
        ):
            return False
        with path.open("rb") as handle:
            return receipt.get("sha256") == hashlib.file_digest(handle, "sha256").hexdigest()
    except (OSError, ValueError, AttributeError):
        return False


def atomic_copy(source, dest):
    """Make the convenience copy without exposing a partial file to recovery."""
    source, dest = Path(source), Path(dest)
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}-", suffix=".part", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as handle, source.open("rb") as input_handle:
            while chunk := input_handle.read(1024 * 1024):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, dest)
        sync_directory(dest.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch(a, world_id, prefix=""):
    """Read existing metadata/assets only. Existing asset files are retained on recovery."""
    directory = Path(a.marble_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _, response = call("GET", f"/marble/v1/worlds/{world_id}")
    world = checked_world(response)
    write_json(directory / f"{a.name}-world.json", world)
    assets = world["assets"]
    semantics = assets["splats"].get("semantics_metadata") or {}
    world_url = f" {world.get('world_marble_url')}" if a.input_type == "video" else ""
    log(
        a,
        f"{prefix}world {world_id}{world_url} metric_scale_factor {semantics.get('metric_scale_factor')} ground_plane_offset {semantics.get('ground_plane_offset')}",
    )
    if a.spz and not Path(a.spz).exists():
        size, sha256 = download(assets["splats"]["spz_urls"]["full_res"], a.spz, require_gzip=True)
        write_spz_receipt(directory, a.name, a.spz, size, sha256)
        log(a, f"spz full_res {size} bytes -> {a.spz}")
        copy = directory / f"{a.name}.spz"
        if Path(a.spz).resolve() != copy.resolve() and not copy.exists():
            atomic_copy(a.spz, copy)
            log(a, f"copy -> {copy}")
        elif Path(a.spz).resolve() != copy.resolve():
            log(a, f"existing SPZ retained without replacement -> {copy}")
    elif a.spz:
        state = (
            "verified by its download receipt"
            if matching_spz_receipt(directory, a.name, a.spz)
            else "not verified or replaced"
        )
        log(a, f"existing SPZ retained; {state} -> {a.spz}")
        if state == "not verified or replaced":
            raise ValueError(
                "Existing SPZ lacks a matching integrity receipt; preserve and inspect it, "
                "or fetch the same world to a fresh destination. No generation is needed."
            )
    if a.thumb and assets.get("thumbnail_url") and not Path(a.thumb).exists():
        size, _ = download(assets["thumbnail_url"], a.thumb)
        log(a, f"thumbnail {size} bytes -> {a.thumb}")


def poll(a, operation_id):
    directory = Path(a.marble_dir)
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        _, operation = call("GET", f"/marble/v1/operations/{operation_id}")
        write_json(directory / f"{a.name}-op.json", {**operation, "operation_id": operation_id})
        progress = (operation.get("metadata") or {}).get("progress") or {}
        print(
            time.strftime("%H:%M:%S"),
            progress.get("status"),
            progress.get("description"),
            flush=True,
        )
        if operation.get("done"):
            break
        time.sleep(a.interval)
    if operation.get("error"):
        log(a, f"op {operation_id} FAILED: {operation['error']}")
        sys.exit(1)
    fetch(a, operation["metadata"]["world_id"], f"op {operation_id} done: ")


def inputs(a):
    """Validate every local input before the first remote call (including later multi views)."""
    prompt = getattr(a, "prompt", None)
    if a.input_type == "multi":
        if not 1 <= len(a.images) <= 8:
            raise ValueError("multi submit needs 1–8 --images")
        items = []
        for spec in a.images:
            path, _, azimuth = spec.partition(":")
            angle = float(azimuth) if azimuth else None
            if angle is not None and not math.isfinite(angle):
                raise ValueError("azimuth must be finite")
            items.append((Path(path), angle))
    else:
        if not a.target:
            raise ValueError(f"{a.input_type} submit needs an input file")
        items = [(Path(a.target), None)]
    if getattr(a, "prompt_file", None):
        prompt = json.loads(Path(a.prompt_file).read_text())["text_prompt"]
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("text_prompt must be a string")
    for path, _ in items:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"input is missing or empty: {path}")
        if path.stat().st_size > UPLOAD_LIMIT:
            raise ValueError(f"input over 100 MB: {path}")
        # Readability is checked for every view before any uploads begin.
        with path.open("rb") as handle:
            handle.read(1)
        if a.input_type == "multi" and not path.suffix:
            raise ValueError(f"image needs a file extension: {path}")
    return items, prompt


def upload(a, path, identity):
    data = path.read_bytes()
    if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
        raise ValueError("Input changed after its submission receipt; inspect before retrying")
    kind = "video" if a.input_type == "video" else "image"
    extension = (
        path.suffix.lstrip(".").lower()
        if a.input_type == "multi"
        else ("mp4" if kind == "video" else "png")
    )
    _, prepared = call(
        "POST",
        "/marble/v1/media-assets:prepare_upload",
        {"file_name": path.name, "kind": kind, "extension": extension},
    )
    asset = prepared["media_asset"]
    asset_id = asset.get("id") or asset.get("media_asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError("Upload preparation returned no media asset ID")
    upload_info = prepared["upload_info"]
    content_type = f"{kind}/{'jpeg' if extension in ('jpg', 'jpeg') else extension}"
    required = upload_info.get("required_headers")
    headers = {
        "Content-Type": content_type,
        **(required if required is not None else {"x-goog-content-length-range": "0,104857600"}),
    }
    status, _ = call(
        upload_info.get("upload_method", "PUT"),
        upload_info["upload_url"],
        raw=data,
        headers=headers,
        timeout=600,
    )
    if a.input_type == "multi":
        log(
            a, f"image {path.name} ({path.stat().st_size} bytes, asset {asset_id}, upload {status})"
        )
    else:
        note = (" " + a.note) if a.input_type == "image" and a.note else ""
        log(
            a,
            f"{kind} (input {path}, {path.stat().st_size} bytes, asset {asset_id} upload {status}){note}",
        )
    return asset_id


def submit(a, poll_after=True):
    if recover_submission(a):
        return
    if a.spz and Path(a.spz).exists():
        raise ValueError(
            "Output SPZ already exists; use --reuse-world or fetch, never generate over it"
        )
    items, prompt = inputs(a)
    receipt_path, receipt = claim_submission(a)
    receipt["inputs"] = []
    for path, angle in items:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        receipt["inputs"].append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "azimuth": angle,
            }
        )
    write_json(receipt_path, receipt)
    entries = []
    for (path, angle), identity in zip(items, receipt["inputs"]):
        asset_id = upload(a, path, identity)
        entry = {"content": {"source": "media_asset", "media_asset_id": asset_id}}
        if angle is not None:
            entry["azimuth"] = angle
        entries.append(entry)
    if a.input_type == "video":
        body = {
            "model": a.model,
            "world_prompt": {"type": "video", "video_prompt": entries[0]["content"]},
        }
        if prompt is not None:
            body["world_prompt"]["text_prompt"] = prompt
    else:
        world_prompt = (
            {
                "type": "image",
                "image_prompt": entries[0]["content"],
                "is_pano": False,
                "disable_recaption": True,
                "text_prompt": prompt,
            }
            if a.input_type == "image"
            else {
                "type": "multi-image",
                "reconstruct_images": True,
                "disable_recaption": True,
                "text_prompt": prompt,
                "multi_image_prompt": entries,
            }
        )
        body = {"display_name": a.name, "model": a.model, "world_prompt": world_prompt}
        if a.seed is not None:
            body["seed"] = a.seed
    body["permission"] = {"public": False}
    write_json(Path(a.marble_dir) / f"{a.name}-request.json", body)
    receipt.update(
        status="submitting",
        request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
    )
    write_json(receipt_path, receipt)
    _, operation = call("POST", "/marble/v1/worlds:generate", body)
    operation_id = operation["operation_id"]
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError(
            "Generation returned no operation ID; keep the receipt and inspect the provider account"
        )
    receipt.update(status="submitted", operation_id=operation_id)
    write_json(receipt_path, receipt)
    if a.input_type == "video":
        detail = f"asset {asset_id}"
    elif a.input_type == "image":
        detail = f"asset {asset_id}, {a.model}, seed {a.seed}, disable_recaption"
    else:
        detail = (
            f"{len(items)} images, {a.model}, seed {a.seed}, reconstruct_images, disable_recaption"
        )
    log(a, f"op {operation_id} submitted ({detail})")
    if a.input_type == "multi":
        log(a, f"prompt: {prompt}")
    if poll_after:
        poll(a, operation_id)


def parser():
    root = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = root.add_subparsers(dest="input_type", required=True)
    for input_type in ("image", "multi", "video"):
        ap = modes.add_parser(input_type)
        ap.add_argument("mode", choices=["submit", "submit-only", "poll", "fetch"])
        ap.add_argument(
            "target",
            nargs="?",
            help="input file (image/video submit), operation id (poll), or world id (fetch)",
        )
        ap.add_argument("--name", required=True)
        ap.add_argument("--marble-dir", required=True)
        ap.add_argument("--spz")
        ap.add_argument("--thumb")
        ap.add_argument("--interval", type=int, default=60)
        ap.add_argument("--model", default="marble-1.1")
        ap.add_argument("--prompt", default="" if input_type == "image" else None)
        ap.add_argument("--prompt-file", help="JSON from scripts/world_prompt.py")
        if input_type != "video":
            ap.add_argument("--ops", help="ops log stem (default name)")
            ap.add_argument("--seed", type=int)
            ap.add_argument("--note", default="")
        if input_type == "multi":
            ap.add_argument("--images", nargs="+", default=[], help="path[:azimuth_deg] per view")
    return root


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.interval < 0 or (a.mode not in {"submit", "submit-only"} and not a.target):
        ap.error("poll/fetch require an existing ID; interval must be nonnegative")
    try:
        if a.mode == "submit":
            submit(a)
        elif a.mode == "submit-only":
            submit(a, poll_after=False)
        elif a.mode == "poll":
            poll(a, a.target)
        else:
            fetch(a, a.target)
    except (ValueError, KeyError, OSError) as error:
        ap.exit(1, f"Marble {a.mode} failed: {error}\n")


if __name__ == "__main__":
    main()
