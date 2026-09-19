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
video intentionally does not send image-only display_name/seed/prompt fields.
All submissions pin their model and request private world permissions.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.worldlabs.ai"
UPLOAD_LIMIT = 104857600


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
    request = urllib.request.Request(url, data=raw, method=method, headers=h)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            return response.status, (
                json.loads(data)
                if data and response.headers.get("Content-Type", "").startswith("application/json")
                else data
            )
    except urllib.error.HTTPError as error:
        sys.exit(f"{method} {path} -> {error.code}: {error.read().decode(errors='replace')[:2000]}")


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


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return dest.stat().st_size


def fetch(a, world_id, prefix=""):
    """Read existing metadata/assets only. Existing asset files are retained on recovery."""
    directory = Path(a.marble_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _, world = call("GET", f"/marble/v1/worlds/{world_id}")
    (directory / f"{a.name}-world.json").write_text(json.dumps(world, indent=1))
    assets = world["assets"]
    semantics = assets["splats"].get("semantics_metadata") or {}
    world_url = f" {world.get('world_marble_url')}" if a.input_type == "video" else ""
    log(
        a,
        f"{prefix}world {world_id}{world_url} metric_scale_factor {semantics.get('metric_scale_factor')} ground_plane_offset {semantics.get('ground_plane_offset')}",
    )
    if a.spz and not Path(a.spz).exists():
        size = download(assets["splats"]["spz_urls"]["full_res"], a.spz)
        log(a, f"spz full_res {size} bytes -> {a.spz}")
        copy = directory / f"{a.name}.spz"
        if Path(a.spz).resolve() != copy.resolve():
            shutil.copy(a.spz, copy)
        log(a, f"copy -> {copy}")
    elif a.spz:
        log(a, f"spz already on disk -> {a.spz}")
    if a.thumb and assets.get("thumbnail_url") and not Path(a.thumb).exists():
        size = download(assets["thumbnail_url"], a.thumb)
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
        if a.prompt_file:
            prompt = json.loads(Path(a.prompt_file).read_text())["text_prompt"]
            if not isinstance(prompt, str):
                raise ValueError("prompt-file text_prompt must be a string")
    else:
        if not a.target:
            raise ValueError(f"{a.input_type} submit needs an input file")
        items = [(Path(a.target), None)]
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


def upload(a, path):
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
    asset_id = prepared["media_asset"]["media_asset_id"]
    content_type = f"{kind}/{'jpeg' if extension in ('jpg', 'jpeg') else extension}"
    status, _ = call(
        "PUT",
        prepared["upload_info"]["upload_url"],
        raw=path.read_bytes(),
        headers={"x-goog-content-length-range": "0,104857600", "Content-Type": content_type},
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


def submit(a):
    if recover_submission(a):
        return
    if a.spz and Path(a.spz).exists():
        raise ValueError(
            "Output SPZ already exists; use --reuse-world or fetch, never generate over it"
        )
    items, prompt = inputs(a)
    receipt_path, receipt = claim_submission(a)
    entries = []
    for path, angle in items:
        asset_id = upload(a, path)
        entry = {"content": {"source": "media_asset", "media_asset_id": asset_id}}
        if angle is not None:
            entry["azimuth"] = angle
        entries.append(entry)
    if a.input_type == "video":
        body = {
            "model": a.model,
            "world_prompt": {"type": "video", "video_prompt": entries[0]["content"]},
        }
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
    if a.input_type == "multi":
        (Path(a.marble_dir) / f"{a.name}-request.json").write_text(json.dumps(body, indent=1))
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
    poll(a, operation_id)


def parser():
    root = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = root.add_subparsers(dest="input_type", required=True)
    for input_type in ("image", "multi", "video"):
        ap = modes.add_parser(input_type)
        ap.add_argument("mode", choices=["submit", "poll", "fetch"])
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
        if input_type != "video":
            ap.add_argument("--ops", help="ops log stem (default name)")
            ap.add_argument("--prompt", default="" if input_type == "image" else None)
            ap.add_argument("--seed", type=int)
            ap.add_argument("--note", default="")
        if input_type == "multi":
            ap.add_argument("--images", nargs="+", default=[], help="path[:azimuth_deg] per view")
            ap.add_argument("--prompt-file", help="JSON from scripts/world_prompt.py")
    return root


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.interval < 0 or (a.mode != "submit" and not a.target):
        ap.error("poll/fetch require an existing ID; interval must be nonnegative")
    try:
        if a.mode == "submit":
            submit(a)
        elif a.mode == "poll":
            poll(a, a.target)
        else:
            fetch(a, a.target)
    except (ValueError, KeyError, OSError) as error:
        ap.exit(1, f"Marble {a.mode} failed: {error}\n")


if __name__ == "__main__":
    main()
