#!/usr/bin/env python3
"""Generate a World Labs Marble world from one 16:9 PNG (image mode, disable_recaption) and
download its splats. Same flow and logging as marble_video_world.py.

  WLT_API_KEY=... python3 scripts/marble_image_world.py submit input.png --name corridor-recon-f0 \
      --marble-dir .context/marble --ops corridor-clean \
      --spz public/marble-corridor-recon-f0.spz --thumb share/corridor-recon-f0-thumb.png \
      --prompt "..." [--seed 7] [--note "source frame 0, SfM camera 0"]
  ... poll <operation_id> --name ... --marble-dir ... --ops ... --spz ... --thumb ...
"""
import argparse, json, os, shutil, sys, time, urllib.error, urllib.request
from pathlib import Path

BASE = "https://api.worldlabs.ai"


def key():
    k = os.environ.get("WLT_API_KEY")
    if not k:
        sys.exit("set WLT_API_KEY")
    return k


def call(method, path, body=None, raw=None, headers=None, timeout=120):
    h = {"WLT-Api-Key": key()}
    if body is not None:
        raw = json.dumps(body).encode(); h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(BASE + path if path.startswith("/") else path, data=raw, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, (json.loads(data) if data and r.headers.get("Content-Type", "").startswith("application/json") else data)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode(errors='replace')[:2000]}")


def log(a, line):
    Path(a.marble_dir).mkdir(parents=True, exist_ok=True)
    with (Path(a.marble_dir) / f"{a.ops or a.name}-ops.txt").open("a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"[{a.name}] " + line + "\n")
    print(line, flush=True)


def download(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    return dest.stat().st_size


def poll(a, op_id):
    while True:
        _, op = call("GET", f"/marble/v1/operations/{op_id}")
        (Path(a.marble_dir) / f"{a.name}-op.json").write_text(json.dumps(op, indent=1))
        prog = (op.get("metadata") or {}).get("progress") or {}
        print(time.strftime("%H:%M:%S"), prog.get("status"), prog.get("description"), flush=True)
        if op.get("done"):
            break
        time.sleep(a.interval)
    if op.get("error"):
        log(a, f"op {op_id} FAILED: {op['error']}"); sys.exit(1)
    fetch(a, op["metadata"]["world_id"], f"op {op_id} done: ")


def fetch(a, world_id, prefix=""):
    """Save an existing world's metadata, splats and thumbnail. Read-only: no generation, no credits."""
    _, world = call("GET", f"/marble/v1/worlds/{world_id}")
    (Path(a.marble_dir) / f"{a.name}-world.json").write_text(json.dumps(world, indent=1))
    sem = world["assets"]["splats"].get("semantics_metadata") or {}
    log(a, f"{prefix}world {world_id} metric_scale_factor {sem.get('metric_scale_factor')} ground_plane_offset {sem.get('ground_plane_offset')}")
    if a.spz and not Path(a.spz).exists():
        n = download(world["assets"]["splats"]["spz_urls"]["full_res"], a.spz); log(a, f"spz full_res {n} bytes -> {a.spz}")
        copy = Path(a.marble_dir) / f"{a.name}.spz"; shutil.copy(a.spz, copy); log(a, f"copy -> {copy}")
    elif a.spz:
        log(a, f"spz already on disk -> {a.spz}")
    if a.thumb and world["assets"].get("thumbnail_url") and not Path(a.thumb).exists():
        n = download(world["assets"]["thumbnail_url"], a.thumb); log(a, f"thumbnail {n} bytes -> {a.thumb}")


def submit(a):
    img = Path(a.target); size = img.stat().st_size
    _, prep = call("POST", "/marble/v1/media-assets:prepare_upload", {"file_name": img.name, "kind": "image", "extension": "png"})
    asset_id = prep["media_asset"]["media_asset_id"]; upload_url = prep["upload_info"]["upload_url"]
    status, _ = call("PUT", upload_url, raw=img.read_bytes(), headers={"x-goog-content-length-range": "0,104857600", "Content-Type": "image/png"}, timeout=600)
    log(a, f"image (input {img}, {size} bytes, asset {asset_id} upload {status}){' ' + a.note if a.note else ''}")
    body = {"display_name": a.name, "model": a.model,
            "world_prompt": {"type": "image", "image_prompt": {"source": "media_asset", "media_asset_id": asset_id},
                             "is_pano": False, "disable_recaption": True, "text_prompt": a.prompt}}
    if a.seed is not None:
        body["seed"] = a.seed
    _, op = call("POST", "/marble/v1/worlds:generate", body)
    op_id = op["operation_id"]
    log(a, f"op {op_id} submitted (asset {asset_id}, {a.model}, seed {a.seed}, disable_recaption)")
    poll(a, op_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "poll", "fetch"])
    ap.add_argument("target", help="png path (submit), operation id (poll) or world id (fetch)")
    ap.add_argument("--name", required=True); ap.add_argument("--marble-dir", required=True); ap.add_argument("--ops", help="ops log stem (default name)")
    ap.add_argument("--spz"); ap.add_argument("--thumb"); ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--prompt", default=""); ap.add_argument("--seed", type=int); ap.add_argument("--model", default="marble-1.1"); ap.add_argument("--note", default="")
    a = ap.parse_args()
    {"submit": submit, "poll": lambda a: poll(a, a.target),
     "fetch": lambda a: fetch(a, a.target)}[a.mode](a)


if __name__ == "__main__":
    main()
