#!/usr/bin/env python3
"""Generate a World Labs Marble world from a (person-free) clip and download its splats.

  WLT_API_KEY=... python3 scripts/marble_video_world.py submit clean.mp4 --name tos-hall-walk-clean \
      --marble-dir .context/marble
  WLT_API_KEY=... python3 scripts/marble_video_world.py poll <operation_id> --name ... --marble-dir ... \
      --spz public/marble-<name>.spz --thumb share/<name>-thumb.png

submit: prepare_upload -> PUT mp4 -> worlds:generate, then keeps polling (same as poll).
poll: GET operations/<id> every 60 s until done, then GET worlds/<id>, save world JSON,
full_res .spz and thumbnail. Every step appends to <marble-dir>/<name>-ops.txt.
fetch: the tail of poll against a world id that already exists -- no generation, no credits.
"""
import argparse, json, os, shutil, sys, time, urllib.request
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return r.status, (json.loads(data) if data and r.headers.get("Content-Type", "").startswith("application/json") else data)


def log(a, line):
    Path(a.marble_dir).mkdir(parents=True, exist_ok=True)
    with (Path(a.marble_dir) / f"{a.name}-ops.txt").open("a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
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
    log(a, f"{prefix}world {world_id} {world.get('world_marble_url')} metric_scale_factor "
           f"{sem.get('metric_scale_factor')} ground_plane_offset {sem.get('ground_plane_offset')}")
    spz_url = world["assets"]["splats"]["spz_urls"]["full_res"]
    if a.spz and not Path(a.spz).exists():
        n = download(spz_url, a.spz); log(a, f"spz full_res {n} bytes -> {a.spz}")
        copy = Path(a.marble_dir) / f"{a.name}.spz"; shutil.copy(a.spz, copy); log(a, f"copy -> {copy}")
    elif a.spz:
        log(a, f"spz already on disk -> {a.spz}")
    if a.thumb and world["assets"].get("thumbnail_url") and not Path(a.thumb).exists():
        n = download(world["assets"]["thumbnail_url"], a.thumb); log(a, f"thumbnail {n} bytes -> {a.thumb}")


def submit(a):
    clip = Path(a.target); size = clip.stat().st_size
    if size > 104857600:
        sys.exit("clip over 100 MB")
    _, prep = call("POST", "/marble/v1/media-assets:prepare_upload", {"file_name": clip.name, "kind": "video", "extension": "mp4"})
    asset_id = prep["media_asset"]["media_asset_id"]; upload_url = prep["upload_info"]["upload_url"]
    status, _ = call("PUT", upload_url, raw=clip.read_bytes(), headers={"x-goog-content-length-range": "0,104857600", "Content-Type": "video/mp4"}, timeout=600)
    log(a, f"video (input {clip}, {size} bytes, asset {asset_id} upload {status})")
    _, op = call("POST", "/marble/v1/worlds:generate", {"world_prompt": {"type": "video", "video_prompt": {"source": "media_asset", "media_asset_id": asset_id}}})
    op_id = op["operation_id"]
    log(a, f"op {op_id} submitted (asset {asset_id})")
    poll(a, op_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "poll", "fetch"])
    ap.add_argument("target", help="clip path (submit), operation id (poll) or world id (fetch)")
    ap.add_argument("--name", required=True); ap.add_argument("--marble-dir", required=True)
    ap.add_argument("--spz"); ap.add_argument("--thumb"); ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    {"submit": submit, "poll": lambda a: poll(a, a.target),
     "fetch": lambda a: fetch(a, a.target)}[a.mode](a)


if __name__ == "__main__":
    main()
