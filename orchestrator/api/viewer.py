"""Read-only, run-scoped delivery of packaged viewer assets."""

from __future__ import annotations

import json
import math
import mimetypes
import posixpath
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

WORLD_SUFFIXES = {"video": "clean", "both": "clean", "image": "image", "multi": "multi"}
ASSET_SUFFIXES = {
    ".json",
    ".ply",
    ".spz",
    ".glb",
    ".gltf",
    ".bin",
    ".f32",
    ".u16",
    ".u8",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def run_name(root: Path) -> str:
    name = read_json(root / "run.json").get("name", "")
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
        or name in {".", ".."}
    ):
        raise ValueError("invalid run name")
    return name


def asset_path(root: Path, relative: str, *, require_file: bool = True) -> Path:
    """Only the selected run's exported scene and copied source, never its working files."""
    name = run_name(root)
    parts = relative.split("/")
    if (
        not parts
        or relative.startswith("/")
        or any(not p or p in {".", ".."} or p.startswith(".") for p in parts)
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError("invalid viewer asset path")
    path = root / "public" / relative
    source = Path(read_json(root / "run.json").get("source") or "missing")
    source_relative = f"clips/{name}/source{source.suffix.lower()}"
    world_files = {
        f"marble-{name}-{suffix}.spz" for suffix in {*WORLD_SUFFIXES.values(), "finetuned"}
    }
    allowed = (
        relative == source_relative
        or relative in world_files
        or (relative.startswith(f"worlds/{name}-4d/") and path.suffix.lower() in ASSET_SUFFIXES)
    )
    public = (root / "public").resolve()
    if (
        not allowed
        or not public.is_relative_to(root.resolve())
        or not path.resolve().is_relative_to(public)
        or (require_file and not path.is_file())
    ):
        raise ValueError("unknown viewer asset")
    return path


def asset_url(run_id: str, relative: str) -> str:
    return f"/api/pipeline/runs/{quote(run_id, safe='')}/viewer/assets/{quote(relative, safe='/')}"


def resolve_reference(root: Path, manifest: str, reference: str) -> str:
    if (
        not isinstance(reference, str)
        or not reference
        or "\\" in reference
        or ":" in reference
        or reference.startswith("//")
    ):
        raise ValueError("invalid viewer reference")
    relative = (
        reference[1:]
        if reference.startswith("/")
        else posixpath.normpath(posixpath.join(posixpath.dirname(manifest), reference))
    )
    asset_path(root, relative)
    return relative


def require_sequence(root: Path, relative: str) -> None:
    sequence = read_json(asset_path(root, relative))
    frames = sequence.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("empty person sequence")
    for frame in frames:
        resolve_reference(root, relative, frame)
    # Compact motion is an optional acceleration. The viewer verifies it and falls
    # back to these authoritative PLYs when its descriptor or payload is unusable.


def scene_manifest(root: Path) -> dict | None:
    described = read_json(root / "run.json")
    try:
        name = run_name(root)
        run_id = described["runId"]
        base = f"worlds/{name}-4d/"
        person = base + "person/sequence.json"
        require_sequence(root, person)
        asset_path(root, base + "cameras.json")
        asset_path(root, base + "audio.json")
        people = base + "people.json"
        if (root / "public" / people).is_file():
            cast = read_json(asset_path(root, people)).get("people")
            if not isinstance(cast, list) or not cast:
                return None
            for entry in cast:
                if not isinstance(entry, dict):
                    return None
                reference = entry.get("sequence") or f"{entry['directory']}/sequence.json"
                require_sequence(root, resolve_reference(root, people, reference))
            person = people
        source = Path(described["source"])
        video = f"clips/{name}/source{source.suffix.lower()}"
        asset_path(root, video)
        options = described.get("options") or {}
        mode = options.get("marble", "video")
        world = "none"
        if mode != "none":
            suffix = "finetuned" if options.get("finetune") else WORLD_SUFFIXES[mode]
            world = asset_url(run_id, f"marble-{name}-{suffix}.spz")
            asset_path(root, f"marble-{name}-{suffix}.spz")
        params = {
            "world": world,
            "person": asset_url(run_id, person),
            "video": asset_url(run_id, video),
            "place": "0",
            # Packaged scenes use camera 0 at the origin looking down -Z. Starting
            # outside the generated world's full bounds can hide the entire replay.
            # This is a navigation default, not evidence of camera registration.
            "cam": "0,0,0,0,0,-1",
        }
        objects = base + "objects.json"
        if not (root / "public" / base / "ruler.json").is_file():
            params["ruler"] = "0"
        params["objects"] = (
            asset_url(run_id, objects) if (root / "public" / objects).is_file() else "0"
        )
        # A diagnostic preview never promotes a placement or a quality verdict. Preserve
        # only the runner's reported global scale/floor, without applying its fitted pose.
        placement = read_json(root / "state.json").get("stages", {}).get("_placement", {})

        def finite(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        scale = placement.get("fittedScale0")
        if finite(scale) and scale > 0:
            params["scale"] = str(scale)
            if finite(placement.get("floor")):
                params["floor"] = str(placement["floor"])
        else:
            provider = read_json(root / "marble" / f"{name}-{WORLD_SUFFIXES.get(mode)}-world.json")
            metadata = ((provider.get("assets") or {}).get("splats") or {}).get(
                "semantics_metadata"
            ) or {}
            factor = metadata.get(
                "metric_scale_factor",
                provider.get("metric_scale_factor", placement.get("metricScaleFactor")),
            )
            ground = metadata.get(
                "ground_plane_offset",
                provider.get("ground_plane_offset", placement.get("groundPlaneOffset")),
            )
            if finite(factor) and factor > 0:
                params["height"] = str(1.7 / factor)
                if finite(ground):
                    params["floor"] = str(-ground / factor)
            else:
                params["scale"] = "1"
        return {"schema": "wander.viewer-scene/1", "params": params}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def rewrite_references(value, root: Path, relative: str):
    """Root-relative exports become relative to this manifest, including concatenating loaders."""
    if isinstance(value, dict):
        return {key: rewrite_references(item, root, relative) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_references(item, root, relative) for item in value]
    if isinstance(value, str) and value.startswith(("/worlds/", "/clips/", "/marble-")):
        target = value.lstrip("/")
        # Keep references confined, but let optional missing assets reach the
        # viewer's existing fallback instead of failing the entire JSON response.
        asset_path(root, target, require_file=False)
        return posixpath.relpath(target, posixpath.dirname(relative))
    return value


def register_viewer_routes(app, actor, directory):
    @app.api_route("/api/pipeline/runs/{run_id}/viewer/scene.json", methods=["GET", "HEAD"])
    def scene(run_id: str, request: Request, user: str = Depends(actor)):
        manifest = scene_manifest(directory(run_id))
        if manifest is None:
            raise HTTPException(404, "packaged scene is not ready for preview")
        response = JSONResponse(manifest, headers={"Cache-Control": "no-store"})
        return Response(headers=response.headers) if request.method == "HEAD" else response

    @app.api_route("/api/pipeline/runs/{run_id}/viewer/assets/{path:path}", methods=["GET", "HEAD"])
    def asset(run_id: str, path: str, request: Request, user: str = Depends(actor)):
        root = directory(run_id)
        try:
            file = asset_path(root, path)
            if file.suffix.lower() == ".json":
                document = rewrite_references(json.loads(file.read_text()), root, path)
                response = JSONResponse(
                    document,
                    headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
                )
                return Response(headers=response.headers) if request.method == "HEAD" else response
        except (OSError, ValueError, TypeError):
            raise HTTPException(404, "unknown viewer asset") from None
        media_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if media_type in {"text/html", "application/xhtml+xml", "image/svg+xml"}:
            media_type = "application/octet-stream"
        return FileResponse(
            file,
            media_type=media_type,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
