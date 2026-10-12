"""Read-only setup diagnostics; never launch inference, build images, or stage models."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from itertools import islice
from fractions import Fraction
from pathlib import Path
from typing import Any

from orchestrator.agent.harness import HarnessPolicy
from orchestrator.config import REPOSITORY_ROOT, resolve_runs_root
from orchestrator.steps import STEPS, RunOptions, planned_steps

PROBE_TIMEOUT = 20
CLIP_TIMEOUT = 90
MODULES = (
    "numpy",
    "scipy",
    "PIL",
    "cv2",
    "plyfile",
    "modal",
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "safetensors",
)
SEGFORMER = "huggingface/hub/models--nvidia--segformer-b0-finetuned-ade-512-512"
LHM = "huggingface/hub/models--3DAIGC--LHM-500M-HF"
PI3X = "huggingface/hub/models--yyfz233--Pi3X"


class ProbeError(RuntimeError):
    pass


def _probe(argv: list[str], *, timeout: int = PROBE_TIMEOUT, env=None, accepted=(0,)):
    """Bound the whole subprocess tree and never include provider stderr in reports."""
    try:
        process = subprocess.Popen(
            argv,
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except OSError:
        raise ProbeError("probe executable could not start") from None
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise ProbeError(f"probe exceeded {timeout}s") from None
    if process.returncode not in accepted:
        raise ProbeError(f"probe exited {process.returncode}; inspect configuration privately")
    return output, process.returncode


def _module_inventory():
    result = {}
    for name in MODULES:
        try:
            importlib.import_module(name)
            result[name] = True
        except Exception:
            result[name] = False
    return result


def _cache_requirements(plan: list[str]) -> dict[str, tuple[str, ...]]:
    required = {
        "wander-overnight-motion-cache": (PI3X, SEGFORMER),
        "wander-overnight-lhm-cache": (
            "data/pretrained_models/human_model_files",
            "data/pretrained_models/gagatracker",
            "data/pretrained_models/dense_sample_points",
            "data/pretrained_models/voxel_grid",
            "data/gfpgan",
            "gfpgan/weights/GFPGANv1.3.pth",
            "torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
            LHM,
        ),
    }
    if any(name in plan for name in ("clean", "clean_first", "clean_multi")):
        required["wander-clean-video-cache"] = ("lama/big-lama.pt", SEGFORMER)
    if "objects" in plan:
        required["wander-image-to-3d-cache"] = (
            "huggingface/hub/models--microsoft--TRELLIS-image-large",
            "huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0",
            "torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
        )
    return required


def _modal_inventory(requirements: dict[str, list[str]]):
    # Import only the SDK. Importing a worker would define image builds and functions.
    import modal
    from modal.types import FileEntryType

    result = {}
    listing_limit = 2000
    for volume_name, paths in requirements.items():
        volume = modal.Volume.from_name(volume_name, create_if_missing=False)
        for path in paths:
            key = f"{volume_name}:{path}"
            try:
                entries = list(islice(volume.iterdir(path, recursive=True), listing_limit))
                listing_truncated = len(entries) == listing_limit
                found = []
                linked_snapshot_entries = False
                for entry in entries:
                    if entry.type == FileEntryType.FILE and entry.size > 0:
                        found.append(entry.path)
                    elif entry.type == FileEntryType.SYMLINK and "/snapshots/" in entry.path:
                        linked_snapshot_entries |= entry.path.endswith(
                            (".safetensors", ".bin", ".pt", ".pth", ".json", ".yaml")
                        )
                if path.startswith("huggingface/"):
                    # An empty cache directory or a leftover .incomplete blob is not a model.
                    available = any(
                        "/snapshots/" in name
                        and name.endswith((".safetensors", ".bin", ".pt", ".pth"))
                        for name in found
                    ) and any(
                        "/snapshots/" in name and name.endswith((".json", ".yaml"))
                        for name in found
                    )
                else:
                    available = any(not name.endswith(".incomplete") for name in found)
                if available:
                    result[key] = "present"
                elif linked_snapshot_entries or listing_truncated:
                    # Modal exposes snapshot symlinks as metadata, but this probe must not
                    # follow them or read their targets. A bounded listing may also omit
                    # required files, so neither condition can establish absence.
                    result[key] = "unverified"
                else:
                    result[key] = "missing"
            except Exception as error:
                # Return only an exception class, never token-bearing SDK messages.
                result[key] = "missing" if type(error).__name__ == "NotFoundError" else "unverified"
    return result


def _modal_configuration() -> bool:
    profile = os.environ.get("MODAL_PROFILE", "dtpu")
    location = Path(os.environ.get("MODAL_CONFIG_PATH", "~/.modal.toml")).expanduser()
    try:
        configured = tomllib.loads(location.read_text()).get(profile, {})
    except (OSError, ValueError):
        configured = {}
    return bool(
        (os.environ.get("MODAL_TOKEN_ID") or configured.get("token_id"))
        and (os.environ.get("MODAL_TOKEN_SECRET") or configured.get("token_secret"))
    )


def _clip_checks(clip: Path, add) -> dict[str, Any] | None:
    if not clip.is_file() or clip.stat().st_size == 0:
        add(
            "clip.metadata",
            "blocked",
            "Clip is missing, empty, or not a regular file.",
            "Supply --clip /absolute/path/to/source.mp4",
        )
        return None
    try:
        raw, _ = _probe(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(clip),
            ]
        )
        document = json.loads(raw)
        streams = document["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        duration = float(video.get("duration") or document.get("format", {}).get("duration") or 0)
        fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
        width, height = int(video["width"]), int(video["height"])
        if not all(math.isfinite(value) and value > 0 for value in (duration, fps, width, height)):
            raise ValueError("invalid dimensions/timing")
        metadata = {
            "width": width,
            "height": height,
            "fps": fps,
            "durationSeconds": duration,
            "videoCodec": video.get("codec_name"),
            "audioCodecs": [
                stream.get("codec_name")
                for stream in streams
                if stream.get("codec_type") == "audio"
            ],
        }
        add(
            "clip.metadata",
            "okay",
            f"{width}x{height}, {fps:.3f} fps, {duration:.3f}s; audio streams: {len(metadata['audioCodecs'])}.",
        )
    except (ProbeError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError):
        add(
            "clip.metadata",
            "blocked",
            "ffprobe could not establish valid video dimensions and timing.",
            "Inspect the source with ffprobe -v error -show_streams -show_format <clip>",
        )
        return None
    with tempfile.TemporaryDirectory(prefix="wander-doctor-") as temporary:
        report = Path(temporary) / "cuts.json"
        try:
            _, code = _probe(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/shot_cuts.py"),
                    "--video",
                    str(clip),
                    "--no-score",
                    "--no-people",
                    "--json",
                    str(report),
                ],
                timeout=CLIP_TIMEOUT,
                accepted=(0, 1),
            )
            cuts = json.loads(report.read_text())
            count = cuts["cutCount"]
            if (
                type(count) is not int
                or count < 0
                or type(cuts["continuous"]) is not bool
                or not isinstance(cuts["cuts"], list)
                or count != len(cuts["cuts"])
                or cuts["continuous"] != (count == 0)
                or code != int(count > 0)
            ):
                raise ValueError("continuity report disagrees with detector")
            metadata["cutCount"] = count
            add(
                "clip.continuity",
                "okay" if count == 0 else "blocked",
                f"Local cut detector found {count} cut(s); this does not prove camera-solve quality.",
                "Save and review a continuity report: uv run --locked python scripts/shot_cuts.py --video <clip> --no-score --no-people --json .context/cuts.json; then trim the reviewed shot."
                if count
                else "",
            )
        except (ProbeError, OSError, ValueError, KeyError, TypeError):
            add(
                "clip.continuity",
                "blocked",
                "Local continuity inspection failed or exceeded its deadline.",
                "Run uv run --locked python scripts/shot_cuts.py --video <clip> --no-score --no-people --json .context/cuts.json",
            )
    return metadata


def inspect_setup(
    options: RunOptions | dict | None = None, clip: Path | None = None, remote: bool = False
) -> dict:
    """Inspect selected-plan configuration without uploading a clip or submitting work."""
    options = (
        options if isinstance(options, RunOptions) else RunOptions.model_validate(options or {})
    )
    plan = planned_steps(options)
    checks = []

    def add(name, status, detail, action=""):
        checks.append({"name": name, "status": status, "detail": detail, "action": action})

    for executable in ("ffmpeg", "ffprobe", "bun"):
        available = shutil.which(executable)
        working = bool(available)
        if available:
            try:
                _probe([available, "--version" if executable == "bun" else "-version"], timeout=5)
            except ProbeError:
                working = False
        add(
            f"tool.{executable}",
            "okay" if working else "blocked",
            f"{executable} starts successfully."
            if working
            else f"{executable} is missing or cannot start.",
            f"Install or repair {executable} and rerun this doctor." if not working else "",
        )
    try:
        policy = HarnessPolicy.from_environment()
        executable = policy.command[0] if policy.kind == "command" else policy.kind
        available = shutil.which(executable)
        add(
            "tool.harness",
            "okay" if available else "blocked",
            "Configured reviewing harness executable is available."
            if available
            else "Configured reviewing harness executable is missing.",
            "Install the selected harness or configure WANDER_REVIEW_AGENT / WANDER_REVIEW_COMMAND."
            if not available
            else "",
        )
    except (ValueError, IndexError):
        add(
            "tool.harness",
            "blocked",
            "Review harness configuration is invalid.",
            "Correct WANDER_REVIEW_AGENT and WANDER_REVIEW_COMMAND.",
        )
    add(
        "harness.access",
        "unverified",
        "Harness account/session access was not tested; no model request was made.",
        "Verify login with the configured harness's account status command.",
    )
    try:
        output, _ = _probe(
            [
                sys.executable,
                "-c",
                "import json; from orchestrator.doctor import _module_inventory; print(json.dumps(_module_inventory()))",
            ]
        )
        modules = json.loads(output)
        for module in MODULES:
            available = modules.get(module) is True
            add(
                f"python.{module}",
                "okay" if available else "blocked",
                f"{module} imports in the current interpreter."
                if available
                else f"{module} is missing or cannot import in the current interpreter.",
                "uv sync --locked --group inference; rerun with uv run --locked --group inference python -m orchestrator.doctor --json"
                if not available
                else "",
            )
    except (ProbeError, ValueError, AttributeError):
        add(
            "python.imports",
            "blocked",
            "Current-interpreter import probe failed or timed out.",
            "uv sync --locked --group inference; inspect imports in that environment.",
        )
    override = os.environ.get("WANDER_PYTHON")
    if override:
        try:
            same = Path(override).expanduser().resolve() == Path(sys.executable).resolve()
        except OSError:
            same = False
        if not same:
            add(
                "python.override",
                "unverified",
                "WANDER_PYTHON selects a different interpreter from the one checked.",
                "Run the doctor using WANDER_PYTHON or remove the override.",
            )
    root = resolve_runs_root()
    ancestor = root
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
    add(
        "runs.root",
        "okay" if writable else "blocked",
        f"Shared runs root: {root}; parent {'is' if writable else 'is not'} writable.",
        "Configure WANDER_RUNS_ROOT to a writable directory." if not writable else "",
    )
    # anchors only calls OpenAI for its opt-in ruler_vlm knob, absent from RunOptions.
    for credential in sorted(
        {name for step in plan if step != "anchors" for name in STEPS[step].credentials}
    ):
        present = bool(os.environ.get(credential, "").strip())
        add(
            f"credential.{credential}",
            "okay" if present else "blocked",
            f"{credential} is configured; its value is never reported."
            if present
            else f"{credential} is missing.",
            f"Set {credential} in private .env.author and load it with: set -a; source .env.author; set +a"
            if not present
            else "",
        )
        if present:
            add(
                f"access.{credential}",
                "unverified",
                f"{credential} account access, balance and quota are unverified.",
                "Confirm account permissions, credits and run budget before a paid launch.",
            )
    configured = _modal_configuration()
    modal_executable = os.environ.get("WANDER_MODAL") or "modal"
    add(
        "tool.modal",
        "okay" if shutil.which(modal_executable) else "blocked",
        "Configured Modal executable is available."
        if shutil.which(modal_executable)
        else "Configured Modal executable is missing.",
        "uv sync --locked",
    )
    add(
        "modal.auth",
        "unverified" if configured else "blocked",
        "Selected Modal profile/token pair is configured; workspace access has not been tested."
        if configured
        else "Selected Modal profile/token pair is absent or incomplete (default profile: dtpu).",
        "uv run --locked modal token new --profile <your-profile>; export MODAL_PROFILE=<your-profile>"
        if not configured
        else "Rerun with --remote to inspect the selected workspace read-only.",
    )
    requirements = _cache_requirements(plan)
    inventory = None
    if remote and configured:
        try:
            environment = {**os.environ, "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "dtpu")}
            output, _ = _probe(
                [
                    sys.executable,
                    "-c",
                    "import json,sys; from orchestrator.doctor import _modal_inventory; print(json.dumps(_modal_inventory(json.loads(sys.argv[1]))))",
                    json.dumps(requirements),
                ],
                env=environment,
            )
            inventory = json.loads(output)
            if not isinstance(inventory, dict):
                raise ValueError("invalid inventory")
            if any(value in {"present", "missing"} for value in inventory.values()):
                next(check for check in checks if check["name"] == "modal.auth").update(
                    status="okay",
                    detail="Read-only volume metadata was accessible with the selected Modal credentials.",
                    action="",
                )
        except (ProbeError, ValueError, TypeError):
            add(
                "modal.remote",
                "unverified",
                "Read-only Modal probe failed or exceeded its deadline.",
                "Inspect uv run --locked modal volume list --json with the selected profile.",
            )
    for volume, paths in requirements.items():
        for path in paths:
            observed = inventory.get(f"{volume}:{path}") if inventory is not None else None
            status = (
                "okay"
                if observed == "present"
                else "blocked"
                if observed == "missing"
                else "unverified"
            )
            add(
                f"cache.{volume}:{path}",
                status,
                "Nonempty model/cache files are visible; hashes and model compatibility are not validated."
                if status == "okay"
                else "Required remote model cache is missing/empty."
                if status == "blocked"
                else "Required remote model cache has not been verified.",
                "Review docs/paid-recovery.md and worker/modal_stage_weights.py before deliberately staging missing weights."
                if status == "blocked"
                else "Rerun with --remote; inspect saved model staging receipts for hashes/revisions.",
            )
    if "finetune" in plan:
        host = os.environ.get("WANDER_GPU_BOX", "").strip()
        available = (
            bool(host) and not host.startswith("-") and not any(char.isspace() for char in host)
        )
        add(
            "finetune.host",
            "unverified" if available else "blocked",
            "WANDER_GPU_BOX is configured; remote trainer access is unverified."
            if available
            else "Fine-tuning needs a valid WANDER_GPU_BOX SSH destination.",
            "Set WANDER_GPU_BOX=user@host, or omit fine-tuning.",
        )
        for tool in ("ssh", "rsync"):
            add(
                f"tool.{tool}",
                "okay" if shutil.which(tool) else "blocked",
                f"{tool} {'is available' if shutil.which(tool) else 'is missing'}.",
                f"Install {tool}.",
            )
        if remote and available and shutil.which("ssh"):
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
            if identity := os.environ.get("WANDER_GPU_KEY"):
                command += ["-i", str(Path(identity).expanduser())]
            try:
                _probe([*command, host, "true"], timeout=12)
                next(check for check in checks if check["name"] == "finetune.host").update(
                    status="okay",
                    detail="Read-only SSH login succeeded; trainer/GPU environment remains unverified.",
                    action="",
                )
            except ProbeError:
                add(
                    "finetune.ssh",
                    "blocked",
                    "Read-only SSH login failed or timed out.",
                    "Verify WANDER_GPU_BOX, WANDER_GPU_KEY and known_hosts privately.",
                )
        add(
            "finetune.trainer",
            "unverified",
            "Remote fine-tune Python/GPU dependencies were not loaded or executed.",
            "Inspect the configured host's ~/venv trainer environment before training.",
        )
    metadata = _clip_checks(Path(clip).expanduser().resolve(), add) if clip is not None else None
    if clip is None:
        add(
            "clip",
            "unverified",
            "No source clip was supplied; timing and continuity are unverified.",
            "Rerun with --clip /absolute/path/to/source.mp4",
        )
    status = (
        "blocked"
        if any(check["status"] == "blocked" for check in checks)
        else "unverified"
        if any(check["status"] == "unverified" for check in checks)
        else "okay"
    )
    return {
        "schema": "wander.pipeline-doctor/1",
        "ready": status == "okay",
        "status": status,
        "remote": remote,
        "options": options.model_dump(),
        "plan": plan,
        "runsRoot": str(root),
        "python": sys.executable,
        "checks": checks,
        "clip": metadata,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--options", default="{}", help="RunOptions as a JSON object")
    parser.add_argument(
        "--remote", action="store_true", help="Bounded read-only Modal/SSH metadata probes"
    )
    args = parser.parse_args(argv)
    try:
        options = RunOptions.model_validate(json.loads(args.options))
        report = inspect_setup(options, args.clip, args.remote)
    except ValueError:
        report = {
            "schema": "wander.pipeline-doctor/1",
            "ready": False,
            "status": "blocked",
            "checks": [
                {
                    "name": "options",
                    "status": "blocked",
                    "detail": "--options must be a valid RunOptions JSON object.",
                    "action": "Use --options '{}' or supported pipeline options.",
                }
            ],
        }
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"Setup: {report['status']} (no inference submitted)")
        for check in report["checks"]:
            print(f"{check['status']:10} {check['name']}: {check['detail']}")
            if check["action"]:
                print(f"           {check['action']}")
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
