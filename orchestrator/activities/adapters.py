"""Command adapters for existing Wander scripts and the compatibility runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from orchestrator.activities.stage import AdapterContext, StageExecution


def one(context: AdapterContext, name: str) -> Path:
    values = context.inputs.get(name, ())
    if len(values) != 1:
        raise ValueError(f"{context.request.node_id} requires exactly one {name} artifact")
    return values[0]


def named(paths: tuple[Path, ...], filename: str) -> Path:
    matches = [path for path in paths if path.name == filename]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {filename}, found {len(matches)}")
    return matches[0]


def bundle_root(paths: tuple[Path, ...], suffix: str) -> Path:
    for path in paths:
        for parent in (path, *path.parents):
            if parent.name.endswith(suffix):
                return parent
    raise ValueError(f"artifact bundle has no {suffix} root")


class AdmissionAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        source = one(context, "source")
        report = context.attempt.outputs / "shots.json"
        command = (
            sys.executable,
            str(context.repository / "scripts/shot_cuts.py"),
            "--video",
            str(source),
            "--json",
            str(report),
            "--no-score",
        )
        return StageExecution(
            command=command,
            cwd=context.repository,
            output_roles={
                "outputs/shots.json": "shots",
                "outputs/source.json": "source_metadata",
                "outputs/shot-clips/*.mp4": "shot_clip",
            },
            # shot_cuts.py exits 1 to report "this clip is not one continuous shot". That is the
            # finding admission exists to make: finalize() then trims each usable shot and shots()
            # fans them out as branches. Only a crash (any other code) is a failed probe.
            success_exit_codes=frozenset({0, 1}),
        )

    def finalize(self, context: AdapterContext) -> None:
        source = one(context, "source")
        with source.open("rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        (context.attempt.outputs / "source.json").write_text(
            json.dumps(
                {
                    "schema": "wander.source/1",
                    "sha256": source_sha256,
                    "bytes": source.stat().st_size,
                },
                indent=2,
            )
        )
        report = json.loads((context.attempt.outputs / "shots.json").read_text())
        if report.get("continuous"):
            return
        clips = context.attempt.outputs / "shot-clips"
        clips.mkdir()
        for shot in report.get("shots", []):
            if shot.get("tooShort"):
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(context.repository / "scripts/shot_cuts.py"),
                    "--report",
                    str(context.attempt.outputs / "shots.json"),
                    "--trim",
                    str(shot["index"]),
                    "--out",
                    str(clips / f"shot-{shot['index']:02d}.mp4"),
                ],
                cwd=context.repository,
                check=True,
            )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        report = json.loads((context.attempt.outputs / "shots.json").read_text())
        if report.get("continuous"):
            return []
        return [
            {
                "key": f"{shot['index']:02d}",
                "relative_path": f"outputs/shot-clips/shot-{shot['index']:02d}.mp4",
                "start_seconds": shot["start"],
                "end_seconds": shot["end"],
            }
            for shot in report.get("shots", [])
            if not shot.get("tooShort")
        ]


class WorldPromptAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        output = context.attempt.outputs / "prompt.json"
        parameters = context.request.parameters
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/world_prompt.py"),
                "--clip",
                str(one(context, "source")),
                "--n",
                str(parameters.get("samples", 6)),
                "--out",
                str(output),
                "--model",
                str(parameters.get("model", "gpt-6-astra")),
            ),
            cwd=context.repository,
            credentials=("OPENAI_API_KEY",),
            output_roles={"outputs/prompt.json": "world_prompt"},
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class WorldModeAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        output = context.attempt.outputs / "mode.json"
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/select_world_mode.py"),
                "--cameras",
                str(one(context, "cameras")),
                "--clip",
                str(one(context, "source")),
                "--still-images",
                "--out",
                str(output),
            ),
            cwd=context.repository,
            output_roles={"outputs/mode.json": "world_mode"},
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class CleanMultiAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        source = one(context, "source")
        mode = json.loads(one(context, "world_mode").read_text())
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,width,height",
                    "-of",
                    "json",
                    str(source),
                ],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        )["streams"][0]
        numerator, denominator = probe["avg_frame_rate"].split("/")
        source_fps = float(numerator) / float(denominator)
        target_fps = float(context.request.parameters.get("fps", 12))
        step = source_fps / target_fps
        only = ",".join(str(int(round(frame / step))) for frame in mode["frames"])
        output = context.attempt.outputs / "clean-multi"
        report = context.attempt.outputs / "clean-multi.json"
        parameters = context.request.parameters
        command = [
            str(context.request.parameters.get("modal", "modal")),
            "run",
            "worker/modal_clean_video.py",
            "--clip",
            str(source),
            "--only",
            only,
            "--frames-out",
            str(output),
            "--report",
            str(report),
            "--fps",
            str(target_fps),
            "--width",
            str(probe["width"]),
            "--height",
            str(probe["height"]),
            "--dilate",
            str(parameters.get("dilate", 20)),
            "--bottom-extra",
            str(parameters.get("bottom_extra", 40)),
            "--lama-px",
            str(parameters.get("lama_px", 960)),
        ]
        if parameters.get("moved_mask"):
            command.append("--moved-mask")
        return StageExecution(
            command=tuple(command),
            cwd=context.repository,
            output_roles={
                "outputs/clean-multi/f_*.png": "clean_frames",
                "outputs/clean-multi.json": "clean_report",
            },
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


def marble_mode(context: AdapterContext) -> str:
    for mode in ("image", "video", "multi"):
        if mode in context.request.stage_type:
            return mode
    raise ValueError(f"cannot determine Marble mode from {context.request.stage_type}")


def marble_name(context: AdapterContext, mode: str) -> str:
    digest = hashlib.sha256(f"{context.request.run_id}\0{mode}".encode()).hexdigest()[:16]
    return f"world-{digest}-{mode}"


class MarbleSubmitAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        mode = marble_mode(context)
        name = marble_name(context, mode)
        directory = context.attempt.outputs / "marble"
        command = [
            sys.executable,
            str(context.repository / "scripts/marble_world.py"),
            mode,
            "submit-only",
        ]
        if mode == "image":
            command.append(str(one(context, "clean_frame")))
        elif mode == "video":
            command.append(str(one(context, "clean_video")))
        prompt = one(context, "prompt")
        command += [
            "--name",
            name,
            "--marble-dir",
            str(directory),
            "--prompt-file",
            str(prompt),
        ]
        if mode == "multi":
            world_mode = json.loads(one(context, "world_mode").read_text())
            frames = context.inputs["clean_frames"]
            if len(frames) != len(world_mode["azimuth"]):
                raise ValueError("cleaned Marble views do not match measured azimuths")
            command += [
                "--images",
                *[f"{frame}:{azimuth}" for frame, azimuth in zip(frames, world_mode["azimuth"])],
            ]
        if mode in {"image", "multi"}:
            command += ["--seed", str(context.request.parameters.get("seed", 7))]
        return StageExecution(
            command=tuple(command),
            cwd=context.repository,
            credentials=("WLT_API_KEY",),
            output_roles={f"outputs/marble/{name}-generation.json": "provider_operation"},
            unknown_on_failure=True,
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class MarblePollAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        mode = marble_mode(context)
        receipt = json.loads(one(context, "provider_operation").read_text())
        operation_id = receipt.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("Marble submission receipt has no operation ID")
        name = marble_name(context, mode)
        directory = context.attempt.outputs / "marble"
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/marble_world.py"),
                mode,
                "poll",
                operation_id,
                "--name",
                name,
                "--marble-dir",
                str(directory),
                "--spz",
                str(context.attempt.outputs / "world.spz"),
                "--thumb",
                str(context.attempt.outputs / "world-thumb.png"),
                "--interval",
                str(context.request.parameters.get("interval", 60)),
            ),
            cwd=context.repository,
            credentials=("WLT_API_KEY",),
            output_roles={
                "outputs/world.spz": "world_splat",
                "outputs/world-thumb.png": "world_thumbnail",
                f"outputs/marble/{name}-world.json": "world_receipt",
            },
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class PersonPrepAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        output = context.attempt.outputs / "prepared-person"
        parameters = context.request.parameters
        command = [
            sys.executable,
            str(context.repository / "scripts/prepare_lhm_person.py"),
            str(one(context, "source")),
            "--out",
            str(output),
            "--method",
            str(parameters.get("method", "maskrcnn")),
        ]
        if parameters.get("frame") is not None:
            command += ["--frame", str(parameters["frame"])]
        return StageExecution(
            command=tuple(command),
            cwd=context.repository,
            output_roles={"outputs/prepared-person/**/*": "prepared_person"},
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class AudioPackageAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        world = one(context, "viewer_world")
        config = one(context, "audio_config")
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/package_audio.py"),
                "--config",
                str(config),
                "--world",
                str(world),
                "--reviewed",
            ),
            cwd=context.repository,
            output_roles={
                "inputs/viewer_world/audio.json": "audio_manifest",
                "inputs/viewer_world/*.wav": "audio_tracks",
            },
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class ObjectDetectAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        output = context.attempt.outputs / "flights.json"
        crops = context.attempt.outputs / "crops"
        tracks = named(context.inputs["track_data"], "tracks.json")
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/detect_object_flights.py"),
                "--clip",
                str(one(context, "source")),
                "--tracks",
                str(tracks),
                "--cameras",
                str(one(context, "cameras")),
                "--people",
                str(one(context, "people")),
                "--out",
                str(output),
                "--crops",
                str(crops),
            ),
            cwd=context.repository,
            output_roles={
                "outputs/flights.json": "detected_objects",
                "outputs/detected-object-*.json": "detected_object",
                "outputs/crops/*.png": "object_crops",
            },
        )

    def finalize(self, context: AdapterContext) -> None:
        report = json.loads((context.attempt.outputs / "flights.json").read_text())
        if report.get("flights"):
            shutil.copy2(
                context.attempt.outputs / "flights.json",
                context.attempt.outputs / "detected-object-auto.json",
            )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        report = json.loads((context.attempt.outputs / "flights.json").read_text())
        return (
            [
                {
                    "key": "auto",
                    "relative_path": "outputs/detected-object-auto.json",
                }
            ]
            if report.get("flights")
            else []
        )

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class ObjectLiftAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        tracks = named(context.inputs["track_data"], "tracks.json")
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/lift_object_3d.py"),
                "--flights",
                str(one(context, "object")),
                "--cameras",
                str(one(context, "cameras")),
                "--people",
                str(one(context, "people")),
                "--tracks-dir",
                str(tracks.parent),
                "--tracks",
                str(tracks),
                "--out",
                str(context.attempt.outputs / "fit3d.json"),
            ),
            cwd=context.repository,
            output_roles={"outputs/fit3d.json": "object_track"},
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class ObjectDescribeAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        crops = context.inputs["object_crops"]
        if not crops:
            raise ValueError("object description requires retained source crops")
        crops_dir = crops[0].parent
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/describe_object.py"),
                "--flights",
                str(one(context, "object")),
                "--crops-dir",
                str(crops_dir),
                "--cameras",
                str(one(context, "cameras")),
                "--out",
                str(context.attempt.outputs / "description.json"),
            ),
            cwd=context.repository,
            output_roles={
                "outputs/description.json": "object_prompt",
                "outputs/best-crop.png": "object_image",
            },
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class ObjectShapeAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        description = json.loads(one(context, "prompt").read_text())["description"]
        return StageExecution(
            command=(
                str(context.request.parameters.get("modal", "modal")),
                "run",
                "worker/modal_image_to_3d.py",
                "--image",
                str(one(context, "image")),
                "--out-dir",
                str(context.attempt.outputs / "shape"),
                "--refine-first",
                "--refine-prompt",
                description["refinePrompt"],
                "--refine-strength",
                str(context.request.parameters.get("refine_strength", 0.85)),
                "--note",
                "orchestrated auto-object branch from retained source crops",
            ),
            cwd=context.repository,
            output_roles={"outputs/shape/object.ply": "object_shape"},
            unknown_on_failure=True,
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


class ObjectPackageAdapter:
    def build(self, context: AdapterContext) -> StageExecution:
        tracks = named(context.inputs["track_data"], "tracks.json")
        world = bundle_root(context.inputs["viewer_world"], "-4d")
        branch_keys = sorted(
            name.removeprefix("track_") for name in context.inputs if name.startswith("track_")
        )
        branches = []
        for key in branch_keys:
            description = json.loads(one(context, f"description_{key}").read_text())
            branches.append(
                {
                    "id": description["id"],
                    "fit": str(one(context, f"track_{key}")),
                    "model": str(one(context, f"shape_{key}")),
                    "description": description,
                }
            )
        config = context.attempt.root / "object-package.json"
        config.write_text(
            json.dumps(
                {
                    "world": str(world),
                    "tracksDir": str(tracks.parent),
                    "cameras": str(one(context, "cameras")),
                    "branches": branches,
                },
                indent=2,
            )
        )
        return StageExecution(
            command=(
                sys.executable,
                str(context.repository / "scripts/package_object_branches.py"),
                "--config",
                str(config),
                "--python",
                sys.executable,
            ),
            cwd=context.repository,
            output_roles={
                "inputs/viewer_world/**/objects.json": "objects_manifest",
            },
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []


# Parameters the policy layer allows alongside a proposal but which are not stage arguments.
CONTROL_PARAMETERS = frozenset({"hypothesis", "estimated_cost_usd", "marble"})


def legacy_people_flags(options: dict[str, Any]) -> list[str]:
    """Ask the legacy pipeline for the graph shape this run was created with.

    run_clip.py builds a single-person graph unless it is told otherwise, and stages such as
    `tracks` exist only in the multiperson graph. Without this the command names a stage its own
    graph does not contain and reports "unknown stage(s)".
    """
    people = options.get("people")
    if isinstance(people, int) and people > 1:
        return ["--people", str(people)]
    if options.get("all_people"):
        return ["--all-people"]
    return []


def legacy_parameter_flags(definition: dict[str, Any], parameters: dict[str, Any]) -> list[str]:
    """Turn a stage's declared parameters into CLI flags for the legacy pipeline.

    Only names the stage's own ``parameter_schema`` declares are forwarded, so an operator or
    the reviewing agent can set exactly what the schema advertises and nothing else. A boolean
    is a bare flag when true and absent when false.
    """
    properties = (definition.get("parameter_schema") or {}).get("properties") or {}
    flags: list[str] = []
    for name in sorted(parameters):
        if name in CONTROL_PARAMETERS or name not in properties:
            continue
        value = parameters[name]
        if value is False or value is None:
            continue
        flags.append(f"--{name.replace('_', '-')}")
        if value is not True:
            flags.append(str(value))
    return flags


class LegacyPipelineAdapter:
    """Run one existing Pipeline method inside attempt-scoped output directories."""

    def __init__(
        self,
        legacy_stage: str,
        output_roles: dict[str, str],
        materialize=None,
        finalizer=None,
    ):
        self.legacy_stage = legacy_stage
        self.output_roles = output_roles
        self.materialize = materialize
        self.finalizer = finalizer

    def build(self, context: AdapterContext) -> StageExecution:
        source = one(context, "source")
        digest = hashlib.sha256(
            f"{context.request.run_id}\0{context.request.node_id}".encode()
        ).hexdigest()[:16]
        name = f"activity-{digest}"
        root = context.attempt.outputs
        run = root / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        dependencies = {
            binding["stage_id"]: {"status": "ok"}
            for binding in context.request.definition.get("inputs", {}).values()
            if binding.get("source") == "stage_output"
        }
        (run / "state.json").write_text(json.dumps({"stages": dependencies}, indent=2))
        if self.materialize:
            self.materialize(context, root, name)
        command = [
            sys.executable,
            str(context.repository / "scripts/run_clip.py"),
            "--clip",
            str(source),
            "--name",
            name,
            "--marble",
            str(context.request.parameters.get("marble", "none")),
            "--only",
            self.legacy_stage,
            "--one-shot",
            "--no-publish",
        ]
        command += legacy_people_flags(context.request.options)
        command += legacy_parameter_flags(context.request.definition, context.request.parameters)
        return StageExecution(
            command=tuple(command),
            cwd=context.repository,
            environment={
                "WANDER_RUNS_DIR": str(root / "runs"),
                "WANDER_PUBLIC_DIR": str(root / "public"),
                "WANDER_SHARE_DIR": str(root / "share"),
                "WANDER_CLIPS_DIR": str(root / "clips"),
                "WANDER_MARBLE_DIR": str(root / "marble"),
            },
            output_roles={
                pattern.replace("{name}", name): role for pattern, role in self.output_roles.items()
            },
            unknown_on_failure=bool(context.request.definition.get("retry", {}).get("paid")),
        )

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []

    def finalize(self, context: AdapterContext) -> None:
        if self.finalizer:
            self.finalizer(context)


def _copy_files(paths: tuple[Path, ...], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination / path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            target = (
                destination / f"{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}-{path.name}"
            )
        if not target.exists():
            shutil.copy2(path, target)


def materialize_pi3x(context: AdapterContext, root: Path, name: str) -> None:
    destination = root / "runs" / name / "pi3x"
    for role in ("cameras", "point_clouds", "pi3x_aux"):
        _copy_files(context.inputs.get(role, ()), destination)


def materialize_person_prep(context: AdapterContext, root: Path, name: str) -> None:
    _copy_files(context.inputs.get("prepared_person", ()), root / "runs" / name / "prepared-person")


def materialize_lhm(context: AdapterContext, root: Path, name: str) -> None:
    materialize_pi3x(context, root, name)
    _copy_files(context.inputs.get("canonical_person", ()), root / "runs" / name / "lhm-frozen")


def materialize_package(context: AdapterContext, root: Path, name: str) -> None:
    materialize_pi3x(context, root, name)
    _copy_files(context.inputs.get("person_motion", ()), root / "runs" / name / "lhm-motion")


def _copy_viewer_world(paths: tuple[Path, ...], destination: Path) -> None:
    for path in paths:
        parts = path.parts
        marker = next((index for index, part in enumerate(parts) if part.endswith("-4d")), None)
        relative = Path(*parts[marker + 1 :]) if marker is not None else Path(path.name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def materialize_world_fit(context: AdapterContext, root: Path, name: str) -> None:
    materialize_pi3x(context, root, name)
    world = one(context, "world")
    destination = root / "public" / f"marble-{name}-clean.spz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(world, destination)
    _copy_viewer_world(
        context.inputs.get("viewer_world", ()),
        root / "public" / "worlds" / f"{name}-4d",
    )
    for receipt in context.inputs.get("world_receipt", ()):
        target = root / "marble" / f"{name}-clean-world.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receipt, target)


def materialize_scale_state(context: AdapterContext, root: Path, name: str) -> None:
    materialize_world_fit(context, root, name)
    scale = one(context, "scale")
    state = json.loads((root / "runs" / name / "state.json").read_text())
    state["stages"]["_scale"] = json.loads(scale.read_text())
    (root / "runs" / name / "state.json").write_text(json.dumps(state, indent=2))


def finalize_scale(context: AdapterContext) -> None:
    state = next((context.attempt.outputs / "runs").glob("*/state.json"))
    value = json.loads(state.read_text())["stages"]["_scale"]
    (context.attempt.outputs / "scale-fit.json").write_text(json.dumps(value, indent=2))


def materialize_placed_world(context: AdapterContext, root: Path, name: str) -> None:
    materialize_scale_state(context, root, name)
    placement = one(context, "placement")
    destination = root / "public" / "worlds" / f"{name}-4d" / "placement.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(placement, destination)


def finalize_anchors(context: AdapterContext) -> None:
    source = next((context.attempt.outputs / "runs").glob("*/anchors.json"))
    shutil.copy2(source, context.attempt.outputs / "anchors.json")


def finalize_finetune(context: AdapterContext) -> None:
    worlds = list((context.attempt.outputs / "public").glob("marble-*-finetuned.spz"))
    if len(worlds) != 1:
        raise ValueError(f"expected one fine-tuned world, found {len(worlds)}")
    shutil.copy2(worlds[0], context.attempt.outputs / "finetuned.spz")
    logs = sorted((context.attempt.outputs / "runs").glob("*/finetune*.log"))
    (context.attempt.outputs / "finetune-receipt.json").write_text(
        json.dumps(
            {
                "schema": "wander.finetune-receipt/1",
                "world": worlds[0].name,
                "logs": [log.name for log in logs],
            },
            indent=2,
        )
    )


def default_adapters() -> dict[str, object]:
    """Adapters are keyed by StageDefinition.executor."""
    common = {
        "admit_source": AdmissionAdapter(),
        "world_prompt": WorldPromptAdapter(),
        "world_mode": WorldModeAdapter(),
        "clean_multi": CleanMultiAdapter(),
        "marble_submit": MarbleSubmitAdapter(),
        "marble_poll": MarblePollAdapter(),
        "person_prep": PersonPrepAdapter(),
        "package_audio": AudioPackageAdapter(),
        "object_detect": ObjectDetectAdapter(),
        "object_lift": ObjectLiftAdapter(),
        "object_describe": ObjectDescribeAdapter(),
        "object_shape": ObjectShapeAdapter(),
        "object_package": ObjectPackageAdapter(),
    }
    legacy = {
        "clean_first": (
            "clean_first",
            {
                "outputs/runs/{name}/first/f_0000.png": "clean_frame",
                "outputs/runs/{name}/clean-first.json": "clean_report",
            },
        ),
        "clean_video": (
            "clean",
            {
                "outputs/clips/{name}-clean.mp4": "clean_video",
                "outputs/runs/{name}/clean-f0.png": "clean_frame",
                "outputs/runs/{name}/masks.npz": "person_masks",
                "outputs/runs/{name}/clean.json": "clean_report",
            },
        ),
        "pi3x": (
            "pi3x",
            {
                "outputs/runs/{name}/pi3x/cameras.json": "cameras",
                "outputs/runs/{name}/pi3x/frame_*.ply": "point_clouds",
                "outputs/runs/{name}/pi3x/anchors.npz": "pi3x_aux",
                "outputs/runs/{name}/pi3x/meta.json": "pi3x_aux",
                "outputs/runs/{name}/pose-guard.json": "motion_report",
            },
        ),
        "frame_align": (
            "frame_align",
            {"outputs/runs/{name}/pi3x/framealign.json": "frame_alignment"},
        ),
        "track_people": (
            "tracks",
            {
                "outputs/runs/{name}/tracks/tracks.json": "person_tracks",
                "outputs/runs/{name}/tracks/**/*": "person_track_data",
            },
        ),
        "lhm_frozen": (
            "lhm_frozen",
            {
                "outputs/runs/{name}/lhm-frozen/**/*": "canonical_person",
                "outputs/runs/{name}/lhm-frozen/recovery-receipt.json": "recovery_receipt",
            },
        ),
        "lhm_motion": (
            "lhm_motion",
            {
                "outputs/runs/{name}/lhm-motion/motion.json": "person_motion",
                "outputs/runs/{name}/lhm-motion/frame_*.ply": "person_frames",
                "outputs/runs/{name}/lhm-motion/recovery-receipt.json": "recovery_receipt",
            },
        ),
        "package_people": (
            "package",
            {
                "outputs/public/worlds/{name}-4d/**/*": "viewer_world",
                "outputs/public/worlds/{name}-4d/people.json": "people_manifest",
                "outputs/public/worlds/{name}-4d/audio.json": "audio_manifest",
            },
        ),
    }
    special = {
        "scale_fit": LegacyPipelineAdapter(
            "scale_fit",
            {"outputs/scale-fit.json": "scale_fit"},
            materialize=materialize_world_fit,
            finalizer=finalize_scale,
        ),
        "place_fit": LegacyPipelineAdapter(
            "place_fit",
            {
                "outputs/public/worlds/{name}-4d/placement.json": "placement",
            },
            materialize=materialize_scale_state,
        ),
        "anchors": LegacyPipelineAdapter(
            "anchors",
            {"outputs/anchors.json": "anchor_report"},
            materialize=materialize_placed_world,
            finalizer=finalize_anchors,
        ),
        "finetune": LegacyPipelineAdapter(
            "finetune",
            {
                "outputs/finetuned.spz": "finetuned_world",
                "outputs/finetune-receipt.json": "finetune_receipt",
            },
            materialize=materialize_placed_world,
            finalizer=finalize_finetune,
        ),
    }
    return {
        **common,
        **{
            executor: LegacyPipelineAdapter(
                stage,
                roles,
                {
                    "frame_align": materialize_pi3x,
                    "track_people": materialize_pi3x,
                    "lhm_frozen": materialize_person_prep,
                    "lhm_motion": materialize_lhm,
                    "package_people": materialize_package,
                }.get(executor),
            )
            for executor, (stage, roles) in legacy.items()
        },
        **special,
    }
