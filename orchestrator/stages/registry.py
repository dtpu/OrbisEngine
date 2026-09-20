"""Typed stage definitions for the existing generation pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.contracts import (
    ArtifactBinding,
    ArtifactContract,
    QualityPolicy,
    ResourcePolicy,
    RetryPolicy,
    StageDefinition,
    StageKind,
)


class GraphOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    marble: Literal["video", "image", "multi", "both", "none"] = "video"
    people: int = Field(default=1, ge=1, le=16)
    all_people: bool = False
    objects: bool = True
    reviewed_audio: bool = False
    finetune: bool = False


def run_input(role: str) -> ArtifactBinding:
    return ArtifactBinding(source="run_input", role=role)


def stage_output(stage_id: str, role: str) -> ArtifactBinding:
    return ArtifactBinding(source="stage_output", stage_id=stage_id, role=role)


def output(
    role: str,
    media_type: str,
    contract: str | None = None,
    *,
    required: bool = True,
    multiple: bool = False,
) -> ArtifactContract:
    return ArtifactContract(
        role=role,
        media_type=media_type,
        contract=contract,
        required=required,
        multiple=multiple,
    )


def local_resources(timeout: int = 3600) -> ResourcePolicy:
    return ResourcePolicy(task_queue="local_cpu", timeout_seconds=timeout)


def modal_resources(timeout: int = 14400, concurrency: str = "modal") -> ResourcePolicy:
    return ResourcePolicy(
        task_queue="modal_gpu",
        concurrency_key=concurrency,
        timeout_seconds=timeout,
        network=True,
    )


def external_resources(timeout: int = 7200, concurrency: str = "external_api") -> ResourcePolicy:
    return ResourcePolicy(
        task_queue="external_api",
        concurrency_key=concurrency,
        timeout_seconds=timeout,
        network=True,
    )


PAID_ONCE = RetryPolicy(
    paid=True,
    automatic_attempts=1,
    requires_changed_hypothesis=True,
    requires_changed_parameters=True,
    recover_before_retry=True,
)
CLEAN_PARAMETERS = {
    "type": "object",
    "properties": {
        "fps": {"type": "number", "minimum": 1, "maximum": 60},
        "dilate": {"type": "integer", "minimum": 0, "maximum": 200},
        "bottom_extra": {"type": "integer", "minimum": 0, "maximum": 400},
        "lama_px": {"type": "integer", "minimum": 256, "maximum": 2048},
        "moved_mask": {"type": "boolean"},
    },
    "additionalProperties": False,
}
MARBLE_PARAMETERS = {
    "type": "object",
    "properties": {
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "interval": {"type": "integer", "minimum": 0, "maximum": 600},
    },
    "additionalProperties": False,
}
OBJECT_SHAPE_PARAMETERS = {
    "type": "object",
    "properties": {
        "refine_strength": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}


def stage_registry() -> dict[str, StageDefinition]:
    """Return semantic stage types; graph instantiation assigns concrete node IDs."""
    stages = [
        StageDefinition(
            id="admission",
            title="Probe source and select continuous shots",
            kind=StageKind.EXPAND,
            executor="admit_source",
            inputs={"source": run_input("source_video")},
            outputs={
                "source_metadata": output("source_metadata", "application/json"),
                "shots": output("shots", "application/json", "wander.shots/1"),
                "shot_clips": output("shot_clip", "video/mp4", required=False, multiple=True),
            },
            resources=local_resources(600),
            quality=QualityPolicy(automatic_checks=("source_identity", "shot_continuity")),
        ),
        StageDefinition(
            id="clean_first",
            title="Clean first frame",
            kind=StageKind.COMPUTE,
            executor="clean_first",
            inputs={"source": run_input("source_video")},
            outputs={
                "clean_frame": output("clean_frame", "image/png"),
                "clean_report": output("clean_report", "application/json"),
            },
            parameter_schema=CLEAN_PARAMETERS,
            resources=modal_resources(concurrency="inpainting"),
            retry=PAID_ONCE,
            quality=QualityPolicy(automatic_checks=("image_readable",)),
        ),
        StageDefinition(
            id="clean",
            title="Remove people from video",
            kind=StageKind.COMPUTE,
            executor="clean_video",
            inputs={"source": run_input("source_video")},
            outputs={
                "clean_video": output("clean_video", "video/mp4"),
                "clean_frame": output("clean_frame", "image/png"),
                "masks": output("person_masks", "application/octet-stream"),
                "clean_report": output("clean_report", "application/json"),
            },
            parameter_schema=CLEAN_PARAMETERS,
            resources=modal_resources(concurrency="inpainting"),
            retry=PAID_ONCE,
            quality=QualityPolicy(
                automatic_checks=("video_readable", "mask_report"),
                agent_rubric="cleaned_scene",
            ),
        ),
        StageDefinition(
            id="clean_multi",
            title="Clean selected world input frames",
            kind=StageKind.COMPUTE,
            executor="clean_multi",
            inputs={
                "source": run_input("source_video"),
                "world_mode": stage_output("world_mode", "world_mode"),
            },
            outputs={
                "clean_frames": output("clean_frames", "image/png", multiple=True),
                "clean_report": output("clean_report", "application/json"),
            },
            parameter_schema=CLEAN_PARAMETERS,
            resources=modal_resources(concurrency="inpainting"),
            retry=PAID_ONCE,
            quality=QualityPolicy(
                automatic_checks=("images_readable",), agent_rubric="cleaned_scene"
            ),
        ),
        StageDefinition(
            id="world_prompt",
            title="Describe source-grounded world",
            kind=StageKind.AGENT,
            executor="world_prompt",
            inputs={"source": run_input("source_video")},
            outputs={"world_prompt": output("world_prompt", "application/json")},
            resources=external_resources(900, "openai"),
            retry=RetryPolicy(paid=True),
            quality=QualityPolicy(automatic_checks=("prompt_grounding",)),
        ),
        StageDefinition(
            id="world_mode",
            title="Select measured world input views",
            kind=StageKind.COMPUTE,
            executor="world_mode",
            inputs={
                "source": run_input("source_video"),
                "cameras": stage_output("pi3x", "cameras"),
            },
            outputs={"world_mode": output("world_mode", "application/json")},
            resources=local_resources(900),
            quality=QualityPolicy(automatic_checks=("world_mode_measurements",)),
        ),
        StageDefinition(
            id="clean_review",
            title="Approve cleaned world input",
            kind=StageKind.HUMAN,
            inputs={
                "clean_report": stage_output("clean", "clean_report"),
                "clean_frame": stage_output("clean", "clean_frame"),
            },
            outputs={"approval": output("approval", "application/json")},
            quality=QualityPolicy(agent_rubric="cleaned_scene", human_approval=True),
        ),
        StageDefinition(
            id="marble_image_submit",
            title="Submit cleaned image world generation",
            kind=StageKind.COMPUTE,
            executor="marble_submit",
            inputs={
                "clean_frame": stage_output("clean_first", "clean_frame"),
                "prompt": stage_output("world_prompt", "world_prompt"),
                "approval": stage_output("clean_review", "approval"),
            },
            outputs={"provider_operation": output("provider_operation", "application/json")},
            parameter_schema=MARBLE_PARAMETERS,
            resources=external_resources(1800, "marble_submit"),
            retry=PAID_ONCE,
            quality=QualityPolicy(automatic_checks=("provider_operation",)),
        ),
        StageDefinition(
            id="marble_image",
            title="Recover generated image world",
            kind=StageKind.COMPUTE,
            executor="marble_poll",
            inputs={
                "provider_operation": stage_output("marble_image_submit", "provider_operation")
            },
            outputs={
                "world": output("world_splat", "application/octet-stream"),
                "world_receipt": output("world_receipt", "application/json"),
                "world_thumbnail": output("world_thumbnail", "image/png"),
            },
            resources=external_resources(14400, "marble"),
            retry=RetryPolicy(automatic_attempts=6),
            quality=QualityPolicy(automatic_checks=("marble_receipt", "world_readable")),
        ),
        StageDefinition(
            id="marble_video_submit",
            title="Submit cleaned video world generation",
            kind=StageKind.COMPUTE,
            executor="marble_submit",
            inputs={
                "clean_video": stage_output("clean", "clean_video"),
                "prompt": stage_output("world_prompt", "world_prompt"),
                "approval": stage_output("clean_review", "approval"),
            },
            outputs={"provider_operation": output("provider_operation", "application/json")},
            parameter_schema=MARBLE_PARAMETERS,
            resources=external_resources(1800, "marble_submit"),
            retry=PAID_ONCE,
            quality=QualityPolicy(automatic_checks=("provider_operation",)),
        ),
        StageDefinition(
            id="marble_video",
            title="Recover generated video world",
            kind=StageKind.COMPUTE,
            executor="marble_poll",
            inputs={
                "provider_operation": stage_output("marble_video_submit", "provider_operation")
            },
            outputs={
                "world": output("world_splat", "application/octet-stream"),
                "world_receipt": output("world_receipt", "application/json"),
                "world_thumbnail": output("world_thumbnail", "image/png"),
            },
            resources=external_resources(14400, "marble"),
            retry=RetryPolicy(automatic_attempts=6),
            quality=QualityPolicy(automatic_checks=("marble_receipt", "world_readable")),
        ),
        StageDefinition(
            id="marble_multi_submit",
            title="Submit selected-frame world generation",
            kind=StageKind.COMPUTE,
            executor="marble_submit",
            inputs={
                "clean_frames": stage_output("clean_multi", "clean_frames"),
                "world_mode": stage_output("world_mode", "world_mode"),
                "prompt": stage_output("world_prompt", "world_prompt"),
                "approval": stage_output("clean_review", "approval"),
            },
            outputs={"provider_operation": output("provider_operation", "application/json")},
            parameter_schema=MARBLE_PARAMETERS,
            resources=external_resources(1800, "marble_submit"),
            retry=PAID_ONCE,
            quality=QualityPolicy(automatic_checks=("provider_operation",)),
        ),
        StageDefinition(
            id="marble_multi",
            title="Recover generated selected-frame world",
            kind=StageKind.COMPUTE,
            executor="marble_poll",
            inputs={
                "provider_operation": stage_output("marble_multi_submit", "provider_operation")
            },
            outputs={
                "world": output("world_splat", "application/octet-stream"),
                "world_receipt": output("world_receipt", "application/json"),
                "world_thumbnail": output("world_thumbnail", "image/png"),
            },
            resources=external_resources(14400, "marble"),
            retry=RetryPolicy(automatic_attempts=6),
            quality=QualityPolicy(automatic_checks=("marble_receipt", "world_readable")),
        ),
        StageDefinition(
            id="pi3x",
            title="Estimate cameras, depth, and people",
            kind=StageKind.COMPUTE,
            executor="pi3x",
            inputs={"source": run_input("source_video")},
            outputs={
                "cameras": output("cameras", "application/json"),
                "point_clouds": output("point_clouds", "application/octet-stream", multiple=True),
                "pi3x_aux": output("pi3x_aux", "application/octet-stream", multiple=True),
                "motion_report": output("motion_report", "application/json"),
            },
            resources=modal_resources(concurrency="pi3x"),
            retry=PAID_ONCE,
            quality=QualityPolicy(automatic_checks=("pose_continuity", "person_depth")),
        ),
        StageDefinition(
            id="frame_align",
            title="Align camera frames to playback time",
            kind=StageKind.COMPUTE,
            executor="frame_align",
            inputs={
                "source": run_input("source_video"),
                "cameras": stage_output("pi3x", "cameras"),
                "point_clouds": stage_output("pi3x", "point_clouds"),
                "pi3x_aux": stage_output("pi3x", "pi3x_aux"),
            },
            outputs={"frame_alignment": output("frame_alignment", "application/json")},
            resources=local_resources(1800),
            quality=QualityPolicy(automatic_checks=("frame_mapping",)),
        ),
        StageDefinition(
            id="tracks",
            title="Track all visible people",
            kind=StageKind.EXPAND,
            executor="track_people",
            inputs={
                "source": run_input("source_video"),
                "cameras": stage_output("pi3x", "cameras"),
            },
            outputs={
                "tracks": output("person_tracks", "application/json", "wander.tracks/1"),
                "person_track": output(
                    "person_track", "application/json", "wander.track/1", multiple=True
                ),
                "track_data": output("person_track_data", "application/octet-stream"),
            },
            resources=modal_resources(concurrency="tracking"),
            retry=PAID_ONCE,
            quality=QualityPolicy(
                automatic_checks=("identity_stability",), agent_rubric="person_tracks"
            ),
        ),
        StageDefinition(
            id="person_prep",
            title="Prepare person reference",
            kind=StageKind.COMPUTE,
            executor="person_prep",
            inputs={
                "source": run_input("source_video"),
                "track": stage_output("tracks", "person_track"),
            },
            outputs={"prepared_person": output("prepared_person", "application/octet-stream")},
            resources=local_resources(3600),
            quality=QualityPolicy(
                automatic_checks=("person_mask",), agent_rubric="person_reference"
            ),
        ),
        StageDefinition(
            id="lhm_frozen",
            title="Build canonical person",
            kind=StageKind.COMPUTE,
            executor="lhm_frozen",
            inputs={"prepared_person": stage_output("person_prep", "prepared_person")},
            outputs={
                "canonical_person": output("canonical_person", "application/octet-stream"),
                "recovery_receipt": output("recovery_receipt", "application/json"),
            },
            resources=modal_resources(concurrency="lhm"),
            retry=PAID_ONCE,
            quality=QualityPolicy(automatic_checks=("canonical_person", "recovery_receipt")),
        ),
        StageDefinition(
            id="lhm_motion",
            title="Animate reconstructed person",
            kind=StageKind.COMPUTE,
            executor="lhm_motion",
            inputs={
                "source": run_input("source_video"),
                "canonical_person": stage_output("lhm_frozen", "canonical_person"),
                "cameras": stage_output("pi3x", "cameras"),
                "point_clouds": stage_output("pi3x", "point_clouds"),
            },
            outputs={
                "person_motion": output(
                    "person_motion", "application/json", "wander.person-motion/1"
                ),
                "person_frames": output("person_frames", "application/octet-stream", multiple=True),
                "recovery_receipt": output("recovery_receipt", "application/json"),
            },
            resources=modal_resources(concurrency="lhm"),
            retry=PAID_ONCE,
            quality=QualityPolicy(
                automatic_checks=("motion_completeness",), agent_rubric="person_motion"
            ),
        ),
        StageDefinition(
            id="package_people",
            title="Package people and original playback",
            kind=StageKind.JOIN,
            executor="package_people",
            inputs={
                "motions": stage_output("lhm_motion", "person_motion"),
                "alignment": stage_output("frame_align", "frame_alignment"),
                "source": run_input("source_video"),
            },
            outputs={
                "viewer_world": output("viewer_world", "application/x-directory"),
                "people_manifest": output("people_manifest", "application/json"),
                "audio_manifest": output("audio_manifest", "application/json", "wander.audio/1"),
            },
            quality=QualityPolicy(automatic_checks=("package_integrity", "audio_timeline")),
        ),
        StageDefinition(
            id="audio_reviewed",
            title="Package reviewed dialogue and ambience",
            kind=StageKind.COMPUTE,
            executor="package_audio",
            inputs={
                "viewer_world": stage_output("package_people", "viewer_world"),
                "audio_config": run_input("reviewed_audio_config"),
            },
            outputs={
                "audio_manifest": output("audio_manifest", "application/json", "wander.audio/1"),
                "audio_tracks": output("audio_tracks", "audio/wav", multiple=True),
            },
            resources=local_resources(1800),
            quality=QualityPolicy(automatic_checks=("audio_timeline",), human_approval=True),
        ),
        StageDefinition(
            id="object_detect",
            title="Detect dynamic objects",
            kind=StageKind.EXPAND,
            executor="object_detect",
            inputs={
                "source": run_input("source_video"),
                "people": stage_output("package_people", "people_manifest"),
                "cameras": stage_output("pi3x", "cameras"),
                "track_data": stage_output("tracks", "person_track_data"),
            },
            outputs={
                "objects": output("detected_objects", "application/json"),
                "detected_object": output(
                    "detected_object",
                    "application/json",
                    required=False,
                    multiple=True,
                ),
                "object_crops": output("object_crops", "image/png", required=False, multiple=True),
            },
            resources=local_resources(3600),
            quality=QualityPolicy(agent_rubric="dynamic_objects"),
        ),
        StageDefinition(
            id="object_lift",
            title="Lift object motion into 3D",
            kind=StageKind.COMPUTE,
            executor="object_lift",
            inputs={
                "object": stage_output("object_detect", "detected_object"),
                "cameras": stage_output("pi3x", "cameras"),
                "people": stage_output("package_people", "people_manifest"),
                "track_data": stage_output("tracks", "person_track_data"),
            },
            outputs={"object_track": output("object_track", "application/json")},
            resources=local_resources(3600),
            quality=QualityPolicy(automatic_checks=("object_track",)),
        ),
        StageDefinition(
            id="object_describe",
            title="Describe object appearance",
            kind=StageKind.AGENT,
            executor="object_describe",
            inputs={
                "object": stage_output("object_detect", "detected_object"),
                "object_crops": stage_output("object_detect", "object_crops"),
                "cameras": stage_output("pi3x", "cameras"),
            },
            outputs={
                "object_prompt": output("object_prompt", "application/json"),
                "object_image": output("object_image", "image/png"),
            },
            resources=external_resources(900, "openai"),
            retry=RetryPolicy(paid=True),
        ),
        StageDefinition(
            id="object_shape",
            title="Generate object shape",
            kind=StageKind.COMPUTE,
            executor="object_shape",
            inputs={
                "prompt": stage_output("object_describe", "object_prompt"),
                "image": stage_output("object_describe", "object_image"),
            },
            outputs={"object_shape": output("object_shape", "application/octet-stream")},
            parameter_schema=OBJECT_SHAPE_PARAMETERS,
            resources=modal_resources(concurrency="object_shape"),
            retry=PAID_ONCE,
            quality=QualityPolicy(agent_rubric="object_shape"),
        ),
        StageDefinition(
            id="object_package",
            title="Package dynamic objects",
            kind=StageKind.JOIN,
            executor="object_package",
            inputs={
                "tracks": stage_output("object_lift", "object_track"),
                "shapes": stage_output("object_shape", "object_shape"),
                "descriptions": stage_output("object_describe", "object_prompt"),
                "viewer_world": stage_output("package_people", "viewer_world"),
                "cameras": stage_output("pi3x", "cameras"),
                "track_data": stage_output("tracks", "person_track_data"),
            },
            outputs={
                "objects_manifest": output(
                    "objects_manifest", "application/json", "wander.objects/2"
                )
            },
            quality=QualityPolicy(automatic_checks=("objects_manifest",)),
        ),
        StageDefinition(
            id="scale_fit",
            title="Fit world scale to reconstructed people",
            kind=StageKind.COMPUTE,
            executor="scale_fit",
            inputs={
                "source": run_input("source_video"),
                "world": stage_output("marble_video", "world_splat"),
                "world_receipt": stage_output("marble_video", "world_receipt"),
                "cameras": stage_output("pi3x", "cameras"),
                "point_clouds": stage_output("pi3x", "point_clouds"),
                "pi3x_aux": stage_output("pi3x", "pi3x_aux"),
                "viewer_world": stage_output("package_people", "viewer_world"),
            },
            outputs={"scale_fit": output("scale_fit", "application/json")},
            resources=local_resources(3600),
            quality=QualityPolicy(automatic_checks=("scale_ratio", "scale_spread")),
        ),
        StageDefinition(
            id="place_fit",
            title="Fit people and objects into the world",
            kind=StageKind.COMPUTE,
            executor="place_fit",
            inputs={
                "source": run_input("source_video"),
                "scale": stage_output("scale_fit", "scale_fit"),
                "world": stage_output("marble_video", "world_splat"),
                "world_receipt": stage_output("marble_video", "world_receipt"),
                "viewer_world": stage_output("package_people", "viewer_world"),
            },
            outputs={"placement": output("placement", "application/json")},
            resources=local_resources(1800),
            quality=QualityPolicy(automatic_checks=("room_size", "contact_residual")),
        ),
        StageDefinition(
            id="anchors",
            title="Check external scale and placement anchors",
            kind=StageKind.COMPUTE,
            executor="anchors",
            inputs={
                "placement": stage_output("place_fit", "placement"),
                "source": run_input("source_video"),
                "world": stage_output("marble_video", "world_splat"),
                "world_receipt": stage_output("marble_video", "world_receipt"),
                "scale": stage_output("scale_fit", "scale_fit"),
                "viewer_world": stage_output("package_people", "viewer_world"),
                "cameras": stage_output("pi3x", "cameras"),
                "point_clouds": stage_output("pi3x", "point_clouds"),
                "pi3x_aux": stage_output("pi3x", "pi3x_aux"),
            },
            outputs={"anchor_report": output("anchor_report", "application/json")},
            resources=external_resources(1800, "openai"),
            retry=RetryPolicy(paid=True),
            quality=QualityPolicy(automatic_checks=("anchor_consistency",)),
        ),
        StageDefinition(
            id="finetune",
            title="Fine-tune generated world",
            kind=StageKind.COMPUTE,
            executor="finetune",
            inputs={
                "source": run_input("source_video"),
                "world": stage_output("marble_video", "world_splat"),
                "world_receipt": stage_output("marble_video", "world_receipt"),
                "scale": stage_output("scale_fit", "scale_fit"),
                "placement": stage_output("place_fit", "placement"),
                "viewer_world": stage_output("package_people", "viewer_world"),
                "cameras": stage_output("pi3x", "cameras"),
                "point_clouds": stage_output("pi3x", "point_clouds"),
                "pi3x_aux": stage_output("pi3x", "pi3x_aux"),
            },
            outputs={
                "finetuned_world": output("finetuned_world", "application/octet-stream"),
                "finetune_receipt": output("finetune_receipt", "application/json"),
            },
            resources=external_resources(28800, "finetune"),
            retry=PAID_ONCE,
            quality=QualityPolicy(agent_rubric="finetuned_world", human_approval=True),
        ),
        StageDefinition(
            id="verify",
            title="Render and review static world fidelity",
            kind=StageKind.HUMAN,
            inputs={
                "world": stage_output("marble_video", "world_splat"),
                "scale": stage_output("scale_fit", "scale_fit"),
                "source": run_input("source_video"),
            },
            outputs={"quality_review": output("quality_review", "application/json")},
            quality=QualityPolicy(
                automatic_checks=("offline_review_binding",),
                agent_rubric="static_world",
                human_approval=True,
            ),
        ),
    ]
    return {stage.id: stage for stage in stages}
