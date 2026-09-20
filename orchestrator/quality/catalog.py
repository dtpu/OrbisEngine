"""Build stage-specific QA contracts from the declarative stage registry."""

from __future__ import annotations

from collections.abc import Iterable

from orchestrator.contracts import StageDefinition
from orchestrator.stages.registry import stage_registry

from .contracts import (
    EvidenceRequirement,
    QualityLayer,
    QualityRubric,
    RubricCriterion,
    StageQualityContract,
)


def _criterion(
    criterion_id: str,
    description: str,
    roles: Iterable[str],
    *,
    from_attempt: bool = True,
) -> RubricCriterion:
    return RubricCriterion(
        id=criterion_id,
        description=description,
        evidence=tuple(EvidenceRequirement(role=role, from_attempt=from_attempt) for role in roles),
    )


AUTOMATIC_CRITERIA: dict[str, tuple[str, tuple[str, ...]]] = {
    "source_identity": (
        "Source identity and metadata are internally consistent",
        ("source_metadata",),
    ),
    "shot_continuity": ("Selected shots satisfy continuity constraints", ("shots",)),
    "image_readable": ("The image artifact decodes successfully", ("clean_frame",)),
    "video_readable": ("The video artifact decodes successfully", ("clean_video",)),
    "mask_report": ("Mask coverage is measured and reported", ("person_masks", "clean_report")),
    "images_readable": ("Every selected image decodes successfully", ("clean_frames",)),
    "prompt_grounding": ("The prompt is traceable to visible source evidence", ("world_prompt",)),
    "world_mode_measurements": ("World input selection includes measured views", ("world_mode",)),
    "provider_operation": ("A durable provider operation receipt exists", ("provider_operation",)),
    "marble_receipt": ("World recovery matches the submitted operation", ("world_receipt",)),
    "world_readable": ("The recovered world artifact can be parsed", ("world_splat",)),
    "pose_continuity": ("Estimated camera poses are temporally continuous", ("cameras",)),
    "person_depth": ("Person depth estimates include measured diagnostics", ("motion_report",)),
    "frame_mapping": ("Camera frames map deterministically to playback time", ("frame_alignment",)),
    "identity_stability": ("Person identities remain stable over their tracks", ("person_tracks",)),
    "person_mask": ("The person reference contains a usable subject mask", ("prepared_person",)),
    "canonical_person": ("The canonical person artifact is complete", ("canonical_person",)),
    "recovery_receipt": ("Recovery records bind outputs to their operation", ("recovery_receipt",)),
    "motion_completeness": ("Motion covers the required playback interval", ("person_motion",)),
    "package_integrity": (
        "The viewer package and people manifest agree",
        ("viewer_world", "people_manifest"),
    ),
    "audio_timeline": ("Audio timing is bound to the source timeline", ("audio_manifest",)),
    "object_track": ("The object track contains valid 3D samples", ("object_track",)),
    "objects_manifest": ("The object manifest references packaged objects", ("objects_manifest",)),
    "scale_ratio": ("The fitted scale ratio is finite and bounded", ("scale_fit",)),
    "scale_spread": ("Scale observations have an acceptable spread", ("scale_fit",)),
    "room_size": ("Placed room dimensions are physically plausible", ("placement",)),
    "contact_residual": ("Ground contact residuals are within tolerance", ("placement",)),
    "anchor_consistency": ("External anchors agree with fitted placement", ("anchor_report",)),
    "offline_review_binding": (
        "The review is bound to the exact static-world artifacts",
        ("quality_review",),
    ),
}


def _agent_criteria(stage: StageDefinition, rubric_name: str) -> tuple[RubricCriterion, ...]:
    output_roles = {output.role for output in stage.outputs.values()}
    from_attempt = stage.id != "clean_review"
    mappings: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
        "cleaned_scene": (
            ("removal", "Removed subjects leave no material residuals", ("clean_report",)),
            (
                "preservation",
                "Background structure and appearance remain coherent",
                ("clean_frames",) if "clean_frames" in output_roles else ("clean_frame",),
            ),
        ),
        "person_tracks": (
            ("identity", "Tracks neither swap identities nor merge people", ("person_tracks",)),
        ),
        "person_reference": (
            (
                "coverage",
                "The reference preserves the complete visible person",
                ("prepared_person",),
            ),
        ),
        "person_motion": (
            (
                "fidelity",
                "Motion follows the recorded action without invented beats",
                ("person_motion",),
            ),
        ),
        "dynamic_objects": (
            (
                "selection",
                "Detected objects are dynamic and relevant to the action",
                ("detected_objects",),
            ),
        ),
        "object_segmentation": (
            (
                "silhouette",
                "Object masks follow the object without person leakage",
                ("object_masks",),
            ),
        ),
        "object_shape": (
            (
                "appearance",
                "Shape and appearance agree with source observations",
                ("object_shape",),
            ),
        ),
        "finetuned_world": (
            (
                "fidelity",
                "Fine-tuning improves fidelity without damaging geometry",
                ("finetuned_world",),
            ),
        ),
        "static_world": (
            (
                "source_fidelity",
                "Static-world evidence agrees with the recorded source",
                ("source_video", "world_splat", "scale_fit"),
            ),
        ),
    }
    if rubric_name not in mappings:
        raise ValueError(f"no agent rubric contract for {rubric_name!r}")
    if rubric_name == "static_world":
        from_attempt = False
    return tuple(
        _criterion(criterion_id, description, roles, from_attempt=from_attempt)
        for criterion_id, description, roles in mappings[rubric_name]
    )


HUMAN_EVIDENCE: dict[str, tuple[str, ...]] = {
    "clean_review": ("approval",),
    "audio_reviewed": ("audio_manifest",),
    "finetune": ("finetuned_world", "finetune_receipt"),
    "verify": ("quality_review",),
}


def quality_contract_for(stage: StageDefinition) -> StageQualityContract:
    automatic = []
    for check in stage.quality.automatic_checks:
        try:
            description, roles = AUTOMATIC_CRITERIA[check]
        except KeyError as error:
            raise ValueError(f"no automatic QA contract for {check!r}") from error
        automatic.append(
            QualityRubric(
                id=f"{stage.id}.automatic.{check}",
                stage_id=stage.id,
                layer=QualityLayer.AUTOMATIC,
                criteria=(_criterion(check, description, roles),),
            )
        )

    agent = None
    if stage.quality.agent_rubric is not None:
        agent = QualityRubric(
            id=f"{stage.id}.agent.{stage.quality.agent_rubric}",
            stage_id=stage.id,
            layer=QualityLayer.AGENT,
            criteria=_agent_criteria(stage, stage.quality.agent_rubric),
        )

    human = None
    if stage.quality.human_approval:
        try:
            roles = HUMAN_EVIDENCE[stage.id]
        except KeyError as error:
            raise ValueError(f"no human QA contract for stage {stage.id!r}") from error
        human = QualityRubric(
            id=f"{stage.id}.human.approval",
            stage_id=stage.id,
            layer=QualityLayer.HUMAN,
            criteria=(
                _criterion(
                    "approval",
                    "An authorized reviewer explicitly approves these exact artifacts",
                    roles,
                ),
            ),
        )

    return StageQualityContract(
        stage_id=stage.id,
        automatic=tuple(automatic),
        agent=agent,
        human=human,
    )


def quality_catalog() -> dict[str, StageQualityContract]:
    """Return QA contracts for every registered stage, including no-op contracts."""

    return {stage_id: quality_contract_for(stage) for stage_id, stage in stage_registry().items()}
