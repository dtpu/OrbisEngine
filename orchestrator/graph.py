"""Deterministic graph instantiation and artifact-aware readiness evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.contracts import (
    ArtifactBinding,
    NodeStatus,
    StageDefinition,
    StageKind,
)
from orchestrator.stages.registry import GraphOptions, stage_registry

GRAPH_VERSION = "wander.generation-graph/1"
TERMINAL_FAILURES = {NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.CANCELED}
TERMINAL_SUCCESS = {NodeStatus.SUCCEEDED, NodeStatus.SKIPPED}


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    stage_type: str
    definition: StageDefinition
    dependencies: tuple[str, ...]
    status: NodeStatus = NodeStatus.QUEUED
    selected_attempt_id: str | None = None
    selected_artifacts: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    blocked_reason: str | None = None
    parent_node_id: str | None = None
    branch_key: str | None = None


class BranchArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    artifact_id: str


class ShotBranch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    clip_artifact_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)


class ChildWorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: str
    parent_run_id: str
    branch_key: str
    clip_artifact_id: str
    canonical_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_key: str
    start_seconds: float
    end_seconds: float


class RunGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = GRAPH_VERSION
    options: GraphOptions
    nodes: dict[str, GraphNode]

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"nodes": {"__all__": {"status"}}})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def select_attempt(
        self, node_id: str, attempt_id: str, artifacts: dict[str, tuple[str, ...]]
    ) -> None:
        node = self.nodes[node_id]
        missing = {
            contract.role
            for contract in node.definition.outputs.values()
            if contract.required and not artifacts.get(contract.role)
        }
        if missing:
            raise ValueError(f"{node_id} attempt lacks required artifact roles: {sorted(missing)}")
        unknown = set(artifacts) - {contract.role for contract in node.definition.outputs.values()}
        if unknown:
            raise ValueError(f"{node_id} attempt has undeclared artifact roles: {sorted(unknown)}")
        node.selected_attempt_id = attempt_id
        node.selected_artifacts = artifacts
        node.status = NodeStatus.SUCCEEDED
        node.blocked_reason = None

    def set_status(self, node_id: str, status: NodeStatus, reason: str | None = None) -> None:
        node = self.nodes[node_id]
        node.status = status
        node.blocked_reason = reason

    def retry_node(self, node_id: str) -> tuple[str, ...]:
        """Clear selections for a node and all descendants while retaining their attempts."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        affected = {node_id}
        changed = True
        while changed:
            changed = False
            for candidate in self.nodes.values():
                if candidate.id not in affected and affected.intersection(candidate.dependencies):
                    affected.add(candidate.id)
                    changed = True
        for identifier in affected:
            node = self.nodes[identifier]
            node.status = NodeStatus.QUEUED
            node.selected_attempt_id = None
            node.selected_artifacts = {}
            node.blocked_reason = None
        return tuple(sorted(affected))

    def evaluate(self, run_inputs: set[str]) -> tuple[str, ...]:
        """Advance queued nodes to ready/waiting/blocked and return runnable node IDs."""
        changed = True
        while changed:
            changed = False
            for node in self.nodes.values():
                if node.status not in {NodeStatus.QUEUED, NodeStatus.READY}:
                    continue
                failed = [
                    dependency
                    for dependency in node.dependencies
                    if self.nodes[dependency].status in TERMINAL_FAILURES
                ]
                if failed:
                    node.status = NodeStatus.BLOCKED
                    node.blocked_reason = f"dependency did not finish: {sorted(failed)}"
                    changed = True
                    continue
                if any(
                    self.nodes[dependency].status not in TERMINAL_SUCCESS
                    for dependency in node.dependencies
                ):
                    if node.status != NodeStatus.QUEUED:
                        node.status = NodeStatus.QUEUED
                        changed = True
                    continue
                missing = self._missing_inputs(node, run_inputs)
                if missing:
                    node.status = NodeStatus.BLOCKED
                    node.blocked_reason = f"required artifacts are unavailable: {sorted(missing)}"
                    changed = True
                    continue
                target = (
                    NodeStatus.WAITING_HUMAN
                    if node.definition.kind == StageKind.HUMAN
                    else NodeStatus.READY
                )
                if node.status != target:
                    node.status = target
                    changed = True
        return tuple(
            sorted(node.id for node in self.nodes.values() if node.status == NodeStatus.READY)
        )

    def _missing_inputs(self, node: GraphNode, run_inputs: set[str]) -> set[str]:
        missing = set()
        for binding in node.definition.inputs.values():
            if binding.source == "run_input":
                if binding.role not in run_inputs:
                    missing.add(f"run:{binding.role}")
                continue
            producer = self.nodes[binding.stage_id]
            if not producer.selected_artifacts.get(binding.role):
                missing.add(f"{binding.stage_id}:{binding.role}")
        return missing

    def expand_people(self, tracks: tuple[BranchArtifact, ...]) -> tuple[str, ...]:
        if "tracks" not in self.nodes or self.nodes["tracks"].status != NodeStatus.SUCCEEDED:
            raise ValueError("tracks must have a selected successful attempt before expansion")
        if not tracks:
            raise ValueError("person expansion requires at least one accepted track")
        if len(tracks) > self.options.people:
            raise ValueError(
                f"{len(tracks)} accepted tracks exceed the configured cap {self.options.people}"
            )
        _unique_branch_keys(tracks)
        registry = stage_registry()
        motion_nodes = []
        created = []
        for track in tracks:
            role = f"person_track:{track.key}"
            self.nodes["tracks"].selected_artifacts[role] = (track.artifact_id,)
            prep = f"person_prep:{track.key}"
            frozen = f"lhm_frozen:{track.key}"
            motion = f"lhm_motion:{track.key}"
            self._add_dynamic(
                registry,
                "person_prep",
                prep,
                bindings={"track": ("tracks", role)},
                parent="tracks",
                branch_key=track.key,
            )
            self._add_dynamic(
                registry,
                "lhm_frozen",
                frozen,
                bindings={"prepared_person": (prep, "prepared_person")},
                parent="tracks",
                branch_key=track.key,
            )
            self._add_dynamic(
                registry,
                "lhm_motion",
                motion,
                bindings={"canonical_person": (frozen, "canonical_person")},
                parent="tracks",
                branch_key=track.key,
            )
            motion_nodes.append(motion)
            created.extend((prep, frozen, motion))
        package_inputs = {
            f"motion_{track.key}": ArtifactBinding(
                source="stage_output",
                stage_id=motion,
                role="person_motion",
            )
            for track, motion in zip(tracks, motion_nodes)
        }
        package = registry["package_people"].model_copy(
            update={
                "inputs": {
                    **package_inputs,
                    "alignment": ArtifactBinding(
                        source="stage_output",
                        stage_id="frame_align",
                        role="frame_alignment",
                    ),
                    "source": registry["package_people"].inputs["source"],
                }
            }
        )
        self._insert_node(package, "package_people", depends_on=tuple(motion_nodes))
        created.append("package_people")
        created.extend(self._add_post_package_nodes(registry))
        return tuple(created)

    def expand_objects(self, objects: tuple[BranchArtifact, ...]) -> tuple[str, ...]:
        detector = self.nodes.get("object_detect")
        if detector is None or detector.status != NodeStatus.SUCCEEDED:
            raise ValueError("object detection must have a selected attempt before expansion")
        _unique_branch_keys(objects)
        if not objects:
            return ()
        registry = stage_registry()
        tracks: dict[str, str] = {}
        shapes: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        created = []
        for item in objects:
            role = f"detected_object:{item.key}"
            detector.selected_artifacts[role] = (item.artifact_id,)
            lift = f"object_lift:{item.key}"
            describe = f"object_describe:{item.key}"
            shape = f"object_shape:{item.key}"
            self._add_dynamic(
                registry,
                "object_lift",
                lift,
                bindings={"object": ("object_detect", role)},
                parent="object_detect",
                branch_key=item.key,
            )
            self._add_dynamic(
                registry,
                "object_describe",
                describe,
                bindings={"object": ("object_detect", role)},
                parent="object_detect",
                branch_key=item.key,
            )
            self._add_dynamic(
                registry,
                "object_shape",
                shape,
                bindings={
                    "prompt": (describe, "object_prompt"),
                    "image": (describe, "object_image"),
                },
                parent="object_detect",
                branch_key=item.key,
            )
            tracks[item.key] = lift
            shapes[item.key] = shape
            descriptions[item.key] = describe
            created.extend((lift, describe, shape))
        package = registry["object_package"].model_copy(
            update={
                "inputs": {
                    **{
                        f"track_{key}": ArtifactBinding(
                            source="stage_output", stage_id=node, role="object_track"
                        )
                        for key, node in tracks.items()
                    },
                    **{
                        f"shape_{key}": ArtifactBinding(
                            source="stage_output", stage_id=node, role="object_shape"
                        )
                        for key, node in shapes.items()
                    },
                    **{
                        f"description_{key}": ArtifactBinding(
                            source="stage_output", stage_id=node, role="object_prompt"
                        )
                        for key, node in descriptions.items()
                    },
                    "viewer_world": ArtifactBinding(
                        source="stage_output",
                        stage_id="package_people",
                        role="viewer_world",
                    ),
                    "cameras": registry["object_package"].inputs["cameras"],
                    "track_data": registry["object_package"].inputs["track_data"],
                }
            }
        )
        self._insert_node(
            package,
            "object_package",
            depends_on=tuple((*tracks.values(), *shapes.values(), *descriptions.values())),
        )
        created.append("object_package")
        return tuple(created)

    def _add_post_package_nodes(self, registry: dict[str, StageDefinition]) -> tuple[str, ...]:
        created = []
        if self.options.reviewed_audio:
            self._add_dynamic(registry, "audio_reviewed", "audio_reviewed")
            created.append("audio_reviewed")
        if self.options.objects:
            self._add_dynamic(registry, "object_detect", "object_detect")
            created.append("object_detect")
        world_node = {
            "image": "marble_image",
            "video": "marble_video",
            "both": "marble_video",
            "multi": "marble_multi",
        }.get(self.options.marble)
        if world_node:
            binding = {
                "world": (world_node, "world_splat"),
                "world_receipt": (world_node, "world_receipt"),
            }
            self._add_dynamic(registry, "scale_fit", "scale_fit", bindings=binding)
            self._add_dynamic(registry, "place_fit", "place_fit", bindings=binding)
            self._add_dynamic(registry, "anchors", "anchors", bindings=binding)
            created.extend(("scale_fit", "place_fit", "anchors"))
            if self.options.finetune:
                self._add_dynamic(registry, "finetune", "finetune", bindings=binding)
                created.append("finetune")
            self._add_dynamic(registry, "verify", "verify", bindings=binding)
            created.append("verify")
        return tuple(created)

    def _add_dynamic(
        self,
        registry: dict[str, StageDefinition],
        stage_type: str,
        node_id: str,
        *,
        bindings: dict[str, tuple[str, str]] | None = None,
        parent: str | None = None,
        branch_key: str | None = None,
    ) -> None:
        definition = _concrete_stage(registry, stage_type, node_id=node_id, bindings=bindings)
        self._insert_node(
            definition,
            stage_type,
            parent=parent,
            branch_key=branch_key,
        )

    def _insert_node(
        self,
        definition: StageDefinition,
        stage_type: str,
        *,
        depends_on: tuple[str, ...] = (),
        parent: str | None = None,
        branch_key: str | None = None,
    ) -> None:
        if definition.id in self.nodes:
            raise ValueError(f"duplicate graph node {definition.id}")
        dependencies = {
            binding.stage_id
            for binding in definition.inputs.values()
            if binding.source == "stage_output"
        }
        dependencies.update(depends_on)
        dependencies.add("admission")
        missing = dependencies - self.nodes.keys()
        if missing:
            raise ValueError(
                f"{definition.id} refers to nodes not yet instantiated: {sorted(missing)}"
            )
        self.nodes[definition.id] = GraphNode(
            id=definition.id,
            stage_type=stage_type,
            definition=definition,
            dependencies=tuple(sorted(dependencies)),
            parent_node_id=parent,
            branch_key=branch_key,
        )


def _concrete_stage(
    registry: dict[str, StageDefinition],
    stage_type: str,
    *,
    node_id: str | None = None,
    bindings: dict[str, tuple[str, str]] | None = None,
    remove_inputs: set[str] | None = None,
) -> StageDefinition:
    stage = registry[stage_type]
    inputs = dict(stage.inputs)
    for name in remove_inputs or set():
        inputs.pop(name, None)
    for name, (producer, role) in (bindings or {}).items():
        inputs[name] = ArtifactBinding(source="stage_output", stage_id=producer, role=role)
    return stage.model_copy(update={"id": node_id or stage_type, "inputs": inputs})


def _unique_branch_keys(items: tuple[BranchArtifact, ...]) -> None:
    keys = [item.key for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError("branch keys must be unique")


def child_workflows_for_shots(
    parent_run_id: str,
    canonical_source_sha256: str,
    shots: tuple[ShotBranch, ...],
) -> tuple[ChildWorkflowSpec, ...]:
    keys = [shot.key for shot in shots]
    if len(keys) != len(set(keys)):
        raise ValueError("shot branch keys must be unique")
    if any(shot.end_seconds <= shot.start_seconds for shot in shots):
        raise ValueError("shot end must be after its start")
    return tuple(
        ChildWorkflowSpec(
            workflow_id=f"{parent_run_id}:shot:{shot.key}",
            parent_run_id=parent_run_id,
            branch_key=shot.key,
            clip_artifact_id=shot.clip_artifact_id,
            canonical_source_sha256=canonical_source_sha256,
            budget_key=canonical_source_sha256,
            start_seconds=shot.start_seconds,
            end_seconds=shot.end_seconds,
        )
        for shot in shots
    )


def instantiate_graph(options: GraphOptions) -> RunGraph:
    registry = stage_registry()
    nodes: dict[str, GraphNode] = {}

    def add(
        stage_type: str,
        *,
        node_id: str | None = None,
        bindings: dict[str, tuple[str, str]] | None = None,
        remove_inputs: set[str] | None = None,
        depends_on: tuple[str, ...] = (),
    ) -> str:
        definition = _concrete_stage(
            registry,
            stage_type,
            node_id=node_id,
            bindings=bindings,
            remove_inputs=remove_inputs,
        )
        identifier = definition.id
        dependencies = {
            binding.stage_id
            for binding in definition.inputs.values()
            if binding.source == "stage_output"
        }
        dependencies.update(depends_on)
        if stage_type != "admission":
            dependencies.add("admission")
        if identifier in nodes:
            raise ValueError(f"duplicate graph node {identifier}")
        missing = dependencies - nodes.keys()
        if missing:
            raise ValueError(
                f"{identifier} refers to nodes not yet instantiated: {sorted(missing)}"
            )
        nodes[identifier] = GraphNode(
            id=identifier,
            stage_type=stage_type,
            definition=definition,
            dependencies=tuple(sorted(dependencies)),
        )
        return identifier

    add("admission")
    add("pi3x")
    add("frame_align")

    world_node = None
    if options.marble in {"image", "both"}:
        add("clean_first")
    if options.marble in {"video", "both", "none"}:
        add("clean", depends_on=("clean_first",) if options.marble == "both" else ())
    if options.marble != "none":
        add("world_prompt")
        if options.marble == "multi":
            add("world_mode")
            add("clean_multi")
            review_source, review_role = "clean_multi", "clean_report"
            review_frame_role = "clean_frames"
        elif options.marble == "image":
            review_source, review_role = "clean_first", "clean_report"
            review_frame_role = "clean_frame"
        else:
            review_source, review_role = "clean", "clean_report"
            review_frame_role = "clean_frame"
        add(
            "clean_review",
            bindings={
                "clean_report": (review_source, review_role),
                "clean_frame": (review_source, review_frame_role),
            },
            depends_on=("clean_first", "clean") if options.marble == "both" else (),
        )
        if options.marble in {"image", "both"}:
            add("marble_image_submit")
            add("marble_image")
        if options.marble in {"video", "both"}:
            add("marble_video_submit")
            add("marble_video")
        if options.marble == "multi":
            add("marble_multi_submit")
            add("marble_multi")
        world_node = {
            "image": "marble_image",
            "video": "marble_video",
            "both": "marble_video",
            "multi": "marble_multi",
        }[options.marble]

    if options.all_people or options.people > 1:
        add("tracks")
        # Person and package nodes are added after the track count is known.
    else:
        add("person_prep", remove_inputs={"track"})
        add("lhm_frozen")
        add("lhm_motion")
        add("package_people")
        if options.reviewed_audio:
            add("audio_reviewed")
        if world_node:
            world_binding = {
                "world": (world_node, "world_splat"),
                "world_receipt": (world_node, "world_receipt"),
            }
            add("scale_fit", bindings=world_binding)
            add("place_fit", bindings=world_binding)
            add("anchors", bindings=world_binding)
            if options.finetune:
                add("finetune", bindings=world_binding)
            add("verify", bindings=world_binding)

    return RunGraph(options=options, nodes=nodes)
