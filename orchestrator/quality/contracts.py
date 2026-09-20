"""Immutable contracts for stage quality rubrics and their review records."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from orchestrator.contracts import Contract, Evidence, Identifier, QualityVerdict

QUALITY_RUBRIC_SCHEMA = "wander.quality-rubric/1"
QUALITY_REVIEW_SCHEMA = "wander.quality-review/1"
PROMOTION_SCHEMA = "wander.promotion-decision/1"


class QualityLayer(StrEnum):
    AUTOMATIC = "automatic"
    AGENT = "agent"
    HUMAN = "human"


class EvidenceRequirement(Contract):
    """Artifacts needed to make one criterion independently auditable."""

    role: Identifier
    minimum: int = Field(default=1, ge=1)
    from_attempt: bool = True


class RubricCriterion(Contract):
    id: Identifier
    description: str = Field(min_length=1)
    evidence: tuple[EvidenceRequirement, ...]

    @model_validator(mode="after")
    def require_unique_evidence_roles(self) -> RubricCriterion:
        roles = [requirement.role for requirement in self.evidence]
        if not roles:
            raise ValueError("criteria require at least one evidence requirement")
        if len(roles) != len(set(roles)):
            raise ValueError("criterion evidence roles must be unique")
        return self


class QualityRubric(Contract):
    schema_version: Literal["wander.quality-rubric/1"] = QUALITY_RUBRIC_SCHEMA
    id: Identifier
    stage_id: Identifier
    layer: QualityLayer
    criteria: tuple[RubricCriterion, ...]

    @model_validator(mode="after")
    def require_unique_criteria(self) -> QualityRubric:
        criterion_ids = [criterion.id for criterion in self.criteria]
        if not criterion_ids:
            raise ValueError("rubrics require at least one criterion")
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("rubric criterion ids must be unique")
        return self


class StageQualityContract(Contract):
    stage_id: Identifier
    automatic: tuple[QualityRubric, ...] = ()
    agent: QualityRubric | None = None
    human: QualityRubric | None = None

    @model_validator(mode="after")
    def validate_rubric_ownership_and_layers(self) -> StageQualityContract:
        rubrics = (*self.automatic, self.agent, self.human)
        present = [rubric for rubric in rubrics if rubric is not None]
        if any(rubric.stage_id != self.stage_id for rubric in present):
            raise ValueError("all rubrics must belong to the quality contract stage")
        if any(rubric.layer != QualityLayer.AUTOMATIC for rubric in self.automatic):
            raise ValueError("automatic rubrics must use the automatic layer")
        if self.agent is not None and self.agent.layer != QualityLayer.AGENT:
            raise ValueError("agent rubric must use the agent layer")
        if self.human is not None and self.human.layer != QualityLayer.HUMAN:
            raise ValueError("human rubric must use the human layer")
        ids = [rubric.id for rubric in present]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric ids must be unique within a stage")
        return self

    @property
    def ordered_rubrics(self) -> tuple[QualityRubric, ...]:
        return (
            *self.automatic,
            *((self.agent,) if self.agent is not None else ()),
            *((self.human,) if self.human is not None else ()),
        )


class CriterionFinding(Contract):
    criterion_id: Identifier
    passed: bool
    rationale: str = Field(min_length=1)
    evidence: tuple[Evidence, ...]


class QualityReview(Contract):
    schema_version: Literal["wander.quality-review/1"] = QUALITY_REVIEW_SCHEMA
    rubric_id: Identifier
    stage_id: Identifier
    attempt_id: Identifier
    layer: QualityLayer
    findings: tuple[CriterionFinding, ...]
    decided_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_findings(self) -> QualityReview:
        criterion_ids = [finding.criterion_id for finding in self.findings]
        if not criterion_ids:
            raise ValueError("reviews require at least one finding")
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("review findings must have unique criterion ids")
        return self

    @property
    def passed(self) -> bool:
        return all(finding.passed for finding in self.findings)


class PromotionDecision(Contract):
    schema_version: Literal["wander.promotion-decision/1"] = PROMOTION_SCHEMA
    stage_id: Identifier
    attempt_id: Identifier
    verdict: QualityVerdict
    promote: bool
    rationale: str = Field(min_length=1)
    next_layer: QualityLayer | None = None
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def protect_promotion(self) -> PromotionDecision:
        if self.promote != (self.verdict == QualityVerdict.PASS):
            raise ValueError("only pass verdicts promote an attempt")
        if (self.verdict == QualityVerdict.REQUEST_HUMAN) != (
            self.next_layer == QualityLayer.HUMAN
        ):
            raise ValueError("only request_human verdicts may select the human layer")
        return self
