"""Deterministic policy gate between agent proposals and workflow mutations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from orchestrator.contracts import StageDefinition


class ProposalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    reasons: tuple[str, ...]
    parameter_fingerprint: str | None = None


@dataclass(frozen=True)
class PolicyContext:
    dependency_statuses: dict[str, str]
    prior_parameter_fingerprints: tuple[str, ...] = ()
    attempts_used: int = 0
    attempt_ceiling: int = 3
    unresolved_provider_claim: bool = False
    remaining_cost: float | None = None


class ProposalPolicy:
    CONTROL_PARAMETERS = {"hypothesis", "estimated_cost_usd"}

    def evaluate(
        self,
        definition: StageDefinition,
        parameters: dict[str, Any],
        context: PolicyContext,
    ) -> ProposalDecision:
        reasons = []
        incomplete = sorted(
            node for node, status in context.dependency_statuses.items() if status != "succeeded"
        )
        if incomplete:
            reasons.append(f"dependencies are not selected: {', '.join(incomplete)}")
        if context.attempts_used >= context.attempt_ceiling:
            reasons.append("attempt ceiling is exhausted")
        if context.unresolved_provider_claim:
            reasons.append("provider outcome must be reconciled before retry")
        schema = definition.parameter_schema
        schema_reasons = validate_parameters(
            parameters,
            schema,
            allowed_controls=self.CONTROL_PARAMETERS,
        )
        reasons.extend(schema_reasons)
        hypothesis = parameters.get("hypothesis")
        if context.attempts_used and definition.retry.requires_changed_hypothesis:
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                reasons.append("retry requires a changed hypothesis")
        fingerprint = parameter_fingerprint(
            {
                key: value
                for key, value in parameters.items()
                if key in schema.get("properties", {}) or key in self.CONTROL_PARAMETERS
            }
        )
        if (
            context.attempts_used
            and definition.retry.requires_changed_parameters
            and fingerprint in context.prior_parameter_fingerprints
        ):
            reasons.append("retry parameters are unchanged")
        estimate = parameters.get("estimated_cost_usd")
        if definition.retry.paid:
            if context.remaining_cost is not None and not isinstance(estimate, (int, float)):
                reasons.append("paid proposal needs estimated_cost_usd")
            elif (
                context.remaining_cost is not None
                and isinstance(estimate, (int, float))
                and estimate > context.remaining_cost
            ):
                reasons.append("paid proposal exceeds remaining cost budget")
        return ProposalDecision(
            accepted=not reasons,
            reasons=tuple(reasons),
            parameter_fingerprint=fingerprint if not schema_reasons else None,
        )


def parameter_fingerprint(parameters: dict[str, Any]) -> str:
    execution = {
        key: value
        for key, value in parameters.items()
        if key not in {"hypothesis", "estimated_cost_usd"}
    }
    encoded = json.dumps(execution, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_parameters(
    parameters: dict[str, Any],
    schema: dict[str, Any],
    *,
    allowed_controls: set[str],
) -> list[str]:
    properties = schema.get("properties", {})
    reasons = []
    unknown = sorted(set(parameters) - set(properties) - allowed_controls)
    if unknown:
        reasons.append(f"undeclared parameters: {', '.join(unknown)}")
    for name in schema.get("required", []):
        if name not in parameters:
            reasons.append(f"missing required parameter: {name}")
    for name, value in parameters.items():
        rule = properties.get(name)
        if rule is None:
            continue
        expected = rule.get("type")
        valid_type = {
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "string": lambda item: isinstance(item, str),
        }.get(expected, lambda item: True)
        if not valid_type(value):
            reasons.append(f"{name} must be {expected}")
            continue
        if "minimum" in rule and value < rule["minimum"]:
            reasons.append(f"{name} is below minimum {rule['minimum']}")
        if "maximum" in rule and value > rule["maximum"]:
            reasons.append(f"{name} is above maximum {rule['maximum']}")
        if "enum" in rule and value not in rule["enum"]:
            reasons.append(f"{name} is not one of the allowed values")
    return reasons
