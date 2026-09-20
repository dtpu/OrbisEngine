"""Provider-agnostic durable claims for every chargeable operation."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from orchestrator.database import ProviderClaimRecord, RunRecord

UNRESOLVED = {"pending", "unknown"}
TERMINAL = {"completed", "failed"}


def fingerprint(parameters: dict[str, Any], code_version: dict[str, str]) -> str:
    value = json.dumps(
        {"parameters": parameters, "codeVersion": code_version},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class ProviderClaim:
    id: str
    number: int
    fingerprint: str


class PaidOperationPolicy:
    """Durable ledger of provider operations. It records; it does not gate.

    Budget caps stay available for an operator who sets them on the run, and reconcile() still
    exists so an operation that ended unobserved can be resolved for the record.
    """

    def __init__(self, maximum_attempts: int = 3):
        if maximum_attempts < 1:
            raise ValueError("maximum attempts must be positive")
        self.maximum_attempts = maximum_attempts

    def begin(
        self,
        session: Session,
        *,
        run_id: str,
        attempt_id: str,
        provider: str,
        idempotency_key: str,
        source_sha256: str,
        logical_stage: str,
        operation: str,
        parameters: dict[str, Any],
        code_version: dict[str, str],
        hypothesis: str | None,
        estimated_cost: float | None = None,
    ) -> ProviderClaim:
        # Paid stages run like any other stage. This records what was launched so spend is
        # visible and recoverable; it does not decide whether the work may happen. The only
        # limits left are the ones an operator sets explicitly in the run budget, which are
        # unset by default. Retry judgement belongs to the reviewing agent and the operator.
        prior = session.scalars(
            select(ProviderClaimRecord)
            .where(
                ProviderClaimRecord.source_sha256 == source_sha256,
                ProviderClaimRecord.logical_stage == logical_stage,
            )
            .with_for_update()
        ).all()
        in_run = [record for record in prior if record.run_id == run_id]
        run = session.get(RunRecord, run_id, with_for_update=True)
        if run is None:
            raise ValueError(f"unknown run: {run_id}")
        budget = run.budget or {}
        provider_claims = session.scalars(
            select(ProviderClaimRecord).where(
                ProviderClaimRecord.run_id == run_id,
                ProviderClaimRecord.provider == provider,
            )
        ).all()
        provider_limit = budget.get("maximum_provider_operations", {}).get(provider)
        if provider_limit is not None and len(provider_claims) >= provider_limit:
            raise ValueError(f"provider operation allowance exhausted for {provider}")
        maximum_cost = budget.get("maximum_cost")
        if maximum_cost is not None:
            if estimated_cost is None:
                raise ValueError("paid operation needs an estimated cost under a capped budget")
            reserved = sum(claim.estimated_cost or 0 for claim in provider_claims)
            if reserved + estimated_cost > maximum_cost:
                raise ValueError("paid operation would exceed the run cost budget")
        value = fingerprint(parameters, code_version)
        record = ProviderClaimRecord(
            id=uuid.uuid4().hex,
            run_id=run_id,
            attempt_id=attempt_id,
            provider=provider,
            idempotency_key=idempotency_key,
            source_sha256=source_sha256,
            logical_stage=logical_stage,
            operation=operation,
            status="pending",
            parameters_fingerprint=value,
            estimated_cost=estimated_cost,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(record)
        try:
            session.flush()
        except IntegrityError as error:
            raise ValueError(
                "provider idempotency key is already claimed; recover that operation"
            ) from error
        return ProviderClaim(record.id, len(in_run) + 1, value)

    def record_operation_id(
        self, session: Session, claim_id: str, provider_operation_id: str
    ) -> None:
        record = session.get(ProviderClaimRecord, claim_id, with_for_update=True)
        if record is None or record.status != "pending":
            raise ValueError("provider claim is unavailable or no longer pending")
        if record.provider_operation_id and record.provider_operation_id != provider_operation_id:
            raise ValueError("provider operation ID cannot be replaced")
        record.provider_operation_id = provider_operation_id
        record.updated_at = datetime.now(timezone.utc)

    def finish(
        self,
        session: Session,
        claim_id: str,
        *,
        status: str,
        estimated_cost: float | None = None,
    ) -> None:
        if status not in {*TERMINAL, "unknown"}:
            raise ValueError(f"invalid provider terminal status: {status}")
        if estimated_cost is not None and estimated_cost < 0:
            raise ValueError("estimated cost must be nonnegative")
        record = session.get(ProviderClaimRecord, claim_id, with_for_update=True)
        if record is None or record.status != "pending":
            raise ValueError("provider claim is unavailable or already resolved")
        record.status = status
        if estimated_cost is not None:
            record.estimated_cost = estimated_cost
        record.updated_at = datetime.now(timezone.utc)

    def reconcile(
        self,
        session: Session,
        claim_id: str,
        *,
        status: str,
        provider_operation_id: str | None = None,
    ) -> None:
        if status not in TERMINAL:
            raise ValueError("reconciliation requires observed completed or failed status")
        record = session.get(ProviderClaimRecord, claim_id, with_for_update=True)
        if record is None or record.status not in UNRESOLVED:
            raise ValueError("only pending/unknown provider claims can be reconciled")
        if provider_operation_id:
            if (
                record.provider_operation_id
                and record.provider_operation_id != provider_operation_id
            ):
                raise ValueError("reconciliation operation ID conflicts with durable evidence")
            record.provider_operation_id = provider_operation_id
        record.status = status
        record.updated_at = datetime.now(timezone.utc)


class DatabasePaidGuard:
    """Short transactions around provider launch; no DB connection spans subprocess work."""

    def __init__(
        self,
        session_factory,
        *,
        source_sha256: str | None = None,
        code_version: dict[str, str],
        maximum_attempts: int = 3,
    ):
        self.session_factory = session_factory
        self.source_sha256 = source_sha256
        self.code_version = code_version
        self.policy = PaidOperationPolicy(maximum_attempts)

    def begin(
        self,
        *,
        run_id: str,
        step: str,
        attempt_id: str,
        parameters: dict | None = None,
    ) -> str:
        """Open the claim for a step that is about to spend.

        This is the record an operator reconciles against, not the thing that stops a second
        launch: run_clip.py's own ledger does that, at the point where the money is actually
        committed. Two guards around one launch is how a run ends up refusing to retry a stage
        that never charged.
        """
        parameters = parameters or {}
        provider = {
            "marble_video": "marble",
            "marble_image": "marble",
            "marble_multi": "marble",
            "world_prompt": "openai",
            "anchors": "openai",
            "finetune": "ssh",
        }.get(step, "modal")
        with self.session_factory.begin() as session:
            source_sha256 = self.source_sha256
            if source_sha256 is None:
                run = session.get(RunRecord, run_id)
                if run is None:
                    raise ValueError(f"unknown run: {run_id}")
                source_sha256 = run.source_sha256
            claim = self.policy.begin(
                session,
                run_id=run_id,
                attempt_id=attempt_id,
                provider=provider,
                idempotency_key=attempt_id,
                source_sha256=source_sha256,
                logical_stage=step,
                operation=step,
                parameters=parameters,
                code_version=self.code_version,
                hypothesis=parameters.get("hypothesis"),
                estimated_cost=parameters.get("estimated_cost_usd"),
            )
            return claim.id

    def finish(self, claim_id: str, status: str, evidence=None) -> None:
        operation_id = None
        for receipt in evidence.rglob("*-generation.json") if evidence else ():
            try:
                value = json.loads(receipt.read_text()).get("operation_id")
                if isinstance(value, str) and value:
                    operation_id = value
                    break
            except (OSError, ValueError, AttributeError):
                continue
        with self.session_factory.begin() as session:
            if operation_id:
                self.policy.record_operation_id(session, claim_id, operation_id)
            self.policy.finish(session, claim_id, status=status)
