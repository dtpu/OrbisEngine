"""Provider-agnostic durable claims for every chargeable operation."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from orchestrator.database import AttemptRecord, ProviderClaimRecord, RunRecord

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
    """Bound source/stage executions and reserve run budgets before provider submission."""

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
        if estimated_cost is not None and (not math.isfinite(estimated_cost) or estimated_cost < 0):
            raise ValueError("estimated cost must be finite and nonnegative")
        value = fingerprint(parameters, code_version)
        # SQLite ignores FOR UPDATE. Reserve its writer before reading the allowance;
        # an existing write transaction already owns that reservation.
        connection = session.connection()
        if connection.dialect.name == "sqlite":
            driver = connection.connection.driver_connection
            if not driver.in_transaction:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
        # Lock a shared source owner even when no provider claim exists yet. Ordered locks
        # serialize new claims across candidate/run renames as well as within one run.
        session.scalars(
            select(RunRecord)
            .where(RunRecord.source_sha256 == source_sha256)
            .order_by(RunRecord.id)
            .with_for_update()
        ).all()
        run = session.get(RunRecord, run_id, with_for_update=True)
        if run is None:
            raise ValueError(f"unknown run: {run_id}")
        if run.source_sha256 != source_sha256:
            raise ValueError("provider claim must use the run's original source SHA-256")
        prior = session.scalars(
            select(ProviderClaimRecord)
            .where(
                ProviderClaimRecord.source_sha256 == source_sha256,
                ProviderClaimRecord.logical_stage == logical_stage,
            )
            .with_for_update()
        ).all()
        budget = run.budget or {}
        run_claims = session.scalars(
            select(ProviderClaimRecord).where(
                ProviderClaimRecord.run_id == run_id,
            )
        ).all()
        provider_claims = [claim for claim in run_claims if claim.provider == provider]
        provider_limit = budget.get("maximum_provider_operations", {}).get(provider)
        if provider_limit is not None and len(provider_claims) >= provider_limit:
            raise ValueError(f"provider operation allowance exhausted for {provider}")
        maximum_cost = budget.get("maximum_cost")
        if maximum_cost is not None:
            if estimated_cost is None:
                raise ValueError("paid operation needs an estimated cost under a capped budget")
            if any(claim.estimated_cost is None for claim in run_claims):
                raise ValueError(
                    "existing paid operations need estimated costs under a capped budget"
                )
            reserved = sum(claim.estimated_cost for claim in run_claims)
            if reserved + estimated_cost > maximum_cost:
                raise ValueError("paid operation would exceed the run cost budget")
        if any(record.status in UNRESOLVED for record in prior):
            raise ValueError("pending/unknown provider claim requires evidence reconciliation")
        limits = budget.get("maximum_attempts_by_stage", {})
        maximum = min(
            self.maximum_attempts,
            limits.get(logical_stage, self.maximum_attempts),
            limits.get(operation, self.maximum_attempts),
        )
        if len(prior) >= maximum:
            raise ValueError("paid source/stage execution allowance exhausted")
        retries = [record for record in prior if record.operation == operation]
        attempt = session.get(AttemptRecord, attempt_id)
        if attempt is None or attempt.run_id != run_id:
            raise ValueError("paid operation requires an existing attempt in its run")
        normalized = " ".join(hypothesis.casefold().split()) if hypothesis else ""
        if retries:
            if not normalized:
                raise ValueError("paid retry requires an explicit new hypothesis")
            for record in retries:
                previous = session.get(AttemptRecord, record.attempt_id)
                previous_hypothesis = previous.hypothesis if previous else None
                if previous_hypothesis and normalized == " ".join(
                    previous_hypothesis.casefold().split()
                ):
                    raise ValueError("paid retry requires a new hypothesis")
            if any(record.parameters_fingerprint == value for record in retries):
                raise ValueError("paid retry requires changed relevant parameters or code")
        if hypothesis:
            if attempt.hypothesis and attempt.hypothesis.strip() != hypothesis.strip():
                raise ValueError("paid operation hypothesis conflicts with its durable attempt")
            attempt.hypothesis = hypothesis.strip()
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
        return ProviderClaim(record.id, len(prior) + 1, value)

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
        if estimated_cost is not None and (not math.isfinite(estimated_cost) or estimated_cost < 0):
            raise ValueError("estimated cost must be finite and nonnegative")
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
