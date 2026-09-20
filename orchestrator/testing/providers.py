"""Deterministic, no-spend provider doubles for orchestrator tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from orchestrator.contracts import CostRecord

Outcome = Literal["completed", "failed", "unknown"]


class IdempotencyConflict(ValueError):
    """The same provider idempotency key was reused for different work."""


@dataclass(frozen=True)
class CostReceipt:
    """Stable cost evidence; integer micros avoid floating-point accounting drift."""

    provider: str
    operation_id: str
    amount_microusd: int
    units: tuple[tuple[str, int], ...] = ()
    estimated: bool = False

    def __post_init__(self) -> None:
        if self.amount_microusd < 0:
            raise ValueError("cost must be nonnegative")

    @property
    def amount_usd(self) -> Decimal:
        return Decimal(self.amount_microusd) / Decimal(1_000_000)

    def as_cost_record(self) -> CostRecord:
        return CostRecord(
            provider=self.provider,
            amount=float(self.amount_usd),
            units={name: float(value) for name, value in self.units},
            estimated=self.estimated,
        )


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    operation_id: str
    status: Outcome
    outputs: dict[str, Any]
    cost: CostReceipt
    error: str | None = None


@dataclass(frozen=True)
class SubmissionReceipt:
    provider: str
    operation_id: str
    status: Literal["pending", "unknown"]
    cost: CostReceipt


@dataclass(frozen=True)
class _SavedOperation:
    fingerprint: str
    result: ProviderResult


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


class _DeterministicProvider:
    def __init__(self, provider: str, *, amount_microusd: int) -> None:
        self.provider = provider
        self.amount_microusd = amount_microusd
        self._operations: dict[str, _SavedOperation] = {}
        self.invocations = 0
        self.charges = 0

    @property
    def operations(self) -> tuple[ProviderResult, ...]:
        return tuple(saved.result for saved in self._operations.values())

    def _execute(
        self,
        *,
        operation: str,
        idempotency_key: str,
        request: dict[str, Any],
        outcome: Outcome,
        outputs: dict[str, Any],
        units: tuple[tuple[str, int], ...] = (),
    ) -> ProviderResult:
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        if outcome not in {"completed", "failed", "unknown"}:
            raise ValueError(f"unsupported outcome: {outcome}")
        fingerprint = _digest(operation, _canonical(request), outcome)
        saved = self._operations.get(idempotency_key)
        if saved:
            if saved.fingerprint != fingerprint:
                raise IdempotencyConflict(
                    f"{self.provider} idempotency key was reused for different work"
                )
            return saved.result

        operation_id = f"{self.provider}:{_digest(operation, idempotency_key)[:20]}"
        receipt = CostReceipt(
            provider=self.provider,
            operation_id=operation_id,
            amount_microusd=self.amount_microusd,
            units=units,
        )
        result = ProviderResult(
            provider=self.provider,
            operation_id=operation_id,
            status=outcome,
            outputs=outputs if outcome == "completed" else {},
            cost=receipt,
            error="deterministic provider failure" if outcome == "failed" else None,
        )
        self._operations[idempotency_key] = _SavedOperation(fingerprint, result)
        self.invocations += 1
        self.charges += 1
        return result


class FakeMarbleProvider(_DeterministicProvider):
    """Asynchronous Marble submit/poll fake with durable operation identities."""

    def __init__(self, *, amount_microusd: int = 2_500_000) -> None:
        super().__init__("marble", amount_microusd=amount_microusd)
        self._by_operation_id: dict[str, ProviderResult] = {}
        self.polls = 0

    def submit(
        self,
        *,
        mode: Literal["image", "video", "multi"],
        input_sha256: str,
        prompt: str,
        idempotency_key: str,
        seed: int = 7,
        outcome: Outcome = "completed",
    ) -> SubmissionReceipt:
        request = {
            "mode": mode,
            "input_sha256": input_sha256,
            "prompt": prompt,
            "seed": seed,
        }
        world_sha256 = _digest("world", _canonical(request))
        result = self._execute(
            operation=f"generate:{mode}",
            idempotency_key=idempotency_key,
            request=request,
            outcome=outcome,
            outputs={
                "world_sha256": world_sha256,
                "thumbnail_sha256": _digest("thumbnail", world_sha256),
                "receipt": {
                    "schema": "wander.fake-marble-world/1",
                    "mode": mode,
                    "seed": seed,
                },
            },
            units=(("generations", 1),),
        )
        self._by_operation_id[result.operation_id] = result
        status = "unknown" if result.status == "unknown" else "pending"
        return SubmissionReceipt(self.provider, result.operation_id, status, result.cost)

    def poll(self, operation_id: str) -> ProviderResult:
        self.polls += 1
        try:
            return self._by_operation_id[operation_id]
        except KeyError as error:
            raise KeyError(f"unknown Marble operation: {operation_id}") from error


class FakeSynchronousJobs(_DeterministicProvider):
    """Modal-style synchronous job fake."""

    def __init__(self, *, amount_microusd: int = 125_000) -> None:
        super().__init__("modal", amount_microusd=amount_microusd)

    def run(
        self,
        job: str,
        parameters: dict[str, Any],
        *,
        idempotency_key: str,
        outcome: Outcome = "completed",
    ) -> ProviderResult:
        request = {"job": job, "parameters": parameters}
        return self._execute(
            operation=job,
            idempotency_key=idempotency_key,
            request=request,
            outcome=outcome,
            outputs={
                "result_sha256": _digest("modal-result", _canonical(request)),
                "job": job,
            },
            units=(("jobs", 1),),
        )


class FakeVLMProvider(_DeterministicProvider):
    """VLM fake returning configured text or a stable prompt-derived response."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        amount_microusd: int = 15_000,
    ) -> None:
        super().__init__("openai", amount_microusd=amount_microusd)
        self.responses = dict(responses or {})

    def respond(
        self,
        prompt: str,
        *,
        image_sha256s: tuple[str, ...] = (),
        idempotency_key: str,
        outcome: Outcome = "completed",
    ) -> ProviderResult:
        request = {"prompt": prompt, "image_sha256s": image_sha256s}
        text = self.responses.get(prompt, f"fixture-response:{_digest(prompt)[:16]}")
        return self._execute(
            operation="vlm-response",
            idempotency_key=idempotency_key,
            request=request,
            outcome=outcome,
            outputs={"text": text, "response_sha256": _digest(text)},
            units=(("requests", 1),),
        )


class FakeObjectGenerator(_DeterministicProvider):
    def __init__(self, *, amount_microusd: int = 750_000) -> None:
        super().__init__("object-generator", amount_microusd=amount_microusd)

    def generate(
        self,
        *,
        image_sha256: str,
        prompt: str,
        idempotency_key: str,
        outcome: Outcome = "completed",
    ) -> ProviderResult:
        request = {"image_sha256": image_sha256, "prompt": prompt}
        return self._execute(
            operation="image-to-3d",
            idempotency_key=idempotency_key,
            request=request,
            outcome=outcome,
            outputs={
                "object_sha256": _digest("object-ply", _canonical(request)),
                "media_type": "application/octet-stream",
            },
            units=(("objects", 1),),
        )


class FakeFineTuner(_DeterministicProvider):
    def __init__(self, *, amount_microusd: int = 4_000_000) -> None:
        super().__init__("ssh", amount_microusd=amount_microusd)

    def fine_tune(
        self,
        *,
        world_sha256: str,
        source_sha256: str,
        parameters: dict[str, Any],
        idempotency_key: str,
        outcome: Outcome = "completed",
    ) -> ProviderResult:
        request = {
            "world_sha256": world_sha256,
            "source_sha256": source_sha256,
            "parameters": parameters,
        }
        return self._execute(
            operation="fine-tune-world",
            idempotency_key=idempotency_key,
            request=request,
            outcome=outcome,
            outputs={
                "world_sha256": _digest("finetuned-world", _canonical(request)),
                "receipt": {
                    "schema": "wander.fake-finetune-receipt/1",
                    "parameters": parameters,
                },
            },
            units=(("jobs", 1),),
        )
