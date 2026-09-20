"""No-network test support for orchestrator consumers."""

from orchestrator.testing.fixtures import (
    FIXTURE_CODE_REVISION,
    FIXTURE_ENVIRONMENT_SHA256,
    FIXTURE_TIME,
    ArtifactFixture,
    RepresentativeRunFixture,
    artifact_fixture,
    representative_run_fixture,
)
from orchestrator.testing.providers import (
    CostReceipt,
    FakeFineTuner,
    FakeMarbleProvider,
    FakeObjectGenerator,
    FakeSynchronousJobs,
    FakeVLMProvider,
    IdempotencyConflict,
    ProviderResult,
    SubmissionReceipt,
)

__all__ = [
    "FIXTURE_CODE_REVISION",
    "FIXTURE_ENVIRONMENT_SHA256",
    "FIXTURE_TIME",
    "ArtifactFixture",
    "CostReceipt",
    "FakeFineTuner",
    "FakeMarbleProvider",
    "FakeObjectGenerator",
    "FakeSynchronousJobs",
    "FakeVLMProvider",
    "IdempotencyConflict",
    "ProviderResult",
    "RepresentativeRunFixture",
    "SubmissionReceipt",
    "artifact_fixture",
    "representative_run_fixture",
]
