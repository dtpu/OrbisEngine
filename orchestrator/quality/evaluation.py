"""Evidence validation and deterministic stage promotion rules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from orchestrator.contracts import Artifact, QualityVerdict

from .contracts import (
    PromotionDecision,
    QualityLayer,
    QualityReview,
    QualityRubric,
    StageQualityContract,
)


def validate_review_evidence(
    rubric: QualityRubric,
    review: QualityReview,
    artifacts: Iterable[Artifact],
    *,
    run_id: str,
) -> tuple[str, ...]:
    """Return all structural and artifact-binding problems in a review."""

    issues: list[str] = []
    if review.rubric_id != rubric.id:
        issues.append(f"review rubric {review.rubric_id!r} does not match {rubric.id!r}")
    if review.stage_id != rubric.stage_id:
        issues.append(f"review stage {review.stage_id!r} does not match {rubric.stage_id!r}")
    if review.layer != rubric.layer:
        issues.append(f"review layer {review.layer!r} does not match {rubric.layer!r}")

    artifact_by_id = {artifact.id: artifact for artifact in artifacts}
    expected = {criterion.id: criterion for criterion in rubric.criteria}
    actual = {finding.criterion_id: finding for finding in review.findings}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        issues.append(f"missing criterion findings: {', '.join(missing)}")
    if unexpected:
        issues.append(f"unexpected criterion findings: {', '.join(unexpected)}")

    for criterion_id in sorted(set(expected) & set(actual)):
        criterion = expected[criterion_id]
        finding = actual[criterion_id]
        evidence_ids = [item.artifact_id for item in finding.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            issues.append(f"{criterion_id}: duplicate evidence artifact ids")

        valid_artifacts: list[Artifact] = []
        for evidence in finding.evidence:
            artifact = artifact_by_id.get(evidence.artifact_id)
            if artifact is None:
                issues.append(f"{criterion_id}: unknown artifact {evidence.artifact_id!r}")
                continue
            if artifact.run_id != run_id:
                issues.append(f"{criterion_id}: artifact {artifact.id!r} belongs to another run")
                continue
            valid_artifacts.append(artifact)

        role_counts = Counter(artifact.role for artifact in valid_artifacts)
        for requirement in criterion.evidence:
            matching = [
                artifact
                for artifact in valid_artifacts
                if artifact.role == requirement.role
                and (
                    not requirement.from_attempt
                    or artifact.producer_attempt_id == review.attempt_id
                )
            ]
            if len(matching) < requirement.minimum:
                ownership = " from this attempt" if requirement.from_attempt else ""
                issues.append(
                    f"{criterion_id}: requires {requirement.minimum} "
                    f"{requirement.role!r} artifact(s){ownership}"
                )
        allowed_roles = {requirement.role for requirement in criterion.evidence}
        extra_roles = sorted(set(role_counts) - allowed_roles)
        if extra_roles:
            issues.append(
                f"{criterion_id}: evidence has undeclared roles: {', '.join(extra_roles)}"
            )

    return tuple(issues)


def evaluate_promotion(
    contract: StageQualityContract,
    reviews: Iterable[QualityReview],
    artifacts: Iterable[Artifact],
    *,
    run_id: str,
    attempt_id: str,
    retry_allowed: bool = True,
) -> PromotionDecision:
    """Apply automatic, then agent, then human gates to one immutable attempt."""

    review_list = tuple(reviews)
    artifact_list = tuple(artifacts)
    by_rubric = {review.rubric_id: review for review in review_list}
    if len(by_rubric) != len(review_list):
        return _blocked(contract, attempt_id, ("duplicate reviews for a rubric",))

    expected_ids = {rubric.id for rubric in contract.ordered_rubrics}
    unexpected = sorted(set(by_rubric) - expected_ids)
    if unexpected:
        return _blocked(
            contract,
            attempt_id,
            (f"reviews reference undeclared rubrics: {', '.join(unexpected)}",),
        )

    for rubric in (*contract.automatic, *((contract.agent,) if contract.agent else ())):
        review = by_rubric.get(rubric.id)
        if review is None:
            return _blocked(
                contract,
                attempt_id,
                (f"required {rubric.layer} review {rubric.id!r} is missing",),
            )
        issues = _review_issues(rubric, review, artifact_list, run_id, attempt_id)
        if issues:
            return _blocked(contract, attempt_id, issues)
        if not review.passed:
            verdict = QualityVerdict.RETRY if retry_allowed else QualityVerdict.BLOCK
            return PromotionDecision(
                stage_id=contract.stage_id,
                attempt_id=attempt_id,
                verdict=verdict,
                promote=False,
                rationale=f"{rubric.layer} quality criteria failed",
            )

    if contract.human is not None:
        review = by_rubric.get(contract.human.id)
        if review is None:
            return PromotionDecision(
                stage_id=contract.stage_id,
                attempt_id=attempt_id,
                verdict=QualityVerdict.REQUEST_HUMAN,
                promote=False,
                rationale="automatic and agent gates passed; human approval is required",
                next_layer=QualityLayer.HUMAN,
            )
        issues = _review_issues(
            contract.human,
            review,
            artifact_list,
            run_id,
            attempt_id,
        )
        if issues:
            return _blocked(contract, attempt_id, issues)
        if not review.passed:
            return PromotionDecision(
                stage_id=contract.stage_id,
                attempt_id=attempt_id,
                verdict=QualityVerdict.BLOCK,
                promote=False,
                rationale="human approval was denied",
            )

    return PromotionDecision(
        stage_id=contract.stage_id,
        attempt_id=attempt_id,
        verdict=QualityVerdict.PASS,
        promote=True,
        rationale="all required quality gates passed",
    )


def _review_issues(
    rubric: QualityRubric,
    review: QualityReview,
    artifacts: tuple[Artifact, ...],
    run_id: str,
    attempt_id: str,
) -> tuple[str, ...]:
    issues = list(validate_review_evidence(rubric, review, artifacts, run_id=run_id))
    if review.attempt_id != attempt_id:
        issues.append(f"review attempt {review.attempt_id!r} does not match {attempt_id!r}")
    return tuple(issues)


def _blocked(
    contract: StageQualityContract,
    attempt_id: str,
    issues: tuple[str, ...],
) -> PromotionDecision:
    return PromotionDecision(
        stage_id=contract.stage_id,
        attempt_id=attempt_id,
        verdict=QualityVerdict.BLOCK,
        promote=False,
        rationale="quality evidence is incomplete or invalid",
        issues=issues,
    )
