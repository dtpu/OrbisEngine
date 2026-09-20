"""QA rubric, evidence-binding, verdict, and promotion tests."""

import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contracts import Artifact, Evidence, QualityVerdict
from orchestrator.quality import (
    CriterionFinding,
    PromotionDecision,
    QualityLayer,
    QualityReview,
    evaluate_promotion,
    quality_catalog,
    validate_review_evidence,
)
from orchestrator.quality.validators import validate_stage_outputs
from orchestrator.stages.registry import stage_registry

HASH = "a" * 64
RUN_ID = "run-1"
ATTEMPT_ID = "attempt-1"


def artifacts_for(rubric, *, run_id=RUN_ID, attempt_id=ATTEMPT_ID):
    artifacts = []
    seen = set()
    for criterion in rubric.criteria:
        for requirement in criterion.evidence:
            key = (requirement.role, requirement.from_attempt)
            if key in seen:
                continue
            seen.add(key)
            artifacts.append(
                Artifact(
                    id=f"artifact-{requirement.role}-{int(requirement.from_attempt)}",
                    run_id=run_id,
                    role=requirement.role,
                    sha256=HASH,
                    size=1,
                    media_type="application/octet-stream",
                    storage_key=f"blobs/{len(artifacts)}",
                    producer_attempt_id=(
                        attempt_id if requirement.from_attempt else "prior-attempt"
                    ),
                )
            )
    return artifacts


def review_for(rubric, artifacts, *, passed=True, attempt_id=ATTEMPT_ID):
    findings = []
    for criterion in rubric.criteria:
        required_roles = {requirement.role for requirement in criterion.evidence}
        matching_artifacts = {
            artifact.id: artifact for artifact in artifacts if artifact.role in required_roles
        }
        evidence = tuple(
            Evidence(artifact_id=artifact.id, observation=f"Observed {artifact.role}")
            for artifact in matching_artifacts.values()
        )
        findings.append(
            CriterionFinding(
                criterion_id=criterion.id,
                passed=passed,
                rationale="Criterion evaluated against bound evidence",
                evidence=evidence,
            )
        )
    return QualityReview(
        rubric_id=rubric.id,
        stage_id=rubric.stage_id,
        attempt_id=attempt_id,
        layer=rubric.layer,
        findings=tuple(findings),
        decided_by=f"{rubric.layer}:test",
    )


class QualityContractTests(unittest.TestCase):
    def test_catalog_covers_every_stage_policy(self):
        stages = stage_registry()
        catalog = quality_catalog()
        self.assertEqual(set(catalog), set(stages))
        for stage_id, stage in stages.items():
            contract = catalog[stage_id]
            with self.subTest(stage=stage_id):
                self.assertEqual(
                    tuple(rubric.criteria[0].id for rubric in contract.automatic),
                    stage.quality.automatic_checks,
                )
                self.assertEqual(
                    contract.agent is not None,
                    stage.quality.agent_rubric is not None,
                )
                self.assertEqual(
                    contract.human is not None,
                    stage.quality.human_approval,
                )

    def test_rubrics_are_stage_specific_and_require_explicit_evidence(self):
        catalog = quality_catalog()
        clean = catalog["clean"]
        clean_multi = catalog["clean_multi"]
        self.assertNotEqual(clean.agent.id, clean_multi.agent.id)
        self.assertEqual(clean.agent.stage_id, "clean")
        self.assertTrue(
            all(
                criterion.evidence
                for rubric in clean.ordered_rubrics
                for criterion in rubric.criteria
            )
        )

    def test_evidence_validator_rejects_missing_wrong_run_and_wrong_attempt(self):
        rubric = quality_catalog()["clean"].automatic[0]
        artifacts = artifacts_for(rubric)
        valid = review_for(rubric, artifacts)
        self.assertEqual(
            validate_review_evidence(rubric, valid, artifacts, run_id=RUN_ID),
            (),
        )

        missing = review_for(rubric, ())
        issues = validate_review_evidence(rubric, missing, (), run_id=RUN_ID)
        self.assertTrue(any("requires" in issue for issue in issues))

        wrong_run_artifacts = artifacts_for(rubric, run_id="run-2")
        wrong_run = review_for(rubric, wrong_run_artifacts)
        issues = validate_review_evidence(
            rubric,
            wrong_run,
            wrong_run_artifacts,
            run_id=RUN_ID,
        )
        self.assertTrue(any("another run" in issue for issue in issues))

        wrong_attempt_artifacts = artifacts_for(rubric, attempt_id="attempt-2")
        wrong_attempt = review_for(rubric, wrong_attempt_artifacts)
        issues = validate_review_evidence(
            rubric,
            wrong_attempt,
            wrong_attempt_artifacts,
            run_id=RUN_ID,
        )
        self.assertTrue(any("from this attempt" in issue for issue in issues))

    def test_automatic_and_agent_gates_must_both_pass(self):
        contract = quality_catalog()["clean"]
        rubrics = contract.ordered_rubrics
        artifacts = tuple(artifact for rubric in rubrics for artifact in artifacts_for(rubric))
        reviews = tuple(review_for(rubric, artifacts) for rubric in rubrics)
        decision = evaluate_promotion(
            contract,
            reviews,
            artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        self.assertEqual(decision.verdict, QualityVerdict.PASS)
        self.assertTrue(decision.promote)

        failed = (review_for(rubrics[0], artifacts, passed=False), *reviews[1:])
        decision = evaluate_promotion(
            contract,
            failed,
            artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        self.assertEqual(decision.verdict, QualityVerdict.RETRY)
        self.assertFalse(decision.promote)

        decision = evaluate_promotion(
            contract,
            failed,
            artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            retry_allowed=False,
        )
        self.assertEqual(decision.verdict, QualityVerdict.BLOCK)

    def test_human_gate_requests_review_then_controls_promotion(self):
        # verify is the remaining stage with both an agent rubric and a human gate. It also
        # carries an automatic check, so the whole ladder runs: automatic, then agent, then the
        # human who decides promotion.
        contract = quality_catalog()["verify"]
        automatic_reviews = []
        agent_artifacts = list(artifacts_for(contract.agent))
        for rubric in contract.automatic:
            evidence = artifacts_for(rubric)
            agent_artifacts.extend(evidence)
            automatic_reviews.append(review_for(rubric, evidence))
        agent_artifacts = list({a.id: a for a in agent_artifacts}.values())
        agent_review = review_for(contract.agent, agent_artifacts)
        earlier_reviews = (*automatic_reviews, agent_review)

        pending = evaluate_promotion(
            contract,
            earlier_reviews,
            agent_artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        self.assertEqual(pending.verdict, QualityVerdict.REQUEST_HUMAN)
        self.assertEqual(pending.next_layer, QualityLayer.HUMAN)

        human_artifacts = artifacts_for(contract.human)
        all_artifacts = (*agent_artifacts, *human_artifacts)
        approved = review_for(contract.human, all_artifacts)
        decision = evaluate_promotion(
            contract,
            (*earlier_reviews, approved),
            all_artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        self.assertEqual(decision.verdict, QualityVerdict.PASS)
        self.assertTrue(decision.promote)

        denied = review_for(contract.human, all_artifacts, passed=False)
        decision = evaluate_promotion(
            contract,
            (*earlier_reviews, denied),
            all_artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        self.assertEqual(decision.verdict, QualityVerdict.BLOCK)
        self.assertFalse(decision.promote)

    def test_invalid_or_stale_evidence_never_promotes(self):
        contract = quality_catalog()["clean_first"]
        rubric = contract.automatic[0]
        artifacts = artifacts_for(rubric)
        stale_review = review_for(rubric, artifacts, attempt_id="attempt-old")
        decision = evaluate_promotion(
            contract,
            (stale_review,),
            artifacts,
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
        )
        self.assertEqual(decision.verdict, QualityVerdict.BLOCK)
        self.assertFalse(decision.promote)
        self.assertTrue(any("does not match" in issue for issue in decision.issues))

    def test_promotion_contract_forbids_non_pass_promotion(self):
        with self.assertRaises(ValidationError):
            PromotionDecision(
                stage_id="clean",
                attempt_id=ATTEMPT_ID,
                verdict=QualityVerdict.BLOCK,
                promote=True,
                rationale="Invalid",
            )


class AutomaticValidatorTests(unittest.TestCase):
    def test_missing_and_invalid_required_outputs_fail_before_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            (root / "outputs/operation.json").write_text('{"status":"submitted"}')
            definition = {
                "outputs": {
                    "provider_operation": {
                        "role": "provider_operation",
                        "media_type": "application/json",
                        "required": True,
                        "multiple": False,
                    }
                },
                "quality": {"automatic_checks": ["provider_operation"]},
            }
            issues = validate_stage_outputs(
                definition,
                root,
                {"outputs/operation.json": "provider_operation"},
            )
            self.assertIn("no provider operation ID", " ".join(issues))

    def test_positive_finite_scale_passes_structural_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            (root / "outputs/scale.json").write_text('{"scale0":1.25}')
            definition = {
                "outputs": {
                    "scale": {
                        "role": "scale_fit",
                        "media_type": "application/json",
                        "required": True,
                        "multiple": False,
                    }
                },
                "quality": {"automatic_checks": ["scale_ratio"]},
            }
            self.assertEqual(
                validate_stage_outputs(
                    definition,
                    root,
                    {"outputs/scale.json": "scale_fit"},
                ),
                (),
            )


if __name__ == "__main__":
    unittest.main()
