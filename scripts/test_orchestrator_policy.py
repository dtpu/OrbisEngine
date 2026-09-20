"""Agent proposal policy tests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contracts import RetryPolicy, StageDefinition, StageKind
from orchestrator.policy import PolicyContext, ProposalPolicy, parameter_fingerprint


class ProposalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.definition = StageDefinition(
            id="clean",
            title="Clean",
            kind=StageKind.COMPUTE,
            executor="clean_video",
            parameter_schema={
                "type": "object",
                "properties": {
                    "dilate": {"type": "integer", "minimum": 0, "maximum": 200},
                    "moved_mask": {"type": "boolean"},
                },
                "required": ["dilate"],
                "additionalProperties": False,
            },
            retry=RetryPolicy(
                paid=True,
                requires_changed_hypothesis=True,
                requires_changed_parameters=True,
            ),
        )
        self.policy = ProposalPolicy()

    def test_accepts_bounded_changed_paid_retry(self):
        previous = parameter_fingerprint({"dilate": 20})
        decision = self.policy.evaluate(
            self.definition,
            {
                "dilate": 40,
                "moved_mask": True,
                "hypothesis": "Expand the residual-person boundary",
                "estimated_cost_usd": 1.0,
            },
            PolicyContext(
                dependency_statuses={"admission": "succeeded"},
                prior_parameter_fingerprints=(previous,),
                attempts_used=1,
                remaining_cost=2.0,
            ),
        )
        self.assertTrue(decision.accepted, decision.reasons)

    def test_rejects_dependency_schema_reconciliation_and_budget_violations(self):
        previous = parameter_fingerprint({"dilate": 20})
        decision = self.policy.evaluate(
            self.definition,
            {
                "dilate": 20,
                "shell": "rm -rf /",
                "estimated_cost_usd": 3.0,
            },
            PolicyContext(
                dependency_statuses={"admission": "failed"},
                prior_parameter_fingerprints=(previous,),
                attempts_used=1,
                unresolved_provider_claim=True,
                remaining_cost=2.0,
            ),
        )
        self.assertFalse(decision.accepted)
        joined = " | ".join(decision.reasons)
        for expected in (
            "dependencies",
            "undeclared parameters",
            "changed hypothesis",
            "unchanged",
            "reconciled",
            "remaining cost",
        ):
            self.assertIn(expected, joined)

    def test_attempt_ceiling_and_bounds_are_enforced(self):
        decision = self.policy.evaluate(
            self.definition,
            {
                "dilate": 201,
                "hypothesis": "Try more",
                "estimated_cost_usd": 0.1,
            },
            PolicyContext(
                dependency_statuses={},
                attempts_used=3,
                attempt_ceiling=3,
                remaining_cost=1,
            ),
        )
        self.assertFalse(decision.accepted)
        self.assertIn("attempt ceiling", " ".join(decision.reasons))
        self.assertIn("above maximum", " ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
