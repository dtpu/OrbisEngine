"""The reviewing agent judges attempts through a coding harness; no model is called here."""

import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.activities.review import ReviewActivities
from orchestrator.agent.harness import HarnessAgent, HarnessPolicy
from orchestrator.artifacts import DatabaseFilesystemArtifactResolver, LocalCAS, freeze_attempt
from orchestrator.contracts import NodeStatus, Run
from orchestrator.database import Base
from orchestrator.graph import instantiate_graph
from orchestrator.persistence import DatabaseAttemptLedger
from orchestrator.repository import PipelineRepository
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import (
    AGENT_RETRY_CAP,
    REVIEWABLE_STATUSES,
    ReviewDecision,
    ReviewInput,
    StageActivityInput,
    StageActivityResult,
    apply_review_decision,
    reviewable,
)
from orchestrator.workspace import RunWorkspace

RUN_ID = "run-review"
SHA = "a" * 64


def fake_harness(root: Path, body: str) -> HarnessPolicy:
    """A 'harness' that is a python script run in the scratch workspace."""
    script = root / "agent.py"
    script.write_text(textwrap.dedent(body))
    return HarnessPolicy(kind="command", command=(sys.executable, str(script)), timeout_seconds=60)


class ReviewActivityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-review-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        engine = create_engine(f"sqlite:///{self.root / 'meta.db'}")
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.repository = PipelineRepository(self.sessions)
        self.store = LocalCAS(self.root / "cas")
        self.store.initialize()
        self.archive = LocalCAS(self.root / "artifacts")
        self.archive.initialize()
        self.ledger = DatabaseAttemptLedger(
            self.sessions, code_revision="test-revision", environment={"container": "test"}
        )
        self.graph = instantiate_graph(GraphOptions())
        self.repository.create_run(
            Run(
                id=RUN_ID,
                graph_version="wander.generation-graph/1",
                code_revision="test-revision",
                source_sha256=SHA,
                source_artifact_id=f"artifact:{RUN_ID}:source",
                created_by="tests",
            ),
            self.graph,
        )
        self.definition = self.graph.nodes["clean"].definition.model_dump(mode="json")
        self.request, self.result = self._finished_attempt()

    def _finished_attempt(self):
        """A succeeded `clean` attempt with a report and a frame, recorded like the runner does."""
        attempt_id = f"{RUN_ID}:2:1"
        stage_input = StageActivityInput(
            run_id=RUN_ID,
            node_id="clean",
            stage_type="clean",
            definition=self.definition,
            selected_inputs={},
        )
        workspace = RunWorkspace(self.root / "runs", RUN_ID)
        workspace.initialize()
        attempt = workspace.create_attempt("clean", attempt_id, {"runId": RUN_ID})
        (attempt.outputs / "clean.json").write_text(json.dumps({"residual": 0.4}))
        (attempt.outputs / "clean.png").write_bytes(b"\x89PNG not really")
        self.ledger.started(stage_input, attempt_id)
        attempt.finalize({"status": "succeeded", "error": None, "command": []})
        manifest = freeze_attempt(
            attempt,
            self.store,
            status="succeeded",
            roles={"outputs/clean.json": "clean_report", "outputs/clean.png": "clean_frame"},
        )
        for frozen in manifest.files:
            self.archive.put_blob(frozen.sha256, self.store.blob_path(frozen.sha256))
        artifacts = {}
        for frozen in manifest.files:
            if frozen.role != "attempt_file":
                artifacts.setdefault(frozen.role, []).append(frozen.artifact_id)
        result = StageActivityResult(
            node_id="clean", attempt_id=attempt_id, status="succeeded", artifacts=artifacts
        )
        self.ledger.finished(stage_input, result, manifest)
        request = ReviewInput(
            run_id=RUN_ID,
            node_id="clean",
            stage_type="clean",
            definition=self.definition,
            attempt_id=attempt_id,
            artifacts=artifacts,
            operator_messages=[{"id": "m1", "author": "daniel", "message": "keep the floor"}],
        )
        return request, result

    def activities(self, policy: HarnessPolicy) -> ReviewActivities:
        return ReviewActivities(
            self.repository,
            workspace_root=self.root / "runs",
            store=self.store,
            outbox_archive=self.archive,
            resolver=DatabaseFilesystemArtifactResolver(self.sessions, self.root / "artifacts"),
            attempt_ledger=self.ledger,
            harness=HarnessAgent(policy),
        )

    def test_pass_verdict_keeps_the_reviewed_attempt_and_is_recorded(self):
        policy = fake_harness(
            self.root,
            """
            import json, pathlib
            task = json.load(open("task.json"))
            assert pathlib.Path("outputs/clean.json").is_file(), "outputs were not hydrated"
            assert "TASK.md" in open("TASK.md").read() or "Review stage" in open("TASK.md").read()
            assert task["operator_messages"][0]["message"] == "keep the floor"
            json.dump({"tool": "quality.verdict", "node_id": "clean",
                       "attempt_id": task["attempts"][-1]["id"], "verdict": "pass",
                       "rationale": "residual 0.4 is within tolerance; background intact"},
                      open("decision.json", "w"))
            """,
        )
        decision = self.activities(policy).review(self.request, f"{RUN_ID}:9:1:agent")
        self.assertEqual(decision.decision, "pass")
        self.assertEqual(decision.attempt_id, self.request.attempt_id)
        self.assertFalse(decision.revised)
        reviews = self.repository.run_summary(RUN_ID)["reviews"]
        self.assertEqual([r["verdict"] for r in reviews], ["pass"])
        self.assertTrue(reviews[0]["decidedBy"].startswith("agent:command"))

    def test_edited_outputs_become_a_new_attempt_that_a_pass_selects(self):
        policy = fake_harness(
            self.root,
            """
            import json
            report = json.load(open("outputs/clean.json")); report["residual"] = 0.1
            json.dump(report, open("outputs/clean.json", "w"))
            json.dump({"tool": "quality.verdict", "node_id": "clean",
                       "attempt_id": json.load(open("task.json"))["attempts"][-1]["id"],
                       "verdict": "pass", "rationale": "corrected the residual estimate"},
                      open("decision.json", "w"))
            """,
        )
        revised_id = f"{RUN_ID}:9:1:agent"
        decision = self.activities(policy).review(self.request, revised_id)
        self.assertEqual(decision.decision, "pass")
        self.assertTrue(decision.revised)
        self.assertEqual(decision.attempt_id, revised_id)
        self.assertIn("clean_report", decision.artifacts)
        self.assertNotEqual(
            decision.artifacts["clean_report"], self.result.artifacts["clean_report"]
        )
        summary = self.repository.run_summary(RUN_ID)
        self.assertEqual([a["id"] for a in summary["attempts"]][-1], revised_id)
        # The revised blob reached the archive so downstream stages can read it.
        revised = next(
            a
            for a in summary["artifacts"]
            if a["attemptId"] == revised_id and a["role"] == "clean_report"
        )
        self.assertTrue(
            (
                self.root / "artifacts" / "blobs" / revised["sha256"][:2] / revised["sha256"]
            ).is_file()
        )

    def test_retry_carries_hypothesis_and_parameters(self):
        policy = fake_harness(
            self.root,
            """
            import json
            json.dump({"tool": "attempt.retry", "node_id": "clean",
                       "attempt_id": json.load(open("task.json"))["attempts"][-1]["id"],
                       "hypothesis": "mask misses the shadow; dilate more",
                       "parameters": {"dilate": 40}}, open("decision.json", "w"))
            """,
        )
        decision = self.activities(policy).review(self.request, f"{RUN_ID}:9:1:agent")
        self.assertEqual(decision.decision, "retry")
        self.assertEqual(decision.parameters, {"dilate": 40})
        self.assertEqual(decision.hypothesis, "mask misses the shadow; dilate more")

    def test_no_decision_or_bad_decision_falls_to_a_human_never_a_pass(self):
        silent = fake_harness(self.root, "print('thinking...')\n")
        decision = self.activities(silent).review(self.request, f"{RUN_ID}:9:1:agent")
        self.assertEqual(decision.decision, "needs_human")
        forbidden = fake_harness(
            self.root,
            """
            import json
            json.dump({"tool": "parameters.propose", "node_id": "clean", "parameters": {},
                       "hypothesis": "x"}, open("decision.json", "w"))
            """,
        )
        decision = self.activities(forbidden).review(self.request, f"{RUN_ID}:9:2:agent")
        self.assertEqual(decision.decision, "needs_human")
        self.assertIn("not permitted", decision.rationale)


class ReviewDecisionApplicationTests(unittest.TestCase):
    def setUp(self):
        self.graph = instantiate_graph(GraphOptions())
        # Every role the clean stage declares as required; select_attempt refuses fewer.
        self.artifacts = {
            "clean_video": ["artifact:v"],
            "clean_frame": ["artifact:f"],
            "person_masks": ["artifact:m"],
            "clean_report": ["artifact:r"],
        }
        self.result = StageActivityResult(
            node_id="clean", attempt_id="run:2:1", status="succeeded", artifacts=self.artifacts
        )

    def test_only_rubric_stages_are_reviewed(self):
        self.assertTrue(reviewable(self.graph.nodes["clean"].definition.model_dump(mode="json")))
        self.assertFalse(
            reviewable(self.graph.nodes["admission"].definition.model_dump(mode="json"))
        )
        self.assertFalse(reviewable(self.graph.nodes["verify"].definition.model_dump(mode="json")))

    def test_pass_selects_the_agent_named_attempt(self):
        decision = ReviewDecision(
            node_id="clean",
            attempt_id="run:9:1:agent",
            decision="pass",
            rationale="ok",
            artifacts={**self.artifacts, "clean_report": ["artifact:r2"]},
            revised=True,
        )
        outcome = apply_review_decision(self.graph, self.result, decision, {}, {})
        self.assertEqual(outcome, "pass")
        node = self.graph.nodes["clean"]
        self.assertEqual(node.status, NodeStatus.SUCCEEDED)
        self.assertEqual(node.selected_attempt_id, "run:9:1:agent")
        self.assertEqual(node.selected_artifacts["clean_report"], ("artifact:r2",))

    def test_retry_is_bounded_and_hands_parameters_to_the_next_attempt(self):
        retry_parameters, agent_retries = {}, {}
        decision = ReviewDecision(
            node_id="clean",
            attempt_id="run:2:1",
            decision="retry",
            rationale="more dilation",
            parameters={"dilate": 40},
            hypothesis="more dilation",
        )
        self.assertEqual(
            apply_review_decision(
                self.graph, self.result, decision, retry_parameters, agent_retries, cap=1
            ),
            "retry",
        )
        self.assertEqual(retry_parameters["clean"], {"dilate": 40, "hypothesis": "more dilation"})
        self.assertEqual(self.graph.nodes["clean"].status, NodeStatus.QUEUED)
        self.assertEqual(
            apply_review_decision(
                self.graph, self.result, decision, retry_parameters, agent_retries, cap=1
            ),
            "needs_human",
        )
        self.assertEqual(self.graph.nodes["clean"].status, NodeStatus.WAITING_HUMAN)
        self.assertIn("confirm to let it try again", self.graph.nodes["clean"].blocked_reason)

    def test_failures_are_reviewed_too_and_cannot_be_passed(self):
        """A failed attempt is exactly where the agent's judgement matters most."""
        self.assertEqual(REVIEWABLE_STATUSES, {"succeeded", "failed", "blocked"})
        failed = StageActivityResult(
            node_id="clean", attempt_id="run:2:1", status="failed", error="command exited 1"
        )
        # It may not rubber-stamp a stage that produced nothing valid.
        passing = ReviewDecision(
            node_id="clean", attempt_id="run:2:1", decision="pass", rationale="looks fine to me"
        )
        self.assertEqual(apply_review_decision(self.graph, failed, passing, {}, {}), "needs_human")
        self.assertEqual(self.graph.nodes["clean"].status, NodeStatus.WAITING_HUMAN)
        self.assertIn("agent passed a failed attempt", self.graph.nodes["clean"].blocked_reason)

    def test_a_failed_attempt_can_be_retried_with_new_parameters(self):
        failed = StageActivityResult(
            node_id="clean", attempt_id="run:2:1", status="failed", error="SVD did not converge"
        )
        retry_parameters: dict = {}
        decision = ReviewDecision(
            node_id="clean",
            attempt_id="run:2:1",
            decision="retry",
            rationale="mask too tight",
            parameters={"dilate": 40},
            hypothesis="mask too tight",
        )
        self.assertEqual(
            apply_review_decision(self.graph, failed, decision, retry_parameters, {}), "retry"
        )
        self.assertEqual(retry_parameters["clean"]["dilate"], 40)

    def test_agent_gets_three_retries_then_asks_the_operator(self):
        self.assertEqual(AGENT_RETRY_CAP, 3)
        retry_parameters, agent_retries = {}, {}
        decision = ReviewDecision(
            node_id="clean",
            attempt_id="run:2:1",
            decision="retry",
            rationale="try again",
            hypothesis="try again",
        )
        for _ in range(AGENT_RETRY_CAP):
            self.assertEqual(
                apply_review_decision(
                    self.graph, self.result, decision, retry_parameters, agent_retries
                ),
                "retry",
            )
        self.assertEqual(agent_retries["clean"], AGENT_RETRY_CAP)
        self.assertEqual(
            apply_review_decision(
                self.graph, self.result, decision, retry_parameters, agent_retries
            ),
            "needs_human",
        )
        reason = self.graph.nodes["clean"].blocked_reason
        self.assertIn(f"agent used its {AGENT_RETRY_CAP} retries", reason)
        self.assertIn("confirm to let it try again", reason)

    def test_question_waits_for_a_human_with_the_question_visible(self):
        decision = ReviewDecision(
            node_id="clean",
            attempt_id="run:2:1",
            decision="needs_human",
            rationale="is the reflection acceptable?",
            question="Is the residual reflection on the floor acceptable?",
        )
        self.assertEqual(
            apply_review_decision(self.graph, self.result, decision, {}, {}), "needs_human"
        )
        self.assertEqual(self.graph.nodes["clean"].blocked_reason, decision.question)


if __name__ == "__main__":
    unittest.main()
