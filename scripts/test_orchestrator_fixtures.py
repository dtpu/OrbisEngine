"""No-spend provider fake and deterministic fixture tests."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.testing import (
    FakeFineTuner,
    FakeMarbleProvider,
    FakeObjectGenerator,
    FakeSynchronousJobs,
    FakeVLMProvider,
    IdempotencyConflict,
    artifact_fixture,
    representative_run_fixture,
)

HASH = "a" * 64


class ProviderFakeTests(unittest.TestCase):
    def test_marble_submit_poll_is_idempotent_and_charged_once(self):
        provider = FakeMarbleProvider()
        arguments = {
            "mode": "video",
            "input_sha256": HASH,
            "prompt": "An observed test room",
            "idempotency_key": "run-1:marble-submit",
        }

        first = provider.submit(**arguments)
        second = provider.submit(**arguments)
        result = provider.poll(first.operation_id)

        self.assertEqual(first, second)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.outputs["world_sha256"]), 64)
        self.assertEqual(provider.invocations, 1)
        self.assertEqual(provider.charges, 1)
        self.assertEqual(result.cost.amount_microusd, 2_500_000)
        self.assertEqual(result.cost.as_cost_record().amount, 2.5)

    def test_idempotency_key_rejects_different_work(self):
        provider = FakeSynchronousJobs()
        provider.run("clean", {"dilate": 20}, idempotency_key="same-key")
        with self.assertRaises(IdempotencyConflict):
            provider.run("clean", {"dilate": 40}, idempotency_key="same-key")

    def test_unknown_outcome_has_no_invented_outputs_and_remains_recoverable(self):
        provider = FakeMarbleProvider()
        receipt = provider.submit(
            mode="image",
            input_sha256=HASH,
            prompt="Unknown result fixture",
            idempotency_key="unknown-operation",
            outcome="unknown",
        )
        result = provider.poll(receipt.operation_id)

        self.assertEqual(receipt.status, "unknown")
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.outputs, {})
        self.assertIsNone(result.error)

    def test_modal_vlm_object_and_finetune_outputs_are_deterministic(self):
        modal = FakeSynchronousJobs()
        vlm = FakeVLMProvider({"describe": "a red ball"})
        objects = FakeObjectGenerator()
        finetune = FakeFineTuner()

        modal_result = modal.run(
            "pi3x",
            {"frames": 12},
            idempotency_key="modal-1",
        )
        vlm_result = vlm.respond(
            "describe",
            image_sha256s=(HASH,),
            idempotency_key="vlm-1",
        )
        object_result = objects.generate(
            image_sha256=HASH,
            prompt=vlm_result.outputs["text"],
            idempotency_key="object-1",
        )
        finetune_result = finetune.fine_tune(
            world_sha256=modal_result.outputs["result_sha256"],
            source_sha256=HASH,
            parameters={"steps": 100},
            idempotency_key="finetune-1",
        )

        self.assertEqual(vlm_result.outputs["text"], "a red ball")
        self.assertEqual(len(object_result.outputs["object_sha256"]), 64)
        self.assertEqual(len(finetune_result.outputs["world_sha256"]), 64)
        self.assertEqual(
            finetune_result,
            finetune.fine_tune(
                world_sha256=modal_result.outputs["result_sha256"],
                source_sha256=HASH,
                parameters={"steps": 100},
                idempotency_key="finetune-1",
            ),
        )
        self.assertEqual(finetune.charges, 1)


class DeterministicFixtureTests(unittest.TestCase):
    def test_representative_run_is_byte_for_byte_repeatable(self):
        first = representative_run_fixture()
        second = representative_run_fixture()

        self.assertEqual(first.run.model_dump_json(), second.run.model_dump_json())
        self.assertEqual(first.attempt.model_dump_json(), second.attempt.model_dump_json())
        self.assertEqual(first.artifacts, second.artifacts)
        self.assertEqual(
            first.run.source_sha256, first.artifact_for_role("source_video").artifact.sha256
        )
        self.assertEqual(first.attempt.costs[0].amount, 2.5)

    def test_artifacts_write_with_contract_hashes_and_no_overwrite(self):
        fixture = representative_run_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            paths = fixture.write_artifacts(Path(temporary))
            for item in fixture.artifacts:
                path = paths[item.artifact.id]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), item.artifact.sha256
                )
                self.assertEqual(path.stat().st_size, item.artifact.size)

            changed = Path(temporary) / fixture.artifacts[0].filename
            changed.write_bytes(b"different")
            with self.assertRaises(FileExistsError):
                fixture.artifacts[0].write(Path(temporary))

    def test_small_artifact_factory_has_content_identity(self):
        first = artifact_fixture("report", '{"ok":true}\n')
        second = artifact_fixture("report", '{"ok":true}\n')
        changed = artifact_fixture("report", '{"ok":false}\n')

        self.assertEqual(first, second)
        self.assertNotEqual(first.artifact.id, changed.artifact.id)
        self.assertEqual(first.artifact.sha256, hashlib.sha256(first.content).hexdigest())


if __name__ == "__main__":
    unittest.main()
