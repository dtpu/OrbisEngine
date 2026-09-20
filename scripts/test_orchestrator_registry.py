"""Coverage and consistency tests for the declarative stage registry."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.activities.adapters import default_adapters
from orchestrator.contracts import StageKind
from orchestrator.stages import GraphOptions, stage_registry


class RegistryTests(unittest.TestCase):
    def test_registry_covers_current_pipeline_and_explicit_substages(self):
        stages = stage_registry()
        expected = {
            "admission",
            "clean_first",
            "clean",
            "clean_multi",
            "world_prompt",
            "world_mode",
            "marble_image_submit",
            "marble_image",
            "marble_video_submit",
            "marble_video",
            "marble_multi_submit",
            "marble_multi",
            "pi3x",
            "frame_align",
            "tracks",
            "person_prep",
            "lhm_frozen",
            "lhm_motion",
            "package_people",
            "audio_reviewed",
            "object_detect",
            "object_lift",
            "object_describe",
            "object_shape",
            "object_package",
            "scale_fit",
            "place_fit",
            "anchors",
            "finetune",
            "verify",
        }
        self.assertEqual(set(stages), expected)

    def test_bindings_reference_declared_stages_and_output_roles(self):
        stages = stage_registry()
        for stage in stages.values():
            for binding in stage.inputs.values():
                if binding.source != "stage_output":
                    continue
                with self.subTest(stage=stage.id, dependency=binding.stage_id, role=binding.role):
                    self.assertIn(binding.stage_id, stages)
                    producer = stages[binding.stage_id]
                    self.assertTrue(
                        any(output.role == binding.role for output in producer.outputs.values()),
                        f"{stage.id} requires undeclared role {binding.role} from {binding.stage_id}",
                    )

    def test_paid_stages_never_enable_automatic_resubmission(self):
        for stage in stage_registry().values():
            if stage.retry.paid:
                with self.subTest(stage=stage.id):
                    self.assertEqual(stage.retry.automatic_attempts, 1)

    def test_human_nodes_have_no_executor(self):
        for stage in stage_registry().values():
            if stage.kind == StageKind.HUMAN:
                self.assertIsNone(stage.executor)

    def test_every_executor_has_an_activity_adapter(self):
        adapters = default_adapters()
        for stage in stage_registry().values():
            if stage.executor:
                self.assertIn(stage.executor, adapters, stage.id)

    def test_graph_options_are_bounded_and_immutable(self):
        options = GraphOptions(all_people=True, people=4, objects=True)
        self.assertEqual(options.people, 4)
        with self.assertRaises(ValueError):
            GraphOptions(people=0)


if __name__ == "__main__":
    unittest.main()
