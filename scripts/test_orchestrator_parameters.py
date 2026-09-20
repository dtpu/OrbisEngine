"""A stage's declared parameters must reach its command line.

The legacy adapter used to forward only a nested ``cli`` key, which every stage schema forbids
with ``additionalProperties: false``. Operator and agent retries therefore ran with defaults and
produced byte-identical output, which looks like a stage that ignores its own tuning.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.activities.adapters import legacy_parameter_flags, legacy_people_flags
from orchestrator.graph import instantiate_graph
from orchestrator.stages import GraphOptions


class LegacyParameterFlagTests(unittest.TestCase):
    def setUp(self):
        graph = instantiate_graph(GraphOptions())
        self.clean = graph.nodes["clean"].definition.model_dump(mode="json")

    def test_declared_parameters_become_flags(self):
        self.assertEqual(
            legacy_parameter_flags(self.clean, {"dilate": 12, "bottom_extra": 0}),
            ["--bottom-extra", "0", "--dilate", "12"],
        )

    def test_a_retry_hypothesis_is_not_a_stage_argument(self):
        flags = legacy_parameter_flags(
            self.clean, {"dilate": 12, "hypothesis": "masks eat the goalpost"}
        )
        self.assertEqual(flags, ["--dilate", "12"])

    def test_booleans_are_bare_flags_and_false_is_absent(self):
        self.assertEqual(legacy_parameter_flags(self.clean, {"moved_mask": True}), ["--moved-mask"])
        self.assertEqual(legacy_parameter_flags(self.clean, {"moved_mask": False}), [])

    def test_undeclared_names_are_refused_including_the_old_cli_key(self):
        self.assertEqual(
            legacy_parameter_flags(self.clean, {"cli": {"dilate": 12}, "whatever": 1}), []
        )

    def test_zero_is_forwarded_rather_than_treated_as_unset(self):
        """bottom_extra=0 is a real instruction: use no downward margin."""
        self.assertIn("--bottom-extra", legacy_parameter_flags(self.clean, {"bottom_extra": 0}))

    def test_a_stage_without_declared_parameters_forwards_nothing(self):
        graph = instantiate_graph(GraphOptions())
        pi3x = graph.nodes["pi3x"].definition.model_dump(mode="json")
        self.assertEqual(legacy_parameter_flags(pi3x, {"dilate": 12}), [])


class MultipersonFlagTests(unittest.TestCase):
    """Stages that only exist in the multiperson graph must ask for that graph."""

    def test_an_explicit_cap_is_passed_through(self):
        self.assertEqual(
            legacy_people_flags({"people": 16, "all_people": True}), ["--people", "16"]
        )

    def test_all_people_without_a_cap_uses_the_flag(self):
        self.assertEqual(legacy_people_flags({"all_people": True}), ["--all-people"])

    def test_a_single_person_run_asks_for_nothing(self):
        self.assertEqual(legacy_people_flags({"people": 1, "all_people": False}), [])

    def test_missing_options_ask_for_nothing(self):
        self.assertEqual(legacy_people_flags({}), [])


if __name__ == "__main__":
    unittest.main()
