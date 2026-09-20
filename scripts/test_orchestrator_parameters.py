"""A step's declared defaults must be the values the underlying script really uses.

The catalogue tells the agent what a step is running at, and the agent reasons from that: it
moves a knob because the value it was shown did not suit what it saw. A default written here
that the script does not have is a lie told in the brief, so both are read from the scripts
themselves rather than restated.
"""

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.steps import STEPS, suggested_order

UNKNOWN = object()

# A step whose script is deliberately run away from its own default. The value is the script's,
# so a change to it still fails here and gets looked at.
OVERRIDDEN = {("person_prep", "method"): "segformer"}

# Which script each step's knobs are passed to. run_clip.py forwards its own flags; the rest
# are called by it and own theirs.
SCRIPTS = {
    "clean": "run_clip.py",
    "clean_first": "run_clip.py",
    "clean_multi": "run_clip.py",
    "tracks": "run_clip.py",
    "scale_fit": "run_clip.py",
    "anchors": "run_clip.py",
    "finetune": "run_clip.py",
    "person_prep": "prepare_lhm_person.py",
    "world_prompt": "world_prompt.py",
}
# The catalogue's name for a flag the script spells differently.
SPELLINGS = {("world_prompt", "samples"): "--n"}


def script_arguments(name: str) -> dict[str, object]:
    """Every ``--flag`` a script accepts, mapped to its argparse default.

    Read from the source rather than by importing, because these scripts pull in torch and
    friends at import time.
    """
    tree = ast.parse((ROOT / "scripts" / name).read_text())
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)
    }
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        default: object = None
        for keyword in node.keywords:
            if keyword.arg == "default":
                if isinstance(keyword.value, ast.Name) and keyword.value.id in constants:
                    default = constants[keyword.value.id]
                else:
                    try:
                        default = ast.literal_eval(keyword.value)
                    except ValueError:
                        default = UNKNOWN
            if (
                keyword.arg == "action"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "store_true"
            ):
                default = False
        for argument in node.args:
            if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                found[argument.value] = default
    return found


class DeclaredDefaultTests(unittest.TestCase):
    def test_every_knob_says_what_it_is_for(self):
        for name, step in STEPS.items():
            for knob, rule in (step.parameters or {}).get("properties", {}).items():
                with self.subTest(step=name, parameter=knob):
                    self.assertTrue(rule.get("description"), "a knob with no description")
                    self.assertIn("type", rule)

    def test_a_declared_default_is_the_one_the_script_really_uses(self):
        for step_name, script in SCRIPTS.items():
            arguments = script_arguments(script)
            for knob, rule in (STEPS[step_name].parameters or {}).get("properties", {}).items():
                flag = SPELLINGS.get((step_name, knob), "--" + knob.replace("_", "-"))
                with self.subTest(step=step_name, parameter=knob):
                    self.assertIn(flag, arguments, f"{script} has no {flag}")
                    expected = OVERRIDDEN.get((step_name, knob), rule.get("default"))
                    if "default" in rule and arguments[flag] is not UNKNOWN:
                        self.assertEqual(expected, arguments[flag])

    def test_an_optional_knob_has_no_default_rather_than_a_null_one(self):
        """`scale0` and the reference `frame` mean "let the step decide" when unset."""
        self.assertNotIn("default", STEPS["scale_fit"].parameters["properties"]["scale0"])
        self.assertNotIn("default", STEPS["person_prep"].parameters["properties"]["frame"])


class CatalogueTests(unittest.TestCase):
    def test_every_step_names_a_stage_run_clip_actually_has(self):
        """The catalogue is what the agent picks from; a name it cannot run is a dead end."""
        source = (ROOT / "scripts/run_clip.py").read_text()

        def named(start: str, until: str) -> set[str]:
            return set(re.findall(r'"([a-z_0-9]+)"', source.split(start)[1].split(until)[0]))

        world = named("def world_half", "def single_graph")
        known = (
            world
            | named("def single_graph", "def multiperson_graph")
            | named("def multiperson_graph", "\nclass ")
        )
        for name in STEPS:
            with self.subTest(step=name):
                self.assertIn(name, known, f"run_clip.py has no stage called {name}")

    def test_the_order_is_advice_that_at_least_holds_together(self):
        order = suggested_order()
        self.assertEqual(set(order), set(STEPS))
        for name, step in STEPS.items():
            for needed in step.after:
                with self.subTest(step=name, after=needed):
                    self.assertLess(order.index(needed), order.index(name))

    def test_what_costs_money_is_marked(self):
        paid = {name for name, step in STEPS.items() if step.paid}
        for name in ("marble_video", "pi3x", "tracks", "lhm_frozen", "lhm_motion", "finetune"):
            self.assertIn(name, paid, f"{name} spends and must say so in the brief")
        self.assertNotIn("frame_align", paid)


if __name__ == "__main__":
    unittest.main()
