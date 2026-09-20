"""A stage's declared parameters must reach its command line.

The legacy adapter used to forward only a nested ``cli`` key, which every stage schema forbids
with ``additionalProperties: false``. Operator and agent retries therefore ran with defaults and
produced byte-identical output, which looks like a stage that ignores its own tuning.

The rest of these check the other half: a stage's overridable defaults are declared in its
``parameter_schema``, so the reviewing agent can see what a stage ran with. A default written
there that the underlying script does not have, or a knob it does not accept, would be a lie
told to the agent in ``task.json``, so both are checked against the scripts themselves.
"""

import ast
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

from types import SimpleNamespace

from orchestrator.activities.adapters import (
    default_adapters,
    flags,
    legacy_parameter_flags,
    legacy_people_flags,
    tuning,
)
from orchestrator.graph import instantiate_graph
from orchestrator.stages import GraphOptions
from orchestrator.stages.registry import stage_registry


def script_arguments(name: str) -> dict[str, object]:
    """Every ``--flag`` a script accepts, mapped to its argparse default.

    Read from the source rather than by importing, because these scripts pull in torch and
    friends at import time. A default named rather than written out is resolved against the
    script's module-level constants; anything else is reported as unknown and not compared.
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


UNKNOWN = object()


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


# Stage -> the script its adapter runs. A legacy stage reaches run_clip.py, which forwards the
# declared names as --flags; the others call their script directly.
LEGACY_STAGES = ("clean_first", "clean", "tracks", "scale_fit", "anchors", "finetune")
DIRECT_STAGES = {
    "object_detect": "detect_object_flights.py",
    "object_lift": "lift_object_3d.py",
    "object_describe": "describe_object.py",
    "person_prep": "prepare_lhm_person.py",
    "world_prompt": "world_prompt.py",
}


# A stage that deliberately runs a script away from the script's own default, and passes the
# flag on every command to do it. The value here is the script's default, so a change to it
# still fails this test and gets looked at rather than silently moving the stage.
OVERRIDDEN_DEFAULTS = {("person_prep", "method"): "segformer"}


def declared(stage_id: str) -> dict[str, dict]:
    schema = stage_registry()[stage_id].parameter_schema or {}
    return schema.get("properties", {})


class GraphShapeTests(unittest.TestCase):
    """Each legacy stage has to name a stage the graph it asks for actually contains.

    run_clip.py calls the per-person stages `lhm_frozen_00`, `lhm_motion_00` and
    `package_people` in its multiperson graph, and `lhm_frozen`, `lhm_motion` and `package` in
    its single-person one. The orchestrator wants the single-person names: it does its own
    fan-out, so each person is a node with its own attempt directory holding one person. Asking
    for the multiperson graph everywhere made those three name something it does not contain.
    """

    def command(self, executor: str, stage_id: str) -> tuple[str, ...]:
        adapter = default_adapters()[executor]
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        context = SimpleNamespace(
            request=SimpleNamespace(
                run_id="run-1",
                node_id=stage_id,
                definition={"inputs": {}, "outputs": {}, "retry": {}},
                parameters={},
                options={"people": 16, "all_people": True},
            ),
            attempt=SimpleNamespace(outputs=root),
            repository=Path("."),
            inputs={"source": (root / "source.mp4",)},
        )
        return adapter.build(context).command

    def test_only_tracking_asks_for_the_multiperson_graph(self):
        stages = {
            "track_people": ("tracks", "tracks", True),
            "lhm_frozen": ("lhm_frozen:00", "lhm_frozen", False),
            "lhm_motion": ("lhm_motion:00", "lhm_motion", False),
            "package_people": ("package_people", "package", False),
        }
        for executor, (node_id, legacy, multiperson) in stages.items():
            with self.subTest(stage=node_id):
                command = self.command(executor, node_id)
                self.assertEqual(command[command.index("--only") + 1], legacy)
                self.assertEqual("--people" in command, multiperson)

    def test_every_named_stage_exists_in_the_graph_it_asks_for(self):
        """Read run_clip.py's own graph builders rather than trusting a list here."""
        source = (ROOT / "scripts/run_clip.py").read_text()

        def named(start: str, end: str) -> set[str]:
            return set(re.findall(r'"([a-z_0-9]+)"', source.split(start)[1].split(end)[0]))

        # Both graphs open with world_half(), so its stages belong to each of them.
        world = named("def world_half", "def single_graph")
        single = world | named("def single_graph", "def multiperson_graph")
        multi = world | named("def multiperson_graph", "\nclass ")
        for executor, node_id in (
            ("track_people", "tracks"),
            ("lhm_frozen", "lhm_frozen:00"),
            ("lhm_motion", "lhm_motion:00"),
            ("package_people", "package_people"),
            ("clean_video", "clean"),
            ("pi3x", "pi3x"),
            ("frame_align", "frame_align"),
        ):
            command = self.command(executor, node_id)
            legacy = command[command.index("--only") + 1]
            wanted = multi if "--people" in command or "--all-people" in command else single
            with self.subTest(stage=node_id, legacy=legacy):
                self.assertIn(legacy, wanted, f"{legacy} is not in the graph this command asks for")


class DeclaredDefaultTests(unittest.TestCase):
    """A declared default must be the value the stage would have used anyway.

    If they drift, ``task.json`` tells the reviewing agent the stage ran at dilate 20 while it
    really ran at 28, and the agent's next retry reasons from a number that was never true.
    """

    def test_every_knob_says_what_it_is_for(self):
        for stage_id, stage in stage_registry().items():
            for name, rule in (stage.parameter_schema or {}).get("properties", {}).items():
                with self.subTest(stage=stage_id, parameter=name):
                    self.assertTrue(rule.get("description"), "a knob with no description")
                    self.assertIn("type", rule)

    def test_legacy_defaults_match_run_clip(self):
        arguments = script_arguments("run_clip.py")
        for stage_id in LEGACY_STAGES:
            for name, rule in declared(stage_id).items():
                flag = "--" + name.replace("_", "-")
                with self.subTest(stage=stage_id, parameter=name):
                    self.assertIn(flag, arguments, f"run_clip.py has no {flag}")
                    if "default" in rule and arguments[flag] is not UNKNOWN:
                        self.assertEqual(rule["default"], arguments[flag])

    def test_directly_called_scripts_accept_every_knob(self):
        for stage_id, script in DIRECT_STAGES.items():
            arguments = script_arguments(script)
            for name, rule in declared(stage_id).items():
                flag = "--" + name.replace("_", "-")
                if stage_id == "world_prompt" and name == "samples":
                    flag = "--n"  # world_prompt.py's own name for the sample count
                with self.subTest(stage=stage_id, parameter=name):
                    self.assertIn(flag, arguments, f"{script} has no {flag}")
                    expected = OVERRIDDEN_DEFAULTS.get((stage_id, name), rule.get("default"))
                    if "default" in rule and arguments[flag] is not UNKNOWN:
                        self.assertEqual(expected, arguments[flag])

    def test_admission_knobs_match_shot_cuts(self):
        arguments = script_arguments("shot_cuts.py")
        knobs = declared("admission")
        self.assertEqual(knobs["cut_threshold"]["default"], arguments["--threshold"])
        self.assertEqual(knobs["min_seconds"]["default"], arguments["--min-seconds"])

    def test_an_optional_knob_has_no_default_rather_than_a_null_one(self):
        """`scale0` and the reference `frame` mean "let the stage decide" when unset.

        A null default would be rendered as a flag value of None by anything that reads
        defaults, so the schema simply omits it and the adapter leaves the flag off.
        """
        self.assertNotIn("default", declared("scale_fit")["scale0"])
        self.assertNotIn("default", declared("person_prep")["frame"])


class ResolvedSettingTests(unittest.TestCase):
    def context(self, stage_id: str, parameters: dict):
        definition = stage_registry()[stage_id].model_dump(mode="json")
        request = SimpleNamespace(definition=definition, parameters=parameters)
        return SimpleNamespace(request=request)

    def test_defaults_resolve_without_any_parameters(self):
        settings = tuning(self.context("clean", {}))
        self.assertEqual(settings["dilate"], 20)
        self.assertEqual(settings["lama_px"], 960)
        self.assertIs(settings["moved_mask"], False)

    def test_one_override_leaves_the_rest_at_their_defaults(self):
        settings = tuning(self.context("clean", {"dilate": 48}))
        self.assertEqual(settings["dilate"], 48)
        self.assertEqual(settings["bottom_extra"], 40)

    def test_an_undeclared_name_is_ignored_rather_than_rendered(self):
        settings = tuning(self.context("clean", {"hypothesis": "halo survives", "nonsense": 1}))
        self.assertNotIn("hypothesis", settings)
        self.assertNotIn("nonsense", settings)

    def test_flags_render_booleans_bare_and_omit_what_is_unset(self):
        self.assertEqual(
            flags({"min_len": 6, "keep_all": True, "quiet": False, "frame": None}),
            ["--keep-all", "--min-len", "6"],
        )


if __name__ == "__main__":
    unittest.main()
