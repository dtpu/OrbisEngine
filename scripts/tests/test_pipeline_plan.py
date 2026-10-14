"""Submission options must describe exactly what the runner executes."""

import itertools
import sys
from graphlib import TopologicalSorter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from pydantic import ValidationError

from orchestrator.cli import RunContext, run_clip_command
from orchestrator.pipeline import MARBLE_MODES, multiperson_graph, single_graph
from orchestrator.steps import (
    RunOptions,
    base_step,
    needs_of,
    people_steps,
    planned_steps,
    stage_graph,
    STEPS,
)


@pytest.mark.parametrize(
    "mode,many,objects,finetune",
    itertools.product(MARBLE_MODES, (False, True), (False, True), (False, True)),
)
def test_submitted_plan_matches_executable_graph(mode, many, objects, finetune):
    options = RunOptions(
        marble=mode, people=2 if many else 1, all_people=many, objects=objects, finetune=finetune
    )
    stages, deps = stage_graph(options)
    actual = (
        multiperson_graph(2, mode, objects, finetune)
        if many
        else single_graph(mode, objects, finetune)
    )
    assert (stages, deps) == actual
    assert set(people_steps(planned_steps(options), options.people, multiperson=many)) == set(
        stages
    )
    for step in stages:
        assert base_step(step) in STEPS
        assert needs_of(step, options.people, many, options) == deps[step]
    assert ("tracks" in stages) == many
    assert ("finetune" in stages) == (finetune and mode != "none")
    if mode == "both":
        assert {"clean_first", "clean", "marble_image", "marble_video"} <= set(stages)
    if mode == "multi":
        assert "pi3x" in deps["clean_multi"]


def test_legacy_disabled_audio_is_accepted_but_never_promises_a_missing_stage():
    assert RunOptions.model_validate({"reviewed_audio": False}) == RunOptions()
    with pytest.raises(ValidationError, match="unsupported"):
        RunOptions.model_validate({"reviewed_audio": True})


def test_commands_preserve_one_person_tracking_and_optional_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDER_STAGE_LEDGER", str(tmp_path / "paid-ledger.json"))
    context = RunContext(
        "test",
        tmp_path / "test",
        "test",
        tmp_path / "source.mp4",
        Path.cwd(),
        {"all_people": True, "people": 1, "objects": False, "finetune": False},
    )
    command = run_clip_command(context, "person_prep_00", [])
    assert command[command.index("--people") + 1] == "1"
    assert "--no-objects" in command and "--skip-finetune" in command
    assert "--reject-cuts" in command and "--one-shot" not in command
    assert command[command.index("--stage-ledger") + 1] == str(tmp_path / "paid-ledger.json")


def test_object_detection_and_review_wait_for_required_outputs():
    _, single = single_graph("video", objects=True)
    assert "place_fit" in single["objects"]
    assert {"objects", "anchors", "finetune"} <= set(single["verify"])
    _, many = multiperson_graph(2, "image", objects=True, finetune=False)
    assert "package_people" in many["objects"]
    assert {"objects", "anchors"} <= set(many["verify"])
    assert "finetune" not in many["verify"]


@pytest.mark.parametrize("mode", MARBLE_MODES)
@pytest.mark.parametrize("many,people", [(False, 1), (True, 1), (True, 3)])
@pytest.mark.parametrize("objects,finetune", itertools.product((False, True), (False, True)))
def test_final_world_is_refined_before_placement_and_anchors(mode, many, people, objects, finetune):
    stages, dependencies = (
        multiperson_graph(people, mode, objects, finetune)
        if many
        else single_graph(mode, objects, finetune)
    )
    # A missing dependency or cycle would leave the supervisor unable to finish this graph.
    ordered = list(TopologicalSorter(dependencies).static_order())
    assert set(ordered) == set(stages)
    assert len(stages) == len(set(stages))
    if mode == "none":
        assert not {"scale_fit", "finetune", "place_fit", "anchors", "verify"} & set(stages)
        return

    assert dependencies["place_fit"] == ["scale_fit"] + (["finetune"] if finetune else [])
    assert dependencies["anchors"] == ["place_fit"]
    assert dependencies["verify"] == (
        ["anchors"] + (["objects"] if objects else []) + (["finetune"] if finetune else [])
    )
    tail = ["scale_fit"]
    if finetune:
        assert dependencies["finetune"] == ["scale_fit"]
        tail.append("finetune")
    else:
        assert "finetune" not in stages
    tail += ["place_fit", "anchors", "verify"]
    # Keep the runner's stage list in the same final-world order as its dependency graph.
    assert [stage for stage in stages if stage in tail] == tail
    assert [stage for stage in ordered if stage in tail] == tail
