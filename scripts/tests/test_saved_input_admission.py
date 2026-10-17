"""Prepared delivery admits one pinned local packaging attempt, never inference."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from orchestrator import cli
from orchestrator.api.progress import stage_snapshot
from scripts.tests import test_run_controls as fixtures

run = fixtures.run


@pytest.fixture
def saved_run(run, monkeypatch):
    options = {"marble": "video", "people": 2, "all_people": True, "finetune": False}
    delivery = {
        "schema": "wander.saved-inputs/1",
        "sourceRunId": "prior",
        "config": "retained/scene.json",
        "inputs": "retained/inputs.json",
        "stage": "package_people",
    }
    run.context = replace(run.context, options=options, retained_delivery=delivery)
    root = run.context.run_dir
    (root / "run.json").write_text(
        json.dumps(
            {
                "runId": run.context.run_id,
                "name": run.context.name,
                "source": str(run.context.source),
                "sourceSha256": run.context.source_sha256,
                "repository": str(run.context.repository),
                "options": run.context.options,
                "retainedDelivery": run.context.retained_delivery,
            }
        )
    )
    (root / "tracks").mkdir()
    (root / "tracks/tracks.json").write_text('{"trackCount":2}')
    verifier = Mock()
    monkeypatch.setattr("orchestrator.retained_inputs.verify_delivery_inputs", verifier)
    run.verifier = verifier
    return run


def args(*tokens):
    return cli.build_parser().parse_args(["step", *tokens])


@pytest.mark.parametrize(
    "tokens",
    [
        ("pi3x",),
        ("tracks",),
        ("marble_video",),
        ("package_people", "--force", "package_people"),
        ("package_people", "--retained-scene", "other.json"),
        ("package_people", "--no-gate"),
    ],
)
def test_prepared_run_rejects_inference_and_overrides(saved_run, monkeypatch, tokens):
    spawn = Mock()
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(saved_run.context, args(*tokens)) == 2
    assert not saved_run.journal.entries()
    spawn.assert_not_called()
    saved_run.verifier.assert_not_called()


def test_one_local_attempt_uses_prepared_config_then_refuses_retry(saved_run, monkeypatch):
    spawn = Mock(return_value=fixtures.stage_process())
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(saved_run.context, args("--wait", "package_people")) == 0
    launched = spawn.call_args.args[0]
    assert launched[-2:] == [
        "--retained-scene",
        str(saved_run.context.run_dir / "retained/scene.json"),
    ]
    assert "--no-publish" in launched
    assert cli.command_step(saved_run.context, args("--wait", "package_people")) == 2
    spawn.assert_called_once()
    assert len([e for e in saved_run.journal.entries() if e.kind == "step.started"]) == 1


def test_detached_delivery_claims_original_attempt_once(saved_run, monkeypatch):
    spawn = Mock(return_value=fixtures.stage_process())
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(saved_run.context, args("package_people")) == 0
    child = cli.build_parser().parse_args(spawn.call_args.args[0][3:])
    assert cli.command_step(saved_run.context, child) == 0
    assert cli.command_step(saved_run.context, child) == 2
    assert cli.command_step(saved_run.context, args("package_people")) == 2
    assert spawn.call_count == 2  # detached waiter plus its one stage process
    assert not saved_run.journal.active_attempts()


def test_changed_inputs_block_execution_and_close_pending_child(saved_run, monkeypatch):
    spawn = Mock(return_value=fixtures.stage_process())
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    assert cli.command_step(saved_run.context, args("package_people")) == 0
    child = cli.build_parser().parse_args(spawn.call_args.args[0][3:])
    saved_run.verifier.side_effect = ValueError("input digest mismatch")
    assert cli.command_step(saved_run.context, child) == 2
    spawn.assert_called_once()
    assert not saved_run.journal.active_attempts()
    result = saved_run.journal.entries()[-1]
    assert result.kind == "step.finished"
    assert result.data["status"] == "blocked"
    assert "digest mismatch" in result.data["error"]


def test_api_cli_and_agent_plan_only_saved_prerequisites_and_packaging(saved_run):
    expected = ["pi3x", "tracks", "lhm_motion_00", "lhm_motion_01", "frame_align", "package_people"]
    assert cli.plan_for(saved_run.context) == expected
    session = fixtures.session(saved_run)
    assert session.plan(saved_run.context.options) == expected
    snapshot = stage_snapshot(saved_run.context.run_dir)
    assert [stage["name"] for stage in snapshot.stages] == expected
    assert all(set(deps) <= set(expected) for deps in snapshot.required_dependencies.values())
