"""The upload service is confined to explicitly automatic run manifests."""

import json
import os

import pytest

from orchestrator import supervisor
from orchestrator.agent.harness import HarnessPolicy
from orchestrator.journal import Journal
from orchestrator.worker_presence import WorkerPresence, worker_status


@pytest.fixture
def inert_worker(monkeypatch):
    monkeypatch.delenv("WANDER_STAGE_AUTHORIZATION", raising=False)
    monkeypatch.setattr(supervisor, "repository_from_environment", lambda: None)
    monkeypatch.setattr(
        HarnessPolicy,
        "from_environment",
        lambda: HarnessPolicy(kind="command", command=("unused",)),
    )


def automatic(root, name):
    return supervisor.open_run(
        root, run_id=name, name=name, source=None, options={"hands_off": True}
    )


@pytest.mark.parametrize(
    "manifest",
    [
        {"runId": "manual"},
        {"runId": "manual", "options": {}},
        {"runId": "manual", "options": {"hands_off": False}},
        {"runId": "manual", "options": {"hands_off": "true"}},
        {"runId": "manual", "options": {"hands_off": 1}},
        {"runId": "manual", "options": []},
        {"runId": "manual", "options": None},
        {"options": {"hands_off": True}},
        {"runId": [], "options": {"hands_off": True}},
        {"runId": "", "options": {"hands_off": True}},
        [],
        None,
        "invalid",
    ],
)
def test_manual_legacy_and_malformed_runs_are_ignored(
    tmp_path, monkeypatch, inert_worker, manifest
):
    run = tmp_path / "clip"
    run.mkdir()
    (run / "run.json").write_text(json.dumps(manifest))
    malformed = tmp_path / "broken-json"
    malformed.mkdir()
    (malformed / "run.json").write_text("{incomplete")
    monkeypatch.setattr(supervisor, "work_on", lambda *_: pytest.fail("out-of-scope work"))
    assert supervisor.main(["--runs", str(tmp_path), "--uploads-only", "--once"]) == 0


def test_symlink_directories_and_manifests_are_ignored(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    outside = automatic(tmp_path, "outside")
    (root / "linked-run").symlink_to(outside, target_is_directory=True)
    linked_manifest = root / "linked-manifest"
    linked_manifest.mkdir()
    (linked_manifest / "run.json").symlink_to(outside / "run.json")
    eligible = automatic(root, "eligible")
    assert supervisor.runs_needing_work(root, uploads_only=True) == [eligible]


@pytest.mark.parametrize("state", ["paused", "canceled", "succeeded", "failed"])
def test_automatic_run_control_and_terminal_states_are_preserved(tmp_path, state):
    run = automatic(tmp_path, "automatic")
    if state == "paused":
        (run / state).touch()
    else:
        Journal(run).append("run.finished", status=state)
    assert supervisor.runs_needing_work(tmp_path, uploads_only=True) == []


def test_automatic_upload_arriving_after_startup_is_picked_up_next_pass(
    tmp_path, monkeypatch, inert_worker
):
    supervisor.open_run(tmp_path, run_id="manual", name="manual", source=None)
    calls = []
    passes = 0

    class StopWatch(Exception):
        pass

    def tick(_):
        nonlocal passes
        passes += 1
        assert worker_status(tmp_path, hands_off=True)["status"] == "available"
        if passes == 1:
            assert calls == []
            automatic(tmp_path, "new-upload")
        else:
            raise StopWatch

    monkeypatch.setattr(supervisor.time, "sleep", tick)
    monkeypatch.setattr(
        supervisor,
        "work_on",
        lambda path, *_: calls.append(path) or supervisor.WorkOutcome("done", 0),
    )
    with pytest.raises(StopWatch):
        supervisor.main(["--runs", str(tmp_path), "--uploads-only"])
    assert calls == [tmp_path / "new-upload"]


@pytest.mark.parametrize("mutation", ["manual", "run-id", "symlink"])
def test_scope_is_checked_again_immediately_before_work(
    tmp_path, monkeypatch, inert_worker, mutation
):
    run = automatic(tmp_path, "selected")

    class ChangedPresence(WorkerPresence):
        def set_active(self, run_id):
            super().set_active(run_id)
            if run_id is not None:
                manifest = run / "run.json"
                if mutation == "symlink":
                    held = tmp_path / "outside.json"
                    manifest.rename(held)
                    manifest.symlink_to(held)
                else:
                    record = json.loads(manifest.read_text())
                    if mutation == "manual":
                        record["options"]["hands_off"] = False
                    else:
                        record["runId"] = "different"
                    manifest.write_text(json.dumps(record))

    monkeypatch.setattr(supervisor, "WorkerPresence", ChangedPresence)
    monkeypatch.setattr(supervisor, "work_on", lambda *_: pytest.fail("scope changed before work"))
    assert supervisor.main(["--runs", str(tmp_path), "--uploads-only", "--run-id", "selected"]) == 0


@pytest.mark.parametrize("permit", ["", "source-specific-permit.json"])
def test_source_scoped_permit_fails_before_presence_or_harness(tmp_path, monkeypatch, permit):
    monkeypatch.setenv("WANDER_STAGE_AUTHORIZATION", permit)
    monkeypatch.setattr(supervisor, "WorkerPresence", lambda *_a, **_k: pytest.fail("registered"))
    monkeypatch.setattr(HarnessPolicy, "from_environment", lambda: pytest.fail("loaded harness"))
    monkeypatch.setattr(
        supervisor, "repository_from_environment", lambda: pytest.fail("opened database")
    )
    root = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        supervisor.main(["--runs", str(root), "--uploads-only", "--once"])
    assert error.value.code == 2
    assert not root.exists()
    assert os.environ["WANDER_STAGE_AUTHORIZATION"] == permit


@pytest.mark.parametrize(
    "mutation",
    [
        {"incomplete": True},
        {"name": None},
        {"name": []},
        {"name": ""},
        {"source": None},
        {"source": []},
        {"repository": None},
        {"repository": []},
        {"repository": ""},
        {"goal": ["bad goal"]},
        {"retainedDelivery": []},
        {"retainedDelivery": "bad delivery"},
        {"options": {"hands_off": True, "people": "invalid"}},
        {"options": {"hands_off": True, "people": "2.0"}},
        {"options": {"hands_off": True, "people": 0}},
        {"options": {"hands_off": True, "people": 17}},
        {"options": {"hands_off": True, "marble": "invalid"}},
        {"options": {"hands_off": True, "marble": []}},
        {"options": {"hands_off": True, "all_people": []}},
    ],
)
def test_malformed_automatic_neighbor_cannot_stop_shared_worker(
    tmp_path, monkeypatch, inert_worker, mutation
):
    from orchestrator.agent.contracts import AgentResult
    from orchestrator.agent.harness import HarnessAgent, HarnessOutcome

    malformed = automatic(tmp_path, "a-malformed")
    manifest_path = malformed / "run.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation.get("incomplete"):
        manifest = {"runId": "a-malformed", "options": {"hands_off": True}}
    else:
        manifest.update(mutation)
    manifest_path.write_text(json.dumps(manifest))
    healthy = automatic(tmp_path, "z-healthy")
    harness_calls = []

    def agent_turn(self, messages, run_dir, brief, **kwargs):
        harness_calls.append(run_dir)
        assert run_dir == healthy
        Journal(run_dir).append("run.finished", status="failed", summary="Inert test complete")
        return HarnessOutcome(AgentResult(status="completed", transcript_path="inert"))

    monkeypatch.setattr(HarnessAgent, "run", agent_turn)
    # Exercise real work_on and RunSession planning; only the provider boundary is inert.
    assert supervisor.main(["--runs", str(tmp_path), "--uploads-only", "--once"]) == 0
    assert harness_calls == [healthy]
    assert Journal(healthy).finished() == "failed"
    assert not (malformed / "BRIEF.md").exists()
    assert not (malformed / "bin").exists()


def test_empty_and_missing_source_remain_eligible_for_stage_preflight(tmp_path):
    run = automatic(tmp_path, "automatic")
    manifest = supervisor.run_manifest(run, uploads_only=True)
    assert manifest is not None and manifest["source"] == ""
    manifest["source"] = str(run / "missing.mp4")
    (run / "run.json").write_text(json.dumps(manifest))
    assert supervisor.runs_needing_work(tmp_path, uploads_only=True) == [run]


@pytest.mark.parametrize("repository", ["absent", "", None])
@pytest.mark.parametrize("people", [1, "2"])
def test_legacy_manual_repository_fallback_and_integer_strings_still_run(
    tmp_path, monkeypatch, inert_worker, repository, people
):
    from orchestrator.agent.contracts import AgentResult
    from orchestrator.agent.harness import HarnessAgent, HarnessOutcome

    run = supervisor.open_run(tmp_path, run_id="legacy", name="legacy", source=None)
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    if repository == "absent":
        del manifest["repository"]
    else:
        manifest["repository"] = repository
    manifest["options"] = {"people": people}
    manifest_path.write_text(json.dumps(manifest))
    assert supervisor.runs_needing_work(tmp_path) == [run]
    assert supervisor.runs_needing_work(tmp_path, uploads_only=True) == []
    harness_calls = []

    def agent_turn(self, messages, run_dir, brief, **kwargs):
        harness_calls.append(run_dir)
        Journal(run_dir).append("run.finished", status="failed", summary="Inert test complete")
        return HarnessOutcome(AgentResult(status="completed", transcript_path="inert"))

    monkeypatch.setattr(HarnessAgent, "run", agent_turn)
    assert supervisor.main(["--runs", str(tmp_path), "--once"]) == 0
    assert harness_calls == [run]
    assert Journal(run).finished() == "failed"


@pytest.mark.parametrize("repository", ["absent", "", None])
def test_retained_manual_manifest_requires_repository(tmp_path, repository):
    run = supervisor.open_run(tmp_path, run_id="retained", name="retained", source=None)
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retainedDelivery"] = {"stage": "package_people"}
    if repository == "absent":
        del manifest["repository"]
    else:
        manifest["repository"] = repository
    manifest_path.write_text(json.dumps(manifest))
    assert supervisor.runs_needing_work(tmp_path) == []


def test_manual_people_value_that_session_cannot_consume_is_ignored(tmp_path):
    run = supervisor.open_run(tmp_path, run_id="legacy", name="legacy", source=None)
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["options"] = {"people": "2.0"}
    manifest_path.write_text(json.dumps(manifest))
    assert supervisor.runs_needing_work(tmp_path) == []
