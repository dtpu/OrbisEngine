"""Run-scoped operator review using real local evidence validation and no providers."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.api.app import ApiSettings
from orchestrator.api.service import create_service
from orchestrator.cli import plan_for
from orchestrator.database import Base
from orchestrator.journal import MANUAL_WORLD_REVIEW, Journal, RunProjection
from orchestrator.repository import PipelineRepository
from orchestrator.world_review import (
    RetainedReviewPipeline,
    quality,
    read_run_context,
)
from quality_gate import (
    ACKNOWLEDGEMENTS,
    CRITERIA,
    EVIDENCE_SCHEMA,
    PASS_FRACTION,
    PLAN_SCHEMA,
    file_sha256,
)

TOKEN = "local-test-token-at-least-24-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ENDPOINT = "/api/pipeline/runs/review-test/world-review"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def retained(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
    settings = ApiSettings(bearer_token=TOKEN, code_revision="review-test-revision")
    root = tmp_path / "runs"
    app = create_service(
        repository=repository,
        settings=settings,
        runs_root=root,
        artifact_root=tmp_path / "artifacts",
    )
    client = TestClient(app)
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=16x16:r=2:d=1.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    submission = client.post(
        "/api/pipeline/runs/upload",
        headers=AUTH,
        files={"source": ("review-test.mp4", source.read_bytes(), "video/mp4")},
        data={
            "requested_id": "review-test",
            "options": '{"marble":"image","objects":false,"hands_off":true}',
        },
    )
    assert submission.status_code == 201, submission.text
    run = next(path.parent for path in root.glob("*/run.json"))
    context = read_run_context(run, "review-test")
    clip = run / "public" / "clips" / run.name / "source.mp4"
    clip.parent.mkdir(parents=True)
    shutil.copy2(source, clip)
    public = run / "public"
    world = public / f"marble-{run.name}-image.spz"
    world.write_bytes(b"synthetic world")
    world_dir = public / "worlds" / f"{run.name}-4d"
    frame = world_dir / "person" / "frame_000.ply"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        b"property float z\nproperty float opacity\nend_header\n0 -1 -2 2\n0 1 -2 2\n"
    )
    write_json(
        frame.parent / "sequence.json",
        {
            "sourceSha256": context.source_sha256,
            "frames": [frame.name],
            "frame_sha256": [file_sha256(frame)],
        },
    )
    write_json(
        world_dir / "audio.json",
        {
            "schema": "wander.audio/1",
            "source": {"hasAudio": False},
            "timeline": {"durationSeconds": 1.5},
        },
    )
    source_times = quality.source_timestamps(clip)
    cameras = world_dir / "cameras.json"
    write_json(
        cameras,
        {
            "cameras": [
                {
                    "sourceIndex": index,
                    "time": timestamp - source_times[0],
                    "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "source_intrinsics": [[10, 0, 8], [0, 10, 8], [0, 0, 1]],
                    "source_image_size": [16, 16],
                }
                for index, timestamp in enumerate(source_times)
            ]
        },
    )
    placement = world_dir / "placement.json"
    write_json(placement, {})
    anchors = run / "anchors.json"
    write_json(anchors, {"ok": True})
    observation = world.with_name(world.stem + "-obs.bin")
    observation.write_bytes(b"synthetic observations")
    diagnostic = run / "verify" / "diagnostic-fixture"
    diagnostic.mkdir(parents=True)
    report_path = diagnostic / "report.json"

    stages = {name: {"status": "ok"} for name in plan_for(context)}
    stages["_scale"] = {"scale0": 1.0}
    stages["verify"] = {"status": "failed"}
    stages["_verify"] = {"status": "blocked", "passed": False, "diagnosticReport": str(report_path)}
    state = {"stages": stages}
    runner = RetainedReviewPipeline(context, state)
    stages["_placement_solve"] = {
        "world": str(world),
        "worldSha256": file_sha256(world),
        "reportSha256": file_sha256(placement),
        "inputReceipt": runner.placement_inputs(world),
    }
    stages["_anchors"] = {
        "worldSha256": file_sha256(world),
        "placementSha256": file_sha256(placement),
        "reportSha256": file_sha256(anchors),
        "observationSha256": file_sha256(observation),
        "inputReceipt": runner.anchor_inputs(world),
    }
    assert runner.anchors_are_current(world)
    write_json(run / "state.json", state)
    inputs = quality.current_inputs(clip, world, cameras, 1.0)
    rows, samples = [], []
    for index, timestamp in enumerate(source_times):
        elapsed = timestamp - source_times[0]
        row = {"frame": index, "sourcePtsSeconds": timestamp, "timeSeconds": elapsed}
        png = quality.source_frame_png(clip, index)
        for field, prefix in (
            ("pair", "pair"),
            ("sourceImage", "source"),
            ("renderImage", "render"),
        ):
            image = diagnostic / f"{prefix}-f{index}.png"
            image.write_bytes(png)
            row[field] = {"path": image.name, "sha256": file_sha256(image)}
        rows.append(row)
        samples.append(
            {
                "id": f"f{index}",
                "sourceFrame": index,
                "timeSeconds": elapsed,
                "critical": True,
                "view": "source-camera",
            }
        )
    write_json(
        report_path,
        {
            "schema": EVIDENCE_SCHEMA,
            "inputs": inputs,
            "frames": rows,
            "evidenceCodeSha256": quality.evidence_code_sha256(),
            "timingMethod": quality.TIMING_METHOD,
            "sourceTimeOriginSeconds": source_times[0],
            "requestedFrames": list(range(len(source_times))),
        },
    )
    write_json(
        diagnostic / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "inputs": inputs,
            "samples": samples,
            "criteria": list(CRITERIA),
            "passFraction": PASS_FRACTION,
            "evidenceReport": {"path": report_path.name, "sha256": file_sha256(report_path)},
        },
    )
    journal = Journal(run, projection=RunProjection(repository, context.run_id))
    journal.append("run.finished", status="failed", summary="Manual review needed")
    original_run = subprocess.run
    calls = []

    def local_only(argv, *args, **kwargs):
        tool = Path(argv[0]).name
        assert tool in {"ffprobe", "ffmpeg"}, argv
        calls.append(tool)
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_only)
    with client:
        yield client, context, journal, diagnostic, calls
    engine.dispose()


def submitted_review(client):
    loaded = client.get(ENDPOINT, headers=AUTH)
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    result = body["reviewTemplate"]
    assert result["reviewer"] == ""
    assert not any(result["acknowledgements"].values())
    result["reviewer"] = "Synthetic test operator"
    result["acknowledgements"] = ACKNOWLEDGEMENTS.copy()
    for sample in result["samples"]:
        for judgment in sample["criteria"].values():
            assert judgment["status"] == "unknown"
            judgment.update(status="pass", reason="Synthetic contract fixture")
    payload = {
        "diagnosticId": body["diagnosticId"],
        "planSha256": body["planSha256"],
        "result": result,
    }
    return body, payload


def test_read_pairs_and_accept_world_without_reopening_work(retained):
    client, context, journal, diagnostic, calls = retained
    body, payload = submitted_review(client)
    assert body["canSubmit"]
    assert client.get(ENDPOINT).status_code == 401
    assert client.post(ENDPOINT, json=payload).status_code == 401
    pair = body["samples"][0]["pairUrl"]
    assert client.get(pair).status_code == 401
    response = client.get(pair, headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert hashlib.sha256(response.content).hexdigest() == body["samples"][0]["pairSha256"]

    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "reviewStatus": "passed",
        "runStatus": "succeeded",
        "errors": [],
        "completionReason": None,
    }
    assert calls == ["ffprobe", "ffmpeg", "ffmpeg", "ffmpeg"]
    state = json.loads((context.run_dir / "state.json").read_text())
    assert state["stages"]["_verify"]["passed"]
    assert state["stages"]["verify"]["status"] == "ok"
    assert len(list(diagnostic.glob("manual-*.json"))) == 1
    assert journal.finished() == "succeeded"
    assert "verify" in journal.done()
    run = client.get(ENDPOINT.removesuffix("/world-review"), headers=AUTH)
    assert run.status_code == 200, run.text
    assert [
        (attempt["nodeId"], attempt["status"])
        for attempt in run.json()["attempts"]
        if attempt["nodeId"] == "verify"
    ] == [("verify", "succeeded")]
    assert [(e.kind, e.data.get("step")) for e in journal.entries()] == [
        ("run.finished", None),
        ("step.finished", "verify"),
        ("run.finished", None),
    ]
    assert journal.attempt_number("verify") == 2
    assert client.post(ENDPOINT, headers=AUTH, json=payload).status_code == 409


def test_short_review_write_does_not_leave_unresolved_or_malformed_journal(retained, monkeypatch):
    client, _, journal, _, _ = retained
    _, payload = submitted_review(client)
    original_write = os.write
    before = journal.path.read_bytes()

    def interrupt(descriptor, payload):
        if MANUAL_WORLD_REVIEW.encode() in payload:
            return original_write(descriptor, payload[: len(payload) // 2])
        return original_write(descriptor, payload)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "write", interrupt)
        assert client.post(ENDPOINT, headers=AUTH, json=payload).status_code == 409
    assert journal.path.read_bytes() == before
    assert journal.finished() == "failed"
    assert not journal.active_attempts()
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["runStatus"] == "succeeded"
    assert "verify" in journal.done()


def test_review_cannot_complete_for_a_different_viewer_scale(retained):
    client, context, journal, _, _ = retained
    _, payload = submitted_review(client)
    write_json(
        context.run_dir / "public" / "worlds" / f"{context.name}-4d" / "scene.json",
        {"schema": "wander.viewer-scene/1", "params": {"scale": "2"}},
    )
    viewer = client.get(ENDPOINT.removesuffix("/world-review") + "/viewer/scene.json", headers=AUTH)
    assert viewer.json()["params"]["scale"] == "2"
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "reviewStatus": "passed",
        "runStatus": "failed",
        "errors": [],
        "completionReason": "The viewer registration scale differs from the reviewed scale",
    }
    assert journal.finished() == "failed"
    assert journal.entries()[-1].data["delivery"]["qualityStatus"] == "blocked"


@pytest.mark.parametrize("changed", ["world", "receipt"])
def test_changes_after_review_write_block_completion(retained, monkeypatch, changed):
    client, context, journal, diagnostic, _ = retained
    _, payload = submitted_review(client)
    original_append = Journal.append

    def mutate_after_record(self, kind, **data):
        event = original_append(self, kind, **data)
        if data.get("origin") == MANUAL_WORLD_REVIEW:
            if changed == "world":
                path = context.run_dir / "public" / f"marble-{context.name}-image.spz"
            else:
                path = next(diagnostic.glob("manual-*.json"))
            path.write_bytes(b"changed after assessment")
        return event

    monkeypatch.setattr(Journal, "append", mutate_after_record)
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["reviewStatus"] == "blocked"
    assert response.json()["runStatus"] == "failed"
    assert journal.finished() == "failed"
    assert journal.last_status("verify") == "blocked"
    state = json.loads((context.run_dir / "state.json").read_text())
    assert not state["stages"]["_verify"]["passed"]


def test_failed_judgment_is_recorded_without_acceptance(retained):
    client, context, journal, diagnostic, _ = retained
    _, payload = submitted_review(client)
    payload["result"]["samples"][0]["criteria"]["room_layout"].update(
        status="fail",
        reason="Synthetic critical layout failure",
    )
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["reviewStatus"] == "failed"
    assert response.json()["runStatus"] == "failed"
    assert len(list(diagnostic.glob("manual-*.json"))) == 1
    assert journal.last_status("verify") == "failed"
    assert not json.loads((context.run_dir / "state.json").read_text())["stages"]["_verify"][
        "passed"
    ]


def test_passed_review_reports_incomplete_stage_without_launching_it(retained):
    client, context, journal, _, calls = retained
    _, payload = submitted_review(client)
    state_path = context.run_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["stages"]["world_prompt"] = {"status": "pending"}
    write_json(state_path, state)
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "reviewStatus": "passed",
        "runStatus": "failed",
        "errors": [],
        "completionReason": "incomplete stages: world_prompt",
    }
    assert journal.finished() == "failed"
    assert calls == ["ffprobe", "ffmpeg", "ffmpeg", "ffmpeg"]


@pytest.mark.parametrize(
    "change",
    [
        "active",
        "paused",
        "canceled",
        "stale_plan",
        "pair",
        "symlink",
        "anchors",
        "source",
        "scale",
        "escaping_report",
    ],
)
def test_controlled_or_changed_evidence_cannot_submit(retained, change):
    client, context, journal, diagnostic, calls = retained
    _, payload = submitted_review(client)
    if change == "active":
        journal.append("step.started", step="anchors", attempt="unfinished")
    elif change == "paused":
        (context.run_dir / "paused").touch()
    elif change == "canceled":
        journal.append("run.finished", status="canceled", summary="operator canceled")
    elif change == "stale_plan":
        payload["planSha256"] = "1" * 64
    elif change == "pair":
        (diagnostic / "pair-f0.png").write_bytes(b"changed")
    elif change == "symlink":
        pair = diagnostic / "pair-f0.png"
        real = pair.rename(pair.with_name("private-other.png"))
        pair.symlink_to(real)
    elif change == "anchors":
        (context.run_dir / "anchors.json").write_text("{}")
    elif change == "source":
        context.source.write_bytes(b"other source")
    elif change in {"scale", "escaping_report"}:
        state_path = context.run_dir / "state.json"
        state = json.loads(state_path.read_text())
        if change == "scale":
            state["stages"]["_scale"]["scale0"] = 2.0
        else:
            state["stages"]["_verify"]["diagnosticReport"] = str(
                context.run_dir.parent / "other" / "verify" / "diagnostic-fixture" / "report.json"
            )
        write_json(state_path, state)
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 409, response.text
    assert not list(diagnostic.glob("manual-*.json"))
    assert journal.last_status("verify") is None
    assert not calls


def test_missing_bundle_and_nonhuman_review_do_not_replace_verdict(retained):
    client, context, journal, diagnostic, _ = retained
    _, payload = submitted_review(client)
    payload["result"]["judgeKind"] = "llm"
    response = client.post(ENDPOINT, headers=AUTH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["reviewStatus"] == "blocked"
    assert not list(diagnostic.glob("manual-*.json"))
    assert journal.last_status("verify") is None
    state_path = context.run_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["stages"]["_verify"].pop("diagnosticReport")
    write_json(state_path, state)
    assert client.get(ENDPOINT, headers=AUTH).status_code == 404
    assert client.post(ENDPOINT, headers=AUTH, json=payload).status_code == 404
