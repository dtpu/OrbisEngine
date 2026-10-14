"""Local acceptance receipts bind the inputs actually measured, without repeating inference."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

import run_clip
import world_quality_review
from test_final_world_selection import pipeline, accept_placement, accept_anchors  # noqa: F401


@pytest.fixture
def accepted(request):
    p = request.getfixturevalue("pipeline")
    for stage in p.stages:
        p.state.record(stage, status="ok")
    wd = run_clip.public_dir() / "worlds/fixture-4d"
    (wd / "people.json").write_text(
        '{"people": [{"id": "person", "track": 0, "sequence": "person/sequence.json"}]}'
    )
    tracks = p.ctx / "tracks"
    (tracks / "track_00").mkdir(parents=True)
    (tracks / "masks.npz").write_bytes(b"observed masks")
    (tracks / "track_00/motion.json").write_text('{"frames": []}')
    accept_placement(p)
    accept_anchors(p)
    assert p.placement_is_current(p.world_spz())
    assert p.anchors_are_current(p.world_spz())
    return p


def input_path(p, kind):
    wd = run_clip.public_dir() / "worlds/fixture-4d"
    return {
        "cameras": wd / "cameras.json",
        "people": wd / "people.json",
        "sequence": wd / "person/sequence.json",
        "frame": wd / "person/frame_000.ply",
        "masks": p.ctx / "tracks/masks.npz",
        "motion": p.ctx / "tracks/track_00/motion.json",
    }[kind]


@pytest.mark.parametrize("kind", ["cameras", "people", "sequence", "frame", "masks", "motion"])
@pytest.mark.parametrize("change", ["mutated", "missing"])
def test_changed_or_missing_placement_inputs_invalidate_placement_and_anchor_acceptance(
    accepted, kind, change
):
    p = accepted
    path = input_path(p, kind)
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n ")
    assert not p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())
    assert p.state.done("finetune")  # Invalidation is local; it does not erase paid work.


@pytest.mark.parametrize("kind", ["scale", "floor", "size-gate"])
def test_effective_registration_and_size_gate_options_are_bound(accepted, kind):
    p = accepted
    if kind == "scale":
        p.state.record("_scale", scale0=1.25)
    elif kind == "floor":
        p.placement.return_value = {"floor": -1.25}
    else:
        p.a.allow_bad_size = True
    assert not p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())


def test_adding_legacy_motion_metadata_invalidates_mask_only_interpretation(accepted):
    p = accepted
    motion = input_path(p, "motion")
    motion.unlink()
    accept_placement(p)
    accept_anchors(p)
    assert p.anchors_are_current(p.world_spz())
    motion.write_text('{"frames": [{"sample":0,"sourceIndex":0,"box":[0,0,100,500]}]}')
    assert not p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())


@pytest.mark.parametrize(
    "kind", ["video", "ruler-tolerance", "ruler-model", "anchor-report", "sidecar"]
)
def test_anchor_specific_inputs_and_outputs_invalidate_only_anchor_acceptance(
    accepted, monkeypatch, kind
):
    p = accepted
    if kind == "video":
        p.clip.write_bytes(b"changed source video")
    elif kind == "ruler-tolerance":
        p.a.ruler_tol = 0.1
    elif kind == "ruler-model":
        monkeypatch.setenv("WANDER_METRIC_DEPTH", "another-metric-model")
    elif kind == "anchor-report":
        (p.ctx / "anchors.json").write_text('{"ok": false}')
    else:
        world = p.world_spz()
        world.with_name(world.stem + "-obs.bin").unlink()
    assert p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())


@pytest.mark.parametrize("kind", ["sequence", "people"])
def test_malformed_manifest_is_stale_without_crashing_reuse(accepted, kind):
    p = accepted
    input_path(p, kind).write_text("not JSON")
    assert not p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())


def test_unreadable_floor_measurement_is_stale_without_crashing_reuse(accepted):
    p = accepted
    p.placement.side_effect = RuntimeError("corrupt person geometry")
    assert not p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())


@pytest.mark.parametrize("legacy", ["placement", "anchors"])
def test_legacy_receipts_schedule_only_local_rechecks(accepted, monkeypatch, legacy):
    p = accepted
    p.state.record("_placement_solve" if legacy == "placement" else "_anchors", inputReceipt=None)
    calls = []
    monkeypatch.setattr(p, "stage_fn", lambda stage: lambda: calls.append(stage))
    monkeypatch.setattr(p, "summary", Mock())
    monkeypatch.setattr(run_clip.time, "sleep", lambda _: None)
    assert p.go() == 0
    assert calls == (
        ["place_fit", "anchors", "verify"] if legacy == "placement" else ["anchors", "verify"]
    )
    assert p.state.done("finetune")


def test_only_verify_cannot_bypass_changed_observations(accepted, monkeypatch):
    p = accepted
    input_path(p, "motion").write_text('{"frames": [], "cropBoundsChanged": true}')
    p.a.only = "verify"
    p.a.quality_plan, p.a.quality_result, p.a.quality_evidence_root = "plan", "result", "evidence"
    assess = Mock(return_value={"status": "passed", "errors": []})
    monkeypatch.setattr(world_quality_review, "assess", assess)
    monkeypatch.setattr(p, "summary", Mock())
    monkeypatch.setattr(run_clip.time, "sleep", lambda _: None)
    assert p.go() == 1
    assess.assert_not_called()
    assert not p.state.done("verify")
    assert p.state.done("finetune")


def test_placement_receipt_is_not_recorded_for_inputs_changed_during_solve(accepted, monkeypatch):
    p = accepted
    receipt = p.state.data["stages"]["_placement_solve"]["inputReceipt"]

    def solve(*args, **kwargs):
        input_path(p, "cameras").write_text('{"cameras": [], "changedWhileSolving": true}')

    monkeypatch.setattr(run_clip, "run", solve)
    with pytest.raises(RuntimeError, match="inputs changed during solve"):
        p.place_fit()
    assert p.state.data["stages"]["_placement_solve"]["inputReceipt"] == receipt
    assert not p.placement_is_current(p.world_spz())


def test_anchor_receipt_is_not_recorded_for_inputs_changed_during_checks(accepted, monkeypatch):
    p = accepted
    receipt = p.state.data["stages"]["_anchors"]["inputReceipt"]

    def export(command, *args, **kwargs):
        Path(command[command.index("--out") + 1]).write_bytes(b"new visibility measurement")

    def check(command, **kwargs):
        Path(command[command.index("--json-out") + 1]).write_text('{"ok": true}')
        p.clip.write_bytes(b"changed while checking")
        return SimpleNamespace(returncode=0, stdout="checked")

    monkeypatch.setattr(run_clip, "run", export)
    monkeypatch.setattr(run_clip.subprocess, "run", check)
    with pytest.raises(RuntimeError, match="anchor inputs changed during checks"):
        p.anchors()
    assert p.state.data["stages"]["_anchors"]["inputReceipt"] == receipt
    assert not p.anchors_are_current(p.world_spz())


def test_input_hash_cache_detects_changed_bytes_even_when_mtime_and_size_are_restored(
    accepted, tmp_path, monkeypatch
):
    path = tmp_path / "measured.bin"
    path.write_bytes(b"before")
    p = accepted
    original = run_clip.file_sha256
    calls = []

    def digest(file):
        calls.append(file)
        return original(file)

    monkeypatch.setattr(run_clip, "file_sha256", digest)
    before = p.input_hash(path)
    assert p.input_hash(path) == before
    assert calls == [path.resolve()]
    timestamp = path.stat().st_mtime_ns
    path.write_bytes(b"after!")
    os.utime(path, ns=(timestamp, timestamp))
    assert p.input_hash(path) != before
    assert len(calls) == 2


def test_failed_candidate_is_not_promoted_by_receipt_checks(accepted):
    p = accepted
    p.state.record("anchors", status="failed", error="quality rejected")
    assert p.placement_is_current(p.world_spz())
    assert not p.anchors_are_current(p.world_spz())
    assert p.state.data["stages"]["anchors"]["error"] == "quality rejected"
