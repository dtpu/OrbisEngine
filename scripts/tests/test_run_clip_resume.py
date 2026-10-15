"""Resume invalidates derived outputs without discarding paid artifacts or widening --only."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_clip


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    p = object.__new__(run_clip.Pipeline)
    p.ctx = tmp_path / "run"
    p.ctx.mkdir()
    p.name = "fixture"
    p.clip = tmp_path / "source.mp4"
    p.clip.write_bytes(b"retained source")
    p.marble = "none"
    p.multi = False
    p.people_limit = 2
    p.a = SimpleNamespace(
        only=None,
        force=None,
        skip_finetune=False,
        reuse_world=None,
        quality_plan=None,
        quality_result=None,
        quality_evidence_root=None,
    )
    p.stages, p.deps = run_clip.single_graph("none", objects=True, finetune=False)
    p.state = run_clip.State(p.ctx / "state.json")
    p.summary = Mock()
    monkeypatch.setattr(run_clip.time, "sleep", lambda _: None)
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(tmp_path / "public"))
    return p


def settle(p):
    for stage in p.stages:
        p.state.record(stage, status="ok")


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_repackage_restores_objects_from_retained_inputs(pipeline, monkeypatch, multi, subset):
    p = pipeline
    package = "package_people" if multi else "package"
    if multi:
        p.multi = True
        p.stages, p.deps = run_clip.multiperson_graph(2, "none", objects=True, finetune=False)
        (p.ctx / "tracks").mkdir()
        (p.ctx / "tracks/tracks.json").write_text(json.dumps({"duration": 1}))
    settle(p)
    for folder in ("lhm-motion", "lhm-motion-00", "lhm-motion-01"):
        directory = p.ctx / folder
        directory.mkdir()
        (directory / "sequence.json").write_text(json.dumps({"duration": 1, "frames": []}))
    receipt = p.ctx / "lhm-motion/recovery-receipt.json"
    receipt.write_bytes(b"retained paid receipt")
    world = run_clip.public_dir() / "worlds/fixture-4d"
    world.mkdir(parents=True)
    objects = world / "objects.json"
    objects.write_bytes(b"retained object package")
    calls = []

    def package_locally(command, log):
        calls.append(package)
        assert p.state.data["stages"]["objects"]["status"] == "pending"
        world.mkdir(parents=True)
        (world / "person").mkdir()

    def restore_objects():
        calls.append("objects")
        assert (world / "person").is_dir()
        objects.write_bytes(b"restored object package")

    monkeypatch.setattr(run_clip, "source_audio_manifest", Mock())
    monkeypatch.setattr(run_clip, "run", package_locally)
    monkeypatch.setattr(p, "objects", restore_objects)
    p.a.force = package
    p.a.only = package if subset else None
    assert p.go() == 0
    if subset:
        assert calls == [package]
        assert not objects.exists()
        assert not p.state.done("objects")
        # A new scheduler instance reads the persisted invalidation after --only finishes.
        p.state = run_clip.State(p.ctx / "state.json")
        p.a.only = p.a.force = None
        assert p.go() == 0
    assert calls == [package, "objects"]
    assert objects.read_bytes() == b"restored object package"
    assert receipt.read_bytes() == b"retained paid receipt"
    assert p.state.done("lhm_motion_00" if multi else "lhm_motion")
    assert p.state.done("objects")


def test_failed_upstream_retry_invalidates_transitive_success_before_launch(pipeline, monkeypatch):
    p = pipeline
    p.stages = ["upstream", "middle", "downstream", "unrelated"]
    p.deps = {"upstream": [], "middle": ["upstream"], "downstream": ["middle"], "unrelated": []}
    settle(p)
    p.state.record("upstream", status="failed")
    p.a.only = "upstream"

    def fail():
        saved = run_clip.State(p.ctx / "state.json")
        assert not saved.done("middle")
        assert not saved.done("downstream")
        assert saved.done("unrelated")
        raise RuntimeError("upstream still unavailable")

    monkeypatch.setattr(p, "stage_fn", lambda _: fail)
    assert p.go() == 1
    # Selecting a descendant alone cannot accept the invalidated dependency.
    p.a.only = "downstream"
    work = Mock()
    monkeypatch.setattr(p, "stage_fn", work)
    assert p.go() == 1
    work.assert_not_called()


@pytest.mark.parametrize("stale", ["place_fit", "anchors"])
def test_receipt_triggered_rerun_invalidates_excluded_descendants(pipeline, monkeypatch, stale):
    p = pipeline
    p.stages = ["place_fit", "anchors", "verify"]
    p.deps = {"place_fit": [], "anchors": ["place_fit"], "verify": ["anchors"]}
    settle(p)
    p.a.only = stale
    monkeypatch.setattr(p, "world_spz", lambda: p.ctx / "world.spz")
    monkeypatch.setattr(p, "placement_is_current", lambda _: stale != "place_fit")
    monkeypatch.setattr(p, "anchors_are_current", lambda _: stale != "anchors")
    calls = []
    monkeypatch.setattr(p, "stage_fn", lambda stage: lambda: calls.append(stage))
    assert p.go() == 0
    assert calls == [stale]
    assert not p.state.done("verify")
    if stale == "place_fit":
        assert not p.state.done("anchors")


def test_cached_skipped_slots_stay_settled_until_tracks_rerun(pipeline, monkeypatch):
    p = pipeline
    p.multi = True
    p.stages, p.deps = run_clip.multiperson_graph(2, "none", objects=True, finetune=False)
    settle(p)
    skipped = {"person_prep_01", "lhm_frozen_01", "lhm_motion_01"}
    for stage in skipped:
        p.state.record(stage, status="skipped", error="one track previously")
    # Old track-count inference must not bypass pending slot work after tracks changes.
    monkeypatch.setattr(p, "vacant", lambda _: True)
    calls = []

    def stage_fn(stage):
        def execute():
            if stage == "package_people":
                assert skipped <= set(calls)
            calls.append(stage)

        return execute

    monkeypatch.setattr(p, "stage_fn", stage_fn)
    assert p.go() == 0
    assert calls == []
    p.a.force = "tracks"
    assert p.go() == 0
    expected = set(p.stages) - {"clean", "pi3x", "frame_align"}
    assert set(calls) == expected
    assert all(p.state.done(stage) for stage in skipped)


@pytest.mark.parametrize("before,after", [(1, 1), (1, 2), (2, 1)])
def test_orchestrated_slots_follow_settled_track_count(pipeline, monkeypatch, before, after):
    p = pipeline
    p.multi = True
    p.stages, p.deps = run_clip.multiperson_graph(3, "none", objects=False, finetune=False)
    tracks = p.ctx / "tracks/tracks.json"
    tracks.parent.mkdir()
    tracks.write_text(json.dumps({"trackCount": before}))
    # CLI plan_for only executes the actual cast, so surplus slots have no stage record.
    for stage in p.stages:
        if not p.vacant(stage) and stage != "package_people":
            p.state.record(stage, status="ok")
    calls = []

    def stage_fn(stage):
        def execute():
            calls.append(stage)
            if stage == "tracks":
                tracks.write_text(json.dumps({"trackCount": after}))

        return execute

    monkeypatch.setattr(p, "stage_fn", stage_fn)
    p.a.only = "package_people"
    assert p.go() == 0
    assert calls == ["package_people"]
    p.a.only = p.a.force = "tracks"
    assert p.go() == 0
    assert p.state.data["stages"]["lhm_motion_02"]["status"] == "pending"
    # Current occupied slots must be rebuilt before packaging, including newly found people.
    p.a.only, p.a.force = "package_people", None
    assert p.go() == 1
    assert calls == ["package_people", "tracks"]
    for index in range(after):
        for base in ("person_prep", "lhm_frozen", "lhm_motion"):
            p.a.only = f"{base}_{index:02d}"
            assert p.go() == 0
    p.a.only = "package_people"
    assert p.go() == 0
    assert calls[-1] == "package_people"
    assert "lhm_motion_02" not in calls


def test_unsettled_tracking_cannot_infer_excluded_vacancy(pipeline, monkeypatch):
    p = pipeline
    p.multi = True
    p.stages, p.deps = run_clip.multiperson_graph(2, "none", objects=False, finetune=False)
    settle(p)
    monkeypatch.setattr(p, "vacant", lambda stage: stage.endswith("_01"))
    monkeypatch.setattr(p, "stage_fn", lambda _: lambda: None)
    p.a.only = p.a.force = "pi3x"
    assert p.go() == 0
    assert not p.state.done("tracks")
    p.a.only, p.a.force = "package_people", None
    work = Mock()
    monkeypatch.setattr(p, "stage_fn", work)
    assert p.go() == 1
    work.assert_not_called()
