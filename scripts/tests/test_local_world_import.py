"""Local world import: real SPZ decoding, no providers or accepted-quality promotion."""

import copy
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_world_import as world
import run_clip


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    clip = root / "original.mp4"
    clip.write_bytes(b"original source bytes")
    spz = root / "retained.spz"
    # Two complete v2 degree-zero splats: canonical decoder validates every byte.
    spz.write_bytes(
        gzip.compress(struct.pack("<IIIBBBB", 0x5053474E, 2, 2, 0, 12, 0, 0) + bytes([64]) * 38)
    )
    evidence = root / "snapshot.json"
    evidence.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "/clips/original.mp4", "sha256": world.digest(clip)},
                    {"path": "/marble-old-clean.spz", "sha256": world.digest(spz)},
                ]
            }
        )
    )
    receipt = root / "receipt.json"
    document = {
        "schema": world.SCHEMA,
        "sourceSha256": world.digest(clip),
        "world": {"path": str(spz), "bytes": spz.stat().st_size, "sha256": world.digest(spz)},
        "associationEvidence": [{"path": str(evidence), "sha256": world.digest(evidence)}],
        "generationLineage": "unknown",
        "worldId": None,
        "operationId": None,
        "execution": "imported_local",
        "freshGeneration": False,
    }
    receipt.write_text(json.dumps(document))
    ctx, public, marble = root / "run", root / "public", root / "marble"
    ctx.mkdir()
    monkeypatch.setattr(run_clip, "MARBLE_DIR", marble)
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    return SimpleNamespace(
        root=root,
        clip=clip,
        spz=spz,
        evidence=evidence,
        receipt=receipt,
        document=document,
        ctx=ctx,
        public=public,
        marble=marble,
        name="fixture",
    )


def prepare(f):
    return world.prepare(f.receipt, f.clip, f.ctx, f.public, f.marble, f.name)


def execute(f):
    return world.import_world(f.receipt, f.clip, f.ctx, f.public, f.marble, f.name)


def test_import_is_hash_pinned_and_idempotent_without_provider_metadata(fixture):
    f = fixture
    plan = prepare(f)
    assert not f.public.exists()
    assert not (f.ctx / "local-world-import.json").exists()
    result = execute(f)
    target = Path(plan["target"])
    assert target.read_bytes() == f.spz.read_bytes()
    assert result["generationLineage"] == "unknown"
    assert result["freshGeneration"] is False
    assert result["worldId"] is None
    assert result["operationId"] is None
    assert result["execution"] == "imported_local"
    assert result["splatCount"] == 2
    assert result["qualityStatus"] == "unreviewed"
    original = target.stat().st_mtime_ns
    assert execute(f) == result
    assert target.stat().st_mtime_ns == original
    assert not f.marble.exists()  # No fake provider -world/-request/-generation receipt.


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(sourceSha256="0" * 64),
        lambda d: d["world"].update(sha256="0" * 64),
        lambda d: d["world"].update(bytes=True),
        lambda d: d["world"].update(bytes=999),
        lambda d: d["world"].update(path="relative.spz"),
        lambda d: d["associationEvidence"][0].update(sha256="0" * 64),
        lambda d: d.update(associationEvidence=[]),
        lambda d: d.update(freshGeneration=True),
        lambda d: d.update(generationLineage="proven"),
        lambda d: d.update(worldId="invented-provider-id"),
    ],
)
def test_invalid_receipt_cannot_mutate_outputs(fixture, change):
    f = fixture
    change(f.document)
    f.receipt.write_text(json.dumps(f.document))
    with pytest.raises(ValueError):
        execute(f)
    assert not f.public.exists()
    assert not (f.ctx / "local-world-import.json").exists()


def test_evidence_must_pair_actual_source_and_asset(fixture):
    f = fixture
    f.evidence.write_text(json.dumps({"sha256": world.digest(f.clip)}))
    f.document["associationEvidence"][0]["sha256"] = world.digest(f.evidence)
    f.receipt.write_text(json.dumps(f.document))
    with pytest.raises(ValueError, match="pair both"):
        execute(f)
    assert not f.public.exists()


@pytest.mark.parametrize(
    "raw",
    [
        b"not gzip",
        gzip.compress(b"short"),
        gzip.compress(struct.pack("<IIIBBBB", 0x5053474E, 2, 2, 0, 12, 0, 0) + b"truncated"),
    ],
)
def test_structure_rejected_even_when_all_hash_pins_match(fixture, raw):
    f = fixture
    f.spz.write_bytes(raw)
    f.document["world"].update(bytes=len(raw), sha256=world.digest(f.spz))
    f.evidence.write_text(json.dumps([world.digest(f.clip), world.digest(f.spz)]))
    f.document["associationEvidence"][0]["sha256"] = world.digest(f.evidence)
    f.receipt.write_text(json.dumps(f.document))
    with pytest.raises(ValueError, match="SPZ structure"):
        execute(f)
    assert not f.public.exists()


@pytest.mark.parametrize("which", ["receipt", "spz", "evidence"])
def test_symlink_inputs_rejected(fixture, which):
    f = fixture
    path = getattr(f, which)
    actual = path.with_name(path.name + ".actual")
    path.rename(actual)
    path.symlink_to(actual)
    with pytest.raises(ValueError, match="symlinks"):
        execute(f)
    assert not f.public.exists()


@pytest.mark.parametrize("kind", ["request", "generation", "op", "world", "log", "active", "state"])
def test_existing_or_uncertain_submission_cannot_be_replaced(fixture, kind):
    f = fixture
    if kind in {"request", "generation", "op", "world"}:
        f.marble.mkdir()
        (f.marble / f"fixture-clean-{kind}.json").write_text("{}")
    elif kind == "log":
        (f.ctx / "marble_clean.log").write_text("submission uncertain")
    elif kind == "active":
        (f.ctx / "journal.jsonl").write_text(
            json.dumps(
                {"kind": "step.started", "data": {"step": "marble_video", "attempt": "unresolved"}}
            )
            + "\n"
        )
    else:
        (f.ctx / "state.json").write_text(
            json.dumps({"stages": {"_marble": {"clean": "saved-operation"}}})
        )
    with pytest.raises(ValueError, match="Marble"):
        execute(f)
    assert not f.public.exists()


def test_existing_unreceipted_world_refused_and_imported_asset_tampering_detected(fixture):
    f = fixture
    f.public.mkdir()
    target = f.public / "marble-fixture-clean.spz"
    target.write_bytes(f.spz.read_bytes())
    with pytest.raises(ValueError, match="without retained"):
        execute(f)
    target.unlink()
    execute(f)
    target.write_bytes(b"different world")
    with pytest.raises(ValueError, match="hash mismatch"):
        execute(f)
    assert target.read_bytes() == b"different world"


def test_prepared_import_recovers_same_pinned_asset_without_resubmission(fixture):
    f = fixture
    result = execute(f)
    retained = f.ctx / "local-world-import.json"
    result["status"] = "prepared"
    retained.write_text(json.dumps(result))
    Path(result["target"]).unlink()
    assert execute(f)["status"] == "complete"
    assert Path(result["target"]).read_bytes() == f.spz.read_bytes()


def test_tampered_retained_receipt_refused(fixture):
    f = fixture
    result = execute(f)
    result["generationLineage"] = "proven"
    (f.ctx / "local-world-import.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not match"):
        execute(f)


def test_runner_import_holds_failed_history_and_invalidates_checks_without_launching_other_stages(
    fixture,
):
    f = fixture
    p = object.__new__(run_clip.Pipeline)
    p.ctx, p.clip, p.name = f.ctx, f.clip, f.name
    p.a = SimpleNamespace(
        import_local_world=str(f.receipt),
        import_local_clean=None,
        clean_candidate=None,
        lhm_frozen_candidate=None,
        orchestrated=False,
        marble="video",
        skip_marble=False,
        only="marble_video",
        force=None,
        reuse_world=None,
        retained_scene=None,
        shot=None,
        all_shots=False,
        skip_finetune=True,
        quality_plan=None,
        quality_result=None,
        quality_evidence_root=None,
    )
    p.marble = "video"
    p.state = run_clip.State(f.ctx / "state.json")
    p.state.record("clean", status="failed", error="retained failed clean")
    p.state.record("review", status="blocked", accepted=False)
    p.state.record("_paid_clean", status="failed", attempt="retained-claim")
    p.state.record("scale_fit", status="ok")
    p.state.record("verify", status="ok")
    p.state.record("_verify", status="passed", passed=True)
    p.stages = [
        "clean",
        "world_prompt",
        "review",
        "marble_video",
        "package",
        "scale_fit",
        "place_fit",
        "anchors",
        "verify",
    ]
    p.deps = dict(
        clean=[],
        world_prompt=[],
        review=["clean", "world_prompt"],
        marble_video=["clean", "review"],
        package=[],
        scale_fit=["package", "marble_video"],
        place_fit=["scale_fit"],
        anchors=["place_fit"],
        verify=["anchors"],
    )
    p.summary = Mock()
    p.paid_run = Mock(side_effect=AssertionError("no paid call"))
    p.clean = Mock(side_effect=AssertionError("no clean"))
    p.world_prompt = Mock(side_effect=AssertionError("no OpenAI"))
    p.marble_world = Mock(side_effect=AssertionError("no provider submission"))
    ledger = f.root / "ledger.json"
    ledger.write_text('{"failed history":1}')
    assert p.go() == 0
    for call in [p.paid_run, p.clean, p.world_prompt, p.marble_world]:
        call.assert_not_called()
    assert p.state.data["stages"]["clean"]["status"] == "failed"
    assert p.state.data["stages"]["review"]["accepted"] is False
    assert p.state.data["stages"]["_paid_clean"]["attempt"] == "retained-claim"
    assert p.state.data["stages"]["scale_fit"]["status"] == "pending"
    assert p.state.data["stages"]["verify"]["status"] == "pending"
    assert p.state.data["stages"]["_verify"]["passed"] is False
    stage = p.state.data["stages"]["marble_video"]
    assert stage["execution"] == "imported_local"
    assert stage["freshGeneration"] is False
    assert stage["freshInference"] is False
    assert json.loads(p.state.path.read_text())["stages"]["marble_video"]["freshInference"] is False
    assert stage["generationLineage"] == "unknown"
    assert ledger.read_text() == '{"failed history":1}'
    p.go()  # An explicit reimport checks the retained import instead of a cached no-op.
    assert p.state.data["stages"]["marble_video"]["execution"] == "imported_local"


def test_flags_do_not_enable_import_via_normal_generation(fixture):
    f = fixture
    args = SimpleNamespace(
        import_local_world=str(f.receipt),
        orchestrated=False,
        marble="video",
        skip_marble=False,
        only="marble_video",
        reuse_world=None,
        retained_scene=None,
        import_local_clean=None,
        shot=None,
        all_shots=False,
        force=None,
    )
    run_clip.validate_local_world_import(args)
    for field, value in [
        ("only", None),
        ("only", "marble_video,clean"),
        ("marble", "image"),
        ("orchestrated", True),
        ("reuse_world", "world-id"),
        ("shot", 0),
        ("force", "clean"),
    ]:
        changed = copy.copy(args)
        setattr(changed, field, value)
        with pytest.raises(ValueError):
            run_clip.validate_local_world_import(changed)


def test_cli_import_skips_cut_models_and_automatic_archiving(fixture, monkeypatch):
    f = fixture
    monkeypatch.setenv("WANDER_RUNS_ROOT", str(f.root / "runs"))
    factory = Mock(return_value=SimpleNamespace(go=lambda: 0))
    shot_selection = Mock(side_effect=AssertionError("import must not score/trim source"))
    external = Mock(side_effect=AssertionError("import must not archive or call providers"))
    monkeypatch.setattr(run_clip, "Pipeline", factory)
    monkeypatch.setattr(run_clip, "resolve_shots", shot_selection)
    monkeypatch.setattr(run_clip.subprocess, "run", external)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_clip.py",
            "--clip",
            str(f.clip),
            "--name",
            "fixture",
            "--marble",
            "video",
            "--only",
            "marble_video",
            "--import-local-world",
            str(f.receipt),
        ],
    )
    with pytest.raises(SystemExit) as result:
        run_clip.main()
    assert result.value.code == 0
    assert factory.call_args.args[0].no_publish is True
    shot_selection.assert_not_called()
    external.assert_not_called()
    assert not (f.root / "runs").exists()  # Fake pipeline does not execute the import.


def test_cli_invalid_receipt_stops_before_pipeline_state_or_archive(fixture, monkeypatch):
    f = fixture
    f.document["sourceSha256"] = "0" * 64
    f.receipt.write_text(json.dumps(f.document))
    monkeypatch.setenv("WANDER_RUNS_ROOT", str(f.root / "runs"))
    factory = Mock(side_effect=AssertionError("no constructor/state mutation"))
    monkeypatch.setattr(run_clip, "Pipeline", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_clip.py",
            "--clip",
            str(f.clip),
            "--name",
            "fixture",
            "--marble",
            "video",
            "--only",
            "marble_video",
            "--import-local-world",
            str(f.receipt),
        ],
    )
    with pytest.raises(SystemExit) as result:
        run_clip.main()
    assert result.value.code == 2
    factory.assert_not_called()
    assert not (f.root / "runs").exists()


def test_retained_import_blocks_later_generation_even_after_stage_invalidation(fixture):
    f = fixture
    execute(f)
    p = object.__new__(run_clip.Pipeline)
    p.ctx = f.ctx
    p.marble_saved_operation = Mock(
        side_effect=AssertionError("must not enter provider recovery/submission")
    )
    with pytest.raises(ValueError, match="blocks new generation"):
        p.marble_submit_locked("video", f.clip, "clean")
    p.marble_saved_operation.assert_not_called()


def test_spz_admission_stays_strict_under_python_optimization(fixture):
    f = fixture
    # A full-size payload with the wrong magic must fail even when shared-reader
    # assert statements are disabled by Python -O.
    f.spz.write_bytes(gzip.compress(struct.pack("<IIIBBBB", 0, 2, 2, 0, 12, 0, 0) + bytes(38)))
    f.document["world"].update(bytes=f.spz.stat().st_size, sha256=world.digest(f.spz))
    f.evidence.write_text(json.dumps([world.digest(f.clip), world.digest(f.spz)]))
    f.document["associationEvidence"][0]["sha256"] = world.digest(f.evidence)
    f.receipt.write_text(json.dumps(f.document))
    script = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from local_world_import import validate_receipt;validate_receipt(sys.argv[2],sys.argv[3])"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            script,
            str(Path(run_clip.__file__).parent),
            str(f.receipt),
            str(f.clip),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SPZ structure is invalid" in result.stderr
    assert not f.public.exists()
