"""Offline candidate/receipt contracts, not model or visual-quality tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import lhm_frozen_candidates as candidates
from worker.stages import lhm_recovery as recovery


def write(path, value):
    path.write_text(json.dumps(value))


def outputs(dest, prepared):
    receipt_path, receipt = recovery.new_receipt(dest, "frozen")
    for name, data in {
        "person-posed.ply": b"ply generated",
        "canonical-state.pt": b"generated state",
        "inference.log": b"generated",
        "modal-run.json": b'{"error":null}',
        "result.json": json.dumps(
            {
                "prepared": json.loads((prepared / "prepared.json").read_text()),
                "sourceImageSha256": recovery.file_hash(prepared / "source.png"),
            }
        ).encode(),
    }.items():
        (dest / name).write_bytes(data)
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": recovery.file_hash(p)}
        for p in dest.iterdir()
        if p.name != recovery.RECEIPT_NAME
    }
    manifest = {
        "schema": recovery.SCHEMA,
        "recoveryId": receipt["recoveryId"],
        "mode": "frozen",
        "status": "complete",
        "finalized": True,
        "files": files,
        "problems": [],
        "inferenceError": False,
        "requireRegistration": False,
    }
    write(dest / "recovery-manifest.json", manifest)
    receipt.update(
        status="recovered",
        resultStatus="complete",
        recoveredDestination=str(dest.resolve()),
        manifestSha256=recovery.file_hash(dest / "recovery-manifest.json"),
    )
    write(receipt_path, receipt)


@pytest.fixture
def pipeline(tmp_path):
    ctx = tmp_path / "run"
    ctx.mkdir()
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"original source")
    prepared = ctx / "prepared-person"
    prepared.mkdir()
    for name in ["source.png", "mask.png"]:
        (prepared / name).write_bytes(name.encode())
    write(prepared / "prepared.json", {"sourceSha256": recovery.file_hash(clip), "frameIndex": 0})
    p = SimpleNamespace(
        ctx=ctx,
        clip=clip,
        a=SimpleNamespace(
            lhm_frozen_candidate="first",
            only="lhm_frozen",
            force="",
            orchestrated=False,
            people=0,
            all_people=False,
        ),
        state=SimpleNamespace(data={"stages": {}}, done=lambda _: False),
    )
    p.paid_run = Mock(side_effect=lambda operation, cmd, log: outputs(Path(cmd[-1]), prepared))
    p.recover_lhm = Mock(return_value=True)
    return p


def test_success_selection_and_no_flag_resume(pipeline):
    p = pipeline
    assert candidates.run_candidate(p)
    output = p.ctx / "lhm-frozen-candidates/first/output"
    assert candidates.selected_directory(p.ctx, p.clip) == output
    assert p.paid_run.call_args.args[0] == "lhm_frozen"
    assert (output.parent / "submission-started.json").is_file()
    p.a.lhm_frozen_candidate = None
    assert candidates.run_candidate(p)
    assert p.paid_run.call_count == 1
    p.recover_lhm.assert_not_called()


def test_failed_candidate_retained_never_resubmitted(pipeline):
    p = pipeline
    p.paid_run.side_effect = RuntimeError("uncertain provider call")
    with pytest.raises(RuntimeError):
        candidates.run_candidate(p)
    root = p.ctx / "lhm-frozen-candidates/first"
    before = (root / "input-receipt.json").read_bytes()
    with pytest.raises(ValueError, match="cannot resubmit"):
        candidates.run_candidate(p)
    assert p.paid_run.call_count == 1
    assert (root / "input-receipt.json").read_bytes() == before
    assert not (p.ctx / candidates.SELECTION).exists()


def test_retained_output_is_recovery_only(pipeline):
    p = pipeline

    def fail(operation, cmd, log):
        Path(cmd[-1]).mkdir()
        raise RuntimeError("interrupted")

    p.paid_run.side_effect = fail
    with pytest.raises(RuntimeError):
        candidates.run_candidate(p)
    output = p.ctx / "lhm-frozen-candidates/first/output"

    def recover(dest, mode, log):
        assert dest == output and mode == "frozen"
        dest.rmdir()
        outputs(dest, p.ctx / "prepared-person")
        return True

    p.recover_lhm.side_effect = recover
    assert candidates.run_candidate(p)
    assert p.paid_run.call_count == 1
    p.recover_lhm.assert_called_once()


@pytest.mark.parametrize(
    "mutation",
    [
        "output",
        "prepared",
        "clip",
        "result",
        "manifest",
        "receipt",
        "partial",
        "symlink",
        "directory-symlink",
        "missing-selection",
        "selection",
    ],
)
def test_selected_evidence_tampering_fails_closed(pipeline, mutation):
    p = pipeline
    candidates.run_candidate(p)
    out = p.ctx / "lhm-frozen-candidates/first/output"
    if mutation == "output":
        (out / "person-posed.ply").write_bytes(b"changed")
    elif mutation == "prepared":
        (p.ctx / "prepared-person/mask.png").write_bytes(b"changed")
    elif mutation == "clip":
        p.clip.write_bytes(b"changed")
    elif mutation == "missing-selection":
        (p.ctx / candidates.SELECTION).unlink()
    elif mutation == "directory-symlink":
        actual = out.with_name("moved")
        out.rename(actual)
        out.symlink_to(actual, target_is_directory=True)
    elif mutation == "symlink":
        original = out / "person-posed.ply"
        original.unlink()
        original.symlink_to(p.clip)
    else:
        path = {
            "result": out / "result.json",
            "manifest": out / "recovery-manifest.json",
            "receipt": out / "recovery-receipt.json",
            "partial": out / "recovery-manifest.json",
            "selection": p.ctx / candidates.SELECTION,
        }[mutation]
        value = json.loads(path.read_text())
        if mutation == "partial":
            value["status"] = "partial"
        elif mutation == "result":
            value["sourceImageSha256"] = "0" * 64
        else:
            value["tampered"] = True
        write(path, value)
    with pytest.raises(ValueError):
        candidates.selected_directory(p.ctx, p.clip)
    assert p.paid_run.call_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lhm_frozen_candidate": "../escape"},
        {"orchestrated": True},
        {"people": 1},
        {"all_people": True},
        {"only": "lhm_frozen,lhm_motion"},
    ],
)
def test_option_scope(pipeline, kwargs):
    p = pipeline
    for key, value in kwargs.items():
        setattr(p.a, key, value)
    with pytest.raises(ValueError):
        candidates.validate_options(p.a)
    p.paid_run.assert_not_called()


def test_cached_force_guard(pipeline):
    p = pipeline
    p.state.done = lambda _: True
    with pytest.raises(ValueError, match="--force lhm_frozen"):
        candidates.validate_change(p)
    p.a.force = "lhm_frozen"
    candidates.validate_change(p)


@pytest.mark.parametrize("blocker", ["motion", "motion-symlink", "running-motion", "ok-package"])
def test_retained_motion_blocks_changing_selection(pipeline, blocker):
    p = pipeline
    if blocker == "motion":
        (p.ctx / "lhm-motion").mkdir()
    elif blocker == "motion-symlink":
        (p.ctx / "lhm-motion").symlink_to(p.ctx / "missing")
    else:
        name, status = (
            ("lhm_motion", "running") if blocker == "running-motion" else ("package", "ok")
        )
        p.state.data["stages"][name] = {"status": status}
    with pytest.raises(ValueError, match="cannot change"):
        candidates.run_candidate(p)
    p.paid_run.assert_not_called()


def test_failed_retry_preserves_selected_receipts(pipeline):
    p = pipeline
    candidates.run_candidate(p)
    before = (p.ctx / candidates.SELECTION).read_bytes()
    history = p.ctx / "lhm-frozen-candidates/first/selection.json"
    original = history.read_bytes()
    p.a.lhm_frozen_candidate = "second"
    p.paid_run.side_effect = RuntimeError("failed retry")
    with pytest.raises(RuntimeError):
        candidates.run_candidate(p)
    assert (p.ctx / candidates.SELECTION).read_bytes() == before
    assert history.read_bytes() == original


def test_legacy_unchanged_and_dangling_selection_rejected(pipeline):
    p = pipeline
    p.a.lhm_frozen_candidate = None
    assert candidates.run_candidate(p) is False
    assert candidates.selected_directory(p.ctx, p.clip) == p.ctx / "lhm-frozen"
    (p.ctx / candidates.SELECTION).symlink_to(p.ctx / "missing")
    with pytest.raises(ValueError, match="symlink"):
        candidates.run_candidate(p)


@pytest.mark.parametrize(
    "wrong", ["prepared", "source-image", "missing-required", "partial", "output-symlink"]
)
def test_invalid_new_outputs_never_selected_even_with_matching_manifest(pipeline, wrong):
    p = pipeline

    def paid(operation, cmd, log):
        out = Path(cmd[-1])
        outputs(out, p.ctx / "prepared-person")
        result = json.loads((out / "result.json").read_text())
        if wrong == "prepared":
            result["prepared"]["frameIndex"] = 143
        elif wrong == "source-image":
            result["sourceImageSha256"] = "0" * 64
        write(out / "result.json", result)
        manifest = json.loads((out / "recovery-manifest.json").read_text())
        manifest["files"]["result.json"] = candidates._pin(out / "result.json")
        if wrong == "missing-required":
            del manifest["files"]["canonical-state.pt"]
            (out / "canonical-state.pt").unlink()
        if wrong == "partial":
            manifest["status"] = "partial"
        if wrong == "output-symlink":
            (out / "extra-symlink").symlink_to(p.clip)
        write(out / "recovery-manifest.json", manifest)
        receipt = json.loads((out / "recovery-receipt.json").read_text())
        receipt["manifestSha256"] = recovery.file_hash(out / "recovery-manifest.json")
        write(out / "recovery-receipt.json", receipt)

    p.paid_run.side_effect = paid
    with pytest.raises(ValueError):
        candidates.run_candidate(p)
    assert not (p.ctx / candidates.SELECTION).exists()
    assert (p.ctx / "lhm-frozen-candidates/first/output/recovery-receipt.json").is_file()


def test_inputs_invalid_before_paid_launch(pipeline):
    p = pipeline
    source = p.ctx / "prepared-person/source.png"
    source.unlink()
    source.symlink_to(p.clip)
    with pytest.raises(ValueError, match="symlink"):
        candidates.run_candidate(p)
    p.paid_run.assert_not_called()


def test_already_selected_reuses_with_retained_motion(pipeline):
    p = pipeline
    candidates.run_candidate(p)
    (p.ctx / "lhm-motion").mkdir()
    assert candidates.run_candidate(p)
    assert p.paid_run.call_count == 1


def test_selection_lock_blocks_paid_and_leaves_no_candidate(pipeline):
    p = pipeline
    with candidates.selection_lock(p.ctx):
        with pytest.raises(ValueError, match="busy"):
            candidates.run_candidate(p)
    p.paid_run.assert_not_called()
    assert not (p.ctx / "lhm-frozen-candidates").exists()


def test_fresh_persisted_motion_status_blocks_stale_pipeline(pipeline):
    p = pipeline
    p.state.path = p.ctx / "state.json"
    write(p.state.path, {"stages": {"lhm_motion": {"status": "running"}}})
    with pytest.raises(ValueError, match="cannot change"):
        candidates.run_candidate(p)
    p.paid_run.assert_not_called()


def test_runner_single_person_hooks_use_selection(pipeline):
    import run_clip

    p = pipeline
    candidates.run_candidate(p)
    p.a.lhm_frozen_candidate = None
    run_clip.Pipeline.lhm_frozen(p)
    assert p.paid_run.call_count == 1
    depth = p.ctx / "pi3x"
    depth.mkdir()
    (depth / "frame_000.ply").write_bytes(b"retained reference")
    p.recover_lhm.return_value = False
    run_clip.Pipeline.lhm_motion(p)
    operation, command, _ = p.paid_run.call_args.args
    assert operation == "lhm_motion"
    assert (
        Path(command[command.index("--canonical") + 1])
        == p.ctx / "lhm-frozen-candidates/first/output"
    )


def test_guard_precedes_scheduler_state_mutation(pipeline):
    import run_clip

    p = pipeline
    p.state.done = lambda _: True
    with pytest.raises(ValueError, match="--force lhm_frozen"):
        run_clip.Pipeline.go(p)
    assert p.state.data == {"stages": {}}
    p.paid_run.assert_not_called()


def fake_completed_volume(output):
    manifest = (output / "recovery-manifest.json").read_bytes()
    document = json.loads(manifest)
    payloads = {name: (output / name).read_bytes() for name in document["files"]}

    def read_file(name):
        if name.endswith("/manifest.json"):
            return [manifest]
        relative = name.split("/output/", 1)[1]
        return [payloads[relative]]

    return SimpleNamespace(read_file=read_file)


@pytest.mark.parametrize("repair_missing_artifact", [False, True])
def test_selected_candidate_survives_actual_fake_volume_recovery(pipeline, repair_missing_artifact):
    p = pipeline
    candidates.run_candidate(p)
    output = candidates.selected_directory(p.ctx, p.clip)
    selection_before = (p.ctx / candidates.SELECTION).read_bytes()
    history_before = (output.parent / "selection.json").read_bytes()
    volume = fake_completed_volume(output)
    if repair_missing_artifact:
        (output / "person-posed.ply").unlink()
    for _ in range(2):
        recovery.recover_outputs(output / "recovery-receipt.json", volume)
        assert candidates.selected_directory(p.ctx, p.clip) == output
    assert (p.ctx / candidates.SELECTION).read_bytes() == selection_before
    assert (output.parent / "selection.json").read_bytes() == history_before
    assert p.paid_run.call_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("functionCallId", "changed-call-id"),
        ("recoveryId", "0" * 32),
        ("destination", "/changed-destination"),
        ("recoveredDestination", "/changed-recovered-destination"),
        ("manifestSha256", "0" * 64),
        ("status", "waiting"),
        ("resultStatus", "failed"),
    ],
)
def test_recovery_identity_stays_pinned_after_recovery(pipeline, field, value):
    p = pipeline
    candidates.run_candidate(p)
    output = candidates.selected_directory(p.ctx, p.clip)
    recovery.recover_outputs(output / "recovery-receipt.json", fake_completed_volume(output))
    receipt_path = output / "recovery-receipt.json"
    document = json.loads(receipt_path.read_text())
    document[field] = value
    write(receipt_path, document)
    with pytest.raises(ValueError):
        candidates.selected_directory(p.ctx, p.clip)
    assert p.paid_run.call_count == 1


def legacy_selection(p):
    candidates.run_candidate(p)
    output = candidates.selected_directory(p.ctx, p.clip)
    path = p.ctx / candidates.SELECTION
    document = json.loads(path.read_text())
    document["files"]["recovery-receipt.json"] = candidates._pin(output / "recovery-receipt.json")
    write(path, document)
    write(output.parent / "selection.json", document)
    return output


def test_healthy_legacy_selection_keeps_history_and_retains_original_identity(pipeline):
    p = pipeline
    output = legacy_selection(p)
    selection_before = (p.ctx / candidates.SELECTION).read_bytes()
    history_before = (output.parent / "selection.json").read_bytes()
    receipt_before = (output / "recovery-receipt.json").read_bytes()
    assert candidates.selected_directory(p.ctx, p.clip) == output
    snapshot = output.parent / "selected-recovery-receipt.json"
    assert snapshot.read_bytes() == receipt_before
    recovery.recover_outputs(output / "recovery-receipt.json", fake_completed_volume(output))
    assert candidates.selected_directory(p.ctx, p.clip) == output
    assert snapshot.read_bytes() == receipt_before
    assert (p.ctx / candidates.SELECTION).read_bytes() == selection_before
    assert (output.parent / "selection.json").read_bytes() == history_before
    document = json.loads((output / "recovery-receipt.json").read_text())
    document["functionCallId"] = "changed-call-id"
    write(output / "recovery-receipt.json", document)
    with pytest.raises(ValueError, match="identity changed"):
        candidates.selected_directory(p.ctx, p.clip)


def test_changed_legacy_receipt_without_prior_identity_snapshot_fails_closed(pipeline):
    p = pipeline
    output = legacy_selection(p)
    recovery.recover_outputs(output / "recovery-receipt.json", fake_completed_volume(output))
    with pytest.raises(ValueError, match="cannot be established"):
        candidates.selected_directory(p.ctx, p.clip)
    assert not (output.parent / "selected-recovery-receipt.json").exists()


def test_legacy_identity_snapshot_cannot_be_tampered(pipeline):
    p = pipeline
    output = legacy_selection(p)
    candidates.selected_directory(p.ctx, p.clip)
    snapshot = output.parent / "selected-recovery-receipt.json"
    snapshot.write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        candidates.selected_directory(p.ctx, p.clip)
