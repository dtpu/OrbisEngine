"""No-spend retained cleaning recovery, selection integrity and downstream contracts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clean_candidates
import run_clip
from stage_attempts import StageAttempts


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    p = object.__new__(run_clip.Pipeline)
    p.ctx = tmp_path / "fixture"
    p.ctx.mkdir()
    p.name = "fixture"
    p.clip = tmp_path / "source.mp4"
    p.clip.write_bytes(b"original source")
    p.info = {"width": 11, "height": 4}
    p.multi = False
    p.marble = "video"
    p.clean_mp4 = tmp_path / "original-clean.mp4"
    p.clean_report = p.ctx / "clean.json"
    p.clean_masks_path = p.ctx / "masks.npz"
    p.first_png = p.ctx / "first/f_0000.png"
    p.selected_clean_candidate = None
    p.a = SimpleNamespace(
        clean_candidate="instance-masks-v1",
        clean_masks=None,
        moved_mask=False,
        fps=12,
        dilate=20,
        bottom_extra=40,
        lama_px=960,
        only="clean",
        force="clean",
        skip_finetune=True,
        reuse_world=None,
        gate_pass=True,
        stage_hypothesis="Use corrected instance masks",
        stage_ledger=str(tmp_path / "ledger.json"),
    )
    p.state = run_clip.State(p.ctx / "state.json")
    p.state.record("clean", status="ok")
    p.originals = {}
    for path in (
        p.clean_mp4,
        p.clean_report,
        p.clean_masks_path,
        p.ctx / "clean-f0.png",
        p.ctx / "clean.log",
    ):
        path.write_bytes(b"retained original " + path.name.encode())
        p.originals[path] = path.read_bytes()
    monkeypatch.setattr(run_clip, "MARBLE_DIR", tmp_path / "marble")
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(tmp_path / "public"))
    p.paid_run = Mock(
        side_effect=lambda operation, command, log: write_clean_outputs(p, command, log)
    )
    return p


def write_clean_outputs(p, command, log):
    outputs = {
        flag: Path(command[command.index(flag) + 1])
        for flag in ("--out", "--frame0", "--masks", "--report")
    }
    outputs["--out"].write_bytes(b"corrected clean video")
    outputs["--frame0"].write_bytes(b"clean first frame")
    outputs["--masks"].write_bytes(b"packed clean masks")
    outputs["--report"].write_text(
        json.dumps(
            {
                "sourceSha256": run_clip.file_sha256(p.clip),
                "outputVideoSha256": run_clip.file_sha256(outputs["--out"]),
                "error": None,
            }
        )
    )
    Path(log).write_text("completed provider output\n")


def assert_originals(p):
    for path, content in p.originals.items():
        assert path.read_bytes() == content


def test_success_selects_fresh_outputs_and_preserves_originals(pipeline):
    p = pipeline
    p.clean()
    p.paid_run.assert_called_once()
    directory = p.ctx / "clean-candidates/instance-masks-v1"
    assert p.clean_mp4 == directory / "clean.mp4"
    assert p.clean_report == directory / "clean.json"
    assert p.clean_masks_path == directory / "masks.npz"
    receipt = json.loads((p.ctx / "clean-selection.json").read_text())
    assert receipt == json.loads((directory / "selection.json").read_text())
    assert receipt["sourceSha256"] == run_clip.file_sha256(p.clip)
    assert set(receipt["files"]) == set(clean_candidates.OUTPUTS)
    assert_originals(p)
    # A fresh runner object loads the successful selection without needing the option again.
    resumed = object.__new__(run_clip.Pipeline)
    resumed.ctx, resumed.clip, resumed.name = p.ctx, p.clip, p.name
    resumed.load_clean_selection()
    assert resumed.clean_mp4 == p.clean_mp4


def test_failed_candidate_keeps_previous_selection_and_all_artifacts(pipeline):
    p = pipeline
    p.clean()
    old_receipt = (p.ctx / "clean-selection.json").read_bytes()
    old_video = p.clean_mp4
    p.a.clean_candidate = "second-candidate"
    p.paid_run.side_effect = RuntimeError("provider failed after launch")
    with pytest.raises(RuntimeError, match="provider failed"):
        p.clean()
    assert p.clean_mp4 == old_video
    assert (p.ctx / "clean-selection.json").read_bytes() == old_receipt
    assert (p.ctx / "clean-candidates/second-candidate").is_dir()
    assert_originals(p)


@pytest.mark.parametrize("collision", ["directory", "file", "symlink", "parent-symlink"])
def test_destination_collision_stops_before_paid_call(pipeline, collision, tmp_path):
    p = pipeline
    directory = p.ctx / "clean-candidates/instance-masks-v1"
    if collision == "parent-symlink":
        (p.ctx / "clean-candidates").symlink_to(tmp_path, target_is_directory=True)
    else:
        directory.parent.mkdir()
        if collision == "directory":
            directory.mkdir()
        elif collision == "file":
            directory.write_bytes(b"existing file")
        else:
            directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        p.clean()
    p.paid_run.assert_not_called()
    assert_originals(p)


@pytest.mark.parametrize(
    "tamper", ["video", "source", "receipt", "path", "missing", "file-symlink", "candidate-symlink"]
)
def test_selected_hashes_and_path_scope_fail_closed(pipeline, tamper, tmp_path):
    p = pipeline
    p.clean()
    receipt = p.ctx / "clean-selection.json"
    directory = p.clean_mp4.parent
    if tamper == "video":
        p.clean_mp4.write_bytes(b"changed")
    elif tamper == "source":
        p.clip.write_bytes(b"different source")
    elif tamper == "receipt":
        receipt.write_text("{broken")
    elif tamper == "path":
        doc = json.loads(receipt.read_text())
        doc["files"]["video"]["path"] = "../outside.mp4"
        receipt.write_text(json.dumps(doc))
        (directory / "selection.json").write_text(json.dumps(doc))
    elif tamper == "missing":
        receipt.unlink()
    elif tamper == "file-symlink":
        external = tmp_path / "outside.mp4"
        external.write_bytes(p.clean_mp4.read_bytes())
        p.clean_mp4.unlink()
        p.clean_mp4.symlink_to(external)
    else:
        moved = tmp_path / "moved-candidate"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        p.load_clean_selection()


@pytest.mark.parametrize("evidence", ["generation", "world", "log", "state", "spz"])
def test_any_prior_world_submission_blocks_clean_replacement(pipeline, evidence):
    p = pipeline
    run_clip.MARBLE_DIR.mkdir()
    if evidence == "generation":
        (run_clip.MARBLE_DIR / "fixture-clean-generation.json").write_text('{"status":"preparing"}')
    elif evidence == "world":
        (run_clip.MARBLE_DIR / "fixture-image-world.json").write_text("{}")
    elif evidence == "log":
        (p.ctx / "marble_clean.log").write_text("outcome unknown")
    elif evidence == "state":
        p.state.record("_marble", clean="recorded-operation")
    else:
        run_clip.public_dir().mkdir()
        (run_clip.public_dir() / "marble-fixture-clean.spz").write_bytes(b"paid world")
    with pytest.raises(ValueError, match="Marble submission"):
        p.clean()
    p.paid_run.assert_not_called()
    assert not (p.ctx / "clean-candidates").exists()
    assert_originals(p)


def test_changed_candidate_requires_force_before_cached_clean_can_be_skipped(pipeline):
    p = pipeline
    p.a.force = None
    with pytest.raises(ValueError, match="--force clean"):
        p.go()
    p.paid_run.assert_not_called()


def test_retained_candidate_does_not_reset_paid_allowance(pipeline, monkeypatch):
    p = pipeline
    ledger = StageAttempts(p.a.stage_ledger)
    for index in range(3):
        attempt = ledger.begin(
            run_clip.file_sha256(p.clip),
            "clean",
            parameters={"version": index},
            code_version={"worker": "test"},
            hypothesis=f"test {index}",
            candidate=p.name,
            log=p.ctx / "old.log",
            results=[],
        )
        ledger.finish(attempt["id"], status="failed", evidence=[])
    p.paid_run = run_clip.Pipeline.paid_run.__get__(p)
    provider = Mock()
    monkeypatch.setattr(run_clip, "run", provider)
    with pytest.raises(run_clip.QualityStop, match="three total"):
        p.clean()
    provider.assert_not_called()
    assert len(json.loads(ledger.path.read_text())["attempts"]) == 3
    assert not (p.ctx / "clean-selection.json").exists()
    assert_originals(p)


def test_review_and_marble_use_selected_success(pipeline, monkeypatch):
    p = pipeline
    p.clean()
    video = p.clean_mp4
    monkeypatch.setattr(run_clip, "probe", lambda _: {"frames": 2, "duration": 1, "fps": 2})
    commands = []

    def extract(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"review frame")

    monkeypatch.setattr(run_clip.subprocess, "run", extract)
    p.review()
    assert all(command[command.index("-i") + 1] == str(video) for command in commands)
    p.marble_world = Mock()
    p.marble_video()
    assert p.marble_world.call_args.args[1] == video


def test_selection_lock_prevents_concurrent_submission_or_candidate(pipeline):
    p = pipeline
    with clean_candidates.selection_lock(p.ctx), pytest.raises(ValueError, match="owns this run"):
        p.clean()
    p.paid_run.assert_not_called()


@pytest.mark.parametrize("slug", ["../escape", "a/b", ".hidden", "", "has space", "x" * 65])
def test_candidate_slug_is_bounded(slug):
    with pytest.raises(ValueError):
        clean_candidates.validate_slug(slug)


@pytest.mark.parametrize(
    "changes",
    [
        {"marble": "multi"},
        {"marble": "image"},
        {"marble": "both"},
        {"orchestrated": True},
        {"reuse_world": "world"},
        {"only": "review"},
    ],
)
def test_incompatible_candidate_options_rejected(changes):
    args = SimpleNamespace(
        clean_candidate="new",
        orchestrated=False,
        skip_marble=False,
        marble="video",
        only="clean",
        reuse_world=None,
        retained_scene=None,
    )
    for key, value in changes.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        run_clip.validate_clean_candidate(args)


@pytest.mark.parametrize("recovery", ["operation", "world", "explicit-world"])
def test_real_constructor_allows_only_existing_marble_recovery_with_bad_clean_selection(
    pipeline, monkeypatch, recovery
):
    p = pipeline
    p.clean()
    p.clean_mp4.write_bytes(b"damaged old clean")
    run_clip.MARBLE_DIR.mkdir()
    if recovery == "operation":
        (run_clip.MARBLE_DIR / "fixture-clean-generation.json").write_text(
            '{"operation_id":"existing-op"}'
        )
    elif recovery == "world":
        (run_clip.MARBLE_DIR / "fixture-clean-world.json").write_text(
            '{"world_id":"existing-world"}'
        )
    # Construct the actual runner against the retained directory. Model/provider interfaces
    # are never used; only the local video probe and certificate command are substituted.
    monkeypatch.setattr(run_clip, "runs_dir", lambda: p.ctx.parent)
    monkeypatch.setattr(run_clip, "probe", lambda _: p.info)
    monkeypatch.setattr(
        run_clip.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="/local/certificates\n")
    )
    args = SimpleNamespace(
        clip=str(p.clip),
        name=p.ctx.name,
        fps=12,
        marble_key="UNUSED_TEST_KEY",
        people=0,
        all_people=False,
        skip_marble=False,
        marble="video",
        object=[],
        no_objects=True,
        skip_finetune=True,
        only="marble_video",
        reuse_world="existing-world" if recovery == "explicit-world" else None,
    )
    constructed = run_clip.Pipeline(args, clip=p.clip, name=p.name)
    assert constructed.selected_clean_candidate is None
    if recovery == "operation":
        constructed.marble_poll = Mock()
        constructed.key = "synthetic-unused"
        constructed.marble_video()
        constructed.marble_poll.assert_called_once_with("video", "existing-op", "clean")


@pytest.mark.parametrize("only", ["marble_video", "review,marble_video"])
def test_real_constructor_still_validates_selection_for_new_or_mixed_submission(
    pipeline, monkeypatch, only
):
    p = pipeline
    p.clean()
    p.clean_mp4.write_bytes(b"damaged old clean")
    monkeypatch.setattr(run_clip, "runs_dir", lambda: p.ctx.parent)
    monkeypatch.setattr(run_clip, "probe", lambda _: p.info)
    args = SimpleNamespace(clip=str(p.clip), name=p.name, only=only, reuse_world=None)
    if only != "marble_video":
        run_clip.MARBLE_DIR.mkdir()
        (run_clip.MARBLE_DIR / "fixture-clean-generation.json").write_text(
            '{"operation_id":"existing-op"}'
        )
    with pytest.raises(ValueError, match="changed"):
        run_clip.Pipeline(args, clip=p.clip, name=p.name)
