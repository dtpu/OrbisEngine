"""An all-person run must not succeed by discarding identity failures."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_clip


@pytest.mark.parametrize(
    "all_people,code,failed,reason,refused",
    [
        (True, 3, [0, 2, 3, 4], None, True),
        (False, 3, [0, 2, 3, 4], None, False),
        (True, 0, [], None, False),
        (False, 0, [], None, False),
        (True, 3, [0, 1, 2, 3, 4], None, True),
        (False, 3, [0, 1, 2, 3, 4], None, True),
        (True, 3, [], "swap control missed", True),
        (False, 3, [], "swap control missed", True),
        (True, 4, [], "insufficient evidence", True),
    ],
)
def test_identity_failure_cannot_remove_people(
    tmp_path, monkeypatch, all_people, code, failed, reason, refused
):
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.a = SimpleNamespace(all_people=all_people)
    pipeline.ctx = tmp_path
    pipeline.clip = tmp_path / "source.mp4"
    evidence = json.dumps({"tracks": list(range(5)), "failedTracks": failed, "reason": reason})
    receipt = tmp_path / "identity.json"
    receipt.write_text(evidence)
    audit = Mock(return_value=SimpleNamespace(returncode=code, stdout="audit result"))
    monkeypatch.setattr(run_clip.subprocess, "run", audit)
    if refused:
        with pytest.raises(RuntimeError, match="identity audit"):
            pipeline.identity_guard()
    else:
        pipeline.identity_guard()
    assert receipt.read_text() == evidence
    audit.assert_called_once()
    assert "--swap-test" in audit.call_args.args[0]


@pytest.mark.parametrize("index", [0, 1])
def test_retained_failed_audit_blocks_all_people_before_avatar_work(tmp_path, index):
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.a = SimpleNamespace(all_people=True)
    pipeline.ctx = tmp_path
    pipeline.track_doc = lambda: {"trackCount": 2, "tracks": [{"track": 0}, {"track": 1}]}
    (tmp_path / "identity.json").write_text(json.dumps({"ok": False, "failedTracks": [0]}))
    with pytest.raises(RuntimeError, match="retained identity audit failed"):
        pipeline.track_or_skip(index)


def test_explicit_subset_keeps_existing_skip_behavior(tmp_path):
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.a = SimpleNamespace(all_people=False)
    pipeline.ctx = tmp_path
    pipeline.track_doc = lambda: {"trackCount": 2, "tracks": [{"track": 0}, {"track": 1}]}
    (tmp_path / "identity.json").write_text(json.dumps({"ok": False, "failedTracks": [0]}))
    with pytest.raises(run_clip.SkipStage):
        pipeline.track_or_skip(0)
    assert pipeline.track_or_skip(1) == {"track": 1}
