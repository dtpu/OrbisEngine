"""Local entrypoints must bound waits without changing inference input arguments."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker import modal_lhm, modal_multiperson
from worker.stages import lhm_recovery, modal_profile


@pytest.mark.parametrize(
    "mode,expected", [("frozen", 1200), ("motion", 1200), ("short_motion", 900)]
)
def test_entrypoint_records_owner_and_effective_deadline(tmp_path, monkeypatch, mode, expected):
    for name in (
        "source.png",
        "mask.png",
        "prepared.json",
        "source.mp4",
        "canonical-state.pt",
        "result.json",
        "cameras.json",
        "frame_000.ply",
        "source-poses.pt",
        "motion.json",
    ):
        (tmp_path / name).write_bytes(name.encode())
    submit = Mock(side_effect=RuntimeError("stop before provider"))
    monkeypatch.setattr(lhm_recovery, "submit_once", submit)
    monkeypatch.setattr(modal_profile, "effective_modal_profile", lambda: "owner-profile")
    module = modal_lhm if mode == "frozen" else modal_multiperson
    monkeypatch.setattr(module, "app", SimpleNamespace(app_id="ap-owned"))
    monkeypatch.setattr(module, "INFERENCE_RESOURCES", {"timeout": 600, "startup_timeout": 600})
    with pytest.raises(RuntimeError, match="stop before provider"):
        if mode == "frozen":
            module.main(prepared=str(tmp_path), out=str(tmp_path / "output"))
        else:
            module.animate(
                video=str(tmp_path / "source.mp4"),
                canonical=str(tmp_path),
                cameras=str(tmp_path / "cameras.json"),
                track_dir=str(tmp_path),
                out=str(tmp_path / "output"),
                execution_timeout=300 if mode == "short_motion" else 0,
            )
    kwargs = submit.call_args.kwargs
    assert kwargs["result_timeout_seconds"] == expected
    assert kwargs["app_id"] == "ap-owned"
    assert kwargs["modal_profile"] == "owner-profile"
    if mode != "frozen":
        assert kwargs["execution_timeout"] == (300 if mode == "short_motion" else 0)
