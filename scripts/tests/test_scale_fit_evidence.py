"""Scale diagnostics stay with the run, including rejected measurements (no inference)."""

import json
import math
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
    p.state = run_clip.State(p.ctx / "state.json")
    p.a = SimpleNamespace(
        scale0=None,
        scale_probes=5,
        allow_scale_spread=False,
        only=None,
        force=None,
        skip_finetune=True,
        reuse_world=None,
    )
    p.stages = ["scale_fit"]
    p.deps = {"scale_fit": []}
    p.marble = "video"
    p.multi = False
    p.summary = Mock()
    p.placement = Mock(return_value={"floorMatchedScale": 0.8})
    p.base_world_spz = Mock(return_value=p.ctx / "world.spz")
    p.cameras = Mock(return_value=p.ctx / "cameras.json")
    p.person_ply = Mock(return_value=p.ctx / "person.ply")
    p.anchor_samples = Mock(return_value="0,1")
    share = p.ctx / "share"
    share.mkdir()
    monkeypatch.setattr(run_clip, "SHARE", share)
    return p


def diagnostic_worker(monkeypatch, rows_by_probe, *, figure=True):
    commands = []

    def invoke(command, log, **kwargs):
        rows = rows_by_probe[len(commands)]
        commands.append(command)
        log.write_text(
            "\n".join(f"(ratio {r:.3f}), p10 {lo:.3f} p90 {hi:.3f}" for r, lo, hi in rows)
        )
        if figure:
            output = Path(command[command.index("--share") + 1])
            tag = command[command.index("--tag") + 1]
            (output / f"{tag}-depth-ratio.png").write_bytes(b"diagnostic fixture")
        assert kwargs["attempts"] == 1

    monkeypatch.setattr(run_clip, "run", invoke)
    return commands


@pytest.mark.parametrize(
    "rows,median_ok,spread_ok",
    [
        ([(0.889, 0.088, 1.456), (1.125, 0.872, 1.115)], False, False),
        ([(0.94, 0.92, 1.0), (1.064, 1.0, 1.08)], False, True),
        ([(1.0, 0.8, 1.2), (1.0, 0.95, 1.05)], True, False),
        ([(1.0, 0.95, 1.05), (1.0, 0.98, 1.02)], True, True),
    ],
)
def test_fitted_measurements_and_figure_survive_quality_verdict(
    pipeline, monkeypatch, rows, median_ok, spread_ok
):
    p = pipeline
    # The first probe changes the scale; the second converges but may still fail the gates.
    commands = diagnostic_worker(monkeypatch, [[(0.9, 0.85, 1.0)], rows])
    accepted = median_ok and spread_ok
    assert p.go() == (0 if accepted else 1)
    assert len(commands) == 2
    stages = json.loads(p.state.path.read_text())["stages"]
    assert stages["scale_fit"]["status"] == ("ok" if accepted else "failed")
    report = json.loads((p.ctx / "scale-fit.json").read_text())
    assert report == stages["_scale"]
    assert report["scale0"] == pytest.approx(0.8 * 0.9)
    assert report["geomean"] == pytest.approx(math.sqrt(rows[0][0] * rows[1][0]))
    assert report["ratios"] == [row[0] for row in rows]
    assert report["p10"] == [row[1] for row in rows]
    assert report["p90"] == [row[2] for row in rows]
    assert report["gateOk"] is accepted
    assert report["medianOk"] is median_ok
    assert report["spreadOk"] is spread_ok
    figure = Path(report["figure"])
    assert figure.parent == p.ctx / "share"
    assert figure.is_file()
    assert len(list(figure.parent.glob("*-depth-ratio.png"))) == 2
    if not accepted:
        assert str(figure) in stages["scale_fit"]["error"]


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_figure_cannot_produce_a_dangling_evidence_reference(
    pipeline, monkeypatch, empty
):
    p = pipeline
    p.a.scale0 = 1.0
    diagnostic_worker(monkeypatch, [[(1.0, 0.98, 1.02)]], figure=False)
    if empty:
        (p.ctx / "share/fixture-diag-s10000-depth-ratio.png").touch()
    assert p.go() == 1
    stages = json.loads(p.state.path.read_text())["stages"]
    assert stages["scale_fit"]["status"] == "failed"
    assert "produced no figure" in stages["scale_fit"]["error"]
    assert "_scale" not in stages
    assert not (p.ctx / "scale-fit.json").exists()
    assert (p.ctx / "diag-given.log").is_file()
