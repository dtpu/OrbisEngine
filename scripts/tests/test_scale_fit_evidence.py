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
        scale_spread_unaccepted=False,
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


@pytest.mark.parametrize(
    "rows,kept",
    [
        ([(1.0, 0.8, 1.2), (1.0, 0.95, 1.05)], True),
        ([(0.94, 0.92, 1.0), (1.064, 1.0, 1.08)], False),
        ([(0.889, 0.088, 1.456), (1.125, 0.872, 1.115)], False),
    ],
)
def test_unaccepted_spread_keeps_the_fit_only_when_every_median_holds(
    pipeline, monkeypatch, rows, kept
):
    p = pipeline
    p.a.scale_spread_unaccepted = True
    diagnostic_worker(monkeypatch, [[(0.9, 0.85, 1.0)], rows])
    assert p.go() == (0 if kept else 1)
    stages = json.loads(p.state.path.read_text())["stages"]
    assert stages["scale_fit"]["status"] == ("ok" if kept else "failed")
    report = stages["_scale"]
    # The verdict is unchanged; only what it stops is.
    assert report["gateOk"] is False
    assert report["scale0"] == pytest.approx(0.8 * 0.9)
    if kept:
        assert report["medianOk"] is True and report["spreadOk"] is False
        assert "exceed the required spread" in report["gateError"]
        assert report["figure"] in report["gateError"]
    else:
        assert report["medianOk"] is False
        assert "are not all inside" in stages["scale_fit"]["error"]


def test_passing_gate_records_no_gate_error(pipeline, monkeypatch):
    p = pipeline
    p.a.scale0 = 1.0
    diagnostic_worker(monkeypatch, [[(1.0, 0.95, 1.05), (1.0, 0.98, 1.02)]])
    assert p.go() == 0
    report = json.loads((p.ctx / "scale-fit.json").read_text())
    assert report["gateOk"] is True and report["gateError"] is None


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


def test_distance_bands_are_reported_without_changing_the_gate(pipeline, monkeypatch):
    p = pipeline
    p.a.scale0 = 1.0
    bands = [
        {
            "frame": f,
            "bands": [
                {"band": 0, "p10": 0.98, "median": 1.0, "p90": 1.05},
                {"band": 3, "p10": 0.7, "median": 0.86, "p90": 1.0},
            ],
        }
        for f in (0, 1)
    ]

    def invoke(command, log, **kwargs):
        lines = ["(ratio 1.000), p10 0.840 p90 1.060"] * 2
        lines += ["depth-ratio-bands " + json.dumps(frame) for frame in bands]
        log.write_text("\n".join(lines))
        tag = command[command.index("--tag") + 1]
        (Path(command[command.index("--share") + 1]) / f"{tag}-depth-ratio.png").write_bytes(b"x")

    monkeypatch.setattr(run_clip, "run", invoke)
    assert p.go() == 1
    stages = json.loads(p.state.path.read_text())["stages"]
    report = json.loads((p.ctx / "scale-fit.json").read_text())
    assert report["distanceBands"] == bands
    assert report["spreadOk"] is False and report["gateOk"] is False
    assert (
        "nearest-quarter p10/median/p90 [(0.98, 1.0, 1.05), (0.98, 1.0, 1.05)]"
        in (stages["scale_fit"]["error"])
    )
    assert "farthest-quarter [(0.7, 0.86, 1.0), (0.7, 0.86, 1.0)]" in stages["scale_fit"]["error"]


def test_logs_without_complete_bands_keep_the_old_report():
    assert run_clip.distance_bands("(ratio 1.000), p10 0.9 p90 1.1", 1) is None
    one = 'depth-ratio-bands {"frame": 0, "bands": []}'
    assert run_clip.distance_bands(one, 2) is None
    assert run_clip.band_summary(None) == ""


def test_band_summary_keeps_gate_precision_and_flags_collapsed_quartiles():
    edge = {"band": 1, "p10": 0.8964, "median": 1.0, "p90": 1.1021}
    split = {"frame": 0, "bands": [edge, {"band": 3, "p10": 0.75, "median": 0.88, "p90": 1.0}]}
    collapsed = {"frame": 1, "bands": [{"band": 3, "p10": 0.9, "median": 1.0, "p90": 1.1}]}
    summary = run_clip.band_summary([split, collapsed])
    assert "(0.896, 1.0, 1.102)" in summary
    assert "(0.9, 1.0, 1.1, 'only one non-empty quartile')" in summary


def test_malformed_band_lines_are_evidence_loss_not_a_crash():
    assert run_clip.distance_bands('depth-ratio-bands {"frame": 0, "bands": [', 1) is None


def probe_sequence(pipeline, monkeypatch, rows_by_probe):
    """Different measured evidence per call; reuse a filename to exercise snapshot ownership."""
    calls, evidence = [], []
    figure = pipeline.ctx / "rounded-scale-depth-ratio.png"

    def diag(scale, probe):
        index = len(calls)
        rows = rows_by_probe[index]
        bands = [
            {"frame": frame, "bands": [{"band": 0, "p10": lo, "median": ratio, "p90": hi}]}
            for frame, (ratio, lo, hi) in enumerate(rows)
        ]
        pixels = f"measured probe {index}".encode()
        figure.write_bytes(pixels)
        calls.append((scale, probe))
        evidence.append((rows, bands, pixels))
        return rows, figure, bands

    monkeypatch.setattr(pipeline, "diag", diag)
    return calls, evidence, figure


@pytest.mark.parametrize("unaccepted", [False, True])
def test_retains_closest_passing_probe_with_consistent_evidence_and_failed_spread(
    pipeline, monkeypatch, unaccepted
):
    p = pipeline
    p.a.scale_spread_unaccepted = unaccepted
    rows = [
        [(1.4, 1.2, 1.6)],
        [(0.98, 0.94, 1.04), (0.98, 0.93, 1.05)],
        [(1.01, 0.81, 1.21), (1.01, 0.85, 1.16)],
        [(0.94, 0.92, 1.06), (1.06, 0.95, 1.08)],
    ]
    calls, evidence, overwritten = probe_sequence(p, monkeypatch, rows)
    assert p.go() == (0 if unaccepted else 1)
    assert len(calls) == 4  # No additional diagnostic or retry.
    report = json.loads((p.ctx / "scale-fit.json").read_text())
    assert report == p.state.data["stages"]["_scale"]
    assert report["selectedProbe"] == 2
    assert report["selectionReason"] == "earlier_median_pass_closest_log_geomean"
    assert report["scale0"] == pytest.approx(calls[2][0])
    assert report["scale0"] == pytest.approx(0.8 * 1.4 * 0.98)
    assert report["geomean"] == pytest.approx(1.01)
    assert report["ratios"] == [1.01, 1.01]
    assert report["p10"] == [0.81, 0.85]
    assert report["p90"] == [1.21, 1.16]
    assert report["distanceBands"] == evidence[2][1]
    assert Path(report["figure"]).read_bytes() == evidence[2][2]
    assert overwritten.read_bytes() == evidence[3][2]
    assert report["medianOk"] is True
    assert report["spreadOk"] is False
    assert report["gateOk"] is False
    assert "exceed the required spread" in report["gateError"]
    assert report["figure"] in report["gateError"]
    assert len(report["probes"]) == 4
    assert report["probes"][3][0] == pytest.approx(calls[3][0])


def test_median_passing_final_keeps_final_even_if_earlier_spread_was_better(pipeline, monkeypatch):
    rows = [[(1.01, 0.97, 1.04)], [(1.001, 0.8, 1.2)]]
    calls, evidence, figure = probe_sequence(pipeline, monkeypatch, rows)
    assert pipeline.go() == 1
    report = json.loads((pipeline.ctx / "scale-fit.json").read_text())
    assert len(calls) == 2
    assert report["selectedProbe"] == 1
    assert report["selectionReason"] == "converged_final"
    assert report["scale0"] == pytest.approx(calls[1][0])
    assert report["ratios"] == [1.001]
    assert report["distanceBands"] == evidence[1][1]
    assert report["figure"] == str(figure)
    assert report["medianOk"] is True and report["spreadOk"] is False
    assert not list(pipeline.ctx.glob("scale-fit-selected-probe-*.png"))


def test_no_acceptable_probe_keeps_final_failure_and_limits_error_claim(pipeline, monkeypatch):
    calls, _, figure = probe_sequence(
        pipeline, monkeypatch, [[(1.2, 1.0, 1.3)], [(0.94, 0.92, 1.08), (1.06, 0.92, 1.08)]]
    )
    assert pipeline.go() == 1
    report = json.loads((pipeline.ctx / "scale-fit.json").read_text())
    assert len(calls) == 2
    assert report["selectedProbe"] == 1
    assert report["scale0"] == pytest.approx(calls[1][0])
    assert report["figure"] == str(figure)
    assert report["medianOk"] is False and report["spreadOk"] is True
    assert "at the evaluated scale" in report["gateError"]
    assert "at any single scale" not in report["gateError"]


def test_acceptable_probe_does_not_bypass_nonconvergence(pipeline, monkeypatch):
    pipeline.a.scale_probes = 2
    calls, _, _ = probe_sequence(pipeline, monkeypatch, [[(1.02, 0.97, 1.04)]] * 2)
    assert pipeline.go() == 1
    assert len(calls) == 2
    stages = pipeline.state.data["stages"]
    assert "did not converge" in stages["scale_fit"]["error"]
    assert "_scale" not in stages
    assert not (pipeline.ctx / "scale-fit.json").exists()
    assert not list(pipeline.ctx.glob("scale-fit-selected-probe-*.png"))


@pytest.mark.parametrize("ratio,accepted", [(1.02, True), (1.07, False)])
def test_explicit_scale_keeps_single_given_measurement(pipeline, monkeypatch, ratio, accepted):
    pipeline.a.scale0 = 0.726089
    calls, evidence, figure = probe_sequence(pipeline, monkeypatch, [[(ratio, 0.96, 1.08)]])
    assert pipeline.go() == (0 if accepted else 1)
    assert calls == [(0.726089, "given")]
    report = json.loads((pipeline.ctx / "scale-fit.json").read_text())
    assert report["scale0"] == 0.726089
    assert report["selectedProbe"] == "given" and report["selectionReason"] == "explicit_scale"
    assert report["probes"] == []
    assert report["figure"] == str(figure)
    assert report["distanceBands"] == evidence[0][1]
    assert report["gateOk"] is accepted


def test_retained_probe_passes_gate_only_when_its_spread_also_passes(pipeline, monkeypatch):
    calls, _, _ = probe_sequence(
        pipeline, monkeypatch, [[(1.02, 0.96, 1.08)], [(0.94, 0.7, 1.3), (1.06, 0.7, 1.3)]]
    )
    assert pipeline.go() == 0
    report = json.loads((pipeline.ctx / "scale-fit.json").read_text())
    assert len(calls) == 2
    assert report["selectedProbe"] == 0
    assert report["scale0"] == 0.8
    assert report["gateOk"] is True and report["gateError"] is None
