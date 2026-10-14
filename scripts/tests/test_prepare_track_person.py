"""Avatar input selection preserves quality ranking when facing preference is unavailable."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_track_person


def record(source, *, height=0.8, score=0.9, occluded=0.0, full=True, rotation=0.0):
    joints = np.zeros((18, 2))
    joints[:, 1] = np.linspace(10, 100, len(joints))
    joints[16, 0], joints[17, 0] = 30, 70
    visible = [True] * len(joints)
    visible[-1] = full
    return {
        "sample": source + 100,
        "sourceIndex": source,
        "projectedBodyJoints": joints.tolist(),
        "rootRotationVector": [0.0, rotation, 0.0],
        "jointProjectionInImage": visible,
        "heightFraction": height,
        "score": score,
        "occludedFraction": occluded,
        "maskBox": [10, 10, 70, 100],
    }


def quality(row):
    return (
        (40 / 90)
        * row["heightFraction"]
        * row["score"]
        * (1.0 if all(row["jointProjectionInImage"]) else 0.2)
        * (1 - row["occludedFraction"])
    )


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_all_back_views_rank_by_existing_quality_instead_of_input_order(order):
    rows = [
        record(0, height=0.8, score=0.6, occluded=0.5),
        record(28, height=0.7, score=0.9, occluded=0.1),
        record(12, height=0.95, score=0.99, full=False),
    ]
    ranked = prepare_track_person.rank([rows[index] for index in order])
    assert [row["sourceIndex"] for row in ranked] == [28, 0, 12]
    assert all(row["selectionPolicy"] == "any-facing-fallback" for row in ranked)
    assert all(row["facingPreferenceAvailable"] is False for row in ranked)
    by_source = {row["sourceIndex"]: row for row in rows}
    for row in ranked:
        assert row["fit"] == pytest.approx(quality(by_source[row["sourceIndex"]]))
    assert ranked[2]["fullyInFrame"] is False
    assert all(row["forwardZ"] > 0 for row in ranked)


def test_positive_facing_preference_and_original_scores_are_preserved():
    rows = [
        record(0, height=1.0, score=1.0),
        record(28, height=0.6, score=0.8, rotation=np.pi),
        record(12, height=0.9, score=0.9, rotation=2 * np.pi / 3),
    ]
    ranked = prepare_track_person.rank(rows)
    assert [row["sourceIndex"] for row in ranked] == [28, 12, 0]
    assert all(row["selectionPolicy"] == "preferred-facing" for row in ranked)
    assert all(row["facingPreferenceAvailable"] is True for row in ranked)
    by_source = {row["sourceIndex"]: row for row in rows}
    for row in ranked:
        assert row["fit"] == pytest.approx(
            max(0.0, -row["forwardZ"]) * quality(by_source[row["sourceIndex"]])
        )
    assert ranked[-1]["fit"] == 0.0


def test_any_facing_retains_existing_quality_scores_and_order():
    rows = [
        record(0, height=1.0, score=1.0),
        record(28, height=0.6, score=0.8, rotation=np.pi),
        record(12, height=0.9, score=0.9, rotation=2 * np.pi / 3),
    ]
    ranked = prepare_track_person.rank(rows, prefer_front=False)
    assert [row["sourceIndex"] for row in ranked] == [0, 12, 28]
    assert all(row["selectionPolicy"] == "any-facing" for row in ranked)
    by_source = {row["sourceIndex"]: row for row in rows}
    for row in ranked:
        assert row["fit"] == pytest.approx(quality(by_source[row["sourceIndex"]]))


def test_no_positive_quality_is_reported_without_inventing_a_valid_front_sample():
    ranked = prepare_track_person.rank([record(0, score=0.0), record(28, score=0.0)])
    assert all(row["fit"] == 0.0 for row in ranked)
    assert all(row["selectionPolicy"] == "no-positive-quality-score" for row in ranked)
    assert all(row["facingPreferenceAvailable"] is False for row in ranked)
    assert prepare_track_person.rank([]) == []


def test_back_view_fallback_prefers_more_complete_projection_over_larger_crop():
    cropped = record(10, height=1.4, score=0.95, full=False)
    cropped["jointProjectionInImage"] = [True] * 8 + [False] * 10
    complete = record(90, height=0.7, score=0.8, full=False)
    complete["jointProjectionInImage"] = [True] * 16 + [False] * 2
    assert quality(cropped) > quality(complete)

    ranked = prepare_track_person.rank([cropped, complete])

    assert [row["sourceIndex"] for row in ranked] == [90, 10]
    assert ranked[0]["projectedJointInFrameFraction"] == pytest.approx(16 / 18)
    assert ranked[0]["selectionPriority"] == "projected-joint-coverage-then-quality"
    # Coverage is a model projection, not proof that the source shows the full body.
    assert ranked[0]["fullyInFrame"] is False


def test_back_view_coverage_does_not_promote_zero_quality_or_override_equal_coverage():
    unsupported = record(0, score=0.0)
    best = record(5, height=0.7, score=0.9, full=False)
    weaker = record(9, height=0.7, score=0.6, full=False)

    ranked = prepare_track_person.rank([unsupported, weaker, best])

    assert [row["sourceIndex"] for row in ranked] == [5, 9, 0]


@pytest.fixture
def prepare_cli(tmp_path, monkeypatch):
    tracks = tmp_path / "tracks"
    track = tracks / "track_00"
    track.mkdir(parents=True)
    output = tmp_path / "prepared"

    def fake_child(command, **kwargs):
        output.mkdir()
        (output / "prepared.json").write_text(json.dumps({"prepared": "local test stub"}))

    child = Mock(side_effect=fake_child)
    monkeypatch.setattr(prepare_track_person.subprocess, "run", child)

    def invoke(records, *extra):
        (track / "motion.json").write_text(json.dumps({"frames": records}))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prepare_track_person.py",
                str(tracks),
                "--track",
                "0",
                "--clip",
                "unread-source.mp4",
                "--out",
                str(output),
                *extra,
            ],
        )
        prepare_track_person.main()

    return SimpleNamespace(invoke=invoke, child=child, output=output)


def test_default_fallback_is_reported_in_log_and_prepared_metadata(prepare_cli, capsys):
    prepare_cli.invoke([record(0, score=0.2), record(28, score=0.9, full=False)])
    # The full-body factor is retained: source 0 still has the higher base score.
    logged = json.loads(capsys.readouterr().out)
    chosen = logged["chosen"]
    assert chosen["sourceIndex"] == 0
    assert chosen["facingPreferenceAvailable"] is False
    assert chosen["selectionPolicy"] == "any-facing-fallback"
    metadata = json.loads((prepare_cli.output / "prepared.json").read_text())
    assert metadata["trackFit"] == chosen
    assert metadata["trackSample"] == chosen["sample"]
    prepare_cli.child.assert_called_once()


def test_explicit_frame_uses_requested_source_even_when_ranked_lower(prepare_cli, capsys):
    prepare_cli.invoke([record(0, rotation=np.pi), record(28)], "--frame", "28")
    chosen = json.loads(capsys.readouterr().out)["chosen"]
    assert chosen["sourceIndex"] == 28
    assert chosen["selectionPolicy"] == "explicit-frame"
    assert chosen["rankingPolicy"] == "preferred-facing"
    assert chosen["forwardZ"] > 0
    command = prepare_cli.child.call_args.args[0]
    assert command[command.index("--frame") + 1] == "28"
    metadata = json.loads((prepare_cli.output / "prepared.json").read_text())
    assert metadata["trackFit"] == chosen


def test_absent_explicit_frame_is_rejected_before_child_or_output(prepare_cli, capsys):
    with pytest.raises(SystemExit) as error:
        prepare_cli.invoke([record(0), record(28)], "--frame", "29")
    assert error.value.code == 2
    assert "source frame 29 is not present in track 0" in capsys.readouterr().err
    prepare_cli.child.assert_not_called()
    assert not prepare_cli.output.exists()


def test_empty_track_is_rejected_before_child_or_output(prepare_cli, capsys):
    with pytest.raises(SystemExit) as error:
        prepare_cli.invoke([])
    assert error.value.code == 2
    assert "track 0 has no frames to prepare" in capsys.readouterr().err
    prepare_cli.child.assert_not_called()
    assert not prepare_cli.output.exists()


def test_any_facing_cli_keeps_best_base_quality_over_available_front_view(prepare_cli, capsys):
    prepare_cli.invoke([record(0, rotation=np.pi, score=0.2), record(28)], "--any-facing")
    chosen = json.loads(capsys.readouterr().out)["chosen"]
    assert chosen["sourceIndex"] == 28
    assert chosen["selectionPolicy"] == "any-facing"
    command = prepare_cli.child.call_args.args[0]
    assert command[command.index("--frame") + 1] == "28"
