"""Exercise the tracking local entrypoint with offline Modal call doubles."""

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from modal import config as modal_config
from modal.cli.entry_point import entrypoint_cli

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from worker import modal_multiperson
from worker.stages import lhm_recovery
from worker.stages.worker_resources import TIMEOUT_ENV, inference_resources


@pytest.fixture
def launch(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video bytes")
    cameras = tmp_path / "cameras.json"
    cameras.write_bytes(b"camera bytes")
    destination = tmp_path / "tracks"
    function = Mock()
    function.remote.side_effect = AssertionError("tracking must use spawn")
    call = function.spawn.return_value
    call.object_id = "fc-track"
    monkeypatch.setattr(modal_multiperson, "track", function)
    monkeypatch.setattr(modal_multiperson, "app", SimpleNamespace(app_id="ap-track"))
    monkeypatch.setattr(
        modal_multiperson,
        "INFERENCE_RESOURCES",
        inference_resources(timeout=3600, memory=65536, environment={TIMEOUT_ENV: "600"}),
    )
    monkeypatch.setattr(modal_config, "_user_config", {"local-alias": {}, "cli-alias": {}})
    monkeypatch.setattr(modal_config, "_profile", "local-alias")
    monkeypatch.setenv("MODAL_PROFILE", "local-alias")
    for name in (
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_OAUTH_REFRESH_TOKEN",
        "MODAL_OAUTH_CLIENT_ID",
        "MODAL_OAUTH_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    def run(**overrides):
        arguments = {"video": str(video), "out": str(destination), "cameras": str(cameras)}
        modal_multiperson.main(**(arguments | overrides))

    def receipt():
        return json.loads((destination / "tracking-call.json").read_text())

    return SimpleNamespace(
        run=run, receipt=receipt, destination=destination, function=function, call=call
    )


def tracking_output(error=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        log = b"tracking evidence\n"
        entry = tarfile.TarInfo("track.log")
        entry.size = len(log)
        archive.addfile(entry, io.BytesIO(log))
    return {"archive": buffer.getvalue(), "report": {"error": error}}


def test_tracking_receipt_is_durable_before_spawn_and_wait(launch, monkeypatch):
    monkeypatch.setitem(sys.modules, "worker.modal_motion", None)
    cli = CliRunner().invoke(entrypoint_cli, ["--profile", "cli-alias", "run", "--help"])
    assert cli.exit_code == 0, cli.output
    initial = {
        "schema": "wander.tracking-call/1",
        "status": "submitting",
        "appId": "ap-track",
        "modalProfile": "cli-alias",
        "functionCallId": None,
        "resultTimeoutSeconds": 1200,
    }

    def spawn(*args):
        assert launch.receipt() == initial
        return launch.call

    def get(*, timeout):
        assert timeout == 1200
        assert launch.receipt() == initial | {"status": "waiting", "functionCallId": "fc-track"}
        return tracking_output()

    launch.function.spawn.side_effect = spawn
    launch.call.get.side_effect = get
    launch.run(fps=15.0, det_thresh=0.3, gate=0.4, max_gap=4, min_track_samples=6, overlay_every=2)
    launch.function.spawn.assert_called_once_with(
        b"video bytes", b"camera bytes", 15.0, 0.3, 0.4, 4, 6, 2, False, None
    )
    launch.call.cancel.assert_not_called()
    assert launch.receipt() == initial | {"status": "result_received", "functionCallId": "fc-track"}
    assert (launch.destination / "track.log").read_bytes() == b"tracking evidence\n"
    assert json.loads((launch.destination / "modal-run.json").read_text()) == {"error": None}


def test_tracking_default_wait_and_reported_inference_failure(launch, monkeypatch):
    monkeypatch.setattr(
        modal_multiperson,
        "INFERENCE_RESOURCES",
        inference_resources(timeout=3600, memory=65536, environment={}),
    )
    monkeypatch.setenv("MODAL_TOKEN_ID", "synthetic-override")
    launch.call.get.return_value = tracking_output("inference failed")
    with pytest.raises(RuntimeError, match="see the downloaded track.log"):
        launch.run(cameras="")
    launch.function.spawn.assert_called_once_with(
        b"video bytes", None, 12, 0.15, 0.22, 8, 8, 0, False, None
    )
    launch.call.get.assert_called_once_with(timeout=7200)
    launch.call.cancel.assert_not_called()
    assert launch.receipt()["status"] == "result_received"
    assert launch.receipt()["modalProfile"] is None
    assert (launch.destination / "track.log").is_file()
    assert json.loads((launch.destination / "modal-run.json").read_text()) == {
        "error": "inference failed"
    }


@pytest.mark.parametrize(
    "error, cancel_fails, status, raised",
    [
        (TimeoutError(), False, "wait_timeout", RuntimeError),
        (TimeoutError(), True, "wait_timeout", RuntimeError),
        (KeyboardInterrupt(), False, "wait_interrupted", KeyboardInterrupt),
    ],
)
def test_wait_failure_cancels_only_owned_call_and_blocks_repeat(
    launch, error, cancel_fails, status, raised
):
    launch.call.get.side_effect = error
    if cancel_fails:
        launch.call.cancel.side_effect = OSError("private cancellation response")
    with pytest.raises(raised) as failure:
        launch.run()
    if isinstance(error, TimeoutError):
        assert failure.value.__cause__ is error
        assert "do not resubmit" in str(failure.value)
    launch.call.cancel.assert_called_once_with()
    receipt = launch.receipt()
    assert receipt["status"] == status
    assert receipt["appId"] == "ap-track"
    assert receipt["modalProfile"] == "local-alias"
    assert receipt["functionCallId"] == "fc-track"
    assert receipt["cancellation"] == ("unknown" if cancel_fails else "requested")
    assert "private cancellation response" not in json.dumps(receipt)
    with pytest.raises(FileExistsError):
        launch.run()
    launch.function.spawn.assert_called_once()
    assert launch.receipt() == receipt


def test_uncertain_submission_blocks_repeat_without_cancelling_unknown_call(launch):
    launch.function.spawn.side_effect = OSError("lost submission response")
    with pytest.raises(OSError, match="lost submission response"):
        launch.run()
    receipt = launch.receipt()
    assert receipt["status"] == "submission_uncertain"
    assert receipt["appId"] == "ap-track"
    assert receipt["modalProfile"] == "local-alias"
    assert receipt["functionCallId"] is None
    with pytest.raises(FileExistsError):
        launch.run()
    launch.function.spawn.assert_called_once()
    launch.call.get.assert_not_called()
    launch.call.cancel.assert_not_called()
    assert launch.receipt() == receipt


def test_receipt_write_failure_after_spawn_cancels_before_waiting(launch, monkeypatch):
    atomic_json = lhm_recovery.atomic_json

    def write(path, receipt):
        if receipt["status"] == "waiting":
            raise OSError("disk write failed")
        atomic_json(path, receipt)

    monkeypatch.setattr(lhm_recovery, "atomic_json", write)
    with pytest.raises(OSError, match="disk write failed"):
        launch.run()
    launch.function.spawn.assert_called_once()
    launch.call.get.assert_not_called()
    launch.call.cancel.assert_called_once_with()
    receipt = launch.receipt()
    assert receipt["status"] == "wait_interrupted"
    assert receipt["functionCallId"] == "fc-track"
    assert receipt["cancellation"] == "requested"


def test_crop_option_forwarded_without_extra_provider_submissions(launch):
    launch.call.get.return_value = tracking_output()
    launch.run(pose_crops=True)
    launch.function.spawn.assert_called_once()
    assert launch.function.spawn.call_args.args[-2] is True


def test_crops_without_calibration_stop_before_claim_or_submission(launch):
    with pytest.raises(ValueError, match="requires source cameras"):
        launch.run(pose_crops=True, cameras="")
    launch.function.spawn.assert_not_called()
    assert not launch.destination.exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_forwards_crop_option_only_when_requested(tmp_path, enabled):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_clip

    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.ctx = tmp_path
    pipeline.clip = tmp_path / "source.mp4"
    pipeline.a = SimpleNamespace(fps=12, det_thresh=0.15, pose_crops=enabled)
    pipeline.paid_run = Mock()
    pipeline.track_doc = lambda: {"trackCount": 0, "tracks": []}
    pipeline.identity_guard = Mock()
    pipeline.tracks()
    pipeline.paid_run.assert_called_once()
    command = pipeline.paid_run.call_args.args[1]
    assert ("--pose-crops" in command) is enabled


def test_reviewed_exclusions_forward_exact_bytes_after_validation(launch, tmp_path, monkeypatch):
    from worker.stages import person_exclusions

    path = tmp_path / "review.json"
    path.write_bytes(b"reviewed bytes")
    validate = Mock(return_value={})
    monkeypatch.setattr(person_exclusions, "load_video_exclusions", validate)
    launch.call.get.return_value = tracking_output()
    launch.run(person_exclusions=str(path))
    validate.assert_called_once()
    launch.function.spawn.assert_called_once()
    assert launch.function.spawn.call_args.args[-1] == b"reviewed bytes"


def test_invalid_review_stops_before_submission(launch, monkeypatch):
    from worker.stages import person_exclusions

    monkeypatch.setattr(
        person_exclusions, "load_video_exclusions", Mock(side_effect=ValueError("stale source"))
    )
    with pytest.raises(ValueError, match="stale source"):
        launch.run(person_exclusions="stale.json")
    launch.function.spawn.assert_not_called()
    assert not launch.destination.exists()


def test_runner_validates_review_before_paid_claim(tmp_path, monkeypatch):
    from worker.stages import person_exclusions
    import run_clip

    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.ctx = tmp_path
    pipeline.clip = tmp_path / "source.mp4"
    pipeline.a = SimpleNamespace(fps=12, det_thresh=0.15, person_exclusions="stale.json")
    pipeline.paid_run = Mock()
    monkeypatch.setattr(
        person_exclusions, "load_video_exclusions", Mock(side_effect=ValueError("stale source"))
    )
    with pytest.raises(ValueError, match="stale source"):
        pipeline.tracks()
    pipeline.paid_run.assert_not_called()
