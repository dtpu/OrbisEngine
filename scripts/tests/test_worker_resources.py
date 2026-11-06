"""Offline resource validation and real pinned-SDK registration; no Modal app runs."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from worker.stages.worker_resources import TIMEOUT_ENV, inference_resources


@pytest.mark.parametrize(
    "timeout,memory,containers", [(3600, 32768, 2), (1500, 32768, 1), (1800, 65536, 1)]
)
def test_absent_limit_preserves_existing_defaults(timeout, memory, containers):
    with patch.dict(os.environ, {}, clear=True):
        assert inference_resources(timeout=timeout, memory=memory, max_containers=containers) == {
            "cpu": 4,
            "memory": memory,
            "timeout": timeout,
            "max_containers": containers,
            "retries": 0,
            "scaledown_window": 2,
        }


@pytest.mark.parametrize("value", ["1", "600", "1500"])
def test_limit_caps_resources_and_startup_with_same_registered_deadline(value):
    with patch.dict(os.environ, {TIMEOUT_ENV: value}, clear=True):
        options = inference_resources(timeout=1500, memory=32768, max_containers=2)
    assert options == {
        "cpu": (4, 4),
        "memory": (32768, 32768),
        "timeout": int(value),
        "startup_timeout": int(value),
        "max_containers": 1,
        "retries": 0,
        "scaledown_window": 2,
        "env": {TIMEOUT_ENV: value},
    }


def test_explicit_environment_is_isolated_from_process_environment():
    with patch.dict(os.environ, {TIMEOUT_ENV: "900"}):
        options = inference_resources(timeout=1500, memory=32768, environment={TIMEOUT_ENV: "600"})
        assert options["timeout"] == 600
        assert inference_resources(timeout=1500, memory=32768, environment={})["timeout"] == 1500
        assert inference_resources(timeout=1500, memory=32768)["timeout"] == 900
        assert os.environ[TIMEOUT_ENV] == "900"


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "True", " 600", "600 ", "1e3", "1501"])
def test_invalid_limits_fail_closed(value):
    with patch.dict(os.environ, {TIMEOUT_ENV: value}, clear=True):
        with pytest.raises(ValueError, match=TIMEOUT_ENV):
            inference_resources(timeout=1500, memory=32768)


@pytest.mark.parametrize("name,maximum", [("clean_video", 3600), ("motion", 1500), ("lhm", 1800)])
def test_invalid_worker_setting_precedes_image_or_volume_setup(name, maximum):
    import modal

    with (
        patch.dict(os.environ, {TIMEOUT_ENV: str(maximum + 1)}),
        patch.object(modal.Image, "debian_slim", side_effect=AssertionError("image setup")),
        patch.object(modal.Image, "from_registry", side_effect=AssertionError("image setup")),
        patch.object(modal.Volume, "from_name", side_effect=AssertionError("volume setup")),
    ):
        sys.modules.pop(f"worker.modal_{name}", None)
        with pytest.raises(ValueError, match=TIMEOUT_ENV):
            importlib.import_module(f"worker.modal_{name}")


@pytest.mark.parametrize("gpu", [None, "L4", "L40S"])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("value", [None, "600"])
def test_script_import_registers_with_pinned_sdk_and_packages_helper(remote, value, gpu):
    # Separate imports reproduce the CLI file path and container module reimport without
    # launching an app, building an image, fetching volumes or contacting providers.
    code = r"""
import os, runpy, sys
from pathlib import Path
from unittest.mock import patch
import modal
root = Path.cwd()
sys.path.insert(0, str(root / "worker/stages"))
original = modal.App.function
registered = []
packaged = []
original_file = modal.Image.add_local_file

def register(self, *args, **kwargs):
    if kwargs.get("gpu") in {"L4", "L40S"}:
        registered.append(kwargs)
    return original(self, *args, **kwargs)

def add_file(self, local_path, remote_path, **kwargs):
    packaged.append((Path(local_path), remote_path))
    return original_file(self, local_path, remote_path, **kwargs)

with patch.object(modal.App, "function", register), patch.object(modal.Image, "add_local_file", add_file), patch.object(modal, "is_local", return_value=REMOTE):
    for name, maximum, memory, containers in [("clean_video",3600,32768,2),("motion",1500,32768,1),("lhm",1800,65536,1)]:
        namespace = runpy.run_path(str(root / f"worker/modal_{name}.py"))
        options = registered[-1]
        assert options["timeout"] == (600 if BOUNDED else maximum)
        assert options["cpu"] == ((4,4) if BOUNDED else 4)
        assert options["memory"] == ((memory,memory) if BOUNDED else memory)
        assert options["max_containers"] == (1 if BOUNDED else containers)
        assert options["retries"] == 0 and options["scaledown_window"] == 2
        assert options["gpu"] == (GPU if name == "lhm" else "L4")
        expected_env = {"WANDER_LHM_GPU":GPU} if name == "lhm" else {}
        if BOUNDED:
            assert options["startup_timeout"] == 600
            expected_env["WANDER_WORKER_TIMEOUT_SECONDS"] = "600"
        else:
            assert "startup_timeout" not in options
        if expected_env:
            assert options["env"] == expected_env
        else:
            assert "env" not in options
        assert packaged[-1] == (root / "worker/stages/worker_resources.py", "/root/worker_resources.py")
assert len(registered) == 3
print("registered three workers without provider calls")
"""
    env = os.environ.copy()
    env.pop(TIMEOUT_ENV, None)
    env.pop("WANDER_LHM_GPU", None)
    if value is not None:
        env[TIMEOUT_ENV] = value
    if gpu is not None:
        env["WANDER_LHM_GPU"] = gpu
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"REMOTE={remote!r}\nBOUNDED={value is not None!r}\nGPU={gpu or 'L4'!r}\n" + code,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "registered three workers" in result.stdout


@pytest.mark.parametrize("gpu", ["L4", "L40S"])
@pytest.mark.parametrize("animate", [False, True])
def test_bounded_lhm_timeout_finalizes_failed_checkpoint_without_retry(tmp_path, animate, gpu):
    import json
    import types
    import uuid
    from unittest.mock import Mock

    sys.path.insert(0, str(ROOT / "worker"))
    from stages import lhm_execution, lhm_recovery
    from worker import modal_lhm

    real_start = lhm_recovery.start_checkpoint
    recovery_id = uuid.uuid4().hex
    commits = []

    def start(_cache, identifier, mode):
        return real_start(tmp_path, identifier, mode)

    def commit():
        manifest = tmp_path / lhm_recovery.remote_root(recovery_id) / "manifest.json"
        commits.append(json.loads(manifest.read_text())["status"])

    def timeout(command, log_path, **kwargs):
        assert kwargs["execution_timeout"] == 600
        assert kwargs["started"] > 0
        assert command[1] == f"/root/stages/lhm_{'animate' if animate else 'person'}.py"
        log_path.write_text("partial inference log\n")
        (log_path.parent / "partial.ply").write_bytes(b"partial output")
        raise subprocess.TimeoutExpired(command, 540)

    hub = types.SimpleNamespace(snapshot_download=Mock(return_value="/cache/staged-model"))
    with (
        patch.dict(sys.modules, {"huggingface_hub": hub}),
        patch.object(modal_lhm, "INFERENCE_RESOURCES", {"timeout": 600, "startup_timeout": 600}),
        patch.object(modal_lhm, "REQUESTED_GPU", gpu),
        patch.object(modal_lhm, "cache", types.SimpleNamespace(commit=commit)),
        patch.object(modal_lhm, "link_model_caches", return_value=[]),
        patch.object(lhm_recovery, "start_checkpoint", side_effect=start),
        patch.object(lhm_execution, "run_logged_inference", side_effect=timeout) as inference,
        patch("os.chdir"),
        patch("builtins.print"),
        pytest.warns(UserWarning, match="executing locally"),
    ):
        result = modal_lhm.frozen.local({}, animate=animate, recovery_id=recovery_id)
    assert commits == ["running", "failed"]
    assert result["status"] == "failed"
    assert result["report"]["timeoutSeconds"] == 600
    report = result["report"]
    gpu_rate = {"L4": 0.000222, "L40S": 0.000542}[gpu]
    expected_rate = gpu_rate + 4 * 0.0000131 + 64 * 0.00000222
    assert report["gpu"] == report["requestedGPU"] == gpu
    assert report["gpuRateUSDPerSecond"] == gpu_rate
    assert report["computeRateUSDPerSecond"] == pytest.approx(expected_rate)
    assert report["estimatedComputeUSD"] == pytest.approx(report["seconds"] * expected_rate)
    assert "TimeoutExpired" in result["report"]["error"]
    assert inference.call_count == 1
    assert hub.snapshot_download.call_args.kwargs["local_files_only"] is True
    output = tmp_path / lhm_recovery.remote_root(recovery_id) / "output"
    assert (output / "partial.ply").read_bytes() == b"partial output"
    assert "partial inference log" in (output / "inference.log").read_text()


@pytest.mark.parametrize("value", ["", "A10", "l4", "L40s", " L40S", "L40S ", "L4,L40S", "L40S:2"])
def test_invalid_lhm_gpu_precedes_modal_setup(value):
    import modal
    from worker.stages.worker_resources import LHM_GPU_ENV, lhm_gpu

    with (
        patch.dict(os.environ, {LHM_GPU_ENV: value}),
        patch.object(modal.Image, "from_registry", side_effect=AssertionError("image setup")),
        patch.object(modal.Volume, "from_name", side_effect=AssertionError("volume setup")),
    ):
        with pytest.raises(ValueError, match=LHM_GPU_ENV):
            lhm_gpu()
        sys.modules.pop("worker.modal_lhm", None)
        with pytest.raises(ValueError, match=LHM_GPU_ENV):
            importlib.import_module("worker.modal_lhm")


def test_lhm_gpu_explicit_environment_and_rates_ignore_process_secrets():
    from worker.stages.worker_resources import LHM_GPU_ENV, lhm_compute_rate, lhm_gpu

    with patch.dict(os.environ, {LHM_GPU_ENV: "L40S", "PRIVATE_KEY": "secret"}):
        assert lhm_gpu() == "L40S"
        assert lhm_gpu(environment={}) == "L4"
        assert lhm_gpu(environment={LHM_GPU_ENV: "L4"}) == "L4"
        assert os.environ[LHM_GPU_ENV] == "L40S"
    assert lhm_compute_rate("L4") == pytest.approx(0.00041648)
    assert lhm_compute_rate("L40S") == pytest.approx(0.00073648)
    with pytest.raises(ValueError, match=LHM_GPU_ENV):
        lhm_compute_rate("A10")
