"""Real ingress, run state, packaging and dashboard; model providers are simulated.

No credentials, GPU or shared demo media are needed. This proves the plumbing, not
reconstruction quality. The paid-ledger suite separately exercises provider accounting.
"""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from plyfile import PlyData, PlyElement
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_clip
from orchestrator import cli
from orchestrator.api.app import ApiSettings
from orchestrator.api.service import create_service
from orchestrator.database import Base
from orchestrator.journal import Journal
from orchestrator.repository import PipelineRepository
from orchestrator.steps import stage_graph


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def synthetic_motion(folder):
    names = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    vertices = np.zeros(3, dtype=[(name, "<f4") for name in names])
    vertices["rot_0"] = 1
    vertices["z"] = -2
    vertices["opacity"] = 2
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(2):
        filename = f"frame_{index:03d}.ply"
        vertices["x"] = index * 0.05
        PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(
            folder / filename
        )
        frames.append(filename)
    write_json(folder / "sequence.json", {"frames": frames, "fps": 12, "count": 2})


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required for real upload probe")
def test_upload_models_failure_resume_package_and_completion(tmp_path, monkeypatch):
    clip = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=96x64:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        check=True,
    )
    database = f"sqlite:///{tmp_path / 'pipeline.db'}"
    engine = create_engine(database, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
    settings = ApiSettings(
        bearer_token="integration-token-0123456789", code_revision="test-revision"
    )
    app = create_service(
        repository=repository,
        settings=settings,
        runs_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
    )
    client = TestClient(app, headers={"Authorization": f"Bearer {settings.bearer_token}"})
    options = {"marble": "none", "objects": False}
    response = client.post(
        "/api/pipeline/runs/upload",
        files={"source": ("source.mp4", clip.read_bytes(), "video/mp4")},
        data={"options": json.dumps(options), "requested_id": "integration"},
    )
    assert response.status_code == 201, response.text
    run_id = response.json()["id"]
    root = app.state.runs._dir(run_id)
    monkeypatch.setenv("WANDER_RUN_DIR", str(root))
    monkeypatch.setenv("WANDER_DATABASE_URL", database)
    monkeypatch.delenv("WANDER_RUN_ID", raising=False)
    context = cli.RunContext.load()
    assert (
        hashlib.sha256(context.source.read_bytes()).digest()
        == hashlib.sha256(clip.read_bytes()).digest()
    )
    calls = []
    fail_motion = [True]

    def provider(self, name):
        calls.append(name)
        if name == "lhm_motion":
            if fail_motion[0]:
                raise RuntimeError("simulated provider failure; retained inputs can be resumed")
            synthetic_motion(self.ctx / "lhm-motion")
        elif name == "pi3x":
            cameras = [
                {
                    "sourceIndex": i,
                    "time": i / 12,
                    "camera_to_world": np.eye(4).tolist(),
                    "source_intrinsics": [[60, 0, 48], [0, 60, 32], [0, 0, 1]],
                    "source_image_size": [96, 64],
                }
                for i in range(2)
            ]
            write_json(self.ctx / "pi3x/cameras.json", {"cameras": cameras})
        elif name == "frame_align":
            write_json(
                self.ctx / "pi3x/framealign.json",
                {"gravityRaw": [0, 1, 0], "tiltFromYDeg": 0, "source": "simulated"},
            )
        elif name == "clean":
            shutil.copy2(self.clip, self.clean_mp4)
        else:
            write_json(self.ctx / name / "simulated-model-output.json", {"provider": "simulated"})

    for stage in ("clean", "pi3x", "frame_align", "person_prep", "lhm_frozen", "lhm_motion"):
        monkeypatch.setattr(
            run_clip.Pipeline, stage, lambda self, stage=stage: provider(self, stage)
        )
    original_popen = subprocess.Popen

    def execute(command, **kwargs):
        if len(command) < 2 or Path(command[1]).name != "run_clip.py":
            return original_popen(command, **kwargs)
        output = StringIO()
        code = 0
        environment = kwargs["env"]
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "argv", command[1:]),
            patch.object(run_clip, "CLIPS", Path(environment["WANDER_CLIPS_DIR"])),
            patch.object(run_clip, "MARBLE_DIR", Path(environment["WANDER_MARBLE_DIR"])),
            patch.object(run_clip, "SHARE", Path(environment["WANDER_SHARE_DIR"])),
            redirect_stdout(output),
        ):
            try:
                run_clip.main()
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 1
        return type(
            "CapturedRunner",
            (),
            {"stdout": BytesIO(output.getvalue().encode()), "wait": lambda _: code},
        )()

    monkeypatch.setattr(subprocess, "Popen", execute)
    stages, _ = stage_graph(options)
    assert cli.main(["finish", "succeeded", "not done yet"]) == 1
    for stage in stages:
        result = cli.main(["step", "--wait", stage])
        assert result == (1 if stage in {"lhm_motion", "package"} else 0), stage
    progress = client.get(f"/api/pipeline/runs/{run_id}/progress").json()
    assert any("simulated provider failure" in (row["reason"] or "") for row in progress["stopped"])
    assert cli.main(["finish", "succeeded", "still incomplete"]) == 1
    fail_motion[0] = False
    # Revisit everything as the supervisor does. Successful provider stages are cached.
    for stage in stages:
        assert cli.main(["step", "--wait", stage]) == 0, stage
    assert calls.count("clean") == 1
    assert calls.count("pi3x") == 1
    assert calls.count("lhm_frozen") == 1
    assert calls.count("lhm_motion") == 2
    world = root / "public/worlds" / f"{context.name}-4d"
    # A completed stage list is not deliverable if a required viewer file was lost.
    audio = world / "audio.json"
    retained_audio = audio.read_bytes()
    audio.unlink()
    assert cli.main(["finish", "succeeded", "missing delivery file"]) == 1
    assert Journal(root).finished() is None
    assert client.get(f"/api/pipeline/runs/{run_id}/progress").json()["viewerPath"] is None
    audio.write_bytes(retained_audio)
    assert cli.main(["finish", "succeeded", "simulated providers; actual packaging passed"]) == 0
    assert Journal(root).finished() == "succeeded"
    assert repository.run_summary(run_id)["status"] == "succeeded"
    packaged = json.loads((world / "person/sequence.json").read_text())
    assert packaged["allNativeGaussianAttributesFinite"] is True
    assert packaged["sourceSha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    for frame, digest in zip(packaged["frames"], packaged["frame_sha256"], strict=True):
        assert hashlib.sha256((world / "person" / frame).read_bytes()).hexdigest() == digest
    scene = client.get(f"/api/pipeline/runs/{run_id}/viewer/scene.json")
    assert scene.status_code == 200, scene.text
    params = scene.json()["params"]
    assert params["world"] == "none" and params["place"] == "0"
    assert client.get(params["video"]).content == clip.read_bytes()
    delivered = client.get(params["person"]).json()
    person_url = params["person"].rsplit("/", 1)[0]
    for frame, digest in zip(delivered["frames"], delivered["frame_sha256"], strict=True):
        response = client.get(f"{person_url}/{frame}")
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == digest
    progress = client.get(f"/api/pipeline/runs/{run_id}/progress").json()
    assert progress["finished"] == "succeeded" and not progress["stopped"]
    source = next(item for item in progress["previews"] if item["label"] == "Source clip")
    assert client.get(source["url"]).content == clip.read_bytes()
    context.source.write_bytes(b"replaced clip")
    before = list(calls)
    assert cli.main(["step", "--wait", "package"]) == 2
    assert calls == before
    assert cli.main(["finish", "succeeded", "cannot accept changed input"]) == 1
    refused = client.get(f"/api/pipeline/runs/{run_id}/progress").json()
    assert any("Source bytes changed" in (row["reason"] or "") for row in refused["stopped"])
    assert any(
        node["id"] == "package" and node["status"] == "blocked"
        for node in client.get(f"/api/pipeline/runs/{run_id}").json()["nodes"]
    )
    assert client.get(f"/api/pipeline/runs/{run_id}/files/run.json").status_code == 404
    secret = tmp_path / "private.png"
    secret.write_bytes(b"never expose")
    (root / "first").mkdir()
    (root / "first/escape.png").symlink_to(secret)
    assert client.get(f"/api/pipeline/runs/{run_id}/files/first/escape.png").status_code == 404


def test_parallel_state_writers_preserve_other_stages(tmp_path):
    first = run_clip.State(tmp_path / "state.json")
    second = run_clip.State(tmp_path / "state.json")
    first.record("clean", status="failed", error="old error")
    second.record("pi3x", status="ok")
    first.record("clean", status="ok")
    state = json.loads(first.path.read_text())["stages"]
    assert state["pi3x"]["status"] == "ok"
    assert state["clean"]["error"] is None


def test_multi_review_uses_hashed_cleaned_selection(tmp_path):
    from types import SimpleNamespace

    runner = run_clip.Pipeline.__new__(run_clip.Pipeline)
    runner.ctx = tmp_path
    runner.state = run_clip.State(tmp_path / "state.json")
    runner.marble = "multi"
    runner.a = SimpleNamespace(reuse_world=None, gate_pass=False, no_gate=False)
    runner.clean_mp4 = tmp_path / "missing.mp4"
    runner.first_png = tmp_path / "missing.png"
    image = tmp_path / "clean-multi/f_0007.png"
    image.parent.mkdir()
    image.write_bytes(b"cleaned-image-fixture")
    write_json(
        tmp_path / "clean-multi.json",
        {
            "keptFrameProvenance": [
                {
                    "frameIndex": 7,
                    "cleanedImageSha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    with pytest.raises(run_clip.GateStop):
        runner.review()
    assert (tmp_path / "review/f_0007.png").read_bytes() == image.read_bytes()
    image.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="input changed"):
        runner.review()
