"""CPU staging only; all download/provider interfaces are mocked."""

import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
import modal_stage_object_weights as staging


@pytest.fixture
def manifest():
    return {
        "schema": staging.SCHEMA,
        "snapshots": [
            {
                "repoId": staging.TRELLIS,
                "revision": "a" * 40,
                "files": ["pipeline.json", "ckpts/model.json", "ckpts/model.safetensors"],
            },
            {"repoId": staging.SDXL, "revision": "b" * 40, "files": sorted(staging.SDXL_REQUIRED)},
        ],
        "dino": {
            "url": staging.DINO_URL,
            "sha256": hashlib.sha256(b"dino").hexdigest(),
            "bytes": 4,
        },
    }


def downloader(**kwargs):
    assert kwargs["token"] is False
    path = (
        Path(kwargs["cache_dir"])
        / ("models--" + kwargs["repo_id"].replace("/", "--"))
        / "snapshots"
        / kwargs["revision"]
        / kwargs["filename"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if kwargs["filename"] == "pipeline.json":
        path.write_text(json.dumps({"args": {"models": {"decoder": "ckpts/model"}}}))
    else:
        path.write_bytes(b"test model bytes")
    return str(path)


def run_stage(manifest, root, **kwargs):
    return staging.stage_cache(
        manifest,
        time.time() + 60,
        root=root,
        commit=kwargs.pop("commit", Mock()),
        downloader=kwargs.pop("downloader", downloader),
        **kwargs,
    )


def test_success_pins_cache_layout_and_repeat_verifies_without_download(tmp_path, manifest):
    receipt = tmp_path / "setup-receipts/object-weights-attempt.json"
    commits = []

    def commit():
        commits.append(json.loads(receipt.read_text())["status"])

    def serve(*args, **kwargs):
        assert commits == ["started"]
        return io.BytesIO(b"dino")

    with patch.object(staging.urllib.request, "urlopen", side_effect=serve) as request:
        result = run_stage(manifest, tmp_path, commit=commit)
    assert result["status"] == "completed"
    assert result["manifestSha256"] == staging.manifest_hash(manifest)
    assert result["fileCount"] == sum(len(s["files"]) for s in manifest["snapshots"]) + 3
    assert "files" not in result
    assert commits == ["started", "completed"]
    request.assert_called_once_with(staging.DINO_URL, timeout=30)
    assert (tmp_path / staging.DINO_PATH).read_bytes() == b"dino"
    for snapshot in manifest["snapshots"]:
        ref = (
            tmp_path
            / "huggingface/hub"
            / ("models--" + snapshot["repoId"].replace("/", "--"))
            / "refs/main"
        )
        assert ref.read_text() == snapshot["revision"]
    forbidden = Mock(side_effect=AssertionError("no downloads on repeat"))
    assert run_stage(manifest, tmp_path, downloader=forbidden) == result
    forbidden.assert_not_called()
    (tmp_path / staging.DINO_PATH).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run_stage(manifest, tmp_path, downloader=forbidden)


@pytest.mark.parametrize(
    "damage",
    [
        "unpinned",
        "extra_repo",
        "missing_fp16",
        "traversal",
        "missing_trellis_pair",
        "untrusted_dino",
        "missing_hash",
    ],
)
def test_invalid_manifest_never_starts_or_downloads(tmp_path, manifest, damage):
    if damage == "unpinned":
        manifest["snapshots"][0]["revision"] = "main"
    elif damage == "extra_repo":
        manifest["snapshots"][0]["repoId"] = "someone/model"
    elif damage == "missing_fp16":
        manifest["snapshots"][1]["files"].pop()
    elif damage == "traversal":
        manifest["snapshots"][0]["files"].append("../outside")
    elif damage == "missing_trellis_pair":
        manifest["snapshots"][0]["files"].pop()
    elif damage == "untrusted_dino":
        manifest["dino"]["url"] = "https://other.example/model"
    else:
        manifest["dino"].pop("sha256")
    download, commit = Mock(), Mock()
    with pytest.raises(ValueError):
        run_stage(manifest, tmp_path, downloader=download, commit=commit)
    download.assert_not_called()
    commit.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("remaining", [-1, 901, float("nan"), float("inf")])
def test_invalid_deadline_prevents_work(tmp_path, manifest, remaining):
    download = Mock()
    with pytest.raises(ValueError, match="deadline"):
        staging.stage_cache(
            manifest, time.time() + remaining, root=tmp_path, commit=Mock(), downloader=download
        )
    download.assert_not_called()


def test_failed_and_pending_claims_block_new_manifest_and_sanitize_receipts(tmp_path, manifest):
    def fail(**kwargs):
        receipt = json.loads((tmp_path / "setup-receipts/object-weights-attempt.json").read_text())
        assert receipt["status"] == "started"
        raise RuntimeError("token=SECRET fixture")

    with pytest.raises(RuntimeError, match="failed") as error:
        run_stage(manifest, tmp_path, downloader=fail)
    receipt = tmp_path / "setup-receipts/object-weights-attempt.json"
    assert "SECRET" not in receipt.read_text()
    assert "SECRET" not in str(error.value)
    saved = json.loads(receipt.read_text())
    assert saved["status"] == "failed"
    manifest["snapshots"][0]["revision"] = "c" * 40
    for state in ("failed", "started"):
        saved["status"] = state
        receipt.write_text(json.dumps(saved))
        blocked = Mock()
        with pytest.raises(RuntimeError, match="reconciliation"):
            run_stage(manifest, tmp_path, downloader=blocked)
        blocked.assert_not_called()


def test_dino_hash_failure_retains_failed_attempt_and_partial_file(tmp_path, manifest):
    with patch.object(staging.urllib.request, "urlopen", return_value=io.BytesIO(b"oops")):
        with pytest.raises(RuntimeError, match="failed"):
            run_stage(manifest, tmp_path)
    assert not (tmp_path / staging.DINO_PATH).exists()
    assert (tmp_path / staging.DINO_PATH).with_suffix(".download").read_bytes() == b"oops"
    assert (
        json.loads((tmp_path / "setup-receipts/object-weights-attempt.json").read_text())["status"]
        == "failed"
    )


def test_existing_conflicting_ref_stops_before_download(tmp_path, manifest):
    ref = tmp_path / "huggingface/hub/models--microsoft--TRELLIS-image-large/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("c" * 40)
    download = Mock()
    with pytest.raises(RuntimeError, match="failed"):
        run_stage(manifest, tmp_path, downloader=download)
    download.assert_not_called()
    assert ref.read_text() == "c" * 40


def test_pipeline_cannot_reference_unstaged_checkpoint(tmp_path, manifest):
    def missing_model(**kwargs):
        result = downloader(**kwargs)
        if kwargs["filename"] == "pipeline.json":
            Path(result).write_text('{"args":{"models":{"decoder":"ckpts/not-in-manifest"}}}')
        return result

    with pytest.raises(RuntimeError, match="failed"):
        run_stage(manifest, tmp_path, downloader=missing_model)


def test_local_submission_persists_before_call_and_blocks_second_call(tmp_path, manifest):
    source, receipt = tmp_path / "manifest.json", tmp_path / "submission.json"
    source.write_text(json.dumps(manifest))

    def remote(*args):
        assert json.loads(receipt.read_text())["status"] == "started"
        raise TimeoutError("SECRET ambiguous provider result")

    with patch.object(staging.stage, "remote", side_effect=remote) as submit:
        with pytest.raises(RuntimeError, match="ambiguous"):
            staging.main(str(source), str(receipt), time.time() + 60)
        with pytest.raises(RuntimeError, match="receipt exists"):
            staging.main(str(source), str(receipt), time.time() + 60)
        submit.assert_called_once()
    assert "SECRET" not in receipt.read_text()


def test_no_gpu_image_or_inference_imports():
    import ast

    tree = ast.parse(Path(staging.__file__).read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        name and name.split(".")[0] in {"torch", "diffusers", "trellis", "modal_image_to_3d"}
        for name in imported
    )
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "function"
    ]
    assert len(functions) == 1
    args = {
        kw.arg: ast.literal_eval(kw.value)
        for kw in functions[0].keywords
        if kw.arg in {"cpu", "memory", "retries", "timeout", "max_containers"}
    }
    assert args == {
        "cpu": (1, 2),
        "memory": (1024, 4096),
        "retries": 0,
        "timeout": 900,
        "max_containers": 1,
    }
    assert not any(kw.arg == "gpu" for kw in functions[0].keywords)
