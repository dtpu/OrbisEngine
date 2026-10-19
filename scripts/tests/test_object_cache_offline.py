"""Pinned cache contracts and loader routing without GPU/model/provider calls."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
import modal_image_to_3d as worker
import modal_stage_object_aux as aux
from modal_stage_object_weights import manifest_hash, file_record


@pytest.fixture
def retained(tmp_path):
    inventory = []

    def write(relative, content=b"fixture"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return file_record(tmp_path, path)

    models = []
    for repo in (worker.MODEL_ID, worker.REFINER_ID):
        relative = "huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots/" + "a" * 40
        config = "pipeline.json" if repo == worker.MODEL_ID else "model_index.json"
        inventory.append(write(relative + "/" + config))
        models.append({"repoId": repo, "revision": "a" * 40})
    inventory.append(write(worker.DINO_CHECKPOINT))
    primary = {
        "schema": "wander.object-cache-attempt/1",
        "status": "completed",
        "files": inventory,
        "inventorySha256": manifest_hash(inventory),
        "snapshots": models,
    }
    code = "torch/hub/dinov2-" + "b" * 40
    inventory = [write(code + "/hubconf.py"), write("u2net/u2net.onnx")]
    auxiliary = {
        "schema": "wander.object-aux-attempt/1",
        "status": "completed",
        "files": inventory,
        "inventorySha256": manifest_hash(inventory),
        "dinoCodePath": code,
        "dinoRevision": "b" * 40,
    }
    root = tmp_path / "setup-receipts"
    root.mkdir()
    for name, doc in zip(worker.CACHE_RECEIPTS, (primary, auxiliary), strict=True):
        (root / name).write_text(json.dumps(doc))
    return tmp_path


def test_cpu_preflight_verifies_hashes_and_pins_snapshots(retained):
    result = worker.checked_object_cache(retained, need_refiner=True)
    assert result["models"][worker.MODEL_ID]["path"].endswith("/snapshots/" + "a" * 40)
    assert (
        worker.checked_object_cache(retained, expected=result["receipts"], hash_files=False)
        == result
    )
    (retained / worker.DINO_CHECKPOINT).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        worker.checked_object_cache(retained)


@pytest.mark.parametrize("damage", ["pending", "missing", "receipt_changed", "escape", "inventory"])
def test_preflight_fails_closed_before_model_loading(retained, damage):
    pins = worker.checked_object_cache(retained)["receipts"]
    path = retained / "setup-receipts/object-aux-attempt.json"
    doc = json.loads(path.read_text())
    if damage == "pending":
        doc["status"] = "started"
    elif damage == "missing":
        (retained / "u2net/u2net.onnx").unlink()
    elif damage == "receipt_changed":
        doc["extra"] = "changed receipt"
    elif damage == "escape":
        doc["files"][0]["path"] = "../escape"
        doc["inventorySha256"] = manifest_hash(doc["files"])
    else:
        doc["inventorySha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        worker.checked_object_cache(retained, expected=pins)


def test_trellis_uses_local_hub_source_and_exact_cached_checkpoint(retained):
    result = worker.checked_object_cache(retained)
    load_weights = Mock(return_value={"exact": "state"})
    torch = SimpleNamespace(load=load_weights)

    def original_hub(path, name, **kwargs):
        assert path == result["dinoCodePath"]
        assert name == "dinov2_vitl14_reg"
        assert kwargs == {"source": "local", "pretrained": True}
        return torch.hub.load_state_dict_from_url(
            aux.U2NET_URL.replace(
                "github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx", "invalid"
            )
        )

    torch.hub = SimpleNamespace(
        load=original_hub, load_state_dict_from_url=Mock(side_effect=AssertionError("network"))
    )
    hf = SimpleNamespace(hf_hub_download=Mock(side_effect=AssertionError("network")))

    class Pipeline:
        @staticmethod
        def from_pretrained(path):
            assert path == result["models"][worker.MODEL_ID]["path"]
            with pytest.raises(ValueError, match="unstaged"):
                hf.hf_hub_download("other", "weights")
            with pytest.raises(ValueError, match="checkpoint"):
                torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
            return torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
                map_location="cpu",
                check_hash=True,
            )

    with patch.dict(sys.modules, {"torch": torch, "huggingface_hub": hf}):
        assert worker.local_trellis_load(Pipeline, result) == {"exact": "state"}
    load_weights.assert_called_once_with(
        result["dinoWeightsPath"], map_location="cpu", weights_only=True
    )
    assert torch.hub.load is original_hub
    hf.hf_hub_download.assert_not_called()


def test_background_removal_model_uses_local_path(retained):
    result = worker.checked_object_cache(retained)

    class Session:
        download_models = Mock(side_effect=AssertionError("must not download"))

    pipeline = SimpleNamespace(preprocess_image=lambda image: Session.download_models())
    with patch.dict(sys.modules, {"rembg.sessions.u2net": SimpleNamespace(U2netSession=Session)}):
        assert worker.local_preprocess(pipeline, object(), result) == result["u2netPath"]
    Session.download_models.assert_not_called()


def test_main_stops_before_gpu_when_cpu_preflight_fails(tmp_path):
    image = tmp_path / "crop.png"
    image.write_bytes(b"no image decode required")
    with (
        patch.object(worker.cache_preflight, "remote", side_effect=ValueError("missing cache")),
        patch.object(worker.refine_crop, "remote") as refine,
        patch.object(worker.image_to_3d, "remote") as shape,
    ):
        with pytest.raises(ValueError, match="missing cache"):
            worker.main(image=str(image), out_dir=str(tmp_path / "out"), refine_first=True)
    refine.assert_not_called()
    shape.assert_not_called()


@pytest.fixture
def auxiliary_manifest():
    return {
        "schema": aux.SCHEMA,
        "dinoCode": {"revision": "c" * 40, "archiveSha256": "d" * 64, "bytes": 100},
        "u2net": {"url": aux.U2NET_URL, "md5": aux.U2NET_MD5, "bytes": 4},
    }


def code_archive(revision, dangerous=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        entry = tarfile.TarInfo(
            "dinov2-" + revision + ("/../escape" if dangerous else "/hubconf.py")
        )
        entry.size = 4
        archive.addfile(entry, io.BytesIO(b"code"))
    return stream.getvalue()


def test_auxiliary_staging_has_separate_receipt_and_repeat_guard(tmp_path, auxiliary_manifest):
    revision = auxiliary_manifest["dinoCode"]["revision"]

    def download(url, path, *args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(code_archive(revision) if "codeload" in url else b"onnx")

    primary = tmp_path / "setup-receipts/object-weights-attempt.json"
    primary.parent.mkdir()
    primary.write_text("primary must remain unchanged")
    with patch.object(aux, "download", side_effect=download) as fetch:
        result = aux.stage_aux(auxiliary_manifest, time.time() + 60, root=tmp_path, commit=Mock())
        assert result["status"] == "completed"
        assert result["u2netSha256"] == hashlib.sha256(b"onnx").hexdigest()
        assert (
            aux.stage_aux(auxiliary_manifest, time.time() + 60, root=tmp_path, commit=Mock())
            == result
        )
        assert fetch.call_count == 2
    assert primary.read_text() == "primary must remain unchanged"
    receipt = tmp_path / "setup-receipts/object-aux-attempt.json"
    saved = json.loads(receipt.read_text())
    saved["status"] = "failed"
    receipt.write_text(json.dumps(saved))
    with (
        patch.object(aux, "download") as fetch,
        pytest.raises(RuntimeError, match="reconciliation"),
    ):
        aux.stage_aux(auxiliary_manifest, time.time() + 60, root=tmp_path, commit=Mock())
    fetch.assert_not_called()


def test_auxiliary_download_checks_hash_and_size(tmp_path):
    with patch.object(aux.urllib.request, "urlopen", return_value=io.BytesIO(b"onnx")) as fetch:
        aux.download(aux.U2NET_URL, tmp_path / "model", 4, "md5", hashlib.md5(b"onnx").hexdigest())
        fetch.assert_called_once_with(aux.U2NET_URL, timeout=30)
    with (
        patch.object(aux.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")),
        pytest.raises(ValueError, match="checksum/size"),
    ):
        aux.download(aux.U2NET_URL, tmp_path / "wrong", 4, "md5", aux.U2NET_MD5)


def test_auxiliary_archive_rejects_escaping_paths(tmp_path):
    archive = tmp_path / "code.tar.gz"
    archive.write_bytes(code_archive("c" * 40, dangerous=True))
    with pytest.raises(ValueError, match="member"):
        aux.extract_code(archive, tmp_path / "out", "c" * 40)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("text_only", [False, True])
def test_refiner_loads_only_pinned_local_snapshot(retained, text_only, monkeypatch):
    from PIL import Image

    selected = worker.checked_object_cache(retained, need_refiner=True)
    buffer = io.BytesIO()
    image = Image.new("RGB", (8, 8))
    image.save(buffer, format="PNG")
    pipeline = Mock(return_value=SimpleNamespace(images=[image]))
    loader = Mock(return_value=pipeline)
    diffusers = SimpleNamespace(
        StableDiffusionXLPipeline=SimpleNamespace(from_pretrained=loader),
        StableDiffusionXLImg2ImgPipeline=SimpleNamespace(from_pretrained=loader),
    )
    torch = SimpleNamespace(
        float16="float16",
        cuda=SimpleNamespace(empty_cache=lambda: None),
        Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed),
    )
    monkeypatch.setattr(worker, "checked_object_cache", lambda **kwargs: selected)
    with (
        patch.dict(sys.modules, {"torch": torch, "diffusers": diffusers}),
        patch.object(worker.cache, "commit"),
        patch.object(worker.cache, "reload"),
        patch.dict("os.environ", {}, clear=False),
    ):
        result = worker.refine_crop.local(
            buffer.getvalue(), size=8, text_only=text_only, cache_receipts=selected["receipts"]
        )
    loader.assert_called_once_with(
        selected["models"][worker.REFINER_ID]["path"],
        torch_dtype="float16",
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    assert result["meta"]["modelRevision"] == selected["models"][worker.REFINER_ID]["revision"]


def test_auxiliary_failure_receipt_is_durable_and_sanitized(tmp_path, auxiliary_manifest):
    committed = []

    def commit():
        committed.append(
            json.loads((tmp_path / "setup-receipts/object-aux-attempt.json").read_text())["status"]
        )

    with (
        patch.object(aux, "download", side_effect=TimeoutError("secret header")),
        pytest.raises(RuntimeError, match="reconcile"),
    ):
        aux.stage_aux(auxiliary_manifest, time.time() + 60, root=tmp_path, commit=commit)
    assert committed == ["started", "failed"]
    assert "secret header" not in (tmp_path / "setup-receipts/object-aux-attempt.json").read_text()


def test_auxiliary_resources_are_bounded_and_cpu_only():
    import ast

    tree = ast.parse(Path(aux.__file__).read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "function"
    ]
    assert len(functions) == 1
    limits = {
        kw.arg: ast.literal_eval(kw.value)
        for kw in functions[0].keywords
        if kw.arg in {"cpu", "memory", "retries", "timeout", "max_containers"}
    }
    assert limits == {
        "cpu": (1, 2),
        "memory": (1024, 4096),
        "retries": 0,
        "timeout": 900,
        "max_containers": 1,
    }
    assert all(kw.arg != "gpu" for kw in functions[0].keywords)


def test_auxiliary_entrypoint_packages_helper_and_imports_without_repository(tmp_path):
    import ast
    import shutil
    import subprocess

    source = Path(aux.__file__)
    tree = ast.parse(source.read_text())
    mounts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_file"
    ]
    assert len(mounts) == 1
    mount = mounts[0]
    assert ast.literal_eval(mount.args[1]) == "/root/modal_stage_object_weights.py"
    assert {kw.arg: ast.literal_eval(kw.value) for kw in mount.keywords} == {"copy": True}
    assert isinstance(mount.args[0], ast.Call)
    assert mount.args[0].func.attr == "with_name"
    assert ast.literal_eval(mount.args[0].args[0]) == "modal_stage_object_weights.py"
    for filename in ("modal_stage_object_aux.py", "modal_stage_object_weights.py"):
        shutil.copyfile(source.with_name(filename), tmp_path / filename)
    # Reproduce the remote import namespace: only the entrypoint and its explicit helper,
    # installed packages, and stdlib; no repository/worker package or PYTHONPATH fallback.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import modal_stage_object_aux as m; assert m.manifest_hash({'a':1}); print('standalone import passed')",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.strip() == "standalone import passed"
