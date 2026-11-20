"""Offline tests for streamed cache extraction and bounded HTTP range recovery."""

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
from modal._functions import synchronizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "scripts"))
from modal_stage_weights import extract_missing_priors, extract_priors
import modal_stage_weights as staging


def archive_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


class Response(io.BytesIO):
    def __init__(self, body, headers, status=200):
        super().__init__(body)
        self.headers = headers
        self.status = status


@pytest.mark.parametrize(
    "phase, mount_paths",
    [
        ("clean-only", ("/clean",)),
        ("clean-motion", ("/source", "/clean", "/motion")),
        ("clean-motion-repair", ("/source", "/clean", "/motion")),
        ("clean-motion-repair-v2", ("/source", "/clean", "/motion")),
        ("clean-instance", ("/clean", "/lhm")),
        ("lhm", ("/lhm",)),
        ("lhm-resume", ("/lhm",)),
        ("dino", ("/lhm",)),
        ("gfpgan", ("/lhm",)),
    ],
)
def test_local_entrypoint_uses_real_sdk_variant_with_only_phase_dependencies(
    tmp_path, phase, mount_paths
):
    base = synchronizer._translate_in(staging.stage)
    volume_type = type(synchronizer._translate_in(staging.VOLUMES["/clean"]))
    assert not any(isinstance(dep, volume_type) for dep in base._deps_())
    report = {"phase": phase, "status": "complete", "steps": []}
    deadline = time.time() + 1200
    out = tmp_path / "receipt.json"
    invocations = []

    def remote(function, selected_phase, deadline_epoch):
        variant = synchronizer._translate_in(function)
        assert variant._base_function is base
        assert tuple(path for path, _ in variant._options.validated_volumes) == mount_paths
        assert variant._deps_() == [
            synchronizer._translate_in(staging.VOLUMES[p]) for p in mount_paths
        ]
        invocations.append((selected_phase, deadline_epoch))
        return report

    with patch.object(type(staging.stage), "remote", remote):
        staging.main(phase, deadline, str(out))
    assert invocations == [(phase, deadline)]
    assert json.loads(out.read_text()) == report


def test_local_entrypoint_rejects_unknown_phase_without_invoking(tmp_path):
    out = tmp_path / "receipt.json"
    with patch.object(staging, "stage") as function, pytest.raises(ValueError, match="unknown"):
        staging.main("unknown", time.time() + 1200, str(out))
    function.with_options.assert_not_called()
    function.remote.assert_not_called()
    assert not out.exists()


@pytest.mark.parametrize(
    "phase, supplied_hash",
    [
        ("lhm-resume-reconciled", None),
        ("lhm-resume-reconciled", ""),
        ("lhm-resume-reconciled", "a" * 63),
        ("lhm-resume-reconciled", "z" * 64),
        ("lhm-resume", "a" * 64),
        ("clean-only", ""),
    ],
)
def test_reconciliation_hash_args_rejected_at_both_entrypoints(tmp_path, phase, supplied_hash):
    raw_stage = synchronizer._translate_in(staging.stage)._raw_f
    deadline = time.time() + 1200
    with patch.object(staging, "stage") as function:
        with pytest.raises(ValueError, match="--prior-resume-sha256"):
            staging.main(phase, deadline, str(tmp_path / "receipt.json"), supplied_hash)
        function.with_options.assert_not_called()
    with pytest.raises(ValueError, match="--prior-resume-sha256"):
        raw_stage(phase, deadline, supplied_hash)


def test_reconciled_local_entrypoint_invokes_one_lhm_variant(tmp_path):
    deadline = time.time() + 1200
    out = tmp_path / "receipt.json"
    with patch.object(staging, "stage") as function:
        remote = function.with_options.return_value.remote
        remote.return_value = {"status": "complete", "steps": []}
        staging.main("lhm-resume-reconciled", deadline, str(out), "A" * 64)
    function.with_options.assert_called_once_with(volumes={"/lhm": staging.VOLUMES["/lhm"]})
    remote.assert_called_once_with("lhm-resume-reconciled", deadline, prior_resume_sha256="a" * 64)
    assert json.loads(out.read_text()) == remote.return_value


@pytest.mark.parametrize(
    "phase", ["clean-only", "clean-motion", "clean-motion-repair", "clean-motion-repair-v2"]
)
def test_clean_staging_outputs_and_receipts(tmp_path, phase):
    combined = phase != "clean-only"
    clean = tmp_path / "clean"
    motion = tmp_path / "motion"
    clean.mkdir()
    motion.mkdir()
    receipts = clean / "setup-receipts"
    receipts.mkdir()
    pi = motion / "huggingface/hub/models--yyfz233--Pi3X"
    segformer_name = "models--nvidia--segformer-b0-finetuned-ade-512-512"
    segformer = clean / "huggingface/hub" / segformer_name
    snapshot = segformer / "snapshots" / staging.SEGFORMER_REV

    if combined:
        source = tmp_path / "source/hub/models--yyfz233--Pi3X/blobs"
        source.mkdir(parents=True)
        if phase == "clean-motion":
            (source / "blob").write_bytes(b"pi3x")
        else:
            (source.parent.parent / "shared").write_bytes(b"pi3x")
            (source / "blob").symlink_to("../../shared")
            pending = pi.with_name(pi.name + ".incomplete") / "blobs"
            pending.mkdir(parents=True)
            (pending / "blob").symlink_to("../../shared")
            (receipts / "clean-motion-attempt.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error": "cache has a broken or escaping symlink",
                        "steps": [],
                    }
                )
            )
    else:
        (pi / "blobs").mkdir(parents=True)
        (pi / "blobs/blob").write_bytes(b"existing inference cache")
        (receipts / "segformer.json").write_text("legacy receipt")
    motion_before = sorted(motion.rglob("*"))

    def container_path(value):
        path = Path(value)
        if path.is_absolute() and path.parts[1] in {"source", "clean", "motion", "lhm"}:
            if not combined:
                assert path.parts[1] == "clean", f"clean-only accessed {path}"
            return tmp_path.joinpath(*path.parts[1:])
        return path

    def download_snapshot(*args, **kwargs):
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"segformer")
        return str(snapshot)

    download = Mock(side_effect=download_snapshot)
    volumes = {mount: Mock() for mount in staging.VOLUMES}
    raw_stage = synchronizer._translate_in(staging.stage)._raw_f
    with (
        patch.object(staging, "Path", side_effect=container_path),
        patch.object(staging, "VOLUMES", volumes),
        patch.object(staging.modal, "current_function_call_id", return_value="fc-offline"),
        patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}),
        patch("signal.signal"),
        patch("signal.alarm") as alarm,
        patch(
            "urllib.request.urlopen", return_value=Response(b"lama", {"Content-Length": "4"})
        ) as fetch,
    ):
        deadline = time.time() + 1800
        report = raw_stage(phase, deadline)
        assert raw_stage(phase, deadline) == report

    download.assert_called_once_with(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        revision=staging.SEGFORMER_REV,
        cache_dir="/clean/huggingface/hub",
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        max_workers=1,
    )
    fetch.assert_called_once_with(staging.LAMA_URL, timeout=45)
    assert alarm.call_args_list == [call(1750), call(0)]
    assert report["status"] == "complete"
    assert report["functionCallId"] == "fc-offline"
    assert json.loads((receipts / f"{phase}-attempt.json").read_text()) == report
    assert [step["name"] for step in report["steps"]] == (
        ["pi3x", "segformer", "lama"] if combined else ["segformer", "lama"]
    )
    assert (clean / "lama/big-lama.pt").read_bytes() == b"lama"
    assert (segformer / "refs/main").read_text() == staging.SEGFORMER_REV
    assert volumes["/clean"].commit.call_count == len(report["steps"]) + 2
    volumes["/source"].commit.assert_not_called()
    volumes["/lhm"].commit.assert_not_called()
    if combined:
        assert (pi / "blobs/blob").read_bytes() == b"pi3x"
        assert (motion / "huggingface/hub" / segformer_name / "refs/main").is_file()
        assert json.loads((receipts / "pi3x.json").read_text())["sourceVolume"] == "wander-weights"
        assert (receipts / "segformer.json").is_file()
        assert (receipts / "lama.json").is_file()
        assert volumes["/motion"].commit.call_count == volumes["/clean"].commit.call_count
        if phase != "clean-motion":
            assert [repair["blob"] for repair in report["repairs"]] == ["blob"]
    else:
        assert (pi / "blobs/blob").read_bytes() == b"existing inference cache"
        assert sorted(motion.rglob("*")) == motion_before
        assert (receipts / "segformer.json").read_text() == "legacy receipt"
        assert (receipts / "clean-only-segformer.json").is_file()
        assert (receipts / "clean-only-lama.json").is_file()
        volumes["/motion"].commit.assert_not_called()


def test_clean_only_refuses_an_incomplete_attempt_before_downloading(tmp_path):
    receipt = tmp_path / "clean/setup-receipts/clean-only-attempt.json"
    receipt.parent.mkdir(parents=True)
    previous = '{"status": "failed", "steps": []}'
    receipt.write_text(previous)
    volumes = {"/clean": Mock()}
    download = Mock()
    raw_stage = synchronizer._translate_in(staging.stage)._raw_f
    with (
        patch.object(staging, "Path", side_effect=lambda path: tmp_path / path.lstrip("/")),
        patch.object(staging, "VOLUMES", volumes),
        patch.object(staging.modal, "current_function_call_id", return_value="fc-offline"),
        patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}),
        patch("urllib.request.urlopen") as fetch,
        pytest.raises(RuntimeError, match="prior incomplete attempt"),
    ):
        raw_stage("clean-only", time.time() + 1200)
    download.assert_not_called()
    fetch.assert_not_called()
    volumes["/clean"].commit.assert_not_called()
    assert receipt.read_text() == previous


@pytest.mark.parametrize("status", ["failed", "complete"])
def test_lhm_resume_preserves_legacy_attempt_receipt(tmp_path, status):
    receipt = tmp_path / "lhm/setup-receipts/lhm-resume-attempt.json"
    receipt.parent.mkdir(parents=True)
    legacy = {"phase": "lhm-resume", "status": status, "steps": []}
    original = json.dumps(legacy)
    receipt.write_text(original)
    raw_stage = synchronizer._translate_in(staging.stage)._raw_f
    with (
        patch.object(staging, "Path", side_effect=lambda path: tmp_path / path.lstrip("/")),
        patch.object(staging.modal, "current_function_call_id", return_value="fc-offline"),
        patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=Mock())}),
        patch.object(staging, "extract_missing_priors") as resume,
    ):
        deadline = time.time() + 1200
        if status == "failed":
            with pytest.raises(RuntimeError, match="prior incomplete attempt"):
                raw_stage("lhm-resume", deadline)
        else:
            assert raw_stage("lhm-resume", deadline) == legacy
    resume.assert_not_called()
    assert receipt.read_text() == original


@pytest.fixture
def reconciliation(tmp_path, monkeypatch):
    lhm = tmp_path / "lhm"
    pending = lhm / "data.incomplete"
    pending.mkdir(parents=True)
    receipts = lhm / "setup-receipts"
    receipts.mkdir()
    initial = receipts / "lhm-attempt.json"
    prior = receipts / "lhm-resume-attempt.json"
    final = receipts / "lhm-resume-2-attempt.json"
    old = {
        "phase": "lhm-resume",
        "functionCallId": "fc-" + "A" * 26,
        "status": "failed",
        "errorType": "TimeoutError",
        "error": "cache staging deadline exhausted",
        "steps": [],
        "rangeBytesReceived": 123,
        "elapsedSeconds": 1750,
    }
    initial.write_bytes(json.dumps(old | {"phase": "lhm"}, indent=4).encode() + b"\n")
    prior.write_bytes(json.dumps(old, indent=4).encode() + b"\n")
    volume = Mock()
    snapshot = Mock(side_effect=OSError("offline stop after prior staging"))
    call_id = "fc-" + "B" * 26
    raw_stage = synchronizer._translate_in(staging.stage)._raw_f

    def container_path(path):
        path = Path(path)
        return tmp_path.joinpath(*path.parts[1:]) if path.parts[:2] == ("/", "lhm") else path

    monkeypatch.setattr(staging, "Path", container_path)
    monkeypatch.setattr(staging, "VOLUMES", {"/lhm": volume})
    monkeypatch.setattr(staging.modal, "current_function_call_id", lambda: call_id)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setattr("signal.signal", Mock())
    monkeypatch.setattr("signal.alarm", Mock())

    def run(prior_hash=None):
        digest = (
            prior_hash if prior_hash is not None else hashlib.sha256(prior.read_bytes()).hexdigest()
        )
        return raw_stage("lhm-resume-reconciled", time.time() + 1200, digest)

    return SimpleNamespace(
        lhm=lhm,
        pending=pending,
        initial=initial,
        prior=prior,
        final=final,
        old=old,
        volume=volume,
        snapshot=snapshot,
        call_id=call_id,
        run=run,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "lhm"},
        {"functionCallId": None},
        {"functionCallId": "ap-" + "A" * 26},
        {"status": "started"},
        {"status": "complete"},
        {"errorType": "ValueError"},
        {"error": "cache staging deadline exhausted "},
        {"steps": [{"name": "lhm-priors"}]},
        {"steps": None},
    ],
)
def test_reconciliation_requires_the_exact_terminal_receipt(reconciliation, changes):
    state = reconciliation
    state.prior.write_text(json.dumps(state.old | changes))
    with patch.object(staging, "extract_missing_priors") as extract:
        with pytest.raises(RuntimeError, match="failed lhm-resume deadline receipt"):
            state.run()
    extract.assert_not_called()
    state.volume.commit.assert_not_called()
    assert not state.final.exists()


@pytest.mark.parametrize(
    "mismatch", ["hash", "initial-error", "initial-steps", "pending", "complete"]
)
def test_reconciliation_requires_reviewed_hash_and_eligible_cache_state(reconciliation, mismatch):
    state = reconciliation
    digest = None
    if mismatch == "hash":
        digest = hashlib.sha256(state.prior.read_bytes()).hexdigest()
        with state.prior.open("ab") as stream:
            stream.write(b" ")
    elif mismatch == "initial-error":
        state.initial.write_text(json.dumps(state.old | {"error": "other error"}))
    elif mismatch == "initial-steps":
        state.initial.write_text(json.dumps(state.old | {"steps": ["lhm-priors"]}))
    elif mismatch == "pending":
        state.pending.rmdir()
    else:
        (state.lhm / "data").mkdir()
    with patch.object(staging, "extract_missing_priors") as extract:
        with pytest.raises(RuntimeError, match="SHA-256 mismatch|diagnosed incomplete extraction"):
            state.run(digest)
    extract.assert_not_called()
    state.volume.commit.assert_not_called()
    assert not state.final.exists()


@pytest.mark.parametrize(
    "contents", [b"", b'{"status":"started"}', b'{"status":"failed"}', b'{"status":"complete"}']
)
def test_final_range_slot_blocks_every_replay(reconciliation, contents):
    state = reconciliation
    state.final.write_bytes(contents)
    with patch.object(staging, "extract_missing_priors") as extract:
        with pytest.raises(RuntimeError, match="slot already consumed"):
            state.run()
    extract.assert_not_called()
    state.volume.commit.assert_not_called()
    assert state.final.read_bytes() == contents


def test_reconciled_resume_persists_ownership_before_range_work(reconciliation, monkeypatch):
    state = reconciliation
    prior_path = "pretrained_models/gagatracker/model.pt2"
    git_path = "pretrained_models/gagatracker/.git/lfs/objects/blob"
    data = archive_bytes(
        {
            prior_path: b"runtime weights",
            git_path: b"x" * (1024 * 1024),
            "pretrained_models/human_model_files/a": b"a",
            "pretrained_models/dense_sample_points/b": b"b",
            "pretrained_models/voxel_grid/c": b"c",
        }
    )
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        skipped = archive.getmember(git_path)
    initial_bytes = state.initial.read_bytes()
    prior_bytes = state.prior.read_bytes()
    digest = hashlib.sha256(prior_bytes).hexdigest()
    requests = []
    seen_at_head = []

    def serve(request, timeout):
        assert request.full_url == staging.PRIOR_URL
        if request.get_method() == "HEAD":
            state.volume.commit.assert_called_once_with()
            seen_at_head.append(json.loads(state.final.read_text()))
            return Response(b"", {"ETag": '"pinned"', "Content-Length": str(len(data))})
        assert request.get_header("If-match") == '"pinned"'
        first, last = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        requests.append((first, last))
        return Response(
            data[first : last + 1],
            {
                "ETag": '"pinned"',
                "Content-Range": f"bytes {first}-{last}/{len(data)}",
            },
            206,
        )

    monkeypatch.setattr(staging, "PRIOR_BYTES", len(data))
    with patch("urllib.request.urlopen", side_effect=serve):
        report = state.run(digest)
        with pytest.raises(RuntimeError, match="slot already consumed"):
            state.run(digest)
    assert len(seen_at_head) == 1
    started = seen_at_head[0]
    assert started["status"] == "started" and started["steps"] == []
    assert started["functionCallId"] == state.call_id
    assert started["priorResumeFunctionCallId"] == state.old["functionCallId"]
    assert started["priorResumeReceiptSha256"] == digest
    assert (started["totalAttempt"], started["rangeAttempt"]) == (3, 2)
    assert requests and not any(
        first < skipped.offset_data + skipped.size and last >= skipped.offset_data
        for first, last in requests
    )
    assert (state.lhm / "data" / prior_path).read_bytes() == b"runtime weights"
    assert not (state.lhm / "data" / git_path).exists()
    assert report["steps"][0]["skippedGitBytes"] == skipped.size
    assert report["steps"][0]["rangeBytesReceived"] == sum(
        last - first + 1 for first, last in requests
    )
    assert state.initial.read_bytes() == initial_bytes
    assert state.prior.read_bytes() == prior_bytes
    assert json.loads(state.final.read_text()) == report
    state.snapshot.assert_called_once()


class StageWeightsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.files = {
            "./pretrained_models/large.pt": b"a" * (512 * 1024),
            "./pretrained_models/small.pt": b"b" * (128 * 1024),
        }
        self.tar = archive_bytes(self.files)

    def server(
        self,
        *,
        bad_range=False,
        bad_etag=False,
        status=206,
        truncated=False,
        head_etag='"fixed"',
        head_length_delta=0,
    ):
        requests = []

        def serve(request, timeout):
            self.assertIn(timeout, (30, 45))
            if request.get_method() == "HEAD":
                return Response(
                    b"",
                    {"ETag": head_etag, "Content-Length": str(len(self.tar) + head_length_delta)},
                )
            headers = dict(request.header_items())
            self.assertEqual(headers["If-match"], '"fixed"')
            first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            self.assertLessEqual(last - first + 1, 16 * 1024 * 1024)
            requests.append((first, last))
            body = self.tar[first : last + 1]
            if truncated:
                body = body[:-1]
            return Response(
                body,
                {
                    "Content-Range": "wrong"
                    if bad_range
                    else f"bytes {first}-{last}/{len(self.tar)}",
                    "ETag": '"changed"' if bad_etag else '"fixed"',
                },
                status,
            )

        return serve, requests

    def test_stream_records_full_hash_and_extracts(self):
        import hashlib

        report = extract_priors(io.BytesIO(self.tar), self.root, len(self.tar))
        self.assertEqual(report["archiveSha256"], hashlib.sha256(self.tar).hexdigest())
        self.assertEqual(report["archiveEntries"], 2)
        for name, contents in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), contents)

    def test_official_auxiliary_root_is_extracted(self):
        data = archive_bytes({"./gfpgan/weights/parsing_parsenet.pth": b"public model"})
        extract_priors(io.BytesIO(data), self.root, len(data))
        self.assertEqual(
            (self.root / "gfpgan/weights/parsing_parsenet.pth").read_bytes(), b"public model"
        )

    def test_stream_skips_exact_git_components_but_hashes_the_full_archive(self):
        import hashlib

        runtime = {
            "pretrained_models/sapiens/model.pt2": b"sapiens",
            "pretrained_models/.gitkeep": b"keep",
            "pretrained_models/runtime.git/config": b"runtime",
            "gfpgan/.github/config": b"config",
        }
        metadata = {
            "./pretrained_models/sapiens/.git/lfs/objects/blob": b"metadata blob",
            "gfpgan/.git": b"git worktree pointer",
        }
        data = archive_bytes(metadata | runtime)
        report = extract_priors(io.BytesIO(data), self.root, len(data))
        self.assertEqual(report["archiveBytes"], len(data))
        self.assertEqual(report["archiveSha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(report["archiveEntries"], len(runtime))
        self.assertEqual(report["skippedGitEntries"], len(metadata))
        self.assertEqual(report["skippedGitBytes"], sum(map(len, metadata.values())))
        for name, contents in runtime.items():
            self.assertEqual((self.root / name).read_bytes(), contents)
        self.assertEqual(list(self.root.rglob(".git")), [])

    def test_stream_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "content length mismatch"):
            extract_priors(io.BytesIO(self.tar), self.root, len(self.tar) + 1)

    def test_stream_rejects_unsafe_member_names(self):
        for name in (
            "../escaped",
            "/pretrained_models/escape",
            "other/root",
            "pretrained_models/../escape",
            "pretrained_models/.git/../../escape",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                data = archive_bytes({name: b"unsafe"})
                extract_priors(io.BytesIO(data), self.root, len(data))

    def test_stream_rejects_escaping_symlink(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            member = tarfile.TarInfo("pretrained_models/escape")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
            archive.addfile(member)
        data = output.getvalue()
        with self.assertRaises(tarfile.FilterError):
            extract_priors(io.BytesIO(data), self.root, len(data))

    def test_range_reuses_complete_files_and_repairs_partial_files(self):
        (self.root / "pretrained_models").mkdir()
        complete = self.root / "pretrained_models/large.pt"
        complete.write_bytes(self.files["./pretrained_models/large.pt"])
        original_mtime = complete.stat().st_mtime_ns
        (self.root / "pretrained_models/small.pt").write_bytes(b"partial")
        serve, requests = self.server()
        with patch("urllib.request.urlopen", side_effect=serve):
            report = extract_missing_priors(
                "https://example.invalid/models.tar", self.root, len(self.tar)
            )
        self.assertEqual(report["reusedMembers"], 1)
        self.assertEqual(report["extractedMembers"], 1)
        self.assertIsNone(report["archiveSha256"])
        self.assertEqual(report["sourceETag"], '"fixed"')
        self.assertLess(report["rangeBytesReceived"], len(self.tar) // 2)
        self.assertEqual(report["rangeRequests"], len(requests))
        self.assertEqual(complete.stat().st_mtime_ns, original_mtime)
        for name, contents in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), contents)

    def test_range_never_requests_git_payload_and_extracts_the_runtime_checkpoint(self):
        model = "pretrained_models/sapiens/checkpoints/sapiens_1b_epoch_173_torchscript.pt2"
        metadata = "pretrained_models/sapiens/checkpoints/sapiens-pretrain-1b-torchscript/.git/lfs/objects/b5/09/excluded-blob"
        weights = b"a" * (2 * 1024 * 1024)
        runtime = {model: weights, "pretrained_models/.gitkeep": b"keep after skipped blob"}
        self.tar = archive_bytes({model: weights, metadata: weights} | runtime)
        with tarfile.open(fileobj=io.BytesIO(self.tar), mode="r:") as archive:
            skipped = archive.getmember(metadata)
        partial = self.root / metadata
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"previous partial data")

        serve, requests = self.server()
        with patch("urllib.request.urlopen", side_effect=serve):
            report = extract_missing_priors(
                "https://example.invalid/models.tar", self.root, len(self.tar)
            )

        self.assertFalse(
            any(
                first < skipped.offset_data + skipped.size and last >= skipped.offset_data
                for first, last in requests
            )
        )
        self.assertEqual(
            report["rangeBytesReceived"], sum(last - first + 1 for first, last in requests)
        )
        self.assertEqual(report["rangeRequests"], len(requests))
        self.assertEqual(report["archiveBytes"], len(self.tar))
        self.assertIsNone(report["archiveSha256"])
        self.assertEqual(report["sourceETag"], '"fixed"')
        self.assertEqual(report["skippedGitEntries"], 1)
        self.assertEqual(report["skippedGitBytes"], skipped.size)
        self.assertEqual(report["extractedMembers"], len(runtime))
        self.assertEqual(report["reusedMembers"], 0)
        for name, contents in runtime.items():
            self.assertEqual((self.root / name).read_bytes(), contents)
        self.assertEqual(partial.read_bytes(), b"previous partial data")
        (partial.parent / "broken").symlink_to("missing")
        inventory = staging.inventory(self.root, skip_git_metadata=True)
        self.assertEqual({entry["path"] for entry in inventory["files"]}, set(runtime))

    def test_range_requires_source_etag_and_exact_archive_length(self):
        for options in ({"head_etag": ""}, {"head_length_delta": 1}):
            with self.subTest(options=options):
                serve, requests = self.server(**options)
                with (
                    patch("urllib.request.urlopen", side_effect=serve),
                    self.assertRaisesRegex(ValueError, "known-size object and ETag"),
                ):
                    extract_missing_priors(
                        "https://example.invalid/models.tar", self.root, len(self.tar)
                    )
                self.assertEqual(requests, [])

    def test_range_fails_closed_on_unpinned_or_truncated_responses(self):
        for options in (
            {"bad_range": True},
            {"bad_etag": True},
            {"status": 200},
            {"truncated": True},
        ):
            with self.subTest(options=options):
                serve, requests = self.server(**options)
                with (
                    patch("urllib.request.urlopen", side_effect=serve),
                    self.assertRaises(ValueError),
                ):
                    extract_missing_priors(
                        "https://example.invalid/models.tar", self.root, len(self.tar)
                    )
                self.assertEqual(len(requests), 1, "transport failure must not retry")

    def test_range_rejects_unsafe_member(self):
        for name in ("pretrained_models/../escape", "pretrained_models/.git/../../escape"):
            with self.subTest(name=name):
                self.tar = archive_bytes({name: b"unsafe"})
                serve, _ = self.server()
                with (
                    patch("urllib.request.urlopen", side_effect=serve),
                    self.assertRaises(ValueError),
                ):
                    extract_missing_priors(
                        "https://example.invalid/models.tar", self.root, len(self.tar)
                    )


if __name__ == "__main__":
    unittest.main()
