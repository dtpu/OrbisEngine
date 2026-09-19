"""Offline tests for streamed cache extraction and bounded HTTP range recovery."""

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from modal_stage_weights import extract_missing_priors, extract_priors


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

    def server(self, *, bad_range=False, bad_etag=False, status=206, truncated=False):
        requests = []

        def serve(request, timeout):
            self.assertIn(timeout, (30, 45))
            if request.get_method() == "HEAD":
                return Response(b"", {"ETag": '"fixed"', "Content-Length": str(len(self.tar))})
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

    def test_stream_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "content length mismatch"):
            extract_priors(io.BytesIO(self.tar), self.root, len(self.tar) + 1)

    def test_stream_rejects_unsafe_member_names(self):
        for name in (
            "../escaped",
            "/pretrained_models/escape",
            "other/root",
            "pretrained_models/../escape",
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
        self.tar = archive_bytes({"pretrained_models/../escape": b"unsafe"})
        serve, _ = self.server()
        with patch("urllib.request.urlopen", side_effect=serve), self.assertRaises(ValueError):
            extract_missing_priors("https://example.invalid/models.tar", self.root, len(self.tar))


if __name__ == "__main__":
    unittest.main()
