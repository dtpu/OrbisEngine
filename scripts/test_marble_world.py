"""Offline contracts for Marble requests and recovery. Never uses credentials or the network."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import marble_world as client
import run_clip


class MarbleContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.png, self.jpg, self.mp4 = [
            self.root / name for name in ("one.png", "two.jpg", "clip.mp4")
        ]
        for path in (self.png, self.jpg, self.mp4):
            path.write_bytes(b"owned-mock-upload")
        self.directory = self.root / "metadata"
        self.requests = []
        self.assets = 0
        self.world = {
            "world_marble_url": "https://example.invalid/world",
            "assets": {
                "splats": {
                    "semantics_metadata": {"metric_scale_factor": 1, "ground_plane_offset": 2},
                    "spz_urls": {"full_res": "https://example.invalid/full.spz"},
                },
                "thumbnail_url": "https://example.invalid/thumb.png",
            },
        }
        self.addCleanup(patch.stopall)
        # Any unexpectedly unmocked path is forbidden from reaching the network.
        patch.object(
            client.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")
        ).start()
        patch.object(client, "call", side_effect=self.call).start()
        patch.object(client.time, "sleep").start()
        self.download = patch.object(client, "download", side_effect=self.save).start()

    def save(self, url, destination, require_gzip=False):
        p = Path(destination)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x1f\x8basset")
        return 7, hashlib.sha256(p.read_bytes()).hexdigest()

    def call(self, method, path, body=None, **kwargs):
        self.requests.append((method, path, body, kwargs))
        if path.endswith("prepare_upload"):
            self.assets += 1
            return 200, {
                "media_asset": {"media_asset_id": f"asset-{self.assets}"},
                "upload_info": {"upload_url": "https://example.invalid/upload"},
            }
        if method == "PUT":
            return 200, b""
        if path.endswith("worlds:generate"):
            return 200, {"operation_id": "operation-1"}
        if "/operations/" in path:
            return 200, {"done": True, "metadata": {"world_id": "world-1"}}
        if "/worlds/" in path:
            return 200, self.world
        self.fail(f"Unexpected request: {method} {path}")

    def invoke(self, input_type, mode, *arguments):
        argv = [
            input_type,
            mode,
            *map(str, arguments),
            "--name",
            "fixture",
            "--marble-dir",
            str(self.directory),
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client.main(argv)

    def generated(self):
        requests = [r for r in self.requests if r[1].endswith("worlds:generate")]
        self.assertEqual(len(requests), 1)
        return requests[0][2]

    def test_image_payload_and_upload_match_original_contract(self):
        self.invoke(
            "image",
            "submit",
            self.png,
            "--prompt",
            "empty room",
            "--seed",
            "7",
            "--model",
            "marble-1.1",
            "--ops",
            "shared",
        )
        content = {"source": "media_asset", "media_asset_id": "asset-1"}
        self.assertEqual(
            self.generated(),
            {
                "display_name": "fixture",
                "model": "marble-1.1",
                "permission": {"public": False},
                "seed": 7,
                "world_prompt": {
                    "type": "image",
                    "image_prompt": content,
                    "is_pano": False,
                    "disable_recaption": True,
                    "text_prompt": "empty room",
                },
            },
        )
        self.assertEqual(
            self.requests[0][2], {"file_name": "one.png", "kind": "image", "extension": "png"}
        )
        self.assertEqual(
            self.requests[1][3],
            {
                "raw": self.png.read_bytes(),
                "headers": {
                    "x-goog-content-length-range": "0,104857600",
                    "Content-Type": "image/png",
                },
                "timeout": 600,
            },
        )
        self.assertIn(
            "[fixture] op operation-1 submitted", (self.directory / "shared-ops.txt").read_text()
        )

    def test_submit_only_persists_operation_without_polling_or_fetching(self):
        self.invoke(
            "video",
            "submit-only",
            self.mp4,
            "--prompt",
            "empty room",
        )
        self.generated()
        self.assertFalse(any("/operations/" in request[1] for request in self.requests))
        self.assertFalse(any("/worlds/" in request[1] for request in self.requests))
        receipt = json.loads((self.directory / "fixture-generation.json").read_text())
        self.assertEqual(receipt["status"], "submitted")
        self.assertEqual(receipt["operation_id"], "operation-1")

    def test_multi_payload_prompt_override_azimuth_and_jpeg_upload(self):
        prompt = self.root / "prompt.json"
        prompt.write_text(json.dumps({"text_prompt": "reviewed prompt"}))
        self.invoke(
            "multi",
            "submit",
            "--images",
            f"{self.png}:359.7",
            self.jpg,
            "--prompt",
            "overridden",
            "--prompt-file",
            prompt,
            "--seed",
            "0",
        )
        expected = {
            "display_name": "fixture",
            "model": "marble-1.1",
            "permission": {"public": False},
            "seed": 0,
            "world_prompt": {
                "type": "multi-image",
                "reconstruct_images": True,
                "disable_recaption": True,
                "text_prompt": "reviewed prompt",
                "multi_image_prompt": [
                    {
                        "content": {"source": "media_asset", "media_asset_id": "asset-1"},
                        "azimuth": 359.7,
                    },
                    {"content": {"source": "media_asset", "media_asset_id": "asset-2"}},
                ],
            },
        }
        self.assertEqual(self.generated(), expected)
        self.assertEqual(
            json.loads((self.directory / "fixture-request.json").read_text()), expected
        )
        self.assertEqual(
            self.requests[2][2], {"file_name": "two.jpg", "kind": "image", "extension": "jpg"}
        )
        self.assertEqual(self.requests[3][3]["headers"]["Content-Type"], "image/jpeg")

    def test_video_payload_without_prompt_keeps_automatic_captioning(self):
        self.invoke("video", "submit", self.mp4)
        self.assertEqual(
            self.generated(),
            {
                "model": "marble-1.1",
                "permission": {"public": False},
                "world_prompt": {
                    "type": "video",
                    "video_prompt": {"source": "media_asset", "media_asset_id": "asset-1"},
                },
            },
        )
        self.assertEqual(
            self.requests[0][2], {"file_name": "clip.mp4", "kind": "video", "extension": "mp4"}
        )
        self.assertEqual(self.requests[1][3]["headers"]["Content-Type"], "video/mp4")
        text = (self.directory / "fixture-ops.txt").read_text()
        self.assertNotIn("[fixture]", text)
        self.assertIn("world world-1 https://example.invalid/world metric_scale_factor", text)

    def test_video_uses_explicit_prompt_as_is_and_records_exact_submitted_input(self):
        prompt = self.root / "video-prompt.json"
        prompt.write_text(json.dumps({"text_prompt": "The recorded kitchen has a central island"}))
        self.invoke("video", "submit", self.mp4, "--prompt-file", prompt, "--model", "marble-1.1")
        request = self.generated()
        self.assertEqual(
            request["world_prompt"]["text_prompt"], "The recorded kitchen has a central island"
        )
        self.assertTrue(request["world_prompt"]["disable_recaption"])
        self.assertNotIn("reconstruct_images", request["world_prompt"])
        upload = next(r for r in self.requests if r[0] == "PUT")
        self.assertEqual(upload[3]["raw"], self.mp4.read_bytes())
        saved = json.loads((self.directory / "fixture-request.json").read_text())
        self.assertEqual(saved, request)
        receipt = json.loads((self.directory / "fixture-generation.json").read_text())
        self.assertEqual(
            receipt["inputs"][0]["sha256"], hashlib.sha256(upload[3]["raw"]).hexdigest()
        )
        self.assertEqual(receipt["inputs"][0]["bytes"], len(upload[3]["raw"]))
        self.assertEqual(
            receipt["request_sha256"],
            hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        )

    def test_video_empty_prompt_keeps_automatic_captioning(self):
        self.invoke("video", "submit", self.mp4, "--prompt", "")
        prompt = self.generated()["world_prompt"]
        self.assertNotIn("text_prompt", prompt)
        self.assertNotIn("disable_recaption", prompt)

    def test_documented_upload_id_and_required_headers_are_used(self):
        def documented_call(method, path, body=None, **kwargs):
            if path.endswith("prepare_upload"):
                return 200, {
                    "media_asset": {"id": "documented-video-id"},
                    "upload_info": {
                        "upload_url": "https://example.invalid/upload",
                        "upload_method": "PUT",
                        "required_headers": {
                            "x-goog-content-length-range": "0,1048576000",
                            "x-required": "value",
                        },
                    },
                }
            return self.call(method, path, body, **kwargs)

        with patch.object(client, "call", side_effect=documented_call):
            self.invoke("video", "submit", self.mp4)
        self.assertEqual(
            self.generated()["world_prompt"]["video_prompt"]["media_asset_id"],
            "documented-video-id",
        )
        upload = next(r for r in self.requests if r[0] == "PUT")
        self.assertEqual(upload[3]["headers"]["x-required"], "value")
        self.assertEqual(upload[3]["headers"]["x-goog-content-length-range"], "0,1048576000")

    def test_invalid_video_prompt_fails_before_any_remote_request(self):
        prompt = self.root / "bad-prompt.json"
        prompt.write_text('{"text_prompt": []}')
        with self.assertRaises(SystemExit):
            self.invoke("video", "submit", self.mp4, "--prompt-file", prompt)
        self.assertEqual(self.requests, [])
        self.assertFalse((self.directory / "fixture-generation.json").exists())

    def test_mutated_input_is_not_uploaded_with_stale_provenance(self):
        original_upload = client.upload

        def mutate(a, path, identity):
            path.write_bytes(b"changed after receipt")
            return original_upload(a, path, identity)

        with patch.object(client, "upload", side_effect=mutate), self.assertRaises(SystemExit):
            self.invoke("video", "submit", self.mp4)
        self.assertEqual(self.requests, [])
        self.assertTrue((self.directory / "fixture-generation.json").exists())

    def test_poll_and_fetch_only_read_and_preserve_downloaded_assets(self):
        for input_type in ("image", "multi", "video"):
            for mode in ("poll", "fetch"):
                with self.subTest(input_type=input_type, mode=mode):
                    self.requests.clear()
                    self.download.reset_mock()
                    self.directory = self.root / input_type / mode
                    spz, thumb = self.directory / "output.spz", self.directory / "thumb.png"
                    self.invoke(input_type, mode, "existing-id", "--spz", spz, "--thumb", thumb)
                    self.assertTrue(all(r[0] == "GET" for r in self.requests))
                    self.assertEqual(self.download.call_count, 2)
                    self.assertTrue((self.directory / "fixture-world.json").exists())
                    self.assertEqual(
                        (self.directory / "fixture.spz").read_bytes(), b"\x1f\x8basset"
                    )
                    if mode == "poll":
                        self.assertTrue((self.directory / "fixture-op.json").exists())
                    self.download.reset_mock()
                    self.requests.clear()
                    self.invoke(input_type, mode, "existing-id", "--spz", spz, "--thumb", thumb)
                    self.download.assert_not_called()
                    self.assertTrue(all(r[0] == "GET" for r in self.requests))

    def test_invalid_inputs_never_make_an_api_call(self):
        bad_prompt = self.root / "bad.json"
        bad_prompt.write_text("{bad")
        invalid = [
            ("image", "submit"),
            ("video", "submit", self.root / "missing.mp4"),
            ("multi", "submit"),
            ("multi", "submit", "--images", self.png, self.root / "missing.png"),
            ("multi", "submit", "--images", f"{self.png}:nan"),
            ("multi", "submit", "--images", self.png, "--prompt-file", bad_prompt),
            ("multi", "submit", "--images", *([self.png] * 9)),
            ("image", "poll"),
            ("video", "fetch"),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.invoke(*args)
            self.assertEqual(self.requests, [])
        with patch.object(client, "UPLOAD_LIMIT", 1), self.assertRaises(SystemExit):
            self.invoke("image", "submit", self.png)
        self.assertEqual(self.requests, [])

    def test_failed_operation_persists_status_without_fetch_or_generation(self):
        with patch.object(
            client, "call", return_value=(200, {"done": True, "error": {"message": "failed"}})
        ) as call:
            with self.assertRaises(SystemExit):
                self.invoke("multi", "poll", "failed-op")
            self.assertEqual(
                call.call_args_list[0].args, ("GET", "/marble/v1/operations/failed-op")
            )
            self.assertEqual(call.call_count, 1)
        self.assertTrue((self.directory / "fixture-op.json").exists())

    def test_wrapped_world_is_saved_as_world_and_malformed_response_keeps_metadata(self):
        self.world = {"world": self.world}
        self.invoke("video", "fetch", "existing-id")
        saved = json.loads((self.directory / "fixture-world.json").read_text())
        self.assertEqual(
            saved["assets"]["splats"]["spz_urls"]["full_res"], "https://example.invalid/full.spz"
        )
        before = (self.directory / "fixture-world.json").read_bytes()
        self.world = {"world": {"assets": {"splats": {}}}}
        with self.assertRaises(SystemExit):
            self.invoke("video", "fetch", "existing-id")
        self.assertEqual((self.directory / "fixture-world.json").read_bytes(), before)

    def test_existing_spz_is_retained_but_not_silently_reported_as_downloaded(self):
        self.directory.mkdir()
        spz = self.directory / "partial.spz"
        spz.write_bytes(b"truncated")
        with self.assertRaises(SystemExit):
            self.invoke("video", "fetch", "existing-id", "--spz", spz)
        self.download.assert_not_called()
        self.assertIn(
            "retained; not verified or replaced",
            (self.directory / "fixture-ops.txt").read_text(),
        )
        self.assertEqual(spz.read_bytes(), b"truncated")

    def test_uncertain_submission_survives_restart_and_never_resubmits(self):
        def lose_response(method, path, body=None, **kwargs):
            result = self.call(method, path, body, **kwargs)
            if path.endswith("worlds:generate"):
                raise OSError("response lost after server may have accepted generation")
            return result

        with patch.object(client, "call", side_effect=lose_response), self.assertRaises(SystemExit):
            self.invoke("image", "submit", self.png)
        receipt = json.loads((self.directory / "fixture-generation.json").read_text())
        self.assertEqual(receipt["status"], "submitting")
        self.assertNotIn("operation_id", receipt)
        self.generated()
        self.requests.clear()
        self.png.unlink()  # Recovery must not need the original input or start new uploads.
        with self.assertRaises(SystemExit):
            self.invoke("image", "submit", self.png)
        self.assertEqual(self.requests, [])

    def test_operation_receipt_recovers_when_process_dies_before_logging_id(self):
        original_log = client.log

        def broken_log(args, line):
            if line.startswith("op "):
                raise OSError("process stopped before operation log was written")
            return original_log(args, line)

        with patch.object(client, "log", side_effect=broken_log), self.assertRaises(SystemExit):
            self.invoke("image", "submit", self.png)
        receipt = json.loads((self.directory / "fixture-generation.json").read_text())
        self.assertEqual(receipt["operation_id"], "operation-1")
        self.generated()
        self.requests.clear()
        self.png.unlink()
        self.invoke("image", "submit", self.png)
        self.assertEqual(self.requests[0][:2], ("GET", "/marble/v1/operations/operation-1"))
        self.assertTrue(all(request[0] == "GET" for request in self.requests))

    def test_exclusive_receipt_blocks_a_competing_process_before_upload(self):
        original_upload = client.upload

        def competing_upload(args, path, identity):
            with self.assertRaises(SystemExit):
                self.invoke("video", "submit", self.mp4)
            self.assertEqual(self.requests, [])
            return original_upload(args, path, identity)

        with patch.object(client, "upload", side_effect=competing_upload):
            self.invoke("image", "submit", self.png)
        self.generated()
        args = client.parser().parse_args(
            [
                "image",
                "submit",
                str(self.png),
                "--name",
                "fixture",
                "--marble-dir",
                str(self.directory),
            ]
        )
        before = (self.directory / "fixture-generation.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already reserved"):
            client.claim_submission(args)
        self.assertEqual((self.directory / "fixture-generation.json").read_bytes(), before)

    def test_existing_world_and_corrupt_receipt_block_generation(self):
        self.directory.mkdir()
        world = self.directory / "fixture-world.json"
        world.write_text(json.dumps({"world_id": "existing"}))
        with self.assertRaises(SystemExit):
            self.invoke("image", "submit", self.png)
        world.unlink()
        (self.directory / "fixture-generation.json").write_text("{interrupted")
        with self.assertRaises(SystemExit):
            self.invoke("image", "submit", self.png)
        self.assertEqual(self.requests, [])

    def test_legacy_operation_history_recovers_without_generation(self):
        self.directory.mkdir()
        (self.directory / "fixture-ops.txt").write_text(
            "previous [fixture] op legacy-op submitted (asset old)\n"
        )
        self.invoke("image", "submit", self.root / "missing.png")
        self.assertEqual(self.requests[0][:2], ("GET", "/marble/v1/operations/legacy-op"))
        self.assertTrue(all(request[0] == "GET" for request in self.requests))

    def test_pipeline_unknown_prior_attempt_and_existing_world_are_hard_stops(self):
        pipeline = object.__new__(run_clip.Pipeline)
        pipeline.ctx, pipeline.name = self.root, "fixture"
        pipeline.key, pipeline.certs = "test-placeholder", "test-cert"
        pipeline.a = SimpleNamespace(reuse_world=None, marble_key="WLT_API_KEY")

        def fail_submit(command, log, *args, **kwargs):
            log.write_text("submission failed with no response\n")
            raise RuntimeError("command failed")

        with patch.object(run_clip, "MARBLE_DIR", self.directory):
            with patch.object(run_clip, "run", side_effect=fail_submit) as run:
                with self.assertRaisesRegex(RuntimeError, "charge status is unknown"):
                    pipeline.marble_submit("image", self.png, "image")
                self.assertEqual(run.call_count, 1)
                run.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "charge status is unknown"):
                    pipeline.marble_submit("image", self.png, "image")
                run.assert_not_called()
            self.directory.mkdir()
            (self.directory / "fixture-image-world.json").write_text("{}")
            with patch.object(run_clip, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "use --reuse-world"):
                    pipeline.marble_world("image", self.png, "image")
                run.assert_not_called()

    def test_pipeline_commands_keep_suffixes_and_recover_without_resubmitting(self):
        pipeline = object.__new__(run_clip.Pipeline)
        pipeline.ctx = self.root
        pipeline.name = "fixture"
        pipeline.key = "test-placeholder"
        pipeline.certs = "test-cert"
        pipeline.a = SimpleNamespace(
            poll_attempts=2, reuse_world="existing-world", marble_key="WLT_API_KEY"
        )
        with (
            patch.object(run_clip, "MARBLE_DIR", self.directory),
            patch.object(run_clip, "run") as run,
        ):
            for kind, suffix in [("image", "image"), ("multi", "multi"), ("video", "clean")]:
                pipeline.marble_submit(kind, self.png if kind != "multi" else None, suffix)
                command = run.call_args.args[0]
                self.assertEqual(command[1:4], ["scripts/marble_world.py", kind, "submit"])
                self.assertEqual(run.call_args.kwargs["attempts"], 1)
                self.assertEqual(run.call_args.args[1].name, f"marble_{suffix}.log")
                pipeline.marble_reuse(kind, suffix)
                self.assertEqual(
                    run.call_args.args[0][1:5],
                    ["scripts/marble_world.py", kind, "fetch", "existing-world"],
                )
            log = self.root / "marble_image.log"
            log.write_text("op saved-op submitted (asset asset-1)\n")
            run.side_effect = RuntimeError("poll failed after submit")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.marble_submit("image", self.png, "image"), "saved-op")
            run.reset_mock()
            run.side_effect = [RuntimeError("transient"), None]
            with patch.object(run_clip.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                pipeline.marble_poll("image", "saved-op", "image")
            self.assertEqual(len(run.call_args_list), 2)
            self.assertTrue(all(c.args[0][3:5] == ["poll", "saved-op"] for c in run.call_args_list))
        with self.assertRaisesRegex(RuntimeError, "refusing to retry"):
            run_clip.run(
                ["python", "scripts/marble_world.py", "image", "submit"],
                self.root / "blocked.log",
                attempts=2,
            )


class MarbleTransport(unittest.TestCase):
    def request_headers(self, path, **kwargs):
        response = io.BytesIO(b"{}")
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        with (
            patch.dict(client.os.environ, {"WLT_API_KEY": "test-only-credential"}),
            patch.object(client.urllib.request, "urlopen", return_value=response) as opened,
        ):
            client.call("PUT", path, raw=b"media", **kwargs)
        request = opened.call_args.args[0]
        return {k.lower(): v for k, v in request.header_items()}

    def test_api_request_authenticates(self):
        headers = self.request_headers("/marble/v1/credits")
        self.assertEqual(headers["wlt-api-key"], "test-only-credential")

    def test_api_credential_is_not_forwarded_by_redirects(self):
        response = io.BytesIO(b"{}")
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        with (
            patch.dict(client.os.environ, {"WLT_API_KEY": "test-only-credential"}),
            patch.object(client.urllib.request, "urlopen", return_value=response) as opened,
        ):
            client.call("GET", "/marble/v1/credits")
        redirected = client.urllib.request.HTTPRedirectHandler().redirect_request(
            opened.call_args.args[0], None, 302, "redirect", {}, "https://example.invalid/next"
        )
        self.assertNotIn("wlt-api-key", {k.lower() for k, _ in redirected.header_items()})

    def test_presigned_upload_never_receives_api_credential(self):
        for host in ("storage.googleapis.com", "api.worldlabs.ai.example.invalid"):
            with self.subTest(host=host):
                headers = self.request_headers(
                    f"https://{host}/upload?signature=test",
                    headers={"Content-Type": "image/png", "wlt-api-key": "also-remove-this"},
                )
                self.assertNotIn("wlt-api-key", headers)
                self.assertEqual(headers["content-type"], "image/png")

    def test_http_and_url_errors_never_expose_query_credentials_or_provider_body(self):
        secret = "signed-secret-should-never-appear"
        http = urllib.error.HTTPError(
            f"https://storage.example.invalid/path/file?X-Goog-Signature={secret}",
            403,
            "forbidden",
            {},
            io.BytesIO(f"provider body {secret}".encode()),
        )
        with (
            patch.object(client.urllib.request, "urlopen", side_effect=http),
            self.assertRaises(SystemExit) as raised,
        ):
            client.call("GET", "https://storage.example.invalid/path/file?token=" + secret)
        text = str(raised.exception)
        self.assertIn("HTTP 403", text)
        self.assertIn("storage.example.invalid/path/file", text)
        self.assertNotIn(secret, text)
        with (
            patch.object(
                client.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("credential=" + secret),
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            client.call("GET", "https://storage.example.invalid/path/file?token=" + secret)
        self.assertIn("network error", str(raised.exception))
        self.assertNotIn(secret, str(raised.exception))

    def test_download_is_atomic_and_enforces_length_and_spz_header(self):
        destination = Path(tempfile.mkdtemp()) / "world.spz"
        bad = io.BytesIO(b"not-gzip")
        bad.headers = {"Content-Length": "8"}
        with (
            patch.object(client.urllib.request, "urlopen", return_value=bad),
            self.assertRaisesRegex(ValueError, "gzip header"),
        ):
            client.download("https://storage.example.invalid/object?secret=no", destination, True)
        self.assertFalse(destination.exists())
        short = io.BytesIO(b"\x1f\x8bshort")
        short.headers = {"Content-Length": "999"}
        with (
            patch.object(client.urllib.request, "urlopen", return_value=short),
            self.assertRaisesRegex(ValueError, "length mismatch"),
        ):
            client.download("https://storage.example.invalid/object", destination, True)
        self.assertFalse(destination.exists())

    def test_download_urlerror_is_secret_safe(self):
        secret = "download-signed-secret"
        destination = Path(tempfile.mkdtemp()) / "world.spz"
        with (
            patch.object(
                client.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("signature=" + secret),
            ),
            self.assertRaisesRegex(ValueError, "network error") as raised,
        ):
            client.download(
                "https://storage.example.invalid/object?signature=" + secret, destination
            )
        self.assertNotIn(secret, str(raised.exception))
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
