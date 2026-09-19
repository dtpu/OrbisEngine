"""Offline contracts for Marble requests and recovery. Never uses credentials or the network."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
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

    def save(self, url, destination):
        p = Path(destination)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"asset")
        return 5

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

    def test_video_payload_does_not_gain_image_only_fields(self):
        self.invoke("video", "submit", self.mp4)
        self.assertEqual(
            self.generated(),
            {
                "world_prompt": {
                    "type": "video",
                    "video_prompt": {"source": "media_asset", "media_asset_id": "asset-1"},
                }
            },
        )
        self.assertEqual(
            self.requests[0][2], {"file_name": "clip.mp4", "kind": "video", "extension": "mp4"}
        )
        self.assertEqual(self.requests[1][3]["headers"]["Content-Type"], "video/mp4")
        text = (self.directory / "fixture-ops.txt").read_text()
        self.assertNotIn("[fixture]", text)
        self.assertIn("world world-1 https://example.invalid/world metric_scale_factor", text)

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
                    self.assertEqual((self.directory / "fixture.spz").read_bytes(), b"asset")
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

        def competing_upload(args, path):
            with self.assertRaises(SystemExit):
                self.invoke("video", "submit", self.mp4)
            self.assertEqual(self.requests, [])
            return original_upload(args, path)

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


if __name__ == "__main__":
    unittest.main()
