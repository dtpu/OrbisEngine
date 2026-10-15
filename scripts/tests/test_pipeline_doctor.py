"""No-spend setup diagnostics: configuration is never proof of inference readiness."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import doctor


class DoctorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = {
            "HOME": str(self.root),
            "WANDER_RUNS_ROOT": str(self.root / "runs"),
            "MODAL_TOKEN_ID": "private-modal-id",
            "MODAL_TOKEN_SECRET": "private-modal-secret",
            "OPENAI_API_KEY": "private-openai-key",
            "WLT_API_KEY": "private-world-key",
        }
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"fixture")
        self.metadata = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": "30000/1001",
                    "duration": "4",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "4"},
        }
        self.remote_result = "present"
        self.cuts = 0
        self.import_failures = set()
        self.probes = []
        self.enterContext(patch.dict(os.environ, self.environment, clear=True))
        self.enterContext(
            patch.object(doctor.shutil, "which", side_effect=lambda name: f"/bin/{name}")
        )
        self.enterContext(patch.object(doctor, "_probe", side_effect=self.probe))

    def probe(self, argv, **kwargs):
        self.probes.append((argv, kwargs))
        if argv[-1] in ("-version", "--version"):
            return "version", 0
        if argv[0] == "ffprobe":
            return json.dumps(self.metadata), 0
        if "_module_inventory" in " ".join(argv):
            return json.dumps(
                {name: name not in self.import_failures for name in doctor.MODULES}
            ), 0
        if "_modal_inventory" in " ".join(argv):
            paths = json.loads(argv[-1])
            return json.dumps(
                {
                    f"{volume}:{path}": self.remote_result
                    for volume, names in paths.items()
                    for path in names
                }
            ), 0
        if any(value.endswith("shot_cuts.py") for value in argv):
            Path(argv[argv.index("--json") + 1]).write_text(
                json.dumps(
                    {
                        "continuous": self.cuts == 0,
                        "cutCount": self.cuts,
                        "cuts": [{} for _ in range(self.cuts)],
                    }
                )
            )
            return "", int(self.cuts > 0)
        if argv[0] == "ssh":
            return "", 0
        raise AssertionError(f"Unexpected diagnostic command: {argv}")

    def check(self, report, name):
        return next(value for value in report["checks"] if value["name"] == name)

    def test_local_configuration_never_claims_inference_ready_or_leaks_secrets(self):
        report = doctor.inspect_setup(clip=self.clip)
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"], "unverified")
        self.assertEqual(self.check(report, "modal.auth")["status"], "unverified")
        self.assertTrue(
            all(
                check["status"] == "unverified"
                for check in report["checks"]
                if check["name"].startswith("cache.")
            )
        )
        serialized = json.dumps(report)
        for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "OPENAI_API_KEY", "WLT_API_KEY"):
            self.assertNotIn(os.environ[name], serialized)
        self.assertFalse(any("_modal_inventory" in " ".join(argv) for argv, _ in self.probes))
        self.assertEqual(report["runsRoot"], str((self.root / "runs").resolve()))

    def test_missing_inference_dependency_is_actionable_blocker(self):
        self.import_failures = {"torch"}
        report = doctor.inspect_setup()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(self.check(report, "python.torch")["status"], "blocked")
        self.assertIn("--group inference", self.check(report, "python.torch")["action"])

    def test_missing_tool_and_invalid_harness_block(self):
        with patch.object(doctor.shutil, "which", return_value=None):
            report = doctor.inspect_setup()
        self.assertEqual(self.check(report, "tool.ffmpeg")["status"], "blocked")
        self.assertEqual(self.check(report, "tool.harness")["status"], "blocked")
        os.environ["WANDER_REVIEW_AGENT"] = "unsupported"
        report = doctor.inspect_setup()
        self.assertEqual(self.check(report, "tool.harness")["status"], "blocked")

    def test_failing_tool_version_probe_blocks(self):
        def fail_version(argv, **kwargs):
            if argv[-1] == "-version":
                raise doctor.ProbeError("unavailable")
            return self.probe(argv, **kwargs)

        with patch.object(doctor, "_probe", side_effect=fail_version):
            report = doctor.inspect_setup()
        self.assertEqual(self.check(report, "tool.ffmpeg")["status"], "blocked")
        self.assertTrue(self.check(report, "tool.ffmpeg")["action"])

    def test_builtin_harness_missing_executable_blocks_without_probe(self):
        for kind in ("codex", "claude"):
            with self.subTest(kind=kind):
                os.environ["WANDER_REVIEW_AGENT"] = kind
                self.probes.clear()
                with patch.object(
                    doctor.shutil,
                    "which",
                    side_effect=lambda name: None if name == kind else f"/bin/{name}",
                ):
                    report = doctor.inspect_setup()
                self.assertEqual(self.check(report, "tool.harness")["status"], "blocked")
                self.assertFalse(any(kind in argv[0] for argv, _ in self.probes))

    def test_builtin_harness_startup_success_leaves_access_unverified(self):
        for kind in ("codex", "claude"):
            with self.subTest(kind=kind):
                os.environ["WANDER_REVIEW_AGENT"] = kind
                self.probes.clear()

                def version_output(argv, **kwargs):
                    result = self.probe(argv, **kwargs)
                    return ("private-version-output", 0) if argv[0] == f"/bin/{kind}" else result

                with patch.object(doctor, "_probe", side_effect=version_output):
                    report = doctor.inspect_setup()
                self.assertEqual(self.check(report, "tool.harness")["status"], "okay")
                self.assertEqual(self.check(report, "harness.access")["status"], "unverified")
                harness_probes = [
                    (argv, kwargs) for argv, kwargs in self.probes if argv[0] == f"/bin/{kind}"
                ]
                self.assertEqual(harness_probes, [([f"/bin/{kind}", "--version"], {"timeout": 5})])
                self.assertNotIn("private-version-output", json.dumps(report))

    def test_builtin_harness_broken_wrapper_or_timeout_blocks(self):
        for kind in ("codex", "claude"):
            for failure in ("probe exited 1", "probe exceeded 5s"):
                with self.subTest(kind=kind, failure=failure):
                    os.environ["WANDER_REVIEW_AGENT"] = kind

                    def fail_harness(argv, **kwargs):
                        if argv[0] == f"/bin/{kind}":
                            self.assertEqual(argv, [f"/bin/{kind}", "--version"])
                            self.assertEqual(kwargs, {"timeout": 5})
                            raise doctor.ProbeError(failure)
                        return self.probe(argv, **kwargs)

                    with patch.object(doctor, "_probe", side_effect=fail_harness):
                        report = doctor.inspect_setup()
                    harness = self.check(report, "tool.harness")
                    self.assertEqual(harness["status"], "blocked")
                    self.assertIn("repair", harness["action"])
                    self.assertFalse(report["ready"])

    def test_custom_harness_only_checks_path_and_never_executes_command(self):
        # Even a command naming a built-in harness must not be probed with appended flags.
        for command in ("custom-harness --submit-work", "codex exec --submit-work"):
            with self.subTest(command=command):
                os.environ["WANDER_REVIEW_COMMAND"] = command
                self.probes.clear()
                report = doctor.inspect_setup()
                harness = self.check(report, "tool.harness")
                self.assertEqual(harness["status"], "unverified")
                self.assertIn("executable was found", harness["detail"])
                self.assertIn("startup was not tested", harness["detail"])
                self.assertEqual(self.check(report, "harness.access")["status"], "unverified")
                self.assertFalse(
                    any(argv[0] == f"/bin/{command.split()[0]}" for argv, _ in self.probes)
                )

    def test_custom_harness_missing_executable_blocks_without_execution(self):
        os.environ["WANDER_REVIEW_COMMAND"] = "missing-custom --submit-work"
        with patch.object(
            doctor.shutil,
            "which",
            side_effect=lambda name: None if name == "missing-custom" else f"/bin/{name}",
        ):
            report = doctor.inspect_setup()
        self.assertEqual(self.check(report, "tool.harness")["status"], "blocked")
        self.assertFalse(any("missing-custom" in argv[0] for argv, _ in self.probes))

    def test_remote_timeout_cannot_promote_configured_credentials(self):
        def fail_remote(argv, **kwargs):
            if "_modal_inventory" in " ".join(argv):
                raise doctor.ProbeError("probe exceeded deadline")
            return self.probe(argv, **kwargs)

        with patch.object(doctor, "_probe", side_effect=fail_remote):
            report = doctor.inspect_setup(remote=True)
        self.assertFalse(report["ready"])
        self.assertEqual(self.check(report, "modal.remote")["status"], "unverified")
        self.assertEqual(self.check(report, "modal.auth")["status"], "unverified")

    def test_malformed_continuity_report_blocks(self):
        def corrupt_continuity(argv, **kwargs):
            output = self.probe(argv, **kwargs)
            if any(value.endswith("shot_cuts.py") for value in argv):
                Path(argv[argv.index("--json") + 1]).write_text(
                    '{"continuous":true,"cutCount":1,"cuts":[{}]}'
                )
            return output

        with patch.object(doctor, "_probe", side_effect=corrupt_continuity):
            report = doctor.inspect_setup(clip=self.clip)
        self.assertEqual(self.check(report, "clip.continuity")["status"], "blocked")

    def test_credentials_and_caches_follow_selected_plan(self):
        os.environ.pop("OPENAI_API_KEY")
        os.environ.pop("WLT_API_KEY")
        report = doctor.inspect_setup({"marble": "none", "objects": False})
        names = [check["name"] for check in report["checks"]]
        self.assertNotIn("credential.OPENAI_API_KEY", names)
        self.assertNotIn("credential.WLT_API_KEY", names)
        self.assertFalse(any("image-to-3d-cache" in name for name in names))
        # The no-world graph still cleans its video, so its clean cache remains required.
        self.assertTrue(any("clean-video-cache" in name for name in names))
        self.assertNotIn("finetune.host", names)
        self.assertEqual(report["status"], "unverified")
        image = doctor.inspect_setup({"marble": "image", "objects": False})
        self.assertEqual(self.check(image, "credential.OPENAI_API_KEY")["status"], "blocked")
        self.assertEqual(self.check(image, "credential.WLT_API_KEY")["status"], "blocked")
        objects = doctor.inspect_setup({"marble": "none", "objects": True})
        self.assertEqual(self.check(objects, "credential.OPENAI_API_KEY")["status"], "blocked")
        self.assertTrue(any("image-to-3d-cache" in check["name"] for check in objects["checks"]))

    def test_selected_modal_profile_must_contain_complete_auth(self):
        os.environ.pop("MODAL_TOKEN_ID")
        os.environ.pop("MODAL_TOKEN_SECRET")
        config = self.root / "modal.toml"
        os.environ["MODAL_CONFIG_PATH"] = str(config)
        os.environ["MODAL_PROFILE"] = "reviewed"
        config.write_text('[other]\ntoken_id="private"\ntoken_secret="private"\n')
        report = doctor.inspect_setup(remote=True)
        self.assertEqual(self.check(report, "modal.auth")["status"], "blocked")
        self.assertFalse(any("_modal_inventory" in " ".join(argv) for argv, _ in self.probes))
        config.write_text('[reviewed]\ntoken_id="private"\ntoken_secret="private"\n')
        report = doctor.inspect_setup()
        self.assertEqual(self.check(report, "modal.auth")["status"], "unverified")

    def test_remote_missing_cache_blocks_but_permission_failure_is_unverified(self):
        self.remote_result = "missing"
        report = doctor.inspect_setup(remote=True)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(self.check(report, "modal.auth")["status"], "okay")
        self.assertTrue(
            any(
                check["name"].startswith("cache.") and check["status"] == "blocked"
                for check in report["checks"]
            )
        )
        self.remote_result = "unverified"
        report = doctor.inspect_setup(remote=True)
        self.assertEqual(report["status"], "unverified")
        self.assertEqual(self.check(report, "modal.auth")["status"], "unverified")

    def test_remote_probes_are_bounded_and_never_launch_workers(self):
        report = doctor.inspect_setup(remote=True)
        self.assertEqual(self.check(report, "modal.auth")["status"], "okay")
        remote = [
            (argv, kwargs) for argv, kwargs in self.probes if "_modal_inventory" in " ".join(argv)
        ]
        self.assertEqual(len(remote), 1)
        self.assertEqual(remote[0][1]["env"]["MODAL_PROFILE"], "dtpu")
        for argv, _ in self.probes:
            self.assertNotIn("modal run", " ".join(argv))
            self.assertNotIn("modal_stage_weights.py", " ".join(argv))
        self.assertFalse(report["ready"])

    def test_finetune_requires_host_and_only_probes_selected_host(self):
        report = doctor.inspect_setup({"finetune": True}, remote=True)
        self.assertEqual(self.check(report, "finetune.host")["status"], "blocked")
        os.environ["WANDER_GPU_BOX"] = "user@reviewed-host"
        report = doctor.inspect_setup({"finetune": True}, remote=True)
        self.assertEqual(self.check(report, "finetune.host")["status"], "okay")
        ssh = [argv for argv, _ in self.probes if argv[0] == "ssh"]
        self.assertEqual(len(ssh), 1)
        self.assertIn("BatchMode=yes", ssh[0])
        self.assertIn("StrictHostKeyChecking=yes", ssh[0])
        self.assertEqual(ssh[0][-1], "true")
        self.probes.clear()
        doctor.inspect_setup({"finetune": False}, remote=True)
        self.assertFalse(any(argv[0] == "ssh" for argv, _ in self.probes))

    def test_clip_metadata_and_cuts_are_checked_without_model_scoring(self):
        report = doctor.inspect_setup(clip=self.clip)
        self.assertEqual(report["clip"]["audioCodecs"], ["aac"])
        self.assertAlmostEqual(report["clip"]["fps"], 30000 / 1001)
        self.assertEqual(self.check(report, "clip.continuity")["status"], "okay")
        command, kwargs = next(
            (argv, kwargs)
            for argv, kwargs in self.probes
            if any(value.endswith("shot_cuts.py") for value in argv)
        )
        self.assertIn("--no-score", command)
        self.assertIn("--no-people", command)
        self.assertEqual(kwargs["timeout"], doctor.CLIP_TIMEOUT)
        self.cuts = 1
        report = doctor.inspect_setup(clip=self.clip)
        self.assertEqual(self.check(report, "clip.continuity")["status"], "blocked")

    def test_bad_clip_metadata_and_missing_clip_block(self):
        self.metadata["streams"][0]["avg_frame_rate"] = "0/0"
        report = doctor.inspect_setup(clip=self.clip)
        self.assertEqual(self.check(report, "clip.metadata")["status"], "blocked")
        report = doctor.inspect_setup(clip=self.root / "missing.mp4")
        self.assertEqual(self.check(report, "clip.metadata")["status"], "blocked")

    def test_runs_root_cannot_be_a_file(self):
        root = self.root / "runs"
        root.write_text("conflicting file")
        report = doctor.inspect_setup()
        self.assertEqual(self.check(report, "runs.root")["status"], "blocked")

    def test_cli_invalid_options_are_json_and_nonzero(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = doctor.main(["--json", "--options", '{"unsupported":"private-input"}'])
        self.assertEqual(code, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["checks"][0]["name"], "options")
        self.assertNotIn("private-input", output.getvalue())


class ProbeTests(unittest.TestCase):
    def test_subprocess_timeout_is_bounded(self):
        with self.assertRaisesRegex(doctor.ProbeError, "exceeded"):
            doctor._probe([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.05)

    def test_failed_probe_never_returns_private_stderr(self):
        with self.assertRaises(doctor.ProbeError) as caught:
            doctor._probe(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('private-stdout'); sys.stderr.write('private-token'); sys.exit(2)",
                ]
            )
        self.assertNotIn("private-token", str(caught.exception))
        self.assertNotIn("private-stdout", str(caught.exception))

    def test_remote_inventory_uses_existing_volumes_and_rejects_partial_model(self):
        from modal.types import FileEntryType

        entries = [
            SimpleNamespace(
                path="models/snapshots/rev/model.safetensors.incomplete",
                size=5,
                type=FileEntryType.FILE,
            ),
            SimpleNamespace(
                path="models/snapshots/rev/config.json", size=5, type=FileEntryType.FILE
            ),
        ]
        volume = SimpleNamespace(iterdir=lambda *args, **kwargs: iter(entries))
        with patch("modal.Volume.from_name", return_value=volume) as from_name:
            result = doctor._modal_inventory({"existing": [doctor.PI3X]})
        from_name.assert_called_once_with("existing", create_if_missing=False)
        self.assertEqual(result[f"existing:{doctor.PI3X}"], "missing")

    def test_remote_inventory_accepts_nonempty_regular_snapshot_pair(self):
        from modal.types import FileEntryType

        entries = [
            SimpleNamespace(
                path=f"{doctor.PI3X}/snapshots/rev/model.safetensors",
                size=5,
                type=FileEntryType.FILE,
            ),
            SimpleNamespace(
                path=f"{doctor.PI3X}/snapshots/rev/config.json",
                size=5,
                type=FileEntryType.FILE,
            ),
        ]
        volume = SimpleNamespace(iterdir=lambda *args, **kwargs: iter(entries))
        with patch("modal.Volume.from_name", return_value=volume):
            result = doctor._modal_inventory({"existing": [doctor.PI3X]})
        self.assertEqual(result[f"existing:{doctor.PI3X}"], "present")

    def test_remote_inventory_marks_snapshot_symlinks_unverified(self):
        from modal.types import FileEntryType

        entries = [
            SimpleNamespace(
                path=f"{doctor.PI3X}/snapshots/rev/model.safetensors",
                size=0,
                type=FileEntryType.SYMLINK,
            ),
            SimpleNamespace(
                path=f"{doctor.PI3X}/snapshots/rev/config.json",
                size=0,
                type=FileEntryType.SYMLINK,
            ),
        ]
        volume = SimpleNamespace(iterdir=lambda *args, **kwargs: iter(entries))
        with patch("modal.Volume.from_name", return_value=volume):
            result = doctor._modal_inventory({"existing": [doctor.PI3X]})
        self.assertEqual(result[f"existing:{doctor.PI3X}"], "unverified")

    def test_remote_inventory_does_not_call_truncated_listing_missing(self):
        from modal.types import FileEntryType

        entries = [
            SimpleNamespace(path=f"{doctor.PI3X}/refs/{index}", size=1, type=FileEntryType.FILE)
            for index in range(2000)
        ]
        volume = SimpleNamespace(iterdir=lambda *args, **kwargs: iter(entries))
        with patch("modal.Volume.from_name", return_value=volume):
            result = doctor._modal_inventory({"existing": [doctor.PI3X]})
        self.assertEqual(result[f"existing:{doctor.PI3X}"], "unverified")


if __name__ == "__main__":
    unittest.main()
