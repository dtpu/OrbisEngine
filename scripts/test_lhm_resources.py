"""Offline resource selection and pre-submission validation; no provider calls."""

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage_attempts import command_identity
from worker import modal_lhm
from worker.stages.lhm_resources import compute_rate, execution_options

ROOT = Path(__file__).resolve().parents[1]


class ResourceTests(unittest.TestCase):
    def test_bounds_and_rates(self):
        self.assertEqual(execution_options()["timeout"], 1800)
        for gpu, rate in (("L4", 0.000222), ("H100", 0.001097)):
            for timeout in (1, 1800):
                options = execution_options(gpu, timeout)
                self.assertEqual(options["gpu"], gpu)
                self.assertEqual(options["timeout"], timeout)
                self.assertEqual(options["cpu"], (4, 4))
                self.assertEqual(options["memory"], (65536, 65536))
                self.assertEqual(options["retries"], 0)
            self.assertAlmostEqual(compute_rate(gpu), rate + 4 * 0.0000131 + 64 * 0.00000222)
        for gpu in ("A100", "h100", "H100:2", ""):
            with self.assertRaises(ValueError):
                execution_options(gpu)
        for timeout in (0, -1, 1801, True, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                execution_options(execution_timeout=timeout)

    def test_resource_options_match_installed_modal_sdk(self):
        # Bind the actual SDK signature: a permissive Mock would miss unsupported keys.
        signature = inspect.signature(modal_lhm.frozen.with_options)
        for gpu in ("L4", "H100"):
            signature.bind(**execution_options(gpu, 120))

    def test_invalid_cli_cannot_stage_create_receipt_or_submit(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "out"
            with (
                patch.object(modal_lhm, "stage") as stage,
                patch.object(modal_lhm, "frozen") as frozen,
            ):
                for kwargs in (
                    {"gpu": "A100"},
                    {"execution_timeout": 0},
                    {"execution_timeout": 1801},
                ):
                    with self.assertRaises(ValueError):
                        modal_lhm.main(out=str(destination), **kwargs)
                    with self.assertRaises(ValueError):
                        modal_lhm.main(prepared="missing", out=str(destination), **kwargs)
                stage.remote.assert_not_called()
                frozen.with_options.assert_not_called()
                frozen.spawn.assert_not_called()
            self.assertFalse(destination.exists())

    def test_h100_options_and_receipt_survive_interrupted_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            for name in ("source.png", "mask.png", "prepared.json"):
                (prepared / name).write_bytes(b"private input")
            function = Mock()
            call = function.with_options.return_value.spawn.return_value
            call.object_id = "fc-offline"
            call.get.side_effect = KeyboardInterrupt
            destination = root / "out"
            with patch.object(modal_lhm, "frozen", function), patch("builtins.print"):
                with self.assertRaises(KeyboardInterrupt):
                    modal_lhm.main(
                        prepared=str(prepared),
                        out=str(destination),
                        gpu="H100",
                        execution_timeout=120,
                    )
                options = function.with_options.call_args.kwargs
                self.assertEqual(options, execution_options("H100", 120))
                submitted = function.with_options.return_value.spawn.call_args.kwargs
                self.assertEqual(submitted["gpu"], "H100")
                self.assertEqual(submitted["execution_timeout"], 120)
                receipt = json.loads((destination / "recovery-receipt.json").read_text())
                self.assertEqual(receipt["execution"]["gpu"], "H100")
                self.assertEqual(receipt["execution"]["timeoutSeconds"], 120)
                self.assertEqual(receipt["functionCallId"], "fc-offline")
                with self.assertRaises(FileExistsError):
                    modal_lhm.main(
                        prepared=str(prepared),
                        out=str(destination),
                        gpu="H100",
                        execution_timeout=120,
                    )
                function.with_options.return_value.spawn.assert_called_once()

    def test_paid_identity_binds_gpu_and_native_timeout_bounds(self):
        command = [
            "modal",
            "run",
            "worker/modal_lhm.py",
            "--gpu",
            "H100",
            "--execution-timeout",
            "120",
        ]
        parameters, code, _ = command_identity(command, ROOT)
        self.assertEqual(
            command_identity(command, ROOT),
            command_identity([*command[:2], "--detach", *command[2:]], ROOT),
        )
        self.assertEqual(parameters["options"]["--gpu"], "H100")
        self.assertIn("worker/stages/lhm_resources.py", code)
        other, _, _ = command_identity([*command[:4], "L4", *command[5:]], ROOT)
        self.assertNotEqual(parameters, other)
        for gpu in ("A100", "H100:2"):
            with self.assertRaises(ValueError):
                command_identity([*command[:4], gpu, *command[5:]], ROOT)
        for timeout in ("0", "1801", "1.2"):
            with self.assertRaises(ValueError):
                command_identity([*command[:-1], timeout], ROOT)


if __name__ == "__main__":
    unittest.main()
