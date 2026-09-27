"""No-spend checks for durable paid-stage accounting and runner integration."""

import sys
from pathlib import Path

# The modules under test are this directory's parent; importing them by name is what
# running from scripts/ used to give for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_clip
from stage_attempts import CODE_FILES, StageAttempts, command_identity

SOURCE = "a" * 64


class StageAttemptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = StageAttempts(self.root / "attempts.json")
        self.log = self.root / "worker.log"
        self.log.write_text("saved provider evidence\n")

    def begin(self, operation="pi3x", number=0, hypothesis=None, candidate="candidate-a"):
        return self.ledger.begin(
            SOURCE,
            operation,
            parameters={"setting": number},
            code_version={"worker.py": "code-v1"},
            hypothesis=hypothesis,
            candidate=candidate,
            log=self.log,
            results=[self.root / candidate / "result"],
        )

    def finish(self, attempt, status="failed"):
        self.ledger.finish(attempt["id"], status=status, evidence=[self.log])

    # 698d16d removed the ledger's gates (the three-execution cap, blocking on pending/unknown,
    # the required new hypothesis and the changed-fingerprint rule): it records paid executions
    # and no longer decides whether they may happen. These tests hold it to that.
    def test_inpainting_aliases_share_one_sequence_with_no_cap(self):
        numbers = []
        for number, operation in enumerate(("clean", "clean_first", "clean", "clean_multi"), 1):
            attempt = self.begin(operation, number, candidate=f"candidate-{number}")
            self.finish(attempt)
            numbers.append(attempt["claim"]["number"])
            self.assertEqual(attempt["claim"]["stage"], "inpainting")
            self.assertEqual(attempt["claim"]["operation"], operation)
        self.assertEqual(numbers, [1, 2, 3, 4], "a fourth execution is ordinary, not refused")
        # The saved ledger must still validate with four claims on one stage.
        self.assertEqual(self.begin("clean", 5)["claim"]["number"], 5)

    def test_pending_and_unknown_are_recorded_and_reconcilable_but_do_not_block(self):
        pending = self.begin(number=1)
        retry = self.begin(number=2, hypothesis="Change overlap")
        self.assertEqual(retry["claim"]["number"], 2)
        evidence = self.root / "provider-result.json"
        evidence.write_text('{"observed":"failed"}')
        self.ledger.reconcile(
            pending["id"], status="failed", evidence=evidence, reason="provider showed failure"
        )
        self.ledger.finish(retry["id"], status="unknown", evidence=[self.log])
        third = self.begin(number=3, hypothesis="Change batch size")
        self.assertEqual(third["claim"]["number"], 3)
        saved = {
            item["id"]: [event["status"] for event in item["events"]]
            for item in json.loads(self.ledger.path.read_text())["attempts"]
        }
        self.assertEqual(saved[pending["id"]][0], "pending")
        self.assertEqual(saved[pending["id"]][-1], "failed")
        self.assertEqual(saved[retry["id"]], ["pending", "unknown"])
        self.assertEqual(saved[third["id"]], ["pending"])

    def test_retries_record_their_hypothesis_and_identity_without_gating(self):
        first = self.begin(number=1)
        self.finish(first)
        self.assertFalse(first["claim"]["hypothesisExplicit"])
        self.assertIn("Workflow: pi3x", first["claim"]["hypothesis"])
        # Unchanged parameters and code under a new candidate name are recorded, not refused;
        # the identical fingerprint is what lets a reviewer see it was a repeat.
        repeat = self.begin(number=1, candidate="renamed")
        self.finish(repeat)
        self.assertEqual(repeat["claim"]["fingerprint"], first["claim"]["fingerprint"])
        self.assertEqual(repeat["claim"]["candidate"], "renamed")
        changed = self.begin(number=2, hypothesis="  A real new idea ")
        self.assertTrue(changed["claim"]["hypothesisExplicit"])
        self.assertEqual(changed["claim"]["hypothesis"], "A real new idea")
        self.assertNotEqual(changed["claim"]["fingerprint"], first["claim"]["fingerprint"])
        self.assertEqual(changed["claim"]["number"], 3)

    def test_concurrent_claims_are_both_recorded_with_distinct_numbers(self):
        barrier = threading.Barrier(2)
        attempts = []
        errors = []

        def claim(candidate):
            barrier.wait()
            try:
                attempts.append(self.begin(number=1, candidate=candidate))
            except ValueError as error:
                errors.append(str(error))

        workers = [
            threading.Thread(target=claim, args=(f"candidate-{index}",)) for index in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        self.assertEqual(errors, [])
        self.assertEqual(sorted(item["claim"]["number"] for item in attempts), [1, 2])
        saved = self.ledger.path.read_text()
        self.assertEqual(saved.count('"status": "pending"'), 2)
        # The lock must leave a ledger that still validates.
        self.assertEqual(self.begin(number=2)["claim"]["number"], 3)

    def test_concurrent_original_and_alias_claim_share_one_owner(self):
        alias = "b" * 64
        self.ledger.path.write_text(
            json.dumps(
                {
                    "schema": "wander.pipeline-attempts/1",
                    "sourceAliases": {alias: SOURCE},
                    "attempts": [],
                }
            )
        )
        barrier = threading.Barrier(2)
        results, errors = [], []

        def claim(source):
            barrier.wait(timeout=5)
            try:
                results.append(
                    self.ledger.begin(
                        source,
                        "pi3x",
                        parameters={"setting": 1},
                        code_version={"worker.py": "v1"},
                        hypothesis=None,
                        candidate=source[:4],
                        log=self.log,
                        results=[],
                    )
                )
            except ValueError as error:
                errors.append(str(error))

        threads = [threading.Thread(target=claim, args=(source,)) for source in (SOURCE, alias)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual({item["claim"]["sourceSha256"] for item in results}, {SOURCE})
        self.assertEqual(
            {item["claim"]["submittedSourceSha256"] for item in results}, {SOURCE, alias}
        )
        self.assertEqual(sorted(item["claim"]["number"] for item in results), [1, 2])

    def test_source_alias_counts_against_the_original_sequence(self):
        alias = "b" * 64
        self.ledger.path.write_text(
            json.dumps(
                {
                    "schema": "wander.pipeline-attempts/1",
                    "sourceAliases": {alias: SOURCE},
                    "attempts": [],
                }
            )
        )
        pending = self.ledger.begin(
            alias,
            "pi3x",
            parameters={"setting": 1},
            code_version={"worker.py": "v1"},
            hypothesis=None,
            candidate="encoded-alias",
            log=self.log,
            results=[],
        )
        self.assertEqual(pending["claim"]["sourceSha256"], SOURCE)
        self.assertEqual(pending["claim"]["submittedSourceSha256"], alias)
        second = self.begin(number=2, hypothesis="Changed overlap")
        self.assertEqual(second["claim"]["number"], 2)
        self.finish(pending)
        self.finish(second)
        third = self.ledger.begin(
            alias,
            "pi3x",
            parameters={"setting": 3},
            code_version={"worker.py": "v1"},
            hypothesis="Changed alignment",
            candidate="alias-again",
            log=self.log,
            results=[],
        )
        self.assertEqual(third["claim"]["number"], 3)
        self.finish(third)
        self.assertEqual(self.begin(number=4, hypothesis="Changed model")["claim"]["number"], 4)

    def test_alias_mapping_cannot_be_added_after_alias_hash_owns_claims(self):
        alias = "b" * 64
        first = self.ledger.begin(
            alias,
            "pi3x",
            parameters={"setting": 1},
            code_version={"worker.py": "v1"},
            hypothesis=None,
            candidate="old",
            log=self.log,
            results=[],
        )
        self.ledger.finish(first["id"], status="failed", evidence=[self.log])
        saved = json.loads(self.ledger.path.read_text())
        saved["sourceAliases"] = {alias: SOURCE}
        self.ledger.path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "already owns"):
            self.begin(number=2, hypothesis="new")

    def test_source_aliases_reject_self_maps_chains_and_cycles(self):
        alias = "b" * 64
        third = "c" * 64
        for aliases in (
            {alias: alias},
            {alias: SOURCE, SOURCE: third},
            {alias: SOURCE, SOURCE: alias},
        ):
            with self.subTest(aliases=aliases):
                self.ledger.path.write_text(
                    json.dumps(
                        {
                            "schema": "wander.pipeline-attempts/1",
                            "sourceAliases": aliases,
                            "attempts": [],
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, "source aliases"):
                    self.begin()


class CommandIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        repository = Path(__file__).resolve().parents[2]
        for entry, dependencies in CODE_FILES.items():
            for relative in (entry.split("::")[0], *dependencies):
                destination = self.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((repository / relative).read_bytes())
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.write("clip.mov", "clip")
        self.write("image.png", "image")
        self.write("cameras.json", '{"cameras":[{"x":1}],"path":"alias-a"}')
        self.write("depth.ply", "depth")
        self.prepared = self.inputs / "prepared"
        self.prepared.mkdir()
        self.write("prepared/source.png", "source")
        self.write("prepared/mask.png", "mask")
        self.write(
            "prepared/prepared.json",
            '{"frame":12,"crop":[1,2,3,4],"path":"candidate-specific"}',
        )
        self.canonical = self.inputs / "canonical"
        self.canonical.mkdir()
        self.write("canonical/canonical-state.pt", "state")
        self.write(
            "canonical/result.json",
            '{"prepared":{"sourceSha256":"abc","sourceWidth":10,"sourceHeight":20,"time":1}}',
        )
        self.track = self.inputs / "track"
        self.track.mkdir()
        self.write("track/source-poses.pt", "poses")
        self.write("track/motion.json", '{"frames":[1]}')

    def write(self, relative, content):
        path = self.inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_every_runner_paid_worker_command_shape_has_an_identity_policy(self):
        modal = "/modal"
        out = self.root / "out"
        commands = [
            [
                modal,
                "run",
                "worker/modal_clean_video.py",
                "--clip",
                str(self.inputs / "clip.mov"),
                "--only",
                "0",
                "--frames-out",
                str(out / "first"),
                "--report",
                str(out / "first.json"),
                "--fps",
                "12",
                "--width",
                "640",
                "--height",
                "360",
                "--dilate",
                "20",
                "--bottom-extra",
                "40",
                "--lama-px",
                "960",
                "--mask-backend",
                "semantic",
            ],
            [
                modal,
                "run",
                "worker/modal_clean_video.py",
                "--clip",
                str(self.inputs / "clip.mov"),
                "--only",
                "1,4",
                "--frames-out",
                str(out / "multi"),
                "--report",
                str(out / "multi.json"),
                "--fps",
                "12",
                "--width",
                "640",
                "--height",
                "360",
                "--dilate",
                "20",
                "--bottom-extra",
                "40",
                "--lama-px",
                "960",
                "--mask-backend",
                "semantic",
            ],
            [
                modal,
                "run",
                "worker/modal_clean_video.py",
                "--clip",
                str(self.inputs / "clip.mov"),
                "--out",
                str(out / "clean.mp4"),
                "--frame0",
                str(out / "frame.png"),
                "--masks",
                str(out / "masks.npz"),
                "--report",
                str(out / "clean.json"),
                "--fps",
                "12",
                "--width",
                "640",
                "--height",
                "360",
                "--dilate",
                "20",
                "--bottom-extra",
                "40",
                "--lama-px",
                "960",
                "--mask-backend",
                "union",
                "--moved-mask",
                "--extend-mask-to-bottom",
            ],
            [
                modal,
                "run",
                "worker/modal_motion.py",
                "--experiment",
                "pi3x",
                "--video",
                str(self.inputs / "clip.mov"),
                "--out",
                str(out / "pi3x"),
                "--allow-person-gaps",
                "--allow-missing-person-reference",
                "--alignment",
                "temporal",
                "--overlap",
                "4",
            ],
            [
                modal,
                "run",
                "worker/modal_multiperson.py::main",
                "--video",
                str(self.inputs / "clip.mov"),
                "--cameras",
                str(self.inputs / "cameras.json"),
                "--fps",
                "12",
                "--det-thresh",
                "0.15",
                "--out",
                str(out / "tracks"),
            ],
            [
                modal,
                "run",
                "worker/modal_lhm.py",
                "--prepared",
                str(self.prepared),
                "--out",
                str(out / "frozen"),
            ],
            [
                modal,
                "run",
                "worker/modal_lhm.py",
                "--canonical",
                str(self.canonical),
                "--video",
                str(self.inputs / "clip.mov"),
                "--cameras",
                str(self.inputs / "cameras.json"),
                "--out",
                str(out / "motion"),
            ],
            [
                modal,
                "run",
                "worker/modal_multiperson.py::animate",
                "--video",
                str(self.inputs / "clip.mov"),
                "--canonical",
                str(self.canonical),
                "--cameras",
                str(self.inputs / "cameras.json"),
                "--track-dir",
                str(self.track),
                "--track-id",
                "0",
                "--depth-roi",
                "1,2,3,4",
                "--depth-reference",
                str(self.inputs / "depth.ply"),
                "--out",
                str(out / "motion-00"),
            ],
            [
                modal,
                "run",
                "worker/modal_image_to_3d.py",
                "--image",
                str(self.inputs / "image.png"),
                "--out-dir",
                str(out / "shape"),
                "--refine-first",
                "--refine-prompt",
                "red ball",
                "--refine-strength",
                "0.85",
                "--note",
                "provenance only",
            ],
        ]
        # Single-person animation also fingerprints the depth companion beside cameras.json.
        self.write("frame_000.ply", "depth companion")
        for command in commands:
            with self.subTest(worker=command[2]):
                parameters, code, outputs = command_identity(
                    command, self.root, {"start": 1.0, "end": 2.0}
                )
                self.assertTrue(code)
                self.assertTrue(outputs)
                self.assertEqual(parameters["sourceSelection"], {"start": 1.0, "end": 2.0})

    def test_prepared_metadata_fingerprints_behavior_but_not_candidate_paths(self):
        command = [
            "/modal",
            "run",
            "worker/modal_lhm.py",
            "--prepared",
            str(self.prepared),
            "--out",
            str(self.root / "out"),
        ]
        first, _, _ = command_identity(command, self.root)
        self.write(
            "prepared/prepared.json",
            '{"frame":12,"crop":[1,2,3,4],"path":"renamed-candidate"}',
        )
        renamed, _, _ = command_identity(command, self.root)
        self.assertEqual(first, renamed)
        self.write(
            "prepared/prepared.json",
            '{"frame":13,"crop":[1,2,3,4],"path":"renamed-candidate"}',
        )
        changed, _, _ = command_identity(command, self.root)
        self.assertNotEqual(first, changed)

    def test_clean_masks_input_hash_changes_paid_command_identity(self):
        masks = self.inputs / "reviewed-masks.npz"
        masks.write_bytes(b"reviewed masks version one")
        command = [
            "/modal",
            "run",
            "worker/modal_clean_video.py",
            "--clip",
            str(self.inputs / "clip.mov"),
            "--out",
            str(self.root / "out" / "clean.mp4"),
            "--masks-in",
            str(masks),
        ]
        first, _, _ = command_identity(command, self.root)
        masks.write_bytes(b"reviewed masks version two")
        changed, _, _ = command_identity(command, self.root)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first["inputs"]["--masks-in"], changed["inputs"]["--masks-in"])


class PaidRunIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = object.__new__(run_clip.Pipeline)
        self.pipeline.ctx = self.root / "candidate"
        self.pipeline.ctx.mkdir()
        self.pipeline.name = "candidate"
        self.pipeline.clip = self.root / "source.mov"
        self.pipeline.clip.write_bytes(b"source")
        self.pipeline.shot = None
        self.pipeline.state = run_clip.State(self.pipeline.ctx / "state.json")
        self.pipeline.a = SimpleNamespace(
            clip=str(self.pipeline.clip),
            source_sha256=SOURCE,
            stage_ledger=str(self.root / "ledger.json"),
            stage_hypothesis=None,
        )

    def command(self, output):
        return [
            run_clip.MODAL,
            "run",
            "worker/modal_motion.py",
            "--experiment",
            "pi3x",
            "--video",
            str(self.pipeline.clip),
            "--out",
            str(output),
        ]

    def test_pending_claim_is_recorded_but_does_not_block_the_provider_call(self):
        output = self.root / "fresh-output"
        command = self.command(output)
        parameters, code, outputs = command_identity(command, run_clip.ROOT)
        StageAttempts(self.pipeline.a.stage_ledger).begin(
            SOURCE,
            "pi3x",
            parameters=parameters,
            code_version=code,
            hypothesis=None,
            candidate="old-name",
            log=self.root / "old.log",
            results=outputs,
        )

        def provider(_cmd, _log, **_kwargs):
            output.mkdir()
            (output / "modal-run.json").write_text(json.dumps({"error": None}))

        with patch.object(run_clip, "run", side_effect=provider) as call:
            self.pipeline.paid_run("pi3x", command, self.root / "new.log")
        call.assert_called_once()
        saved = json.loads(Path(self.pipeline.a.stage_ledger).read_text())["attempts"]
        self.assertEqual([item["claim"]["number"] for item in saved], [1, 2])
        self.assertEqual(saved[0]["events"][-1]["status"], "pending")
        self.assertEqual(saved[1]["events"][-1]["status"], "completed")

    def test_existing_output_is_preserved_and_blocks_before_provider_call(self):
        output = self.root / "paid-output"
        output.mkdir()
        retained = output / "artifact.ply"
        retained.write_bytes(b"paid result")
        with (
            patch.object(run_clip, "run") as provider,
            self.assertRaisesRegex(run_clip.QualityStop, "retained"),
        ):
            self.pipeline.paid_run("pi3x", self.command(output), self.root / "new.log")
        provider.assert_not_called()
        self.assertEqual(retained.read_bytes(), b"paid result")
        self.assertFalse(Path(self.pipeline.a.stage_ledger).exists())

    def test_real_paid_stage_methods_preserve_existing_outputs(self):
        self.pipeline.first_png = self.pipeline.ctx / "first/f_0000.png"
        self.pipeline.clean_mp4 = self.root / "clean.mp4"
        self.pipeline.info = {"fps": 30}
        self.pipeline.a.fps = 12
        self.pipeline.a.det_thresh = 0.3
        self.pipeline.clean_flags = lambda: ["--fps", "12"]
        cameras = self.pipeline.ctx / "pi3x/cameras.json"
        cameras.parent.mkdir()
        cameras.write_text('{"cameras": []}')
        cases = [
            ("clean_first", self.pipeline.first_png.parent, True),
            ("clean", self.pipeline.clean_mp4, False),
            ("clean_multi", self.pipeline.ctx / "clean-multi", True),
            ("pi3x", self.pipeline.ctx / "pi3x", True),
            ("tracks", self.pipeline.ctx / "tracks", True),
        ]
        for method, output, directory in cases:
            with self.subTest(method=method):
                if directory:
                    output.mkdir(exist_ok=True)
                retained = output / "retained.ply" if directory else output
                retained.write_bytes(b"previously paid output")
                with (
                    patch.object(
                        self.pipeline,
                        "world_mode",
                        return_value={"mode": "multi-image", "frames": [0]},
                    ),
                    # This test isolates retained-output protection; source/camera
                    # correspondence has its own decoded-frame fixture tests.
                    patch.object(
                        self.pipeline,
                        "multi_source_selection",
                        return_value={"frames": [{"frameIndex": 0}]},
                    ),
                    patch.object(run_clip, "run") as provider,
                    self.assertRaisesRegex(run_clip.QualityStop, "retained"),
                ):
                    getattr(self.pipeline, method)()
                provider.assert_not_called()
                self.assertEqual(retained.read_bytes(), b"previously paid output")
        self.assertFalse(Path(self.pipeline.a.stage_ledger).exists())

    def test_both_mode_sequences_shared_inpainting_allowance(self):
        for graph in (
            lambda: run_clip.single_graph("both"),
            lambda: run_clip.multiperson_graph(2, "both"),
        ):
            _stages, dependencies = graph()
            self.assertEqual(dependencies["clean_first"], [])
            self.assertEqual(dependencies["clean"], ["clean_first"])
            self.assertIn("clean", dependencies["review"])

    def latest_event(self, operation):
        saved = json.loads(Path(self.pipeline.a.stage_ledger).read_text())
        attempt = next(
            item for item in saved["attempts"] if item["claim"]["operation"] == operation
        )
        return attempt["events"][-1]

    def test_runner_records_success_provider_failure_and_uncertain_exit_with_costs(self):
        cases = (
            ("success", None, None, "completed"),
            ("provider_failure", "model failed", None, "failed"),
            ("uncertain", None, RuntimeError("transport ended after launch"), "unknown"),
        )
        for operation, reported_error, raised, expected in cases:
            output = self.root / operation
            command = self.command(output)

            def provider(
                _cmd, _log, *, raised=raised, output=output, reported_error=reported_error, **kwargs
            ):
                self.assertEqual(kwargs["attempts"], 1)
                self.assertTrue(kwargs["append"])
                if raised:
                    raise raised
                output.mkdir()
                (output / "modal-run.json").write_text(
                    json.dumps(
                        {
                            "error": reported_error,
                            "estimatedComputeUSD": 1.25,
                            "seconds": 45,
                        }
                    )
                )

            with (
                self.subTest(operation=operation),
                patch.object(run_clip, "run", side_effect=provider),
            ):
                if expected == "completed":
                    self.pipeline.paid_run(operation, command, self.root / f"{operation}.log")
                elif expected == "failed":
                    with self.assertRaises(run_clip.QualityStop):
                        self.pipeline.paid_run(operation, command, self.root / f"{operation}.log")
                else:
                    with self.assertRaisesRegex(RuntimeError, "transport ended"):
                        self.pipeline.paid_run(operation, command, self.root / f"{operation}.log")
            event = self.latest_event(operation)
            self.assertEqual(event["status"], expected)
            if expected == "unknown":
                self.assertEqual(event["costs"], {})
            else:
                self.assertEqual(event["costs"]["estimatedComputeUSD"], 1.25)
                self.assertEqual(event["costs"]["reportedWorkerSeconds"], 45)


if __name__ == "__main__":
    unittest.main()
