"""No-spend checks for durable paid-stage accounting and runner integration."""

import hashlib
import sys
from pathlib import Path

# The modules under test are this directory's parent; importing them by name is what
# running from scripts/ used to give for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_clip
from stage_attempts import AUTHORIZATION_SCHEMA, CODE_FILES, StageAttempts, command_identity

SOURCE = "a" * 64


class StageAttemptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = StageAttempts(self.root / "attempts.json")
        self.log = self.root / "worker.log"
        self.log.write_text("saved provider evidence\n")

    def begin(
        self,
        operation="pi3x",
        number=0,
        hypothesis=None,
        candidate="candidate-a",
        authorization=None,
        source=SOURCE,
    ):
        return self.ledger.begin(
            source,
            operation,
            parameters={"setting": number},
            code_version={"worker.py": "code-v1"},
            hypothesis=hypothesis,
            candidate=candidate,
            log=self.log,
            results=[self.root / candidate / "result"],
            authorization=authorization,
        )

    def finish(self, attempt, status="failed"):
        self.ledger.finish(attempt["id"], status=status, evidence=[self.log])

    def test_aliases_share_cap_and_candidate_rename_does_not_make_a_retry_new(self):
        first = self.begin("clean", 1)
        self.finish(first)
        # clean_first is a separate requested workflow, but shares the inpainting allowance.
        second = self.begin("clean_first", 2, candidate="renamed-candidate")
        self.finish(second)
        with self.assertRaisesRegex(ValueError, "explicit new --stage-hypothesis"):
            self.begin("clean", 3, candidate="another-name")
        third = self.begin("clean", 3, hypothesis="Test a wider semantic mask")
        self.finish(third)
        with self.assertRaisesRegex(ValueError, "three total"):
            self.begin("clean_multi", 4)

    def test_pending_and_unknown_require_evidence_reconciliation(self):
        pending = self.begin(number=1)
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(number=2, hypothesis="Change overlap")
        evidence = self.root / "provider-result.json"
        evidence.write_text('{"observed":"failed"}')
        self.ledger.reconcile(
            pending["id"], status="failed", evidence=evidence, reason="provider showed failure"
        )
        reconciled = json.loads(self.ledger.path.read_text())["attempts"][0]
        self.assertEqual(reconciled["claim"]["number"], 1)
        self.assertEqual(
            reconciled["events"][-1]["evidenceSha256"],
            hashlib.sha256(evidence.read_bytes()).hexdigest(),
        )
        retry = self.begin(number=2, hypothesis="Change overlap")
        self.ledger.finish(retry["id"], status="unknown", evidence=[self.log])
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(number=3, hypothesis="Change batch size")

    def test_retry_needs_new_hypothesis_and_meaningful_identity_change(self):
        first = self.begin(number=1)
        self.finish(first)
        with self.assertRaisesRegex(ValueError, "explicit new"):
            self.begin(number=2)
        with self.assertRaisesRegex(ValueError, "parameters or code"):
            self.begin(number=1, hypothesis="A real new idea", candidate="renamed")
        second = self.begin(number=2, hypothesis="A real new idea")
        self.finish(second)
        with self.assertRaisesRegex(ValueError, "new hypothesis"):
            self.begin(number=3, hypothesis="  A REAL   NEW IDEA ")

    def test_concurrent_claims_allow_exactly_one_pending_execution(self):
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

        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("evidence reconciliation", errors[0])
        saved = self.ledger.path.read_text()
        self.assertEqual(saved.count('"status": "pending"'), 1)

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
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(results[0]["claim"]["sourceSha256"], SOURCE)
        self.assertIn("evidence reconciliation", errors[0])

    def test_source_alias_cannot_bypass_pending_or_three_execution_cap(self):
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
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(number=2, hypothesis="Changed overlap")
        self.finish(pending)
        second = self.begin(number=2, hypothesis="Changed overlap")
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
        self.finish(third)
        with self.assertRaisesRegex(ValueError, "three total"):
            self.begin(number=4, hypothesis="Changed model")

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

    def test_legacy_over_cap_history_can_be_reconciled_and_never_reset(self):
        first = self.begin(number=1)
        self.finish(first)
        saved = json.loads(self.ledger.path.read_text())
        # Simulate a ledger created while the cap was disabled. Restoring the guard
        # must retain every execution, including an unresolved fourth execution.
        for number in (2, 3, 4):
            legacy = json.loads(json.dumps(saved["attempts"][0]))
            legacy["id"] = f"legacy-{number}"
            legacy["claim"]["number"] = number
            if number == 4:
                legacy["events"] = legacy["events"][:1]
            saved["attempts"].append(legacy)
        self.ledger.path.write_text(json.dumps(saved))
        self.ledger.reconcile(
            "legacy-4",
            status="failed",
            evidence=self.log,
            reason="Observed terminal provider failure",
        )
        with self.assertRaisesRegex(ValueError, "three total"):
            self.begin(number=5, hypothesis="A fresh idea", candidate="renamed")
        retained = json.loads(self.ledger.path.read_text())["attempts"]
        self.assertEqual(len(retained), 4)
        self.assertEqual(retained[:3], saved["attempts"][:3])
        self.assertTrue(retained[-1]["events"][-1]["reconciliation"])

    def permit(self, **changes):
        permit = {
            "schema": AUTHORIZATION_SCHEMA,
            "id": "reviewed-rebuild-1",
            "sourceSha256": SOURCE,
            "candidate": "candidate-a",
            "reason": "Reviewed full source rebuild",
            "operatorApproval": "okay now start it end to end",
            "expiresAtEpoch": time.time() + 3600,
            "operations": {"pi3x": 0},
        }
        permit.update(changes)
        path = self.root / "authorization.json"
        path.write_text(json.dumps(permit))
        return path

    def test_authorization_allows_one_unchanged_over_cap_execution_preserving_history(self):
        for number in range(3):
            attempt = self.begin(number=number, hypothesis=f"Reviewed variation {number}")
            self.finish(attempt)
        history = json.loads(self.ledger.path.read_text())["attempts"]
        permit = self.permit(operations={"pi3x": 3})
        approved = self.begin(number=2, authorization=permit)
        self.assertEqual(approved["claim"]["number"], 4)
        saved = approved["claim"]["authorization"]
        self.assertEqual(saved["id"], "reviewed-rebuild-1")
        self.assertEqual(saved["sha256"], hashlib.sha256(permit.read_bytes()).hexdigest())
        self.assertEqual(saved["path"], str(permit.resolve()))
        self.assertEqual(saved["operatorApproval"], "okay now start it end to end")
        self.assertEqual(json.loads(self.ledger.path.read_text())["attempts"][:3], history)
        self.finish(approved)
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.begin(authorization=permit)
        with self.assertRaisesRegex(ValueError, "three total"):
            self.begin(number=4, hypothesis="Another variation")

    def test_authorization_is_required_for_unchanged_retry_below_cap(self):
        self.finish(self.begin())
        with self.assertRaisesRegex(ValueError, "parameters or code"):
            self.begin(hypothesis="Requested rebuild")
        attempt = self.begin(authorization=self.permit(operations={"pi3x": 1}))
        self.finish(attempt)

    def test_authorization_rejects_wrong_source_candidate_operation_and_count(self):
        cases = (
            ({"sourceSha256": "b" * 64}, "canonical source"),
            ({"candidate": "candidate-b"}, "candidate"),
            ({"operations": {"tracks": 0}}, "operation"),
            ({"operations": {"pi3x": 1}}, "count"),
            ({"expiresAtEpoch": time.time() - 1}, "expired"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.begin(authorization=self.permit(**changes))
            self.assertFalse(self.ledger.path.exists())

    def test_authorization_rejects_malformed_fields_and_duplicate_json_keys(self):
        cases = (
            {"schema": "unknown"},
            {"id": ""},
            {"id": 7},
            {"candidate": " "},
            {"reason": ""},
            {"operatorApproval": None},
            {"expiresAtEpoch": True},
            {"expiresAtEpoch": float("nan")},
            {"expiresAtEpoch": float("inf")},
            {"expiresAtEpoch": 10**400},
            {"expiresAtEpoch": "tomorrow"},
            {"operations": {}},
            {"operations": []},
            {"operations": {"*": 0}},
            {"operations": {"inpainting": 0}},
            {"operations": {"pi3x": True}},
            {"operations": {"pi3x": -1}},
            {"operations": {"pi3x": 0.0}},
            {"operations": {"pi3x": 0, "tracks": "1"}},
            {"extra": "unrecognized"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.begin(authorization=self.permit(**changes))
        path = self.permit()
        for raw in ("null", "[1]", "{", '{"id":"first","id":"second"}'):
            path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.begin(authorization=path)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.begin(authorization=path)
        self.assertFalse(self.ledger.path.exists())

    def test_authorization_shared_inpainting_operations_use_logical_counts(self):
        permit = self.permit(operations={"clean_first": 0, "clean": 1})
        first = self.begin("clean_first", authorization=permit)
        self.finish(first)
        second = self.begin("clean", authorization=permit)
        self.assertEqual(second["claim"]["number"], 2)
        self.finish(second)
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.begin("clean_first", authorization=permit)

    def test_authorization_changed_file_cannot_reuse_id_for_new_operation(self):
        permit = self.permit()
        self.finish(self.begin(authorization=permit))
        modified = self.permit(operations={"tracks": 0})
        with self.assertRaisesRegex(ValueError, "already bound"):
            self.begin("tracks", authorization=modified)

    def test_authorization_must_name_canonical_source_even_for_alias_submission(self):
        alias = "b" * 64
        self.ledger.path.write_text(
            json.dumps(
                {
                    "schema": "wander.pipeline-attempts/1",
                    "attempts": [],
                    "sourceAliases": {alias: SOURCE},
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "canonical source"):
            self.begin(source=alias, authorization=self.permit(sourceSha256=alias))
        permit = self.permit()
        attempt = self.begin(source=alias, authorization=permit)
        self.assertEqual(attempt["claim"]["sourceSha256"], SOURCE)
        self.assertEqual(attempt["claim"]["submittedSourceSha256"], alias)
        self.finish(attempt)
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.begin(authorization=permit)

    def test_authorization_never_bypasses_pending_or_unknown_claims(self):
        pending = self.begin()
        permit = self.permit(operations={"pi3x": 1})
        for status in ("pending", "unknown"):
            if status == "unknown":
                self.finish(pending, status=status)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "reconciliation"):
                self.begin(authorization=permit)

    def test_authorized_unknown_execution_is_consumed_after_reconciliation(self):
        permit = self.permit()
        attempt = self.begin(authorization=permit)
        self.finish(attempt, status="unknown")
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            self.begin(authorization=permit)
        self.ledger.reconcile(
            attempt["id"], status="failed", evidence=self.log, reason="Observed failure"
        )
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.begin(authorization=permit)

    def test_concurrent_authorization_claims_consume_exactly_once(self):
        permit = self.permit()
        barrier = threading.Barrier(2)
        attempts, errors = [], []

        def claim():
            barrier.wait(timeout=5)
            try:
                attempts.append(self.begin(authorization=permit))
            except ValueError as error:
                errors.append(str(error))

        workers = [threading.Thread(target=claim) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("reconciliation", errors[0])

    def test_expired_saved_authorization_does_not_prevent_reconciliation(self):
        attempt = self.begin(authorization=self.permit())
        with patch("stage_attempts.time.time", return_value=time.time() + 7200):
            self.ledger.reconcile(
                attempt["id"], status="failed", evidence=self.log, reason="Observed failure"
            )

    def test_malformed_saved_authorization_blocks_further_claims(self):
        attempt = self.begin(authorization=self.permit())
        self.finish(attempt)
        saved = json.loads(self.ledger.path.read_text())
        saved["attempts"][0]["claim"]["authorization"]["operations"]["pi3x"] = 1
        self.ledger.path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "saved paid-stage authorization"):
            self.begin("tracks")


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

    def test_pending_claim_blocks_before_provider_call(self):
        command = self.command(self.root / "fresh-output")
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
        with (
            patch.object(run_clip, "run") as provider,
            self.assertRaisesRegex(run_clip.QualityStop, "evidence reconciliation"),
        ):
            self.pipeline.paid_run("pi3x", command, self.root / "new.log")
        provider.assert_not_called()

    def test_runner_passes_authorization_and_rejects_reuse_before_provider(self):
        path = self.root / "authorization.json"
        path.write_text(
            json.dumps(
                {
                    "schema": AUTHORIZATION_SCHEMA,
                    "id": "integration-approval",
                    "sourceSha256": SOURCE,
                    "candidate": self.pipeline.name,
                    "reason": "Reviewed full rebuild",
                    "operatorApproval": "start it end to end",
                    "expiresAtEpoch": time.time() + 3600,
                    "operations": {"pi3x": 0},
                }
            )
        )
        self.pipeline.a.stage_authorization = str(path)
        with patch.object(run_clip, "run") as provider:
            self.pipeline.paid_run("pi3x", self.command(self.root / "output"), self.root / "log")
        provider.assert_called_once()
        saved = json.loads(Path(self.pipeline.a.stage_ledger).read_text())["attempts"]
        self.assertEqual(saved[0]["claim"]["authorization"]["id"], "integration-approval")
        with (
            patch.object(run_clip, "run") as provider,
            self.assertRaisesRegex(run_clip.QualityStop, "consumed"),
        ):
            self.pipeline.paid_run(
                "pi3x", self.command(self.root / "other-output"), self.root / "log2"
            )
        provider.assert_not_called()

    def test_cli_authorization_environment_default_and_explicit_override(self):
        for extra, expected in (
            ([], "/env-permit.json"),
            (["--stage-authorization", "/cli-permit.json"], "/cli-permit.json"),
        ):
            with (
                self.subTest(extra=extra),
                patch.dict(os.environ, {"WANDER_STAGE_AUTHORIZATION": "/env-permit.json"}),
                patch.object(
                    sys,
                    "argv",
                    ["run_clip.py", "--clip", "unused", "--name", "test", "--no-publish", *extra],
                ),
                patch.object(run_clip, "resolve_shots", return_value=[]) as resolve,
                self.assertRaises(SystemExit) as exited,
            ):
                run_clip.main()
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(resolve.call_args.args[0].stage_authorization, expected)

    def test_cli_rejects_authorization_for_orchestrated_submission(self):
        with (
            patch.dict(os.environ, {"WANDER_STAGE_AUTHORIZATION": "/env-permit.json"}),
            patch.object(
                sys, "argv", ["run_clip.py", "--clip", "unused", "--name", "test", "--orchestrated"]
            ),
            patch.object(run_clip, "resolve_shots") as resolve,
            patch("sys.stderr"),
            self.assertRaises(SystemExit) as exited,
        ):
            run_clip.main()
        self.assertEqual(exited.exception.code, 2)
        resolve.assert_not_called()

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
        for graph, clean_dependencies in (
            (lambda: run_clip.single_graph("both"), ["clean_first"]),
            (lambda: run_clip.multiperson_graph(2, "both"), ["clean_first", "tracks"]),
        ):
            _stages, dependencies = graph()
            self.assertEqual(dependencies["clean_first"], [])
            self.assertEqual(dependencies["clean"], clean_dependencies)
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
