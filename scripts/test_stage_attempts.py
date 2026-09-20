"""No-spend checks for durable paid-stage accounting and runner integration."""

import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_clip
from stage_attempts import (
    CODE_FILES,
    MAX_DISTINCT_WINDOWS,
    StageAttempts,
    command_identity,
    normalized_window,
    same_work_unit,
)

SOURCE = "a" * 64


class LedgerFixture:
    """Temporary ledger and claim helpers shared by the accounting tests."""

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
        window=None,
        source=SOURCE,
    ):
        return self.ledger.begin(
            source,
            operation,
            parameters={"setting": number, "sourceSelection": window},
            code_version={"worker.py": "code-v1"},
            hypothesis=hypothesis,
            candidate=candidate,
            log=self.log,
            results=[self.root / candidate / "result"],
        )

    def finish(self, attempt, status="failed"):
        self.ledger.finish(attempt["id"], status=status, evidence=[self.log])


class StageAttemptTests(LedgerFixture, unittest.TestCase):
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


class SourceWindowTests(LedgerFixture, unittest.TestCase):
    """Distinct shot windows of one source are distinct work, not retries of each other."""

    shots = tuple({"start": index * 4.0, "end": index * 4.0 + 3.5} for index in range(8))

    def test_window_normalization_and_work_unit_rules(self):
        self.assertIsNone(normalized_window(None))
        self.assertIsNone(normalized_window({"start": 1.0, "end": None}))
        self.assertEqual(normalized_window({"start": 1.0004, "end": 2.0006}), (1.0, 2.001))
        self.assertEqual(normalized_window({"start": 1, "end": 2}), (1.0, 2.0))
        for bad in ({"start": 0.0, "end": float("inf")}, {"start": "0", "end": "1"}, [0, 1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalized_window(bad)
        with self.assertRaisesRegex(ValueError, "end before it starts"):
            normalized_window({"start": 5.0, "end": 4.0})
        cases = [
            (None, None, True),
            (None, (2.0, 3.0), True),  # the whole source covers every window
            ((0.0, 10.0), (0.0, 10.0), True),
            ((0.0, 10.0), (1.0, 10.5), True),  # 9.0s of a 9.5s window
            ((0.0, 10.0), (4.0, 14.0), True),  # 60% of the shorter window
            ((0.0, 10.0), (6.0, 16.0), False),  # 40% of the shorter window
            ((0.0, 10.0), (5.0, 15.0), False),  # exactly half is not "more than half"
            ((0.0, 10.0), (10.0, 20.0), False),  # neighbouring shots share only a boundary
            ((4.0, 4.0), (0.0, 10.0), True),  # an instant inside a window
            ((4.0, 4.0), (5.0, 10.0), False),
        ]
        for first, second, expected in cases:
            with self.subTest(first=first, second=second):
                self.assertIs(same_work_unit(first, second), expected)
                self.assertIs(same_work_unit(second, first), expected)

    def test_each_shot_window_gets_its_own_first_attempt(self):
        for index, window in enumerate(self.shots):
            with self.subTest(shot=index):
                attempt = self.begin(number=index, window=window, candidate=f"shot-{index:02d}")
                self.assertFalse(attempt["claim"]["hypothesisExplicit"])
                self.assertEqual(attempt["claim"]["windowNumber"], 1)
                self.assertEqual(attempt["claim"]["number"], index + 1)
                self.finish(attempt)
        saved = json.loads(self.ledger.path.read_text())
        self.assertEqual(len(saved["attempts"]), len(self.shots))

    def test_fourth_attempt_on_one_window_is_refused(self):
        window = self.shots[3]
        for index in range(3):
            attempt = self.begin(
                number=index,
                window=window,
                hypothesis=None if index == 0 else f"Idea {index}",
                candidate=f"try-{index}",
            )
            self.assertEqual(attempt["claim"]["windowNumber"], index + 1)
            self.finish(attempt)
        with self.assertRaisesRegex(ValueError, "three total"):
            self.begin(number=9, window=window, hypothesis="A fourth idea")
        # A different shot of the same source is unaffected by the exhausted window.
        other = self.begin(number=9, window=self.shots[4], candidate="other-shot")
        self.assertEqual(other["claim"]["windowNumber"], 1)

    def test_retry_on_one_window_still_needs_a_new_hypothesis_and_identity(self):
        window = self.shots[0]
        self.finish(self.begin(number=1, window=window))
        with self.assertRaisesRegex(ValueError, "explicit new --stage-hypothesis"):
            self.begin(number=2, window=window, candidate="renamed")
        with self.assertRaisesRegex(ValueError, "parameters or code"):
            self.begin(number=1, window=window, hypothesis="A real new idea")
        second = self.begin(number=2, window=window, hypothesis="A real new idea")
        self.finish(second)
        with self.assertRaisesRegex(ValueError, "new hypothesis"):
            self.begin(number=3, window=window, hypothesis="  A REAL   NEW IDEA ")

    def test_pending_window_blocks_itself_but_not_another_shot(self):
        pending = self.begin(number=1, window=self.shots[0])
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(number=2, window=self.shots[0], hypothesis="Change overlap")
        second = self.begin(number=2, window=self.shots[1], candidate="shot-01")
        self.ledger.finish(second["id"], status="unknown", evidence=[self.log])
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(number=3, window=self.shots[1], hypothesis="Change batch size")
        evidence = self.root / "provider-result.json"
        evidence.write_text('{"observed":"failed"}')
        self.ledger.reconcile(
            pending["id"], status="failed", evidence=evidence, reason="provider showed failure"
        )
        retry = self.begin(number=3, window=self.shots[0], hypothesis="Change overlap")
        self.assertEqual(retry["claim"]["windowNumber"], 2)

    def test_overlapping_or_float_noise_windows_are_the_same_work_unit(self):
        self.finish(self.begin(number=1, window={"start": 0.0, "end": 10.0}))
        # Nudging the boundaries by a few frames does not buy a fresh allowance.
        with self.assertRaisesRegex(ValueError, "explicit new --stage-hypothesis"):
            self.begin(number=2, window={"start": 0.125, "end": 10.125}, candidate="nudged")
        # Sub-millisecond noise on the same shot is the same window, not a new one.
        with self.assertRaisesRegex(ValueError, "explicit new --stage-hypothesis"):
            self.begin(number=3, window={"start": 0.0000004, "end": 9.9999998}, candidate="noise")
        # A whole-source claim covers the shot windows of that source.
        with self.assertRaisesRegex(ValueError, "explicit new --stage-hypothesis"):
            self.begin(number=4, window=None, candidate="whole-source")
        # Less than half of the shorter window is separate work.
        separate = self.begin(number=5, window={"start": 6.0, "end": 16.0}, candidate="next-shot")
        self.assertEqual(separate["claim"]["windowNumber"], 1)
        self.assertEqual(separate["claim"]["number"], 2)

    def test_distinct_window_cap_stops_a_runaway_shot_list(self):
        for index in range(MAX_DISTINCT_WINDOWS):
            window = {"start": index * 10.0, "end": index * 10.0 + 5.0}
            self.finish(self.begin(number=index, window=window, candidate=f"shot-{index:02d}"))
        with self.assertRaisesRegex(ValueError, "distinct windows"):
            self.begin(number=99, window={"start": 1000.0, "end": 1005.0}, candidate="one-too-many")
        # An already-claimed window still has its own remaining allowance.
        again = self.begin(
            number=99, window={"start": 0.0, "end": 5.0}, hypothesis="Retry the first shot"
        )
        self.assertEqual(again["claim"]["windowNumber"], 2)

    def test_alias_shares_the_canonical_window_allowance_but_not_other_windows(self):
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
        first = self.begin(number=1, window=self.shots[0], source=alias, candidate="alias-shot-0")
        self.assertEqual(first["claim"]["sourceSha256"], SOURCE)
        self.finish(first)
        with self.assertRaisesRegex(ValueError, "explicit new --stage-hypothesis"):
            self.begin(number=2, window=self.shots[0], candidate="canonical-shot-0")
        fresh = self.begin(number=2, window=self.shots[1], source=alias, candidate="alias-shot-1")
        self.assertEqual(fresh["claim"]["windowNumber"], 1)
        self.assertEqual(fresh["claim"]["submittedSourceSha256"], alias)
        self.finish(fresh)
        for index in (2, 3):
            self.finish(
                self.begin(
                    number=index + 10,
                    window=self.shots[0],
                    hypothesis=f"Canonical idea {index}",
                    candidate=f"canonical-{index}",
                )
            )
        with self.assertRaisesRegex(ValueError, "three total"):
            self.begin(
                number=20,
                window=self.shots[0],
                source=alias,
                hypothesis="Alias idea",
                candidate="x",
            )

    def test_concurrent_claims_allow_one_pending_execution_per_window(self):
        windows = self.shots[:3]
        barrier = threading.Barrier(2 * len(windows))
        attempts, errors = [], []
        guard = threading.Lock()

        def claim(index, window):
            barrier.wait(timeout=10)
            try:
                attempt = self.begin(number=index, window=window, candidate=f"candidate-{index}")
            except ValueError as error:
                with guard:
                    errors.append(str(error))
            else:
                with guard:
                    attempts.append(attempt)

        threads = [
            threading.Thread(target=claim, args=(index * 2 + repeat, window))
            for index, window in enumerate(windows)
            for repeat in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())

        self.assertEqual(len(attempts), len(windows))
        self.assertEqual(len(errors), len(windows))
        self.assertTrue(all("evidence reconciliation" in error for error in errors))
        claimed = {
            tuple(sorted(item["claim"]["parameters"]["sourceSelection"].items()))
            for item in attempts
        }
        self.assertEqual(len(claimed), len(windows))
        self.assertEqual(self.ledger.path.read_text().count('"status": "pending"'), len(windows))


class CommandIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        repository = Path(__file__).resolve().parent.parent
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
        workspace = patch.dict(run_clip.LOCAL_ENV, {"MODAL_PROFILE": "test-workspace"})
        workspace.start()
        self.addCleanup(workspace.stop)

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

    def test_unset_workspace_blocks_before_claim_and_provider_call(self):
        with (
            patch.dict(run_clip.LOCAL_ENV, clear=True),
            patch.object(run_clip, "run") as provider,
            self.assertRaisesRegex(run_clip.QualityStop, "MODAL_PROFILE is unset"),
        ):
            self.pipeline.paid_run("pi3x", self.command(self.root / "out"), self.root / "new.log")
        provider.assert_not_called()
        self.assertFalse(Path(self.pipeline.a.stage_ledger).exists())

    def test_claim_records_paying_workspace_and_it_changes_the_fingerprint(self):
        fingerprints = []
        for index, workspace in enumerate(("workspace-a", "workspace-b")):
            self.pipeline.a.stage_ledger = str(self.root / f"ledger-{index}.json")
            with (
                patch.dict(run_clip.LOCAL_ENV, {"MODAL_PROFILE": workspace}),
                patch.object(run_clip, "run"),
            ):
                self.pipeline.paid_run(
                    "pi3x", self.command(self.root / f"out-{index}"), self.root / "new.log"
                )
            claim = json.loads(Path(self.pipeline.a.stage_ledger).read_text())["attempts"][0][
                "claim"
            ]
            self.assertEqual(claim["parameters"]["workspace"], workspace)
            fingerprints.append(claim["fingerprint"])
        self.assertNotEqual(*fingerprints)

    def test_stage_environment_is_claimed_and_malformed_values_block(self):
        self.pipeline.a.stage_environment = ["lamaWeightsSha256=abc", "cacheVolume=v1"]
        with patch.object(run_clip, "run"):
            self.pipeline.paid_run("pi3x", self.command(self.root / "out"), self.root / "new.log")
        ledger = json.loads(Path(self.pipeline.a.stage_ledger).read_text())
        self.assertEqual(
            ledger["attempts"][0]["claim"]["parameters"]["environment"],
            {"lamaWeightsSha256": "abc", "cacheVolume": "v1"},
        )
        for bad in (["novalue"], ["=x"], ["k="], ["k=1", "k=2"]):
            self.pipeline.a.stage_environment = bad
            with (
                self.subTest(bad=bad),
                patch.object(run_clip, "run") as provider,
                self.assertRaisesRegex(run_clip.QualityStop, "stage-environment"),
            ):
                self.pipeline.paid_run(
                    "pi3x", self.command(self.root / "other"), self.root / "new.log"
                )
            provider.assert_not_called()
        self.assertEqual(
            len(json.loads(Path(self.pipeline.a.stage_ledger).read_text())["attempts"]), 1
        )

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
