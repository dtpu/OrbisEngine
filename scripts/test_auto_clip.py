"""No-network, no-spend checks for the clip orchestrator's playbook decisions.

Every test drives scripts/auto_clip.py with a FAKE runner: no subprocess is started, no provider is
contacted and no file outside a temporary directory is touched. What is checked is the DECISION --
which rule fired, on what measured input, and what the log and report then say about it.

  uv run --locked python scripts/test_auto_clip.py
"""

import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import auto_clip
from auto_clip import AutoClip, Result

PROBE = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "hevc",
            "pix_fmt": "yuv420p10le",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "60/1",
            "avg_frame_rate": "60/1",
        }
    ],
    "format": {"duration": "180.0"},
}


def namespace(root, clip, **overrides):
    values = {
        "clip": str(clip),
        "name": "candidate",
        "mode": "best-shot",
        "dry_run": False,
        "resume": False,
        "truth": None,
        "force_window": None,
        "reason": None,
        "judge_segment": False,
        "operator_override": None,
        "people_cap": auto_clip.PEOPLE_CAP,
        "sequence_cap": auto_clip.SEQUENCE_SHOT_CAP,
        "marble": "video",
        "fps": 12,
        "model": "test-model",
        "stage_ledger": str(root / "attempts.json"),
        "source_sha256": None,
        "objects": False,
        "finetune": False,
        "separate_audio": False,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeRunner:
    """Matches a command by substring, returns a scripted result, and may write its outputs."""

    def __init__(self):
        self.calls = []
        self.rules = []

    def on(self, needle, *responses, first=False):
        """`responses` are (returncode, stdout, writer); consumed in order, the last repeating."""
        rule = {"needle": needle, "responses": list(responses), "used": 0}
        self.rules.insert(0, rule) if first else self.rules.append(rule)
        return self

    def __call__(self, argv, *, cwd=None, log=None, env=None):
        argv = [str(part) for part in argv]
        self.calls.append(argv)
        line = " ".join(argv)
        for rule in self.rules:
            if rule["needle"] in line:
                index = min(rule["used"], len(rule["responses"]) - 1)
                rule["used"] += 1
                code, stdout, writer = rule["responses"][index]
                if writer is not None:
                    writer(argv)
                return Result(argv=argv, returncode=code, stdout=stdout)
        return Result(argv=argv, returncode=0, stdout="")

    def matching(self, needle):
        return [argv for argv in self.calls if needle in " ".join(argv)]


class Playbook(unittest.TestCase):
    """A temporary run directory plus the happy-path fake for every stage."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # The orchestrator narrates every step; capture it so a test run stays readable, and so
        # the dry-run test can assert on what an operator would actually see.
        self.out = io.StringIO()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(self.out))
        self.root = Path(temporary.name)
        (self.root / "public" / "clips").mkdir(parents=True)
        self.clip = self.root / "source.mp4"
        self.clip.write_bytes(b"pretend this is a video file")
        self.ctx = self.root / ".context" / "run" / "candidate"
        self.ctx.mkdir(parents=True)
        self.runner = FakeRunner()
        self.clean_judge_calls = 0

    # -- writers -----------------------------------------------------------------------------

    def write_selection(self, *, person_fraction=0.10, crop=None, start=10.0, end=34.0):
        chosen = [
            {
                "id": "w0",
                "rank": 1,
                "startSeconds": start,
                "endSeconds": end,
                "score": 0.71,
                "why": "clean camera travel, one subject, no internal cut",
                "suggestedCrop": crop or {"needed": False},
                "features": {"people": {"personPixelFractionMedian": person_fraction}},
            },
            {
                "id": "w1",
                "rank": 2,
                "startSeconds": 60.0,
                "endSeconds": 80.0,
                "score": 0.63,
                "why": "second best",
                "suggestedCrop": crop or {"needed": False},
                "features": {},
            },
        ]
        (self.ctx / "selection.json").write_text(
            json.dumps(
                {
                    "video": {"path": str(self.clip), "sha256": "0" * 64},
                    "chosen": chosen,
                    "coverageNotSelected": {
                        "seconds": 136.0,
                        "fraction": 0.76,
                        "gaps": [[0, start], [end, 180]],
                    },
                    "override": None,
                    "blockedBest": None,
                    "judgeRequest": str(self.ctx / "sheets" / "judge_request.json"),
                }
            )
        )

    def write_clean_judge(self, *, verdict="pass", action="none", flags=(), frames=()):
        self.clean_judge_calls += 1
        path = self.ctx / f"clean-judge-{self.clean_judge_calls}.json"
        path.write_text(
            json.dumps(
                {
                    "verdict": verdict,
                    "recommendedAction": action,
                    "actionReason": "the judge's one clause",
                    "actionFlags": list(flags),
                    "passShare": 1.0 if verdict == "pass" else 0.25,
                    "passThreshold": 0.75,
                    "framesJudged": [0, 1, 2, 3],
                    "criticalFailures": [],
                    "frames": list(frames),
                }
            )
        )

    def write_tracks(self, count=2):
        directory = self.ctx / "tracks"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tracks.json").write_text(json.dumps({"trackCount": count}))

    def write_people_judge(self, selected=(0,), labels=None):
        labels = labels or [
            {"id": 0, "label": "main", "reason": "carries the action"},
            {"id": 1, "label": "secondary", "reason": "the opponent"},
        ]
        (self.ctx / "people-judge.json").write_text(
            json.dumps(
                {
                    "judge": "people",
                    "trackCount": len(labels),
                    "labels": labels,
                    "selected": list(selected),
                    "unresolved": [],
                }
            )
        )

    def write_identity(self, document):
        (self.ctx / "identity.json").write_text(json.dumps(document))

    def write_state(self, placement=None, stages=None):
        (self.ctx / "state.json").write_text(
            json.dumps(
                {
                    "stages": {
                        **(stages or {}),
                        "_placement": placement
                        if placement is not None
                        else {"floor": -1.2, "fittedScale0": 0.83, "scaleGateOk": True},
                    }
                }
            )
        )

    def write_pose_guard(self, document):
        (self.ctx / "pose-guard.json").write_text(json.dumps(document))

    # -- the happy path ----------------------------------------------------------------------

    def happy(self, *, person_fraction=0.10, selected=(0,), identity=None, crop=None):
        run = self.runner
        run.on("ffprobe", (0, json.dumps(PROBE), None))
        run.on(
            "select_segment.py",
            (
                0,
                "",
                lambda argv: self.write_selection(person_fraction=person_fraction, crop=crop),
            ),
        )
        run.on("ffmpeg", (0, "", None))
        run.on("--only clean", (0, "", None))
        run.on("judge_clean.py", (0, "", lambda argv: self.write_clean_judge()))
        run.on("--only pi3x,frame_align", (0, "", lambda argv: self.write_pose_guard({"ok": True})))
        run.on("--only world_prompt", (0, "", None))
        run.on("--only review,marble_video", (0, "", None))
        run.on("--only tracks", (0, "", lambda argv: self.write_tracks()))
        run.on(
            "judge_people.py",
            (
                0,
                "",
                lambda argv: (
                    self.write_people_judge(selected=selected),
                    self.write_identity(identity if identity is not None else {"ok": True}),
                ),
            ),
        )
        run.on("--only person_prep_00", (0, "", None))
        run.on("--only scale_fit", (0, "", lambda argv: self.write_state()))
        return run

    def auto(self, **overrides):
        return AutoClip(
            namespace(self.root, self.clip, **overrides), runner=self.runner, root=self.root
        )

    def log_document(self):
        return json.loads((self.ctx / "auto-log.json").read_text())

    def decision(self, ident):
        entries = [d for d in self.log_document()["decisions"] if d["id"] == ident]
        self.assertTrue(entries, f"no decision {ident!r} was recorded")
        return entries[-1]


# ---- constants ---------------------------------------------------------------------------------


class Constants(unittest.TestCase):
    def test_crowd_threshold_matches_the_worker_it_cites(self):
        """The orchestrator copies dense_pi3x's crowd threshold; the copy must not drift."""
        source = (auto_clip.ROOT / "worker/stages/dense_pi3x.py").read_text()
        match = re.search(r"^CROWD_PERSON_FRACTION\s*=\s*([0-9.]+)", source, re.MULTILINE)
        self.assertIsNotNone(match, "dense_pi3x.py no longer defines CROWD_PERSON_FRACTION")
        self.assertEqual(float(match.group(1)), auto_clip.CROWD_PERSON_FRACTION)

    def test_execution_cap_matches_the_ledger(self):
        import stage_attempts

        self.assertEqual(auto_clip.MAX_STAGE_EXECUTIONS, stage_attempts.MAX_EXECUTIONS)

    def test_merge_flags_replaces_the_mask_mode_and_keeps_the_rest(self):
        merged = auto_clip.merge_flags(
            ["--people-mask", "semantic", "--dilate", "20"], ["--moved-mask"]
        )
        self.assertEqual(merged, ["--dilate", "20", "--moved-mask"])
        merged = auto_clip.merge_flags(
            ["--people-mask", "foreground"], ["--dilate", "40", "--bottom-extra", "80"]
        )
        self.assertEqual(
            merged, ["--people-mask", "foreground", "--dilate", "40", "--bottom-extra", "80"]
        )


# ---- 3. clean ----------------------------------------------------------------------------------


class MaskMode(Playbook):
    def test_crowd_selects_the_foreground_mask(self):
        self.happy(person_fraction=auto_clip.CROWD_PERSON_FRACTION + 0.17)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        self.assertEqual(auto.mask_mode(), "foreground")
        decision = self.decision("clean-mask-mode")
        self.assertTrue(decision["choice"]["crowd"])
        self.assertIn("CROWD_PERSON_FRACTION", decision["rule"])
        self.assertEqual(decision["inputs"]["threshold"], auto_clip.CROWD_PERSON_FRACTION)

    def test_a_single_subject_keeps_the_semantic_mask(self):
        self.happy(person_fraction=0.08)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        self.assertEqual(auto.mask_mode(), "semantic")
        self.assertFalse(self.decision("clean-mask-mode")["choice"]["crowd"])


class CleanRetry(Playbook):
    def test_one_mapped_retry_then_a_pass(self):
        self.happy()
        self.runner.on(
            "judge_clean.py",
            (
                auto_clip.JUDGE_RETRY,
                "",
                lambda argv: self.write_clean_judge(
                    verdict="retry",
                    action="increase_dilation",
                    flags=["--dilate", "40", "--bottom-extra", "80"],
                ),
            ),
            (0, "", lambda argv: self.write_clean_judge()),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        cleans = self.runner.matching("--only clean")
        self.assertEqual(len(cleans), 2, "exactly one re-clean after the judge asked for it")
        self.assertIn("--dilate", cleans[1])
        self.assertIn("40", cleans[1])
        self.assertIn("--force", cleans[1])
        self.assertIn("--stage-hypothesis", cleans[1])
        self.assertEqual(auto.stages[-1].status, "passed")
        self.assertEqual(self.decision("clean")["choice"]["executions"], 2)

    def test_non_human_subject_maps_to_the_moved_mask(self):
        self.happy()
        self.runner.on(
            "judge_clean.py",
            (
                auto_clip.JUDGE_RETRY,
                "",
                lambda argv: self.write_clean_judge(
                    verdict="retry",
                    action=auto_clip.NON_HUMAN_SUBJECT_ACTION,
                    flags=["--moved-mask"],
                ),
            ),
            (0, "", lambda argv: self.write_clean_judge()),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        retry = self.runner.matching("--only clean")[1]
        self.assertIn("--moved-mask", retry)
        self.assertNotIn("--people-mask", retry)

    def test_the_retry_cap_stops_with_a_report_instead_of_looping(self):
        self.happy()
        self.runner.on(
            "judge_clean.py",
            (
                auto_clip.JUDGE_RETRY,
                "",
                lambda argv: self.write_clean_judge(
                    verdict="retry", action="increase_dilation", flags=["--dilate", "40"]
                ),
            ),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        self.assertEqual(len(self.runner.matching("--only clean")), 2)
        self.assertEqual(auto.stages[-1].status, "failed")
        self.assertIn("CLEAN_JUDGE_RETRIES", auto.stages[-1].reason)
        self.assertIsNotNone(auto.stop_reason)

    def test_a_ledger_refusal_is_surfaced_as_a_blocked_stage(self):
        self.happy()
        refusal = (
            "!! clean quality/recovery/accounting blocked: Paid clean was not submitted: "
            "Paid-stage execution allowance exhausted (three total per original source/stage "
            "and source selection window)"
        )
        self.runner.on("--only clean", (1, refusal, None), first=True)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        self.assertEqual(auto.stages[-1].status, "blocked")
        self.assertIn("allowance exhausted", auto.stages[-1].reason)
        self.assertEqual(len(self.runner.matching("judge_clean.py")), 0)

    def test_a_late_frame_filling_subject_shortens_the_window(self):
        self.happy()
        frames = [
            {"index": 0, "answers": {"smearFraction": 0.02, "subjectsRemoved": "all"}},
            {"index": 1, "answers": {"smearFraction": 0.03, "subjectsRemoved": "all"}},
            {"index": 8, "answers": {"smearFraction": 0.81, "subjectsRemoved": "partial"}},
            {"index": 9, "answers": {"smearFraction": 0.77, "subjectsRemoved": "partial"}},
        ]
        self.runner.on(
            "judge_clean.py",
            (
                auto_clip.JUDGE_RETRY,
                "",
                lambda argv: self.write_clean_judge(
                    verdict="retry",
                    action="increase_dilation",
                    flags=["--dilate", "40"],
                    frames=frames,
                ),
            ),
            (0, "", lambda argv: self.write_clean_judge()),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        self.assertEqual(auto.segment["startSeconds"], 10.0)
        self.assertEqual(auto.segment["endSeconds"], 22.0)
        dropped = auto.dropped[0]
        self.assertEqual((dropped["startSeconds"], dropped["endSeconds"]), (22.0, 34.0))
        shortened = self.decision("work-segment-seg-short")
        self.assertIn("early", shortened["because"])


# ---- 4. solve ----------------------------------------------------------------------------------


class Teleport(Playbook):
    def test_a_teleport_retries_once_on_the_longer_side(self):
        self.happy()
        guard = {
            "ok": False,
            "worst": {"frame": 71, "timeSeconds": 8.0},
            "message": "camera teleport at frame 71 (t=8.00s)",
        }
        self.runner.on(
            "--only pi3x,frame_align",
            (1, guard["message"], lambda argv: self.write_pose_guard(guard)),
            (0, "", lambda argv: self.write_pose_guard({"ok": True})),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.solve()
        self.assertEqual(len(self.runner.matching("--only pi3x,frame_align")), 2)
        # the window is 10-34 s and the teleport is 8 s in, at 18 s: the tail is the longer side
        self.assertEqual((auto.segment["startSeconds"], auto.segment["endSeconds"]), (18.0, 34.0))
        dropped = auto.dropped[0]
        self.assertEqual((dropped["startSeconds"], dropped["endSeconds"]), (10.0, 18.0))
        self.assertEqual(auto.stages[-1].status, "passed")
        self.assertEqual(self.decision("solve-teleport")["choice"]["keep"], "tail")

    def test_a_split_below_the_minimum_stops_instead_of_retrying(self):
        self.happy()
        self.write_selection(start=0.0, end=11.0)
        guard = {"ok": False, "worst": {"frame": 60, "timeSeconds": 5.5}}
        self.runner.on(
            "select_segment.py",
            (0, "", lambda argv: self.write_selection(start=0.0, end=11.0)),
            first=True,
        )
        self.runner.on(
            "--only pi3x,frame_align",
            (1, "camera teleport at frame 60 (t=5.50s)", lambda argv: self.write_pose_guard(guard)),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.solve()
        self.assertEqual(len(self.runner.matching("--only pi3x,frame_align")), 1)
        self.assertEqual(auto.stages[-1].status, "failed")
        self.assertIn("MIN_SPLIT_SECONDS", auto.stages[-1].reason)

    def test_non_overlapping_anchors_take_the_next_best_window(self):
        self.happy()
        message = (
            "!! pi3x FAILED: the measured anchors and the prediction do not agree on the same "
            "scene. Select a shorter segment."
        )
        self.runner.on(
            "--only pi3x,frame_align",
            (1, message, None),
            (0, "", lambda argv: self.write_pose_guard({"ok": True})),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.solve()
        self.assertEqual((auto.segment["startSeconds"], auto.segment["endSeconds"]), (60.0, 80.0))
        self.assertEqual(auto.segment["windowId"], "w1")
        self.assertEqual(auto.stages[-1].status, "passed")


# ---- 5. world ----------------------------------------------------------------------------------


class WorldGate(Playbook):
    def test_the_gate_opens_only_on_a_judge_pass(self):
        self.happy()
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        auto.world()
        marble = self.runner.matching("--only review,marble_video")
        self.assertEqual(len(marble), 1, "exactly one generation for this segment")
        self.assertIn("--gate-pass", marble[0])
        self.assertFalse(self.decision("world-gate")["choice"]["override"])

    def test_a_failed_judge_spends_nothing_without_an_override(self):
        self.happy()
        self.runner.on(
            "judge_clean.py",
            (
                auto_clip.JUDGE_FAIL,
                "",
                lambda argv: self.write_clean_judge(verdict="fail", action="none"),
            ),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.clean()
        auto.stop_reason = None  # ask the world step directly: would it spend?
        auto.world()
        self.assertEqual(self.runner.matching("--only review,marble_video"), [])
        self.assertFalse(self.decision("world-gate")["choice"]["pass"])

    def test_an_operator_override_is_logged_as_an_override_not_a_pass(self):
        self.happy()
        self.runner.on(
            "judge_clean.py",
            (
                auto_clip.JUDGE_FAIL,
                "",
                lambda argv: self.write_clean_judge(verdict="fail", action="none"),
            ),
            first=True,
        )
        auto = self.auto(operator_override="the producer looked at the frames and accepted them")
        auto.admit()
        auto.choose_segment()
        auto.clean()
        auto.stop_reason = None
        auto.world()
        gate = self.decision("world-gate")["choice"]
        self.assertTrue(gate["pass"])
        self.assertTrue(gate["override"])
        self.assertIn("OPERATOR OVERRIDE, not a pass", gate["because"])


# ---- 6. people ---------------------------------------------------------------------------------


class People(Playbook):
    def test_look_alike_subjects_reconstruct_one_and_stay_unverified(self):
        self.happy(
            selected=(0, 1),
            identity={
                "ok": False,
                "reason": (
                    "the swap control was not detected, so this audit has no power on this clip "
                    "and its 'ok' means nothing"
                ),
            },
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.cast()
        self.assertEqual(self.decision("identity")["choice"]["selected"], [0])
        self.assertIs(auto.identity_verified, False)
        reconstruct = self.runner.matching("--only person_prep_00")[0]
        self.assertIn("--people", reconstruct)
        self.assertEqual(reconstruct[reconstruct.index("--people") + 1], "1")
        self.assertNotIn("person_prep_01", " ".join(reconstruct))

    def test_a_verified_audit_keeps_every_selected_subject(self):
        self.happy(selected=(0, 1), identity={"ok": True})
        self.runner.on("--only person_prep_00", (0, "", None), first=True)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.cast()
        self.assertEqual(self.decision("identity")["choice"]["selected"], [0, 1])
        self.assertIs(auto.identity_verified, True)

    def test_a_prep_refusal_makes_the_person_unresolved_and_the_clip_world_only(self):
        self.happy()
        refusal = (
            "!! person_prep_00 FAILED (3s): the subject must be fully inside the frame. The "
            "subject is never fully visible in this clip -- a close-up or a crop cannot be fitted."
        )
        self.runner.on("--only person_prep_00", (1, refusal, None), first=True)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.cast()
        person = auto.people[0]
        self.assertFalse(person.reconstructed)
        self.assertIn("never fully visible", person.unresolvedReason)
        self.assertEqual(auto.stages[-1].status, "failed")
        self.assertIn("world-only", auto.stages[-1].reason)
        self.assertIsNone(auto.stop_reason, "a world-only clip is a valid outcome, not a stop")

    def test_a_pose_refusal_is_recorded_with_its_own_reason(self):
        self.happy()
        refusal = "!! lhm_frozen_00 FAILED (9s): No source person pose detected; refusing"
        self.runner.on("--only person_prep_00", (1, refusal, None), first=True)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        auto.cast()
        self.assertIn("no source person pose", auto.people[0].unresolvedReason)


# ---- 7. reconcile ------------------------------------------------------------------------------


class Reconcile(Playbook):
    def ledger_with_pending(self, *, log_text, results_name="pi3x", manifest=None):
        log = self.ctx / "pi3x.log"
        log.write_text(log_text)
        results = self.ctx / results_name
        results.mkdir(parents=True, exist_ok=True)
        (results / "cameras.json").write_text("{}")
        if manifest is not None:
            (results / "recovery-manifest.json").write_text(json.dumps(manifest))
        ledger = self.root / "attempts.json"
        ledger.write_text(
            json.dumps(
                {
                    "schema": "wander.pipeline-attempts/1",
                    "attempts": [
                        {
                            "id": "a" * 32,
                            "claim": {
                                "candidate": "candidate",
                                "stage": "pi3x",
                                "operation": "pi3x",
                                "log": str(log),
                                "results": [str(results)],
                                "startedAtEpoch": 1.0,
                            },
                            "events": [{"status": "unknown", "reason": "modal exited 1"}],
                        }
                    ],
                }
            )
        )
        return log, results

    def test_a_provider_refusal_is_archived_reconciled_and_unblocked(self):
        log, results = self.ledger_with_pending(
            log_text="Error: workspace has exceeded its spend limit for dtpu\n"
        )
        self.write_identity({"ok": True})
        self.write_state(stages={"pi3x": {"status": "failed"}})
        auto = self.auto()
        outcome = auto.reconcile("pi3x")
        self.assertEqual(outcome["status"], "done")
        self.assertEqual(outcome["kind"], "provider_refusal")
        self.assertFalse(results.exists(), "the leftover output directory was renamed, not kept")
        self.assertTrue((self.ctx / "pi3x.failed-attempt1").is_dir())
        self.assertFalse((self.ctx / "identity.json").exists())
        state = json.loads((self.ctx / "state.json").read_text())
        self.assertEqual(state["stages"]["pi3x"]["status"], "stale")
        self.assertIn("spend limit", state["stages"]["pi3x"]["staleReason"])
        command = self.runner.matching("stage_attempts.py")[0]
        self.assertIn("--reconcile", command)
        self.assertEqual(command[command.index("--status") + 1], "failed")
        self.assertEqual(command[command.index("--evidence") + 1], str(log))

    def test_a_worker_precheck_refusal_is_provable(self):
        self.ledger_with_pending(
            log_text="RuntimeError: Stage public LHM prior assets before GPU inference\n"
        )
        auto = self.auto()
        outcome = auto.reconcile("pi3x")
        self.assertEqual(outcome["kind"], "worker_precheck_refusal")

    def test_a_partial_recovery_manifest_is_provable(self):
        self.ledger_with_pending(
            log_text="downloaded outputs\n",
            manifest={"status": "partial", "reason": "coverage is partial or unverified"},
        )
        auto = self.auto()
        outcome = auto.reconcile("pi3x")
        self.assertEqual(outcome["kind"], "recovery_manifest")

    def test_ambiguous_evidence_stops_and_reports(self):
        _, results = self.ledger_with_pending(log_text="Stopping app - local entrypoint\n")
        auto = self.auto()
        outcome = auto.reconcile("pi3x")
        self.assertEqual(outcome["status"], "stopped")
        self.assertEqual(outcome["kind"], "no_evidence")
        self.assertTrue(results.is_dir(), "nothing is archived on ambiguous evidence")
        self.assertEqual(self.runner.matching("stage_attempts.py"), [])
        self.assertIn("unknown", self.decision("reconcile-pi3x")["because"])

    def test_a_refusal_after_inference_started_is_ambiguous(self):
        self.ledger_with_pending(
            log_text=(
                '{"estimatedComputeUSD": 0.057, "inferenceWallSeconds": 161.4}\n'
                "Error: workspace has exceeded its spend limit\n"
            )
        )
        auto = self.auto()
        outcome = auto.reconcile("pi3x")
        self.assertEqual(outcome["status"], "stopped")
        self.assertEqual(outcome["kind"], "refusal_after_start")
        self.assertIn("partial charge", outcome["because"])
        self.assertEqual(self.runner.matching("stage_attempts.py"), [])

    def test_no_pending_claim_is_not_an_error(self):
        (self.root / "attempts.json").write_text(json.dumps({"attempts": []}))
        auto = self.auto()
        self.assertEqual(auto.reconcile("pi3x")["status"], "none")


# ---- 8. preflight ------------------------------------------------------------------------------


class Preflight(unittest.TestCase):
    def test_the_checklist_is_read_from_the_worker_sources(self):
        requirements = auto_clip.worker_requirements()
        self.assertIn("wander-clean-video-cache", requirements["volumes"])
        self.assertIn("wander-overnight-lhm-cache", requirements["volumes"])
        self.assertIn("lhm", requirements["stagingPhases"])
        self.assertIn("gfpgan", requirements["stagingPhases"])
        paths = [item["path"] for item in requirements["weights"]]
        self.assertIn("/cache/lama/big-lama.pt", paths)
        self.assertIn("/cache/data/pretrained_models", paths)

    def test_build_paths_do_not_swamp_the_checklist(self):
        text = (
            'image = base.run_commands("git clone x /opt/basicsr")\n'
            'LAMA_PT = "/cache/lama/big-lama.pt"\n'
            'if not os.path.exists("/cache/data/pretrained_models"):\n'
            '    raise RuntimeError("Stage public LHM prior assets before GPU inference")\n'
        )
        found = [item["path"] for item in auto_clip.demanded_paths(text, "worker/x.py")]
        self.assertIn("/cache/lama/big-lama.pt", found)
        self.assertIn("/cache/data/pretrained_models", found)
        self.assertNotIn("/opt/basicsr", found)

    def test_dry_run_prints_the_checklist_and_lists_nothing_remotely(self):
        runner = FakeRunner()
        args = Namespace(profile="dtpu", check=True, dry_run=True, json=None)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(auto_clip.command_preflight(args, runner=runner), 0)
        self.assertEqual(runner.calls, [], "a dry preflight contacts nothing")
        self.assertIn("wander-clean-video-cache", printed.getvalue())
        self.assertIn("modal_stage_weights.py --phase lhm", printed.getvalue())

    def test_the_staging_commands_name_every_missing_phase(self):
        requirements = auto_clip.worker_requirements()
        _, commands = auto_clip.preflight_checklist("dtpu", requirements)
        text = "\n".join(" ".join(str(p) for p in c) for c in commands)
        for phase in requirements["stagingPhases"]:
            self.assertIn(f"--phase {phase}", text)


# ---- 9. dry run, resume, schema, report --------------------------------------------------------


class DryRun(Playbook):
    def test_a_dry_run_prints_the_plan_and_spends_nothing(self):
        self.happy()
        auto = self.auto(dry_run=True)
        self.assertEqual(auto.go(), 0)
        self.assertEqual(self.runner.calls, [], "--dry-run starts no subprocess at all")
        document = self.log_document()
        self.assertTrue(document["dryRun"])
        self.assertTrue(all(entry["dryRun"] for entry in document["commands"]))
        planned = "\n".join(" ".join(entry["argv"]) for entry in document["commands"])
        for needle in (
            "select_segment.py",
            "run_clip.py",
            "--only clean",
            "--only pi3x,frame_align",
            "judge_clean.py",
            "judge_people.py",
        ):
            self.assertIn(needle, planned, f"the plan does not print {needle}")
        self.assertEqual(document["report"]["outcome"], "planned")
        printed = self.out.getvalue()
        self.assertIn("[PAID] ", printed, "the plan marks which commands would cost money")
        self.assertIn("scripts/select_segment.py", printed)
        self.assertNotIn(" $ ", printed, "a dry run never claims to have run anything")

    def test_a_dry_run_never_renames_or_rewrites_anything(self):
        _, results = Reconcile.ledger_with_pending(
            self, log_text="Error: workspace has exceeded its spend limit\n"
        )
        self.write_identity({"ok": True})
        auto = self.auto(dry_run=True)
        auto.reconcile("pi3x")
        self.assertTrue(results.is_dir())
        self.assertTrue((self.ctx / "identity.json").exists())
        self.assertEqual(self.runner.calls, [])


class Resume(Playbook):
    def test_resume_reuses_the_decisions_already_recorded(self):
        self.happy()
        self.assertEqual(self.auto().go(), 0)
        first = len(self.runner.calls)
        self.assertGreater(first, 0)
        again = self.auto(resume=True)
        self.assertEqual(again.go(), 0)
        self.assertEqual(
            len(self.runner.calls), first, "a resumed run repeats no recorded decision"
        )
        resumed = [d for d in self.log_document()["decisions"] if d["resumed"]]
        self.assertTrue(
            {"segment", "clean", "solve", "world", "people"} <= {d["id"] for d in resumed}
        )
        self.assertEqual(again.segment["windowId"], "w0")
        self.assertIs(again.identity_verified, True)

    def test_the_log_is_append_only_across_a_resume(self):
        self.happy()
        self.auto().go()
        before = len(self.log_document()["decisions"])
        self.auto(resume=True).go()
        self.assertGreater(len(self.log_document()["decisions"]), before)


class LogSchema(Playbook):
    def test_the_decision_log_carries_its_schema_and_every_required_field(self):
        self.happy()
        auto = self.auto()
        self.assertEqual(auto.go(), 0)
        document = self.log_document()
        self.assertEqual(document["schema"], "wander.auto-clip/1")
        for key in (
            "name",
            "clip",
            "sourceSha256",
            "mode",
            "decisions",
            "commands",
            "costs",
            "report",
        ):
            self.assertIn(key, document)
        for entry in document["decisions"]:
            for key in ("id", "step", "rule", "inputs", "choice", "because", "evidence", "at"):
                self.assertIn(key, entry, f"decision {entry.get('id')} is missing {key}")
            self.assertTrue(entry["because"], f"decision {entry['id']} records no reason")
            self.assertTrue(entry["rule"], f"decision {entry['id']} names no rule")
        for entry in document["commands"]:
            for key in ("decision", "argv", "returncode", "dryRun"):
                self.assertIn(key, entry)

    def test_the_report_states_coverage_people_identity_and_what_is_invented(self):
        self.happy()
        auto = self.auto()
        auto.go()
        report = self.log_document()["report"]
        self.assertEqual(report["outcome"], "packaged")
        self.assertEqual(report["segment"]["coverageNotSelected"]["seconds"], 136.0)
        self.assertEqual(report["people"][0]["label"], "main")
        self.assertTrue(report["people"][0]["reconstructed"])
        self.assertIs(report["identityVerified"], True)
        self.assertTrue(report["placement"]["fitted"])
        self.assertTrue(report["observed"])
        self.assertTrue(any("inpainted" in text for text in report["invented"]))
        self.assertTrue(any(stage["stage"] == "clean" for stage in report["stages"]))

    def test_a_failed_scale_gate_is_never_reported_as_fitted(self):
        self.happy()
        self.runner.on(
            "--only scale_fit",
            (
                0,
                "",
                lambda argv: self.write_state(
                    placement={"floor": -1.2, "fittedScale0": 0.83, "scaleGateOk": False}
                ),
            ),
            first=True,
        )
        auto = self.auto()
        auto.go()
        placement = self.log_document()["report"]["placement"]
        self.assertFalse(placement["fitted"])
        self.assertIn("NOT fitted", placement["note"])
        self.assertIn("fallback", placement["note"])

    def test_the_costs_block_comes_from_the_ledger(self):
        self.happy()
        (self.root / "attempts.json").write_text(
            json.dumps(
                {
                    "attempts": [
                        {
                            "id": "b" * 32,
                            "claim": {
                                "candidate": "candidate",
                                "stage": "pi3x",
                                "sourceSha256": auto_clip.sha256_file(self.clip),
                                "startedAtEpoch": 1.0,
                            },
                            "events": [
                                {
                                    "status": "completed",
                                    "costs": {
                                        "estimatedComputeUSD": 0.0575,
                                        "reportedWorkerSeconds": 166.4,
                                    },
                                }
                            ],
                        }
                    ]
                }
            )
        )
        auto = self.auto()
        auto.go()
        costs = self.log_document()["costs"]
        self.assertEqual(costs["estimatedComputeUSD"], 0.0575)
        self.assertEqual(costs["byStage"]["pi3x"]["executions"], 1)
        self.assertEqual(
            costs["byStage"]["pi3x"]["executionsRemaining"], auto_clip.MAX_STAGE_EXECUTIONS - 1
        )
        self.assertIn("estimate", costs["note"])


# ---- 1, 2 and 9: admission, selection and sequence ---------------------------------------------


class Admission(Playbook):
    def test_letterbox_bars_are_cropped_into_the_working_segment(self):
        crop = {
            "needed": True,
            "ffmpeg": "crop=1920:804:0:138",
            "barFrameFraction": 1.0,
            "x": 0,
            "y": 138,
            "width": 1920,
            "height": 804,
        }
        self.happy(crop=crop)
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        ffmpeg = self.runner.matching("ffmpeg")[0]
        self.assertIn("-vf", ffmpeg)
        self.assertIn("crop=1920:804:0:138", ffmpeg)
        self.assertFalse(self.decision("work-segment-seg")["choice"]["streamCopy"])

    def test_a_whole_uncropped_constant_rate_clip_is_stream_copied(self):
        self.happy()
        self.runner.on(
            "select_segment.py",
            (0, "", lambda argv: self.write_selection(start=0.0, end=180.0)),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        ffmpeg = self.runner.matching("ffmpeg")[0]
        self.assertIn("-c", ffmpeg)
        self.assertIn("copy", ffmpeg)
        self.assertTrue(self.decision("work-segment-seg")["choice"]["streamCopy"])

    def test_a_variable_frame_rate_source_is_re_encoded(self):
        probe = json.loads(json.dumps(PROBE))
        probe["streams"][0]["avg_frame_rate"] = "59935946/1000000"
        self.runner.on("ffprobe", (0, json.dumps(probe), None), first=True)
        self.happy()
        self.runner.on(
            "select_segment.py",
            (0, "", lambda argv: self.write_selection(start=0.0, end=180.0)),
            first=True,
        )
        auto = self.auto()
        auto.admit()
        self.assertTrue(self.decision("admission")["choice"]["variableFrameRate"])
        auto.choose_segment()
        self.assertFalse(self.decision("work-segment-seg")["choice"]["streamCopy"])
        self.assertIn("variable frame rate", self.decision("work-segment-seg")["because"])

    def test_the_source_hash_and_offsets_are_recorded(self):
        self.happy()
        auto = self.auto()
        auto.admit()
        auto.choose_segment()
        self.assertEqual(
            self.decision("admission")["choice"]["sourceSha256"], auto_clip.sha256_file(self.clip)
        )
        choice = self.decision("work-segment-seg")["choice"]
        self.assertEqual((choice["sourceStartSeconds"], choice["sourceEndSeconds"]), (10.0, 34.0))

    def test_a_forced_window_without_a_reason_is_refused(self):
        self.happy()
        auto = self.auto(force_window=[12.0, 24.0], reason=None)
        auto.admit()
        with self.assertRaises(SystemExit):
            auto.choose_segment()

    def test_nothing_selectable_is_an_answer_not_a_crash(self):
        self.happy()
        self.runner.on(
            "select_segment.py",
            (
                1,
                "",
                lambda argv: (self.ctx / "selection.json").write_text(
                    json.dumps(
                        {
                            "chosen": [],
                            "blockedBest": {"id": "w3", "vetoes": ["too-dark"]},
                            "coverageNotSelected": {"seconds": 0, "fraction": 0},
                            "override": None,
                        }
                    )
                ),
            ),
            first=True,
        )
        auto = self.auto()
        self.assertEqual(auto.go(), 1)
        report = self.log_document()["report"]
        self.assertEqual(report["outcome"], "stopped")
        self.assertIn("too-dark", report["stopReason"])


class Sequence(Playbook):
    def test_sequence_mode_plans_the_longest_usable_shots_under_the_cap(self):
        self.happy()
        shots = {
            "cutCount": 4,
            "shotCount": 5,
            "shots": [
                {"index": 0, "start": 0.0, "end": 2.0, "seconds": 2.0, "tooShort": True},
                {"index": 1, "start": 2.0, "end": 20.0, "seconds": 18.0, "tooShort": False},
                {"index": 2, "start": 20.0, "end": 26.0, "seconds": 6.0, "tooShort": False},
                {"index": 3, "start": 26.0, "end": 60.0, "seconds": 34.0, "tooShort": False},
                {"index": 4, "start": 60.0, "end": 70.0, "seconds": 10.0, "tooShort": False},
            ],
        }
        self.runner.on(
            "shot_cuts.py",
            (0, "", lambda argv: (self.ctx / "cuts.json").write_text(json.dumps(shots))),
            first=True,
        )
        auto = self.auto(mode="sequence", sequence_cap=2)
        chosen = auto.plan_shots()
        self.assertEqual([s["index"] for s in chosen], [1, 3], "the two longest, in source order")
        self.assertEqual(self.decision("shots")["choice"]["shots"], [1, 3])
        auto.sequence(chosen)
        command = self.runner.matching("package_shot_sequence.py")[0]
        self.assertIn("--source", command)
        listing = json.loads((self.ctx / "shot-sequence.json").read_text())
        self.assertEqual(len(listing["shots"]), 2)
        self.assertEqual(listing["shots"][0]["sourceStart"], 2.0)


class SegmentJudge(Playbook):
    def test_the_judge_is_an_opinion_and_never_changes_the_ranking(self):
        self.happy()
        self.runner.on(
            "judge-segment",
            (
                0,
                "",
                lambda argv: (self.ctx / "segment-judge.json").write_text(
                    json.dumps({"windowId": "w1", "reason": "this is the famous play"})
                ),
            ),
            first=True,
        )
        os.environ["OPENAI_API_KEY"] = "test-only-not-a-key"
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)
        auto = self.auto(judge_segment=True)
        auto.admit()
        auto.choose_segment()
        opinion = auto.segment["judgeOpinion"]
        self.assertEqual(opinion["windowId"], "w1")
        self.assertFalse(opinion["applied"])
        self.assertEqual(auto.segment["windowId"], "w0", "the measured rank 1 still wins")
        self.assertIn("applied=false", self.decision("segment-judge")["rule"])

    def test_no_key_means_no_opinion_was_bought(self):
        self.happy()
        os.environ.pop("OPENAI_API_KEY", None)
        auto = self.auto(judge_segment=True)
        auto.admit()
        auto.choose_segment()
        self.assertFalse(auto.segment["judgeOpinion"]["asked"])
        self.assertEqual(self.runner.matching("judge-segment"), [])


if __name__ == "__main__":
    unittest.main()
