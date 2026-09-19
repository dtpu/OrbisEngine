"""Offline, synthetic contract evidence; these fixtures are not human or live-model reviews."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import world_quality_review as review
from quality_gate import (
    ACKNOWLEDGEMENTS,
    CRITERIA,
    EVIDENCE_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    file_sha256,
)


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2))


def create_fixture(root: Path, count: int = 3):
    """Return assess kwargs and mutable documents; mock the two source decoding helpers."""
    root.mkdir(parents=True, exist_ok=True)
    clip, world, cameras = root / "clip.mp4", root / "world.spz", root / "cameras.json"
    clip.write_bytes(b"synthetic source")
    world.write_bytes(b"synthetic world")
    rows = [
        {
            "sourceIndex": index,
            "time": float(index),
            "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "source_intrinsics": [[10, 0, 8], [0, 10, 8], [0, 0, 1]],
            "source_image_size": [16, 16],
        }
        for index in range(count)
    ]
    write_json(cameras, {"cameras": rows})
    evidence = root / "evidence"
    evidence.mkdir()
    report = {
        "schema": EVIDENCE_SCHEMA,
        "inputs": review.current_inputs(clip, world, cameras, 1.0),
        "evidenceCodeSha256": review.evidence_code_sha256(),
        "timingMethod": review.TIMING_METHOD,
        "sourceTimeOriginSeconds": 0.0,
        "requestedFrames": list(range(count)),
        "frames": [],
    }
    for index in range(count):
        row = {"frame": index, "timeSeconds": float(index), "sourcePtsSeconds": float(index)}
        for field, name in (("pair", "pair"), ("sourceImage", "source"), ("renderImage", "render")):
            path = evidence / f"{name}-{index}.png"
            path.write_bytes(f"{name}-{index}".encode())
            row[field] = {"path": path.name, "sha256": file_sha256(path)}
        report["frames"].append(row)
    report_path = evidence / "report.json"
    write_json(report_path, report)
    plan = {
        "schema": PLAN_SCHEMA,
        "inputs": report["inputs"],
        "criteria": list(CRITERIA),
        "passFraction": 0.75,
        "evidenceReport": {"path": "report.json", "sha256": file_sha256(report_path)},
        "samples": [
            {
                "id": f"f{index}",
                "sourceFrame": index,
                "timeSeconds": float(index),
                "view": "source-camera",
                "critical": False,
            }
            for index in range(count)
        ],
    }
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    result = {
        "schema": RESULT_SCHEMA,
        "planSha256": file_sha256(plan_path),
        "judgeKind": "manual",
        "reviewer": "Synthetic test reviewer, not a real sign-off",
        "acknowledgements": ACKNOWLEDGEMENTS.copy(),
        "samples": [
            {
                "id": f"f{index}",
                "pairSha256": report["frames"][index]["pair"]["sha256"],
                "criteria": {
                    key: {
                        "status": "pass",
                        "criticalFailure": False,
                        "reason": "Synthetic contract fixture",
                    }
                    for key in CRITERIA
                },
            }
            for index in range(count)
        ],
    }
    result_path = root / "result.json"
    write_json(result_path, result)
    return (
        {
            "plan_path": plan_path,
            "result_path": result_path,
            "evidence_root": evidence,
            "clip": clip,
            "world": world,
            "cameras": cameras,
            "scale0": 1.0,
        },
        {"plan": plan, "result": result, "report": report, "cameras": {"cameras": rows}},
    )


class WorldReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.kwargs, self.docs = create_fixture(self.root)
        self.times = patch.object(review, "source_timestamps", return_value=[0.0, 1.0, 2.0])
        self.frames = patch.object(
            review, "source_frame_png", side_effect=lambda clip, index: f"source-{index}".encode()
        )
        self.times.start()
        self.frames.start()
        self.addCleanup(self.times.stop)
        self.addCleanup(self.frames.stop)

    def save(self):
        write_json(self.kwargs["cameras"], self.docs["cameras"])
        write_json(self.kwargs["evidence_root"] / "report.json", self.docs["report"])
        self.docs["plan"]["evidenceReport"]["sha256"] = file_sha256(
            self.kwargs["evidence_root"] / "report.json"
        )
        write_json(self.kwargs["plan_path"], self.docs["plan"])
        self.docs["result"]["planSha256"] = file_sha256(self.kwargs["plan_path"])
        write_json(self.kwargs["result_path"], self.docs["result"])

    def assert_blocked(self, fragment=None):
        result = review.assess(**self.kwargs)
        self.assertEqual(result["status"], "blocked", result)
        if fragment:
            self.assertIn(fragment, " ".join(result["errors"]))

    def test_manual_import_is_read_only_and_preserves_both_legacy_ledger_shapes(self):
        for name, claims in (
            ("daniel", [{"status": "pending", "reservedRequests": 5}]),
            ("austin", {"source:stage": {"status": "running"}}),
        ):
            write_json(
                self.root / f"{name}-ledger.json",
                {"schema": "wander.quality-attempts/1", "attempts": claims},
            )
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(
            review.subprocess, "run", side_effect=AssertionError("unexpected execution")
        ):
            result = review.assess(**self.kwargs)
        self.assertEqual(result["status"], "passed", result)
        after = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertIn("no geometric", result["scope"])

    def test_llm_result_blocks_even_with_calibration_claims(self):
        self.docs["result"].update(
            judgeKind="llm", model="claimed-model", calibration={"passed": True}
        )
        self.save()
        self.assert_blocked("LLM acceptance is disabled")

    def test_input_mutations_block(self):
        for name in ("clip", "world", "cameras"):
            with self.subTest(name=name):
                path = self.kwargs[name]
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.assert_blocked("current inputs")
                path.write_bytes(original)
        self.kwargs["scale0"] = 1.1
        self.assert_blocked("current inputs")

    def test_missing_malformed_and_unknown_results_block(self):
        for document in ([], {}, {"schema": RESULT_SCHEMA}):
            write_json(self.kwargs["result_path"], document)
            self.assert_blocked()
        self.docs["result"]["samples"][0]["criteria"]["room_layout"]["status"] = "unknown"
        self.save()
        self.assert_blocked("f0/room_layout")
        self.kwargs["result_path"].unlink()
        self.assert_blocked()

    def test_changed_plan_pair_source_render_report_and_code_block(self):
        self.kwargs["plan_path"].write_text(self.kwargs["plan_path"].read_text() + "\n")
        self.assert_blocked("plan bytes changed")
        self.save()
        for name in ("pair", "source", "render"):
            path = self.kwargs["evidence_root"] / f"{name}-0.png"
            original = path.read_bytes()
            path.write_bytes(b"changed")
            self.assert_blocked("evidence bytes changed")
            path.write_bytes(original)
        self.docs["report"]["evidenceCodeSha256"] = "0" * 64
        self.save()
        self.assert_blocked("code changed")

    def test_exact_camera_and_actual_source_times_are_required(self):
        self.docs["plan"]["samples"][1]["timeSeconds"] = 1.01
        self.save()
        self.assert_blocked("planned sample time")
        self.docs["plan"]["samples"][1]["timeSeconds"] = 1.0
        self.save()
        with patch.object(review, "source_timestamps", return_value=[0, 1.01, 2]):
            self.assert_blocked("actual source decoded PTS")
        with patch.object(review, "source_frame_png", return_value=b"wrong source pixels"):
            self.assert_blocked("source evidence pixels")

    def test_camera_numeric_invalidity_and_duplicate_samples_block(self):
        for value in (float("nan"), True, "1"):
            self.docs["cameras"]["cameras"][0]["camera_to_world"][0][0] = value
            self.save()
            inputs = review.current_inputs(
                self.kwargs["clip"], self.kwargs["world"], self.kwargs["cameras"], 1.0
            )
            self.docs["plan"]["inputs"] = inputs
            self.docs["report"]["inputs"] = inputs
            self.save()
            self.assert_blocked("finite")

    def test_coverage_sample_identity_and_threshold_cannot_be_weakened(self):
        original = copy.deepcopy(self.docs)
        mutations = [
            lambda: self.docs["plan"].update(passFraction=0.5),
            lambda: self.docs["plan"]["samples"][1].update(sourceFrame=0),
            lambda: self.docs["result"]["samples"].pop(),
            lambda: self.docs["result"]["samples"][1].update(id="f0"),
            lambda: self.docs["result"]["samples"][0]["criteria"].pop("appearance"),
            lambda: self.docs["result"].update(reviewer=""),
            lambda: self.docs["result"].update(acknowledgements={}),
        ]
        for change in mutations:
            self.docs = copy.deepcopy(original)
            change()
            self.save()
            self.assert_blocked()

    def test_critical_failure_cannot_be_averaged_away(self):
        kwargs, docs = create_fixture(self.root / "five", count=5)
        decision = docs["result"]["samples"][0]["criteria"]["room_layout"]
        decision.update(status="fail", criticalFailure=True)
        write_json(kwargs["result_path"], docs["result"])
        with patch.object(review, "source_timestamps", return_value=[0, 1, 2, 3, 4]):
            result = review.assess(**kwargs)
        self.assertEqual(result["criteria"]["room_layout"]["passingFraction"], 0.8)
        self.assertEqual(result["status"], "failed")

        decision["criticalFailure"] = False
        write_json(kwargs["result_path"], docs["result"])
        with patch.object(review, "source_timestamps", return_value=[0, 1, 2, 3, 4]):
            self.assertEqual(review.assess(**kwargs)["status"], "passed")
        docs["plan"]["samples"][0]["critical"] = True
        write_json(kwargs["plan_path"], docs["plan"])
        docs["result"]["planSha256"] = file_sha256(kwargs["plan_path"])
        write_json(kwargs["result_path"], docs["result"])
        with patch.object(review, "source_timestamps", return_value=[0, 1, 2, 3, 4]):
            self.assertEqual(review.assess(**kwargs)["status"], "failed")

    def test_path_escape_and_symlink_evidence_block(self):
        source = self.kwargs["evidence_root"] / "source-0.png"
        outside = self.root / "outside.png"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        self.assert_blocked("outside evidence root")
        self.docs["plan"]["evidenceReport"]["path"] = "../outside.png"
        self.save()
        self.assert_blocked("safe relative path")

    def test_duplicate_json_keys_block(self):
        self.kwargs["plan_path"].write_text('{"schema": "one", "schema": "two"}')
        self.assert_blocked("duplicate JSON key")

    def test_evidence_mutated_during_inspection_blocks(self):
        def decode(clip, index):
            if index == 2:
                path = self.kwargs["evidence_root"] / "pair-0.png"
                path.write_bytes(b"changed after earlier frame was validated")
            return f"source-{index}".encode()

        with patch.object(review, "source_frame_png", side_effect=decode):
            self.assert_blocked("changed during inspection")

    def test_nonfinite_scale_and_boolean_input_scale_block(self):
        for scale in (True, float("nan"), float("inf"), 0):
            with self.subTest(scale=scale):
                self.kwargs["scale0"] = scale
                self.assert_blocked()
        self.kwargs["scale0"] = 1.0
        self.docs["plan"]["inputs"]["registrationScale"] = True
        self.save()
        self.assert_blocked("finite")


class DecodedSourceTests(unittest.TestCase):
    def test_real_local_decode_uses_frame_ordinal_and_pts_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "tiny.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=16x16:rate=2",
                    "-frames:v",
                    "3",
                    "-c:v",
                    "ffv1",
                    str(clip),
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(review.source_timestamps(clip), [0.0, 0.5, 1.0])
            first, last = review.source_frame_png(clip, 0), review.source_frame_png(clip, 2)
            self.assertTrue(first.startswith(b"\x89PNG"))
            self.assertNotEqual(first, last)
            self.assertEqual(review.source_frame_png(clip, 2), last)


if __name__ == "__main__":
    unittest.main()
