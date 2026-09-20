"""Judge lifecycle tests. Every transport here is a fake; nothing in this file reaches a network.

The synthetic tracks directory mirrors the real `tracks` stage output, which is written by
worker/stages/track_people.py: the top-level document at lines 526-560, the per-track `report`
entries at lines 490-511 (`track`, `rawId`, `quality`, `candidateSamples`, `samples`), and
`track_<rank:02d>/motion.json` holding `frames`, each record built at lines 420-447 with `sample`,
`sourceIndex`, `box` and `maskBox` in source pixels. Its consumers agree: scripts/run_clip.py:500
and :1404 read `tracks/tracks.json` and `tracks/track_NN/motion.json`, and
scripts/prepare_track_person.py:82 reads the same per-track file.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import cv2
import judge_people
import judge_shots
import numpy as np
import vlm_once

USAGE = {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}

WIDTH, HEIGHT = 160, 120
SOURCE_FPS = 10.0
SOURCE_FRAMES = 60
SAMPLE_COUNT = 20


class FakeResponse:
    def __init__(self, payload, *, request_id="req-test", status=200):
        self.payload = payload
        self.headers = {"x-request-id": request_id}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def completion(content, *, usage=USAGE):
    body = content if isinstance(content, str) else json.dumps(content)
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "model": "gpt-6-astra",
            "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
            "usage": usage,
        }
    ).encode()


def must_not_call(*_args, **_kwargs):
    raise AssertionError("transport must not be called")


def counting_transport(payload):
    calls = []

    def transport(*_args, **_kwargs):
        calls.append(True)
        return FakeResponse(payload)

    return transport, calls


def write_video(path, frames=SOURCE_FRAMES):
    """A short MJPG/AVI clip whose frames differ, so ffmpeg and cv2 both have something to read."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), SOURCE_FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        return False
    for ordinal in range(frames):
        image = np.full((HEIGHT, WIDTH, 3), 30, np.uint8)
        image[:, : 4 + (ordinal % 20) * 6] = 200
        cv2.circle(image, (20 + 3 * ordinal, 60), 12, (0, 0, 255), -1)
        writer.write(image)
    writer.release()
    return True


def motion_record(sample, source_index, box):
    """One `frames` record, with the fields track_people.py writes that this judge reads."""
    x0, y0, x1, y1 = box
    return {
        "sample": sample,
        "sourceIndex": source_index,
        "time": round(source_index / SOURCE_FPS, 3),
        "detectedPeople": 2,
        "score": 0.9,
        "originalSourceTime": round(source_index / SOURCE_FPS, 3),
        "projectedBodyJoints": [[float(x0), float(y0)]] * 22,
        "jointProjectionInImage": [True] * 22,
        "confidenceSemantics": "Score is person detection confidence, not per-joint accuracy.",
        "source_intrinsics": [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
        "rootCamera": [0.0, 0.0, 3.0],
        "smplTranslationCamera": [0.0, 0.0, 3.0],
        "feetCamera": [[0.0, 1.0, 3.0]] * 4,
        "footJointOrder": ["leftAnkle", "rightAnkle", "leftFoot", "rightFoot"],
        "rootRotationVector": [0.0, 3.1, 0.0],
        "detectionThreshold": 0.15,
        "trackId": 0,
        "box": [float(x0), float(y0), float(x1), float(y1)],
        "maskBox": [float(x0), float(y0), float(x1), float(y1)],
        "maskIou": 0.8,
        "occludedFraction": 0.0,
        "depth": 3.0,
        "heightFraction": (y1 - y0) / HEIGHT,
        "maskAreaFraction": 0.1,
    }


def write_tracks(root, video):
    """A `tracks` stage output directory: tracks.json plus track_NN/motion.json."""
    tracks_dir = Path(root) / "tracks"
    source_indices = [index * 2 for index in range(SAMPLE_COUNT)]
    plans = {
        0: (range(SAMPLE_COUNT), lambda s: (10 + s, 20, 10 + s + 40, 110)),
        1: (range(5, SAMPLE_COUNT), lambda s: (100, 40, 130, 90)),
    }
    entries = []
    for track_id, (samples, box_of) in plans.items():
        samples = list(samples)
        records = [motion_record(s, source_indices[s], box_of(s)) for s in samples]
        for record in records:
            record["trackId"] = track_id
        folder = tracks_dir / f"track_{track_id:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "motion.json").write_text(json.dumps({"frames": records}, indent=1))
        entries.append(
            {
                "track": track_id,
                "rawId": track_id,
                "quality": {
                    "samples": len(samples),
                    "coverage": len(samples) / SAMPLE_COUNT,
                    "firstSample": samples[0],
                    "lastSample": samples[-1],
                    "gaps": [],
                    "fullyInFrameFraction": 1.0,
                    "medianHeightFraction": 0.6,
                    "meanOccludedFraction": 0.0,
                    "maxOccludedFraction": 0.0,
                    "meanScore": 0.9,
                    "score": 0.8 - 0.1 * track_id,
                },
                "candidateSamples": [
                    {
                        "sample": s,
                        "sourceIndex": source_indices[s],
                        "box": list(map(float, box_of(s))),
                        "maskBox": list(map(float, box_of(s))),
                        "heightFraction": 0.6,
                        "fullyInFrame": True,
                        "occludedFraction": 0.0,
                        "score": 0.9,
                    }
                    for s in samples[:12]
                ],
                "samples": samples,
            }
        )
    document = {
        "video": str(video),
        "sourceSha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(),
        "sourceFps": SOURCE_FPS,
        "sourceFrames": SOURCE_FRAMES,
        "duration": SOURCE_FRAMES / SOURCE_FPS,
        "fps": SOURCE_FPS,
        "samples": SAMPLE_COUNT,
        "sourceIndices": source_indices,
        "timestamps": [round(index / SOURCE_FPS, 4) for index in source_indices],
        "width": WIDTH,
        "height": HEIGHT,
        "detectionThreshold": 0.15,
        "gate": 0.22,
        "maxGap": 8,
        "minTrackSamples": 8,
        "camerasSha256": None,
        "trackCount": len(entries),
        "discardedShortTracks": 0,
        "maskrcnnDetectionsWithoutPose": 0,
        "tracks": entries,
        "method": "synthetic fixture standing in for the MultiHMR tracker",
        "seconds": 1.0,
    }
    (tracks_dir / "tracks.json").write_text(json.dumps(document, indent=1))
    return tracks_dir


def write_cut_report(root, video):
    """A scripts/shot_cuts.py --json report: `shots` with index/start/end/seconds/tooShort."""
    duration = SOURCE_FRAMES / SOURCE_FPS
    shots = [
        {"index": 0, "start": 0.0, "end": 1.0, "seconds": 1.0, "tooShort": True},
        {"index": 1, "start": 1.0, "end": 3.0, "seconds": 2.0, "tooShort": False},
        {"index": 2, "start": 3.0, "end": duration, "seconds": duration - 3.0, "tooShort": False},
    ]
    report = Path(root) / "cuts.json"
    report.write_text(
        json.dumps(
            {
                "video": str(video),
                "source": {
                    "width": WIDTH,
                    "height": HEIGHT,
                    "fps": SOURCE_FPS,
                    "frames": SOURCE_FRAMES,
                    "duration": duration,
                },
                "threshold": 0.1,
                "minSeconds": 1.5,
                "cuts": [{"time": 1.0, "frame": 10, "cut": True}],
                "cutCount": 1,
                "cleared": [],
                "shots": shots,
                "shotCount": len(shots),
                "continuous": False,
            },
            indent=1,
        )
    )
    return report


PEOPLE_ANSWER = {
    "tracks": [
        {"id": 1, "label": "secondary", "reason": "stands beside the subject throughout"},
        {"id": 0, "label": "main", "reason": "crosses the frame and carries the action"},
    ],
    "untracked": [{"frame": 2, "where": "far left edge", "why": "walks in with the subject"}],
}

SHOTS_ANSWER = {
    "groups": [
        {"id": 0, "place": "one bare room with a bright left wall", "shots": [1, 2]},
    ],
    "shots": [
        {"index": 1, "subjects": "a red marker crossing the frame"},
        {"index": 2, "subjects": "the same marker, further right"},
    ],
}


class JudgeCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "clip.avi"
        if not write_video(self.video):
            self.skipTest("no usable MJPG encoder for cv2.VideoWriter here")
        self.key = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret"})
        self.key.start()

    def tearDown(self):
        self.key.stop()
        self.temporary.cleanup()

    def receipt(self, out):
        return json.loads(vlm_once.receipt_path(out).read_text())


class JudgePeopleTest(JudgeCase):
    def setUp(self):
        super().setUp()
        self.tracks = write_tracks(self.root, self.video)
        self.out = self.root / "people-judge.json"

    def judge(self, transport, out=None, **kwargs):
        return judge_people.judge_people(
            self.tracks, self.video, out or self.out, urlopen=transport, **kwargs
        )

    def test_success_labels_every_track_and_orders_selection_by_prominence(self):
        transport, calls = counting_transport(completion(PEOPLE_ANSWER))
        record = self.judge(transport)

        self.assertEqual(len(calls), 1)
        self.assertEqual(record["trackCount"], 2)
        self.assertEqual([entry["id"] for entry in record["labels"]], [0, 1])
        self.assertEqual(record["labels"][0]["label"], "main")
        self.assertEqual(record["labels"][1]["label"], "secondary")
        # Track 0 is 40x90 over 20 samples, track 1 is 30x50 over 15 samples.
        self.assertEqual(record["selected"], [0, 1])
        self.assertGreater(record["labels"][0]["boxAreaOverTime"], 0.0)
        self.assertEqual(record["unresolved"], [])
        self.assertEqual(record["untracked"][0]["where"], "far left edge")
        self.assertEqual(record["untracked"][0]["frameShown"], 2 in record["frames"])
        self.assertEqual(len(record["framePaths"]), 6)
        self.assertEqual(len(record["frames"]), 6)
        for path in record["framePaths"]:
            self.assertTrue(Path(path).is_file())
            self.assertIsNotNone(cv2.imread(path))
        self.assertIn("opinion", record)
        self.assertEqual(self.receipt(self.out)["status"], "completed")
        self.assertEqual(self.receipt(self.out)["response"]["usage"], USAGE)
        self.assertEqual(self.receipt(self.out)["identity"]["request"]["model"], "gpt-6-astra")

    def test_unresolved_collects_everything_not_main_or_secondary(self):
        answer = {
            "tracks": [
                {"id": 0, "label": "main", "reason": "the subject"},
                {"id": 1, "label": "not_a_person", "reason": "a poster on the wall"},
            ]
        }
        transport, _calls = counting_transport(completion(answer))
        record = self.judge(transport)
        self.assertEqual(record["selected"], [0])
        self.assertEqual([entry["id"] for entry in record["unresolved"]], [1])
        self.assertEqual(record["unresolved"][0]["reason"], "a poster on the wall")
        self.assertEqual(record["untracked"], [])

    def test_completed_result_is_reused_without_a_second_call(self):
        transport, calls = counting_transport(completion(PEOPLE_ANSWER))
        first = self.judge(transport)
        with mock.patch.dict(os.environ, {}, clear=True):
            second = self.judge(must_not_call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second, first)
        self.assertEqual(second, json.loads(self.out.read_text()))

    def test_invalid_answers_fail_the_receipt_keep_the_response_and_write_no_output(self):
        cases = {
            "missing-track": {
                "tracks": [{"id": 0, "label": "main", "reason": "the subject"}],
            },
            "duplicate-track": {
                "tracks": [
                    {"id": 0, "label": "main", "reason": "the subject"},
                    {"id": 0, "label": "secondary", "reason": "again"},
                    {"id": 1, "label": "background", "reason": "passing"},
                ],
            },
            "bad-label": {
                "tracks": [
                    {"id": 0, "label": "protagonist", "reason": "the subject"},
                    {"id": 1, "label": "background", "reason": "passing"},
                ],
            },
            "unknown-track": {
                "tracks": [
                    {"id": 0, "label": "main", "reason": "the subject"},
                    {"id": 1, "label": "background", "reason": "passing"},
                    {"id": 7, "label": "crowd", "reason": "invented"},
                ],
            },
            "no-reason": {
                "tracks": [
                    {"id": 0, "label": "main", "reason": ""},
                    {"id": 1, "label": "background", "reason": "passing"},
                ],
            },
        }
        for name, answer in cases.items():
            with self.subTest(name=name):
                out = self.root / name / "people-judge.json"
                payload = completion(answer)
                transport, calls = counting_transport(payload)
                with self.assertRaises(RuntimeError) as caught:
                    self.judge(transport, out=out)

                self.assertEqual(len(calls), 1)
                self.assertFalse(out.exists())
                self.assertNotIn("sk-test-secret", str(caught.exception))
                receipt = self.receipt(out)
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["errorCategory"], "invalid_response")
                response = receipt["response"]
                self.assertEqual(response["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(response["usage"], USAGE)
                self.assertEqual(response["httpStatus"], 200)
                self.assertEqual(
                    vlm_once.response_raw_path(out).read_bytes(),
                    payload,
                )

                with self.assertRaisesRegex(RuntimeError, "failed"):
                    self.judge(must_not_call, out=out)
                self.assertEqual(len(calls), 1)
                self.assertFalse(out.exists())

    def test_a_clip_that_is_not_the_tracked_video_is_refused(self):
        other = self.root / "other.avi"
        if not write_video(other, frames=SOURCE_FRAMES - 4):
            self.skipTest("no usable MJPG encoder for cv2.VideoWriter here")
        with self.assertRaisesRegex(ValueError, "different file"):
            judge_people.judge_people(self.tracks, other, self.out, urlopen=must_not_call)
        self.assertFalse(vlm_once.receipt_path(self.out).exists())


class JudgeShotsTest(JudgeCase):
    def setUp(self):
        super().setUp()
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed here")
        self.report = write_cut_report(self.root, self.video)
        self.out = self.root / "shot-groups.json"

    def judge(self, transport, out=None, **kwargs):
        return judge_shots.judge_shots(self.report, out or self.out, urlopen=transport, **kwargs)

    def test_success_groups_every_judged_shot_and_skips_the_short_one(self):
        transport, calls = counting_transport(completion(SHOTS_ANSWER))
        record = self.judge(transport)

        self.assertEqual(len(calls), 1)
        self.assertEqual(record["judged"], [1, 2])
        self.assertEqual(record["unjudged"], [])
        self.assertEqual(
            record["groups"],
            [{"id": 0, "place": SHOTS_ANSWER["groups"][0]["place"], "shots": [1, 2]}],
        )
        self.assertEqual([entry["index"] for entry in record["subjects"]], [1, 2])
        self.assertEqual(record["subjectsMissing"], [])
        self.assertEqual(len(record["framePaths"]), 2)
        for path in record["framePaths"]:
            tile = cv2.imread(path)
            self.assertIsNotNone(tile)
            # Two frames side by side, so the tile is wider than it is tall.
            self.assertGreater(tile.shape[1], tile.shape[0])
        self.assertIn("registration", record["proposal"])
        self.assertEqual(self.receipt(self.out)["status"], "completed")

    def test_missing_subjects_are_recorded_rather_than_failing_a_billed_answer(self):
        answer = {
            "groups": [
                {"id": 0, "place": "a bare room", "shots": [1]},
                {"id": 1, "place": "a different bare room", "shots": [2]},
            ],
            "shots": [{"index": 1, "subjects": "a red marker"}],
        }
        transport, _calls = counting_transport(completion(answer))
        record = self.judge(transport)
        self.assertEqual([group["id"] for group in record["groups"]], [0, 1])
        self.assertEqual(record["subjectsMissing"], [2])
        self.assertIsNone(record["subjects"][1]["subjects"])

    def test_the_longest_shots_are_judged_and_the_rest_recorded_as_unjudged(self):
        transport, _calls = counting_transport(
            completion({"groups": [{"id": 0, "place": "a bare room", "shots": [2]}], "shots": []})
        )
        record = self.judge(transport, limit=1)
        self.assertEqual(record["judged"], [2])
        self.assertEqual([entry["index"] for entry in record["unjudged"]], [1])
        self.assertIn("longest", record["unjudged"][0]["reason"])
        self.assertEqual(record["subjectsMissing"], [2])

    def test_invalid_groupings_fail_the_receipt_keep_the_response_and_write_no_output(self):
        cases = {
            "shot-in-two-groups": {
                "groups": [
                    {"id": 0, "place": "a bare room", "shots": [1, 2]},
                    {"id": 1, "place": "the same room again", "shots": [2]},
                ],
                "shots": [],
            },
            "shot-left-out": {
                "groups": [{"id": 0, "place": "a bare room", "shots": [1]}],
                "shots": [],
            },
            "unjudged-shot": {
                "groups": [{"id": 0, "place": "a bare room", "shots": [1, 2, 9]}],
                "shots": [],
            },
            "unnamed-place": {
                "groups": [{"id": 0, "place": "  ", "shots": [1, 2]}],
                "shots": [],
            },
        }
        for name, answer in cases.items():
            with self.subTest(name=name):
                out = self.root / name / "shot-groups.json"
                payload = completion(answer)
                transport, calls = counting_transport(payload)
                with self.assertRaises(RuntimeError):
                    self.judge(transport, out=out)

                self.assertEqual(len(calls), 1)
                self.assertFalse(out.exists())
                receipt = self.receipt(out)
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["errorCategory"], "invalid_response")
                self.assertEqual(receipt["response"]["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(vlm_once.response_raw_path(out).read_bytes(), payload)

                with self.assertRaisesRegex(RuntimeError, "failed"):
                    self.judge(must_not_call, out=out)
                self.assertEqual(len(calls), 1)

    def test_completed_result_is_reused_without_a_second_call(self):
        transport, calls = counting_transport(completion(SHOTS_ANSWER))
        first = self.judge(transport)
        with mock.patch.dict(os.environ, {}, clear=True):
            second = self.judge(must_not_call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second, first)


class VlmOnceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "tile.png"
        self.image.write_bytes(b"pretend png bytes")
        self.out = self.root / "judge.json"
        self.key = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret"})
        self.key.start()

    def tearDown(self):
        self.key.stop()
        self.temporary.cleanup()

    def request(self, transport, out=None, validate=None):
        return vlm_once.request_once(
            out or self.out,
            brief="a fixed brief",
            images=[self.image],
            model=vlm_once.DEFAULT_MODEL,
            validate=validate or (lambda record: record),
            identity_extra={"fixture": "vlm-once"},
            urlopen=transport,
        )

    def test_default_model_and_token_bound_reach_the_request(self):
        seen = []

        def transport(request, **options):
            seen.append((request, options))
            return FakeResponse(completion({"ok": True}))

        record = self.request(transport)
        self.assertEqual(record, {"ok": True})
        body = json.loads(seen[0][0].data)
        self.assertEqual(body["model"], "gpt-6-astra")
        self.assertEqual(body["max_completion_tokens"], vlm_once.MAX_COMPLETION_TOKENS)
        self.assertEqual(body["messages"][0]["content"][0]["text"], "a fixed brief")
        self.assertTrue(
            body["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertEqual(seen[0][1]["timeout"], vlm_once.REQUEST_TIMEOUT_SECONDS)

    def test_http_status_selects_rejection_or_ambiguous_billing(self):
        cases = {
            "rejected": (400, "failed", "http_rejected"),
            "rate-limited": (429, "failed", "http_rejected"),
            "server": (503, "unknown", "http_server"),
            "gateway": (500, "unknown", "http_server"),
        }
        for name, (code, status, category) in cases.items():
            with self.subTest(name=name):
                out = self.root / f"http-{name}" / "judge.json"

                def failing(*_args, _code=code, **_kwargs):
                    raise urllib.error.HTTPError(
                        vlm_once.ENDPOINT,
                        _code,
                        "sk-test-secret raw provider failure",
                        {"x-request-id": "req-error"},
                        None,
                    )

                with self.assertRaises(RuntimeError) as caught:
                    self.request(failing, out=out)

                self.assertNotIn("sk-test-secret", str(caught.exception))
                self.assertNotIn("raw provider", str(caught.exception))
                self.assertFalse(out.exists())
                self.assertFalse(vlm_once.response_raw_path(out).exists())
                receipt = json.loads(vlm_once.receipt_path(out).read_text())
                self.assertEqual(receipt["status"], status)
                self.assertEqual(receipt["errorCategory"], category)
                self.assertEqual(receipt["response"]["httpStatus"], code)
                self.assertEqual(receipt["response"]["requestId"], "req-error")
                self.assertEqual(vlm_once.receipt_status(out), status)

                with self.assertRaises(RuntimeError):
                    self.request(must_not_call, out=out)

    def test_recovers_from_saved_bytes_with_no_key_and_no_network(self):
        payload = completion({"ok": True})
        transport, calls = counting_transport(payload)
        with (
            mock.patch.object(vlm_once, "_parse_response", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.request(transport)

        receipt = json.loads(vlm_once.receipt_path(self.out).read_text())
        self.assertEqual(receipt["status"], "response_received")
        self.assertFalse(self.out.exists())
        self.assertEqual(vlm_once.response_raw_path(self.out).read_bytes(), payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            recovered = self.request(must_not_call)

        self.assertEqual(len(calls), 1)
        self.assertEqual(recovered, {"ok": True})
        self.assertEqual(recovered, json.loads(self.out.read_text()))
        completed = json.loads(vlm_once.receipt_path(self.out).read_text())
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["response"]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_recovery_refuses_when_the_saved_bytes_are_missing_or_changed(self):
        cases = {
            "missing": lambda path: path.unlink(),
            "changed": lambda path: path.write_bytes(completion({"ok": True}) + b" "),
        }
        for name, damage in cases.items():
            with self.subTest(name=name):
                out = self.root / f"recover-{name}" / "judge.json"
                transport, _calls = counting_transport(completion({"ok": True}))
                with (
                    mock.patch.object(vlm_once, "_parse_response", side_effect=KeyboardInterrupt),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    self.request(transport, out=out)
                damage(vlm_once.response_raw_path(out))

                with self.assertRaisesRegex(RuntimeError, "do not resubmit"):
                    self.request(must_not_call, out=out)
                self.assertFalse(out.exists())

    def test_identity_ignores_absolute_paths_recorded_on_another_machine(self):
        transport, _calls = counting_transport(completion({"ok": True}))
        self.request(transport)
        path = vlm_once.receipt_path(self.out)
        receipt = json.loads(path.read_text())
        receipt["identity"]["images"][0]["path"] = "/Users/aayan/wander/tile.png"
        receipt["output"]["path"] = "/Users/aayan/wander/judge.json"
        path.write_text(json.dumps(receipt))

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.request(must_not_call), {"ok": True})

    def test_changed_inputs_ask_for_a_new_out(self):
        transport, _calls = counting_transport(completion({"ok": True}))
        self.request(transport)
        self.image.write_bytes(b"different tile bytes entirely")
        with self.assertRaisesRegex(ValueError, "new --out"):
            self.request(must_not_call)

    def test_a_pending_receipt_never_resubmits(self):
        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.request(interrupted)
        self.assertEqual(vlm_once.receipt_status(self.out), "pending")
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.request(must_not_call)


if __name__ == "__main__":
    unittest.main()
