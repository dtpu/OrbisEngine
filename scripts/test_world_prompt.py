import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import stat
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import world_prompt

USAGE = {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}
FRAME_BYTES = b"sampled frame pixels"


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


def completion(content=None, *, usage=USAGE):
    if content is None:
        content = {
            "space": "a test room",
            "architecture": "four visible walls",
            "text_prompt": "A test room. Do not add unseen fixtures.",
        }
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "model": "gpt-6-astra",
            "choices": [
                {
                    "message": {"content": json.dumps(content)},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    ).encode()


def sampling(frames, paths, *, requested=None, undecoded=(), times=None):
    """The shape `world_prompt.sample` returns: ordinals decoded, plus what was asked for."""
    selected = [int(frame) for frame in frames]
    return {
        "requested": [int(frame) for frame in (selected if requested is None else requested)],
        "selected": selected,
        "undecoded": [int(frame) for frame in undecoded],
        "paths": [str(path) for path in paths],
        "containerTimeMs": list(times) if times is not None else [None] * len(selected),
    }


def must_not_call(*_args, **_kwargs):
    raise AssertionError("transport must not be called")


def frame_level(ordinal):
    """Each synthetic frame is a flat grey whose level encodes its ordinal."""
    return 20 + ordinal * 15


def write_synthetic_video(path, count=12, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    if not writer.isOpened():
        return False
    for ordinal in range(count):
        writer.write(np.full((size[1], size[0], 3), frame_level(ordinal), dtype=np.uint8))
    writer.release()
    return True


class StubCapture:
    """A container that advertises more frames than its stream can actually decode."""

    def __init__(self, decodable, claimed):
        self.decodable = decodable
        self.claimed = claimed
        self.ordinal = -1

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.claimed)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float(self.ordinal * 40)
        return 0.0

    def grab(self):
        if self.ordinal + 1 >= self.decodable:
            return False
        self.ordinal += 1
        return True

    def retrieve(self):
        return True, np.full((8, 8, 3), frame_level(self.ordinal), dtype=np.uint8)

    def release(self):
        return None


def race_sampling(_clip, frames, _n, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = out_dir / "frame_0003.png"
    image.write_bytes(FRAME_BYTES)
    return sampling(frames or [3], [image], times=[125.0])


def race_child(index, clip, out, marker_dir, result_dir):
    """One racing process: mark every transport call, then record what it returned."""
    world_prompt.sample = race_sampling
    os.environ["OPENAI_API_KEY"] = "sk-test-secret"

    def slow_transport(*_args, **_kwargs):
        marker = Path(marker_dir) / f"call-{os.getpid()}"
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        time.sleep(0.6)
        return FakeResponse(completion())

    try:
        record = world_prompt.generate_prompt(
            clip, None, 1, out, "gpt-6-astra", urlopen=slow_transport
        )
        outcome = {"result": "returned", "text": record["text_prompt"]}
    except RuntimeError as error:
        outcome = {"result": "refused", "error": str(error)}
    except BaseException as error:  # noqa: BLE001
        outcome = {"result": "error", "error": repr(error)}
    Path(result_dir, f"child-{index}.json").write_text(json.dumps(outcome))


class WorldPromptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clip = self.root / "source.mp4"
        self.clip.write_bytes(b"original source video")
        self.image = self.root / "sample.png"
        self.image.write_bytes(FRAME_BYTES)
        self.out = self.root / "prompt.json"
        self.key = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret"})
        self.key.start()
        self.real_sample = world_prompt.sample
        self.sampler = mock.patch.object(
            world_prompt,
            "sample",
            side_effect=lambda _clip, frames, _n, _out_dir: sampling(
                frames or [3], [self.image], times=[125.0]
            ),
        )
        self.sampler.start()

    def tearDown(self):
        self.sampler.stop()
        self.key.stop()
        self.temporary.cleanup()

    def generate(self, transport, **kwargs):
        return world_prompt.generate_prompt(
            self.clip,
            None,
            1,
            self.out,
            "gpt-6-astra",
            urlopen=transport,
            **kwargs,
        )

    def crash_after_response(self, payload):
        """Leave a `response_received` receipt exactly as an interrupted run would."""
        calls = []

        def transport(*_args, **_kwargs):
            calls.append(True)
            return FakeResponse(payload)

        with (
            mock.patch.object(world_prompt, "_parse_response", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.generate(transport)
        return calls

    def test_success_is_bounded_and_records_identity_response_and_usage(self):
        calls = []

        def transport(request, **options):
            calls.append((request, options))
            return FakeResponse(completion())

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = self.generate(transport)

        self.assertEqual(len(calls), 1)
        body = json.loads(calls[0][0].data)
        self.assertEqual(body["model"], "gpt-6-astra")
        self.assertEqual(body["max_completion_tokens"], 4096)
        self.assertEqual(calls[0][1]["timeout"], world_prompt.REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(result["text_prompt"], "A test room. Do not add unseen fixtures.")
        self.assertEqual(result["clip"], str(self.clip))
        self.assertEqual(result["frames"], [3])
        self.assertEqual(result["framePaths"], [str(self.image)])
        self.assertEqual(result["requestedFrames"], [3])
        self.assertEqual(result["undecodedFrames"], [])
        self.assertEqual(result["frameTimeMs"], [125.0])
        self.assertIn("approximate", result["frameTimeNote"])

        reported = stderr.getvalue().strip().splitlines()
        self.assertEqual(len(reported), 1)
        for expected in ("prompt=120", "completion=40", "total=160", "chatcmpl-test"):
            self.assertIn(expected, reported[0])

        receipt = json.loads(world_prompt.receipt_path(self.out).read_text())
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["response"]["usage"], USAGE)
        self.assertEqual(receipt["response"]["requestId"], "req-test")
        self.assertEqual(receipt["identity"]["source"]["sha256"], world_prompt._sha256(self.clip))
        self.assertEqual(
            receipt["identity"]["frameImages"][0]["sha256"], world_prompt._sha256(self.image)
        )
        self.assertEqual(receipt["identity"]["request"]["maxCompletionTokens"], 4096)
        self.assertEqual(receipt["identity"]["requestedFrames"], [3])
        self.assertEqual(receipt["identity"]["selectedFrames"], [3])

    def test_completed_result_reuses_without_call_and_rejects_mutations(self):
        self.generate(lambda *_args, **_kwargs: FakeResponse(completion()))

        blocked_calls = []

        def should_not_call(*_args, **_kwargs):
            blocked_calls.append(True)
            raise AssertionError("transport must not be called")

        with mock.patch.dict(os.environ, {}, clear=True):
            reused = self.generate(should_not_call)
        self.assertEqual(reused, json.loads(self.out.read_text()))
        self.assertEqual(blocked_calls, [])

        original_source = self.clip.read_bytes()
        self.clip.write_bytes(b"mutated source")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.generate(should_not_call)
        self.clip.write_bytes(original_source)

        with (
            mock.patch.object(world_prompt, "BRIEF", world_prompt.BRIEF + " changed"),
            self.assertRaisesRegex(ValueError, "does not match"),
        ):
            self.generate(should_not_call)

        self.out.write_text(self.out.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "was modified"):
            self.generate(should_not_call)
        self.assertEqual(blocked_calls, [])

    def test_http_timeout_and_malformed_results_make_one_call_and_no_output(self):
        cases = {
            "http": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                urllib.error.HTTPError(
                    "https://api.openai.com/v1/chat/completions",
                    500,
                    "sk-test-secret raw provider failure",
                    {},
                    None,
                )
            ),
            "timeout": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TimeoutError("sk-test-secret raw timeout")
            ),
            "bad-json": lambda *_args, **_kwargs: FakeResponse(b"{not-json"),
            "empty": lambda *_args, **_kwargs: FakeResponse(completion({"text_prompt": ""})),
            "no-usage": lambda *_args, **_kwargs: FakeResponse(completion(usage=None)),
        }
        for name, failing_transport in cases.items():
            with self.subTest(name=name):
                case_root = self.root / name
                case_root.mkdir()
                self.out = case_root / "prompt.json"
                calls = []

                def counted(*args, _calls=calls, _transport=failing_transport, **kwargs):
                    _calls.append(True)
                    return _transport(*args, **kwargs)

                with self.assertRaises(RuntimeError) as caught:
                    self.generate(counted)
                self.assertEqual(calls, [True])
                self.assertFalse(self.out.exists())
                receipt_text = world_prompt.receipt_path(self.out).read_text()
                self.assertNotIn("sk-test-secret", str(caught.exception))
                self.assertNotIn("raw provider", str(caught.exception))
                self.assertNotIn("sk-test-secret", receipt_text)
                self.assertIn(json.loads(receipt_text)["status"], {"failed", "unknown"})

    def test_interrupted_pending_and_failed_receipts_never_resubmit(self):
        calls = []

        def interrupted(*_args, **_kwargs):
            calls.append("interrupted")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.generate(interrupted)
        pending = json.loads(world_prompt.receipt_path(self.out).read_text())
        self.assertEqual(pending["status"], "pending")

        def should_not_call(*_args, **_kwargs):
            calls.append("retried")
            return FakeResponse(completion())

        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.generate(should_not_call)
        self.assertEqual(calls, ["interrupted"])

        failed_out = self.root / "failed" / "prompt.json"
        self.out = failed_out
        with self.assertRaises(RuntimeError):
            self.generate(lambda *_args, **_kwargs: FakeResponse(b"not-json"))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.generate(should_not_call)
        self.assertEqual(calls, ["interrupted"])

    def test_billed_response_is_persisted_before_its_content_is_trusted(self):
        blank = completion({"space": "a test room", "text_prompt": "   "})
        cases = {
            "unusable-content": (blank, "chatcmpl-test", "stop", USAGE),
            "unparseable-body": (b"<html>bad gateway</html>", None, None, None),
        }
        for name, (payload, identifier, finish, usage) in cases.items():
            with self.subTest(name=name):
                case_root = self.root / f"persist-{name}"
                case_root.mkdir()
                self.out = case_root / "prompt.json"

                with self.assertRaises(RuntimeError):
                    self.generate(lambda *_args, _body=payload, **_kwargs: FakeResponse(_body))

                raw_path = world_prompt.response_raw_path(self.out)
                self.assertEqual(raw_path.read_bytes(), payload)
                self.assertEqual(stat.S_IMODE(raw_path.stat().st_mode), 0o600)
                self.assertFalse(self.out.exists())

                receipt = json.loads(world_prompt.receipt_path(self.out).read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["errorCategory"], "invalid_response")
                response = receipt["response"]
                self.assertEqual(response["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(response["bytes"], len(payload))
                self.assertEqual(response["httpStatus"], 200)
                self.assertEqual(response["requestId"], "req-test")
                self.assertEqual(response["id"], identifier)
                self.assertEqual(response["finishReason"], finish)
                self.assertEqual(response["usage"], usage)

    def test_response_received_completes_from_saved_bytes_without_network(self):
        payload = completion()
        calls = self.crash_after_response(payload)

        receipt = json.loads(world_prompt.receipt_path(self.out).read_text())
        self.assertEqual(receipt["status"], "response_received")
        self.assertFalse(self.out.exists())

        with mock.patch.dict(os.environ, {}, clear=True):
            recovered = self.generate(must_not_call)

        self.assertEqual(calls, [True])
        self.assertEqual(recovered["text_prompt"], "A test room. Do not add unseen fixtures.")
        self.assertEqual(recovered["frames"], [3])
        self.assertEqual(recovered["framePaths"], [str(self.image)])
        self.assertEqual(recovered, json.loads(self.out.read_text()))
        completed = json.loads(world_prompt.receipt_path(self.out).read_text())
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["response"]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_response_received_refuses_when_saved_bytes_are_missing_or_changed(self):
        cases = {
            "missing": lambda path: path.unlink(),
            "changed": lambda path: path.write_bytes(completion() + b" "),
        }
        for name, damage in cases.items():
            with self.subTest(name=name):
                case_root = self.root / f"recover-{name}"
                case_root.mkdir()
                self.out = case_root / "prompt.json"
                self.crash_after_response(completion())
                damage(world_prompt.response_raw_path(self.out))

                with self.assertRaisesRegex(RuntimeError, "do not resubmit"):
                    self.generate(must_not_call)
                self.assertFalse(self.out.exists())
                receipt = json.loads(world_prompt.receipt_path(self.out).read_text())
                self.assertEqual(receipt["status"], "response_received")

    def test_changed_source_asks_for_a_new_out(self):
        self.generate(lambda *_args, **_kwargs: FakeResponse(completion()))
        self.clip.write_bytes(b"a different source video entirely")
        with self.assertRaisesRegex(ValueError, r"new --out"):
            self.generate(must_not_call)

    def test_http_status_selects_rejection_or_ambiguous_billing(self):
        cases = {
            "rejected": (400, "failed", "http_rejected"),
            "server": (503, "unknown", "http_server"),
        }
        for name, (code, status, category) in cases.items():
            with self.subTest(name=name):
                case_root = self.root / f"http-{name}"
                case_root.mkdir()
                self.out = case_root / "prompt.json"

                def failing(*_args, _code=code, **_kwargs):
                    raise urllib.error.HTTPError(
                        "https://api.openai.com/v1/chat/completions",
                        _code,
                        "sk-test-secret raw provider failure",
                        {"x-request-id": "req-error"},
                        None,
                    )

                with self.assertRaises(RuntimeError) as caught:
                    self.generate(failing)

                self.assertNotIn("sk-test-secret", str(caught.exception))
                self.assertNotIn("raw provider", str(caught.exception))
                self.assertFalse(self.out.exists())
                self.assertFalse(world_prompt.response_raw_path(self.out).exists())
                receipt = json.loads(world_prompt.receipt_path(self.out).read_text())
                self.assertEqual(receipt["status"], status)
                self.assertEqual(receipt["errorCategory"], category)
                self.assertEqual(receipt["response"]["httpStatus"], code)
                self.assertEqual(receipt["response"]["requestId"], "req-error")

                with self.assertRaises(RuntimeError):
                    self.generate(must_not_call)

    def test_http_error_keeps_enumerated_provider_codes_but_never_its_message(self):
        body = json.dumps(
            {
                "error": {
                    "message": "quota exceeded for key sk-test-secret",
                    "type": "insufficient_quota",
                    "code": "Not An Identifier sk-test-secret",
                }
            }
        ).encode()

        def failing(*_args, **_kwargs):
            raise urllib.error.HTTPError(
                "https://api.openai.com/v1/chat/completions",
                429,
                "Too Many Requests",
                {"x-request-id": "req-error"},
                io.BytesIO(body),
            )

        with self.assertRaises(RuntimeError) as caught:
            self.generate(failing)
        receipt_text = world_prompt.receipt_path(self.out).read_text()
        self.assertNotIn("sk-test-secret", receipt_text + str(caught.exception))
        response = json.loads(receipt_text)["response"]
        self.assertEqual(response["errorType"], "insufficient_quota")
        self.assertNotIn("errorCode", response)

    def test_existing_receipt_samples_into_scratch_and_keeps_recorded_frames(self):
        directories = []

        def recording_sampler(_clip, frames, _n, out_dir):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            directories.append(out_dir)
            image = out_dir / "frame_0003.png"
            image.write_bytes(FRAME_BYTES)
            return sampling(frames or [3], [image], times=[125.0])

        frames_dir = self.out.parent / (self.out.stem + "-frames")
        with mock.patch.object(world_prompt, "sample", recording_sampler):
            self.generate(lambda *_args, **_kwargs: FakeResponse(completion()))
            self.assertEqual(directories, [frames_dir])
            recorded = frames_dir / "frame_0003.png"
            stamp = recorded.stat().st_mtime_ns

            with mock.patch.dict(os.environ, {}, clear=True):
                reused = self.generate(must_not_call)

        self.assertEqual(reused["text_prompt"], "A test room. Do not add unseen fixtures.")
        self.assertEqual(len(directories), 2)
        self.assertNotEqual(directories[1], frames_dir)
        self.assertFalse(directories[1].exists())
        self.assertEqual([path.name for path in frames_dir.iterdir()], ["frame_0003.png"])
        self.assertEqual(recorded.stat().st_mtime_ns, stamp)

    def test_identity_ignores_absolute_paths_recorded_on_another_machine(self):
        self.generate(lambda *_args, **_kwargs: FakeResponse(completion()))

        receipt_file = world_prompt.receipt_path(self.out)
        receipt = json.loads(receipt_file.read_text())
        receipt["identity"]["source"]["path"] = "/Users/aayan/wander/source.mp4"
        receipt["identity"]["frameImages"][0]["path"] = "/Users/aayan/wander/frames/f.png"
        receipt["output"]["path"] = "/Users/aayan/wander/prompt.json"
        receipt_file.write_text(json.dumps(receipt))

        with mock.patch.dict(os.environ, {}, clear=True):
            reused = self.generate(must_not_call)
        self.assertEqual(reused, json.loads(self.out.read_text()))

    def test_sample_decodes_exact_ordinals_from_a_real_video(self):
        video = self.root / "synthetic.avi"
        if not write_synthetic_video(video):
            self.skipTest("no usable MJPG encoder for cv2.VideoWriter here")

        decoded = self.real_sample(video, [0, 4, 9], 3, self.root / "decoded")
        self.assertEqual(decoded["requested"], [0, 4, 9])
        self.assertEqual(decoded["selected"], [0, 4, 9])
        self.assertEqual(decoded["undecoded"], [])
        for ordinal, path, moment in zip(
            [0, 4, 9], decoded["paths"], decoded["containerTimeMs"], strict=True
        ):
            self.assertEqual(Path(path).name, f"frame_{ordinal:04d}.png")
            measured = float(cv2.imread(path).mean())
            self.assertLess(abs(measured - frame_level(ordinal)), 4.0, f"ordinal {ordinal}")
            # Approximate container time only: the clip is written at 10 fps.
            self.assertLess(abs(moment - ordinal * 100.0), 1.0, f"ordinal {ordinal}")

        spread = self.real_sample(video, None, 3, self.root / "spread")
        self.assertEqual(spread["requested"], [0, 6, 11])
        self.assertEqual(spread["selected"], [0, 6, 11])
        self.assertEqual(spread["undecoded"], [])

        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            self.real_sample(video, [1, 99], 2, self.root / "beyond")

    def test_default_spread_records_a_shortfall_instead_of_unread_frames(self):
        with mock.patch.object(world_prompt.cv2, "VideoCapture", lambda _path: StubCapture(5, 20)):
            short = self.real_sample(self.root / "absent.mp4", None, 4, self.root / "short")
        self.assertEqual(short["requested"], [0, 6, 13, 19])
        self.assertEqual(short["selected"], [0])
        self.assertEqual(short["undecoded"], [6, 13, 19])
        self.assertEqual(len(short["paths"]), 1)
        self.assertEqual(len(short["containerTimeMs"]), 1)

    def test_two_processes_racing_one_out_submit_exactly_once(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("the fork start method is unavailable here")
        context = multiprocessing.get_context("fork")
        marker_dir = self.root / "markers"
        result_dir = self.root / "results"
        marker_dir.mkdir()
        result_dir.mkdir()
        self.out = self.root / "race" / "prompt.json"

        children = [
            context.Process(
                target=race_child,
                args=(index, self.clip, self.out, marker_dir, result_dir),
            )
            for index in range(2)
        ]
        for child in children:
            child.start()
        for child in children:
            child.join(90)
        self.assertEqual([child.exitcode for child in children], [0, 0])

        markers = sorted(path.name for path in marker_dir.iterdir())
        self.assertEqual(len(markers), 1, f"exactly one paid submission is allowed: {markers}")
        outcomes = [json.loads(path.read_text()) for path in sorted(result_dir.iterdir())]
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(any(outcome["result"] == "returned" for outcome in outcomes))
        for outcome in outcomes:
            self.assertIn(outcome["result"], {"returned", "refused"}, outcome)
            if outcome["result"] == "returned":
                self.assertEqual(outcome["text"], "A test room. Do not add unseen fixtures.")
        receipt = json.loads(world_prompt.receipt_path(self.out).read_text())
        self.assertEqual(receipt["status"], "completed")


if __name__ == "__main__":
    unittest.main()
