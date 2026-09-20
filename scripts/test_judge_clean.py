"""Clean-judge tests. Every transport here is a fake; nothing in this file reaches a network.

The synthetic run directory mirrors what scripts/run_clip.py leaves behind before its credit gate:
`clean.json` as worker/modal_clean_video.py writes it (`indices`, `keptIndices`, `peopleMask`,
`dilate`, `bottomExtra`, `maskFraction`, the foreground instance counts), the bit-packed
`masks.npz` that stage saves, `state.json` naming the source clip in its `_run` record, and the six
`review/clean_NNNN.png` frames `Pipeline.review()` pulls out of the cleaned video.

Every source frame carries a different background and a person rectangle in a different place, so a
composite that pairs the wrong original with a cleaned frame cannot match byte for byte.
"""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import cv2
import judge_clean
import numpy as np
import vlm_once

USAGE = {"prompt_tokens": 400, "completion_tokens": 90, "total_tokens": 490}

WIDTH, HEIGHT = 160, 120
SOURCE_FPS = 10.0
SOURCE_FRAMES = 40
CLEAN_FPS = 5.0
CLEAN_FRAMES = 20
REVIEW_ORDINALS = (0, 3, 7, 11, 15, 19)
PERSON_TOP, PERSON_BOTTOM = 30, 100


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


def background(ordinal):
    """A striped, ordinal-dependent plate, so every frame's background is its own."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    columns = (np.arange(WIDTH) + 3 * ordinal) % 24
    image[:, :, 0] = np.where(columns < 12, 40, 190)[None, :]
    image[:, :, 1] = (np.arange(HEIGHT)[:, None] * 2 % 200).astype(np.uint8)
    image[:, :, 2] = np.where(columns < 6, 220, 60)[None, :]
    return image


def person_box(ordinal):
    left = 10 + 3 * ordinal
    return left, PERSON_TOP, min(left + 26, WIDTH - 1), PERSON_BOTTOM


def source_frame(ordinal):
    image = background(ordinal)
    x0, y0, x1, y1 = person_box(ordinal)
    cv2.rectangle(image, (x0, y0), (x1, y1), (245, 245, 245), -1)
    cv2.circle(image, ((x0 + x1) // 2, y0 + 6), 6, (200, 180, 160), -1)
    return image


def write_video(path, frames=SOURCE_FRAMES):
    """A short MJPG/AVI clip whose frames all differ, so a sequential decode has work to do."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), SOURCE_FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        return False
    for ordinal in range(frames):
        writer.write(source_frame(ordinal))
    writer.release()
    return True


def mask_for(source_ordinal):
    """The clean stage's person mask: the box, grown the way a dilation grows it."""
    x0, y0, x1, y1 = person_box(source_ordinal)
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[max(0, y0 - 4) : min(HEIGHT, y1 + 5), max(0, x0 - 4) : min(WIDTH, x1 + 5)] = True
    return mask


def make_run(
    root,
    video,
    name="clip-v1",
    *,
    people_mask="foreground",
    moved=False,
    with_masks=True,
    review_ordinals=REVIEW_ORDINALS,
):
    """A run directory as it stands at the credit gate, with its cleaned review frames."""
    run_dir = Path(root) / ".context" / "run" / name
    (run_dir / "review").mkdir(parents=True, exist_ok=True)
    indices = [2 * index for index in range(CLEAN_FRAMES)]
    originals = judge_clean.read_frames(video, indices)

    masks = np.zeros((CLEAN_FRAMES, HEIGHT, WIDTH), bool)
    for resampled, source_ordinal in enumerate(indices):
        masks[resampled] = mask_for(source_ordinal)
    if with_masks:
        np.savez_compressed(
            run_dir / "masks.npz",
            masks=np.packbits(masks, axis=-1),
            shape=np.array([CLEAN_FRAMES, HEIGHT, WIDTH]),
            indices=np.array(indices),
        )

    for ordinal in review_ordinals:
        source_ordinal = indices[ordinal]
        cleaned = originals[source_ordinal].copy()
        plate = background(source_ordinal)
        mask = masks[ordinal]
        cleaned[mask] = plate[mask]
        cv2.imwrite(str(run_dir / "review" / f"clean_{ordinal:04d}.png"), cleaned)

    (run_dir / "clean.json").write_text(
        json.dumps(
            {
                "seconds": 120.0,
                "error": None,
                "frames": CLEAN_FRAMES,
                "inpainted": CLEAN_FRAMES,
                "indices": indices,
                "keptIndices": list(range(CLEAN_FRAMES)),
                "sourceFps": SOURCE_FPS,
                "sourceSha256": hashlib.sha256(Path(video).read_bytes()).hexdigest(),
                "moved": moved,
                "peopleMask": people_mask,
                "foregroundSelected": 24 if people_mask == "foreground" else None,
                "foregroundRejectedSmall": 9 if people_mask == "foreground" else None,
                "fps": CLEAN_FPS,
                "width": WIDTH,
                "height": HEIGHT,
                "dilate": 20,
                "bottomExtra": 40,
                "maskFraction": float(masks.mean()),
                "framesWithPeople": CLEAN_FRAMES,
            },
            indent=1,
        )
    )
    (run_dir / "state.json").write_text(
        json.dumps({"stages": {"_run": {"clip": str(video), "name": name, "fps": CLEAN_FPS}}})
    )
    return run_dir


def frame_answer(index, **overrides):
    answer = {
        "index": index,
        "subjects_removed": "all",
        "subjects_remaining": "",
        "floating_remnants": [],
        "edge_people": [],
        "background_destroyed": "none",
        "background_note": "",
        "smear_severity": "none",
        "smear_fraction": 0,
        "non_human_subject_left": "none",
        "verdict": "pass",
        "reason": "the plate reads as the empty room",
    }
    answer.update(overrides)
    return answer


def answer(frames=None, overall=None, indices=REVIEW_ORDINALS):
    return {
        "frames": frames if frames is not None else [frame_answer(index) for index in indices],
        "overall": overall
        or {
            "verdict": "pass",
            "recommended_action": "none",
            "reason": "every sampled frame is empty and intact",
        },
    }


class CleanJudgeCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "clip.avi"
        if not write_video(self.video):
            self.skipTest("no usable MJPG encoder for cv2.VideoWriter here")
        self.run_dir = make_run(self.root, self.video)
        self.out = self.run_dir / "clean-judge.json"
        self.key = mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret"})
        self.key.start()

    def tearDown(self):
        self.key.stop()
        self.temporary.cleanup()

    def receipt(self, out):
        return json.loads(vlm_once.receipt_path(out).read_text())

    def judge(self, transport, out=None, run_dir=None, **kwargs):
        return judge_clean.judge_clean(
            run_dir or self.run_dir, out or self.out, urlopen=transport, **kwargs
        )


class VerdictTest(CleanJudgeCase):
    def test_a_clean_plate_passes_and_records_what_was_measured(self):
        transport, calls = counting_transport(completion(answer()))
        record = self.judge(transport)

        self.assertEqual(len(calls), 1)
        self.assertEqual(record["schema"], "wander.clean-judge/1")
        self.assertEqual(record["verdict"], "pass")
        self.assertEqual(record["recommendedAction"], "none")
        self.assertFalse(record["autoRetryable"])
        self.assertEqual(record["criticalFailures"], [])
        self.assertEqual(record["passShare"], 1.0)
        self.assertEqual(record["passThreshold"], 0.75)
        self.assertEqual(record["framesJudged"], list(REVIEW_ORDINALS))
        self.assertEqual(record["calibration"], "uncalibrated")
        self.assertIn("not proof of visual quality", record["opinion"])
        self.assertIn("75%", record["decisionRules"])
        self.assertEqual(record["clean"]["peopleMask"], "foreground")
        self.assertEqual(record["clean"]["dilate"], 20)
        self.assertEqual(record["clean"]["foregroundSelected"], 24)
        self.assertEqual(record["model"], "gpt-6-astra")
        self.assertEqual(self.receipt(self.out)["status"], "completed")

        for frame in record["frames"]:
            self.assertTrue(Path(frame["compositePath"]).is_file())
            self.assertIsNotNone(cv2.imread(frame["compositePath"]))
            signals = frame["signals"]
            # Only the masked person region was repainted in the fixture.
            self.assertGreater(signals["changedFraction"], 0.0)
            self.assertEqual(signals["changedOutsideMaskFraction"], 0.0)
            self.assertEqual(signals["meanAbsDifferenceOutsideMask"], 0.0)
            self.assertGreater(signals["meanAbsDifferenceInsideMask"], 0.0)
            self.assertIsNotNone(signals["blurRatioInChangedRegion"])
            self.assertIsNotNone(signals["blurRatioAgainstPlate"])
        self.assertGreater(record["signals"]["meanChangedFraction"], 0.0)
        self.assertEqual(record["signals"]["maxChangedOutsideMaskFraction"], 0.0)
        self.assertEqual(record["signals"]["framesWithMask"], len(REVIEW_ORDINALS))
        self.assertIn("Nothing here changes the verdict", record["signalsNote"])

    def test_each_recommended_action_comes_back_with_the_flags_it_means(self):
        cases = {
            "increase_dilation": ("foreground", ["--dilate", "40", "--bottom-extra", "80"]),
            "switch_to_semantic_mask": ("foreground", ["--people-mask", "semantic"]),
            "switch_to_foreground_mask": ("semantic", ["--people-mask", "foreground"]),
            "enable_moved_mask": ("semantic", ["--moved-mask"]),
        }
        for action, (mode, flags) in cases.items():
            with self.subTest(action=action):
                run_dir = make_run(self.root / action, self.video, people_mask=mode)
                out = run_dir / "clean-judge.json"
                frames = [
                    frame_answer(
                        index,
                        floating_remnants=["yellow glove"],
                        verdict="retry",
                        reason="a glove hangs where his hand was",
                    )
                    if position < 3
                    else frame_answer(index)
                    for position, index in enumerate(REVIEW_ORDINALS)
                ]
                payload = completion(
                    answer(
                        frames=frames,
                        overall={
                            "verdict": "retry",
                            "recommended_action": action,
                            "reason": "held objects survive the mask",
                        },
                    )
                )
                transport, calls = counting_transport(payload)
                record = self.judge(transport, out=out, run_dir=run_dir)

                self.assertEqual(len(calls), 1)
                self.assertEqual(record["verdict"], "retry")
                self.assertEqual(record["recommendedAction"], action)
                self.assertEqual(record["actionFlags"], flags)
                self.assertFalse(record["actionIsNoOp"])
                self.assertTrue(record["autoRetryable"])
                self.assertEqual(record["actionReason"], "held objects survive the mask")
                self.assertEqual(record["passShare"], 0.5)
                self.assertEqual(record["passingFrames"], list(REVIEW_ORDINALS[3:]))

    def test_a_change_that_is_already_the_current_setting_is_not_automatic(self):
        payload = completion(
            answer(
                frames=[
                    frame_answer(index, verdict="retry", reason="a faint outline survives")
                    for index in REVIEW_ORDINALS
                ],
                overall={
                    "verdict": "retry",
                    "recommended_action": "switch_to_foreground_mask",
                    "reason": "the stands were erased",
                },
            )
        )
        transport, _calls = counting_transport(payload)
        record = self.judge(transport)
        self.assertEqual(record["verdict"], "retry")
        self.assertTrue(record["actionIsNoOp"])
        self.assertFalse(record["autoRetryable"])
        self.assertEqual(record["actionFlags"], [])

    def test_one_critical_frame_beats_a_five_in_six_pass_rate(self):
        cases = {
            "subject-left": {
                "subjects_removed": "none",
                "subjects_remaining": "the running man, centre frame",
                "verdict": "fail",
                "reason": "the runner was never removed",
            },
            "background-destroyed": {
                "background_destroyed": "major",
                "background_note": "the crowd in the stands is a grey wall",
                "verdict": "fail",
                "reason": "the stands were erased",
            },
            "non-human-subject": {
                "non_human_subject_left": "the costumed mascot, left of frame",
                "verdict": "fail",
                "reason": "the mascot is still standing there",
            },
        }
        for name, damage in cases.items():
            with self.subTest(case=name):
                run_dir = make_run(self.root / name, self.video)
                out = run_dir / "clean-judge.json"
                frames = [frame_answer(index) for index in REVIEW_ORDINALS[:-1]]
                frames.append(frame_answer(REVIEW_ORDINALS[-1], **damage))
                payload = completion(
                    answer(
                        frames=frames,
                        overall={
                            "verdict": "retry",
                            "recommended_action": "switch_to_semantic_mask",
                            "reason": "the detector missed a subject",
                        },
                    )
                )
                transport, _calls = counting_transport(payload)
                record = self.judge(transport, out=out, run_dir=run_dir)

                # Five of six frames pass, which is over the 75% share, and it does not matter.
                self.assertEqual(record["passShare"], round(5 / 6, 4))
                self.assertGreater(record["passShare"], record["passThreshold"])
                self.assertEqual(record["verdict"], "retry")
                self.assertEqual(
                    record["criticalFailures"],
                    [{"index": REVIEW_ORDINALS[-1], "kinds": expected_kinds(damage)}],
                )

    def test_a_critical_frame_with_nothing_left_to_change_fails(self):
        frames = [frame_answer(index) for index in REVIEW_ORDINALS[:-1]]
        frames.append(
            frame_answer(
                REVIEW_ORDINALS[-1],
                background_destroyed="major",
                background_note="the stands are a grey wall",
                verdict="fail",
                reason="the crowd was erased",
            )
        )
        payload = completion(
            answer(
                frames=frames,
                overall={
                    "verdict": "pass",
                    "recommended_action": "none",
                    "reason": "the subject is gone from every frame",
                },
            )
        )
        transport, _calls = counting_transport(payload)
        record = self.judge(transport)
        self.assertEqual(record["verdict"], "fail")
        self.assertEqual(record["recommendedAction"], "none")
        self.assertFalse(record["autoRetryable"])

    def test_the_models_overall_verdict_can_only_make_the_result_worse(self):
        payload = completion(
            answer(
                overall={
                    "verdict": "retry",
                    "recommended_action": "increase_dilation",
                    "reason": "outlines survive in frames I was not shown",
                }
            )
        )
        transport, _calls = counting_transport(payload)
        record = self.judge(transport)
        self.assertEqual(record["passShare"], 1.0)
        self.assertEqual(record["verdict"], "retry")
        self.assertEqual(record["modelOverall"]["verdict"], "retry")

    def test_a_run_without_saved_masks_is_judged_without_mask_signals(self):
        run_dir = make_run(self.root / "nomasks", self.video, with_masks=False)
        out = run_dir / "clean-judge.json"
        transport, _calls = counting_transport(completion(answer()))
        record = self.judge(transport, out=out, run_dir=run_dir)
        self.assertEqual(record["verdict"], "pass")
        self.assertIn("masks.npz", record["maskNote"])
        for frame in record["frames"]:
            self.assertIsNone(frame["signals"]["maskFraction"])
            self.assertIsNone(frame["signals"]["changedOutsideMaskFraction"])
            self.assertIsNone(frame["signals"]["blurRatioAgainstPlate"])
            self.assertGreater(frame["signals"]["changedFraction"], 0.0)


def expected_kinds(damage):
    """The critical kind a damaged fixture answer is expected to raise."""
    if damage.get("subjects_removed", "all") != "all":
        return ["subject_not_removed"]
    if damage.get("non_human_subject_left", "none") != "none":
        return ["non_human_subject_left"]
    return ["background_destroyed"]


class CompositeTest(CleanJudgeCase):
    def test_every_pair_carries_the_original_of_that_cleaned_frame(self):
        clean = json.loads((self.run_dir / "clean.json").read_text())
        frames = judge_clean.review_frames(self.run_dir, clean)
        self.assertEqual([frame["index"] for frame in frames], list(REVIEW_ORDINALS))
        self.assertEqual(
            [frame["sourceIndex"] for frame in frames],
            [2 * ordinal for ordinal in REVIEW_ORDINALS],
        )

        built = judge_clean.build_composites(self.video, frames, self.root / "pairs")
        originals = judge_clean.read_frames(self.video, [2 * o for o in REVIEW_ORDINALS])
        for frame in built:
            canvas = cv2.imread(frame["compositePath"])
            self.assertEqual(canvas.shape, (HEIGHT, 2 * WIDTH + judge_clean.SEPARATOR_PX, 3))
            left = canvas[:, :WIDTH]
            right = canvas[:, WIDTH + judge_clean.SEPARATOR_PX :]
            expected_left = originals[frame["sourceIndex"]]
            expected_right = cv2.imread(str(frame["cleanPath"]))
            # Below the drawn captions the two halves are the source bytes, unchanged.
            np.testing.assert_array_equal(left[PERSON_TOP:], expected_left[PERSON_TOP:])
            np.testing.assert_array_equal(right[PERSON_TOP:], expected_right[PERSON_TOP:])
            # A pair of a different frame's original would not match: every frame differs.
            other = originals[2 * REVIEW_ORDINALS[-1] if frame["index"] != 19 else 0]
            self.assertFalse(np.array_equal(left[PERSON_TOP:], other[PERSON_TOP:]))

    def test_more_review_frames_than_the_cap_are_spread_over_the_clip(self):
        run_dir = make_run(
            self.root / "many", self.video, review_ordinals=tuple(range(CLEAN_FRAMES))
        )
        clean = json.loads((run_dir / "clean.json").read_text())
        frames = judge_clean.review_frames(run_dir, clean, judge_clean.MAX_FRAMES)
        self.assertEqual(len(frames), judge_clean.MAX_FRAMES)
        self.assertEqual(frames[0]["index"], 0)
        self.assertEqual(frames[-1]["index"], CLEAN_FRAMES - 1)
        self.assertEqual([f["index"] for f in frames], sorted({f["index"] for f in frames}))

        three = judge_clean.review_frames(run_dir, clean, 3)
        self.assertEqual([frame["index"] for frame in three], [0, 10, CLEAN_FRAMES - 1])
        transport, calls = counting_transport(
            completion(answer(indices=[frame["index"] for frame in three]))
        )
        record = judge_clean.judge_clean(
            run_dir, run_dir / "clean-judge.json", max_frames=3, urlopen=transport
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(record["framesJudged"], [0, 10, CLEAN_FRAMES - 1])
        self.assertEqual(record["verdict"], "pass")


class ReceiptTest(CleanJudgeCase):
    def test_invalid_answers_fail_the_receipt_keep_the_response_and_write_no_output(self):
        cases = {
            "missing-frame": answer(frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]),
            "duplicate-frame": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS]
                + [frame_answer(REVIEW_ORDINALS[0])]
            ),
            "unknown-frame": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS] + [frame_answer(4242)]
            ),
            "bad-removed-enum": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], subjects_removed="mostly")]
            ),
            "bad-damage-enum": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], background_destroyed="somewhat")]
            ),
            "bad-smear-enum": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], smear_severity="a bit")]
            ),
            "partial-without-who": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], subjects_removed="partial")]
            ),
            "damage-without-note": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], background_destroyed="major")]
            ),
            "smear-fraction-out-of-range": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], smear_severity="major", smear_fraction=1.4)]
            ),
            "no-reason": answer(
                frames=[frame_answer(i) for i in REVIEW_ORDINALS[:-1]]
                + [frame_answer(REVIEW_ORDINALS[-1], reason="  ")]
            ),
            "unknown-action": answer(
                overall={
                    "verdict": "retry",
                    "recommended_action": "rerun_with_a_bigger_model",
                    "reason": "invented",
                }
            ),
            "pass-with-an-action": answer(
                overall={
                    "verdict": "pass",
                    "recommended_action": "increase_dilation",
                    "reason": "contradictory",
                }
            ),
            "retry-without-an-action": answer(
                overall={
                    "verdict": "retry",
                    "recommended_action": "none",
                    "reason": "nothing to do about it",
                }
            ),
            "no-overall": {"frames": [frame_answer(i) for i in REVIEW_ORDINALS]},
            "not-json": "the plate looks fine to me",
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                out = self.root / name / "clean-judge.json"
                payload = completion(body)
                transport, calls = counting_transport(payload)
                with self.assertRaises(RuntimeError) as caught:
                    self.judge(transport, out=out)

                self.assertEqual(len(calls), 1)
                self.assertFalse(out.exists())
                self.assertNotIn("sk-test-secret", str(caught.exception))
                receipt = self.receipt(out)
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["errorCategory"], "invalid_response")
                self.assertEqual(receipt["response"]["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(receipt["response"]["usage"], USAGE)
                self.assertEqual(vlm_once.response_raw_path(out).read_bytes(), payload)

                with self.assertRaisesRegex(RuntimeError, "failed"):
                    self.judge(must_not_call, out=out)
                self.assertEqual(len(calls), 1)
                self.assertFalse(out.exists())

    def test_completed_result_is_reused_without_a_second_call(self):
        transport, calls = counting_transport(completion(answer()))
        first = self.judge(transport)
        with mock.patch.dict(os.environ, {}, clear=True):
            second = self.judge(must_not_call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second, first)
        self.assertEqual(second, json.loads(self.out.read_text()))

    def test_a_clip_that_is_not_the_cleaned_video_is_refused(self):
        other = self.root / "other.avi"
        if not write_video(other, frames=SOURCE_FRAMES - 6):
            self.skipTest("no usable MJPG encoder for cv2.VideoWriter here")
        with self.assertRaisesRegex(ValueError, "different file"):
            self.judge(must_not_call, clip=other)
        self.assertFalse(vlm_once.receipt_path(self.out).exists())


class CommandLineTest(CleanJudgeCase):
    def run_main(self, transport, *extra, out=None, run_dir=None):
        """main() with its streams captured, so a test run stays readable."""
        argv = ["--run", str(run_dir or self.run_dir), "--out", str(out or self.out), *extra]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return judge_clean.main(argv, urlopen=transport)

    def test_exit_codes_say_pass_retry_fail_and_error(self):
        transport, _calls = counting_transport(completion(answer()))
        self.assertEqual(self.run_main(transport), 0)

        retry_run = make_run(self.root / "retry", self.video)
        retry = completion(
            answer(
                frames=[
                    frame_answer(index, verdict="retry", reason="an outline survives")
                    for index in REVIEW_ORDINALS
                ],
                overall={
                    "verdict": "retry",
                    "recommended_action": "increase_dilation",
                    "reason": "the masks are a little tight",
                },
            )
        )
        transport, _calls = counting_transport(retry)
        self.assertEqual(
            self.run_main(transport, out=retry_run / "clean-judge.json", run_dir=retry_run),
            10,
        )

        fail_run = make_run(self.root / "fail", self.video)
        fail = completion(
            answer(
                frames=[
                    frame_answer(
                        index,
                        subjects_removed="none",
                        subjects_remaining="the runner, mid frame",
                        verdict="fail",
                        reason="nobody was removed",
                    )
                    for index in REVIEW_ORDINALS
                ],
                overall={
                    "verdict": "fail",
                    "recommended_action": "switch_to_semantic_mask",
                    "reason": "the detector saw nobody",
                },
            )
        )
        transport, _calls = counting_transport(fail)
        self.assertEqual(
            self.run_main(transport, out=fail_run / "clean-judge.json", run_dir=fail_run), 11
        )

        empty = self.root / "empty-run"
        empty.mkdir()
        self.assertEqual(
            self.run_main(must_not_call, out=empty / "clean-judge.json", run_dir=empty),
            judge_clean.ERROR_EXIT,
        )

    def test_a_human_verdict_is_appended_and_takes_over_the_exit_code(self):
        transport, calls = counting_transport(completion(answer()))
        self.assertEqual(self.run_main(transport), 0)

        ledger = self.out.parent / judge_clean.CALIBRATION_NAME
        self.assertFalse(ledger.exists())

        with mock.patch.dict(os.environ, {}, clear=True):
            code = self.run_main(
                must_not_call,
                "--human-verdict",
                "retry",
                "--human-note",
                "a glove is still floating in frame 11",
            )
        self.assertEqual(code, 10)
        self.assertEqual(len(calls), 1)

        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["schema"], "wander.clean-judge-calibration/1")
        self.assertEqual(entry["judgeVerdict"], "pass")
        self.assertEqual(entry["humanVerdict"], "retry")
        self.assertEqual(entry["humanNote"], "a glove is still floating in frame 11")
        self.assertFalse(entry["agreed"])
        self.assertTrue(entry["falsePass"])
        self.assertFalse(entry["falseFail"])
        self.assertEqual(entry["run"], self.run_dir.name)
        self.assertEqual(entry["judgementSha256"], vlm_once.file_identity(self.out)["sha256"])

        with mock.patch.dict(os.environ, {}, clear=True):
            code = self.run_main(
                must_not_call, "--human-verdict", "pass", "--human-note", "agreed on review"
            )
        self.assertEqual(code, 0)
        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[1]["agreed"])
        self.assertFalse(entries[1]["falsePass"])

    def test_a_human_verdict_without_a_note_is_an_error_and_writes_nothing(self):
        transport, _calls = counting_transport(completion(answer()))
        self.assertEqual(self.run_main(transport), 0)
        with mock.patch.dict(os.environ, {}, clear=True):
            code = self.run_main(must_not_call, "--human-verdict", "fail")
        self.assertEqual(code, judge_clean.ERROR_EXIT)
        self.assertFalse((self.out.parent / judge_clean.CALIBRATION_NAME).exists())

    def test_the_judged_record_is_printed_as_json_with_a_summary_on_stderr(self):
        transport, _calls = counting_transport(completion(answer()))
        self.assertEqual(self.run_main(transport), 0)
        self.assertEqual(json.loads(self.stdout.getvalue())["verdict"], "pass")
        self.assertIn("clean judge: pass", self.stderr.getvalue())
        self.assertIn("not proof of visual quality", self.stderr.getvalue())


class DecisionUnitTest(unittest.TestCase):
    """The deterministic layer on its own, with no images, no run and no transport."""

    clean: ClassVar[dict] = {
        "peopleMask": "foreground",
        "moved": False,
        "dilate": 30,
        "bottomExtra": 45,
    }

    def test_three_of_four_passing_is_a_pass_and_two_of_four_is_not(self):
        good = [frame_answer(index) for index in range(4)]
        bad = dict(good[3], edge_people=["a shoulder, bottom right"], verdict="retry")
        frames = judge_clean.check_frames({"frames": good[:3] + [bad]}, [0, 1, 2, 3])
        judged = [{"index": index, "answers": frames[index]} for index in range(4)]
        overall = {
            "verdict": "retry",
            "recommendedAction": "increase_dilation",
            "reason": "an edge person survives",
        }
        decision = judge_clean.decide(judged, overall, self.clean)
        self.assertEqual(decision["passShare"], 0.75)
        # 3/4 clears the share, but the model's own overall retry is the worse answer.
        self.assertEqual(decision["verdict"], "retry")

        decision = judge_clean.decide(
            judged[:2] + [{"index": 2, "answers": frames[3]}, judged[3]],
            overall,
            self.clean,
        )
        self.assertEqual(decision["passShare"], 0.5)
        self.assertEqual(decision["verdict"], "retry")
        self.assertTrue(decision["autoRetryable"])

    def test_the_dilation_suggestion_doubles_what_the_run_used(self):
        self.assertEqual(
            judge_clean.suggested_flags("increase_dilation", self.clean),
            ["--dilate", "60", "--bottom-extra", "90"],
        )
        self.assertEqual(
            judge_clean.suggested_flags("increase_dilation", {"dilate": 0, "bottomExtra": 0}),
            ["--dilate", "10", "--bottom-extra", "10"],
        )

    def test_a_no_op_recommendation_is_recognised_for_every_mode(self):
        semantic = {"peopleMask": "semantic", "moved": False}
        moved = {"peopleMask": "semantic", "moved": True}
        self.assertTrue(judge_clean.action_is_noop("switch_to_semantic_mask", semantic))
        self.assertFalse(judge_clean.action_is_noop("switch_to_semantic_mask", moved))
        self.assertTrue(judge_clean.action_is_noop("switch_to_foreground_mask", self.clean))
        self.assertTrue(judge_clean.action_is_noop("enable_moved_mask", moved))
        self.assertFalse(judge_clean.action_is_noop("increase_dilation", self.clean))

    def test_none_and_a_spelled_out_none_are_the_same_empty_list(self):
        frames = judge_clean.check_frames(
            {
                "frames": [
                    frame_answer(0, floating_remnants=None, edge_people=["none"]),
                ]
            },
            [0],
        )
        self.assertEqual(frames[0]["floatingRemnants"], [])
        self.assertEqual(frames[0]["edgePeople"], [])


class SpreadTest(unittest.TestCase):
    def test_spread_keeps_the_ends_and_never_repeats(self):
        self.assertEqual(judge_clean.spread(list(range(5)), 8), [0, 1, 2, 3, 4])
        picked = judge_clean.spread(list(range(20)), 8)
        self.assertEqual(len(picked), 8)
        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 19)
        self.assertEqual(picked, sorted(set(picked)))
        with self.assertRaises(ValueError):
            judge_clean.spread(list(range(4)), 0)


if __name__ == "__main__":
    unittest.main()
