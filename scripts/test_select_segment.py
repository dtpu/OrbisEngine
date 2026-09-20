"""Segment selection, tested where the decisions are: window geometry, scoring, vetoes, output.

Every test below feeds SYNTHETIC per-window feature tables, so nothing here decodes a video,
downloads a model or reaches the network. That is deliberate: the interesting failures of this
stage are "it picked a window that straddles a cut" and "it preferred one person to four", and
both are decided by pure functions that never see a pixel.

One optional smoke test builds a tiny clip with ffmpeg and runs the whole stage over it with the
detector stubbed out; it skips itself when ffmpeg is missing.

  uv run --locked python scripts/test_select_segment.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_segment as sel

FPS = 24.0
PARAMS = sel._params(
    min_window=6.0,
    target_window=12.0,
    max_window=20.0,
    min_usable=2.5,
    stride=3.0,
    people_cap=4,
    sample_fps=3.0,
    people_max_frames=48,
    top_k=3,
    max_overlap=0.0,
)
WEIGHTS = dict(sel.DEFAULT_WEIGHTS)


# ---------------------------------------------------------------- synthetic features
def features(
    *,
    seconds: float = 12.0,
    ratio: float | None = 0.70,
    subjects: int = 1,
    presence: float = 1.0,
    stability: float = 1.0,
    height: float = 0.6,
    full_body: float = 1.0,
    person_pixels: float = 0.1,
    sharpness: float = 400.0,
    dark: float = 0.02,
    bright: float = 0.0,
    overlay: float = 0.0,
    min_carry: float = 0.9,
    endpoint_overlap: float | None = 0.15,
    samples: int = 36,
    people_available: bool = True,
    cap: int = 4,
) -> dict:
    """One window's measurements, in the exact shape select_segment builds from real frames."""
    camera = (
        dict(
            available=True,
            verdict="rotation-only" if ratio >= sel.ROTATION_ONLY_RATIO else "has-parallax",
            widestGapSeconds=3.0,
            pairs=12,
            widestGapRatio=ratio,
            widestGapHomographyResidualPx=2.0,
            threshold=sel.ROTATION_ONLY_RATIO,
            byGap={},
        )
        if ratio is not None
        else dict(available=False, verdict="inconclusive", reason="synthetic", byGap={})
    )
    people = (
        dict(
            available=True,
            reason=None,
            frames=6,
            presenceFraction=presence,
            subjectCountMedian=subjects,
            subjectCountMax=subjects,
            countStability=stability,
            subjectHeightFractionMedian=height,
            fullBodyFraction=full_body,
            personPixelFractionMedian=person_pixels,
            personPixelFractionMax=person_pixels,
            peopleCap=cap,
        )
        if people_available
        else dict(available=False, reason="synthetic: no detector", frames=0)
    )
    return dict(
        seconds=seconds,
        samples=samples,
        camera=camera,
        people=people,
        image=dict(
            available=samples >= 2,
            samples=samples,
            sharpness=sharpness,
            darkFraction=dark,
            brightFraction=bright,
            meanLuma=110.0,
        ),
        overlay=dict(
            available=True,
            staticPixelFraction=overlay,
            letterboxFraction=0.0,
            overlayFraction=overlay,
            staticCameraSuspected=False,
        ),
        continuity=dict(
            available=samples >= 2,
            samplePairs=max(samples - 1, 0),
            minCarry=min_carry,
            medianCarry=max(min_carry, 0.9),
            minCarryAtSeconds=1.0,
        ),
        place=dict(
            available=True,
            endpointOverlap=endpoint_overlap,
            endpointPairs=4,
            halfOverlap=endpoint_overlap,
            spanSeconds=seconds,
        )
        if endpoint_overlap is not None
        else dict(available=False, reason="synthetic: too featureless"),
    )


def score(**kwargs) -> float:
    return sel.score_window(features(**kwargs), WEIGHTS, PARAMS)["score"]


def raw_score(**kwargs) -> float:
    """The weighted sum before vetoes: what the terms say, with no refusal on top of them.

    The ranking tests use this so that a term being worth more than another is tested on the sum
    itself; whether a value is ALSO a veto is a separate question, tested in Vetoes.
    """
    return sel.score_window(features(**kwargs), WEIGHTS, PARAMS)["scoreBeforeVetoes"]


def candidate(cid: str, start: float, end: float, value: float, vetoes=()) -> dict:
    return dict(
        id=cid,
        startSeconds=start,
        endSeconds=end,
        seconds=end - start,
        score=value,
        vetoes=list(vetoes),
    )


# ---------------------------------------------------------------- window geometry
class Windows(unittest.TestCase):
    def test_a_short_shot_is_one_whole_window_inside_its_margins(self):
        shot = dict(index=0, start=10.0, end=22.0)
        got = sel.windows_for_shot(shot, FPS, PARAMS)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0]["whole"])
        frame = 1.0 / FPS
        self.assertAlmostEqual(got[0]["start"], 10.0 + 0.5 * frame, places=6)
        self.assertAlmostEqual(got[0]["end"], 22.0 - 1.5 * frame, places=6)

    def test_the_margin_matches_what_shot_cuts_trim_actually_cuts(self):
        """A chosen window must be trimmable by trim()'s own convention, or it gains a foreign frame."""
        shot = dict(index=0, start=4.0, end=12.0, seconds=8.0)
        lo, hi = sel.usable_span(shot, FPS)
        frame = 1.0 / FPS
        trim_start = shot["start"] + 0.5 * frame
        trim_end = trim_start + max(shot["seconds"] - 2.0 * frame, 0.1)
        self.assertAlmostEqual(lo, trim_start, places=9)
        self.assertAlmostEqual(hi, trim_end, places=9)

    def test_a_shot_under_the_usable_floor_yields_nothing(self):
        self.assertEqual(sel.windows_for_shot(dict(index=0, start=0.0, end=2.0), FPS, PARAMS), [])

    def test_a_long_shot_is_cut_into_sliding_target_length_windows(self):
        got = sel.windows_for_shot(dict(index=3, start=0.0, end=60.0), FPS, PARAMS)
        self.assertGreater(len(got), 1)
        for w in got:
            self.assertAlmostEqual(w["end"] - w["start"], PARAMS["targetWindowSeconds"], places=6)
            self.assertEqual(w["shotIndex"], 3)
        starts = [w["start"] for w in got]
        self.assertEqual(starts, sorted(starts))
        self.assertAlmostEqual(starts[1] - starts[0], PARAMS["strideSeconds"], places=6)

    def test_windows_never_cross_a_cut(self):
        shots = sel.shots_from_cut_times([30.0, 45.0], 90.0)
        got = sel.candidate_windows(shots, FPS, PARAMS)
        self.assertTrue(got)
        bounds = {s["index"]: (s["start"], s["end"]) for s in shots}
        for w in got:
            lo, hi = bounds[w["shotIndex"]]
            self.assertGreaterEqual(w["start"], lo)
            self.assertLessEqual(w["end"], hi)
            for cut in (30.0, 45.0):
                self.assertFalse(w["start"] < cut < w["end"], f"{w} straddles the cut at {cut}")

    def test_the_last_window_reaches_the_end_of_a_long_shot(self):
        shot = dict(index=0, start=0.0, end=41.0)
        got = sel.windows_for_shot(shot, FPS, PARAMS)
        _, hi = sel.usable_span(shot, FPS)
        self.assertAlmostEqual(got[-1]["end"], hi, places=6)

    def test_every_window_is_within_the_length_bounds(self):
        shots = sel.shots_from_cut_times([9.0, 70.0], 120.0)
        for w in sel.candidate_windows(shots, FPS, PARAMS):
            length = w["end"] - w["start"]
            self.assertGreaterEqual(length, PARAMS["minUsableSeconds"] - 1e-6)
            self.assertLessEqual(length, PARAMS["maxWindowSeconds"] + 1e-6)

    def test_truth_boundaries_replace_the_detectors(self):
        with tempfile.TemporaryDirectory() as work:
            truth = Path(work) / "truth.json"
            truth.write_text(json.dumps([{"time": 12.0}, {"time": 25.0}]))
            times, meta = sel.cut_source(Path("unused.mp4"), None, truth, 40.0)
        self.assertEqual(times, [12.0, 25.0])
        self.assertEqual(meta["source"], "truth")

    def test_a_cut_report_supplies_the_boundaries_without_re_detecting(self):
        with tempfile.TemporaryDirectory() as work:
            report = Path(work) / "cuts.json"
            report.write_text(json.dumps(dict(video="x.mp4", cuts=[{"time": 5.5}])))
            times, meta = sel.cut_source(Path("unused.mp4"), report, None, 20.0)
        self.assertEqual(times, [5.5])
        self.assertEqual(meta["source"], "cut-report")

    def test_a_sample_pair_that_does_not_correlate_is_a_suspected_break(self):
        rows = [
            dict(index=i, time=i / 6.0, carryFromPrevious=None if i == 0 else carry)
            for i, carry in enumerate([1.0, 0.9, 0.88, 0.05, 0.91, 0.85])
        ]
        got = sel.suspected_breaks(rows)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0]["fromSeconds"], 2 / 6.0, places=4)
        self.assertAlmostEqual(got[0]["toSeconds"], 3 / 6.0, places=4)
        self.assertEqual(got[0]["carry"], 0.05)

    def test_a_break_splits_its_shot_and_the_interval_itself_is_dropped(self):
        shots = sel.shots_from_cut_times([], 40.0)
        got = sel.split_shots_at(shots, [dict(fromSeconds=20.0, toSeconds=20.1667, carry=0.05)])
        self.assertEqual(len(got), 2)
        self.assertEqual((got[0]["start"], got[0]["end"]), (0.0, 20.0))
        self.assertEqual((got[1]["start"], got[1]["end"]), (20.1667, 40.0))
        self.assertEqual(got[0]["endSource"], "suspected-break")
        self.assertEqual(got[1]["startSource"], "suspected-break")
        self.assertEqual([s["index"] for s in got], [0, 1])

    def test_windows_never_cross_a_suspected_break_either(self):
        shots = sel.split_shots_at(
            sel.shots_from_cut_times([], 60.0),
            [dict(fromSeconds=25.0, toSeconds=25.1667, carry=0.05)],
        )
        for w in sel.candidate_windows(shots, FPS, PARAMS):
            self.assertFalse(w["start"] < 25.0 < w["end"])
            self.assertFalse(w["start"] < 25.1667 < w["end"])

    def test_a_break_outside_a_shot_leaves_it_alone(self):
        shots = sel.shots_from_cut_times([30.0], 60.0)
        got = sel.split_shots_at(shots, [dict(fromSeconds=45.0, toSeconds=45.1667, carry=0.02)])
        self.assertEqual(len(got), 3)
        self.assertEqual((got[0]["start"], got[0]["end"]), (0.0, 30.0))

    def test_cut_times_outside_the_clip_are_ignored(self):
        shots = sel.shots_from_cut_times([-1.0, 0.0, 10.0, 99.0], 30.0)
        self.assertEqual([(s["start"], s["end"]) for s in shots], [(0.0, 10.0), (10.0, 30.0)])


# ---------------------------------------------------------------- letterbox
class Letterbox(unittest.TestCase):
    """Bars are 27% of a 2.39:1 movie frame; measuring through them corrupts exposure AND carry."""

    def stack(self, top: int, bottom: int, left: int = 0, right: int = 0, n: int = 8):
        import numpy as np

        rng = np.random.default_rng(7)
        frames = rng.integers(40, 220, size=(n, 100, 200), dtype=np.uint8)
        if top:
            frames[:, :top, :] = 0
        if bottom:
            frames[:, -bottom:, :] = 0
        if left:
            frames[:, :, :left] = 0
        if right:
            frames[:, :, -right:] = 0
        return frames

    def test_bars_are_found_on_every_edge(self):
        got = sel.letterbox_bars(self.stack(top=13, bottom=13, left=6, right=6))
        self.assertEqual((got["topPx"], got["bottomPx"]), (13, 13))
        self.assertEqual((got["leftPx"], got["rightPx"]), (6, 6))
        self.assertGreater(got["fractionOfFrame"], 0.3)

    def test_a_clip_without_bars_reports_none(self):
        got = sel.letterbox_bars(self.stack(top=0, bottom=0))
        self.assertEqual(got["fractionOfFrame"], 0.0)
        self.assertEqual((got["topPx"], got["bottomPx"]), (0, 0))

    def test_a_dark_picture_is_not_mistaken_for_a_bar(self):
        import numpy as np

        frames = np.full((6, 100, 200), 40, np.uint8)  # dim, but above LETTERBOX_LUMA
        frames[:, :10, :] = 0
        self.assertEqual(sel.letterbox_bars(frames)["topPx"], 10)
        self.assertEqual(sel.letterbox_bars(frames)["bottomPx"], 0)

    def test_bars_survive_a_minority_of_unletterboxed_frames(self):
        """The 99 s movie scene is 2.39:1 for 84% of its length and full-frame for the rest."""
        frames = self.stack(top=13, bottom=13, n=20)
        frames[16:, :, :] = 200  # the last 4 frames of 20 have no bars at all
        got = sel.letterbox_bars(frames, sample_every=1)
        self.assertEqual((got["topPx"], got["bottomPx"]), (13, 13))
        self.assertAlmostEqual(got["barFrameFraction"], 0.8, places=3)

    def test_bars_in_too_few_frames_are_not_cropped(self):
        frames = self.stack(top=13, bottom=13, n=20)
        frames[10:, :, :] = 200  # only half the clip is letterboxed
        got = sel.letterbox_bars(frames, sample_every=1)
        self.assertEqual(got["fractionOfFrame"], 0.0)

    def test_one_dark_shot_is_not_a_bar(self):
        frames = self.stack(top=0, bottom=0, n=20)
        frames[:4, :, :] = 0  # four entirely black frames, no bars anywhere
        self.assertEqual(sel.letterbox_bars(frames, sample_every=1)["fractionOfFrame"], 0.0)

    def test_the_search_never_leaves_the_edge_band(self):
        import numpy as np

        frames = np.zeros((4, 100, 200), np.uint8)  # an entirely black clip
        got = sel.letterbox_bars(frames)
        self.assertLessEqual(got["topPx"], int(100 * sel.LETTERBOX_MAX_BAND))
        self.assertLessEqual(got["leftPx"], int(200 * sel.LETTERBOX_MAX_BAND))

    def test_cropping_removes_exactly_the_bars_from_grey_and_rgb(self):
        import numpy as np

        bars = sel.letterbox_bars(self.stack(top=13, bottom=13))
        grey = np.zeros((3, 100, 200), np.uint8)
        rgb = np.zeros((3, 100, 200, 3), np.uint8)
        self.assertEqual(sel.crop_bars(grey, bars).shape, (3, 74, 200))
        self.assertEqual(sel.crop_bars(rgb, bars).shape, (3, 74, 200, 3))

    def test_a_crop_that_would_leave_nothing_is_refused(self):
        import numpy as np

        everything = dict(topFraction=0.5, bottomFraction=0.5, leftFraction=0.0, rightFraction=0.0)
        frames = np.zeros((2, 40, 60), np.uint8)
        self.assertEqual(sel.crop_bars(frames, everything).shape, (2, 40, 60))

    def test_the_overlay_measure_reports_bars_it_was_handed(self):
        import numpy as np

        moving = np.random.default_rng(3).integers(0, 255, size=(6, 50, 80), dtype=np.uint8)
        got = sel.overlay_features(moving, letterbox_fraction=0.27)
        self.assertEqual(got["letterboxFraction"], 0.27)
        self.assertGreaterEqual(got["overlayFraction"], 0.27)
        self.assertFalse(got["staticCameraSuspected"])

    def test_a_locked_off_camera_falls_back_to_the_bars_alone(self):
        import numpy as np

        still = np.full((6, 50, 80), 120, np.uint8)
        got = sel.overlay_features(still, letterbox_fraction=0.1)
        self.assertTrue(got["staticCameraSuspected"])
        self.assertEqual(got["overlayFraction"], 0.1)


# ---------------------------------------------------------------- scoring
class Scoring(unittest.TestCase):
    def test_more_camera_translation_scores_higher(self):
        # A lower homography/fundamental ratio IS more translation; see parallax_probe.
        ladder = [raw_score(ratio=r) for r in (0.95, 0.90, 0.86, 0.82, 0.70)]
        self.assertEqual(ladder, sorted(ladder))
        self.assertLess(ladder[0], ladder[-1])

    def test_translation_is_the_dominant_term(self):
        """Removing parallax must cost more than removing any other single term."""
        base = raw_score()
        drops = dict(
            translation=base - raw_score(ratio=0.95),
            people=base - raw_score(subjects=0),
            continuity=base - raw_score(min_carry=sel.INTERNAL_CUT_CARRY + 0.001),
            fullBody=base - raw_score(full_body=0.0),
            subjectHeight=base - raw_score(height=0.2),
            image=base - raw_score(sharpness=sel.SHARPNESS_FLOOR),
            duration=base - raw_score(seconds=sel.MIN_WINDOW_SECONDS),
            place=base - raw_score(endpoint_overlap=0.0),
        )
        for term, cost in drops.items():
            if term == "translation":
                continue
            self.assertGreater(drops["translation"], cost, f"{term} outweighs camera translation")

    def test_no_subject_scores_below_one_subject(self):
        self.assertLess(score(subjects=0, presence=0.0), score(subjects=1))

    def test_subjects_up_to_the_cap_are_not_penalised(self):
        equal = [score(subjects=n) for n in range(1, sel.PEOPLE_CAP + 1)]
        self.assertEqual(
            len(set(equal)), 1, f"a crowd within the cap was scored differently: {equal}"
        )

    def test_above_the_cap_the_people_term_decays_but_stays_above_zero(self):
        over = sel.count_term(sel.PEOPLE_CAP + 2, sel.PEOPLE_CAP)
        self.assertLess(over, 1.0)
        self.assertGreaterEqual(over, sel.PEOPLE_OVER_CAP_FLOOR)
        self.assertGreater(score(subjects=sel.PEOPLE_CAP * 2), score(subjects=0, presence=0.0))

    def test_a_frame_dominated_by_people_is_penalised(self):
        self.assertLess(score(person_pixels=0.75), score(person_pixels=0.1))

    def test_intermittent_subjects_score_below_a_constant_cast(self):
        self.assertLess(score(presence=0.2), score(presence=1.0))
        self.assertLess(score(stability=0.2), score(stability=1.0))

    def test_a_missing_detector_degrades_loudly_rather_than_scoring_zero(self):
        result = sel.score_window(features(people_available=False), WEIGHTS, PARAMS)
        self.assertIn("no person detector", result["why"])
        self.assertEqual(result["terms"]["people"], sel.PEOPLE_UNKNOWN)

    def test_darkness_blur_and_overlays_all_cost_score(self):
        self.assertLess(score(dark=0.5), score(dark=0.0))
        self.assertLess(score(sharpness=30.0), score(sharpness=400.0))
        self.assertLess(score(overlay=0.25), score(overlay=0.0))

    def test_a_suspected_internal_cut_costs_score_before_it_vetoes(self):
        self.assertLess(score(min_carry=0.35), score(min_carry=0.9))

    def test_place_is_measured_and_reported_but_not_ranked_on_by_default(self):
        """Measured on real footage it pointed the wrong way, so its default weight is zero.

        The term itself still rises with overlap, so `--weights` can switch it on for footage
        where it has been shown to work; what is pinned here is that the SHIPPED default does not
        move a score by a measurement that failed its own test.
        """
        self.assertEqual(sel.DEFAULT_WEIGHTS["place"], 0.0)
        self.assertEqual(raw_score(endpoint_overlap=0.0), raw_score(endpoint_overlap=0.4))
        terms = [
            sel.place_term(dict(available=True, endpointOverlap=o))
            for o in (0.0, sel.PLACE_OVERLAP_FLOOR, 0.05, sel.PLACE_OVERLAP_CLEAR, 0.4)
        ]
        self.assertEqual(terms, sorted(terms))
        self.assertEqual(terms[-2], terms[-1], "the place term saturates once the ends overlap")
        heavy = {**sel.DEFAULT_WEIGHTS, "place": 0.5}
        self.assertLess(
            sel.score_window(features(endpoint_overlap=0.0), heavy, PARAMS)["score"],
            sel.score_window(features(endpoint_overlap=0.4), heavy, PARAMS)["score"],
        )

    def test_an_unmeasurable_place_degrades_rather_than_scoring_zero(self):
        result = sel.score_window(features(endpoint_overlap=None), WEIGHTS, PARAMS)
        self.assertEqual(result["terms"]["place"], sel.PLACE_UNKNOWN)
        self.assertEqual(result["vetoes"], [])

    def test_the_score_stays_inside_zero_and_one(self):
        best = score(ratio=0.5, subjects=2, sharpness=5000.0, person_pixels=0.0, min_carry=1.0)
        self.assertLessEqual(best, 1.0)
        self.assertGreaterEqual(score(ratio=0.99, subjects=0, presence=0.0, sharpness=0.0), 0.0)

    def test_the_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(sel.DEFAULT_WEIGHTS.values()), 1.0, places=9)


class Weights(unittest.TestCase):
    def test_an_override_is_normalised_and_recorded(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "w.json"
            path.write_text(json.dumps({"people": 0.8}))
            weights, source = sel.resolve_weights(path)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertEqual(source["requested"], {"people": 0.8})
        self.assertGreater(weights["people"], sel.DEFAULT_WEIGHTS["people"])

    def test_an_unknown_term_is_refused(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "w.json"
            path.write_text(json.dumps({"vibes": 1.0}))
            with self.assertRaises(ValueError):
                sel.resolve_weights(path)

    def test_a_reweighted_score_can_change_the_order(self):
        """A caller who cares more about subjects than about parallax must be able to say so."""
        travelling = features(ratio=0.70, presence=0.2, subjects=1)  # moves, barely anyone in it
        crowded = features(ratio=0.90, presence=1.0, subjects=4)  # four subjects, little parallax
        people_heavy = {k: (0.9 if k == "people" else 0.1 / 6) for k in sel.DEFAULT_WEIGHTS}
        self.assertGreater(
            sel.score_window(travelling, WEIGHTS, PARAMS)["score"],
            sel.score_window(crowded, WEIGHTS, PARAMS)["score"],
        )
        self.assertLess(
            sel.score_window(travelling, people_heavy, PARAMS)["score"],
            sel.score_window(crowded, people_heavy, PARAMS)["score"],
        )


# ---------------------------------------------------------------- vetoes
class Vetoes(unittest.TestCase):
    def kinds(self, **kwargs) -> list[str]:
        return [
            v.split(":")[0] for v in sel.score_window(features(**kwargs), WEIGHTS, PARAMS)["vetoes"]
        ]

    def test_a_clean_window_is_not_vetoed(self):
        self.assertEqual(self.kinds(), [])

    def test_too_short(self):
        self.assertIn("too-short", self.kinds(seconds=4.0))

    def test_no_decodable_frames(self):
        self.assertIn("no-samples", self.kinds(samples=1))

    def test_a_dark_window_is_vetoed(self):
        self.assertIn("too-dark", self.kinds(dark=sel.DARK_VETO + 0.05))
        self.assertNotIn("too-dark", self.kinds(dark=sel.DARK_VETO - 0.05))

    def test_a_suspected_internal_cut_is_vetoed(self):
        self.assertIn("internal-cut", self.kinds(min_carry=sel.INTERNAL_CUT_CARRY - 0.05))

    def test_violent_motion_inside_a_single_take_is_penalised_but_not_vetoed(self):
        """0.166 is the worst sample pair in 1638 of creed-long-take, a VERIFIED single take.

        At the old 0.30 gate that clip was cut into 47 pieces and no 12 s window survived, so this
        is the measurement the threshold is pinned to, not an opinion about how bad motion is.
        """
        self.assertEqual(self.kinds(min_carry=0.166), [])
        self.assertLess(score(min_carry=0.166), score(min_carry=0.9))

    def test_a_veto_zeroes_the_score_and_keeps_the_terms(self):
        result = sel.score_window(features(dark=0.9), WEIGHTS, PARAMS)
        self.assertEqual(result["score"], 0.0)
        self.assertGreater(result["scoreBeforeVetoes"], 0.0)
        self.assertGreater(result["terms"]["translation"], 0.0)
        self.assertTrue(result["why"].startswith("VETOED"))

    def test_a_pan_only_camera_is_vetoed_and_says_so(self):
        """parallax_probe calls this a collapse, not a penalty, so it refuses rather than ranks."""
        result = sel.score_window(features(ratio=0.98), WEIGHTS, PARAMS)
        self.assertIn("rotation-only", self.kinds(ratio=0.98))
        self.assertEqual(result["terms"]["translation"], 0.0)
        self.assertIn("0.98", result["why"])  # the measured ratio, not just the verdict
        self.assertNotIn("rotation-only", self.kinds(ratio=sel.ROTATION_ONLY_RATIO - 0.01))

    def test_an_unmeasurable_camera_is_not_vetoed(self):
        """Inconclusive is not rotation-only: a window nothing could be matched in still ranks."""
        self.assertEqual(self.kinds(ratio=None), [])

    def test_a_traverse_is_never_vetoed(self):
        """The measured overlap of a traverse and of one room cross; a gate on it would be a guess."""
        self.assertEqual(self.kinds(endpoint_overlap=0.0), [])
        self.assertEqual(self.kinds(endpoint_overlap=None), [])


# ---------------------------------------------------------------- the sibling module
class ShotCuts(unittest.TestCase):
    """shot_cuts.py is edited by other people; a half-saved edit must not take this stage down."""

    def thumbs(self):
        import numpy as np

        rng = np.random.default_rng(11)
        a = rng.integers(0, 255, size=(sel.THUMB_HEIGHT, sel.THUMB_WIDTH), dtype="uint8")
        b = np.roll(a, 3, axis=1)  # the same picture, shifted: carry should stay high
        return a, b

    def test_the_local_carry_fallback_agrees_with_the_real_one(self):
        a, b = self.thumbs()
        if sel.shot_cuts is None:
            self.skipTest("shot_cuts did not import; the fallback is already what runs")
        self.assertAlmostEqual(sel.shot_cuts.carry(a, b), sel._fallback_carry(a, b), places=6)

    def test_carry_still_works_with_the_sibling_missing(self):
        a, b = self.thumbs()
        original = sel.shot_cuts
        sel.shot_cuts = None
        try:
            self.assertGreater(sel.carry(a, b), 0.9)
        finally:
            sel.shot_cuts = original

    def test_detecting_cuts_without_the_sibling_is_refused_in_words(self):
        original, sel.shot_cuts = sel.shot_cuts, None
        try:
            with self.assertRaises(ValueError) as caught:
                sel.cut_source(Path("clip.mp4"), None, None, 30.0)
        finally:
            sel.shot_cuts = original
        self.assertIn("--truth", str(caught.exception))


# ---------------------------------------------------------------- selection
class TopK(unittest.TestCase):
    def test_ties_are_broken_by_the_earlier_start(self):
        pool = [candidate("b", 30.0, 42.0, 0.5), candidate("a", 5.0, 17.0, 0.5)]
        self.assertEqual([c["id"] for c in sel.choose_top(pool, 1, 0.0)], ["a"])
        self.assertEqual([c["id"] for c in sel.choose_top(list(reversed(pool)), 1, 0.0)], ["a"])

    def test_the_picks_do_not_overlap(self):
        pool = [
            candidate("a", 0.0, 12.0, 0.9),
            candidate("b", 3.0, 15.0, 0.85),
            candidate("c", 30.0, 42.0, 0.6),
            candidate("d", 33.0, 45.0, 0.55),
        ]
        got = sel.choose_top(pool, 3, 0.0)
        self.assertEqual([c["id"] for c in got], ["a", "c"])
        self.assertEqual([c["rank"] for c in got], [1, 2])

    def test_an_overlap_allowance_admits_a_partly_overlapping_window(self):
        pool = [candidate("a", 0.0, 12.0, 0.9), candidate("b", 9.0, 21.0, 0.8)]
        self.assertEqual([c["id"] for c in sel.choose_top(pool, 2, 0.0)], ["a"])
        self.assertEqual([c["id"] for c in sel.choose_top(pool, 2, 0.3)], ["a", "b"])

    def test_vetoed_windows_are_never_chosen(self):
        pool = [candidate("a", 0.0, 12.0, 0.0, ["too-dark: x"]), candidate("b", 20.0, 32.0, 0.1)]
        self.assertEqual([c["id"] for c in sel.choose_top(pool, 3, 0.0)], ["b"])

    def test_everything_vetoed_selects_nothing(self):
        pool = [candidate("a", 0.0, 12.0, 0.0, ["too-short: x"])]
        self.assertEqual(sel.choose_top(pool, 3, 0.0), [])

    def test_coverage_not_selected_names_the_time_left_out(self):
        got = sel.coverage_not_selected(
            [dict(startSeconds=10.0, endSeconds=22.0), dict(startSeconds=40.0, endSeconds=52.0)],
            60.0,
        )
        self.assertEqual(got["gaps"], [[0.0, 10.0], [22.0, 40.0], [52.0, 60.0]])
        self.assertAlmostEqual(got["seconds"], 36.0, places=6)
        self.assertAlmostEqual(got["fraction"], 0.6, places=4)

    def test_coverage_of_nothing_is_the_whole_clip(self):
        self.assertAlmostEqual(sel.coverage_not_selected([], 30.0)["fraction"], 1.0, places=6)


# ---------------------------------------------------------------- override and judge
class Override(unittest.TestCase):
    def test_an_override_without_a_reason_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            sel.select(Path("does-not-exist.mp4"), force_window=(1.0, 5.0), force_reason="  ")
        self.assertIn("--reason", str(caught.exception))

    def test_a_judge_opinion_is_recorded_as_an_opinion(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "op.json"
            path.write_text(json.dumps(dict(windowId="w002", reason="the dunk is here")))
            got = sel.load_judge_opinion(path, {"w001", "w002"})
        self.assertEqual(got["windowId"], "w002")
        self.assertTrue(got["known"])
        self.assertFalse(got["applied"])

    def test_an_opinion_about_an_unknown_window_is_marked_unknown(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "op.json"
            path.write_text(json.dumps(dict(windowId="nope", reason="")))
            self.assertFalse(sel.load_judge_opinion(path, {"w001"})["known"])


# ---------------------------------------------------------------- end to end, detector stubbed
def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class Smoke(unittest.TestCase):
    """The whole stage over a tiny generated clip, with Mask R-CNN replaced by a stub.

    This checks the parts the pure tests cannot: that ffmpeg decodes, that the per-frame cache and
    the per-window aggregation line up, and that selection.json has the shape the runner will read.
    """

    @classmethod
    def setUpClass(cls):
        if not have_ffmpeg():
            raise unittest.SkipTest("ffmpeg/ffprobe not on PATH")
        cls.work = tempfile.TemporaryDirectory(prefix="select-segment-smoke-")
        cls.clip = Path(cls.work.name) / "clip.mp4"
        # A moving pattern over 16 s: enough for sliding windows, small enough to be quick.
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=12:duration=16",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(cls.clip),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "work"):
            cls.work.cleanup()

    def run_stage(self, **kwargs):
        original = sel.person_measurements

        def stub(rgb_frames, times, min_height_frac):
            return dict(
                available=True,
                reason=None,
                frames=[
                    dict(
                        time=round(t, 4),
                        detections=1,
                        subjects=1,
                        tallestHeightFraction=0.5,
                        fullBody=True,
                        personPixelFraction=0.08,
                    )
                    for t in times
                ],
            )

        sel.person_measurements = stub
        try:
            return sel.select(self.clip, truth=self.truth_file(), **kwargs)
        finally:
            sel.person_measurements = original

    def truth_file(self) -> Path:
        path = Path(self.work.name) / "truth.json"
        path.write_text(json.dumps([]))  # one continuous shot; no cut detection needed
        return path

    def test_the_document_has_the_schema_the_runner_will_read(self):
        doc = self.run_stage(top_k=2, sample_fps=2.0, people_max_frames=6)
        self.assertEqual(doc["schema"], "wander.segment-selection/1")
        for key in (
            "video",
            "cuts",
            "parameters",
            "weights",
            "weightsSource",
            "featuresAvailable",
            "candidates",
            "chosen",
            "coverageNotSelected",
            "override",
            "judge",
            "runtime",
        ):
            self.assertIn(key, doc)
        self.assertEqual(len(doc["video"]["sha256"]), 64)
        self.assertTrue(doc["candidates"])
        for c in doc["candidates"]:
            for key in ("id", "startSeconds", "endSeconds", "startFrame", "endFrame", "features"):
                self.assertIn(key, c)
            for feature in ("camera", "people", "image", "overlay", "continuity", "place"):
                self.assertIn(feature, c["features"])
            self.assertTrue(c["why"])
        self.assertEqual(len(doc["cuts"]["sha256"]), 64)  # the exact cut list this was made from
        self.assertTrue(doc["chosen"])
        first = doc["chosen"][0]
        self.assertEqual(first["rank"], 1)
        # The runner reads start, end and crop off one object; a clip without bars says so.
        self.assertIn("suggestedCrop", first)
        self.assertFalse(first["suggestedCrop"]["needed"])
        self.assertLess(first["startSeconds"], first["endSeconds"])
        self.assertLessEqual(first["endSeconds"], doc["video"]["durationSeconds"] + 1e-6)
        json.dumps(doc)  # the whole document must be JSON-serialisable

    def test_an_override_records_the_score_it_replaced(self):
        doc = self.run_stage(
            top_k=1,
            sample_fps=2.0,
            people_max_frames=6,
            force_window=(2.0, 10.0),
            force_reason="the producer wants the entrance",
        )
        self.assertIsNotNone(doc["override"])
        self.assertEqual(doc["override"]["reason"], "the producer wants the entrance")
        self.assertEqual(doc["chosen"][0]["startSeconds"], 2.0)
        self.assertTrue(doc["chosen"][0]["forced"])
        self.assertTrue(doc["override"]["wouldHaveChosen"])

    def test_a_contact_sheet_and_judge_request_are_written_on_request(self):
        sheets = Path(self.work.name) / "sheets"
        doc = self.run_stage(top_k=2, sample_fps=2.0, people_max_frames=6, contact_sheet_dir=sheets)
        self.assertTrue(doc["contactSheets"])
        for path in doc["contactSheets"].values():
            self.assertTrue(Path(path).exists())
        request = json.loads(Path(doc["judgeRequest"]).read_text())
        self.assertEqual(request["schema"], "wander.segment-judge-request/1")
        self.assertTrue(request["windows"])
        self.assertEqual(len(request["windows"][0]["contactSheetSha256"]), 64)


def main() -> int:
    unittest.main(argv=[sys.argv[0]], verbosity=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
