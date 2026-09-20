#!/usr/bin/env python3
"""Chained alignment for the Pi3X dense solve; numpy only, no torch, GPU or media.

One global anchor set needs all eight of its views to be co-visible, which a camera that
TRAVELS never is: `anchor_spread` correctly refuses game-run at 3.4 scene depths and game-s1 at
12.6, and the product wants those segments solved rather than trimmed. The chained mode splits
such a clip into overlapping windows that each pass the same spread limit and registers
consecutive windows on the frames they both predict.

Nothing here runs Pi3X. The windows' "predictions" are a synthetic corridor placed in an
arbitrary similarity frame per window, with noise, which is exactly the input shape the real
stage hands these functions: two predictions of the same physical frames, each in its own
coordinates and at its own arbitrary scale.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

from dense_pi3x import (  # noqa: E402
    ALIGNMENT_CHAINED,
    ALIGNMENT_GLOBAL,
    ANCHOR_SPREAD_LIMIT,
    CHAIN_MIN_WINDOW,
    CHAIN_OVERLAP_SAMPLES,
    CHAIN_TARGET_SPREAD,
    LINK_DEPTH_RATIO_BAND,
    MAX_LINK_RESIDUAL,
    adapt_window,
    anchor_spread,
    apply_similarity,
    chain_transforms,
    compose_similarity,
    drift_bound,
    identity_similarity,
    link_quality,
    localise_break,
    measure_spread,
    next_window,
    plan_windows,
    similarity,
    travel_profile,
)

# The two clips this mode exists for, as their saved `pi3x/anchors.npz` measures them: the eight
# global anchor samples, and the camera baseline between consecutive anchors as a multiple of the
# clip's median anchor depth. game-run is a 6.5 s chase camera down a causeway (78 samples, median
# depth 15.05, largest baseline 3.43 depths); game-s1 is 22 s of third-person traversal across
# several areas (269 samples, median depth 4.09 over the four anchors that measured one, largest
# baseline 12.55 depths). Both are refused outright by the single global anchor set.
GAME_RUN = dict(
    count=78,
    anchors=[0, 11, 22, 33, 44, 55, 66, 77],
    steps=[0.555, 0.527, 0.319, 0.238, 0.554, 0.616, 0.868],
    spread=3.433,
    path=3.677,
    windows=2,
)
GAME_S1 = dict(
    count=269,
    anchors=[0, 38, 77, 115, 153, 191, 230, 268],
    steps=[4.257, 6.166, 1.935, 2.425, 0.832, 1.022, 2.171],
    spread=12.552,
    path=18.808,
    windows=13,
)


def straight_poses(steps):
    """Camera-to-world poses walking one way, one entry longer than `steps`.

    A chase camera and a runner followed up a flight of steps both travel in one direction, so a
    straight walk is the case the planner has to be right about: the largest baseline between two
    views IS the path between them, and no amount of revisiting shortens it.
    """
    centres = np.concatenate([[0.0], np.cumsum(np.asarray(steps, dtype=float))])
    poses = np.tile(np.eye(4), (len(centres), 1, 1))
    poses[:, 0, 3] = centres
    return poses


def measured_profile(clip):
    """The travel profile the planner would build from that clip's saved anchor prediction."""
    return travel_profile(straight_poses(clip["steps"]), clip["anchors"], 1.0, clip["count"])


def random_similarity(rng, scale=(0.4, 2.5)):
    """One window's arbitrary prediction frame: Pi3X fixes no scale, origin or gravity."""
    rotation = np.linalg.qr(rng.normal(0, 1, (3, 3)))[0]
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return (
        float(rng.uniform(*scale)),
        rotation,
        rng.normal(0, 5, 3),
    )


def corridor(count=480, path=120.0, depth=4.0, seed=0):
    """A causeway several tens of scene depths long, and what each frame of it can see.

    Returns the ground-truth camera centres and, per frame, the points that frame observes: a
    slab of wall, floor and ceiling about one scene depth away. Consecutive frames see almost the
    same slab, frames a window apart see none of the same surface -- which is the whole reason one
    global anchor set cannot carry this clip.
    """
    rng = np.random.default_rng(seed)
    along = np.linspace(0.0, path, count)
    centres = np.stack([rng.normal(0, 0.02, count), np.full(count, 1.2), along], axis=1)
    seen = []
    for centre in centres:
        ahead = rng.uniform(0.3 * depth, 1.8 * depth, 900)
        seen.append(
            np.stack(
                [
                    rng.uniform(-2.0, 2.0, 900),
                    rng.uniform(0.0, 3.0, 900),
                    centre[2] + ahead,
                ],
                axis=1,
            )
        )
    return centres, seen


def scene_depth(points, centre):
    """The same median distance from a camera to its own points the stage measures."""
    return float(np.median(np.linalg.norm(np.asarray(points) - np.asarray(centre), axis=1)))


class WindowPlanning(unittest.TestCase):
    def test_a_co_visible_clip_plans_one_window_and_keeps_the_global_path(self):
        # hp-fly-s63 measured 0.01 scene depths, img5594 0.68: a whole clip in one window, which
        # is what the global solve already is.
        poses = straight_poses([0.02] * 7)
        spread, depth = measure_spread(poses, [1.0] * 8)
        self.assertLess(spread, ANCHOR_SPREAD_LIMIT)
        self.assertEqual(spread, anchor_spread(poses, [1.0] * 8)[0])
        self.assertEqual(depth, 1.0)
        profile = travel_profile(poses, list(range(0, 78, 11))[:8], depth, 78)
        self.assertEqual(plan_windows(profile, 78), [(0, 77)])

    def test_a_traversal_is_split_instead_of_refused(self):
        profile = measured_profile(GAME_RUN)
        windows = plan_windows(profile, GAME_RUN["count"])
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], GAME_RUN["count"] - 1)

    def test_every_planned_window_fits_inside_the_spread_budget(self):
        for clip in (GAME_RUN, GAME_S1):
            profile = measured_profile(clip)
            windows = plan_windows(profile, clip["count"])
            for first, last in windows[:-1]:
                # The surveyed path a window spends, which for a one-way walk IS its largest
                # baseline, stays inside the planned share of the limit.
                self.assertLessEqual(
                    profile[last] - profile[first],
                    CHAIN_TARGET_SPREAD * ANCHOR_SPREAD_LIMIT + 1e-9,
                    f"{clip['count']} samples, window {first}-{last}",
                )

    def test_windows_cover_the_clip_and_overlap_by_exactly_the_shared_frames(self):
        for clip in (GAME_RUN, GAME_S1):
            windows = plan_windows(measured_profile(clip), clip["count"])
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], clip["count"] - 1)
            owned = list(range(windows[0][0], windows[0][1] + 1))
            for (_, previous_last), (first, last) in zip(windows, windows[1:]):
                self.assertEqual(first, previous_last - CHAIN_OVERLAP_SAMPLES + 1)
                owned.extend(range(previous_last + 1, last + 1))
            self.assertEqual(owned, list(range(clip["count"])))

    def test_no_window_is_shorter_than_the_floor(self):
        for clip in (GAME_RUN, GAME_S1):
            for first, last in plan_windows(measured_profile(clip), clip["count"]):
                self.assertGreaterEqual(last - first + 1, CHAIN_MIN_WINDOW)

    def test_the_plan_reports_what_it_would_choose_on_the_two_real_game_runs(self):
        # game-run: 78 samples over a surveyed 3.68-depth path, cut into 0-47 and 44-77.
        # game-s1: 269 samples over 18.81 depths, cut into 13 windows of 12 to 57 samples -- it
        # spends seven short windows on its first 73, where the survey puts 10.4 of those depths,
        # and long ones afterwards. Both clips are refused outright today.
        for clip in (GAME_RUN, GAME_S1):
            profile = measured_profile(clip)
            self.assertAlmostEqual(float(profile[-1]), clip["path"], places=3)
            self.assertEqual(len(plan_windows(profile, clip["count"])), clip["windows"])
        self.assertEqual(plan_windows(measured_profile(GAME_RUN), 78), [(0, 47), (44, 77)])

    def test_an_absorbed_tail_is_the_only_window_allowed_past_the_planned_budget(self):
        # game-run's second window takes 2.04 surveyed depths rather than 1.79, because the nine
        # samples it would otherwise have left behind cannot be a window. That is still well
        # inside the limit each window is then measured against, and the measurement is what
        # decides: `adapt_window` shrinks any window the survey got wrong.
        profile = measured_profile(GAME_RUN)
        windows = plan_windows(profile, GAME_RUN["count"])
        spent = [float(profile[last] - profile[first]) for first, last in windows]
        self.assertLess(max(spent), ANCHOR_SPREAD_LIMIT)
        self.assertLessEqual(spent[0], CHAIN_TARGET_SPREAD * ANCHOR_SPREAD_LIMIT + 1e-9)

    def test_a_tail_too_short_for_its_own_window_is_absorbed(self):
        # A budget that would leave four samples over at the end must not plan a four-sample
        # window whose eight anchors are four views.
        profile = np.linspace(0.0, 4.0, 60)
        for first, last in plan_windows(profile, 60):
            self.assertGreaterEqual(last - first + 1, CHAIN_MIN_WINDOW)

    def test_a_still_camera_plans_one_window_however_long_the_clip(self):
        self.assertEqual(plan_windows(np.zeros(400), 400), [(0, 399)])

    def test_planning_refuses_a_clip_too_short_to_fill_a_window(self):
        with self.assertRaises(RuntimeError) as caught:
            plan_windows(np.zeros(4), 4)
        self.assertIn("one window", str(caught.exception))

    def test_the_profile_needs_a_pose_per_anchor_and_a_positive_depth(self):
        with self.assertRaises(RuntimeError) as caught:
            travel_profile(straight_poses([1.0, 1.0]), [0, 10], 1.0, 20)
        self.assertIn("pose per anchor sample", str(caught.exception))
        with self.assertRaises(RuntimeError) as caught:
            travel_profile(straight_poses([1.0]), [0, 10], 0.0, 20)
        self.assertIn("positive scene depth", str(caught.exception))

    def test_the_profile_interpolates_between_the_samples_the_anchors_sit_on(self):
        profile = travel_profile(straight_poses([2.0, 2.0]), [0, 10, 20], 1.0, 21)
        self.assertAlmostEqual(profile[0], 0.0)
        self.assertAlmostEqual(profile[5], 1.0)
        self.assertAlmostEqual(profile[20], 4.0)

    def test_next_window_never_goes_backwards_on_a_camera_that_jumps(self):
        profile = np.concatenate([[0.0], np.full(59, 99.0)])
        self.assertEqual(next_window(profile, 0, 59, 1.8), 0 + CHAIN_MIN_WINDOW - 1)


class WindowAdaptation(unittest.TestCase):
    def test_a_window_inside_the_limit_is_accepted(self):
        action, end = adapt_window(0, 39, 0.6 * ANCHOR_SPREAD_LIMIT, 200)
        self.assertEqual((action, end), ("accept", 39))

    def test_an_over_spread_window_shrinks_by_how_far_over_it_came(self):
        action, end = adapt_window(0, 99, 2 * ANCHOR_SPREAD_LIMIT, 400)
        self.assertEqual(action, "shrink")
        self.assertLess(end, 99)
        self.assertGreaterEqual(end, CHAIN_MIN_WINDOW - 1)

    def test_a_window_the_survey_over_estimated_grows_once(self):
        action, end = adapt_window(0, 39, 0.1 * ANCHOR_SPREAD_LIMIT, 400)
        self.assertEqual(action, "grow")
        self.assertGreater(end, 39)
        self.assertLessEqual(end - 0 + 1, 2 * 40)

    def test_the_last_window_of_a_clip_never_grows_past_it(self):
        action, end = adapt_window(360, 399, 0.01, 399)
        self.assertEqual((action, end), ("accept", 399))

    def test_a_window_already_at_the_floor_that_still_fails_says_so(self):
        action, _ = adapt_window(0, CHAIN_MIN_WINDOW - 1, 20.0, 400)
        self.assertEqual(action, "fail")

    def test_a_window_with_no_measurable_spread_fails_rather_than_resizing(self):
        action, _ = adapt_window(0, 99, np.nan, 400)
        self.assertEqual(action, "fail")


class TransformComposition(unittest.TestCase):
    def test_composing_with_the_identity_leaves_a_transform_bit_for_bit(self):
        # The global path composes every batch transform with window 0's identity, so this has to
        # be exact or a co-visible clip's geometry would move.
        rng = np.random.default_rng(7)
        inner = random_similarity(rng)
        scale, rotation, offset = compose_similarity(identity_similarity(), inner)
        self.assertEqual(scale, inner[0])
        np.testing.assert_array_equal(rotation, inner[1])
        np.testing.assert_array_equal(offset, inner[2])

    def test_a_composed_transform_moves_points_exactly_like_the_two_in_turn(self):
        rng = np.random.default_rng(8)
        outer, inner = random_similarity(rng), random_similarity(rng)
        points = rng.normal(0, 3, (500, 3))
        np.testing.assert_allclose(
            apply_similarity(compose_similarity(outer, inner), points),
            apply_similarity(outer, apply_similarity(inner, points)),
            atol=1e-9,
        )

    def test_applying_the_identity_keeps_the_caller_s_own_arithmetic(self):
        points = np.random.default_rng(9).normal(0, 1, (64, 3)).astype("float32")
        scale, rotation, offset = random_similarity(np.random.default_rng(10))
        np.testing.assert_array_equal(
            apply_similarity((scale, rotation, offset), points),
            points @ rotation.T * scale + offset,
        )

    def test_the_chain_starts_at_window_zero_s_own_frame(self):
        transforms = chain_transforms([])
        self.assertEqual(len(transforms), 1)
        self.assertEqual(transforms[0][0], 1.0)
        np.testing.assert_array_equal(transforms[0][1], np.eye(3))


class ChainedCorridor(unittest.TestCase):
    """A camera running the length of a corridor, solved as windows and put back together."""

    def build(self, seed=1, noise=0.002, count=480, path=120.0, depth=4.0, break_window=None):
        """A corridor solved window by window, each window in its own arbitrary frame."""
        centres, seen = corridor(count=count, path=path, depth=depth, seed=seed)
        rng = np.random.default_rng(seed + 100)
        anchors = np.unique(np.linspace(0, count - 1, 8).round().astype(int))
        poses = np.tile(np.eye(4), (len(anchors), 1, 1))
        poses[:, :3, 3] = centres[anchors]
        profile = travel_profile(poses, anchors, depth, count)
        windows = plan_windows(profile, count)
        placements, frames = [], {}
        for window, (first, last) in enumerate(windows):
            place = random_similarity(rng)
            placements.append(place)
            for i in range(first, last + 1):
                points = seen[i]
                if break_window is not None and window == break_window and i < first + 20:
                    # A window that registered onto the wrong surface: the same pixels, a
                    # different piece of corridor. This is what a failed link looks like.
                    points = seen[(i + count // 3) % count]
                frames[(window, i)] = dict(
                    points=apply_similarity(place, points)
                    + rng.normal(0, noise * depth * place[0], points.shape),
                    centre=apply_similarity(place, centres[i]),
                )
        return dict(
            centres=centres,
            windows=windows,
            frames=frames,
            placements=placements,
            depth=depth,
            profile=profile,
        )

    def truth(self, built, sample):
        """Where window 0's own frame puts a sample, which is what the chain has to reproduce."""
        return apply_similarity(built["placements"][0], built["centres"][sample])

    def unit(self, built):
        """One scene depth, in window 0's units."""
        base = built["frames"][(0, 0)]
        return scene_depth(base["points"], base["centre"])

    def links_of(self, built):
        """Fit every window onto the previous one from the frames they share, as the stage does."""
        links, transforms = [], []
        for index in range(1, len(built["windows"])):
            first = built["windows"][index][0]
            shared = list(range(first, first + CHAIN_OVERLAP_SAMPLES))
            here = [built["frames"][(index, i)] for i in shared]
            there = [built["frames"][(index - 1, i)] for i in shared]
            scale, rotation, offset, rms = similarity(
                np.concatenate([frame["points"] for frame in here]),
                np.concatenate([frame["points"] for frame in there]),
            )
            moved = apply_similarity(
                (scale, rotation, offset), np.array([frame["centre"] for frame in here])
            )
            camera_rms = float(
                np.sqrt(
                    np.mean(
                        np.sum((moved - np.array([frame["centre"] for frame in there])) ** 2, 1)
                    )
                )
            )
            links.append(
                link_quality(
                    index - 1,
                    shared,
                    sum(len(frame["points"]) for frame in here),
                    rms,
                    scale,
                    float(np.median([scene_depth(f["points"], f["centre"]) for f in there])),
                    float(np.median([scene_depth(f["points"], f["centre"]) for f in here])),
                    camera_rms,
                    remaining_path=float(len(built["windows"]) - index),
                )
            )
            transforms.append((scale, rotation, offset))
        return links, transforms

    def test_the_corridor_is_far_too_long_for_one_global_anchor_set(self):
        built = self.build()
        anchors = np.linspace(0, len(built["centres"]) - 1, 8).round().astype(int)
        poses = np.tile(np.eye(4), (8, 1, 1))
        poses[:, :3, 3] = built["centres"][anchors]
        with self.assertRaises(RuntimeError) as caught:
            anchor_spread(poses, [built["depth"]] * 8)
        self.assertIn("scene depths", str(caught.exception))
        self.assertGreater(len(built["windows"]), 4)

    def errors(self, built):
        """Every sample's recovered position against window 0's truth, in scene depths."""
        links, fits = self.links_of(built)
        transforms = chain_transforms(fits)
        unit = self.unit(built)
        found = {}
        for window, (first, last) in enumerate(built["windows"]):
            owned = first if window == 0 else first + CHAIN_OVERLAP_SAMPLES
            for i in range(owned, last + 1):
                placed = apply_similarity(
                    transforms[window], built["frames"][(window, i)]["centre"]
                )
                found[i] = float(np.linalg.norm(placed - self.truth(built, i))) / unit
        return links, found

    def test_the_chain_recovers_the_whole_camera_path_in_window_zero_s_frame(self):
        built = self.build()
        links, found = self.errors(built)
        self.assertTrue(all(link["ok"] for link in links), [link["failures"] for link in links])
        self.assertEqual(sorted(found), list(range(len(built["centres"]))))
        worst = max(found.values())
        self.assertLess(
            worst, 0.05, f"{worst:.4f} scene depths at sample {max(found, key=found.get)}"
        )

    def test_the_error_the_chain_leaves_grows_away_from_window_zero(self):
        # Drift is the honest cost of chaining: nothing ties the far end back to window 0.
        _, found = self.errors(self.build())
        early = np.mean([found[i] for i in range(0, 40)])
        late = np.mean([found[i] for i in range(len(found) - 40, len(found))])
        self.assertGreater(late, early)

    def test_the_far_end_of_the_chain_stays_inside_the_reported_drift_bound(self):
        built = self.build()
        links, found = self.errors(built)
        bound = drift_bound(links)
        self.assertGreater(bound["boundDepths"], 0.0)
        self.assertEqual(bound["links"], len(links))
        worst = max(found.values())
        self.assertLess(worst, bound["boundDepths"], f"{worst:.4f} over {bound['boundDepths']:.4f}")

    def test_a_noisier_solve_drifts_further_and_says_so(self):
        quiet = drift_bound(self.errors(self.build(noise=0.002))[0])
        loud = drift_bound(self.errors(self.build(noise=0.02))[0])
        self.assertGreater(loud["residualDepths"], quiet["residualDepths"])
        self.assertGreater(loud["boundDepths"], quiet["boundDepths"])

    def test_the_drift_bound_grows_with_the_chain(self):
        short = drift_bound([dict(rmsDepths=0.02, remainingPathDepths=4.0)])
        long = drift_bound([dict(rmsDepths=0.02, remainingPathDepths=4.0)] * 5)
        self.assertGreater(long["boundDepths"], short["boundDepths"])
        self.assertAlmostEqual(long["residualDepths"], 0.1)
        self.assertIn("cannot measure its own drift", long["note"])

    def test_a_link_that_never_fitted_is_ignored_by_the_bound_not_silently_zeroed(self):
        bound = drift_bound([dict(rmsDepths=float("nan"), remainingPathDepths=4.0)])
        self.assertEqual(bound["links"], 1)
        self.assertEqual(bound["boundDepths"], 0.0)

    def test_a_broken_link_is_detected_and_localised(self):
        built = self.build(break_window=3)
        links, _ = self.links_of(built)
        broken = [link for link in links if not link["ok"]]
        self.assertTrue(broken, "the deliberately wrong window registered anyway")
        self.assertEqual([link["link"] for link in broken], [2])
        self.assertTrue(broken[0]["failures"])
        found = localise_break(built["windows"], links)
        self.assertFalse(found["ok"])
        self.assertEqual(found["failedLinks"], [2])
        self.assertEqual(found["weakestLink"]["link"], 2)
        # Windows 0..2 registered to each other, so their samples are the prefix to trim to.
        self.assertEqual(found["registeredPrefix"]["windows"], [0, 2])
        self.assertEqual(found["registeredPrefix"]["firstSample"], 0)
        self.assertEqual(found["registeredPrefix"]["lastSample"], built["windows"][2][1])
        self.assertEqual(found["registeredSuffix"]["windows"], [3, len(built["windows"]) - 1])
        self.assertEqual(found["registeredSuffix"]["firstSample"], built["windows"][3][0])


class LinkJudgement(unittest.TestCase):
    def good(self, **over):
        fields = dict(
            index=0,
            samples=[40, 41, 42, 43],
            points=6000,
            rms=0.1,
            scale=2.0,
            depth_previous=4.0,
            depth_current=2.0,
            camera_rms=0.05,
        )
        fields.update(over)
        return link_quality(**fields)

    def test_a_clean_link_passes_and_reports_what_it_measured(self):
        record = self.good()
        self.assertTrue(record["ok"])
        self.assertEqual(record["failures"], [])
        self.assertAlmostEqual(record["rmsDepths"], 0.025)
        self.assertAlmostEqual(record["depthRatio"], 1.0)
        self.assertAlmostEqual(record["cameraRmsDepths"], 0.0125)
        self.assertEqual(record["sharedSamples"], [40, 41, 42, 43])

    def test_an_arbitrary_link_scale_is_not_itself_a_complaint(self):
        # Each Pi3X prediction picks its own scale, so a link scale of 7 is ordinary as long as
        # the shared frames measure the same depth once it has been applied.
        record = self.good(scale=7.0, depth_current=4.0 / 7.0)
        self.assertTrue(record["ok"], record["failures"])
        self.assertEqual(record["scale"], 7.0)

    def test_a_link_that_registered_onto_the_wrong_surface_fails_on_its_residual(self):
        record = self.good(rms=4.0 * MAX_LINK_RESIDUAL * 1.5)
        self.assertFalse(record["ok"])
        self.assertIn("alignment residual", record["failures"][0])

    def test_a_link_that_disagrees_about_the_shared_frames_depth_fails(self):
        record = self.good(depth_current=2.0 * LINK_DEPTH_RATIO_BAND * 1.2)
        self.assertFalse(record["ok"])
        self.assertTrue(any("deep in this window" in why for why in record["failures"]))

    def test_the_shared_frames_camera_centres_check_the_fit_they_were_left_out_of(self):
        record = self.good(camera_rms=4.0)
        self.assertFalse(record["ok"])
        self.assertTrue(any("camera centres" in why for why in record["failures"]))

    def test_a_link_with_no_measurable_depth_is_a_failure_not_a_nan(self):
        record = self.good(depth_previous=0.0)
        self.assertFalse(record["ok"])
        self.assertIn("no positive median depth", record["failures"][0])

    def test_the_remaining_path_is_carried_for_the_drift_bound(self):
        record = self.good(remaining_path=6.5)
        self.assertEqual(record["remainingPathDepths"], 6.5)


class BreakLocalisation(unittest.TestCase):
    def windows(self, count=5, length=30):
        first = 0
        out = []
        for _ in range(count):
            out.append((first, first + length - 1))
            first += length - CHAIN_OVERLAP_SAMPLES
        return out

    def test_a_chain_that_closed_reports_the_whole_clip(self):
        windows = self.windows()
        found = localise_break(windows, [dict(ok=True, rmsDepths=0.01)] * 4)
        self.assertTrue(found["ok"])
        self.assertEqual(found["registeredPrefix"]["windows"], [0, 4])
        self.assertEqual(found["registeredSuffix"]["windows"], [0, 4])
        self.assertEqual(found["failedLinks"], [])

    def test_the_weakest_link_of_a_closed_chain_is_its_worst_residual(self):
        windows = self.windows()
        links = [dict(ok=True, rmsDepths=r, link=i) for i, r in enumerate([0.01, 0.09, 0.02, 0.03])]
        self.assertEqual(localise_break(windows, links)["weakestLink"]["link"], 1)

    def test_a_mid_chain_break_gives_both_a_prefix_and_a_suffix_to_trim_to(self):
        windows = self.windows()
        links = [
            dict(ok=True, rmsDepths=0.01, link=0),
            dict(ok=False, rmsDepths=0.9, link=1),
            dict(ok=True, rmsDepths=0.02, link=2),
            dict(ok=True, rmsDepths=0.02, link=3),
        ]
        found = localise_break(windows, links)
        self.assertFalse(found["ok"])
        self.assertEqual(found["weakestLink"]["link"], 1)
        self.assertEqual(found["registeredPrefix"]["windows"], [0, 1])
        self.assertEqual(found["registeredPrefix"]["firstSample"], windows[0][0])
        self.assertEqual(found["registeredPrefix"]["lastSample"], windows[1][1])
        self.assertEqual(found["registeredSuffix"]["windows"], [2, 4])
        self.assertEqual(found["registeredSuffix"]["firstSample"], windows[2][0])
        self.assertEqual(found["registeredSuffix"]["lastSample"], windows[4][1])

    def test_a_link_the_run_never_reached_ends_a_run_and_is_named_separately(self):
        # The stage fails fast, so the windows past the break are planned, not registered: their
        # range is not offered as something that did register.
        windows = self.windows()
        links = [
            dict(ok=True, rmsDepths=0.01, link=0),
            dict(ok=False, rmsDepths=0.9, link=1),
            dict(ok=None, link=2),
            dict(ok=None, link=3),
        ]
        found = localise_break(windows, links)
        self.assertEqual(found["unattemptedLinks"], [2, 3])
        self.assertEqual(found["registeredPrefix"]["windows"], [0, 1])
        self.assertEqual(found["registeredSuffix"]["windows"], [4, 4])

    def test_a_break_at_the_first_link_leaves_window_zero_as_the_prefix(self):
        windows = self.windows(count=3)
        links = [dict(ok=False, rmsDepths=1.0, link=0), dict(ok=True, rmsDepths=0.01, link=1)]
        found = localise_break(windows, links)
        self.assertEqual(found["registeredPrefix"]["windows"], [0, 0])
        self.assertEqual(found["registeredPrefix"]["lastSample"], windows[0][1])
        self.assertEqual(found["registeredSuffix"]["windows"], [1, 2])

    def test_a_window_count_and_link_count_that_disagree_raise(self):
        with self.assertRaises(RuntimeError) as caught:
            localise_break(self.windows(count=3), [dict(ok=True)])
        self.assertIn("need 2 links", str(caught.exception))


class ModeChoice(unittest.TestCase):
    def test_the_two_modes_are_the_names_the_metadata_discloses(self):
        self.assertEqual(ALIGNMENT_GLOBAL, "global")
        self.assertEqual(ALIGNMENT_CHAINED, "chained")

    def test_the_limit_that_chooses_between_them_is_the_one_windows_are_held_to(self):
        # The mode is chosen on the same measurement each window then has to pass on its own, so
        # a clip under the limit is solved by one window and nothing about it changes.
        poses = straight_poses([0.1] * 7)
        spread, _ = measure_spread(poses, [1.0] * 8)
        self.assertLessEqual(spread, ANCHOR_SPREAD_LIMIT)
        anchor_spread(poses, [1.0] * 8)

    def test_the_two_real_game_runs_both_land_in_chained_mode(self):
        for clip in (GAME_RUN, GAME_S1):
            # What their saved anchor predictions actually measured, and what a straight walk of
            # the same consecutive baselines comes to: the walk is the longer of the two, because
            # a path is at least the distance between its ends.
            self.assertGreater(clip["spread"], ANCHOR_SPREAD_LIMIT)
            spread, _ = measure_spread(straight_poses(clip["steps"]), [1.0] * 8)
            self.assertAlmostEqual(spread, clip["path"], places=3)
            self.assertGreaterEqual(spread, clip["spread"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
