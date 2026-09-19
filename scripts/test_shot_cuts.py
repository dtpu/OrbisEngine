"""Cut detection against synthetic clips whose cuts are known by construction.

The detector's job is to separate an edit from fast motion, and only footage with both can show
whether it does. So the fixtures here are rendered, not recorded: a panorama is panned behind a
window, the exposure is integrated over the shutter to give real directional blur, and a cut is a
swap of the panorama on a named frame. That makes the expected answer exact -- cut at frame 45, no
cut anywhere else -- which recorded footage never is.

Fixtures are built in a TemporaryDirectory at test time and deleted with it. No media is committed.

  uv run --locked python scripts/test_shot_cuts.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot_cuts

FIXTURE_WIDTH = 640
FIXTURE_HEIGHT = 360
FIXTURE_FPS = 30.0
FIXTURE_FRAMES = 90
CUT_FRAME = 45  # every cut fixture cuts here, so the expected time is CUT_FRAME / FIXTURE_FPS
SHUTTER_SAMPLES = 8  # sub-exposures averaged per frame; this is what makes the blur directional
# A 180-degree shutter, the film standard: the sensor is open for half the frame interval, so a
# frame smears over half its own displacement. Integrating the whole interval instead (a 360-degree
# shutter, which nothing shoots) blurs a whip pan into porridge and makes every fixture easier than
# real footage.
SHUTTER_ANGLE = 0.5


# ---------------------------------------------------------------- rendering
SPECTRUM = 0.85  # octave amplitude falls as scale**SPECTRUM; see panorama()


def panorama(seed: int, width: int, height: int, warm: bool = True):
    """A wide backdrop built from octaves of noise, coarse to fine, plus drawn structures.

    The octave weighting matters more than it looks. A photograph's power spectrum falls off with
    spatial frequency: most of its energy is in large shapes, and fine detail rides on top. Flat
    per-pixel noise instead decorrelates completely under a 30 px blur, so a whip pan over it turns
    every frame pair into two unrelated images -- the frame difference across a fast pan becomes as
    large as the difference across a cut, and no detector could separate them. Real footage is not
    like that, and neither is this.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    img, scale = np.zeros((height, width, 3), np.float32), 256
    while scale >= 2:
        small = rng.normal(0.0, 1.0, (max(height // scale, 2), max(width // scale, 2), 3))
        octave = cv2.resize(
            small.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
        )
        img += octave * scale**SPECTRUM
        scale //= 2
    img = 128.0 + 52.0 * img / max(float(img.std()), 1e-6)
    # Grit: the leaf, gravel and fabric scale. It is what ORB actually keys on in a sharp frame and
    # the first thing a 40 px smear removes, so a fixture without it cannot show the blur-aware
    # bracket search doing anything.
    speck = rng.normal(0.0, 1.0, (max(height // 2, 2), max(width // 2, 2), 3)).astype(np.float32)
    img += cv2.resize(speck, (width, height), interpolation=cv2.INTER_LANCZOS4) * 17.0
    tint = np.array([1.12, 1.0, 0.86] if warm else [0.86, 1.0, 1.12], np.float32)
    img = np.clip(img * tint, 0, 255).astype(np.uint8)
    for _ in range(width // 26):
        a = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        b = (a[0] + int(rng.integers(-220, 220)), a[1] + int(rng.integers(-220, 220)))
        colour = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.line(img, a, b, colour, int(rng.integers(2, 9)))
        cv2.circle(img, b, int(rng.integers(5, 26)), colour, -1)
    return img


def _crop(pano, x: float, y: float, w: int, h: int):
    import numpy as np

    height, width = pano.shape[:2]
    x0 = int(np.clip(round(x), 0, width - w))
    y0 = int(np.clip(round(y), 0, height - h))
    return pano[y0 : y0 + h, x0 : x0 + w]


def _exposure(pano, previous: tuple[float, float], now: tuple[float, float], w: int, h: int):
    """One frame, integrated over the shutter from `previous` to `now`.

    Averaging the sub-exposures is how a camera makes motion blur, so a fast pan here blurs the way
    a fast pan does and a slow one stays sharp, with no blur parameter to tune.
    """
    import numpy as np

    acc = np.zeros((h, w, 3), np.float32)
    for k in range(SHUTTER_SAMPLES):
        f = 1.0 - SHUTTER_ANGLE * (k / (SHUTTER_SAMPLES - 1))
        acc += _crop(pano, *(previous[i] + (now[i] - previous[i]) * f for i in (0, 1)), w, h)
    return (acc / SHUTTER_SAMPLES).astype(np.uint8)


def pan(pano, xs: list[float], ys: list[float], w: int = FIXTURE_WIDTH, h: int = FIXTURE_HEIGHT):
    """Frames along a trajectory.

    The position before the first frame is extrapolated rather than repeated. Repeating it leaves
    frame 0 perfectly sharp while every other frame carries its motion blur, and in a cut fixture
    that hands the detector a cue no real cut has: the first frame of the incoming shot is blurred
    like any other.
    """
    before = [(2 * xs[0] - xs[1], 2 * ys[0] - ys[1])] + list(zip(xs, ys))[:-1]
    return [_exposure(pano, before[i], (xs[i], ys[i]), w, h) for i in range(len(xs))]


def ramp(start: float, speeds: list[tuple[int, float]], ease: float = 6.0) -> list[float]:
    """Positions from (frame count, pixels per frame) runs, eased between runs."""
    xs, x, previous = [], start, speeds[0][1]
    for count, speed in speeds:
        for i in range(count):
            blend = min((i + 1) / max(ease, 1.0), 1.0)
            x += previous + (speed - previous) * blend
            xs.append(x)
        previous = speed
    return xs


def handheld(seed: int, n: int, amplitude: float):
    """Frame-to-frame wobble of an operator's hands, added on top of the intended move.

    This is not decoration. ffmpeg's scene score is min(mafd, |mafd - previous mafd|)/100, so a
    perfectly smooth move scores near zero however fast it is: the frame difference is large but
    constant. Real footage is never smooth, and it is the wobble -- displacement that changes every
    frame -- that drives the score up and produces candidates in the middle of continuous motion.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, amplitude, (n, 2))
    kernel = np.array([0.25, 0.5, 0.25])
    return np.stack([np.convolve(raw[:, i], kernel, mode="same") for i in (0, 1)], axis=1)


def encode(frames: list, path: Path, fps: float = FIXTURE_FPS) -> Path:
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "veryfast",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True,
    )
    return path


# ---------------------------------------------------------------- the fixtures
# Each returns (frames, expected cut frame indices). "Continuous" fixtures return no cut frames and
# must produce no cuts at all; a cut fixture must be found at exactly its frame.
def _shaky(pano, xs: list[float], ys: list[float], seed: int, amplitude: float):
    wobble = handheld(seed, len(xs), amplitude)
    return pan(
        pano,
        [x + w for x, w in zip(xs, wobble[:, 0])],
        [y + w for y, w in zip(ys, wobble[:, 1])],
    )


def _wide_enough(seed: int, xs: list[float], height: int = 700, warm: bool = True):
    """A panorama sized to the move it has to carry, so a whip never runs off its own backdrop."""
    return panorama(seed, int(max(xs)) + FIXTURE_WIDTH + 160, height, warm=warm)


def fixture_a_slow_pan():
    """A. Slow handheld pan over texture. The easy continuous case."""
    pano = panorama(11, 1400, 600)
    xs = ramp(60.0, [(FIXTURE_FRAMES, 1.1)])
    ys = [140 + 0.3 * i for i in range(FIXTURE_FRAMES)]
    return _shaky(pano, xs, ys, 12, 1.6), []


def fixture_b_whip_pan():
    """B. Whip pans, blurred and erratic. The false-positive class this work is about.

    An operator whipping to follow action does not accelerate smoothly: the camera sits, snaps,
    overshoots, drags back and snaps again. Every speed change is a step in the frame difference,
    which is exactly what the scene score responds to.
    """
    xs = ramp(
        120.0,
        [
            (8, 3.0),
            (16, 132.0),
            (6, 84.0),
            (7, -40.0),
            (6, 5.0),
            (5, 118.0),
            (20, 96.0),
            (22, 6.0),
        ],
        ease=1.5,
    )
    ys = ramp(200.0, [(20, 2.0), (14, -11.0), (26, 7.0), (30, -3.0)], ease=1.5)
    return _shaky(_wide_enough(23, xs), xs, ys, 24, 5.0), []


def fixture_b2_whip_beyond_search():
    """B2. A whip fast enough that the coarse carry search cannot follow it.

    Past about a fifth of the frame width per frame the shift search runs out of room, so carry
    collapses for a run of frames even though nothing was cut. Nothing but temporal isolation can
    clear that: the frames around a cut agree with each other, the frames around this one do not.
    """
    xs = ramp(
        120.0, [(6, 4.0), (24, 235.0), (8, 150.0), (10, 6.0), (20, 210.0), (22, 8.0)], ease=1.5
    )
    ys = ramp(220.0, [(30, 4.0), (30, -12.0), (30, 5.0)], ease=1.5)
    return _shaky(_wide_enough(211, xs), xs, ys, 212, 6.0), []


def fixture_c_foreground():
    """C. Large foreground subjects crossing fast over a moving background, blurred.

    The subjects enter and leave the frame rather than gliding across it, so the amount of the
    frame that is changing steps up and down the way it does when players cross camera.
    """
    import cv2
    import numpy as np

    xs = ramp(120.0, [(26, 18.0), (10, 62.0), (18, 6.0), (36, 30.0)], ease=1.5)
    pano = _wide_enough(37, xs)
    ys = [200 + 9.0 * np.sin(i / 5.0) for i in range(FIXTURE_FRAMES)]
    wobble = handheld(38, FIXTURE_FRAMES, 4.0)
    xs = [x + w for x, w in zip(xs, wobble[:, 0])]
    ys = [y + w for y, w in zip(ys, wobble[:, 1])]
    # The second subject passes close to the lens: it is wide, dark and crosses in a few frames, so
    # the fraction of the frame that is changing steps up and down instead of drifting.
    movers = [
        (-520.0, 78.0, 30, (232, 228, 210), (62, 108)),
        (1500.0, -210.0, 10, (26, 30, 44), (190, 300)),
    ]
    frames = []
    for i in range(FIXTURE_FRAMES):
        acc = np.zeros((FIXTURE_HEIGHT, FIXTURE_WIDTH, 3), np.float32)
        for k in range(SHUTTER_SAMPLES):
            f = i - SHUTTER_ANGLE * (k / (SHUTTER_SAMPLES - 1))
            grid = range(FIXTURE_FRAMES)
            layer = _crop(
                pano, np.interp(f, grid, xs), np.interp(f, grid, ys), FIXTURE_WIDTH, FIXTURE_HEIGHT
            ).copy()
            for x0, speed, top, colour, (rx, ry) in movers:
                cx = int(x0 + speed * f)
                cv2.ellipse(layer, (cx, top + ry), (rx, ry), 0, 0, 360, colour, -1)
                cv2.ellipse(layer, (cx, top), (rx // 2, rx // 2), 0, 0, 360, colour, -1)
            acc += layer
        frames.append((acc / SHUTTER_SAMPLES).astype("uint8"))
    return frames, []


def fixture_d_cut_places():
    """D. A hard cut between two different places."""
    first = panorama(51, 1400, 600, warm=True)
    second = panorama(67, 1400, 600, warm=False)
    head = _shaky(first, ramp(60.0, [(CUT_FRAME, 1.4)]), [140.0] * CUT_FRAME, 52, 1.8)
    n = FIXTURE_FRAMES - CUT_FRAME
    tail = _shaky(second, ramp(300.0, [(n, 1.6)]), [200.0] * n, 68, 1.8)
    return head + tail, [CUT_FRAME]


def fixture_e_cut_same_place():
    """E. A hard cut between two views of the SAME place: same palette, different content."""
    pano = panorama(83, 3200, 700)
    n = FIXTURE_FRAMES - CUT_FRAME
    head = _shaky(pano, ramp(120.0, [(CUT_FRAME, 1.3)]), [90.0] * CUT_FRAME, 84, 1.8)
    tail = _shaky(pano, ramp(2300.0, [(n, 1.5)]), [300.0] * n, 85, 1.8)
    return head + tail, [CUT_FRAME]


def fixture_f_cut_in_motion():
    """F. A hard cut in the middle of fast blurred motion: both shots are whipping."""
    n = FIXTURE_FRAMES - CUT_FRAME
    head_x = ramp(120.0, [(10, 8.0), (20, 96.0), (15, 78.0)], ease=1.5)
    tail_x = ramp(200.0, [(n, 88.0)], ease=1.5)
    head = _shaky(_wide_enough(97, head_x), head_x, ramp(240.0, [(CUT_FRAME, -3.0)]), 98, 5.0)
    tail = _shaky(
        _wide_enough(113, tail_x, warm=False), tail_x, ramp(260.0, [(n, 2.0)], ease=1.5), 114, 5.0
    )
    return head + tail, [CUT_FRAME]


def fixture_g_grain():
    """G. A static shot buried in film grain."""
    import numpy as np

    rng = np.random.default_rng(131)
    still = _crop(panorama(127, 1200, 600), 200, 150, FIXTURE_WIDTH, FIXTURE_HEIGHT).astype(
        np.float32
    )
    return [
        np.clip(still + rng.normal(0, 21.0, still.shape), 0, 255).astype(np.uint8)
        for _ in range(FIXTURE_FRAMES)
    ], []


def fixture_h_flash(saturating: bool = False):
    """H. A one-to-two frame exposure jump inside a continuous shot.

    A camera flash, a strobe or lightning is not an edit, so this must stay continuous. The
    saturating variant blows the frame to white: the content is then genuinely absent from those
    frames and no frame-pair evidence can separate it from a cut, which is why it is characterised
    but not asserted.
    """
    import numpy as np

    pano = panorama(149, 1400, 600)
    frames = _shaky(pano, ramp(60.0, [(FIXTURE_FRAMES, 1.2)]), [140.0] * FIXTURE_FRAMES, 150, 1.6)
    for i in (CUT_FRAME, CUT_FRAME + 1):
        f = frames[i].astype(np.float32)
        frames[i] = np.clip(
            f * (0.15 + 240.0 / max(f.mean(), 1.0)) if saturating else f * 1.55, 0, 255
        ).astype(np.uint8)
    return frames, []


FIXTURES = {
    "A_slow_pan": fixture_a_slow_pan,
    "B_whip_pan": fixture_b_whip_pan,
    "B2_whip_beyond_search": fixture_b2_whip_beyond_search,
    "C_foreground": fixture_c_foreground,
    "D_cut_places": fixture_d_cut_places,
    "E_cut_same_place": fixture_e_cut_same_place,
    "F_cut_in_motion": fixture_f_cut_in_motion,
    "G_grain": fixture_g_grain,
    "H_flash": fixture_h_flash,
    "H_flash_saturating": lambda: fixture_h_flash(saturating=True),
}


def build(name: str, directory: Path) -> tuple[Path, list[float]]:
    """Render and encode one fixture; returns its path and the exact times of its cuts."""
    frames, cut_frames = FIXTURES[name]()
    path = encode(frames, directory / f"{name}.mp4")
    return path, [n / FIXTURE_FPS for n in cut_frames]


# ---------------------------------------------------------------- tests
CONTINUOUS = [
    "A_slow_pan",
    "B_whip_pan",
    "B2_whip_beyond_search",
    "C_foreground",
    "G_grain",
    "H_flash",
    "H_flash_saturating",
]
CUTS = ["D_cut_places", "E_cut_same_place", "F_cut_in_motion"]
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
# Comfortably above CARRY_CARRIES: these cases are not marginal, and a test that only just passed
# would not say what it means to say.
CARRY_CLEARLY_CARRIES = 0.9


class Clips:
    """One TemporaryDirectory of rendered fixtures, built on first use and shared by the tests."""

    directory: tempfile.TemporaryDirectory | None = None
    built: ClassVar[dict[str, tuple[Path, list[float]]]] = {}

    @classmethod
    def get(cls, name: str) -> tuple[Path, list[float]]:
        if cls.directory is None:
            cls.directory = tempfile.TemporaryDirectory(prefix="shotcuts-fixtures-")
        if name not in cls.built:
            cls.built[name] = build(name, Path(cls.directory.name))
        return cls.built[name]

    @classmethod
    def clear(cls):
        cls.built = {}
        if cls.directory is not None:
            cls.directory.cleanup()
            cls.directory = None


def tearDownModule():
    Clips.clear()


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class ContinuousClipTests(unittest.TestCase):
    """Nothing in these clips is an edit, so nothing in them may be reported as a cut."""

    def test_no_cuts(self):
        for name in CONTINUOUS:
            with self.subTest(fixture=name):
                video, _ = Clips.get(name)
                found = shot_cuts.detect_cuts(video)
                cuts = [c for c in found if c["cut"]]
                self.assertEqual(
                    cuts,
                    [],
                    f"{name}: {len(cuts)} false cut(s): "
                    + "; ".join(f"{c['time']}s {c['why']}" for c in cuts),
                )

    def test_every_cleared_candidate_says_why(self):
        for name in CONTINUOUS:
            with self.subTest(fixture=name):
                video, _ = Clips.get(name)
                for c in shot_cuts.detect_cuts(video):
                    self.assertTrue(
                        c["why"].strip(), f"{name}: candidate at {c['time']}s has no why"
                    )
                    self.assertIsNotNone(c["carry"])

    def test_the_fixtures_reach_the_detector_at_all(self):
        """A clip that produces no candidate proves nothing, so check the hard ones produce some.

        B, B2, C and H exist to be argued with. If a change to the renderer quietened them below
        the candidate gate, test_no_cuts would still pass while testing nothing.
        """
        for name in ("B_whip_pan", "B2_whip_beyond_search", "C_foreground", "H_flash"):
            with self.subTest(fixture=name):
                video, _ = Clips.get(name)
                self.assertTrue(
                    shot_cuts.detect_cuts(video), f"{name} produced no candidate to clear"
                )


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class CutClipTests(unittest.TestCase):
    """Each of these is cut once, on a frame this file chose, and must be found there."""

    def test_found_at_the_right_frame(self):
        for name in CUTS:
            with self.subTest(fixture=name):
                video, expected = Clips.get(name)
                cuts = [c for c in shot_cuts.detect_cuts(video) if c["cut"]]
                self.assertEqual(len(cuts), 1, f"{name}: {[c['time'] for c in cuts]}")
                off = abs(cuts[0]["time"] - expected[0]) * FIXTURE_FPS
                self.assertLessEqual(
                    off,
                    1.0001,
                    f"{name}: cut reported at {cuts[0]['time']}s, expected {expected[0]}s "
                    f"({off:.2f} frames off)",
                )

    def test_the_cut_time_is_the_first_frame_of_the_new_shot(self):
        """shots_from_cuts and trim() both read `time` that way; the report must mean it."""
        video, expected = Clips.get("D_cut_places")
        report = shot_cuts.cut_report(video, score=False)
        self.assertFalse(report["continuous"])
        self.assertEqual(report["cutCount"], 1)
        self.assertEqual(report["shotCount"], 2)
        self.assertAlmostEqual(report["shots"][0]["end"], report["cuts"][0]["time"], places=3)
        self.assertAlmostEqual(report["shots"][1]["start"], report["cuts"][0]["time"], places=3)
        self.assertAlmostEqual(report["cuts"][0]["time"], expected[0], delta=1.0 / FIXTURE_FPS)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class ReportContractTests(unittest.TestCase):
    """run_clip.py reads this report and this exit status; both are a contract."""

    def test_continuous_report_shape_and_exit_status(self):
        video, _ = Clips.get("B_whip_pan")
        report = shot_cuts.cut_report(video, score=False)
        self.assertTrue(report["continuous"])
        self.assertEqual(report["cutCount"], 0)
        self.assertEqual(report["cuts"], [])
        self.assertEqual(report["shotCount"], 1)
        self.assertTrue(report["cleared"])
        self.assertEqual(self.run_cli(video), 0)

    def test_cut_report_shape_and_exit_status(self):
        video, _ = Clips.get("E_cut_same_place")
        report = shot_cuts.cut_report(video, score=False)
        self.assertEqual(report["cutCount"], len(report["cuts"]))
        self.assertEqual(report["continuous"], report["cutCount"] == 0)
        self.assertEqual(self.run_cli(video), 1)

    def test_every_candidate_carries_its_evidence(self):
        video, _ = Clips.get("D_cut_places")
        report = shot_cuts.cut_report(video, score=False)
        for c in report["cuts"] + report["cleared"]:
            for field in ("time", "frame", "score", "carry", "cut", "why"):
                self.assertIn(field, c)
            self.assertTrue(c["why"].strip())

    def run_cli(self, video: Path) -> int:
        with tempfile.TemporaryDirectory() as work:
            out = Path(work) / "cuts.json"
            done = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "shot_cuts.py"),
                    "--video",
                    str(video),
                    "--json",
                    str(out),
                    "--no-score",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertTrue(out.exists(), done.stdout)
            doc = json.loads(out.read_text())
            self.assertEqual(done.returncode, int(not doc["continuous"]), done.stdout)
        return done.returncode


class IsolationTests(unittest.TestCase):
    """The isolation ratio, on written-out series. No video, no ffmpeg."""

    def test_a_spike_in_a_calm_stretch_is_isolated(self):
        series = [0.01] * 40
        series[20] = 0.80
        self.assertGreater(shot_cuts.isolation(series, 20), shot_cuts.ISOLATION_MIN)

    def test_a_plateau_is_not_isolated_however_high_it_sits(self):
        series = [0.01] * 10 + [0.75] * 24 + [0.01] * 10
        for i in (14, 20, 28):
            with self.subTest(index=i):
                self.assertLess(shot_cuts.isolation(series, i), shot_cuts.ISOLATION_MIN)

    def test_a_spike_on_a_plateau_is_measured_against_the_plateau(self):
        """The whole point: a cut inside fast motion is still a cut, and the motion is still not."""
        series = [0.01] * 10 + [0.20] * 24 + [0.01] * 10
        series[22] = 0.95
        self.assertAlmostEqual(shot_cuts.isolation(series, 22), 0.95 / 0.20, places=6)
        self.assertLess(shot_cuts.isolation(series, 18), shot_cuts.ISOLATION_MIN)

    def test_the_immediate_neighbours_are_excluded(self):
        """A cut drags the frames either side of it up; counting them would hide the cut."""
        series = [0.02] * 40
        series[19] = series[20] = series[21] = 0.90
        self.assertAlmostEqual(shot_cuts.isolation(series, 20), 0.90 / 0.02, places=6)

    def test_the_floor_bounds_a_spike_in_a_dead_still_shot(self):
        series = [0.0] * 40
        series[20] = 0.5
        self.assertAlmostEqual(shot_cuts.isolation(series, 20), 0.5 / shot_cuts.ISOLATION_FLOOR)

    def test_the_window_shortens_at_the_start_and_the_end(self):
        series = [0.01] * 40
        series[1] = series[38] = 0.60
        for i in (1, 38):
            with self.subTest(index=i):
                self.assertGreater(shot_cuts.isolation(series, i), shot_cuts.ISOLATION_MIN)

    def test_nothing_to_compare_against_counts_as_isolated(self):
        self.assertEqual(shot_cuts.isolation([0.5], 0), float("inf"))
        self.assertEqual(shot_cuts.isolation([0.1, 0.5], 1), float("inf"))


class CarryTests(unittest.TestCase):
    """The coarse similarity, on made-up thumbnails. No video, no ffmpeg."""

    def thumb(self, seed: int):
        import cv2
        import numpy as np

        rng = np.random.default_rng(seed)
        small = rng.integers(0, 255, (shot_cuts.THUMB_HEIGHT // 4, shot_cuts.THUMB_WIDTH // 4))
        return cv2.resize(
            small.astype("uint8"),
            (shot_cuts.THUMB_WIDTH, shot_cuts.THUMB_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

    def test_a_frame_carries_into_itself(self):
        a = self.thumb(1)
        self.assertGreater(shot_cuts.carry(a, a), 0.999)

    def test_a_shift_within_the_search_still_carries(self):
        import numpy as np

        a = self.thumb(2)
        shifted = np.roll(a, (shot_cuts.CARRY_SEARCH_X - 4, 3), axis=(1, 0))
        self.assertGreater(shot_cuts.carry(a, shifted), CARRY_CLEARLY_CARRIES)

    def test_an_exposure_jump_still_carries(self):
        """Normalised correlation is why a flash frame is not an edit."""
        import numpy as np

        a = self.thumb(3)
        brighter = np.clip(a.astype(np.float32) * 1.6 + 20, 0, 255).astype("uint8")
        self.assertGreater(shot_cuts.carry(a, brighter), CARRY_CLEARLY_CARRIES)

    def test_an_unrelated_frame_does_not_carry(self):
        self.assertLess(shot_cuts.carry(self.thumb(4), self.thumb(5)), shot_cuts.CARRY_CARRIES)


def main() -> int:
    unittest.main(argv=[sys.argv[0]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
