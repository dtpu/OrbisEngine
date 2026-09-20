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


DOUBLED_CUT_FRAME = 2 * (CUT_FRAME // 2)  # the container frame the doubled fixture cuts on


def fixture_i_duplicated_frames():
    """I. A hard cut in 30 fps content carried in a 60 fps container: every frame is doubled.

    This is what a phone or a broadcast conversion hands the detector, and it used to defeat the
    ORB stage outright: the frame "before" the cut could be the cut frame's own duplicate, which
    matches itself perfectly and clears the cut. The cut is on content frame CUT_FRAME // 2, which
    is container frame DOUBLED_CUT_FRAME.
    """
    half = CUT_FRAME // 2
    first = panorama(311, 1400, 600, warm=True)
    second = panorama(317, 1400, 600, warm=False)
    head = _shaky(first, ramp(60.0, [(half, 2.2)]), [140.0] * half, 312, 1.8)
    n = FIXTURE_FRAMES // 2 - half
    tail = _shaky(second, ramp(300.0, [(n, 2.4)]), [200.0] * n, 318, 1.8)
    doubled = [f for frame in head + tail for f in (frame, frame.copy())]
    return doubled, [DOUBLED_CUT_FRAME]


def _dim(frame, gain: float = 0.11, lift: float = 5.0):
    """Crush a frame into the bottom of the range: a night exterior, not a graded image."""
    import numpy as np

    return np.clip(frame.astype(np.float32) * gain + lift, 0, 255).astype(np.uint8)


def fixture_j_dark_cut():
    """J. A hard cut between two dark low-contrast shots, which the scene score cannot see.

    ffmpeg's score is a mean absolute difference: squeeze the picture into fifteen grey levels and
    a splice between two unrelated places scores like nothing happening. Four of the real labelled
    cuts were missed exactly this way, so the fixture asserts the score really is under the gate as
    well as asserting the cut is found.
    """
    first = panorama(331, 1400, 600, warm=True)
    second = panorama(347, 1400, 600, warm=False)
    head = _shaky(first, ramp(60.0, [(CUT_FRAME, 1.0)]), [140.0] * CUT_FRAME, 332, 1.2)
    n = FIXTURE_FRAMES - CUT_FRAME
    tail = _shaky(second, ramp(300.0, [(n, 1.1)]), [200.0] * n, 348, 1.2)
    return [_dim(f) for f in head + tail], [CUT_FRAME]


FADE_FRAMES = 12  # each side of the fade in fixture K


def fixture_k_fade_to_black():
    """K. Two shots joined by a fade out to black and a fade up out of it.

    Nothing in the frame-pair evidence can see this: the scene score reads the per-frame luma step
    as nothing happening, and `carry` is a normalised correlation, so a dimmer copy of the frame
    before still correlates with it. It has to be found as a ramp, and reported as the region it
    occupies, or one of the two shots keeps a stretch of frames with no picture in them.
    """
    import numpy as np

    pano_a = panorama(353, 1400, 600, warm=True)
    pano_b = panorama(359, 1400, 600, warm=False)
    hold = CUT_FRAME - FADE_FRAMES
    head = _shaky(pano_a, ramp(60.0, [(hold + FADE_FRAMES, 1.2)]), [140.0] * CUT_FRAME, 354, 1.2)
    n = FIXTURE_FRAMES - CUT_FRAME
    tail = _shaky(pano_b, ramp(300.0, [(n, 1.3)]), [200.0] * n, 360, 1.2)
    frames = []
    for i, f in enumerate(head):
        k = i - hold
        scale = 1.0 if k < 0 else max(1.0 - (k + 1) / FADE_FRAMES, 0.0)
        frames.append((f.astype(np.float32) * scale).astype(np.uint8))
    for i, f in enumerate(tail):
        scale = min((i + 1) / FADE_FRAMES, 1.0)
        frames.append((f.astype(np.float32) * scale).astype(np.uint8))
    return frames, [CUT_FRAME]


def fixture_l_occlusion_returns():
    """L. Two frames of something crossing hard against the lens, and the view comes back.

    A player, a bludger or an arm blacks out most of the frame for a frame or two. The pair either
    side of it looks exactly like a cut; the only thing that says otherwise is that the view is
    back a few frames later, which is what `returns` is for.
    """
    import cv2
    import numpy as np

    pano = panorama(367, 1400, 600)
    frames = _shaky(pano, ramp(60.0, [(FIXTURE_FRAMES, 1.1)]), [140.0] * FIXTURE_FRAMES, 368, 1.5)
    for k, i in enumerate((CUT_FRAME, CUT_FRAME + 1)):
        layer = frames[i].copy()
        cx = int(FIXTURE_WIDTH * (0.35 + 0.30 * k))
        cv2.ellipse(
            layer,
            (cx, FIXTURE_HEIGHT // 2),
            (FIXTURE_WIDTH, FIXTURE_HEIGHT // 2 + 40),
            0,
            0,
            360,
            (18, 16, 22),
            -1,
        )
        frames[i] = np.clip(layer, 0, 255).astype(np.uint8)
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
    "I_duplicated_frames": fixture_i_duplicated_frames,
    "J_dark_cut": fixture_j_dark_cut,
    "K_fade_to_black": fixture_k_fade_to_black,
    "L_occlusion_returns": fixture_l_occlusion_returns,
}
# A fixture whose container runs at a different rate than its content says so here.
FIXTURE_RATES = {"I_duplicated_frames": 2 * FIXTURE_FPS}


def build(name: str, directory: Path) -> tuple[Path, list[float]]:
    """Render and encode one fixture; returns its path and the exact times of its cuts."""
    frames, cut_frames = FIXTURES[name]()
    rate = FIXTURE_RATES.get(name, FIXTURE_FPS)
    path = encode(frames, directory / f"{name}.mp4", rate)
    return path, [n / rate for n in cut_frames]


# ---------------------------------------------------------------- tests
CONTINUOUS = [
    "A_slow_pan",
    "B_whip_pan",
    "B2_whip_beyond_search",
    "C_foreground",
    "G_grain",
    "H_flash",
    "H_flash_saturating",
    "L_occlusion_returns",
]
CUTS = ["D_cut_places", "E_cut_same_place", "F_cut_in_motion", "I_duplicated_frames", "J_dark_cut"]
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
                off = abs(cuts[0]["time"] - expected[0]) * FIXTURE_RATES.get(name, FIXTURE_FPS)
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


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class PairFetchTests(unittest.TestCase):
    """The frame pair a cut is judged on has to be that pair, and nothing else.

    Fetching it by TIME cannot do that: a printed timestamp does not name a frame boundary exactly,
    and one frame of slip puts a frame of the outgoing shot on the incoming side, where it matches
    the outgoing shot perfectly and clears the cut. These tests hold the fetch to the frame index.
    """

    def small(self, frame):
        import cv2

        return cv2.resize(
            frame, (shot_cuts.THUMB_WIDTH, shot_cuts.THUMB_HEIGHT), interpolation=cv2.INTER_AREA
        )

    def test_a_fetched_frame_is_the_frame_that_was_asked_for(self):
        video, _ = Clips.get("D_cut_places")
        times, scores, thumbs, stamps = shot_cuts.scene_pass(video)
        wanted = {10, CUT_FRAME - 1, CUT_FRAME, CUT_FRAME + 1, 70}
        got = shot_cuts.frames_at(video, wanted, stamps)
        self.assertEqual(set(got), wanted)
        for i in sorted(wanted):
            here = shot_cuts.carry(thumbs[i], self.small(got[i]))
            self.assertGreater(here, 0.97, f"frame {i} is not the frame the scoring pass saw")
            for neighbour in (i - 1, i + 1):
                if 0 <= neighbour < len(thumbs):
                    self.assertGreater(
                        here,
                        shot_cuts.carry(thumbs[neighbour], self.small(got[i])),
                        f"frame {i} matches its neighbour {neighbour} better than itself",
                    )

    def test_the_pair_across_a_cut_is_one_frame_of_each_shot(self):
        video, _ = Clips.get("D_cut_places")
        times, scores, thumbs, stamps = shot_cuts.scene_pass(video)
        got = shot_cuts.frames_at(video, {CUT_FRAME - 1, CUT_FRAME}, stamps)
        across = shot_cuts.orb_fraction(got[CUT_FRAME - 1], got[CUT_FRAME])
        self.assertIsNotNone(across)
        self.assertLess(across, shot_cuts.CUT_MATCH_MAX, "the pair straddling the cut matched")
        inside = shot_cuts.frames_at(video, {CUT_FRAME - 2, CUT_FRAME - 1}, stamps)
        within = shot_cuts.orb_fraction(inside[CUT_FRAME - 2], inside[CUT_FRAME - 1])
        self.assertGreater(within, shot_cuts.CUT_MATCH_MAX, "a pair inside one shot did not match")

    def test_a_duplicated_frame_is_not_its_own_predecessor(self):
        video, _ = Clips.get("I_duplicated_frames")
        times, scores, thumbs, stamps = shot_cuts.scene_pass(video)
        # The incoming shot's first content frame is container frames n and n + 1; the outgoing
        # shot's last is n - 2 and n - 1. The frame before the second copy is the FIRST copy, which
        # is the same picture, so the pair has to reach back past it.
        n = DOUBLED_CUT_FRAME
        self.assertEqual(shot_cuts.previous_distinct(thumbs, n + 1), n - 1)
        self.assertEqual(shot_cuts.previous_distinct(thumbs, n), n - 1)
        self.assertEqual(shot_cuts.previous_distinct(thumbs, n - 1), n - 3)

    def test_a_failed_pair_fetch_says_so_instead_of_blaming_the_footage(self):
        """A second decode that does not come back must not read as "too blurred to judge"."""
        video, _ = Clips.get("D_cut_places")

        def refuse(*args, **kwargs):
            raise RuntimeError("frame 44 came back with pts 99, not the 45 the scoring pass saw")

        original = shot_cuts.frames_at
        shot_cuts.frames_at = refuse
        try:
            judged = [c for c in shot_cuts.detect_cuts(video) if "matchedAgainstFrame" in c]
        finally:
            shot_cuts.frames_at = original
        self.assertTrue(judged, "no candidate reached the ORB stage, so nothing was tested")
        for c in judged:
            self.assertIn("came back with pts", c["matchFailed"])
            self.assertIn("could not be fetched", c["why"])
            self.assertNotIn("too blurred", c["why"])
            self.assertIsNone(c["match"])

    def test_the_duplicated_container_cut_is_judged_on_the_right_pair(self):
        video, expected = Clips.get("I_duplicated_frames")
        cuts = [c for c in shot_cuts.detect_cuts(video) if c["cut"]]
        self.assertEqual(len(cuts), 1, f"{[c['time'] for c in cuts]}")
        self.assertLess(cuts[0]["matchedAgainstFrame"], cuts[0]["frame"])
        self.assertIsNotNone(cuts[0]["match"])
        self.assertLess(cuts[0]["match"], shot_cuts.CUT_MATCH_MAX)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class DarkCutTests(unittest.TestCase):
    """A cut the scene score cannot see at all must still be found."""

    def test_the_scene_score_really_is_under_the_gate(self):
        video, expected = Clips.get("J_dark_cut")
        times, scores, thumbs, stamps = shot_cuts.scene_pass(video)
        self.assertLess(
            max(scores),
            shot_cuts.SCENE_CANDIDATE,
            "the dark fixture is not dark enough to test the second gate",
        )

    def test_the_carry_gate_finds_it_anyway(self):
        video, expected = Clips.get("J_dark_cut")
        cuts = [c for c in shot_cuts.detect_cuts(video) if c["cut"]]
        self.assertEqual(len(cuts), 1, f"{[c['time'] for c in cuts]}")
        self.assertAlmostEqual(cuts[0]["time"], expected[0], delta=1.5 / FIXTURE_FPS)
        self.assertLess(cuts[0]["carry"], shot_cuts.CARRY_CANDIDATE)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class FadeTests(unittest.TestCase):
    """A fade is a region, and neither shot may contain it."""

    def test_the_fade_is_found_as_a_region(self):
        video, expected = Clips.get("K_fade_to_black")
        report = shot_cuts.cut_report(video, score=False)
        self.assertEqual(len(report["fades"]), 1, report["fades"])
        fade = report["fades"][0]
        self.assertEqual(fade["through"], "black")
        self.assertGreaterEqual(fade["frames"], 2 * FADE_FRAMES - 2)
        cuts = report["cuts"]
        self.assertEqual([c["kind"] for c in cuts], ["fade"], [c["why"] for c in cuts])
        self.assertAlmostEqual(cuts[0]["fadeStart"], fade["start"], places=3)
        self.assertAlmostEqual(cuts[0]["time"], fade["end"], places=3)

    def test_neither_shot_holds_the_faded_frames(self):
        video, expected = Clips.get("K_fade_to_black")
        report = shot_cuts.cut_report(video, score=False)
        fade = report["fades"][0]
        self.assertEqual(report["shotCount"], 2, report["shots"])
        self.assertLessEqual(report["shots"][0]["end"], fade["start"] + 1e-6)
        self.assertGreaterEqual(report["shots"][1]["start"], fade["end"] - 1e-6)
        self.assertGreater(fade["end"] - fade["start"], 0.5 * FADE_FRAMES / FIXTURE_FPS)

    def test_a_hard_cut_to_a_black_card_is_not_a_fade(self):
        """One frame of black is an edit; FADE_MIN_FRAMES of ramp is a fade."""
        video, _ = Clips.get("D_cut_places")
        self.assertEqual(shot_cuts.cut_report(video, score=False)["fades"], [])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are needed to render the fixtures")
class OcclusionTests(unittest.TestCase):
    """Something crossing the lens for two frames is not an edit, because the view comes back."""

    def test_the_occlusion_is_a_candidate_and_is_cleared(self):
        video, _ = Clips.get("L_occlusion_returns")
        found = shot_cuts.detect_cuts(video)
        near = [c for c in found if abs(c["frame"] - CUT_FRAME) <= 3]
        self.assertTrue(near, "the occlusion did not even become a candidate")
        self.assertEqual([c for c in found if c["cut"]], [])
        self.assertTrue(
            any(c["returns"] is not None and c["returns"] >= shot_cuts.CARRY_RETURNS for c in near),
            f"cleared for the wrong reason: {[(c['frame'], c['why']) for c in near]}",
        )


class EvaluationTests(unittest.TestCase):
    """The scoring harness itself, on written-out reports and labels. No video, no ffmpeg."""

    def truth(self, **over):
        base = dict(
            video="x.mp4",
            fps=25.0,
            cuts=[
                dict(time=1.00, kind="hard", confidence="high"),
                dict(time=2.00, kind="hard", confidence="high"),
                dict(time=3.00, kind="hard", confidence="medium"),
                dict(time=8.00, kind="fade-out-to-black", confidence="high"),
            ],
            reviewedNonCuts=[dict(time=5.0)],
            unclear=[dict(time=6.0)],
        )
        base.update(over)
        return base

    def report(self, times, fades=()):
        cuts = []
        for t in times:
            cuts.append(dict(time=t, frame=int(t * 25), kind="hard", carry=0.2, score=0.3))
        for start, end in fades:
            cuts.append(
                dict(
                    time=end,
                    frame=int(end * 25),
                    kind="fade",
                    fadeStart=start,
                    carry=0.2,
                    score=0.01,
                )
            )
        return dict(video="x.mp4", cuts=sorted(cuts, key=lambda c: c["time"]))

    def test_a_hit_inside_the_tolerance_and_a_miss_outside_it(self):
        got = shot_cuts.evaluate_truth(self.report([1.04, 2.20]), self.truth())
        self.assertEqual(got["hits"], 1)  # 1.04 is one frame off; 2.20 is five
        self.assertEqual([m["time"] for m in got["missed"]], [2.00])
        self.assertEqual([f["time"] for f in got["falsePositives"]], [2.20])
        self.assertEqual(got["recall"], 0.5)

    def test_a_medium_label_is_neither_a_hit_nor_a_false_positive(self):
        got = shot_cuts.evaluate_truth(self.report([1.0, 2.0, 3.0]), self.truth())
        self.assertEqual((got["hits"], got["required"], got["optional"]), (2, 2, 1))
        self.assertEqual(got["falsePositives"], [])
        self.assertEqual((got["precision"], got["recall"]), (1.0, 1.0))

    def test_a_fade_matches_a_label_anywhere_inside_its_region(self):
        got = shot_cuts.evaluate_truth(self.report([1.0, 2.0], fades=[(7.5, 8.6)]), self.truth())
        self.assertEqual(got["optional"], 1)
        self.assertEqual(got["falsePositives"], [])
        self.assertEqual(got["required"], 2, "a fade label is not required of the cut detector")

    def test_windows_score_only_what_is_inside_them(self):
        got = shot_cuts.evaluate_truth(
            self.report([1.0, 2.0, 9.9]), self.truth(), windows=[[0.5, 2.5]]
        )
        self.assertEqual((got["detections"], got["required"], got["hits"]), (2, 2, 2))
        self.assertEqual(got["falsePositives"], [])

    def test_verified_only_ignores_detections_nobody_looked_at(self):
        strict = shot_cuts.evaluate_truth(self.report([1.0, 4.4]), self.truth())
        self.assertEqual([f["time"] for f in strict["falsePositives"]], [4.4])
        loose = shot_cuts.evaluate_truth(self.report([1.0, 4.4]), self.truth(), verified_only=True)
        self.assertEqual(loose["falsePositives"], [])
        self.assertEqual(loose["detections"], 1)

    def test_a_reviewed_non_cut_is_a_false_positive(self):
        got = shot_cuts.evaluate_truth(self.report([1.0, 2.0, 5.0]), self.truth())
        self.assertEqual([f["time"] for f in got["falsePositives"]], [5.0])
        self.assertEqual(got["precision"], round(2 / 3, 3))


class DuplicateFrameTests(unittest.TestCase):
    """previous_distinct, on made-up thumbnails. No video, no ffmpeg.

    The ORB stage is only as good as the frame it compares against, and in a container running at
    twice its content's rate the frame before a candidate is the candidate again.
    """

    def series(self, seeds: list[int]):
        import numpy as np

        rng = np.random.default_rng(0)
        pictures = {
            s: rng.integers(0, 255, (shot_cuts.THUMB_HEIGHT, shot_cuts.THUMB_WIDTH), dtype="uint8")
            for s in set(seeds)
        }
        return np.stack([pictures[s] for s in seeds])

    def test_the_previous_frame_is_the_previous_frame_when_none_repeat(self):
        thumbs = self.series([1, 2, 3, 4, 5])
        self.assertEqual(shot_cuts.previous_distinct(thumbs, 4), 3)

    def test_a_doubled_container_reaches_back_past_the_duplicate(self):
        """[A, A, B, B, C, C]: the frame before each second copy is its own first copy."""
        thumbs = self.series([1, 1, 2, 2, 3, 3])
        self.assertEqual(shot_cuts.previous_distinct(thumbs, 5), 3)
        self.assertEqual(shot_cuts.previous_distinct(thumbs, 4), 3)
        self.assertEqual(shot_cuts.previous_distinct(thumbs, 3), 1)
        self.assertEqual(shot_cuts.previous_distinct(thumbs, 2), 1)

    def test_a_frozen_run_reaches_back_to_the_last_moving_frame(self):
        thumbs = self.series([1, 2, 3, 3, 3, 3])
        self.assertEqual(shot_cuts.previous_distinct(thumbs, 5), 1)


class SuppressNeighbourTests(unittest.TestCase):
    """One camera move accuses several frames; one edit is one of them. No video, no ffmpeg."""

    def record(self, time: float, carry: float) -> dict:
        return dict(time=time, frame=int(time * 30), carry=carry, cut=True, why="a cut")

    def test_a_run_inside_the_gap_collapses_to_its_least_carrying_frame(self):
        run = [self.record(10.0, 0.40), self.record(10.03, 0.12), self.record(10.06, 0.33)]
        shot_cuts.suppress_neighbours(run)
        self.assertEqual([r["cut"] for r in run], [False, True, False])
        self.assertIn("10.03", run[0]["why"])

    def test_cuts_further_apart_than_the_gap_all_stand(self):
        run = [self.record(10.0, 0.40), self.record(10.0 + 2 * shot_cuts.MIN_CUT_SECONDS, 0.12)]
        shot_cuts.suppress_neighbours(run)
        self.assertEqual([r["cut"] for r in run], [True, True])

    def test_records_are_taken_in_time_order_whatever_order_they_arrive_in(self):
        run = [self.record(10.06, 0.33), self.record(10.0, 0.40), self.record(10.03, 0.12)]
        shot_cuts.suppress_neighbours(run)
        self.assertEqual([r["cut"] for r in run], [False, False, True])


class ShotsFromCutsTests(unittest.TestCase):
    """The segments between the cuts, on written-out cuts. No video, no ffmpeg."""

    def fade(self, start: float, end: float) -> dict:
        return dict(time=end, frame=int(end * 30), kind="fade", fadeStart=start, carry=0.1)

    def hard(self, time: float) -> dict:
        return dict(time=time, frame=int(time * 30), kind="hard", carry=0.1)

    def test_a_hard_cut_starts_the_next_shot_on_its_own_frame(self):
        shots = shot_cuts.shots_from_cuts([self.hard(4.0)], 10.0, 1.5)
        self.assertEqual([(s["start"], s["end"]) for s in shots], [(0.0, 4.0), (4.0, 10.0)])

    def test_neither_shot_either_side_of_a_fade_holds_its_frames(self):
        shots = shot_cuts.shots_from_cuts([self.fade(4.0, 5.0)], 10.0, 1.5)
        self.assertEqual([(s["start"], s["end"]) for s in shots], [(0.0, 4.0), (5.0, 10.0)])

    def test_a_fade_that_opens_the_clip_leaves_no_empty_shot_in_front_of_it(self):
        shots = shot_cuts.shots_from_cuts([self.fade(0.0, 0.7)], 10.0, 1.5)
        self.assertEqual([(s["start"], s["end"]) for s in shots], [(0.7, 10.0)])
        self.assertEqual([s["index"] for s in shots], [0])

    def test_a_fade_that_closes_the_clip_leaves_no_empty_shot_behind_it(self):
        shots = shot_cuts.shots_from_cuts([self.fade(9.2, 10.0)], 10.0, 1.5)
        self.assertEqual([(s["start"], s["end"]) for s in shots], [(0.0, 9.2)])

    def test_a_frozen_card_is_flagged_static_and_moving_footage_is_not(self):
        times = [i / 30 for i in range(300)]
        changes = [0.0] + [3.0 if i < 150 else 0.001 for i in range(1, 300)]
        shots = shot_cuts.shots_from_cuts([self.hard(5.0)], 10.0, 1.5, times, changes)
        self.assertEqual([s.get("static") for s in shots], [False, True])
        self.assertLess(shots[1]["change"], shot_cuts.STATIC_CHANGE)
        self.assertGreater(shots[0]["change"], shot_cuts.STATIC_CHANGE)

    def test_a_shot_is_not_flagged_static_without_the_series_to_say_so(self):
        shots = shot_cuts.shots_from_cuts([self.hard(5.0)], 10.0, 1.5)
        self.assertEqual([("change" in s or "static" in s) for s in shots], [False, False])


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
