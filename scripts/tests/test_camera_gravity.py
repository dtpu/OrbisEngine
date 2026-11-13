"""Camera height in body-heights from footage geometry, and the size check that uses it."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import anchor_checks  # noqa: E402
import camera_gravity as cg  # noqa: E402
import place_solve  # noqa: E402

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = (640, 480)


def pitched_camera(height, pitch_deg, z=0.0):
    """OpenGL camera_to_world at (0, height, z), looking down -z, pitched down by pitch_deg."""
    a = np.radians(-pitch_deg)
    M = np.eye(4)
    M[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    M[:3, 3] = [0.0, height, z]
    return M


def project(M, points):
    """Image (u, v) of world points through an OpenGL camera with intrinsics K."""
    local = (np.asarray(points, float) - M[:3, 3]) @ M[:3, :3]
    depth = -local[:, 2]
    return np.c_[K[0, 0] * local[:, 0] / depth + K[0, 2], K[1, 1] * -local[:, 1] / depth + K[1, 2]]


def walker_row(M, source, distance, stature=1.0):
    foot, head = project(M, [[0, 0, M[2, 3] - distance], [0, stature, M[2, 3] - distance]])
    return dict(
        sourceIndex=source,
        top=float(head[1]),
        bottom=float(foot[1]),
        left=float(foot[0] - 12),
        right=float(foot[0] + 12),
    )


def frame_with_poles(M):
    """A grey frame with world-vertical poles drawn through camera M."""
    import cv2

    gray = np.full((SIZE[1], SIZE[0]), 90, np.uint8)
    z0 = M[2, 3]
    for x, dz in [(-3.0, 6), (-2.2, 9), (-4.0, 12), (2.6, 7), (3.4, 10), (2.0, 14), (-1.6, 16)]:
        a, b = project(M, [[x, 0.0, z0 - dz], [x, 4.0, z0 - dz]])
        cv2.line(
            gray,
            tuple(np.round(a).astype(int)),
            tuple(np.round(b).astype(int)),
            230,
            4,
            cv2.LINE_AA,
        )
    return gray


def test_body_heights_is_the_camera_height_over_the_walker():
    for height, pitch, distance in [(2.2, 6.0, 8.0), (1.45, 2.0, 5.0), (0.9, -3.0, 6.0)]:
        M = pitched_camera(height, pitch)
        row = walker_row(M, 0, distance, stature=1.7)
        down = cg.camera_down(M, [0, -1, 0])
        value = cg.body_heights(
            K, down, row["top"], row["bottom"], (row["left"] + row["right"]) / 2
        )
        assert value == pytest.approx(height / 1.7, rel=1e-6)


def test_tilt_follows_the_camera_and_unmeasurable_views_refuse():
    pitch, roll = cg.tilt_degrees(cg.camera_down(pitched_camera(2, 5.0), [0, -1, 0]))
    assert pitch == pytest.approx(5.0) and roll == pytest.approx(0.0, abs=1e-9)
    down = cg.camera_down(pitched_camera(2, 0.0), [0, -1, 0])
    # Feet above the horizon: the camera is below the floor the formula assumes.
    assert cg.body_heights(K, down, 100.0, 200.0, 320.0) is None


def test_vertical_lines_recover_gravity_from_a_pitched_camera():
    M = pitched_camera(2.2, 6.0)
    segments = cg.vertical_segments(frame_with_poles(M))
    planes = cg.segment_planes(segments, K)
    found = cg.common_direction(planes, segments[:, 4], [0, 1, 0])
    assert found is not None
    pitch, roll = cg.tilt_degrees(found[0])
    assert pitch == pytest.approx(6.0, abs=0.3) and roll == pytest.approx(0.0, abs=0.3)


def scene(height=2.2, pitch=6.0, frames=6):
    cameras, rows, gray = [], [], {}
    for i in range(frames):
        M = pitched_camera(height, pitch, z=-0.4 * i)
        cameras.append(
            dict(
                sourceIndex=i,
                camera_to_world=M.tolist(),
                source_intrinsics=K.tolist(),
                source_image_size=list(SIZE),
            )
        )
        rows.append(walker_row(M, i, 8.0, stature=1.7))
        gray[i] = frame_with_poles(M)
    return cameras, rows, gray


def test_footage_measures_an_elevated_camera_when_both_tilts_agree():
    cameras, rows, gray = scene()
    geometry = cg.measure(rows, cameras, [0, 1, 0], gray)
    assert geometry["registration"]["bodyHeightsMedian"] == pytest.approx(2.2 / 1.7, rel=1e-6)
    assert geometry["image"]["bodyHeightsMedian"] == pytest.approx(2.2 / 1.7, rel=0.02)
    assert geometry["image"]["framesSupporting"] == 6
    assert geometry["image"]["tiltFromFloorNormalDeg"] < 0.5
    verdict = cg.footage_verdict(geometry, 2.2 / 1.7)
    assert verdict == dict(verified=True, fittedBodyHeights=2.2 / 1.7, why=[])


def test_a_tilted_registration_is_caught_by_the_frames_own_vertical_lines():
    cameras, rows, gray = scene()
    tilt = np.radians(8.0)
    normal = [0.0, np.cos(tilt), np.sin(tilt)]  # the floor fit disagrees with gravity by 8 degrees
    geometry = cg.measure(rows, cameras, normal, gray)
    assert geometry["image"]["tiltFromFloorNormalDeg"] == pytest.approx(8.0, abs=0.5)
    registration = geometry["registration"]["bodyHeightsMedian"]
    verdict = cg.footage_verdict(geometry, registration)
    assert not verdict["verified"]
    assert "vertical lines" in verdict["why"][0] and "apart" in verdict["why"][0]


def test_too_few_vertical_lines_conclude_nothing():
    cameras, rows, gray = scene(frames=3)
    geometry = cg.measure(rows, cameras, [0, 1, 0], gray)
    verdict = cg.footage_verdict(geometry, geometry["registration"]["bodyHeightsMedian"])
    assert not verdict["verified"] and "need 30 from 5" in verdict["why"][0]
    blank = {i: np.full((SIZE[1], SIZE[0]), 90, np.uint8) for i in gray}
    geometry = cg.measure(rows, cameras, [0, 1, 0], blank)
    verdict = cg.footage_verdict(geometry, geometry["registration"]["bodyHeightsMedian"])
    assert not verdict["verified"] and "no common vertical direction" in verdict["why"][0]


def geometry(registration, image, *, inliers=80, frames=12):
    return dict(
        registration=dict(bodyHeightsMedian=registration),
        image=dict(
            bodyHeightsMedian=image,
            inliers=inliers,
            framesSupporting=frames,
            framesSampled=12,
            segments=240,
        ),
        tolerance=cg.TOLERANCE,
        minFrames=cg.MIN_FRAMES,
        minInliers=cg.MIN_INLIERS,
    )


def size_check(camera_m, footage=None, slope=0.0):
    sc = dict(
        statureUnits=0.5,
        metresPerWorldUnit=3.4,
        cameraHeightM=camera_m,
        floorToCeilingM=3.0,
        cameraHeightTrace=dict(
            metres=[camera_m] * 4,
            medianM=camera_m,
            p10M=camera_m,
            p90M=camera_m,
            endToEndSlopeM=slope,
            note="",
        ),
    )
    if footage is not None:
        sc["footageGeometry"] = footage
    return place_solve.judge_size(sc)


def test_the_handheld_band_still_decides_without_footage():
    inside = size_check(1.5)
    assert inside["pass_"] and inside["basis"] == "handheld band"
    assert inside["cameraHeightTrace"]["verdict"] == "flat in band"
    high = size_check(2.21)
    assert not high["pass_"] and high["cameraHeightTrace"]["verdict"] == "flat out of band"
    with pytest.raises(SystemExit, match=r"expected 1\.15–1\.85\)\. Inspect"):
        place_solve.require_valid_size(high, True)
    assert size_check(1.5, slope=0.3)["cameraHeightTrace"]["verdict"] == "drift"


def test_a_measured_elevated_camera_passes_with_its_basis_recorded():
    sc = size_check(2.21, geometry(1.303, 1.278))
    assert sc["pass_"] and sc["basis"] == "measured from footage"
    assert sc["cameraHeightTrace"]["verdict"] == "flat, measured from footage"
    assert sc["footageGeometry"]["verdict"]["fittedBodyHeights"] == pytest.approx(2.21 / 1.70)
    place_solve.require_valid_size(sc, True)
    trace = anchor_checks.check_camera_trace({"surface": {"sizeCheck": sc}})
    assert trace["ok"] and trace["basis"] == "measured from footage"


@pytest.mark.parametrize(
    "footage,reason",
    [
        (geometry(1.303, 1.05), "vertical lines"),
        (geometry(1.303, 1.29, inliers=12, frames=3), "need 30 from 5"),
        (geometry(1.05, 1.05), "fitted placement"),
    ],
)
def test_footage_that_does_not_measure_the_height_keeps_the_failure(footage, reason):
    sc = size_check(2.21, footage)
    assert not sc["pass_"] and sc["basis"] == "handheld band"
    assert sc["cameraHeightTrace"]["verdict"] == "flat out of band"
    with pytest.raises(SystemExit, match="did not establish that height either") as raised:
        place_solve.require_valid_size(sc, True)
    assert reason in str(raised.value)
    trace = anchor_checks.check_camera_trace({"surface": {"sizeCheck": sc}})
    assert not trace["ok"] and "does not measure the camera there" in trace["why"][0]


def test_sampling_spreads_frames_across_the_clip():
    rows = [dict(sourceIndex=i) for i in range(0, 180, 2)]
    picks = cg.sample_sources(rows, limit=12)
    assert len(picks) == 12 and picks[0] == 0 and picks[-1] == 178
    assert cg.sample_sources(rows[:4]) == [0, 2, 4, 6]
