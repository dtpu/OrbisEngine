#!/usr/bin/env python3
"""The one frame every packaged clip lives in, and the one place it is defined.

The packagers re-express a Pi3X reconstruction so that source camera 0 sits at the origin. Until
now its ORIENTATION was the phone's body frame at frame 0 ("camera 0 = identity"), so the y axis of
person PLYs, cameras.json and objects.json was whichever way up pointed out of the handset. Marble
builds its worlds gravity-levelled, and fourd.html models SfM -> Marble as scale + translation, so
the phone's own pitch at frame 0 (15 deg on hpwide, 19 deg on the gym) went straight into every
placement as a translation that nearly works.

With a `framealign.json` beside the Pi3X cameras.json (written by `frame_align.py graph`, a stage
of run_clip.py) the reframe keeps camera 0 at the origin and puts GRAVITY on +y instead: camera 0
comes out pitched by exactly what the phone was pitched, which is how Marble already has it.
Without the file the old convention is returned unchanged, so every historical package is
byte-identical. Every script that re-anchors the raw frame on camera 0 -- the packagers,
bake_video_colours, finetune_export_bedroom -- goes through here, so there is one convention.

numpy only: this is imported by the packagers and by bake_video_colours, which frame_align.py
itself imports.
"""
import json
from pathlib import Path

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def orthonormal(R):
    u, _, vt = np.linalg.svd(np.asarray(R, float))
    return u @ vt


def align_rotation(g):
    """Minimal rotation taking g to +y (no yaw about gravity)."""
    g = unit(g)
    v = np.cross(g, UP); s = np.linalg.norm(v); c = float(g @ UP)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0, 0, 0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return [float(x) for x in q]


def framealign_path(cameras_path) -> Path:
    return Path(cameras_path).with_name("framealign.json")


def read_framealign(cameras_path):
    p = framealign_path(cameras_path)
    return json.loads(p.read_text()) if p.exists() else None


def camera0_reframe(cameras_path, origin_frame: int = 0):
    """(R, t, framealign): raw Pi3X frame -> packaged frame, p' = R p + t.

    Camera `origin_frame` goes to the origin either way. With a framealign.json its gravity goes to
    +y (camera 0 keeps the phone's real pitch and roll); without one camera 0's own axes do, which
    is the historical 'camera 0 = identity' convention.
    """
    cams = json.loads(Path(cameras_path).read_text())["cameras"]
    c0 = np.array(cams[origin_frame]["camera_to_world"], float)
    R0 = orthonormal(c0[:3, :3]); t0 = c0[:3, 3]
    R, t = R0.T, -R0.T @ t0
    fa = read_framealign(cameras_path)
    if fa is not None:
        Rg = align_rotation(R0.T @ np.asarray(fa["gravityRaw"], float))   # gravity as camera 0 sees it
        R, t = Rg @ R, Rg @ t
    return R, t, fa


def origin_c2w(cameras_path, origin_frame: int = 0):
    """The 4x4 that bake_video_colours / finetune_export_bedroom call `origin_c2w`: the inverse of
    the reframe, i.e. the camera-to-world of the levelled origin. Equals cameras[origin_frame]
    exactly when there is no framealign.json."""
    R, t, fa = camera0_reframe(cameras_path, origin_frame)
    M = np.eye(4)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ t
    return M, fa


def describe(fa) -> str:
    if fa is None:
        return "camera 0 = identity (origin, y up, looking -z)"
    return (f"camera 0 at the origin, gravity on +y (camera 0 pitched {fa['tiltFromYDeg']:.1f} deg, "
            f"frame_align.py {fa['source']})")
