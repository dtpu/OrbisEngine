#!/usr/bin/env python3
"""Package an lhm_animate.py motion run as public/worlds/<clip>-4d/person/ in the corridor layout.

  uv run --locked --group inference python scripts/package_person_sequence.py .context/<clip>/lhm-motion \
      --cameras .context/<clip>/pi3x/cameras.json --clip public/clips/<clip>.mp4 \
      --out public/worlds/<clip>-4d

The motion run's Gaussians live in the Pi3X OpenGL world (y up, -z forward), where camera 0 is
close to but not exactly the identity. Every frame and camera is re-expressed with the inverse
of camera 0's rigid pose so the clip's first camera sits at the origin looking down -z, the same
convention as overnight-corridor-video-projection/person. Scale is untouched (body scale stays
the run's first-frame body-depth registration in SfM units).

Writes person/frame_NNN.ply (same 17 float fields), person/sequence.json, cameras.json.
"""

import argparse, hashlib, json, shutil
from pathlib import Path

import numpy as np

import package_person_motion
from sfm_frame import camera0_reframe, describe as describe_frame
from plyfile import PlyData, PlyElement


def orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        raise ValueError("reflected camera")
    return R


def quat_from_matrix(R):
    """(w, x, y, z) from a rotation matrix, matching pytorch3d.matrix_to_quaternion."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        w = s / 4
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = s / 4
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = s / 4
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = s / 4
    return np.array([w, x, y, z])


def quat_mul(a, b):
    """(w,x,y,z) Hamilton product; a is (4,), b is (N,4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b.T
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        1,
    )


def transform_ply(src, dst, R, t):
    ply = PlyData.read(src)
    v = ply["vertex"].data.copy()
    xyz = np.column_stack([v["x"], v["y"], v["z"]]) @ R.T + t
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    q = np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]])
    q = quat_mul(quat_from_matrix(R), q)
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = q.T.astype(np.float32)
    PlyData([PlyElement.describe(v, "vertex")], text=False, byte_order="<").write(dst)
    finite = all(np.isfinite(v[n]).all() for n in v.dtype.names)
    return len(v), finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--lower-body",
        default="Source-observed: legs and feet are in frame for the whole shot; independent MultiHMR pose per sample, no inferred gait",
        help="lowerBodyProvenance text written to sequence.json (say so when legs are hidden or cut)",
    )
    a = ap.parse_args()
    motion = Path(a.motion)
    out = Path(a.out)
    person = out / "person"
    person.mkdir(parents=True, exist_ok=True)
    seq = json.loads((motion / "sequence.json").read_text())
    cams = json.loads(Path(a.cameras).read_text())
    # world -> the packaged frame: camera 0 at the origin, gravity on +y when the run's framealign.json
    # exists beside cameras.json, camera 0's own axes otherwise (scripts/sfm_frame.py)
    R, t, fa = camera0_reframe(a.cameras)
    frame_desc = describe_frame(fa)
    print(f"frame: {frame_desc}", flush=True)

    frames, hashes, counts, finite_all = [], [], set(), True
    for name in seq["frames"]:
        n, finite = transform_ply(motion / name, person / name, R, t)
        counts.add(n)
        finite_all &= finite
        hashes.append(hashlib.sha256((person / name).read_bytes()).hexdigest())
        frames.append(name)
    for extra in ("motion.json", "registration.json", "missing-poses.json"):
        if (motion / extra).exists():
            shutil.copy(motion / extra, person / extra)

    out_cams = []
    for c in cams["cameras"]:
        m = np.array(c["camera_to_world"])
        m[:3, :3] = orthonormal(m[:3, :3])
        m2 = np.eye(4)
        m2[:3, :3] = R @ m[:3, :3]
        m2[:3, 3] = R @ m[:3, 3] + t
        out_cams.append(
            dict(
                sourceIndex=c["sourceIndex"],
                time=c["time"],
                camera_to_world=m2.tolist(),
                source_intrinsics=c["source_intrinsics"],
                source_image_size=c.get("source_image_size"),
            )
        )
    (out / "cameras.json").write_text(
        json.dumps(
            dict(
                coordinates=f"OpenGL, {frame_desc}. Same frame as person/frame_*.ply. Pi3X SfM of the clip, units = Pi3X normalized (not metric).",
                frameAlign=fa,
                source=str(a.cameras),
                sourceClip=str(a.clip),
                cameras=out_cams,
            ),
            indent=1,
        )
    )

    seq_out = dict(seq)
    seq_out.update(
        frames=frames,
        count=len(frames),
        frame_sha256=hashes,
        gaussiansPerFrame=sorted(counts)[0] if len(counts) == 1 else sorted(counts),
        allNativeGaussianAttributesFinite=finite_all,
        people_only=True,
        coordinates=f"OpenGL Pi3X-SfM world re-expressed so {frame_desc}; body scale = first-frame body-depth registration (registration.json)",
        camera0Reframe=dict(
            rotation=R.tolist(),
            translation=t.tolist(),
            source="cameras.json camera 0 of the Pi3X export",
        ),
        lowerBodyProvenance=a.lower_body,
        sourceClip=str(a.clip),
        sourceSha256=hashlib.sha256(Path(a.clip).read_bytes()).hexdigest(),
    )
    (person / "sequence.json").write_text(json.dumps(seq_out, indent=2))
    motion_track = add_motion_track(person)
    print(
        json.dumps(
            dict(
                frames=len(frames),
                gaussians=sorted(counts),
                finite=finite_all,
                motionTrack=motion_track,
                out=str(out),
            ),
            indent=1,
        )
    )


def add_motion_track(person: Path):
    """The compact per-frame track the viewer prefers over one PLY per frame; the PLYs stay.

    A sequence whose colour, opacity or scale change between frames is refused by the packer and
    ships without a track, which the viewer handles by reading the PLYs as before.
    """
    try:
        record = package_person_motion.package(person)
    except ValueError as error:
        print(f"motion track skipped: {error}", flush=True)
        return None
    return dict(file=record["file"], bytes=record["bytes"], plyBytes=record["plyBytes"])


if __name__ == "__main__":
    main()
