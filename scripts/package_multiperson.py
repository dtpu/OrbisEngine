#!/usr/bin/env python3
"""Package N per-track lhm_animate runs as ONE public/worlds/<clip>-4d/ with a people manifest.

  uv run --locked --group inference python scripts/package_multiperson.py \
      --motion 0=.context/mp/<clip>/lhm-motion-00 --motion 1=.context/mp/<clip>/lhm-motion-01 \
      --tracks .context/mp/<clip>/tracks --cameras .context/mp/<clip>/pi3x/cameras.json \
      --clip public/clips/<clip>.mp4 --out public/worlds/<clip>-4d

Layout, chosen so a single world directory holds everybody and today's presets keep working:

  <world>/cameras.json      camera-0 frame, written once and shared by every person
  <world>/person/           the PRIMARY track, in exactly the layout package_person_sequence.py
                            writes today, so ?person=/worlds/<clip>-4d/person/sequence.json is
                            unchanged for existing single-person presets
  <world>/person_01/ ...    every further track, same layout
  <world>/people.json       the manifest: who is in this world, when each is visible, and each
                            one's own transform

Every track is lifted into the world by the SAME Pi3X cameras, so the per-person transforms are
identity by construction and one global placement (floor/height/scale) covers everybody. The
transform block exists anyway because it is the only place a future per-person world solve (the
GVHMR route, which fits each subject its own trajectory) could put its result, and because a
person can then be nudged without repackaging.
"""

import argparse, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from package_person_sequence import (  # noqa: E402
    add_motion_track,
    orthonormal,
    quat_from_matrix,
    quat_mul,
    transform_ply,
)
from sfm_frame import camera0_reframe, describe as describe_frame  # noqa: E402


def body_stats(ply):
    from plyfile import PlyData

    v = PlyData.read(str(ply))["vertex"].data
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    y = v["y"][op > 0.5]
    lo, hi = float(np.percentile(y, 1)), float(np.percentile(y, 99))
    return hi - lo, lo


def measure_body(plys, R, t):
    """Per-frame (height, feetY) of a motion run as it will be after the camera-0 reframe.

    One frame is not enough: the vertical extent of a walking body changes with the stride, and
    frame 0 can catch one person mid-reach and the other standing, which is a ~20 % difference.
    The median over the clip is the standing height.
    """
    from plyfile import PlyData

    heights, feet = [], []
    for ply in plys:
        v = PlyData.read(str(ply))["vertex"].data
        op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
        y = (np.column_stack([v["x"], v["y"], v["z"]])[op > 0.5] @ R.T + t)[:, 1]
        lo, hi = float(np.percentile(y, 1)), float(np.percentile(y, 99))
        heights.append(hi - lo)
        feet.append(lo)
    return np.array(heights), np.array(feet)


def transform_ply_sized(src, dst, R, t, scale, centre):
    """Rigid reframe plus a uniform `scale` about `centre` (this frame's camera in the output frame).

    Scaling about the source camera is exactly what the LHM registration did, so redoing it about
    the same point changes the body's world size and depth while leaving its projection into that
    frame bit-for-bit where it was: the reprojection check stays valid and the avatar stops being
    the wrong size next to the other person.
    """
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(src))
    v = ply["vertex"].data.copy()
    xyz = np.column_stack([v["x"], v["y"], v["z"]]) @ R.T + t
    if scale != 1.0:
        xyz = centre + (xyz - centre) * scale
        for n in ("scale_0", "scale_1", "scale_2"):
            v[n] = (v[n].astype(np.float64) + np.log(scale)).astype(
                np.float32
            )  # PLY holds log sigma
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    q = np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]])
    q = quat_mul(quat_from_matrix(R), q)
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = q.T.astype(np.float32)
    PlyElement.describe(v, "vertex")
    PlyData([PlyElement.describe(v, "vertex")], text=False, byte_order="<").write(str(dst))
    finite = all(np.isfinite(v[n]).all() for n in v.dtype.names)
    return len(v), finite


def runs(samples):
    """Contiguous [first, last] sample runs, so the viewer can hide an absent person."""
    out = []
    for s in samples:
        if out and s == out[-1][1] + 1:
            out[-1][1] = s
        else:
            out.append([s, s])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--motion",
        action="append",
        required=True,
        metavar="TRACK=DIR",
        help="repeatable; the lhm_animate output for one track",
    )
    ap.add_argument("--tracks", required=True, help="tracking stage output (tracks.json)")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lower-body", default="Per-track MultiHMR pose per sample, no inferred gait")
    ap.add_argument(
        "--height-stride",
        type=int,
        default=4,
        help="sample every Nth frame when measuring standing height and the floor",
    )
    ap.add_argument(
        "--shared-scale",
        default="auto",
        help="auto | none | <track id>. Each track's LHM depth registration is fitted "
        "independently, so two people the same height in life come out different "
        "sizes. 'auto' rescales everyone to the track whose registration had the "
        "most supporting depth points; 'none' keeps each track's own fit.",
    )
    ap.add_argument(
        "--floor-y",
        type=float,
        default=None,
        help="absolute floor Y in the camera-0 frame, measured from a real Marble world. "
        "Implies --shared-scale floor against THAT plane instead of the median of the "
        "tracks' own feet: each track is rescaled about its own source camera until its "
        "median feet land on it. With a real floor this is a genuine constraint, not a "
        "guess, and the resulting body heights are then a check rather than an assumption.",
    )
    a = ap.parse_args()
    if a.floor_y is not None:
        a.shared_scale = "floor"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks_doc = json.loads((Path(a.tracks) / "tracks.json").read_text())
    quality = {t["track"]: t["quality"] for t in tracks_doc["tracks"]}

    cams = json.loads(Path(a.cameras).read_text())
    # world -> the packaged frame: camera 0 at the origin, gravity on +y when the run's framealign.json
    # exists beside cameras.json, camera 0's own axes otherwise (scripts/sfm_frame.py)
    R, t, fa = camera0_reframe(a.cameras)
    frame_desc = describe_frame(fa)
    print(f"frame: {frame_desc}", flush=True)

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
                coordinates=f"OpenGL, {frame_desc}. Same frame as every person/frame_*.ply. "
                "Pi3X SfM of the clip, units = Pi3X normalized (not metric).",
                frameAlign=fa,
                source=str(a.cameras),
                sourceClip=str(a.clip),
                # people.json and every sequence.json carry this hash; without it here a consumer
                # that pairs cameras with people (scripts/calibrate_person_floor.py) can only bind
                # them by file path.
                sourceSha256=hashlib.sha256(Path(a.clip).read_bytes()).hexdigest(),
                cameras=out_cams,
            ),
            indent=1,
        )
    )

    pairs = []
    for spec in a.motion:
        tid, _, folder = spec.partition("=")
        pairs.append((int(tid), Path(folder)))
    pairs.sort(key=lambda p: -quality.get(p[0], {}).get("score", 0))

    # ---- one body size for everybody ---------------------------------------
    heights, roi_points, regs, measured = {}, {}, {}, {}
    cam_y = {c["sourceIndex"]: np.array(c["camera_to_world"])[1, 3] for c in out_cams}
    for tid, motion in pairs:
        seq0 = json.loads((motion / "sequence.json").read_text())
        picks = list(range(0, len(seq0["frames"]), max(1, a.height_stride)))
        h, f = measure_body([motion / seq0["frames"][i] for i in picks], R, t)
        cy = np.array([cam_y[seq0["sourceIndices"][i]] for i in picks])
        measured[tid] = dict(heights=h, feet=f, cameraY=cy)
        heights[tid] = float(np.median(h))
        regs[tid] = (
            json.loads((motion / "registration.json").read_text())
            if (motion / "registration.json").exists()
            else {}
        )
        roi_points[tid] = regs[tid].get("depthRoiPoints") or 0
    if a.shared_scale in ("none",) or len(pairs) < 2:
        reference, multipliers = None, {tid: 1.0 for tid, _ in pairs}
    else:
        reference = (
            max(roi_points, key=roi_points.get)
            if a.shared_scale in ("auto", "floor")
            else int(a.shared_scale)
        )
        multipliers = {tid: heights[reference] / heights[tid] for tid, _ in pairs}
    shared = dict(
        mode=a.shared_scale,
        referenceTrack=reference,
        reason=(
            "Body size is whatever each track's own first-frame depth registration made it, and "
            "those are fitted independently; the reference is the track whose registration had "
            "the most Pi3X person points inside its box, and everyone is rescaled about their "
            "own source camera to match its body height."
            if reference is not None
            else "Each track keeps its own registration scale."
        ),
        depthRoiPoints=roi_points,
        heightsBeforeUnits={str(k): v for k, v in heights.items()},
        multipliers={str(k): v for k, v in multipliers.items()},
        assumption="People in one clip are assumed the same standing height; pass --shared-scale none "
        "when they are not.",
        medianHeightFrames=len(next(iter(measured.values()))["heights"]),
    )
    # Honest residual: after the size fix, do everybody's feet land on ONE floor? Scaling about the
    # camera moves a body along its view ray, so a body whose depth was wrong ends up off the floor.
    floors = {}
    for tid, m in measured.items():
        mm = multipliers[tid]
        floors[tid] = m["cameraY"] + mm * (m["feet"] - m["cameraY"])
    all_f = np.concatenate(list(floors.values()))
    shared["floorCheck"] = dict(
        medianFeetYPerTrack={str(k): float(np.median(v)) for k, v in floors.items()},
        floorSpreadUnits=float(
            max(np.median(v) for v in floors.values()) - min(np.median(v) for v in floors.values())
        ),
        floorSpreadAsBodyHeights=float(
            (
                max(np.median(v) for v in floors.values())
                - min(np.median(v) for v in floors.values())
            )
            / heights[pairs[0][0]]
        ),
        commonFloorY=float(np.median(all_f)),
        note="Feet Y of every person after the size fix. A non-zero spread is how far apart the "
        "independently fitted per-track depth registrations put the same floor.",
    )
    # The alternative: size each body so its feet land on the common floor instead of matching heights.
    if reference is not None:
        target = shared["floorCheck"]["commonFloorY"] if a.floor_y is None else float(a.floor_y)
        alt = {}
        for tid, m in measured.items():
            denom = np.median(m["feet"] - m["cameraY"])
            alt[str(tid)] = (
                float((target - np.median(m["cameraY"])) / denom) if abs(denom) > 1e-6 else 1.0
            )
        # At which floor height do the two constraints STOP fighting? Rescaling track t about its own
        # camera to put its feet on a plane T gives multiplier (T - C_t)/F_t and height H_t (T - C_t)/F_t.
        # Equal heights and a common floor are then the same answer at exactly one T. If a real world's
        # measured floor is near that T, the floor is the missing absolute and the disagreement is gone;
        # if it is far from it, the two reconstructions really are different sizes and no plane fixes it.
        ks = sorted(measured)
        if len(ks) == 2:
            A = [
                heights[k] / float(np.median(measured[k]["feet"] - measured[k]["cameraY"]))
                for k in ks
            ]
            C = [float(np.median(measured[k]["cameraY"])) for k in ks]
            den = A[0] - A[1]
            shared["consistentFloorY"] = (
                float((A[0] * C[0] - A[1] * C[1]) / den) if abs(den) > 1e-9 else None
            )
            shared["consistentFloorNote"] = (
                "The one floor plane at which matching the two bodies' heights "
                "and standing them on one floor are the same answer."
            )
        shared["floorMatchedMultipliers"] = dict(
            multipliers=alt,
            mode="--shared-scale floor",
            targetFloorY=target,
            targetSource=(
                "median of the tracks' own feet"
                if a.floor_y is None
                else "--floor-y (measured Marble world)"
            ),
            resultingHeightRatio=float(
                max(alt[str(k)] * heights[k] for k in heights)
                / min(alt[str(k)] * heights[k] for k in heights)
            ),
            note="What --shared-scale floor would apply instead. Equal heights and a common floor are "
            "the same constraint only if the per-track depth registrations agree; the gap between "
            "these two answers is the disagreement.",
        )
    if a.shared_scale == "floor" and reference is not None:
        multipliers = {
            tid: shared["floorMatchedMultipliers"]["multipliers"][str(tid)] for tid, _ in pairs
        }
        shared["applied"] = "floor"
    else:
        shared["applied"] = "height" if reference is not None else "none"
    print(json.dumps(shared, indent=1))
    cam_centre = {c["sourceIndex"]: np.array(c["camera_to_world"])[:3, 3] for c in out_cams}

    people = []
    for rank, (tid, motion) in enumerate(pairs):
        name = "person" if rank == 0 else f"person_{rank:02d}"
        person = out / name
        shutil.rmtree(person, ignore_errors=True)
        person.mkdir(parents=True)
        seq = json.loads((motion / "sequence.json").read_text())
        frames, hashes, counts, finite_all = [], [], set(), True
        mult = multipliers[tid]
        for fi, fname in enumerate(seq["frames"]):
            centre = cam_centre[seq["sourceIndices"][fi]]
            n, finite = transform_ply_sized(motion / fname, person / fname, R, t, mult, centre)
            counts.add(n)
            finite_all &= finite
            hashes.append(hashlib.sha256((person / fname).read_bytes()).hexdigest())
            frames.append(fname)
        for extra in ("motion.json", "registration.json", "missing-poses.json"):
            if (motion / extra).exists():
                shutil.copy(motion / extra, person / extra)

        seq_out = dict(seq)
        seq_out.update(
            frames=frames,
            count=len(frames),
            frame_sha256=hashes,
            gaussiansPerFrame=sorted(counts)[0] if len(counts) == 1 else sorted(counts),
            allNativeGaussianAttributesFinite=finite_all,
            people_only=True,
            coordinates=f"OpenGL Pi3X-SfM world re-expressed so {frame_desc}; body scale = "
            "this track's first-frame body-depth registration (registration.json)",
            camera0Reframe=dict(
                rotation=R.tolist(),
                translation=t.tolist(),
                source="cameras.json camera 0 of the Pi3X export"
                + (" + framealign.json gravity" if fa else ""),
                tiltFromYDeg=None if fa is None else fa["tiltFromYDeg"],
            ),
            lowerBodyProvenance=a.lower_body,
            trackIndex=tid,
            personId=name,
            manifest="../people.json",
            sharedScaleMultiplier=mult,
            sharedScaleReferenceTrack=reference,
            sourceClip=str(a.clip),
            sourceSha256=hashlib.sha256(Path(a.clip).read_bytes()).hexdigest(),
        )
        (person / "sequence.json").write_text(json.dumps(seq_out, indent=2))
        add_motion_track(person)

        body, feet = body_stats(person / frames[0])
        reg = (
            json.loads((motion / "registration.json").read_text())
            if (motion / "registration.json").exists()
            else {}
        )
        present = sorted(int(s) for s in seq.get("sampleIndices", [])) or None
        if present is None:
            src = list(tracks_doc["sourceIndices"])
            present = [src.index(i) for i in seq["sourceIndices"] if i in src]
        people.append(
            dict(
                id=name,
                track=tid,
                sequence=f"{name}/sequence.json",
                directory=name,
                label=f"person {tid}",
                frames=len(frames),
                fps=seq["fps"],
                sourceIndices=seq["sourceIndices"],
                timestamps=seq["timestamps"],
                visibleSampleRuns=runs(present),
                firstSample=present[0],
                lastSample=present[-1],
                gaussiansPerFrame=seq_out["gaussiansPerFrame"],
                bodyHeightUnits=body,
                feetY=feet,
                registrationScale=reg.get("uniformScale"),
                registration=reg,
                sizeCorrection=dict(
                    multiplier=mult,
                    appliedAboutSourceCamera=True,
                    bodyHeightBeforeUnits=heights[tid],
                    referenceTrack=reference,
                ),
                quality=quality.get(tid),
                transform=dict(
                    translation=[0.0, 0.0, 0.0],
                    quaternionXYZW=[0.0, 0.0, 0.0, 1.0],
                    scale=1.0,
                    note="Identity: every track was lifted by the same Pi3X cameras, so all people "
                    "already share one world. Present so a per-person world solve or a manual "
                    "nudge has somewhere to live.",
                ),
            )
        )
        print(
            json.dumps(
                {
                    k: people[-1][k]
                    for k in (
                        "id",
                        "track",
                        "frames",
                        "bodyHeightUnits",
                        "feetY",
                        "registrationScale",
                    )
                },
                indent=1,
            )
        )

    heights = [p["bodyHeightUnits"] for p in people]
    manifest = dict(
        schema="wander.people/1",
        clip=str(a.clip),
        sourceSha256=hashlib.sha256(Path(a.clip).read_bytes()).hexdigest(),
        fps=people[0]["fps"],
        samples=tracks_doc["samples"],
        duration=tracks_doc["duration"],
        sourceIndices=tracks_doc["sourceIndices"],
        timestamps=tracks_doc["timestamps"],
        cameras="cameras.json",
        primary=people[0]["id"],
        peopleCount=len(people),
        people=people,
        sharedScale=shared,
        coordinates=f"OpenGL, {frame_desc}. Every person is in this one frame; per-person transforms "
        "are identity.",
        frameAlign=None
        if fa is None
        else dict(
            tiltFromYDeg=fa["tiltFromYDeg"],
            source=fa["source"],
            adopted=fa["adopted"],
            file=str(a.cameras),
        ),
        sharedPlacement=dict(
            bodyHeightUnits=people[0]["bodyHeightUnits"],
            feetY=people[0]["feetY"],
            note="The viewer must derive ONE scale (from `primary`) and apply it to every person, or two "
            "people who are the same height in life come out different sizes on screen.",
            bodyHeightSpread=float(max(heights) - min(heights)),
            bodyHeightRatio=float(max(heights) / min(heights)) if min(heights) > 0 else None,
        ),
        tracking=dict(
            source=str(Path(a.tracks) / "tracks.json"),
            method=tracks_doc["method"],
            trackCount=tracks_doc["trackCount"],
            discardedShortTracks=tracks_doc["discardedShortTracks"],
        ),
        backwardCompatible="person/ holds the highest-quality track in the single-person layout, so "
        "?person=<world>/person/sequence.json still loads exactly one person.",
    )
    (out / "people.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            dict(
                out=str(out),
                people=len(people),
                bodyHeights=[round(h, 4) for h in heights],
                bodyHeightRatio=manifest["sharedPlacement"]["bodyHeightRatio"],
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
