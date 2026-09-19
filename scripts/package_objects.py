#!/usr/bin/env python3
"""Package an object as a first-class citizen of a 4D world. Format: `wander.objects/2`.

Writes `objects.json` beside `people.json` in the same raw SfM world coordinates the person PLYs
use, so the viewer applies the same shared scale, the same `transform.translation` and the same
`floorFit.sharedCameraDrift` it already applies to a person.

`docs/objects.md` is the format. The short version: one schema, four motion cases, differing only
in how pose is determined per frame.

  attached      a fixed rigid offset on a body joint          worn backpack, held phone, umbrella
  free          an analytic trajectory                        a thrown bottle, a dropped ball
  handoff       attached -> free -> attached                  a throw and a catch
  worldDynamic  its own world track, on no one                a swinging door, a pulled suitcase

`blend` segments stitch the seams of a handoff and carry the distance the two fits disagree by, so
a stitch can never be mistaken for a reconstruction.

Every case bakes to the same per-source-frame table of position AND orientation, so a viewer
implements one thing and gets all four. This supersedes both `wander.objects/1` (thrown only,
proxy appearance, no orientation) and `wander-rigid-object/1` (worn only, camera-space track);
`--from-rigid-object` converts metadata from the latter; converted entries still need baked tracks.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from source_clock import camera_source_fps, resolve_source_fps

CV = np.diag([1.0, -1.0, -1.0])


def load_cameras(path):
    with open(path) as handle:
        cams = json.load(handle)["cameras"]
    idx = np.array([c["sourceIndex"] for c in cams], float)
    source_fps = camera_source_fps(cams)
    R = []
    for c in cams:
        m = np.array(c["camera_to_world"])[:3, :3]
        u, _, vt = np.linalg.svd(m)
        R.append(u @ vt)
    R = np.array(R)
    t = np.array([np.array(c["camera_to_world"])[:3, 3] for c in cams])
    return dict(
        idx=idx,
        source_fps=source_fps,
        t=t,
        slerp=Slerp(idx, Rotation.from_matrix(R)),
        K=np.array(cams[0]["source_intrinsics"]),
    )


def cam_at(C, sf):
    sf = float(np.clip(sf, C["idx"][0], C["idx"][-1]))
    return C["slerp"]([sf]).as_matrix()[0], np.array(
        [np.interp(sf, C["idx"], C["t"][:, k]) for k in range(3)]
    )


def ballistic_position(p0, v0, gravity, frame, start_frame, source_fps):
    """Evaluate a fitted ballistic curve at a source-frame time."""
    t = (float(frame) - float(start_frame)) / float(source_fps)
    return np.asarray(p0) + np.asarray(v0) * t + 0.5 * np.asarray(gravity) * t * t


def joint_world(track_dir, C, scale, joint):
    ps = torch.load(
        os.path.join(track_dir, "source-poses.pt"), map_location="cpu", weights_only=False
    )
    idx = np.asarray(ps["sourceIndices"], float)
    j = np.array([np.asarray(p["j3d"])[joint] for p in ps["poses"]])

    def at(sf):
        jc = np.array([np.interp(sf, idx, j[:, k]) for k in range(3)])
        R, t = cam_at(C, sf)
        return (R @ CV) @ jc * scale + t

    return at


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ply_gaussian_count(path):
    with open(path, "rb") as fh:
        for _ in range(64):
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def _align_y_to(d):
    """Minimal rotation taking the model's local +y onto direction `d`."""
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d = d / n
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, d))
    if c > 1 - 1e-9:
        return Rotation.identity()
    if c < -1 + 1e-9:
        return Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    ax = np.cross(y, d)
    return Rotation.from_rotvec(ax / np.linalg.norm(ax) * np.arccos(np.clip(c, -1, 1)))


def bake_orientation(frames, positions, orient, segments, fps=30.0):
    """-> (N, 4) xyzw quaternions, one per source frame, for any of the four motion cases.

    Orientation is per SEGMENT, because the four cases want different things and baking one rule
    over the whole clip gets both halves wrong. A bottle spinning at 2.7 rev/s while it sits in
    someone's hand is as wrong as one gliding through the air axis-aligned.

      attached      held upright, plus the segment's own fixed `quaternionXYZW` when it has one.
                    A carried bottle is upright; velocity-aligning it to the walk would lay it flat.
      free          local +y along the direction of travel, plus the measured spin, with the spin's
                    clock starting at the segment's own t0 rather than at frame 0.
      blend         slerp between the orientations either side of the seam.
      worldDynamic  velocity-aligned, which is the only thing a baked track on its own supports.
    """
    n = len(frames)
    fr = np.asarray(frames, float)
    v = np.gradient(positions, axis=0)
    out = [None] * n

    def spin_of(sg):
        o = (sg or {}).get("orientation") or orient or {}
        r = float(o.get("spinRevPerSec", 0.0)) if o.get("mode") == "measuredRate" else 0.0
        ax = np.array(o.get("spinAxisWorld", [0.0, 0.0, 1.0]), float)
        return r, ax / (np.linalg.norm(ax) or 1.0)

    def seg_of(f):
        for sg in segments:
            if sg["fromSourceFrame"] <= f <= sg["toSourceFrame"]:
                return sg
        return None

    last = Rotation.identity()
    for k in range(n):
        sg = seg_of(fr[k])
        kind = sg["kind"] if sg else "worldDynamic"
        if kind == "attached":
            q = sg.get("quaternionXYZW") if sg else None
            r = Rotation.from_quat(q) if q else Rotation.identity()
        elif kind in ("free", "worldDynamic"):
            r = _align_y_to(v[k]) or last
            rev, axis = spin_of(sg if kind == "free" else None)
            if kind == "free" and abs(rev) > 1e-9:
                t = (fr[k] - float(sg.get("t0SourceFrame", sg["fromSourceFrame"]))) / fps
                r = Rotation.from_rotvec(axis * (2 * np.pi * rev * t)) * r
        else:
            r = None  # blend: filled in below
        out[k] = r
        if r is not None:
            last = r

    # blend seams: slerp between the nearest resolved orientation either side
    for k in range(n):
        if out[k] is not None:
            continue
        lo = next((j for j in range(k, -1, -1) if out[j] is not None), None)
        hi = next((j for j in range(k, n) if out[j] is not None), None)
        if lo is None and hi is None:
            out[k] = Rotation.identity()
        elif lo is None:
            out[k] = out[hi]
        elif hi is None:
            out[k] = out[lo]
        else:
            u = (k - lo) / max(hi - lo, 1)
            out[k] = Slerp([0.0, 1.0], Rotation.concatenate([out[lo], out[hi]]))([u])[0]
    return np.array([r.as_quat() for r in out])


def from_rigid_object(path, mpu, last_source_frame=None):
    """Convert a `wander-rigid-object/1` package (branch aayan/vm-objects) to a /2 object.

    The old format's `attachment.transform` is 16 column-major
    floats; it is a rigid transform, so it decomposes to a translation and a quaternion with
    nothing lost, and a viewer wants those rather than a matrix.
    """
    o = json.load(open(os.path.join(path, "object.json")))
    M = np.array(o["attachment"]["transform"], float).reshape(4, 4, order="F")
    R = Rotation.from_matrix(M[:3, :3] / np.cbrt(max(abs(np.linalg.det(M[:3, :3])), 1e-12)))
    kind = o.get("kind", "worn")
    seg = dict(
        kind="attached",
        fromSourceFrame=0,
        toSourceFrame=last_source_frame,
        parent=None,
        joint=None,
        jointName=o["attachment"].get("anchor"),
        offsetLocalMetres=[float(v) * mpu for v in M[:3, 3]],
        quaternionXYZW=[float(v) for v in R.as_quat()],
        anchorDefinition=o.get("anchor_definition", ""),
    )
    m = o.get("measured", {})
    return dict(
        id=o.get("label", "object"),
        label=o.get("label", "object"),
        prompt=o.get("prompt", ""),
        motion="attached" if kind in ("worn", "carried") else "free",
        objectClass=kind,
        appearance=dict(
            kind="gaussians",
            model=os.path.join(os.path.basename(path), "object.ply"),
            modelSha256=o.get("model_sha256"),
            gaussians=o.get("gaussians"),
            frame=o.get("frame", ""),
            shapeProvenance=o.get("shapeProvenance", ""),
            colourProvenance=o.get("colourProvenance", ""),
            observedAngularSpreadDeg=m.get("observed_angular_spread_deg"),
            invented=False,
        ),
        pose=dict(
            segments=[seg],
            orientation=dict(mode="fixed", provenance="inherited from the anchor frame"),
        ),
        evidence=dict(
            silhouetteIouFixed=o["attachment"].get("silhouette_iou_after_refine"),
            medianCentroidReprojectionPx=o["attachment"].get("median_centroid_reprojection_px"),
            usableFrames=m.get("usable_frames"),
            fusedFrames=m.get("fused_frames"),
            medianIcpResidual=m.get("median_icp_residual"),
            migratedFrom="wander-rigid-object/1",
        ),
    )


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def seam_sizes(flights, nsrc, blend_frames):
    """Per-seam blend length, so two throws close together cannot produce overlapping blends."""
    n = len(flights)
    out = []
    for i, fl in enumerate(flights):
        fr, fc = fl["flight"]
        prev_fc = flights[i - 1]["flight"][1] if i else -1
        next_fr = flights[i + 1]["flight"][0] if i + 1 < n else nsrc - 1
        out.append(
            (
                int(max(1, min(blend_frames, (fr - prev_fc) // 2))),
                int(max(1, min(blend_frames, (next_fr - fc) // 2))),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument(
        "--fit", required=True, help="wander.object-fit/2 (N flights) or /1 (one flight)"
    )
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--thrower", type=int, default=None, help="only needed for a /1 fit")
    ap.add_argument("--catcher", type=int, default=None)
    ap.add_argument(
        "--holder",
        type=int,
        default=None,
        help="track that carries the object when there is no free span at all",
    )
    ap.add_argument("--joint", type=int, default=21)
    ap.add_argument("--joint-name", default="rightWrist")
    ap.add_argument("--blend-frames", type=int, default=10)
    ap.add_argument(
        "--size-m",
        default="0.060,0.120,0.060",
        help="x,y,z in metres in the OBJECT frame, where +y is the long axis",
    )
    ap.add_argument("--color-srgb", default="0.72,0.78,0.86")
    ap.add_argument("--opacity", type=float, default=0.9)
    ap.add_argument("--id", default="bottle")
    ap.add_argument("--label", default="water bottle")
    ap.add_argument(
        "--prompt", default="", help="the word that segments this object; see object_segment.py"
    )
    ap.add_argument(
        "--motion",
        default="",
        choices=["", "attached", "free", "handoff", "worldDynamic"],
        help="default: read off the fit - handoff when there is a free span, attached when not",
    )
    ap.add_argument("--object-class", default="thrown")
    ap.add_argument(
        "--model",
        default="",
        help="appearance PLY, copied into <world>/objects/<id>/ "
        "and referenced relative to objects.json",
    )
    ap.add_argument(
        "--model-frame", default="object-local; origin at the model centroid; +y is the long axis"
    )
    ap.add_argument("--shape-provenance", default="")
    ap.add_argument("--colour-provenance", default="")
    ap.add_argument(
        "--invented",
        action="store_true",
        help="set when a generative model supplied structure the footage does not show",
    )
    ap.add_argument("--observed-views", type=int, default=0)
    ap.add_argument("--observed-spread-deg", type=float, default=0.0)
    ap.add_argument(
        "--orientation",
        default="",
        help="orientation.json from object_orientation.py; a comma list gives one per free span, "
        "a single file is applied to every span and labelled as such",
    )
    ap.add_argument(
        "--flights-json", default="", help="detect_object_flights.py output, recorded as evidence"
    )
    ap.add_argument(
        "--from-rigid-object",
        default="",
        help="convert a wander-rigid-object/1 package directory instead of fitting",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    world = a.world
    with open(os.path.join(world, "people.json")) as handle:
        pj = json.load(handle)
    with open(a.fit) as handle:
        fit = json.load(handle)
    ff = pj["floorFit"]
    mpu = ff["metresPerWorldUnit"]
    C = load_cameras(a.cameras)

    src = np.array(pj["sourceIndices"], float)
    drift = ff["sharedCameraDrift"]
    doff = np.array(drift["offsetYUnits"], float)
    dsrc = src[np.array(drift["samples"], int)]
    lvC = ff["levels"][[k for k in ff["levels"] if k.startswith("C_")][0]]["translationY"]
    by_track = {p["track"]: p for p in pj["people"]}
    scale = {
        t: by_track[t]["registrationScale"]
        * by_track[t].get("sizeCorrection", {}).get("multiplier", 1.0)
        for t in by_track
    }
    cconst = {t: lvC[by_track[t]["id"]] for t in by_track}

    def drift_at(f):
        return float(np.interp(f, dsrc, doff))

    # ---- normalise /1 and /2 fits to one list of flights
    if fit.get("schema") == "wander.object-fit/2":
        flights = fit["flights"]
    else:
        fr, fc = fit["flight"]
        flights = [
            dict(
                flight=[int(fr), int(fc)],
                thrower=a.thrower,
                catcher=a.catcher,
                fits=fit["fits"],
                frames=fit["frames"],
                spanM=fit["spanM"],
                flightSec=fit["flightSec"],
                constantY=fit.get("constantY", 0.0),
            )
        ]
    for fl in flights:
        if fl.get("thrower") is None or fl.get("catcher") is None:
            raise SystemExit(
                "flight %s has no thrower/catcher; pass --thrower/--catcher" % fl["flight"]
            )
    g_units = fit.get("gravityUnitsPerS2") or 9.80665 / mpu
    try:
        source_fps = resolve_source_fps(fit=fit.get("fps"), camera=C["source_fps"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    g = np.array([0.0, -g_units, 0.0])
    nsrc = int(pj["sourceIndices"][-1]) + 1

    involved = sorted({t for fl in flights for t in (fl["thrower"], fl["catcher"])}) or [
        a.holder if a.holder is not None else sorted(by_track)[0]
    ]
    obj_const = float(np.mean([cconst[t] for t in involved]))

    hands = {
        t: joint_world(os.path.join(a.tracks_dir, f"track_{t:02d}"), C, scale[t], a.joint)
        for t in by_track
    }

    def hand_corr(t, f):
        return hands[t](f) + np.array([0, cconst[t] + drift_at(f), 0])

    def arc_of(fl):
        b = fl["fits"]["handAnchored"]
        p0 = np.array(b["p0"])
        v0 = np.array(b["v0"])
        fr = fl["flight"][0]

        def arc(f):
            return ballistic_position(p0, v0, g, f, fr, source_fps)

        return arc

    arcs = [arc_of(fl) for fl in flights]
    seams = seam_sizes(flights, nsrc, a.blend_frames)
    d_rel = [
        arcs[i](flights[i]["flight"][0]) - hand_corr(flights[i]["thrower"], flights[i]["flight"][0])
        for i in range(len(flights))
    ]
    d_cat = [
        arcs[i](flights[i]["flight"][1]) - hand_corr(flights[i]["catcher"], flights[i]["flight"][1])
        for i in range(len(flights))
    ]

    def holder_at(f):
        """Whose hand the object is in at an ATTACHED frame: the last catcher, else the next thrower."""
        prev = [i for i, fl in enumerate(flights) if fl["flight"][1] < f]
        if prev:
            return flights[prev[-1]]["catcher"]
        nxt = [i for i, fl in enumerate(flights) if fl["flight"][0] > f]
        if nxt:
            return flights[nxt[0]]["thrower"]
        return a.holder if a.holder is not None else sorted(by_track)[0]

    def pos_corrected(f):
        for i, fl in enumerate(flights):
            fr, fc = fl["flight"]
            if fr <= f <= fc:
                return arcs[i](f)
        p = hand_corr(holder_at(f), f)
        for i, fl in enumerate(flights):
            fr, fc = fl["flight"]
            b_in, b_out = seams[i]
            if fr - b_in <= f < fr:
                p = p + smoothstep((f - (fr - b_in)) / b_in) * d_rel[i]
            if fc < f <= fc + b_out:
                p = p + (1 - smoothstep((f - fc) / b_out)) * d_cat[i]
        return p

    frames = np.arange(0, nsrc)
    raw = np.array(
        [
            pos_corrected(float(f)) - np.array([0.0, drift_at(float(f)) + obj_const, 0.0])
            for f in frames
        ]
    )

    size = [float(v) for v in a.size_m.split(",")]
    col = [float(v) for v in a.color_srgb.split(",")]

    appearance = dict(
        kind="proxy", shape="capsule", axis="local +y", why=fit.get("appearanceNote", "")
    )
    if a.model:
        dest_dir = os.path.join(world, "objects", a.id)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "object.ply")
        if os.path.abspath(a.model) != os.path.abspath(dest):
            with open(a.model, "rb") as srcf, open(dest, "wb") as dst:
                dst.write(srcf.read())
        appearance = dict(
            kind="gaussians",
            model=f"objects/{a.id}/object.ply",
            modelSha256=sha256(dest),
            gaussians=ply_gaussian_count(dest),
            frame=a.model_frame,
        )
    appearance.update(
        sizeMetres=size,
        sizeWorldUnits=[v / mpu for v in size],
        colorSRGB=col,
        opacity=a.opacity,
        translucent=True,
        shapeProvenance=a.shape_provenance,
        colourProvenance=a.colour_provenance,
        observedViews=a.observed_views or None,
        observedAngularSpreadDeg=a.observed_spread_deg or None,
        invented=bool(a.invented),
    )

    # ---- orientation: one per free span when supplied, else one shared and said so
    ofiles = [p for p in a.orientation.split(",") if p] if a.orientation else []
    if not ofiles:
        orients = [
            dict(mode="velocityAligned", spinRevPerSec=0.0, provenance="no rotation was measured")
        ] * max(len(flights), 1)
    elif len(ofiles) == 1 and len(flights) > 1:
        o = json.load(open(ofiles[0]))
        o.pop("perpendicularExtentPx", None)
        o = dict(
            o,
            appliedTo="every free span",
            provenanceCaveat=(
                "this rate was measured on ONE free span and applied to all %d; "
                "the other spans' tumble is assumed, not measured" % len(flights)
            ),
        )
        orients = [o] * len(flights)
    else:
        orients = []
        for p in ofiles:
            o = json.load(open(p))
            o.pop("perpendicularExtentPx", None)
            orients.append(o)
        while len(orients) < len(flights):
            orients.append(orients[-1])

    segments = []
    cursor = 0
    ev_flights = []
    for i, fl in enumerate(flights):
        fr, fc = fl["flight"]
        b_in, b_out = seams[i]
        best = fl["fits"]["handAnchored"]
        p0 = np.array(best["p0"])
        v0 = np.array(best["v0"])
        if fr - b_in > cursor:
            segments.append(
                dict(
                    kind="attached",
                    fromSourceFrame=int(cursor),
                    toSourceFrame=int(fr - b_in),
                    parent=by_track[holder_at(cursor)]["id"],
                    joint=a.joint,
                    jointName=a.joint_name,
                    evidence="no free span detected here; the object is assumed to stay in "
                    "that hand between the throws that were detected",
                )
            )
        segments.append(
            dict(
                kind="blend",
                fromSourceFrame=int(fr - b_in),
                toSourceFrame=int(fr),
                from_="attached",
                to="free",
                offsetWorldUnits=d_rel[i].tolist(),
                offsetMetres=float(np.linalg.norm(d_rel[i]) * mpu),
                why="the fitted arc starts this far from the avatar wrist joint; see OBJECTS-STATUS",
            )
        )
        seg = dict(
            kind="free",
            model="ballistic",
            fromSourceFrame=int(fr),
            toSourceFrame=int(fc),
            p0WorldUnits=p0.tolist(),
            v0WorldUnitsPerSec=v0.tolist(),
            gWorldUnitsPerSec2=g.tolist(),
            t0SourceFrame=int(fr),
            fps=source_fps,
            space="driftCorrected",
            flightIndex=i,
            throwerTrack=int(fl["thrower"]),
            catcherTrack=int(fl["catcher"]),
            orientation=orients[i],
            evidence=dict(
                tracked2dFrames=len(fl.get("frames", [])),
                fittedWindowSourceFrames=[int(fr), int(fc)],
                fittedWindowSeconds=fl.get("flightSec"),
                spanMetres=fl.get("spanM"),
                reprojectionPxRms=best["reprojPxRms"],
                reprojectionPxMax=best["reprojPxMax"],
                releaseToWristMetres=float(np.linalg.norm(d_rel[i]) * mpu),
                catchToWristMetres=float(np.linalg.norm(d_cat[i]) * mpu),
                depthSigmaMetres=best["p0SigmaM"][2],
                throwSpeedMetresPerSec=best["speedMs"],
                imageOnlyReprojectionPxRms=fl["fits"].get("imageOnly", {}).get("reprojPxRms"),
            ),
        )
        segments.append(seg)
        segments.append(
            dict(
                kind="blend",
                fromSourceFrame=int(fc),
                toSourceFrame=int(fc + b_out),
                from_="free",
                to="attached",
                offsetWorldUnits=d_cat[i].tolist(),
                offsetMetres=float(np.linalg.norm(d_cat[i]) * mpu),
            )
        )
        cursor = int(fc + b_out)
        ev_flights.append(
            seg["evidence"]
            | dict(
                freeSegmentSourceFrames=[int(fr), int(fc)],
                throwerTrack=int(fl["thrower"]),
                catcherTrack=int(fl["catcher"]),
            )
        )
    if cursor <= nsrc - 1:
        segments.append(
            dict(
                kind="attached",
                fromSourceFrame=int(cursor),
                toSourceFrame=int(nsrc - 1),
                parent=by_track[holder_at(cursor)]["id"],
                joint=a.joint,
                jointName=a.joint_name,
                evidence=(
                    "object visible in that hand in the source frames"
                    if not flights
                    else "after the last detected throw; assumed held, not tracked"
                ),
            )
        )

    motion = a.motion or ("handoff" if len(flights) > 1 else "handoff" if flights else "attached")
    obj = dict(
        id=a.id,
        label=a.label,
        prompt=a.prompt,
        motion=motion,
        objectClass=a.object_class,
        appearance=appearance,
        pose=dict(
            orientation=orients[0]
            if flights
            else dict(mode="fixed", provenance="no free span; orientation is not measured"),
            segments=segments,
        ),
        transform=dict(translation=[0.0, obj_const, 0.0], quaternionXYZW=[0, 0, 0, 1], scale=1.0),
        appliesSharedCameraDrift=True,
        evidence=dict(
            freeSpans=len(flights),
            tracked2dFrames=int(sum(len(fl.get("frames", [])) for fl in flights)),
            totalSourceFrames=int(nsrc),
            interpolated2dFrames=0,
            freeSourceFrames=int(sum(fl["flight"][1] - fl["flight"][0] + 1 for fl in flights)),
            attachedSourceFrames=int(
                nsrc - sum(fl["flight"][1] - fl["flight"][0] + 1 for fl in flights)
            ),
            # scalar, and the WORST of the throws: a viewer HUD reads this one number, and an
            # average would flatter the fit. The per-throw values are right below it.
            reprojectionPxRms=round(
                max([fl["fits"]["handAnchored"]["reprojPxRms"] for fl in flights] or [0.0]), 3
            ),
            reprojectionPxMax=round(
                max([fl["fits"]["handAnchored"]["reprojPxMax"] for fl in flights] or [0.0]), 3
            ),
            reprojectionPxRmsPerFlight=[
                round(float(fl["fits"]["handAnchored"]["reprojPxRms"]), 3) for fl in flights
            ],
            reprojectionPxMaxPerFlight=[
                round(float(fl["fits"]["handAnchored"]["reprojPxMax"]), 3) for fl in flights
            ],
            spanMetres=round(float(sum(fl.get("spanM") or 0.0 for fl in flights)), 3),
            spanMetresPerFlight=[round(float(fl.get("spanM") or 0.0), 3) for fl in flights],
            perFlight=ev_flights,
            detection=json.load(open(a.flights_json))["rule"] if a.flights_json else None,
            attachedProvenance=(
                "every attached span is an ASSUMPTION: the detector only sees the object "
                "when it is clear of the person masks, so a held object is never "
                "directly observed. What is measured is when it was in the AIR."
            ),
        ),
    )

    obj["bakedTrack"] = dict(
        space="raw SfM world, identical to the person PLYs before transform.translation",
        fps=source_fps,
        sourceFrames=frames.tolist(),
        sampleIndex=(frames / (pj["sourceIndices"][-1] / (pj["samples"] - 1))).round(4).tolist(),
        positions=np.round(raw, 5).tolist(),
        quaternionsXYZW=np.round(
            bake_orientation(
                frames,
                raw,
                orients[0] if flights else {},
                obj["pose"]["segments"],
                fps=source_fps,
            ),
            5,
        ).tolist(),
        visible=[True] * len(frames),
    )

    out = dict(
        schema="wander.objects/2",
        clip=pj.get("clip"),
        fps=pj["fps"],
        samples=pj["samples"],
        sourceIndices=pj["sourceIndices"],
        timestamps=pj["timestamps"],
        metresPerWorldUnit=mpu,
        coordinates=(
            "Raw SfM world, OpenGL, camera 0 = identity - the same frame the person "
            "PLYs are written in. Apply the shared scale, then transform.translation, "
            "then floorFit.sharedCameraDrift, exactly as for a person."
        ),
        people="people.json",
        objects=[obj],
    )
    for extra in a.from_rigid_object.split(",") if a.from_rigid_object else []:
        out["objects"].append(from_rigid_object(extra.strip(), mpu, int(nsrc - 1)))
    path = a.out or os.path.join(world, "objects.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print("wrote", path, os.path.getsize(path), "bytes")
    print(
        "%d free spans, %d segments, object constantY %.4f u"
        % (len(flights), len(segments), obj_const)
    )
    for i, fl in enumerate(flights):
        print(
            "  flight %d f%d-%d  track %d -> %d  release %.3f m  catch %.3f m  reproj %.2f px"
            % (
                i,
                fl["flight"][0],
                fl["flight"][1],
                fl["thrower"],
                fl["catcher"],
                np.linalg.norm(d_rel[i]) * mpu,
                np.linalg.norm(d_cat[i]) * mpu,
                fl["fits"]["handAnchored"]["reprojPxRms"],
            )
        )


if __name__ == "__main__":
    main()
