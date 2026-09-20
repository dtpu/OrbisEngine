#!/usr/bin/env python3
"""Pick one track's best avatar-input frame and build the LHM prepared-person folder for it.

  uv run --locked --group inference python scripts/prepare_track_person.py .context/mp/<clip>/tracks \
      --track 0 --clip public/clips/<clip>.mp4 --out .context/mp/<clip>/prepared-00

LHM wants a big, full-body, unoccluded, camera-facing crop. The tracker's own `candidateSamples`
rank on size and visibility only, which on a two-person clip picks whichever sample is largest --
often a back view, because shoulder spread is the same from behind. This adds the facing term:

  forward = R(rootRotationVector) @ [0,0,1] in the OpenCV source camera, so forward.z < 0 means
  the body faces the lens. Checked against a frame read by eye (elevator track 1, sample 85, a
  3/4 front view) which scores -0.63; back views of the same clip score around +0.8.

The mask then comes from Mask R-CNN restricted to this track's box, so a two-person frame yields
the right identity instead of whichever person Mask R-CNN scored higher.

Samples whose person is cropped by a frame edge are excluded, not merely down-weighted, whenever
any whole-person sample exists: a cropped sample's mask is a torso, prepare_lhm_person.py needs a
portrait person, and the old down-weight still put hp-fly-s63's first frame (a boy on a broom,
head and feet outside the frame) on top and failed the build. When nothing is whole the best
cropped sample is used and the reason is recorded, and when even that cannot yield a portrait
person -- a movie close-up, a tight broadcast crop -- this refuses with that limit named.
"""

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def rodrigues(v):
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = np.asarray(v) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# A subject standing entirely inside the frame is no taller than the frame, so a tracked height
# above this means the body already runs past an edge and the mask can only hold the part that
# stayed inside. That is exactly how hp-fly-s63's top-ranked sample (heightFraction 1.79, head
# and feet cropped) produced a torso-only mask wider than it was tall, which
# prepare_lhm_person.py's portrait check then refused.
FULL_BODY_HEIGHT_FRACTION = 1.0

# The fallback floor when no sample is fully in frame. `heightFraction` is the tracked body
# height measured in frame heights, so its reciprocal is roughly the share of that body the
# frame can show: half a body still crops to something portrait-shaped, a movie close-up at
# heightFraction 6 (about a sixth of a body) never does.
MIN_VISIBLE_BODY_FRACTION = 0.5


class NoUsableSample(RuntimeError):
    """No sample of this track can meet the prepared-person input's own limits."""


def visible_body_fraction(height_fraction):
    """Roughly how much of the tracked body's height the frame can hold."""
    return min(1.0, 1.0 / max(float(height_fraction), 1e-9))


def portrait_mask(box):
    """Is this track's mask box taller than it is wide, as the prepared person must be?"""
    x0, y0, x1, y1 = [float(v) for v in box]
    return (y1 - y0) > (x1 - x0)


def rank(records, prefer_front=True):
    """Score every sample of one track, best first, on the terms LHM's input actually needs."""
    out = []
    for r in records:
        j = np.array(r["projectedBodyJoints"])
        height = float(j[:, 1].max() - j[:, 1].min())
        spread = float(abs(j[16, 0] - j[17, 0]) / max(height, 1.0))  # 16/17 = SMPL-X shoulders
        forward_z = float((rodrigues(r["rootRotationVector"]) @ np.array([0.0, 0.0, 1.0]))[2])
        facing = max(0.0, -forward_z) if prefer_front else 1.0
        full = 1.0 if all(r["jointProjectionInImage"]) else 0.2
        clear = 1.0 - float(r.get("occludedFraction", 0.0))
        # Capped on purpose: a body taller than the frame is a closer crop, not a bigger
        # subject, and rewarding that size is what put a cropped close-up on top.
        size = min(float(r["heightFraction"]), FULL_BODY_HEIGHT_FRACTION)
        fit = facing * spread * size * r["score"] * full * clear
        out.append(
            dict(
                fit=fit,
                sample=r["sample"],
                sourceIndex=r["sourceIndex"],
                forwardZ=forward_z,
                shoulderSpread=spread,
                heightFraction=r["heightFraction"],
                visibleBodyFraction=visible_body_fraction(r["heightFraction"]),
                score=r["score"],
                fullyInFrame=full == 1.0,
                portraitMask=portrait_mask(r["maskBox"]),
                occludedFraction=r.get("occludedFraction", 0.0),
                box=r["maskBox"],
            )
        )
    out.sort(key=lambda d: -d["fit"])
    return out


def in_frame(candidate):
    """A whole, uncropped person: what the prepared-person input is supposed to be built from."""
    return (
        candidate["fullyInFrame"]
        and candidate["heightFraction"] <= FULL_BODY_HEIGHT_FRACTION
        and candidate["portraitMask"]
    )


def usable(candidate):
    """Cropped, but still enough of a portrait body for a mask the avatar build can accept."""
    return (
        candidate["portraitMask"] and candidate["visibleBodyFraction"] >= MIN_VISIBLE_BODY_FRACTION
    )


def unusable_reason(ranked):
    """Why this track can offer nothing, in the terms the prepared-person input refuses on."""
    best = max(ranked, key=lambda c: c["visibleBodyFraction"])
    landscape = sum(1 for c in ranked if not c["portraitMask"])
    return (
        f"no sample of this track can be prepared as an avatar input: of {len(ranked)} tracked "
        f"samples, {sum(1 for c in ranked if not usable(c))} fail the prepared-person limits "
        f"({landscape} have a mask box wider than tall; the rest show less than "
        f"{MIN_VISIBLE_BODY_FRACTION:.0%} of the body). The best sample is {best['sample']} "
        f"(source frame {best['sourceIndex']}), whose tracked body is {best['heightFraction']:.2f} "
        f"frame heights tall, so at most {best['visibleBodyFraction']:.0%} of the person is ever "
        "inside the frame. The subject is never fully visible in this clip -- a close-up or a "
        "tight broadcast crop -- and prepare_lhm_person.py needs a portrait mask of a person, "
        "so this track has no avatar input to build from."
    )


def select(records, prefer_front=True, frame=None):
    """Choose one sample to prepare, and record what that choice excluded and why."""
    ranked = rank(records, prefer_front=prefer_front)
    if not ranked:
        raise NoUsableSample("this track has no tracked samples")
    if frame is not None:
        chosen = next((c for c in ranked if c["sourceIndex"] == frame), None)
        if chosen is None:
            raise NoUsableSample(f"source frame {frame} is not a sample of this track")
        reason = f"explicitly requested source frame {frame}; ranking and limits not applied"
        return dict(chosen=chosen, ranked=ranked, basis="requested", reason=reason, excluded=0)
    preferred = [c for c in ranked if in_frame(c)]
    if preferred:
        return dict(
            chosen=preferred[0],
            ranked=ranked,
            basis="fully-in-frame",
            reason=(
                f"best fit among {len(preferred)} samples showing the whole person inside the "
                f"frame; {len(ranked) - len(preferred)} cropped or landscape-mask samples were "
                "excluded before ranking"
            ),
            excluded=len(ranked) - len(preferred),
        )
    fallback = [c for c in ranked if usable(c)]
    if not fallback:
        raise NoUsableSample(unusable_reason(ranked))
    return dict(
        chosen=fallback[0],
        ranked=ranked,
        basis="cropped-fallback",
        reason=(
            f"no sample shows the whole person: every one of the {len(ranked)} samples is "
            f"cropped by a frame edge. Chosen from the {len(fallback)} that still keep at least "
            f"{MIN_VISIBLE_BODY_FRACTION:.0%} of the body in a portrait mask box "
            f"({len(ranked) - len(fallback)} excluded); the avatar is built from a partly "
            "visible person and its unseen parts stay inferred"
        ),
        excluded=len(ranked) - len(fallback),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", help="tracking stage output directory (holds tracks.json)")
    ap.add_argument("--track", type=int, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, help="override the chosen source frame")
    ap.add_argument(
        "--pad", type=float, default=0.06, help="fraction of box size added around the box"
    )
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--method", default="maskrcnn", choices=["segformer", "maskrcnn"])
    ap.add_argument("--any-facing", action="store_true", help="do not prefer camera-facing samples")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    tracks = Path(a.tracks)
    records = json.loads((tracks / f"track_{a.track:02d}" / "motion.json").read_text())["frames"]
    try:
        selection = select(records, prefer_front=not a.any_facing, frame=a.frame)
    except NoUsableSample as refusal:
        sys.exit(f"track {a.track}: {refusal}")
    chosen, ranked = selection["chosen"], selection["ranked"]
    x0, y0, x1, y1 = chosen["box"]
    px, py = (x1 - x0) * a.pad, (y1 - y0) * a.pad
    box = f"{x0 - px:.0f},{y0 - py:.0f},{x1 + px:.0f},{y1 + py:.0f}"
    print(
        json.dumps(
            dict(
                track=a.track,
                chosen=chosen,
                selectionBasis=selection["basis"],
                selectionReason=selection["reason"],
                excludedSamples=selection["excluded"],
                consideredSamples=len(ranked),
                paddedBox=box,
                runnerUp=ranked[1] if len(ranked) > 1 else None,
            ),
            indent=1,
        )
    )
    cmd = [
        a.python,
        str(ROOT / "scripts" / "prepare_lhm_person.py"),
        a.clip,
        "--frame",
        str(chosen["sourceIndex"]),
        "--out",
        a.out,
        "--method",
        a.method,
        "--dilate",
        str(a.dilate),
        # Attached form: a box that runs off the left or top edge starts with a minus sign, and
        # argparse would otherwise read it as an option name instead of this option's value.
        f"--bbox={box}",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    meta = Path(a.out) / "prepared.json"
    doc = json.loads(meta.read_text())
    doc.update(
        trackIndex=a.track,
        trackSample=chosen["sample"],
        trackFit=chosen,
        trackSelectionBasis=selection["basis"],
        trackSelectionReason=selection["reason"],
        trackSamplesExcluded=selection["excluded"],
        trackSamplesConsidered=len(ranked),
        trackSource=str(tracks / "tracks.json"),
    )
    meta.write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
