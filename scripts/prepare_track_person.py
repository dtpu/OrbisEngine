#!/usr/bin/env python3
"""Pick one track's best avatar-input frame and build the LHM prepared-person folder for it.

  worker/.venv-da3/bin/python scripts/prepare_track_person.py .context/mp/<clip>/tracks \
      --track 0 --clip public/clips/<clip>.mp4 --out .context/mp/<clip>/prepared-00

LHM wants a big, full-body, unoccluded, camera-facing crop. The tracker's own `candidateSamples`
rank on size and visibility only, which on a two-person clip picks whichever sample is largest --
often a back view, because shoulder spread is the same from behind. This adds the facing term:

  forward = R(rootRotationVector) @ [0,0,1] in the OpenCV source camera, so forward.z < 0 means
  the body faces the lens. Checked against a frame read by eye (elevator track 1, sample 85, a
  3/4 front view) which scores -0.63; back views of the same clip score around +0.8.

The mask then comes from Mask R-CNN restricted to this track's box, so a two-person frame yields
the right identity instead of whichever person Mask R-CNN scored higher.
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


def rank(records, prefer_front=True):
    out = []
    for r in records:
        j = np.array(r["projectedBodyJoints"])
        height = float(j[:, 1].max() - j[:, 1].min())
        spread = float(abs(j[16, 0] - j[17, 0]) / max(height, 1.0))  # 16/17 = SMPL-X shoulders
        forward_z = float((rodrigues(r["rootRotationVector"]) @ np.array([0.0, 0.0, 1.0]))[2])
        facing = max(0.0, -forward_z) if prefer_front else 1.0
        full = 1.0 if all(r["jointProjectionInImage"]) else 0.2
        clear = 1.0 - float(r.get("occludedFraction", 0.0))
        fit = facing * spread * r["heightFraction"] * r["score"] * full * clear
        out.append(dict(fit=fit, sample=r["sample"], sourceIndex=r["sourceIndex"], forwardZ=forward_z,
                        shoulderSpread=spread, heightFraction=r["heightFraction"], score=r["score"],
                        fullyInFrame=full == 1.0, occludedFraction=r.get("occludedFraction", 0.0),
                        box=r["maskBox"]))
    out.sort(key=lambda d: -d["fit"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", help="tracking stage output directory (holds tracks.json)")
    ap.add_argument("--track", type=int, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, help="override the chosen source frame")
    ap.add_argument("--pad", type=float, default=0.06, help="fraction of box size added around the box")
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--method", default="maskrcnn", choices=["segformer", "maskrcnn"])
    ap.add_argument("--any-facing", action="store_true", help="do not prefer camera-facing samples")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    tracks = Path(a.tracks)
    records = json.loads((tracks / f"track_{a.track:02d}" / "motion.json").read_text())["frames"]
    ranked = rank(records, prefer_front=not a.any_facing)
    chosen = next((r for r in ranked if r["sourceIndex"] == a.frame), ranked[0]) if a.frame is not None else ranked[0]
    x0, y0, x1, y1 = chosen["box"]
    px, py = (x1 - x0) * a.pad, (y1 - y0) * a.pad
    box = f"{x0 - px:.0f},{y0 - py:.0f},{x1 + px:.0f},{y1 + py:.0f}"
    print(json.dumps(dict(track=a.track, chosen=chosen, paddedBox=box,
                          runnerUp=ranked[1] if len(ranked) > 1 else None), indent=1))
    cmd = [a.python, str(ROOT / "scripts" / "prepare_lhm_person.py"), a.clip,
           "--frame", str(chosen["sourceIndex"]), "--out", a.out, "--method", a.method,
           "--dilate", str(a.dilate), "--bbox", box]
    subprocess.run(cmd, cwd=ROOT, check=True)
    meta = Path(a.out) / "prepared.json"
    doc = json.loads(meta.read_text())
    doc.update(trackIndex=a.track, trackSample=chosen["sample"], trackFit=chosen,
               trackSource=str(tracks / "tracks.json"))
    meta.write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
