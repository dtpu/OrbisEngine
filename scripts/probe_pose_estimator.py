#!/usr/bin/env python3
"""Ask MultiHMR whether the scorer's top reference candidates can actually be used.

  KMP_DUPLICATE_LIB_OK=TRUE worker/.venv-da3/bin/python scripts/probe_pose_estimator.py \
      public/clips/bedroom.mp4 --scores .context/pose/frames/bedroom.json --top 8 \
      --work .context/pose/framefix/probe/bedroom --out .context/pose/framefix/probe-bedroom.json

scripts/score_reference_frames.py ranks frames on the identity they carry and stops there. The
downstream build can still refuse the winner: `worker/experiments/lhm_person.py` runs MultiHMR on
the person cut out onto white, and on bedroom's top-scored frame 198 -- a man seated behind a desk
-- that returns no detection and the whole canonical build dies. A frame that scores well and
cannot be used is worse than useless, so the scorer must gate on it.

This prepares the same cut-out prepare_lhm_person.py would write for each candidate, sends the
batch to worker/modal_pose_probe.py (one L4 call, seconds), and writes the probe file that
score_reference_frames.py --estimator-probe consumes. A local COCO detector is not a substitute:
it reads bedroom f198 at 0.999 and would pass the frame MultiHMR refuses.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
DEFAULT_MODAL = os.environ.get("WANDER_MODAL", str(ROOT / "worker/.venv-da3/bin/modal"))


def candidates(scores, top, extra):
    rows = sorted([r for r in scores["rows"] if r.get("score", 0) > 0], key=lambda r: -r["score"])
    idx = [r["index"] for r in rows[:top]]
    for e in extra:
        if e not in idx:
            idx.append(e)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--scores", required=True, help="score_reference_frames.py output for this clip")
    ap.add_argument("--work", required=True, help="where the candidate cut-outs are written")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--also", default="", help="extra frame indices to probe (e.g. the frame in use today)")
    ap.add_argument("--clip-name", help="key used in the probe file; defaults to the video stem")
    ap.add_argument("--bbox", help="x0,y0,x1,y1 track box, for a multi-person clip")
    ap.add_argument("--track-motion", help="one track's motion.json: use THAT identity's own box in "
                                           "each frame (the box moves, so a single --bbox will not do)")
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--modal", default=DEFAULT_MODAL)
    ap.add_argument("--det-thresh", type=float, default=0.3)
    a = ap.parse_args()

    from prepare_lhm_person import box_report, decode, person_mask
    scores = json.loads(Path(a.scores).read_text())
    extra = [int(x) for x in a.also.split(",") if x.strip()]
    name = a.clip_name or Path(a.video).stem
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    seeds = {}
    if a.track_motion:
        from score_reference_frames import track_seeds
        seeds = track_seeds(a.track_motion)

    reqs, skipped = [], {}
    for i in candidates(scores, a.top, extra):
        d = work / f"f{i}"; d.mkdir(exist_ok=True)
        if not (d / "source.png").exists():
            rgb, _, _ = decode(a.video, i)
            m = person_mask(rgb, a.dilate, "maskrcnn", seeds.get(i, bbox))
            rep = box_report(m, rgb.shape[1], rgb.shape[0])
            if rep.get("empty") or not rep["portrait"]:
                # lhm_person.py rejects a non-portrait crop before MultiHMR ever runs, so this
                # candidate is unusable for the same reason and needs no GPU to say so.
                skipped[str(i)] = dict(detected=False, score=0.0, reason=f"unusable person mask: {rep}")
                continue
            Image.fromarray(rgb).save(d / "source.png")
            Image.fromarray(m).save(d / "mask.png")
        reqs.append(dict(clip=name, index=i, source=str(d / "source.png"), mask=str(d / "mask.png")))

    req_path = work / "probe-in.json"
    req_path.write_text(json.dumps(reqs, indent=1))
    if reqs:
        cmd = [a.modal, "run", str(ROOT / "worker" / "modal_pose_probe.py"),
               "--requests", str(req_path), "--out", str(a.out), "--det-thresh", str(a.det_thresh)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    doc = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else dict(clips={})
    if skipped:
        doc["clips"].setdefault(name, {}).update(skipped)
        Path(a.out).write_text(json.dumps(doc, indent=1))
    rows = doc["clips"].get(name, {})
    ok = sorted([i for i, v in rows.items() if v["detected"]], key=int)
    bad = sorted([i for i, v in rows.items() if not v["detected"]], key=int)
    print(f"{name}: readable {ok}")
    print(f"{name}: UNREADABLE {bad}")


if __name__ == "__main__":
    main()
