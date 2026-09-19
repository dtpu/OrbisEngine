"""Say what a detected thrown object is, from its own pixels, so nothing has to be typed.

`detect_object_flights.py` finds the flights and cuts a crop per observation. This picks the
sharpest few of those crops, shows them to a VLM and gets back the words the rest of the object
pipeline used to be given by hand: a label, the segmentation word, a refine prompt for the
image-to-3D generator, and a colour. Alongside that it estimates the object's physical size from
the tracklet's apparent scale and the depth the ballistic fit put it at: a blob `s` pixels across
at depth `z` metres under focal `f` is `s * z / f` metres, an UPPER bound because the difference
footprint is inflated by the motion within a frame.

Writes description.json and best-crop.png (the sharpest crop, upscaled, for the generator).
"""

import argparse, json, os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "worker", "stages"))

PROMPT = (
    "These are crops of ONE small object, cut from consecutive frames of a phone video while it was "
    "in free flight after being thrown or tossed by a person. They are tiny and motion-blurred; the "
    "object is roughly {size_cm} cm along its longest side (estimated from the video's own geometry). "
    "Identify it as best the pixels allow. Answer ONLY with JSON: "
    "{{\"label\": \"<2-4 words, e.g. 'plastic water bottle', 'tennis ball'>\", "
    '"category": "bottle|ball|can|cup|phone|keys|other", '
    '"segmentWord": "<a short noun phrase to segment it with, e.g. \'a plastic water bottle\'>", '
    '"refinePrompt": "<one sentence describing this exact object for a product photo generator: '
    'material, colour, label or markings, proportions>", '
    '"colourSRGB": [r, g, b] in 0..1 of the object\'s dominant colour, '
    '"longOverShort": <ratio of long axis to short axis, 1.0 for a sphere>, '
    '"translucent": true|false, "confidence": 0..1, "notes": "<what is uncertain>"}}'
)


def pick_crops(fl, crops_dir, n):
    """Sharpest crops across all flights, at most ceil(n / flights) from any one flight so a long
    throw does not crowd out a short one."""
    per = []
    for i, r in enumerate(fl["flights"]):
        cs = [c for c in r.get("crops", []) if os.path.exists(os.path.join(crops_dir, c["file"]))]
        cs.sort(key=lambda c: -c["laplacianVar"] * (c["box"][2] - c["box"][0]))
        per.append([(i, c) for c in cs])
    k = max(1, -(-n // max(1, len(per))))
    out = [c for cs in per for c in cs[:k]]
    out.sort(key=lambda ic: -ic[1]["laplacianVar"])
    return out[:n]


def size_estimate(fl, cameras):
    """-> per-flight and overall apparent size in metres from bbox px x depth / focal."""
    K = np.array(json.load(open(cameras))["cameras"][0]["source_intrinsics"], float)
    f = float(0.5 * (K[0, 0] + K[1, 1]))
    per = []
    for r in fl["flights"]:
        z = r.get("depthM")
        if not z:
            continue
        long_px = np.array(r["sizePx"], float)
        short_px = np.array([min(b[2], b[3]) for b in r.get("bboxPx", [])] or long_px, float)
        per.append(
            dict(
                flight=r["frames"][:1] + r["frames"][-1:],
                depthM=z,
                focalPx=f,
                longAxisMetresMedian=float(np.median(long_px) * z / f),
                longAxisMetresMin=float(long_px.min() * z / f),
                shortAxisMetresMedian=float(np.median(short_px) * z / f),
                shortAxisMetresMin=float(short_px.min() * z / f),
            )
        )
    if not per:
        return dict(perFlight=[], longAxisMetres=None, shortAxisMetres=None)
    return dict(
        perFlight=per,
        longAxisMetres=float(np.median([p["longAxisMetresMin"] for p in per])),
        shortAxisMetres=float(np.median([p["shortAxisMetresMin"] for p in per])),
        rule=(
            "bbox side in px x fitted depth / focal, per observation; the MIN over a flight is "
            "taken because the differencing footprint is inflated by intra-frame motion, then "
            "the median over flights. Upper bounds, not measurements of a resolved silhouette."
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", required=True, help="detect_object_flights.py output with crops")
    ap.add_argument("--crops-dir", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument(
        "--out", required=True, help="description.json; best-crop.png is written beside it"
    )
    ap.add_argument("--n-crops", type=int, default=6)
    ap.add_argument("--upscale", type=int, default=6)
    ap.add_argument("--model", default=None, help="default: vlm_judge.SMART")
    ap.add_argument(
        "--no-vlm", action="store_true", help="size and crops only; label falls back to generic"
    )
    a = ap.parse_args()

    fl = json.load(open(a.flights))
    picks = pick_crops(fl, a.crops_dir, a.n_crops)
    if not picks:
        raise SystemExit("no crops in " + a.flights)
    size = size_estimate(fl, a.cameras)
    odir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(odir, exist_ok=True)

    ups = []
    for k, (i, c) in enumerate(picks):
        im = cv2.imread(os.path.join(a.crops_dir, c["file"]))
        im = cv2.resize(im, None, fx=a.upscale, fy=a.upscale, interpolation=cv2.INTER_CUBIC)
        p = os.path.join(odir, "vlm-crop%02d.png" % k)
        cv2.imwrite(p, im)
        ups.append(p)
        if k == 0:
            cv2.imwrite(os.path.join(odir, "best-crop.png"), im)

    desc = dict(
        label="small thrown object",
        category="other",
        segmentWord="a small thrown object",
        refinePrompt="a small handheld object",
        colourSRGB=None,
        longOverShort=None,
        translucent=None,
        confidence=0.0,
        notes="VLM not consulted",
    )
    if not a.no_vlm:
        from vlm_judge import ask_images, SMART

        model = a.model or SMART
        cm = (size["longAxisMetres"] or 0.1) * 100
        r = ask_images(model, ups, PROMPT.format(size_cm="%.0f" % cm), detail="high")
        if "error" in r:
            desc["notes"] = "VLM error: " + r["error"]
        else:
            desc.update({k: r.get(k, desc.get(k)) for k in desc})
            desc["model"] = model
            desc["raw"] = r

    # the model's own colour reading is checked against the crop pixels: median of the central
    # third of the sharpest crop, in sRGB 0..1
    im = cv2.imread(os.path.join(a.crops_dir, picks[0][1]["file"]))
    h, w = im.shape[:2]
    core = im[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3].reshape(-1, 3)
    med = np.median(core, axis=0)[::-1] / 255.0
    desc["colourSRGBFromPixels"] = [float(v) for v in med]
    if not desc.get("colourSRGB"):
        desc["colourSRGB"] = desc["colourSRGBFromPixels"]

    long_m = size["longAxisMetres"]
    short_m = size["shortAxisMetres"]
    if long_m and desc.get("longOverShort"):
        # the short axis is the one least inflated by blur; the model's aspect turns it into a length
        short_m = min(short_m, long_m)
        long_m = min(long_m, short_m * float(desc["longOverShort"]))
    size_m = [short_m or 0.06, long_m or 0.1, short_m or 0.06]

    out = dict(
        schema="wander.object-description/1",
        flights=a.flights,
        cropsUsed=[dict(flight=i, **c) for i, c in picks],
        description=desc,
        size=size,
        sizeMetres=[float(v) for v in size_m],
        sizeRule=(
            "[short, long, short] in the object frame where +y is the long axis; short from the "
            "least-blurred bbox side, long capped by the VLM aspect ratio"
        ),
        id=(desc.get("category") if desc.get("category") not in (None, "other") else "object"),
    )
    json.dump(out, open(a.out, "w"), indent=1)
    print(
        json.dumps(
            dict(
                label=desc["label"],
                category=desc["category"],
                sizeMetres=out["sizeMetres"],
                confidence=desc["confidence"],
                crops=[c["file"] for _, c in picks],
            ),
            indent=1,
        )
    )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
