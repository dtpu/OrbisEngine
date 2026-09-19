#!/usr/bin/env python3
"""VLM judgement for the Wander pipeline.

Three jobs, each validated against renders whose verdict we already know:
  screen   per-frame accept/reject of generated or rendered imagery (high volume)
  scope    is a clip in scope, and what does it contain (once per clip)
  boundary should an opening be walled off or does the space continue

Usage:
  vlm_judge.py screen   IMG [IMG ...]      [--walkable "desc"] [--model M]
  vlm_judge.py scope    IMG [IMG ...]
  vlm_judge.py boundary IMG

Prints one JSON object per image, one per line. Exit 0 always; read the JSON.
Requires OPENAI_API_KEY (see ~/.openai-env).
"""

import argparse, base64, json, os, ssl, sys, time, urllib.error, urllib.request

CTX = ssl.create_default_context()
try:
    import certifi

    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

FAST = "gpt-5.4-mini"  # per-frame: same verdicts as astra, ~1.5s, far cheaper
SMART = "gpt-6-astra"  # low volume, harder semantic calls

SCREEN = (
    "You are screening one rendered frame from a 3D reconstruction of a real place, "
    "for a product where the viewer walks around inside the scene.\n"
    'Answer ONLY with JSON: {"acceptable": true|false, "reason": "<10 words>", '
    '"blocking_object": true|false}\n'
    "REJECT: shredded or spiky geometry, smeared or melted surfaces, figures that are "
    "blobs, large black or grey voids, stretched needle artefacts.\n"
    "ACCEPT: plausible real place, even if soft, low-detail, or containing invented "
    "furniture, signage or objects. Invented content is fine; broken geometry is not.\n"
    "Set blocking_object true ONLY if a solid object stands in open floor where a "
    "person would walk. Objects against walls or in alcoves do not count."
)
SCOPE = (
    "You are triaging a video frame to decide whether this clip suits photogrammetric "
    "3D reconstruction.\n"
    'Answer ONLY with JSON: {"in_scope": true|false, "reason": "<12 words>", '
    '"effects": true|false, "people": "none|partial|full_body", '
    '"setting": "interior|exterior", "motion_blur": "none|mild|severe"}\n'
    "OUT OF SCOPE: heavy fire, smoke, water, fog or particle effects; mostly featureless "
    "surfaces; extreme motion blur throughout.\n"
    "people: full_body means feet are visible, partial means cropped."
)
BOUNDARY = (
    "You are looking at a rendered view of a reconstructed place. Beyond the edge of "
    "what the camera recorded, we must either wall the space off or invent a "
    "continuation. Walling is cheap and safe; it is wrong if the architecture is "
    "visibly open.\n"
    'Answer ONLY with JSON: {"action": "wall|continue", "reason": "<12 words>", '
    '"surface": "<what the wall should look like, or what continues>"}'
)


def ask_images(model, images, prompt, detail="low", retries=3, timeout=180):
    """One chat call carrying `prompt` plus every image in `images`. Returns the parsed JSON object,
    or {"error": ...}. `detail` is the OpenAI image fidelity knob: "low" is enough to screen a frame,
    "high" is needed to read materials, signage and layout off a full-resolution pair."""
    content = [{"type": "text", "text": prompt}]
    for img in images:
        b = base64.b64encode(open(img, "rb").read()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b}", "detail": detail},
            }
        )
    body = {"model": model, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for i in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=timeout, context=CTX))
            txt = r["choices"][0]["message"]["content"].strip()
            txt = txt.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(txt)
        except Exception as e:
            if i == retries - 1:
                return {"error": str(e)[:200]}
            # a 429 is the account's rate limit, not a transient: back off for tens of seconds and
            # honour Retry-After when it is given, so a burst of parallel runs does not get the key blocked
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                try:
                    wait = float(e.headers.get("Retry-After") or 0)
                except ValueError:
                    wait = 0
                time.sleep(max(wait, 15 * 2**i))
            else:
                time.sleep(2**i)


def ask(model, img, prompt, retries=3):
    return ask_images(model, [img], prompt, retries=retries, timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["screen", "scope", "boundary"])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model")
    a = ap.parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        print(json.dumps({"error": "OPENAI_API_KEY unset; source ~/.openai-env"}))
        return
    prompt = {"screen": SCREEN, "scope": SCOPE, "boundary": BOUNDARY}[a.job]
    model = a.model or (FAST if a.job == "screen" else SMART)
    for img in a.images:
        out = ask(model, img, prompt)
        out["image"] = os.path.basename(img)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
