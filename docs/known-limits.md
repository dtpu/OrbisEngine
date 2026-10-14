# Known limits

Approaches already measured to fail.
Do not retry one without a new idea, and add a line here when you measure a new dead end.

- Refining body pose against the footage makes it worse: a single camera leaves a null space.
- Faces below about 64 px of face height are unreadable; per-clip head refits do not help.
- Smoke, fire, and water cannot be reconstructed from one moving camera.
- Clips with too little parallax, or too dark, fail reconstruction.
- GEN3C as a static fill generator lost to the FLUX panorama fill.
- A dominant low horizontal plane can be a countertop rather than a walking floor. Kitchen source-labelled floor tiles lie about 0.61 body-heights below its selected counter plane; validate surface identity before using camera-height or walker-floor estimates.
- A track with no usable complete-body height does not constrain body scale through vertical silhouette edges alone. In the retained hall-walk shot, head/floor fitting increased visible body area and reduced source overlap despite passing the camera-height check. Tracker-derived crop flags are also fallible: at six seconds its projected feet lie outside the image although the source boots are visible. A visible-width experiment improved overlap but still failed area and contact-noise checks. Preserve the failed verdict and investigate tracking disagreement; a size-check pass does not establish accurate placement.
- Main-thread `toBlob` and `OffscreenCanvas.convertToBlob` took about four seconds per spectator image during native Quest XR sessions. Moving JPEG encoding to a dedicated worker reduced observed encoding latency to roughly 25–40 ms; simulated XR did not reproduce the main-thread stall.
