# Known limits

Approaches already measured to fail.
Do not retry one without a new idea, and add a line here when you measure a new dead end.

- Refining body pose against the footage makes it worse: a single camera leaves a null space.
- Faces below about 64 px of face height are unreadable; per-clip head refits do not help.
- Smoke, fire, and water cannot be reconstructed from one moving camera.
- Clips with too little parallax, or too dark, fail reconstruction.
- GEN3C as a static fill generator lost to the FLUX panorama fill.
