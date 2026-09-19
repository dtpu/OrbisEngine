# Kitchen continuation checkpoint

The user reported disappearing objects, James intersecting cabinets/walls, and restricted walking.
These are unresolved acceptance failures. The accepted gym and the previously published kitchen
remain available; the source-coloured kitchen candidate below is an experiment, not a replacement
certified to fix them.

The newly supplied Modal profile `austin-jian547` authenticated successfully. It had no active apps
or model volumes at preflight. Credentials were stored privately with mode 600; other profiles were
preserved. No new paid job has been submitted. Existing receipts and attempt counts remain intact;
the requested additional spending cap has not yet been supplied.

## Recovered observed geometry

The missing eight-anchor Pi3X archive was recovered with verified ranged S3 transfers:
54,200,874 bytes, SHA-256 `74e54e99b45a6707fc280ed8332676b033b1574fa86c3b54485d762d5b44fcea`.
Its original RGB/depth, poses, confidence and masks belong to the already verified kitchen derivative.
An independent reconstruction of `static-scene.npy` matched all 1,541,687 points exactly, confirming
the cache coordinate convention. This required no new inference.

`scripts/package_static_anchors.py` exports original RGB as Gaussian SH0 in the packaged people/camera
OpenGL frame. It checks source hashes, camera/time correspondence, coordinate consistency, mask shape,
finite positive-depth points and bounded radii. Explicit source-bound track rectangles and anchor
selection can exclude missed people and inconsistent fixture states. No empty regions are filled.
Run `uv run --locked scripts/test_package_static_anchors.py` for the four geometry/mask regressions.

The initial eight-anchor result contains missed-person ghosts and inconsistent open/closed fridge
states. It is quarantined as evidence. The conservative candidate uses six closed-fridge anchors,
0–36.169 seconds, plus expanded recorded person rectangles. It contains 1,115,485 splats. Its static
coverage stops at those selected observations; the existing people/audio timeline still covers all
50.73 seconds. It cannot replay fridge motion, and moving cookware remains inconsistent.

The real viewer exposes a more faithful sink and kitchen layout, but also large unfilled masked
regions and floating feet. It is **not accepted**. Publication of its review checkpoint records this
failure; it does not silently replace the current kitchen.

## Floor and contact diagnosis

The dominant low plane in this kitchen is the countertop (native Y about -1.22), not the floor.
Five manually reviewed source polygons on visible floor tiles support a lower plane near -3.11.
Of 21,367 labelled depth samples, 18,400 lie within the fitted tolerance. Their plane-fit RMS is
0.00183 body-heights; this measures consistency of estimated depth, not a physical survey.

Next work must preserve recorded objects, keep James aligned with the room through the full clip,
and validate movement through observed open space. A camera-centred depth calibration is being
investigated: it can adjust floor contact while preserving each recorded camera projection, without
optimizing articulation. It has not been implemented or accepted at this checkpoint.

Private evidence is under `.context/evidence/kitchen-continuation/`. The current source, frames,
previews and published snapshot are documented in [the delivery report](gym-kitchen-delivery.md).
