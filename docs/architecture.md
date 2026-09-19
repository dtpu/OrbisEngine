# Architecture

Wander replays a processed clip from a movable viewpoint. It does not simulate alternate actions.
The generated environment and inferred hidden appearance extend beyond recorded evidence.

## Viewer

`index.html` redirects to `demo.html`, the presentation shell and scene picker. It embeds
`fourd.html`, which owns the Three.js/Spark scene, source video, playback clock, animated people,
objects, and walk camera. `window.wander` exposes its transport and diagnostic state to the shell
and capture scripts. Scene presets point to media manifests; those media files are not in Git.

`src/walk-map.js` renders the overhead position picker. Floor and collision manifests constrain
walking; the camera eye height derives from measured character stature. `src/xr/fourd-xr.ts`
adds the headset rig, teleport or smooth movement, snap turning, and adaptive splat budget.
Head motion comes from the XR runtime. Report frame rates measured on the physical device.

`src/inspection/video-projection.ts` is the retained projection layer used by the current viewer;
its manifest types live alongside it. The previous standalone inspection application is archived.
`src/audio/fourd-audio.ts` keeps original or reviewed spatial audio synchronized to the source
clock and uses the desktop camera or headset center pose as the listener. Its manifest, provenance,
and review requirements are documented in [audio.md](audio.md).

## Private assets

`server/shared-assets.mjs` is a Vite development middleware. It reads server-only credentials,
pins one complete S3 snapshot, verifies content hashes, and caches immutable blobs under `.context/`.
It serves logical same-origin asset URLs and supports byte ranges. Remote failures do not fall back
to unpublished local media. `WANDER_ASSETS_MODE=local` is an explicit author/offline mode.

`scripts/publish-runs.mjs` publishes assets and run archives with checksum verification, immutable
snapshots, and a conditional latest-pointer update. A running viewer keeps its existing snapshot.
The private teammate identity can read viewer assets only; author archive access is separate.
See [shared-assets.md](shared-assets.md). Vite builds skip `public/` copying and never bundle keys.

## Clip pipeline

`scripts/run_clip.py` orchestrates resumable stages in `.context/run/<name>/state.json`:

1. Detect cuts and choose a continuous shot.
2. Solve source cameras with Pi3X; track people and audit identity continuity.
3. Remove people for a clean environment input; review that input before Marble generation.
4. Build and animate each retained person, aligning all tracks to the same camera solve.
5. Detect free object flights and fit their motion from source observations.
6. Align gravity, fit environment scale and person placement, and run visual/geometry checks.
7. Optionally fine-tune on a configured host, package manifests, and publish assets and run evidence.

The scripts call bounded Modal workers under `worker/` and their retained Python helpers.
No GPU job, model download, or world generation is part of frontend installation or build.
`worker/requirements-pipeline.txt` lists the local author environment; Modal image definitions pin
their own remote dependencies. Existing model caches/access must be configured before launching.

A successful process exit does not establish visual quality. Review the real viewer against source
frames, label generated regions, and retain evidence outside Git. Metric scales depend on assumed
human stature; prefer body-heights when presenting measurements.
