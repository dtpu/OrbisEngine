# Accepted gym delivery

The subsequent [live raw-video evaluation](video-input-evaluation.md) generated a separate gym
candidate. It was not promoted; this accepted environment, people/props and viewer snapshot remain
unchanged. Generation success did not establish improved geometry or fewer hallucinations.

Austin's current gym reconstruction was accepted for this delivery. His source record is
`52b7759d89c216be76d347567122d7c4ea1b6d2c`; the accepted private viewer snapshot is
`viewer/snapshots/0cbba181-b115-473f-b67d-1e88d1a1889e.json`.

Open [the gym comparison](http://127.0.0.1:5399/reviews/gym-repair/index.html?quality=detail)
and select **Walk around the bench** for the standalone candidate. The original `demo=gym`
preset remains the comparison baseline. A running server pins its asset snapshot: restart your
own demo server to adopt a newer publication. Do not stop another person's server.

The published page, `preview-status.json`, collider manifest and preview filename retain older
work-in-progress wording. The acceptance above supersedes that delivery status; it does not erase
the recorded body-shape, foot/upper-back contact, inferred-dumbbell and late-person limitations.
The larger-model experiment was not promoted. No new generation or actor repair was performed
as part of this integration.

## What main adds

The focused port from Austin's `2ba38d7ca08e9a64365a21c2106e7c94598dfeda` adds optional oriented
box colliders, continuous walker collision checks and finite seat support. It preserves main's
lossless animation reader, verified original-PLY fallback, loading/retry behavior, deterministic
stance, object timing and recorded-audio handling. Colliders constrain the viewer's walker;
they do not simulate the actor or repair prerecorded contact. See [objects](objects.md).

The review's already published standalone controller supplies comparison controls. Its candidate
uses the exact placement/frame alignment and disables stance, foot locking and camera drift as
specified by Austin's package. Both review panels explicitly disable audio. The default scenes'
original-audio behavior is unchanged.

## Asset identity

- Source video SHA-256: `fef56b00ef0004614bf0f68fd046898158f3b319ab4da8916a0364289c482226`.
- Generated room SHA-256: `f76c470932c83b1f985de8be5598928f8bb5a6933ad6144f92a8986221451520`.
- Person: 213 original PLY samples at 12 fps, duration 18.0666667 seconds, retaining original
  filenames/timestamps and the package's declared omitted-sample/interpolation evidence.
- Props: two original 542-sample tracks at 30 fps, with their relative PLY models.
- Collision: measured/inferred seat, backrest and frame boxes in final viewer coordinates.

No animation was quantized, resampled, renumbered or repacked. Media and verification captures
remain in private storage and ignored local evidence, outside Git.

## Verification

The combined build/typecheck, full Ruff/Prettier check and 41 Bun regressions passed, including
nine static-collider tests. All 226 accepted-package dependencies were downloaded and checked
against the pinned catalog's size and SHA-256. A later shared snapshot retained all checked gym
entries and the six previously published stairs/elevator lossless assets unchanged.

Playwright tested current source at port 5399, routing the new gym assets through the independently
pinned private asset server because the existing Vite server retained an older snapshot. All 213
person samples, both props and three colliders loaded with no page errors or object-track warnings.
Playback and seeks reached the final sample; front, profile and opposite views were inspected
alongside the source. Four cardinal walking approaches stopped outside the bench, stationary seat
support remained stable, and map selection excluded blocked furniture positions.

The production build's full-detail comparison loaded both panels and passed synchronized
pause/play, seek, restart, source playback and reversible people/props visibility checks. The
standalone view used standard detail; the comparison exercised full detail. These are desktop
Chrome checks, not physical-headset or cold-network performance measurements. Heavy browser checks
ran sequentially; the laptop reported elevated memory pressure but no thermal warnings.

Default stairs/elevator regressions retained float32 playback with no fallback, advancing original
audio, stair ascent/descent, corridor wall stops and the map interface. Explicit missing and malformed
collider requests failed without false readiness. The malformed-manifest check exposed a late frame
progress update overwriting the startup error; main now keeps that error visible while downloads
settle. The corrected failure case and subsequent successful lossless scene load both passed.
