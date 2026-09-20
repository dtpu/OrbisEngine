# Two friends outside the venue

The `img5594-video-baseline` picker entry loads the two-person plaza recording with its saved
ground placement. Desktop and headset menus share the same entry.

- Original recording: `/clips/img5594-video-baseline/source.mov`, 10.928333 seconds.
- Browser playback: `/clips/img5594-video-baseline/playback-original-audio.mp4`.
- World: `/marble-img5594-video-baseline-clean.spz`.
- People, cameras and placement: `/worlds/img5594-video-baseline-4d/`.

The supplied placement flags are retained: `place=1`, `rot=0,0,0`, `rotfix=0`,
`feetmode=sfm`, `feetlock=0`, `stance=0`, and `camdrift=0`. Both people load through the
shared people manifest.

## Delivery integrity

The shared snapshot already contained the scene and float32 motion when this integration was
prepared. Its 270 unchanged archive entries include the original PLYs, world, cameras and
placement. Both published motion tracks were checked against all original frames: 130 samples
for `person` and 127 for `person_01`, each with 40,000 splats. All 71.96 million position and
rotation values match bit for bit. Static appearance, original frame hashes and non-motion
sequence metadata are unchanged.

The earlier browser MP4 re-encoded the soundtrack. The selected playback copy keeps its H.264
video packets unchanged and remuxes the original AAC stream. All 515 audio packet payloads,
timestamps, durations and priming metadata match the original MOV. The browser video is
10.942 seconds long; the viewer loops at the scene's original 10.928333-second duration.

Snapshot `viewer/snapshots/9e2bc0bd-1bf8-4f51-aa98-00cd5e49a955.json` adds only the
19,113,899-byte playback copy, preserving all 18,047 previous viewer entries. Its SHA-256 is
`b151efc1de649b9170dc264fe3ac105fdacadddb87fa9981130b4beee4245ae2`.
A fresh S3 download matched the published size and hash. A running server must adopt this
snapshot or a later one; follow the shared-assets restart guidance and preserve servers owned
by other sessions.

## Review and limits

Chrome loaded all 257 person samples without compact-track fallback or page errors, using the
supplied placement and both per-person size corrections. The six-entry picker, seeking,
playback loop, original-audio unlock and walking were checked. Full-body views at 1, 5 and
9.5 seconds were inspected from a moved observer camera. These are desktop checks, not headset
performance evidence. Downloads used a verified warm cache; no cold-load benchmark is claimed.

The ground placement remains an approximation: its saved manifest labels it an offline
calibration candidate, with geometric foot proxies and a tilted ground plane. Soft or incomplete
feet, imperfect contact and motion/appearance differences remain. The brown-hoodie friend's
backpack is missing, the building sign is garbled, and unobserved areas are generated. The fixed
walk camera does not follow actors as they move; use the look and walking controls to keep them
in view.

Private screenshots, packet checks and motion comparisons are under
`.context/evidence/img5594/` in the integration checkout. No media or generated scene files are
included in Git.

