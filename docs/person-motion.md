# Compact person motion

The default package is **lossless float32**: every recorded frame, splat centre and rotation is
preserved bit-for-bit. The first original PLY supplies colour, opacity, scale and other static
attributes. Packing refuses a sequence if any static attribute changes, any property layout or
splat count differs, or any value is nonfinite. This is a delivery optimization of the existing
reconstruction; it does not improve geometry or registration.

Both `package_person_sequence.py` and `package_multiperson.py` add the lossless track when they
package a run. To add or refresh one without inference or publication:

```sh
uv run --locked scripts/package_person_motion.py public/worlds/<clip>-4d/person
```

Original PLYs remain available. Frame names/order, fps, exact timestamps, duration, source offsets
and other sequence metadata are retained. The packer refreshes `frame_sha256` from the original
PLYs and adds `motion`. A refused repack removes any stale motion descriptor before reporting the
error. It does not remove or rewrite PLYs. Packing and publishing must run after all frame writers
have stopped.

## Format and validation

`sequence.json.motion.schema` is `wander.person-motion/1`. The record contains:

- `file`, `dtype` (`float32` by default), `bytes`, and payload `sha256`.
- `frames`, `splats`, ordered `frameFiles`, and ordered `channels`:
  `x,y,z,rot_0,rot_1,rot_2,rot_3` (rotation is w,x,y,z).
- `base.file`, `base.bytes`, and `base.sha256` binding the track to its exact appearance PLY.
- `min`, `scale` and measured `maxAbsError` per channel; all errors are zero for float32.

Payload order is frame, splat, channel, with little-endian scalar values. Float32 stores the
original values directly. The browser validates schema, dimensions, frame order and byte length,
verifies both SHA-256 hashes, then decodes the complete track before installing any compact keys.
It preserves the normal PLY playback timing and interpolation. Stance stabilization derives complete
predecessor chains, so seek order and frame arrival speed cannot permanently cache incomplete
corrections. Existing viewer flags for explicitly
selecting fewer splats continue to work; compact packaging itself never reduces splats or frames.

Missing, corrupt, mismatched and older unbound motion records fall back to the original PLYs.
The reason is logged and exposed as `wander.people[i].motionFallback`. The interpolated PLY path verifies frame hashes
when `frame_sha256` is present; invalid PLY bodies, mismatched layouts/counts or missing
required frames cause a load error. A failed fallback sets `wander.people[i].loadError`, finishes
its loading attempt (`settled`), and never reports ready. The demo wrapper displays the failure
and offers reload. `?motiontrack=0` forces the original PLY path for comparisons.

## Optional quantization

`--quantize` explicitly opts into lossy uint16 storage. The browser also requires
`?motionquantize=1` before using a uint16 track; otherwise it loads the original PLYs. This flag
is for direct viewer review paths. Quantization is never inferred from asset metadata or enabled
by the pipeline. `--lossless` remains an accepted explicit spelling of the default.

For uint16, `value = code * scale[channel] + min[channel]`, rounded to float32 by the viewer.
`maxAbsError` measures that final float32 error. Existing historical byte/speed measurements for
quantized packages do not establish sizes or performance for the new lossless default.

## Checks and credits

Run `bun run test:person-motion` for Python packaging checks and real Python-to-TypeScript
round trips, including zero-error float32, static appearance rejection, preserved timing,
SHA-256 failures, wrong keyframes, reordered frames, missing payloads, explicit quantization
failure propagation when original frames also fail, and stance invariance under seeks/delayed frames. Tests need no media, network, GPU,
compression streams or provider credentials, and run on Bun 1.2.21.

The packer, decoder, pipeline hooks, PLY parser and round-trip tests were selected from James's
branch commits `d08706d`, `4d3c95b`, `6a34c12`, `265397f`, `6f18244`, `57c8954`, and `5f020e9`.
Their original author Claude and co-author Claude Fable 5.1 remain credited in the ported commits.
Codex adapted the integration for lossless defaults, format identity, integrity verification and
safe fallback. Live rendered comparisons and actual network timing remain separate acceptance
checks; unit tests do not establish visual quality or headset performance.
