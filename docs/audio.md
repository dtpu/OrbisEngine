# Audio in the walk viewer

`fourd.html` owns sound for both the standalone viewer and the `demo.html` wrapper. Enable sound with the sound button, a first pointer gesture in the viewer, or Enter/Space. A later gesture preserves the user's mute choice. Enter still toggles playback once. The source inset can be hidden without losing sound. Opening the recorded reel pauses the world; closing it leaves the world paused.

The original mono/stereo soundtrack remains a non-positional mix. Independent voices become positional only when a package supplies reviewed mono speaker stems. The implementation never extracts identities or invents sound from a silent video.

The published elevator, lobby, stairs2, atrium, and TOS scene packages currently provide authentic original stereo mixes, not isolated voices. The HP wide-shot package explicitly declares no available soundtrack. These packages do not demonstrate per-character dialogue; that path is implemented and tested with synthetic audio, awaiting reviewed real stems.

## Packaging original sound

The viewer discovers `audio.json` next to `people.json`, or in the world directory above a `person` sequence. Presets using an alternate avatar directory must declare the original audio manifest path explicitly; the atrium preset does this for its alternate reconstruction. Override discovery with `?audio=/worlds/example-4d/audio.json`; `?audio=0` skips package discovery, retaining embedded video audio. The demo wrapper forwards this option and XR options.

```json
{
  "schema": "wander.audio/1",
  "source": { "hasAudio": false },
  "timeline": { "durationSeconds": 10 },
  "original": { "url": "audio/original.wav", "offsetSeconds": 0 },
  "defaultMode": "original"
}
```

`source.hasAudio` says whether the **packaged video** contains an audio stream. It remains false when authentic sound is recovered from an earlier source file into `original.wav`. The optional external `original` replaces embedded audio, never doubles it. After loading, a package with `hasAudio: false`, no decoded external original, and no usable spatial mix disables the sound button as “No soundtrack available.” Without a manifest, the button offers the embedded soundtrack if present; portable media APIs cannot guarantee track detection.

The timeline duration must match the viewer's people/sequence duration within 0.1 seconds. A track's sample zero occurs at `offsetSeconds` on the clip timeline, so sample time equals clip time minus offset. Trim original audio against the same original-video edit, retaining matching silence and verifying offsets. URLs resolve relative to `audio.json`. Additional provenance fields are allowed. Optional `provenance.attribution`, `provenance.license`, and `provenance.licenseUrl` show a visible audio excerpt credit while the original mix is selected; license links accept only HTTP(S) URLs. The recovered TOS package uses this credit for Blender Foundation's CC BY 3.0 source. Keep media and manifests in gitignored `public/worlds/<clip>-4d/` or `public/clips/audio/`; publish through the existing shared-assets snapshot only when authorized. PCM WAV and MP3 MIME entries already exist. No keys belong in manifests or client code. The attribution strip is currently shown only in original mode; it does not render per-stem credits or remain visible in spatial mode.

## Reviewed character dialogue

Extend the package with `spatialMixComplete: true` and `tracks`. Each concurrent character needs an independent mono signal. An isolated dialogue stem with non-overlapping turns can be prepared as separate speaker files offline, preserving timing and tails. A raw stereo channel or arbitrary time slice of a full mix is not proof of an isolated speaker.

```json
{
  "spatialMixComplete": true,
  "defaultMode": "spatial",
  "tracks": [
    {
      "id": "voice-a", "kind": "dialogue", "url": "audio/voice-a.wav",
      "offsetSeconds": 0, "personId": "person-1",
      "reviewed": true, "provenance": "supplied-stem",
      "anchor": {
        "url": "person-1/head.json", "space": "person-local",
        "offsetBodyHeights": [0, 0, 0]
      }
    },
    { "id": "bed", "kind": "ambience", "url": "audio/bed.wav", "offsetSeconds": 0 }
  ]
}
```

This fragment belongs inside the complete manifest above. Use actual `people.json` IDs; the single-person sequence fallback uses `person`. Supported track kinds are `dialogue` and `ambience`. Ambience goes directly to the output mix without position or head rotation. `spatialMixComplete` is a review assertion that the supplied tracks form the intended replacement mix; the software cannot verify that a bed is dialogue-free. Overlapping speakers play simultaneously through separate HRTF panners. The complete stem mix replaces the original; a missing/invalid stem or anchor keeps the original route instead. A malformed spatial section also preserves a valid original track. With no usable embedded or external original, selecting the fallback is silent.

Original mode is the default when `defaultMode` is omitted. `defaultMode: "spatial"` takes effect only after a complete package containing at least one reviewed dialogue track has loaded successfully. The mode button is hidden until then; its labels are “Sound: original” and “Dialogue: spatial.” The sound button separately controls enable/mute. These are browser-page controls, not an immersive controller menu: enable sound before entering VR. Load failures appear in the sound button tooltip and `audioState.error`.

Head sidecars supply `positions` (preferred when both exist) or `eyes` arrays of local XYZ triples and optional strictly increasing `times`. Without `times`, the person's sequence timestamps are used and lengths must match. The sampler interpolates these positions, applies the optional body-height offset in person-local axes, then the final person world matrix. Supply mouth positions where verified; otherwise describe these as head anchors. Review offsets and source tracks rather than treating eye positions as measured lips. The current sampler holds the first/last anchor outside its time range: do not approve a package with unlocated speech beyond that range. No bounding-box or speaker-identity guesses are made.

The listener follows the desktop camera's world pose. During XR rendering it uses the headset's center pose composed with the final locomotion rig before Three's stereo-frustum adjustment. Spatial distances use one shared body-height scale. Head turns and walking continue to update the listener when playback is paused. XR exit and hidden pages pause playback. Real headset interruptions, comfort, and perceived localization still require device testing.

## Transport and limits

Video remains the master clock. External buffers share a common Web Audio start time and are invalidated on pause, seeking, rate changes, loop wrap, and drift exceeding 80 ms. Stalled or paused video does not advance the scene's clock; explicitly absent/failed video retains silent animation fallback. Source video longer than the scene loops at the scene interval.

Spatial dialogue uses 1× speed. A different rate returns to the original mix on the next active playback update; the viewer API accepts rates from 0.25× through 4×. External-original buffers follow the selected rate with pitch change; embedded media follows browser pitch behavior. For deterministic frame-stepped export, call `play(false)` and `setMuted(true)` before `reelSet` or other programmatic seeks, then mux aligned audio offline; a programmatic seek does not itself disable embedded audio. Short clips are decoded fully in memory; long recordings need a streaming transport before deployment. Neither source separation, object-localized effects, acoustic occlusion, room reverb, nor pitch-preserving stem stretching is implemented. Runtime buffer decoding does not verify source-video edits or wording. The offline packager checks file format, duration, and head-timeline shape; human review must establish content and synchronization.

`window.wander` exposes `unlockAudio()`, `setMuted(boolean)`, `setAudioMode('original'|'spatial')`, `setPlaybackRate(rate)`, and `audioState` alongside existing `play` and `setTime`. Inspect `audioState.error`, `hasAudio`, `spatialReady`, `context`, `activeSources` (scheduled buffer nodes), and signed scheduling drift when diagnosing silence. Scheduling drift is not measured hardware output latency.

## Checks

- `bun test ./scripts/test-fourd-audio.ts`: schema/anchor rejection, overlap, shared scheduling, source movement, pause/seek/loop/rate cancellation, original fallback, silent clips, and mute persistence using deterministic graph doubles.
- `bun scripts/test-fourd-audio-browser.ts`: native Chromium OfflineAudioContext renders an owned mono tone through the production spatial graph and checks that head rotation reverses left/right ear dominance. It launches and closes its own headless Chrome, uses no server, and writes only bundled code under `.context`.
- `bun scripts/capture-audio-demo.ts`: integrated real-viewer checks at the existing local server, including actual preset IDs, S3 audio assets, output graph signal, Enter, mute, seek, loop, reel exclusion, silent clips, and unobscured source credit. Screenshots/reports default to gitignored `.context/evidence/audio`; `AUDIO_URL`, `AUDIO_CLIPS`, and `AUDIO_OUT` override the target. It creates and closes its own headless browsers. Its default cases require the published five original-mix packages plus silent HP; its S3 header assertions intentionally do not pass in local-assets mode.
- `bun run build`: TypeScript and all current viewer production entries, including `fourd.html` and `demo.html`. Validate playback and private assets through the actual viewer at the prescribed local port as well.

These checks establish graph behavior and rendered binaural direction for a synthetic fixture. They do not establish real-clip lip sync, correct speaker attribution, audible quality, Quest frame rate, or real-device output latency. Review actual audio/visual onset, overlapping words, loop edges, parent/iframe Enter, reel exclusion, clipping, and XR pose behavior before a public spatial-dialogue demonstration.

The design uses the documented [Web Audio media source](https://webaudio.github.io/web-audio-api/#MediaElementAudioSourceNode), [buffer scheduling](https://webaudio.github.io/web-audio-api/#AudioBufferSourceNode), and [Three.js XR camera](https://threejs.org/docs/pages/WebXRManager.html) APIs. Audio unlock follows [Chrome's official autoplay guidance](https://developer.chrome.com/blog/autoplay/).


## Preparing supplied speaker stems offline

The recovered originals are 48 kHz stereo mixes whose channels have no verified speaker identity. No separation or diarization stage is shipped in this repository. Moving the original stereo mix onto a character would move every other voice and environmental sound with it. Use authentic isolated dialogue inputs rather than treating stereo channels as speaker stems.

The shortest reliable next step is a user-supplied isolated mono WAV per character (or clean, manually reviewed dialogue-only turns padded to a full-length speaker track), plus a complete residual bed when needed. Overlapping voices require independently isolated signals. The packaging tool cannot remove competing voices, restore missing words, decide who is speaking, or certify the residual bed. Do not assert a human review solely because this validator passes.

`scripts/package_audio.py` uses only Python's standard library. It requires an existing world's `audio.json` with `source.hasAudio`, a positive duration, and `original.url` pointing to an existing relative local PCM WAV. The fallback must start at offset zero and match the clip duration within 50 ms. Embedded audio alone or a remote original URL does not meet this packaging contract. Supply a configuration file like:

```json
{
  "personIds": ["person-1", "person-2"],
  "tracks": [
    {
      "id": "voice-1", "kind": "dialogue", "file": "voice-1.wav",
      "personId": "person-1", "reviewed": true,
      "provenance": "supplied-isolated-dialogue",
      "anchorFile": "person-1-head.json",
      "offsetBodyHeights": [0, 0, 0]
    },
    {
      "id": "voice-2", "kind": "dialogue", "file": "voice-2.wav",
      "personId": "person-2", "reviewed": true,
      "anchorFile": "person-2-head.json"
    },
    { "id": "room-bed", "kind": "ambience", "file": "dialogue-free-bed.wav" }
  ]
}
```

Use the target viewer's actual person IDs; this list is an explicit declaration, not inferred identity. The CLI checks speaker membership against this caller-supplied list, not against `people.json`; the viewer separately validates IDs against its loaded cast. Each dialogue track needs a distinct person ID and `reviewed: true`. Input paths resolve relative to the configuration file. WAVs must be uncompressed 16-, 24-, or 32-bit PCM, use the original sample rate, and span the full normalized clip (within 50 ms); pad silence rather than shifting turns to sample zero. Speaker files must be mono; ambience may be mono or stereo. The packager always emits zero offsets and does not trim, resample, time-shift, or normalize input samples. Track IDs must be unique ASCII letters/digits/underscore/hyphen, from 1 to 64 characters. The ambience entry is optional only when the supplied voices already form the complete intended mix. Never use the full original as the bed; the tool rejects the identical original file, but cannot identify leaked speech or a re-encoded duplicate.

Anchors need `positions` (or `eyes`) and matching `times`: at least two samples, starting at zero, strictly increasing within the clip, and ending within `max(final sample interval, 1 ms) + 1 ms` of its end. An optional `sequenceFile` supplies `timestamps` when an existing head sidecar lacks times. Positions are in the same **person-local coordinates** as the active reconstruction, before the person's placement matrix. Mouth/head offsets use that person's body-height fraction in local axes; the viewer transforms positions into the shared world and uses a common body-height scale for attenuation. A head sidecar from a different avatar reconstruction needs review even if its track ID matches. The CLI validates array shape and timeline coverage, not physical accuracy, lip motion, or speaker identity.

```sh
uv run --locked scripts/package_audio.py --config .context/dialogue/config.json --world public/worlds/example-4d --reviewed --check
uv run --locked scripts/package_audio.py --config .context/dialogue/config.json --world public/worlds/example-4d --reviewed
uv run --locked scripts/tests/test_package_audio.py
```

Only pass `--reviewed` after listening to each speaker and the complete sum, checking overlapping speech, all words and tails, speaker-to-person identity, visible motion timing, and head-anchor alignment. The command refuses to write without this human-review gate. Validation failures preserve the existing manifest. Success copies exact PCM assets and normalized head sidecars into a content-addressed `audio-reviewed/` directory and atomically updates `audio.json`, retaining its original audio fallback and source provenance. It defaults to **original** mode for comparison; select spatial dialogue in the viewer to audition the reviewed package. No source media is synthesized, no model is downloaded, and no service or GPU job is launched. Publication remains a separate authorized step through shared assets.

The packaging tests generate owned tones in temporary directories and cover the review gate, exact sample preservation, fallback/provenance retention, idempotent output, rejected stereo dialogue, duration mismatches, unknown speakers, bad head timelines, duplicate original mix, and validation-only behavior. Actual prepared character dialogue still needs a listening and headset check; these tests do not manufacture usable stems.
