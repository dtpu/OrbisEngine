# Audio in the walk viewer

`fourd.html` owns sound for both the standalone viewer and the `demo.html` wrapper. Enable sound with the sound button, a first pointer gesture in the viewer, or Enter/Space. A later gesture preserves the user's mute choice. Enter still toggles playback once. The source inset can be hidden without losing sound. Opening the recorded reel pauses the world; closing it leaves the world paused.

The original mono/stereo soundtrack remains a non-positional mix. Independent voices become positional only when a package supplies reviewed mono speaker stems. The implementation never extracts identities or invents sound from a silent video.

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

`source.hasAudio` says whether the **packaged video** contains an audio stream. It remains false when authentic sound is recovered from an earlier source file into `original.wav`. The optional external `original` replaces embedded audio, never doubles it. A package with `hasAudio: false` and no playable external audio displays “No soundtrack available.” Without a manifest, the button offers the embedded soundtrack if present; portable media APIs cannot guarantee track detection.

The timeline duration must match the viewer's people/sequence duration within 0.1 seconds. A track's sample zero occurs at `offsetSeconds` on the clip timeline, so sample time equals clip time minus offset. Trim original audio against the same original-video edit, retaining matching silence and verifying offsets. URLs resolve relative to `audio.json`. Additional provenance fields are allowed. Optional `provenance.attribution`, `provenance.license`, and `provenance.licenseUrl` show a visible audio excerpt credit while the original mix is selected; license links accept only HTTP(S) URLs. The recovered TOS package uses this credit for Blender Foundation's CC BY 3.0 source. Keep media and manifests in gitignored `public/worlds/<clip>-4d/` or `public/clips/audio/`; publish through the existing shared-assets snapshot only when authorized. PCM WAV and MP3 MIME entries already exist. No keys belong in manifests or client code.

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

This fragment belongs inside the complete manifest above. Use actual `people.json` IDs. `spatialMixComplete` is a review assertion that the supplied tracks form the intended replacement mix; the software cannot verify that a bed is dialogue-free. Overlapping speakers play simultaneously through separate HRTF panners. The complete stem mix replaces the original; a missing/invalid stem or anchor keeps the original route instead. A malformed spatial section also preserves a valid original track.

Head sidecars supply `positions` (or `eyes`) arrays of local XYZ triples and optional strictly increasing `times`. Without `times`, the person's sequence timestamps are used and lengths must match. The sampler interpolates these positions, applies the optional body-height offset in person-local axes, then the final person world matrix. Supply mouth positions where verified; otherwise describe these as head anchors. Review offsets and source tracks rather than treating eye positions as measured lips. The current sampler holds the first/last anchor outside its time range: do not approve a package with unlocated speech beyond that range. No bounding-box or speaker-identity guesses are made.

The listener follows the desktop camera's world pose. During XR rendering it uses the headset's center pose composed with the final locomotion rig before Three's stereo-frustum adjustment. Spatial distances use one shared body-height scale. Head turns and walking continue to update the listener when playback is paused. XR exit and hidden pages pause playback. Real headset interruptions, comfort, and perceived localization still require device testing.

## Transport and limits

Video remains the master clock. External buffers share a common Web Audio start time and are invalidated on pause, seeking, rate changes, loop wrap, and drift exceeding 80 ms. Stalled or paused video does not advance the scene's clock; explicitly absent/failed video retains silent animation fallback. Source video longer than the scene loops at the scene interval.

Spatial dialogue uses 1× speed. Changing rate returns to the original mix. External-original buffers follow the selected rate with pitch change; embedded media follows browser pitch behavior. Frame-stepped export should remain silent and receive aligned audio when muxing offline. Short clips are decoded fully in memory; long recordings need a streaming transport before deployment. Neither source separation, acoustic occlusion, room reverb, nor pitch-preserving stem stretching is implemented.

`window.wander` exposes `unlockAudio()`, `setMuted(boolean)`, `setAudioMode('original'|'spatial')`, `setPlaybackRate(rate)`, and `audioState` alongside existing `play` and `setTime`. Inspect `audioState.error`, `hasAudio`, `spatialReady`, `context`, active source count, and signed scheduling drift when diagnosing silence. Scheduling drift is not measured hardware output latency.

## Checks

- `node scripts/test-fourd-audio.mjs`: schema/anchor rejection, overlap, shared scheduling, source movement, pause/seek/loop/rate cancellation, original fallback, silent clips, and mute persistence using deterministic graph doubles.
- `node scripts/test-fourd-audio-browser.mjs`: native Chromium OfflineAudioContext renders an owned mono tone through the production spatial graph and checks that head rotation reverses left/right ear dominance. It launches and closes its own headless Chrome, uses no server, and writes only bundled code under `.context`.
- `node scripts/capture-audio-demo.mjs`: integrated real-viewer checks at the existing local server, including actual preset IDs, S3 audio assets, output graph signal, Enter, mute, seek, loop, reel exclusion, silent clips, and unobscured source credit. Screenshots/reports default to gitignored `.context/evidence/audio`; `AUDIO_URL`, `AUDIO_CLIPS`, and `AUDIO_OUT` override the target. It creates and closes its own headless browsers.
- `npm run build`: TypeScript and existing production entries. Current Vite production entry configuration does not include `fourd.html`/`demo.html`; validate these through the actual viewer at the prescribed local port after integration.

These checks establish graph behavior and rendered binaural direction for a synthetic fixture. They do not establish real-clip lip sync, correct speaker attribution, audible quality, Quest frame rate, or real-device output latency. Review actual audio/visual onset, overlapping words, loop edges, parent/iframe Enter, reel exclusion, clipping, and XR pose behavior before a public spatial-dialogue demonstration.

The design uses the documented [Web Audio media source](https://webaudio.github.io/web-audio-api/#MediaElementAudioSourceNode), [buffer scheduling](https://webaudio.github.io/web-audio-api/#AudioBufferSourceNode), and [Three.js XR camera](https://threejs.org/docs/pages/WebXRManager.html) APIs. Audio unlock follows [Chrome's official autoplay guidance](https://developer.chrome.com/blog/autoplay/).


## Preparing supplied speaker stems offline

The recovered elevator, lobby, stairs2, atrium, and TOS originals are 48 kHz stereo mixes with no separate speaker tracks. Their channels have no verified speaker identity. The inspected local Python environment has no installed speaker-separation package, and there is no existing separation/diarization pipeline in this repository. Moving the original stereo mix onto a character would falsely move every other voice and environmental sound with it. The available street-ambience asset is unrelated to these recordings.

The shortest reliable next step is a user-supplied isolated mono WAV per character (or clean, manually reviewed dialogue-only turns padded to a full-length speaker track), plus a complete residual bed when needed. Overlapping voices require independently isolated signals. The packaging tool cannot remove competing voices, restore missing words, decide who is speaking, or certify the residual bed. Do not assert a human review solely because this validator passes.

`scripts/package_audio.py` uses only Python's standard library. It accepts an existing world's original-fallback `audio.json` and a configuration file like:

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

Use the target viewer's actual person IDs; this list is an explicit mapping, not inferred identity. Input paths resolve relative to the configuration file. WAVs must be uncompressed PCM, use the original sample rate, and span the full normalized clip (within 50 ms); pad silence rather than shifting turns to sample zero. Speaker files must be mono. The ambience entry is optional only when the supplied voices already form the complete intended mix. Never use the full original as the bed; the tool rejects the identical original file, but cannot identify leaked speech or a re-encoded duplicate.

Anchors need `positions` (or `eyes`) and matching `times`, covering the full clip through the final sampled frame. An optional `sequenceFile` supplies `timestamps` when an existing head sidecar lacks times. Positions are in the same **person-local coordinates** as the active reconstruction, before the person's placement matrix. Mouth/head offsets use that person's body-height fraction in local axes; the viewer transforms positions into the shared world and uses a common body-height scale for attenuation. A head sidecar from a different avatar reconstruction needs review even if its track ID matches. The CLI validates shape/timing, not physical accuracy.

```sh
python3 scripts/package_audio.py --config .context/dialogue/config.json --world public/worlds/example-4d --reviewed --check
python3 scripts/package_audio.py --config .context/dialogue/config.json --world public/worlds/example-4d --reviewed
python3 scripts/test_package_audio.py
```

Only pass `--reviewed` after listening to each speaker and the complete sum, checking overlapping speech, all words and tails, speaker-to-person identity, visible motion timing, and head-anchor alignment. The command refuses to write without this human-review gate. Validation failures preserve the existing manifest. Success copies exact PCM assets and normalized head sidecars into a content-addressed `audio-reviewed/` directory and atomically updates `audio.json`, retaining its original audio fallback and source provenance. It defaults to **original** mode for comparison; select spatial dialogue in the viewer to audition the reviewed package. No source media is synthesized, no model is downloaded, and no service or GPU job is launched. Publication remains a separate authorized step through shared assets.

The packaging tests generate owned tones in temporary directories and cover the review gate, exact sample preservation, fallback/provenance retention, idempotent output, rejected stereo dialogue, duration mismatches, unknown speakers, bad head timelines, duplicate original mix, and validation-only behavior. Actual prepared character dialogue still needs a listening and headset check; these tests do not manufacture usable stems.
