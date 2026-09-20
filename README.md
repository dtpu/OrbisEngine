# Wander

Stand inside a processed video and walk around while its people and objects replay the recorded
action. The demo combines a generated environment, animated people, fitted objects, and original
audio. New viewpoints can expose generated content; the viewer is an observer and does not change
what happened.

## Run the demo

Use [Bun](https://bun.sh/docs/installation) 1.2.21+ and Chrome. Clone only `main` so the archived branches and their history are not downloaded:

```sh
git clone --no-tags --single-branch --branch main https://github.com/StockerMC/wander.git
cd wander
bun install --frozen-lockfile
# Save the read-only teammate credentials shared privately as .env.local.
bun run demo
```

Open **http://127.0.0.1:5399/demo.html**. The root URL redirects here. The local Bun server loads
scene assets from private S3 and caches them on demand; credentials never enter browser code.
The shared migration is complete. Restart Vite to adopt a newly published snapshot. See
[shared assets](docs/shared-assets.md) for credentials, publishing, and recovery.

The picker contains five scenes: the two-person lift lobby, Tears of Steel shot 31, the waving
lobby subject, Austin's accepted gym, and the latest kitchen fixture repair. Desktop and headset
menus share this selection. Gym and kitchen load their reviewed asset manifests, including
animated people, moving objects and walking colliders; their recorded limitations still apply.

For an offline asset bundle shared by the author, extract its existing `public/` structure from
the repository root and explicitly select local mode:

```sh
tar -xzf /path/to/wander-sample-elevator.tar.gz
WANDER_ASSETS_MODE=local bun run demo
```

Select only scenes installed by that bundle. No model downloads run during `bun install --frozen-lockfile`. If another
Vite server already owns port 5399, use it and coordinate changes instead of stopping it.

Austin’s accepted gym reconstruction is available in the [gym review](http://127.0.0.1:5399/reviews/gym-repair/index.html?quality=detail).
Use its **Walk around the bench** link for a single-scene view. The original gym preset remains
the comparison baseline. See [the gym delivery notes](docs/gym-delivery.md) for the accepted asset
version, retained limitations, and the older review page’s historical work-in-progress labels.

## Controls

- **Enter / Space:** play or pause.
- **W/A/S/D:** walk; drag to look around, or click for mouse look and **Esc** to release it.
- **Shift:** move faster. **R:** reset the view.
- **M (with the scene focused):** open the overhead map; click supported ground to choose a position.
- Use the sound controls to enable the original mix, mute it, or select reviewed spatial tracks
  when a scene provides them. Sound starts after an interaction. See [audio](docs/audio.md).

Walk mode is the default. `?walk=0` restores the authoring camera. The source clip, reconstruction,
and audio share one playback timeline. Recorded stereo mixes do not imply separated speakers.

## Interactive bottle prototype

The opt-in elevator prototype is available locally at
[fourd.html?demo=elevator&interact=1&xr=1](http://127.0.0.1:5399/fourd.html?demo=elevator&interact=1&xr=1).
It starts the recorded exchange normally in VR, then a deliberate close approach, directed
microphone speech, or a valid bottle grip pauses the recording for local bottle handling and an
in-scene fictional character with brief, natural dialogue. Enter VR also starts the OpenAI microphone connection,
subject to headset permission. Press **X on the left controller** to replay; there is no floating
control bar or text chat. Replay restores and plays the recorded state while retaining the
visitor's pose and the current connection's voice history. Neither a returned bottle nor VR re-entry
restarts or loops the recording. People automatically turn toward the visitor during conversation.
A bottle locator makes the recorded prop easier to find; an invented, clearer bottle model appears
when interacting and follows the visitor's grip when held.

Voice needs a private `OPENAI_API_KEY` on the local Vite server. The browser receives a short-lived
session secret; static build output cannot create one. Recorded bodies stay paused apart from an
automatic whole-body turn: there is no new walking, reach, eye, or lip animation. See
[interactive bottle prototype](docs/interactive-world.md) for manifest, head-anchor, and
verification limits.

## Quest

Enable Developer Mode and accept the headset's USB debugging prompt. With the demo running:

```sh
adb devices
adb reverse tcp:5399 tcp:5399
```

In Meta Browser open `http://localhost:5399/demo.html?xr=1`, start playback, and select **Enter VR**.
The desktop demo includes a small **Quest view** window showing the active headset's left-eye
scene. Open both devices through the same local demo server, then enter VR on the Quest. The
preview follows head and joystick movement, including the in-scene hands and body. It does not
include Meta system menus or audio. Close the window to stop receiving; capture pauses when
no visible window is watching. The local relay retains only the latest image in memory, with
no recording. The window fits the headset image's proportions; drag its top-left corner to
resize it. Capture and display target 60 frames per second at up to 512 pixels; the panel shows
the actual displayed rate. Speed depends on headset rendering, encoding and the connection, and
cannot exceed the headset's rendered frame rate. Capture overlaps the preceding upload, with
at most one following frame and no accumulating queue. `xrview=0` disables sharing from the headset.
Default movement is teleport with snap turning. `?xr=1&xrmove=smooth` enables smooth walking
with the left joystick and continuous turning with the right joystick. Right-stick turning has a
15% deadzone, stops on release, and turns at 90 degrees/second at full deflection. Add
`&xrturnspeed=60` to adjust that rate (0–180 degrees/second), or `&xrturnmode=snap` to keep
snap turning while walking smoothly. `&xrturn=0` disables joystick turning in either mode.
Left-stick walking follows the direction you face, with analog speed. Physical leaning, walking
and crouching remain tracked. For a small play space, `?xr=1&xrmove=smooth&xrwalkgain=1.5` makes horizontal
physical steps cover up to 50% more scene distance. `xrwalkgain` defaults to 1 and accepts 1–2;
only the extra travel is limited by the scene boundary and, in walk mode, obstacle checks.
Physical head tracking remains unrestricted; the gain does not magnify head rotation, eye height
or crouching. Re-enter VR after changing these options.
VR shows illustrative gloves at your tracked controller poses, with finger curls driven by the
trigger and grip buttons. When the headset supplies hand tracking, the gloves follow its finger
joints. Their appearance is invented; tracking does not reconstruct your real skin or clothing.
Use `xrhands=0` to hide them.
A simple torso, arms, legs and shoes provide a first-person body when you look down. The arms
reach toward tracked hands/controllers; the torso and walking steps are estimated from head pose
and movement. This is an illustrative avatar, not measured full-body or foot tracking. `xrbody=0`
hides the body while keeping hands available.
Entering VR starts playback, which loops until paused; exiting VR pauses it. With `walk=1`,
smooth joystick movement uses the desktop floor and obstacle checks, including known stair heights.
Default teleport mode aims at supported, unblocked floor cells and moves the rig and avatar to
the selected height. Both modes follow supported floor beneath the head; unknown geometry cannot
be selected as a teleport landing. Exiting VR restores the saved desktop camera and floor.
`personsize=0.9` makes recorded people 10% smaller around their moving foot anchor without changing
the scene's scale or your eye height. The default is 1. This is a visual adjustment, not a new
measurement of the recorded person's height or a repair of missing reconstructed geometry.
Measure the physical headset's frame rate before demonstrating it. Simulated XR tests check code
paths only. The desktop viewer is the fallback when headset performance is inadequate.

While in VR, press **B on the right controller** to open the scene sidebar where you point.
It stays anchored there while you browse. It uses
the same thumbnails and clip names as the desktop rail, including the spare scenes. Move either
joystick up/down to browse, then press **A** to load the highlighted clip and its environment.
Pointing at a row and pressing A or pulling the trigger also selects it.
Press B again or point at the close button and pull the trigger to resume the current scene.
Playback and artificial movement pause while browsing; your actual head movement remains tracked.
Switching clips keeps the same VR session. The current scene stays visible while the next loads;
choose another clip to cancel a slow load, or choose the current clip to return to it.
A failed load restores the previous scene and shows a retry message in the sidebar. Closing the
message restores the playback state from before browsing. Release a held joystick after opening
the sidebar before using it to browse.
The previous scene stays prepared in memory for quick return visits. Loading a third scene
replaces that spare before allocating another, so at most two scenes are resident. First visits
still download or read cached assets and prepare their 3D data; the cache clears on page reload.
To speed up first visits, [prepare the worlds on the Mac](docs/shared-assets.md#prepared-world-cache)
once. The viewer then loads the same packed splats and detail tree without rebuilding them on the
Quest. These disk caches survive page reloads; people and video still need to load.

## Code map

| Path                                         | Responsibility                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `demo.html`                                  | Scene picker, loading state, source comparison, and demo controls                  |
| `fourd.html`                                 | Three.js/Spark scene, video-master playback, people, objects, and grounded walking |
| `src/fourd-session.js`, `src/fourd-runtime.js` | Persistent renderer and disposable scene loading without ending the VR session |
| `src/scene-catalog.ts`                       | Shared desktop and VR clip list |
| `src/video-projection.ts`, `src/walk-map.ts` | Recorded-image projection and the overhead position picker                         |
| `src/xr/`, `src/audio/`                      | Headset locomotion and synchronized, position-aware audio                          |
| `server/shared-assets.ts`                    | Private S3 delivery with pinned snapshots, verified caching, and byte ranges       |
| `scripts/run_clip.py`, `worker/stages/`      | Resumable reconstruction and its active worker implementations                     |

`window.wander` exposes transport and diagnostic state to the wrapper and capture scripts.
The source video owns the timeline; the XR runtime owns head motion. Scene manifests and media
stay outside Git. Viewer installation and builds do not launch inference or download model weights.

## Process a clip

`scripts/run_clip.py` is the current end-to-end pipeline. It checks shot continuity, runs camera,
people, and world stages, fits placement, packages assets, and publishes results to private S3.
Automatic completion archives outputs without changing the shared viewer. Explicit author
publication follows separate visual acceptance.
This is author tooling: it needs [uv](https://docs.astral.sh/uv/getting-started/installation/),
Python 3.11–3.12, FFmpeg, local inference dependencies, configured
Modal access and model caches, Marble credentials, and an OpenAI key for visual review. GPU/model
setup is explicit and separate from installing the viewer. `uv` manages the root `.venv` using
`pyproject.toml` and `uv.lock`; `uv sync --locked` installs local tools, and the `inference` group
adds local model libraries. Model weights are downloaded only when inference runs.

### Author credentials

For a teammate running reconstruction, follow [author credential setup](docs/shared-assets.md#author-credentials):
join the shared Modal workspace, use profile `dtpu`, and configure `WLT_API_KEY`, `OPENAI_API_KEY`,
and AWS publishing access. `.env.example` lists the variables. Keep author secrets in private
`.env.author` and viewer read-only credentials in `.env.local`; Python needs the author file
explicitly exported into its launching shell.

```sh
uv sync --locked --group inference
# Complete the linked credential setup and confirm model-cache access first.
set -a
source .env.author
set +a
uv run --locked --group inference scripts/run_clip.py --clip /path/to/source.mp4 --name example \
  --marble video --all-people --fps 12 --skip-finetune
```

Video is the default Marble input. It submits the cleaned clip at the configured processing frame
rate with the source description, a pinned model and private permissions. `--marble image` and
`--marble multi` are explicit still-image alternatives; camera spread alone does not select them.
This policy preserves more temporal input, but reduced hallucination has not been established by a
controlled comparison. See [the kitchen review](docs/kitchen-review.md#video-first-policy) and
[live kitchen/gym results](docs/video-input-evaluation.md). The real video path works, but the new
candidates were not promoted; prompted video now explicitly disables automatic recaptioning.

The run stops at the cleaned-frame review gate before generating its Marble world. Inspect the
reported frames, then repeat the command with `--gate-pass` only after approving that input.
Unattended runs pass `--no-gate` to skip this stop.
`--marble none` avoids Marble generation but can still run paid GPU/API stages. `--no-publish`
keeps results local. Repeating a command resumes its saved stages; `--only` and `--force` select
stages explicitly. Run `uv run --locked --group inference scripts/run_clip.py --help` for all options.
The world `verify` stage now requires an explicit offline manual review of its static room evidence.
Without one it retains CPU-rendered diagnostics and stops as blocked. Legacy vision scores cannot
accept a world. See [quality judging](docs/quality-judging.md) for the review contract and remaining
calibration limits; static-world review alone does not certify people, motion or walking.
Paid Modal stages retain their outputs and share a local execution ledger. See
[paid stage accounting and recovery](docs/paid-recovery.md) before retrying an interrupted run.

Optional fine-tuning requires an explicitly configured GPU host (`--gpu-box` or `WANDER_GPU_BOX`)
and an optional SSH identity (`--gpu-key` or `WANDER_GPU_KEY`); the remote host must already have
its trainer dependencies and `~/venv/bin/python`. No command here provisions a GPU host.

Output directories default to `.context/{share,clips,marble}` and `.context/run/<name>`; override
with `WANDER_SHARE_DIR`, `WANDER_CLIPS_DIR`, and `WANDER_MARBLE_DIR`. Use `WANDER_EVIDENCE_DIR`
when publishing an external evidence directory. Packaged viewer assets live under `public/` and
remain untracked. See the [code map](#code-map) and [object packaging](docs/objects.md).

For an unattended improvement run, write the run brief and follow
[unattended runs](docs/unattended-runs.md). It covers recovering source clips from S3,
checking caches, budgets and model selection, isolated candidate runs, visual acceptance,
and recovery without duplicate Marble generations.

## Checks

```sh
bun run build
bun run format:check
bun run test:shared-assets
bun run test:pull-assets
bun run test:publish-hook
bun run test:marble
bun run test:audio
bun run test:audio-browser
bun run test:audio-package
bun run test:person-motion
bun run test:static-colliders
bun run test:walk-collision
bun run test:walk-clearance-browser
bun run test:xr-turning
```

Python checks and formatting need `uv`; viewer-only use needs just Bun.
The walk-clearance browser check needs Chrome, the local server, and the shared kitchen assets.
It exercises the reported kitchen corner with keyboard movement and checks that the adjacent
counter still blocks. After building, `WALK_TEST_DIST=dist bun run test:walk-clearance-browser`
uses this checkout's compiled viewer with the running server's scene assets.
Use `bun run format` to apply pinned Ruff and Prettier formatting; `format:python` and
`format:web` select one toolchain. Builds type-check the migrated TypeScript modules.
The formatted inline scripts in `demo.html` and `fourd.html` still use JavaScript.
The audio browser check needs installed Chrome. With the live demo running, use `bun run smoke:xr`
for a simulated XR smoke check and `bun run capture:shared /absolute/evidence/directory` for
S3-backed viewer captures. Walk collision captures accept `--out` for an evidence directory.
`bun run test:xr` runs all XR and person-size unit tests without Chrome or a server.
With Chrome installed and that server running, `bun run test:xr-browser` runs the simulated XR
browser suite: physical gain and wall checks, smooth/snap turning, tracked height, re-entry,
teleporting between levels, stair support, body stepping/crouching, playback controls and runtime person scaling.
`bun run test:xr-locomotion` retains the focused movement check; `bun run test:person-size-browser`
runs the authoring-scale regression alone. Simulated captures stay under `.context/evidence/`.
The turning browser check also needs that server. After building, run
`XR_TEST_DIST=dist bun run test:xr-turning` to test this checkout's compiled code
while using the running server for scene assets. Evidence stays under `.context/evidence/`.
`bun test ./scripts/test-xr-scene-sidebar.ts` checks sidebar input edges, ray targets, scrolling,
and bounded thumbnail loading. `bun test ./scripts/test-scene-session.js` checks cancellation,
playback restoration, activation rollback, cache bounds and disposal. `bun scripts/test-fourd-startup-browser.ts`
checks startup failures and retry using isolated browser fixtures.
`XR_SIDEBAR_URL=http://127.0.0.1:5399 bun scripts/test-xr-scene-sidebar-browser.ts`
checks scene switching, preserved XR sessions, paused movement, and failed-load recovery with
synthetic XR against the running viewer. These are not physical-headset measurements.

`bun test ./scripts/test-quest-view.ts` checks the spectator relay's ownership, expiry and upload
limits. `bun scripts/test-quest-view-panel-browser.ts` checks the preview window's connection and
visibility states. After building, `bun scripts/test-quest-view-browser.ts` checks real rendered
pixels through the encoder, relay and panel with synthetic XR input on an isolated local server.
Native Quest capture still needs a headset check; main-thread canvas encoding can stall during
immersive sessions, so JPEG encoding runs in a dedicated worker.
`bun run capture:audio` exercises the real demo's audio; set `AUDIO_OUT` for its evidence path.
Compact animation is lossless by default; see [person motion](docs/person-motion.md) for integrity
checks, original-PLY fallback and explicit quantization opt-in. The [kitchen review](docs/kitchen-review.md)
separates input, generation, export, rendering and registration findings.
Build output contains app code only, not public media. Use `bun run demo` for local testing with
the private asset middleware; the compiled files alone are not a complete deployment.

The [pipeline walkthrough plan](docs/pipeline-walkthrough.md) describes a simple visual presentation
of each stage and its saved outputs for judges. It is a plan, not an implemented viewer feature.

## Repository scope

`main` contains the active viewer, its pipeline dependency closure, tests, and current docs.
Earlier viewers, experiments, the historical handoff, and investigation records are preserved on
`archived-main`.
History has been consolidated; save any uncommitted work and use a fresh clone instead of pulling
the rewritten history into an old checkout. Date labels were removed from tracked text, paths, and
commit messages; Git retains its required internal author/committer timestamps.
Use a clean single-branch clone of `main` for future public distribution; do not publish the
archive branch, private bundles, caches, credentials, or old local checkout contents with it.
An asset's availability in private storage does not grant permission to redistribute it.

Tears of Steel credit: **(CC) Blender Foundation | [mango.blender.org](https://mango.blender.org/)**,
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The demo uses trimmed/transcoded excerpts.
Phone clips are private supplied footage. Active presets use these source excerpts:

| Preset     | Source         | Excerpt                                                     |
| ---------- | -------------- | ----------------------------------------------------------- |
| `elevator` | `IMG_2876.MOV` | First 10 seconds                                            |
| `lobby`    | `IMG_5410.MOV` | Full take, sampled at 30 fps                                |
| `stairs2`  | `IMG_2877.MOV` | Full 4.1-second take                                        |
| `atrium`   | `IMG_2879.MOV` | 1.5–6.9 seconds                                             |
| `tos31`    | Tears of Steel | Shot near 158.7 seconds, with its first 3.5 seconds removed |

Tears of Steel's official [download](https://mango.blender.org/download/) and
[sharing](https://mango.blender.org/sharing/) pages provide its source and attribution terms.
Audio manifests record exact source hashes and sample offsets. Other direct presets, including
Harry Potter excerpts, are supplied assets outside the five-scene picker; private storage access
does not grant redistribution rights.

Contributors: **Aayan Karmali (StockerMC), Austin Jian, and Daniel Pu**. The archive preserves the
original development history and human authorship. To move this lean app to a future public repo,
start from the single-branch clone above and push **main only** to that separate remote.
