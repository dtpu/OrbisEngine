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

For an offline asset bundle shared by the author, extract its existing `public/` structure from
the repository root and explicitly select local mode:

```sh
tar -xzf /path/to/wander-sample-elevator.tar.gz
WANDER_ASSETS_MODE=local bun run demo
```

Select only scenes installed by that bundle. No model downloads run during `bun install --frozen-lockfile`. If another
Vite server already owns port 5399, use it and coordinate changes instead of stopping it.

## Controls

- **Enter / Space:** play or pause.
- **W/A/S/D:** walk; drag to look around, or click for mouse look and **Esc** to release it.
- **Shift:** move faster. **R:** reset the view.
- **M (with the scene focused):** open the overhead map; click supported ground to choose a position.
- Use the sound controls to enable the original mix, mute it, or select reviewed spatial tracks
  when a scene provides them. Sound starts after an interaction. See [audio](docs/audio.md).

Walk mode is the default. `?walk=0` restores the authoring camera. The source clip, reconstruction,
and audio share one playback timeline. Recorded stereo mixes do not imply separated speakers.

## Quest

Enable Developer Mode and accept the headset's USB debugging prompt. With the demo running:

```sh
adb devices
adb reverse tcp:5399 tcp:5399
```

In Meta Browser open `http://localhost:5399/demo.html?xr=1`, start playback, and select **Enter VR**.
Default movement is teleport with snap turning; `?xr=1&xrmove=smooth` enables smooth locomotion.
Measure the physical headset's frame rate before demonstrating it. Simulated XR tests check code
paths only. The desktop viewer is the fallback when headset performance is inadequate.

## Code map

| Path                                         | Responsibility                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `demo.html`                                  | Scene picker, loading state, source comparison, and demo controls                  |
| `fourd.html`                                 | Three.js/Spark scene, video-master playback, people, objects, and grounded walking |
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
  --marble image --all-people --fps 12 --skip-finetune
```

The run stops at the cleaned-frame review gate before generating its Marble world. Inspect the
reported frames, then repeat the command with `--gate-pass` only after approving that input.
Unattended runs pass `--no-gate` to skip this stop.
`--marble none` avoids Marble generation but can still run paid GPU/API stages. `--no-publish`
keeps results local. Repeating a command resumes its saved stages; `--only` and `--force` select
stages explicitly. Run `uv run --locked --group inference scripts/run_clip.py --help` for all options.
Paid Modal stages retain their outputs and share a local execution ledger. See
[paid stage accounting and recovery](docs/paid-recovery.md) before retrying an interrupted run.

Optional fine-tuning requires an explicitly configured GPU host (`--gpu-box` or `WANDER_GPU_BOX`)
and an optional SSH identity (`--gpu-key` or `WANDER_GPU_KEY`); the remote host must already have
its trainer dependencies and `~/venv/bin/python`. No command here provisions a GPU host.

Output directories default to `.context/{share,clips,marble}` and `.context/run/<name>`; override
with `WANDER_SHARE_DIR`, `WANDER_CLIPS_DIR`, and `WANDER_MARBLE_DIR`. Use `WANDER_EVIDENCE_DIR`
when publishing an external evidence directory. Packaged viewer assets live under `public/` and
remain untracked. See the [code map](#code-map) and [object packaging](docs/objects.md).

For an unattended improvement run, fill [TONIGHT.md](docs/overnight/TONIGHT.md), then follow the
[overnight runbook](docs/overnight/RUNBOOK.md). It covers recovering source clips from S3,
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
```

Python checks and formatting need `uv`; viewer-only use needs just Bun.
Use `bun run format` to apply pinned Ruff and Prettier formatting; `format:python` and
`format:web` select one toolchain. Builds type-check the migrated TypeScript modules.
The formatted inline scripts in `demo.html` and `fourd.html` still use JavaScript.
The audio browser check needs installed Chrome. With the live demo running, use `bun run smoke:xr`
for a simulated XR smoke check and `bun run capture:shared /absolute/evidence/directory` for
S3-backed viewer captures. Walk collision captures accept `--out` for an evidence directory.
`bun run capture:audio` exercises the real demo's audio; set `AUDIO_OUT` for its evidence path.
Build output contains app code only, not public media. Use `bun run demo` for local testing with
the private asset middleware; the compiled files alone are not a complete deployment.

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
