# Wander

Stand inside a processed video and walk around while its people and objects replay the recorded
action. The demo combines a generated environment, animated people, fitted objects, and original
audio. New viewpoints can expose generated content; the viewer is an observer and does not change
what happened.

## Run the demo

Use Node 22.12+ or Node 24 and Chrome. Clone only `main` so the archived branches and their history are not downloaded:

```sh
git clone --no-tags --single-branch --branch main https://github.com/StockerMC/wander.git
cd wander
npm ci
# Save the read-only teammate credentials shared privately as .env.local.
npm run demo
```

Open **http://127.0.0.1:5399/demo.html**. The root URL redirects here. The local Node server loads
scene assets from private S3 and caches them on demand; credentials never enter browser code.
The shared migration is complete. Restart Vite to adopt a newly published snapshot. See
[shared assets](docs/shared-assets.md) for credentials, publishing, and recovery.

For an offline asset bundle shared by the author, extract its existing `public/` structure from
the repository root and explicitly select local mode:

```sh
tar -xzf /path/to/wander-sample-elevator.tar.gz
WANDER_ASSETS_MODE=local npm run demo
```

Select only scenes installed by that bundle. No model downloads run during `npm ci`. If another
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

## Process a clip

`scripts/run_clip.py` is the current end-to-end pipeline. It checks shot continuity, runs camera,
people, and world stages, fits placement, packages assets, and publishes results to private S3.
This is author tooling: it needs Python 3.11+, FFmpeg, local inference dependencies, configured
Modal access and model caches, Marble credentials, and an OpenAI key for visual review. GPU/model
setup is explicit and separate from installing the viewer.

```sh
python3 -m venv worker/.venv-da3
worker/.venv-da3/bin/pip install -r worker/requirements-pipeline.txt
# Configure Modal credentials and the project's existing model caches before processing.
export MODAL_PROFILE=dtpu
export WANDER_PYTHON="$PWD/worker/.venv-da3/bin/python"
export WANDER_MODAL="$PWD/worker/.venv-da3/bin/modal"
python3 scripts/run_clip.py --clip /path/to/source.mp4 --name example \
  --marble image --all-people --fps 12 --skip-finetune
```

The run stops at the cleaned-frame review gate before generating its Marble world. Inspect the
reported frames, then repeat the command with `--gate-pass` only after approving that input.
`--marble none` avoids Marble generation but can still run paid GPU/API stages. `--no-publish`
keeps results local. Repeating a command resumes its saved stages; `--only` and `--force` select
stages explicitly. Run `python3 scripts/run_clip.py --help` for all options.

Optional fine-tuning requires an explicitly configured GPU host (`--gpu-box` or `WANDER_GPU_BOX`)
and an optional SSH identity (`--gpu-key` or `WANDER_GPU_KEY`); the remote host must already have
its trainer dependencies and `~/venv/bin/python`. No command here provisions a GPU host.

Output directories default to `.context/{share,clips,marble}` and `.context/run/<name>`; override
with `WANDER_SHARE_DIR`, `WANDER_CLIPS_DIR`, and `WANDER_MARBLE_DIR`. Use `WANDER_EVIDENCE_DIR`
when publishing an external evidence directory. Packaged viewer assets live under `public/` and
remain untracked. See [architecture](docs/architecture.md) and [object packaging](docs/objects.md).

## Checks

```sh
npm run build
npm run test:shared-assets
npm run test:publish-hook
npm run test:audio
npm run test:audio-browser
npm run test:audio-package
```

The audio browser check needs installed Chrome. With the live demo running, use `npm run smoke:xr`
for a simulated XR smoke check and `npm run capture:shared -- /absolute/evidence/directory` for
S3-backed viewer captures. Walk collision captures accept `--out` for an evidence directory.
`npm run capture:audio` exercises the real demo's audio; set `AUDIO_OUT` for its evidence path.
Build output contains app code only, not public media. Use `npm run demo` for local testing with
the private asset middleware; the compiled files alone are not a complete deployment.

## Repository scope

`main` contains the active viewer, its pipeline dependency closure, tests, and current docs.
Earlier viewers, experiments, and detailed investigation records are preserved on `archived-main`.
History has been consolidated; save any uncommitted work and use a fresh clone instead of pulling
the rewritten history into an old checkout. Date labels were removed from tracked text, paths, and
commit messages; Git retains its required internal author/committer timestamps.
Use a clean single-branch clone of `main` for future public distribution; do not publish the
archive branch, private bundles, caches, credentials, or old local checkout contents with it.
An asset's availability in private storage does not grant permission to redistribute it.

Tears of Steel credit: **(CC) Blender Foundation | [mango.blender.org](https://mango.blender.org/)**,
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The demo uses trimmed/transcoded excerpts.
Phone clips are private supplied footage. Source records are in [assets/CLIPS.md](assets/CLIPS.md).

Contributors: **Aayan Karmali (StockerMC), Austin Jian, and Daniel Pu**. The archive preserves the
original development history and human authorship. To move this lean app to a future public repo,
start from the single-branch clone above and push **main only** to that separate remote.
