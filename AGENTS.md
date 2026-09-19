# Working on Wander

Read `README.md` first. Audio and asset contracts are in
`docs/audio.md` and `docs/shared-assets.md`.

- Measure before claiming success. Check the real viewer at `http://127.0.0.1:5399`;
  keep screenshots and measurements outside Git. Simulated XR is not headset evidence.
- Use Bun for viewer commands and `uv run --locked` for local Python tools. Keep `bun.lock` and
  `uv.lock` current. Run `bun run format:check` after Python edits; use ordinary readable blocks.
- At most two subagents may work at once. Give them independent tasks; do not poll them.
- Start Vite only with `bunx --bun vite --port 5399 --host 127.0.0.1`. Do not kill someone else's
  server, pipeline, or Modal job. `RECORD=1` disables reload during captures.
- Use `MODAL_PROFILE=dtpu` for GPU launches. Existing model volumes may belong to the primary
  profile; check availability before launching. Launch paid jobs only within the user-authorized task and budget.
- Marble: one generation credit per new clip after every no-spend gate passes; tell the user
  before spending. The pipeline's cleaned-frame review gate remains required.
- Never commit footage, generated scenes, model weights, run outputs, screenshots, credentials,
  or keys. Keep media in private shared storage. Do not paste secrets into chat or logs.
- Never use bare `git stash`; worktrees share it. Preserve other agents' changes.
- No per-clip constants in runtime code. Use measured manifests and general flags.
- Quote body-heights; metre estimates assume a subject height. Label unobserved geometry and
  invented appearance. Preserve recorded audio words and timing; do not invent speaker stems.
- Check results adversarially against source footage before calling them settled.
