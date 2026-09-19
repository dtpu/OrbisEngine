# Working on Wander

This file is durable: rules that hold for every session. Do not edit it or record progress here.
Read `README.md` for context and commands.
Before changing an area, read its doc: `docs/audio.md`, `docs/objects.md`, or `docs/shared-assets.md`.
Check `docs/known-limits.md` before trying a new approach.
When running unattended, also follow `docs/overnight/RULES.md`.

## Never

- Commit footage, generated scenes, weights, run outputs, screenshots, credentials, or keys; media lives in private shared storage.
- Paste secrets into chat or logs.
- Use bare `git stash` (worktrees share it), rewrite pushed history, or discard other agents' changes.
- Kill a Vite server, pipeline run, or Modal job you did not start.
- Skip, weaken, or delete a test to get to green.
- Add per-clip constants to runtime code; use measured manifests and general flags.

## Tooling

- Bun for viewer commands, `uv run --locked` for Python tools. Keep `bun.lock` and `uv.lock` current.
- Run `bun run format:check` after code edits (Ruff for Python, Prettier for web code); use ordinary readable blocks.
- Start Vite only with `bunx --bun vite --port 5399 --host 127.0.0.1`. `RECORD=1` disables reload during captures.
- At most two subagents at once, on independent tasks; do not poll them.

## Evidence

- Exit codes and passing tests do not prove visual quality. Check the real viewer at `http://127.0.0.1:5399` against the source footage, adversarially.
- Keep screenshots and measurements under `.context/evidence/`, never in Git. Simulated XR is not headset evidence.
- Quote body-heights, not metres (metres assume a 1.70 m subject). Label unobserved geometry and invented appearance.
- Preserve recorded audio words and timing; do not invent speaker stems.
