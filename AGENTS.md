# Working on Wander

This file is durable: rules that stay true across sessions. Do not edit it or record progress here.
Task list: `TONIGHT.md`. Progress: `docs/overnight/LOG.md`. Context and commands: `README.md`.
Read the matching `docs/*.md` before changing audio, objects, or shared assets.

## Unattended

- No human is available. Never stop to ask: log the question, state your assumption, continue.
- Read the log before starting; append to it after every experiment.
- Blocked: log why and move to the next item. All done or blocked: retry the first unverified item from a new angle.
- The one exception: if `TONIGHT.md` still holds only its template, log that and stop. Never invent a backlog.
- Run `scripts/run_clip.py` with `--no-gate`. No spend cap. Stop Modal jobs you started once idle.
- Use `MODAL_PROFILE=dtpu` for GPU launches. Model volumes may belong to the primary profile; check availability first.

## Never

- Commit footage, generated scenes, weights, run outputs, screenshots, credentials, or keys; media lives in private shared storage.
- Paste secrets into chat or logs.
- Use bare `git stash` (worktrees share it), rewrite pushed history, or discard other agents' changes.
- Kill a Vite server, pipeline run, or Modal job you did not start.
- Skip, weaken, or delete a test to get to green.
- Add per-clip constants to runtime code; use measured manifests and general flags.

## Tooling

- Bun for viewer commands, `uv run --locked` for Python tools. Keep `bun.lock` and `uv.lock` current.
- Run `bun run format:check` after Python edits; use ordinary readable blocks.
- Start Vite only with `bunx --bun vite --port 5399 --host 127.0.0.1`. `RECORD=1` disables reload during captures.
- At most two subagents at once, on independent tasks; do not poll them.

## Evidence

- Exit codes and passing tests do not prove visual quality. Check the real viewer at `http://127.0.0.1:5399` against the source footage, adversarially.
- Keep screenshots and measurements under `.context/evidence/`, never in Git. Simulated XR is not headset evidence.
- Quote body-heights, not metres (metres assume a 1.70 m subject). Label unobserved geometry and invented appearance.
- Preserve recorded audio words and timing; do not invent speaker stems.

## Dead ends (measured; do not retry without a new idea)

- Refining body pose against footage makes it worse.
- Faces under ~64 px are unreadable; head refits do not help.
- Smoke, fire, water, and low-parallax or dark clips do not reconstruct.
- GEN3C static fill lost to FLUX panorama fill.
