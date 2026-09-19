# Working on Wander

This file is durable: rules that stay true across sessions. Do not edit it or record progress here.
Task list: `TONIGHT.md`. Progress: `docs/overnight/LOG.md`. Context and commands: `README.md`.
Read the matching `docs/*.md` before changing audio, objects, or shared assets.

## Unattended

- No human is available. Never stop to ask: log the question, state your assumption, continue.
- Read the log before starting; append to it after every experiment.
- Blocked: log why and move to the next item. All done or blocked: retry the first unverified item from a new angle.
- The one exception: if `TONIGHT.md` still holds only its template, log that and stop. Never invent a backlog.
- Run `scripts/run_clip.py` with `--no-gate`. No spend cap; `MODAL_PROFILE=dtpu`. Stop Modal jobs you started once idle.

## Never

- Commit media, generated scenes, weights, run outputs, screenshots, or keys; paste secrets anywhere.
- Use bare `git stash`, rewrite pushed history, or discard other agents' changes.
- Kill a Vite server, pipeline run, or Modal job you did not start.
- Skip, weaken, or delete a test to get to green.
- Add per-clip constants to runtime code.

Use Bun and `uv run --locked` as the README shows; keep both lockfiles current and run `bun run format:check` after Python edits.

## Evidence

- Exit codes and passing tests do not prove visual quality. Check the real viewer at `http://127.0.0.1:5399` against the source footage, adversarially.
- Keep screenshots and measurements under `.context/evidence/`, never in Git. Simulated XR is not headset evidence.
- Quote body-heights, not metres (metres assume a 1.70 m subject). Label invented geometry and appearance; never invent audio.

## Dead ends (measured; do not retry without a new idea)

- Refining body pose against footage makes it worse.
- Faces under ~64 px are unreadable; head refits do not help.
- Smoke, fire, water, and low-parallax or dark clips do not reconstruct.
- GEN3C static fill lost to FLUX panorama fill.
