# Working on Wander

This file is durable: rules that hold for every session. Do not edit it or record progress here.
Read `README.md` for context and commands.
Before changing an area, read its doc: `docs/audio.md`, `docs/objects.md`, or `docs/shared-assets.md`.
Check `docs/known-limits.md` before trying a new approach.
When running unattended, also follow `docs/overnight/RULES.md` and `docs/overnight/TONIGHT.md`.

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

## Planning and model use

- Use the main chat as the orchestrator: own the plan, task boundaries, spending, integration,
  and final review. Use a strongest available model with deep reasoning for architecture,
  ambiguous failures, visual judgments, and decisions that could spend credits or damage results.
  Aayan's preferred examples are Fable or GPT-6 Astra with extra-high (`xhigh`) reasoning,
  when that exact model and setting are available in the host.
- Delegate bounded implementation to the least expensive available model/effort that can meet
  the same acceptance checks. GPT-6 Astra at `medium` suits straightforward execution;
  use `high` for work needing more analysis. A lighter capable model is also appropriate for
  mechanical edits, focused tests, and inventory work. These are task-based choices, not a
  blanket instruction to lower quality or change the pipeline's inference models.
- Give each worker a concrete goal, owned files, relevant context, and verifiable done checks.
  Avoid full-history copies, duplicate investigations, and agents whose only task is waiting.
  The orchestrator should continue useful independent work while workers run.
- Escalate when a worker finds ambiguous requirements, a new design decision, a failed
  acceptance check it cannot explain, or needs an unchanged retry. Do not burn usage by
  repeatedly asking a weaker model to solve the same unresolved problem.
- Review worker diffs and evidence before integration. Model savings never justify skipped
  tests, reduced visual checks, or claiming unmeasured results. Use concise handoffs and logs.
- Select only models and effort settings actually exposed by the session. If model switching
  is unavailable, report that constraint and keep the task bounded; do not claim a switch happened.
  `CLAUDE.md` imports this file so both hosts follow the same policy.

## Evidence

- Exit codes and passing tests do not prove visual quality. Check the real viewer at `http://127.0.0.1:5399` against the source footage, adversarially.
- Keep screenshots and measurements under `.context/evidence/`, never in Git. Simulated XR is not headset evidence.
- Quote body-heights, not metres (metres assume a 1.70 m subject). Label unobserved geometry and invented appearance.
- Preserve recorded audio words and timing; do not invent speaker stems.
