# Unattended runs

These rules apply only when no human is available, such as an overnight run.
The current policy decisions (git, gate, spend, publishing, Modal account) live here so they can change without touching `AGENTS.md`.

To start a run, the human fills in `TONIGHT.md` and launches the agent with a prompt such as:
"You are running unattended. Follow `AGENTS.md`, `docs/overnight/RULES.md`, and `TONIGHT.md`."

## Loop

- Tasks come from `TONIGHT.md`, worked in order. Progress goes in `docs/overnight/LOG.md`.
- Read the log before starting; it is your memory after context compaction.
- Append a log entry after every experiment, before doing anything else.
- Never stop to ask. Log the question, state the assumption you chose, and continue.
- Blocked: log why and move to the next item.
- Once code backlog items are verified or explicitly blocked, complete the required final pipeline
  rerun in `TONIGHT.md`: all existing source clips plus the specified new clips. Passing unit tests
  alone does not complete the run.
- A code change after rerunning invalidates affected clip results; rerun those stages and update
  the evidence. Do not repeatedly launch an unchanged failing job without a new diagnosis.
- Finish with the per-clip pass/fail/blocked report, output and evidence references, and remaining
  issues. Stop when the backlog and final report are complete; do not invent unrelated work.
- The one exception: if `TONIGHT.md` still contains the `<item>` placeholder, log that and stop.

## Current policy

- Git: work on the branch named in `TONIGHT.md`, created from `main` if it does not exist. Never commit or push to `main`.
- Commit each verified change on its own with a message that says what changed and why; commit the log with it. Push the branch after each commit.
- Run `scripts/run_clip.py` with `--no-gate`; the cleaned-frame review gate is not in use.
- Keep rerun candidates separate from the existing demo paths until viewer comparisons pass.
  Reuse existing Marble worlds (`--reuse-world`); allow at most one new generation per new clip.
- Let `run_clip.py` publish to shared S3 as it does by default. A publish failure leaves results local; retry with `bun run runs:publish` and move on rather than treating it as a blocker.
- No spend cap. Stop every Modal job you started once it is idle.
- Use `MODAL_PROFILE=dtpu` for GPU launches. Model volumes may belong to the primary profile; check availability first.
