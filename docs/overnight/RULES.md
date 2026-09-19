# Unattended runs

These rules apply only when no human is available, such as an overnight run.
The current policy decisions (git, gate, spend, publishing, Modal account) live here so they can change without touching `AGENTS.md`.

To start a run, the human fills in [TONIGHT.md](TONIGHT.md) and launches the agent with a prompt such as:
"You are running unattended. Follow `AGENTS.md`, `docs/overnight/RULES.md`, and `docs/overnight/TONIGHT.md`."

## Loop

- Tasks and priorities come from `TONIGHT.md`. Make verified progress on as many goals as possible
  within its time and spending limits; move past blocked items and reserve final validation time.
  Progress goes in `docs/overnight/LOG.md`.
- Read the log before starting; it is your memory after context compaction.
- Record the run's start and deadline once; resumes and compaction do not reset its clock, budget,
  or retry counters. Follow the seven-hour checkpoints in `TONIGHT.md` and stop by its deadline.
- Follow [RUNBOOK.md](RUNBOOK.md) for S3 recovery, preflight, candidate execution, and failure
  recovery. Follow `AGENTS.md` for orchestrator/worker model selection and review.
- Append a log entry after every experiment, before doing anything else.
- Never stop to ask. Log the question, state a bounded assumption, and continue independent
  work. Missing credentials, source provenance, or budget are blockers, not permission to guess
  or launch paid work.
- Blocked: log why and move to the next item.
- At the validation checkpoint, defer unfinished feature work and perform the required final pipeline
  rerun in `TONIGHT.md`: all existing source clips plus the specified new clips, within the remaining
  time and resources. Report uncovered rows as blocked. Passing unit tests alone does not complete
  visual acceptance; incomplete coverage must remain explicit.
- A code change after rerunning invalidates affected clip results; rerun those stages and update
  the evidence. Do not repeatedly launch an unchanged failing job without a new diagnosis.
- Notable failures on old or new inputs trigger the investigation policy in `TONIGHT.md`:
  research uncertain causes, try bounded general fixes, and validate against other clips.
  A failed row alone is insufficient when useful diagnosis remains feasible within the time and budget.
- Finish with the per-clip pass/fail/blocked report, output and evidence references, and remaining
  issues plus a status for every requested goal. Stop by the deadline, or earlier when useful
  authorized work and the report are complete or blocked; do not invent unrelated work.
- The one exception: if `TONIGHT.md` still contains the `<item>` placeholder, log that and stop.

## Current policy

- Git: work on the branch named in `TONIGHT.md`, created from `main` if it does not exist. Never commit or push to `main`.
- Commit each verified change on its own with a message that says what changed and why; commit the log with it. Push the branch after each commit.
- Run `scripts/run_clip.py` with `--no-gate`; the human cleaned-frame stop is not in use. This
  does not waive visual review or any automated quality gates implemented during the run.
- Apply the rubric, critical-failure vetoes, and retry ceilings in `TONIGHT.md`. Never loosen a
  threshold, hide missing evidence, or reset attempts to turn a failure into a pass. Time, spend,
  and Marble generation limits override every retry allowance.
- Keep rerun candidates separate from the existing demo paths until viewer comparisons pass.
  Reuse existing Marble worlds (`--reuse-world`); allow at most one new generation per new clip.
- Preserve finished runs in shared S3. Use `--no-publish` while stages or other agents might still
  write assets; publish explicitly once writers are idle. The default auto-publish also runs after
  failures/gates, so use it only when that upload is intended and the shared tree is stable.
  Retry a publish failure with `bun run runs:publish`; keep local results and record a blocked
  archive if it still fails. Publishing evidence does not mean a candidate passed or was promoted.
- Enforce the deadline and resource ceilings in `TONIGHT.md`. Check the resource ledger before each paid
  stage/batch, reserve for in-flight work and final validation, and stop launching work if the
  remaining time/budget is insufficient or cannot be bounded. Bound job timeouts by the remaining
  window and preserve partial outputs before stopping your jobs at the deadline. Stop every Modal
  job you started once it is idle; never stop another teammate's job.
- Use `MODAL_PROFILE=dtpu` for GPU launches. Model volumes may belong to the primary profile; check availability first.
