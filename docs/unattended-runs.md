# Unattended runs

Durable policy and procedure for runs with no human available (for example overnight). Read the
root `AGENTS.md` first; this document adds the rules that only apply when nobody can answer.
Run-specific facts (branch, deadline, spend caps, clip list, backlog) live in a per-run brief
written by the human, not here. Quality criteria and retry caps live in
[quality-rubric.md](quality-rubric.md).

## The run brief

Before launching, the human writes a short brief (untracked or on the working branch) containing:

- Branch name. Default: `unattended/<owner>-<run-id>`, created from current `main`.
- Time window with UTC start and deadline, plus reserved time for final validation and handoff.
- Spend ceilings per provider (Modal, OpenAI, Marble) and the credential aliases authorized for
  each. Ceilings apply to the whole run across every key, worker, clip and resumed chat.
- Existing clips to preserve and any new clips to admit, with source path, exact trim, and any
  required `--people` cap or processing options.
- Prioritized goals with a done condition each.

Launch with a prompt such as: "You are running unattended. Follow `AGENTS.md`,
`docs/unattended-runs.md`, and the brief at `<path>`." If the brief still contains a placeholder,
log that and stop.

## Loop

- Make verified progress on as many goals as possible within the brief's limits; move past blocked
  items and reserve final validation time. Progress goes in an append-only run log under
  `.context/evidence/<run-id>/LOG.md`; read it before starting, since it is your memory after
  context compaction. Append an entry after every experiment, before doing anything else:
  tried, exact command, measured result with evidence path, assumptions made, next step.
- Record the run's start and deadline once; resumes and compaction do not reset the clock, budget,
  or retry counters. Stop by the deadline.
- Never stop to ask. Log the question, state a bounded assumption, and continue independent work.
  Missing credentials, source provenance, or budget are blockers, not permission to guess or to
  launch paid work. Blocked: log why and move to the next item.
- At the validation checkpoint, defer unfinished feature work and rerun the pipeline for every
  existing clip plus the brief's new clips within the remaining allowance. A code change after a
  rerun invalidates affected clip results; rerun those stages and refresh evidence. Do not relaunch
  an unchanged failing job without a new diagnosis.
- A notable clip failure is an engineering task, not just a failed row: capture it in the real
  viewer, identify trigger/stage/affected input class, check `known-limits.md`, form a bounded
  hypothesis with expected result and cost, validate a fix on the triggering clip, another relevant
  clip and an unaffected baseline, and add a focused regression check. Prefer general fixes based
  on measured inputs; never branch on clip names. If no sound fix fits, preserve the investigation
  and report the row failed/blocked with the precise limitation.
- Finish with the per-clip pass/fail/blocked report described under [Report](#report).

## Git

- Work only on the brief's branch. Never commit or push to `main`; never force-push to resolve drift.
- Commit each verified change on its own with a message saying what changed and why; push after
  each commit.
- Preserve original author identities and co-author credits when porting teammates' commits;
  adapted work retains contributor credit and source-commit references. Agent credit is additional.

## Setup and preflight

1. Fetch `origin`, inspect new commits, and start the working branch from current `main`.
2. `bun install --frozen-lockfile` and `uv sync --locked --group inference`. FFmpeg and Chrome are
   required. Follow [author credentials](shared-assets.md#author-credentials); the read-only viewer
   key cannot publish or read archive intermediates. Export `.env.author` into the launching shell
   without printing it or using shell tracing.
3. Record the commit, private S3 snapshot, active viewer mode, free disk, clip inventory, and
   current provider usage in the log. Keep manifests and measurements under `.context/evidence/`.
4. Run local checks before paying for a baseline with a known code failure: `bun run build`,
   `bun run format:check`, `bun run test:pull-assets`, `bun run test:marble`, plus the audio,
   object and viewer checks described in their docs.
5. Read-only Modal preflight (`MODAL_PROFILE=dtpu`):

   ```sh
   uv run --locked modal profile list --json
   uv run --locked modal volume list --profile dtpu --json
   uv run --locked modal volume ls --profile dtpu wander-clean-video-cache /lama --json
   uv run --locked modal volume ls --profile dtpu wander-overnight-lhm-cache /data/pretrained_models --json
   uv run --locked modal volume ls --profile dtpu wander-overnight-motion-cache /huggingface --json
   uv run --locked modal app list --profile dtpu --json
   ```

   Missing model caches are a setup blocker: stage caches deliberately, validate model/revision and
   path, then run one bounded smoke inference when authorized. Environment Modal tokens override the
   saved profile; confirm the actual workspace. `modal run worker/modal_lhm.py` without arguments is
   not a free preflight. Do not substitute a silent CPU/low-quality fallback or omit people to make a
   command succeed.

Before processing a new input, record its SHA-256, byte size, duration, frame rate, dimensions,
codecs/audio streams, and continuity result. Keep originals unchanged and work from ignored copies.
The candidate input basename must match the runner `--name`; the head-track packager resolves
artifacts by basename, so a new run name with an old source basename can pick up old head tracks.

## Recover inputs and previous runs

```sh
bun run assets:pull --out .context/inputs --list
bun run assets:pull --out .context/inputs --path /clips/elevator.mp4
# Authors only:
bun run assets:pull --out .context/archive --archive --list
bun run assets:pull --out .context/archive --archive --prefix runs/elevator/
```

Use exact paths from the returned manifest; `.wander-pull.json` pins the snapshot, so use a new
output directory per snapshot. See [shared assets](shared-assets.md#recover-input-clips-and-runs).
Published `/clips/` holds trimmed/transcoded inputs; full originals are not guaranteed to be in S3,
and the picker presets are not the complete inventory. Exclude generated outputs, deduplicate
identical source/trim aliases, and confirm provenance before treating an uploaded MP4 as a source.

For each input, record SHA-256, duration, trim, previous options, world/operation ID, and baseline
manifest. Missing inputs or world IDs become explicit blocked rows; never regenerate a world because
metadata is missing. Restore prior audio manifests/WAVs from the same snapshot before comparing:
the packaged videos are silent and their mixes are external sidecars.

## Credentials, key pool, and job ownership

Supplied credentials are an authorized pool: rotate among keys/accounts explicitly provided by
teammates when capacity requires it, keeping one shared run budget and retry ledger. A key change
never grants a new clip generation or resets attempts. Do not create accounts, obtain trials, buy
credits, or raise limits. Keep secret values only in ignored files; logs and reports use aliases.

Main reads one active `WLT_API_KEY` and `OPENAI_API_KEY` and does not rotate a pool itself: select
the alias in the orchestrator and export it for that process. Before launching, inventory the pool
(provider, key alias, account alias, verified balance/quota, caps, model/cache access); record
unavailable checks as unknown. Credentials alone do not establish funds.

Selection and recovery order:

1. Resolve source hash, exact trim and aliases against the existing world/run inventory. Reuse an
   existing world (`--reuse-world`). Assign one owner per new source before submission.
2. Select an account with verified capacity and reserve the possible cost in the shared ledger
   (`--stage-ledger`, see [paid-recovery.md](paid-recovery.md)). Record aliases, source/request
   hashes, attempt number and output path before launching. If a shared lock is unavailable,
   partition clip ownership explicitly.
3. Persist the provider operation/app/function-call ID immediately with its owning account alias.
   Poll, cancel or recover with that account's credential; switching keys does not transfer ownership.
4. On a confirmed pre-submission quota refusal with no accepted job, retire it atomically and pick
   another supplied account. A real failed generation still consumes the one-generation allowance.
5. On throttling, honor the provider's retry delay and bound retries. Never relaunch a paid GPU
   command because its output mentions `rate limit`. On a timeout or ambiguous response, recover the
   existing job with the owning account; if the outcome stays unknown, block that source's new
   submission and continue other work. Do not try another key to resolve an unknown job.
6. OpenAI requests may retry within the recorded request/cost allowance; generation launches need
   the ownership checks above.

## Bounded execution and provenance

- Start one paid candidate at a time until costs are measured. Declare a finite queue size before
  increasing concurrency and reserve budget for active calls, cold builds and recovery. Per-process
  retry limits are not a shared concurrency limit.
- Run `scripts/run_clip.py` with `--no-gate` (the human cleaned-frame stop is not in use). This does
  not waive visual review or any automated quality gate.
- Use a fresh candidate name per run and never run two processes against the same run name. Keep
  candidates separate from demo paths until viewer comparisons pass. Bounded diagnostic example:

  ```sh
  mkdir -p public/clips
  cp -n .context/inputs/clips/elevator.mp4 public/clips/elevator-candidate.mp4
  MODAL_PROFILE=dtpu uv run --locked --group inference scripts/run_clip.py \
    --clip public/clips/elevator-candidate.mp4 --name elevator-candidate \
    --marble image --reuse-world <world-id-from-generation-record> \
    --all-people --fps 12 --skip-finetune --no-objects --no-gate --no-publish
  ```

  This is paid. `--reuse-world` only prevents a new Marble generation; cleaning, prompting,
  verification and GPU reconstruction still cost. The final regression must retain each clip's
  recorded settings and stages. Keep `--marble none` for clips on that lane.
- The runner cache is stage-status based and does not invalidate on changed code, source, settings
  or missing outputs; `--force` does not invalidate descendants and `--only` skips prerequisites.
  Prefer fresh candidate state, log every stage that actually ran, and rerun all affected
  descendants. Copying old `state.json` defeats a rerun.
- Treat each stage as a recorded contract: source hash/time mapping, parameters/model/code version,
  input/output hashes, status, attempt, timing, provider IDs and cost. Exit codes and manifests alone
  do not prove execution; check models loaded and outputs cover the expected frames/people/objects.
  Keep camera, mask, ROI, depth and registration sample times together.
- Packaging replaces the candidate world directory; generate head/audio sidecars after the final
  package step (`scripts/package_head_track.py --world <name>-4d`) and preserve them before
  repackaging. Consult [audio.md](audio.md) before restoring sound; changed trims need new offsets.
- Never auto-enable bad-size/weak-anchor/scale-spread flags from log text or disable the visual
  judge after an exception. Degraded fallbacks need an explicit contract and visible status.

## View, measure, and preserve

- Open the URL printed by the pipeline with `&walk=1`. Check frame zero and representative playback
  against source and baseline, then sideways views, floor contact, map placement, collisions and
  audio timing. Read `verify/report.json`; a failed fidelity verdict can accompany a zero exit code.
- Capture comparisons at matched source-camera timestamps and, where a baseline exists, matched
  baseline paths, including beginning/middle/end, difficult motion/occlusion and known failures.
  The manual full-video review stays authoritative; stills alone cannot establish temporal quality.
- Pin baseline and candidate snapshots independently; the shared latest pointer may include other
  branches' outputs. Label new clips without an old output **no prior baseline**.
- Unpublished candidates need local asset mode: `WANDER_ASSETS_MODE=local bun run demo` when port
  5399 is free. If another server owns it, coordinate; do not kill it or start a second server.
  Without a viewer, preserve the candidate and mark visual acceptance blocked.
- Preserve finished runs in shared S3. Use `--no-publish` while stages or other agents might still
  write; publish once writers are idle with `bun run runs:publish --evidence-dir <dir>`. Retry a
  failure with `bun run runs:publish`; keep local results and record a blocked archive if it still
  fails. A publish conflict means another author moved the pointer: inspect and retry, never
  overwrite. Publishing is not a pass or a promotion. Only promote candidates with passing visual
  comparisons, and keep the previous snapshot available.
- Preserve `*-generation.json`, operation/world metadata and logs from `WANDER_MARBLE_DIR` (default
  `.context/marble`) under the evidence directory before publishing; the default archive omits them.

## Budget and failure recovery

- Maintain a resource ledger: provider/account alias, starting usage, confirmed spend, estimates,
  in-flight reserved cost, remaining allowance, last evidence checked. Check it before every paid
  stage; include cold builds, idle GPU time, retries and teammates' spend. Bound job timeouts by the
  remaining window and stop launching work whose cost cannot fit.
- Apply the rubric, critical-failure vetoes and retry ceilings in [quality-rubric.md](quality-rubric.md).
  Never loosen a threshold, hide missing evidence, or reset attempts. Time, spend and Marble limits
  override every retry allowance.
- Marble: reuse existing worlds for existing clips; at most one new generation per new clip, and
  zero replacements. Recover known operation IDs by polling and world IDs by fetching, never by
  resubmitting. A timeout with no saved operation ID may still have spent credits: preserve the
  record and block generation until operation history resolves it. Do not delete receipts or rename
  inputs to bypass the guard, which is per generation name/mode, not cross-machine.
- `estimatedComputeUSD` and elapsed times are estimates, not invoices. `vlm_judge.ask_images` drops
  token usage and may retry three times; missing token counts do not mean zero cost. There is no
  global automatic dollar cutoff in the runner, and usage dashboards do not enforce the budget.
  Monthly account limits are not a fresh per-run allowance.
- Inspect an existing Modal app without relaunching:

  ```sh
  uv run --locked modal app logs APP_ID --profile dtpu --tail 100 --timestamps --show-function-call-id
  ```

- Diagnose once before retrying. Stop only jobs you started; preserve partial outputs and evidence
  even when a failure prevents the full rerun.

## Report

Finish with one row per inventory input: source/trim/hash, baseline and candidate references,
final commit/options, stages **executed** / **verified reuse** / **not run**, outcome
(**passed**, **failed**, **blocked**) with a change label (improved, unchanged, regressed,
unverified), key measurements, viewer/comparison links, and the concrete failure or blocking
reason. Include a goal-by-goal status, spend with estimates distinguished from confirmed usage,
elapsed time and why work stopped, and features actually exercised versus proposals.

Build a comparison index with labeled source/before/after screenshots and short playback/walk
recordings under untracked `public/reviews/<run-id>/`, publish it to private S3 when writers are
idle, and record the snapshot so teammates can open
`http://127.0.0.1:5399/reviews/<run-id>/index.html`. Commit the text report on the working branch.

## Out of scope

Repository rewrites or history cleanup; adding unprovided accounts or raising budgets;
regenerating existing Marble worlds without authorization; overwriting shipping assets before
comparisons pass; public media publication; purchasing domains or hosting. Simulated XR does not
establish headset performance; record that limitation and complete desktop comparisons.
