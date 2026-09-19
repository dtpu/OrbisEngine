# Overnight execution runbook

Read the root `AGENTS.md`, [TONIGHT.md](TONIGHT.md), and [RULES.md](RULES.md) first. The task list and resource
ceilings come from `TONIGHT.md`; this document explains how to execute them. Do not launch
inference merely to test access. Fill the task placeholder before handing off an unattended run.

## Seven-hour clock

Start the seven-hour clock when unattended execution actually begins, not when the plan or prompt
is written. Record that UTC start and the resulting deadline in `LOG.md`. Context compaction,
restarts, and resumed sessions keep the original clock and resource ledger; they do not start a new
seven-hour allowance. The deadline and the budget ceilings in `TONIGHT.md` override full coverage.
When either prevents a required run or comparison, stop launching work and leave an explicit
blocked inventory row with the missing evidence and reason.

- **0:00–1:00:** establish the inventory and baseline, watch every available source and baseline,
  record the manual flaw inventory, and trace each applicable pipeline step.
- **1:00–4:30:** implement the highest-priority bounded general fixes and measure them. A parallel
  frontend task is useful only when it has independent files, acceptance checks, and budget.
- **4:30–6:30:** freeze code early enough for final affected-stage reruns, matched source/baseline/
  candidate comparisons, and reporting of rows that cannot complete. Earlier reruns are welcome;
  rerun again only when a later change invalidates them.
- **6:30–7:00:** publish authorized private outputs, finish the report and recovery references,
  stop jobs started by this run, and preserve partial evidence. At the deadline, stop even if the
  complete inventory has not run.

## Start and record the baseline

1. Fetch `origin` and inspect new commits. Start the named working branch from current `main`;
   preserve teammates' work and push only that working branch. Never force-push to resolve drift.
2. Install with `bun install --frozen-lockfile` and `uv sync --locked --group inference`.
   FFmpeg and Chrome are also required. Follow [author credentials](../shared-assets.md#author-credentials).
   The read-only viewer key cannot publish or read archive intermediates. Export `.env.author`
   into the launching shell; never print its contents or use shell tracing around secrets.
3. Record the commit, private S3 snapshot, active viewer mode, local disk available, clip inventory,
   and current provider usage in `LOG.md`. Keep detailed manifests and measurements under
   `.context/evidence/overnight/`. Record model/effort choices and any unavailable capabilities.
4. Run relevant local checks before paying for a baseline that already has a known code failure.
   `bun run build`, `bun run format:check`, `bun run test:pull-assets`, and `bun run test:marble`
   cover build, formatting, recovery, and Marble submission contracts. Select audio/object/viewer
   checks for affected work as described in their docs.

For this run, admit `assets/test1.mov` and `assets/test2.mov` relative to the repository root as
new inputs using their full clips unless `TONIGHT.md` specifies an exact trim. Label their
comparison rows **no prior baseline** unless a verified prior output is recovered. Before processing, record each original's
SHA-256, byte size, duration, frame rate, dimensions, codecs/audio streams, and continuity result.
Keep the originals unchanged and preserve their recorded audio words and timing. Work from ignored
copies where a stage requires another location or basename. The candidate input basename must
match the runner `--name`, because downstream packaging resolves artifacts by that basename.

Watch every available source video and baseline from beginning to end before choosing fixes. Build
a manual flaw inventory with clip, source time/range, view or camera path, severity, observed flaw,
and evidence reference. For every applicable pipeline step, record its input, output, observed
quality, failure or limitation, start/end/elapsed time, and estimated then confirmed cost. Missing
or unreadable evidence is an unknown, not a pass.

Read-only Modal preflight after credential setup:

```sh
uv run --locked modal profile list --json
uv run --locked modal volume list --profile dtpu --json
uv run --locked modal volume ls --profile dtpu wander-clean-video-cache /lama --json
uv run --locked modal volume ls --profile dtpu wander-overnight-lhm-cache /data/pretrained_models --json
uv run --locked modal volume ls --profile dtpu wander-overnight-motion-cache /huggingface --json
uv run --locked modal app list --profile dtpu --json
```

Missing model caches are a setup blocker. Check the intended shared workspace/profile before
building or downloading replacements. Environment Modal tokens override saved-profile tokens;
the `dtpu` label alone does not identify the billed workspace. Do not print tokens when diagnosing.
`modal run worker/modal_lhm.py` without arguments is **not** a free preflight: it can build images
and invoke staging. The uv-based GPU image definitions passed local checks, but cold builds
have not been verified on paid GPU infrastructure.

## Recover inputs and previous runs

```sh
# Bun loads the private viewer environment; listing pins the current snapshot without media download.
bun run assets:pull --out .context/overnight-inputs --list
bun run assets:pull --out .context/overnight-inputs --path /clips/elevator.mp4
# Authors only: inspect the archive before selecting an existing run prefix.
bun run assets:pull --out .context/overnight-archive --archive --list
bun run assets:pull --out .context/overnight-archive --archive --prefix runs/elevator/
```

Use exact paths/prefixes present in the returned manifest. The downloader preserves hierarchy
under `--out`, verifies SHA-256 and size, and resumes matching files. `.wander-pull.json` pins
the snapshot; use a new output directory for another snapshot. `--snapshot` can select a
historical manifest. It refuses differing local files without `--overwrite` and rejects symlink
destinations. See [shared assets](../shared-assets.md#recover-input-clips-and-runs) for bulk selection.

Published `/clips/` contains trimmed/transcoded inputs, including elevator, lobby, stairs2,
atrium, HP, and Tears of Steel. Full-length camera originals and full films are **not guaranteed**
to be in S3. The five picker presets are not the complete input inventory: `tos31` actually uses
`tos31d.mp4`, and direct presets include HP. Exclude generated cinematic outputs, deduplicate
identical source/trim aliases, and confirm provenance before treating an uploaded MP4 as a source.

For each input, record its SHA-256, duration, trim, previous options, world/operation ID, and
baseline manifest. Missing inputs, existing world IDs, or required settings become explicit
blocked rows. Do not regenerate a world simply because the metadata is missing. Restore prior
audio manifests/WAVs and evidence from the same snapshot before comparing: the five main
packaged videos themselves are silent; their authentic mixes are external sidecars.

## Provided key pool and job ownership

Aayan authorizes use of the existing provider keys supplied by the team, including rotation across
those supplied accounts. Do not stop an entire run because the first key has no credits when
another authorized account has capacity. Keys for the same account share its limits; changing a
key may not resolve throttling. Keep secret values only in ignored private environment/config
files. Logs and commands shown in reports use aliases, never key values or fragments.

Before launching, inventory the authorized pool with provider, key alias, account/workspace alias,
verified balance/quota, any account-specific cap, and model/cache access. Record unavailable checks
as unknown. Aggregate all spending against the run caps in `TONIGHT.md`; do not multiply those
caps by the number of keys. Do not create accounts, claim new trials, purchase credits, or raise
limits automatically. Provided credentials do not themselves establish that an account has funds.

Use this selection and recovery order:

1. Resolve source hash, exact trim and verified aliases against the team's existing world/run
   inventory. Reuse an existing world. Assign one owner for every new source before submission.
2. Select a supplied account with verified capacity and reserve the possible cost in the shared
   ledger. Record key/account aliases, source/request hashes, attempt number and output path
   **before** launching. Use an atomic claim across workers; local receipts do not coordinate
   different machines. If a shared lock/ledger is unavailable, partition clip ownership explicitly.
3. Persist the provider operation/app/function-call ID immediately, together with its owning
   account alias. Poll, fetch, cancel or recover using that account's credential. Switching the
   default key does not transfer ownership of an existing operation or access to its assets.
4. On a confirmed pre-submission quota/billing refusal with no accepted job, retire the refusal
   atomically and select another supplied account with capacity. Preserve the refusal record and
   count requests. Only one worker may reserve the next attempt. A real failed generation still
   consumes the one-generation allowance; key rotation cannot reset it.
5. On throttling, honor the provider's retry delay and bound retries. Do not relaunch a paid GPU
   command merely because its output contains `rate limit`; a job may already have run. On a
   timeout, disconnect or ambiguous response, recover the existing job or inspect operation
   history with the owning account. If the outcome remains unknown, block that source's new
   submission and continue independent work. Do not try another key to resolve an unknown job.
6. For a permitted new call on another supplied account, retain the original source allowance,
   attempt ledger, deadline and aggregate budget. Never rotate accounts to bypass an access
   restriction. OpenAI requests may be retried within the recorded request/cost allowance;
   non-idempotent generation launches require the stronger ownership checks above.

For Modal, keep `MODAL_PROFILE=dtpu` and verify the actual workspace selected by environment
credentials. Recover jobs and access volumes in their owning workspace. A missing volume/model
is a setup problem: stage caches deliberately, validate model/revision and path, then perform one
bounded smoke inference when authorized. Separate setup/build cost from actual GPU inference.
Do not substitute a silent CPU/low-quality fallback or omit people just to make a command succeed.

## Bounded execution and output provenance

Start with one paid pipeline and measure stage time, memory, provider limits and cost. Declare a
finite queue size before increasing concurrency; reserve budget for all active calls, cold builds,
and recovery. Coordinate at the account level across teammates. Spawning dozens of run processes,
then limiting only each process's own retries or launch spacing, is not a shared concurrency limit.
Keep publishing out of that queue: use `--no-publish` for active candidates and one publisher after
writers are idle. Never delete another run's `.part` file without establishing its owner is inactive.

Treat each stage as a recorded contract: source hash/time mapping, parameters/model/code version,
input/output hashes, status, attempt, start/end time, provider IDs and estimated/confirmed cost.
A manifest file or exit code alone does not establish correct execution. Check that stages really
ran, required models were loaded, and outputs cover the expected frames/people/objects. Link every
published scene to those receipts. Preserve camera, mask, ROI, depth and registration sample time
together; mixing references from different frames can corrupt placement even when inference passes.

Before merging an overnight harness, test at least ambiguous submit failures, simultaneous claims,
resume after interruption, changed source/settings/code, missing/corrupt outputs, forced descendants,
exhausted attempts/budget and failed publication. Never auto-enable bad-size/weak-anchor/scale-spread
flags from log text or disable the visual judge after an arbitrary exception. Such fallbacks need
an explicit contract, visible degraded status and source comparison, not a success label.

## Execute isolated candidates

Use a fresh name for each candidate and the same basename for its input video. The head-track
packager derives its run path from the clip basename. A new run name with the old source basename
can accidentally use old head tracks. Never run two processes against the same run name.

For example, after downloading elevator above, first check that neither `elevator-candidate`
outputs nor its run directory exist. Copy the unchanged clip without overwriting any existing file:

```sh
mkdir -p public/clips
cp -n .context/overnight-inputs/clips/elevator.mp4 public/clips/elevator-candidate.mp4

MODAL_PROFILE=dtpu uv run --locked --group inference scripts/run_clip.py \
  --clip public/clips/elevator-candidate.mp4 --name elevator-candidate \
  --marble image --reuse-world 065f7002-cb5c-42b6-a872-88e7fabe2c21 \
  --all-people --fps 12 --skip-finetune --no-objects --no-gate --no-publish
```

That world ID comes from elevator's original generation record. This is a **paid, bounded
diagnostic example**, not a free setup check or a command for every clip. It omits objects and
fine-tuning; the final regression must retain each clip's recorded settings and applicable
stages. `--all-people` caps people at four; record when a clip needs an explicit `--people` cap.
`--reuse-world` prevents a new Marble generation only: cleaning, world prompting, verification,
and GPU reconstruction can still cost money. Fetching an existing world requires the Marble key
unless matching metadata and splats are already local. Keep `--marble none` for existing clips
that use that lane. Fine-tuning needs separately configured SSH/GPU infrastructure.

The runner's cache is stage-status based. It does not automatically invalidate on changed code,
source, settings, or missing outputs. `--force` does not invalidate descendants, and `--only`
does not run prerequisites automatically. Prefer fresh candidate state; log every stage that
actually ran. Copying old `state.json` defeats a final rerun. Reuse weights and existing worlds,
not stale reconstruction results. If a fix affects several descendants, rerun all of them.

Packaging replaces the candidate world directory. Generate head/audio sidecars after the final
package step and preserve them before repackaging. `package_head_track.py` is a separate command:

```sh
uv run --locked --group inference scripts/package_head_track.py --world elevator-candidate-4d
```

Consult [audio](../audio.md) before restoring sound. An unchanged clip can use its verified
original mix; changed trims require new sample offsets and duration checks. Do not copy spatial
anchors from a different reconstruction or fabricate speaker stems/review approvals. The current
HP wide package remains silent because no soundtrack has been recovered.

## View, measure, and preserve

Use the URL printed by the pipeline, retaining its fitted floor/scale values, and add `&walk=1`.
Use `place=1`/`rotfix=1` only when the intended placement/alignment artifacts are present. Check
frame zero and representative playback frames against source and baseline, then inspect sideways
views, floor contact, map placement, collisions, and audio timing. Read `verify/report.json` and
its comparison sheet: a failed fidelity verdict can still accompany a zero pipeline exit code.

Capture comparisons in batches at matching source-camera timestamps and, when a baseline exists,
matching baseline timestamps and paths. Add off-axis and walk views that expose geometry hidden
from the recorded camera. Keep the manual full-video review authoritative evidence alongside any
sampled automation; still frames alone cannot establish temporal quality.

An LLM quality judge is a proposed addition for this run, not an existing end-to-end quality gate.
Before changing it, inspect the current `vlm_judge`/`verify_world.py` integration, prompts, image
limits, retry behavior, outputs, and token accounting, then reuse it only where its contract fits.
The criterion definitions and critical checkpoints live in `TONIGHT.md`. The initial proposal is
at least 75% for every applicable criterion, every critical checkpoint passing, and any critical
failure vetoing acceptance. Calibrate that proposal against clear manual passes and failures before
using it; it is a triage signal, not proof of visual quality. Unknown, missing, or incomparable
evidence blocks the affected criterion and final acceptance rather than receiving a passing score.

Pin immutable baseline and candidate snapshots independently; the shared latest pointer may
include outputs from other branches. Use matching code/manifest formats, exact source times and
cameras, and record missing capabilities rather than substituting unrelated demo presets.

Benchmark compression and speed on representative fixed inputs: record original and packaged
bytes, settings, compression ratio, encode/package wall time, viewer transfer/load-to-first-frame,
and playback behavior, with matched visual evidence for any quality claim. Separate first visible
world, first usable motion and complete readiness. Measure fresh-browser/cold-server-cache and
warm-cache runs separately, record timeouts and failed requests, and test one active renderer at a
time. Report repeat count, network/cache conditions and frame pacing; compression does not imply
better geometry. Inject missing first/later motion frames to verify failure states, not just success. Check the actual viewer
at narrow and wide desktop sizes for usable controls, source comparison/projector presentation,
resize behavior, loading/error states, and interaction responsiveness. Record physical projector
or headset checks only when that hardware was actually observed.

Unpublished candidates require **local asset mode**. When port 5399 is free, start:

```sh
WANDER_ASSETS_MODE=local bun run demo
```

If a server already owns that port, use it and coordinate a mode/snapshot restart with its owner.
Do not kill it or silently start a second server on another port. S3 mode deliberately returns
404 for unpublished local assets. If you cannot obtain a suitable viewer, preserve the candidate
and mark visual acceptance blocked. A CLI success is not a substitute. A new published snapshot
also requires a coordinated restart before the existing S3 viewer can see it.

When all writers are idle, publish candidates and evidence with the configured author AWS identity:

```sh
bun run runs:publish --evidence-dir .context/evidence/overnight
```

This publishes `public/`, `.context/run/`, and selected evidence, not arbitrary local source
folders. The supplied originals for this run stay in gitignored `assets/` for local use; do not
copy those originals into publication trees. Git ignore rules do not control S3 publication.
Keep candidate asset paths separate; uploading them does not require changing demo presets.
Record the immutable snapshot and archive keys. A publish conflict means another author changed
the pointer; inspect their result and retry when writes are idle, never overwrite their changes.
Only promote candidates with passing visual comparisons. Finish with every inventory row marked
passed, failed, or blocked and its evidence, spend, source hash, and final code commit.
Private asset access authorizes only the private recovery and publication paths specified in
`TONIGHT.md` and the shared-assets documentation. Do not purchase credits, publish media publicly,
or deploy to an unspecified target. In the final report, distinguish features actually implemented
and exercised from proposals or planned work, and identify unverified behavior explicitly.

## Budget and failure recovery

Maintain a resource ledger with: provider/account alias, starting usage/balance, confirmed spend
since start, estimates, in-flight reserved cost, remaining allowance, and last evidence checked.
Use the limits in `TONIGHT.md`, shared across all workers and clips. Start one bounded paid
candidate at a time until actual costs are measured. Before each next stage/batch, include cold
image builds, idle GPU time, retries, and other teammates' spend. Set bounded job timeouts and
record the stop condition. Do not launch work whose remaining cost cannot fit the allowance.

For a quality failure, allow at most two quality-motivated retries after the first stage execution
(three total executions of that stage). Every retry needs a changed, justified parameter or code
hypothesis and recorded before/after evidence, timing, and cost. Never loosen an acceptance
threshold to make a retry pass. Read-only API transport errors may use a separately bounded retry/backoff
policy. Paid launch wrappers must not repeat a submitted or ambiguous job. Count every request and
its elapsed time and possible cost; a transport retry does not grant another quality attempt. The shared deadline and provider budgets can stop retries sooner.
The one-generation-per-new-clip Marble rule overrides every retry allowance: recover or poll the
recorded operation instead of submitting another generation. `--no-gate` skips only the current
human cleaned-frame stop; it does not waive manual review or bypass any future automated quality
gate added by the run.

Worker `estimatedComputeUSD` values and elapsed times are estimates, not provider invoices.
`vlm_judge.ask_images` currently drops OpenAI token-usage metadata and may retry up to three times;
only some artifacts, such as world-ruler answers, preserve usage. Missing token counts do not mean
zero cost. There is no global automatic dollar/token cutoff in the runner. Check provider usage
and the coding host's separate usage view; record unknowns rather than claiming exact totals.

Do not assume usage dashboards or spend alerts enforce the run budget. Verify the account
control actually available and keep local reservations regardless; this repository does not
change account settings. Monthly account limits are not a fresh per-run allowance. A billing
refusal stops calls on that exhausted account. Another team-supplied account may be selected
under the shared run cap and [key-pool recovery rules](#provided-key-pool-and-job-ownership).
Do not top up, raise limits, or add an unprovided account automatically.
See [OpenAI spend controls](https://developers.openai.com/api/docs/guides/spend-limits).

Keep exact commands, exit statuses, provider app/function-call IDs, Marble operation/world IDs,
and bounded logs. To inspect an existing Modal app without relaunching:

```sh
uv run --locked modal app logs APP_ID --profile dtpu --tail 100 --timestamps --show-function-call-id
```

Diagnose once before retrying. Rate limits may justify a bounded delay; unchanged model failures
need a new hypothesis. Known Marble operation IDs are recovered by polling, existing world IDs
by fetching/`--reuse-world`, never by resubmitting. A timeout with no saved operation ID may still
have spent credits: preserve the submission record and block generation until the account's
operation history resolves it. Do not delete receipts or rename an input to bypass this protection.
The guard covers one generation name/mode in its retained metadata directory; it is not
cross-machine or source-hash idempotency. Assign one owner per clip. Preserve the matching
`*-generation.json`, operation/world metadata, and operation logs from `WANDER_MARBLE_DIR`
(default `.context/marble`) under the selected evidence directory before publishing. That metadata
directory is not otherwise included by the default run archive. Restore those records before
recovering on another machine; never start a second submission because its local history is absent.
Stop only jobs you started, and preserve partial outputs/evidence even when a budget or service
failure prevents the full final rerun.
