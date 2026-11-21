# Pipeline setup diagnostics

For upload-once processing and its delivery/quality distinction, see
[automatic uploads](automatic-uploads.md). The dashboard opts new runs into that mode;
the supervisor must be configured and running before it can process them.

Run the read-only doctor in the same shell and Python environment as the pipeline:

```sh
uv run --locked python -m orchestrator.doctor --json
uv run --locked --group inference python -m orchestrator.doctor \
  --json --clip /absolute/path/to/source.mp4 \
  --options '{"marble":"video","all_people":true,"objects":false,"finetune":false}'
```

The first command can identify missing local inference dependencies without installing them.
To install the local mask, reference-frame and pose-artifact dependencies, use
`uv sync --locked --group inference`, then rerun the second command. The local group does not
install the LHM, Pi3X, TRELLIS or gsplat GPU worker environments.

The report uses three statuses:

- `okay`: this particular check succeeded, such as importing a module, starting FFmpeg, or
  reading nonempty remote cache file metadata.
- `blocked`: a required executable, import, credential, cache, host or clip check failed.
  Its `action` describes the next setup step.
- `unverified`: configuration exists but its usability was not established, or a read-only probe
  could not complete. This includes untested account access, quotas and model compatibility.

`ready` is true only if every check is `okay`. Finding credentials in the environment does not
establish inference readiness. The default report leaves Modal workspace access and remote
caches unverified. With `--remote`, a selected World Labs stage also reads the authenticated
API credit balance using [`GET /marble/v1/credits`](https://docs.worldlabs.ai/api/reference/credits/get). A zero balance blocks setup; a positive
balance only confirms that credits were visible at that instant. It does not establish that
they cover the selected model, grant generation permission, or authorize spending. An unavailable
or malformed response remains unverified. The request is bounded, does not retry or follow
redirects, and never includes credentials or provider response bodies in the report.
Harness login, OpenAI access and credits, and World Labs generation permissions and quota remain
unverified even with remote inspection: the doctor never sends a model request or generates
a world. Exit status is 1 for a blocker and 0 otherwise; automation must also read
the report's status and checks, since exit 0 can mean unverified.

The doctor checks FFmpeg, FFprobe, Bun and the configured harness executable; imports local
pipeline modules in the current interpreter in an isolated process; and resolves the same
`WANDER_RUNS_ROOT` as the supervisor and API (default: repository `.context/runs`). The direct
runner also accepts the legacy `WANDER_RUNS_DIR` override. `WANDER_STAGE_LEDGER` selects a
shared paid-stage ledger; its default stays at repository `.context/pipeline-attempts.json`
to retain previous accounting. The `wander` command and direct runner share that ledger.
Keep the same absolute ledger across candidates and worktrees that share an allowance.

For an explicitly authorized run, set `WANDER_STAGE_AUTHORIZATION` to the absolute path of
its bounded paid-stage authorization and `WANDER_VLM_BUDGET` to the absolute path of its VLM
request budget before starting the supervisor. These two settings pass through the explicit
environment allowlist to the reviewing harness and its stage subprocesses; their values are
not included in the generated prompt or brief. Restrict that supervisor to the selected run
and remove these settings before supervising unrelated runs. They do not create approvals:
the runner still validates the authorization's source, candidate, operation, execution count
and expiry against the same shared ledger, and the VLM budget separately limits requests.
See [paid recovery](paid-recovery.md) for the authorization and recovery procedure.

A differing `WANDER_PYTHON` is explicitly
unverified. The default reviewing harness is Codex; `WANDER_REVIEW_AGENT` and
`WANDER_REVIEW_COMMAND` select alternatives. Built-in Codex and Claude harnesses must pass a
five-second `--version` startup probe; a found but broken wrapper blocks setup. Custom commands
are only checked for executable presence and remain unverified because their startup is not
tested. Harness probe output is never included in the report. No harness model session is started.

Reviewing agents use `workspace-write` by default. If a shared SQLite database or paid-stage
ledger lives outside the run directory, explicitly grant only its required parent directory:

```sh
export WANDER_REVIEW_WRITE_ROOTS='["/absolute/path/to/shared state"]'
```

This variable accepts a JSON list of nonempty directory strings, preserving spaces in paths;
use absolute paths. Its default is an empty list. Built-in Codex sessions receive `--add-dir`
for each root; resumed sessions use the equivalent
`sandbox_workspace_write.writable_roots` config override because `codex exec resume` lacks
`--add-dir`. Claude receives additional `--add-dir` arguments. Custom command arguments are
unchanged. The harness never infers shared roots or silently grants the whole repository;
SQLite needs its directory writable for journal/WAL files as well as the database itself.

For authorized network-backed stages, set `WANDER_REVIEW_NETWORK=true` before starting the
supervisor. This explicitly enables network access for commands in the Codex `workspace-write`
sandbox, for both fresh and resumed sessions, without changing filesystem permissions.
The default is off, explicitly enforced in both fresh and resumed Codex commands so host
configuration or a previous session cannot implicitly enable it.
Values `true/false`, `1/0`, `yes/no`, and `on/off` are accepted without
case sensitivity; invalid values fail setup. Enabling it with another harness or sandbox
mode is rejected. This setting is translated to Codex configuration, not passed through as
an environment variable, and never triggers a fallback to a broader sandbox. Stage permits
and request budgets remain separate requirements.

Credentials come from the selected shared stage plan. World generation needs `WLT_API_KEY`,
source-grounded world prompts and object descriptions need `OPENAI_API_KEY`, and Modal needs
a complete token pair or the selected saved profile. The runner's default Modal profile is
`dtpu`. Set `MODAL_PROFILE` to your own configured profile when appropriate. To use a different
TOML file, set `MODAL_CONFIG_PATH` to its absolute path so it resolves from the run directory.
The supervisor forwards these settings and the optional `WANDER_WORKER_TIMEOUT_SECONDS` and
`WANDER_LHM_GPU` controls to stage subprocesses. See [paid recovery](paid-recovery.md) for the
supported workers and values. The doctor reports credential names and configuration status.
It does not load dotenv files automatically:

```sh
set -a
source .env.author
set +a
```

The directory runner's paid-stage ledger limits execution count and unresolved submissions,
not dollars. The provider-agnostic database policy reserves explicit cost budgets only for
callers using that policy; a setup report does not establish a universal spending ceiling.

Use `--remote` for bounded, read-only inspection of the selected Modal workspace and, when
world generation is selected, the Marble API credit balance:

```sh
uv run --locked --group inference python -m orchestrator.doctor \
  --json --remote --clip /absolute/path/to/source.mp4 \
  --options '{"marble":"video","objects":false,"finetune":false}'
```

The Modal check only reads volume file metadata with `create_if_missing=False`. It never imports worker
entrypoints, builds images, starts containers, downloads weights or stages caches. The entire
Modal probe has a 20-second deadline; unavailable services or permission failures remain
unverified. Known missing/empty cache paths block. Requirements follow the worker paths:

| Selected work                                | Required existing volume/assets                                                                                 |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Cleaning, including the no-world graph       | `wander-clean-video-cache`: LaMa and SegFormer                                                                  |
| Camera solve                                 | `wander-overnight-motion-cache`: Pi3X and SegFormer                                                             |
| Person reconstruction/animation and tracking | `wander-overnight-lhm-cache`: LHM model, human priors, tracker/sample/voxel assets, GFPGAN and DINO checkpoints |
| Object shape/refinement                      | `wander-image-to-3d-cache`: TRELLIS, SDXL and DINO checkpoint                                                   |

Cache checks establish nonempty file presence, not hashes, active model revisions, complete
sharded snapshots, license access or compatibility with a future image build. Inspect saved
staging receipts and [paid recovery](paid-recovery.md) before deliberately staging missing assets.
The doctor does not initiate staging or repair. Local reference-frame models may still need
their separately reviewed Torch/Hugging Face cache preparation before processing a source.

### Stage missing Modal model caches

These commands start one paid CPU staging call per invocation. Select the saved local profile
alias for the intended workspace and use a fresh absolute Unix deadline in seconds and a unique
local receipt path for each authorized phase. Run phases serially. Set `STAGING_PROFILE` and
`STAGING_RECEIPT_DIR`, and set `DEADLINE_EPOCH` immediately before each call. For example,
`DEADLINE_EPOCH=$(( $(date +%s) + 1800 ))` allows 30 minutes from launch. Defining the function
below is local and does not start staging.

```sh
stage_cache() {
  uv run --locked modal --profile "$STAGING_PROFILE" run worker/modal_stage_weights.py::main \
    --phase "$1" --deadline-epoch "$2" --out "$STAGING_RECEIPT_DIR/$1.json"
}
```

| Phase command | Volume mounts and effect |
| --- | --- |
| `stage_cache clean-only "$DEADLINE_EPOCH"` | Cleaning cache only. Stages LaMa and pinned SegFormer in `wander-clean-video-cache`. The motion cache is outside the function's mounts. |
| `stage_cache lhm "$DEADLINE_EPOCH"` | LHM cache only. Stages LHM priors, model snapshot and Mask R-CNN in `wander-overnight-lhm-cache`. |
| `stage_cache dino "$DEADLINE_EPOCH"` | LHM cache only. Stages the DINO checkpoint. |
| `stage_cache gfpgan "$DEADLINE_EPOCH"` | LHM cache only. Stages the GFPGAN checkpoint. |
| `stage_cache lhm-resume "$DEADLINE_EPOCH"` | LHM cache only. Range recovery requires a matching failed `lhm` receipt and incomplete extraction. |
| `stage_cache clean-instance "$DEADLINE_EPOCH"` | Cleaning and LHM caches. Copies the checksum-verified Mask R-CNN checkpoint from the LHM cache into the cleaning cache. |
| `stage_cache clean-motion "$DEADLINE_EPOCH"` | Existing `wander-weights` source, cleaning and motion caches. Copies Pi3X into a fresh motion destination, stages SegFormer in both destination caches and stages LaMa in the cleaning cache. |
| `stage_cache clean-motion-repair "$DEADLINE_EPOCH"` | Same mounts as `clean-motion`. Repairs the diagnosed legacy Pi3X symlink failure using its receipt and incomplete copy. |
| `stage_cache clean-motion-repair-v2 "$DEADLINE_EPOCH"` | Same repair checks and mounts, with its own legacy attempt receipt. |

The bare script path also selects `main` in Modal 1.5.5. Use the local entrypoint to select
mounts. `::stage` is the underlying function and has no static mounts. Only the three combined
cleaning and motion phases look up `wander-weights`, which must already exist. `clean-only`
records `clean-only-attempt.json`, `clean-only-segformer.json` and `clean-only-lama.json` under
the cleaning volume's `setup-receipts/`. Legacy attempt and step receipt paths are preserved.
An incomplete attempt requires inspection and reconciliation. A complete attempt returns its
saved report without downloading again, though a new CLI invocation can still launch a container.

LHM prior staging excludes entries with an exact `.git` path component. This skips the Git LFS
object stored alongside the runtime Sapiens `.pt2` checkpoint and preserves the checkpoint at
its original path. A fresh `lhm` run still reads and hashes the entire source archive. `lhm-resume`
validates the declared source length and pins the ETag on every range. It reads the headers for
skipped entries and seeks past their file payloads. `skippedGitEntries` counts excluded members
and `skippedGitBytes` sums their declared sizes. `rangeBytesReceived` reports the bytes actually
fetched during resume, including headers. Any partial `.git` files left by an older attempt stay
on the volume and are omitted from the final runtime inventory. Existing receipt and recovery
checks still apply.

#### Reconcile a failed LHM range recovery

The final recovery slot is `lhm-resume-reconciled`. Before launching it, verify that the prior
Modal app and call have terminated. Compute `PRIOR_RESUME_SHA256` over the exact saved bytes of
the volume's `lhm-resume-attempt.json`, including whitespace. This phase requires a failed
`lhm-resume` receipt with a call ID, `TimeoutError`, the exact message
`cache staging deadline exhausted`, and an empty step list. The original `lhm` failure guard
still applies, and `data.incomplete` must exist with `data` absent.

Use one authorized launcher. Before invoking Modal, exclusively create and durably persist a
fixed local launch claim for this workspace, environment and LHM volume. Use exclusive-create
mode (`open(..., "x")`) and retain the claim even if launch or completion is uncertain. An
existing claim blocks another launch. The Modal volume receipt alone cannot enforce an atomic
cap across independent launchers.

```sh
uv run --locked modal --profile "$STAGING_PROFILE" run worker/modal_stage_weights.py::main \
  --phase lhm-resume-reconciled --deadline-epoch "$DEADLINE_EPOCH" \
  --prior-resume-sha256 "$PRIOR_RESUME_SHA256" \
  --out "$STAGING_RECEIPT_DIR/lhm-resume-2.json"
```

The worker commits `setup-receipts/lhm-resume-2-attempt.json` before reading archive ranges.
It records both call IDs, the prior receipt hash, `totalAttempt: 3` and `rangeAttempt: 2`.
The original receipts and their counters are preserved. Any existing final receipt consumes the
slot, including a partial or completed receipt. Inspect it directly after a lost response. This
path allows one initial staging attempt and at most two range recoveries. It uses the same
Git-metadata exclusion and resource limits as ordinary range recovery.

Staging is outside the paid-stage ledger. The function requests one CPU and 1 GiB of memory,
with limits of two CPUs and 4 GiB, one container per function configuration, zero function retries
and a two-second scaledown window. Modal's execution timeout is 1,800 seconds. On function entry,
the worker uses the smaller of 1,750 seconds and the remaining time to `--deadline-epoch`,
rejecting an exhausted deadline. Queueing and startup can consume the absolute time budget before
the worker enters. Neither timeout bounds total CLI wall time or spend. The receipt's
`computeUpperEstimateUSD` uses elapsed worker time at the resource limits (about $0.00003508/s).
Review provider charges for startup, build, transfer and storage separately.

Single-person placement now measures an undilated Mask R-CNN instance at every packaged source
ordinal, using the cached COCO_V1 checkpoint also used by the default reference-frame preparation.
This local pass never downloads weights. It records the exact source, decoder, checkpoint and
frame hashes in `single-person-masks.npz`; changed inputs invalidate reuse. Frames with zero or
multiple confident people remain unavailable, and fewer than three usable observations block the
fit. Inpainting masks cannot substitute for these observations. A fitted silhouette still must
pass the existing world-scale, size, contact and final quality checks.

Fine-tuning checks are included only when that stage is in the selected plan. Set
`WANDER_GPU_BOX=user@host` and optionally `WANDER_GPU_KEY=/absolute/path/to/key`. Remote mode
uses SSH batch mode, strict existing host-key verification, a bounded connection timeout and
the command `true`. It never uploads code or starts training. A successful login does not
verify the host's `~/venv`, CUDA, gsplat or GPU capacity; those remain unverified.

With `--clip`, FFprobe records dimensions, frame rate, duration, video codec and audio streams.
The existing local cut detector runs with `--no-score --no-people` under a 90-second deadline
and uses only a temporary report. It performs no person-model inference or shot ranking.
Cuts, malformed timing and an incomplete detector report block the supplied clip. Save and
review a report before trimming a continuous shot:

```sh
uv run --locked python scripts/shot_cuts.py --video /absolute/path/to/source.mp4 \
  --no-score --no-people --json .context/cuts.json
uv run --locked python scripts/shot_cuts.py --report .context/cuts.json \
  --trim 0 --out .context/reviewed-shot.mp4
```

Review the chosen index instead of assuming shot 0 is suitable. Continuity checks do not prove
parallax, person identity, downstream registration, visual quality or headset behavior. Source
clips are never uploaded by the doctor, and no paid operation is launched.

## After updating the checkout

Restart the local app and worker processes you own after pulling or switching code. Python API
processes retain imported stage definitions while later requests may load newer modules from disk;
a mixed process can fail setup checks even though a fresh doctor process succeeds. For example,
an API started before conditional paid-step metadata was added raised an `AttributeError` on
`Step.paid_with` after the checkout changed. Restarting uses one consistent code version; do not
relax credential checks to accommodate stale processes. Coordinate with another process's owner
rather than terminating it.

## Submission and recovery

### Prepare a delivery from saved inputs

On an existing run's admin page, choose **Use saved inputs**, then **Prepare paused run**.
The source run must be paused or finished, with no unresolved stage attempts. This supports
world-enabled tracked-person runs without fine-tuning, with complete local camera, tracking,
animation and exported scene files. Object models must be self-contained PLY or GLB files;
external glTF buffers and textures are not copied implicitly.

The server verifies the source and exported frame hashes, copies the supported inputs, and
pins every copied file before publishing a separate paused run. It retains the source run link,
room, presentation, original audio and objects. It does not copy agent sessions, attempts,
quality decisions, credentials or provider archives. Missing, changed or symlinked inputs stop
preparation without publishing a partial run. Checking and preparing use no provider calls.

After an operator resumes the prepared run, the local worker may launch **one** `package_people`
attempt using its pinned scene configuration. Other stages, extra parameters and retries are
rejected; the packager also disables its internal retry loop. The agent then asks for viewer
review. Reused stages are explicitly labeled as saved inputs, and packaging grants no fresh
inference or reconstruction quality acceptance. Preparation alone does not start a worker.

The API exposes `GET /api/pipeline/runs/{id}/saved-inputs` for availability and
`POST /api/pipeline/runs/{id}/saved-inputs` with an optional `id` for preparation.

### Submit new work

The admin reads its option defaults and supported world modes from `GET /api/pipeline/options`.
`POST /api/pipeline/doctor` inspects the API server locally and checks for a live worker matching
the requested automatic/manual mode. Its local dependency and credential checks describe the API
process, not a separate worker's environment. A heartbeat does not establish provider access or
spending authorization. Uploads can be queued while worker setup is repaired. The API and supervisor
must share `WANDER_RUNS_ROOT`. `create_service()` accepts local artifact storage through
`WANDER_ARTIFACT_ROOT`; importing it no longer requires cloud configuration.

For a dedicated automatic-upload worker, use `orchestrator.supervisor --uploads-only --watch`.
This leaves manual/legacy runs with their current workers and rejects source-specific exception
permits. See [automatic uploads](automatic-uploads.md#worker-for-automatic-uploads) for configuration
and worker-version compatibility.

New runs retain a source SHA-256 in `run.json`. Stages refuse missing or changed source digests
before launching; existing outputs stay on disk. Legacy manifests without a digest need a new
submission instead of blindly reusing cached state. Orchestrated runs reject cuts instead of
silently trimming into another run directory. The direct runner retains its explicit shot controls.
Reject-only cut checks skip shot ranking and its model loads; their detector arguments match
the cache key. Cut detection and rejection still run before any stage.

`wander step` admits one unresolved attempt per step in a run directory. A filesystem lock
serializes admission, then releases before execution so independent steps can proceed. Detached
children must match their parent's recorded command and claim execution once; overlapping
agents or supervisor restarts cannot launch that attempt again or overwrite its log. Completion
only clears the matching attempt, including when older journals contain overlapping starts.
API pause/cancel commands share the launch lock. They prevent subsequent launches and retain
results from work that already started; the lock is never held while waiting for that work.
The supervisor yields after a turn leaves detached work running, instead of immediately
requesting another agent turn to start the same work.

Pending `wander ask` questions also hold new agent and stage launches. Questions share the
admission lock with launches; already-running work keeps its completion records. Answering
wakes the supervisor on its next pass, including a reply that arrives as the prior turn ends.
A stalled session is discarded so the next turn can recover from the journal.

To watch just one run while answering questions, use:

```sh
uv run --locked python -m orchestrator.supervisor --run-id RUN_ID --watch
```

Without `--watch`, `--run-id` still makes one pass and exits. `--once` and `--watch` are
mutually exclusive. The worker's `codex --version` and `codex exec --help` must start normally;
an API doctor check does not test that executable or the worker's model sign-in.

An executor that disappears without recording completion remains unresolved. Inspect its log,
provider receipt and retained outputs before reconciling it; restarting the supervisor, removing
the lock file, or inventing a new attempt must not resubmit uncertain work. Older unresolved starts
without the child-claim record also require inspection. This guard applies to `wander step` on
one shared filesystem, not direct `run_clip.py` or worker invocations. The shared paid ledger
and [provider recovery rules](paid-recovery.md) still apply.

Run budgets (`maximum_cost` and custom per-run attempt/provider caps) are rejected by the API
because the directory runner does not enforce those database-policy settings. Use provider-side
limits alongside the shared ledger; a configured API budget must never look like a spending cap
that runtime commands ignore.

Single-person object processing detects thrown objects. Lifting and packaging them in 3D needs
tracked-person mode (`all_people=true`, even with `people=1`). The single-person detector waits
for placement's scale when a world is selected. Fine-tuning remains opt-in. Verification waits
for placement, anchors, selected object work and fine-tuning, and still requires explicit review.

The directory is authoritative. Its journal can explicitly resume a finished run; cached successful
stages are reused while failed stages remain visible. Source hashes and packaged frame hashes are
checked before accepting completion, along with the viewer's required files and exported source
bytes. A rerun invalidates downstream stages before replacing outputs. `--only` does not start
excluded descendants; they remain pending until selected or a full resume. Paid artifacts and
receipts remain subject to their existing recovery and attempt limits.
This does not yet fingerprint every successful stage's code
and parameters: use a new candidate when those inputs change, and retain the paid ledger.
