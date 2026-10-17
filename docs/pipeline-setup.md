# Pipeline setup diagnostics

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
caches unverified. Harness login, OpenAI/World Labs account permissions and credits remain
unverified even with remote inspection, because the doctor never sends a model request or
generates a world. Exit status is 1 for a blocker and 0 otherwise; automation must also read
the report's status and checks, since exit 0 can mean unverified.

The doctor checks FFmpeg, FFprobe, Bun and the configured harness executable; imports local
pipeline modules in the current interpreter in an isolated process; and resolves the same
`WANDER_RUNS_ROOT` as the supervisor and API (default: repository `.context/runs`). The direct
runner also accepts the legacy `WANDER_RUNS_DIR` override. `WANDER_STAGE_LEDGER` selects a
shared paid-stage ledger; its default stays at repository `.context/pipeline-attempts.json`
to retain previous accounting. The `wander` command and direct runner share that ledger.
Keep the same absolute ledger across candidates and worktrees that share an allowance.
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

Credentials come from the selected shared stage plan. World generation needs `WLT_API_KEY`,
source-grounded world prompts and object descriptions need `OPENAI_API_KEY`, and Modal needs
a complete token pair or the selected saved profile. The runner's default Modal profile is
`dtpu`; set `MODAL_PROFILE` to your own configured profile when appropriate. `MODAL_CONFIG_PATH`
can select a different TOML file. The doctor reports credential names and configuration status,
never their values. It does not load dotenv files automatically:

```sh
set -a
source .env.author
set +a
```

The directory runner's paid-stage ledger limits execution count and unresolved submissions,
not dollars. The provider-agnostic database policy reserves explicit cost budgets only for
callers using that policy; a setup report does not establish a universal spending ceiling.

Use `--remote` for bounded, read-only inspection of the selected Modal workspace:

```sh
uv run --locked --group inference python -m orchestrator.doctor \
  --json --remote --clip /absolute/path/to/source.mp4 \
  --options '{"marble":"video","objects":false,"finetune":false}'
```

This only reads volume file metadata with `create_if_missing=False`. It never imports worker
entrypoints, builds images, starts containers, downloads weights or stages caches. The entire
Modal probe has a 20-second deadline; unavailable services or permission failures remain
unverified. Known missing/empty cache paths block. Requirements follow the worker paths:

| Selected work | Required existing volume/assets |
| --- | --- |
| Cleaning, including the no-world graph | `wander-clean-video-cache`: LaMa and SegFormer |
| Camera solve | `wander-overnight-motion-cache`: Pi3X and SegFormer |
| Person reconstruction/animation and tracking | `wander-overnight-lhm-cache`: LHM model, human priors, tracker/sample/voxel assets, GFPGAN and DINO checkpoints |
| Object shape/refinement | `wander-image-to-3d-cache`: TRELLIS, SDXL and DINO checkpoint |

Cache checks establish nonempty file presence, not hashes, active model revisions, complete
sharded snapshots, license access or compatibility with a future image build. Inspect saved
staging receipts and [paid recovery](paid-recovery.md) before deliberately staging missing assets.
The doctor does not initiate staging or repair. Local reference-frame models may still need
their separately reviewed Torch/Hugging Face cache preparation before processing a source.

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
`POST /api/pipeline/doctor` inspects the API server locally; it does not establish that a separate
worker is ready. Uploads can be queued while worker setup is repaired. The API and supervisor
must share `WANDER_RUNS_ROOT`. `create_service()` accepts local artifact storage through
`WANDER_ARTIFACT_ROOT`; importing it no longer requires cloud configuration.

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
