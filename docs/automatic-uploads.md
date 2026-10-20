# Automatic uploads

New dashboard uploads select `options.hands_off: true`. The API retains the original source
and publishes the run for the supervisor immediately. The browser can be closed after upload;
the server and supervisor must remain running. API clients can select the same option on
`POST /api/pipeline/runs/upload` or source-artifact submission. Existing runs and callers that
omit the option keep the manual workflow.

Configure the worker's credentials, model caches, authorized provider access and provider-side
spending controls once, following [pipeline setup](pipeline-setup.md). A supervisor restricted
with `--run-id` only processes that run; a service intended to process new uploads must watch
its configured runs root. Do not carry a source-specific paid-stage permit into a worker for
unrelated uploads. This change does not provision a hosted worker or turn a local API into a
public multi-user service.

## Worker for automatic uploads

Use a dedicated supervisor for the upload form:

```sh
uv run --locked --group inference python -m orchestrator.supervisor \
  --runs /absolute/path/to/shared/runs --uploads-only --watch
```

Keep that process running alongside the API. It discovers new automatic uploads on each pass,
including uploads submitted after startup. Manual and legacy runs remain with their existing
workers; paused and finished runs remain untouched. `--uploads-only --once` makes a single pass
over eligible uploads and exits. Adding `--run-id` further restricts the worker to that run.

Start the upload worker from its own configuration with the intended provider credentials,
spending controls, shared `WANDER_STAGE_LEDGER`, and any separately prepared VLM allowance.
The worker refuses `--uploads-only` when `WANDER_STAGE_AUTHORIZATION` is present, even if empty:
that setting grants a source-specific exception, which cannot authorize unrelated uploads.
Do not reuse an exhausted allowance or reset the shared ledger to enable a service. Provider
spending authorization is still required before enabling paid processing.

### Limit the number of uploads admitted

For a finite trial, add `--max-uploads 1`:

```sh
uv run --locked --group inference python -m orchestrator.supervisor \
  --runs /absolute/path/to/shared/runs --uploads-only --watch --max-uploads 1
```

This persistently limits how many distinct uploads this runs root can admit. An admission is
recorded before starting its agent. Restarting the supervisor uses the same history; failures
and cancellation do not refund a slot. A different limit is refused. Keep the runs root and its
admission history together; deleting or replacing that history is not a supported reset.

The bounded worker processes one admitted upload at a time. A paused upload retains its place.
A finished upload retains its place while detached stages remain unresolved. The next upload
can start only after that work settles and if unused admission slots remain. Missing or changed
run identity and malformed history stop admission instead of freeing a slot. Existing automatic
work that already started cannot silently be adopted into a new allowance.

While occupied, the worker reports availability only for its admitted run. Once its allowance
is exhausted, it no longer advertises availability for new uploads. The API can still store new
uploads with **Queue for later**; storing a video does not consume an admission or start inference.
Updated supervisors sharing this root respect its persisted policy even if they omit the limit;
they cannot process automatic uploads outside the bounded worker. Update all supervisors before
using this control. Manual work remains separate.

This is an upload admission count, not a dollar ceiling, provider-operation allowance, or total
runtime limit. It covers these supervisors; direct runner, `wander`, and provider invocations
remain outside it. Provider spending controls, the shared stage ledger, and the intended VLM
request allowance remain required. A one-upload allowance does not authorize a new paid run.

Worker presence records include their processing mode. An uploads-only worker is available for
`hands_off: true` submissions and runs, and unavailable for manual ones. On a bounded root, a
generic worker continues serving manual runs and reports that restricted capability. Update the
API with the worker so it understands that scope; older APIs ignore version-three records instead
of reporting a worker that cannot process their request. The updated reader still accepts
version-one unrestricted workers and version-two workers with automatic-upload scope.

The upload setup check verifies that a recent supervisor heartbeat exists for the API's runs
root and accepts new uploads. A worker scoped with `--run-id`, a one-pass worker, or a worker
watching another directory does not satisfy that check. When no matching worker is connected,
the form offers **Queue for later** and retains the upload for a worker to pick up. API clients
may still submit while workers are offline; submission stores a run, not a promise of immediate
execution.

The run page updates worker availability while processing. Heartbeats refresh every five seconds,
including during long agent turns, and expire after 45 seconds without an update. A disconnect
does not fail the run or repeat a provider call; detached stages may still finish. Restarting a
matching supervisor resumes the existing directory under the ordinary admission checks. Paused,
canceled and finished runs retain their existing state.

Worker status is a liveness signal, not proof of provider access, spending authorization,
available GPU capacity, or successful reconstruction. It requires the API and worker to share
the same runs directory. Older workers without these heartbeat records appear unavailable until
updated. The legacy `WANDER_HEARTBEAT` timestamp remains supported for process monitoring.

Automatic processing skips the human cleaned-frame pause with the existing `--no-gate`
option. The agent still inspects outputs and runs the existing reconstruction checks. It cannot
change the submitted run through extra stage flags or automatically retry a paid stage.
Existing provider ledgers and unresolved-operation guards continue to apply. A known provider
operation must be recovered separately rather than submitted again.

Once a paid stage that the package depends on fails or is blocked, the run cannot produce a
preview, because hands-off mode never repeats a paid stage. The next paid launch is then
refused and the run finishes `failed`, so it does not spend on a world generation or avatars
that can no longer be packaged. Stages already running finish and keep their outputs. While an
unpaid package prerequisite (such as `person_prep`) is failed, paid launches are held instead:
rerunning that step is still admitted, and paid work resumes only once it succeeds. Unpaid steps
use the same fixed defaults on a rerun, so a failure caused by the footage itself, such as no
fully visible person, repeats and the run ends without spending on generation.

The shared stage ledger counts executions per source SHA-256 across all runs, including
historical imports. A second execution needs an explicit new hypothesis, which a hands-off run
cannot supply. Uploading a video that was already processed therefore blocks at its first
previously executed paid stage. Use a source that has not been processed before. An upload
does not reuse earlier outputs or worlds.

When the agent needs a decision, exhausts its useful work, or stalls, the run records a final
failure with its reason instead of waiting for a question in the UI. Existing outputs and
evidence remain available. A readable packaged scene may be delivered as an **unaccepted
preview** alongside that failure. This is a delivery outcome, not a quality pass: the run
status remains `failed`, the failed checks remain visible, and no human approval is invented.
The final journal entry records `delivery.status`, `qualityStatus`, and the reason. The progress
endpoint rechecks that the preview files remain available.

`succeeded` still requires all existing stage, source, package, audio and quality checks. In
particular, [static-world acceptance](quality-judging.md) still requires a real manual review;
an automatically delivered preview does not satisfy that contract. Calibrated automatic quality
acceptance and autonomous geometry repair remain separate work. A fresh upload is therefore not
guaranteed to produce an accepted reconstruction.

Pause and cancellation still stop new work. Finished automatic runs require an explicit operator
resume before any new stage starts, and resuming does not restore a consumed paid attempt. Manual
mode retains explicit stage retries and records a resume when one reopens a finished run. Already
submitted provider work can finish and retain its output after the run stops.

The directory runner does **not** enforce a per-run dollar cap. Monetary-capped API submissions
continue to be rejected; the form does not pretend to set one. The attempt controls do not
replace provider spending controls or grant additional credit authorization.

Validation uses the real upload API, SQLite storage, runner and packager with simulated model
providers, plus tests for terminal outcomes, source identity, missing files, paid retry refusal,
pause/cancel and restart behavior. It does not establish fresh provider success or visual quality.
