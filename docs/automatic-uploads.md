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
