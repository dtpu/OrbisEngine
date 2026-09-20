# Paid stage accounting and recovery

The runner records a durable claim before each supported paid Modal subprocess. The default
ledger is `.context/pipeline-attempts.json`. Use the same absolute `--stage-ledger` path for
all candidates, aliases, and worktrees that must share an allowance. File locks coordinate
callers on that filesystem; separate machines or copied ledgers do not share ownership.
This is not a universal spending cap: direct worker invocations, model-cache staging, SSH
fine-tuning, OpenAI, and Marble generation are outside this ledger. Reported compute costs
are estimates, not independently verified bills.

The allowance is three executions per original source SHA-256 and logical stage. The clean,
first-frame clean, and multi-image clean operations share the inpainting allowance; individual
LHM track suffixes have separate allowances. `--marble both` sequences its two clean operations
so their claims do not overlap. A candidate rename or new output path does not create a new
allowance. A retry needs a new `--stage-hypothesis` and changed relevant parameters or worker
code. Pending or unknown attempts block further submissions until evidence is reconciled.

The runner hashes the original `--clip` by default, including when processing a derived shot.
For independently encoded aliases or pretrimmed inputs, use a reviewed canonical
`--source-sha256`, or record verified `{aliasSha256: canonicalSha256}` entries in the ledger's
`sourceAliases` object before claims exist for the alias. The runner cannot infer that different
bytes depict the same original recording. Review prior receipts before using this guard for an
existing source: a fresh ledger does not establish that no earlier jobs were paid for.

`--force` schedules a stage again but does not delete paid outputs or bypass accounting.
Existing result paths are retained. LHM destinations with compatible recovery receipts use
read-only recovery; legacy or incompatible destinations stop for inspection. Other paid stages
need retained-output inspection and, when justified, a fresh candidate with the same ledger.
A successful subprocess is recorded as a completed execution, not a visual quality verdict.

If the user explicitly authorizes one exception after three terminal executions, author tooling
can call `StageAttempts.authorize_one_extra(source, operation, evidence, reason)`. This records
a hash-bound approval artifact, canonical source, logical stage and the prior three attempt IDs.
Only that scope may execute a fourth time; a fifth, reused approval, unresolved prior execution,
or changed approval file is rejected. The ordinary limit remains three, history is never reset,
and this is an operator-recorded authorization rather than authenticated user identity. It does
not raise the spending cap or authorize a different source/stage. Older clients that cannot read
a fourth claim fail closed; use the updated accounting code in every process sharing that ledger.

Additional selected shots share their original source's allowance and are treated as retries of
the same stage. Automatic `--all-shots` processing can therefore stop after the first shot;
review subsequent shots separately with distinct hypotheses instead of changing the source hash.

## Recover LHM outputs without inference

The native LHM worker checkpoints generated files and a hash manifest in its Modal volume before
returning a small receipt. The local `recovery-receipt.json` retains the remote path and call ID
when available. A submission with an uncertain outcome is never automatically repeated. The
receipt lock prevents concurrent local submission and recovery; an active owner causes a busy
error, so preserve the receipt and retry recovery after that owner finishes. Do not delete lock
files to bypass ownership.

```sh
uv run --locked python worker/stages/lhm_recovery.py \
  --receipt .context/run/example/lhm-frozen/recovery-receipt.json
```

This command only reads the existing Modal volume. It verifies sizes, hashes, and output coverage,
reuses matching local files, and refuses conflicting local files. Each transfer owns its temporary
file. Failed or partial outputs remain available with explicit status and are not promoted to a
complete reconstruction. Original prepared inputs remain outside the generated-output checkpoint.
A running or unavailable checkpoint remains unresolved; recovery does not submit inference.
Both `modal_lhm.py` and `modal_multiperson.py::animate` use this checkpoint and recovery path.
Multiperson animation records input hashes before submission and the call ID before waiting.
An existing output destination stops before another submission; recover its receipt instead.

Saved-track animation accepts `--fixed-world-scale` to retain a measured positive scale below 10
without fitting depth again. `--execution-timeout` selects a provider limit of 1–3600 seconds;
omitting it retains the existing 3600-second default. Explicit limits also cap CPU at four physical
cores and memory at 64 GiB. The subprocess stops before the provider deadline to reserve up to
60 seconds for output hashing and volume commit. Hard kills or failed commits can still leave an
unresolved checkpoint and must not trigger resubmission.

Native `modal_lhm.py` accepts `--gpu L4|H100` and `--execution-timeout 1..1800`;
its defaults remain L4 and 1800 seconds, with four CPU cores and 64 GiB memory.
Invalid selections stop locally before receipt creation or submission. Both GPU selections use a shared derived
image that rebuilds PyTorch3D, diff-gaussian-rasterization, and simple-knn for CUDA
`8.9;9.0+PTX`, reusing cached base layers. The first launch on either GPU needs CPU
image-build time in addition to its GPU execution allowance. The worker checkpoints its detected CUDA device
name before inference and refuses a device mismatch. Receipts record selected resources and
input hashes; output reports include selected and detected GPU and estimated compute cost.
The estimate uses [Modal pricing](https://modal.com/pricing): L4 $0.000222/s or H100
$0.001097/s, plus four CPU cores at $0.0000131/core/s and 64 GiB at $0.00000222/GiB/s.
Image builds, final output hashing and volume commits are outside the reported inference estimate.
The inference child reserves up to 60 seconds within the selected provider timeout for
checkpoint finalization. Use `modal run --detach worker/modal_lhm.py ... --gpu H100` when
remote execution must survive CLI disconnection; retain the receipt for read-only recovery.
Direct Modal commands remain outside the runner ledger unless the caller explicitly claims them.
The runner's paid-command identity accepts `modal run --detach` without granting a new
execution identity or retry allowance. The optional `lhm-larger` CPU cache staging phase
downloads the pinned 1B checkpoint alone, retaining the existing 500M model and priors.

A sparse saved-pose input retains its full source grid. Exporting all intended non-null seeds does
not mean the full recording was reconstructed: the recovery manifest stays partial and the local
command exits with that status after downloading outputs. Validate exact intended sample IDs and
hashes before explicitly merging them with retained outputs. The wrapper never fills missing poses.

Tracked animation selects the earliest sample with both a retained pose and declared person
depth. `--registration-sample` binds that reference to its own source camera, source index and
track ROI; it does not discard earlier poses. A one-pose prepass computes the same fixed median
depth ratio before chronological export. Missing declared files or mismatched source records
stop before submission. The option cannot be combined with `--fixed-world-scale`, and unset
standalone calls retain their previous first-pose registration behavior. This resolves missing
opening depth, not bad poses, appearance, floor contact or camera reconstruction.

After inspecting saved provider or recovery evidence, reconcile the corresponding ledger claim:

```sh
uv run --locked python scripts/stage_attempts.py \
  --ledger /absolute/shared/pipeline-attempts.json --reconcile ATTEMPT_ID \
  --status completed --evidence /absolute/path/recovery-manifest.json \
  --reason 'Inspected the finalized complete recovery manifest'
```

Use `failed` only with evidence of a terminal failure. Reconciliation records the evidence hash
and preserves the execution count. It does not submit work, verify invoices, or prove that the
supplied evidence justifies the operator's conclusion. An ambiguous result must remain pending
or unknown.

## Offline checks

`test_stage_weights.py`, `test_lhm_recovery.py`, `test_run_clip_recovery.py`, and
`test_stage_attempts.py` use local fixtures and mocked provider interfaces. They cover receipt
ownership, interrupted transfers, source-alias claims, unknown outcomes, retained outputs,
and no automatic resubmission. They do not build cloud images or stage models.
