# Paid stage accounting and recovery

The runner records a durable claim before each supported paid Modal subprocess. The default
ledger is `.context/pipeline-attempts.json`. Use the same absolute `--stage-ledger` path for
all candidates, aliases, and worktrees that must share an allowance. File locks coordinate
callers on that filesystem; separate machines or copied ledgers do not share ownership.
This is not a universal spending cap: direct worker invocations, model-cache staging, SSH
fine-tuning, OpenAI, and Marble generation are outside this ledger. Reported compute costs
are estimates, not independently verified bills.

The allowance is three executions per original source SHA-256, logical stage, and source selection
window. The clean, first-frame clean, and multi-image clean operations share the inpainting
allowance; individual LHM track suffixes have separate allowances. `--marble both` sequences its two
clean operations so their claims do not overlap. A candidate rename or new output path does not
create a new allowance. A retry needs a new `--stage-hypothesis` and changed relevant parameters or
worker code. Pending or unknown attempts block further submissions for that window until evidence is
reconciled; a pending window does not block the other windows of the same source.

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

## Source selection windows

A claim records the shot's source window in `parameters.sourceSelection` as `{start, end}` seconds
of the original recording, or `null` for the whole source. Each distinct window of one source and
stage carries its own three-execution allowance, so a multi-shot clip is processed shot by shot and
`--all-shots` runs every shot without a hypothesis for each first attempt. A second attempt on the
same window is still a retry: it needs a new `--stage-hypothesis` and changed relevant parameters or
worker code.

Windows are rounded to whole milliseconds, so float noise cannot invent a new window. Two windows
that overlap by more than half of the shorter one are the same unit of paid work, and their attempts
share one allowance; nudging boundaries by a few frames therefore buys nothing. A `null` window
covers every window of that source. Neighbouring shots that only touch at a boundary are separate
work. At most 24 distinct windows exist per original source and stage; beyond that the runner stops
and asks for the selected shots to be reviewed rather than claiming more. Marble generation stays
outside this ledger entirely, per window or otherwise, and must be accounted for separately.

A claim's `number` remains the per source and stage ordinal that older ledgers recorded; the added
`windowNumber` is the one to three ordinal inside its window. Saved ledgers written before windows
existed carry no `windowNumber` and stay valid.

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
This durable checkpoint path belongs to `modal_lhm.py`. Multiperson animation through
`modal_multiperson.py::animate` retains local outputs but does not yet produce these recovery
receipts; an existing destination therefore stops for manual inspection.

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
