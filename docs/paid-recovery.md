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
code. Each distinct clean operation can make its initial execution without a retry hypothesis,
but all consume the same three-execution allowance. Pending or unknown attempts block further
submissions for that source/logical stage until evidence is reconciled. Existing over-cap ledger
history remains readable and reconcilable; it never grants another execution.

The provider-agnostic `orchestrator/paid.py` policy likewise caps executions across run names
by original source and logical stage. It persists retry hypotheses on the associated stage attempt
and blocks unresolved claims before submission. Explicit run budgets can stop work sooner:
provider operation limits count every claim, and the cost ceiling reserves estimates across
all providers, including failed or unresolved operations. A capped cost budget requires an
estimate for every reserved operation. These are estimates, not verified invoices. This policy
only protects callers that use it; it does not add accounting to direct provider invocations.

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

The same deadline is available through `scripts/run_clip.py --execution-timeout 600`
with `--people` or `--all-people`, and through a tracked run's stage command:

```sh
wander step --wait lhm_motion_00 --execution-timeout 600
```

The runner forwards it only to tracked-person animation; it does not limit tracking, camera,
or canonical generation. Zero keeps the worker default. A nonzero value is rejected for
single-person animation and `run_clip.py --orchestrated` submission, which cannot carry this
stage parameter. The deadline bounds a function's execution time, not total provider spending;
retain the shared ledger, reviewed retry hypothesis, and provider spending controls.

For local tracked-person packaging, `run_clip.py --shared-scale auto|none|<track id>` forwards
the existing packager choice only to `package_people`. The default `auto` uses the registration
with the most supporting depth points; `none` preserves each track's scale. A reviewed explicit
track id preserves the intended body-height reference when retained fixed scales have no new
depth-fit point counts. The id must be within the configured `--people` slots (four with
`--all-people`) and have a completed animation. For an orchestrated tracked run, use
`wander step --wait package_people --shared-scale <track id>` for an unfinished package. If the
stage is already complete, add `--force package_people` to this command or the direct runner;
changing the flag alone does not rebuild cached output. Existing authored audio still blocks
replacement, so use a fresh candidate when preserving an authored package. Nondefault values are
rejected by single-person mode and `run_clip.py --orchestrated`, whose API submission cannot carry stage knobs.
This is local packaging, not a depth refit or a paid inference retry. Review the resulting scale
and placement before adoption.

## Deliver a retained tracked scene locally

When new packaging uses exactly the same source and person geometry, preserve the reviewed
room, person transforms, floor fit, original audio, object models and timing, and viewer
settings with a pinned config:

```sh
uv run --locked python scripts/retained_scene.py \
  --package /absolute/retained/public/worlds/example-4d \
  --world /absolute/retained/public/marble-example-clean.spz \
  --clip /absolute/source.mp4 --out /absolute/retained-scene.json

wander step --wait package_people --retained-scene /absolute/retained-scene.json
```

The first command hashes the supported manifests and their referenced assets without copying
them. The second runs the ordinary packager, validates the result, and delivers the retained
assets in the same local step. Use a **fresh candidate package destination** with completed
tracked-person animations and the correct camera inputs. The direct runner equivalent requires
`--people N --only package_people --retained-scene CONFIG --skip-finetune` and a world-enabled
Marble mode. The run must have finetuning disabled. It does not generate or fine-tune a world.
`run_clip.py --orchestrated` cannot transport this stage option; use `wander step` instead.

The retained scale reference is selected automatically; a conflicting explicit `--shared-scale`
is rejected. Compatibility requires the exact source hash, cast identities, sample grid,
camera matrices and effective alignment, canonical hashes, per-person scale and reframe, and
every freshly packaged PLY hash. Retained sequence manifests can come from compact archives
without PLY files; any retained PLYs that are present must match their recorded hashes too.
A mismatch stops publication. Config, source, and retained
asset hashes are checked again before applying the overlay. Existing package directories and
conflicting destination SPZs are preserved; changing the option does not rebuild an already
successful cached stage. Use another fresh candidate instead of overwriting an authored package.

This bounded path supports `wander.objects/2` and original audio, including a hash-pinned
original recording paired with a silent source clip. Spatial stems are not supported. It does
not rerun providers, invent audio, solve placement, or advance downstream quality checks.
`retained-delivery.json` records the copied hashes and `acceptance: none`. Placement, collision,
and frame-alignment diagnostic reports are excluded because their world provenance may be stale;
old placement, anchor, and verification state is never imported. The run remains subject to normal placement,
object, quality, and completion gates. Inspect the delivered viewer before adoption.

## Repair incomplete saved-track output

A sparse saved-pose input retains its full source grid. Exporting all intended non-null seeds does
not mean the full recording was reconstructed: the recovery manifest stays partial and the local
command exits with that status after downloading outputs. Validate exact intended sample IDs and
hashes before explicitly merging them with retained outputs. The wrapper never fills missing poses.

For a reviewed missing detection, `--recover-missing-poses` explicitly enables new pose inference
from those source frames while retaining every non-null saved pose. This is a new paid candidate,
not output recovery: use a fresh destination and the same source ledger, and supply
`--fixed-world-scale` from the retained registration so the repair cannot refit the body's scale.
The default remains saved-track-only animation. Review the recovered person's identity, projected
joints and continuity against the source before adoption, especially for a leading gap where no
previous pose can guide association. Complete coverage and hash validation still apply; this flag
does not fill gaps by copying or interpolating poses, and a remaining gap still fails completeness.

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
