# Paid stage accounting and recovery

The runner records a durable claim before each supported paid Modal subprocess. The default
ledger is `WANDER_STAGE_LEDGER` when configured, otherwise `.context/pipeline-attempts.json`.
An explicit `--stage-ledger` overrides that setting. Use the same absolute ledger path for
all candidates, aliases, and worktrees that must share an allowance. File locks coordinate
callers on that filesystem; separate machines or copied ledgers do not share ownership.
This is not a universal spending cap: direct worker invocations, model-cache staging, SSH
fine-tuning, OpenAI, and Marble generation are outside this ledger. Reported compute costs
are estimates, not independently verified bills.

For a bounded run, `WANDER_VLM_BUDGET` can point to an existing private JSON file with
`schema: "wander.vlm-budget/1"`, positive integer `maxCalls` and `maxCompletionTokens`, an
`allowedModels` list, and an initially empty `calls` list. Calls through
`worker/stages/vlm_judge.py` reserve a request in that file before network access, include the
completion-token limit, and do not retry. Duplicate request bodies are refused across processes
and resumes, including after a failed or ambiguous response. Keep the same file for the whole
run; recreating it would lose its accounting. Recorded usage excludes prompts, images, keys,
and provider error text. This bounds requests and completion tokens for that helper, not input
tokens, monetary charges, or other OpenAI clients. Without the environment variable, calls remain
unbudgeted; the HTTP 429 handling below still applies.

Source descriptions use the existing GPT-6 Astra model with explicit `low` reasoning for factual
frame extraction, one request, and a 300-second HTTP socket timeout. Explicit alternate models keep
their provider reasoning default. The output records `model`, `reasoningEffort`, and `timeoutSeconds`
on success or request failure. The socket timeout is not an overall run deadline. A timeout still
consumes its claim; this setting does not authorize a retry or relax duplicate-request protection.

Bounded failures retain the numeric HTTP status and finite recognized provider error-code, type,
and parameter identifiers when available. HTTP error bodies are read once, up to 16 KiB; raw bodies,
messages, headers and arbitrary strings are never retained. The returned `errorGuidance` is static
operator guidance derived only from these sanitized identifiers, and is preserved in source-prompt
error output. A failed claim still consumes its allowance; neither fixing credentials nor learning
the error cause refunds or resets it.

The [OpenAI API error guide](https://developers.openai.com/api/docs/guides/error-codes) distinguishes
`credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`,
`organization_usage_limit_exceeded`, and `insufficient_quota` from request-rate errors. Inspect the
identified organization's/project's credit balance or corresponding limit before another request.
`rate_limit_exceeded` and `slow_down` instead direct the operator to reduce request rate and wait.
Known `error.type` values are retained separately as `providerType`, without relabelling them as
error codes. Unknown codes/types are omitted. An HTTP 429 with no recognized cause remains unknown;
it is not classified as billing exhaustion or a transient rate limit from its status alone.

Bounded calls never retry, including recognized request-rate errors. Unbudgeted calls stop immediately
on billing-related or unknown HTTP 429 errors; only explicitly recognized `rate_limit_exceeded` or
`slow_down` codes retain the caller's finite retry allowance and numeric `Retry-After` backoff. Other
unbudgeted retry behavior is unchanged. These diagnostics do not authorize another request or repair
historical receipts whose provider response was already discarded.

The allowance is three executions per original source SHA-256 and logical stage. The clean,
first-frame clean, and multi-image clean operations share the inpainting allowance; individual
LHM track suffixes have separate allowances. `--marble both` sequences its two clean operations
so their claims do not overlap. A candidate rename or new output path does not create a new
allowance. A retry needs a new `--stage-hypothesis` and changed relevant parameters or worker
code. Each distinct clean operation can make its initial execution without a retry hypothesis,
but all consume the same three-execution allowance. Pending or unknown attempts block further
submissions for that source/logical stage until evidence is reconciled. Existing over-cap ledger
history remains readable and reconcilable; history alone never grants another execution.

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
Full-video cleaning also supports the named same-run candidates described below.
A successful subprocess is recorded as a completed execution, not a visual quality verdict.
Additional selected shots share their original source's allowance and are treated as retries of
the same stage. Automatic `--all-shots` processing can therefore stop after the first shot;
review subsequent shots separately with distinct hypotheses instead of changing the source hash.

## Bounded polling of an existing Modal call

In the pinned Modal 1.5.5 client, a `FunctionCall.get(timeout=...)` poll with no result raises
Python's built-in `TimeoutError`. It is distinct from `modal.exception.TimeoutError` and
provider execution failures. Catching only the provider class can accidentally exit `app.run()`
and stop a healthy paid job when the first short poll expires.

Direct diagnostic launchers can use `scripts/modal_polling.py::wait_for_call` after recording
the existing call ID. It polls that same call within a finite overall timeout, continuing only
after an exact built-in polling timeout whose traceback identifies the pinned SDK's local poll.
Unknown origins or a changed SDK fail closed; remote timeouts, cancellation and other errors propagate.
At the overall deadline it invokes the supplied cancellation callback once, then fails. The caller
must supply a callback that cancels only its owned call and records that request; the helper does
not establish ownership, submit work, retry a generation, or settle the paid ledger. Retain and
recover outputs and provider state before reconciling an interrupted attempt.

## Optional single-person worker resource ceilings

Set `WANDER_WORKER_TIMEOUT_SECONDS=600` in the launching process to limit the GPU functions
`modal_clean_video.py::clean`, `modal_motion.py::reconstruct`, and `modal_lhm.py::frozen`
(including its canonical animation mode). This is a registration-time setting: use a fresh Modal
invocation after changing it. It must be a positive integer no greater than that function's
existing maximum: 3600 seconds for cleaning, 1500 for camera/motion reconstruction, and 1800 for
LHM frozen or animated inference. A shared value for all three must therefore be at most 1500.
Empty, malformed, zero, negative, or excessive values fail before image or volume setup.
Omitting the variable preserves the existing resource requests, concurrency and timeouts.
The direct runner records the effective resource policy in each new paid claim, including
defaults when no limit is set. Changing a deadline changes the attempt identity; unrelated
environment values and credentials are excluded. Earlier claims remain unchanged.

When configured, the function decorator sets both execution and startup timeouts to that value,
CPU request/limit to `(4, 4)`, memory request/limit to `(32768, 32768)` MiB for cleaning and motion
or `(65536, 65536)` MiB for LHM, one maximum container, no automatic retries, and a two-second
scaledown window. The small nonsecret timeout setting is forwarded to containers so their module
reimport uses the same policy. Startup and execution are separate deadlines; the setting does
not bound their combined wall time, image builds, finalization, or unrelated workers. It is not
a universal dollar cap and does not replace the paid ledger or provider spending controls.
The model, precision, input sampling, inference arguments and output-quality requirements stay
unchanged. CPU staging functions and tracked multiperson workers retain their existing policy.

Bounded single-person LHM uses the existing logged-subprocess deadline guard to stop its own
inference child before the provider deadline, reserving up to 60 seconds (10% for short limits)
for output hashing and volume commit. Partial outputs are finalized as failed, never retried
or accepted as complete. Hard provider termination or failed finalization can still leave an
unresolved checkpoint; recover existing outputs instead of resubmitting. Cleaning and motion
use the provider deadline without a durable-output checkpoint, so a timeout can lose their
return artifacts. Inspect provider status and the existing attempt before authorizing any retry.

Offline checks in `scripts/tests/test_worker_resources.py` cover default preservation, invalid
settings before worker setup, actual registration against the locked Modal SDK, packaged helper
imports, and failed-checkpoint preservation for both bounded LHM modes. They do not build cloud
images or establish runtime cost or inference quality.

## Explicit one-run operator authorization

After explicit operator approval, the direct runner accepts `--stage-authorization /absolute/permit.json`
(default `WANDER_STAGE_AUTHORIZATION`). The runner never creates a permit. A reviewed permit may
authorize unchanged parameters or an execution above the normal cap, without resetting history.
It does not bypass unresolved claims, retained-output checks, quality gates, or other provider budgets.
It applies only to this runner's supported paid Modal operations; `--orchestrated` rejects it.

The JSON must contain exactly these fields:

- `schema`: `wander.paid-stage-authorization/1`.
- `id`: a unique identifier (a UUID is suitable).
- `sourceSha256`: the canonical original source's lowercase SHA-256, even when submitting an alias.
- `candidate`: the exact runner candidate name.
- `reason`: the reviewed reason for this execution.
- `operatorApproval`: the operator's approval verbatim.
- `expiresAtEpoch`: a finite positive Unix timestamp, later than the claim time.
- `operations`: a nonempty object mapping concrete operation names to nonnegative integer counts
  of **prior logical-stage executions** expected at submission. For example, `{"clean_first": 3,
"clean": 4}` authorizes those two operations in order against the shared inpainting count;
  it does not authorize a retry of either one. Inspect the actual shared ledger when setting counts.

Each claim reads and validates the permit under the ledger lock, records its exact file hash,
absolute path, and full contents, and consumes that ID's operation immediately. Failed and unknown
executions still consume it. Reconciliation never restores it. Reusing an ID with modified file
contents is rejected, including attempts to change counts or add operations. Keep the same permit
file unchanged while its distinct approved operations run. Candidate/source mismatches, expired or
malformed permits, unlisted operations, and stale counts stop submission; they never fall back to
the ordinary allowance. The shared ledger remains required across candidates, aliases, and worktrees.

## Retain a corrected full-video clean within the same run

Before any Marble submission for the run, `--clean-candidate <slug>` writes a fresh full-video
clean under `clean-candidates/<slug>/` in that run. The slug uses 1–64 letters, digits, underscores
or hyphens and starts with a letter or digit. Existing candidate directories are retained and
refused. This option supports `--marble video` or `none`; it is rejected with still-image modes,
`--orchestrated`, `--reuse-world`, or retained-world delivery. If `--only` is used, it must include
`clean`. Use `--force clean` when replacing a cached successful clean; changing the slug alone
cannot silently reuse that cached result.

The fresh directory keeps the video, first frame, output masks, worker report, log, and any locally
generated tracking masks. Original outputs remain unchanged. The ordinary shared paid ledger and
retry/authorization rules still apply to the concrete `clean` operation and the same runner name;
a stage-candidate slug grants no new execution allowance.

After a successful clean with a matching source/video report, the runner retains a candidate
`selection.json` and atomically writes `clean-selection.json` at the run root. The receipt pins
the actual source SHA-256 and each selected artifact's path and SHA-256. Subsequent review and
Marble commands use that selection without needing the candidate flag again. Missing, modified,
or escaping artifacts and inconsistent receipts fail closed. A failed candidate leaves the prior
selection intact, while the ordinary failed-stage/dependency state continues to block progression.
Normal reruns invalidate downstream results; successful cleaning still requires visual review.

Recorded Marble requests, uncertain submissions, operations, logs, or generated worlds block
selection replacement before another clean is paid for. A run lock also serializes new Marble
submissions with candidate cleaning. Existing operation polling and explicit world recovery remain
read-only and do not require a new clean. This path cannot replace the input of an already-paid world.

## Import an offline CPU clean without another paid claim

`scripts/local_clean_video.py` can use an explicitly cached TorchScript LaMa file on CPU,
without a downloading wrapper or provider client. Its diagnostics are not an accepted clean.
A pipeline import requires a completed, **full-selection** helper run made with `--encode`,
plus the original exact-source packed mask archive. A sparse first-frame or diagnostic subset
cannot be imported as a full video, even if every selected frame completed.

Import into a new same-run candidate, before any Marble submission:

```sh
uv run --locked python scripts/run_clip.py \
  --clip /absolute/path/to/source.mp4 --name existing-run --fps 12 \
  --marble video --no-objects --skip-finetune --no-publish \
  --only clean --force clean --clean-candidate local-cpu-v1 \
  --import-local-clean /absolute/path/to/offline-output/clean.json \
  --clean-masks /absolute/path/to/original-exact-source-masks.npz \
  --stage-ledger /absolute/path/to/shared/pipeline-attempts.json
```

The import does not load a model, launch inference, create a paid claim, or erase previous
paid history. `--force clean` is required when cleaning is cached. Import is restricted to
`--only clean`; omit the import/candidate/masks flags on later resumes. The existing selection
lock and Marble-history guard reject replacement after a known or uncertain world submission.
Original candidates and failed-import evidence remain retained, and candidate names cannot
be reused.

The runner checks source SHA, dimensions and FPS, requires every sample in the locally observed
full FFmpeg FPS mapping, re-decodes those exact source frames, and validates each PNG's encoded
and RGB hashes, input-mask hash, source/outside RGB hashes, and measured outside-mask preservation.
It checks the silent video's hash, decoded frame count, dimensions and FPS. Missing, changed,
symlinked or incomplete inputs fail before selection. This first import path intentionally excludes
alternate tracker sampling grids and upsampled duplicate-source grids. Decoder-version RGB
differences also fail closed; a source hash and timestamp match do not prove identical decoded RGB.

The new candidate retains all PNGs, the original mask archive and video. `clean.log` retains the
original helper JSON byte-for-byte and is pinned by the existing selection receipt; `clean.json`
adds an explicit `provider: local` compatibility report. Stage state records imported local work,
with no inference during import. The scheduler invalidates downstream review/verification;
import grants no visual acceptance, paid retry allowance, or world-generation approval. Inspect
the full output and perform the ordinary separate `review --gate-pass` stage only when warranted.
Existing LHM output directories retain their recovery-only behavior; importing cleanup does not
repair or authorize a failed person-reconstruction attempt.

### Optional measured source patches

The CPU helper accepts `--source-patches /absolute/recipe.json` to run LaMa normally and
then compose measured source-observed background into the removal mask. It does not estimate
transforms, load another model, expand the removal mask, or establish quality acceptance.
Omitting the option preserves the existing isolated LaMa path. Run into a fresh output directory;
a previous diagnostic postprocess cannot be relabelled as a new helper execution.

A data-only `wander.source-patches/1` recipe contains exactly `schema`, `sourceSha256`,
`sourceTimeBase`, `sourceIndices`, `sourcePts`, `fps`, `width`, `height`, and `patches`. The source
fields must match the entire exact-source mask grid. Each patch contains exactly `donorSample`,
`targetSample`, `homography`, `donorRoiXYXY`, `featherPx`, and `maskMarginPx`. Samples are archive
sample IDs, not decoded source ordinals. This initial schema permits one patch per target,
bounded integer XYXY donor rectangles, an eight-pixel inward feather and eight-pixel donor-mask
margin. Matrices must be finite nonsingular 3×3 donor-to-target homographies normalized to
`H[2][2] = 1`, without a projective pole through the ROI. Rectangles overlapping the dilated donor
removal mask are rejected. All recipe checks run before model loading; donor samples needed by
a selected target are decoded even when omitted from `--only`.

The helper warps the observed donor RGB, rectangle and valid unmasked support with bilinear
interpolation, requiring full interpolation coverage. It intersects this support with the original
target removal mask, feathers inward over eight pixels, and leaves all other LaMa pixels intact.
The receipt identifies **LaMa plus measured source-guided composite**, embeds the complete recipe
and its canonical JSON SHA-256, and also pins the original recipe bytes as `source-patches.json`.
Existing helper code hashes cover the runtime algorithm. Each recipe target retains a
`lama_<sample>.png` pre-patch image with byte/RGB hashes, plus donor provenance, support hashes,
coverage and preservation counts. Empty support is recorded honestly and retains the LaMa base.

Import revalidates the recipe/source grid and original recipe bytes, verifies the pre-patch PNGs
and their unchanged outside-mask pixels, then replays the measured composite from those bases
and decoded source donors. The final pixels and source-patch evidence must match exactly. It
copies the recipe and bases along with the other evidence. This verifies the composition, not a
re-execution or independent certification of LaMa inference. Human visual review remains required;
parallax, incorrect measured transforms, uncovered LaMa artifacts and temporal seams can still fail.

## Reuse a local room when its provider ID is unavailable

The direct runner accepts `--import-local-world /absolute/receipt.json` only with
`--marble video --only marble_video`, without another import/reuse mode or source-shot selection.
This is a local recovery route: it skips cut scoring, source prompts and cleaning, creates no paid
claim or provider request, and disables automatic remote archiving for that invocation.

```sh
uv run --locked python scripts/run_clip.py \
  --clip /absolute/path/to/original.mp4 --name existing-run \
  --marble video --only marble_video --skip-finetune --no-publish \
  --import-local-world /absolute/path/to/receipt.json \
  --stage-ledger /absolute/path/to/shared/pipeline-attempts.json
```

The data-only receipt must contain exactly:

- `schema: "wander.local-world-import/1"` and the original clip's `sourceSha256`.
- `world: {"path": "/absolute/retained.spz", "bytes": <positive integer>, "sha256": "<digest>"}`.
- A nonempty `associationEvidence` list of `{"path": "/absolute/evidence.json", "sha256": "<digest>"}`.
  At least one pinned JSON document must contain both the original-source and world hashes,
  recording their retained association. Co-packaging does **not** prove which input generated a world.
- `generationLineage: "unknown"`, `worldId: null`, `operationId: null`,
  `execution: "imported_local"`, and `freshGeneration: false`.

Before mutation the importer checks source/evidence/SPZ sizes and hashes, rejects symlinks, and
uses the existing CPU SPZ decoder to validate complete, nonempty finite v2 degree-zero geometry.
It refuses retained/pending/uncertain Marble submissions and conflicting assets. A successful import
copies only the SPZ into this run's standard `marble-<name>-clean.spz`, retaining a separate
`local-world-import.json`; it never fabricates provider generation metadata. Interrupted imports
can resume from the same pinned receipt. Reimports verify that receipt and asset instead of
submitting or replacing anything. New generation is blocked while the local import is retained.

Only the explicit import invocation removes the world stage's clean/review/prompt dependencies.
Normal generation admission is unchanged. Failed clean records and the shared ledger remain
unchanged, and downstream scale, placement, anchors and verification are invalidated. Later work
must select its intended unfinished stages explicitly; this import does not approve a failed clean,
complete packaging, adopt old people/cameras/placement, or transfer a quality verdict. Fresh person
packaging and registration checks still apply, and manual whole-scene review remains required.
The room is labelled saved local input with unknown generation lineage, never a fresh generation.

Single-person packaging defaults to unreviewed lower-body visibility and labels hidden/off-frame
geometry as inferred. For a measured source-specific statement, pass
`run_clip.py --lower-body-provenance "<observed visibility and inference limits>"` when running
`package`; it forwards to the packager's existing `--lower-body` option. This flag is limited to
direct single-person runs and requires `--force package` if packaging is already cached.
It records provenance text, not a visibility measurement or quality acceptance.

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

## Prepare optional local person-instance clean masks

`scripts/instance_clean_masks.py` creates masks on CPU using only an already cached torchvision
Mask R-CNN checkpoint. It never downloads weights or submits inference to a provider. The model
and backbone constructors disable pretrained downloads; model loading and prediction run with
socket connections blocked, four Torch CPU threads, and a checked checkpoint hash. If packages
are already installed, use offline dependency resolution too:

```sh
uv run --locked --offline --group inference python scripts/instance_clean_masks.py \
  --clip /absolute/source.mp4 --fps 12 --dilate 2 \
  --out .context/evidence/instance-masks/new-candidate --contact-sheets
```

The output directory must be new. The default checkpoint is
`maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` in the Torch hub checkpoint cache; `--weights` can name
an existing copy. A missing or mismatched checkpoint stops without fetching a replacement.
All COCO person instances above score 0.5 are unioned at mask probability 0.5, followed only by
small symmetric padding (`--dilate`, default two pixels). There is no semantic-class union,
whole-column extension, or moved-content/shadow union. Zero supported people, an empty supported
instance, or malformed model output fails closed—even on a genuinely empty frame. This helper
does not silently fill missing detections or offer an empty-frame override.

A successful directory contains `masks.npz`, `report.json`, raw per-frame instance archives and
optional contact sheets covering every sampled frame. The report records parameters, model and
generator hashes, scores/boxes, and exact FFmpeg source indices, PTS, timebase and source hash.
A failed run retains partial evidence and a failed report; it does not publish `masks.npz`.
Review all samples, including hands, motion blur, reflections, occlusions and nearby fixtures.
Padding can still erase a fixture beside a person, and the model does not guarantee carried-object,
shadow, identity or temporal coverage. Mask completion is not accepted inpainting quality.

The bounded lobby trial with default two-pixel padding preserved the checked fixtures but left
prominent person-shaped remnants after LaMa. It failed the full-video review. This helper is an
optional mask preparation and inspection tool, not a validated replacement cleaning algorithm;
see [known limits](known-limits.md) before spending on another candidate.

The resulting archive uses the existing direct runner's `--clean-masks` contract, which forwards
it to the worker's `--masks-in` path and bypasses the worker's semantic masks and legacy growth.
Keep the original FPS and source clip. This option changes no current pipeline default and does
not authorize another paid clean, reset the shared inpainting allowance, or replace a failed
quality verdict. Inpainting remains subject to the same ledger, approvals and visual review.

## Experimental isolation of inpainting context

The direct runner and clean worker accept `--inpaint-mode isolated`; `legacy` remains the default.
The orchestrator's `wander step` clean steps expose the same `inpaint_mode` parameter;
`run_clip.py --orchestrated` rejects a nondefault mode rather than discarding it. This experimental mode
removes masked RGB before resizing, conservatively resizes mask coverage, and grows only the
model's context mask to cover the Lanczos filter support. It then writes the generated fill at
full opacity inside the original removal mask and preserves source pixels outside that mask
before video encoding (lossy H.264 encoding can still change them).
The wider model mask does not enlarge the output write region. Upsampling scales the filter
margin per axis. Repeated passes neutralize the same region again, rather than refine an earlier fill.

Offline mutation checks on eleven retained lobby frames found 40,744 changed low-resolution
pixels outside the old nearest-neighbor mask after changing only masked RGB. The isolated
preparation eliminated that dependence. Lossless first-frame evidence also places major ghosts
inside fully opaque fill regions, so final-alpha changes alone cannot explain or repair them.
A subsequent explicitly approved first-frame trial removed most large hair, torso and trouser
remnants, but visible outlines and shoe remnants remained. It failed visual review; the full clip
was not rerun. The existing mask still misses some fingers and reflected people, and hard mask
boundaries can introduce visible seams. Keep the mode experimental and do not promote the failed
clean or generate a world until the actual output meets the unchanged review criteria.

Worker reports identify `inpaintMode`, `maskSource`, the supplied archive hash, and applied mask
growth separately from requested growth. The helper and mode are included in paid-attempt
identity; neither this option nor a new candidate name grants another execution. Deterministic
stub-model tests cover signal isolation, exact outside preservation, thin masks, upsampling,
legacy pixel parity, source timing and paid-command forwarding without inference or provider calls.

Isolated-mode workers now compare every selected frame against the same worker's decoded RGB
outside the removal mask, before writing the output PNG or encoding video. Any changed pixel
fails the run. `compositeChecks` records source index/PTS, decoded RGB and removal-mask hashes,
the number of outside-mask pixels, changed-pixel count, and before/after outside-RGB hashes.
Hashes use row-major bytes: interleaved uint8 RGB for images, one byte (0/1) per mask pixel, and
only the selected outside RGB triples for the outside hashes. Input bindings and completed
checks survive a later failure; a failing frame is not written or returned. A fully masked
frame has zero outside pixels and proves nothing about preservation of surroundings.

`keptFrameProvenance` also records each saved frame's decoded RGB hash. Legacy mode retains its
existing compositing behavior and has no outside-mask checks; an empty or missing check list is
not a preservation pass. These receipts describe pixels before lossy encoding, do not establish
mask coverage or visual quality, and do not retroactively verify older outputs. Comparing with
a separate local decode can produce differences caused by the decoder itself; use the worker
receipt for this exact-byte check and retain independent visual review.

### Retained single-person frozen candidates

A direct single-person retry can use `--only lhm_frozen --lhm-frozen-candidate SLUG`
(and `--force lhm_frozen` when that stage is cached). This retains earlier failed outputs and
receipts. Each fresh candidate pins the exact clip and three `prepared-person` inputs before
calling the existing `lhm_frozen` paid-stage guard; it does not reset the shared allowance or
grant authorization. A retained candidate is recovery-only, including when its first submission
failed or was interrupted. Use a fresh slug only for a justified, authorized new execution.

Successful recovered artifacts are checked against their complete manifest and input pins before
an atomic selection receipt is published. Single-person motion uses that validated selection;
changing it is blocked after retained motion or running/successful motion or packaging. Missing
selection history, changed inputs/outputs, partial exports and symlinks fail closed. Selection
certifies artifact identity/completeness, not visual quality or acceptance.

### Explicit LHM GPU scheduling choice

`WANDER_LHM_GPU=L40S` requests one L40S for the single-person LHM worker; absence keeps
one L4. Only exact `L4` and `L40S` values are accepted, before image/volume setup. Both match
this image's CUDA SM8.9 build. This is one selected GPU, without a fallback list or an automatic
retry. The setting is forwarded to the container and included in the shared paid-attempt
fingerprint, even for default L4. Changing it does not grant another execution or reset counts.
Model, precision, prepared inputs, CPU/memory requests and configured deadlines stay unchanged.

The worker records `requestedGPU`, its GPU rate, the combined compute rate, and an estimate
for its measured worker interval. Rates used here are L4 $0.000222/s or L40S $0.000542/s,
plus four CPU cores at $0.0000131/core/s and 64GiB at $0.00000222/GiB/s: combined
$0.00041648/s or $0.00073648/s. The estimate excludes container startup, final output hashing
and volume commit, so it is not an invoice or a universal dollar cap. Execution/startup
limits do not establish an end-to-end queue-wait deadline. Review scheduling evidence and
spending authorization separately before any changed-GPU submission.
