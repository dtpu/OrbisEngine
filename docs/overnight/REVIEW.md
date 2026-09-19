# Overnight integration review

Keep the measured improvements as separate changes. None of the three branches currently
justifies replacing all of `main`, and a published world is not a completed reconstructed replay.
Austin's gym world is a visual improvement in the tested front and sideways views and a candidate to retain;
James's compact person animation gives the clearest measured delivery improvement. The compact
packages are opt-in review paths; the default demo wrapper does not forward arbitrary person-path
overrides, so opening James's ordinary demo alone does not reproduce the compact-package comparison. Daniel has
valuable reconstruction and paid-job recovery work, with different integration requirements.

## Versions and evidence

Baseline viewer: `00ab4f5`. Initial audits: Daniel `6ea996f`, Austin `6b023d5`, James `a45f4fd`.
Later deltas inspected: Daniel `ffe0700`, James `3414228`. Daniel's newer tip fixes its earlier
TypeScript failure and removes progressive pose streaming; its typecheck and Vite bundle pass.
James's later commits change the pipeline, not the viewer tested below. Documentation-only main
changes do not change the baseline viewer.

Independent checks include 222 focused Daniel/Austin Python tests, 50 focused James tests,
build/type checks, additional frontend/object-timing tests, mocked failure/race reproductions,
and Playwright captures of the actual built viewers with private S3 assets. Passing checks are
not universal: both gzip motion test suites fail on the installed supported-minimum Bun 1.2.21
because `DecompressionStream` is unavailable there. Daniel also leaks a mocked global `fetch`
between tests; its storage suite passes independently. Fix tooling compatibility and isolation
before integrating those suites; do not weaken them.

Browser evidence and detailed branch audits live outside Git in
`.context/evidence/branch-comparison/`. Reproduction/build sources are in `.context/comparison/`.
The private asset snapshots are immutable:

| Variant             | Viewer snapshot                        |
| ------------------- | -------------------------------------- |
| Baseline            | `db3be2e3-a4b2-4251-b273-37818b4c5fbc` |
| Daniel              | `d7f10a3f-eff2-4f1b-b823-27010d543cdd` |
| Austin continuation | `078b9b43-52bc-4f27-90c2-efc48a2c6201` |
| James motion        | `5644d828-daad-45e2-bfe6-3797067fd9e5` |

Prefix each ID with `viewer/snapshots/` and append `.json` when recovering assets. The viewer's
latest pointer alone cannot select historical outputs. Use a matching branch build plus pinned
asset recovery/proxy; do not silently serve newer assets to an older decoder.

## Measured output and loading results

Chrome/Metal, 1280×720, one automated scene at a time, one cold visit and one repeat visit per
case. Other user activity and network variation were not controlled. Cold cases had a 150-second
readiness cutoff. Repeat visits after a cutoff can still finish outstanding server downloads,
so they are not clean fully-warm benchmarks. Person byte counts below come from completed
loads; total response-header byte sums for interrupted visits are not actual completed wire bytes.

| Comparison                    | Baseline                                   | Candidate                                                   | Interpretation                                                                                       |
| ----------------------------- | ------------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Stairs person payload         | 136.0 MB                                   | James 30.7 MB                                               | 77.4% smaller; visible geometry/placement remains essentially the same.                              |
| Stairs cold full readiness    | 47.1 s                                     | James 22.6 s                                                | Delivery improvement on this connection; both approximately 60 FPS during the short playback sample. |
| Elevator person payload       | 652.9 MB                                   | James 139.9 MB                                              | 78.6% smaller.                                                                                       |
| Elevator cold full readiness  | Not ready by 150 s                         | James 52.0 s                                                | Strong delivery improvement; does not repair person/world registration.                              |
| Gym world, matched front view | Fragmented foreground bench and floor      | Austin clearer bench, cleaner floor and equipment           | Retain the candidate; confirm placement and complete walking-route behavior before promotion.        |
| Source-coloured stairs        | Existing generated world with person       | Daniel recognizable recorded stairs, large voids, no person | Useful experimental reconstruction lane; no walkability acceptance.                                  |
| Source-coloured kitchen       | No prior completed reconstruction baseline | Daniel room plus 608-pose sequence                          | Recognizable layout and person; smeared foreground, holes and missing handled objects remain.        |

The initial kitchen measurements use Daniel's earlier streaming implementation. Its current tip
waits for the complete pose set; do not advertise the removed first-usable streaming behavior.
Optional ruler metadata was missing in the tested packaged scenes. Gym and the kitchen review
also lacked the requested audio manifest. A successful render does not establish recovered or
spatially separated dialogue. Some cold baseline visits had a stalled source video despite a
60 FPS rendering loop; rendering FPS is not proof of advancing playback.

Interaction checks on main, James and Austin stairs passed for Enter play/pause, advancing video,
map opening and walking. At 390×844, 1920×1080 and 2560×1440 the canvases fit without horizontal
page overflow; this alone does not establish touch or projector usability. Original-mix audio
was running with measured diagnostic offsets below 7 ms in the short test; spatial dialogue was
unavailable. Injecting first-person-frame download failure exposed a useful Austin improvement:
it displays “Scene could not load. Reload to retry.” Main and James remain at 0/50 frames with
an uncaught fetch error. Austin's wrapper also exposes a retry button for this failure; main and James keep their loading overlay. Keep that explicit failure/retry flow.

The repeat gym comparison disabled the boundary fade and used identical origin and half-body-height
sideways camera offsets for both worlds. Austin's bench, floor and equipment also improve in that
sideways view; the earlier black diagnostic frames included the boundary fade and were not proof
of missing geometry. Both versions still place the person too high relative to the bench. Prefer
Austin's environment while treating person/bench fitting as a separate unresolved task. These are
matched world-coordinate views, not a verified calibrated source-camera overlay.

Current Daniel `ffe0700` loaded all 608 kitchen poses in 14.9 seconds with already-cached assets
and advanced through the six-second recording without page errors. This is a warm local playback
check, not a cold-network speed claim or acceptance of the complete recording. Its review URL
explicitly disables the source video/audio. Six short silent browser recordings preserve the
baseline/candidate elevator and gym, James elevator and current Daniel kitchen checks.

## Keep, reconcile, or exclude

| Area                                         | Decision                              | Integration requirements                                                                                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Austin gym world                             | **Keep candidate**                    | Preserve its immutable asset path and baseline. Fit/verify people, handled objects, cameras and collision support against this world before changing the shipping preset.                                                                                           |
| Austin elevator world                        | **Hold as alternative**               | Layout and rail/stair geometry differ; no demonstrated overall win sufficient for default replacement.                                                                                                                                                              |
| James motion packaging                       | **Keep after hardening**              | Preserve frame/timing contracts; verify motion payload integrity and base-PLY identity. Reconcile competing formats before rollout.                                                                                                                                 |
| Daniel lossless chunked motion               | **Evaluate alongside James format**   | Current tip fully preloads; do not restore removed progressive behavior silently. Fix Bun test compatibility, choose one canonical format or explicitly support both.                                                                                               |
| Daniel source-coloured environment stage     | **Keep experimental option**          | Useful source-layout fidelity and shared coordinates; holes and uncertain geometry remain. Label provenance and missing collision/people capabilities.                                                                                                              |
| Daniel durable paid-call accounting/recovery | **Keep/reconcile first**              | Bring runner and worker recovery receipts together; retain pending/unknown outcomes. Its local ledger does not coordinate separate machines or impose a dollar cap.                                                                                                 |
| Austin cache/resume provenance               | **Keep/reconcile**                    | Combine source/options/code checks and forced-descendant protections with Daniel's recovery. Avoid relying on stage status alone.                                                                                                                                   |
| Quality judges                               | **Combine deliberately**              | Retain Daniel's calibration/holdout/request accounting and Austin's source/world/camera binding. APIs/ledgers conflict; no verified autonomous repair-and-promote loop exists.                                                                                      |
| Storage transport                            | **Select bounded recovery/streaming** | Prefer bounded streaming publication, explicit timeouts and preserved failures. Reconcile Austin/James patches; avoid unbounded buffered uploads and deleting another writer's partial files.                                                                       |
| Audio/object/runtime fixes                   | **Keep focused fixes**                | Daniel legacy AudioParam fallback; Austin exact timestamps, source-FPS object timing, malformed-track diagnostics, API credential-origin restriction/private worlds; James watcher exclusions and readiness diagnostics. Retest their shared dependencies together. |
| James late fanout and permissive fallbacks   | **Exclude pending repair**            | Do not promote scenes by dropping failed/small tracks, ignoring verification failures, weakening identity gates, or auto-enabling quality overrides.                                                                                                                |

## Reproduced blockers

- An ambiguous rate-limit log caused Austin's paid-command wrapper to attempt four mocked
  Modal invocations; Daniel attempted one. A nominal one-attempt Marble submit caused James's
  wrapper to invoke seven times, versus one on main. These establish duplicate-launch risk,
  not proof that seven provider charges occurred.
- Two concurrent James retries of one refused generation receipt both claimed it. Keep atomic
  reservation and reconcile the prior operation before allowing another submission/key.
- James selects a largest-mask depth reference without forwarding its sample time through the
  animation registration path. Mask/ROI/depth and camera can therefore refer to different frames.
- James's identity shortcut accepts an `overlap` reason that includes insufficient nonzero overlap;
  its world-only fallback does not produce a complete world-only replay package. Missing people
  are not a successful people-reconstruction result.
- The latest James tip additionally skips tracks occupying less than 2% of the image and tracks
  with failed depth registration. That can hide real subjects and needs explicit degraded-output
  semantics and source validation. New log-based key selection does not fix the reproduced races.

## Were World Labs and GPUs used?

Yes, but use and effective completion are different. Archived World Labs completion records were
independently retrieved for Daniel's kitchen and MMA worlds: both completed without a provider
error, at 1,600 and 1,580 credits respectively. Twenty-two archived overnight Modal run records
identify L4 execution; five contain non-null errors. These are archived execution records, not
fresh dashboard queries or independently reconciled invoices. The archived resource ledger
records estimates and a shared-account meter increment, which can include other users.

Austin's logs report replacement elevator/gym generations. James's late logs report additional
worlds and GPU fanout, but the latest shared snapshot inspected is still Austin's continuation
snapshot and contains no completed new `kitchen-cooking` people/sequence package or world blob.
Some reported debit totals cannot be reconciled from the committed narrative alone. Recover
operation IDs, account aliases and complete source/job/output lineage before asserting spend or
full completion. No new paid inference or world generation was launched for this review.

## Integration order and acceptance

Austin has an active goal improving the gym. The tested asset is a pinned checkpoint, not a claim
about his eventual result. Refresh his branch and publication receipts at integration time and
rerun affected comparisons; do not overwrite his work or replace this evidence silently.

Preserve original authorship when porting changes to `main`. Prefer cherry-picking the original
commits with their author identity and co-author trailers. When dependencies require adapting or
combining work, retain the original contributors' credits and record the source commits; Codex
credit is additional and does not replace human authorship.

1. Protect spend and recovery first: one shared source/job ownership ledger, atomic claims,
   non-resubmitting launch wrappers and tests for ambiguous outcomes.
2. Reconcile cache/resume provenance and storage transport; test partial/failed publication and
   interruption without touching shipping assets.
3. Land the small audio/object/viewer fixes with combined typecheck, formatting and regression tests.
4. Choose and harden the motion format using matching assets, payload/base hashes, missing-frame
   tests, cold/repeat loading and actual advancing playback.
5. Retain Austin's gym candidate and Daniel's experimental environment lane, then validate/refit
   each candidate against source cameras, motion and support geometry. Promote per clip, preserving
   a rollback snapshot; do not promote a whole branch merely because one world looks better.

The supplied-key rotation policy is now explicit in the runbook. Main does not yet implement an
automatic key-pool selector or all the harness safeguards described here. This review is a
selection and integration plan, not a claim that those implementation changes have landed.
Full-inventory reconstruction, physical Quest performance, live billing reconciliation and every
possible failure mode remain outside the independently completed checks above.
