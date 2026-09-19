# Tonight

The human rewrites this file before each overnight run.
The agent works through the backlog in order, then completes the final pipeline rerun below,
under `docs/overnight/RULES.md`. The agent does not edit this file.
Progress goes in `docs/overnight/LOG.md`.

Branch: `overnight-main-1`

## Backlog

In priority order.
Each item needs a done check the agent can verify on its own.

1. **<item>**
   - Done when: <a measurable check, e.g. a capture, a test, or a number from the real viewer>
   - Notes: <clips, flags, runs, or files to start from>

## New clips for this run

None specified yet. Add each new clip's source path or shared-storage key, a unique name,
its exact trim if applicable, and any intended processing options here. A viewer preset alone
is not a source clip. Do not invent inputs or treat another preset of the same input as a new clip.

## Final pipeline rerun (required)

After the code backlog is verified or explicitly blocked, run the updated pipeline for **every
existing source clip**, plus every new clip listed above. This is a final validation phase,
not an optional extra after unit tests.

- Build the clip inventory from shared/local clip manifests and saved run records, including
  clips outside the five-scene picker. Deduplicate aliases for the same source and exact trim.
  Record missing source inputs as blocked rows; do not silently omit them.
- Record the code commit and baseline S3 snapshot/output for each existing clip before rerunning.
  Retain the recorded trim, frame rate, and processing options unless a change is intentional
  and logged. Use separate candidate run names/output paths so the demo and baseline remain usable.
- Actually rerun reconstruction and packaging with the final code. A successful exit that only
  resumes cached completed stages is not a rerun: use fresh stage state or explicit `--force`
  stage names, and record which stages ran. Reuse downloaded weights and existing generated
  Marble worlds (`--reuse-world`); keep `--marble none` for clips using that lane. Do not buy
  replacement worlds for existing clips.
- Process specified new clips through the applicable pipeline, with at most one new Marble
  generation per new clip. Record any admission failure instead of forcing unsuitable footage
  through. Apply the unattended execution policy in `docs/overnight/RULES.md`.
- Open each completed candidate in the real viewer and compare against its source and baseline.
  Capture representative playback and walk views; check people/placement, objects where present,
  collisions, and original audio timing/availability. Log missing soundtrack as a limitation,
  not invented audio or evidence of spatial dialogue.
- Preserve runs and evidence in private S3. Only promote a candidate to the normal demo paths
  after its comparison passes. Keep the previous snapshot available for recovery.
- If this phase triggers another code fix, rerun the affected clips/stages against that final
  commit and refresh their evidence. Do not report results from an older implementation as current.

Done when: every inventory row is **passed**, **failed**, or **blocked**, with its source/trim,
commit, rerun stages, output/S3 reference, viewer evidence, and a concrete failure or blocking
reason where needed. Failed or blocked rows mean the full regression has not passed; report
those explicitly in the handoff, alongside any improvements and remaining regressions.

## Out of scope

- <areas the agent must not touch tonight>
