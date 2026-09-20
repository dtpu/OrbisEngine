# Live video-first evaluation: kitchen and gym

September 19, 2026. These are private, separately retained evaluation candidates, **not replacements**
for the old kitchen or accepted Austin gym. Daniel's walkthrough implementation is unchanged.

## Three different questions

1. **Does video upload/generation work?** Yes for both kitchen and gym. The production
   `scripts/marble_world.py video submit` path uploaded the complete files, received HTTP 200,
   persisted operation IDs, polled completion and retrieved private `marble-1.1` video worlds.
   Independent provider downloads, local exports and browser-served bytes matched. Full gzip CRC,
   SPZ headers and actual Spark rendering passed. This exercised the real provider, not mock S3.
2. **Are the new rooms acceptable?** Not for promotion as faithful, walkable reconstructions.
   The first kitchen has appealing frontal detail but unsupported cabinetry and off-axis artifacts.
   The prompt-preserved follow-up still has a visibly smeared floor and unverified invented areas.
   The new gym distorts the central bench and does not improve the accepted room consistently.
   An environment preview intentionally excludes separate animated people/props; this is not an
   authorization to omit either recorded person from a complete gym delivery.
3. **Does this prove fewer hallucinations from video?** No. Selected views improve while others
   invent or distort structure. Inputs, prompts and generated coordinate systems are not controlled
   tightly enough for a causal video-versus-stills claim. API completion is not visual acceptance.

## Input provenance and limits

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| Kitchen submitted historical union-cleaned MP4 | 20,916,685 | `abed03c5c52831a61612edec70a0f9cf72b3e3e473a59afc456d853df06f03a6` |
| Available kitchen source encoding, `kitchen-cooking.mov` | 143,639,523 | `15084804fb92ee6bcf37aa9433452aa534c4ae7f8d43dac3b6dbaf29389d8598` |
| Gym submitted complete raw `gym.mp4` | 26,637,508 | `fef56b00ef0004614bf0f68fd046898158f3b319ab4da8916a0364289c482226` |

The kitchen upload was recovered by verified, read-only shared-storage download: 609 frames at
12 fps, 1280×720, 50.75 seconds. Its historical cleaning report was also hash-verified
(`5181d2ae23c171665be025b905aedb2e39ffb9930a9084818a839af595c2786c`).
Eleven source/cleaned time samples show the same action, but substantial removal smears remain
around the stove, cabinetry and moving person. The fridge opens late in the recording. The old
five-still selection emphasized a closed fridge; the new prompt describes the changing fixture
state and asks for one coherent room. That is an explicit input/prompt difference, not an ablation.

The exact historical `test1.mov` encoding is unavailable locally. The available source encoding has
3,040 frames and a different hash/frame rate. Similar footage does **not** establish decoded-frame
correspondence. The approximate upstream index mapping and previously reported 305/609 label
mismatches remain unresolved; we did not reuse their indices as exact camera correspondences.

The gym's historical cleaned MP4 could not be recovered. We therefore submitted all 542 raw frames
at 30 fps (18.0667 seconds), retaining the complete lifter action and late second person's presence
in the input. Ten source samples were inspected across the recording. Cleaning/input differences
prevent treating the accepted-room comparison as a controlled generation experiment. No new person
reconstruction, body-motion alteration, cleaning inference or subject-drop decision was performed.

## Provider receipts

| Candidate | Operation ID | World ID |
| --- | --- | --- |
| Kitchen video, automatic recaption | `4fd87e3d-1656-4406-8f40-26dc352fc505` | `2a900b18-4e13-4084-88f6-6a9aca29628a` |
| Gym raw video, automatic recaption | `f0e168a8-f4a8-46d3-8a2e-0d99ea564e4f` | `fcf65577-99e4-481f-8f76-c34b0aa196fa` |
| Kitchen video, prompt preserved | `12f88c07-33c2-46ea-afe3-d3751c1e8c30` | `0a4db6aa-ba0c-4013-9c0f-677fe6f4e3e2` |

All three completed in approximately six to seven minutes and each operation reported 1,600
credits: 100 for the panorama and 1,500 for the world. Total settled spend was **4,800 credits**;
the selected credential's reported balance fell from 7,000 to 2,200. All runs use a supplied
credential loaded from the ignored, permission-restricted handoff file; values are not logged,
committed or embedded in the viewer.

| Export | Bytes | SHA-256 |
| --- | ---: | --- |
| Preserved five-still kitchen | 28,769,418 | `20362b6c9bb20d171e82dfc1443ddd016c15ef2cfe0ff1a82cce21c240e9ff1f` |
| First kitchen video | 28,933,411 | `74a2277da90caafb3098e3867c0c5cf96cd5f97cdb71a5d26ae47dc4bf288eb2` |
| Prompt-preserved kitchen video | 29,033,734 | `7c70d387385d48de3853e174bc8f7f88e180d5e16e86dd7eec2aadc1c54de63c` |
| Preserved accepted gym | 28,794,871 | `f76c470932c83b1f985de8be5598928f8bb5a6933ad6144f92a8986221451520` |
| New gym video | 29,146,983 | `1c190af78d9294f419f1fec689d0544342d37ef7335601c30b94936ed819d8b4` |

The original gym comparison environment (`996033d85eacc9ab145178108c0aec621e1f2c4f2c2b1f98286c5177fedc8ce4`)
was inspected separately; it is not the accepted Austin asset. The old kitchen is provider world
`94c51e35-7829-4fe2-92c1-ee853e62ddf2`, not another file merely named `marble-kitchen.spz`.

## Defect exposed by the live requests

The first two returned worlds contained rewritten captions despite sending source-grounded text.
The client had incorrectly treated `disable_recaption` as image-only. The current official
[video request schema](https://docs.worldlabs.ai/api/reference/worlds/generate.md) supports that field.
Prompted video submissions now explicitly set it to true. Absent/empty prompts retain automatic
captioning. Image/multi-image behavior is unchanged; neither request intent nor provider-generated
claims such as “faultless” are evidence of geometric correctness.

A separately authorized, hypothesis-driven kitchen follow-up uses identical uploaded bytes, model,
privacy and text with only `disable_recaption=true` added. Its operation is
`12f88c07-33c2-46ea-afe3-d3751c1e8c30`. This is not an automatic regeneration loop, and stochastic
generation remains an uncontrolled difference. The completed world retains the supplied text
**exactly**. The response omits the disable flag itself; exact returned text is the observed retention
evidence. Its video type, model, privacy, downloaded/served bytes and gzip/SPZ integrity also pass.
This validates the corrected request path, not adherence to every instruction or room accuracy.

## Browser and visual method

Playwright drove a locally bundled Spark 2.2 environment preview with the recorded source video
alongside it. Each candidate was checked sequentially in desktop Chrome at both 800k practical LoD
and full 1,920,000-splat detail: origin, left/right turns, opposite direction and two displaced views.
Capture reports record world hash, camera position/quaternion, FOV, clipping, scale, source time,
playback/reset checks and page errors. Additional downward views inspect the bench and floor.
All six environments passed loading, advancing source playback and reset checks with no page
errors: 72 standard captures plus 10 downward inspection captures. Full detail did not remove the
identified defects; practical-detail examples retain them too.
All numerical camera translations are **viewer units**, not measured metres. Provider scale metadata
was recorded, not used to claim cross-world registration. Identical numeric cameras do not represent
matched physical viewpoints in independently generated worlds.

Source playback is a reference, not synchronized camera reconstruction. Full video inputs also mean
these time samples are not held out from generation. There is no newly passing held-out alignment,
validated collision model, person placement, headset trial or authenticated human quality approval.
Private provider-native viewing was not used to establish a matched-camera renderer comparison.

Visual findings from direct inspection:

- First kitchen: sharper stove/fridge and countertop detail in some frontal views, but a new
  cabinet/counter/coffee-appliance arrangement against the blue-wall side is not supported by the
  inspected footage. Dark/smeared geometry remains visible beneath the counter when displaced.
- Prompt-preserved kitchen: recognizable frontal kitchen detail and exact text retention, but the
  downward displaced view shows a broad stretched/smeared floor surface around the open appliance.
  A glazed door and differently furnished rear room are inferred areas, not verified reconstruction.
  It does not earn promotion, and prompt retention alone does not prove improved geometry.
- Gym: broad mirrored-wall/dumbbell-rack appearance is recognizable, but the main bench becomes
  thin, broken-looking geometry; reflections and equipment layout are not reliable reconstruction.
  The accepted environment retains a substantially more coherent bench. The original gym also
  exhibits large floating/elongated artifacts, so outperforming that older baseline alone would not
  justify replacing the accepted scene.

No default scene, accepted asset, shared viewer snapshot or catalog pointer was changed. The review
server is isolated at `http://127.0.0.1:5511`; existing teammate servers are untouched. Raw footage,
generated worlds, receipts and captures stay under ignored `.context/evidence/`, outside Git.

## Delivery and remaining work

The independent cut-cache correctness fix, complete Node publisher/runtime pair, bound offline
manual quality-review gate, archive-only automatic publication and Marble recovery/integrity
hardening have landed with source authorship/attribution preserved. Actual Node upload streaming,
checksums, sizes and conditional headers were tested with an offline loopback fixture, not real S3
publication. See [quality judging](quality-judging.md) and [shared assets](shared-assets.md).

Final local checks passed: 34 Marble/selector tests; 89 cut-cache, recovery, quality-import,
offline-capture and archive-hook Python tests; 13 Node transport/policy tests; 14 shared-asset/pull
Bun tests; the complete Ruff/Prettier check; and production TypeScript/Vite build. Negative-case
fixtures intentionally emit blocked/failed stage logs; all test suites passed without skips or
weakened assertions. The prompted-video cases cover explicit, absent and empty prompts plus
unchanged image contracts. Heavy browser/build work ran sequentially; memory pressure was elevated
intermittently, disk retained over 90 GiB free and no thermal/performance warnings were reported.

The calibrated live AI harness and autonomous repair are **not** enabled or validated.
The incompatible list/dictionary quality-attempt ledgers require a versioned, lossless migration
before adopting either implementation. Next: repair decoded-index/PTS provenance; establish measured
source/camera/scale registration and held-out geometry checks; curate structurally sound inputs;
then design a matched-input, controlled, bounded comparison before claiming fewer hallucinations.
