# Kitchen continuation checkpoint

The user reported disappearing objects, James intersecting cabinets/walls, and restricted walking.
These are unresolved acceptance failures. The accepted gym and the previously published kitchen
remain available; the source-coloured kitchen candidate below is an experiment, not a replacement
certified to fix them.

The newly supplied Modal profile `austin-jian547` authenticated successfully. It had no active apps
or model volumes at preflight. Credentials were stored privately with mode 600; other profiles were
preserved. No new paid job has been submitted. Existing receipts and attempt counts remain intact;
the requested additional spending cap has not yet been supplied.

## Recovered observed geometry

The missing eight-anchor Pi3X archive was recovered with verified ranged S3 transfers:
54,200,874 bytes, SHA-256 `74e54e99b45a6707fc280ed8332676b033b1574fa86c3b54485d762d5b44fcea`.
Its original RGB/depth, poses, confidence and masks belong to the already verified kitchen derivative.
An independent reconstruction of `static-scene.npy` matched all 1,541,687 points exactly, confirming
the cache coordinate convention. This required no new inference.

`scripts/package_static_anchors.py` exports original RGB as Gaussian SH0 in the packaged people/camera
OpenGL frame. It checks source hashes, camera/time correspondence, coordinate consistency, mask shape,
finite positive-depth points and bounded radii. Explicit source-bound track rectangles and anchor
selection can exclude missed people and inconsistent fixture states. No empty regions are filled.
Run `uv run --locked scripts/tests/test_package_static_anchors.py` for the four geometry/mask regressions.

The initial eight-anchor result contains missed-person ghosts and inconsistent open/closed fridge
states. It is quarantined as evidence. The conservative candidate uses six closed-fridge anchors,
0–36.169 seconds, plus expanded recorded person rectangles. It contains 1,115,485 splats. Its static
coverage stops at those selected observations; the existing people/audio timeline still covers all
50.73 seconds. It cannot replay fridge motion, and moving cookware remains inconsistent.

The real viewer exposes a more faithful sink and kitchen layout, but also large unfilled masked
regions and floating feet. It is **not accepted**. Publication of its review checkpoint records this
failure; it does not silently replace the current kitchen.

## Floor and contact diagnosis

The dominant low plane in this kitchen is the countertop (native Y about -1.22), not the floor.
Five manually reviewed source polygons on visible floor tiles support a lower plane near -3.11.
Of 21,367 labelled depth samples, 18,400 lie within the fitted tolerance. Their plane-fit RMS is
0.00183 body-heights; this measures consistency of estimated depth, not a physical survey.

Next work must preserve recorded objects, keep James aligned with the room through the full clip,
and validate movement through observed open space. A camera-centred depth calibration is being
investigated: it can adjust floor contact while preserving each recorded camera projection, without
optimizing articulation. The later calibration experiment below implements this transform; it is not a whole-scene acceptance.

Private evidence is under `.context/evidence/kitchen-continuation/`. The current source, frames,
previews and published snapshot are documented in [the delivery report](gym-kitchen-delivery.md).

## Main and shared-storage checkpoint

The focused delivery changes landed on `dtpu/htn2026` main at `119260a`, on top of the newer
video-input evaluation work. The author branch remains `austin/gym-kitchen` at `919eac8` before
this receipt update. Neither push rewrote history or included the other session's unfinished XR work.

The verified private S3 viewer/archive snapshot is
`ebdbb33d-1cc7-4d57-8764-c3a76464a538` in `wander-shared-797639045717`.
All five new/updated viewer entries passed remote SHA-256 and size checks. Fresh downloads of both
previews, both review pages and the people manifest matched. The 16,878 unaffected viewer entries
and all 25,729 prior archive entries were preserved; the delivery index is the sole updated prior
viewer path. New repair evidence is namespaced separately from prior receipts.

Open [both scenes and previews](http://127.0.0.1:5399/reviews/gym-kitchen/index.html), or the
[unaccepted observed-geometry experiment](http://127.0.0.1:5399/reviews/kitchen-continuation/index.html).
A server pins its snapshot at startup; only restart a server you own. This checkout's immediate
[local review link](http://127.0.0.1:5399/@fs/Users/austinjian/htn2026/public/reviews/gym-kitchen/index.html)
works through the existing server without interrupting it.

The new-main video evaluation records two additional kitchen generations and one gym generation,
4,800 provider credits total across the three. Those receipts remain separate historical spending;
this continuation submitted none. Neither the new credentials nor those evaluation results reset
stage allowances. See [the evaluation](video-input-evaluation.md).

## Floor calibration experiment

`scripts/calibrate_person_floor.py` uses a fixed positive depth scale per appearance track about
each recorded camera centre. Its placement manifest preserves original PLY/f32 geometry, poses,
source indices and timing; it validates input hashes, the measured source correspondence and floor
uncertainty. Final-viewer offsets include the global registration scale. Four synthetic regressions
check pixel invariance, floor contact, tilted planes, sequence gaps and unstable-scale rejection.

At native world scale, the early/late factors are 1.123970 and 1.131966. Opaque low-point contact RMS
falls from 0.1068/0.1062 to 0.0166/0.0161 body-heights. Maximum residuals remain 0.0920/0.0444
body-heights. These are geometric proxies, not measured anatomical contacts. All actual compact
splat positions preserve their matched source-camera pixels to numerical precision at sampled
times. Intermediate camera interpolation and opposing views are separate checks.

Eight source-camera views plus two displaced views loaded in the real viewer with no page errors.
The transform and visibility handover are applied correctly. Visible room holes, static person
remnants, absent held objects and the closed refrigerator still fail reconstruction acceptance.
Floor contact alone does not establish valid cabinet/wall contact or repair body articulation.

Initial desktop walking measurements also confirm the user's restriction. A 1.8-second backward
input from the opening position travels 0.149 body-heights in the current image-generated room,
versus 0.940 in the observed-geometry experiment. Forward movement stops near the front counter
in both. These are individual movement checks, not certification of all free space or headset use.

The old automatic object detector retained two putative flights. Sequentially decoded source
frames 2130, 2158, 2159 and 2173 show both detections following stationary foreground plate/dish-rack
texture as the camera moves. Neither is a supported thrown-object track. Their original output is
retained with rejection evidence; they must not be packaged as flying props. Held cookware, the
bag and appliance doors still need distinct source-bound tracking.

## Refined source masks and moving-door candidate

A further local candidate replaces the added person rectangles with traced source silhouettes,
retaining the original saved person masks. Selected late geometry excludes the opened fridge,
cook/bag and changed microwave region; the blurred final anchor remains excluded. This retains
1,397,563 source-coloured splats, 25.3% more than the rectangular-mask candidate. Reviewed overlays
bind each polygon to its actual source index and RGB hash. Static ghost fragments and missing
surfaces are reduced but remain visible from displaced viewpoints.

The separate door experiment extracts a planar freezer exterior from the closed source at
36.169 seconds, fits its hinge to recovered depth and estimates opening angles from labelled
corners at 37.497–39.500 seconds. It removes 51,276 static door splats and restores 2,889 observed
interior splats. A 12 fps baked object track opens the door and holds the last measured angle.
Sparse in-sample corner RMS is 5.87–12.07 pixels at 672-pixel image width. No dense or held-out
tracking acceptance is claimed. The reverse face/thickness are unobserved; the planar reverse
appearance is explicitly labelled inferred. Interior holes, missing handle detail and imperfect
hand contact remain. The ten inspected source-camera views and two opposing views load in the
real viewer with no page errors, but this remains an experimental reconstruction.

The actor completion inventory confirms all 609 source-bound poses already exist. The minimum
appearance repair would preserve 559 early frames and animate 50 late frames plus one control
with the early canonical and recorded fixed scale. The current wrapper requires a fixed-scale
passthrough, durable checkpoint and bounded execution before submission. The new workspace has no
model volumes, required native body assets are absent locally, and one read-only check of the old
`dtpu` profile returned an authentication failure. No retry, model download or paid job was launched.
The full contract and receipt/ledger inventory are retained in private `actor-plan/` evidence.

## Repair preview and object-loading checkpoint

The repair candidate and full-length preview were published in private viewer/archive snapshot
`33d0d6a0-db4e-4734-96aa-92fb534c7052`. All seven new/updated viewer entries passed remote
SHA-256/size checks; fresh downloads of its object manifest, preview, review pages and unchanged gym
preview matched. The 16,882 unaffected viewer entries and all 25,756 prior archive entries remained
identical. The original accepted gym was not changed.

The repair preview contains 609 actual viewer captures over 50.73 seconds. All 2,380 original AAC
packet payloads, timestamps, durations and priming values are identical; video ends within one
1/12288-second tick of the source endpoint. SHA-256:
`e201da3998de99733d64a18deec5f215d6a85cdf346610679366e1c411bf5198`.
Full decoding and an actual playback loop pass, with exactly one visible cook in every captured
sample. Four-direction walking passes in the candidate; backward travel is 0.828 body-heights in
1.8 seconds after the new body calibration. The forward stop remains near the counter.

The user's subsequent report of absent objects is accurate for the earlier kitchen: its
`objects.json` is empty. The new repair's explicitly selected manifest renders one Gaussian
`freezer-door` object, confirmed by viewer state and screenshots with zero object-load warnings.
It does not yet render the handled pan, kettle or bag. Do not describe it as complete object
reconstruction. The review entry now distinguishes the new repair from the earlier empty-object
scene. Pending source/hand-contact validation of further props remains separate work.

## Continued object repair and authorized actor job

The third pan hypothesis improves held-out centre errors to 3.10, 5.07 and 7.66 pixels at
672-pixel source width. All 17 sampled mesh-handle proximity checks meet 0.05 body-heights.
The real viewer renders the inferred held pan through 16.033 seconds, then switches to the
retained observed counter pan. Exactly 2,678 source-selected Gaussian points were separated
from the room for that timed static layer; all other room points are unchanged. Nineteen source
views, two displaced views and four-direction walking were inspected. The mesh has invented
appearance and approximate tilt, especially around 13 seconds; uncertain early tracking and
other static remnants remain. This is an object-repair experiment, not whole-kitchen acceptance.

Its full 609-frame preview preserves all 2,380 original AAC packet payloads, timestamps, durations
and priming values. Exactly one cook and one pan layer are visible at every captured sample.
Preview SHA-256: `32f64af62ae8cf80333e667e50fcdb17d4e10aa858e088005010d42292d94352`.

Source inspection identifies the pouring object as a water-filter pitcher, correcting the earlier
kettle label. The third pitcher hypothesis passes only 3/6 independent projection checks and
fails pickup; the bag's third hypothesis also fails projection/contact/coverage. Both remain
excluded from the delivered scene, with all attempts and review-only artifacts retained.

The spending audit verified 24 existing logs and seven completed historical kitchen apps.
Their observed $0.74521365 is already included in earlier workspace billing and must not be added
again. Earlier overwritten failures and incomplete billing prevent certifying the original $27
remainder. This does not establish that it is exhausted.

The user subsequently authorized **up to $10 additional Modal spend** for the concrete saved-actor
repair: reuse the early canonical for the final 50 samples plus sample 0 as a control, then preserve
the 559 existing early outputs. This approval covers that corrective job and necessary cache setup;
it does not reset other attempts or authorize automatic retries. A fresh read-only preflight now
finds another session's completed LHM, native-body, DINO and face-model caches in `austin-jian547`.
Reuse them; do not duplicate setup or stop the other session's jobs.

The tested animation wrapper now preserves a fixed world scale, supports a 600-second execution
limit and durably checkpoints generated outputs with input hashes and a recovery receipt. Eight
new offline checks plus 35 existing recovery/accounting checks pass. A sparse 51-sample export
must remain partial until its intended IDs, control output and explicit full-clip merge are verified.

## Saved-appearance repair execution

The explicitly authorized corrective job ran once in `austin-jian547`: app
`ap-YkBOW5PfyeiaBghCN0qTdj`, call `fc-01M2Y5ZXA4MWYFJZHFNSWJAAXY`.
It reused completed shared model caches, with no additional setup submission. The bounded
worker finished in 55.774 seconds, reporting a compute estimate of $0.023229. This excludes
final hashing/commit, transfer and storage; provider billing for this app has not posted.
Keep the $2.249888 reservation against the additional $10 authorization until reconciliation.
Other sessions' jobs and costs remain separate. No retry was submitted.

All 51 intended outputs were recovered with verified hashes. The worker correctly reports
partial full-grid coverage because its saved seeds contain only sample 0 and samples 559–608.
After the original packaging transform, the control PLY is byte-identical to the retained
first frame. Every tail frame has 40,000 finite Gaussians; colour, opacity and scale attributes
are bit-identical to the early canonical. A fresh candidate references all 559 original early
frames and their unchanged compact motion, then appends 50 new frames. Its lossless tail
motion SHA-256 is `0195d7c19bad78b4eed956fe379d44bb0a51a3079e6bb79c55505fbaaff761b8`.

The early floor placement is unchanged. The tail's independently measured fixed camera-centred
factor is 1.125916; opaque low-point floor residual RMS is 0.01596 body-heights, maximum 0.04328.
Source-camera pixels remain unchanged to numerical precision. These are geometric proxies,
not proof of anatomical, cabinet or wall contact.

The `b2c36c3` main build passed nine source-camera checks, including both sides of the appearance
handover, and two full-body opposing views. James retains the same saved appearance, with one
visible cook. Four-direction walking remains available; backward travel is 0.827 body-heights in
1.8 seconds, with a forward stop near the counter. Pitcher/bag motion, room holes, stretched
static fragments and imperfect articulation/contact still prevent whole-kitchen acceptance.
No headset validation is claimed. Evidence and the deterministic merge script are archived
under the actor-repair delivery evidence namespace rather than committed as generated assets.

## Late door motion and final preview

The original source keeps the freezer open to its final frame, but the door progressively swings
partly closed. It genuinely occludes James around 46.49–46.575 seconds. The second local door
candidate preserves opening samples 0–474 and both pan tracks, replacing the indefinite late hold
with source-measured angles. At 50.497 seconds, corner RMS improves from 64.55 to 4.75 pixels at
672-pixel review width. Only four of six independent checkpoints pass when uncertainty counts as
failure; near-edge-on 41 seconds, clipped 46.49 seconds and final camera alignment remain deficient.
The track holds its last measured angle after 50.497 seconds. It is a better inspection candidate,
not an accepted door reconstruction. No further door hypothesis or paid call was submitted.

The combined preview is `/reviews/kitchen-continuation/actor-pan-door-preview.mp4`, SHA-256
`0006f67d0c3451079f26eaebfc70749456e85ce157f69bc66617fad865a83c22` (19,926,597 bytes).
It contains 609 viewer captures, with one visible cook and one pan layer at every sample. The first
474 captures are verified unchanged; later samples were recaptured after the actor/door changes.
All 2,380 original AAC packet payloads, timestamps, durations and priming values are identical.
Full decoding and playback pass; video ends within one 1/12288-second tick of 50.73 seconds.
The earlier previews and every accepted gym dependency remain unchanged.

Eleven source-camera checks on the `aa08f23` main build confirm the revised door loads alongside
the pan and actor. Original floor/walking limitations remain. Another session subsequently took
port 5399 from a separate checkout; it was left running. A separate local static preview at
[port 5401](http://127.0.0.1:5401/reviews/gym-kitchen/picker-local.html?clip=kitchen-repair)
serves the committed viewer build and verified local media, using the existing shared viewer for
unchanged baseline clips. The portable picker retains all eleven cards. This preview uses no Vite
restart and exposes only public media/build files. Shared-storage review pages use the normal
`/fourd.html` from the teammate's checkout.

## Verified publication and user review

Both scenes and previews are published in `wander-shared-797639045717`, viewer/archive snapshot
`2884fb59-fec1-4a16-8a37-bd6115a13f6e`. All 65 new or updated viewer entries passed remote SHA-256
and size checks. Fresh downloads of the previews, actor motion, manifests, picker and review pages
matched. The 16,887 unaffected viewer entries and all 25,800 prior archive entries were preserved.
Every accepted gym dependency remains unchanged. New outputs, source observations, failed object
hypotheses, original spending ledger and the additional authorized job receipt are archived under
`evidence/kitchen-actor-pan-v4/`. The local verification report is
`.context/evidence/kitchen-continuation/actor-repair-10usd/publication-verification.json`.

The user reviewed the kitchen as good and specifically identified the missing Brita water-filter
pitcher. Use this version as a reference for similar kitchen clips, with the recorded object,
geometry and contact limitations retained. The [reusable kitchen workflow](kitchen-workflow.md)
captures that process and adds explicit prop inventory and interaction coverage before cleaning.
Its procedures still include manual authoring and review; documenting them adds no automatic
held-object reconstruction feature or authorization for future paid runs.

## Main picker aisle regression

The curated `kitchen-repair` entry already loads the reviewed room-access floor bridges, but
main updated only their support height and kept the observed grid's outside boundary. Desktop
walking therefore stopped at the end of the peninsula despite an unoccupied supported aisle.
The runtime now unions reachable finite support with the navigation footprint. Occupancy,
capsule sweeps, step limits and the existing camera bounds remain active; media is unchanged.

The browser regression in `scripts/test-walk-support-browser.ts` accepts a private JSON fixture
with `url`, X/Z `waypoints`, map/collision `probes` and an optional `blockedApproach`. Set
`WALK_TEST_CONFIG` to that file and optionally `WALK_TEST_DIST` to a candidate build. It tests
actual keyboard input in either the picker iframe or the standalone viewer and saves screenshots.
The kitchen passed ten waypoints around the peninsula into the cooking aisle and back, with
no occupied route samples. A direct countertop approach stopped after 0.168 body-heights;
the aisle map probe became selectable while countertop and outside-grid probes remained refused.
Evidence is private under `.context/evidence/kitchen-aisle/` in the kitchen walk worktree.

These are desktop checks, not headset evidence. The separate XR `walk.advance` diagnostic still
has an existing support-height smoothing issue on slightly inclined boxes; that path was not
changed here. Previously recorded inferred-room, object and actor-contact limitations remain.

### Open counter entrance

The later report exposed a gap beside the earlier dog-leg test route: at X=3, walking toward
Z=-1 stopped near Z=0.06 with no occupied cells. The inferred floor bridge covered only the
narrow outer aisle, leaving the visibly open approach outside the navigation footprint.
The revised private `room-access/colliders-open-entry.json` extends that bridge toward the
counter, retaining its far edge, fitted floor plane, three bodies and all obstacle checks.
This support is inferred geometry; no rendered room, actor or prop asset was changed.

The kitchen catalog now uses `selected-scenes/kitchen-open-entry.json`. Both new manifests
are published in `viewer/snapshots/f87c64c4-7113-43d7-acf9-182e59e65a1d.json` in
`wander-shared-797639045717`; downloaded SHA-256 and sizes match local files, and all 18,047
entries from the previously pinned snapshot remain unchanged. The new collider SHA-256 is
`e5695967f1a2130324a8bbca18af91c4657144029d34c962fe75449eb56c7682`.

The regression now includes the previously blocked straight approach, alternate aisle positions,
the return trip, an occupied countertop approach and map/outer-boundary probes. Twelve keyboard
waypoints passed in the standalone viewer. The direct countertop approach still stops outside
occupied cells after about 0.186 body-heights. Evidence, including the failing baseline, is under
`.context/evidence/kitchen-open-aisle/` in the kitchen walk worktree. Fixtures may specify
`expectedParams` so a test fails if the viewer did not load its intended collider candidate.
