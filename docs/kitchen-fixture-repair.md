# Kitchen fixture repair

The latest walkthrough exposed a perforated freezer door, missing wall faces beside the fridge,
a static microwave door and a stationary small-pot lid. This repair addresses those four areas
using existing reconstruction outputs and original decoded footage. It makes no new paid call
and does not change the cook, source cameras, recorded audio, pan tracks or accepted gym.

Open the [kitchen picker](http://127.0.0.1:5399/reviews/gym-kitchen/picker-local.html?clip=kitchen-repair&v=fixtures1),
[standalone review](http://127.0.0.1:5399/reviews/kitchen-continuation/fixture-repair/index.html),
or [preview](http://127.0.0.1:5399/reviews/kitchen-continuation/fixture-repair/preview.mp4).
Use the normal local server and shared assets described in [shared-assets.md](shared-assets.md).
Media and review pages live in private storage, outside Git.

## What changed

The old freezer door used 12,414 sparse points on a single plane. Its replacement uses 190,870
dense overlapping splats, with front colour sampled from original decoded frame 2167 at
36.161667 seconds. An inferred reverse face and edges close the shell. The original door angles,
hinge and timing are preserved by compensating for the viewer's automatic asset centring.
The lower refrigerator front is also resampled from this original frame. This improves its
front appearance; it does not recover details that were never resolved in the footage.

The freezer gets an explicitly inferred cavity enclosure and shelf, retaining the existing
observed interior fragments. No food or package text is invented. Two wall planes fitted to
6,136 and 519 source depth samples fill the gap beside the refrigerator. Their fit RMS is
0.00249 and 0.00207 body-heights; extending their shape and colour beyond those observations
is inferred. A narrow microwave-front exclusion also removes 211 overlapping generated points; surviving
packed attributes remain unchanged.

The microwave door now opens around 34.8–35.8 seconds and closes by 36.997 seconds. Sparse source
corners constrain its hinge angle, with interpolation during occlusion. An independent visible
corner at 35.595 seconds is 4.46 pixels from the fit in the 960-pixel-wide reference. The front
texture comes from original frame 1393 at 23.246667 seconds. Its thickness and dark reverse are
inferred. The partial observed interior remains incomplete. Bounded door volumes remove 3,901
observed points, including a front extrusion restricted to the door footprint; that removes
old fragments covering the new face without broad removal of the adjacent control panel.

The small rear stainless pot has a separate lid. Its visible lowering is constrained at
22.995–23.496667 seconds; the hidden pickup at 21.75–22.745 seconds is invented interpolation.
The circular-disc model uses the seated source ellipse and 64 depths. Two independent seated
centres differ by 0.75 and 1.03 pixels at reference width 960. Depth, thickness, underside and
occluded motion remain approximate; neither handle geometry nor exact grip is reconstructed. Its fixed top texture is too bright
from some raised views; material reflections are not reconstructed.
Only 156 stationary lid points are removed. The larger black pot remains closed, as observed.

Original decode ordinals and PTS bind every new observation. The original SHA-256 remains
`15084804fb92ee6bcf37aa9433452aa534c4ae7f8d43dac3b6dbaf29389d8598`.
The CFR derivative has different ordinals; those are not substituted for original frame labels.

## Checks and remaining failures

The combined scene passes 14 source-camera checkpoints plus displaced front, side and reverse
views without page errors or object warnings. An isolated freezer-front test changes its
background from red to blue: all 221,940 tested interior pixels remain identical. This certifies
opacity at that view, not shape or contact accuracy. In a fixed camera wall-face crop, near-black
pixels fall from 77.83% to 0%; this measures local coverage, not complete room accuracy.

Actual keyboard walking through the clip picker still reaches James around the counter, with
supported, unblocked sampled positions and advancing playback. The prior floor supports and
actor are retained. Moving doors have no dynamic collision. The scene still contains seams,
some holes, blurred small objects and imperfect hand/cabinet contact. This is 50.73-second temporal
coverage, not a complete spatial or object reconstruction. Desktop Chrome checks are not headset
evidence.

The carried bag is still absent. One changed, local rigid-pouch model fit source silhouettes
and candidate palm surfaces, then failed: only four of eight independent centres passed, with
critical hand-contact errors above the existing 0.04 body-height limit. The old wrist failures
and three historical bag/pitcher hypotheses are preserved. No appearance was generated from
that failed candidate, and no budget or attempt ledger was reset. Brita motion remains absent.
The source inventory records the bag at 40.501667–44.490 and 49.496667–50.663333 seconds; hidden
package identity/transport is uncertain. The pitcher interaction is around 4.5–9.25 seconds.

Reproduction scripts, exact source bindings, rejected candidates and browser evidence are kept
under `.context/evidence/kitchen-fixture-repair/` and archived with the publication. See the
[delivery history](kitchen-continuation.md) for earlier actor repair, spending and scene limits.

## Preview and delivery verification

The new preview contains 609 newly captured viewer frames across 50.730 seconds. One cook,
one pan layer and the three moving fixture layers are active at every sampled time; source
occlusion can still hide their pixels. Full decoding passes. All 2,380 original AAC payload
hashes, PTS/DTS, durations and priming values are unchanged. Preview SHA-256 is
`8d3c3d25ec045757a64dd7e0e5ce1d4191c4184b5ba25e3231e6a8e130a84a3d`.

The real viewer completed a playback loop. After the last microwave cleanup, the source-camera
checks, appliance close-ups and keyboard route through the actual clip picker were repeated.
There were no page errors or object warnings. Final static world: 1,756,110 splats; source-bound
fixture manifests and model hashes validate. All other clip-picker cards remain unchanged.
The repository format check passes; no runtime code or test was changed by this asset repair.

Published to private bucket `wander-shared-797639045717`, viewer/archive snapshot
`670344c7-b047-41d6-ae26-89b01d2bd275`. Every one of the twelve new or updated viewer paths
passed remote checksum/size and fresh-download SHA-256 checks. All 16,963 unaffected viewer
paths and 35,667 previous archive entries remain identical, including the accepted gym.
Publication had no exclusions. Source bindings, rejected candidates, geometry scripts, browser
checks and preview captures are under `evidence/kitchen-fixture-repair-v1/`. Local verification:
`.context/evidence/kitchen-fixture-repair/publication-verification.json`.

## Shareable team video

The [team MP4](http://127.0.0.1:5399/reviews/kitchen-continuation/fixture-repair/kitchen-team-share.mp4)
is a standalone 1280×720 H.264/AAC file, 20,721,178 bytes, suitable for sending directly to the team.
At the user's request, audio ends at 48.730 seconds, leaving the final two seconds silent while
keeping the complete 50.730-second video. All 609 compressed video packets and their timestamps
match the reviewed preview exactly; video has not been re-encoded. The earlier audio is re-encoded
as AAC at 192 kb/s. The original preview and reconstruction audio remain available.

SHA-256: `cd918e4a7368d18ce0c6910b1f1a8c8c6f8ab70a01fef4d2aaee4cfba7b34ad3`.
Full decoding, stream durations and actual browser tail playback pass. The review page has a
**Download team MP4** link. Export verification is under `.context/evidence/kitchen-share-export/`.
