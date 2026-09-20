# Austin kitchen clip demo

Branch: `austin-kitchen-clip-demo` in `dtpu/htn2026`.
This branch preserves the complete committed `austin/gym-kitchen` history through `a57f05c`
and adds this teammate entry point. It includes the kitchen repair code and workflow,
plus the accepted gym's viewer and collider support.

## Run and review

Follow the branch [README](../README.md) to install with Bun and start the local viewer.
Obtain the team's read-only `.env.local` privately using the
[shared asset instructions](shared-assets.md#teammates). Viewing these published scenes
does not require Modal credentials or a new reconstruction run.

- [Clip picker: kitchen selected](http://127.0.0.1:5399/reviews/gym-kitchen/picker-local.html?clip=kitchen-repair)
- [Kitchen scene and review](http://127.0.0.1:5399/reviews/kitchen-continuation/index.html)
- [Kitchen preview video](http://127.0.0.1:5399/reviews/kitchen-continuation/actor-pan-door-preview.mp4)
- [Accepted gym scene](http://127.0.0.1:5399/reviews/gym-repair/index.html?quality=detail)
- [Gym preview video](http://127.0.0.1:5399/reviews/gym-repair/gym-contact-wip.mp4)

Use WASD to walk, drag to look, and Enter to play or pause. Enable original audio using the
player controls. The gym comparison and unchanged gym preview are silent.
A running server pins its asset snapshot; restart only a server you own to adopt a newer
publication. Preserve other sessions' servers. `/api/shared-assets` reports the active snapshot.

## What this version contains

The kitchen animation covers the full 50.73-second recording with original audio/timing and 609 actor
samples. This is temporal coverage, **not a complete room reconstruction**. The static room uses
seven selected source observations and has large missing surfaces, especially when walking away
from the recorded camera path. Those black areas are absent geometry, not unfinished downloads
or a detail setting. There is currently no validated complete kitchen walkthrough.

James's final 50 samples reuse his early saved appearance; the 559 early samples
are unchanged. The scene includes the inferred held pan, its handoff to the observed resting
pan, and the freezer door's opening and late partial closing. The final actor pose is held
for 0.067 seconds; the last measured door angle is held after 50.497 seconds.

The user found this kitchen clip good and identified the missing Brita filter pitcher.
Brita and bag motion remain absent. Room holes, static fragments, approximate pan/door motion
and imperfect cabinet/wall contact remain documented. This is the useful reviewed reference,
with those limitations preserved. The accepted gym and its original assets are unchanged.

Media lives in private bucket `wander-shared-797639045717`, outside Git. The verified delivery
uses viewer/archive snapshot `2884fb59-fec1-4a16-8a37-bd6115a13f6e`; its viewer manifest is
`viewer/snapshots/2884fb59-fec1-4a16-8a37-bd6115a13f6e.json`. All 65 updated/new viewer assets
passed uploaded hash/size checks. Author evidence and retained attempts are under
`evidence/kitchen-actor-pan-v4/` in the archive snapshot.

For follow-up work, read the [reusable kitchen workflow](kitchen-workflow.md),
[delivery and validation record](kitchen-continuation.md), [object contract](objects.md),
[gym delivery](gym-delivery.md), and [paid recovery rules](paid-recovery.md).
The workflow includes manual object authoring and review; it is not a new automatic
held-object reconstruction feature.
