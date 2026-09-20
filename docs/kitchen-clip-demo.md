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
- [Kitchen with repaired wall and aisle](http://127.0.0.1:5399/reviews/kitchen-continuation/room-access/index.html)
- [Current kitchen preview](http://127.0.0.1:5399/reviews/kitchen-continuation/room-access/preview.mp4)
- [Earlier kitchen repair notes](http://127.0.0.1:5399/reviews/kitchen-continuation/index.html)
- [Previous kitchen preview](http://127.0.0.1:5399/reviews/kitchen-continuation/actor-pan-door-preview.mp4)
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
or a detail setting.

The current picker adds filtered surroundings from the existing generated kitchen. It restores
much of the doorway, ceiling and adjacent room while preserving the observed cooking area,
animation and object files. These surrounding surfaces and their appearance are inferred;
they do not establish a source-accurate complete room. The previous version remains in
**Spares and experiments** as **Kitchen · previous partial room** (`clip=kitchen-observed`).
The blue paint is present in the source footage. A fitted planar continuation now closes its
ragged gaps, with extrapolated surface explicitly labelled as inferred. Three bounded supports
on the measured tile plane connect the starting area to the aisle beside James. Step back slightly,
go right around the counter, then walk into the kitchen. Counter collision remains active;
this does not make every generated surface walkable. See the
[wall and aisle repair](kitchen-continuation.md#wall-and-aisle-repair).

James's final 50 samples reuse his early saved appearance; the 559 early samples
are unchanged. The scene includes the inferred held pan, its handoff to the observed resting
pan, and the freezer door's opening and late partial closing. The final actor pose is held
for 0.067 seconds; the last measured door angle is held after 50.497 seconds.

The user found this kitchen clip good and identified the missing Brita filter pitcher.
Brita and bag motion remain absent. Room holes, static fragments, approximate pan/door motion
and imperfect cabinet/wall contact remain documented. This is the useful reviewed reference,
with those limitations preserved. The accepted gym and its original assets are unchanged.

Media lives in private bucket `wander-shared-797639045717`, outside Git. The current wall/aisle
delivery uses viewer/archive snapshot `870f901a-462f-4f13-9c01-b839cb0ceb5f`; its viewer manifest is
`viewer/snapshots/870f901a-462f-4f13-9c01-b839cb0ceb5f.json`. New geometry, preview and walking
evidence are archived under `evidence/kitchen-room-access-v1/`. The earlier surroundings remain
in snapshot `1a852341-85b3-4179-8ab6-cb964dc5d504` and `evidence/kitchen-room-context-v1/`.
The actor repair, spending receipts and accepted gym remain preserved from snapshot
`2884fb59-fec1-4a16-8a37-bd6115a13f6e` and its `evidence/kitchen-actor-pan-v4/` namespace.

For follow-up work, read the [reusable kitchen workflow](kitchen-workflow.md),
[delivery and validation record](kitchen-continuation.md), [object contract](objects.md),
[gym delivery](gym-delivery.md), and [paid recovery rules](paid-recovery.md).
The workflow includes manual object authoring and review; it is not a new automatic
held-object reconstruction feature.
