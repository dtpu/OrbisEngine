# Interactive world exploration

This is a proposal, not an implemented feature. The first experiment is to enter a reconstructed
scene, select a person, and have a conversation grounded in that scene. A later experiment adds
one movable prop and lets the character respond to changes the viewer actually made.

## Feasibility in the current viewer

| Interaction                                                 | Assessment                     | Main work                                                                                   |
| ----------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| Point at a person or object and inspect it                  | Small first step               | Selection volumes, labels, and desktop/XR input routing.                                    |
| Speak to a person and hear an answer from their location    | Feasible prototype             | Scene context, a voice session, a live audio route, and conversation controls.              |
| Ask a character to highlight an object or replay a moment   | Feasible prototype             | Explicit viewer actions with validated arguments and results.                               |
| Pick up an independently loaded prop                        | Moderate                       | Hand attachment, ownership of its transform, release behavior, and reset.                   |
| Open a door or move furniture embedded in the room          | Larger asset task              | Separate the object, repair the revealed background, and update collision geometry.         |
| Make a recorded person walk over, gesture, or accept a prop | Larger animation task          | A controllable body representation, new motion, navigation, and contact handling.           |
| Match new speech with convincing facial movement            | Research task for these assets | A controllable face and speech animation; the current recorded frames do not supply either. |

These are engineering assessments from code inspection, not measured schedules or demonstrated
interaction quality.

`fourd.html` exposes `wander.people`, `wander.objects`, playback controls, and the Three.js scene.
Loaded props have IDs, groups, meshes, and recorded tracks. `applyTime()` updates every person and
prop from the video clock, so a new transform would be overwritten unless ownership changes.
Most room geometry is a combined splat world, not a collection of named movable objects.
See [objects](objects.md) and [person motion](person-motion.md).

The audio module already handles listener pose and reviewed person anchors. Its sources are
recorded buffers synchronized to video; live conversation needs its own transport and clock.
Published original mixes do not contain isolated per-person dialogue. See [audio](audio.md).
XR controller triggers currently serve teleport aiming; interaction must have explicit input
priority so selecting a character does not also teleport the user.

## Reusing the reconstruction skeleton

The missing runtime skeleton does not mean the reconstruction started without one.
`worker/stages/lhm_person.py` supplies SMPL-X root, body, hand, and jaw pose parameters to LHM.
It saves canonical appearance attributes, query points, neutral transforms, and parameters in
`canonical-state.pt`. `worker/stages/lhm_animate.py` reuses that state with different source poses
through `animation_infer_gs`, then exports the resulting Gaussian frames.

The pinned [LHM renderer](https://github.com/aigc3d/LHM/blob/4f88aaeb3629249fbbddb4d0784a06962d9e1338/LHM/models/rendering/gs_renderer.py)
uses SMPL-X transforms to animate Gaussian positions and orientations. This gives us a concrete
route to investigate retaining the person's appearance while adding joint control. The browser
package currently contains the results of those transforms, not an exported controllable rig.
The saved state alone is not a standalone browser asset; extraction also needs the matching
model implementation and its body/skinning data.

The first technical proof should load one retained canonical avatar, export the joint hierarchy,
rest transforms, Gaussian attributes and deformation data, then reproduce a known source pose
in the browser. Compare that result against the existing baked frame before trying a new arm
pose. Preserve the model's deformation corrections and coordinate transforms; adding bones to
a posed PLY alone does not establish that the surface is correctly bound to them.

Once joint control works, add idle and walk motion, transitions, navigation, and a hand target.
An agent can then select these actions. A skeleton by itself supplies neither locomotion nor
convincing bottle contact. A simple rigged stand-in can test those mechanics earlier, but would
be an illustrative replacement rather than the reconstructed person's original appearance.
Exact retained source-state availability for the elevator cast must be verified independently
of the existence of the generic LHM export pipeline. No conversion or headset performance has
been demonstrated by this investigation.

## Suggested first experience

Select a person, choose **Talk**, and pause the recorded scene at its current time. Ask a question
about something visible. Hear a generated answer from the selected person's position, with a
transcript and a clear indication that this is AI dialogue. Ask the character to highlight a
known object. End the conversation and restore the previous playback state.

Start with a paused body and a speaking indicator. This proves conversation and scene awareness
without claiming new body or lip animation. Use a standard synthetic voice. A character's persona
is authored; footage alone does not establish the real person's knowledge, memories, or beliefs.
Keep generated responses separate from the original recorded soundtrack and transcript.

## Proposed implementation shape

1. **Entity registry:** adapt existing person/prop IDs into selectable entities with labels,
   position anchors, selection volumes, context, and supported actions. Extra static hotspots
   belong in scene manifests, not per-scene runtime constants.
2. **Interaction controller:** desktop pointing and XR selection emit the same events. Begin with
   one active conversation, explicit microphone activation, cancellation, and a text fallback.
3. **Agent context:** provide the selected entity, reviewed scene facts, playback time, nearby
   entity IDs, and current interaction state. Add session memory for the conversation. Geometry
   and rendering alone do not give a language model knowledge of the scene.
4. **Action interface:** expose a short list such as `inspect_entity`, `highlight_entity`, and
   `seek_recording`. The viewer validates targets and supported operations, executes actions,
   and returns the actual result. The model chooses actions; local code handles rendering,
   collision, and animation on every frame.
5. **Live voice:** use a separate live audio source attached to a reviewed head anchor, with a
   documented approximate fallback when needed. Keep original playback paused during the first
   experiment. Stop microphone tracks and responses on exit, scene change, or cancellation.

One candidate is the [OpenAI Realtime API](https://developers.openai.com/api/docs/guides/realtime),
which supports browser speech conversations, interruptions, and conversation state. Its browser
flow uses a server-created ephemeral credential and WebRTC. Keep the project key on the server.
[Function tools](https://developers.openai.com/api/docs/guides/realtime-mcp) let application code
execute actions requested during the conversation. This is an integration option, not a selected
model, cost estimate, or tested Quest configuration.

## Next experiment: one prop that changes the conversation

Use a prop already rendered separately. Give it explicit recorded, held, and released states so
the recorded track and hand controller never both write its transform. Support reset to the
recording. Validate collision and the visual result before adding throwing or a physical handoff.
Send successful pickup/release events to the character so its response reflects actual state.

A lamp or door can use deterministic rules; it does not need a separate language-model session.
Multiple characters can have different contexts and memories while sharing the same agent
implementation. Autonomous conversations between characters can follow after one reliable turn.

## Evidence needed before calling the prototype successful

- Selection consistently targets the intended visible entity on desktop and a physical headset.
- Speech comes from the selected anchor and follows head turns; original audio does not double.
- Interruption and cancellation stop speech, microphone capture, and pending actions correctly.
- Scene changes cannot apply a late response or action to an entity in the new scene.
- A claimed highlight, seek, or pickup is confirmed by actual viewer state.
- Measure response delay, API usage, and headset frame rate; none is established by this proposal.

The first product choice is whether the character explains the recorded moment or participates
in a fictional continuation. The second is whether conversation alone makes a compelling demo,
or whether manipulating one prop is essential to the experience.
