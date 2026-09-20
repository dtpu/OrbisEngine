# Interactive world exploration

This is a proposal, not an implemented feature. The first experiment is a VR bottle-tossing
scene: speak to a character through the headset microphone, catch the bottle, and throw it back.
The character reacts to what actually happens. Conversation is voice-only, with no typed chat,
chat panel, or transcript in the intended experience.

## First implementation goal

Make the existing bottle-tossing scene react to a visitor: approaching can prompt a greeting,
catching interrupts the recorded exchange, a character converses and faces the visitor, and a
return throw can be accepted at an assisted catch target. The scene starts in interaction mode
with the recording paused. Replay is an explicit visitor action; Reset restores the starting
scene state and stays paused.
This scope uses the current recorded bodies; exporting a controllable skeleton is a later task.

The scene script is a small state machine. Seed its throw and handoff timing and participants
from the existing object manifest. New dialogue and reactions are generated. The viewer owns
the bottle state and executes validated actions; the agent receives actual events and chooses
what to say or which supported action to request. Proximity greetings require an explicitly
started conversation and a cooldown so walking around does not trigger repeated introductions.

Implement and review these stages in order:

1. **Playback, interrupt, and reset:** opt-in scene entry paused, an immersive Replay control,
   XR selection, grab/hold/release, recorded-motion ownership, and a reliable reset.
   Validate the exact bottle scene visually.
2. **Throw and return:** bounded simulated throws, supported floor/wall contacts, an explicit
   assisted catch region near a recorded receiving pose, misses, and pickup after a miss.
3. **Conversation:** explicit headset microphone activation, spoken answers, scene/event context,
   proximity and bottle reactions, interruption, and cleanup. No connection starts merely on page
   load. If microphone access fails, offer retry or mute status; do not introduce a text fallback.
4. **Facing and actions:** whole-body turning around the character's position, a return marker,
   and an offer to replay. The agent cannot start or loop the recording on its own.
5. **Acceptance:** demonstrate approach, catch, conversation, successful return, miss, reset,
   manual replay, pause, VR re-entry, and scene exit. Check real provider/audio behavior and physical headset behavior separately
   from offline and simulated XR tests. Document any checks that cannot be performed.

Natural reaching, independent head/eye movement, lip sync, and new walking animation are outside
this first implementation. Initial engineering scaffolding exists locally for bottle physics
and the agent connection; it is not yet integrated or a demonstrated viewer feature.

## Voice and event logic

The recording, bottle, and conversation have independent state:

| System       | States and authority                                                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recording    | Paused interaction or a manually started, single replay. Only visitor controls can start playback.                                                   |
| Bottle       | Recorded track, held by visitor, simulated flight, resting after a miss, or accepted by a character. Exactly one system owns its position at a time. |
| Conversation | Off, connecting, listening, speaking, or suspended during replay. Microphone use requires explicit activation and browser/headset permission.        |

Stopping the recording freezes its people and soundtrack, not the application. Head and hand
tracking, locomotion, bottle simulation, and live voice continue on their own updates. The first
prototype does not synthesize new body animation while the recorded person is paused.

Choose one active conversational character by pointing and selecting **Talk** in VR. Once voice
is enabled, approaching that character can trigger a greeting. Use separate arrival/departure
distances and a cooldown so small movements do not repeat it. Nearby people do not all answer,
and walking away and back does not silently switch the active character. Use small selection,
microphone, and speaking cues plus audible feedback; the experience must not require reading a
conversation. Internal speech events may be used by the connection without becoming a chat UI.

Seed the scene script from the manifest's recorded participants, held spans, and flights. This
provides the original exchange's timing and roles, not a complete inferred personality or a fixed
dialogue. During interaction, process actual events in this order:

1. Validate and apply the local event immediately: proximity, grab, release, assisted return,
   miss, replay, or reset. Controller grip grabs a reachable bottle; releasing grip supplies the
   throw velocity. Do not wait for the agent before transferring ownership or advancing physics.
2. Update scene context with the bottle owner, recording state, selected character, and event.
   Send meaningful changes, not every frame, to the active voice session.
3. Let the agent choose a spoken response and optionally request a supported action such as
   facing the visitor, showing a return target, or offering replay. Validate that the action is
   still allowed in the current scene and return its actual result.
4. Cancel obsolete speech and pending actions when the visitor interrupts, resets, replays, or
   leaves. A late response must not act on a newer scene state. Keep conversation history across
   a replay/reset but explicitly tell the agent what physical state changed.

An assisted return succeeds only when the simulated bottle reaches an available receiving
region with a clear path. Otherwise it continues its flight or lands for pickup. The agent cannot
declare a successful catch, teleport the bottle, start playback, or issue unsupported walking or
joint animation. The initial receiving region is approximate and needs visual review against a
paused pose; accepting the bottle does not imply an animated reach or hand closure.

While listening, normal speech starts a turn; speaking over an answer interrupts it. A VR mic
control mutes/unmutes the input, and ending Talk or leaving the scene releases the microphone
and connection. Source replay suspends live input and answers. Connection failure leaves local
movement and bottle interaction working, with a retry cue and no keyboard requirement.

## Manual replay while staying in VR

Interaction mode is the default for this experiment. Entering or re-entering VR does not start
the source recording. Walking, hand input, and an explicitly connected conversation keep working
while the recording is paused. The current bodies hold their recorded poses until we add further
animation; whole-body facing remains an illustrative interaction.

Provide these controls inside the immersive scene, accessible with controllers:

- **Talk / microphone:** select a character and enable voice, with mute and end-conversation
  controls. Show whether input is active, suspended, or unavailable without a chat panel.
- **Replay recording:** deliberately restore the original character/bottle state and play once
  from the beginning. Keep the visitor's position and orientation. Make the bottle reset clear
  in the control's description. Retain conversation history and notify the agent of the reset.
- **Pause / continue:** stop or continue that single playback without leaving VR. A catch can
  also interrupt playback and transfer the bottle to the visitor. Continue is available only
  while the original exchange is still intact. Once the visitor takes ownership of the bottle,
  use Replay to explicitly restore the recorded exchange instead of silently snapping it back.
- **Reset scene:** restore the original starting state and remain in interaction mode, paused.

At the end of a replay, stop at a valid final pose and return to interaction mode. Returning the
bottle, completing a conversation, reconnecting an agent, or re-entering VR must never restart
the recording automatically. During source playback, suspend agent speech and microphone input
so the recorded soundtrack does not become a conversational turn; restore the user's prior
conversation/microphone choice when playback stops. An agent may offer replay, but the visitor
initiates it through the control.

Implementation must coordinate the video's autoplay/loop flags, the viewer's initial transport
state and modulo loop, and the existing XR-entry auto-play handler. An HTML-only button is not
an immersive VR control. The current standard viewer's looping behavior remains outside this
opt-in experiment. Verify that the scene still accepts input after the recording has ended.

## Feasibility in the current viewer

| Interaction                                                 | Assessment                     | Main work                                                                                   |
| ----------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| Point at a person or object and inspect it                  | Small first step               | Selection volumes, selection cues, and XR input routing.                                    |
| Speak to a person and hear an answer from their location    | Feasible prototype             | Scene context, a voice session, a live audio route, and conversation controls.              |
| Ask a character for a return target or a replay offer       | Feasible prototype             | Validated viewer actions; the visitor starts replay manually.                               |
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
The local elevator sample archive contains both cast tracks and motion sidecars, but no
`canonical-state.pt`, `source-poses.pt`, or GLB/FBX rig export. A canonical state is retained for
one kitchen reconstruction; that is not the elevator cast. Recover the matching elevator state
from the original author outputs before attempting an appearance-preserving export. If that
state cannot be recovered, regenerating it may change the person's appearance. No conversion
or headset performance has been demonstrated by this investigation.

## Suggested first experience

Enter the paused scene, select a person, and choose **Talk** to enable the headset microphone.
Approach and greet the character. Hear a generated answer from their position, with a speaking
cue and an audible introduction that identifies this as an AI character. Use **Replay recording**
to watch the original exchange once; live conversation is suspended during playback. Catch the
bottle with grip to interrupt the exchange, then talk while holding it. Throw it toward the
assisted receiving region or miss and pick it up. The character reacts to the result. Remain in
interaction mode until deliberately choosing Replay or Reset; returning the bottle never starts
the source recording on its own.

Start with a paused body and a speaking indicator. This proves conversation and scene awareness
without claiming new body or lip animation. Use a standard synthetic voice. A character's persona
is authored; footage alone does not establish the real person's knowledge, memories, or beliefs.
Keep generated responses separate from the original recorded soundtrack and transcript.

## Proposed implementation shape

1. **Entity registry:** adapt existing person/prop IDs into selectable entities with labels,
   position anchors, selection volumes, context, and supported actions. Extra static hotspots
   belong in scene manifests, not per-scene runtime constants.
2. **Interaction controller:** XR selection chooses one active conversation. Use explicit
   microphone activation, voice interruption, and cancellation. Route immersive menu selections
   before teleport input, and use grip for bottle handling. No typed input is part of this scope.
3. **Agent context:** provide the selected entity, reviewed scene facts, playback time, nearby
   entity IDs, and current interaction state. Add session memory for the conversation. Geometry
   and rendering alone do not give a language model knowledge of the scene.
4. **Action interface:** expose a short list such as `face_player`, `show_return_target`, and
   `offer_replay`. The last action offers the existing manual control; it never starts playback.
   The viewer validates targets and supported operations, executes actions, and returns the
   actual result. The model chooses actions; local code handles rendering, collision, and
   animation on every frame.
5. **Live voice:** use a separate live audio source attached to a reviewed head anchor, with a
   documented approximate fallback when needed. Keep original playback paused during live
   conversation. Stop microphone tracks and responses on exit, scene change, or ending Talk.
   Interrupting an answer cancels its speech and pending actions while keeping listening active.

One candidate is the [OpenAI Realtime API](https://developers.openai.com/api/docs/guides/realtime),
which supports browser speech conversations, interruptions, and conversation state. Its browser
flow uses a server-created ephemeral credential and WebRTC. Keep the project key on the server.
[Function tools](https://developers.openai.com/api/docs/guides/realtime-mcp) let application code
execute actions requested during the conversation. This is an integration option, not a selected
model, cost estimate, or tested Quest configuration.

## Bottle ownership and later expansion

Use the bottle already rendered separately. Give it explicit ownership states so the recorded
track, hand controller, and physics never write its transform simultaneously. Validate collision,
release motion, and the receiving region against the visible scene. Send confirmed pickup,
return, and miss events to the character so its responses reflect actual state.

A lamp or door can use deterministic rules; it does not need a separate language-model session.
Multiple characters can have different contexts and memories while sharing the same agent
implementation. Autonomous conversations between characters can follow after one reliable turn.

## Evidence needed before calling the prototype successful

- Selection consistently targets the intended visible entity on a physical headset, with no
  typing needed to select, converse, handle the bottle, replay, or reset.
- Speech comes from the selected anchor and follows head turns; original audio does not double.
- Speaking over an answer cancels its speech and pending actions while listening stays active.
  Ending Talk, leaving the scene, or exiting VR releases microphone capture and the connection.
- Scene changes cannot apply a late response or action to an entity in the new scene.
- A claimed turn, return target, pickup, or return is confirmed by actual viewer state.
- Replay and Reset behave as specified after a grab, missed throw, return, voice interruption,
  disconnect, and VR re-entry. Continue cannot overwrite visitor-owned bottle state.
- Measure response delay, API usage, and headset frame rate; none is established by this proposal.

The chosen first experience is a fictional continuation grounded in the recorded scene, with
voice and bottle interaction both required. Runtime integration, assisted-contact quality,
headset microphone behavior, and device performance remain to be demonstrated.
