# Interactive world exploration

This is a proposal, not an implemented feature. The first experiment is a VR bottle-tossing
scene: speak to a character through the headset microphone, catch the bottle, and throw it back.
The character reacts to what actually happens. Conversation is voice-only, with no typed chat,
chat panel, or transcript in the intended experience.

## First implementation goal

Make the existing bottle-tossing scene react to a visitor: approaching can prompt a greeting,
catching interrupts the recorded exchange, a character converses and faces the visitor, and a
return throw can be accepted at an assisted catch target. The clip starts playing normally while
the visitor watches. Approaching a person, addressing them by voice, or grabbing the bottle
interrupts playback at that moment and enters live interaction. No preliminary Pause or Talk
selection is required. After interruption, Replay is an explicit visitor action; Reset restores
the starting scene state and stays paused.
This scope uses the current recorded bodies; exporting a controllable skeleton is a later task.

The scene script is a small state machine. Seed its throw and handoff timing and participants
from the existing object manifest. New dialogue and reactions are generated. The viewer owns
the bottle state and executes validated actions; the agent receives actual events and chooses
what to say or which supported action to request. A proximity event can select a character and
start interaction without a prior conversation. Use a cooldown so walking around does not trigger
repeated introductions. Voice capture still requires one explicit microphone activation and
browser/headset permission; that setup is separate from interrupting the scene.

Implement and review these stages in order:

1. **Playback, interrupt, and reset:** normal initial playback, proximity/voice/grab interruption,
   an immersive Replay control, XR selection, grab/hold/release, recorded-motion ownership, and a reliable reset.
   Validate the exact bottle scene visually.
2. **Throw and return:** bounded simulated throws, supported floor/wall contacts, an explicit
   assisted catch region near a recorded receiving pose, misses, and pickup after a miss.
3. **Conversation:** explicit headset microphone activation, spoken answers, scene/event context,
   proximity and bottle reactions, interruption, and cleanup. No connection starts merely on page
   load. If microphone access fails, offer retry or mute status; do not introduce a text fallback.
4. **Facing and actions:** whole-body turning around the character's position, a return marker,
   and an offer to replay. After interruption, the agent cannot restart or loop the recording on its own.
5. **Acceptance:** demonstrate approach, catch, conversation, successful return, miss, reset,
   manual replay, pause, VR re-entry, and scene exit. Check real provider/audio behavior and physical headset behavior separately
   from offline and simulated XR tests. Document any checks that cannot be performed.

Natural reaching, independent head/eye movement, lip sync, and new walking animation are outside
this first implementation. Initial engineering scaffolding exists locally for bottle physics
and the agent connection; it is not yet integrated or a demonstrated viewer feature.

## Voice and event logic

The recording, bottle, and conversation have independent state:

| System       | States and authority                                                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Recording    | Normal playback on initial entry, paused live interaction after a trigger, or a manually started replay. After interruption, only visitor controls can restart playback. |
| Bottle       | Recorded track, held by visitor, simulated flight, resting after a miss, or accepted by a character. Exactly one system owns its position at a time.                     |
| Conversation | Off, connecting, listening for an interaction trigger, or conversing after interruption. Microphone use requires explicit activation and browser/headset permission.     |

Stopping the recording freezes its people and soundtrack, not the application. Head and hand
tracking, locomotion, bottle simulation, and live voice continue on their own updates. The first
prototype does not synthesize new body animation while the recorded person is paused.

Approaching a person can interrupt playback and select them as the active conversational
character. Addressing a nearby character by voice or grabbing the bottle are alternative triggers;
none requires a preliminary Talk selection. Resolve one target from proximity, viewing direction,
or the recorded bottle participants, with optional controller selection when needed. Use separate
arrival/departure distances and a cooldown so small movements do not repeat the trigger. Once
interaction begins, latch that target until the visitor changes it. Nearby people do not all
answer, and walking away and back does not silently switch the active character. Use small selection,
microphone, and speaking cues plus audible feedback; the experience must not require reading a
conversation. Internal speech events may be used by the connection without becoming a chat UI.

### Deliberately close approach

The approach zone is a small sensor, not a physical collider or a room-wide pause trigger.
As initial tuning values, require the visitor's head projected onto the floor to be within
0.30 scene body-heights of the character's current torso axis, facing generally toward them,
for about 0.4 seconds. Require departure beyond 0.45 body-heights before a fresh approach can
re-arm. These are proposed interaction settings, not measured geometry or accepted headset
values. Store them as general configuration and tune them against the real scene in VR.

Sample the final headset world pose after both physical movement and locomotion. Use a reviewed
moving character anchor where available; otherwise validate an approximate anchor derived from
the current visible body frame. The person's group origin and the viewer's combined cast centre
are not reliable per-character proximity anchors. Reject hidden people, incompatible vertical
positions, and paths blocked by known collision geometry. At initial entry or replay, do not
fire merely because the visitor already occupies the zone; arm after leaving it. Require visitor
approach rather than a recorded character passing a stationary observer.

Grabbing is independent of this dwell: grip near the visible bottle should interrupt immediately,
even when the visitor's head is outside a person's approach zone. Validate controller-to-bottle
reach and known occlusion, and check motion between frames so a fast bottle is not missed solely
because it crossed the hand between samples. Directed speech is another independent trigger once
the microphone is enabled. Resolve a simultaneous valid grab before an approach greeting so the
agent receives the actual bottle owner in its first context update.

Seed the scene script from the manifest's recorded participants, held spans, and flights. This
provides the original exchange's timing and roles, not a complete inferred personality or a fixed
dialogue. During interaction, process actual events in this order:

1. Validate and apply the local event immediately: proximity, grab, release, assisted return,
   speech onset, miss, replay, or reset. The first interaction trigger pauses the recording and
   original soundtrack at the current time, captures the current scene state, and enters live
   interaction. Controller grip grabs a reachable bottle; releasing grip supplies the
   throw velocity. Do not wait for the agent before transferring ownership or advancing physics.
2. Update scene context with the bottle owner, recording state, selected character, and event.
   Send meaningful changes, not every frame, to the active voice session.
3. Let the agent choose a spoken response and optionally request a supported action such as
   facing the visitor, showing a return target, or offering replay. Validate that the action is
   still allowed in the current scene and return its actual result.
4. Cancel obsolete speech and pending actions when the visitor interrupts, resets, replays, or
   leaves. A late response must not act on a newer scene state. Keep conversation history across
   a replay/reset but explicitly tell the agent what physical state changed.

The transition captures the recording time, character poses, bottle pose, and current owner in
one local state change. If a proximity or voice interruption occurs during a recorded flight,
transfer the bottle's current position and sampled velocity into local physics instead of leaving
it suspended in the air. If it is held by a character, retain that attachment until a valid grab.
A valid visitor grab takes precedence and attaches it to that hand. Derive these states from the
manifest, not clip-specific times. This recorded-flight handoff still needs implementation;
the existing local physics scaffold currently covers grabbing and release from the visitor.

An assisted return succeeds only when the simulated bottle reaches an available receiving
region with a clear path. Otherwise it continues its flight or lands for pickup. The agent cannot
declare a successful catch, teleport the bottle, start playback, or issue unsupported walking or
joint animation. The initial receiving region is approximate and needs visual review against a
paused pose; accepting the bottle does not imply an animated reach or hand closure.

After microphone activation, listen for the visitor addressing a character even while the clip
plays. On a valid speech trigger, pause the original soundtrack before starting an agent answer
and preserve the visitor's first words. Do not disable all input during playback: that would make
voice interruption impossible. Recorded audio must not trigger the agent; echo rejection and
false-trigger behavior need physical headset validation. Generated answers wait until playback
has stopped. While conversing, speaking over an answer interrupts it. A VR mic control
mutes/unmutes input, and ending the conversation or leaving the scene releases the microphone
and connection. Connection failure leaves local movement, interruption, and bottle interaction
working, with a retry cue and no keyboard requirement.

For the voice prototype, prepare the connection after explicit microphone activation and gate
automatic answers until a valid interaction supplies the active character and scene state.
OpenAI's [conversation controls](https://developers.openai.com/api/docs/guides/realtime-conversations#keep-vad-but-disable-automatic-responses)
allow speech detection with automatic responses disabled. Speech activity alone does not prove
that the visitor addressed a character; combine it with target eligibility and headset audio
validation. Keep the connection bounded with timeout and idle cleanup, and report disconnection
without affecting local physics. The current scaffold's automatic responses, typed-input path,
and `resume_recording` tool need revision before integration to match this plan.

## Manual replay while staying in VR

Initial entry starts the source recording normally. An interaction trigger pauses it; subsequent
VR re-entry must preserve that interrupted state rather than automatically restarting playback.
Walking, hand input, and a connected conversation keep working while the recording is paused.
The current bodies hold their recorded poses until we add further animation; whole-body facing
remains an illustrative interaction.

Provide these controls inside the immersive scene, accessible with controllers:

- **Microphone / conversation:** enable voice once, with mute and end-conversation controls.
  Optional character selection changes the target; it is not required for proximity or grab
  interruption. Show whether input is active or unavailable without a chat panel.
- **Replay recording:** deliberately restore the original character/bottle state and play once
  from the beginning. Keep the visitor's position and orientation. Make the bottle reset clear
  in the control's description. Retain conversation history and notify the agent of the reset.
  Cancel current agent speech and allow fresh interaction events to interrupt this replay too.
  Do not immediately pause again merely because the visitor is still inside the same proximity
  region; require departure and a new approach for that trigger, or a fresh grab or speech event.
- **Pause / continue:** stop or continue that single playback without leaving VR. A catch can
  also interrupt playback and transfer the bottle to the visitor. Continue is available only
  while the original exchange is still intact. Once the visitor takes ownership of the bottle,
  use Replay to explicitly restore the recorded exchange instead of silently snapping it back.
- **Reset scene:** restore the original starting state and remain in interaction mode, paused.

At the end of initial playback or a replay, stop at a valid final pose and enter interaction mode;
repeat playback remains manual. Returning the
bottle, completing a conversation, reconnecting an agent, or re-entering VR must never restart
the recording automatically. During source playback, suppress agent answers while keeping
permissioned microphone input available to detect a fresh interaction. Preserve the user's mute
choice. An agent may offer replay, but the visitor initiates it through the control.

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

Enter VR and watch the original exchange play normally. Enable the headset microphone once for
voice interaction. Walk up to a person, address them, or catch the bottle with grip: that event
stops the recording at its current moment and starts live interaction. Hear a generated answer
from the selected character's position, with a speaking cue and an audible introduction that
identifies this as an AI character. Talk while holding the bottle, throw it toward the assisted
receiving region, or miss and pick it up. The character reacts to the result. Remain in
interaction mode until deliberately choosing Replay or Reset; returning the bottle never starts
the source recording on its own.

After interruption, use a paused body and a speaking indicator. This proves conversation and scene awareness
without claiming new body or lip animation. Use a standard synthetic voice. A character's persona
is authored; footage alone does not establish the real person's knowledge, memories, or beliefs.
Keep generated responses separate from the original recorded soundtrack and transcript.

## Proposed implementation shape

1. **Entity registry:** adapt existing person/prop IDs into selectable entities with labels,
   position anchors, selection volumes, context, and supported actions. Extra static hotspots
   belong in scene manifests, not per-scene runtime constants.
2. **Interaction controller:** proximity, directed speech, or a bottle grab interrupts playback
   and resolves one active character; XR selection can override the target. Use explicit
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
- Initial playback runs until an interaction trigger or the end of the clip. Approach, speech,
  and grabbing independently pause it without a preliminary Talk selection. Recorded audio
  cannot trigger interaction, and the visitor's first words are retained.
- Replay and Reset behave as specified after a grab, missed throw, return, voice interruption,
  disconnect, and VR re-entry. Continue cannot overwrite visitor-owned bottle state.
- Measure response delay, API usage, and headset frame rate; none is established by this proposal.

The chosen first experience is a fictional continuation grounded in the recorded scene, with
voice and bottle interaction both required. Runtime integration, assisted-contact quality,
headset microphone behavior, and device performance remain to be demonstrated.
