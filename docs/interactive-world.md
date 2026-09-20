# Interactive bottle prototype

The opt-in first prototype pauses a recorded elevator exchange for a local, headset-driven bottle
interaction. It uses the existing recorded people and separately packaged bottle. The spoken
character is explicitly fictional and generated; it does not claim to be, remember, or speak for
the people in the source video.

Run the local viewer, then open
[the elevator prototype](http://127.0.0.1:5399/fourd.html?demo=elevator&interact=1&xr=1). `?interact=1`
enables the feature; without it, the normal viewer and its loop behavior remain unchanged.
`?interactObject=<object-id>` selects a different separately loaded thrown object when a scene has
more than one. The feature is unavailable if the selected scene has no such object.

## Use in VR

Enter VR starts the recording normally and opens the OpenAI voice connection. Allow microphone
access when the headset asks; it stays available while this VR session is visible. The source stops,
including its recorded audio, when one locally validated event occurs:

- Move deliberately and very close to a visible person while facing them. The default sensor is
  0.30 body-heights, has a 0.4-second dwell, requires visitor movement, and does not re-arm until
  departure beyond 0.45 body-heights. A person walking past a stationary visitor does not trigger
  it.
- Address a nearby, faced person through the microphone. Speech has no typed-chat fallback.
- Squeeze near the visible bottle. The reach check uses the controller and bottle paths between
  frames plus known floor and obstacle geometry, so a fast crossing is not accepted through a wall.

There is no floating control panel. Press **X on the left controller** to replay the recording;
holding X does not repeatedly restart it. Leaving VR closes the microphone connection. A connection
failure leaves local bottle handling available. After fixing permission or server configuration,
re-enter VR to retry. Provider errors do not trigger automatic retries.

When the recording is paused, the visitor can hold and release the bottle. Its local simulation
uses the reviewed floor and collision callbacks. A green receiving region is an assisted target
near a recorded held pose: a successful return attaches the bottle to that paused pose and keeps
the recording paused. It is not an animated catch, hand closure, or new body motion. A miss stays
available for pickup; press X to restore an out-of-reach bottle.

Replay restores the recorded bottle, people, and source time to the start and plays once. It keeps
the visitor's headset position and orientation; an existing voice session keeps its conversation
history while it receives the reset state. Returning the bottle, ending playback, reconnecting
voice, or leaving and re-entering VR does not loop or restart the recording automatically. After
an interruption, press X to restore the recorded exchange.

## Runtime boundaries

The recording, bottle, and voice session have separate ownership. The viewer freezes the recorded
people at an interruption. It samples a recorded airborne bottle's current position and velocity
into local physics instead of suspending it. An accepted agent action can only face the paused body,
show the return target, or offer the manual Replay control. It cannot move a person, declare a
catch, teleport the bottle, or start playback.

Character selection uses current per-person head anchors. A packaged `head.json` is sampled when
valid; otherwise the visible mesh's approximate head is used. The group origin and a combined cast
centre are not person anchors. Candidate visibility, vertical compatibility, and obstruction are
checked locally. The bottle participants, held/free spans, dimensions, and sampled poses come from
the measured object and person manifests, rather than per-clip runtime constants. See
[objects](objects.md) for their timing, coordinate, and evidence limits.

Whole-body facing rotates a paused recorded group about its current anchor. There is no controllable
walking, reaching, eye movement, lip sync, or real character rig. Generated voice is spatialized at
the selected anchor; the original clip only supplies its recorded mix, not isolated character
speech.

## Local voice setup

Voice is available only through the Vite development server. Put `OPENAI_API_KEY` in the private
`.env.local` used to start it; `WANDER_AGENT_MODEL` can select the supported realtime model. Do not
prefix either with `VITE_` and do not put them in browser code. The development plugin creates a
short-lived provider client secret on a same-origin local request and the browser uses the realtime
SDK with that secret. A static build has no middleware and therefore cannot mint voice sessions.

The server limits and validates local session requests and supplies only the selected scene/person/
object identifiers. Each provider connection lasts at most three minutes and answers are short.
Normal expiration renews the connection while VR remains active and visible, using the current
scene state; prior conversation history is not carried across connections. Hiding VR stops capture and pauses the
recording. Returning to visible VR reconnects a previously healthy connection without restarting
the clip. Failed connections are not automatically retried. Input transcription is disabled; there
is no transcript UI. Chrome must run on localhost (for Quest, use the documented ADB reverse
connection) so both microphone permission and the local credential endpoint are available.

## Verification status

Run `bun run test:interaction` for ownership, collision, approach, metadata, credential, SDK, and
scene integration checks. `bun run test:interaction-browser` exercises the real elevator viewer
with synthetic headset/controller input. When another checkout serves the shared media, build
first and set `VIEWER_BUILD_DIR=dist` to serve this build through the browser test's route handler.
No provider calls occur by default. `WANDER_TEST_LIVE_VOICE=1` explicitly enables a short paid voice
smoke test and requires `OPENAI_API_KEY` in that test process's environment.

On 2026-09-20 the real-asset browser check verified close approach interrupting advancing playback,
source-audio pause, controller X replay, a grip and short throw into the assisted return region,
whole-body facing, reset of the facing transform, and paused VR re-entry. A comparison against the
source footage confirmed the basic held-pose target; this is not a precise new wrist reconstruction.
The ordinary viewer still loops when the experiment is absent. Focused tests, existing XR/audio/
collision regressions, TypeScript/build, and repository formatting were checked separately.

The first live provider check returned HTTP 429 `credit_balance_exhausted`. After switching the
private local credential on 2026-09-20, the live browser test connected successfully and measured
generated audio through the spatial output graph. The full live interaction browser check passed.
SDK lifecycle/audio-gate unit checks use mocked transport.

A connected Quest entered an earlier build in a real immersive session and granted microphone
permission. The updated page is loaded there, but the headset was asleep during the new credential
check. Successful generated speech has therefore been measured in the browser, not heard on the
headset. Controller comfort, hand tracking, source-soundtrack echo rejection, perceived latency,
and spatial sound still need headset testing. Screenshots and measurements stay in ignored
`.context/evidence/bottle-agent/`.
