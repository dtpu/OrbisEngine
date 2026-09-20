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

- Move deliberately toward a visible person while facing them. The default sensor is
  0.95 body-heights with a 0.3-second dwell and at least 0.035 body-heights of visitor approach.
  Starting nearby works without backing away first; after firing, departure beyond 1.28
  body-heights rearms that person. A person walking past a stationary visitor does not trigger it.
- Address a nearby, faced person through the microphone, up to 1.8 body-heights away. During an
  existing conversation the limit is 2.52 body-heights and looking down at the bottle is allowed.
  Accepted speech pauses playback immediately; the character answers after the utterance commits.
  Clearly addressing another person changes the speaker without changing bottle ownership.
  Obstructed and hidden people remain unavailable. Speech has no typed-chat fallback.
- Squeeze within 0.12 body-heights of the visible bottle. The reach check uses the controller and bottle paths between
  frames plus known floor and obstacle geometry, so a fast crossing is not accepted through a wall.

There is no floating control panel. Press **X on the left controller** to replay the recording;
holding X does not repeatedly restart it. Leaving VR closes the microphone connection. A connection
failure leaves local bottle handling available. After fixing permission or server configuration,
re-enter VR to retry. Provider errors do not trigger automatic retries.

If Quest reports “The specified session configuration is not supported” despite previously
working, restart Meta Browser and reopen the local viewer. In a headset check, even a bare
`immersive-vr` request without microphone capture failed with the browser's internal runtime
initialization error; restarting the browser restored immersive entry and voice. This error alone
does not identify an unsupported scene or an optional-feature problem.

When the recording is paused, the visitor can hold and release the bottle. Its local simulation
uses the reviewed floor and collision callbacks. A green receiving region is an assisted target
near a recorded held pose: a successful return attaches the bottle to that paused pose and keeps
the recording paused. It is not an animated catch, hand closure, or new body motion. A miss stays
available for pickup; press X to restore an out-of-reach bottle.

Interaction loads the measured collision grid even when desktop walking is disabled. Loose bottles
also use inferred support across single-cell gaps inside the measured floor region,
requiring at least two neighboring floor samples that agree within one grid cell of height.
This repairs sparse sampling gaps, not larger unknown areas or the outside boundary. The visible
bottle has a separate conservative floor clearance so its base does not sink into the support.

A pulsing ring and yellow beacon locate the bottle during playback and after release. During
interaction a cyan bottle with an orange cap replaces the source prop, centered in the visitor's
grip when held; its marker hides in the hand. This is invented appearance, scaled from manifest
dimensions and clamped to 0.09–0.12 body-heights tall for visibility, with a narrow profile that
leaves the glove visible. The locator stays upright when the bottle rotates. It does not replace measured
positions or enlarge the measured wall collider. The separate floor clearance encloses the
replacement bottle in any orientation. The source bottle returns on replay.
Resting support includes an additional 0.03 body-heights of visual clearance above the sampled
floor to keep the bottle visible where that collision surface sits below the rendered floor.

Coarse occupied floor cells can extend above their measured surface. A descending bottle sweeps to
the first blocking boundary near known floor and settles there; side walls and ceilings still bounce
it. This is conservative collision support, not a more precise reconstruction of the floor.

Replay restores the recorded bottle, people, and source time to the start and plays once. It keeps
the visitor's headset position and orientation; an existing voice session keeps its conversation
history while it receives the reset state. Returning the bottle, ending playback, reconnecting
voice, or leaving and re-entering VR does not loop or restart the recording automatically. After
an interruption, press X to restore the recorded exchange.

Main's in-VR scene sidebar remains available on right-controller B, with A selecting a scene.
Switching scenes stops the old microphone. Returning to a cached interrupted scene keeps its bottle
and paused recording, reconnects voice, and preserves the native XR session. The interaction opt-in
carries across scene choices; a scene without a separately packaged thrown object uses normal playback.

## Runtime boundaries

The recording, bottle, and voice session have separate ownership. The viewer freezes the recorded
people at an interruption. It samples a recorded airborne bottle's current position and velocity
into local physics instead of suspending it. The selected person smoothly turns toward the visitor
locally while in conversation range. An accepted agent action can only face the paused body,
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

The shared voice prompt speaks in first person as the selected fictional character, using their
manifest role, current activity, and an invented per-person conversational style: deadpan skeptic,
cocky competitor, or blunt observer. They can disagree, tease, and give dry comebacks rather than
constantly praise or pitch activities. Replies are short ordinary dialogue without assistant
introductions or action narration. Direct questions about whether the
character is real receive a truthful answer. The prompt does not invent actual identities or
recorded memories and cannot grant actions beyond the local viewer's tools.

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
the clip. A `response_cancel_not_active` race during an established connection does not close the
microphone; the already-finished reply remains silent and later speech can get a response.
Other errors still close capture and retain only a sanitized provider code in diagnostic state.
Failed connections are not automatically retried. Input transcription is disabled; there
is no transcript UI. Chrome must run on localhost (for Quest, use the documented ADB reverse
connection) so both microphone permission and the local credential endpoint are available.

## Verification status

Run `bun run test:interaction` for ownership, collision, approach, metadata, credential, SDK, and
scene integration checks. `bun run test:interaction-browser` exercises the real elevator viewer
with synthetic headset/controller input. When another checkout serves the shared media, build
first and set `VIEWER_BUILD_DIR=dist` to serve this build through the browser test's route handler.
No provider calls occur by default. `WANDER_TEST_LIVE_VOICE=1` explicitly enables a short paid voice
smoke test and requires `OPENAI_API_KEY` in that test process's environment.
With that opt-in, `WANDER_TEST_VOICE_WAV=/path/to/owned-speech.wav` injects a local speech fixture
into the test microphone and checks real provider VAD, source pause, and two consecutive replies.

On 2026-09-20 the real-asset browser check verified close approach interrupting advancing playback,
source-audio pause, controller X replay, a grip and short throw into the assisted return region,
automatic smooth whole-body facing, reset of the facing transform, and paused VR re-entry.
The expanded range, locator, rotated controller grip alignment, and recorded/proxy bottle visibility
passed in the actual viewer; screenshots were reviewed with the held bottle clear of the source
actor. The final focused interaction suite contains 95 passing tests. A comparison against the
source footage confirmed the basic held-pose target; this is not a precise new wrist reconstruction.
The ordinary viewer still loops when the experiment is absent. Focused tests, existing XR/audio/
collision regressions, TypeScript/build, and repository formatting were checked separately.

The first live provider check returned HTTP 429 `credit_balance_exhausted`. After switching the
private local credential on 2026-09-20, the live browser test connected successfully and measured
generated audio through the spatial output graph. The full live interaction browser check passed;
the revised prompt produced a short scene-specific greeting without an assistant introduction.
SDK lifecycle/audio-gate unit checks use mocked transport.

The latest real-asset check also dropped bottles at the reported sparse-floor locations using
the production floor callback and visual clearance; both settled above support. Two synthetic
spoken turns through the live provider paused source playback and produced measured audio while
the microphone stayed connected. The replies followed the new dry, competitive style. This checks
the full speech path in Chrome, not the Quest microphone or perceived conversational quality.
The harmless cancellation race and a fatal credential error have separate SDK regression checks.
A later physical Quest failure logged `response_cancel_not_active`: the old client had incorrectly
closed its microphone for that error. After reloading the fix, the headset entered visible VR with
a live microphone track and completed generated responses.

A connected Quest entered a real immersive session and granted microphone permission. The new
credential connected successfully; live microphone samples, outgoing audio, accepted speech events,
and completed provider responses were observed on the headset. This establishes the connection,
not perceived speech quality. Controller comfort, hand tracking, source-soundtrack echo rejection, perceived latency,
and spatial sound still need headset testing. Screenshots and measurements stay in ignored
`.context/evidence/bottle-agent/`.

After merging current main (`c93119a`), the updated character/bottle build entered a visible
immersive session on the Quest with microphone permission granted and voice connected. Browser
interaction checks, TypeScript/build, and formatting also passed after the merge.

Main through `0546438` is now integrated with its persistent scene lifecycle and B/A sidebar.
The combined live-provider browser check passed speech, three reported drop locations with and
without walking enabled, switching away/back without replay, and grabbing after cached reactivation.
Main's sidebar suite also passed cache/session preservation and failure recovery. The final build
entered visible immersive VR on the physical Quest with voice connected (provider HTTP 201).
