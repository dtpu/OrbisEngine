import {
  OpenAIRealtimeWebRTC,
  RealtimeAgent,
  RealtimeSession,
  tool,
} from '@openai/agents/realtime';
import type { Quaternion, Vector3 } from 'three';
import { sceneCharacterInstructions } from './character-prompt';

export type AgentIdentity = {
  sceneId: string;
  personId: string;
  personLabel: string;
  objectId: string;
};

type Options = {
  identity: () => AgentIdentity;
  status: (message: string) => void;
  action: (name: string, args: unknown) => string | Promise<string>;
  speaking: (value: boolean) => void;
  speechStarted: () => boolean;
  /** Normal session-cap cleanup only; failures and explicit stops never request renewal. */
  expired?: () => void;
};
type ToolCall = Parameters<OpenAIRealtimeWebRTC['sendFunctionCallOutput']>[0];
type ClientEvent = Parameters<OpenAIRealtimeWebRTC['sendEvent']>[0];

function providerErrorCode(event: unknown): string | null {
  // RealtimeSession forwards the transport's `{ type: 'error', error: rawEvent }` envelope.
  // Keep this deliberately narrow so arbitrary SDK, auth, quota, or transport failures stay fatal.
  if (!event || typeof event !== 'object' || (event as { type?: unknown }).type !== 'error')
    return null;
  const raw = (event as { error?: unknown }).error;
  if (!raw || typeof raw !== 'object' || (raw as { type?: unknown }).type !== 'error') return null;
  const code = (raw as { error?: { code?: unknown } }).error?.code;
  return typeof code === 'string' && /^[a-z0-9_]{1,96}$/.test(code) ? code : null;
}

/** SDK tool continuations must pass the same local gate as microphone turns. */
class SceneVoiceTransport extends OpenAIRealtimeWebRTC {
  allowResponse = () => false;
  finishTool: (call: ToolCall, output: string) => void = () => {};

  override sendEvent(event: ClientEvent) {
    if (event.type === 'response.create' && !this.allowResponse()) return;
    super.sendEvent(event);
  }

  override requestResponse(response?: Record<string, unknown>) {
    // All responses, including tool followups, are explicitly scheduled by the client.
    if (response && this.allowResponse()) super.requestResponse(response);
  }

  override sendFunctionCallOutput(call: ToolCall, output: string) {
    this.finishTool(call, output);
  }

  completeTool(call: ToolCall, output: string) {
    super.sendFunctionCallOutput(call, output, false);
  }
}

function setupFailureMessage(error: unknown): string {
  // SDK error events wrap an Error, while connect() rejects with that Error directly.
  // Inspect only for a known condition; provider details never become UI text or logs.
  let detail = error;
  let rateLimited = false;
  for (let depth = 0; depth < 3 && detail && typeof detail === 'object'; depth++) {
    const record = detail as {
      code?: unknown;
      message?: unknown;
      error?: unknown;
      status?: unknown;
    };
    const message = typeof record.message === 'string' ? record.message.slice(0, 4096) : '';
    if (
      record.code === 'credit_balance_exhausted' ||
      /credit_balance_exhausted/i.test(message) ||
      (/credit balance/i.test(message) && /exhausted|depleted|insufficient/i.test(message))
    ) {
      return 'Server OpenAI account needs credits · microphone off';
    }
    rateLimited ||=
      record.status === 429 || /Realtime call request failed with status 429\b/i.test(message);
    detail = record.error;
  }
  if (rateLimited) return 'Voice provider HTTP 429: account quota or rate limit · microphone off';
  return error instanceof DOMException && error.name === 'NotAllowedError'
    ? 'Microphone permission was declined. Enable access and try again.'
    : 'Could not start voice. Re-enter VR to retry.';
}

/** Permissioned microphone conversation; the recording retains its own transport. */
export class BottleAgentClient {
  private session: RealtimeSession | null = null;
  private transport: SceneVoiceTransport | null = null;
  private microphone: MediaStream | null = null;
  private audio: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private panner: PannerNode | null = null;
  private gain: GainNode | null = null;
  private remote: HTMLAudioElement | null = null;
  private abort: AbortController | null = null;
  private generation = 0;
  private epoch = 0;
  private ready = false;
  private playing = true;
  private muted = false;
  private busy = false;
  private speechStarting = false;
  private speech: { id: string; epoch: number } | null = null;
  private latestState = '';
  private pendingReaction = false;
  private responseKey = '';
  private inFlightKey = '';
  private inFlightId = '';
  private responses = new Map<string, number>();
  private calls = new Map<string, { epoch: number; executed: boolean; completed: boolean }>();
  private toolFollowup = false;
  private pendingTools = 0;
  private toolRounds = 0;
  private sequence = 0;
  private settleConnection: (() => void) | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private responseTimer: ReturnType<typeof setTimeout> | null = null;
  private providerErrorCode: string | null = null;

  constructor(private readonly options: Options) {}

  get connected() {
    return this.ready;
  }

  /** A code-only provider diagnostic; provider messages and credentials are never retained. */
  get lastProviderErrorCode() {
    return this.providerErrorCode;
  }

  async connect(): Promise<void> {
    const pendingReaction = this.pendingReaction;
    this.disconnect();
    this.providerErrorCode = null;
    this.pendingReaction = pendingReaction && !this.playing;
    const generation = this.generation;
    const current = () => generation === this.generation;
    const abort = new AbortController();
    this.abort = abort;
    const completion = new Promise<void>((resolve) => {
      this.settleConnection = resolve;
    });
    this.options.status('Connecting · allow microphone access');
    this.timer = setTimeout(() => {
      if (current()) this.fail('Connection timed out. Re-enter VR to retry.');
    }, 25_000);
    const start = async () => {
      try {
        const audio = new AudioContext();
        this.audio = audio;
        // Start both permissioned browser operations within the Enter VR gesture.
        // The caller can immediately request XR without awaiting network setup.
        const resume = audio.resume();
        const capture = navigator.mediaDevices
          .getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
          })
          .then((microphone) => {
            if (!current()) {
              microphone.getTracks().forEach((track) => track.stop());
              return null;
            }
            this.microphone = microphone;
            return microphone;
          });
        const [microphone] = await Promise.all([capture, resume]);
        if (!current() || !microphone) return;
        microphone.getAudioTracks().forEach((track) => (track.enabled = !this.muted));
        const response = await fetch('/api/bottle-agent/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.options.identity()),
          signal: abort.signal,
        });
        if (!current()) return;
        if (!response.ok) {
          this.fail(
            response.status === 503
              ? 'The character needs the server’s OpenAI connection configured.'
              : 'Could not connect. Re-enter VR to retry.',
          );
          return;
        }
        const secret: { value?: string; model?: string } = await response.json();
        if (!current()) return;
        if (!/^ek_[A-Za-z0-9_-]{1,512}$/.test(secret.value ?? '')) {
          this.fail('Invalid conversation credentials.');
          return;
        }
        const remote = document.createElement('audio');
        remote.muted = true; // The only audible route is our spatial Web Audio graph.
        this.remote = remote;
        const transport = new SceneVoiceTransport({
          mediaStream: microphone,
          audioElement: remote,
          changePeerConnection: (peer) => {
            peer.addEventListener('track', (event) => {
              if (!current() || !event.streams[0]) return;
              this.source?.disconnect();
              this.source = audio.createMediaStreamSource(event.streams[0]);
              this.panner ??= audio.createPanner();
              this.panner.panningModel = 'HRTF';
              this.panner.distanceModel = 'inverse';
              this.panner.refDistance = 1;
              this.panner.rolloffFactor = 0.5;
              this.gain ??= audio.createGain();
              this.gain.gain.value = 0;
              this.panner.disconnect();
              this.gain.disconnect();
              this.source.connect(this.panner).connect(this.gain).connect(audio.destination);
              void remote.play().catch(() => {});
            });
            return peer;
          },
        });
        this.transport = transport;
        transport.allowResponse = () => current() && this.ready && !this.playing;
        transport.finishTool = (call, output) => {
          const entry = this.calls.get(call.callId);
          if (!current() || !this.allowed(entry?.epoch) || !entry || entry.completed) return;
          entry.completed = true;
          this.pendingTools = Math.max(0, this.pendingTools - 1);
          transport.completeTool(call, output.slice(0, 3000));
          this.toolFollowup = true;
          this.flushFollowup();
        };
        const agent = new RealtimeAgent({
          name: 'Scene character',
          instructions: sceneCharacterInstructions(),
          voice: 'marin',
          tools: [
            ['face_player', 'Face the visitor if currently possible.'],
            ['show_return_target', 'Show the available bottle receiving target.'],
            [
              'offer_replay',
              'Tell the visitor to press X on the left controller to replay, without starting playback.',
            ],
          ].map(([name, description]) =>
            tool({
              name,
              description,
              parameters: {
                type: 'object',
                properties: {},
                required: [],
                additionalProperties: false,
              },
              execute: async (args, _context, details) => {
                const id = details?.toolCall?.callId;
                const entry = id ? this.calls.get(id) : undefined;
                if (!current() || !entry || entry.executed || !this.allowed(entry.epoch)) {
                  return 'Action cancelled: the scene changed.';
                }
                entry.executed = true;
                if (
                  !args ||
                  typeof args !== 'object' ||
                  Array.isArray(args) ||
                  Object.keys(args).length
                ) {
                  return 'Action rejected: expected empty arguments.';
                }
                try {
                  return await this.options.action(name, args);
                } catch {
                  return 'Action failed.';
                }
              },
            }),
          ),
        });
        const session = new RealtimeSession(agent, {
          transport,
          model: secret.model || 'gpt-realtime',
          tracingDisabled: true,
          config: {
            outputModalities: ['audio'],
            audio: {
              input: {
                // The SDK otherwise enables a separate transcription model by default.
                transcription: null,
                turnDetection: {
                  type: 'server_vad',
                  createResponse: false,
                  interruptResponse: false,
                },
              },
              output: { voice: 'marin' },
            },
            providerData: { max_output_tokens: 256 },
          },
        });
        this.session = session;
        transport.on('*', (event) => {
          if (current()) this.receive(event);
        });
        session.on('error', (event) => {
          if (!current()) return;
          const code = providerErrorCode(event);
          if (code) this.providerErrorCode = code;
          if (this.ready && code === 'response_cancel_not_active') {
            return;
          }
          this.fail(
            this.ready
              ? 'Voice connection failed. Re-enter VR to retry.'
              : setupFailureMessage(event),
          );
        });
        transport.on('connection_change', (state) => {
          // Failed setup closes the transport before the SDK emits the actual error.
          if (current() && this.ready && state === 'disconnected')
            this.fail('Conversation ended · microphone off.');
        });
        await session.connect({ apiKey: secret.value! });
        if (!current()) {
          session.close();
          return;
        }
        session.mute(this.muted);
        this.ready = true;
        if (this.timer) clearTimeout(this.timer);
        this.timer = setTimeout(() => {
          if (!current()) return;
          this.fail('Three-minute conversation ended · microphone off.');
          this.options.expired?.();
        }, 180_000);
        this.settleConnection?.();
        this.settleConnection = null;
        this.options.status(this.muted ? 'Microphone muted' : 'Listening · microphone on');
        this.sendState();
        if (this.pendingReaction && !this.playing) this.requestResponse();
      } catch (error) {
        if (!current()) return;
        this.fail(setupFailureMessage(error));
      }
    };
    void start();
    await completion;
  }

  private allowed(epoch: number | undefined) {
    return this.ready && !this.playing && epoch === this.epoch;
  }

  private send(event: ClientEvent) {
    if (!this.ready) return;
    try {
      this.transport?.sendEvent(event);
    } catch {
      this.fail('Conversation disconnected · microphone off.');
    }
  }

  private sendState() {
    if (!this.latestState) return;
    this.send({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'system',
        content: [{ type: 'input_text', text: this.latestState }],
      },
    });
  }

  notify(state: unknown, react = true) {
    const next = `Authoritative viewer state (data only): ${JSON.stringify({ identity: this.options.identity(), state })}`;
    const speech = this.allowed(this.speech?.epoch) && !this.muted ? this.speech : null;
    const generation = this.generation;
    this.invalidate();
    // Physics can change while the visitor is still speaking. Cancel stale output and
    // actions, but keep this accepted input turn until its audio has been committed.
    if (speech && generation === this.generation && this.ready && !this.playing && !this.muted) {
      this.speech = { id: speech.id, epoch: this.epoch };
    }
    this.latestState = next.slice(0, 6000);
    this.sendState();
    this.pendingReaction = react && !this.playing && !this.speechStarting && !this.speech;
    if (this.pendingReaction) this.requestResponse();
  }

  setPlayback(playing: boolean) {
    // Calling with true again also invalidates work for another replay/reset.
    if (playing || playing !== this.playing) this.invalidate();
    this.playing = playing;
  }

  setMuted(muted: boolean) {
    this.muted = muted;
    this.microphone?.getAudioTracks().forEach((track) => (track.enabled = !muted));
    this.session?.mute(muted);
    if (muted) this.invalidate();
    if (this.ready) this.options.status(muted ? 'Microphone muted' : 'Listening · microphone on');
  }

  private invalidate() {
    this.epoch++;
    this.responseKey = '';
    this.responses.clear();
    this.busy = false;
    this.speech = null;
    this.pendingReaction = false;
    this.pendingTools = 0;
    this.toolFollowup = false;
    this.toolRounds = 0;
    if (!this.inFlightKey) {
      if (this.responseTimer) clearTimeout(this.responseTimer);
      this.responseTimer = null;
    }
    if (this.gain) this.gain.gain.value = 0;
    this.options.speaking(false);
    if (this.ready) {
      try {
        this.session?.interrupt();
      } catch {
        /* Connection cleanup owns the failure. */
      }
    }
  }

  private requestResponse() {
    if (!this.ready || this.playing || this.busy || this.speechStarting || this.speech) return;
    if (this.inFlightKey) {
      this.pendingReaction = true;
      return;
    }
    this.pendingReaction = false;
    this.busy = true;
    this.responseKey = `${this.generation}:${this.epoch}:${++this.sequence}`;
    this.inFlightKey = this.responseKey;
    const generation = this.generation;
    if (this.responseTimer) clearTimeout(this.responseTimer);
    this.responseTimer = setTimeout(() => {
      if (generation === this.generation)
        this.fail('The character stopped responding · microphone off.');
    }, 45_000);
    this.send({
      type: 'response.create',
      response: { metadata: { wander_turn: this.responseKey } },
    });
  }

  private flushFollowup() {
    if (!this.toolFollowup || this.busy || this.pendingTools || this.playing || !this.ready) return;
    this.toolFollowup = false;
    if (++this.toolRounds > 3) {
      this.options.status('The character reached its action limit. Speak again to continue.');
      return;
    }
    this.requestResponse();
  }

  private receive(event: Record<string, any>) {
    switch (event.type) {
      case 'input_audio_buffer.speech_started': {
        if (!this.ready || this.muted || typeof event.item_id !== 'string') return;
        this.speechStarting = true;
        let accepted = false;
        try {
          accepted = this.options.speechStarted();
        } finally {
          this.speechStarting = false;
        }
        if (accepted && !this.playing && this.ready) {
          this.invalidate();
          this.speech = { id: event.item_id, epoch: this.epoch };
        }
        break;
      }
      case 'input_audio_buffer.committed': {
        const accepted =
          this.speech?.id === event.item_id && this.allowed(this.speech?.epoch) && !this.muted;
        if (this.speech?.id === event.item_id) this.speech = null;
        if (accepted) this.requestResponse();
        else if (typeof event.item_id === 'string')
          this.send({ type: 'conversation.item.delete', item_id: event.item_id });
        break;
      }
      case 'response.created': {
        const id = event.response?.id;
        if (event.response?.metadata?.wander_turn === this.inFlightKey && typeof id === 'string')
          this.inFlightId = id;
        if (
          !this.allowed(this.epoch) ||
          !this.busy ||
          !this.responseKey ||
          event.response?.metadata?.wander_turn !== this.responseKey ||
          typeof id !== 'string'
        ) {
          if (typeof id === 'string') this.send({ type: 'response.cancel', response_id: id });
          break;
        }
        this.responses.set(id, this.epoch);
        break;
      }
      case 'output_audio_buffer.started':
        if (this.allowed(this.responses.get(event.response_id))) {
          if (this.gain) this.gain.gain.value = 1;
          this.options.speaking(true);
        } else {
          if (this.gain) this.gain.gain.value = 0;
          this.send({ type: 'output_audio_buffer.clear' });
        }
        break;
      case 'output_audio_buffer.stopped':
      case 'output_audio_buffer.cleared':
        if (this.gain) this.gain.gain.value = 0;
        this.options.speaking(false);
        break;
      case 'response.output_item.done': {
        const item = event.item;
        if (
          item?.type !== 'function_call' ||
          !this.allowed(this.responses.get(event.response_id)) ||
          typeof item.call_id !== 'string' ||
          item.call_id.length > 128 ||
          this.calls.has(item.call_id)
        )
          return;
        if (this.calls.size >= 64) {
          this.fail('Conversation action limit reached · microphone off.');
          return;
        }
        this.calls.set(item.call_id, { epoch: this.epoch, executed: false, completed: false });
        this.pendingTools++;
        break;
      }
      case 'response.done':
        if (event.response?.id === this.inFlightId) {
          this.inFlightId = '';
          this.inFlightKey = '';
          if (!this.pendingTools) {
            if (this.responseTimer) clearTimeout(this.responseTimer);
            this.responseTimer = null;
          }
          // Let the SDK finish its response sequencing before another explicit request.
          const generation = this.generation;
          queueMicrotask(() => {
            if (generation !== this.generation) return;
            if (this.pendingReaction) this.requestResponse();
            else this.flushFollowup();
          });
        }
        if (!this.allowed(this.responses.get(event.response?.id))) return;
        if (!this.pendingTools) {
          if (this.responseTimer) clearTimeout(this.responseTimer);
          this.responseTimer = null;
        }
        this.busy = false;
        this.responseKey = '';
        if (event.response.status !== 'completed') {
          this.invalidate();
          this.options.status('The character could not answer. Speak again to retry.');
        }
        break;
    }
  }

  updateAudio(source: Vector3, listener: Vector3, rotation: Quaternion, stature: number) {
    if (!this.audio || !this.panner) return;
    const scale = 1 / Math.max(stature, 0.001);
    this.panner.positionX.value = source.x * scale;
    this.panner.positionY.value = source.y * scale;
    this.panner.positionZ.value = source.z * scale;
    const ears = this.audio.listener;
    ears.positionX.value = listener.x * scale;
    ears.positionY.value = listener.y * scale;
    ears.positionZ.value = listener.z * scale;
    const forward = source.clone().set(0, 0, -1).applyQuaternion(rotation);
    const up = source.clone().set(0, 1, 0).applyQuaternion(rotation);
    ears.forwardX.value = forward.x;
    ears.forwardY.value = forward.y;
    ears.forwardZ.value = forward.z;
    ears.upX.value = up.x;
    ears.upY.value = up.y;
    ears.upZ.value = up.z;
  }

  private fail(message: string) {
    this.disconnect();
    this.options.status(message);
  }

  disconnect() {
    this.generation++;
    this.ready = false;
    this.inFlightKey = '';
    this.inFlightId = '';
    this.invalidate();
    this.settleConnection?.();
    this.settleConnection = null;
    this.abort?.abort();
    this.abort = null;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const session = this.session;
    this.session = null;
    this.transport = null;
    try {
      session?.close();
    } catch {
      /* Continue releasing our media if SDK close fails. */
    }
    this.microphone?.getTracks().forEach((track) => track.stop());
    this.microphone = null;
    this.source?.disconnect();
    this.source = null;
    this.panner?.disconnect();
    this.panner = null;
    this.gain?.disconnect();
    this.gain = null;
    if (this.remote) {
      this.remote.pause();
      this.remote.srcObject = null;
      this.remote = null;
    }
    if (this.audio) void this.audio.close().catch(() => {});
    this.audio = null;
    this.calls.clear();
    this.inFlightKey = '';
    this.inFlightId = '';
  }
}
