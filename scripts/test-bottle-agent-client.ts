import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { Quaternion, Vector3 } from 'three';
import { BottleAgentClient } from '../src/interaction/bottle-agent-client';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
}
const tick = async () => {
  for (let i = 0; i < 150; i++) await Promise.resolve();
};
class Channel extends EventTarget {
  readyState = 'connecting';
  sent: any[] = [];
  open() {
    this.readyState = 'open';
    this.dispatchEvent(new Event('open'));
  }
  send(value: string) {
    this.sent.push(JSON.parse(value));
  }
  emit(event: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }));
  }
  close() {
    this.readyState = 'closed';
    this.dispatchEvent(new Event('close'));
  }
  acknowledge() {
    const session = this.sent.find((event) => event.type === 'session.update').session;
    this.emit({
      type: 'session.created',
      event_id: 'session',
      session: { ...session, tracing: null },
    });
    this.emit({
      type: 'session.updated',
      event_id: 'ack',
      session,
    });
  }
  responses() {
    return this.sent.filter((e) => e.type === 'response.create');
  }
  created(id = 'response') {
    const request = this.responses().at(-1);
    this.emit({
      type: 'response.created',
      event_id: 'created',
      response: { id, status: 'in_progress', output: [], metadata: request?.response.metadata },
    });
  }
  done(id = 'response', status = 'completed') {
    this.emit({ type: 'response.done', event_id: 'done', response: { id, status, output: [] } });
  }
  speech(id = 'utterance') {
    this.emit({
      type: 'input_audio_buffer.speech_started',
      event_id: 'speech',
      item_id: id,
      audio_start_ms: 0,
    });
  }
  commit(id = 'utterance') {
    this.emit({
      type: 'input_audio_buffer.speech_stopped',
      event_id: 'stop',
      item_id: id,
      audio_end_ms: 1000,
    });
    this.emit({
      type: 'input_audio_buffer.committed',
      event_id: 'commit',
      item_id: id,
      previous_item_id: null,
    });
  }
  tool(id: string, name = 'face_player', args = '{}', responseId = 'response') {
    this.emit({
      type: 'response.output_item.done',
      event_id: 'tool',
      response_id: responseId,
      output_index: 0,
      item: {
        id: `item_${id}`,
        type: 'function_call',
        status: 'completed',
        call_id: id,
        name,
        arguments: args,
      },
    });
  }
}
class Peer extends EventTarget {
  static instances: Peer[] = [];
  channel = new Channel();
  closed = false;
  connectionState = 'new';
  ontrack: ((event: any) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  constructor() {
    super();
    Peer.instances.push(this);
  }
  addTrack() {}
  getSenders() {
    return [];
  }
  createDataChannel() {
    return this.channel;
  }
  async createOffer() {
    return { type: 'offer', sdp: 'fixture offer' };
  }
  async setLocalDescription() {}
  async setRemoteDescription() {}
  close() {
    this.closed = true;
  }
  track() {
    const event = new Event('track') as Event & { streams: MediaStream[] };
    event.streams = [stream()];
    this.ontrack?.(event);
    this.dispatchEvent(event);
  }
}
const parameter = () => ({ value: 0 });
class AudioNode {
  disconnected = false;
  gain = parameter();
  positionX = parameter();
  positionY = parameter();
  positionZ = parameter();
  connect(node: AudioNode) {
    return node;
  }
  disconnect() {
    this.disconnected = true;
  }
}
class Audio {
  static instances: Audio[] = [];
  closed = false;
  destination = new AudioNode();
  nodes: AudioNode[] = [];
  listener = {
    positionX: parameter(),
    positionY: parameter(),
    positionZ: parameter(),
    forwardX: parameter(),
    forwardY: parameter(),
    forwardZ: parameter(),
    upX: parameter(),
    upY: parameter(),
    upZ: parameter(),
  };
  constructor() {
    Audio.instances.push(this);
  }
  node() {
    const node = new AudioNode();
    this.nodes.push(node);
    return node;
  }
  createMediaStreamSource() {
    return this.node();
  }
  createPanner() {
    return this.node();
  }
  createGain() {
    return this.node();
  }
  async resume() {}
  async close() {
    this.closed = true;
  }
}
const originals = new Map<string, PropertyDescriptor | undefined>();
function replace(name: string, value: unknown) {
  if (!originals.has(name)) originals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
}
let clients: BottleAgentClient[];
let stopped: number;
let requests: string[];
let captures: MediaTrackConstraints[];
let tracks: { enabled: boolean; stop: () => void }[];
const stream = () => {
  const track = { enabled: true, stop: () => stopped++ };
  tracks.push(track);
  return { getTracks: () => [track], getAudioTracks: () => [track] } as unknown as MediaStream;
};
function setup(
  action: (name: string, args: unknown) => string | Promise<string> = () => 'ok',
  expired?: () => void,
) {
  const statuses: string[] = [];
  const actions: string[] = [];
  const speaking: boolean[] = [];
  let acceptSpeech = true;
  const identity = {
    sceneId: 'bottle',
    personId: 'person',
    personLabel: 'Character',
    objectId: 'bottle',
  };
  const client = new BottleAgentClient({
    identity: () => identity,
    expired,
    status: (value) => statuses.push(value),
    speaking: (value) => speaking.push(value),
    speechStarted: () => {
      if (acceptSpeech) {
        client.setPlayback(false);
        client.notify({ event: 'directed speech' });
      }
      return acceptSpeech;
    },
    action: (name, args) => {
      actions.push(name);
      return action(name, args);
    },
  });
  clients.push(client);
  return {
    client,
    statuses,
    actions,
    speaking,
    identity,
    rejectSpeech: () => {
      acceptSpeech = false;
    },
  };
}
async function connected(
  action?: (name: string, args: unknown) => string | Promise<string>,
  expired?: () => void,
) {
  const fixture = setup(action, expired);
  const promise = fixture.client.connect();
  await tick();
  const peer = Peer.instances.at(-1)!;
  expect(peer).toBeDefined();
  const channel = peer.channel;
  channel.open();
  channel.acknowledge();
  await promise;
  expect(fixture.client.connected).toBe(true);
  return { ...fixture, channel, peer };
}
function react(client: BottleAgentClient, channel: Channel, responseId = 'response') {
  client.setPlayback(false);
  client.notify({ event: 'bottle caught' });
  channel.created(responseId);
}
beforeEach(() => {
  clients = [];
  stopped = 0;
  requests = [];
  captures = [];
  tracks = [];
  Peer.instances = [];
  Audio.instances = [];
  replace('RTCPeerConnection', Peer);
  replace('AudioContext', Audio);
  replace('document', {
    createElement: () => ({ muted: false, srcObject: null, pause() {}, async play() {} }),
  });
  replace('navigator', {
    mediaDevices: {
      getUserMedia: async (constraints: { audio: MediaTrackConstraints }) => {
        captures.push(constraints.audio);
        return stream();
      },
    },
  });
  replace('fetch', async (url: string) => {
    requests.push(String(url));
    return String(url).startsWith('/api')
      ? Response.json({ value: 'ek_fixture', model: 'gpt-realtime' })
      : new Response('fixture answer');
  });
});
afterEach(() => {
  clients.forEach((client) => client.disconnect());
  for (const [name, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
  originals.clear();
});

describe('SDK bottle voice lifecycle and local response authority', () => {
  test('connect starts capture and audio unlock synchronously before XR can consume the gesture', async () => {
    const resumed = deferred<void>();
    const events: string[] = [];
    class GestureAudio extends Audio {
      override resume() {
        events.push('audio');
        return resumed.promise;
      }
    }
    replace('AudioContext', GestureAudio);
    replace('navigator', {
      mediaDevices: {
        getUserMedia: async () => {
          events.push('microphone');
          return stream();
        },
      },
    });
    const { client } = setup();
    const pending = client.connect();
    // No microtask has run: the parent can request immersive XR right here.
    expect(events).toEqual(['audio', 'microphone']);
    expect(requests).toHaveLength(0);
    await tick();
    expect(requests).toHaveLength(0);
    client.disconnect();
    await pending;
    expect(stopped).toBe(1);
    resumed.resolve();
    await tick();
    expect(requests).toHaveLength(0);
  });

  test('normal session expiry offers renewal once after cleanup; stops and failures never do', async () => {
    const callbacks = new Map<number, () => void>();
    replace('setTimeout', (callback: () => void, milliseconds: number) => {
      callbacks.set(milliseconds, callback);
      return milliseconds;
    });
    replace('clearTimeout', () => {});
    let expirations = 0;
    const expired = () => {
      expirations++;
      expect(Audio.instances.at(-1)!.closed).toBe(true);
      expect(Peer.instances.at(-1)!.closed).toBe(true);
    };
    const first = await connected(undefined, expired);
    const firstCap = callbacks.get(180_000)!;
    firstCap();
    firstCap();
    expect(expirations).toBe(1);
    expect(first.client.connected).toBe(false);
    expect(stopped).toBe(1);
    const second = await connected(undefined, expired);
    const secondCap = callbacks.get(180_000)!;
    second.client.disconnect();
    secondCap();
    expect(expirations).toBe(1);
    const third = await connected(undefined, expired);
    const thirdCap = callbacks.get(180_000)!;
    third.peer.connectionState = 'failed';
    third.peer.onconnectionstatechange?.();
    thirdCap();
    expect(expirations).toBe(1);
    expect(stopped).toBe(3);
    replace('fetch', async (url: string) =>
      String(url).startsWith('/api')
        ? Response.json({ value: 'ek_fixture' })
        : Response.json({ error: { code: 'credit_balance_exhausted' } }, { status: 429 }),
    );
    const failed = setup(undefined, expired);
    await failed.client.connect();
    expect(expirations).toBe(1);
    expect(stopped).toBe(4);
  });

  test('SDK config acknowledgement gates readiness; connection never greets during playback', async () => {
    const { client } = setup();
    client.notify({ event: 'watching' }, false);
    const promise = client.connect();
    await tick();
    const channel = Peer.instances[0].channel;
    channel.open();
    await tick();
    expect(client.connected).toBe(false);
    channel.acknowledge();
    await promise;
    const config = channel.sent.find((e) => e.type === 'session.update').session;
    expect(config.model).toBe('gpt-realtime');
    expect(config.audio.input.turn_detection).toMatchObject({
      create_response: false,
      interrupt_response: false,
    });
    expect(config.max_output_tokens).toBe(256);
    expect(config.tools.map((t: any) => t.name)).toEqual([
      'face_player',
      'show_return_target',
      'offer_replay',
    ]);
    expect(channel.responses()).toHaveLength(0);
    const state = channel.sent.find((e) => e.item?.type === 'message');
    expect(state.item.role).toBe('system');
    expect(state.item.content[0].text).toContain('watching');
    expect(captures[0]).toMatchObject({ echoCancellation: true, noiseSuppression: true });
    expect('sendText' in client).toBe(false);
  });

  test('selecting a character after microphone setup updates identity before the first answer', async () => {
    const { client, channel, identity, actions } = await connected();
    expect(channel.responses()).toHaveLength(0);
    identity.personId = 'selected-person';
    identity.personLabel = 'Selected character';
    react(client, channel);
    const messages = channel.sent.filter((event) => event.item?.type === 'message');
    const state = messages.at(-1);
    expect(state.item.role).toBe('system');
    expect(state.item.content[0].text).toContain('"personId":"selected-person"');
    expect(state.item.content[0].text).toContain('Selected character');
    expect(channel.sent.indexOf(state)).toBeLessThan(channel.sent.indexOf(channel.responses()[0]));
    channel.tool('old-character');
    identity.personId = 'another-person';
    identity.personLabel = 'Another character';
    client.notify({ event: 'selected character changed' }, false);
    await tick();
    expect(actions).toHaveLength(0);
    expect(requests.filter((url) => url.startsWith('/api'))).toHaveLength(1);
    expect(client.connected).toBe(true);
  });

  test('SDK uses only the ephemeral credential and server-selected model for provider setup', async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    replace('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url: String(url), init });
      return String(url).startsWith('/api')
        ? Response.json({ value: 'ek_fixture', model: 'configured-realtime' })
        : new Response('fixture answer');
    });
    const { channel, statuses } = await connected();
    expect(calls.map((call) => call.url)).toEqual([
      '/api/bottle-agent/session',
      'https://api.openai.com/v1/realtime/calls',
    ]);
    expect(JSON.parse(calls[0].init.body as string)).toEqual({
      sceneId: 'bottle',
      personId: 'person',
      personLabel: 'Character',
      objectId: 'bottle',
    });
    expect(new Headers(calls[0].init.headers).has('Authorization')).toBe(false);
    expect(new Headers(calls[1].init.headers).get('Authorization')).toBe('Bearer ek_fixture');
    expect(new Headers(calls[1].init.headers).get('Content-Type')).toBe('application/sdp');
    expect(calls[1].init.body).toBe('fixture offer');
    const config = channel.sent.find((event) => event.type === 'session.update').session;
    expect(config.model).toBe('configured-realtime');
    expect(
      channel.sent.some(
        (event) => event.type === 'session.update' && event.session.tracing === null,
      ),
    ).toBe(true);
    expect(config.audio.input.transcription).toBe(null);
    expect(JSON.stringify(channel.sent)).not.toContain('ek_fixture');
    expect(statuses.join(' ')).not.toContain('ek_fixture');
  });

  test('a project credential returned accidentally is rejected before any SDK provider call', async () => {
    replace('fetch', async (url: string) => {
      requests.push(String(url));
      return Response.json({ value: 'sk-test-project-secret', model: 'gpt-realtime' });
    });
    const { client, statuses } = setup();
    await client.connect();
    expect(requests).toEqual(['/api/bottle-agent/session']);
    expect(Peer.instances).toHaveLength(0);
    expect(statuses.join(' ')).not.toContain('sk-test-project-secret');
    expect(statuses.at(-1)).toBe('Invalid conversation credentials.');
    expect(stopped).toBe(1);
    expect(client.connected).toBe(false);
  });

  test('SDP credit exhaustion reports a sanitized setup failure and releases all media', async () => {
    const sensitive = 'sk-test-provider-detail';
    const cases = [
      {
        error: {
          code: 'credit_balance_exhausted',
          message: `Your organization has exhausted its credit balance. ${sensitive}`,
        },
      },
      { error: { code: 'credit_balance_exhausted', diagnostic: sensitive } },
    ];
    const logs: unknown[][] = [];
    replace('console', {
      ...console,
      error: (...args: unknown[]) => logs.push(args),
      warn: (...args: unknown[]) => logs.push(args),
    });
    for (const body of cases) {
      replace('fetch', async (url: string) => {
        requests.push(String(url));
        return String(url).startsWith('/api')
          ? Response.json({ value: 'ek_fixture', model: 'gpt-realtime' })
          : Response.json(body, { status: 429 });
      });
      const { client, statuses } = setup();
      await client.connect();
      expect(statuses.at(-1)).toBe('Server OpenAI account needs credits · microphone off');
      expect(statuses.join(' ')).not.toContain(sensitive);
      expect(statuses.join(' ')).not.toContain('ek_fixture');
      expect(statuses).not.toContain('Conversation ended · microphone off.');
      expect(client.connected).toBe(false);
      expect(Peer.instances.at(-1)!.closed).toBe(true);
      expect(Audio.instances.at(-1)!.closed).toBe(true);
    }
    expect(stopped).toBe(cases.length);
    expect(
      requests.filter((url) => url === 'https://api.openai.com/v1/realtime/calls'),
    ).toHaveLength(cases.length);
    expect(JSON.stringify(logs)).not.toContain(sensitive);
    expect(JSON.stringify(logs)).not.toContain('ek_fixture');
  });

  test('other SDP 429 failures identify account quota or rate limits without exposing details', async () => {
    replace('fetch', async (url: string) =>
      String(url).startsWith('/api')
        ? Response.json({ value: 'ek_fixture' })
        : Response.json(
            { error: { code: 'rate_limit_exceeded', message: 'sk-test-provider-detail' } },
            { status: 429 },
          ),
    );
    const { client, statuses } = setup();
    await client.connect();
    expect(statuses.at(-1)).toBe(
      'Voice provider HTTP 429: account quota or rate limit · microphone off',
    );
    expect(statuses.join(' ')).not.toContain('sk-test-provider-detail');
    expect(statuses).not.toContain('Conversation ended · microphone off.');
    expect(stopped).toBe(1);
    expect(client.connected).toBe(false);
    expect(Peer.instances[0].closed).toBe(true);
    expect(Audio.instances[0].closed).toBe(true);
  });

  test('an established connection failure still reports that the conversation ended', async () => {
    const { client, peer, statuses } = await connected();
    peer.connectionState = 'failed';
    peer.onconnectionstatechange?.();
    expect(statuses.at(-1)).toBe('Conversation ended · microphone off.');
    expect(client.connected).toBe(false);
    expect(stopped).toBe(1);
  });

  test('speech detection stays active during playback, with answers after accepted audio commit only', async () => {
    const { client, channel } = await connected();
    channel.speech();
    expect(tracks[0].enabled).toBe(true);
    expect(channel.responses()).toHaveLength(0);
    channel.commit();
    expect(channel.responses()).toHaveLength(1);
    channel.created();
    channel.done();
    client.setPlayback(true);
    client.notify({ event: 'replay' });
    expect(channel.responses()).toHaveLength(1);
  });

  test('rejected speech cannot start a reply and its committed audio leaves conversation context', async () => {
    const { channel, rejectSpeech } = await connected();
    rejectSpeech();
    channel.speech();
    channel.commit();
    expect(channel.responses()).toHaveLength(0);
    expect(channel.sent).toContainEqual({ type: 'conversation.item.delete', item_id: 'utterance' });
  });

  test('a stale committed utterance cannot consume a newer accepted speech turn', async () => {
    const { client, channel } = await connected();
    channel.speech('old');
    client.setPlayback(true);
    channel.speech('new');
    channel.commit('old');
    expect(channel.responses()).toHaveLength(0);
    channel.commit('new');
    expect(channel.responses()).toHaveLength(1);
  });

  test('physics context updates preserve accepted speech and wait for its audio commit', async () => {
    const { client, channel } = await connected();
    channel.speech();
    client.notify({ event: 'released', owner: 'free' }, false);
    client.notify({ event: 'landed', owner: 'floor' });
    client.notify({ event: 'grabbed', owner: 'visitor' });
    expect(channel.responses()).toHaveLength(0);
    const state = channel.sent.filter((event) => event.item?.type === 'message').at(-1);
    expect(state.item.role).toBe('system');
    expect(state.item.content[0].text).toContain('grabbed');
    channel.commit();
    expect(channel.responses()).toHaveLength(1);
    expect(channel.sent).not.toContainEqual({
      type: 'conversation.item.delete',
      item_id: 'utterance',
    });
  });

  test('reset and replay discard accepted speech before subsequent context updates', async () => {
    for (const playing of [false, true]) {
      const { client, channel } = await connected();
      channel.speech();
      client.notify({ event: 'released' }, false);
      client.setPlayback(true);
      client.setPlayback(playing);
      client.notify({ event: playing ? 'replayed' : 'reset' }, false);
      channel.commit();
      expect(channel.responses()).toHaveLength(0);
      expect(channel.sent).toContainEqual({
        type: 'conversation.item.delete',
        item_id: 'utterance',
      });
      client.disconnect();
    }
  });

  test('muting or ending voice prevents context updates from restoring obsolete speech', async () => {
    const { client, channel } = await connected();
    channel.speech('muted');
    client.setMuted(true);
    client.notify({ event: 'landed' }, false);
    client.setMuted(false);
    channel.commit('muted');
    expect(channel.responses()).toHaveLength(0);
    expect(channel.sent).toContainEqual({ type: 'conversation.item.delete', item_id: 'muted' });
    channel.speech('ended');
    client.disconnect();
    client.notify({ event: 'landed' }, false);
    channel.commit('ended');
    expect(channel.responses()).toHaveLength(0);
    expect(client.connected).toBe(false);
  });

  test('local reaction prior to microphone setup is answered only once connected and paused', async () => {
    const { client } = setup();
    client.setPlayback(false);
    client.notify({ event: 'approach' });
    const pending = client.connect();
    await tick();
    const channel = Peer.instances[0].channel;
    channel.open();
    channel.acknowledge();
    await pending;
    expect(channel.responses()).toHaveLength(1);
  });

  test('interruption keeps unsent replies local so replay cannot release the SDK response queue', async () => {
    const { client, channel } = await connected();
    react(client, channel);
    channel.speech('interrupt');
    channel.commit('interrupt');
    expect(channel.responses()).toHaveLength(1);
    client.setPlayback(true);
    channel.done('response', 'cancelled');
    await tick();
    expect(channel.responses()).toHaveLength(1);
    channel.speech('fresh');
    channel.commit('fresh');
    expect(channel.responses()).toHaveLength(2);
  });

  test('late response creation after replay is cancelled before tool or audio dispatch', async () => {
    const { client, channel, actions, speaking } = await connected();
    client.setPlayback(false);
    client.notify({ event: 'approach' });
    client.setPlayback(true);
    channel.created();
    channel.tool('late');
    channel.emit({
      type: 'output_audio_buffer.started',
      event_id: 'late',
      response_id: 'response',
    });
    channel.done('response', 'cancelled');
    await tick();
    expect(actions).toHaveLength(0);
    expect(speaking.at(-1)).toBe(false);
    expect(channel.responses()).toHaveLength(1);
    expect(channel.sent).toContainEqual({ type: 'response.cancel', response_id: 'response' });
  });

  test('a directed interruption can answer once the prior cancelled response finishes', async () => {
    const { client, channel } = await connected();
    react(client, channel);
    channel.speech('interrupt');
    channel.commit('interrupt');
    channel.done('response', 'cancelled');
    await tick();
    expect(channel.responses()).toHaveLength(2);
    channel.created('fresh');
    channel.done('fresh');
  });

  test('SDK tools execute once and offer_replay cannot restart source playback', async () => {
    const { client, channel, actions } = await connected();
    react(client, channel);
    channel.tool('a');
    channel.tool('a');
    channel.tool('b', 'show_return_target');
    channel.tool('c', 'offer_replay');
    await tick();
    expect(actions).toEqual(['face_player', 'show_return_target', 'offer_replay']);
    expect(channel.sent.filter((e) => e.item?.type === 'function_call_output')).toHaveLength(3);
    channel.done();
    await tick();
    expect(channel.responses()).toHaveLength(2);
  });

  test('unsupported and malformed tool requests never reach local actions', async () => {
    for (const [name, args] of [
      ['resume_recording', '{}'],
      ['face_player', '{"extra":1}'],
      ['face_player', '[]'],
      ['face_player', '{'],
    ]) {
      const { client, channel, actions } = await connected();
      react(client, channel);
      channel.tool('bad', name, args);
      await tick();
      expect(actions).toHaveLength(0);
      client.disconnect();
    }
  });

  test('replay invalidates pending tools, late outputs and late audio without losing microphone', async () => {
    const result = deferred<string>();
    const { client, channel, actions, peer, speaking } = await connected(() => result.promise);
    peer.track();
    react(client, channel);
    channel.tool('a');
    await tick();
    expect(actions).toEqual(['face_player']);
    client.setPlayback(true);
    channel.tool('late');
    result.resolve('ok');
    await tick();
    channel.emit({
      type: 'output_audio_buffer.started',
      event_id: 'audio',
      response_id: 'response',
    });
    channel.done();
    expect(actions).toEqual(['face_player']);
    expect(channel.sent.filter((e) => e.item?.type === 'function_call_output')).toHaveLength(0);
    expect(channel.responses()).toHaveLength(1);
    expect(speaking.at(-1)).toBe(false);
    expect(Audio.instances[0].nodes[2].gain.value).toBe(0);
    expect(tracks[0].enabled).toBe(true);
    expect(client.connected).toBe(true);
  });

  test('reset while paused invalidates a tool queued inside SDK asynchronous dispatch', async () => {
    const { client, channel, actions } = await connected();
    react(client, channel);
    channel.tool('a');
    client.notify({ event: 'reset' }, false);
    await tick();
    expect(actions).toHaveLength(0);
    expect(channel.sent.filter((e) => e.item?.type === 'function_call_output')).toHaveLength(0);
  });

  test('tool followup waits for response completion and asynchronous action result', async () => {
    const result = deferred<string>();
    const { client, channel } = await connected(() => result.promise);
    react(client, channel);
    channel.tool('a');
    await tick();
    channel.done();
    expect(channel.responses()).toHaveLength(1);
    result.resolve('ok');
    await tick();
    expect(channel.responses()).toHaveLength(2);
    expect(channel.sent.at(-2).item.type).toBe('function_call_output');
  });

  test('tool loops stop after three followups', async () => {
    const { client, channel, statuses } = await connected();
    react(client, channel);
    for (let i = 0; i < 4; i++) {
      if (i) channel.created(`r${i}`);
      channel.tool(String(i), 'face_player', '{}', i ? `r${i}` : 'response');
      await tick();
      channel.done(i ? `r${i}` : 'response');
      await tick();
    }
    expect(channel.responses()).toHaveLength(4);
    expect(statuses.at(-1)).toContain('action limit');
  });

  test('spatial output opens only for an authorized response and updates source/listener pose', async () => {
    const { client, channel, peer, speaking } = await connected();
    peer.track();
    react(client, channel);
    channel.emit({
      type: 'output_audio_buffer.started',
      event_id: 'audio',
      response_id: 'response',
    });
    const audio = Audio.instances[0];
    expect(audio.nodes[2].gain.value).toBe(1);
    expect(speaking.at(-1)).toBe(true);
    client.updateAudio(new Vector3(2, 4, 6), new Vector3(1, 2, 3), new Quaternion(), 2);
    expect(audio.nodes[1].positionX.value).toBe(1);
    expect(audio.listener.positionY.value).toBe(1);
    expect(audio.listener.forwardZ.value).toBe(-1);
    client.disconnect();
    expect(audio.closed).toBe(true);
    expect(audio.nodes.every((n) => n.disconnected)).toBe(true);
  });

  test('unsolicited responses are cancelled and cannot run tools', async () => {
    const { client, channel, actions } = await connected();
    client.setPlayback(false);
    channel.emit({
      type: 'response.created',
      event_id: 'unexpected',
      response: { id: 'rogue', status: 'in_progress', output: [] },
    });
    channel.tool('a', 'face_player', '{}', 'rogue');
    await tick();
    expect(actions).toHaveLength(0);
    expect(channel.sent).toContainEqual({ type: 'response.cancel', response_id: 'rogue' });
  });

  test('mute choice survives replay and microphone reconnection', async () => {
    const { client, channel } = await connected();
    client.setMuted(true);
    client.setPlayback(true);
    channel.speech();
    channel.commit();
    expect(tracks[0].enabled).toBe(false);
    expect(channel.responses()).toHaveLength(0);
    const pending = client.connect();
    await tick();
    const next = Peer.instances.at(-1)!.channel;
    next.open();
    next.acknowledge();
    await pending;
    expect(tracks.at(-1)!.enabled).toBe(false);
    client.setMuted(false);
    expect(tracks.at(-1)!.enabled).toBe(true);
  });

  test('disconnect settles pending permission and stops tracks acquired afterward', async () => {
    const permission = deferred<MediaStream>();
    replace('navigator', { mediaDevices: { getUserMedia: () => permission.promise } });
    const { client } = setup();
    const pending = client.connect();
    await tick();
    client.disconnect();
    await pending;
    permission.resolve(stream());
    await tick();
    expect(stopped).toBe(1);
    expect(requests).toHaveLength(0);
    expect(Audio.instances[0].closed).toBe(true);
  });

  test('overlapping connects close the older SDK connection and discard its tools', async () => {
    const { client, channel, peer, actions } = await connected();
    react(client, channel);
    const pending = client.connect();
    channel.tool('late');
    await tick();
    const next = Peer.instances.at(-1)!.channel;
    next.open();
    next.acknowledge();
    await pending;
    expect(peer.closed).toBe(true);
    expect(actions).toHaveLength(0);
    expect(stopped).toBe(1);
  });

  test('disconnect during SDK SDP negotiation prevents late readiness', async () => {
    const answer = deferred<Response>();
    replace('fetch', async (url: string) =>
      String(url).startsWith('/api') ? Response.json({ value: 'ek_fixture' }) : answer.promise,
    );
    const { client } = setup();
    const pending = client.connect();
    await tick();
    const peer = Peer.instances[0];
    client.disconnect();
    await pending;
    answer.resolve(new Response('late answer'));
    await tick();
    peer.channel.open();
    expect(client.connected).toBe(false);
    expect(peer.closed).toBe(true);
    expect(stopped).toBe(1);
  });

  test('three-minute session cap and response timeout release capture', async () => {
    const callbacks = new Map<number, () => void>();
    replace('setTimeout', (callback: () => void, milliseconds: number) => {
      callbacks.set(milliseconds, callback);
      return milliseconds;
    });
    replace('clearTimeout', () => {});
    const { client, channel, statuses } = await connected();
    react(client, channel);
    callbacks.get(45_000)!();
    expect(client.connected).toBe(false);
    expect(stopped).toBe(1);
    expect(statuses.at(-1)).toContain('stopped responding');
    const next = await connected();
    callbacks.get(180_000)!();
    expect(next.client.connected).toBe(false);
    expect(stopped).toBe(2);
    expect(next.statuses.at(-1)).toContain('Three-minute');
  });

  test('permission denial has a microphone retry cue and releases audio', async () => {
    replace('navigator', {
      mediaDevices: {
        getUserMedia: async () => {
          throw new DOMException('denied', 'NotAllowedError');
        },
      },
    });
    const { client, statuses } = setup();
    await client.connect();
    expect(statuses.at(-1)).toContain('permission was declined');
    expect(statuses.at(-1)).not.toContain('text');
    expect(Audio.instances[0].closed).toBe(true);
    expect(client.connected).toBe(false);
  });
});
