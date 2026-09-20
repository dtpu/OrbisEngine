import type { Quaternion, Vector3 } from 'three';
import { BottleAgentClient, type AgentIdentity } from './bottle-agent-client';
import { CHARACTER_VOICES, type CharacterVoice } from './character-voice';

type Options = {
  characters: AgentIdentity[];
  identity: () => AgentIdentity;
  status: (message: string) => void;
  speaking: (value: boolean) => void;
  speechStarted: (voice: CharacterVoice) => boolean;
  action: (name: string, args: unknown) => string | Promise<string>;
  expired?: () => void;
};

/** Realtime voices cannot change after audio starts. Keep at most two warm connections
 * so addressing another character preserves the visitor's first spoken turn. */
export class CharacterVoiceCast {
  private readonly clients = new Map<CharacterVoice, BottleAgentClient>();
  private readonly statuses = new Map<CharacterVoice, string>();
  private playing = true;
  private routedVoice: CharacterVoice | null = null;
  private generation = 0;
  private failureCode: string | null = null;

  constructor(private readonly options: Options) {
    for (const voice of CHARACTER_VOICES) {
      const character = options.characters.find((person) => (person.voice ?? 'ash') === voice);
      if (!character) continue;
      this.clients.set(
        voice,
        new BottleAgentClient({
          identity: () => {
            const selected = options.identity();
            return (selected.voice ?? 'ash') === voice ? selected : character;
          },
          canRespond: () => this.selectedVoice === voice,
          status: (message) => {
            this.statuses.set(voice, message);
            if (this.selectedVoice === voice) options.status(message);
          },
          speaking: (value) => {
            if (this.selectedVoice === voice) options.speaking(value);
          },
          action: (name, args) =>
            this.selectedVoice === voice
              ? options.action(name, args)
              : 'Action cancelled: another character is selected.',
          failed: (message) => {
            this.failureCode = this.clients.get(voice)?.lastProviderErrorCode ?? null;
            // A failed pair must not leave a second microphone running behind “Mic off”.
            this.disconnect();
            options.status(message);
          },
          speechStarted: () => {
            // The scene checks this voice before changing the addressed character.
            if (!options.speechStarted(voice)) return false;
            this.route();
            return this.selectedVoice === voice;
          },
          expired: () => {
            // Renew the pair together; disconnect cancels the other expiration timer.
            this.disconnect();
            options.expired?.();
          },
        }),
      );
    }
    if (!this.clients.size) throw new Error('A voice cast needs at least one character.');
  }

  private get selectedVoice(): CharacterVoice {
    return this.options.identity().voice ?? 'ash';
  }

  get activeClient(): BottleAgentClient {
    return this.clients.get(this.selectedVoice)!;
  }

  get connected() {
    return this.activeClient.connected;
  }

  get lastProviderErrorCode() {
    return this.failureCode ?? this.activeClient.lastProviderErrorCode;
  }

  private route() {
    const selected = this.selectedVoice;
    if (this.routedVoice !== selected) {
      this.options.speaking(false);
      this.options.status(this.statuses.get(selected) ?? 'Mic off');
      this.routedVoice = selected;
    }
    for (const [voice, client] of this.clients) {
      client.setPlayback(this.playing || voice !== selected);
    }
  }

  async connect() {
    const generation = ++this.generation;
    this.failureCode = null;
    this.route();
    // Start every permission request synchronously within the Enter VR gesture.
    await Promise.all(
      [...this.clients.values()].map((client) =>
        generation === this.generation ? client.connect() : Promise.resolve(),
      ),
    );
  }

  notify(state: unknown, react = true) {
    this.route();
    this.activeClient.notify(state, react);
  }

  setPlayback(playing: boolean) {
    this.playing = playing;
    this.route();
  }

  setMuted(muted: boolean) {
    for (const client of this.clients.values()) client.setMuted(muted);
  }

  updateAudio(source: Vector3, listener: Vector3, rotation: Quaternion, stature: number) {
    this.activeClient.updateAudio(source, listener, rotation, stature);
  }

  disconnect() {
    this.generation++;
    for (const client of this.clients.values()) client.disconnect();
    this.options.speaking(false);
    this.statuses.clear();
    this.routedVoice = null;
  }
}
