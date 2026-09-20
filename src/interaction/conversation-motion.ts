export type ConversationMode = 'off' | 'listening' | 'speaking';

export interface ConversationPose {
  /** Additive head angles in radians, relative to the paused recorded pose. */
  pitch: number;
  yaw: number;
  roll: number;
  /** Invented forward shoulder swing and elbow flexion, in radians. */
  leftShoulder: number;
  rightShoulder: number;
  leftElbow: number;
  rightElbow: number;
  /** Additive chest displacement in body-heights. */
  breath: number;
  weight: number;
  mode: ConversationMode;
}

const FADE_SECONDS = 0.5;

function zeroPose(): ConversationPose {
  return {
    pitch: 0,
    yaw: 0,
    roll: 0,
    leftShoulder: 0,
    rightShoulder: 0,
    leftElbow: 0,
    rightElbow: 0,
    breath: 0,
    weight: 0,
    mode: 'off',
  };
}

/** Invented conversational motion, not recovered performance, a skeleton or lip sync.
 * Apply each output to the original paused pose; never add it to last frame's result.
 */
export class ConversationMotion {
  private readonly phase: number;
  private time = 0;
  private mode: ConversationMode = 'off';
  private fadeTime = FADE_SECONDS;
  private listening = 0;
  private speaking = 0;
  private startListening = 0;
  private startSpeaking = 0;
  private pose = zeroPose();

  constructor(personId: string) {
    let hash = 2166136261;
    for (let i = 0; i < personId.length; i++) {
      hash = Math.imul(hash ^ personId.charCodeAt(i), 16777619);
    }
    this.phase = ((hash >>> 0) / 4294967296) * Math.PI * 2;
  }

  update(deltaSeconds: number, mode: ConversationMode): ConversationPose {
    const dt = Number.isFinite(deltaSeconds) ? Math.min(0.1, Math.max(0, deltaSeconds)) : 0;
    if (mode !== this.mode) {
      this.startListening = this.listening;
      this.startSpeaking = this.speaking;
      this.fadeTime = 0;
      this.mode = mode;
    }
    this.fadeTime = Math.min(FADE_SECONDS, this.fadeTime + dt);
    const progress = this.fadeTime / FADE_SECONDS;
    const blend = progress * progress * (3 - 2 * progress);
    this.listening = this.startListening * (1 - blend) + (mode === 'listening' ? blend : 0);
    this.speaking = this.startSpeaking * (1 - blend) + (mode === 'speaking' ? blend : 0);
    const weight = this.listening + this.speaking;
    this.time += dt;
    if (weight === 0) {
      this.pose = { ...zeroPose(), mode };
      return this.snapshot();
    }

    const t = this.time;
    const p = this.phase;
    // A slow envelope separates the listening nods. Speech adds a second cadence,
    // without using audio words or claiming to reconstruct the recorded speaker.
    const nod = (0.5 + 0.5 * Math.sin(t * 0.9 + p)) ** 2;
    // Alternate explanatory beats, with a short rest between arms. Squaring the
    // positive lobe gives each beat a smooth start and finish; listening rests.
    const beat = Math.sin(t * 2.4 + p);
    const leftGesture =
      this.speaking *
      Math.max(0, (beat - 0.18) / 0.82) ** 2 *
      (0.8 + 0.2 * Math.sin(t * 0.7 + p) ** 2);
    const rightGesture =
      this.speaking *
      Math.max(0, (-beat - 0.18) / 0.82) ** 2 *
      (0.8 + 0.2 * Math.sin(t * 0.7 + p + 1.2) ** 2);
    this.pose = {
      pitch:
        this.listening * 0.022 * nod * Math.sin(t * 3.1 + p) +
        this.speaking * (0.036 * Math.sin(t * 3.8 + p) + 0.014 * Math.sin(t * 6.1 + p * 2)),
      yaw:
        this.listening * 0.012 * Math.sin(t * 0.8 + p * 2) +
        this.speaking * 0.035 * Math.sin(t * 1.3 + p * 2),
      roll:
        this.listening * 0.006 * Math.sin(t * 1.1 + p * 3) +
        this.speaking * 0.018 * Math.sin(t * 1.7 + p * 3),
      breath: weight * 0.002 * Math.sin(t * 1.5 + p),
      leftShoulder: 0.16 * leftGesture,
      rightShoulder: 0.16 * rightGesture,
      leftElbow: 0.38 * leftGesture,
      rightElbow: 0.38 * rightGesture,
      weight,
      mode,
    };
    return this.snapshot();
  }

  reset(): void {
    this.time = 0;
    this.mode = 'off';
    this.fadeTime = FADE_SECONDS;
    this.listening = 0;
    this.speaking = 0;
    this.startListening = 0;
    this.startSpeaking = 0;
    this.pose = zeroPose();
  }

  snapshot(): ConversationPose {
    return { ...this.pose };
  }
}
