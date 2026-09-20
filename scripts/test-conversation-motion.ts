import { describe, expect, test } from 'bun:test';
import {
  ConversationMotion,
  type ConversationMode,
  type ConversationPose,
} from '../src/interaction/conversation-motion';

const arms = ['leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow'] as const;
const channels = ['pitch', 'yaw', 'roll', 'breath', 'weight', ...arms] as const;
const zero: ConversationPose = {
  pitch: 0,
  yaw: 0,
  roll: 0,
  breath: 0,
  weight: 0,
  leftShoulder: 0,
  rightShoulder: 0,
  leftElbow: 0,
  rightElbow: 0,
  mode: 'off',
};

function advance(motion: ConversationMotion, seconds: number, mode: ConversationMode, fps = 60) {
  for (let i = 0; i < Math.round(seconds * fps); i++) motion.update(1 / fps, mode);
  return motion.snapshot();
}

describe('invented conversation overlay', () => {
  test('off is silent and settles exactly onto the recorded pose', () => {
    const motion = new ConversationMotion('person');
    expect(advance(motion, 10, 'off')).toEqual(zero);
    advance(motion, 2, 'speaking');
    const fading = motion.update(1 / 60, 'off');
    expect(fading.weight).toBeGreaterThan(0);
    expect(advance(motion, 1, 'off')).toEqual(zero);
    expect(advance(motion, 10, 'off')).toEqual(zero);
  });

  test('all channels remain finite and anatomically modest over sustained mode changes', () => {
    const motion = new ConversationMotion('long-running-person');
    const limits: Record<(typeof channels)[number], number> = {
      pitch: 0.06,
      yaw: 0.045,
      roll: 0.025,
      breath: 0.0025,
      weight: 1,
      leftShoulder: 0.16,
      rightShoulder: 0.16,
      leftElbow: 0.38,
      rightElbow: 0.38,
    };
    for (let frame = 0; frame < 36000; frame++) {
      const mode = (['off', 'listening', 'speaking'] as const)[Math.floor(frame / 173) % 3];
      const pose = motion.update(1 / 60, mode);
      for (const channel of channels) {
        expect(Number.isFinite(pose[channel])).toBe(true);
        expect(Math.abs(pose[channel])).toBeLessThanOrEqual(limits[channel]);
      }
      expect(pose.weight).toBeGreaterThanOrEqual(0);
      for (const arm of arms) expect(pose[arm]).toBeGreaterThanOrEqual(0);
    }
  });

  test('mode changes preserve the current pose and move smoothly including interrupted fades', () => {
    const motion = new ConversationMotion('smooth');
    let previous = advance(motion, 2, 'listening');
    for (const mode of ['speaking', 'off', 'listening', 'speaking', 'off'] as const) {
      const switched = motion.update(0, mode);
      for (const channel of channels) expect(switched[channel]).toBeCloseTo(previous[channel], 12);
      for (let frame = 0; frame < 13; frame++) {
        const pose = motion.update(1 / 60, mode);
        for (const channel of ['pitch', 'yaw', 'roll'] as const) {
          expect(Math.abs(pose[channel] - previous[channel])).toBeLessThan(0.006);
        }
        expect(Math.abs(pose.breath - previous.breath)).toBeLessThan(0.0002);
        for (const arm of arms) {
          expect(Math.abs(pose[arm] - previous[arm])).toBeLessThan(0.04);
        }
        previous = pose;
      }
    }
  });

  test('30, 60 and 90 Hz agree at equal times through fades and changing cadence', () => {
    const results = [30, 60, 90].map((fps) => {
      const motion = new ConversationMotion('same-person');
      return (['listening', 'speaking', 'off', 'speaking'] as const).map((mode) =>
        advance(motion, mode === 'off' ? 2.1 : 0.3, mode, fps),
      );
    });
    for (let sample = 0; sample < results[0].length; sample++) {
      for (const channel of channels) {
        expect(results[1][sample][channel]).toBeCloseTo(results[0][sample][channel], 10);
        expect(results[2][sample][channel]).toBeCloseTo(results[0][sample][channel], 10);
      }
    }
  });

  test('ids are deterministic but people do not move in unison', () => {
    const a = advance(new ConversationMotion('person'), 4, 'speaking');
    expect(advance(new ConversationMotion('person'), 4, 'speaking')).toEqual(a);
    const b = advance(new ConversationMotion('person_01'), 4, 'speaking');
    expect(Math.abs(a.pitch - b.pitch) + Math.abs(a.yaw - b.yaw)).toBeGreaterThan(0.005);
  });

  test('speech is more active than listening, and both breathe', () => {
    function energy(mode: ConversationMode) {
      const motion = new ConversationMotion('energy');
      advance(motion, 1, mode);
      let head = 0;
      let chest = 0;
      for (let i = 0; i < 1200; i++) {
        const pose = motion.update(1 / 60, mode);
        head += pose.pitch ** 2 + pose.yaw ** 2 + pose.roll ** 2;
        chest += pose.breath ** 2;
      }
      return { head, chest };
    }
    const listening = energy('listening');
    const speaking = energy('speaking');
    expect(speaking.head).toBeGreaterThan(listening.head * 2);
    expect(listening.chest).toBeGreaterThan(0.001);
    expect(speaking.chest).toBeGreaterThan(0.001);
  });

  test('speaking alternates visible arm beats with pauses while listening rests', () => {
    const motion = new ConversationMotion('gestures');
    advance(motion, 1, 'speaking');
    let leftBeats = 0;
    let rightBeats = 0;
    let restFrames = 0;
    let previousSide = '';
    let switches = 0;
    let previous = motion.snapshot();
    for (let frame = 0; frame < 1200; frame++) {
      const pose = motion.update(1 / 60, 'speaking');
      for (const arm of arms) {
        expect(Math.abs(pose[arm] - previous[arm])).toBeLessThan(0.025);
      }
      const left = pose.leftShoulder > 0.1 && pose.leftElbow > 0.24;
      const right = pose.rightShoulder > 0.1 && pose.rightElbow > 0.24;
      expect(left && right).toBe(false);
      if (left) leftBeats++;
      if (right) rightBeats++;
      const side = left ? 'left' : right ? 'right' : '';
      if (side && side !== previousSide) {
        switches++;
        previousSide = side;
      }
      if (arms.every((arm) => pose[arm] === 0)) restFrames++;
      previous = pose;
    }
    expect(leftBeats).toBeGreaterThan(100);
    expect(rightBeats).toBeGreaterThan(100);
    expect(switches).toBeGreaterThan(10);
    expect(restFrames).toBeGreaterThan(80);

    const switched = motion.update(0, 'listening');
    for (const arm of arms) expect(switched[arm]).toBe(previous[arm]);
    advance(motion, 0.5, 'listening');
    for (let frame = 0; frame < 600; frame++) {
      const pose = motion.update(1 / 60, 'listening');
      for (const arm of arms) expect(pose[arm]).toBe(0);
    }
  });

  test('arm gestures share the half-second activation and exact fade-out', () => {
    const motion = new ConversationMotion('arm-fade');
    const halfway = advance(motion, 0.25, 'speaking');
    expect(halfway.weight).toBeCloseTo(0.5, 12);
    for (const arm of arms) {
      expect(halfway[arm]).toBeLessThanOrEqual(arm.endsWith('Shoulder') ? 0.08 : 0.19);
    }
    expect(advance(motion, 0.25, 'speaking').weight).toBeCloseTo(1, 12);
    expect(advance(motion, 0.25, 'off').weight).toBeCloseTo(0.5, 12);
    expect(advance(motion, 0.3, 'off')).toEqual(zero);
  });

  test('invalid and negative deltas freeze time; a stalled frame advances at most 0.1 seconds', () => {
    const motion = new ConversationMotion('guard');
    const before = advance(motion, 2, 'speaking');
    for (const dt of [NaN, Infinity, -Infinity, -1, 0]) {
      expect(motion.update(dt, 'speaking')).toEqual(before);
    }
    const reference = new ConversationMotion('guard');
    advance(reference, 2, 'speaking');
    expect(motion.update(999, 'speaking')).toEqual(reference.update(0.1, 'speaking'));
  });

  test('reset is exact and reproducible; callers cannot mutate internal state', () => {
    const motion = new ConversationMotion('reset');
    const first = advance(motion, 2, 'speaking');
    const snapshot = motion.snapshot();
    snapshot.pitch = 999;
    expect(motion.snapshot()).toEqual(first);
    motion.reset();
    expect(motion.snapshot()).toEqual(zero);
    expect(advance(motion, 2, 'speaking')).toEqual(first);
    const returned: ConversationPose = motion.update(1 / 60, 'listening');
    returned.weight = 999;
    expect(motion.snapshot().weight).toBeLessThanOrEqual(1);
  });
});
