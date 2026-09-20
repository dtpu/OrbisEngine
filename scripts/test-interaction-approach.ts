import { describe, expect, test } from 'bun:test';
import { ApproachDetector } from '../src/interaction/approach';

const visitor = (x: number, z = 0) => [x, 1, z] as [number, number, number];
const candidate = (id: string, x: number, z = 0, eligible = true) => ({
  id,
  position: [x, 1, z] as [number, number, number],
  eligible,
});
const detector = () =>
  new ApproachDetector({
    enterDistance: 1,
    exitDistance: 1.5,
    dwellSeconds: 0.4,
    minApproachDistance: 0.2,
    facingCos: 0.5,
  });
const update = (
  subject: ApproachDetector,
  position: [number, number, number],
  candidates: ReturnType<typeof candidate>[],
  dt = 0.1,
  forward: [number, number, number] = [-1, 0, 0],
) => subject.update({ position, forward, candidates }, dt);

describe('proximity approach detection', () => {
  test('fires one deliberate close approach after dwell and visitor movement', () => {
    const subject = detector();
    expect(update(subject, visitor(3), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.9), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.7), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.65), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.6), [candidate('alex', 0)])).toBe('alex');
  });

  test('initial or reset-inside positions do not fire until departure beyond exit and a new approach', () => {
    const subject = detector();
    for (let i = 0; i < 8; i++)
      expect(update(subject, visitor(0.5), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(2), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.8), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.6), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.5), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(0.45), [candidate('alex', 0)])).toBe('alex');

    subject.reset(visitor(0.5));
    for (let i = 0; i < 8; i++)
      expect(update(subject, visitor(0.5), [candidate('alex', 0)])).toBeNull();
  });

  test('a moving actor cannot trigger against a stationary visitor', () => {
    const subject = detector();
    expect(update(subject, visitor(0), [candidate('alex', 3)])).toBeNull();
    expect(update(subject, visitor(0), [candidate('alex', 0.9)])).toBeNull();
    for (let i = 0; i < 8; i++)
      expect(update(subject, visitor(0), [candidate('alex', 0.5)])).toBeNull();
  });

  test('does not trigger while passing outside the zone or facing away', () => {
    const subject = detector();
    expect(update(subject, visitor(3), [candidate('alex', 0)])).toBeNull();
    for (let i = 0; i < 8; i++)
      expect(update(subject, visitor(1.1), [candidate('alex', 0)])).toBeNull();
    expect(update(subject, visitor(3), [candidate('alex', 0)])).toBeNull();
    for (let i = 0; i < 8; i++) {
      expect(update(subject, visitor(0.6), [candidate('alex', 0)], 0.1, [1, 0, 0])).toBeNull();
    }
  });

  test('requires exit before rearming and retains the first dwell target during jitter', () => {
    const subject = detector();
    expect(
      update(subject, visitor(3), [candidate('bravo', 0), candidate('alpha', -0.1)]),
    ).toBeNull();
    expect(
      update(subject, visitor(0.9), [candidate('bravo', 0), candidate('alpha', -0.1)]),
    ).toBeNull();
    expect(
      update(subject, visitor(0.7), [candidate('bravo', 0), candidate('alpha', -0.1)]),
    ).toBeNull();
    // Alpha becomes closer after the detector has selected bravo; it must not steal this dwell.
    expect(
      update(subject, visitor(0.62), [candidate('bravo', 0), candidate('alpha', 0.55)]),
    ).toBeNull();
    expect(update(subject, visitor(0.55), [candidate('bravo', 0), candidate('alpha', 0.54)])).toBe(
      'bravo',
    );
    for (let i = 0; i < 8; i++) {
      expect(
        update(subject, visitor(0.52), [candidate('bravo', 0), candidate('alpha', 0.54)]),
      ).toBeNull();
    }
  });

  test('uses deterministic nearest-then-id target choice and explicit eligibility', () => {
    const subject = detector();
    expect(
      update(subject, visitor(3), [
        candidate('zeta', 0),
        candidate('alpha', 0),
        candidate('hidden', 0, 0, false),
      ]),
    ).toBeNull();
    expect(
      update(subject, visitor(0.9), [
        candidate('zeta', 0),
        candidate('alpha', 0),
        candidate('hidden', 0, 0, false),
      ]),
    ).toBeNull();
    expect(
      update(subject, visitor(0.7), [
        candidate('zeta', 0),
        candidate('alpha', 0),
        candidate('hidden', 0, 0, false),
      ]),
    ).toBeNull();
    expect(
      update(subject, visitor(0.6), [
        candidate('zeta', 0),
        candidate('alpha', 0),
        candidate('hidden', 0, 0, false),
      ]),
    ).toBeNull();
    expect(
      update(subject, visitor(0.5), [
        candidate('zeta', 0),
        candidate('alpha', 0),
        candidate('hidden', 0, 0, false),
      ]),
    ).toBe('alpha');
  });
});
