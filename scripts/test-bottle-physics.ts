import { describe, expect, test } from 'bun:test';
import { BottlePhysics, type Vec3 } from '../src/interaction/bottle-physics';

function bottle(overrides: Partial<ConstructorParameters<typeof BottlePhysics>[0]> = {}) {
  const result = new BottlePhysics({
    radius: 0.05,
    gravity: 9.8,
    maxSpeed: 30,
    floorAt: () => 0,
    blockedAt: () => false,
    ...overrides,
  });
  result.reset([0, 1, 0]);
  return result;
}
function toss(result: BottlePhysics, velocity: Vec3) {
  expect(result.grab('left', result.snapshot().position, 0)).toBe(true);
  expect(result.release('left', velocity)).toBe(true);
}

describe('bottle interaction simulation', () => {
  test('recorded poses remain authoritative until grabbed; holding has one owner and no gravity', () => {
    const b = bottle();
    b.step(1);
    expect(b.snapshot().position).toEqual([0, 1, 0]);
    b.setRecordedPosition([1, 1, 0]);
    expect(b.grab('left', [0, 1, 0], 0.1)).toBe(false);
    expect(b.grab('left', [1, 1, 0], 0.1)).toBe(true);
    expect(b.grab('right', [1, 1, 0], 0.1)).toBe(false);
    b.moveHand('right', [4, 1, 0], 0.1);
    b.setRecordedPosition([5, 1, 0]);
    expect(b.release('right')).toBe(false);
    expect(b.step(1, { position: [1, 1, 0], radius: 1 })).toEqual([]);
    expect(b.snapshot()).toEqual({
      mode: 'held',
      holder: 'left',
      position: [1, 1, 0],
      velocity: [0, 0, 0],
    });
  });

  test('release carries measured hand velocity and limits extreme throws', () => {
    const b = bottle({ gravity: 0 });
    b.grab('left', [0, 1, 0], 0);
    b.moveHand('left', [0.1, 1, 0], 0.05);
    b.release('left');
    b.step(0.1);
    expect(b.snapshot().position[0]).toBeCloseTo(0.3);
    expect(b.snapshot().velocity).toEqual([2, 0, 0]);
    b.grab('left', b.snapshot().position, 0);
    b.release('left', [Number.MAX_VALUE, Number.MAX_VALUE, 0]);
    expect(Math.hypot(...b.snapshot().velocity)).toBeCloseTo(30);
  });

  test('a drop settles on measured support and reports once; unknown support stays unknown', () => {
    const b = bottle();
    toss(b, [1, 0, 0]);
    const events = [];
    for (let i = 0; i < 600; i++) events.push(...b.step(1 / 60));
    expect(events.map((event) => event.type)).toEqual(['dropped']);
    expect(b.snapshot().mode).toBe('resting');
    expect(b.snapshot().position[1]).toBeCloseTo(0.05);
    expect(b.grab('left', [10, 1, 0], 0.1)).toBe(false);
    expect(b.grab('left', b.snapshot().position, 0)).toBe(true);
    const falling = bottle({ floorAt: () => null });
    toss(falling, [0, 0, 0]);
    for (let i = 0; i < 120; i++) falling.step(1 / 60);
    expect(falling.snapshot().position[1]).toBeLessThan(-5);
    expect(falling.snapshot().mode).toBe('free');
  });

  test('a fast drop honors the visible support extent without widening wall collision', () => {
    const contacts: number[] = [];
    const b = bottle({
      floorRadius: 0.11,
      bounceSpeed: 100,
      settleSpeed: 1,
      blockedAt: (_position, radius) => {
        contacts.push(radius);
        return false;
      },
    });
    expect(b.startFlight([0, 1, 0], [0, -30, 0])).toBe(true);
    const events = b.step(0.1);
    expect(events).toEqual([{ type: 'dropped', position: [0, 0.11, 0] }]);
    expect(b.snapshot()).toMatchObject({
      mode: 'resting',
      position: [0, 0.11, 0],
      velocity: [0, 0, 0],
    });
    expect(contacts).toContain(0.05);

    const unsupported = bottle({ floorRadius: 0.11, floorAt: () => null });
    expect(unsupported.startFlight([0, 1, 0], [0, -30, 0])).toBe(true);
    unsupported.step(0.1);
    expect(unsupported.snapshot().position[1]).toBeLessThan(0);
    expect(unsupported.snapshot().mode).toBe('free');
  });

  test('lifted landing support keeps lower recorded poses reachable without allowing pickup through real floor', () => {
    const b = bottle({ floorLift: 0.5 });
    b.reset([0, 0.2, 0]);
    expect(b.grab('left', [0, 0.2, 0], 0)).toBe(true);
    b.release('left', [0, 0, 0]);
    for (let i = 0; i < 600 && b.snapshot().mode !== 'resting'; i++) b.step(1 / 60);
    expect(b.snapshot().mode).toBe('resting');
    expect(b.snapshot().position[1]).toBeCloseTo(0.55);
    expect(b.grab('left', b.snapshot().position, 0)).toBe(true);
    b.reset([0, 0.2, 0]);
    expect(b.grab('left', [0, -0.1, 0], 0.5)).toBe(false);
    const unknown = bottle({ floorLift: 0.5, floorAt: () => null });
    unknown.startFlight([0, 0.2, 0], [0, -30, 0]);
    unknown.step(0.1);
    expect(unknown.snapshot().position[1]).toBeLessThan(0);
    expect(unknown.snapshot().mode).toBe('free');
    expect(() => bottle({ floorLift: -1 })).toThrow();
  });

  test('a downward occupied floor bin settles without making walls or ceilings into support', () => {
    const occupiedFloor = bottle({
      blockedAt: ([, y]) => y <= 0.15,
    });
    expect(occupiedFloor.startFlight([0, 1, 0], [0, -30, 0])).toBe(true);
    const events = [];
    for (let i = 0; i < 720; i++) {
      events.push(...occupiedFloor.step(1 / 72));
      if (occupiedFloor.snapshot().mode === 'resting') break;
    }
    expect(events.map((event) => event.type)).toEqual(['dropped']);
    expect(occupiedFloor.snapshot().mode).toBe('resting');
    expect(occupiedFloor.snapshot().position[1]).toBeGreaterThanOrEqual(0.15);

    const wall = bottle({
      gravity: 0,
      floorAt: () => null,
      blockedAt: ([x]) => x >= 0.2,
    });
    expect(wall.startFlight([0, 1, 0], [10, -1, 0])).toBe(true);
    wall.step(0.1);
    expect(wall.snapshot().mode).toBe('free');
    expect(wall.snapshot().position[0]).toBeLessThan(0.2);

    const floorWall = bottle({
      gravity: 0,
      blockedAt: ([x]) => x >= 0.2,
    });
    expect(floorWall.startFlight([0, 0.11, 0], [10, -1, 0])).toBe(true);
    floorWall.step(0.1);
    expect(floorWall.snapshot().mode).toBe('free');
    expect(floorWall.snapshot().position[0]).toBeLessThan(0.2);

    const ceiling = bottle({
      gravity: 0,
      blockedAt: ([, y]) => y >= 1.1,
    });
    expect(ceiling.startFlight([0, 1, 0], [0, 10, 0])).toBe(true);
    ceiling.step(0.1);
    expect(ceiling.snapshot().mode).toBe('free');
    expect(ceiling.snapshot().velocity[1]).toBeLessThan(0);
  });

  test('radius collision prevents a fast throw crossing a thin occupied wall', () => {
    // Sphere against an occupied cell spanning x=[0.5,0.52], y=[0,2], z=[-1,1].
    const b = bottle({
      gravity: 0,
      blockedAt: ([x, y, z], radius) => {
        const dx = Math.max(0.5 - x, 0, x - 0.52);
        const dy = Math.max(-y, 0, y - 2);
        const dz = Math.max(-1 - z, 0, z - 1);
        return Math.hypot(dx, dy, dz) < radius;
      },
    });
    toss(b, [30, 0, 0]);
    expect(b.step(0.1, { position: [0.6, 1, 0], radius: 0.25 })).toEqual([]);
    expect(b.snapshot().position[0]).toBeLessThan(0.45);
    expect(b.snapshot().velocity[0]).toBeLessThan(0);
    expect(b.snapshot().mode).toBe('free');
  });

  test('swept free throws catch a small target, lock its pose and report once', () => {
    const b = bottle({ gravity: 0 });
    toss(b, [30, 0, 0]);
    const target = { position: [0.61, 1, 0] as Vec3, radius: 0.005 };
    expect(b.step(0.1, target)).toEqual([{ type: 'returned', position: target.position }]);
    target.position[0] = 4;
    b.setRecordedPosition([2, 2, 2]);
    expect(b.step(1, target)).toEqual([]);
    expect(b.snapshot().position).toEqual([0.61, 1, 0]);
    expect(b.grab('left', [5, 1, 0], 0.1)).toBe(false);
    expect(b.grab('left', b.snapshot().position, 0)).toBe(true);
  });

  test('invalid input is rejected, snapshots are copies and reset clears motion and ownership', () => {
    const b = bottle();
    b.reset([NaN, 0, 0]);
    b.setRecordedPosition([Infinity, 0, 0]);
    expect(b.grab('left', [NaN, 1, 0], 1)).toBe(false);
    expect(b.grab('left', [0, 1, 0], Infinity)).toBe(false);
    b.grab('left', [0, 1, 0], 0);
    b.moveHand('left', [NaN, 1, 0], 1);
    b.moveHand('left', [1, 1, 0], 0);
    expect(b.release('left', [NaN, 0, 0])).toBe(false);
    expect(b.snapshot().mode).toBe('held');
    b.release('left');
    expect(b.step(NaN)).toEqual([]);
    expect(b.step(Infinity)).toEqual([]);
    expect(b.step(-1)).toEqual([]);
    const snapshot = b.snapshot();
    snapshot.position[0] = 999;
    snapshot.velocity[0] = 999;
    expect(b.snapshot().position).toEqual([0, 1, 0]);
    expect(b.snapshot().velocity).toEqual([0, 0, 0]);
    b.reset([2, 3, 4]);
    expect(b.snapshot()).toEqual({
      mode: 'recorded',
      position: [2, 3, 4],
      velocity: [0, 0, 0],
      holder: null,
    });
    expect(() => bottle({ radius: 0 })).toThrow();
    expect(() => bottle({ floorRadius: 0 })).toThrow();
    expect(() => bottle({ gravity: NaN })).toThrow();
  });

  test('excessive frame deltas are bounded and free grabs still require reach', () => {
    const b = bottle({ gravity: 0 });
    toss(b, [1, 0, 0]);
    expect(b.grab('right', [4, 1, 0], 0.1)).toBe(false);
    b.step(100);
    expect(b.snapshot().position[0]).toBeCloseTo(0.1);
    expect(b.grab('right', b.snapshot().position, 0)).toBe(true);
  });

  test('recorded flight handoff preserves its sampled velocity and never replaces a held bottle', () => {
    const b = bottle({ gravity: 0 });
    expect(b.startFlight([0, 1, 0], [4, 0, 0])).toBe(true);
    expect(b.snapshot()).toEqual({
      mode: 'free',
      position: [0, 1, 0],
      velocity: [4, 0, 0],
      holder: null,
    });
    b.step(0.1);
    expect(b.snapshot().position[0]).toBeCloseTo(0.4);
    expect(b.grab('left', b.snapshot().position, 0)).toBe(true);
    expect(b.startFlight([2, 1, 0], [1, 0, 0])).toBe(false);
    expect(b.release('right')).toBe(false);
    expect(b.release('left')).toBe(true);
  });

  test('attached recorded and returned positions stay synchronized without an artificial swept grab', () => {
    const b = bottle({ gravity: 0 });
    expect(b.setAttachedPosition([1, 1, 0])).toBe(true);
    expect(b.snapshot().position).toEqual([1, 1, 0]);
    expect(b.startFlight([0, 1, 0], [30, 0, 0])).toBe(true);
    expect(b.step(0.1, { position: [0.61, 1, 0], radius: 0.005 })).toEqual([
      { type: 'returned', position: [0.61, 1, 0] },
    ]);
    expect(b.setAttachedPosition([2, 1, 0])).toBe(true);
    expect(b.canGrabSegment([0.61, 1, 0], 0)).toBe(false);
    expect(b.canGrabSegment([2, 1, 0], 0)).toBe(true);
    expect(b.grab('left', [2, 1, 0], 0)).toBe(true);
    expect(b.setAttachedPosition([3, 1, 0])).toBe(false);
    expect(b.release('left')).toBe(true);
    expect(b.setAttachedPosition([3, 1, 0])).toBe(false);
  });

  test('swept hand and bottle paths permit a fast legitimate catch but reject blocked or below-floor grabs', () => {
    const fast = bottle({ gravity: 0 });
    expect(fast.startFlight([0, 1, 0], [30, 0, 0])).toBe(true);
    fast.step(0.1);
    expect(fast.canGrabSegment([1, 1, 0], 0.01)).toBe(true);
    expect(fast.grab('right', [1, 1, 0], 0.01, [1, 1, 0])).toBe(true);

    const fastHand = bottle({ gravity: 0 });
    fastHand.setRecordedPosition([1, 1, 0]);
    expect(fastHand.grab('left', [2, 1, 0], 0.01, [0, 1, 0])).toBe(true);

    const blocked = bottle({
      blockedAt: ([x, y], radius) => Math.abs(x - 0.5) < radius && Math.abs(y - 1) < radius,
    });
    blocked.setRecordedPosition([0, 1, 0]);
    expect(blocked.grab('left', [1, 1, 0], 2)).toBe(false);

    const floorBlocked = bottle();
    floorBlocked.setRecordedPosition([0, 0.05, 0]);
    expect(floorBlocked.grab('left', [0, -0.1, 0], 1)).toBe(false);
  });
});
