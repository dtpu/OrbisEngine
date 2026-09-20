import { describe, expect, test } from 'bun:test';
import { personVisibility } from '../src/person-visibility';

describe('person visibility on the source timeline', () => {
  test('retained poses after an occlusion appear immediately at their source times', () => {
    // Three PLYs at samples 1, 4 and 5. Their compact indices are 0, 1 and 2.
    const visible = personVisibility(
      [
        [1, 1],
        [4, 5],
      ],
      [0, 0.09, 0.18, 0.29, 0.41, 0.5],
      0.6,
    );
    expect(visible(0)).toBe(false);
    expect(visible(0.09)).toBe(true);
    expect(visible(0.179)).toBe(true);
    expect(visible(0.18)).toBe(false);
    expect(visible(0.4)).toBe(false);
    expect(visible(0.41)).toBe(true);
    expect(visible(0.59)).toBe(true);
    expect(visible(0.6)).toBe(false);
  });

  test('two fragments of one subject hand over without overlap or a missing interval', () => {
    const timeline = [0, 0.08, 0.17, 0.25];
    const early = personVisibility([[0, 1]], timeline, 0.34);
    const late = personVisibility([[2, 3]], timeline, 0.34);
    for (const time of [0, 0.08, 0.169, 0.17, 0.25, 0.339]) {
      expect(Number(early(time)) + Number(late(time))).toBe(1);
    }
    expect(early(0.34) || late(0.34)).toBe(false);
  });

  test('single sequences retain legacy visibility; malformed shared runs fail', () => {
    expect(personVisibility(undefined, undefined, 0)(10)).toBe(true);
    expect(personVisibility([], [0, 1], 2)(0)).toBe(false);
    for (const runs of [
      [[0, 2]],
      [[1, 0]],
      [[-1, 0]],
      [[0.5, 1]],
      [
        [0, 1],
        [1, 1],
      ],
    ]) {
      expect(() => personVisibility(runs, [0, 1], 2)).toThrow();
    }
    expect(() => personVisibility([[0, 0]], [0, 0], 2)).toThrow();
    expect(() => personVisibility([[0, 0]], undefined, 2)).toThrow();
  });
});
