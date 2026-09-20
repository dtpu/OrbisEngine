import { expect, test } from 'bun:test';
import { ReplayButton } from '../src/xr/replay-button';

function source(handedness: XRHandedness = 'left', mapping: GamepadMappingType = 'xr-standard') {
  return {
    handedness,
    gamepad: {
      mapping,
      buttons: Array.from({ length: 6 }, () => ({ pressed: false, touched: false, value: 0 })),
    } as unknown as Gamepad,
  };
}
function press(input: ReturnType<typeof source>, index: number, down: boolean) {
  Object.assign(input.gamepad.buttons[index], { pressed: down, value: down ? 1 : 0 });
}

test('left X replays once per press and never repeats while held', () => {
  const button = new ReplayButton();
  const left = source();
  expect(button.update([left])).toBe(false);
  press(left, 4, true);
  expect(button.update([left])).toBe(true);
  for (let i = 0; i < 100; i++) expect(button.update([left])).toBe(false);
  press(left, 4, false);
  expect(button.update([left])).toBe(false);
  press(left, 4, true);
  expect(button.update([left])).toBe(true);
});

test('entry and reconnect require releasing an already held X button', () => {
  const button = new ReplayButton();
  const left = source();
  press(left, 4, true);
  expect(button.update([left])).toBe(false);
  press(left, 4, false);
  button.update([left]);
  button.reset();
  press(left, 4, true);
  expect(button.update([left])).toBe(false);
  button.update([]);
  expect(button.update([left])).toBe(false);
  press(left, 4, false);
  button.update([left]);
  press(left, 4, true);
  expect(button.update([left])).toBe(true);
});

test('right A, left grip/trigger/Y, unmapped pads, and missing buttons cannot replay', () => {
  for (const input of [source('right'), source('left', '')]) {
    const button = new ReplayButton();
    button.update([input]);
    press(input, 4, true);
    expect(button.update([input])).toBe(false);
  }
  const button = new ReplayButton();
  const left = source();
  button.update([left]);
  for (const index of [0, 1, 2, 3, 5]) press(left, index, true);
  expect(button.update([left])).toBe(false);
  expect(button.update([{ handedness: 'left', gamepad: undefined }])).toBe(false);
});
