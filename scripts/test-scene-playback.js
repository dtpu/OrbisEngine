import { expect, test } from 'bun:test';
import { resolveSceneConfig } from '../src/scene-config.ts';
import { cameraReelMode, playbackState, togglePlayback } from '../src/scene-playback.js';

test('a scene manifest without a tour keeps clip playback at the current camera', async () => {
  const query = await resolveSceneConfig(new URLSearchParams('scene=/scenes/custom.json'), [], {
    abort: new AbortController(),
    fetch: async () =>
      Response.json({
        schema: 'wander.viewer-scene/1',
        params: { world: '/room.ply', cam: '0,0,0,0,0,-1', walk: '1' },
      }),
  });
  const calls = [];
  const api = {
    cameraReel: cameraReelMode(query) !== 'stationary',
    playing: false,
    time: 31.7,
    play(on) {
      this.playing = on;
      calls.push(['play', on]);
    },
    showreel() {
      throw new Error('An unconfigured camera tour must not start');
    },
  };
  expect(playbackState(api).label).toBe('Play clip');
  togglePlayback(api);
  expect(playbackState(api)).toEqual({ camera: false, running: true, label: 'Pause clip' });
  togglePlayback(api);
  expect(api.playing).toBe(false);
  expect(api.time).toBe(31.7);
  expect(calls).toEqual([
    ['play', true],
    ['play', false],
  ]);
});

test('authored keys and legacy relative paths retain camera playback', () => {
  expect(cameraReelMode(new URLSearchParams('reelkeys=0:0,0,0;1:1,0,0'))).toBe('keys');
  expect(cameraReelMode(new URLSearchParams('reel=A=0,0,0;B=1,0,0'))).toBe('behind');
  expect(cameraReelMode(new URLSearchParams('reelmode=behind'))).toBe('behind');
  expect(cameraReelMode(new URLSearchParams('reelmode=orbit'))).toBe('orbit');
  expect(cameraReelMode(new URLSearchParams('reelmode=stationary&reelkeys=0:0,0,0;1:1,0,0'))).toBe(
    'stationary',
  );
  expect(cameraReelMode(new URLSearchParams('world=/custom.spz'))).toBe('stationary');
  const calls = [];
  const api = {
    cameraReel: true,
    playing: true,
    reel: null,
    showreel(on) {
      this.reel = on ? {} : null;
      calls.push(on);
    },
    play() {
      throw new Error('The camera-tour control must retain its existing behavior');
    },
  };
  expect(playbackState(api).label).toBe('Play camera move');
  togglePlayback(api);
  expect(playbackState(api).running).toBe(true);
  togglePlayback(api);
  expect(calls).toEqual([true, false]);
});

test('button state follows keyboard playback and switching between camera and clip modes', () => {
  expect(playbackState({ cameraReel: false, playing: true }).label).toBe('Pause clip');
  expect(playbackState({ cameraReel: false, playing: false }).label).toBe('Play clip');
  expect(playbackState({ cameraReel: true, playing: true, reel: null }).label).toBe(
    'Play camera move',
  );
  expect(playbackState({ cameraReel: true, reel: {} }).label).toBe(
    'Playing — press any key to stop',
  );
});

test.each([false, true])('pausing preserves mute with cameraReel=%s', (cameraReel) => {
  let unlocks = 0;
  const api = {
    cameraReel,
    playing: true,
    reel: cameraReel ? {} : null,
    muted: true,
    unlockAudio() {
      unlocks++;
      this.muted = false;
    },
    play(on) {
      this.playing = on;
    },
    showreel(on) {
      this.reel = on ? {} : null;
    },
  };
  togglePlayback(api);
  expect(playbackState(api).running).toBe(false);
  expect(unlocks).toBe(0);
  expect(api.muted).toBe(true);
  togglePlayback(api);
  expect(playbackState(api).running).toBe(true);
  expect(unlocks).toBe(1);
});
