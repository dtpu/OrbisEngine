// Native Web Audio rendering of owned in-memory tones; no server or media files required.
import assert from 'node:assert/strict';
import { build } from 'vite';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import type * as THREE from 'three';
import type { FourDAudio } from '../src/audio/fourd-audio.ts';
declare global {
  interface Window {
    WanderAudio: typeof import('../src/audio/fourd-audio.ts');
  }
}
type AudioPerson = ConstructorParameters<typeof FourDAudio>[2][number];
await build({
  configFile: false,
  logLevel: 'error',
  build: {
    outDir: '.context/audio-browser',
    emptyOutDir: true,
    lib: {
      entry: 'src/audio/fourd-audio.ts',
      formats: ['iife'],
      name: 'WanderAudio',
      fileName: () => 'audio.js',
    },
  },
});
const code = await readFile('.context/audio-browser/audio.js', 'utf8');
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.setContent('<body></body>');
  await page.addScriptTag({ content: code });
  const results = await page.evaluate(async () => {
    // One second, mono PCM sine generated solely for this test.
    const rate = 48000,
      n = rate,
      wav = new ArrayBuffer(44 + n * 2),
      view = new DataView(wav);
    const str = (at: number, s: string) =>
      [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
    str(0, 'RIFF');
    view.setUint32(4, 36 + n * 2, true);
    str(8, 'WAVE');
    str(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    str(36, 'data');
    view.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++)
      view.setInt16(44 + i * 2, Math.sin((i / rate) * Math.PI * 2 * 440) * 12000, true);
    const m = {
      schema: 'wander.audio/1',
      source: { hasAudio: false },
      timeline: { durationSeconds: 1 },
      defaultMode: 'spatial',
      spatialMixComplete: true,
      tracks: [
        {
          id: 'voice',
          kind: 'dialogue',
          url: 'tone.wav',
          personId: 'person',
          reviewed: true,
          provenance: 'owned-synthetic-tone',
          anchor: { url: 'head.json', space: 'person-local' },
        },
      ],
    };
    Reflect.set(
      window,
      'fetch',
      async (url: RequestInfo | URL) =>
        new Response(
          String(url).endsWith('.wav')
            ? wav.slice(0)
            : JSON.stringify(
                String(url).endsWith('audio.json')
                  ? m
                  : {
                      positions: [
                        [-1, 0, 0],
                        [-1, 0, 0],
                      ],
                      times: [0, 1],
                    },
              ),
        ),
    );
    const render = async (yaw: number) => {
      let context!: OfflineAudioContext;
      // Render the production graph deterministically, replacing only media input and hardware clock.
      Reflect.set(
        window,
        'AudioContext',
        class extends OfflineAudioContext {
          constructor() {
            super(2, rate, rate);
            context = this;
          }
          get state(): AudioContextState {
            return 'running';
          }
          resume() {
            return Promise.resolve();
          }
          close() {
            return Promise.resolve();
          }
          createMediaElementSource() {
            return this.createGain();
          }
        },
      );
      const video = document.createElement('video');
      Object.defineProperties(video, { paused: { value: false }, readyState: { value: 4 } });
      const person = {
        id: 'person',
        bodyH: 1,
        ts: [0, 1],
        group: {
          updateWorldMatrix() {},
          matrixWorld: { elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
        },
      };
      const audio = new window.WanderAudio.FourDAudio(
        video,
        1,
        [person as unknown as AudioPerson],
        () => 1,
      );
      await audio.load('http://fixture/audio.json');
      await audio.unlock();
      audio.tick(
        0,
        true,
        { x: 0, y: 0, z: 0 } as THREE.Vector3,
        { x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) } as THREE.Quaternion,
      );
      const buffer = await context.startRendering();
      const energy = [0, 1].map((channel) =>
        buffer
          .getChannelData(channel)
          .slice(4800)
          .reduce((sum, x) => sum + x * x, 0),
      );
      audio.dispose();
      return energy;
    };
    return { facingForward: await render(0), facingBack: await render(Math.PI) };
  });
  assert.ok(
    results.facingForward[0] > results.facingForward[1] * 1.2,
    'left-positioned voice is stronger in left ear',
  );
  assert.ok(
    results.facingBack[1] > results.facingBack[0] * 1.2,
    'head turn reverses ear dominance',
  );
  console.log(JSON.stringify({ passed: true, ...results }));
} finally {
  await browser.close();
}
