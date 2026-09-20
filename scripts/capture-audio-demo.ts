// Verify the integrated viewer; screenshots/report stay outside Git. No asset mutations.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium, type Browser } from 'playwright-core';
import type { FourDAudio } from '../src/audio/fourd-audio.ts';
import type {} from './viewer-types.ts';
declare global {
  interface Window {
    __audioMeters: { context: AudioContext; analyser: AnalyserNode }[];
  }
}
type AudioSample = { rms: number; drift?: number; sources?: number };
type CapturedState = {
  demo: string | null;
  t: number;
  duration: number;
  playing: boolean;
  mediaPaused: boolean;
  mediaTime: number;
  audio: FourDAudio['state'];
  button?: string | null;
  credit?: string;
};
type ClipResult = {
  clip: string;
  checks: { name: string; pass: boolean | null; detail: unknown }[];
  errors: string[];
  assets: { url: string; status: number; source?: string; snapshot?: string }[];
  samples: AudioSample[];
  final?: CapturedState;
  failure?: string;
};
const base = process.env.AUDIO_URL || 'http://127.0.0.1:5399';
const out = process.env.AUDIO_OUT || path.resolve('.context/evidence/audio');
const clips = (process.env.AUDIO_CLIPS || 'elevator,lobby,stairs2,atrium,tos31,hpwide').split(',');
await mkdir(out, { recursive: true });
const report: { url: string; clips: ClipResult[]; limitations: string[] } = {
  url: base,
  clips: [],
  limitations: [
    'Headless browser validates graph signal and transport; no listening or Quest perceptual verdict.',
  ],
};
let browser: Browser | undefined;
try {
  for (const clip of clips) {
    const result: ClipResult = { clip, checks: [], errors: [], assets: [], samples: [] };
    report.clips.push(result);
    browser = await chromium.launch({
      channel: 'chrome',
      headless: true,
      args: ['--use-angle=metal', '--ignore-gpu-blocklist'],
    });
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.on('pageerror', (error) => result.errors.push(error.message));
    page.on('response', (response) => {
      if (/\/audio\.json|\/audio\/.*\.(wav|mp3)/.test(response.url()))
        result.assets.push({
          url: response.url(),
          status: response.status(),
          source: response.headers()['x-wander-asset-source'],
          snapshot: response.headers()['x-wander-snapshot'],
        });
    });
    await page.addInitScript(() => {
      const Native = window.AudioContext;
      window.__audioMeters = [];
      window.AudioContext = class extends Native {
        private __meterAttached = false;
        createGain() {
          const gain = super.createGain();
          if (!this.__meterAttached) {
            this.__meterAttached = true;
            const analyser = this.createAnalyser();
            analyser.fftSize = 2048;
            gain.connect(analyser);
            window.__audioMeters.push({ context: this, analyser });
          }
          return gain;
        }
      };
    });
    const check = (name: string, ok: unknown, detail: unknown) => {
      result.checks.push({ name, pass: !!ok, detail });
      if (!ok) console.log(`${clip}: FAIL ${name}`, detail);
    };
    try {
      const direct = clip === 'hpwide';
      await page.goto(`${base}/${direct ? 'fourd.html?demo=' : 'demo.html?clip='}${clip}`, {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
      });
      const frame = direct
        ? page.mainFrame()
        : await page
            .locator('#frame')
            .elementHandle()
            .then((el) => el.contentFrame());
      assert.ok(frame, 'Viewer frame must exist');
      await frame.waitForFunction(
        () => window.wander?.ready && !window.wander.audioState.loading,
        null,
        { timeout: 150000 },
      );
      const state = () =>
        frame.evaluate(() => {
          const w = window.wander;
          return {
            demo: w.demo,
            t: w.t,
            duration: w.dur,
            playing: w.playing,
            mediaPaused: w.video.paused,
            mediaTime: w.video.currentTime,
            audio: w.audioState,
            button: document.querySelector('#audio-controls button')?.textContent,
            credit: document.querySelector<HTMLElement>('#audio-credit')?.innerText,
          };
        });
      check('actual requested preset', (await state()).demo === clip, await state());
      if (direct) {
        result.final = await state();
        check(
          'silent clip is explicitly unavailable',
          result.final.audio.hasAudio === false &&
            result.final.button === 'No soundtrack available',
          result.final,
        );
        check(
          'no spatial claim without stems',
          !result.final.audio.spatialReady && result.final.audio.mode === 'original',
          result.final.audio,
        );
        await page.screenshot({ path: path.join(out, `${clip}-unavailable.png`) });
        continue;
      }
      await frame.locator('#audio-controls button').first().click();
      await frame.waitForFunction(
        () => window.wander.audioState.context === 'running' && !window.wander.audioState.muted,
        null,
        { timeout: 10000 },
      );
      await frame.evaluate(() => window.wander.play(true));
      await page.waitForTimeout(350);
      check(
        'sound unlocked',
        (await state()).audio.unlocked && !(await state()).audio.muted,
        await state(),
      );
      for (let i = 0; i < 10; i++) {
        await page.waitForTimeout(130);
        result.samples.push(
          await frame.evaluate(() => {
            const a = window.__audioMeters[0]?.analyser;
            if (!a) return { rms: 0 };
            const data = new Float32Array(a.fftSize);
            a.getFloatTimeDomainData(data);
            return {
              rms: Math.sqrt(data.reduce((sum, v) => sum + v * v, 0) / data.length),
              drift: window.wander.audioState.driftSeconds,
              sources: window.wander.audioState.activeSources,
            };
          }),
        );
      }
      check(
        'original route has signal',
        result.samples.some((s) => s.rms > 0.00001),
        result.samples,
      );
      check(
        'exactly one external source',
        result.samples.every((s) => s.sources === 1),
        result.samples,
      );
      // Remove button focus so Enter reaches the viewer's transport, not button activation.
      await frame.evaluate(() => {
        if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
        window.focus();
      });
      await page.keyboard.press('Enter');
      await page.waitForTimeout(130);
      const paused = await state();
      await page.waitForTimeout(250);
      const still = await state();
      check(
        'Enter pauses once',
        !paused.playing &&
          paused.mediaPaused &&
          paused.audio.activeSources === 0 &&
          Math.abs(still.t - paused.t) < 0.01,
        { paused, still },
      );
      await page.keyboard.press('Enter');
      await page.waitForTimeout(250);
      check(
        'Enter resumes once',
        (await state()).playing && (await state()).audio.activeSources === 1,
        await state(),
      );
      await frame.evaluate(() => {
        window.wander.play(false);
        window.wander.setTime(1);
      });
      await page.waitForTimeout(200);
      check(
        'paused seek is silent',
        !(await state()).playing &&
          (await state()).audio.activeSources === 0 &&
          Math.abs((await state()).t - 1) < 0.02,
        await state(),
      );
      await frame.evaluate(() => {
        window.wander.setTime(window.wander.dur - 0.2);
        window.wander.play(true);
      });
      await page.waitForTimeout(800);
      const looped = await state();
      check('loop restarts one source', looped.t < 2 && looped.audio.activeSources === 1, looped);
      await frame.locator('#audio-controls button').first().click();
      await page.waitForTimeout(100);
      check(
        'mute stops external sources',
        (await state()).audio.muted && (await state()).audio.activeSources === 0,
        await state(),
      );
      await frame.evaluate(() => {
        if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
        window.focus();
      });
      await page.keyboard.press('Enter');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(150);
      check('Enter preserves user mute', (await state()).audio.muted, await state());
      await frame.locator('#audio-controls button').first().click();
      await page.waitForTimeout(150);
      if (clip === 'tos31') {
        check(
          'visible CC source attribution',
          /Blender Foundation.*CC BY 3.0/.test((await state()).credit ?? ''),
          (await state()).credit,
        );
        const credit = await frame.locator('#audio-credit').boundingBox(),
          bar = await page.locator('#bar').boundingBox();
        check('credit clears wrapper controls', credit && bar && credit.y + credit.height < bar.y, {
          credit,
          bar,
        });
      }
      await page.screenshot({ path: path.join(out, `${clip}-sound.png`) });
      if (clip === 'elevator') {
        // Test the wrapper's trusted event forwarding explicitly.
        await page.evaluate(() => {
          if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
          window.focus();
        });
        await page.keyboard.press('Enter');
        await page.waitForTimeout(200);
        check('parent Enter forwards once', (await state()).playing, await state());
      }
      check(
        'audio assets served by S3 snapshot',
        result.assets.some(
          (a) => /audio\/original.wav/.test(a.url) && a.status === 200 && a.source === 's3',
        ),
        result.assets,
      );
      result.final = await state();
    } catch (error) {
      result.failure = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.log(`${clip}: ERROR ${result.failure}`);
    } finally {
      await page.close();
      await browser.close();
      await writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
    }
    console.log(
      `${clip}: ${result.checks.filter((c) => c.pass).length}/${result.checks.length} checks, ${result.errors.length} page errors`,
    );
  }
} finally {
  await browser?.close();
  await writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
}
assert.ok(
  report.clips.every(
    (c) => !c.failure && !c.errors.length && c.checks.every((k) => k.pass !== false),
  ),
  'One or more live audio checks failed; inspect report.json',
);
