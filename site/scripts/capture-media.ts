// Renders the landing page media that only the real viewer can produce, into public/media/:
// the hero loop, the "same second, new angle" renders, three viewer-feature loops, and the
// pipeline-strip stills. Everything comes from the viewer at VIEWER_URL (the repo root's
// `bun run demo`, S3-backed) driven through its `window.wander` diagnostics; no scene asset is
// modified. Frames are read straight from the WebGL canvas, so software GL (a headless server
// without a GPU) works, only slowly. Run after `bun run scripts/pull-media.ts`.
//
//   bun run scripts/capture-media.ts [--only hero,compare,viewer,steps] [--force]
//
// Camera offsets are reported in body-heights: the viewer's mean character stature for the
// scene, which the walk mode also uses as the visitor's height. The numbers printed at the end
// are what the compare captions quote.
import { mkdir, writeFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { chromium, type Browser, type Frame, type Page } from 'playwright-core';

const root = resolve(import.meta.dirname, '..');
const outRoot = join(root, 'public', 'media');
const workRoot = join(root, 'node_modules', '.cache', 'capture-media');
const base = process.env.VIEWER_URL || 'http://127.0.0.1:5399';
const ffmpeg = process.env.FFMPEG || 'ffmpeg';
// Without a GPU the full 800k-splat budget renders at ~16 s a frame; LOD_COUNT=150000 renders
// the same views at page size with no visible loss and about three times faster.
const lodCount = process.env.LOD_COUNT;
const argv = process.argv.slice(2);
const force = argv.includes('--force');
const onlyArg = argv[argv.indexOf('--only') + 1];
const only = argv.includes('--only') && onlyArg ? onlyArg.split(',') : null;

function chromePath(): string {
  const candidates = [
    process.env.CHROME_PATH,
    '/home/ubuntu/.local/bin/google-chrome',
    '/usr/bin/google-chrome',
    '/opt/pw-browsers/chromium',
  ].filter((p): p is string => Boolean(p));
  for (const p of candidates) if (existsSync(p)) return p;
  return chromium.executablePath();
}

type Vec = [number, number, number];
type Pose = { t: number; pos?: Vec; look?: Vec };

// The subset of the viewer's `window.wander` diagnostics this script drives; the full shape is
// documented in the root `scripts/viewer-types.ts`.
type V3 = {
  x: number;
  y: number;
  z: number;
  set(x: number, y: number, z: number): V3;
  clone(): V3;
  add(v: V3): V3;
  normalize(): V3;
  multiplyScalar(s: number): V3;
  toArray(): number[];
};
type WanderDiagnostics = {
  ready: boolean;
  dur: number;
  upm: number;
  camera: {
    position: V3;
    getWorldDirection(v: V3): V3;
    lookAt(x: number, y: number, z: number): void;
  };
  THREE: { Vector3: new (x?: number, y?: number, z?: number) => V3 };
  spark: { renderer: { domElement: HTMLCanvasElement } };
  clampBox: { min: V3; max: V3 };
  walk: { stature: number; speed: number; advance(from: V3, delta: V3, dt: number): V3 } | null;
  play(value: boolean): void;
  setTime(value: number): void;
};
declare global {
  interface Window {
    wander: WanderDiagnostics;
  }
}
type SceneInfo = {
  upm: number;
  stature: number;
  home: Vec;
  dir: Vec;
  homeLook: Vec;
  box: [Vec, Vec];
  dur: number;
};

// Hide every overlay the viewer draws over its canvas: HUD, help, hints, the source video inset,
// the edge fade, and the audio credit. The demo shell hides the same set.
const HIDE_CSS = `
  #hud, #help, #hint, #pip, #edgefade, #audio-controls, #audio-credit, #walk-map, #xrBtn,
  #wander-audio, .wander-overlay { display: none !important; }`;

async function openScene(browser: Browser, demo: string, walk: boolean, size: [number, number]) {
  const page = await browser.newPage({ viewport: { width: size[0], height: size[1] } });
  const params = new URLSearchParams({ demo, pip: '0', dpr: '1' });
  if (walk) params.set('walk', '1');
  if (lodCount) params.set('lodcount', lodCount);
  await page.goto(`${base}/fourd.html?${params}`, { waitUntil: 'domcontentloaded' });
  await page.addStyleTag({ content: HIDE_CSS });
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 600000 });
  const frame = page.mainFrame();
  await frame.evaluate(() => {
    window.wander.play(false);
    window.wander.setTime(0);
  });
  // The world keeps streaming in after "ready"; wait until successive frames stop changing size.
  let last = 0;
  for (let i = 0; i < 40; i++) {
    const bytes = (await grabJpeg(frame)).length;
    if (last && Math.abs(bytes - last) / last < 0.02 && bytes > 40000) break;
    last = bytes;
    await page.waitForTimeout(3000);
  }
  const info = await frame.evaluate((walk) => {
    const w = window.wander;
    const dir = new w.THREE.Vector3();
    w.camera.getWorldDirection(dir);
    const look = w.camera.position.clone().add(dir.multiplyScalar(3));
    return {
      upm: w.upm,
      stature: walk && w.walk ? w.walk.stature : 1.7 * w.upm,
      home: w.camera.position.toArray() as Vec,
      dir: dir.clone().normalize().toArray() as Vec,
      homeLook: look.toArray() as Vec,
      box: [w.clampBox.min.toArray(), w.clampBox.max.toArray()] as [Vec, Vec],
      dur: w.dur,
    };
  }, walk);
  console.log(
    `${demo}${walk ? ' (walk)' : ''}: 1 body-height = ${info.stature.toFixed(3)} u, home ${fmt(info.home)}, box ${fmt(info.box[0])}..${fmt(info.box[1])}`,
  );
  return { page, frame, info };
}

const fmt = (v: Vec) => `(${v.map((n) => n.toFixed(2)).join(', ')})`;

async function grabJpeg(frame: Frame, quality = 0.92) {
  const url = await frame.evaluate(
    (quality) =>
      new Promise<string>((res) =>
        requestAnimationFrame(() =>
          res(window.wander.spark.renderer.domElement.toDataURL('image/jpeg', quality)),
        ),
      ),
    quality,
  );
  return Buffer.from(url.split(',')[1], 'base64');
}

async function render(frame: Frame, pose: Pose) {
  await frame.evaluate((pose) => {
    const w = window.wander;
    w.setTime(pose.t);
    if (pose.pos) w.camera.position.set(pose.pos[0], pose.pos[1], pose.pos[2]);
    if (pose.look) w.camera.lookAt(pose.look[0], pose.look[1], pose.look[2]);
  }, pose);
  // The splat sort runs off the main thread and lags a camera jump by several frames. The
  // renderer is deterministic, so a frame is final once two grabs in a row are identical.
  let image = await grabJpeg(frame);
  for (let i = 0; i < 12; i++) {
    const next = await grabJpeg(frame);
    if (next.equals(image)) break;
    image = next;
  }
  return image;
}

function run(args: string[]) {
  const r = spawnSync(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', ...args], {
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  if (r.error) throw new Error(`${ffmpeg} is not available: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`${ffmpeg} exited with ${r.status}`);
}

async function still(frame: Frame, pose: Pose, out: string) {
  const dest = join(outRoot, out);
  await mkdir(join(dest, '..'), { recursive: true });
  await writeFile(dest, await render(frame, pose));
  console.log(`made ${out}`);
}

async function sequence(
  frame: Frame,
  out: string,
  fps: number,
  seconds: number,
  poseAt: (s: number) => Pose | Promise<Pose>,
) {
  const n = Math.round(fps * seconds);
  const work = join(workRoot, out.replace(/[/.]/g, '_'));
  await rm(work, { recursive: true, force: true });
  await mkdir(work, { recursive: true });
  const started = Date.now();
  for (let i = 0; i < n; i++) {
    const pose = await poseAt(i / fps);
    await writeFile(join(work, `f_${String(i).padStart(4, '0')}.jpg`), await render(frame, pose));
    if (i % 10 === 9)
      console.log(
        `  ${out}: ${i + 1}/${n} frames, ${((Date.now() - started) / 1000).toFixed(0)} s`,
      );
  }
  const dest = join(outRoot, out);
  await mkdir(join(dest, '..'), { recursive: true });
  run([
    '-framerate',
    String(fps),
    '-i',
    join(work, 'f_%04d.jpg'),
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    '-crf',
    '20',
    '-preset',
    'medium',
    '-movflags',
    '+faststart',
    dest,
  ]);
  console.log(`made ${out}`);
}

const clampTo = (p: Vec, box: [Vec, Vec], margin = 0.05): Vec =>
  p.map((v, i) => Math.min(box[1][i] - margin, Math.max(box[0][i] + margin, v))) as Vec;

// A point in body-heights relative to the home pose: right of, in front of, and above it, along
// the direction the home pose faces.
function relative(info: SceneInfo, right: number, forward: number, up = 0): Vec {
  const s = info.stature;
  const [dx, , dz] = info.dir;
  const len = Math.hypot(dx, dz) || 1;
  const fx = dx / len;
  const fz = dz / len;
  return [
    info.home[0] + (-fz * right + fx * forward) * s,
    info.home[1] + up * s,
    info.home[2] + (fx * right + fz * forward) * s,
  ];
}
const offset = (info: SceneInfo, right: number, forward: number, up = 0) =>
  clampTo(relative(info, right, forward, up), info.box);
const describe = (info: SceneInfo, pos: Vec) => {
  const s = info.stature;
  const [ddx, , ddz] = info.dir;
  const len = Math.hypot(ddx, ddz) || 1;
  const fx = ddx / len;
  const fz = ddz / len;
  const ox = pos[0] - info.home[0];
  const oz = pos[2] - info.home[2];
  const dx = (-fz * ox + fx * oz) / s;
  const dz = (fx * ox + fz * oz) / s;
  const dy = (pos[1] - info.home[1]) / s;
  return `${dx.toFixed(2)} right, ${dz.toFixed(2)} forward, ${dy.toFixed(2)} up (body-heights)`;
};

const want = (group: string, out: string) =>
  (!only || only.includes(group)) && (force || !existsSync(join(outRoot, out)));

const measurements: string[] = [];
const browser = await chromium.launch({
  executablePath: chromePath(),
  headless: true,
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
    '--autoplay-policy=no-user-gesture-required',
    '--mute-audio',
  ],
});
try {
  // Hero: the elevator clip while the viewpoint walks off the recorded path and back.
  if (want('hero', 'hero/wander.mp4')) {
    const { page, frame, info } = await openScene(browser, 'elevator', false, [1280, 720]);
    const look = info.homeLook;
    await sequence(frame, 'hero/wander.mp4', 10, info.dur, (s) => {
      const k = s / info.dur;
      const swing = Math.sin(Math.PI * k);
      return { t: s, pos: offset(info, 0.7 * swing, 0.9 * swing, 0.1 * swing), look };
    });
    await page.close();
  }

  // Compare: one instant per scene from a standing position the recording never had.
  // Standing position (right, forward) and the point looked at (lookRight, lookForward), in
  // body-heights from the walk start pose, which is where the recording begins.
  const compare: {
    id: string;
    demo: string;
    t: number;
    right: number;
    forward: number;
    lookRight: number;
    lookForward: number;
  }[] = [
    {
      id: 'elevator',
      demo: 'elevator',
      t: 6.0,
      right: 1.0,
      forward: 0.6,
      lookRight: 0,
      lookForward: 3,
    },
    {
      id: 'lobby',
      demo: 'lobby',
      t: 4.733,
      right: 0.8,
      forward: -1.0,
      lookRight: 0,
      lookForward: 0.6,
    },
    {
      id: 'gym',
      demo: 'gym-accepted',
      t: 8.0,
      right: 1.0,
      forward: 0.4,
      lookRight: 0,
      lookForward: 3,
    },
  ];
  for (const c of compare) {
    if (!want('compare', `${c.id}/render.jpg`)) continue;
    const { page, frame, info } = await openScene(browser, c.demo, true, [1280, 720]);
    const pos = offset(info, c.right, c.forward);
    const look = relative(info, c.lookRight, c.lookForward);
    await still(frame, { t: c.t, pos, look }, `${c.id}/render.jpg`);
    const actual = (await frame.evaluate(() => window.wander.camera.position.toArray())) as Vec;
    measurements.push(`${c.id} t=${c.t}s: ${describe(info, actual)}`);
    console.log(`  ${measurements[measurements.length - 1]}`);
    await page.close();
  }

  // Viewer features, 4:3 loops of five seconds.
  if (want('viewer', 'viewer/walk.mp4')) {
    const { page, frame, info } = await openScene(browser, 'elevator', true, [960, 720]);
    // Walk forward with the runtime's own collision step, at its walking speed, so the loop
    // shows exactly where a visitor can go.
    let last = 0;
    await sequence(frame, 'viewer/walk.mp4', 10, 5, async (s) => {
      const dt = s - last;
      last = s;
      const pos = (await frame.evaluate((dt) => {
        const w = window.wander;
        const dir = new w.THREE.Vector3();
        w.camera.getWorldDirection(dir);
        dir.y = 0;
        dir.normalize().multiplyScalar(w.walk!.speed * dt);
        return w.walk!.advance(w.camera.position, dir, dt).toArray();
      }, dt)) as Vec;
      return { t: 2 + s, pos };
    });
    measurements.push(`walk loop: ${info.stature.toFixed(3)} u per body-height`);
    await page.close();
  }
  if (want('viewer', 'viewer/rewind.mp4')) {
    const { page, frame } = await openScene(browser, 'elevator', false, [960, 720]);
    await sequence(frame, 'viewer/rewind.mp4', 10, 5, (s) => ({ t: 8 - s }));
    await page.close();
  }
  if (want('viewer', 'viewer/audio.mp4')) {
    const { page, frame, info } = await openScene(browser, 'tos31', false, [960, 720]);
    await sequence(frame, 'viewer/audio.mp4', 10, 5, (s) => ({
      t: 2 + s,
      pos: offset(info, -0.2 * (s / 5), 0.5 * (s / 5)),
      look: info.homeLook,
    }));
    await page.close();
  }

  // Pipeline strip: the built room with the people placed, and the fit at a throw instant.
  if (want('steps', 'steps/4.jpg') || want('steps', 'steps/5.jpg')) {
    const { page, frame, info } = await openScene(browser, 'elevator', false, [1280, 720]);
    if (want('steps', 'steps/4.jpg')) await still(frame, { t: 0 }, 'steps/4.jpg');
    if (want('steps', 'steps/5.jpg'))
      await still(
        frame,
        { t: 6.0, pos: offset(info, -0.9, 1.2, -0.3), look: info.homeLook },
        'steps/5.jpg',
      );
    await page.close();
  }
  if (want('steps', 'steps/6.jpg')) {
    const page: Page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await page.goto(`${base}/demo.html?clip=elevator&walk=1`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.querySelector('iframe')?.contentWindow?.wander?.ready,
      null,
      { timeout: 600000 },
    );
    await page.waitForTimeout(20000);
    await mkdir(join(outRoot, 'steps'), { recursive: true });
    await page.screenshot({
      path: join(outRoot, 'steps', '6.jpg'),
      type: 'jpeg',
      quality: 90,
      timeout: 300000,
    });
    console.log('made steps/6.jpg');
    await page.close();
  }
} finally {
  await browser.close();
}
if (measurements.length) console.log(['Viewpoint offsets:', ...measurements].join('\n  '));
