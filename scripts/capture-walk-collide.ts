// Verify the ?walk=1 collision grid (wander.collision/2) against the running dev server, on the
// presets exactly as they ship (no extra flags, R9: the clamp used to stop the stairs2 climb):
//   stairs2  walk straight ahead from the start pose (the flight is at -z): the eye must rise in
//            steps, so the trace of camera y / walk floor against time is what to read
//   elevator walk 2.5 s in, then strafe left into the escalator well wall (-x) and right into the
//            escalator (+x): x must stop short of where the grid says the wall is
// Trace is sampled every 100 ms and printed; screenshots go to --out.
//   bun scripts/capture-walk-collide.ts [--out dir] [--base http://127.0.0.1:5399]
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { chromium, type Page, type BrowserContext } from 'playwright-core';

const rest = process.argv.slice(2);
const flag = (name: string, d: string) => {
  const i = rest.indexOf('--' + name);
  return i < 0 ? d : rest[i + 1];
};
const OUT = flag('out', path.resolve('.context/evidence/walk'));
const BASE = flag('base', 'http://127.0.0.1:5399');
const W = 1280,
  H = 720,
  SAMPLE_MS = 100;

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--use-angle=metal', '--ignore-gpu-blocklist'],
});
const state = () => {
  const w = window.wander,
    p = w.camera.position;
  return {
    x: +p.x.toFixed(3),
    y: +p.y.toFixed(3),
    z: +p.z.toFixed(3),
    floor: +w.walk!.floor.toFixed(3),
    eyeRatio: +((p.y - w.walk!.floor) / w.walk!.stature).toFixed(3),
    cellFloor: w.walk!.cellAt(p.x, p.z).floor,
    blockedUnder: w.walk!.blockedAt(p.x, p.z),
  };
};
type WalkState = ReturnType<typeof state>;
type TraceSample = WalkState & { t: number };
interface CommonReport {
  errors: string[];
  grid: import('./viewer-types.ts').WalkGridSummary | null;
  upm: number;
  start: WalkState;
}
interface StairsReport extends CommonReport {
  stepUp: number;
  trace: TraceSample[];
  end?: WalkState;
  rise?: unknown;
}
interface ElevatorReport extends CommonReport {
  fwd: TraceSample[];
  left: TraceSample[];
  right: TraceSample[];
  afterFwd?: WalkState;
  afterLeft?: WalkState;
  afterRight?: WalkState;
  rowBlocked?: unknown;
  stops?: unknown;
}
async function hold(page: Page, code: string, ms: number, trace: TraceSample[]) {
  await page.keyboard.down(code);
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    await page.waitForTimeout(SAMPLE_MS);
    trace.push({
      t: +((performance.now() - t0) / 1000).toFixed(2),
      ...(await page.evaluate(state)),
    });
  }
  await page.keyboard.up(code);
}
async function open(ctx: BrowserContext, url: string) {
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.wander?.ready, null, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await page.keyboard.press('Space');
  return { page, errors };
}
const out: Record<string, unknown> = {};
try {
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  {
    const { page, errors } = await open(ctx, `${BASE}/fourd.html?demo=stairs2&walk=1`);
    const r: StairsReport = {
      errors,
      grid: await page.evaluate(() => window.wander.walk!.grid),
      stepUp: await page.evaluate(() => +window.wander.walk!.stepUp.toFixed(3)),
      upm: await page.evaluate(() => +window.wander.upm.toFixed(4)),
      start: await page.evaluate(state),
      trace: [],
    };
    await page.screenshot({ path: path.join(OUT, 'stairs2-collide-0-start.png') });
    await hold(page, 'KeyW', 3000, r.trace);
    await page.screenshot({ path: path.join(OUT, 'stairs2-collide-1-midflight.png') });
    await hold(page, 'KeyW', 3000, r.trace);
    await page.screenshot({ path: path.join(OUT, 'stairs2-collide-2-top.png') });
    r.end = await page.evaluate(state);
    const floors = r.trace.map((s) => s.floor);
    r.rise = {
      units: +(r.end.floor - r.start.floor).toFixed(3),
      metres: +((r.end.floor - r.start.floor) / r.upm).toFixed(2),
      distinctFloorLevels: new Set(floors.map((f) => f.toFixed(2))).size,
      largestJump: +Math.max(...floors.slice(1).map((f, i) => f - floors[i])).toFixed(3),
      everDescended: floors.slice(1).some((f, i) => f < floors[i] - 1e-3),
    };
    out.stairs2 = r;
    await page.close();
  }
  {
    const { page, errors } = await open(ctx, `${BASE}/fourd.html?demo=elevator&walk=1`);
    const r: ElevatorReport = {
      errors,
      grid: await page.evaluate(() => window.wander.walk!.grid),
      upm: await page.evaluate(() => +window.wander.upm.toFixed(4)),
      start: await page.evaluate(state),
      fwd: [],
      left: [],
      right: [],
    };
    await hold(page, 'KeyW', 2500, r.fwd);
    r.afterFwd = await page.evaluate(state);
    await page.screenshot({ path: path.join(OUT, 'elevator-collide-0-in-corridor.png') });
    // where the grid says the wall is on this row, either side
    r.rowBlocked = await page.evaluate(() => {
      const w = window.wander,
        p = w.camera.position,
        g = w.walk!.grid!,
        row = [];
      for (let x = -2.5; x <= 3; x += g.cell) {
        const c = w.walk!.cellAt(x, p.z);
        row.push(
          w.walk!.blockedAt(x, p.z) ? '#' : Number.isNaN(c.floor) ? (c.inside ? '~' : ' ') : '.',
        );
      }
      return {
        from: -2.5,
        cell: +g.cell.toFixed(4),
        z: +p.z.toFixed(2),
        row: row.join(''),
        legend:
          '# footprint blocked at the walker height, . known floor, ~ unknown inside, space outside',
      };
    });
    await hold(page, 'KeyA', 4000, r.left);
    r.afterLeft = await page.evaluate(state);
    await page.screenshot({ path: path.join(OUT, 'elevator-collide-1-left-wall.png') });
    await hold(page, 'KeyD', 8000, r.right);
    r.afterRight = await page.evaluate(state);
    await page.screenshot({ path: path.join(OUT, 'elevator-collide-2-right-escalator.png') });
    const settled = (tr: TraceSample[], k: 'x' | 'y' | 'z') => {
      const last = tr.slice(-10).map((s) => s[k]);
      return +(Math.max(...last) - Math.min(...last)).toFixed(4);
    };
    r.stops = {
      leftX: r.afterLeft.x,
      leftMovingOverLastSecond: settled(r.left, 'x'),
      rightX: r.afterRight.x,
      rightMovingOverLastSecond: settled(r.right, 'x'),
    };
    out.elevator = r;
    await page.close();
  }
} finally {
  await browser.close();
}
console.log(JSON.stringify(out, null, 1));
