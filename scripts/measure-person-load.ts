// Measures what the real viewer downloads and how long it takes until every person keyframe is in
// memory, for one demo URL. Evidence goes outside git; run it against a local-mode server so the
// numbers are the packaged bytes and not S3 or cache behaviour. No GPU is needed for the part this
// measures: keyframes are fetched and decoded before anything is drawn.
//
//   WANDER_ASSETS_MODE=local bunx --bun vite --port 5399 --host 127.0.0.1 &
//   bun scripts/measure-person-load.ts "http://127.0.0.1:5399/demo.html?clip=stairs2&walk=1" out.json
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import path from 'node:path';

const [url, out] = process.argv.slice(2);
if (!url || !out) throw new Error('Usage: bun scripts/measure-person-load.ts URL OUT.json');
await mkdir(path.dirname(out), { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await (
    await browser.newContext({ viewport: { width: 1280, height: 720 } })
  ).newPage();
  const responses: { url: string; bytes: number; status: number }[] = [];
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'warning' || m.type() === 'error') errors.push(`${m.type()}: ${m.text()}`);
  });
  page.on('response', async (r) => {
    let bytes = Number(r.headers()['content-length'] || 0);
    if (!bytes && r.status() === 200) {
      try {
        bytes = (await r.body()).byteLength;
      } catch {}
    }
    responses.push({ url: r.url(), bytes, status: r.status() });
  });
  const t0 = performance.now();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  const people = () =>
    page.evaluate(() => {
      const frame = document.querySelector('iframe') as HTMLIFrameElement | null;
      const w = (frame ? frame.contentWindow : window) as any;
      return (w?.wander?.people ?? []).map((p: any) => ({ loaded: p.loaded, nF: p.nF }));
    });
  let firstKey: number | null = null,
    allKeys: number | null = null;
  const deadline = t0 + 600000;
  while (performance.now() < deadline) {
    const ps = await people();
    if (ps.length && firstKey === null && ps.every((p) => p.loaded >= 1))
      firstKey = performance.now() - t0;
    if (ps.length && ps.every((p) => p.loaded === p.nF)) {
      allKeys = performance.now() - t0;
      break;
    }
    await page.waitForTimeout(250);
  }
  const person = responses.filter((r) => /\/person[^/]*\//.test(r.url));
  const sum = (rs: typeof responses) => rs.reduce((a, r) => a + r.bytes, 0);
  const report = {
    url,
    people: await people(),
    seconds: {
      firstKeyframe: firstKey && firstKey / 1000,
      allKeyframes: allKeys && allKeys / 1000,
    },
    bytes: {
      total: sum(responses),
      person: sum(person),
      personPly: sum(person.filter((r) => r.url.endsWith('.ply'))),
      personMotion: sum(person.filter((r) => /motion\.(u16|f32)$/.test(r.url))),
    },
    requests: { total: responses.length, person: person.length },
    failed: responses.filter((r) => r.status >= 400).map((r) => `${r.status} ${r.url}`),
    errors: errors.slice(0, 20),
  };
  await writeFile(out, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
