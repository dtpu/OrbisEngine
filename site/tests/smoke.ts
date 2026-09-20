import { existsSync } from 'node:fs';
import { chromium, type Page } from 'playwright-core';

function chromePath(): string {
  const candidates = [
    process.env.CHROME_PATH,
    '/home/ubuntu/.local/bin/google-chrome',
    '/usr/bin/google-chrome',
  ].filter((p): p is string => Boolean(p));
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return chromium.executablePath();
}

const BASE = 'http://127.0.0.1:5400';
const SECTION_IDS = [
  'hero',
  'gallery',
  'compare',
  'how-it-works',
  'viewer',
  'limits',
  'team',
  'cta',
];

async function serverRunning(): Promise<boolean> {
  try {
    const res = await fetch(BASE, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function waitForServer(timeoutMs = 30000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await serverRunning()) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`vite dev server did not start at ${BASE}`);
}

async function collectErrors(run: (page: Page) => Promise<void>): Promise<string[]> {
  const browser = await chromium.launch({
    executablePath: chromePath(),
  });
  try {
    const page = await browser.newPage();
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(String(err)));
    await run(page);
    return errors;
  } finally {
    await browser.close();
  }
}

type Check = { name: string; run: () => Promise<void> };

const checks: Check[] = [
  {
    name: 'page loads with no console errors',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'all 8 section ids exist',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        for (const id of SECTION_IDS) {
          const el = await page.$(`section#${id}`);
          if (!el) throw new Error(`missing section #${id}`);
        }
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'every <video> has poster and src',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const bad = await page.$$eval('video', (videos) =>
          videos
            .filter(
              (v) =>
                !v.getAttribute('poster') ||
                (!v.getAttribute('src') && !v.querySelector('source[src]')),
            )
            .map((v) => v.outerHTML.slice(0, 80)),
        );
        if (bad.length > 0) throw new Error(`videos missing poster/src: ${bad.join(' | ')}`);
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'dev gallery (?dev=1) loads with no console errors',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/?dev=1`, { waitUntil: 'networkidle' });
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  // --- sections append below ---
  {
    name: 'viewer: 3 loops, kbd caps, headset line',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const videos = await page.$$('section#viewer video');
        if (videos.length !== 3) throw new Error(`expected 3 videos, got ${videos.length}`);
        const kbds = await page.$$eval('section#viewer kbd', (els) =>
          els.map((el) => el.textContent?.trim()),
        );
        for (const k of ['W', 'A', 'S', 'D', 'Space', 'Shift', 'R', 'M']) {
          if (!kbds.includes(k)) throw new Error(`missing key cap ${k}`);
        }
        const text = (await page.textContent('section#viewer')) ?? '';
        if (!text.includes('Works in a headset')) throw new Error('missing headset line');
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'hero: copy, CTAs and two labeled loops',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const h1 = await page.textContent('section#hero h1');
        if (h1?.trim() !== 'Step inside a video.') throw new Error(`hero h1: ${h1}`);
        for (const href of ['/demo.html', '#how-it-works']) {
          if (!(await page.$(`section#hero a[href="${href}"]`)))
            throw new Error(`hero missing CTA ${href}`);
        }
        const videos = await page.$$('section#hero video');
        if (videos.length !== 2) throw new Error(`hero has ${videos.length} videos, want 2`);
        const labels = await page.$$eval('section#hero figcaption', (els) =>
          els.map((e) => e.textContent?.trim()),
        );
        if (labels.join(',') !== 'Recorded,Wander') throw new Error(`hero labels: ${labels}`);
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'gallery renders 9 scene cards with viewer links',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const count = await page.$$eval('section#gallery article', (els) => els.length);
        if (count !== 9) throw new Error(`expected 9 gallery cards, got ${count}`);
        const links = await page.$$eval('section#gallery a[href^="/demo.html?clip="]', (els) =>
          els.map((a) => a.getAttribute('href')),
        );
        if (links.length < 9) throw new Error(`expected viewer links, got ${links.length}`);
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'compare: 3 sliders with Recorded/Wander labels and body-height captions',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const result = await page.$eval('section#compare', (section) => {
          const sliders = section.querySelectorAll('input[type="range"][aria-label="Compare"]');
          const imgs = section.querySelectorAll('img');
          const captions = Array.from(section.querySelectorAll('li > p')).map(
            (p) => p.textContent ?? '',
          );
          return {
            h2: section.querySelector('h2')?.textContent ?? '',
            sliders: sliders.length,
            imgs: imgs.length,
            recorded: section.textContent?.includes('Recorded') ?? false,
            wander: section.textContent?.includes('Wander') ?? false,
            captions,
          };
        });
        if (result.h2 !== 'Same second, new angle') throw new Error(`bad h2: ${result.h2}`);
        if (result.sliders !== 3) throw new Error(`expected 3 sliders, got ${result.sliders}`);
        if (result.imgs !== 6) throw new Error(`expected 6 images, got ${result.imgs}`);
        if (!result.recorded || !result.wander) throw new Error('missing slider labels');
        if (result.captions.length !== 3 || result.captions.some((c) => !/body-height/.test(c))) {
          throw new Error(`captions must mention body-heights: ${result.captions.join(' | ')}`);
        }
        if (result.captions.some((c) => /\bm\b|metre|meter/i.test(c))) {
          throw new Error('captions must not use metres');
        }
      });
      const relevant = errors.filter((e) => !/media\//.test(e));
      if (relevant.length > 0) throw new Error(`console errors: ${relevant.join(' | ')}`);
    },
  },
  {
    name: 'how-it-works renders six steps with images and footer',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const count = await page.$$eval('section#how-it-works ol li', (items) => items.length);
        if (count !== 6) throw new Error(`expected 6 steps, got ${count}`);
        const imgs = await page.$$eval('section#how-it-works ol li img', (els) =>
          els.map((img) => [img.getAttribute('src') ?? '', img.getAttribute('alt') ?? '']),
        );
        if (imgs.length !== 6 || imgs.some(([src, alt]) => !src || !alt))
          throw new Error(`step images missing src/alt: ${JSON.stringify(imgs)}`);
        const text = await page.$eval('section#how-it-works', (el) => el.textContent ?? '');
        if (!text.includes('Completed processing is not accepted quality'))
          throw new Error('footer line missing');
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'limits renders heading and 4 items with no console errors',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const h2 = await page.textContent('section#limits h2');
        if (!h2?.includes('What Wander can\u2019t do yet.'))
          throw new Error(`bad limits h2: ${h2}`);
        const n = await page.$$eval('section#limits li', (els) => els.length);
        if (n !== 4) throw new Error(`expected 4 limits, got ${n}`);
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'team renders 4 members with GitHub links',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const hrefs = await page.$$eval('section#team li a', (els) =>
          els.map((a) => a.getAttribute('href')),
        );
        if (hrefs.length !== 4 || hrefs.some((h) => !h?.startsWith('https://github.com/')))
          throw new Error(`bad team links: ${hrefs.join(', ')}`);
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
  {
    name: 'cta is dark and has demo + code buttons',
    run: async () => {
      const errors = await collectErrors(async (page) => {
        await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
        const theme = await page.getAttribute('section#cta', 'data-theme');
        if (theme !== 'dark') throw new Error(`cta theme: ${theme}`);
        const hrefs = await page.$$eval('section#cta a', (els) =>
          els.map((a) => a.getAttribute('href')),
        );
        if (!hrefs.includes('/demo.html') || !hrefs.includes('https://github.com/dtpu/htn2026'))
          throw new Error(`cta links: ${hrefs.join(', ')}`);
      });
      if (errors.length > 0) throw new Error(`console errors: ${errors.join(' | ')}`);
    },
  },
];

let server: Bun.Subprocess | null = null;
try {
  if (!(await serverRunning())) {
    server = Bun.spawn({
      cmd: ['bunx', '--bun', 'vite', '--port', '5400', '--host', '127.0.0.1'],
      cwd: new URL('..', import.meta.url).pathname,
      stdout: 'inherit',
      stderr: 'inherit',
    });
    await waitForServer();
  }

  let failures = 0;
  for (const check of checks) {
    try {
      await check.run();
      console.log(`PASS ${check.name}`);
    } catch (err) {
      failures++;
      console.error(`FAIL ${check.name}: ${err}`);
    }
  }
  process.exit(failures > 0 ? 1 : 0);
} finally {
  server?.kill();
}
