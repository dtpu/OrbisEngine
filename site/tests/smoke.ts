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
