// Exercise the real HTML entry points without scene assets or an existing viewer server.
import assert from 'node:assert/strict';
import { chromium, type Page } from 'playwright-core';
import { build } from 'vite';

const root = new URL('../', import.meta.url).pathname;
const source = await Bun.file(new URL('../fourd.html', import.meta.url)).text();
const built = await build({
  configFile: false,
  root,
  logLevel: 'error',
  publicDir: false,
  build: {
    write: false,
    rollupOptions: { input: { fourd: `${root}fourd.html`, demo: `${root}demo.html` } },
  },
});
assert(!('on' in built), 'Expected a single build, not a watcher');
const outputs = (Array.isArray(built) ? built : [built]).flatMap((result) => result.output);
const files = new Map(
  outputs.map((output) => [
    `/${output.fileName}`,
    output.type === 'chunk' ? output.code : output.source,
  ]),
);
const productionHtml = String(files.get('/fourd.html'));
const productionEntry = productionHtml.match(/<script[^>]+src="([^"]+)"/)?.[1];
assert(productionEntry, 'Built viewer must have a module entry');
const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  fetch(request) {
    const path = new URL(request.url).pathname;
    const content = path === '/source.html' ? source : files.get(path);
    if (content === undefined) return new Response('Missing fixture asset', { status: 404 });
    const type = path.endsWith('.html')
      ? 'text/html'
      : path.endsWith('.css')
        ? 'text/css'
        : path.endsWith('.js')
          ? 'application/javascript'
          : 'application/octet-stream';
    return new Response(typeof content === 'string' ? content : new Uint8Array(content), {
      headers: { 'Content-Type': type },
    });
  },
});
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const base = `http://127.0.0.1:${server.port}`;
async function injectFailures(page: Page) {
  await page.evaluate(() => {
    window.dispatchEvent(new ErrorEvent('error', { message: 'injected error' }));
    window.dispatchEvent(
      new PromiseRejectionEvent('unhandledrejection', {
        promise: Promise.resolve(),
        reason: new Error('injected rejection'),
      }),
    );
    void Promise.reject(new Error('injected asynchronous rejection'));
  });
  await page.waitForTimeout(50);
}
try {
  for (const [path, entry] of [
    ['/source.html', '/src/fourd-session.js'],
    ['/source.html', '/src/fourd-runtime.js'],
    ['/fourd.html', productionEntry],
  ]) {
    const page = await browser.newPage();
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    if (entry === '/src/fourd-runtime.js') {
      await page.route('**/src/fourd-session.js', (route) =>
        route.fulfill({
          contentType: 'application/javascript',
          body: `import '/src/fourd-runtime.js'; export function startFourD() {}`,
        }),
      );
    }
    await page.route(`**${entry}`, (route) => route.abort());
    await page.goto(`${base}${path}`);
    await page.locator('#startup-error').waitFor({ state: 'visible' });
    assert.equal(await page.locator('#startup-error').getAttribute('role'), 'alert');
    assert.match(await page.locator('#startup-error').innerText(), /Reload to retry/);
    assert.equal(await page.locator('.scene-root').count(), 0);
    assert.equal(await page.evaluate('window.__wanderStartupError'), true);
    await injectFailures(page);
    assert.equal(await page.locator('#startup-error').count(), 1);
    assert.deepEqual(errors, ['injected asynchronous rejection']);
    await page.close();
  }

  // An unrelated later error must not mark an already healthy scene as failed.
  const healthy = await browser.newPage();
  await healthy.route('**/src/fourd-session.js', (route) =>
    route.fulfill({
      contentType: 'application/javascript',
      body: `export function startFourD(template) {
        const root = document.createElement('div');
        root.className = 'scene-root';
        root.append(template.content.cloneNode(true));
        document.body.append(root);
        document.querySelector('#st').textContent = 'Healthy scene';
        window.wander = { ready: true };
      }`,
    }),
  );
  await healthy.goto(`${base}/source.html`);
  await healthy.waitForFunction('window.wander?.ready');
  await injectFailures(healthy);
  assert.equal(await healthy.locator('#startup-error').count(), 0);
  assert.equal(await healthy.locator('#st').innerText(), 'Healthy scene');
  assert.equal(await healthy.evaluate('window.__wanderStartupError'), false);
  await healthy.close();

  // Use the actual desktop wrapper and compiled entry URL, including hashed bundle paths.
  const wrapper = await browser.newPage();
  const wrapperErrors: string[] = [];
  wrapper.on('pageerror', (error) => wrapperErrors.push(error.message));
  await wrapper.route(`**${productionEntry}`, (route) => route.abort());
  await wrapper.goto(`${base}/demo.html?clip=elevator&xrview=0`);
  await wrapper.locator('#loading.failed').waitFor({ state: 'visible' });
  assert.equal(await wrapper.locator('#loadretry').isVisible(), true);
  assert.match(await wrapper.locator('#loadmsg').innerText(), /could not finish loading/);
  assert.equal(await wrapper.frameLocator('#frame').locator('#startup-error').count(), 1);
  await Promise.all([
    wrapper.waitForRequest((request) => new URL(request.url()).pathname === productionEntry),
    wrapper.click('#loadretry'),
  ]);
  await wrapper.locator('#loading.failed').waitFor({ state: 'visible' });
  assert.equal(await wrapper.frameLocator('#frame').locator('#startup-error').count(), 1);
  assert.deepEqual(wrapperErrors, []);
  await wrapper.close();
  console.log(
    'FourD startup failures: source, production, repeated events, healthy scene, wrapper passed.',
  );
} finally {
  await browser.close();
  server.stop(true);
}
