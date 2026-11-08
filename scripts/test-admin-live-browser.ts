/** Browser smoke for an isolated local API/admin pair; no supervisor or inference required. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const clip = process.argv[2];
if (!clip)
  throw new Error('Usage: bun scripts/test-admin-live-browser.ts /absolute/path/to/clip.mp4');
const base = process.env.ADMIN_SMOKE_URL ?? 'http://127.0.0.1:5401';
const output = resolve(process.env.ADMIN_SMOKE_OUTPUT ?? '.context/evidence/admin-live-smoke');
const runId = `admin-smoke-${Date.now()}`;
await mkdir(output, { recursive: true });
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
const browser = executablePath
  ? await chromium.launch({ executablePath, headless: true })
  : await chromium
      .launch({ channel: 'chrome', headless: true })
      .catch(() => chromium.launch({ headless: true }));
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
page.setDefaultTimeout(30_000);
const pageErrors: string[] = [];
const failedResponses: { url: string; status: number }[] = [];
page.on('pageerror', (error) => pageErrors.push(error.message));
page.on('response', (response) => {
  if (response.status() >= 400)
    failedResponses.push({ url: response.url(), status: response.status() });
});

try {
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.locator('#option-marble').waitFor();
  const schemaResponse = await page.request.get(`${base}/api/pipeline/options`);
  assert.equal(schemaResponse.status(), 200);
  const schema = await schemaResponse.json();
  assert.equal(await page.locator('#option-marble').inputValue(), schema.properties.marble.default);
  await page.locator('#requested-id').fill(runId);
  await page.locator('input[type=file]').setInputFiles(resolve(clip));
  await page.locator('#option-marble').selectOption('none');
  await page.locator('#option-objects').uncheck();
  await page.screenshot({ path: `${output}/create.png`, fullPage: true });

  // Attach rejection handlers immediately: upload may still be pending if a setup assertion fails.
  const doctorResponse = page
    .waitForResponse((response) => response.url().endsWith('/api/pipeline/doctor'), {
      timeout: 60_000,
    })
    .catch((error: Error) => error);
  const submission = page
    .waitForResponse((response) => response.url().endsWith('/api/pipeline/runs/upload'), {
      timeout: 90_000,
    })
    .catch((error: Error) => error);
  await page.getByRole('button', { name: 'Upload and process', exact: true }).click();
  const doctor = await doctorResponse;
  if (doctor instanceof Error) throw doctor;
  assert.equal(doctor.status(), 200);
  const setup = await doctor.json();
  await page
    .getByRole('heading', { name: /Local setup checks passed|Setup needs attention/ })
    .waitFor();
  await page.screenshot({ path: `${output}/doctor.png`, fullPage: true });
  if (setup.checks.some((check: { status: string }) => check.status === 'blocked')) {
    await page.getByRole('button', { name: 'Queue for later', exact: true }).click();
  }
  const uploaded = await submission;
  if (uploaded instanceof Error) throw uploaded;
  assert.equal(uploaded.status(), 201, await uploaded.text());
  assert.equal((await uploaded.json()).id, runId);

  const runUrl = `${base}/api/pipeline/runs/${runId}`;
  const summary = async () => (await page.request.get(runUrl)).json();
  assert.equal((await summary()).status, 'queued');
  await page.goto(`${base}/runs/${runId}`, { waitUntil: 'domcontentloaded' });
  await page.getByText('Source clip', { exact: true }).first().waitFor();
  await page.waitForFunction(
    () => [...document.querySelectorAll('video')].some((video) => video.readyState >= 2),
    null,
    { timeout: 30_000 },
  );
  await page
    .locator('video')
    .first()
    .evaluate(async (element) => {
      const video = element as HTMLVideoElement;
      await new Promise<void>((done) => {
        video.addEventListener('seeked', () => done(), { once: true });
        video.currentTime = Math.min(1, video.duration / 2);
      });
    });
  const source = await page
    .locator('video')
    .first()
    .evaluate((element) => {
      const video = element as HTMLVideoElement;
      return {
        duration: video.duration,
        readyState: video.readyState,
        currentTime: video.currentTime,
      };
    });
  assert.ok(source.duration > 0);
  await page.screenshot({ path: `${output}/run.png`, fullPage: true });

  await page.getByRole('button', { name: 'Pause run', exact: true }).click();
  await page.getByRole('button', { name: 'Resume run', exact: true }).waitFor();
  assert.equal((await summary()).status, 'paused');
  await page.screenshot({ path: `${output}/paused.png`, fullPage: true });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Resume run', exact: true }).click();
  await page.getByRole('button', { name: 'Pause run', exact: true }).waitFor();
  assert.equal((await summary()).live.paused, false);
  await page.getByRole('button', { name: 'Pause run', exact: true }).click();
  await page.getByRole('button', { name: 'Resume run', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Cancel run', exact: true }).click();
  await page.getByRole('button', { name: 'Confirm cancel', exact: true }).click();
  await page.getByRole('heading', { name: 'Run canceled', exact: true }).waitFor();
  const canceled = await summary();
  assert.equal(canceled.status, 'canceled');
  assert.equal(canceled.recordedStatus, 'canceled');
  assert.equal(canceled.live.paused, false);
  await page.screenshot({ path: `${output}/canceled.png`, fullPage: true });
  assert.equal(failedResponses.length, 0, JSON.stringify(failedResponses));

  // A second upload opens the existing outcome, including a cancellation. It cannot
  // silently create a new queued run or resume the one the operator stopped.
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.locator('#option-marble').waitFor();
  await page.locator('#requested-id').fill(`${runId}-repeat`);
  await page.locator('input[type=file]').setInputFiles(resolve(clip));
  await page.locator('#option-marble').selectOption('none');
  await page.locator('#option-objects').uncheck();
  const repeatSetup = page.waitForResponse((response) =>
    response.url().endsWith('/api/pipeline/doctor'),
  );
  await page.getByRole('button', { name: 'Upload and process', exact: true }).click();
  assert.equal((await repeatSetup).status(), 200);
  const repeatResponse = page.waitForResponse((response) =>
    response.url().endsWith('/api/pipeline/runs/upload'),
  );
  await page.getByRole('button', { name: 'Queue for later', exact: true }).click();
  const reopened = await repeatResponse;
  assert.equal(reopened.status(), 200, await reopened.text());
  const repeated = await reopened.json();
  assert.equal(repeated.id, runId);
  assert.equal(repeated.reused, true);
  assert.equal(repeated.status, 'canceled');
  await page.getByRole('status').filter({ hasText: 'Opened existing run' }).waitFor();
  assert.deepEqual(await summary(), canceled);
  const listed = await (await page.request.get(`${base}/api/pipeline`)).json();
  assert.deepEqual(
    listed.runs.map((run: { id: string }) => run.id),
    [runId],
  );
  await page.screenshot({ path: `${output}/reopened.png`, fullPage: true });

  await page.goto(`${base}/runs/${runId}-missing`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('alert').filter({ hasText: 'unknown run' }).waitFor();
  await page.screenshot({ path: `${output}/missing-run.png`, fullPage: true });
  assert.ok(failedResponses.every((response) => response.status === 404));
  assert.deepEqual(pageErrors, []);
  await writeFile(
    `${output}/report.json`,
    JSON.stringify(
      { runId, source, setup, canceled, repeated, failedResponses, pageErrors },
      null,
      2,
    ),
  );
  console.log(JSON.stringify({ runId, source, status: canceled.status, output }));
} finally {
  await browser.close();
}
