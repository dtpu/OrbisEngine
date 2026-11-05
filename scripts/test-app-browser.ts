/** Exercise the real local stack with an isolated database and no worker or paid providers. */
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, open } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const evidence = path.join(root, '.context/evidence/app-browser');
await mkdir(evidence, { recursive: true });
const state = await mkdtemp(path.join(evidence, 'state-'));
const base = 'http://127.0.0.1:5399';
// Never attach this test to an operator's running app or upload into their database.
try {
  await fetch(base, { signal: AbortSignal.timeout(1000) });
  throw new Error('Port 5399 is already in use. Stop your local app before running test:e2e.');
} catch (error) {
  if ((error as Error).message.includes('already in use')) throw error;
}
const env = {
  ...process.env,
  WANDER_APP_LOAD_ENV: '0',
  WANDER_APP_STATE: state,
  WANDER_DATABASE_URL: `sqlite:///${path.join(state, 'pipeline.sqlite')}`,
  WANDER_RUNS_ROOT: path.join(state, 'runs'),
  WANDER_ARTIFACT_ROOT: path.join(state, 'artifacts'),
  WANDER_API_TOKEN: 'isolated-browser-test-token-no-provider-access',
  WANDER_ASSETS_MODE: 'local',
  WANDER_API_PORT: '18000',
  WANDER_ADMIN_PORT: '15401',
  ADMIN_SMOKE_URL: base,
  ADMIN_SMOKE_OUTPUT: path.join(evidence, 'admin'),
  SITE_SMOKE_URL: `${base}/about/`,
};
const clip = path.join(state, 'source.mp4');
function run(command: string[], cwd = root) {
  const result = spawnSync(command[0], command.slice(1), { cwd, env, stdio: 'inherit' });
  assert.equal(result.status, 0, `${command.join(' ')} failed`);
}
run([
  'ffmpeg',
  '-hide_banner',
  '-loglevel',
  'error',
  '-f',
  'lavfi',
  '-i',
  'testsrc2=size=640x360:rate=24',
  '-f',
  'lavfi',
  '-i',
  'sine=frequency=440:sample_rate=48000',
  '-t',
  '2',
  '-c:v',
  'libx264',
  '-pix_fmt',
  'yuv420p',
  '-c:a',
  'aac',
  clip,
]);
const log = await open(path.join(evidence, 'services.log'), 'w');
const app = spawn('bun', ['scripts/start-app.ts'], {
  cwd: root,
  env,
  stdio: ['ignore', log.fd, log.fd],
});
try {
  const deadline = Date.now() + 60_000;
  let ready = false;
  while (Date.now() < deadline && app.exitCode === null) {
    try {
      const response = await fetch(`${base}/api/pipeline/options`, {
        signal: AbortSignal.timeout(1000),
      });
      if (response.ok) {
        ready = true;
        break;
      }
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  assert.ok(ready, `App did not start; inspect ${evidence}/services.log`);
  const dashboard = await (await fetch(base)).text();
  assert.ok(dashboard.includes('href="/demo.html"'), 'Viewer must use the same origin');
  assert.ok(dashboard.includes('href="/about/"'), 'About must be connected');
  assert.ok(!dashboard.includes(env.WANDER_API_TOKEN), 'API token must remain server-side');
  const viewer = await fetch(`${base}/fourd.html`);
  assert.equal(viewer.status, 200);
  assert.ok((await viewer.text()).includes('id="scene-template"'));
  run(['bun', 'scripts/test-admin-live-browser.ts', clip]);
  run(['bun', 'run', 'test:smoke'], path.join(root, 'site'));
  console.log(`Full app browser checks passed. Evidence: ${evidence}`);
} finally {
  app.kill('SIGTERM');
  await new Promise<void>((resolve) => {
    if (app.exitCode !== null) return resolve();
    app.once('exit', () => resolve());
  });
  await log.close();
}
