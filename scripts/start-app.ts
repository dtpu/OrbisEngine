/** Start the complete local app. Paid upload processing is an explicit, bounded option. */
import { spawn, execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { loadEnv } from 'vite';
import { assertPortsAvailable, waitForService } from '../server/app-startup.ts';

const root = path.resolve(import.meta.dirname, '..');
const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log(
    'bun run app [--worker --max-uploads N]\nDefault: local UI, API and viewer; uploads queue until a worker is enabled.',
  );
  process.exit(0);
}
const worker = args.includes('--worker');
const maxIndex = args.indexOf('--max-uploads');
const maxUploads = maxIndex >= 0 ? Number(args[maxIndex + 1]) : 1;
const unknown = args.filter(
  (arg, i) =>
    arg !== '--worker' && arg !== '--max-uploads' && !(maxIndex >= 0 && i === maxIndex + 1),
);
if (
  unknown.length ||
  !Number.isSafeInteger(maxUploads) ||
  maxUploads < 1 ||
  (maxIndex >= 0 && !worker)
)
  throw new Error('Usage: bun run app [--worker --max-uploads N]');

const env: Record<string, string | undefined> = {
  ...(process.env.WANDER_APP_LOAD_ENV === '0' ? {} : loadEnv('author', root, '')),
  ...process.env,
};
const state = path.resolve(root, env.WANDER_APP_STATE || '.context/app');
await mkdir(state, { recursive: true, mode: 0o700 });
const tokenFile = path.join(state, 'api-token');
if (!env.WANDER_API_TOKEN) {
  try {
    env.WANDER_API_TOKEN = (await readFile(tokenFile, 'utf8')).trim();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    env.WANDER_API_TOKEN = randomBytes(32).toString('hex');
    await writeFile(tokenFile, env.WANDER_API_TOKEN, { flag: 'wx', mode: 0o600 });
  }
}
if (env.WANDER_API_TOKEN.length < 24)
  throw new Error('WANDER_API_TOKEN must have at least 24 characters.');
function port(name: string, fallback: number) {
  const value = Number(env[name] || fallback);
  if (!Number.isSafeInteger(value) || value < 1 || value > 65535)
    throw new Error(`Invalid ${name}`);
  return value;
}
const apiPort = port('WANDER_API_PORT', 8000);
const adminPort = port('WANDER_ADMIN_PORT', 5401);
env.WANDER_DATABASE_URL ||= `sqlite:///${path.join(state, 'pipeline.sqlite')}`;
env.WANDER_RUNS_ROOT ||= path.join(state, 'runs');
env.WANDER_ARTIFACT_ROOT ||= path.join(state, 'artifacts');
env.WANDER_CODE_REVISION = execFileSync('git', ['rev-parse', 'HEAD'], {
  cwd: root,
  encoding: 'utf8',
}).trim();
env.WANDER_PIPELINE_API = `http://127.0.0.1:${apiPort}`;
env.WANDER_ADMIN_URL = `http://127.0.0.1:${adminPort}`;
env.WANDER_VIEWER_ORIGIN = '';
env.WANDER_APP_ORIGIN = 'http://127.0.0.1:5399';
if (worker && 'WANDER_STAGE_AUTHORIZATION' in env)
  throw new Error('An upload worker cannot inherit a source-specific WANDER_STAGE_AUTHORIZATION.');
for (const file of ['.venv/bin/python', 'admin/.next/BUILD_ID', 'site/dist/index.html']) {
  try {
    await access(path.join(root, file));
  } catch {
    throw new Error(`Missing ${file}. Run bun run setup first.`);
  }
}
await mkdir(env.WANDER_RUNS_ROOT, { recursive: true });
await mkdir(env.WANDER_ARTIFACT_ROOT, { recursive: true });
const groups = new Set<number>();
let stopping = false;
function stop(code: number) {
  if (stopping) return;
  stopping = true;
  for (const group of groups) {
    try {
      process.kill(-group, 'SIGTERM');
    } catch {}
  }
  process.exitCode = code;
  setTimeout(() => {
    // A wrapper may exit before its descendants. The process group still belongs to this app.
    for (const group of groups) {
      try {
        process.kill(-group, 'SIGKILL');
      } catch {}
    }
    process.exit(code);
  }, 3000);
}
process.once('SIGINT', () => stop(0));
process.once('SIGTERM', () => stop(0));
function start(name: string, command: string[], cwd = root) {
  if (stopping) throw new Error('Local app is stopping.');
  const child = spawn(command[0], command.slice(1), { cwd, env, stdio: 'inherit', detached: true });
  if (child.pid) groups.add(child.pid);
  child.once('error', () => {
    console.error(`${name} could not start.`);
    stop(1);
  });
  child.once('exit', (code) => {
    if (!stopping) {
      console.error(`${name} stopped (${code}). Stopping the local app.`);
      stop(code || 1);
    }
  });
  return child;
}
try {
  await assertPortsAvailable([apiPort, adminPort, 5399]);
  if (stopping) throw new Error('Local app is stopping.');
  // Alembic handles both new and existing databases, without resetting stored runs.
  execFileSync(path.join(root, '.venv/bin/python'), ['-m', 'alembic', 'upgrade', 'head'], {
    cwd: root,
    env,
    stdio: 'inherit',
  });
  const api = start('API', [
    path.join(root, '.venv/bin/python'),
    '-m',
    'uvicorn',
    'orchestrator.api.service:create_service',
    '--factory',
    '--host',
    '127.0.0.1',
    '--port',
    String(apiPort),
  ]);
  await waitForService(`${env.WANDER_PIPELINE_API}/api/pipeline/options`, api, () => stopping, {
    authorization: `Bearer ${env.WANDER_API_TOKEN}`,
  });
  const dashboard = start(
    'Dashboard',
    ['bun', 'run', 'start', '--port', String(adminPort), '--hostname', '127.0.0.1'],
    path.join(root, 'admin'),
  );
  await waitForService(`${env.WANDER_ADMIN_URL}/api/pipeline/options`, dashboard, () => stopping);
  const viewer = start('Viewer', [
    'bunx',
    '--bun',
    'vite',
    '--port',
    '5399',
    '--host',
    '127.0.0.1',
  ]);
  await waitForService('http://127.0.0.1:5399/api/pipeline/options', viewer, () => stopping);
  if (worker)
    start('Upload worker', [
      path.join(root, '.venv/bin/python'),
      '-m',
      'orchestrator.supervisor',
      '--runs',
      env.WANDER_RUNS_ROOT,
      '--uploads-only',
      '--watch',
      '--max-uploads',
      String(maxUploads),
    ]);
  console.log(
    `\nOrbis Engine: http://127.0.0.1:5399\nAbout: http://127.0.0.1:5399/about/\nUpload worker: ${worker ? `enabled (${maxUploads} total admitted uploads)` : 'off — uploads are saved in the queue'}\n`,
  );
} catch (error) {
  console.error((error as Error).message);
  stop(1);
}
