import { spawnSync } from 'node:child_process';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const nodeCheck = spawnSync(
  'node',
  ['-e', 'if (Number(process.versions.node.split(".")[0]) < 24) process.exit(1)'],
  { cwd: root },
);
if (nodeCheck.status !== 0) {
  console.error(
    'Node.js 24+ is required. Select Node 24 (for example, volta install node@24) and retry.',
  );
  process.exit(1);
}
for (const [folder, command] of [
  ['', ['bun', 'install', '--frozen-lockfile']],
  ['', ['uv', 'sync', '--locked', '--group', 'dev', '--group', 'inference']],
  ['admin', ['bun', 'install', '--frozen-lockfile']],
  ['site', ['bun', 'install', '--frozen-lockfile']],
  ['', ['bun', 'run', 'build']],
  ['admin', ['bun', 'run', 'build']],
  ['site', ['bun', 'run', 'build']],
] as const) {
  const result = spawnSync(command[0], command.slice(1), {
    cwd: path.join(root, folder),
    stdio: 'inherit',
  });
  if (result.status !== 0) process.exit(result.status || 1);
}
console.log('\nSetup complete. Run bun run app, then open http://127.0.0.1:5399.');
