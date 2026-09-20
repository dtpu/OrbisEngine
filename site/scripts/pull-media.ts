// Pulls media files listed in media-manifest.json into public/media/.
// Manifest shape: { "files": [{ "url": "https://...", "path": "scene-id/poster.jpg" }] }
import { mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const manifestPath = join(root, 'media-manifest.json');
const outRoot = join(root, 'public', 'media');

type Entry = { url: string; path: string };

const manifest: { files: Entry[] } = JSON.parse(await Bun.file(manifestPath).text());

let downloaded = 0;
let skipped = 0;
let failed = 0;

for (const { url, path } of manifest.files) {
  const dest = join(outRoot, path);
  if (existsSync(dest)) {
    skipped++;
    continue;
  }
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await mkdir(dirname(dest), { recursive: true });
    await writeFile(dest, Buffer.from(await res.arrayBuffer()));
    downloaded++;
    console.log(`downloaded ${path}`);
  } catch (err) {
    failed++;
    console.error(`failed ${path}: ${err}`);
  }
}

console.log(`done: ${downloaded} downloaded, ${skipped} skipped, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
