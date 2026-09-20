// Pulls the landing page media listed in media-manifest.json into public/media/.
//
// Every entry names one file below public/media/ and says where it comes from:
//   { "path": "elevator/source.mp4", "asset": "/clips/elevator.mp4" }
//       a file in the pinned viewer snapshot of the private shared bucket (teammate credentials)
//   { "path": "steps/2.jpg", "archive": "runs/elevator/tracks/track-overlay-000.jpg" }
//       a file in the author archive (author AWS credentials only; mark it "optional" so the
//       pull still succeeds for teammates without archive access)
//   { "path": "sample.mp4", "url": "https://..." }
//       a public URL
//   { "path": "elevator/poster.jpg", "frame": { "from": "elevator/wander.mp4", "at": 1.4 } }
//       one still taken with ffmpeg from a file pulled earlier in the list
//   { "path": "kitchen/wander.mp4", "clip": { "from": "...", "start": 3, "duration": 8 } }
//       a trimmed, re-encoded loop made with ffmpeg from a file pulled earlier in the list
//   { "path": "steps/3.jpg", "ffmpeg": { "inputs": ["a.png", "b.png"], "args": ["-filter_complex", "..."] } }
//       any other ffmpeg composition of files pulled earlier in the list
//   { "path": "elevator/render.jpg", "capture": true }
//       produced by scripts/capture-media.ts from the running viewer; listed here so the
//       manifest stays the one inventory of what the page expects
//
// "frame" and "clip" accept an optional ffmpeg "crop" ("w:h:x:y") and "scale" ("w:h") filter.
// Existing files are kept; delete a file to rebuild it. Nothing here writes to Git or S3.
import { mkdir, writeFile, rename, unlink } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { createWriteStream } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { spawnSync } from 'node:child_process';
import { GetObjectCommand } from '@aws-sdk/client-s3';
import {
  config,
  client,
  getJSON,
  hashFile,
  type AssetCatalog,
  type CatalogPointer,
} from '../../scripts/lib/shared-storage';

const root = resolve(import.meta.dirname, '..');
const manifestPath = join(root, 'media-manifest.json');
const outRoot = join(root, 'public', 'media');
const ffmpeg = process.env.FFMPEG || 'ffmpeg';

type Derivation = { from: string; crop?: string; scale?: string };
type Entry = {
  path: string;
  url?: string;
  asset?: string;
  archive?: string;
  frame?: Derivation & { at: number };
  clip?: Derivation & { start: number; duration: number; fps?: number };
  ffmpeg?: { inputs: string[]; args: string[] };
  capture?: boolean;
  optional?: boolean;
  note?: string;
};

const manifest: { files: Entry[] } = JSON.parse(await Bun.file(manifestPath).text());
const captured = new Set(manifest.files.filter((e) => e.capture).map((e) => e.path));

const safePath = (p: string) => {
  if (!/^[\w.-]+(\/[\w.-]+)*$/.test(p)) throw new Error(`Unsafe media path: ${p}`);
  return join(outRoot, p);
};

let s3: ReturnType<typeof client> | undefined;
const catalogs: Partial<Record<'viewer' | 'archive', AssetCatalog>> = {};
async function catalog(mode: 'viewer' | 'archive') {
  s3 ??= client(process.env);
  if (!catalogs[mode]) {
    const pointer = (await getJSON<CatalogPointer>(s3, config.catalogKey)).value;
    const key = mode === 'viewer' ? pointer.snapshot : pointer.archive;
    if (!key) throw new Error(`The shared pointer has no ${mode} snapshot`);
    catalogs[mode] = (await getJSON<AssetCatalog>(s3, key)).value;
  }
  return catalogs[mode]!;
}

async function download(dest: string, body: NodeJS.ReadableStream) {
  await mkdir(dirname(dest), { recursive: true });
  const temp = `${dest}.part`;
  try {
    await pipeline(body, createWriteStream(temp, { flags: 'w', mode: 0o644 }));
    await rename(temp, dest);
  } finally {
    await unlink(temp).catch(() => {});
  }
}

async function fromBucket(mode: 'viewer' | 'archive', logical: string, dest: string) {
  const entry = (await catalog(mode)).files[logical];
  if (!entry) throw new Error(`not in the pinned ${mode} snapshot: ${logical}`);
  const response = await s3!.send(new GetObjectCommand({ Bucket: config.bucket, Key: entry.key }));
  await download(dest, response.Body as NodeJS.ReadableStream);
  const sha = await hashFile(dest);
  if (sha !== entry.sha256) {
    await unlink(dest);
    throw new Error(`checksum mismatch for ${logical}`);
  }
}

async function fromUrl(url: string, dest: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
}

function filters(d: Derivation, extra: string[] = []) {
  const chain = [...extra];
  if (d.crop) chain.push(`crop=${d.crop}`);
  if (d.scale) chain.push(`scale=${d.scale}`);
  return chain.length ? ['-vf', chain.join(',')] : [];
}

function run(args: string[]) {
  const r = spawnSync(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', ...args], {
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  if (r.error) throw new Error(`${ffmpeg} is not available: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`${ffmpeg} exited with ${r.status}`);
}

function pulled(p: string) {
  const file = safePath(p);
  if (!existsSync(file)) throw new Error(`source not pulled yet: ${p}`);
  return file;
}

async function derive(entry: Entry, dest: string) {
  await mkdir(dirname(dest), { recursive: true });
  if (entry.ffmpeg) {
    const inputs = entry.ffmpeg.inputs.flatMap((p) => ['-i', pulled(p)]);
    run([...inputs, ...entry.ffmpeg.args, dest]);
    return;
  }
  const d = entry.frame ?? entry.clip!;
  const source = pulled(d.from);
  if (entry.frame) {
    // No seek at 0: ffmpeg's image demuxer drops a still's only frame when asked to seek.
    run([
      ...(entry.frame.at > 0 ? ['-ss', String(entry.frame.at)] : []),
      '-i',
      source,
      ...filters(entry.frame),
      '-frames:v',
      '1',
      '-q:v',
      '2',
      dest,
    ]);
    return;
  }
  const clip = entry.clip!;
  run([
    '-ss',
    String(clip.start),
    '-t',
    String(clip.duration),
    '-i',
    source,
    ...filters(clip, clip.fps ? [`fps=${clip.fps}`] : []),
    '-an',
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    '-crf',
    '23',
    '-preset',
    'medium',
    '-movflags',
    '+faststart',
    dest,
  ]);
}

let done = 0;
let skipped = 0;
let failed = 0;
let waiting = 0;

for (const entry of manifest.files) {
  const dest = safePath(entry.path);
  if (existsSync(dest)) {
    skipped++;
    continue;
  }
  const sources = [entry.frame?.from, entry.clip?.from, ...(entry.ffmpeg?.inputs ?? [])];
  if (
    entry.capture ||
    sources.some((from) => from && captured.has(from) && !existsSync(safePath(from)))
  ) {
    waiting++;
    continue;
  }
  try {
    if (entry.url) await fromUrl(entry.url, dest);
    else if (entry.asset) await fromBucket('viewer', entry.asset, dest);
    else if (entry.archive) await fromBucket('archive', entry.archive, dest);
    else if (entry.frame || entry.clip || entry.ffmpeg) await derive(entry, dest);
    else throw new Error('entry has no source');
    done++;
    console.log(`made ${entry.path}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (entry.optional) {
      console.warn(`skipped optional ${entry.path}: ${message}`);
      skipped++;
    } else {
      failed++;
      console.error(`failed ${entry.path}: ${message}`);
    }
  }
}

console.log(
  `done: ${done} made, ${skipped} kept or skipped, ${failed} failed` +
    (waiting ? `, ${waiting} waiting for scripts/capture-media.ts` : ''),
);
s3?.destroy();
process.exit(failed > 0 ? 1 : 0);
