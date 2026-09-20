#!/usr/bin/env bun
// Pre-download a snapshot's blobs into the cache the dev-server middleware reads, so the first
// visit to a scene is as fast as the second. It reuses server/shared-assets.ts, including its
// stall watchdog, retry and checksum verification, and writes no second copy of anything.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { blobCache } from '../server/shared-assets.ts';
import { ROOT } from './lib/shared-storage.ts';
import type { AssetCatalog, AssetEntry } from './lib/shared-storage.ts';

class WarmError extends Error {}

/** The scene ids the demo shell offers, in its own order. */
export function demoSceneIds(demoHtml: string): string[] {
  const ids = [...demoHtml.matchAll(/^\s*id: '([\w-]+)',$/gm)].map((m) => m[1]);
  if (!ids.length) throw new WarmError('No demo scene ids found in demo.html.');
  return [...new Set(ids)];
}

function braced(source: string, from: number) {
  const open = source.indexOf('{', from);
  for (let i = open, depth = 0; i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}' && --depth === 0) return source.slice(open, i + 1);
  }
  throw new WarmError('Unbalanced preset table in fourd.html.');
}

/** Every asset path a viewer preset names, read from the preset table instead of a fixed list. */
export function presetPaths(fourdHtml: string, ids: string[]): Map<string, string[]> {
  const table = braced(fourdHtml, fourdHtml.indexOf('const DEMOS = {'));
  const found = new Map<string, string[]>();
  for (const m of table.matchAll(/^ {4}'?([\w-]+)'?: \{/gm)) {
    if (!ids.includes(m[1])) continue;
    const entry = braced(table, m.index);
    found.set(m[1], [...new Set([...entry.matchAll(/'(\/[^']*)'/g)].map((x) => x[1]))]);
  }
  const missing = ids.filter((id) => !found.has(id));
  if (missing.length) throw new WarmError(`No viewer preset for: ${missing.join(', ')}`);
  return found;
}

/** A named world directory is warmed whole: the viewer derives collision, audio, placement and
 * frame paths from the sequence it was given, so warming only the named files would miss them. */
export function selectPaths(catalog: AssetCatalog, named: Iterable<string>): string[] {
  const all = Object.keys(catalog.files);
  const exact = new Set<string>();
  const prefixes = new Set<string>();
  for (const value of named) {
    const world = /^(\/worlds\/[^/]+\/)/.exec(value);
    if (world) prefixes.add(world[1]);
    else if (Object.hasOwn(catalog.files, value)) exact.add(value);
  }
  return all
    .filter((p) => exact.has(p) || [...prefixes].some((prefix) => p.startsWith(prefix)))
    .sort();
}

export async function warmAssets(
  cache: {
    catalog(): Promise<AssetCatalog & { snapshot: string }>;
    cached(f: AssetEntry): Promise<string>;
  },
  selection: { scenes?: string[]; paths?: string[]; prefixes?: string[]; all?: boolean },
  root = ROOT,
  concurrency = 6,
  report: (done: number, total: number, bytes: number) => void = () => {},
) {
  const catalog = await cache.catalog();
  let wanted: string[];
  if (selection.all) wanted = Object.keys(catalog.files).sort();
  else {
    // Only what a scene names is expanded to its world directory; --path means that one file.
    const named = new Set<string>();
    const scenes = selection.scenes ?? [];
    if (scenes.length) {
      const [demoHtml, fourdHtml] = await Promise.all([
        readFile(path.join(root, 'demo.html'), 'utf8'),
        readFile(path.join(root, 'fourd.html'), 'utf8'),
      ]);
      const ids = scenes.includes('all') ? demoSceneIds(demoHtml) : scenes;
      for (const paths of presetPaths(fourdHtml, ids).values()) for (const p of paths) named.add(p);
    }
    wanted = selectPaths(catalog, named);
    for (const p of selection.paths ?? []) if (Object.hasOwn(catalog.files, p)) wanted.push(p);
    for (const prefix of selection.prefixes ?? [])
      for (const p of Object.keys(catalog.files)) if (p.startsWith(prefix)) wanted.push(p);
    wanted = [...new Set(wanted)].sort();
  }
  if (!wanted.length)
    throw new WarmError('Nothing selected; use --scene, --path, --prefix or --all.');
  // One blob can back several paths; the cache is content-addressed, so download each blob once.
  const blobs = new Map<string, AssetEntry>();
  for (const p of wanted) blobs.set(catalog.files[p].sha256, catalog.files[p]);
  const entries = [...blobs.values()];
  const total = entries.reduce((n, f) => n + f.size, 0);
  let done = 0;
  let bytes = 0;
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(concurrency, entries.length) }, async () => {
      while (next < entries.length) {
        const f = entries[next++];
        await cache.cached(f);
        done++;
        bytes += f.size;
        report(done, entries.length, bytes);
      }
    }),
  );
  return { snapshot: catalog.snapshot, paths: wanted.length, blobs: entries.length, bytes: total };
}

const help = `Fill the local shared-asset cache the demo server reads (.context/shared-assets/blobs).

  bun run assets:warm                      Every scene in the demo picker (the pre-demo command)
  bun run assets:warm --scene lobby        One scene, by its demo id
  bun run assets:warm --prefix /clips/     Any snapshot prefix
  bun run assets:warm --all                Every file in the pinned snapshot

--scene ID     Demo scene id; repeatable. Defaults to every scene demo.html offers.
--path PATH    Exact snapshot path; repeatable.
--prefix P     Snapshot path prefix; repeatable.
--all          Warm the whole pinned snapshot instead of a selection.
--quiet        Print only the final summary.
--help         Show this text without contacting S3.

Downloads are verified by size and SHA-256, retried once, and abandoned after 30 s of silence.
Files already in the cache are skipped, so re-running is cheap. Nothing is uploaded or deleted,
and no second copy is written: this is the same cache a running server fills on demand.
`;

if (import.meta.main) {
  let cache: ReturnType<typeof blobCache> | undefined;
  try {
    const { values } = parseArgs({
      options: {
        scene: { type: 'string', multiple: true },
        path: { type: 'string', multiple: true },
        prefix: { type: 'string', multiple: true },
        all: { type: 'boolean' },
        quiet: { type: 'boolean' },
        help: { type: 'boolean' },
      },
    });
    if (values.help) console.log(help);
    else {
      const scenes =
        values.scene ?? (values.path || values.prefix || values.all ? undefined : ['all']);
      cache = blobCache(process.env);
      let last = 0;
      const result = await warmAssets(
        cache,
        { scenes, paths: values.path, prefixes: values.prefix, all: values.all },
        ROOT,
        6,
        (done, count, bytes) => {
          if (values.quiet || Date.now() - last < 2000) return;
          last = Date.now();
          console.error(`  ${done}/${count} blobs, ${(bytes / 1e9).toFixed(2)} GB`);
        },
      );
      console.log(JSON.stringify(result, null, 2));
    }
  } catch (error) {
    console.error(
      error instanceof WarmError
        ? error.message
        : 'Warming failed; check credentials, network, and disk space.',
    );
    process.exitCode = 1;
  } finally {
    cache?.destroy();
  }
}
