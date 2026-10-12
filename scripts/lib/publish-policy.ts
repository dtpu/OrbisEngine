import path from 'node:path';
import { homedir } from 'node:os';
import { realpath } from 'node:fs/promises';
import { filesUnder } from './shared-storage.ts';
import type { ScannedFile } from './shared-storage.ts';

export interface PublishMode {
  archiveOnly?: boolean;
  assetsOnly?: boolean;
}

export function publishInventory(
  root: string,
  mode: PublishMode,
  evidenceDir?: string,
  env: Record<string, string | undefined> = process.env,
) {
  if (mode.archiveOnly && mode.assetsOnly)
    throw new Error('--archive-only and --assets-only cannot be combined');
  const roots: {
    directory: string;
    logicalPrefix: string;
    kind: 'files' | 'archives';
    prefix: 'viewer' | 'archive';
    exclusionKey: string;
  }[] = [
    {
      directory: path.join(root, 'public'),
      logicalPrefix: mode.archiveOnly ? 'public/' : '/',
      kind: mode.archiveOnly ? 'archives' : 'files',
      prefix: mode.archiveOnly ? 'archive' : 'viewer',
      exclusionKey: 'public',
    },
  ];
  if (!mode.assetsOnly) {
    const runRoots: [string, string | undefined][] = [
      ['runs', '.context/runs'],
      ['runs-legacy', '.context/run'],
      ['runs-configured', env.WANDER_RUNS_ROOT],
      ['runs-dir', env.WANDER_RUNS_DIR],
    ];
    for (const [exclusionKey, directory] of runRoots) {
      if (!directory) continue;
      const expanded =
        directory === '~' || directory.startsWith('~/')
          ? path.join(homedir(), directory.slice(1))
          : directory;
      const resolved = path.resolve(root, expanded);
      if (roots.some((source) => source.logicalPrefix === 'runs/' && source.directory === resolved))
        continue;
      roots.push({
        directory: resolved,
        logicalPrefix: 'runs/',
        kind: 'archives',
        prefix: 'archive',
        exclusionKey,
      });
    }
    if (evidenceDir)
      roots.push({
        directory: path.resolve(evidenceDir),
        logicalPrefix: 'evidence/',
        kind: 'archives',
        prefix: 'archive',
        exclusionKey: 'evidence',
      });
  }
  return roots;
}

export interface PublishTask extends ScannedFile {
  kind: 'files' | 'archives';
  prefix: 'viewer' | 'archive';
}

// Resolve the complete local inventory before uploading anything. Two distinct runs
// cannot share runs/<name>, even when their current sets of files do not overlap.
export async function scanPublishInventory(inventory: ReturnType<typeof publishInventory>) {
  const tasks: PublishTask[] = [];
  const exclusions: Record<string, string[]> = {};
  const seenRoots = new Set<string>();
  const runOwners = new Map<string, string>();
  for (const source of inventory) {
    let directory: string;
    try {
      directory = await realpath(source.directory);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      exclusions[source.exclusionKey] = [];
      continue;
    }
    const identity = `${source.kind}:${source.logicalPrefix}:${directory}`;
    if (seenRoots.has(identity)) continue;
    seenRoots.add(identity);
    // Preserve the public scanner's existing symlink policy in both modes.
    const scan = await filesUnder(
      directory,
      source.exclusionKey === 'public' ? '/' : source.logicalPrefix,
    );
    if (source.logicalPrefix === 'runs/') {
      const names = new Set(
        scan.files.map((file) => file.path.slice('runs/'.length).split('/')[0]),
      );
      for (const name of names) {
        const owner = await realpath(path.join(directory, name));
        const previous = runOwners.get(name);
        if (previous && previous !== owner)
          throw new Error(
            `Archive run collision at runs/${name}: ${previous} and ${owner}. Rename or consolidate the local runs before publishing.`,
          );
        runOwners.set(name, owner);
      }
    }
    tasks.push(
      ...scan.files.map((file) => ({
        ...file,
        path:
          source.exclusionKey === 'public' && source.kind === 'archives'
            ? `${source.logicalPrefix}${file.path.slice(1)}`
            : file.path,
        kind: source.kind,
        prefix: source.prefix,
      })),
    );
    exclusions[source.exclusionKey] = scan.excluded;
  }
  return { tasks, exclusions };
}

export interface SnapshotWrite {
  key: string;
  value: unknown;
  condition: { IfMatch?: string; IfNoneMatch?: string };
}

// Keep pointer promotion last. Archive-only updates retain every existing viewer field.
export function publishMetadata(input: {
  archiveOnly: boolean;
  previous: Record<string, unknown>;
  etag: string | null | undefined;
  archiveKey: string;
  snapshot: string;
  catalogKey: string;
  schema: string;
  createdAt: string;
  files: unknown;
  archives: unknown;
  exclusions: unknown;
}): SnapshotWrite[] {
  const writes: SnapshotWrite[] = [
    {
      key: input.archiveKey,
      value: {
        schema: input.schema,
        createdAt: input.createdAt,
        files: input.archives,
        exclusions: input.exclusions,
      },
      condition: { IfNoneMatch: '*' },
    },
  ];
  if (!input.archiveOnly)
    writes.push({
      key: input.snapshot,
      value: { schema: input.schema, createdAt: input.createdAt, files: input.files },
      condition: { IfNoneMatch: '*' },
    });
  // Without a prior viewer pointer, retain the archive by key without creating a viewer catalog.
  if (!input.archiveOnly || input.etag)
    writes.push({
      key: input.catalogKey,
      value: input.archiveOnly
        ? { ...input.previous, archive: input.archiveKey }
        : {
            schema: input.schema,
            snapshot: input.snapshot,
            archive: input.archiveKey,
            createdAt: input.createdAt,
          },
      condition: input.etag ? { IfMatch: input.etag } : { IfNoneMatch: '*' },
    });
  return writes;
}

export async function writePublishMetadata(
  writes: SnapshotWrite[],
  put: (write: SnapshotWrite) => Promise<unknown>,
) {
  for (const write of writes) await put(write);
}
