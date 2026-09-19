import path from 'node:path';

export interface PublishMode {
  archiveOnly?: boolean;
  assetsOnly?: boolean;
}

export function publishInventory(root: string, mode: PublishMode, evidenceDir?: string) {
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
    roots.push({
      directory: path.join(root, '.context/run'),
      logicalPrefix: 'runs/',
      kind: 'archives',
      prefix: 'archive',
      exclusionKey: 'runs',
    });
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
