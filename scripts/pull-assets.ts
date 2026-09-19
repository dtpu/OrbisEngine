#!/usr/bin/env bun
import { constants } from 'node:fs';
import { createWriteStream } from 'node:fs';
import { lstat, mkdir, open, readFile, writeFile, link, rename, unlink } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { parseArgs } from 'node:util';
import { GetObjectCommand } from '@aws-sdk/client-s3';
import {
  ROOT,
  config,
  client,
  getJSON,
  hashFile,
  forbidden,
  type AssetCatalog,
  type AssetEntry,
  type CatalogPointer,
  type ObjectReader,
} from './lib/shared-storage.ts';

export interface PullOptions {
  out: string;
  paths?: string[];
  prefixes?: string[];
  list?: boolean;
  archive?: boolean;
  snapshot?: string;
  overwrite?: boolean;
  includeCinematic?: boolean;
}
interface Receipt {
  schema: 'wander.pull/1';
  bucket: string;
  mode: 'viewer' | 'archive';
  snapshot: string;
  manifestSha256: string;
  files: Record<string, AssetEntry>;
}
class PullError extends Error {}
const receiptName = '.wander-pull.json';
const missing = (error: unknown) => (error as NodeJS.ErrnoException).code === 'ENOENT';

function logicalPath(value: string, archive: boolean, prefix = false): string {
  if (typeof value !== 'string' || /[\\%\x00-\x1f\x7f:]/.test(value)) {
    throw new PullError(
      'Logical paths must be plain paths, without encoding or control characters.',
    );
  }
  if (archive ? !/^(runs|evidence)\//.test(value) : !value.startsWith('/')) {
    throw new PullError('Use /paths for viewer files, or runs/... and evidence/... for archives.');
  }
  const relative = archive ? value : value.slice(1);
  const parts = (prefix ? relative.replace(/\/$/, '') : relative).split('/');
  if (
    parts.some((part) => !part || part === '.' || part === '..') ||
    forbidden(relative) ||
    parts[0].startsWith('.wander-pull')
  ) {
    throw new PullError('Unsafe or reserved logical path.');
  }
  return relative;
}

async function checkedDirectory(root: string, relative: string): Promise<string> {
  let current = root;
  for (const part of relative.split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    try {
      await mkdir(current);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
    }
    const info = await lstat(current);
    if (info.isSymbolicLink() || !info.isDirectory())
      throw new PullError('Output directories must not contain symlinks or non-directories.');
  }
  return current;
}
async function regularFile(file: string) {
  try {
    const info = await lstat(file);
    if (info.isSymbolicLink() || !info.isFile())
      throw new PullError('Refusing a symlink or non-file destination.');
    return info;
  } catch (error) {
    if (missing(error)) return null;
    throw error;
  }
}
async function matches(file: string, entry: AssetEntry): Promise<boolean> {
  const handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const before = await handle.stat();
    if (!before.isFile() || before.size !== entry.size) return false;
    const hash = createHash('sha256');
    for await (const chunk of handle.createReadStream({ autoClose: false })) hash.update(chunk);
    const after = await handle.stat();
    return (
      before.size === after.size &&
      before.mtimeMs === after.mtimeMs &&
      hash.digest('hex') === entry.sha256
    );
  } finally {
    await handle.close();
  }
}
async function saveReceipt(file: string, receipt: Receipt, fresh: boolean, root: string) {
  await checkedDirectory(root, path.relative(root, path.dirname(file)));
  const temp = `${file}.${randomUUID()}.part`;
  try {
    await writeFile(temp, JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
    await regularFile(file);
    if (fresh) await link(temp, file);
    else await rename(temp, file);
  } finally {
    await unlink(temp).catch(() => {});
  }
}
function checkedSnapshot(key: string, mode: 'viewer' | 'archive') {
  if (!new RegExp(`^${mode}/snapshots/[A-Za-z0-9.-]+\\.json$`).test(key)) {
    throw new PullError(
      'Snapshot must be a viewer/snapshots/*.json or archive/snapshots/*.json key for the chosen mode.',
    );
  }
  return key;
}
function validateCatalog(value: AssetCatalog, archive: boolean) {
  if (
    value?.schema !== 'wander.shared/1' ||
    !value.files ||
    typeof value.files !== 'object' ||
    Array.isArray(value.files)
  ) {
    throw new PullError('Invalid shared manifest.');
  }
  const namespace = archive ? 'archive' : 'viewer';
  for (const [name, entry] of Object.entries(value.files)) {
    logicalPath(name, archive);
    if (
      !entry ||
      !/^[a-f0-9]{64}$/.test(entry.sha256) ||
      entry.key !== `${namespace}/blobs/${entry.sha256}` ||
      !Number.isSafeInteger(entry.size) ||
      entry.size < 0 ||
      typeof entry.contentType !== 'string'
    ) {
      throw new PullError('Invalid asset checksum, size, or blob key.');
    }
  }
}

export async function pullAssets(options: PullOptions, s3: ObjectReader, root = ROOT) {
  const mode = options.archive ? 'archive' : 'viewer';
  const out = path.resolve(root, options.out);
  const relativeOut = path.relative(root, out);
  if (
    !relativeOut.startsWith(`.context${path.sep}`) ||
    relativeOut.split(path.sep).some((part) => part === '..')
  ) {
    throw new PullError(
      '--out must be an explicit subdirectory of this checkout’s .context directory.',
    );
  }
  const exact = options.paths ?? [];
  const prefixes = options.prefixes ?? [];
  if (!options.list && exact.length + prefixes.length === 0)
    throw new PullError('Choose --path or --prefix; no files are downloaded by default.');
  for (const value of exact) logicalPath(value, !!options.archive);
  for (const value of prefixes) logicalPath(value, !!options.archive, true);
  await checkedDirectory(root, relativeOut);
  const receiptFile = path.join(out, receiptName);
  let receipt: Receipt | undefined;
  if (await regularFile(receiptFile)) {
    receipt = JSON.parse(await readFile(receiptFile, 'utf8')) as Receipt;
    if (
      receipt.schema !== 'wander.pull/1' ||
      receipt.bucket !== config.bucket ||
      receipt.mode !== mode ||
      !receipt.files ||
      typeof receipt.files !== 'object' ||
      Array.isArray(receipt.files) ||
      !/^[a-f0-9]{64}$/.test(receipt.manifestSha256)
    ) {
      throw new PullError(
        'Output receipt does not match this bucket/mode. Choose a new output directory.',
      );
    }
    if (options.snapshot && options.snapshot !== receipt.snapshot)
      throw new PullError('Output is pinned to another snapshot. Choose a new output directory.');
  }
  let snapshot = receipt?.snapshot ?? options.snapshot;
  if (!snapshot) {
    const pointer = (await getJSON<CatalogPointer>(s3, config.catalogKey)).value;
    snapshot = options.archive ? pointer.archive : pointer.snapshot;
  }
  if (!snapshot) throw new PullError('The current pointer has no requested snapshot.');
  checkedSnapshot(snapshot, mode);
  const catalog = (await getJSON<AssetCatalog>(s3, snapshot)).value;
  validateCatalog(catalog, !!options.archive);
  const manifestSha256 = createHash('sha256').update(JSON.stringify(catalog)).digest('hex');
  if (receipt && receipt.manifestSha256 !== manifestSha256)
    throw new PullError('Pinned manifest changed unexpectedly.');
  if (!receipt) {
    receipt = {
      schema: 'wander.pull/1',
      bucket: config.bucket,
      mode,
      snapshot,
      manifestSha256,
      files: {},
    };
    await saveReceipt(receiptFile, receipt, true, root);
  }
  for (const name of exact)
    if (!Object.hasOwn(catalog.files, name))
      throw new PullError(`Path is not in the pinned snapshot: ${name}`);
  const defaultList = options.list && exact.length + prefixes.length === 0;
  const selectionPrefixes = defaultList ? [options.archive ? 'runs/' : '/clips/'] : prefixes;
  const selected = Object.entries(catalog.files)
    .filter(([name]) => {
      if (exact.includes(name)) return true;
      if (!selectionPrefixes.some((prefix) => name.startsWith(prefix))) return false;
      const cinematic = name.startsWith('/clips/cinematic/');
      if (
        cinematic &&
        !options.includeCinematic &&
        !prefixes.some((prefix) => prefix.startsWith('/clips/cinematic/'))
      )
        return false;
      return !defaultList || options.archive || /\.(mp4|mov|webm|mkv)$/i.test(name);
    })
    .sort(([a], [b]) => a.localeCompare(b));
  if (!options.list && selected.length === 0)
    throw new PullError('No files match the requested prefixes.');
  const result: {
    path: string;
    size: number;
    sha256: string;
    status: 'listed' | 'downloaded' | 'verified';
  }[] = [];
  for (const [name, entry] of selected) {
    const row = { path: name, size: entry.size, sha256: entry.sha256 };
    if (options.list) {
      result.push({ ...row, status: 'listed' });
      continue;
    }
    const relative = logicalPath(name, !!options.archive);
    const parent = await checkedDirectory(
      root,
      path.relative(root, path.dirname(path.join(out, relative))),
    );
    const target = path.join(parent, path.basename(relative));
    const existing = await regularFile(target);
    if (existing && (await matches(target, entry))) {
      receipt.files[name] = entry;
      await saveReceipt(receiptFile, receipt, false, root);
      result.push({ ...row, status: 'verified' });
      continue;
    }
    if (existing && !options.overwrite)
      throw new PullError(
        `Local file differs: ${name}; use --overwrite only to replace it intentionally.`,
      );
    const response = await s3.send(new GetObjectCommand({ Bucket: config.bucket, Key: entry.key }));
    await checkedDirectory(root, path.relative(root, parent));
    const temp = path.join(parent, `.wander-pull-${randomUUID()}.part`);
    try {
      await pipeline(
        response.Body as NodeJS.ReadableStream,
        createWriteStream(temp, { flags: 'wx', mode: 0o600 }),
      );
      const info = await lstat(temp);
      if (info.size !== entry.size || (await hashFile(temp)) !== entry.sha256)
        throw new PullError(`Downloaded checksum/size mismatch: ${name}`);
      await checkedDirectory(root, path.relative(root, parent));
      await regularFile(target);
      if (options.overwrite) await rename(temp, target);
      else await link(temp, target);
      receipt.files[name] = entry;
      await saveReceipt(receiptFile, receipt, false, root);
      result.push({ ...row, status: 'downloaded' });
    } finally {
      await unlink(temp).catch(() => {});
    }
  }
  return { mode, snapshot, out, files: result };
}

const help = `Pull selected immutable S3 assets into an ignored local directory.
  bun scripts/pull-assets.ts --out .context/inputs --list
  bun scripts/pull-assets.ts --out .context/inputs --path /clips/elevator.mp4
  bun scripts/pull-assets.ts --out .context/recovered --archive --prefix runs/example/

--out DIR              Required subdirectory of this checkout's .context.
--list                 List only; defaults to source video paths under /clips/.
--path PATH            Exact manifest path; repeat to choose several files.
--prefix PREFIX        Choose a manifest prefix; repeatable, no implicit bulk download.
--include-cinematic    Include /clips/cinematic/ in broad prefix/list selections.
--archive              Read author archive using normal AWS credentials.
--snapshot KEY         Historical snapshot; otherwise pin latest on first use.
--overwrite            Replace differing local files after checksum verification.
--help                 Show this text without contacting S3.

Paths preserve their hierarchy below --out (e.g. clips/elevator.mp4).
.wander-pull.json pins the snapshot; use a new directory to select a newer snapshot.
Exact cinematic paths/prefixes explicitly opt in. Verified local files are reused.
Viewer mode uses WANDER_ASSET_* credentials when set, otherwise normal AWS credentials.
`;
if (import.meta.main) {
  let s3: ObjectReader | undefined;
  try {
    const { values } = parseArgs({
      options: {
        out: { type: 'string' },
        list: { type: 'boolean' },
        path: { type: 'string', multiple: true },
        prefix: { type: 'string', multiple: true },
        archive: { type: 'boolean' },
        snapshot: { type: 'string' },
        overwrite: { type: 'boolean' },
        'include-cinematic': { type: 'boolean' },
        help: { type: 'boolean' },
      },
    });
    if (values.help) console.log(help);
    else {
      if (!values.out) throw new PullError('--out is required. Use --help for examples.');
      s3 = client(values.archive ? {} : process.env);
      console.log(
        JSON.stringify(
          await pullAssets(
            {
              out: values.out,
              paths: values.path,
              prefixes: values.prefix,
              list: values.list,
              archive: values.archive,
              snapshot: values.snapshot,
              overwrite: values.overwrite,
              includeCinematic: values['include-cinematic'],
            },
            s3,
          ),
          null,
          2,
        ),
      );
    }
  } catch (error) {
    console.error(
      error instanceof PullError
        ? error.message
        : 'Asset pull failed; check credentials, network, and output permissions.',
    );
    process.exitCode = 1;
  } finally {
    s3?.destroy();
  }
}
