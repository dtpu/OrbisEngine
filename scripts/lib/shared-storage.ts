import { readFile, readdir, stat, realpath } from 'node:fs/promises';
import path from 'node:path';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
export interface StorageConfig {
  bucket: string;
  region: string;
  catalogKey: string;
}
export type StorageEnvironment = Record<string, string | undefined>;
export interface AssetEntry {
  key: string;
  size: number;
  sha256: string;
  contentType: string;
}
export interface AssetCatalog {
  schema: string;
  createdAt?: string;
  files: Record<string, AssetEntry>;
}
export interface CatalogPointer {
  snapshot: string;
  archive?: string;
  createdAt?: string;
}
// A narrow read interface keeps the proxy testable without an AWS client implementation.
export interface ObjectReader {
  send(command: GetObjectCommand): Promise<{ Body?: unknown; ETag?: string }>;
  destroy(): void;
}
export interface ScannedFile {
  file: string;
  path: string;
  size: number;
  mtimeMs: number;
}
export const ROOT = path.resolve(import.meta.dirname, '../..');
export const config: StorageConfig = JSON.parse(
  await readFile(path.join(ROOT, 'wander-storage.json'), 'utf8'),
);
export const mime = (p: string): string =>
  (
    ({
      '.json': 'application/json',
      '.mp4': 'video/mp4',
      '.webm': 'video/webm',
      '.mov': 'video/quicktime',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.svg': 'image/svg+xml',
      '.wasm': 'application/wasm',
      '.mjs': 'text/javascript',
      '.js': 'text/javascript',
      '.txt': 'text/plain',
      '.html': 'text/html',
      '.css': 'text/css',
      '.wav': 'audio/wav',
      '.mp3': 'audio/mpeg',
    }) as Record<string, string>
  )[path.extname(p).toLowerCase()] || 'application/octet-stream';
export function client(env: StorageEnvironment = {}) {
  const id = env.WANDER_ASSET_ACCESS_KEY_ID,
    secret = env.WANDER_ASSET_SECRET_ACCESS_KEY;
  if (!!id !== !!secret)
    throw new Error('Set both WANDER_ASSET_ACCESS_KEY_ID and WANDER_ASSET_SECRET_ACCESS_KEY');
  return new S3Client({
    region: config.region,
    ...(id
      ? {
          credentials: {
            accessKeyId: id,
            secretAccessKey: secret!,
            ...(env.WANDER_ASSET_SESSION_TOKEN
              ? { sessionToken: env.WANDER_ASSET_SESSION_TOKEN }
              : {}),
          },
        }
      : {}),
  });
}
export async function getJSON<T = unknown>(s3: ObjectReader, key: string) {
  const r = await s3.send(new GetObjectCommand({ Bucket: config.bucket, Key: key }));
  return {
    value: JSON.parse(
      await (r.Body as { transformToString(): Promise<string> }).transformToString(),
    ) as T,
    etag: r.ETag,
  };
}
export async function hashFile(file: string) {
  const h = createHash('sha256');
  for await (const chunk of createReadStream(file)) h.update(chunk);
  return h.digest('hex');
}
export function forbidden(name: string) {
  return (
    /(^|\/)(\.env[^/]*|\.aws|\.ssh|node_modules|\.git|__pycache__|\.venv[^/]*|credentials|[^/]*\.(pem|key|p12|pfx))($|\/)/i.test(
      name,
    ) || /(^|\/)[^/]*(secret|credential)[^/]*$/i.test(name)
  );
}
export function containsSecret(text: string) {
  return /(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|\bhf_[A-Za-z0-9]{24,}|(?:aws_secret_access_key|OPENAI_API_KEY|MARBLE_API_KEY|MODAL_TOKEN_SECRET|WLT_API_KEY|ANTHROPIC_API_KEY)["']?\s*[=:]\s*["']?[A-Za-z0-9/+_-]{20,}/i.test(
    text,
  );
}
export async function filesUnder(dir: string, prefix = '') {
  const files: ScannedFile[] = [],
    excluded: string[] = [];
  async function walk(base: string, rel: string, ancestors = new Set<string>()) {
    let resolved;
    try {
      resolved = await realpath(base);
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return;
      throw e;
    }
    if (ancestors.has(resolved)) {
      excluded.push(rel);
      return;
    }
    ancestors = new Set([...ancestors, resolved]);
    let entries;
    try {
      entries = await readdir(base, { withFileTypes: true });
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return;
      throw e;
    }
    for (const e of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      const name = rel ? `${rel}/${e.name}` : e.name,
        full = path.join(base, e.name);
      if (forbidden(name) || e.name === '.DS_Store') {
        excluded.push(name);
        continue;
      }
      if (e.isDirectory()) await walk(full, name, ancestors);
      else if (e.isSymbolicLink() && prefix === 'runs/') {
        // Pipeline reuse links point to sibling stage outputs within .context.
        const target = await realpath(full);
        if (
          target.startsWith(path.join(ROOT, '.context') + path.sep) &&
          (await stat(full)).isDirectory()
        )
          await walk(full, name, ancestors);
        else excluded.push(name);
      }
      // The demo reel is intentionally a symlink to the locally recorded video.
      else if (e.isFile() || (e.isSymbolicLink() && name === 'reel-65s.mp4' && prefix === '/')) {
        const s = await stat(full);
        if (/\.(json|txt|log|md|yaml|yml|csv|html|js|mjs|ts|tsx|mts|cts|py|sh)$/i.test(name)) {
          if (containsSecret(await readFile(full, 'utf8'))) {
            excluded.push(name);
            continue;
          }
        }
        files.push({ file: full, path: prefix + name, size: s.size, mtimeMs: s.mtimeMs });
      } else if (e.isSymbolicLink()) excluded.push(name);
    }
  }
  await walk(dir, '');
  return { files, excluded };
}
