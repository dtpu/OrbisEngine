import type { PackedSplats } from '@sparkjsdev/spark';
import { decodePreparedWorld } from './prepared-world';

declare const __WANDER_SPARK_BUILD_ID__: string;

/** Optional local acceleration. Original published assets remain the fallback and authority. */
export async function loadPreparedWorld(
  url: string,
  fetchAsset: typeof fetch,
): Promise<PackedSplats | null> {
  try {
    const source = await fetchAsset(url, { method: 'HEAD' });
    if (!source.ok) return null;
    const hash = /^"([a-f0-9]{64})"$/.exec(source.headers.get('etag') || '')?.[1];
    if (!hash) return null;
    const response = await fetchAsset(
      `/api/prepared-world/v1/${__WANDER_SPARK_BUILD_ID__}/${hash}.bin`,
    );
    if (!response.ok) return null;
    return await decodePreparedWorld(await response.arrayBuffer(), {
      sourceHash: hash,
      buildId: __WANDER_SPARK_BUILD_ID__,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    console.warn('Prepared world unavailable; loading the original scene.');
    return null;
  }
}
