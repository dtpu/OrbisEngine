import { SplatMesh } from '@sparkjsdev/spark';
import { encodePreparedWorld } from '../src/prepared-world';

declare global {
  interface Window {
    prepareWorld: (
      source: string,
      sourceHash: string,
      buildId: string,
    ) => Promise<{ milliseconds: number; bytes: number; splats: number }>;
  }
}

window.prepareWorld = async (source, sourceHash, buildId) => {
  const start = performance.now();
  const mesh = new SplatMesh({ url: source, lod: true });
  try {
    await mesh.initialized;
    const packed = mesh.packedSplats;
    if (!packed?.lodSplats) throw new Error('The world did not produce a prepared LoD tree');
    const bytes = await encodePreparedWorld(packed, { sourceHash, buildId });
    const response = await fetch('/prepared', { method: 'POST', body: bytes });
    if (!response.ok) throw new Error('Could not save prepared world');
    return {
      milliseconds: Math.round(performance.now() - start),
      bytes: bytes.byteLength,
      splats: packed.lodSplats.numSplats,
    };
  } finally {
    mesh.dispose();
  }
};
