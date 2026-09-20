'use client';

import { useEffect, useState } from 'react';
import { MeshBlock } from '@/components/blocks/mesh-block';
import { SplatBlock } from '@/components/blocks/splat-block';
import { plyFlavour, type PlyFlavour } from '@/lib/blocks';

/**
 * A `.ply` whose renderer is chosen from its header rather than its extension.
 *
 * The pipeline writes both kinds under the same name: `object.ply` is Gaussian, while the person
 * frames are plain `x y z red green blue` point clouds. Handing a point cloud to the splat
 * renderer fails, and handing Gaussians to the point renderer throws their shape away, so the
 * block reads the ASCII header first -- a couple of kilobytes over a range request.
 */

const HEADER_BYTES = 8192;

export function PlyBlock({ url, name, size }: { url: string; name: string; size: number }) {
  const [flavour, setFlavour] = useState<PlyFlavour | 'reading' | 'unreadable'>('reading');

  useEffect(() => {
    let cancelled = false;
    setFlavour('reading');
    fetch(url, { headers: { Range: `bytes=0-${HEADER_BYTES - 1}` }, cache: 'force-cache' })
      .then(async (response) => {
        if (!response.ok) throw new Error(`${response.status}`);
        const head = new TextDecoder().decode(await response.arrayBuffer());
        if (!cancelled) setFlavour(head.startsWith('ply') ? plyFlavour(head) : 'unreadable');
      })
      .catch(() => {
        // A server that will not serve a range still serves the whole file to the renderer;
        // Gaussian is the safer guess there, since it is what the shape stages write.
        if (!cancelled) setFlavour('gaussian');
      });
    return () => {
      cancelled = true;
    };
  }, [url]);

  if (flavour === 'reading') {
    return (
      <div className="block3d">
        <p className="block3d__status">Reading header…</p>
      </div>
    );
  }
  if (flavour === 'unreadable') {
    return <p className="viewer__note">This file does not start with a PLY header.</p>;
  }
  if (flavour === 'gaussian') return <SplatBlock url={url} name={name} size={size} />;
  return <MeshBlock url={url} extension="ply" size={size} points={flavour === 'points'} />;
}
