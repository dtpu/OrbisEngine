'use client';

import { useEffect, useRef, useState } from 'react';
import { formatBytes } from '@/lib/format';

/**
 * A Gaussian splat artifact you can orbit: `.spz` worlds, and the `.ply` objects and person
 * frames the packager writes. Nothing here is loaded until the block is on screen, because a
 * world is tens of megabytes and every canvas costs a WebGL context.
 *
 * Worlds leave the pipeline Y-down, the same convention `fourd.html` corrects with
 * `world.rotation.x = Math.PI`; without the same flip every world previews upside down.
 */

type Phase =
  | { at: 'idle' }
  | { at: 'loading'; fraction: number | null }
  | { at: 'ready'; splats: number }
  | { at: 'failed'; reason: string };

export function SplatBlock({ url, name, size }: { url: string; name: string; size: number }) {
  const host = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>({ at: 'idle' });
  const [armed, setArmed] = useState(false);

  // A world is worth megabytes of transfer, so it waits until the operator looks at it.
  useEffect(() => {
    const element = host.current;
    if (!element || armed) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) setArmed(true);
      },
      { rootMargin: '200px' },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [armed]);

  useEffect(() => {
    const element = host.current;
    if (!element || !armed) return;

    let disposed = false;
    const cleanups: Array<() => void> = [];
    setPhase({ at: 'loading', fraction: null });

    void (async () => {
      try {
        const [THREE, spark, controlsModule] = await Promise.all([
          import('three'),
          import('@sparkjsdev/spark'),
          import('three/examples/jsm/controls/OrbitControls.js'),
        ]);
        if (disposed) return;

        const renderer = new THREE.WebGLRenderer({ antialias: false });
        renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
        renderer.setSize(element.clientWidth, element.clientHeight, false);
        renderer.domElement.className = 'block3d__canvas';
        element.append(renderer.domElement);
        cleanups.push(() => {
          renderer.domElement.remove();
          renderer.dispose();
          renderer.forceContextLoss();
        });

        const scene = new THREE.Scene();
        scene.add(new spark.SparkRenderer({ renderer }));
        const camera = new THREE.PerspectiveCamera(
          55,
          element.clientWidth / Math.max(1, element.clientHeight),
          0.02,
          2000,
        );
        const controls = new controlsModule.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        cleanups.push(() => controls.dispose());

        const response = await fetch(url, { cache: 'force-cache' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const bytes = await response.arrayBuffer();
        if (disposed) return;

        const mesh = new spark.SplatMesh({
          fileBytes: bytes,
          fileName: name,
          // Large worlds only stay interactive with level of detail turned on.
          lod: size > 32 * 1024 * 1024,
          onProgress: (event) =>
            setPhase({
              at: 'loading',
              fraction: event.lengthComputable ? event.loaded / event.total : null,
            }),
        });
        await mesh.initialized;
        if (disposed) {
          mesh.dispose();
          return;
        }
        mesh.rotation.x = Math.PI;
        scene.add(mesh);
        cleanups.push(() => mesh.dispose());

        mesh.updateMatrixWorld(true);
        const box = mesh.getBoundingBox().applyMatrix4(mesh.matrixWorld);
        const centre = box.getCenter(new THREE.Vector3());
        const span = Math.max(0.2, box.getSize(new THREE.Vector3()).length());
        camera.position.copy(centre).add(new THREE.Vector3(0, span * 0.12, span * 0.62));
        camera.near = span / 500;
        camera.far = span * 20;
        camera.updateProjectionMatrix();
        controls.target.copy(centre);
        controls.update();

        const resize = new ResizeObserver(() => {
          const width = element.clientWidth;
          const height = Math.max(1, element.clientHeight);
          renderer.setSize(width, height, false);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
        });
        resize.observe(element);
        cleanups.push(() => resize.disconnect());

        renderer.setAnimationLoop(() => {
          controls.update();
          renderer.render(scene, camera);
        });
        cleanups.push(() => renderer.setAnimationLoop(null));

        setPhase({ at: 'ready', splats: mesh.numSplats });
      } catch (cause) {
        if (!disposed) setPhase({ at: 'failed', reason: (cause as Error).message });
      }
    })();

    return () => {
      disposed = true;
      for (const cleanup of cleanups.reverse()) cleanup();
    };
  }, [armed, url, name, size]);

  return (
    <div className="block3d" ref={host}>
      <p className="block3d__status" role="status">
        {phase.at === 'idle' ? `${formatBytes(size)} — scroll into view to load` : null}
        {phase.at === 'loading'
          ? `Loading ${formatBytes(size)}${
              phase.fraction === null ? '…' : ` — ${Math.round(phase.fraction * 100)}%`
            }`
          : null}
        {phase.at === 'ready' ? `${phase.splats.toLocaleString()} splats — drag to orbit` : null}
        {phase.at === 'failed' ? `Could not read this file: ${phase.reason}` : null}
      </p>
    </div>
  );
}
