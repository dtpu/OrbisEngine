'use client';

import { useEffect, useRef, useState } from 'react';
import { formatBytes } from '@/lib/format';

/**
 * A polygon model you can orbit: `.glb`, `.gltf`, `.obj`, `.stl`, `.fbx`. The pipeline does not
 * write these yet -- its 3D is Gaussian, handled by `SplatBlock` -- but reference models an
 * operator attaches to a stage arrive here instead of as an undifferentiated download.
 *
 * Models come in unlit and in whatever units their author used, so the block supplies its own
 * three-point-ish lighting and frames the camera on the measured bounding box.
 */

type Phase =
  | { at: 'loading' }
  | { at: 'ready'; count: number; unit: string }
  | { at: 'failed'; reason: string };

export function MeshBlock({
  url,
  extension,
  size,
  points = false,
}: {
  url: string;
  extension: string;
  size: number;
  /** A PLY with no faces: draw its vertices, since a surface would be invented. */
  points?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>({ at: 'loading' });

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    let disposed = false;
    const cleanups: Array<() => void> = [];
    setPhase({ at: 'loading' });

    void (async () => {
      try {
        const [THREE, controlsModule] = await Promise.all([
          import('three'),
          import('three/examples/jsm/controls/OrbitControls.js'),
        ]);
        if (disposed) return;

        const object = await loadModel(THREE, extension, url, points);
        if (disposed) return;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
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
        scene.add(new THREE.HemisphereLight(0xffffff, 0x404048, 2.2));
        const key = new THREE.DirectionalLight(0xffffff, 1.6);
        key.position.set(1, 2, 1.5);
        scene.add(key);
        scene.add(object);
        cleanups.push(() => disposeTree(object));

        const camera = new THREE.PerspectiveCamera(
          45,
          element.clientWidth / Math.max(1, element.clientHeight),
          0.01,
          1000,
        );
        const controls = new controlsModule.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        cleanups.push(() => controls.dispose());

        const box = new THREE.Box3().setFromObject(object);
        const centre = box.getCenter(new THREE.Vector3());
        const span = Math.max(0.01, box.getSize(new THREE.Vector3()).length());
        camera.position.copy(centre).add(new THREE.Vector3(span * 0.5, span * 0.35, span * 0.8));
        camera.near = span / 200;
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

        setPhase({ at: 'ready', ...measure(object, points) });
      } catch (cause) {
        if (!disposed) setPhase({ at: 'failed', reason: (cause as Error).message });
      }
    })();

    return () => {
      disposed = true;
      for (const cleanup of cleanups.reverse()) cleanup();
    };
  }, [url, extension, size, points]);

  return (
    <div className="block3d" ref={host}>
      <p className="block3d__status" role="status">
        {phase.at === 'loading' ? `Loading ${formatBytes(size)}…` : null}
        {phase.at === 'ready'
          ? `${phase.count.toLocaleString()} ${phase.unit} — drag to orbit`
          : null}
        {phase.at === 'failed' ? `Could not read this file: ${phase.reason}` : null}
      </p>
    </div>
  );
}

/** Geometry-only formats carry no material, so they get one the surround can light. */
async function loadModel(
  THREE: typeof import('three'),
  extension: string,
  url: string,
  points: boolean,
): Promise<import('three').Object3D> {
  if (extension === 'ply') {
    const { PLYLoader } = await import('three/examples/jsm/loaders/PLYLoader.js');
    const geometry = await new PLYLoader().loadAsync(url);
    const coloured = Boolean(geometry.attributes.color);
    if (points) {
      // Point size is in world units, so it is scaled from the cloud's own extent.
      geometry.computeBoundingSphere();
      const radius = geometry.boundingSphere?.radius ?? 1;
      return new THREE.Points(
        geometry,
        new THREE.PointsMaterial({
          size: Math.max(radius / 400, 1e-4),
          vertexColors: coloured,
          color: coloured ? 0xffffff : 0xbfbdb6,
        }),
      );
    }
    geometry.computeVertexNormals();
    return new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        vertexColors: coloured,
        color: coloured ? 0xffffff : 0xbfbdb6,
        roughness: 0.8,
        metalness: 0,
      }),
    );
  }
  if (extension === 'glb' || extension === 'gltf') {
    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
    return (await new GLTFLoader().loadAsync(url)).scene;
  }
  if (extension === 'obj') {
    const { OBJLoader } = await import('three/examples/jsm/loaders/OBJLoader.js');
    return await new OBJLoader().loadAsync(url);
  }
  if (extension === 'fbx') {
    const { FBXLoader } = await import('three/examples/jsm/loaders/FBXLoader.js');
    return await new FBXLoader().loadAsync(url);
  }
  if (extension === 'stl') {
    const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js');
    const geometry = await new STLLoader().loadAsync(url);
    geometry.computeVertexNormals();
    return new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({ color: 0xbfbdb6, roughness: 0.75, metalness: 0 }),
    );
  }
  throw new Error(`no loader for .${extension}`);
}

/** What the status line counts: vertices for a cloud, triangles for a surface. */
function measure(
  object: import('three').Object3D,
  points: boolean,
): { count: number; unit: string } {
  let total = 0;
  object.traverse((child) => {
    const geometry = (child as import('three').Mesh).geometry;
    const position = geometry?.attributes?.position;
    if (!position) return;
    total += points ? position.count : (geometry.index?.count ?? position.count) / 3;
  });
  return { count: Math.round(total), unit: points ? 'points' : 'triangles' };
}

function disposeTree(object: import('three').Object3D): void {
  object.traverse((child) => {
    const mesh = child as import('three').Mesh;
    mesh.geometry?.dispose?.();
    for (const material of [mesh.material].flat()) material?.dispose?.();
  });
}
