import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { worldPreviewAssetUrl } from './world-preview-path';

const status = document.querySelector<HTMLElement>('#status')!;
const reset = document.querySelector<HTMLButtonElement>('#reset')!;
const overview = document.querySelector<HTMLButtonElement>('#overview')!;
let renderer: THREE.WebGLRenderer | undefined;
let controls: OrbitControls | undefined;
let world: SplatMesh | undefined;
let spark: SparkRenderer | undefined;
let disposed = false;
const abort = new AbortController();

function dispose() {
  if (disposed) return;
  disposed = true;
  reset.disabled = true;
  overview.disabled = true;
  abort.abort();
  renderer?.setAnimationLoop(null);
  controls?.dispose();
  world?.dispose();
  spark?.dispose();
  renderer?.dispose();
  renderer?.domElement.remove();
}
window.addEventListener('pagehide', dispose, { once: true });

async function start() {
  const worldPath = new URLSearchParams(location.search).get('world');
  const url = worldPreviewAssetUrl(worldPath);
  const response = await fetch(url, { signal: abort.signal });
  if (!response.ok)
    throw new Error('The local SPZ is unavailable. Wait for generation to finish and reload.');
  const bytes = await response.arrayBuffer();
  if (disposed) return;

  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setSize(innerWidth, innerHeight);
  document.querySelector('#viewport')!.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#16171a');
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 10000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  spark = new SparkRenderer({ renderer, maxStdDev: Math.sqrt(5) });
  scene.add(spark);
  world = new SplatMesh({ fileBytes: bytes, fileName: worldPath!, lod: false });
  // Match fourd's Marble/OpenCV Y-down to OpenGL Y-up convention. No fitted scale is invented.
  world.rotation.x = Math.PI;
  scene.add(world);
  await world.initialized;
  if (disposed) return;
  world.updateMatrixWorld(true);
  const box = world.getBoundingBox().applyMatrix4(world.matrixWorld);
  if (box.isEmpty() || ![...box.min, ...box.max].every(Number.isFinite) || !world.numSplats) {
    throw new Error('The SPZ contains no finite visible world bounds.');
  }
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.1);
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  const home = center.clone().add(new THREE.Vector3(0, radius * 0.15, distance));
  const resetView = (fitWorld = false) => {
    // Marble's generation origin faces -Z after the X flip, matching fourd's convention.
    // This is a navigation starting point, not a measured source-camera registration.
    controls!.target.copy(fitWorld ? center : new THREE.Vector3(0, 0, -Math.max(radius / 10, 0.1)));
    camera.position.copy(fitWorld ? home : new THREE.Vector3());
    camera.near = Math.max(radius / 10000, 0.001);
    camera.far = Math.max(distance * 100, 100);
    camera.updateProjectionMatrix();
    controls!.update();
  };
  resetView();
  reset.addEventListener('click', () => resetView(), { signal: abort.signal });
  overview.addEventListener('click', () => resetView(true), { signal: abort.signal });
  reset.disabled = false;
  overview.disabled = false;
  status.textContent = `Loaded ${worldPath} · ${world.numSplats.toLocaleString()} splats · preview only`;
  document.body.dataset.state = 'ready';
  const resize = () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer!.setSize(innerWidth, innerHeight);
  };
  window.addEventListener('resize', resize, { signal: abort.signal });
  renderer.setAnimationLoop(() => {
    try {
      controls!.update();
      renderer!.render(scene, camera);
    } catch {
      showError();
    }
  });
}

function showError() {
  if (disposed) return;
  dispose();
  document.body.dataset.state = 'error';
  status.textContent =
    'Could not load the local SPZ. Check the world path, completed file, and WebGL support, then reload.';
}

void start().catch(showError);
