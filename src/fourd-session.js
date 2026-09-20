import * as THREE from 'three';
import { createFourD } from './fourd-runtime.js';
import { ALL } from './scene-catalog.ts';
import { createSceneSession } from './scene-session.js';

/** Keep one renderer and one native XRSession while replacing the recorded scene. */
export function startFourD(template) {
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  document.body.prepend(renderer.domElement);
  const session = createSceneSession({
    renderer,
    template,
    initial: new URLSearchParams(location.search),
    clips: ALL.map((clip) => ({
      id: clip.id,
      title: clip.title,
      sub: clip.sub,
      meta: clip.place.split(' · ').slice(1).join(' · '),
      src: clip.src,
      posterTime: clip.poster,
    })),
    createRuntime: createFourD,
  });
  window.addEventListener('pagehide', session.dispose, { once: true });
  void session.ready.catch(() => {});
  return session;
}
