import * as THREE from 'three';
import { createFourD } from './fourd-runtime.js';
import { ALL, SPARES } from './scene-catalog.ts';
import { SceneScope } from './scene-scope.js';

/** Keep one renderer and one native XRSession while replacing the recorded scene. */
export function startFourD(template) {
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  document.body.prepend(renderer.domElement);
  const initial = new URLSearchParams(location.search);
  const clips = ALL.map((clip) => ({
    id: clip.id,
    title: clip.title,
    sub: clip.sub,
    meta: clip.place.split(' · ').slice(1).join(' · '),
    src: clip.src,
    posterTime: clip.poster,
    spare: SPARES.some((spare) => spare.id === clip.id),
  }));
  let active = null;
  let cached = null;
  let pending = null;
  let generation = 0;
  const cacheState = { retained: [], hits: 0, misses: 0, lastLoad: null };
  window.__sceneCache = cacheState;
  const publishCache = () => {
    cacheState.retained = cached ? [cached.api.demo] : [];
  };
  const notify = (detail) =>
    window.dispatchEvent(new CustomEvent('wander:scenechange', { detail }));

  const optionsFor = (id) => {
    const options = new URLSearchParams({ demo: id });
    for (const [key, value] of initial) {
      if (
        ['walk', 'audio', 'personsize', 'dpr'].includes(key) ||
        key.startsWith('xr') ||
        key.startsWith('fake')
      )
        options.set(key, value);
    }
    return options;
  };

  async function load(search) {
    const id = search.get('demo');
    if (active?.api.demo === id && !pending) return;
    if (pending) throw new Error('A scene is already loading. Wait for it to finish.');
    const epoch = ++generation;
    const started = performance.now();
    const sorted = new URLSearchParams(search);
    sorted.sort();
    const key = sorted.toString();
    const reused = cached?.key === key ? cached : null;
    // A third scene evicts the spare BEFORE allocation: never retain more than the
    // active scene plus one cached or loading scene, even while switching in VR.
    if (!reused) cached?.scope.dispose();
    cached = null;
    publishCache();
    const scope = reused?.scope || new SceneScope();
    const root = reused?.root || document.createElement('div');
    if (!reused) {
      root.className = 'scene-root';
      root.append(template.content.cloneNode(true));
      scope.onDispose(() => {
        for (const video of root.querySelectorAll('video')) {
          video.pause();
          video.removeAttribute('src');
          video.load();
        }
        root.remove();
      });
    }
    root.hidden = !!active;
    document.body.append(root);
    pending = { id, scope, root, api: reused?.api || null };
    window.__loadingScene = pending;
    window.__wanderStartupError = false;
    const previous = active;
    previous?.api.play(false);
    notify({ id, loading: true });
    let timeout;
    try {
      const prepared = (async () => {
        const runtime = reused || (await createFourD({ search, renderer, root, scope }));
        scope.abort.signal.throwIfAborted();
        pending.api = runtime.api;
        if (!active) window.wander = runtime.api;
        await Promise.all(runtime.api.people.map((person) => person.allLoaded));
        scope.abort.signal.throwIfAborted();
        if (!runtime.api.ready || runtime.api.people.some((person) => person.loadError))
          throw new Error('The scene could not finish loading. Choose it again to retry.');
        return runtime;
      })();
      const runtime = await Promise.race([
        prepared,
        new Promise((_, reject) => {
          timeout = setTimeout(
            () => reject(new Error('Loading timed out. Check the connection and try again.')),
            180000,
          );
        }),
      ]);
      if (epoch !== generation) throw new Error('Scene load cancelled');
      clearTimeout(timeout);
      const previousAudio = previous?.api.audioState;
      const wasMuted = previousAudio?.muted ?? true;
      previous?.deactivate();
      previous?.root.remove();
      active = { ...runtime, root, scope, key };
      window.wander = runtime.api;
      runtime.api.loadScene = select;
      window.__stages = runtime.stages;
      root.hidden = false;
      if (reused) runtime.api.setTime(0);
      // Sound permission belongs to this viewing session, including fresh audio graphs.
      // resume() may wait for another browser gesture, so it must not hold scene activation.
      // unlock() initially unmutes synchronously; restore the user's explicit choice below.
      if (previousAudio?.unlocked)
        void runtime.api
          .unlockAudio()
          .catch((error) => console.warn('Audio resume failed:', error));
      runtime.api.setMuted(wasMuted);
      await runtime.activate({ clips, currentId: id, select });
      scope.abort.signal.throwIfAborted();
      cached = previous;
      publishCache();
      pending = null;
      window.__loadingScene = null;
      if (reused) cacheState.hits++;
      else cacheState.misses++;
      cacheState.lastLoad = {
        id,
        reused: !!reused,
        milliseconds: Math.round(performance.now() - started),
      };
      const url = new URL(location.href);
      url.search = search.toString();
      history.replaceState(null, '', url);
      notify({ id, loading: false });
    } catch (error) {
      clearTimeout(timeout);
      scope.dispose();
      if (epoch === generation) {
        // Preparation failures leave the previous scene running. If activation failed
        // after parking it, restore its live renderer/XR hooks before showing the error.
        if (active?.scope === scope) {
          active = previous;
          if (previous) {
            document.body.append(previous.root);
            previous.root.hidden = false;
            window.wander = previous.api;
            window.__stages = previous.stages;
            await previous.activate({ clips, currentId: previous.api.demo, select });
            previous.api.play(false);
          }
        }
        pending = null;
        window.__loadingScene = null;
        if (!active) {
          window.__wanderStartupError = true;
          const message = document.createElement('p');
          message.textContent = error.message;
          document.body.append(message);
        }
        notify({
          id: active?.api.demo || id,
          requestedId: id,
          loading: false,
          error: error.message,
        });
      }
      throw error;
    }
  }

  async function select(id) {
    if (!ALL.some((clip) => clip.id === id)) throw new Error('Unknown scene');
    await load(optionsFor(id));
  }

  window.addEventListener(
    'pagehide',
    () => {
      generation++;
      pending?.scope.dispose();
      cached?.scope.dispose();
      active?.scope.dispose();
      cached = null;
      publishCache();
    },
    { once: true },
  );
  void load(initial).catch(() => {});
}
