import { SceneScope } from './scene-scope.js';

/** Transactional scene ownership, separate from WebGL and asset preparation. */
export function createSceneSession({ renderer, template, initial, clips, createRuntime }) {
  let active = null;
  let cached = null;
  let pending = null;
  let generation = 0;
  let closed = false;
  let transitions = Promise.resolve();
  const cacheState = { retained: [], hits: 0, misses: 0, lastLoad: null };
  window.__sceneCache = cacheState;
  const publishCache = () => {
    cacheState.retained = cached ? [cached.api.demo] : [];
  };
  const notify = (detail) =>
    window.dispatchEvent(new CustomEvent('wander:scenechange', { detail }));
  const publishActive = () => {
    window.wander = active?.api;
    window.__stages = active?.stages;
    if (active) active.api.loadScene = select;
  };
  const navigation = (id) => ({ clips, currentId: id, select });
  const transition = (work) => {
    const result = transitions.then(work);
    transitions = result.catch(() => {});
    return result;
  };

  const optionsFor = (id) => {
    const options = new URLSearchParams({ demo: id });
    for (const [key, value] of initial)
      if (
        ['walk', 'audio', 'personsize', 'dpr', 'interact', 'interactObject'].includes(key) ||
        key.startsWith('xr') ||
        key.startsWith('fake')
      )
        options.set(key, value);
    return options;
  };

  async function load(search, options = {}) {
    if (closed) return;
    const id = search.get('demo');
    if (pending?.id === id) return pending.done;
    const previousPlaying =
      options.resumePlayback ?? pending?.previousPlaying ?? !!active?.api.playing;
    const epoch = ++generation;
    pending?.scope.dispose();
    pending = null;
    window.__loadingScene = null;
    if (active?.api.demo === id && !active.needsActivation) {
      await transition(async () => {});
      if (closed || epoch !== generation) return;
      // The cancelled activation may itself have failed to restore this scene.
      if (active.needsActivation) return load(search, { resumePlayback: previousPlaying });
      active.api.play(previousPlaying);
      notify({ id, loading: false, cancelled: true, resumePlayback: previousPlaying });
      return;
    }
    const started = performance.now();
    const sorted = new URLSearchParams(search);
    sorted.sort();
    const key = sorted.toString();
    const reused = cached?.key === key ? cached : null;
    // Free the spare/cancelled runtime before allocating its replacement.
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
    const request = { id, scope, root, api: reused?.api || null, previousPlaying, done: null };
    pending = request;
    window.__loadingScene = request;
    window.__wanderStartupError = false;
    document.getElementById('scene-load-error')?.remove();
    active?.api.play(false);
    notify({ id, loading: true });
    const assertCurrent = () => {
      scope.abort.signal.throwIfAborted();
      if (closed || epoch !== generation)
        throw new DOMException('Scene load cancelled', 'AbortError');
    };
    request.done = (async () => {
      let timeout;
      let moved = false;
      try {
        const prepared = (async () => {
          const runtime = reused || (await createRuntime({ search, renderer, root, scope }));
          assertCurrent();
          request.api = runtime.api;
          if (!active) window.wander = runtime.api;
          await Promise.all(runtime.api.people.map((person) => person.allLoaded));
          assertCurrent();
          if (!runtime.api.ready || runtime.api.people.some((person) => person.loadError))
            throw new Error('The scene could not finish loading. Choose it again to retry.');
          return runtime;
        })();
        const runtime = await Promise.race([
          prepared,
          new Promise((_, reject) => {
            scope.listen(scope.abort.signal, 'abort', () => reject(scope.abort.signal.reason), {
              once: true,
            });
            timeout = setTimeout(
              () => reject(new Error('Loading timed out. Choose another scene or retry.')),
              180000,
            );
          }),
        ]);
        // Activation and rollback alone are serialized. A stalled asset fetch is never
        // part of this queue, so choosing another clip can cancel it immediately.
        await transition(async () => {
          assertCurrent();
          const previous = active;
          const previousAudio = previous?.api.audioState;
          const wasMuted = previousAudio?.muted ?? true;
          previous?.deactivate();
          previous?.root.remove();
          root.hidden = false;
          moved = true;
          try {
            if (reused) runtime.api.setTime(0);
            // Audio resume may await a browser gesture. It must not block activation;
            // unlock initially unmutes, so restore the user's choice immediately after.
            if (previousAudio?.unlocked)
              void runtime.api
                .unlockAudio()
                .catch((error) => console.warn('Audio resume failed:', error));
            runtime.api.setMuted(wasMuted);
            await runtime.activate(navigation(id));
            assertCurrent();
            active = { ...runtime, root, scope, key, needsActivation: false };
            publishActive();
            cached = previous;
            publishCache();
          } catch (error) {
            scope.dispose();
            if (previous && !closed) {
              document.body.append(previous.root);
              previous.root.hidden = false;
              previous.needsActivation = true;
              publishActive();
              try {
                await previous.activate(navigation(previous.api.demo));
                previous.needsActivation = false;
                previous.api.play(epoch === generation && previousPlaying);
              } catch (restoreError) {
                previous.deactivate();
                throw new Error(
                  `${error.message} The previous scene could not resume; choose a clip to retry.`,
                  { cause: restoreError },
                );
              }
            }
            throw error;
          }
        });
        assertCurrent();
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
        pending = null;
        window.__loadingScene = null;
        notify({ id, loading: false });
      } catch (error) {
        scope.dispose();
        if (closed || epoch !== generation) return;
        if (!moved) active?.api.play(previousPlaying);
        publishActive();
        if (!active) {
          window.__wanderStartupError = true;
          const message = document.createElement('p');
          message.id = 'scene-load-error';
          message.setAttribute('role', 'alert');
          message.style.cssText =
            'position:fixed;inset:24px auto auto 24px;z-index:100;color:white;background:#211;padding:16px';
          message.textContent = error.message;
          document.body.append(message);
        }
        notify({
          id: active?.api.demo || id,
          requestedId: id,
          loading: false,
          error: error.message,
          resumePlayback: previousPlaying,
        });
        throw error;
      } finally {
        clearTimeout(timeout);
        // Includes a failed rollback: it must never strand subsequent selections.
        if (pending === request) {
          pending = null;
          window.__loadingScene = null;
        }
      }
    })();
    return request.done;
  }

  async function select(id, options) {
    if (!clips.some((clip) => clip.id === id)) throw new Error('Unknown scene');
    await load(optionsFor(id), options);
  }
  function dispose() {
    closed = true;
    generation++;
    pending?.scope.dispose();
    cached?.scope.dispose();
    active?.scope.dispose();
    pending = null;
    window.__loadingScene = null;
    cached = null;
    publishCache();
  }
  return { select, dispose, ready: load(initial) };
}
