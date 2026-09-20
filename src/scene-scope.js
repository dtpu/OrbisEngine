// A scene owns its listeners, requests and GPU resources; the renderer/XR session outlive it.
export class SceneScope {
  active = false;
  disposed = false;
  abort = new AbortController();
  cleanups = [];

  onDispose(cleanup) {
    if (this.disposed) cleanup();
    else this.cleanups.push(cleanup);
  }

  listen(target, type, listener, options) {
    target.addEventListener(type, listener, options);
    this.onDispose(() => target.removeEventListener(type, listener, options));
  }

  listenActive(target, type, listener, options) {
    this.listen(
      target,
      type,
      (...args) => {
        if (this.active && !this.disposed) listener(...args);
      },
      options,
    );
  }

  fetch(url, options) {
    this.abort.signal.throwIfAborted();
    return fetch(url, { ...options, signal: this.abort.signal });
  }

  tracked(Base) {
    const scope = this;
    return class extends Base {
      constructor(...args) {
        scope.abort.signal.throwIfAborted();
        super(...args);
        let disposed = false;
        const dispose = this.dispose.bind(this);
        this.dispose = () => {
          if (disposed) return;
          disposed = true;
          dispose();
        };
        scope.onDispose(() => this.dispose());
      }
    };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.active = false;
    this.abort.abort();
    for (const cleanup of this.cleanups.splice(0).reverse()) {
      try {
        cleanup();
      } catch (error) {
        console.warn('Scene cleanup failed:', error);
      }
    }
  }
}

export function disposeScene(scene) {
  const released = new Set();
  const release = (resource) => {
    if (!resource?.dispose || released.has(resource)) return;
    released.add(resource);
    resource.dispose();
  };
  scene.traverse((object) => {
    release(object.geometry);
    for (const material of [object.material].flat()) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) release(value);
      release(material);
    }
  });
  scene.clear();
}
