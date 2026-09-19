// A synthetic immersive-vr session, injected before page scripts by scripts/smoke-xr.ts.
//
// There is no headset on this machine and Chrome's own navigator.xr has no device, so the only way
// to run the REAL code path - three's WebXRManager, the ArrayCamera, two eye viewports, one Spark
// sort for both, fourd-xr's rig and vignette - is to hand the page a session that behaves like a
// Quest 3's. This is that session. It renders into the default framebuffer at a Quest 3's
// dimensions, so the frame times it produces are the true cost of the stereo frame on THIS GPU.
//
// It is a test fixture. It is never loaded by fourd.html and is not shipped.
//
//   ?fakexr=1            install it
//   ?fakew= ?fakeh=      per-eye resolution (default 2064x2208, Quest 3 native)
//   ?fakevfov=           vertical field of view in degrees (default 96, Quest 3's)
//
// window.__fakeXR is the driver: .head {x,y,z,yaw} in METRES of reference space, .axes.left/.right
// as xr-standard thumbsticks, .frames the frame times the session has served.
(() => {
  const P = new URLSearchParams(location.search);
  if (P.get('fakexr') !== '1') return;
  const n = (k, d) => (P.get(k) == null || isNaN(+P.get(k)) ? d : +P.get(k));
  const EW = n('fakew', 2064),
    EH = n('fakeh', 2208),
    VFOV = n('fakevfov', 96),
    IPD = 0.063;

  const drv = {
    head: { x: 0, y: 1.6, z: 0, yaw: 0 },
    axes: { left: [0, 0, 0, 0], right: [0, 0, 0, 0] },
    frames: [],
    presenting: false,
  };
  window.__fakeXR = drv;

  // Column-major 4x4 for a yaw-only pose, which is all a scripted head needs.
  const pose = (x, y, z, yaw) => {
    const c = Math.cos(yaw),
      s = Math.sin(yaw);
    return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]);
  };
  const mul = (a, b) => {
    // a * b, both column-major
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++) {
        let v = 0;
        for (let k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = v;
      }
    return o;
  };
  const proj = (vfov, aspect, near, far) => {
    const f = 1 / Math.tan((vfov * Math.PI) / 360);
    return new Float32Array([
      f / aspect,
      0,
      0,
      0,
      0,
      f,
      0,
      0,
      0,
      0,
      (far + near) / (near - far),
      -1,
      0,
      0,
      (2 * far * near) / (near - far),
      0,
    ]);
  };

  class Emitter {
    constructor() {
      this._l = {};
    }
    addEventListener(t, f) {
      (this._l[t] ??= []).push(f);
    }
    removeEventListener(t, f) {
      this._l[t] = (this._l[t] ?? []).filter((x) => x !== f);
    }
    dispatchEvent(e) {
      for (const f of this._l[e.type] ?? []) f.call(this, e);
    }
  }

  class FakeSpace {}
  class FakeRefSpace extends Emitter {
    getOffsetReferenceSpace() {
      return this;
    }
  }

  class FakeLayer {
    constructor(session, gl, init = {}) {
      const s = init.framebufferScaleFactor ?? 1;
      this.framebufferWidth = Math.round(EW * 2 * s);
      this.framebufferHeight = Math.round(EH * s);
      this.framebuffer = null; // the default framebuffer: real pixels, real cost
      this.ignoreDepthValues = false;
      this.fixedFoveation = 0;
      this._eyeW = Math.round(EW * s);
      this._eyeH = Math.round(EH * s);
      gl.canvas.width = this.framebufferWidth;
      gl.canvas.height = this.framebufferHeight;
    }
    getViewport(view) {
      return {
        x: view.eye === 'left' ? 0 : this._eyeW,
        y: 0,
        width: this._eyeW,
        height: this._eyeH,
      };
    }
  }
  window.XRWebGLLayer = FakeLayer;
  delete window.XRWebGLBinding; // forces three down the baseLayer path, as a Quest 3 does

  class FakeFrame {
    constructor(session) {
      this.session = session;
    }
    getViewerPose() {
      const h = drv.head;
      const base = pose(h.x, h.y, h.z, h.yaw);
      const pm = proj(
        VFOV,
        EW / EH,
        this.session.renderState.depthNear,
        this.session.renderState.depthFar,
      );
      return {
        transform: { matrix: base },
        views: [-1, 1].map((sgn) => ({
          eye: sgn < 0 ? 'left' : 'right',
          transform: { matrix: mul(base, pose((sgn * IPD) / 2, 0, 0, 0)) },
          projectionMatrix: pm,
        })),
      };
    }
    getPose(space) {
      return {
        transform: { matrix: space._matrix ?? pose(0, 1.2, -0.2, 0) },
        emulatedPosition: false,
      };
    }
    getDepthInformation() {
      return null;
    }
  }

  const source = (handedness, dx) => {
    const targetRaySpace = new FakeSpace();
    // Held out in front and tilted down 35 deg, so a teleport ray meets the floor.
    const p = Math.cos(-0.61),
      q = Math.sin(-0.61);
    targetRaySpace._matrix = new Float32Array([
      1,
      0,
      0,
      0,
      0,
      p,
      q,
      0,
      0,
      -q,
      p,
      0,
      dx,
      1.1,
      -0.25,
      1,
    ]);
    return {
      handedness,
      targetRayMode: 'tracked-pointer',
      targetRaySpace,
      gripSpace: targetRaySpace,
      profiles: ['oculus-touch-v3'],
      hand: undefined,
      gamepad: {
        get axes() {
          return drv.axes[handedness] ?? [0, 0, 0, 0];
        },
        buttons: [],
        mapping: 'xr-standard',
        connected: true,
      },
    };
  };

  class FakeSession extends Emitter {
    constructor(mode, init) {
      super();
      this.mode = mode;
      this.environmentBlendMode = 'opaque';
      this.visibilityState = 'visible';
      this.enabledFeatures = (init?.optionalFeatures ?? []).filter((f) => f === 'local-floor');
      this.renderState = { baseLayer: null, depthNear: 0.1, depthFar: 1000, layers: undefined };
      this.inputSources = [source('left', -0.2), source('right', 0.2)];
      this._cbs = [];
      this._raf = 0;
      this._ended = false;
      this._last = 0;
    }
    updateRenderState(s) {
      Object.assign(this.renderState, s);
    }
    async requestReferenceSpace() {
      return new FakeRefSpace();
    }
    requestAnimationFrame(cb) {
      this._cbs.push(cb);
      if (!this._raf) this._raf = requestAnimationFrame((t) => this._pump(t));
      return this._cbs.length;
    }
    cancelAnimationFrame() {}
    _pump(t) {
      this._raf = 0;
      if (this._ended) return;
      if (this._last) drv.frames.push(t - this._last);
      this._last = t;
      const cbs = this._cbs;
      this._cbs = [];
      for (const cb of cbs) cb(t, new FakeFrame(this));
    }
    async end() {
      this._ended = true;
      drv.presenting = false;
      this.dispatchEvent({ type: 'end', session: this });
    }
  }

  const xr = new Emitter();
  xr.isSessionSupported = async (mode) => mode === 'immersive-vr';
  xr.requestSession = async (mode, init) => {
    drv.presenting = true;
    drv.frames.length = 0;
    return new FakeSession(mode, init);
  };
  Object.defineProperty(navigator, 'xr', { value: xr, configurable: true });
})();
