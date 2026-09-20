// WebXR mode for fourd.html. Opt-in, behind ?xr=1, and inert otherwise: fourd.html's only
// concession is a guarded dynamic import at the end of its module script, so with the flag absent
// nothing in this file is fetched, parsed or run and the desktop path is what it was.
//
// WHAT IS DIFFERENT IN A HEADSET, and why this is a module rather than a branch in the loop:
//
// 1. THE POSE IS NOT OURS. Outside XR fourd.html writes camera.position directly. Presenting,
//    three overwrites it from the headset every frame (WebXRManager.updateUserCamera), so the
//    desktop WASD/clamp/reel code keeps running and is simply discarded. Locomotion therefore has
//    to move something UNDER the camera: the rig below. The rig is the only thing this file moves.
//
// 2. HEAD MOTION IS NEVER DAMPED. The desktop soft boundary eases the camera to a stop at the edge
//    of the measured region. Doing that to someone's head is a direct route to sickness (BRAIN
//    section 6), so the same limits are enforced in two different ways: locomotion gets the soft
//    gain, and physical leaning gets a vignette without resistance. Optional physical walk gain
//    adds horizontal rig travel; it never changes the tracked pose. You can always lean out.
//
// 3. THE EDGE FADE IS DOM. fourd.html grades the edge with a CSS filter on the canvas and a fixed
//    overlay div. Neither exists inside a headset - the session renders to its own framebuffer -
//    so the same grading is redrawn here as a head-locked shell, in DEGREES off the forward axis
//    rather than in screen UV (Meta's own correction, and the form Fernandes & Feiner tested).
//
// 4. THE SCENE IS NOT IN METRES. WebXR reference space always is; these worlds are not (tos31 is
//    1 unit = 2.06 m). rig.scale = wander.upm maps the headset's metres onto the world's units, or
//    the user arrives three and a half metres tall.
//
// URL PARAMETERS (all no-ops without ?xr=1)
//   ?xr=1            enable; adds an Enter VR button when the browser reports immersive-vr
//   ?xrmove=         teleport (default) | smooth
//   ?xrwalkgain=     physical horizontal travel multiplier, 1..2 (default 1)
//   ?xrview=0        disable the local single-eye spectator preview
//   ?xrhands=0      hide the illustrative controller gloves and tracked hands
//   ?xrbody=0       hide the estimated first-person body
//   ?xrlod=          splat budget while presenting (default 500000, Spark's own WebXR figure)
//   ?xradapt=0       disable the closed loop that lowers that budget when frames run long
//   ?xrfps=          frame rate the loop holds the budget against (default 72)
//   ?xrfoveation=    three's fixed foveation, 0 none .. 1 max (default 1)
//   ?xrscale=        override world units per metre (default wander.upm)
//   ?xreye=          eye height in metres when the runtime grants no floor (default 1.6)
//   ?xrturn=         snap turn in degrees (default 30); 0 disables turning
//   ?xrturnmode=     smooth | snap (default follows xrmove; teleport uses snap)
//   ?xrturnspeed=    smooth turn speed in degrees/s at full deflection (default 90, 0..180)
//   ?xrspeed=        smooth locomotion speed in m/s (default 1.4, a walk)
//   ?xrpolyfill=1    force webxr-polyfill (phones, Cardboard); matches the rest of the repo
import * as THREE from 'three';
import { createAvatarBody } from './avatar-body';
import { createAvatarHands } from './avatar-hands';
import { PhysicalWalk, parseWalkGain } from './physical-walk';
import { raycastWalkFloor } from './teleport';
import { createQuestView } from './quest-view';
import { createSceneSidebar, type SceneSidebarClip } from './scene-sidebar';

type Wander = {
  spark: { lodSplatCount?: number };
  playing?: boolean;
  play?: (playing: boolean) => void;
  upm: number;
  floorY: number;
  clampOn: boolean;
  clampBox: THREE.Box3;
  pathPts: THREE.Vector3[];
  pathR: number;
  softMargin: number;
  overshoot: number;
  edgeAt: (p: THREE.Vector3) => number;
  params: Record<string, string>;
  walk?: {
    eye: number;
    floor: number;
    grid: { cell: number } | null;
    cellAt: (
      x: number,
      z: number,
      maximumSupport?: number,
    ) => { floor: number; inside: boolean | number };
    blockedAt: (x: number, z: number, floor?: number) => number;
    advance: (from: THREE.Vector3, delta: THREE.Vector3, dt: number) => THREE.Vector3;
  } | null;
  possess?: { head: THREE.Vector3; yaw: number } | null; // fourd.html ?possess=: the ridden person's head (world units) and yaw, updated every frame
};

export type XrInit = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  wander: Wander;
  q: URLSearchParams;
  sceneNavigation?: {
    clips: SceneSidebarClip[];
    currentId: string;
    select: (id: string, options?: { resumePlayback?: boolean }) => Promise<void>;
  };
};

export type XrRuntime = { dispose(): void };

const DEG = Math.PI / 180;
const MIN_BUDGET = 150_000;
const ADAPT_WINDOW = 90; // frames between budget decisions: ~1.25 s at 72 Hz

export async function initXR({
  renderer,
  scene,
  camera,
  wander,
  q,
  sceneNavigation,
}: XrInit): Promise<XrRuntime> {
  let disposed = false;
  let switching = false;
  let sidebarOpen = false;
  let resumePlayback = false;
  let errorResume: boolean | null = null;
  let teleportBlockedUntilRelease = renderer.xr.isPresenting;
  const num = (k: string, d: number) => {
    const v = q.get(k);
    return v == null || v === '' || isNaN(+v) ? d : +v;
  };

  // ---- the two bits of DOM -----------------------------------------------------------------------
  const btn = document.createElement('button');
  btn.id = 'xrBtn';
  btn.textContent = 'Enter VR';
  btn.style.cssText =
    'position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:20;' +
    'padding:10px 18px;border:1px solid #567;border-radius:6px;background:#000c;color:#dff;' +
    'font:14px system-ui;cursor:pointer;display:none';
  document.body.appendChild(btn);
  const hud = document.createElement('div');
  hud.id = 'xrHud';
  hud.style.cssText =
    'position:fixed;left:12px;bottom:12px;z-index:20;background:#000a;padding:6px 9px;' +
    'border-radius:6px;font:11px ui-monospace,monospace;color:#9cf;white-space:pre;display:none';
  document.body.appendChild(hud);
  const note = (msg: string) => {
    hud.style.display = '';
    hud.textContent = msg;
  };

  if (q.get('xrpolyfill') === '1' && !renderer.xr.isPresenting) {
    // How this gets TESTED without a headset: the polyfill grants a real immersive-vr session on a
    // desktop browser, so three's whole XR path, the rig, the anchor and the render wrapper run for
    // real. It declines to install where a navigator.xr already exists - desktop Chrome has one and
    // it has no device - so that one has to be removed first (same trick as src/main.ts). There are
    // no controllers in a Cardboard session, so it proves the session, not the locomotion.
    try {
      delete (Navigator.prototype as unknown as Record<string, unknown>).xr;
    } catch {
      /* non-configurable */
    }
    for (const k of Object.getOwnPropertyNames(window)) {
      if (/^XR[A-Z]/.test(k)) {
        try {
          delete (window as unknown as Record<string, unknown>)[k];
        } catch {
          /* keep */
        }
      }
    }
    const { default: WebXRPolyfill } = await import('webxr-polyfill');
    new WebXRPolyfill({ cardboard: true, allowCardboardOnDesktop: true });
  }
  const xr = navigator.xr;
  if (!xr) {
    note('xr=1: this browser has no navigator.xr');
    return {
      dispose() {
        btn.remove();
        hud.remove();
      },
    };
  }

  // ---- budget ------------------------------------------------------------------------------------
  // Spark only has a budget to spend if the LoD slice is running at all; the baked-colour and
  // observation-confidence paths need the plain packed array and turn it off (fourd.html: lodOn).
  const p = wander.params;
  const lodOn =
    p.lod === '1' || (p.lod !== '0' && !p.bakedweights && !(p.obs && +(p.obsfade ?? 0) > 0));
  const deskLod = wander.spark.lodSplatCount;
  const xrLod = Math.max(MIN_BUDGET, num('xrlod', 500_000));
  const adapt = q.get('xradapt') !== '0' && lodOn;
  const targetMs = 1000 / Math.max(30, num('xrfps', 72));
  let budget = xrLod;

  // ---- rig ---------------------------------------------------------------------------------------
  // The camera hangs off this while presenting and is handed back parentless on exit, which is how
  // fourd.html had it. Scale converts the headset's metres into this world's units.
  const upm = Math.max(1e-4, num('xrscale', wander.upm || 1));
  const mode = q.get('xrmove') === 'smooth' ? 'smooth' : 'teleport';
  const physicalWalk = new PhysicalWalk(parseWalkGain(q.get('xrwalkgain')));
  const requestedTurnMode = q.get('xrturnmode');
  const turnMode =
    requestedTurnMode === 'smooth' || requestedTurnMode === 'snap'
      ? requestedTurnMode
      : mode === 'smooth'
        ? 'smooth'
        : 'snap';
  const requestedTurnSpeed = num('xrturnspeed', 90);
  const turnSpeed = Number.isFinite(requestedTurnSpeed)
    ? THREE.MathUtils.clamp(requestedTurnSpeed, 0, 180)
    : 90;
  const report: Record<string, unknown> = {
    mode,
    walkGain: physicalWalk.gain,
    turnMode,
    turnSpeed,
    upm: +upm.toFixed(4),
    askedBudget: xrLod,
    adapt,
    targetMs: +targetMs.toFixed(2),
  };
  const publish = () => {
    (window as unknown as { __xr: unknown }).__xr = report;
  };
  publish();

  const rig = new THREE.Group();
  rig.name = 'xr-rig';
  rig.scale.setScalar(upm);
  scene.add(rig);

  const avatarHands = q.get('xrhands') === '0' ? null : createAvatarHands(renderer, rig);
  if (avatarHands) report.hands = avatarHands.state;
  const avatarBody = q.get('xrbody') === '0' ? null : createAvatarBody(renderer, rig);
  if (avatarBody) report.body = avatarBody.state;
  const questView =
    q.get('xrview') === '0' ? null : createQuestView(renderer, q.get('demo') || 'scene');
  if (questView) report.questView = questView.state;

  const vignette = buildVignette();
  const marker = buildMarker(upm);
  scene.add(marker.group); // world space: the rig is scaled and turns under it

  // ---- the boundary, in the two forms a headset needs ---------------------------------------------
  const box = wander.clampBox;
  const boxC = box.getCenter(new THREE.Vector3());
  const boxH = box
    .getSize(new THREE.Vector3())
    .multiplyScalar(0.5)
    .max(new THREE.Vector3(1e-4, 1e-4, 1e-4));
  const softLo = 1 - THREE.MathUtils.clamp(wander.softMargin ?? 0.25, 0, 0.95);
  const softHi = 1 + Math.max(0, wander.overshoot ?? 0.35);
  const softGain = (a: number) => {
    if (a <= softLo) return 1;
    const u = THREE.MathUtils.clamp((a - softLo) / Math.max(1e-6, softHi - softLo), 0, 1);
    return 1 - u * u * (3 - 2 * u);
  };
  const seg = new THREE.Vector3(),
    segBest = new THREE.Vector3(),
    segTmp = new THREE.Vector3();
  function nearestOnPath(v: THREE.Vector3) {
    let best = Infinity;
    for (let i = 0; i + 1 < wander.pathPts.length; i++) {
      const a = wander.pathPts[i],
        b = wander.pathPts[i + 1];
      const ab = seg.copy(b).sub(a);
      const t = THREE.MathUtils.clamp(
        segTmp.copy(v).sub(a).dot(ab) / Math.max(1e-9, ab.lengthSq()),
        0,
        1,
      );
      const c = ab.multiplyScalar(t).add(a),
        d = c.distanceTo(v);
      if (d < best) {
        best = d;
        segBest.copy(c);
      }
    }
    return best;
  }
  const onLeash = wander.pathPts.length >= 2 && wander.pathR > 0;
  /** Inside the measured region? Used to refuse a teleport, never to stop a head. */
  const reachable = (v: THREE.Vector3) =>
    !wander.clampOn || (box.containsPoint(v) && (!onLeash || nearestOnPath(v) <= wander.pathR));

  // ---- session -------------------------------------------------------------------------------------
  void xr
    .isSessionSupported('immersive-vr')
    .then((ok) => {
      if (disposed) return;
      if (ok) btn.style.display = '';
      else note('xr=1: WebXR is here but no immersive-vr device is');
    })
    .catch(() => {
      if (!disposed) note('xr=1: immersive-vr unsupported');
    });

  renderer.xr.enabled = true;
  renderer.xr.setFoveation(THREE.MathUtils.clamp(num('xrfoveation', 1), 0, 1));
  // Resolution is the bigger of the two levers and the one that cannot move inside a session: three
  // only reads it when the session is created. Measured at Quest 3 native per-eye, dropping to 0.8
  // (0.64x the pixels) bought more than halving the splat budget did (scripts/bench-xr-stereo.mjs),
  // so the default spends sharpness on frame rate and ?xrfbscale=1 hands it back.
  const fbScale = THREE.MathUtils.clamp(num('xrfbscale', 0.8), 0.3, 2);
  if (!renderer.xr.isPresenting) renderer.xr.setFramebufferScaleFactor(fbScale);
  report.fbScale = fbScale;

  let floorRef = renderer.xr.getSession()?.enabledFeatures?.includes('local-floor') ?? false;
  let groundFloor = wander.walk?.floor ?? wander.floorY;
  const onButtonClick = () =>
    void (async () => {
      if (disposed) return;
      const live = renderer.xr.getSession();
      if (live) {
        await live.end();
        return;
      }
      try {
        const session = await xr.requestSession('immersive-vr', {
          optionalFeatures:
            avatarHands || avatarBody ? ['local-floor', 'hand-tracking'] : ['local-floor'],
        });
        if (disposed) {
          // The scene may have changed while the browser permission dialog was open.
          await session.end();
          return;
        }
        floorRef = session.enabledFeatures?.includes('local-floor') ?? false;
        renderer.xr.setReferenceSpaceType(floorRef ? 'local-floor' : 'local');
        await renderer.xr.setSession(session);
      } catch (err) {
        if (!disposed)
          note('VR unavailable: ' + (err instanceof Error ? err.message : String(err)));
      }
    })();
  btn.addEventListener('click', onButtonClick);

  // Where the rig sits so the head lands on the desktop camera's pose. Sampled ONCE per session
  // (BRAIN section 6 rules out a continuous recentre; 0.2 Hz is the worst frequency there is).
  const homeParent = camera.parent;
  const home = camera.position.clone();
  const homeQ = camera.quaternion.clone();
  let homeYaw = yawOf(homeQ);
  let anchored = false;
  const frames: number[] = [];
  let lastFrameStart = 0;
  let lastAdapt = 0;
  let walkReferenceSpace: XRReferenceSpace | null = null;
  const resetPhysicalWalk = () => {
    physicalWalk.reset();
    avatarBody?.reset();
  };

  let sessionStartTimer = 0;
  const onSessionStart = () => {
    if (disposed) return;
    floorRef = renderer.xr.getSession()?.enabledFeatures?.includes('local-floor') ?? floorRef;
    btn.textContent = sceneNavigation ? 'Exit VR · B: scenes · A: select' : 'Exit VR';
    home.copy(camera.position);
    homeQ.copy(camera.quaternion);
    homeYaw = yawOf(homeQ);
    anchored = false;
    groundFloor = wander.walk?.floor ?? wander.floorY;
    report.groundFloor = groundFloor;
    report.teleportTarget = null;
    report.teleportValid = false;
    targetOk = false;
    physicalWalk.reset();
    walkReferenceSpace?.removeEventListener('reset', resetPhysicalWalk);
    walkReferenceSpace = renderer.xr.getReferenceSpace();
    walkReferenceSpace?.addEventListener('reset', resetPhysicalWalk);
    vel.set(0, 0, 0);
    snapLatch = false;
    aimingPrev = false;
    blink = 0;
    frames.length = 0;
    lastFrameStart = 0;
    lastAdapt = 0;
    budget = xrLod;
    if (lodOn) wander.spark.lodSplatCount = budget;
    else note('xr: LoD is off for this preset (baked/obs colours), so ?xrlod= cannot apply');
    rig.position.copy(home);
    rig.quaternion.identity();
    rig.add(camera);
    camera.add(vignette.mesh);
    for (const c of controllers) rig.add(c.grip);
    // The one number nobody can look up: what this device actually asked for, per eye. It is not
    // there on the first frame, so read it once the first viewer pose has landed.
    window.clearTimeout(sessionStartTimer);
    sessionStartTimer = window.setTimeout(() => {
      if (disposed || !renderer.xr.isPresenting) return;
      const vp = (renderer.xr.getCamera().cameras[0] as { viewport?: THREE.Vector4 } | undefined)
        ?.viewport;
      const layer = renderer.xr.getSession()?.renderState.baseLayer;
      report.perEye = vp ? `${vp.z}x${vp.w}` : 'unknown';
      report.framebuffer = layer
        ? `${layer.framebufferWidth}x${layer.framebufferHeight}`
        : 'unknown';
      report.floorRef = floorRef;
      publish();
      note(summary());
    }, 800);
  };

  const onSessionEnd = () => {
    window.clearTimeout(sessionStartTimer);
    resumePlayback = false;
    sidebar?.close();
    btn.textContent = 'Enter VR';
    walkReferenceSpace?.removeEventListener('reset', resetPhysicalWalk);
    walkReferenceSpace = null;
    physicalWalk.reset();
    rig.remove(camera);
    homeParent?.add(camera);
    camera.remove(vignette.mesh);
    for (const c of controllers) {
      rig.remove(c.grip);
      c.aiming = false;
      c.triggerHeld = false;
      c.ray.visible = false;
    }
    marker.set(null, false);
    vignette.set(0);
    // Hand the pose back exactly where the desktop code left it.
    camera.position.copy(home);
    camera.quaternion.copy(homeQ);
    camera.updateMatrixWorld(true);
    if (lodOn) wander.spark.lodSplatCount = deskLod;
    renderer.setPixelRatio(Math.min(+(p.dpr ?? 1) || 1, devicePixelRatio));
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    finish();
    note(summary());
  };

  // ---- controllers -----------------------------------------------------------------------------------
  const controllers = [0, 1].map((i) => {
    const grip = renderer.xr.getController(i);
    const ray = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, -1),
      ]),
      new THREE.LineBasicMaterial({
        color: 0x66ccff,
        transparent: true,
        opacity: 0.6,
        depthTest: false,
      }),
    );
    ray.renderOrder = 999;
    ray.frustumCulled = false;
    ray.visible = false;
    grip.add(ray); // grip is in metres inside the scaled rig, so the line is in metres too
    const c = {
      grip,
      ray,
      aiming: false,
      triggerHeld: false,
      onSelectStart() {
        c.triggerHeld = true;
        c.aiming = mode === 'teleport' && !sidebarOpen && !teleportBlockedUntilRelease;
      },
      onSelectEnd() {
        c.triggerHeld = false;
        c.aiming = false;
      },
    };
    grip.addEventListener('selectstart', c.onSelectStart);
    grip.addEventListener('selectend', c.onSelectEnd);
    return c;
  });

  // ---- locomotion --------------------------------------------------------------------------------------
  const SNAP = num('xrturn', 30) * DEG;
  const SPEED = num('xrspeed', 1.4) * upm; // m/s -> world units/s
  const EYE = (q.get('walk') === '1' && wander.walk?.eye) || num('xreye', 1.6) * upm; // walk mode: the desktop's own eye-height rule
  const ACCEL = 9;
  const DEAD = 0.2;
  const TURN_DEAD = 0.15;
  const BLINK = 0.12; // seconds of black over a teleport or a snap turn
  const axis = (v: number | undefined, dead = DEAD) =>
    v === undefined || !Number.isFinite(v) || Math.abs(v) <= dead
      ? 0
      : (THREE.MathUtils.clamp(v, -1, 1) - Math.sign(v) * dead) / (1 - dead);

  const sticks = { moveX: 0, moveY: 0, turnX: 0 };
  function readSticks() {
    sticks.moveX = sticks.moveY = sticks.turnX = 0;
    const session = renderer.xr.getSession();
    if (!session) return;
    for (const src of session.inputSources) {
      const a = src.gamepad?.axes;
      if (!a) continue;
      // xr-standard puts the thumbstick on 2/3 and a touchpad on 0/1; not every runtime fills both.
      const x = Math.abs(a[2] ?? 0) > Math.abs(a[0] ?? 0) ? a[2] : a[0];
      const y = Math.abs(a[3] ?? 0) > Math.abs(a[1] ?? 0) ? a[3] : a[1];
      if (src.handedness === 'right') {
        sticks.turnX = axis(x, turnMode === 'smooth' ? TURN_DEAD : DEAD);
      } else {
        sticks.moveX = axis(x);
        sticks.moveY = axis(y);
      }
    }
  }

  const head = new THREE.Vector3(); // head, world space
  const headLocal = new THREE.Vector3(); // head, rig space, metres
  const headQuaternion = new THREE.Quaternion();
  const headMatrix = new THREE.Matrix4();
  const headScale = new THREE.Vector3();
  const fwd = new THREE.Vector3(),
    right = new THREE.Vector3(),
    want = new THREE.Vector3();
  const vel = new THREE.Vector3(),
    step = new THREE.Vector3(),
    nrm = new THREE.Vector3();
  const groundFrom = new THREE.Vector3();
  const physicalStep = new THREE.Vector3();
  const bodyRigStart = new THREE.Vector3();
  const bodyTravel = new THREE.Vector3();
  const bodyInverseRotation = new THREE.Quaternion();
  const target = new THREE.Vector3(),
    tmpV = new THREE.Vector3(),
    tmpQ = new THREE.Quaternion();
  let targetOk = false;
  let aimingPrev = false;
  let snapLatch = false;
  let blink = 0;
  let headYaw = 0;

  /** The head's offset from the rig origin, in world units and in the rig's current orientation. */
  const headOffset = (out: THREE.Vector3) =>
    out.copy(headLocal).multiplyScalar(upm).applyQuaternion(rig.quaternion);

  /** Walk mode intersects known support, including stair levels; authoring mode uses its plane. */
  function aim(from: THREE.Object3D): boolean {
    targetOk = false;
    report.teleportTarget = null;
    report.teleportValid = false;
    const o = from.getWorldPosition(tmpV);
    const d = new THREE.Vector3(0, 0, -1).applyQuaternion(from.getWorldQuaternion(tmpQ));
    if (d.y > -1e-3) return false;
    const walk = wander.walk;
    if (walk) {
      const hit = raycastWalkFloor(
        o,
        d,
        (x, z) => {
          const cell = walk.cellAt(x, z, o.y);
          return cell.inside ? cell.floor : NaN;
        },
        Math.min(walk.grid?.cell ?? 0.1 * upm, 0.1 * upm) / 2,
        20 * upm,
        target,
      );
      if (!hit) return false;
      targetOk = reachable(target) && !walk.blockedAt(target.x, target.z, target.y);
    } else {
      const t = (groundFloor - o.y) / d.y;
      if (t < 0 || t > 20 * upm) return false;
      target.copy(o).addScaledVector(d, t);
      targetOk = reachable(target);
    }
    report.teleportTarget = target.toArray();
    report.teleportValid = targetOk;
    return true;
  }

  function commit() {
    if (!targetOk) return;
    const off = headOffset(new THREE.Vector3());
    rig.position.x = target.x - off.x;
    rig.position.z = target.z - off.z;
    // Move the reference floor by the landing height delta, retaining local-space eye/crouch pose.
    rig.position.y += target.y - groundFloor;
    groundFloor = target.y;
    report.groundFloor = groundFloor;
    targetOk = false;
    report.teleportValid = false;
    vel.set(0, 0, 0);
    avatarBody?.reset();
    blink = 1;
  }

  const turnUp = new THREE.Vector3(0, 1, 0);
  const turnQuaternion = new THREE.Quaternion();
  function turnAroundHead(angle: number) {
    // Rotate about the HEAD, not the rig origin, or a turn also slides you sideways.
    turnQuaternion.setFromAxisAngle(turnUp, angle);
    rig.position.sub(head).applyQuaternion(turnQuaternion).add(head);
    rig.quaternion.premultiply(turnQuaternion).normalize();
    // Walking in this same frame must use the new heading.
    headYaw = yawOf(tmpQ.copy(rig.quaternion).multiply(headQuaternion));
  }

  function snapTurn(dir: number) {
    if (!SNAP) return;
    turnAroundHead(-dir * SNAP);
    avatarBody?.reset();
    blink = 0.7;
  }

  function travel(dt: number) {
    want.set(0, 0, 0);
    fwd.set(-Math.sin(headYaw), 0, -Math.cos(headYaw));
    right.set(Math.cos(headYaw), 0, -Math.sin(headYaw));
    want.addScaledVector(fwd, -sticks.moveY).addScaledVector(right, sticks.moveX);
    want.clampLength(0, 1).multiplyScalar(SPEED);
    // Critically damped rather than instant velocity: no lurch on push, no dead stop on release.
    vel.lerp(want, 1 - Math.exp(-ACCEL * dt));
    if (vel.lengthSq() < 1e-10) vel.set(0, 0, 0);
    step.addScaledVector(vel, dt);
  }

  function moveRig(dt: number) {
    // Turning around an offset head translates the rig origin without walking. Start the body
    // travel measurement after that pivot, so only accepted locomotion can animate a step.
    bodyRigStart.copy(rig.position);
    // fourd.html's own soft boundary, applied to the RIG. Outward motion past the measured face is
    // scaled towards zero so you coast to a stop; inward motion is never damped, so it is always one
    // nudge back into the good region. The head is untouched by any of this.
    if (wander.clampOn) {
      for (let i = 0; i < 3; i++) {
        const s = (head.getComponent(i) - boxC.getComponent(i)) / boxH.getComponent(i);
        const dv = step.getComponent(i);
        if (dv * s > 0) step.setComponent(i, dv * softGain(Math.abs(s)));
      }
      if (onLeash) {
        const r0 = nearestOnPath(head);
        nrm.copy(head).sub(segBest);
        nrm.y = 0; // The path leash may slow horizontal walking, never lift or lower the rig.
        if (nrm.lengthSq() > 1e-12) {
          nrm.normalize();
          const out = step.dot(nrm);
          if (out > 0) step.addScaledVector(nrm, -out * (1 - softGain(r0 / wander.pathR)));
        }
      }
    }
    if (wander.walk?.advance) {
      groundFrom.set(head.x, groundFloor, head.z);
      const next = wander.walk.advance(groundFrom, step, dt);
      rig.position.x += next.x - groundFrom.x;
      rig.position.y += next.y - groundFloor;
      rig.position.z += next.z - groundFrom.z;
      groundFloor = next.y;
      report.groundFloor = groundFloor;
    } else rig.position.add(step);
  }

  // ---- the frame ----------------------------------------------------------------------------------------
  // Wrapped rather than spliced into fourd.html's loop: at this point three has already read this
  // frame's eye poses but has not yet composed them with the rig, so moving the rig here lands in
  // the SAME frame it was computed for. No lag, and fourd.html keeps its loop.
  const origRender = renderer.render;
  renderer.render = function (sc: THREE.Object3D, cam: THREE.Camera) {
    const xrFrame = renderer.xr.isPresenting && sc === scene && cam === camera;
    if (xrFrame && !disposed) tick();
    origRender.call(renderer, sc, cam);
    if (xrFrame) questView?.afterRender();
  } as typeof renderer.render;

  function suppressLocomotion() {
    vel.set(0, 0, 0);
    step.set(0, 0, 0);
    physicalWalk.reset();
    aimingPrev = false;
    targetOk = false;
    snapLatch = false;
    teleportBlockedUntilRelease = true;
    report.teleportTarget = null;
    report.teleportValid = false;
    marker.set(null, false);
    for (const c of controllers) {
      c.aiming = false;
      c.ray.visible = false;
    }
  }

  const sidebar = sceneNavigation
    ? createSceneSidebar({
        renderer,
        scene,
        onOpenChange(open) {
          sidebarOpen = open;
          suppressLocomotion();
          if (open) {
            if (errorResume !== null) {
              resumePlayback = errorResume;
              errorResume = null;
            } else if (!switching) resumePlayback = !!wander.playing;
            wander.play?.(false);
          } else {
            if (resumePlayback && !switching && !disposed && renderer.xr.isPresenting)
              wander.play?.(true);
            if (!switching) resumePlayback = false;
          }
        },
        async onSelect(id) {
          if (disposed) return;
          if (id === sceneNavigation.currentId && !switching) {
            sidebar?.close();
            return;
          }
          try {
            await sceneNavigation.select(id, { resumePlayback });
          } catch {
            // The current binding receives the session event, even if this one was
            // disposed during a failed activation. Obsolete requests cannot clear it.
          }
        },
      })
    : null;
  if (sidebar && sceneNavigation) {
    sidebar.setCatalog(sceneNavigation.clips, sceneNavigation.currentId);
    report.sidebar = sidebar.state;
  }
  const onSceneChange = (event: Event) => {
    if (disposed || !sidebar) return;
    const detail = (event as CustomEvent).detail;
    switching = !!detail.loading;
    if (switching) sidebar.setStatus('Loading… choose another clip to cancel', true);
    else if (detail.error) {
      resumePlayback = detail.resumePlayback ?? resumePlayback;
      if (renderer.xr.isPresenting) {
        errorResume = sidebarOpen ? null : resumePlayback;
        sidebar.showError(detail.error);
        if (sidebarOpen) wander.play?.(false);
      } else sidebar.setStatus(detail.error, false);
    } else {
      sidebar.setStatus('', false);
      if (detail.cancelled) {
        resumePlayback = detail.resumePlayback ?? resumePlayback;
        sidebar.close();
      }
    }
  };
  window.addEventListener('wander:scenechange', onSceneChange);

  function tick() {
    const now = performance.now();
    const dt = lastFrameStart ? Math.min(0.1, (now - lastFrameStart) / 1000) : 1 / 72;
    if (lastFrameStart) frames.push(now - lastFrameStart);
    lastFrameStart = now;

    avatarHands?.update();

    const frame = renderer.xr.getFrame();
    const referenceSpace = renderer.xr.getReferenceSpace();
    const pose = frame && referenceSpace ? frame.getViewerPose(referenceSpace) : null;
    if (!pose) {
      physicalWalk.reset();
      avatarBody?.reset();
      return;
    }
    // Three's array camera starts at the first eye, then shifts for the stereo frustum union.
    // Track the runtime's actual viewer center so head rotation cannot masquerade as a step.
    headMatrix.fromArray(pose.transform.matrix);
    if (!headMatrix.elements.every(Number.isFinite)) {
      physicalWalk.reset();
      avatarBody?.reset();
      return;
    }
    headMatrix.decompose(headLocal, headQuaternion, headScale);
    const hf = new THREE.Vector3(0, 0, -1).applyQuaternion(headQuaternion);

    if (!anchored) {
      anchored = true;
      // Face the way the preset framed the shot, then put the head exactly where its camera was.
      rig.quaternion.setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        homeYaw - Math.atan2(-hf.x, -hf.z),
      );
      const off = headOffset(tmpV);
      rig.position.set(
        home.x - off.x,
        floorRef ? groundFloor : groundFloor + EYE - off.y,
        home.z - off.z,
      );
    }
    rig.updateMatrixWorld(true);
    head.copy(headLocal).applyMatrix4(rig.matrixWorld);
    headYaw = yawOf(tmpQ.copy(rig.quaternion).multiply(headQuaternion));

    sidebar?.update({
      headPosition: head,
      headQuaternion: tmpQ.copy(rig.quaternion).multiply(headQuaternion),
      upm,
      dt,
    });
    bodyRigStart.copy(rig.position);

    if (sidebarOpen || switching) {
      suppressLocomotion();
    } else if (wander.possess) {
      physicalWalk.reset();
      // Possession: the rig follows the person's head so the viewer's eyes land on it. His yaw
      // reaches the rig only as snap turns through the blink (header point 2: never a smooth turn
      // of someone's head, never his pitch or roll); ?possessyaw=smooth opts into a continuous yaw.
      // The headset does every other rotation. No locomotion.
      const want = wander.possess.yaw,
        diff = Math.atan2(
          Math.sin(want - yawOf(rig.quaternion)),
          Math.cos(want - yawOf(rig.quaternion)),
        );
      if (q.get('possessyaw') === 'smooth')
        rig.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), want);
      else if (SNAP && Math.abs(diff) >= SNAP) snapTurn(-Math.sign(diff));
      const off = headOffset(tmpV);
      rig.position.set(
        wander.possess.head.x - off.x,
        wander.possess.head.y - off.y,
        wander.possess.head.z - off.z,
      );
      rig.updateMatrixWorld(true);
      head.copy(headLocal).applyMatrix4(rig.matrixWorld);
    } else {
      readSticks();
      if (teleportBlockedUntilRelease) {
        const triggerPressed =
          controllers.some((c) => c.triggerHeld) ||
          Array.from(renderer.xr.getSession()?.inputSources ?? []).some(
            (source) => source.gamepad?.buttons[0]?.pressed,
          );
        if (!triggerPressed && sticks.moveY >= -0.5) teleportBlockedUntilRelease = false;
      }
      if (turnMode === 'smooth') {
        // Apply the stick's analog rate directly every frame: no latch, blink, or release coast.
        // Pitch and roll stay with the headset's tracked pose.
        if (SNAP && sticks.turnX) turnAroundHead(-sticks.turnX * turnSpeed * DEG * dt);
      } else if (SNAP) {
        if (Math.abs(sticks.turnX) > 0.7) {
          if (!snapLatch) {
            snapTurn(Math.sign(sticks.turnX));
            snapLatch = true;
          }
        } else if (Math.abs(sticks.turnX) < 0.4) snapLatch = false;
      }

      // The runtime's real head displacement remains untouched. Only extra gain and joystick
      // displacement pass through the shared boundary/collision solver, in either movement mode.
      headYaw = yawOf(tmpQ.copy(rig.quaternion).multiply(headQuaternion));
      step.copy(physicalWalk.update(headLocal, rig.quaternion, upm, physicalStep));
      if (mode === 'smooth') travel(dt);
      moveRig(dt);
      rig.updateMatrixWorld(true);
      head.copy(headLocal).applyMatrix4(rig.matrixWorld);

      if (mode === 'teleport') {
        // Aim with the left stick pushed forward, or with either trigger held. Release to go.
        const held = controllers.find((c) => c.aiming);
        const aiming = !teleportBlockedUntilRelease && (!!held || sticks.moveY < -0.5);
        const src = (held ?? controllers[0]).grip;
        if (aiming && aim(src)) {
          marker.set(target, targetOk);
          for (const c of controllers) {
            c.ray.visible = c.grip === src;
            if (c.grip === src)
              c.ray.scale.setScalar(src.getWorldPosition(tmpV).distanceTo(target) / upm);
          }
        } else {
          marker.set(null, false);
          for (const c of controllers) c.ray.visible = false;
        }
        if (aimingPrev && !aiming) commit();
        aimingPrev = aiming;
      }
    }

    rig.updateMatrixWorld(true);
    head.copy(headLocal).applyMatrix4(rig.matrixWorld);

    // The body consumes reference-space metres. Account for artificial travel so a planted foot
    // stays in world space while the rig walks; head tracking is already included in headLocal.
    // Possession has a recorded body of its own, so hide this illustrative observer body there.
    avatarBody?.setSessionActive(!wander.possess);
    if (avatarBody && !wander.possess) {
      bodyTravel
        .subVectors(rig.position, bodyRigStart)
        .applyQuaternion(bodyInverseRotation.copy(rig.quaternion).invert())
        .divideScalar(upm);
      avatarBody.update({
        headLocal,
        headQuaternion,
        floorY: (groundFloor - rig.position.y) / upm,
        dt,
        travel: bodyTravel,
      });
    }

    // Grading. edgeAt is fourd.html's own measure - 0 inside the box, 1 at the hard limit - read off
    // the HEAD, so leaning out is what darkens the world, exactly as walking out does on the desktop.
    // Nothing here resists; it only tells you.
    const e = wander.clampOn ? THREE.MathUtils.clamp(wander.edgeAt(head), 0, 1) : 0;
    blink = Math.max(0, blink - dt / BLINK);
    vignette.set(Math.max(e * e * (3 - 2 * e), blink));

    if (adapt) adaptBudget();
    report.edge = +e.toFixed(2);
  }

  // Closed loop on the splat budget. Every measurement in this repo was taken on a desktop GPU and
  // is an upper bound on what a Quest 3 will do, so the device settles the real number itself: if
  // the median of the last ADAPT_WINDOW frames misses the target the budget drops, if it clears it
  // with room to spare the budget climbs back, never above what was asked for. One direction per
  // decision and ~1.25 s between decisions, so it cannot visibly oscillate.
  function adaptBudget() {
    if (frames.length < ADAPT_WINDOW || frames.length - lastAdapt < ADAPT_WINDOW) return;
    lastAdapt = frames.length;
    const win = frames.slice(-ADAPT_WINDOW).sort((a, b) => a - b);
    const med = win[ADAPT_WINDOW >> 1];
    report.medianMs = +med.toFixed(2);
    if (med > targetMs * 1.06 && budget > MIN_BUDGET) budget = Math.max(MIN_BUDGET, budget * 0.8);
    else if (med < targetMs * 0.8 && budget < xrLod) budget = Math.min(xrLod, budget * 1.12);
    else return;
    wander.spark.lodSplatCount = Math.round(budget);
    report.budget = Math.round(budget);
    publish();
  }

  function finish() {
    if (!frames.length) return;
    const s = [...frames].sort((a, b) => a - b);
    const at = (f: number) => s[Math.min(s.length - 1, Math.floor(s.length * f))];
    report.frames = s.length;
    report.medianMs = +at(0.5).toFixed(2);
    report.p95Ms = +at(0.95).toFixed(2);
    report.p99Ms = +at(0.99).toFixed(2);
    report.overBudget = +(s.filter((v) => v > targetMs).length / s.length).toFixed(3);
    report.budget = Math.round(budget);
    publish();
    console.log('[xr]', report);
  }

  const summary = () =>
    Object.entries(report)
      .map(([k, v]) => `${k} ${v}`)
      .join('\n');

  renderer.xr.addEventListener('sessionstart', onSessionStart);
  renderer.xr.addEventListener('sessionend', onSessionEnd);
  if (renderer.xr.isPresenting) onSessionStart();

  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      window.clearTimeout(sessionStartTimer);
      renderer.xr.removeEventListener('sessionstart', onSessionStart);
      renderer.xr.removeEventListener('sessionend', onSessionEnd);
      walkReferenceSpace?.removeEventListener('reset', resetPhysicalWalk);
      btn.removeEventListener('click', onButtonClick);
      sidebar?.dispose();
      window.removeEventListener('wander:scenechange', onSceneChange);
      avatarHands?.dispose();
      avatarBody?.dispose();
      questView?.dispose();
      for (const c of controllers) {
        c.grip.removeEventListener('selectstart', c.onSelectStart);
        c.grip.removeEventListener('selectend', c.onSelectEnd);
        c.grip.remove(c.ray);
        rig.remove(c.grip);
        c.ray.geometry.dispose();
        c.ray.material.dispose();
      }
      camera.remove(vignette.mesh);
      vignette.mesh.geometry.dispose();
      vignette.mesh.material.dispose();
      marker.dispose();
      rig.remove(camera);
      homeParent?.add(camera);
      camera.position.copy(home);
      camera.quaternion.copy(homeQ);
      camera.updateMatrixWorld(true);
      rig.removeFromParent();
      renderer.render = origRender;
      if (lodOn) wander.spark.lodSplatCount = deskLod;
      btn.remove();
      hud.remove();
    },
  };
}

function yawOf(qt: THREE.Quaternion) {
  const f = new THREE.Vector3(0, 0, -1).applyQuaternion(qt);
  return Math.atan2(-f.x, -f.z);
}

/**
 * The edge of the recording, redrawn for a headset: an aperture that closes in from the periphery,
 * specified in DEGREES off the forward axis. Head-locked by being a child of the camera, depth-test
 * off so it grades the splats, the avatar and the video surface together. Doubles as the blink over
 * a teleport or a snap turn, which is why it goes fully black rather than only closing.
 */
function buildVignette() {
  const uniforms = { uAmount: { value: 0 }, uWash: { value: 0.55 } };
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.25, 20, 14),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      uniforms,
      vertexShader:
        'varying vec3 vPos; void main(){ vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: `
        varying vec3 vPos;
        uniform float uAmount, uWash;
        void main() {
          float ang = acos(clamp(-normalize(vPos).z, -1.0, 1.0));
          float inner = mix(1.75, 0.38, uAmount);          // 100 deg wide open -> 22 deg at the limit
          float a = smoothstep(inner - 0.21, inner, ang);  // 12 deg feather, never a hard ring
          gl_FragColor = vec4(0.0, 0.0, 0.0, max(a, uWash * uAmount));
        }`,
    }),
  );
  mesh.renderOrder = 1000;
  mesh.frustumCulled = false;
  mesh.visible = false;
  return {
    mesh,
    set(amount: number) {
      uniforms.uAmount.value = amount;
      mesh.visible = amount > 0.002; // inside the region there is no intervention at all
    },
  };
}

/** Teleport landing ring, in world units. Green where the recording still holds, red where it stops. */
function buildMarker(upm: number) {
  const group = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.22 * upm, 0.3 * upm, 32),
    new THREE.MeshBasicMaterial({
      color: 0x66ffaa,
      transparent: true,
      opacity: 0.85,
      depthTest: false,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.renderOrder = 999;
  group.add(ring);
  group.visible = false;
  return {
    group,
    dispose() {
      group.removeFromParent();
      ring.geometry.dispose();
      ring.material.dispose();
    },
    set(at: THREE.Vector3 | null, ok: boolean) {
      group.visible = at !== null;
      if (!at) return;
      group.position.copy(at);
      (ring.material as THREE.MeshBasicMaterial).color.setHex(ok ? 0x66ffaa : 0xff5544);
    },
  };
}
