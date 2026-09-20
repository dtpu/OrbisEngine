import * as THREE from 'three';

export type SceneSidebarClip = {
  id: string;
  title: string;
  sub?: string;
  meta?: string;
  src: string;
  spare?: boolean;
  poster?: string;
  posterTime?: number;
};

const WIDTH = 640;
const HEIGHT = 1200;
const LIST_TOP = 154;
const LIST_BOTTOM = 1028;
const ROW_HEIGHT = 156;
const MAX_THUMBNAILS = 24;

/** A desktop-style scene rail, anchored where the right controller points when opened.
 * Head poses and upm are in scene/world coordinates. Call before locomotion each XR frame.
 */
export function createSceneSidebar({
  renderer,
  scene,
  onSelect,
  onOpenChange,
}: {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  onSelect: (id: string) => void;
  onOpenChange: (open: boolean) => void;
}) {
  const canvas = document.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Scene sidebar needs a canvas context');
  const ctx = context;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
    side: THREE.DoubleSide,
  });
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(WIDTH / HEIGHT, 1), material);
  panel.name = 'xr-scene-sidebar';
  panel.renderOrder = 10000;
  panel.visible = false;
  panel.frustumCulled = false;
  scene.add(panel);
  const state = {
    open: false,
    selected: null as string | null,
    busy: false,
    error: '',
    status: '',
    focus: 0,
    hover: -1,
    scroll: 0,
    clipCount: 0,
    thumbnails: 0,
    thumbnailLoads: 0,
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
  };
  let clips: SceneSidebarClip[] = [];
  let dirty = true;
  let disposed = false;
  let session: XRSession | null = null;
  let primed = false;
  let menuHeld = false;
  let confirmHeld = false;
  let stickDirection = 0;
  let stickNeedsRelease = false;
  let openRequested = false;
  let repeatRemaining = 0;
  let stickSelection = false;
  let thumbnailGeneration = 0;
  const thumbnails = new Map<string, HTMLCanvasElement>();
  const attempted = new Set<string>();
  const pending = new Map<string, () => void>();
  const forward = new THREE.Vector3();
  const raycaster = new THREE.Raycaster();
  const rayOrigin = new THREE.Vector3();
  const rayDirection = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const hitPoint = new THREE.Vector3();
  const pointerMaterial = new THREE.LineBasicMaterial({
    color: 0x547c79,
    transparent: true,
    opacity: 0.8,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const inputs = [0, 1].map((index) => {
    const controller = renderer.xr.getController(index);
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      pointerMaterial,
    );
    line.name = `xr-sidebar-pointer-${index}`;
    line.renderOrder = 10001;
    line.visible = false;
    line.frustumCulled = false;
    scene.add(line);
    const input = {
      controller,
      line,
      source: null as XRInputSource | null,
      trigger: false,
      focusOrigin: new THREE.Vector3(),
      focusDirection: new THREE.Vector3(),
    };
    const connected = (event: { data: XRInputSource }) => {
      input.source = event.data;
      // A controller connected with a held trigger must release it before selecting.
      input.trigger = true;
    };
    const disconnected = () => {
      input.source = null;
      input.trigger = false;
      line.visible = false;
    };
    controller.addEventListener('connected', connected);
    controller.addEventListener('disconnected', disconnected);
    return { controller, line, connected, disconnected, input };
  });

  function maxScroll() {
    return Math.max(0, clips.length * ROW_HEIGHT - (LIST_BOTTOM - LIST_TOP));
  }

  function revealFocus() {
    const top = state.focus * ROW_HEIGHT;
    state.scroll = Math.max(
      0,
      Math.min(
        maxScroll(),
        Math.max(top + ROW_HEIGHT - (LIST_BOTTOM - LIST_TOP), Math.min(state.scroll, top)),
      ),
    );
    dirty = true;
  }

  function text(value: string, x: number, y: number, maxWidth: number) {
    let fitted = value;
    if (ctx.measureText(fitted).width > maxWidth) {
      while (fitted && ctx.measureText(`${fitted}…`).width > maxWidth) fitted = fitted.slice(0, -1);
      fitted += '…';
    }
    ctx.fillText(fitted, x, y);
  }

  function draw() {
    ctx.fillStyle = '#f7f5f0';
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.strokeStyle = '#d6d2c8';
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, WIDTH - 2, HEIGHT - 2);
    ctx.fillStyle = '#1e2927';
    ctx.font = '52px Georgia, serif';
    ctx.fillText('Scenes', 34, 76);
    ctx.font = '22px sans-serif';
    ctx.fillStyle = '#727970';
    ctx.fillText(`${clips.length} clips · browse your collection`, 36, 117);
    ctx.fillStyle = state.hover === -2 ? '#d8dfd8' : '#e9e7df';
    ctx.fillRect(WIDTH - 100, 30, 70, 60);
    ctx.fillStyle = '#263730';
    ctx.font = '34px sans-serif';
    ctx.fillText('×', WIDTH - 76, 72);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, LIST_TOP, WIDTH, LIST_BOTTOM - LIST_TOP);
    ctx.clip();
    const first = Math.floor(state.scroll / ROW_HEIGHT);
    const last = Math.min(
      clips.length,
      Math.ceil((state.scroll + LIST_BOTTOM - LIST_TOP) / ROW_HEIGHT),
    );
    for (let i = first; i < last; i++) {
      const clip = clips[i];
      const y = LIST_TOP + i * ROW_HEIGHT - state.scroll;
      const selected = clip.id === state.selected;
      const focused = i === state.focus;
      const hovered = i === state.hover;
      if (selected || focused || hovered) {
        ctx.fillStyle = hovered ? '#dce5df' : selected ? '#e7e8df' : '#eeeee7';
        ctx.fillRect(0, y, WIDTH, ROW_HEIGHT);
      }
      if (selected) {
        ctx.fillStyle = '#2d554b';
        ctx.fillRect(0, y, 6, ROW_HEIGHT);
      }
      if (focused) {
        ctx.strokeStyle = '#63847b';
        ctx.lineWidth = 2;
        ctx.strokeRect(12, y + 5, WIDTH - 34, ROW_HEIGHT - 10);
      }
      ctx.strokeStyle = '#dcd9d0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(30, y + ROW_HEIGHT);
      ctx.lineTo(WIDTH - 30, y + ROW_HEIGHT);
      ctx.stroke();
      ctx.fillStyle = '#deded5';
      ctx.fillRect(32, y + 31, 148, 84);
      const thumb = thumbnails.get(clip.id);
      if (thumb) ctx.drawImage(thumb, 32, y + 31, 148, 84);
      else {
        ctx.fillStyle = '#7a827a';
        ctx.font = '21px sans-serif';
        ctx.fillText('▶', 94, y + 82);
      }
      ctx.fillStyle = '#293b33';
      ctx.font = '33px Georgia, serif';
      text(clip.title, 202, y + 51, WIDTH - 232);
      ctx.fillStyle = '#626c62';
      ctx.font = '21px sans-serif';
      text(clip.sub || 'Recorded scene', 202, y + 86, WIDTH - 232);
      ctx.font = '18px sans-serif';
      text(
        [selected ? 'CURRENT' : clip.spare ? 'MORE SCENES' : '', clip.meta]
          .filter(Boolean)
          .join(' · '),
        202,
        y + 120,
        WIDTH - 232,
      );
    }
    ctx.restore();
    if (maxScroll() > 0) {
      const track = LIST_BOTTOM - LIST_TOP;
      const length = Math.max(56, (track * track) / (clips.length * ROW_HEIGHT));
      ctx.fillStyle = '#a9b6ab';
      ctx.fillRect(
        WIDTH - 10,
        LIST_TOP + (state.scroll / maxScroll()) * (track - length),
        5,
        length,
      );
    }
    ctx.fillStyle = '#eae9e1';
    ctx.fillRect(1, LIST_BOTTOM, WIDTH - 2, HEIGHT - LIST_BOTTOM - 1);
    ctx.fillStyle = state.error ? '#95402c' : '#355047';
    ctx.font = '24px sans-serif';
    text(state.status || 'A or trigger · Open selected scene', 32, 1072, WIDTH - 64);
    ctx.fillStyle = '#626c62';
    ctx.font = '22px sans-serif';
    ctx.fillText('Joystick ↑ ↓  Browse', 32, 1118);
    ctx.fillText('B  Close sidebar', 32, 1159);
    texture.needsUpdate = true;
    dirty = false;
  }

  function sameOrigin(value: string) {
    try {
      const url = new URL(value, window.location.href);
      return url.origin === window.location.origin && ['http:', 'https:'].includes(url.protocol)
        ? url.href
        : null;
    } catch {
      return null;
    }
  }

  function loadThumbnail(clip: SceneSidebarClip) {
    const generation = thumbnailGeneration;
    const src = sameOrigin(clip.poster || clip.src);
    attempted.add(clip.id);
    if (!src) return;
    const media = document.createElement(clip.poster ? 'img' : 'video');
    let finished = false;
    const timeout = window.setTimeout(() => finish(), 10000);
    function cleanup() {
      window.clearTimeout(timeout);
      media.onload = media.onerror = null;
      if (media instanceof HTMLVideoElement) {
        media.onloadeddata = media.onloadedmetadata = media.onseeked = null;
        media.pause();
        media.removeAttribute('src');
        media.load();
      } else media.removeAttribute('src');
    }
    function finish(success = false) {
      if (finished) return;
      finished = true;
      if (!disposed && generation === thumbnailGeneration) {
        pending.delete(clip.id);
        if (success) {
          const thumb = document.createElement('canvas');
          thumb.width = 192;
          thumb.height = 108;
          const thumbContext = thumb.getContext('2d');
          if (thumbContext) {
            const w = media instanceof HTMLVideoElement ? media.videoWidth : media.naturalWidth;
            const h = media instanceof HTMLVideoElement ? media.videoHeight : media.naturalHeight;
            if (w > 0 && h > 0) {
              const scale = Math.max(thumb.width / w, thumb.height / h);
              try {
                thumbContext.drawImage(
                  media,
                  (thumb.width - w * scale) / 2,
                  (thumb.height - h * scale) / 2,
                  w * scale,
                  h * scale,
                );
                thumbnails.set(clip.id, thumb);
                while (thumbnails.size > MAX_THUMBNAILS) {
                  const oldest = thumbnails.keys().next().value;
                  if (oldest === undefined) break;
                  thumbnails.delete(oldest);
                  attempted.delete(oldest);
                }
              } catch {
                // Missing or undecodable source footage keeps the neutral thumbnail.
              }
            }
          }
        }
        state.thumbnails = thumbnails.size;
        state.thumbnailLoads = pending.size;
        dirty = true;
      }
      cleanup();
    }
    pending.set(clip.id, () => finish());
    state.thumbnailLoads = pending.size;
    media.onerror = () => finish();
    if (media instanceof HTMLVideoElement) {
      media.muted = true;
      media.playsInline = true;
      media.preload = 'auto';
      media.onloadedmetadata = () => {
        if (finished || disposed || generation !== thumbnailGeneration) return;
        const requested = Number.isFinite(clip.posterTime) ? Math.max(0, clip.posterTime!) : 0.5;
        const target = Number.isFinite(media.duration)
          ? Math.min(requested, Math.max(0, media.duration - 0.05))
          : 0;
        if (target > 0) {
          media.onseeked = () => finish(true);
          media.currentTime = target;
        } else media.onloadeddata = () => finish(true);
      };
      media.src = src;
      media.load();
    } else {
      media.onload = () => finish(true);
      media.src = src;
    }
  }

  function requestVisibleThumbnails() {
    if (!state.open || disposed) return;
    const first = Math.floor(state.scroll / ROW_HEIGHT);
    const last = Math.min(
      clips.length,
      Math.ceil((state.scroll + LIST_BOTTOM - LIST_TOP) / ROW_HEIGHT),
    );
    for (let i = first; i < last && pending.size < 2; i++) {
      const clip = clips[i];
      if (!attempted.has(clip.id)) loadThumbnail(clip);
    }
  }

  function close() {
    openRequested = false;
    if (!state.open) return;
    state.open = panel.visible = false;
    state.hover = -1;
    stickSelection = false;
    for (const { line } of inputs) line.visible = false;
    onOpenChange(false);
  }

  function resetSession() {
    close();
    session = null;
    primed = false;
    menuHeld = false;
    confirmHeld = false;
    stickDirection = 0;
    stickNeedsRelease = false;
    repeatRemaining = 0;
    for (const { input } of inputs) {
      input.trigger = false;
      input.source = null;
    }
  }
  renderer.xr.addEventListener('sessionend', resetSession);

  function open(
    headPosition: THREE.Vector3,
    headQuaternion: THREE.Quaternion,
    upm: number,
    pointer?: THREE.Group,
  ) {
    if (pointer?.visible) {
      pointer.updateWorldMatrix(true, false);
      pointer.getWorldPosition(panel.position);
      pointer.getWorldQuaternion(rotation);
      forward.set(0, 0, -1).applyQuaternion(rotation).normalize();
    } else {
      panel.position.copy(headPosition);
      forward.set(0, 0, -1).applyQuaternion(headQuaternion).normalize();
    }
    panel.position.addScaledVector(forward, 1.15 * upm);
    panel.lookAt(headPosition);
    panel.scale.setScalar(1.35 * upm);
    panel.updateMatrixWorld(true);
    state.position = panel.position.toArray();
    state.quaternion = panel.quaternion.toArray();
    state.open = panel.visible = true;
    state.focus = Math.max(
      0,
      clips.findIndex((clip) => clip.id === state.selected),
    );
    state.hover = -1;
    stickSelection = false;
    revealFocus();
    onOpenChange(true);
  }

  function targetAt(x: number, y: number) {
    if (x >= WIDTH - 110 && y >= 20 && y <= 105) return -2;
    if (y < LIST_TOP || y >= LIST_BOTTOM) return -1;
    const index = Math.floor((y - LIST_TOP + state.scroll) / ROW_HEIGHT);
    return index >= 0 && index < clips.length ? index : -1;
  }

  function readStick(sources: XRInputSource[]) {
    let axis = 0;
    for (const source of sources) {
      const axes = source.gamepad?.axes;
      const value = axes && axes.length >= 4 ? axes[3] : axes?.[1] || 0;
      if (Math.abs(value) > Math.abs(axis)) axis = value;
    }
    return Math.abs(axis) > 0.55 ? Math.sign(axis) : 0;
  }

  function update({
    headPosition,
    headQuaternion,
    upm,
    dt,
  }: {
    headPosition: THREE.Vector3;
    headQuaternion: THREE.Quaternion;
    upm: number;
    dt: number;
  }) {
    if (disposed) return;
    const activeSession = renderer.xr.getSession();
    if (!activeSession || !renderer.xr.isPresenting) {
      if (session || state.open) resetSession();
      return;
    }
    if (session !== activeSession) {
      const requested = openRequested;
      close();
      openRequested = requested;
      primed = false;
      stickDirection = 0;
      repeatRemaining = 0;
      session = activeSession;
    }
    const sources = Array.from(activeSession.inputSources);
    const rightSource = sources.find((source) => source.handedness === 'right');
    const menu = !!rightSource?.gamepad?.buttons[5]?.pressed;
    const menuPressed = primed && menu && !menuHeld;
    menuHeld = menu;
    const confirm = !!rightSource?.gamepad?.buttons[4]?.pressed;
    const confirmPressed = primed && confirm && !confirmHeld;
    confirmHeld = confirm;
    // Snapshot A and triggers even while hidden/busy, so held buttons cannot click through.
    const pressed = inputs.map(({ input }, index) => {
      const retainedSource = renderer.xr.getControllerGrip?.(index)?.userData.wanderInputSource as
        XRInputSource | undefined;
      const source =
        input.source && sources.includes(input.source)
          ? input.source
          : retainedSource && sources.includes(retainedSource)
            ? retainedSource
            : sources[index];
      const trigger = !!source?.gamepad?.buttons[0]?.pressed;
      const edge = primed && trigger && !input.trigger;
      input.trigger = trigger;
      return { source, edge };
    });
    primed = true;
    if (menuPressed || openRequested) {
      if (state.open && menuPressed) close();
      else {
        const index = pressed.findIndex(
          ({ source }) => source === rightSource && source?.targetRayMode === 'tracked-pointer',
        );
        open(headPosition, headQuaternion, Math.max(0.001, upm), inputs[index]?.controller);
        stickDirection = readStick(sources);
        stickNeedsRelease = stickDirection !== 0;
        repeatRemaining = 0;
        openRequested = false;
      }
      if (state.open) {
        requestVisibleThumbnails();
        draw();
      }
      return;
    }
    if (!state.open) return;
    let hover = -1;
    let selectedTarget = -1;
    for (let index = 0; index < inputs.length; index++) {
      const { controller, line, input } = inputs[index];
      const { source, edge } = pressed[index];
      line.visible = false;
      if (!source || source.targetRayMode !== 'tracked-pointer' || !controller.visible) continue;
      controller.updateWorldMatrix(true, false);
      controller.getWorldPosition(rayOrigin);
      controller.getWorldQuaternion(rotation);
      rayDirection.set(0, 0, -1).applyQuaternion(rotation).normalize();
      // Joystick browsing owns selection until the user deliberately aims again.
      // The panel now opens under the ray, so a resting pointer must not override that focus.
      const pointerMoved =
        stickSelection &&
        (rayOrigin.distanceToSquared(input.focusOrigin) > (0.025 * upm) ** 2 ||
          rayDirection.dot(input.focusDirection) < Math.cos(0.05));
      raycaster.set(rayOrigin, rayDirection);
      const hit = raycaster.intersectObject(panel, false)[0];
      if (hit?.uv) {
        const target = targetAt(hit.uv.x * WIDTH, (1 - hit.uv.y) * HEIGHT);
        if (target !== -1) {
          hover = target;
          if (pointerMoved) stickSelection = false;
        }
        if (edge) selectedTarget = target;
        hitPoint.copy(hit.point);
      } else hitPoint.copy(rayOrigin).addScaledVector(rayDirection, upm * 2);
      const positions = line.geometry.getAttribute('position') as THREE.BufferAttribute;
      positions.setXYZ(0, rayOrigin.x, rayOrigin.y, rayOrigin.z);
      positions.setXYZ(1, hitPoint.x, hitPoint.y, hitPoint.z);
      positions.needsUpdate = true;
      line.visible = true;
    }
    if (hover !== state.hover) {
      state.hover = hover;
      dirty = true;
    }
    {
      const direction = readStick(sources);
      if (!direction) stickNeedsRelease = false;
      repeatRemaining -= Math.max(0, Math.min(0.1, dt));
      if (
        !stickNeedsRelease &&
        direction &&
        (direction !== stickDirection || repeatRemaining <= 0)
      ) {
        state.focus = Math.max(0, Math.min(clips.length - 1, state.focus + direction));
        repeatRemaining = direction !== stickDirection ? 0.38 : 0.14;
        stickSelection = true;
        for (const { controller, input } of inputs) {
          controller.getWorldPosition(input.focusOrigin);
          controller.getWorldQuaternion(rotation);
          input.focusDirection.set(0, 0, -1).applyQuaternion(rotation).normalize();
        }
        revealFocus();
      }
      stickDirection = direction;
      if (stickSelection && pressed.some(({ edge }) => edge)) {
        selectedTarget = state.focus;
      }
      if (confirmPressed) selectedTarget = stickSelection || hover === -1 ? state.focus : hover;
      if (selectedTarget === -2) close();
      else if (selectedTarget >= 0 && clips[selectedTarget]) {
        state.focus = selectedTarget;
        dirty = true;
        onSelect(clips[selectedTarget].id);
      }
    }
    requestVisibleThumbnails();
    if (dirty) draw();
  }

  return {
    state,
    setCatalog(next: SceneSidebarClip[], currentId: string | null) {
      if (disposed) return;
      thumbnailGeneration++;
      for (const cancel of pending.values()) cancel();
      pending.clear();
      thumbnails.clear();
      attempted.clear();
      clips = next.map((clip) => ({ ...clip }));
      state.clipCount = clips.length;
      state.selected = currentId;
      state.thumbnails = state.thumbnailLoads = 0;
      state.focus = Math.max(
        0,
        clips.findIndex((clip) => clip.id === currentId),
      );
      revealFocus();
    },
    setStatus(message: string, busy: boolean) {
      if (disposed) return;
      state.status = message;
      state.busy = busy;
      state.error = busy ? '' : message;
      dirty = true;
    },
    showError(message: string) {
      if (disposed) return;
      state.status = state.error = message;
      state.busy = false;
      if (!state.open) openRequested = true;
      dirty = true;
    },
    close,
    update,
    dispose() {
      if (disposed) return;
      close();
      disposed = true;
      thumbnailGeneration++;
      for (const cancel of pending.values()) cancel();
      pending.clear();
      thumbnails.clear();
      renderer.xr.removeEventListener('sessionend', resetSession);
      for (const { controller, connected, disconnected, line } of inputs) {
        controller.removeEventListener('connected', connected);
        controller.removeEventListener('disconnected', disconnected);
        line.removeFromParent();
        line.geometry.dispose();
      }
      panel.removeFromParent();
      panel.geometry.dispose();
      material.dispose();
      texture.dispose();
      pointerMaterial.dispose();
    },
  };
}
