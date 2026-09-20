import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import * as THREE from 'three';
import { createSceneSidebar } from '../src/xr/scene-sidebar';

const originals = {
  document: globalThis.document,
  window: globalThis.window,
  HTMLVideoElement: globalThis.HTMLVideoElement,
};
const disposers: (() => void)[] = [];
const videos: FakeVideo[] = [];
class FakeVideo {
  src = '';
  currentTime = 0;
  duration = 10;
  videoWidth = 640;
  videoHeight = 360;
  onloadedmetadata: (() => void) | null = null;
  onloadeddata: (() => void) | null = null;
  onseeked: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onload: (() => void) | null = null;
  load() {}
  pause() {}
  removeAttribute() {
    this.src = '';
  }
}
beforeEach(() => {
  const context = new Proxy(
    {},
    {
      get: (_, key) =>
        key === 'measureText' ? (value: string) => ({ width: value.length * 10 }) : () => {},
      set: () => true,
    },
  );
  Object.assign(globalThis, {
    document: {
      createElement(tag: string) {
        if (tag === 'canvas') return { width: 0, height: 0, getContext: () => context };
        if (tag === 'video') {
          const video = new FakeVideo();
          videos.push(video);
          return video;
        }
        throw new Error(`Unexpected element ${tag}`);
      },
    },
    window: {
      location: { href: 'http://localhost/demo.html', origin: 'http://localhost' },
      setTimeout,
      clearTimeout,
    },
    HTMLVideoElement: FakeVideo,
  });
});
afterEach(() => {
  for (const dispose of disposers.splice(0)) dispose();
  videos.length = 0;
  Object.assign(globalThis, originals);
});

function fixture(count = 12, thumbnails = false) {
  const scene = new THREE.Scene();
  const controllers = [new THREE.Group(), new THREE.Group()];
  const rig = new THREE.Group();
  rig.add(...controllers);
  scene.add(rig);
  const sources = ['left', 'right'].map((handedness) => ({
    handedness,
    targetRayMode: 'tracked-pointer',
    gamepad: { buttons: Array.from({ length: 6 }, () => ({ pressed: false })), axes: [0, 0, 0, 0] },
  }));
  let session = { inputSources: sources };
  const xr = Object.assign(new THREE.EventDispatcher(), {
    isPresenting: true,
    getSession: () => session,
    getController: (index: number) => controllers[index],
  });
  const selected: string[] = [];
  const opened: boolean[] = [];
  const sidebar = createSceneSidebar({
    renderer: { xr } as unknown as THREE.WebGLRenderer,
    scene,
    onSelect: (id) => selected.push(id),
    onOpenChange: (value) => opened.push(value),
  });
  disposers.push(sidebar.dispose);
  sidebar.setCatalog(
    Array.from({ length: count }, (_, index) => ({
      id: `scene-${index}`,
      title: `Scene ${index}`,
      spare: index >= 5,
      src: thumbnails ? `/clips/${index}.mp4` : 'https://other.test/source.mp4',
      posterTime: 2,
    })),
    'scene-0',
  );
  const pose = {
    headPosition: new THREE.Vector3(0, 1.7, 0),
    headQuaternion: new THREE.Quaternion(),
    upm: 1,
    dt: 1 / 60,
  };
  const tick = () => sidebar.update(pose);
  const button = (hand: number, index: number, pressed: boolean) => {
    sources[hand].gamepad.buttons[index].pressed = pressed;
    tick();
  };
  const open = () => {
    tick();
    button(1, 5, true);
    button(1, 5, false);
  };
  const pointAt = (index: number, x: number, y: number) => {
    const panel = scene.getObjectByName('xr-scene-sidebar')!;
    const target = panel.localToWorld(
      new THREE.Vector3(((x / 640 - 0.5) * 640) / 1200, 0.5 - y / 1200, 0),
    );
    const controller = controllers[index];
    const direction = target.sub(controller.getWorldPosition(new THREE.Vector3())).normalize();
    controller.quaternion
      .setFromUnitVectors(new THREE.Vector3(0, 0, -1), direction)
      .premultiply(rig.getWorldQuaternion(new THREE.Quaternion()).invert());
  };
  return {
    sidebar,
    scene,
    rig,
    controllers,
    sources,
    selected,
    opened,
    pose,
    tick,
    button,
    open,
    pointAt,
    xr,
    reenter: () => {
      xr.dispatchEvent({ type: 'sessionend' } as never);
      session = { inputSources: sources };
      tick();
    },
  };
}

describe('XR scene sidebar', () => {
  test('right B toggles once, ignores held buttons on entry, and stays anchored while walking', () => {
    const f = fixture();
    f.button(1, 5, true);
    expect(f.sidebar.state.open).toBe(false);
    f.button(1, 5, false);
    f.button(0, 4, true);
    f.button(0, 5, true);
    f.button(1, 4, true);
    expect(f.sidebar.state.open).toBe(false);
    f.button(1, 4, false);
    f.pose.upm = 3;
    f.controllers[1].position.copy(f.pose.headPosition);
    f.button(1, 5, true);
    expect(f.sidebar.state.open).toBe(true);
    expect(f.sidebar.state.position[0]).toBeCloseTo(0);
    expect(f.sidebar.state.position[2]).toBeCloseTo(-3.45);
    f.tick();
    expect(f.sidebar.state.open).toBe(true); // Holding B does not toggle again.
    // The complete rail must fit a 90-degree view at opening, including the left thumbnails.
    const camera = new THREE.PerspectiveCamera(90, 1, 0.01, 100);
    camera.position.copy(f.pose.headPosition);
    camera.updateMatrixWorld(true);
    const panel = f.scene.getObjectByName('xr-scene-sidebar') as THREE.Mesh;
    const corners = panel.geometry.getAttribute('position');
    for (let i = 0; i < corners.count; i++) {
      const projected = new THREE.Vector3()
        .fromBufferAttribute(corners, i)
        .applyMatrix4(panel.matrixWorld)
        .project(camera);
      expect(Math.abs(projected.x)).toBeLessThan(1);
      expect(Math.abs(projected.y)).toBeLessThan(1);
    }
    const anchor = [...f.sidebar.state.position];
    f.pose.headPosition.set(8, 9, 10);
    f.tick();
    expect(f.sidebar.state.position).toEqual(anchor);
    f.button(1, 5, false);
    f.button(1, 5, true);
    expect(f.sidebar.state.open).toBe(false);
    expect(f.opened).toEqual([true, false]);
    f.reenter();
    expect(f.sidebar.state.open).toBe(false);
    f.button(1, 5, false);
    f.button(1, 5, true);
    expect(f.sidebar.state.open).toBe(true);
  });

  test('centers on the tracked aim after rig rotation and scaling, and reanchors only on opening', () => {
    const f = fixture();
    f.rig.position.set(2, -0.5, 4);
    f.rig.rotation.y = 0.7;
    f.rig.scale.setScalar(0.45);
    f.pose.upm = 0.45;
    f.pose.headPosition.copy(f.rig.localToWorld(new THREE.Vector3(0, 1.7, 0)));
    f.rig.getWorldQuaternion(f.pose.headQuaternion);
    const controller = f.controllers[1];
    controller.position.set(0.22, 1.2, -0.15);
    controller.rotation.set(-0.13, 0.25, 0);
    f.tick();
    f.button(1, 5, true);
    f.button(1, 5, false);
    const origin = controller.getWorldPosition(new THREE.Vector3());
    const direction = new THREE.Vector3(0, 0, -1).applyQuaternion(
      controller.getWorldQuaternion(new THREE.Quaternion()),
    );
    const panel = f.scene.getObjectByName('xr-scene-sidebar')!;
    const hit = new THREE.Raycaster(origin, direction).intersectObject(panel)[0];
    expect(hit.uv!.x).toBeCloseTo(0.5, 6);
    expect(hit.uv!.y).toBeCloseTo(0.5, 6);
    expect(hit.distance).toBeCloseTo(1.15 * f.pose.upm, 6);
    const line = f.scene.getObjectByName('xr-sidebar-pointer-1') as THREE.Line;
    const endpoint = new THREE.Vector3().fromBufferAttribute(
      line.geometry.getAttribute('position'),
      1,
    );
    expect(endpoint.distanceTo(panel.position)).toBeLessThan(1e-6);
    const anchor = [...f.sidebar.state.position];
    controller.position.x += 0.2;
    controller.rotation.y += 0.2;
    f.tick();
    expect(f.sidebar.state.position).toEqual(anchor);
    f.button(1, 5, true);
    f.button(1, 5, false);
    f.button(1, 5, true);
    expect(f.sidebar.state.position).not.toEqual(anchor);
  });

  test('joystick focus wins over the resting pointer until the user aims again', () => {
    const f = fixture();
    f.open();
    expect(f.sidebar.state.hover).toBe(2);
    f.sources[0].gamepad.axes[3] = 1;
    f.tick();
    f.sources[0].gamepad.axes[3] = 0;
    f.tick();
    expect(f.sidebar.state.focus).toBe(1);
    f.button(1, 4, true);
    expect(f.selected).toEqual(['scene-1']);
    f.button(1, 4, false);
    f.pointAt(1, 320, 154 + 156 * 4 + 70);
    f.tick();
    f.button(1, 4, true);
    expect(f.selected).toEqual(['scene-1', 'scene-4']);
  });

  test('A selects once, ignores left X, and can replace a pending selection without repeating held input', () => {
    const f = fixture();
    f.button(1, 4, true);
    f.open();
    expect(f.selected).toEqual([]);
    f.button(1, 4, false);
    f.button(0, 4, true);
    expect(f.selected).toEqual([]);
    f.pointAt(1, 320, 154 + 156 * 3 + 70);
    f.button(1, 4, true);
    f.tick();
    expect(f.selected).toEqual(['scene-3']);
    f.sidebar.setStatus('Loading scene', true);
    f.pointAt(1, 320, 154 + 156 * 4 + 70);
    f.button(1, 4, false);
    f.button(1, 4, true);
    f.sidebar.setStatus('', false);
    f.tick();
    expect(f.selected).toEqual(['scene-3', 'scene-4']);
    f.button(1, 4, false);
    f.button(1, 4, true);
    expect(f.selected).toEqual(['scene-3', 'scene-4', 'scene-4']);
  });

  test('actual Three ray intersections select a row once and close target works', () => {
    const f = fixture();
    f.open();
    f.pointAt(1, 320, 154 + 156 * 2 + 70);
    f.tick();
    expect(f.sidebar.state.hover).toBe(2);
    f.button(1, 0, true);
    f.tick();
    expect(f.selected).toEqual(['scene-2']);
    f.button(1, 0, false);
    f.pointAt(1, 580, 60);
    f.button(1, 0, true);
    expect(f.sidebar.state.open).toBe(false);
  });

  test('scrolls to spare clips during loading and held triggers cannot click through opening', () => {
    const f = fixture(15);
    f.button(1, 0, true);
    f.open();
    f.pointAt(1, 320, 210);
    f.tick();
    expect(f.selected).toEqual([]);
    f.sources[0].gamepad.axes[3] = 1;
    f.pose.dt = 0.1;
    for (let frame = 0; frame < 50; frame++) f.tick();
    expect(f.sidebar.state.focus).toBe(14);
    expect(f.sidebar.state.scroll).toBeGreaterThan(0);
    f.sources[0].gamepad.axes[3] = 0;
    // Point away from the newly aim-centered menu; trigger still chooses joystick focus.
    f.controllers[1].rotation.y = Math.PI;
    f.button(1, 0, false);
    f.button(1, 0, true);
    expect(f.selected).toEqual(['scene-14']);
    f.sidebar.setStatus('Loading scene', true);
    f.sources[0].gamepad.axes[3] = -1;
    f.button(1, 0, false);
    f.button(1, 0, true);
    expect(f.sidebar.state.focus).toBe(13);
    expect(f.selected).toEqual(['scene-14', 'scene-13']);
    f.button(1, 5, true);
    expect(f.sidebar.state.open).toBe(false);
  });

  test('a stick held while opening must return to neutral before browsing or repeating', () => {
    const f = fixture();
    f.sources[0].gamepad.axes[3] = 1;
    f.open();
    f.pose.dt = 0.1;
    for (let frame = 0; frame < 20; frame++) f.tick();
    expect(f.sidebar.state.focus).toBe(0);
    // Reversing without releasing is still the held input that opened the menu.
    f.sources[0].gamepad.axes[3] = -1;
    f.tick();
    expect(f.sidebar.state.focus).toBe(0);
    f.sources[0].gamepad.axes[3] = 0;
    f.tick();
    f.sources[0].gamepad.axes[3] = 1;
    f.tick();
    expect(f.sidebar.state.focus).toBe(1);
    for (let frame = 0; frame < 5; frame++) f.tick();
    expect(f.sidebar.state.focus).toBeGreaterThan(1);
    f.button(1, 5, true);
    f.button(1, 5, false);
    f.button(1, 5, true);
    const reopenedFocus = f.sidebar.state.focus;
    for (let frame = 0; frame < 20; frame++) f.tick();
    expect(f.sidebar.state.focus).toBe(reopenedFocus);
  });

  test('an error opens the restored sidebar on its next tracked pose without consuming held A', () => {
    const f = fixture();
    f.sources[1].gamepad.buttons[4].pressed = true;
    f.sidebar.showError('Could not switch. Choose another clip.');
    f.tick();
    expect(f.sidebar.state.open).toBe(true);
    expect(f.sidebar.state.error).toContain('Could not switch');
    expect(f.sidebar.state.busy).toBe(false);
    f.tick();
    expect(f.selected).toEqual([]);
    f.button(1, 4, false);
    f.button(1, 4, true);
    expect(f.selected).toHaveLength(1);
  });

  test('decodes only two visible source posters and safely cancels stale callbacks', () => {
    const f = fixture(30, true);
    f.open();
    expect(videos.length).toBe(2);
    expect(f.sidebar.state.thumbnailLoads).toBe(2);
    videos[0].onloadedmetadata!();
    expect(videos[0].currentTime).toBe(2);
    videos[0].onseeked!();
    expect(f.sidebar.state.thumbnails).toBe(1);
    f.tick();
    expect(videos.length).toBe(3);
    expect(f.sidebar.state.thumbnailLoads).toBe(2);
    const stale = videos[1].onloadedmetadata!;
    f.sidebar.setCatalog([], null);
    stale();
    expect(f.sidebar.state.thumbnailLoads).toBe(0);
    expect(f.sidebar.state.thumbnails).toBe(0);
    f.sidebar.dispose();
    expect(f.scene.getObjectByName('xr-scene-sidebar')).toBeUndefined();
    f.tick();
  });
});
