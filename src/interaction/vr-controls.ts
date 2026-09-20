import * as THREE from 'three';

export type ControlAction = 'microphone' | 'replay' | 'pause' | 'reset' | 'end';
type Button = {
  action: ControlAction;
  mesh: THREE.Mesh;
  canvas: HTMLCanvasElement;
  texture: THREE.CanvasTexture;
};

/** Small controller-operated panel rendered into the XR scene, not a DOM overlay. */
export class VrInteractionControls {
  readonly group = new THREE.Group();
  private readonly buttons: Button[] = [];
  private readonly raycaster = new THREE.Raycaster();
  private readonly statusCanvas = document.createElement('canvas');
  private readonly statusTexture: THREE.CanvasTexture;
  private previous = '';
  private hovered: ControlAction | null = null;

  constructor(
    private readonly stature: number,
    private readonly invoke: (action: ControlAction) => void,
  ) {
    this.group.name = 'interaction-controls';
    const names: ControlAction[] = ['microphone', 'replay', 'pause', 'reset', 'end'];
    for (let i = 0; i < names.length; i++) {
      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 160;
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(0.102 * stature, 0.052 * stature),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          depthTest: false,
          depthWrite: false,
          toneMapped: false,
        }),
      );
      mesh.position.x = (i - 2) * 0.11 * stature;
      mesh.renderOrder = 10010;
      mesh.userData.action = names[i];
      this.group.add(mesh);
      this.buttons.push({ action: names[i], mesh, canvas, texture });
    }
    this.statusCanvas.width = 1400;
    this.statusCanvas.height = 140;
    this.statusTexture = new THREE.CanvasTexture(this.statusCanvas);
    this.statusTexture.colorSpace = THREE.SRGBColorSpace;
    const status = new THREE.Mesh(
      new THREE.PlaneGeometry(0.54 * stature, 0.045 * stature),
      new THREE.MeshBasicMaterial({
        map: this.statusTexture,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    status.position.y = 0.053 * stature;
    status.renderOrder = 10010;
    this.group.add(status);
    this.group.visible = false;
  }

  update(
    position: THREE.Vector3,
    rotation: THREE.Quaternion,
    text: string,
    microphone: boolean,
    playing: boolean,
  ) {
    this.group.position
      .set(0, -0.25 * this.stature, -0.65 * this.stature)
      .applyQuaternion(rotation)
      .add(position);
    this.group.quaternion.copy(rotation);
    this.group.updateMatrixWorld(true);
    const key = JSON.stringify([text, microphone, playing, this.hovered]);
    if (this.previous === key) return;
    this.previous = key;
    for (const button of this.buttons) {
      const ctx = button.canvas.getContext('2d')!;
      ctx.fillStyle = this.hovered === button.action ? '#37675d' : '#142a29';
      ctx.fillRect(0, 0, 320, 160);
      ctx.strokeStyle = button.action === 'microphone' && microphone ? '#a9efab' : '#779994';
      ctx.lineWidth = 5;
      ctx.strokeRect(3, 3, 314, 154);
      ctx.fillStyle = '#f4f9ed';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = 'bold 42px sans-serif';
      const label = {
        microphone: microphone ? 'Mute mic' : 'Enable mic',
        replay: 'Replay',
        pause: playing ? 'Pause' : 'Continue',
        reset: 'Reset',
        end: 'End voice',
      }[button.action];
      ctx.fillText(label, 160, 80);
      button.texture.needsUpdate = true;
    }
    const ctx = this.statusCanvas.getContext('2d')!;
    ctx.fillStyle = '#102322';
    ctx.fillRect(0, 0, 1400, 140);
    ctx.fillStyle = '#f4f9ed';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '38px sans-serif';
    ctx.fillText(text.slice(0, 85), 700, 70);
    this.statusTexture.needsUpdate = true;
  }

  hit(origin: THREE.Vector3, direction: THREE.Vector3): ControlAction | null {
    if (!this.group.visible) return null;
    this.raycaster.set(origin, direction.clone().normalize());
    return (
      this.raycaster.intersectObjects(
        this.buttons.map((b) => b.mesh),
        false,
      )[0]?.object.userData.action ?? null
    );
  }
  hover(origin: THREE.Vector3, direction: THREE.Vector3): boolean {
    this.hovered = this.hit(origin, direction);
    return this.hovered !== null;
  }
  select(origin: THREE.Vector3, direction: THREE.Vector3): boolean {
    const action = this.hit(origin, direction);
    if (!action) return false;
    this.invoke(action);
    return true;
  }
  dispose() {
    this.group.traverse((node) => {
      if (node instanceof THREE.Mesh) {
        node.geometry.dispose();
        const material = node.material as THREE.MeshBasicMaterial;
        material.map?.dispose();
        material.dispose();
      }
    });
    this.group.removeFromParent();
  }
}
