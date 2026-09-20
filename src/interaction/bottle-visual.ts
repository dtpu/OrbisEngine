import * as THREE from 'three';

export type BottleVisualUpdate = {
  /** World-space centre supplied by the interaction owner. */
  position: THREE.Vector3;
  /** World-space orientation supplied by the interaction owner. */
  quaternion: THREE.Quaternion;
  /** Shows the procedural bottle replacement. */
  proxyVisible: boolean;
  /** Shows the loose-bottle locator independently of the proxy. */
  locatorVisible: boolean;
  /** Lets the locator soften when a visitor is already reaching for the bottle. */
  nearHand?: boolean;
  /** Frame duration available to a caller that does not have an elapsed scene time. */
  dt?: number;
  /** Monotonic scene time in seconds for the locator pulse. */
  elapsed?: number;
};

type BottleSize = {
  bodyHeight: number;
  fullHeight: number;
  radius: number;
  neckRadius: number;
};

const finitePositive = (value: number | undefined) =>
  value !== undefined && Number.isFinite(value) && value > 0;

function bottleSize(stature: number, dimensions?: THREE.Vector3): BottleSize {
  const fallbackHeight = 0.105 * stature;
  const measuredHeight = dimensions?.y;
  // Object dimensions are useful only when they fit inside the procedural controller fist. A tiny
  // or malformed proxy dimension otherwise makes a held bottle disappear inside the glove.
  const fullHeight = finitePositive(measuredHeight)
    ? THREE.MathUtils.clamp(measuredHeight ?? fallbackHeight, 0.09 * stature, 0.12 * stature)
    : fallbackHeight;
  const measuredRadius = dimensions && Math.max(dimensions.x, dimensions.z) / 2;
  const radius = finitePositive(measuredRadius)
    ? THREE.MathUtils.clamp(measuredRadius ?? fullHeight * 0.15, 0.013 * stature, 0.017 * stature)
    : THREE.MathUtils.clamp(fullHeight * 0.15, 0.013 * stature, 0.017 * stature);
  const neckRadius = radius * 0.58;
  const shoulderHeight = fullHeight * 0.15;
  const neckHeight = fullHeight * 0.15;
  const capHeight = fullHeight * 0.09;
  return {
    bodyHeight: fullHeight - shoulderHeight - neckHeight - capHeight,
    fullHeight,
    radius,
    neckRadius,
  };
}

/**
 * A deliberately invented, scene-space visual proxy for a tracked bottle. It owns no gameplay
 * state: callers choose its pose and visibility from the recorded/physics authority.
 */
export class BottleVisual {
  readonly group = new THREE.Group();
  readonly proxy = new THREE.Group();
  readonly locator = new THREE.Group();
  private readonly locatorMaterial: THREE.MeshBasicMaterial;
  private readonly beaconMaterial: THREE.MeshBasicMaterial;
  private disposed = false;

  constructor(stature: number, dimensions?: THREE.Vector3) {
    if (!Number.isFinite(stature) || stature <= 0)
      throw new Error('Bottle visual requires a positive finite stature');

    const size = bottleSize(stature, dimensions);
    const shoulderHeight = size.fullHeight * 0.15;
    const neckHeight = size.fullHeight * 0.15;
    const capHeight = size.fullHeight * 0.09;
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(size.radius * 0.93, size.radius, size.bodyHeight, 20),
      new THREE.MeshStandardMaterial({
        color: 0x2d91a1,
        emissive: 0x092a30,
        roughness: 0.35,
        metalness: 0.04,
      }),
    );
    body.name = 'bottle-proxy-body';
    this.proxy.add(body);

    const bodyBottom = -size.fullHeight / 2;
    body.position.y = bodyBottom + size.bodyHeight / 2;
    const bodyTop = bodyBottom + size.bodyHeight;
    const shoulder = new THREE.Mesh(
      new THREE.CylinderGeometry(size.neckRadius, size.radius * 0.93, shoulderHeight, 20),
      new THREE.MeshStandardMaterial({
        color: 0x4aaebb,
        emissive: 0x0b3238,
        roughness: 0.32,
        metalness: 0.03,
      }),
    );
    shoulder.name = 'bottle-proxy-shoulder';
    shoulder.position.y = bodyTop + shoulderHeight / 2;
    this.proxy.add(shoulder);

    const neck = new THREE.Mesh(
      new THREE.CylinderGeometry(size.neckRadius, size.neckRadius, neckHeight, 16),
      new THREE.MeshStandardMaterial({ color: 0x63c2ca, emissive: 0x103f43, roughness: 0.28 }),
    );
    neck.name = 'bottle-proxy-neck';
    neck.position.y = bodyTop + shoulderHeight + neckHeight / 2;
    this.proxy.add(neck);

    const cap = new THREE.Mesh(
      new THREE.CylinderGeometry(size.neckRadius * 1.12, size.neckRadius * 1.12, capHeight, 16),
      new THREE.MeshStandardMaterial({
        color: 0xff9c3d,
        emissive: 0x4a1c03,
        roughness: 0.46,
      }),
    );
    cap.name = 'bottle-proxy-cap';
    cap.position.y = bodyTop + shoulderHeight + neckHeight + capHeight / 2;
    this.proxy.add(cap);

    const label = new THREE.Mesh(
      new THREE.CylinderGeometry(
        size.radius * 1.012,
        size.radius * 1.012,
        size.bodyHeight * 0.34,
        20,
      ),
      new THREE.MeshBasicMaterial({ color: 0xf5f3de }),
    );
    label.name = 'bottle-proxy-label';
    label.position.y = body.position.y - size.bodyHeight * 0.04;
    this.proxy.add(label);

    const labelBand = new THREE.Mesh(
      new THREE.CylinderGeometry(
        size.radius * 1.018,
        size.radius * 1.018,
        size.bodyHeight * 0.07,
        20,
      ),
      new THREE.MeshBasicMaterial({ color: 0x145d77 }),
    );
    labelBand.name = 'bottle-proxy-label-band';
    labelBand.position.y = label.position.y;
    this.proxy.add(labelBand);

    const locatorRadius = Math.max(size.radius * 1.8, 0.035 * stature);
    this.locatorMaterial = new THREE.MeshBasicMaterial({
      color: 0x9dffd0,
      transparent: true,
      opacity: 0.88,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(locatorRadius, Math.max(0.0025 * stature, size.radius * 0.11), 8, 32),
      this.locatorMaterial,
    );
    ring.name = 'bottle-locator-ring';
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -size.fullHeight * 0.42;
    this.locator.add(ring);

    this.beaconMaterial = new THREE.MeshBasicMaterial({
      color: 0xffdc6b,
      transparent: true,
      opacity: 0.88,
    });
    const beacon = new THREE.Mesh(
      new THREE.OctahedronGeometry(Math.max(0.012 * stature, size.radius * 0.48), 1),
      this.beaconMaterial,
    );
    beacon.name = 'bottle-locator-beacon';
    beacon.position.y = size.fullHeight * 0.7;
    this.locator.add(beacon);

    this.group.name = 'bottle-interaction-visual';
    this.proxy.name = 'interaction-bottle-proxy';
    this.locator.name = 'interaction-bottle-locator';
    this.group.add(this.proxy, this.locator);
  }

  update({
    position,
    quaternion,
    proxyVisible,
    locatorVisible,
    nearHand = false,
    dt,
    elapsed,
  }: BottleVisualUpdate) {
    if (this.disposed) return;
    this.group.position.copy(position);
    this.group.quaternion.copy(quaternion);
    // The bottle can inherit recorded spin, while its discovery aid stays upright above it.
    this.locator.quaternion.copy(this.group.quaternion).invert();
    this.proxy.visible = proxyVisible;
    this.locator.visible = locatorVisible;

    const time = Number.isFinite(elapsed) ? elapsed! : Number.isFinite(dt) ? Math.max(0, dt!) : 0;
    const pulse = 0.68 + 0.32 * Math.sin(time * Math.PI * 2.4);
    const attention = nearHand ? 0.72 : 1;
    this.locatorMaterial.opacity = 0.32 + pulse * 0.56 * attention;
    this.beaconMaterial.opacity = 0.38 + pulse * 0.5 * attention;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.group.removeFromParent();
    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    this.group.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.geometry) geometries.add(mesh.geometry);
      const meshMaterials = mesh.material;
      if (Array.isArray(meshMaterials))
        meshMaterials.forEach((material) => materials.add(material));
      else if (meshMaterials) materials.add(meshMaterials);
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    this.group.clear();
  }
}
