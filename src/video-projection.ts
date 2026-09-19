import * as THREE from 'three';
interface ProjectionSource {
  url: string;
  sha256: string;
  width: number;
  height: number;
  fps: number;
  duration: number;
}

export interface VideoProjection {
  url: string;
  sha256: string;
  // Alignment of the projection cameras relative to the output root, like a composite component.
  transform?: {
    position: [number, number, number];
    quaternion: [number, number, number, number];
    scale: number;
  };
  // World-unit distance and degrees of turn from the recorded pose at which the video weight is zero.
  falloff: { distance: number; angle: number };
  // Fraction of the frame over which the video fades at the frustum edge.
  feather: number;
  // 'depth' (default): scene content nearer than the surface draws over the frame (a 3D person in
  // front of a doorframe). 'video': nothing draws over the frame; for worlds whose only splats
  // are the environment, where anything in front of the recorded surface is by definition wrong.
  priority?: 'depth' | 'video';
  // 'time' (default): the layer shows the frame at the source time. 'pose': while the source is
  // paused, the layer steers it to the recorded frame whose camera best faces the viewer within
  // selectRadius (world units, default falloff.distance), so turning at a station follows the gaze.
  select?: 'time' | 'pose';
  selectRadius?: number;
  // Linear-light gain applied to the splat pass in the composite so it matches the footage's
  // exposure; measured offline as mean(video)/mean(splats) where both are valid. 1 = none.
  exposure?: number;
}

/**
 * Opt-in video-projection layer. The recorded frame is projected from its own camera onto a
 * depth surface that was rendered from the trained splats offline, so the video sits ON the
 * splat geometry instead of floating in front of it: a doorframe in the frame occludes a
 * 3D person behind it because the surface writes real depth. Every frame is resident before
 * the output is ready; nothing is guessed at playback.
 *
 * Compositing runs as three renders per frame into offscreen targets:
 *   A  the scene with the video surface (opaque, depth-tested against everything);
 *   B  the scene without it (what the default path shows);
 *   W  the surface alone, emitting its blend weight (frustum feather x pose falloff x person mask).
 * out = mix(B, A, W). Splats in front of the surface appear in both, so they win either way; the
 * weight only decides how much of the splats BEHIND the surface show through as the viewer leaves
 * the recorded pose or reaches the edge of the frame. With W = 0 everywhere the output is B.
 */

export interface ProjectionHeader {
  version: 1;
  width: number;
  height: number;
  frames: number;
  fps: number;
  // Depth layers can be fewer than source frames; depthIndex maps each source frame to its layer.
  depthFrames: number;
  depthIndex: number[];
  depthSources: number[];
  maskLayers: number;
  source: { width: number; height: number; sha256: string };
  intrinsics: { fx: number; fy: number; cx: number; cy: number };
  depthRange: { near: number; far: number };
  edgeThreshold: number;
  cameras: number[][];
  registeredFrames: number[];
}

function check(ok: unknown, message: string): asserts ok {
  if (!ok) throw new Error(`Video projection: ${message}`);
}
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const positiveInt = (x: unknown): x is number => Number.isInteger(x) && (x as number) > 0;

async function gunzip(bytes: ArrayBuffer): Promise<Uint8Array> {
  check(typeof DecompressionStream === 'function', 'this browser cannot decompress gzip');
  try {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    throw new Error('Video projection: not a gzip container');
  }
}

/** Fail closed: the container must describe exactly the experiment's source and be complete. */
export async function parseProjectionContainer(
  bytes: ArrayBuffer,
  source: ProjectionSource,
): Promise<{ header: ProjectionHeader; data: Uint8Array }> {
  const raw = await gunzip(bytes);
  check(
    raw.byteLength > 8 && String.fromCharCode(...raw.subarray(0, 4)) === 'WVP1',
    'not a WVP1 container',
  );
  const headerLength = new DataView(raw.buffer, raw.byteOffset, raw.byteLength).getUint32(4, true);
  check(headerLength > 0 && 8 + headerLength <= raw.byteLength, 'truncated header');
  const h = JSON.parse(
    new TextDecoder().decode(raw.subarray(8, 8 + headerLength)),
  ) as ProjectionHeader;
  check(
    h?.version === 1 &&
      positiveInt(h.width) &&
      positiveInt(h.height) &&
      positiveInt(h.frames) &&
      finite(h.fps) &&
      h.fps > 0,
    'invalid dimensions',
  );
  check(
    h.source?.width === source.width &&
      h.source?.height === source.height &&
      h.source?.sha256 === source.sha256,
    'depth was built for a different source video',
  );
  check(
    Math.abs(h.fps - source.fps) < 1e-6 && h.frames <= Math.round(source.duration * source.fps) + 1,
    'frame rate or count disagrees with the source',
  );
  const k = h.intrinsics;
  check(
    k && [k.fx, k.fy].every((x) => finite(x) && x > 0) && [k.cx, k.cy].every(finite),
    'invalid intrinsics',
  );
  check(
    h.depthRange &&
      finite(h.depthRange.near) &&
      finite(h.depthRange.far) &&
      h.depthRange.near > 0 &&
      h.depthRange.far > h.depthRange.near,
    'invalid depth range',
  );
  check(finite(h.edgeThreshold) && h.edgeThreshold > 0, 'invalid discontinuity threshold');
  check(
    Array.isArray(h.cameras) &&
      h.cameras.length === h.frames &&
      h.cameras.every((m) => Array.isArray(m) && m.length === 16 && m.every(finite)),
    'one camera_to_world per frame is required',
  );
  check(
    Array.isArray(h.registeredFrames) &&
      h.registeredFrames.every((f) => Number.isInteger(f) && f >= 0 && f < h.frames),
    'invalid registered frame list',
  );
  check(
    positiveInt(h.depthFrames) &&
      h.depthFrames <= h.frames &&
      Array.isArray(h.depthIndex) &&
      h.depthIndex.length === h.frames &&
      h.depthIndex.every((i) => Number.isInteger(i) && i >= 0 && i < h.depthFrames),
    'every source frame needs a depth layer',
  );
  check(
    Array.isArray(h.depthSources) &&
      h.depthSources.length === h.depthFrames &&
      h.depthSources.every((f) => Number.isInteger(f) && f >= 0 && f < h.frames),
    'invalid depth layer sources',
  );
  check(
    h.maskLayers === 0 ? h.depthFrames === h.frames : h.maskLayers === h.frames,
    'person masks must cover every source frame',
  );
  const data = raw.subarray(8 + headerLength);
  check(
    data.byteLength === h.depthFrames * h.width * h.height * 4 + h.maskLayers * h.width * h.height,
    'payload size does not match the header',
  );
  return { header: h, data };
}

const vertexShader = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2DArray;
uniform sampler2DArray depthTex;
uniform sampler2DArray maskTex;
uniform int depthLayer;
uniform int maskLayer;
uniform vec4 intrinsics;   // fx fy cx cy in source pixels
uniform vec2 sourceSize;
uniform vec2 depthRange;   // near far
uniform ivec2 texels;
out vec2 vUv;
out vec2 vTexel;
out float vLogDepth;
out float vSupport;
out float vMask;
#include <fog_pars_vertex>
void main() {
  ivec2 texel = ivec2(int(uv.x * float(texels.x)), int(uv.y * float(texels.y)));
  vec4 t = texelFetch(depthTex, ivec3(texel, depthLayer), 0);
  float q = (t.r * 255.0 + t.g * 255.0 * 256.0) / 65535.0;
  float depth = 1.0 / mix(1.0 / depthRange.y, 1.0 / depthRange.x, q);
  // Unsupported texels (no splat coverage) carry nearest-neighbour filled depth: exact at the
  // recorded pose, and weighted down as the viewer leaves it.
  vSupport = t.a > 0.25 ? 1.0 : 0.0;
  vMask = texelFetch(maskTex, ivec3(texel, maskLayer), 0).r;
  vec2 px = uv * sourceSize;
  vec3 pc = vec3((px.x - intrinsics.z) / intrinsics.x * depth, -(px.y - intrinsics.w) / intrinsics.y * depth, -depth);
  vLogDepth = log(depth);
  vTexel = uv * vec2(texels);
  vUv = vec2(uv.x, 1.0 - uv.y);
  vec4 mvPosition = modelViewMatrix * vec4(pc, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`;

// Shared by both fragment programs: drop triangles that bridge a depth discontinuity. Log depth
// is linear across a triangle, so its slope per texel is exact from the screen derivatives.
// At the recorded pose a stretched triangle projects exactly onto its own pixels, so the cut
// only matters as the viewer leaves the path: the threshold relaxes with the pose weight.
const discardStretched = /* glsl */ `
  mat2 A = mat2(dFdx(vTexel), dFdy(vTexel));
  if (abs(determinant(A)) < 1e-9) discard;
  vec2 slope = vec2(dFdx(vLogDepth), dFdy(vLogDepth)) * inverse(A);
  float slopeMax = max(abs(slope.x), abs(slope.y));
  // A pure turn about the recorded camera reprojects exactly whatever the depth, so cuts are
  // only needed as the viewer TRANSLATES: the threshold relaxes with the distance term alone.
  float threshold = edgeThreshold / max(0.02, 1.0 - translationWeight);
  if (slopeMax > threshold) discard;`;

const colorFragment = /* glsl */ `
precision highp float;
out vec4 fragColor;
#define gl_FragColor fragColor
uniform sampler2D videoTex;
uniform float edgeThreshold;
uniform float poseWeight;
uniform float translationWeight;
uniform float videoPriority;
in vec2 vUv;
in vec2 vTexel;
in float vLogDepth;
in float vSupport;
in float vMask;
#include <fog_pars_fragment>
void main() {
  ${discardStretched}
  gl_FragColor = vec4(texture(videoTex, vUv).rgb, 1.0);
  #include <fog_fragment>
}`;

const weightFragment = /* glsl */ `
precision highp float;
out vec4 fragColor;
uniform float edgeThreshold;
uniform float feather;
uniform float poseWeight;
uniform float translationWeight;
uniform float videoPriority;
in vec2 vUv;
in vec2 vTexel;
in float vLogDepth;
in float vSupport;
in float vMask;
void main() {
  ${discardStretched}
  vec2 e = smoothstep(0.0, feather, vUv) * smoothstep(0.0, feather, 1.0 - vUv);
  // Fade out before the discontinuity cut so the cut lands where the video already yielded to the splats.
  float edgeFade = 1.0 - smoothstep(0.5 * threshold, threshold, slopeMax);
  float support = mix(poseWeight, 1.0, vSupport);
  float w = e.x * e.y * poseWeight * edgeFade * support * (1.0 - smoothstep(0.25, 0.75, vMask));
  fragColor = vec4(w, 0.0, 0.0, 1.0);
}`;

const compositeFragment = /* glsl */ `
precision highp float;
uniform sampler2D withVideo;
uniform sampler2D withoutVideo;
uniform sampler2D weight;
uniform float exposure;  // linear gain on the splat pass so it matches the footage
uniform int debug; // 0 composite, 1 weight, 2 video only, 3 splats only
varying vec2 vUv;
void main() {
  vec3 a = texture2D(withVideo, vUv).rgb, b = texture2D(withoutVideo, vUv).rgb * exposure;
  float w = texture2D(weight, vUv).r;
  vec3 c = mix(b, a, w);
  if (debug == 1) c = vec3(w);
  else if (debug == 2) c = a;
  else if (debug == 3) c = b;
  gl_FragColor = vec4(c, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export type ProjectionDebug = 'off' | 'weight' | 'video' | 'splats';
const DEBUG: Record<ProjectionDebug, number> = { off: 0, weight: 1, video: 2, splats: 3 };
const VIDEO_LAYER = 1;

export class VideoProjectionLayer {
  readonly object = new THREE.Group();
  readonly mesh: THREE.Mesh;
  private readonly material: THREE.ShaderMaterial;
  private readonly weightMaterial: THREE.ShaderMaterial;
  private readonly depthTexture: THREE.DataArrayTexture;
  private readonly maskTexture: THREE.DataArrayTexture;
  private videoTexture: THREE.VideoTexture | null = null;
  private video: HTMLVideoElement | null = null;
  private frame = -1;
  private requested = -1;
  private lastSteer = 0;
  /** The harness sets this around its own seeks so steering never fights a verified seek. */
  suspended = false;
  private poseWeight = 0;
  private targets: {
    a: THREE.WebGLRenderTarget;
    b: THREE.WebGLRenderTarget;
    w: THREE.WebGLRenderTarget;
  } | null = null;
  private readonly compositeMaterial: THREE.ShaderMaterial;
  private readonly compositeScene = new THREE.Scene();
  private readonly compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  debug: ProjectionDebug = 'off';

  constructor(
    readonly header: ProjectionHeader,
    data: Uint8Array,
    readonly options: VideoProjection,
  ) {
    const { width, height, frames, depthFrames } = header;
    const depthBytes = depthFrames * width * height * 4;
    this.depthTexture = new THREE.DataArrayTexture(
      data.subarray(0, depthBytes),
      width,
      height,
      depthFrames,
    );
    this.depthTexture.format = THREE.RGBAFormat;
    this.depthTexture.type = THREE.UnsignedByteType;
    this.depthTexture.minFilter = THREE.NearestFilter;
    this.depthTexture.magFilter = THREE.NearestFilter;
    this.depthTexture.generateMipmaps = false;
    this.depthTexture.needsUpdate = true;
    let masks: Uint8Array;
    if (header.maskLayers) masks = data.subarray(depthBytes);
    else {
      masks = new Uint8Array(frames * width * height);
      for (let i = 0; i < masks.length; i++) masks[i] = data[i * 4 + 2];
    }
    this.maskTexture = new THREE.DataArrayTexture(masks, width, height, frames);
    this.maskTexture.format = THREE.RedFormat;
    this.maskTexture.type = THREE.UnsignedByteType;
    this.maskTexture.minFilter = THREE.NearestFilter;
    this.maskTexture.magFilter = THREE.NearestFilter;
    this.maskTexture.generateMipmaps = false;
    this.maskTexture.needsUpdate = true;

    // One vertex per depth texel at its centre; two triangles per cell.
    const geometry = new THREE.BufferGeometry();
    const uv = new Float32Array(width * height * 2);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 2;
        uv[i] = (x + 0.5) / width;
        uv[i + 1] = (y + 0.5) / height;
      }
    const index = new Uint32Array((width - 1) * (height - 1) * 6);
    let n = 0;
    for (let y = 0; y < height - 1; y++)
      for (let x = 0; x < width - 1; x++) {
        const a = y * width + x,
          b = a + 1,
          c = a + width,
          d = c + 1;
        index[n++] = a;
        index[n++] = c;
        index[n++] = b;
        index[n++] = b;
        index[n++] = c;
        index[n++] = d;
      }
    geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    // Positions come from the depth texture in the vertex shader; three still wants an attribute for bounds.
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(width * height * 3), 3),
    );
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    const uniforms = {
      depthTex: { value: this.depthTexture },
      maskTex: { value: this.maskTexture },
      depthLayer: { value: 0 },
      maskLayer: { value: 0 },
      intrinsics: {
        value: new THREE.Vector4(
          header.intrinsics.fx,
          header.intrinsics.fy,
          header.intrinsics.cx,
          header.intrinsics.cy,
        ),
      },
      sourceSize: { value: new THREE.Vector2(header.source.width, header.source.height) },
      depthRange: { value: new THREE.Vector2(header.depthRange.near, header.depthRange.far) },
      texels: { value: new THREE.Vector2(width, height) },
      edgeThreshold: { value: header.edgeThreshold },
      poseWeight: { value: 0 },
      translationWeight: { value: 0 },
      videoPriority: { value: options.priority === 'video' ? 1 : 0 },
    };
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader: colorFragment,
      fog: true,
      uniforms: {
        ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
        ...uniforms,
        videoTex: { value: null },
      },
      side: THREE.DoubleSide,
    });
    this.weightMaterial = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader: weightFragment,
      uniforms: {
        ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
        ...uniforms,
        feather: { value: options.feather },
      },
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.matrixAutoUpdate = false;
    this.mesh.frustumCulled = false;
    this.mesh.layers.set(VIDEO_LAYER);
    this.object.add(this.mesh);
    this.compositeMaterial = new THREE.ShaderMaterial({
      vertexShader:
        'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: compositeFragment,
      uniforms: {
        withVideo: { value: null },
        withoutVideo: { value: null },
        weight: { value: null },
        exposure: { value: options.exposure ?? 1 },
        debug: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.compositeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial));
    this.setTime(0);
  }

  /** The harness's decoded source element; the same frame the reviewer sees beside the render. */
  attach(video: HTMLVideoElement) {
    if (this.videoTexture?.image === video) return;
    this.videoTexture?.dispose();
    this.video = video;
    this.videoTexture = new THREE.VideoTexture(video);
    this.videoTexture.colorSpace = THREE.SRGBColorSpace;
    this.videoTexture.minFilter = THREE.LinearFilter;
    this.videoTexture.magFilter = THREE.LinearFilter;
    this.videoTexture.generateMipmaps = false;
    this.material.uniforms.videoTex.value = this.videoTexture;
  }

  setTime(time: number) {
    const frame = Math.min(this.header.frames - 1, Math.max(0, Math.round(time * this.header.fps)));
    if (frame === this.frame) return;
    this.frame = frame;
    for (const m of [this.material, this.weightMaterial]) {
      m.uniforms.depthLayer.value = this.header.depthIndex[frame];
      m.uniforms.maskLayer.value = frame;
    }
    const m = this.header.cameras[frame];
    this.mesh.matrix.set(
      m[0],
      m[1],
      m[2],
      m[3],
      m[4],
      m[5],
      m[6],
      m[7],
      m[8],
      m[9],
      m[10],
      m[11],
      m[12],
      m[13],
      m[14],
      m[15],
    );
    this.mesh.matrixWorldNeedsUpdate = true;
  }

  state() {
    return {
      frame: this.frame,
      time: this.frame / this.header.fps,
      depthLayer: this.header.depthIndex[this.frame],
      registered: this.header.registeredFrames.includes(this.frame),
      poseWeight: this.poseWeight,
      debug: this.debug,
      attached: !!this.videoTexture,
      select: this.options.select ?? 'time',
      requested: this.requested,
    };
  }

  /**
   * Pose-driven frame selection: among recorded cameras within selectRadius of the viewer, the one
   * whose optical axis best matches the viewer's, with hysteresis so a steady gaze does not flicker
   * between neighbours. Only steers a paused, non-seeking source; the decoded frame then arrives
   * through the harness's frame callback and setTime() switches the layer.
   */
  private steer(camera: THREE.Camera) {
    if (
      this.options.select !== 'pose' ||
      this.suspended ||
      !this.video ||
      !this.video.paused ||
      this.video.seeking
    )
      return;
    const now = performance.now();
    if (now - this.lastSteer < 150) return;
    this.object.updateMatrixWorld(true);
    const world = this.object.matrixWorld,
      viewPosition = new THREE.Vector3(),
      viewForward = new THREE.Vector3(0, 0, -1);
    camera.getWorldPosition(viewPosition);
    viewForward.transformDirection(camera.matrixWorld);
    const radius = this.options.selectRadius ?? this.options.falloff.distance;
    const position = new THREE.Vector3(),
      forward = new THREE.Vector3();
    let best = -1,
      bestAngle = Infinity,
      nearest = -1,
      nearestDistance = Infinity,
      currentAngle = Infinity;
    for (let i = 0; i < this.header.frames; i++) {
      const m = this.header.cameras[i];
      position.set(m[3], m[7], m[11]).applyMatrix4(world);
      const d = position.distanceTo(viewPosition);
      if (d < nearestDistance) {
        nearestDistance = d;
        nearest = i;
      }
      if (d > radius) continue;
      forward.set(-m[2], -m[6], -m[10]).transformDirection(world);
      const angle = Math.acos(THREE.MathUtils.clamp(forward.dot(viewForward), -1, 1));
      if (i === this.frame) currentAngle = angle;
      if (angle < bestAngle) {
        bestAngle = angle;
        best = i;
      }
    }
    if (best < 0) best = nearest;
    if (best < 0 || best === this.frame || best === this.requested) return;
    // hysteresis: a candidate must beat the shown frame by 5 degrees unless the shown frame is out of range
    if (Number.isFinite(currentAngle) && bestAngle > currentAngle - THREE.MathUtils.degToRad(5))
      return;
    this.requested = best;
    this.lastSteer = now;
    this.video.currentTime = (best + 0.4) / this.header.fps;
  }

  /** Weight from how far the viewer has left the frame's camera: 1 at the recorded pose, 0 at the falloff. */
  private updatePoseWeight(camera: THREE.Camera) {
    this.mesh.updateMatrixWorld(true);
    const framePosition = new THREE.Vector3(),
      frameForward = new THREE.Vector3(0, 0, -1);
    framePosition.setFromMatrixPosition(this.mesh.matrixWorld);
    frameForward.transformDirection(this.mesh.matrixWorld);
    const viewPosition = new THREE.Vector3(),
      viewForward = new THREE.Vector3(0, 0, -1);
    camera.getWorldPosition(viewPosition);
    viewForward.transformDirection(camera.matrixWorld);
    const distance = viewPosition.distanceTo(framePosition);
    const angle = THREE.MathUtils.radToDeg(
      Math.acos(THREE.MathUtils.clamp(viewForward.dot(frameForward), -1, 1)),
    );
    const { distance: d, angle: a } = this.options.falloff;
    const translationWeight = 1 - THREE.MathUtils.smoothstep(distance, 0, d);
    this.poseWeight = translationWeight * (1 - THREE.MathUtils.smoothstep(angle, 0, a));
    for (const m of [this.material, this.weightMaterial]) {
      m.uniforms.poseWeight.value = this.poseWeight;
      m.uniforms.translationWeight.value = translationWeight;
    }
  }

  private ensureTargets(renderer: THREE.WebGLRenderer) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    if (this.targets && this.targets.a.width === size.x && this.targets.a.height === size.y)
      return this.targets;
    this.targets?.a.dispose();
    this.targets?.b.dispose();
    this.targets?.w.dispose();
    const make = (type: THREE.TextureDataType) => {
      const t = new THREE.WebGLRenderTarget(size.x, size.y, {
        type,
        samples: 0,
        depthBuffer: true,
        colorSpace: THREE.LinearSRGBColorSpace,
      });
      t.texture.minFilter = THREE.NearestFilter;
      t.texture.magFilter = THREE.NearestFilter;
      t.texture.generateMipmaps = false;
      return t;
    };
    this.targets = {
      a: make(THREE.HalfFloatType),
      b: make(THREE.HalfFloatType),
      w: make(THREE.UnsignedByteType),
    };
    this.compositeMaterial.uniforms.withVideo.value = this.targets.a.texture;
    this.compositeMaterial.uniforms.withoutVideo.value = this.targets.b.texture;
    this.compositeMaterial.uniforms.weight.value = this.targets.w.texture;
    return this.targets;
  }

  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera) {
    if (!this.videoTexture) throw new Error('Video projection has no source video attached');
    const targets = this.ensureTargets(renderer);
    this.steer(camera);
    this.updatePoseWeight(camera);
    const layers = camera.layers.mask,
      target = renderer.getRenderTarget();
    const clearColor = renderer.getClearColor(new THREE.Color()),
      clearAlpha = renderer.getClearAlpha();
    try {
      // Pass A: with 'video' priority the surface is drawn alone, so no splat can stand in front
      // of the recorded pixels; with 'depth' priority nearer scene content draws over it.
      if (this.options.priority === 'video') camera.layers.set(VIDEO_LAYER);
      else {
        camera.layers.enable(0);
        camera.layers.enable(VIDEO_LAYER);
      }
      renderer.setRenderTarget(targets.a);
      renderer.render(scene, camera);
      camera.layers.enable(0);
      camera.layers.disable(VIDEO_LAYER);
      renderer.setRenderTarget(targets.b);
      renderer.render(scene, camera);
      camera.layers.set(VIDEO_LAYER);
      this.mesh.material = this.weightMaterial;
      renderer.setClearColor(0x000000, 1);
      renderer.setRenderTarget(targets.w);
      renderer.render(scene, camera);
    } finally {
      this.mesh.material = this.material;
      camera.layers.mask = layers;
      renderer.setClearColor(clearColor, clearAlpha);
      renderer.setRenderTarget(target);
    }
    this.compositeMaterial.uniforms.debug.value = DEBUG[this.debug];
    renderer.render(this.compositeScene, this.compositeCamera);
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.weightMaterial.dispose();
    this.depthTexture.dispose();
    this.maskTexture.dispose();
    this.videoTexture?.dispose();
    this.videoTexture = null;
    this.targets?.a.dispose();
    this.targets?.b.dispose();
    this.targets?.w.dispose();
    this.targets = null;
    this.compositeMaterial.dispose();
    (this.compositeScene.children[0] as THREE.Mesh).geometry.dispose();
  }
}

export async function loadVideoProjection(
  bytes: ArrayBuffer,
  experiment: { source: ProjectionSource },
  options: VideoProjection,
): Promise<VideoProjectionLayer> {
  const { header, data } = await parseProjectionContainer(bytes, experiment.source);
  return new VideoProjectionLayer(header, data, options);
}
