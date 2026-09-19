export type Position = [number, number, number];
export type Quaternion = [number, number, number, number];
export interface Pose { position: Position; quaternion: Quaternion }
export interface Checkpoint extends Pose { id: string; label: string; time: number; route: Pose[] }
export interface Benchmark {
  version: 1; id: string; sourceSha256: string; coordinates: string;
  camera: { fov: number; aspect: number; near: number; far: number; movementStep: number };
  checkpoints: Checkpoint[];
}
export interface Asset { url: string; sha256?: string }
export interface ConfidenceFog { color: string; density: number }
/** Appearance of the measured-clearance contact shadow; the measurements themselves live in `url`. */
export interface ContactShadow {
  url: string; sha256?: string; color?: string;
  contactClearance?: number; maxClearance?: number;
  maxOpacity?: number; minOpacity?: number; spread?: number;
  /** Multiplies the measured footprint. The mark a shoe casts is not the shoe's own outline. */
  footprintScale?: number;
}
/**
 * How sharply to draw this output. Every field is optional but at least one must be present,
 * and each is validated independently so a partial declaration cannot silently fall back.
 *
 * `pixelRatio` is the inspection canvas backing-store ratio; absent, the viewer keeps its own
 * min(2, devicePixelRatio). `splatBlur` and `splatFocalAdjustment` are Spark's `blurAmount` and
 * `focalAdjustment`: Spark's stock defaults (0.3 / 1.0) assume a trainer that baked the
 * anti-aliasing dilation into the Gaussians, and a scene trained without it is dilated twice.
 */
export interface RenderQuality { pixelRatio?: number; splatBlur?: number; splatFocalAdjustment?: number }
export interface Transform { position: Position; quaternion: Quaternion; scale: number }
/** Gaussians baked from recorded pixels warped out of neighbouring frames of the same walk. */
export interface WarpComposite extends Asset {
  sha256: string; provenance: 'recorded-warp-v1';
  donorFrames: number; splats: number; transform?: Transform;
}
export interface VideoProjection extends Asset {
  sha256: string;
  // Alignment of the projection cameras relative to the output root, like a composite component.
  transform?: { position: Position; quaternion: Quaternion; scale: number };
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
export interface Experiment {
  version: 1; id: string; title: string; hypothesis: string; limitations: string;
  source: Asset & { sha256: string; width: number; height: number; fps: number; duration: number };
  benchmark: string;
  output: Asset & {
    format: 'glb' | 'ply' | 'gaussian-ply' | 'composite'; timing: { kind: 'static-environment' } | { kind: 'dynamic' } | { kind: 'frozen'; time: number };
    coverage: { kind: 'observed' | 'inferred' | 'mixed' | 'unknown'; description: string };
    vertexColorSpace: 'srgb' | 'linear'; doubleSided: boolean;
    // Optional alignment is explicit: never auto-center or rescale an experiment.
    transform?: { position: Position; quaternion: Quaternion; scale: number };
    pointSize?: number;
    sizeAttenuation?: boolean;
    // Absent retains the installed Spark encoder for reproducible older scenes.
    shPacking?: 'ceil-v1';
    // Absent renders the trained footprint alone; present appends warped recorded pixels
    // beyond it. Only recorded colour is ever stored, so no signage or text can be invented.
    warpComposite?: WarpComposite;
    // Absent casts nothing, which is the previous behaviour of every published scene.
    contactShadow?: ContactShadow;
    // Absent keeps the viewer's own pixel ratio and the installed Spark splat defaults.
    renderQuality?: RenderQuality;
    // Absent renders at full contrast and opacity; present attenuates mesh and splats alike.
    confidenceFog?: ConfidenceFog;
    // Absent renders splats everywhere; present shows the recorded frame on splat-rendered depth near the path.
    videoProjection?: VideoProjection;
  };
}
function check(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const positive = (x: unknown) => finite(x) && x > 0;
const text = (x: unknown) => typeof x === 'string' && x.trim().length > 0;
const vector = (x: unknown, n: number) => Array.isArray(x) && x.length === n && x.every(finite);
const quaternion = (x: unknown) => vector(x, 4) && Math.abs(Math.hypot(...x as number[]) - 1) < 1e-6;
const pose = (x: Pose) => !!x && vector(x.position, 3) && quaternion(x.quaternion);
const hash = (x: unknown) => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const asset = (x: Asset) => !!x && text(x.url) && (x.sha256 === undefined || hash(x.sha256));

const unit = (x: unknown) => x === undefined || (finite(x) && x > 0 && x <= 1);
function contactShadow(x: ContactShadow) {
  if (!asset(x)) return false;
  if (x.color !== undefined && !/^#[0-9a-f]{6}$/.test(x.color)) return false;
  if (![x.contactClearance, x.maxClearance].every(v => v === undefined || (finite(v) && v >= 0))) return false;
  // A shadow that grew darker as the foot rose would advertise float as contact.
  if (!unit(x.maxOpacity) || !unit(x.minOpacity)) return false;
  if (x.maxOpacity !== undefined && x.minOpacity !== undefined && x.minOpacity > x.maxOpacity) return false;
  if (x.spread !== undefined && !(finite(x.spread) && x.spread >= 1)) return false;
  if (x.footprintScale !== undefined && !(finite(x.footprintScale) && x.footprintScale > 0)) return false;
  return (x.contactClearance ?? 0) < (x.maxClearance ?? Infinity);
}

const range = (x: unknown, lo: number, hi: number) => x === undefined || (finite(x) && x >= lo && x <= hi);
const renderQuality = (q: RenderQuality) =>
  !!q && typeof q === 'object' && Object.keys(q).every(k => ['pixelRatio', 'splatBlur', 'splatFocalAdjustment'].includes(k))
  && [q.pixelRatio, q.splatBlur, q.splatFocalAdjustment].some(x => x !== undefined)
  && range(q.pixelRatio, 1, 3) && range(q.splatBlur, 0, 1) && range(q.splatFocalAdjustment, 0.5, 4);

export function parseExperiment(value: unknown): Experiment {
  const e = value as Experiment;
  check(e?.version === 1 && [e.id, e.title, e.hypothesis, e.limitations, e.benchmark].every(text), 'Invalid experiment identity, hypothesis, limitations, or benchmark URL');
  check(asset(e.source) && hash(e.source.sha256) && [e.source.width, e.source.height].every(x => Number.isInteger(x) && x > 0) && positive(e.source.fps) && positive(e.source.duration), 'Invalid source video metadata or SHA-256');
  const o = e.output;
  check(asset(o) && ['glb', 'ply', 'gaussian-ply', 'composite'].includes(o.format), 'Output must be GLB, PLY, Gaussian PLY, or a composite');
  check(o.timing?.kind === 'static-environment' || o.timing?.kind === 'dynamic' || (o.timing?.kind === 'frozen' && finite(o.timing.time) && o.timing.time >= 0 && o.timing.time < e.source.duration), 'Output timing must declare static, dynamic, or a valid frozen time');
  check(o.coverage && ['observed', 'inferred', 'mixed', 'unknown'].includes(o.coverage.kind) && text(o.coverage.description), 'Declare observed/inferred coverage and its limitations');
  check(['srgb', 'linear'].includes(o.vertexColorSpace) && typeof o.doubleSided === 'boolean', 'Declare vertex color space and surface sidedness');
  check(o.transform === undefined || (pose(o.transform) && positive(o.transform.scale)), 'Invalid output alignment');
  const w = o.warpComposite;
  check(w === undefined || (asset(w) && hash(w.sha256) && w.provenance === 'recorded-warp-v1'
    && Number.isInteger(w.donorFrames) && w.donorFrames > 0 && Number.isInteger(w.splats) && w.splats > 0
    && (w.transform === undefined || (pose(w.transform) && positive(w.transform.scale)))
    && o.format === 'composite'), 'Warp composite needs recorded-warp-v1 provenance, a SHA-256, positive donor and splat counts, and a composite output');
  check(o.contactShadow === undefined || contactShadow(o.contactShadow), 'Contact shadow requires a measurement URL, an optional #rrggbb colour, clearances in ascending order and opacities within 0..1');
  check(o.renderQuality === undefined || renderQuality(o.renderQuality), 'Render quality needs at least one of pixelRatio 1-3, splatBlur 0-1 or splatFocalAdjustment 0.5-4');
  check(o.pointSize === undefined || positive(o.pointSize), 'Invalid point size');
  check(o.sizeAttenuation === undefined || typeof o.sizeAttenuation === 'boolean', 'Invalid point attenuation');
  check(o.shPacking === undefined || (o.shPacking === 'ceil-v1' && o.format === 'gaussian-ply' && hash(o.sha256)), 'SH packing requires ceil-v1, Gaussian PLY and an asset SHA-256');
  check(o.confidenceFog === undefined || (/^#[0-9a-f]{6}$/.test(o.confidenceFog.color) && positive(o.confidenceFog.density)), 'Confidence fog requires a #rrggbb colour and a positive density');
  const v = o.videoProjection;
  check(v === undefined || (asset(v) && hash(v.sha256) && ['gaussian-ply', 'composite'].includes(o.format)
    && (v.transform === undefined || (pose(v.transform) && positive(v.transform.scale)))
    && positive(v.falloff?.distance) && positive(v.falloff?.angle) && v.falloff.angle < 180
    && finite(v.feather) && v.feather >= 0 && v.feather < 0.5
    && (v.priority === undefined || ['depth', 'video'].includes(v.priority))
    && (v.select === undefined || ['time', 'pose'].includes(v.select)) && (v.selectRadius === undefined || positive(v.selectRadius))
    && (v.exposure === undefined || (positive(v.exposure) && v.exposure <= 4))),
    'Video projection requires a SHA-256 asset on a splat or composite output, a valid alignment, positive falloff distance, a falloff angle below 180 degrees, a feather below one half, a priority of depth or video, a select of time or pose, a positive select radius and an exposure gain between 0 and 4');
  return e;
}

export function parseBenchmark(value: unknown, experiment: Experiment): Benchmark {
  const b = value as Benchmark;
  check(b?.version === 1 && text(b.id) && text(b.coordinates) && b.sourceSha256 === experiment.source.sha256, 'Benchmark does not match the source footage');
  const c = b.camera;
  check(c && [c.fov, c.aspect, c.near, c.far, c.movementStep].every(positive) && c.fov < 170 && c.far > c.near, 'Invalid benchmark camera');
  check(Array.isArray(b.checkpoints) && b.checkpoints.length > 0, 'Benchmark has no checkpoints');
  const ids = new Set<string>();
  for (const p of b.checkpoints) {
    check(pose(p) && typeof p.id === 'string' && /^[a-z0-9-]+$/.test(p.id) && !ids.has(p.id) && text(p.label), 'Invalid or duplicate checkpoint');
    check(finite(p.time) && p.time >= 0 && p.time < experiment.source.duration, `Invalid source time at ${p.id}`);
    check(Array.isArray(p.route) && p.route.length > 0 && p.route.every(pose), `Missing or invalid movement route at ${p.id}`);
    if (experiment.output.timing.kind === 'frozen') check(Math.abs(p.time - experiment.output.timing.time) < 1e-6, `Frozen output cannot be compared to source time ${p.time} at ${p.id}`);
    ids.add(p.id);
  }
  return b;
}

export function resolveAsset(relative: string, base: string): string {
  const url = new URL(relative, base);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Assets must use HTTP(S) URLs');
  if (url.username || url.password) throw new Error('Do not put credentials in asset URLs');
  return url.href;
}

export interface AssetRecord { url: string; bytes: number; sha256: string }
export async function fetchAsset(url: string, records: AssetRecord[], expectedHash?: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Asset ${response.status}: ${url}`);
  if (response.headers.get('content-type')?.includes('text/html')) throw new Error(`Asset missing or returned HTML: ${url}`);
  const bytes = await response.arrayBuffer();
  if (!bytes.byteLength) throw new Error(`Empty asset: ${url}`);
  const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
  if (expectedHash && sha256 !== expectedHash) throw new Error(`SHA-256 mismatch: ${url}`);
  records.push({ url, bytes: bytes.byteLength, sha256 });
  return bytes;
}
export const readJson = (bytes: ArrayBuffer): unknown => JSON.parse(new TextDecoder().decode(bytes));
