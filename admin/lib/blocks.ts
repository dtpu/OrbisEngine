import type { RunArtifact } from './types';

/**
 * What an artifact *is*, as the operator reads it, rather than what the HTTP layer calls it.
 *
 * The pipeline hands almost every 3D output over as `application/octet-stream`, so the file name
 * has to decide where the media type cannot: `world.spz` is a world you orbit, `masks.npz` is an
 * array nobody can look at, and both arrive as the same opaque type. Every block on the page --
 * the tile in a graph node, the filmstrip thumbnail, the full panel in the inspector -- is chosen
 * from this one classification, so a stage's output looks the same wherever it is shown.
 */
export type BlockType =
  | 'image'
  | 'video'
  | 'audio'
  | 'splat'
  | 'points'
  | 'mesh'
  | 'tensor'
  | 'data'
  | 'text'
  | 'archive'
  | 'binary';

export interface BlockKind {
  type: BlockType;
  /** The word under the block: what the operator is looking at, in their language. */
  noun: string;
  /** Mono mark drawn when the block has no picture of its own. */
  mark: string;
  /** True when the full-size block draws from the bytes and needs its own WebGL canvas. */
  canvas: boolean;
}

/** Formats that are Gaussian splats by definition. */
const SPLAT = new Set(['spz', 'splat', 'ksplat']);
/** Polygon models. Nothing in the pipeline emits these yet; hand-uploaded references do. */
const MESH = new Set(['glb', 'gltf', 'obj', 'stl', 'fbx']);
/** Arrays with no picture: masks, patches, anchors, weights. */
const TENSOR = new Set(['npz', 'npy', 'pt', 'pth', 'safetensors']);
/**
 * Readable files the API cannot serve inline. Anything outside its short allowlist is downgraded
 * to `application/octet-stream` on the way out, which is why a stage log arrives looking binary.
 */
const TEXT = new Set(['log', 'txt', 'md', 'csv', 'tsv', 'yaml', 'yml', 'jsonl', 'ndjson']);
const ARCHIVE = new Set(['gz', 'tgz', 'zip', 'tar', 'bz2', 'xz', 'zst']);

export function extensionOf(artifact: RunArtifact): string {
  const name = (artifact.name ?? '').toLowerCase();
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1) : '';
}

export function classify(artifact: RunArtifact): BlockKind {
  const media = artifact.mediaType.split(';', 1)[0].trim().toLowerCase();
  const extension = extensionOf(artifact);

  // File name first: a `.spz` served as octet-stream is still a world.
  if (SPLAT.has(extension)) {
    return {
      type: 'splat',
      noun: extension === 'spz' ? '3D world' : '3D splats',
      mark: '◈',
      canvas: true,
    };
  }
  // `.ply` says nothing about what is inside it: this pipeline writes both Gaussian objects and
  // plain coloured point clouds under that extension. Only the header settles it, so the block
  // reads it before choosing a renderer.
  if (extension === 'ply') return { type: 'points', noun: '3D points', mark: '◈', canvas: true };
  if (MESH.has(extension)) return { type: 'mesh', noun: '3D model', mark: '◇', canvas: true };
  if (TENSOR.has(extension)) return { type: 'tensor', noun: 'arrays', mark: '▦', canvas: false };
  if (ARCHIVE.has(extension)) return { type: 'archive', noun: 'archive', mark: '▤', canvas: false };
  if (TEXT.has(extension)) {
    return { type: 'text', noun: extension === 'log' ? 'log' : 'text', mark: '¶', canvas: false };
  }

  // SVG is markup, not a picture: it is never rendered inline under this origin.
  if (media.startsWith('image/') && media !== 'image/svg+xml') {
    return { type: 'image', noun: 'frame', mark: '▣', canvas: false };
  }
  if (media.startsWith('video/')) return { type: 'video', noun: 'clip', mark: '▶', canvas: false };
  if (media.startsWith('audio/')) return { type: 'audio', noun: 'audio', mark: '♪', canvas: false };
  if (media === 'application/json' || media.endsWith('+json')) {
    return { type: 'data', noun: 'report', mark: '{ }', canvas: false };
  }
  if (media.startsWith('text/')) return { type: 'text', noun: 'text', mark: '¶', canvas: false };
  return { type: 'binary', noun: 'file', mark: '·', canvas: false };
}

/**
 * Which artifact a stage leads with: the thing an operator judges by eye. A world or a clip beats
 * a still, a still beats a report, and an opaque blob comes last however big it is.
 */
const LEAD: BlockType[] = [
  'splat',
  'points',
  'mesh',
  'video',
  'image',
  'audio',
  'data',
  'text',
  'tensor',
  'archive',
  'binary',
];

/** Block types that put a picture on the screen rather than a number or a word. */
const VISUAL = new Set<BlockType>(['splat', 'points', 'mesh', 'video', 'image']);

export function isVisual(artifact: RunArtifact): boolean {
  return VISUAL.has(classify(artifact).type);
}

export function leadArtifact(artifacts: RunArtifact[]): RunArtifact | undefined {
  return [...artifacts].sort(
    (a, b) => LEAD.indexOf(classify(a).type) - LEAD.indexOf(classify(b).type),
  )[0];
}

/** Vertices and faces a `.ply` declares, so a tile can say how big the cloud is. */
export function plyCounts(header: string): { vertices: number | null; faces: number | null } {
  const vertices = /^element\s+vertex\s+(\d+)/m.exec(header);
  const faces = /^element\s+face\s+(\d+)/m.exec(header);
  return {
    vertices: vertices ? Number(vertices[1]) : null,
    faces: faces ? Number(faces[1]) : null,
  };
}

export type PlyFlavour = 'gaussian' | 'mesh' | 'points';

/**
 * What a `.ply` actually holds, read from its ASCII header.
 *
 * A Gaussian PLY carries the spherical-harmonic and covariance properties a splat renderer needs
 * (`f_dc_*`, `scale_*`, `rot_*`); a mesh declares faces; anything else is a bare point cloud.
 * Guessing from the extension gets this wrong in both directions -- the person frames this
 * pipeline writes are `x y z red green blue`, not Gaussians.
 */
export function plyFlavour(header: string): PlyFlavour {
  const properties = new Set(
    header
      .split('\n')
      .filter((line) => line.startsWith('property '))
      .map((line) => line.trim().split(/\s+/).pop() ?? ''),
  );
  if (properties.has('f_dc_0') && properties.has('scale_0') && properties.has('rot_0')) {
    return 'gaussian';
  }
  const faces = /^element\s+face\s+(\d+)/m.exec(header);
  return faces && Number(faces[1]) > 0 ? 'mesh' : 'points';
}
