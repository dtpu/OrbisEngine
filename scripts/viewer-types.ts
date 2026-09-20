// The public diagnostic subset consumed after the live viewer reports ready.
import type * as THREE from 'three';
import type { FourDAudio } from '../src/audio/fourd-audio.ts';

export interface WalkGridSummary {
  cell: number;
  width: number;
  height: number;
  hullCells: number;
  coverage: number;
  source: string;
}
export interface WalkDiagnostics {
  floor: number;
  stature: number;
  stepUp: number;
  grid: WalkGridSummary | null;
  cellAt(
    x: number,
    z: number,
    maximumSupport?: number,
  ): { floor: number; dist: number; inside: boolean };
  blockedAt(x: number, z: number, floor?: number): number;
  advance(from: THREE.Vector3, delta: THREE.Vector3, dt: number): THREE.Vector3;
}
/** One window of a merged shot sequence, as `wander.shots.windows` reports it. */
export interface ShotWindowSummary {
  index: number;
  world: string;
  sourceStart: number;
  sourceEnd: number;
}
/**
 * Present only for a package from scripts/package_shot_sequence.py: several worlds on one source
 * clock. `active` is -1 while the clip is in a gap, where no shot was ever reconstructed.
 */
export interface ShotDiagnostics {
  count: number;
  windows: ShotWindowSummary[];
  gaps: { start: number; end: number }[];
  /** What the viewer shows in a gap. 'blank' hides the world outright. */
  gapMode: 'blank' | 'hold-dimmed';
  readonly active: number;
  /** Shot indices whose world is resident in memory, most recently used first. */
  readonly resident: number[];
  at(seconds: number): number;
}
export interface ViewerDiagnostics {
  ready: boolean;
  demo: string | null;
  camera: THREE.PerspectiveCamera;
  THREE: typeof THREE;
  spark: { renderer: THREE.WebGLRenderer };
  people: { loaded: number; nF: number }[];
  walk: WalkDiagnostics | null;
  /** null for every scene that is not a merged shot sequence. */
  shots: ShotDiagnostics | null;
  /** The world currently on screen; a shot sequence swaps it at every cut. */
  world: { numSplats: number } | null;
  /** The active world's floor height, which changes with the world. */
  floorY: number;
  clampOn: boolean;
  clampBox: THREE.Box3;
  upm: number;
  edge: number;
  playing: boolean;
  t: number;
  dur: number;
  video: HTMLVideoElement;
  audioState: FourDAudio['state'];
  play(value: boolean): void;
  setTime(value: number): void;
}
declare global {
  interface Window {
    wander: ViewerDiagnostics;
    __fakeXR: {
      frames: number[];
      head: { x: number; y: number; z: number; yaw: number; pitch: number };
      axes: { left: number[]; right: number[] };
      buttons: { left: number[]; right: number[] };
    };
    __xr: unknown;
  }
}
