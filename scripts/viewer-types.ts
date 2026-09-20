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
  cellAt(x: number, z: number): { floor: number; dist: number; inside: boolean };
  blockedAt(x: number, z: number): boolean | 0 | 1;
}
export interface ViewerDiagnostics {
  ready: boolean;
  demo: string | null;
  camera: THREE.PerspectiveCamera;
  THREE: typeof THREE;
  spark: { renderer: THREE.WebGLRenderer };
  people: { loaded: number; nF: number }[];
  walk: WalkDiagnostics | null;
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
    };
    __xr: unknown;
  }
}
