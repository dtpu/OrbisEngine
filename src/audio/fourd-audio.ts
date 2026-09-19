import * as THREE from 'three';

export type Track = {
  id: string;
  kind: 'dialogue' | 'ambience';
  url: string;
  offsetSeconds?: number;
  personId?: string;
  reviewed?: boolean;
  provenance?: string;
  anchor?: { url: string; space: 'person-local'; offsetBodyHeights?: number[] };
};
export type AudioManifest = {
  schema: 'wander.audio/1';
  source: { hasAudio: boolean };
  timeline: { durationSeconds: number };
  original?: { url: string; offsetSeconds?: number };
  provenance?: { attribution?: string; license?: string; licenseUrl?: string };
  spatialMixComplete?: boolean;
  defaultMode?: 'original' | 'spatial';
  tracks?: Track[];
};
type Person = { id: string; group: THREE.Object3D; ts: number[]; bodyH: number };
type Anchor = { positions: number[][]; times: number[]; offset: number[]; person: Person };
type Loaded = { track: Track; buffer: AudioBuffer; anchor?: Anchor; panner?: PannerNode };
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

export function validateManifest(
  value: unknown,
  duration: number,
  personIds: string[],
): AudioManifest {
  const m = value as AudioManifest;
  if (
    !m ||
    m.schema !== 'wander.audio/1' ||
    typeof m.source?.hasAudio !== 'boolean' ||
    !finite(m.timeline?.durationSeconds) ||
    Math.abs(m.timeline.durationSeconds - duration) > 0.1
  )
    throw new Error('Audio manifest does not match the clip timeline');
  const validOffset = (x: unknown) => x === undefined || finite(x);
  if (
    m.original &&
    (typeof m.original.url !== 'string' ||
      !m.original.url ||
      !validOffset(m.original.offsetSeconds))
  )
    throw new Error('Invalid original audio track');
  if (m.tracks !== undefined && !Array.isArray(m.tracks)) throw new Error('Invalid audio tracks');
  const ids = new Set<string>();
  for (const t of m.tracks ?? []) {
    if (
      !t ||
      typeof t.id !== 'string' ||
      ids.has(t.id) ||
      typeof t.url !== 'string' ||
      !t.url ||
      !validOffset(t.offsetSeconds)
    )
      throw new Error('Invalid or duplicate audio track');
    ids.add(t.id);
    if (t.kind !== 'dialogue' && t.kind !== 'ambience') throw new Error('Unknown audio track kind');
    if (
      t.kind === 'dialogue' &&
      (t.reviewed !== true ||
        !t.provenance ||
        !personIds.includes(t.personId!) ||
        t.anchor?.space !== 'person-local' ||
        typeof t.anchor.url !== 'string')
    )
      throw new Error('Dialogue needs a reviewed speaker and head anchor');
    const o = t.anchor?.offsetBodyHeights;
    if (o && (o.length !== 3 || !o.every(finite))) throw new Error('Invalid head offset');
  }
  return m;
}

export function validateAnchor(
  data: { positions?: number[][]; eyes?: number[][]; times?: number[] },
  fallbackTimes: number[],
) {
  const positions = data.positions ?? data.eyes;
  const times = data.times ?? fallbackTimes;
  if (
    !positions?.length ||
    positions.length !== times.length ||
    !positions.every((p) => p.length === 3 && p.every(finite)) ||
    !times.every((t, i) => finite(t) && (i === 0 || t > times[i - 1]))
  )
    throw new Error('Invalid head timeline');
  return { positions, times };
}

export function sampleAnchor(
  positions: number[][],
  times: number[],
  time: number,
  out: THREE.Vector3,
) {
  let i = 0;
  while (i + 1 < times.length && times[i + 1] <= time) i++;
  const j = Math.min(i + 1, times.length - 1);
  const u = j === i ? 0 : THREE.MathUtils.clamp((time - times[i]) / (times[j] - times[i]), 0, 1);
  return out.fromArray(positions[i]).lerp(new THREE.Vector3().fromArray(positions[j]), u);
}

/** One output owner. Video is the clock; external tracks share a single buffer schedule. */
export class FourDAudio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private originalGain: GainNode | null = null;
  private mediaSource: MediaElementAudioSourceNode | null = null;
  private original: AudioBuffer | null = null;
  private loaded: Loaded[] = [];
  private voices: AudioBufferSourceNode[] = [];
  private startedAt = 0;
  private clipAt = 0;
  private scheduledRate = 1;
  private lastTime = -1;
  private manifest: AudioManifest | null = null;
  private dead = false;
  private loading = false;
  private spatialReady = false;
  private unlocked = false;
  private error = '';
  private muted = true;
  private mode: 'original' | 'spatial' = 'original';
  private suspended = false;
  private abort = new AbortController();
  private pos = new THREE.Vector3();
  private forward = new THREE.Vector3();
  private up = new THREE.Vector3();
  private lastLabel = '';
  private creditKey = '';
  private credit: HTMLDivElement;
  readonly button: HTMLButtonElement;
  readonly modeButton: HTMLButtonElement;
  private listeners: (() => void)[] = [];

  constructor(
    private video: HTMLVideoElement,
    private duration: number,
    private people: Person[],
    private bodyHeight: () => number,
  ) {
    this.button = document.createElement('button');
    this.modeButton = document.createElement('button');
    const controls = document.createElement('div');
    controls.id = 'audio-controls';
    controls.style.cssText =
      'position:fixed;right:16px;bottom:54px;z-index:30;display:flex;gap:6px;font:13px system-ui';
    for (const b of [this.button, this.modeButton]) {
      b.style.cssText =
        'background:#151b24;color:white;border:1px solid #667080;border-radius:8px;padding:9px 12px;cursor:pointer';
      controls.append(b);
    }
    this.button.addEventListener('click', () => {
      if (this.muted || !this.unlocked) {
        this.setMuted(false);
        void this.unlock();
      } else this.setMuted(true);
    });
    this.modeButton.addEventListener('click', () =>
      this.setMode(this.mode === 'original' ? 'spatial' : 'original'),
    );
    this.credit = document.createElement('div');
    this.credit.id = 'audio-credit';
    this.credit.hidden = true;
    this.credit.style.cssText =
      'position:fixed;right:16px;bottom:18px;z-index:30;max-width:70vw;padding:5px 8px;border-radius:5px;background:#111d;color:#ddd;font:11px system-ui';
    document.body.append(controls, this.credit);
    for (const event of ['pause', 'seeking', 'waiting', 'ended', 'ratechange', 'emptied']) {
      const fn = () => this.invalidate();
      video.addEventListener(event, fn);
      this.listeners.push(() => video.removeEventListener(event, fn));
    }
    this.paint();
  }

  get state() {
    return {
      muted: this.muted,
      unlocked: this.unlocked,
      mode: this.mode,
      spatialReady: this.spatialReady,
      loading: this.loading,
      hasAudio: this.manifest
        ? this.manifest.source.hasAudio || !!this.original || this.spatialReady
        : null,
      context: this.ctx?.state ?? 'locked',
      activeSources: this.voices.length,
      error: this.error,
      driftSeconds:
        this.voices.length && this.ctx
          ? this.clipAt +
            (this.ctx.currentTime - this.startedAt) * this.scheduledRate -
            this.video.currentTime
          : 0,
    };
  }

  private context() {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0;
      this.master.connect(this.ctx.destination);
      this.originalGain = this.ctx.createGain();
      this.originalGain.gain.value = 0;
      this.originalGain.connect(this.master);
    }
    return this.ctx;
  }

  async load(url: string) {
    this.loading = true;
    this.paint();
    try {
      const response = await fetch(url, { signal: this.abort.signal });
      if (response.status === 404) return; // Optional package; embedded source remains usable.
      if (!response.ok) throw new Error('Audio package unavailable');
      const raw = await response.json();
      // Validate the original independently: malformed speaker metadata must not discard it.
      let m = validateManifest(
        { ...raw, tracks: undefined, spatialMixComplete: false, defaultMode: 'original' },
        this.duration,
        this.people.map((p) => p.id),
      );
      try {
        m = validateManifest(
          raw,
          this.duration,
          this.people.map((p) => p.id),
        );
      } catch {
        this.error = 'Spatial manifest invalid; using original soundtrack';
      }
      if (this.dead) return;
      this.manifest = m;
      const base = new URL(url, location.href);
      const decode = async (path: string) => {
        const r = await fetch(new URL(path, base), { signal: this.abort.signal });
        if (!r.ok) throw new Error('Audio track unavailable');
        const bytes = await r.arrayBuffer();
        if (this.dead) throw new Error('Audio disposed');
        return this.context().decodeAudioData(bytes);
      };
      if (m.original) {
        try {
          this.original = await decode(m.original.url);
        } catch {
          this.error = 'Original audio unavailable';
        }
      }
      if (m.spatialMixComplete === true && m.tracks?.some((t) => t.kind === 'dialogue')) {
        try {
          const loaded: Loaded[] = [];
          for (const track of m.tracks) {
            const buffer = await decode(track.url);
            if (this.dead) return;
            if (track.kind === 'dialogue' && buffer.numberOfChannels !== 1)
              throw new Error('Dialogue stem must be mono');
            let anchor: Anchor | undefined;
            if (track.kind === 'dialogue') {
              const person = this.people.find((p) => p.id === track.personId)!;
              const r = await fetch(new URL(track.anchor!.url, base), {
                signal: this.abort.signal,
              });
              if (!r.ok) throw new Error('Speaker head track unavailable');
              anchor = {
                ...validateAnchor(await r.json(), person.ts),
                person,
                offset: track.anchor!.offsetBodyHeights ?? [0, 0, 0],
              };
            }
            loaded.push({ track, buffer, anchor });
          }
          if (this.dead) return;
          this.loaded = loaded;
          this.spatialReady = true;
          if (m.defaultMode === 'spatial') this.mode = 'spatial';
        } catch {
          this.error = 'Spatial dialogue unavailable; using original soundtrack';
          this.mode = 'original';
        }
      }
      this.invalidate();
      this.route();
    } catch (e) {
      if (!this.dead) this.error = e instanceof Error ? e.message : 'Audio unavailable';
    } finally {
      this.loading = false;
      if (!this.dead) this.paint();
    }
  }

  async unlock() {
    if (this.dead) return;
    const ctx = this.context();
    if (this.unlocked && ctx.state === 'running') return;
    const firstUnlock = !this.unlocked;
    if (!this.mediaSource) {
      this.mediaSource = ctx.createMediaElementSource(this.video);
      this.mediaSource.connect(this.originalGain!);
    }
    // Invoke resume in the trusted event stack, not after a fetch/session request.
    const pending = ctx.resume();
    this.unlocked = true;
    if (firstUnlock) this.muted = false;
    this.video.muted = false;
    this.route();
    try {
      await pending;
      if (ctx.state !== 'running') throw new Error('Audio suspended');
    } catch {
      this.unlocked = false;
      this.muted = true;
      this.error = 'Tap Enable sound to allow audio';
    }
    this.route();
    this.paint();
  }

  setMuted(value: boolean) {
    this.muted = value;
    this.invalidate();
    this.route();
    this.paint();
  }
  setMode(value: 'original' | 'spatial') {
    this.mode = value === 'spatial' && this.spatialReady ? 'spatial' : 'original';
    this.invalidate();
    this.route();
    this.paint();
  }
  setSuspended(value: boolean) {
    this.suspended = value;
    this.invalidate();
    this.route();
  }
  invalidate() {
    for (const v of this.voices) {
      try {
        v.stop();
      } catch {
        /* already ended */
      }
      v.disconnect();
    }
    this.voices = [];
    this.lastTime = -1;
  }

  private route() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const audible = this.unlocked && !this.muted && !this.suspended;
    this.master!.gain.cancelScheduledValues(now);
    this.master!.gain.setTargetAtTime(audible ? 1 : 0, now, 0.008);
    this.originalGain!.gain.value =
      this.mode === 'original' && !this.original && this.manifest?.source.hasAudio !== false
        ? 1
        : 0;
  }

  tick(time: number, running: boolean, position: THREE.Vector3, rotation: THREE.Quaternion) {
    if (this.dead || !this.ctx) return;
    const ctx = this.ctx,
      now = ctx.currentTime;
    const unit = Math.max(1e-6, this.bodyHeight());
    const l = ctx.listener;
    this.forward.set(0, 0, -1).applyQuaternion(rotation);
    this.up.set(0, 1, 0).applyQuaternion(rotation);
    const listenerPosition = this.pos.copy(position).divideScalar(unit);
    if (l.positionX) {
      for (const [params, vector] of [
        [[l.positionX, l.positionY, l.positionZ], listenerPosition],
        [[l.forwardX, l.forwardY, l.forwardZ], this.forward],
        [[l.upX, l.upY, l.upZ], this.up],
      ] as [AudioParam[], THREE.Vector3][]) {
        params[0].value = vector.x;
        params[1].value = vector.y;
        params[2].value = vector.z;
      }
    } else {
      // Firefox has no listener AudioParams. Without this the assignment above threw on every
      // animation frame, which both silenced spatial audio and made the whole viewer crawl.
      l.setPosition(listenerPosition.x, listenerPosition.y, listenerPosition.z);
      l.setOrientation(
        this.forward.x,
        this.forward.y,
        this.forward.z,
        this.up.x,
        this.up.y,
        this.up.z,
      );
    }
    for (const item of this.loaded)
      if (item.anchor) {
        const a = item.anchor;
        sampleAnchor(a.positions, a.times, time, this.pos);
        this.pos.addScaledVector(new THREE.Vector3().fromArray(a.offset), a.person.bodyH);
        a.person.group.updateWorldMatrix(true, false);
        this.pos.applyMatrix4(a.person.group.matrixWorld).divideScalar(unit);
        if (!item.panner) {
          item.panner = ctx.createPanner();
          item.panner.panningModel = 'HRTF';
          item.panner.refDistance = 0.6;
          item.panner.rolloffFactor = 0.5;
          item.panner.connect(this.master!);
        }
        if (item.panner.positionX) {
          item.panner.positionX.value = this.pos.x;
          item.panner.positionY.value = this.pos.y;
          item.panner.positionZ.value = this.pos.z;
        } else item.panner.setPosition(this.pos.x, this.pos.y, this.pos.z);
      }
    const active =
      running &&
      !this.video.paused &&
      !this.video.seeking &&
      this.video.readyState >= 3 &&
      this.unlocked &&
      !this.muted &&
      !this.suspended &&
      ctx.state === 'running';
    if (!active) {
      if (this.voices.length) this.invalidate();
      this.paint();
      return;
    }
    // All dialogue stems are reviewed at normal speed. Other rates preserve the original mix.
    if (this.mode === 'spatial' && this.video.playbackRate !== 1) {
      this.setMode('original');
      this.error = 'Spatial dialogue uses normal speed';
    }
    const tracks =
      this.mode === 'spatial'
        ? this.loaded
        : this.original
          ? [
              {
                buffer: this.original,
                track: { offsetSeconds: this.manifest?.original?.offsetSeconds ?? 0 },
              },
            ]
          : [];
    const predicted = this.clipAt + (now - this.startedAt) * this.scheduledRate;
    if (this.voices.length && (time < this.lastTime - 0.01 || Math.abs(predicted - time) > 0.08))
      this.invalidate();
    if (!this.voices.length && tracks.length) {
      this.startedAt = now + 0.015;
      this.clipAt = time + 0.015 * this.video.playbackRate;
      this.scheduledRate = this.video.playbackRate;
      for (const item of tracks) {
        const offset = this.clipAt - (item.track.offsetSeconds ?? 0);
        if (offset >= item.buffer.duration) continue;
        const source = ctx.createBufferSource();
        source.buffer = item.buffer;
        source.playbackRate.value = this.scheduledRate;
        source.connect(('panner' in item && item.panner) || this.master!);
        source.start(
          this.startedAt + Math.max(0, -offset) / this.scheduledRate,
          Math.max(0, offset),
        );
        this.voices.push(source);
      }
    }
    this.lastTime = time;
    this.paint();
  }

  private paint() {
    const p = this.manifest?.provenance;
    const attribution = typeof p?.attribution === 'string' ? p.attribution : '';
    const license = typeof p?.license === 'string' ? p.license : '';
    const href =
      typeof p?.licenseUrl === 'string' && /^https?:\/\//.test(p.licenseUrl) ? p.licenseUrl : '';
    const key = JSON.stringify([attribution, license, href]);
    if (key !== this.creditKey) {
      this.creditKey = key;
      this.credit.textContent = attribution ? `Audio excerpt: ${attribution}` : '';
      if (license) {
        const link = document.createElement('a');
        link.textContent = ` · ${license}`;
        link.style.color = '#c3ddff';
        if (href) {
          link.href = href;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
        }
        this.credit.append(link);
      }
    }
    this.credit.hidden = !attribution || this.mode !== 'original';
    const unavailable =
      this.manifest?.source.hasAudio === false &&
      !this.original &&
      !this.spatialReady &&
      !this.loading;
    const label = this.loading
      ? 'Loading sound…'
      : unavailable
        ? 'No soundtrack available'
        : this.muted || !this.unlocked
          ? 'Enable sound'
          : 'Mute sound';
    if (label !== this.lastLabel) {
      this.button.textContent = label;
      this.lastLabel = label;
    }
    this.button.disabled = unavailable;
    this.button.setAttribute('aria-pressed', String(!this.muted));
    this.button.title =
      this.error ||
      (this.manifest
        ? 'Source soundtrack'
        : 'Play the original soundtrack if this video contains audio');
    this.modeButton.hidden = !this.spatialReady;
    this.modeButton.textContent = this.mode === 'spatial' ? 'Dialogue: spatial' : 'Sound: original';
  }

  dispose() {
    this.dead = true;
    this.abort.abort();
    this.invalidate();
    this.listeners.forEach((f) => f());
    this.mediaSource?.disconnect();
    this.loaded.forEach((i) => i.panner?.disconnect());
    this.video.pause();
    this.video.muted = true;
    void this.ctx?.close();
    this.button.parentElement?.remove();
    this.credit.remove();
  }
}
