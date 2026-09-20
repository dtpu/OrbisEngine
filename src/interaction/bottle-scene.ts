import * as THREE from 'three';
import { BottlePhysics, type Vec3 } from './bottle-physics';
import { ApproachDetector } from './approach';
import { BottleAgentClient } from './bottle-agent-client';
import { VrInteractionControls, type ControlAction } from './vr-controls';
import {
  nearestHeldTime,
  parseHeadTrack,
  recordedState,
  sampleVector,
  yawBetween,
  type HeadTrack,
  type ObjectMetadata,
} from './recorded-object';

export type InteractionHand = {
  id: string;
  position: THREE.Vector3;
  rotation: THREE.Quaternion;
  squeeze: boolean;
};
export type InteractionPerson = {
  id: string;
  label: string;
  meta: { track?: number; head?: string };
  seqUrl: string;
  group: THREE.Object3D;
  frame: number;
  ts: number[];
  pmesh?: THREE.Object3D & { getBoundingBox(): THREE.Box3 };
  meshes?: Array<THREE.Object3D & { getBoundingBox(): THREE.Box3 }>;
};
export type InteractionObject = {
  id: string;
  label: string;
  meta: ObjectMetadata;
  track: { fps: number };
  group: THREE.Object3D;
  mesh: THREE.Object3D;
  interactionOwned: boolean;
  samplePosition(time: number): THREE.Vector3;
};
export type BottleSceneHost = {
  scene: THREE.Scene;
  people: InteractionPerson[];
  objects: InteractionObject[];
  stature: number;
  sceneId: string;
  params: URLSearchParams;
  time(): number;
  playing(): boolean;
  play(playing: boolean): void;
  seek(time: number): void;
  floorAt(x: number, z: number): number | null;
  blockedAt(position: Vec3, radius: number): boolean;
};
type Attachment = { person: InteractionPerson; local: THREE.Vector3 };
type SavedTransform = {
  object: THREE.Object3D;
  parent: THREE.Object3D;
  position: THREE.Vector3;
  rotation: THREE.Quaternion;
  scale: THREE.Vector3;
};
const tuple = (v: THREE.Vector3): Vec3 => [v.x, v.y, v.z];
const from = (v: Vec3) => new THREE.Vector3(...v);

/** Owns the opt-in scene's local state. The voice model never writes transforms directly. */
export class BottleScene {
  readonly controls: VrInteractionControls;
  readonly bottle: InteractionObject;
  readonly physics: BottlePhysics;
  private readonly approach: ApproachDetector;
  private readonly client: BottleAgentClient;
  private readonly head = new THREE.Vector3();
  private readonly rotation = new THREE.Quaternion();
  private readonly tracks = new Map<string, HeadTrack>();
  private readonly hands = new Map<string, InteractionHand>();
  private readonly armedHands = new Set<string>();
  private readonly saved: SavedTransform[] = [];
  private readonly pivots = new Map<string, THREE.Group>();
  private readonly fetchAbort = new AbortController();
  private readonly marker: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  private readonly speaker: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  private active: InteractionPerson | null = null;
  private target: Attachment | null = null;
  private attached: Attachment | null = null;
  private handOffset = new THREE.Quaternion();
  private bottleRotation = new THREE.Quaternion();
  private xrActive = false;
  private interrupted = false;
  private hasEntered = false;
  private disposed = false;
  private muted = false;
  private voiceStarting = false;
  private voiceStatus = 'Mic off';
  private status = 'Playing · approach closely or grip the bottle';
  private lastEvent = 'playing';
  private epoch = 0;
  private speaking = false;
  private airborneSeconds = 0;
  private lost = false;
  private showTarget = false;
  private lastFrame = 0;
  private readonly onPageHide = () => this.dispose();
  private readonly onVisibility = () => {
    if (document.hidden) this.suspend();
  };

  constructor(
    private readonly host: BottleSceneHost,
    bottle: InteractionObject,
  ) {
    this.bottle = bottle;
    const size = bottle.meta.appearance?.sizeWorldUnits;
    const scale = bottle.group.getWorldScale(new THREE.Vector3()).length() / Math.sqrt(3);
    const radius =
      size?.length === 3 && size.every(Number.isFinite)
        ? (Math.max(...size) * scale) / 2
        : 0.025 * host.stature;
    this.physics = new BottlePhysics({
      radius: Math.max(radius, 0.005 * host.stature),
      gravity: 5.77 * host.stature,
      maxSpeed: 8 * host.stature,
      floorAt: host.floorAt,
      blockedAt: host.blockedAt,
    });
    this.physics.reset(tuple(bottle.mesh.getWorldPosition(new THREE.Vector3())));
    const setting = (key: string, fallback: number, lo: number, hi: number) => {
      const value = host.params.has(key) ? Number(host.params.get(key)) : fallback;
      return Number.isFinite(value) ? THREE.MathUtils.clamp(value, lo, hi) : fallback;
    };
    const enter = setting('interactDistance', 0.3, 0.12, 0.6) * host.stature;
    this.approach = new ApproachDetector({
      enterDistance: enter,
      exitDistance: enter * 1.5,
      dwellSeconds: setting('interactDwell', 0.4, 0.1, 1.5),
      facingCos: 0.5,
    });
    this.controls = new VrInteractionControls(host.stature, (action) => this.control(action));
    host.scene.add(this.controls.group);
    this.marker = new THREE.Mesh(
      new THREE.RingGeometry(0.068 * host.stature, 0.075 * host.stature, 48),
      new THREE.MeshBasicMaterial({
        color: 0x80efbd,
        opacity: 0.8,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.marker.name = 'assisted-return-target';
    this.marker.renderOrder = 1002;
    this.marker.visible = false;
    this.speaker = new THREE.Mesh(
      new THREE.SphereGeometry(0.018 * host.stature, 12, 8),
      new THREE.MeshBasicMaterial({
        color: 0x82eabb,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.speaker.name = 'agent-speaking';
    this.speaker.renderOrder = 1002;
    this.speaker.visible = false;
    host.scene.add(this.marker, this.speaker);
    this.client = new BottleAgentClient({
      identity: () => ({
        sceneId: host.sceneId,
        personId: this.active?.id ?? host.people[0].id,
        personLabel: this.active?.label ?? host.people[0].label,
        objectId: bottle.id,
      }),
      status: (message) => {
        this.voiceStatus = message;
      },
      speaking: (value) => {
        this.speaking = value;
      },
      action: (name, args) => this.agentAction(name, args),
      speechStarted: () => this.onSpeech(),
    });
    for (const person of host.people) void this.loadHead(person);
    addEventListener('pagehide', this.onPageHide, { once: true });
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  private async loadHead(person: InteractionPerson) {
    try {
      const url = new URL('head.json', new URL(person.seqUrl, location.href));
      const result = await fetch(url, { signal: this.fetchAbort.signal });
      if (!result.ok) return;
      const track = parseHeadTrack(await result.json(), person.ts);
      if (track && !this.disposed) this.tracks.set(person.id, track);
    } catch {
      /* A missing head track uses the visible frame's approximate head anchor. */
    }
  }

  private anchor(person: InteractionPerson, time = this.host.time()): THREE.Vector3 | null {
    if (!person.group.visible) return null;
    person.group.updateWorldMatrix(true, false);
    const track = this.tracks.get(person.id);
    if (track)
      return sampleVector(track.positions, track.times, time).applyMatrix4(
        person.group.matrixWorld,
      );
    const mesh = person.pmesh ?? person.meshes?.[person.frame];
    if (!mesh) return null;
    mesh.updateWorldMatrix(true, false);
    const box = mesh.getBoundingBox().clone().applyMatrix4(mesh.matrixWorld);
    if (box.isEmpty() || !box.min.toArray().concat(box.max.toArray()).every(Number.isFinite))
      return null;
    return new THREE.Vector3(
      (box.min.x + box.max.x) / 2,
      box.max.y - 0.08 * this.host.stature,
      (box.min.z + box.max.z) / 2,
    );
  }

  private forward(person: InteractionPerson) {
    const track = this.tracks.get(person.id);
    if (!track?.forward) return null;
    return sampleVector(track.forward, track.times, this.host.time()).transformDirection(
      person.group.matrixWorld,
    );
  }

  private clearPath(a: THREE.Vector3, b: THREE.Vector3) {
    const steps = Math.ceil(a.distanceTo(b) / (0.025 * this.host.stature));
    if (steps > 300) return false;
    for (let i = 1; i < steps; i++) {
      const position = a.clone().lerp(b, i / steps);
      if (this.host.blockedAt(tuple(position), 0.006 * this.host.stature)) return false;
    }
    return true;
  }

  private choosePerson() {
    const facing = new THREE.Vector3(0, 0, -1).applyQuaternion(this.rotation);
    return (
      this.host.people
        .map((person) => ({ person, anchor: this.anchor(person) }))
        .filter(
          (item): item is { person: InteractionPerson; anchor: THREE.Vector3 } => !!item.anchor,
        )
        .filter(
          ({ anchor }) =>
            anchor.distanceTo(this.head) < 0.8 * this.host.stature &&
            anchor.clone().sub(this.head).normalize().dot(facing) > 0.45 &&
            this.clearPath(this.head, anchor),
        )
        .sort(
          (a, b) => a.anchor.distanceToSquared(this.head) - b.anchor.distanceToSquared(this.head),
        )[0]?.person ?? null
    );
  }

  private canAddress(person: InteractionPerson) {
    const anchor = this.anchor(person);
    if (!anchor) return false;
    const toward = anchor.clone().sub(this.head);
    return (
      toward.length() < 0.8 * this.host.stature &&
      toward.normalize().dot(new THREE.Vector3(0, 0, -1).applyQuaternion(this.rotation)) > 0.45 &&
      this.clearPath(this.head, anchor)
    );
  }

  private recordingState() {
    return recordedState(
      this.bottle.meta,
      this.bottle.track.fps,
      this.host.time(),
      this.host.people.map((person) => ({ id: person.id, track: person.meta.track })),
    );
  }

  private localAttachment(person: InteractionPerson, point: THREE.Vector3): Attachment {
    person.group.updateWorldMatrix(true, false);
    return { person, local: person.group.worldToLocal(point.clone()) };
  }

  private targetPosition(): THREE.Vector3 | null {
    if (!this.target || !this.target.person.group.visible) return null;
    return this.target.person.group.localToWorld(this.target.local.clone());
  }

  private prepareTarget(person: InteractionPerson) {
    const time = nearestHeldTime(
      this.bottle.meta,
      this.bottle.track.fps,
      this.host.time(),
      person.id,
    );
    const nowHead = this.anchor(person);
    if (time === null || !nowHead) {
      this.target = null;
      return;
    }
    const thenHead = this.anchor(person, time);
    if (!thenHead) {
      this.target = null;
      return;
    }
    const point = this.bottle.samplePosition(time).sub(thenHead).add(nowHead);
    // Bad source seams cannot create a return target across the room.
    if (
      point.distanceTo(nowHead) > 0.65 * this.host.stature ||
      this.host.blockedAt(tuple(point), 0.02 * this.host.stature)
    ) {
      this.target = null;
      return;
    }
    this.target = this.localAttachment(person, point);
  }

  interrupt(reason: 'approached' | 'speech' | 'grabbed', personId?: string): boolean {
    if (!this.xrActive || this.disposed) return false;
    const recorded = this.recordingState();
    const person =
      this.host.people.find((p) => p.id === personId) ??
      this.active ??
      this.host.people.find((p) => p.id === recorded.thrower) ??
      this.choosePerson();
    if (!person || !person.group.visible) return false;
    if (!this.interrupted) {
      this.interrupted = true;
      this.epoch++;
      this.host.play(false);
      this.client.setPlayback(false);
      this.active = person;
      this.bottle.interactionOwned = true;
      const position = this.bottle.mesh.getWorldPosition(new THREE.Vector3());
      const step = 1 / this.bottle.track.fps;
      if (recorded.free && reason !== 'grabbed') {
        const before = Math.max(0, this.host.time() - step);
        const velocity = this.bottle
          .samplePosition(this.host.time() + step)
          .sub(this.bottle.samplePosition(before))
          .divideScalar(this.host.time() + step - before);
        this.physics.startFlight(tuple(position), tuple(velocity));
      } else if (!recorded.free) {
        const owner = this.host.people.find((p) => p.id === recorded.owner);
        this.attached = owner ? this.localAttachment(owner, position) : null;
      }
      this.bottle.mesh.getWorldQuaternion(this.bottleRotation);
      this.prepareTarget(person);
    }
    this.lastEvent = reason;
    this.status =
      reason === 'grabbed'
        ? 'Bottle caught · release grip to throw'
        : 'Interacting · speak or reach for the bottle';
    if (reason !== 'grabbed') this.client.notify(this.snapshot(), reason !== 'speech');
    return true;
  }

  private onSpeech(): boolean {
    if (!this.xrActive || this.muted || this.disposed) return false;
    const person = this.interrupted ? this.active : this.choosePerson();
    if (!person || !person.group.visible || !this.anchor(person)) return false;
    if (!this.canAddress(person)) return false;
    if (!this.interrupted) return this.interrupt('speech', person.id);
    return true;
  }

  private capture(object: THREE.Object3D, pivot: THREE.Group) {
    if (!object.parent || this.saved.some((item) => item.object === object)) return;
    this.saved.push({
      object,
      parent: object.parent,
      position: object.position.clone(),
      rotation: object.quaternion.clone(),
      scale: object.scale.clone(),
    });
    pivot.attach(object);
  }

  private facePerson(): boolean {
    const person = this.active;
    if (!person) return false;
    const position = this.anchor(person),
      facing = this.forward(person);
    if (!position || !facing) return false;
    let pivot = this.pivots.get(person.id);
    if (!pivot) {
      pivot = new THREE.Group();
      pivot.name = `interaction-facing-${person.id}`;
      pivot.position.copy(position);
      pivot.position.y =
        this.host.floorAt(position.x, position.z) ?? position.y - 0.9 * this.host.stature;
      this.host.scene.add(pivot);
      this.capture(person.group, pivot);
      for (const object of this.host.objects) {
        if (object === this.bottle) continue;
        const state = recordedState(object.meta, object.track.fps, this.host.time(), []);
        if (state.owner === person.id) {
          object.interactionOwned = true;
          this.capture(object.mesh, pivot);
        }
      }
      this.pivots.set(person.id, pivot);
    }
    const desired = this.head.clone().sub(position);
    desired.y = 0;
    facing.y = 0;
    if (desired.lengthSq() < 1e-8 || facing.lengthSq() < 1e-8) return false;
    pivot.quaternion.premultiply(yawBetween(facing, desired));
    pivot.updateMatrixWorld(true);
    return true;
  }

  private agentAction(name: string, args: unknown): string {
    if (
      !this.xrActive ||
      !this.interrupted ||
      this.disposed ||
      !args ||
      typeof args !== 'object' ||
      Array.isArray(args) ||
      Object.keys(args).length
    )
      return 'Action unavailable in the current state.';
    if (name === 'face_player')
      return this.facePerson() ? 'Character turned toward visitor.' : 'Facing anchor unavailable.';
    if (name === 'show_return_target') {
      this.showTarget = !!this.target;
      return this.target
        ? 'Assisted return target shown. No bottle catch has occurred.'
        : 'No valid receiving target is available.';
    }
    if (name === 'offer_replay') {
      this.status = 'Replay is available on your VR controls';
      return 'Replay offered; recording remains paused.';
    }
    return 'Unsupported action.';
  }

  private restore() {
    for (const item of this.saved) {
      item.parent.add(item.object);
      item.object.position.copy(item.position);
      item.object.quaternion.copy(item.rotation);
      item.object.scale.copy(item.scale);
      item.object.updateMatrixWorld(true);
    }
    this.saved.length = 0;
    for (const pivot of this.pivots.values()) pivot.removeFromParent();
    this.pivots.clear();
    for (const object of this.host.objects) object.interactionOwned = false;
    this.attached = this.target = null;
    this.active = null;
    this.showTarget = false;
    this.lost = false;
    this.bottle.mesh.visible = true;
    this.airborneSeconds = 0;
    this.marker.visible = false;
  }

  replay(play = true) {
    this.epoch++;
    this.client.setPlayback(true);
    this.host.play(false);
    this.restore();
    this.interrupted = false;
    this.host.seek(0);
    this.physics.reset(tuple(this.bottle.mesh.getWorldPosition(new THREE.Vector3())));
    this.approach.reset(tuple(this.head));
    this.hands.clear();
    this.armedHands.clear();
    this.lastEvent = play ? 'replayed' : 'reset';
    this.status = play ? 'Playing · approach closely or grip the bottle' : 'Scene reset · paused';
    this.host.play(play);
    this.client.setPlayback(play);
    this.client.notify(this.snapshot(), false);
  }

  control(action: ControlAction) {
    if (!this.xrActive || this.disposed) return;
    if (action === 'replay') this.replay(true);
    else if (action === 'reset') this.replay(false);
    else if (action === 'pause') {
      if (this.interrupted) {
        this.status = 'Use Replay to restore the recorded exchange';
        return;
      }
      this.host.play(!this.host.playing());
      this.client.setPlayback(this.host.playing());
      this.status = this.host.playing()
        ? 'Playing · approach closely or grip the bottle'
        : 'Recording paused';
    } else if (action === 'end') {
      this.client.disconnect();
      this.voiceStarting = false;
      this.muted = false;
      this.client.setMuted(false);
      this.voiceStatus = 'Mic off';
    } else if (action === 'microphone') {
      if (this.voiceStarting) return;
      if (this.client.connected) {
        this.muted = !this.muted;
        this.client.setMuted(this.muted);
      } else {
        this.muted = false;
        this.client.setMuted(false);
        this.voiceStarting = true;
        this.client.setPlayback(this.host.playing());
        this.client.notify(this.snapshot(), this.interrupted);
        void this.client
          .connect()
          .catch(() => {
            this.voiceStatus = 'Mic unavailable · select Enable mic to retry';
          })
          .finally(() => {
            this.voiceStarting = false;
          });
      }
    }
  }

  get canPlay() {
    return !this.interrupted;
  }
  transportChanged(playing: boolean) {
    this.client.setPlayback(playing);
  }
  sessionStart() {
    this.xrActive = true;
    this.controls.group.visible = true;
    this.lastFrame = performance.now();
    this.approach.reset();
    this.hands.clear();
    this.armedHands.clear();
    if (!this.hasEntered) {
      this.hasEntered = true;
      if (!this.interrupted) this.host.play(true);
    }
  }
  private suspend() {
    this.client.disconnect();
    this.voiceStarting = false;
    this.voiceStatus = 'Mic off';
    this.host.play(false);
    this.hands.clear();
    this.armedHands.clear();
    const held = this.physics.snapshot();
    if (held.holder) {
      this.physics.release(held.holder, [0, 0, 0]);
      this.lastEvent = 'released_on_exit';
    }
  }
  sessionEnd() {
    this.suspend();
    this.xrActive = false;
    this.controls.group.visible = false;
    this.marker.visible = false;
    this.speaker.visible = false;
  }
  ended() {
    this.host.play(false);
    this.status = 'Recording finished · approach, talk, or Replay';
    this.client.setPlayback(false);
  }
  select(origin: THREE.Vector3, direction: THREE.Vector3) {
    return this.controls.select(origin, direction);
  }
  hover(origin: THREE.Vector3, direction: THREE.Vector3) {
    return this.controls.hover(origin, direction);
  }

  frame(
    head: THREE.Vector3,
    rotation: THREE.Quaternion,
    inputs: InteractionHand[],
    deltaSeconds?: number,
  ) {
    if (!this.xrActive || this.disposed) return;
    const now = performance.now();
    const dt = deltaSeconds ?? Math.min(0.1, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    this.head.copy(head);
    this.rotation.copy(rotation);
    if (!this.interrupted)
      this.physics.setRecordedPosition(
        tuple(this.bottle.mesh.getWorldPosition(new THREE.Vector3())),
      );
    if (this.attached && ['recorded', 'returned'].includes(this.physics.snapshot().mode)) {
      const position = this.attached.person.group.localToWorld(this.attached.local.clone());
      this.physics.setAttachedPosition(tuple(position));
    }
    for (const input of inputs) {
      if (!input.squeeze) this.armedHands.add(input.id);
      const previous = this.hands.get(input.id);
      let state = this.physics.snapshot();
      const reachable =
        !this.lost &&
        this.bottle.group.visible &&
        this.physics.canGrabSegment(
          tuple(input.position),
          0.055 * this.host.stature,
          previous && tuple(previous.position),
        );
      if (input.squeeze && this.armedHands.has(input.id) && !state.holder && reachable) {
        const recorded = this.recordingState();
        if (
          this.interrupt('grabbed', this.active?.id ?? recorded.thrower ?? undefined) &&
          this.physics.grab(
            input.id,
            tuple(input.position),
            0.055 * this.host.stature,
            previous && tuple(previous.position),
          )
        ) {
          this.attached = null;
          this.bottle.interactionOwned = true;
          this.showTarget = true;
          this.bottle.mesh.getWorldQuaternion(this.bottleRotation);
          this.handOffset.copy(input.rotation).invert().multiply(this.bottleRotation);
          this.client.notify(this.snapshot());
        }
      }
      state = this.physics.snapshot();
      if (state.holder === input.id) {
        this.physics.moveHand(input.id, tuple(input.position), dt);
        this.bottleRotation.copy(input.rotation).multiply(this.handOffset);
        if (!input.squeeze) {
          this.physics.release(input.id);
          this.airborneSeconds = 0;
          this.lastEvent = 'released';
          this.status = 'Bottle released · aim for the green return target';
          this.client.notify(this.snapshot(), false);
        }
      }
    }
    const held = this.physics.snapshot();
    if (held.holder && !inputs.some((input) => input.id === held.holder)) {
      this.armedHands.delete(held.holder);
      this.physics.release(held.holder, [0, 0, 0]);
      this.lastEvent = 'tracking_lost';
      this.client.notify(this.snapshot(), false);
    }
    this.hands.clear();
    for (const input of inputs)
      this.hands.set(input.id, {
        ...input,
        position: input.position.clone(),
        rotation: input.rotation.clone(),
      });
    if (!this.interrupted) {
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(rotation);
      const personId = this.approach.update(
        {
          position: tuple(head),
          forward: tuple(forward),
          candidates: this.host.people.map((person) => {
            const anchor = this.anchor(person);
            return {
              id: person.id,
              position: tuple(anchor ?? new THREE.Vector3()),
              eligible:
                !!anchor &&
                Math.abs(anchor.y - head.y) < 0.5 * this.host.stature &&
                this.clearPath(head, anchor),
            };
          }),
        },
        dt,
      );
      if (personId) this.interrupt('approached', personId);
    }
    if (this.interrupted && !this.lost) {
      const target = this.targetPosition();
      const events = this.physics.step(
        dt,
        target ? { position: tuple(target), radius: 0.075 * this.host.stature } : undefined,
      );
      for (const event of events) {
        this.lastEvent = event.type;
        if (event.type === 'returned' && this.target) {
          this.attached = this.target;
          this.showTarget = false;
          this.status = 'Bottle returned · recording stays paused';
        } else {
          this.status = 'Bottle landed · grip to pick it up';
          this.showTarget = false;
        }
        this.client.notify(this.snapshot());
      }
      const state = this.physics.snapshot();
      if (state.mode === 'free') this.airborneSeconds += dt;
      if (
        this.airborneSeconds > 20 ||
        from(state.position).distanceTo(head) > 15 * this.host.stature
      ) {
        this.lost = true;
        this.status = 'Bottle out of reach · Reset restores the scene';
        this.lastEvent = 'out_of_reach';
        this.client.notify(this.snapshot());
      }
      this.bottle.mesh.parent!.updateWorldMatrix(true, false);
      this.bottle.mesh.position.copy(this.bottle.mesh.parent!.worldToLocal(from(state.position)));
      const parentRotation = this.bottle.mesh.parent!.getWorldQuaternion(new THREE.Quaternion());
      this.bottle.mesh.quaternion.copy(parentRotation.invert().multiply(this.bottleRotation));
      this.bottle.mesh.visible = !this.lost;
      this.bottle.mesh.updateMatrixWorld(true);
    }
    const target = this.targetPosition();
    this.marker.visible = this.showTarget && !!target && !this.lost;
    if (target) this.marker.position.copy(target);
    this.marker.quaternion.copy(rotation);
    const anchor = this.active && this.anchor(this.active);
    this.speaker.visible = !!anchor && this.speaking;
    if (anchor) {
      this.speaker.position.copy(anchor).add(new THREE.Vector3(0, 0.1 * this.host.stature, 0));
      this.client.updateAudio(anchor, head, rotation, this.host.stature);
    }
    this.controls.update(
      head,
      rotation,
      `${this.status} · ${this.voiceStatus}`,
      this.client.connected && !this.muted,
      this.host.playing(),
    );
  }

  snapshot() {
    return {
      epoch: this.epoch,
      sceneId: this.host.sceneId,
      recordingTime: this.host.time(),
      playing: this.host.playing(),
      interrupted: this.interrupted,
      activePersonId: this.active?.id ?? null,
      lastEvent: this.lastEvent,
      bottle: {
        id: this.bottle.id,
        ...this.physics.snapshot(),
        personOwner: this.attached?.person.id ?? null,
        lost: this.lost,
      },
      returnTarget: this.targetPosition()?.toArray() ?? null,
      voiceConnected: this.client.connected,
      voiceStatus: this.voiceStatus,
      microphoneMuted: this.muted,
      poseType: 'paused recorded pose; whole-body turn only',
      returnType: 'assisted target, not animated reach',
      availableActions: this.interrupted
        ? ['face_player', 'show_return_target', 'offer_replay']
        : [],
    };
  }

  dispose() {
    if (this.disposed) return;
    this.sessionEnd();
    this.disposed = true;
    this.fetchAbort.abort();
    this.restore();
    this.controls.dispose();
    for (const mesh of [this.marker, this.speaker]) {
      mesh.removeFromParent();
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    removeEventListener('pagehide', this.onPageHide);
    document.removeEventListener('visibilitychange', this.onVisibility);
  }
}
