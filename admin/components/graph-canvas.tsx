'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ArtifactBlock } from '@/components/blocks';
import { StateMark } from '@/components/state-mark';
import { Progress } from '@/components/progress';
import { classify, hasPoster, isVisual, leadArtifact } from '@/lib/blocks';
import {
  elapsedMs,
  formatBytes,
  formatMs,
  isInFlight,
  statusLabel,
  statusTone,
} from '@/lib/format';
import { dependenciesOf, directDependencies, orderStages } from '@/lib/graph';
import type { PipelineRun, RunArtifact, RunAttempt, RunNode } from '@/lib/types';

const NODE_W = 248;
const NODE_H = 208;
const GAP_X = 72;
const GAP_Y = 28;
/** How far outside the block a skipping edge runs, and how far apart two of them sit. */
const LANE = 22;
/** Vertical space between bands: enough for a label and for an edge to route through. */
const BAND_GAP = 60;
const LANE_STEP = 12;
const CORNER = 14;

interface Placed {
  node: RunNode;
  x: number;
  y: number;
  band: number;
}

interface Band {
  key: string;
  label: string;
  top: number;
  height: number;
}

interface Point {
  x: number;
  y: number;
}

/** A point `distance` away from `from`, heading towards `toward`. */
function along(from: Point, toward: Point, distance: number): Point {
  const dx = toward.x - from.x;
  const dy = toward.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  const step = Math.min(distance, length);
  return { x: from.x + (dx / length) * step, y: from.y + (dy / length) * step };
}

/** A polyline with its corners rounded off, so a routed edge reads as one continuous move. */
function roundedPath(points: Point[]): string {
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = points[index - 1];
    const corner = points[index];
    const next = points[index + 1];
    const radius = Math.min(
      CORNER,
      Math.hypot(corner.x - previous.x, corner.y - previous.y) / 2,
      Math.hypot(next.x - corner.x, next.y - corner.y) / 2,
    );
    const entry = along(corner, previous, radius);
    const exit = along(corner, next, radius);
    d += ` L ${entry.x} ${entry.y} Q ${corner.x} ${corner.y}, ${exit.x} ${exit.y}`;
  }
  const last = points[points.length - 1];
  return `${d} L ${last.x} ${last.y}`;
}

/** The branch an expanded stage belongs to: `lhm_frozen:00` is branch `00`. */
function branchKeyOf(id: string): string | null {
  const colon = id.lastIndexOf(':');
  return colon > 0 ? id.slice(colon + 1) : null;
}

/**
 * Which band a stage is drawn in.
 *
 * The fixed pipeline is one band. Stages an expand created at runtime -- one set per person, per
 * object -- are their own band each, because they are not steps of the pipeline so much as the
 * same few steps repeated for each thing found in the footage. Human gates are their own band
 * too: what a person owes the run reads better collected than scattered down the flow.
 */
function bandOf(node: RunNode, deepestExpand: Map<string, string>): { key: string; label: string } {
  const branch = branchKeyOf(node.id);
  if (branch) {
    const origin = deepestExpand.get(node.id);
    return {
      key: `branch:${branch}`,
      label: origin ? `${origin} ${branch}` : `branch ${branch}`,
    };
  }
  if (node.definition.kind === 'human') return { key: 'review', label: 'review' };
  return { key: 'pipeline', label: 'pipeline' };
}

/**
 * Columns by dependency depth, rows grouped into bands, so footage flows left to right and each
 * band reads as one track of work.
 */
function place(nodes: RunNode[]): {
  placed: Placed[];
  bands: Band[];
  width: number;
  height: number;
} {
  const ordered = orderStages(nodes);
  const depthOf = new Map(ordered.map((entry) => [entry.node.id, entry.depth]));
  const byId = new Map(nodes.map((node) => [node.id, node]));

  /*
   * A branch is named after the expand that made it. `admission` is an expand too -- it fans a
   * clip into shots -- so the nearest one is ambiguous and the deepest is taken instead: the
   * person branches descend from `tracks`, which sits well below `admission`.
   */
  const deepestExpand = new Map<string, string>();
  for (const node of nodes) {
    let best: string | undefined;
    const seen = new Set<string>([node.id]);
    const queue = [...dependenciesOf(node)];
    while (queue.length) {
      const id = queue.shift() as string;
      if (seen.has(id)) continue;
      seen.add(id);
      const parent = byId.get(id);
      if (!parent) continue;
      if (
        parent.definition.kind === 'expand' &&
        (best === undefined || (depthOf.get(id) ?? 0) > (depthOf.get(best) ?? 0))
      ) {
        best = id;
      }
      queue.push(...dependenciesOf(parent));
    }
    if (best) deepestExpand.set(node.id, best);
  }

  const groups = new Map<string, { label: string; columns: Map<number, RunNode[]> }>();
  for (const { node, depth } of ordered) {
    const band = bandOf(node, deepestExpand);
    const group = groups.get(band.key) ?? { label: band.label, columns: new Map() };
    group.columns.set(depth, [...(group.columns.get(depth) ?? []), node]);
    groups.set(band.key, group);
  }

  const rank = (key: string) => (key === 'pipeline' ? 0 : key === 'review' ? 2 : 1);
  const keys = [...groups.keys()].sort(
    (a, b) => rank(a) - rank(b) || a.localeCompare(b, undefined, { numeric: true }),
  );

  const placed: Placed[] = [];
  const bands: Band[] = [];
  let top = 0;
  let deepest = 0;
  keys.forEach((key, index) => {
    const group = groups.get(key) as { label: string; columns: Map<number, RunNode[]> };
    const rows = Math.max(1, ...[...group.columns.values()].map((column) => column.length));
    const height = rows * NODE_H + (rows - 1) * GAP_Y;
    for (const [depth, column] of group.columns) {
      deepest = Math.max(deepest, depth);
      column.forEach((node, row) => {
        placed.push({
          node,
          x: depth * (NODE_W + GAP_X),
          y: top + row * (NODE_H + GAP_Y),
          band: index,
        });
      });
    }
    bands.push({ key, label: group.label, top, height });
    top += height + BAND_GAP;
  });

  return {
    placed,
    bands,
    width: (deepest + 1) * NODE_W + deepest * GAP_X,
    height: Math.max(0, top - BAND_GAP),
  };
}

/**
 * The artifact a node's card shows.
 *
 * A card is a tile, and a tile can only draw a frame or a clip: it has no WebGL context to spend
 * on a world or a point cloud, so those fall back to a vertex count. That makes the ranking here
 * different from the inspector's. `marble_video` writes both a `.spz` world and the thumbnail
 * beside it, and the thumbnail is the one worth showing at this size even though the world
 * outranks it everywhere else; `lhm_frozen` writes point clouds and eight PNGs of the person it
 * fitted. So the card leads with whatever the stage itself produced that can be drawn.
 *
 * Failing that it takes the stage's own lead -- a count or a gist still beats a bare mark -- and
 * only then borrows a picture from elsewhere in the attempt, saying that it is showing the input
 * rather than a result. A submit stage that hands a clip to a provider and gets back an operation
 * id has nothing else worth looking at.
 */
function previewFor(
  run: PipelineRun,
  node: RunNode,
): { artifact?: RunArtifact; attempt?: RunAttempt; borrowed?: boolean } {
  const attempts = (run.attempts ?? []).filter((attempt) => attempt.nodeId === node.id);
  const preferred =
    attempts.find((attempt) => attempt.id === node.selectedAttemptId) ??
    attempts[attempts.length - 1];
  if (!preferred) return {};
  const mine = (run.artifacts ?? []).filter((artifact) => artifact.attemptId === preferred.id);
  const own = mine.filter((artifact) => artifact.role !== 'attempt_file');
  const drawable = leadArtifact(own.filter(hasPoster));
  if (drawable) return { artifact: drawable, attempt: preferred };
  const lead = leadArtifact(own);
  if (lead && isVisual(lead)) return { artifact: lead, attempt: preferred };
  const borrowed = leadArtifact(mine.filter(hasPoster)) ?? leadArtifact(mine.filter(isVisual));
  if (borrowed) return { artifact: borrowed, attempt: preferred, borrowed: true };
  return { artifact: lead, attempt: preferred };
}

export function GraphCanvas({
  run,
  selectedId,
  onSelect,
  now,
  typical,
}: {
  run: PipelineRun;
  selectedId: string | null;
  onSelect: (id: string) => void;
  now: number;
  typical: Map<string, number>;
}) {
  const { placed, bands, width, height } = useMemo(() => place(run.nodes), [run.nodes]);
  const byId = useMemo(() => new Map(placed.map((entry) => [entry.node.id, entry])), [placed]);
  const edgesInto = useMemo(() => directDependencies(run.nodes), [run.nodes]);

  /*
   * An edge between neighbouring columns has the gutter to itself, so it takes the short curve.
   * One that skips a column would otherwise cross whatever sits in between and come out the far
   * side looking cut in half, so it leaves the block entirely: down the column gutter, along a
   * lane outside the cards, and back up into its target. Lanes are numbered so two of them never
   * share a line.
   */
  const edges = useMemo(() => {
    const spans: { from: Placed; to: Placed }[] = [];
    for (const target of placed) {
      for (const id of edgesInto.get(target.node.id) ?? []) {
        const source = byId.get(id);
        if (source) spans.push({ from: source, to: target });
      }
    }
    let lanes = 0;
    return spans.map(({ from, to }) => {
      const x1 = from.x + NODE_W;
      const y1 = from.y + NODE_H / 2;
      const x2 = to.x;
      const y2 = to.y + NODE_H / 2;
      if (x2 - x1 <= GAP_X + 1) {
        const bend = (x2 - x1) / 2;
        return {
          from,
          to,
          d: `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`,
        };
      }
      const band = bands[from.band];
      const offset = lanes * LANE_STEP;
      lanes += 1;
      const lane =
        band && band.top + band.height < height
          ? band.top + band.height + BAND_GAP / 2 + offset
          : height + LANE + offset;
      const gutterIn = x1 + GAP_X / 2;
      const gutterOut = x2 - GAP_X / 2;
      return {
        from,
        to,
        d: roundedPath([
          { x: x1, y: y1 },
          { x: gutterIn, y: y1 },
          { x: gutterIn, y: lane },
          { x: gutterOut, y: lane },
          { x: gutterOut, y: y2 },
          { x: x2, y: y2 },
        ]),
      };
    });
  }, [placed, bands, byId, edgesInto, height]);
  const frame = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ x: 24, y: 24, k: 1 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);

  function fit() {
    const box = frame.current?.getBoundingClientRect();
    if (!box) return;
    const k = Math.min(1, (box.width - 48) / width, (box.height - 48) / height);
    setView({ x: (box.width - width * k) / 2, y: (box.height - height * k) / 2, k });
  }

  // Fit once the graph is known; afterwards the operator's pan and zoom win.
  useEffect(fit, [width, height]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const element = frame.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const box = element.getBoundingClientRect();
      const px = event.clientX - box.left;
      const py = event.clientY - box.top;
      setView((current) => {
        const k = Math.min(2.5, Math.max(0.25, current.k * (event.deltaY < 0 ? 1.1 : 0.9)));
        const ratio = k / current.k;
        return { k, x: px - (px - current.x) * ratio, y: py - (py - current.y) * ratio };
      });
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, []);

  const zoom = (factor: number) => {
    const box = frame.current?.getBoundingClientRect();
    const px = (box?.width ?? 0) / 2;
    const py = (box?.height ?? 0) / 2;
    setView((current) => {
      const k = Math.min(2.5, Math.max(0.25, current.k * factor));
      const ratio = k / current.k;
      return { k, x: px - (px - current.x) * ratio, y: py - (py - current.y) * ratio };
    });
  };

  return (
    <div className="canvas">
      <div className="canvas__tools">
        <span className="canvas__hint">
          Drag to pan, scroll to zoom. Each card shows what that stage produced.
        </span>
        <button
          type="button"
          className="button button--small button--quiet"
          onClick={() => zoom(0.85)}
          aria-label="Zoom out"
        >
          −
        </button>
        <button
          type="button"
          className="button button--small button--quiet"
          onClick={() => zoom(1.18)}
          aria-label="Zoom in"
        >
          +
        </button>
        <button type="button" className="button button--small button--quiet" onClick={fit}>
          Fit
        </button>
      </div>
      <div
        className="canvas__frame"
        ref={frame}
        onPointerDown={(event) => {
          if ((event.target as HTMLElement).closest('.gnode')) return;
          drag.current = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y };
          (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          const start = drag.current;
          setView((current) => ({
            ...current,
            x: start.vx + (event.clientX - start.x),
            y: start.vy + (event.clientY - start.y),
          }));
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
      >
        <div
          className="canvas__world"
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})`,
            width,
            height,
          }}
        >
          {bands.map((band) =>
            band.top > 0 ? (
              <span
                className="gband"
                key={band.key}
                style={{ top: band.top - BAND_GAP / 2, width }}
                aria-hidden="true"
              >
                {band.label}
              </span>
            ) : null,
          )}
          <svg className="canvas__edges" width={width} height={height} aria-hidden="true">
            {edges.map(({ from, to, d }) => {
              const lit = from.node.id === selectedId || to.node.id === selectedId;
              const tone = statusTone(from.node.status, from.node.blockedReason);
              return (
                <path
                  key={`${from.node.id}->${to.node.id}`}
                  className={`gedge${lit ? ` gedge--lit gedge--${tone}` : ''}`}
                  d={d}
                />
              );
            })}
          </svg>
          {placed.map(({ node, x, y }) => {
            const tone = statusTone(node.status, node.blockedReason);
            const { artifact, attempt, borrowed } = previewFor(run, node);
            const inFlight = isInFlight(node.status);
            const elapsed =
              inFlight && attempt ? elapsedMs(attempt.startedAt, attempt.finishedAt, now) : null;
            const typicalMs = typical.get(node.id);
            return (
              <button
                key={node.id}
                type="button"
                className={`gnode gnode--${tone}${node.id === selectedId ? ' gnode--selected' : ''}`}
                style={{ left: x, top: y, width: NODE_W, height: NODE_H }}
                onClick={() => onSelect(node.id)}
                aria-pressed={node.id === selectedId}
              >
                <span className="gnode__preview">
                  {artifact ? (
                    <ArtifactBlock runId={run.id} artifact={artifact} density="tile" />
                  ) : (
                    <StateMark tone={tone} label={statusLabel(node.status, node.blockedReason)} />
                  )}
                </span>
                <span className="gnode__body">
                  <span className="gnode__head">
                    <span className="gnode__id">{node.id}</span>
                    {artifact ? (
                      <span className="gnode__kind">
                        {borrowed ? 'sent ' : ''}
                        {classify(artifact).noun}
                        <span className="mono"> {formatBytes(artifact.size)}</span>
                      </span>
                    ) : null}
                  </span>
                  <span className={`gnode__state gnode__state--${tone}`}>
                    {statusLabel(node.status, node.blockedReason)}
                    {elapsed !== null ? <span className="mono"> {formatMs(elapsed)}</span> : null}
                  </span>
                  {inFlight ? (
                    <Progress
                      fraction={
                        elapsed !== null && typicalMs ? Math.min(0.96, elapsed / typicalMs) : null
                      }
                      label={`${node.id} in progress`}
                    />
                  ) : null}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
