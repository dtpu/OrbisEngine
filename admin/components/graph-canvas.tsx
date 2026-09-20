'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ArtifactBlock } from '@/components/blocks';
import { StateMark } from '@/components/state-mark';
import { Progress } from '@/components/progress';
import { classify, leadArtifact } from '@/lib/blocks';
import {
  elapsedMs,
  formatBytes,
  formatMs,
  isInFlight,
  statusLabel,
  statusTone,
} from '@/lib/format';
import { dependenciesOf, orderStages } from '@/lib/graph';
import type { PipelineRun, RunArtifact, RunAttempt, RunNode } from '@/lib/types';

const NODE_W = 248;
const NODE_H = 208;
const GAP_X = 72;
const GAP_Y = 28;

interface Placed {
  node: RunNode;
  x: number;
  y: number;
}

/** Columns by dependency depth, rows stacked and centred, so footage flows left to right. */
function place(nodes: RunNode[]): { placed: Placed[]; width: number; height: number } {
  const ordered = orderStages(nodes);
  const columns = new Map<number, RunNode[]>();
  for (const { node, depth } of ordered) columns.set(depth, [...(columns.get(depth) ?? []), node]);
  const tallest = Math.max(1, ...[...columns.values()].map((column) => column.length));
  const height = tallest * NODE_H + (tallest - 1) * GAP_Y;
  const placed: Placed[] = [];
  for (const [depth, column] of columns) {
    const columnHeight = column.length * NODE_H + (column.length - 1) * GAP_Y;
    const offset = (height - columnHeight) / 2;
    column.forEach((node, row) => {
      placed.push({ node, x: depth * (NODE_W + GAP_X), y: offset + row * (NODE_H + GAP_Y) });
    });
  }
  const width = columns.size * NODE_W + (columns.size - 1) * GAP_X;
  return { placed, width, height };
}

/** The artifact a node's card shows: the newest attempt's primary output, never a staged input. */
function previewFor(
  run: PipelineRun,
  node: RunNode,
): { artifact?: RunArtifact; attempt?: RunAttempt } {
  const attempts = (run.attempts ?? []).filter((attempt) => attempt.nodeId === node.id);
  const preferred =
    attempts.find((attempt) => attempt.id === node.selectedAttemptId) ??
    attempts[attempts.length - 1];
  if (!preferred) return {};
  const outputs = (run.artifacts ?? []).filter(
    (artifact) => artifact.attemptId === preferred.id && artifact.role !== 'attempt_file',
  );
  return { artifact: leadArtifact(outputs), attempt: preferred };
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
  const { placed, width, height } = useMemo(() => place(run.nodes), [run.nodes]);
  const byId = useMemo(() => new Map(placed.map((entry) => [entry.node.id, entry])), [placed]);
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
          <svg className="canvas__edges" width={width} height={height} aria-hidden="true">
            {placed.flatMap(({ node, x, y }) =>
              dependenciesOf(node)
                .map((dependency) => byId.get(dependency))
                .filter((source): source is Placed => Boolean(source))
                .map((source) => {
                  const x1 = source.x + NODE_W;
                  const y1 = source.y + NODE_H / 2;
                  const x2 = x;
                  const y2 = y + NODE_H / 2;
                  const bend = (x2 - x1) / 2;
                  return (
                    <path
                      key={`${source.node.id}->${node.id}`}
                      className={`gedge gedge--${statusTone(source.node.status, source.node.blockedReason)}`}
                      d={`M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`}
                    />
                  );
                }),
            )}
          </svg>
          {placed.map(({ node, x, y }) => {
            const tone = statusTone(node.status, node.blockedReason);
            const { artifact, attempt } = previewFor(run, node);
            const inFlight = isInFlight(node.status);
            const elapsed =
              inFlight && attempt ? elapsedMs(attempt.startedAt, attempt.finishedAt, now) : null;
            const typicalMs = typical.get(node.id);
            return (
              <button
                key={node.id}
                type="button"
                className={`gnode${node.id === selectedId ? ' gnode--selected' : ''}`}
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
