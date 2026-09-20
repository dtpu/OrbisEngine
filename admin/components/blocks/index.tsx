'use client';

import { useEffect, useState } from 'react';
import { MeshBlock } from '@/components/blocks/mesh-block';
import { PlyBlock } from '@/components/blocks/ply-block';
import { SplatBlock } from '@/components/blocks/splat-block';
import { classify, extensionOf, plyCounts } from '@/lib/blocks';
import { artifactUrl } from '@/lib/client';
import { excerpt, summarize, type Summary } from '@/lib/facts';
import { formatBytes } from '@/lib/format';
import type { RunArtifact } from '@/lib/types';

/**
 * One artifact, drawn at the density the surrounding page needs.
 *
 * `tile` sits inside a graph node, `thumb` inside the filmstrip, `full` inside the inspector.
 * All three route through the same classification, so a world reads as a world in the graph and
 * in the inspector, and a block never grows past the box it was given -- the frame has the size,
 * the content is fitted into it.
 *
 * Only `full` is ever interactive. A graph shows a dozen stages at once and a WebGL context per
 * node would exhaust the browser's supply, so tiles and thumbnails stay flat.
 */
export type Density = 'tile' | 'thumb' | 'full';

export function ArtifactBlock({
  runId,
  artifact,
  density,
}: {
  runId: string;
  artifact: RunArtifact;
  density: Density;
}) {
  const kind = classify(artifact);
  const url = artifactUrl(runId, artifact.id);
  const empty = artifact.size === 0;

  if (density === 'full') {
    if (empty) return <p className="viewer__note">This file is empty.</p>;
    switch (kind.type) {
      case 'image':
        // Artifacts are unsized on the API, so the box sets the layout and the image is fitted.
        // eslint-disable-next-line @next/next/no-img-element
        return <img className="viewer__media" src={url} alt={artifact.name ?? artifact.role} />;
      case 'video':
        return (
          <video
            key={artifact.id}
            className="viewer__media"
            src={url}
            controls
            preload="metadata"
          />
        );
      case 'audio':
        return <audio key={artifact.id} className="viewer__audio" src={url} controls />;
      case 'splat':
        return (
          <SplatBlock
            key={artifact.id}
            url={url}
            name={artifact.name ?? `artifact.${extensionOf(artifact)}`}
            size={artifact.size}
          />
        );
      case 'points':
        return (
          <PlyBlock
            key={artifact.id}
            url={url}
            name={artifact.name ?? 'artifact.ply'}
            size={artifact.size}
          />
        );
      case 'mesh':
        return (
          <MeshBlock
            key={artifact.id}
            url={url}
            extension={extensionOf(artifact)}
            size={artifact.size}
          />
        );
      case 'data':
      case 'text':
        return <TextBody artifact={artifact} runId={runId} json={kind.type === 'data'} />;
      default:
        return (
          <p className="viewer__note">
            No inline preview for {kind.noun} ({artifact.mediaType}). Download it to inspect.
          </p>
        );
    }
  }

  return (
    <span className={`block block--${density} block--${kind.type}`}>
      <span className="block__frame">
        <Poster
          artifact={artifact}
          url={url}
          type={kind.type}
          mark={kind.mark}
          empty={empty}
          runId={runId}
        />
      </span>
    </span>
  );
}

/** The picture inside a tile or thumbnail: a real frame where one exists, a typed mark otherwise. */
function Poster({
  artifact,
  url,
  type,
  mark,
  empty,
  runId,
}: {
  artifact: RunArtifact;
  url: string;
  type: string;
  mark: string;
  empty: boolean;
  runId: string;
}) {
  if (!empty && type === 'image') {
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={url} alt="" loading="lazy" />;
  }
  if (!empty && type === 'video') {
    // A poster needs one decoded frame, so the fragment seeks past a black first frame.
    return <video src={`${url}#t=0.5`} muted preload="metadata" tabIndex={-1} />;
  }
  // A report and a prompt have no picture, but they do have content worth reading at a glance.
  if (!empty && (type === 'data' || type === 'text')) {
    return <Gist artifact={artifact} runId={runId} json={type === 'data'} mark={mark} />;
  }
  // A point cloud cannot be drawn here -- a WebGL context per tile would exhaust the browser --
  // but its header says how big it is, and that reads better than a bare mark.
  if (!empty && type === 'points') {
    return <PointCount artifact={artifact} url={url} mark={mark} />;
  }
  return (
    <span className="block__mark" aria-hidden="true" title={artifact.mediaType}>
      {mark}
    </span>
  );
}

/**
 * Vertex count from a PLY header. The header is ASCII at the head of the file, so a range request
 * reads it without pulling a cloud that may be tens of megabytes.
 */
function PointCount({ artifact, url, mark }: { artifact: RunArtifact; url: string; mark: string }) {
  const [vertices, setVertices] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setVertices(null);
    fetch(url, { headers: { Range: 'bytes=0-2047' }, cache: 'force-cache' })
      .then(async (response) => {
        if (!response.ok) throw new Error(String(response.status));
        const head = await response.text();
        if (!cancelled) setVertices(plyCounts(head).vertices);
      })
      .catch(() => {
        // Without a header the tile keeps its mark; the inspector renders the real thing.
      });
    return () => {
      cancelled = true;
    };
  }, [url]);

  if (vertices === null) {
    return (
      <span className="block__mark" aria-hidden="true" title={artifact.mediaType}>
        {mark}
      </span>
    );
  }
  const shown =
    vertices >= 1_000_000
      ? `${(vertices / 1_000_000).toFixed(1)}M`
      : vertices >= 1000
        ? `${Math.round(vertices / 1000)}k`
        : String(vertices);
  return (
    <span className="gist gist--facts">
      <span className="gist__fact">
        <b>{shown}</b>
        <i>points</i>
      </span>
    </span>
  );
}

/** How much of a report or prompt a tile will read before falling back to its mark. */
const GIST_LIMIT = 64_000;

/**
 * The readable part of a report or prompt, shown inside its tile: headline facts for JSON, the
 * opening words for text. Artifacts are immutable and served with a long cache, so the same fetch
 * that fills the inspector fills every tile without touching the network twice.
 */
function Gist({
  artifact,
  runId,
  json,
  mark,
}: {
  artifact: RunArtifact;
  runId: string;
  json: boolean;
  mark: string;
}) {
  const [state, setState] = useState<{ id: string; summary?: Summary }>({ id: artifact.id });

  useEffect(() => {
    let cancelled = false;
    setState({ id: artifact.id });
    if (artifact.size > GIST_LIMIT) return;
    fetch(artifactUrl(runId, artifact.id), { cache: 'force-cache' })
      .then(async (response) => {
        if (!response.ok) throw new Error(String(response.status));
        const body = await response.text();
        if (cancelled) return;
        if (json) {
          try {
            const summary = summarize(JSON.parse(body));
            if (summary) {
              setState({ id: artifact.id, summary });
              return;
            }
          } catch {
            // Not valid JSON after all; read it as words.
          }
        }
        setState({ id: artifact.id, summary: { kind: 'text', text: excerpt(body) } });
      })
      .catch(() => {
        // A tile is not the place to report a failed fetch; the inspector says why.
      });
    return () => {
      cancelled = true;
    };
  }, [artifact.id, artifact.size, json, runId]);

  const summary = state.id === artifact.id ? state.summary : undefined;
  if (summary?.kind === 'facts') {
    return (
      <span className="gist gist--facts">
        {summary.facts.map((fact) => (
          <span className="gist__fact" key={fact.label}>
            <b>{fact.value}</b>
            <i>{fact.label}</i>
          </span>
        ))}
      </span>
    );
  }
  if (summary?.kind === 'text') {
    return <span className="gist gist--text">{summary.text}</span>;
  }
  return (
    <span className="block__mark" aria-hidden="true" title={artifact.mediaType}>
      {mark}
    </span>
  );
}

const TEXT_LIMIT = 200_000;

function TextBody({
  artifact,
  runId,
  json,
}: {
  artifact: RunArtifact;
  runId: string;
  json: boolean;
}) {
  const [state, setState] = useState<{ id: string; text?: string; error?: string }>({
    id: artifact.id,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ id: artifact.id });
    if (artifact.size > TEXT_LIMIT) {
      setState({
        id: artifact.id,
        error: `Too large to show inline (${formatBytes(artifact.size)}).`,
      });
      return;
    }
    fetch(artifactUrl(runId, artifact.id), { cache: 'force-cache' })
      .then(async (response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const text = await response.text();
        if (cancelled) return;
        if (json) {
          try {
            setState({ id: artifact.id, text: JSON.stringify(JSON.parse(text), null, 2) });
            return;
          } catch {
            // Not valid JSON after all; show it verbatim.
          }
        }
        setState({ id: artifact.id, text });
      })
      .catch((cause: Error) => {
        if (!cancelled) setState({ id: artifact.id, error: cause.message });
      });
    return () => {
      cancelled = true;
    };
  }, [artifact.id, artifact.size, json, runId]);

  if (state.id !== artifact.id || (state.text === undefined && !state.error)) {
    return <p className="viewer__note">Loading…</p>;
  }
  if (state.error) return <p className="viewer__note">{state.error}</p>;
  return <pre className="viewer__text">{state.text}</pre>;
}
