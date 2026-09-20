import { orderStages } from '@/lib/graph';
import { statusLabel, statusTone } from '@/lib/format';
import type { RunNode } from '@/lib/types';

/**
 * One cell per stage, in dependency order, coloured by status. It is the whole run at a glance:
 * how far along it is, where it is stuck, and whether anything is waiting on the operator.
 */
export function StageStrip({ nodes, compact = false }: { nodes: RunNode[]; compact?: boolean }) {
  const ordered = orderStages(nodes);
  return (
    <ol className={`strip${compact ? ' strip--compact' : ''}`} aria-label="Stages in order">
      {ordered.map(({ node }) => (
        <li
          key={node.id}
          className={`strip__cell strip__cell--${statusTone(node.status, node.blockedReason)}`}
          title={`${node.id}: ${statusLabel(node.status, node.blockedReason)}`}
        >
          <span className="sr-only">
            {node.id} {statusLabel(node.status, node.blockedReason)}
          </span>
        </li>
      ))}
    </ol>
  );
}
