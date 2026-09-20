import type { PipelineStage } from './types.ts';

export interface DagNode {
  id: string;
  x: number;
  y: number;
}

export interface DagLayout {
  nodes: DagNode[];
  width: number;
  height: number;
}

export function layoutDag(stages: PipelineStage[]): DagLayout {
  const ids = new Set(stages.map((stage) => stage.id));
  const depth = new Map<string, number>();
  const visiting = new Set<string>();

  function visit(id: string): number {
    if (depth.has(id)) return depth.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const current = stages.find((stage) => stage.id === id);
    const parents = current?.dependsOn.filter((parent) => ids.has(parent)) ?? [];
    const value = parents.length ? Math.max(...parents.map(visit)) + 1 : 0;
    visiting.delete(id);
    depth.set(id, value);
    return value;
  }

  stages.forEach((stage) => visit(stage.id));
  const lane = new Map<number, number>();
  const nodes = stages.map((stage) => {
    const column = depth.get(stage.id) ?? 0;
    const row = lane.get(column) ?? 0;
    lane.set(column, row + 1);
    return { id: stage.id, x: 28 + column * 210, y: 26 + row * 92 };
  });
  const columns = Math.max(1, ...nodes.map((node) => (depth.get(node.id) ?? 0) + 1));
  const rows = Math.max(1, ...lane.values());
  return {
    nodes,
    width: Math.max(300, 28 + columns * 210),
    height: Math.max(150, 26 + rows * 92),
  };
}
