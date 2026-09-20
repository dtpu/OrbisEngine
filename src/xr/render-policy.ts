/** Keep CPU colour data intact while allowing headset scenes to use a splat budget. */
export function sceneUsesLod(q: URLSearchParams): boolean {
  if (q.get('lod') === '1') return true;
  const cpuColours = !!q.get('bakedweights') || (!!q.get('obs') && Number(q.get('obsfade')) > 0);
  if (cpuColours) return false;
  const xrOverride = q.get('xr') === '1' && q.get('xrworldlod') !== '0';
  return xrOverride || q.get('lod') !== '0';
}

/** Apply the scene's boundary fade without suppressing a teleport or snap-turn blink. */
export function xrVignetteAmount(edge: number, blink: number, edgeFade: number): number {
  const finiteEdge = Number.isFinite(edge) ? edge : 0;
  const finiteBlink = Number.isFinite(blink) ? blink : 0;
  const fade = Number.isFinite(edgeFade) ? Math.max(0, edgeFade) : 1;
  const scaledEdge = Math.max(0, Math.min(1, finiteEdge * fade));
  const boundaryAmount = scaledEdge * scaledEdge * (3 - 2 * scaledEdge);
  return Math.max(boundaryAmount, Math.max(0, Math.min(1, finiteBlink)));
}
