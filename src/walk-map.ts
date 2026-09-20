// Overhead floor picker. All placement validation stays in the walk controller.
export type MapPoint = { x: number; z: number };
export type WalkMapGrid = {
  width: number;
  height: number;
  floor: ArrayLike<number>;
  inside: ArrayLike<number | boolean>;
  cell: number;
  ox: number;
  oz: number;
};
export type WalkMapOptions = {
  container?: HTMLElement;
  grid: WalkMapGrid;
  position: () => MapPoint;
  yaw: () => number;
  people: () => readonly MapPoint[];
  path: readonly MapPoint[];
  canStand: (x: number, z: number) => boolean;
  stand: (x: number, z: number) => boolean;
};

export function createWalkMap({
  container = document.body,
  grid,
  position,
  yaw,
  people,
  path,
  canStand,
  stand,
}: WalkMapOptions) {
  const panel = document.createElement('section');
  panel.id = 'walk-map';
  panel.innerHTML = `<button class="map-toggle" aria-expanded="false">Map <kbd>M</kbd></button>
    <div class="map-panel" hidden><strong>Pick where to stand</strong>
    <canvas width="320" height="240" tabindex="0" aria-label="Overhead map. Arrow keys choose a spot; Enter moves there."></canvas>
    <div class="map-legend"><span>● You</span><span>● People</span><span>Mapped ground</span></div>
    <p role="status">Click a lit floor tile to move.</p></div>`;
  const style = document.createElement('style');
  style.textContent = `#walk-map{position:fixed;right:14px;top:14px;z-index:12;font:13px system-ui;color:#edf0f3}
    #walk-map button{float:right;background:#141920ed;border:1px solid #647080;border-radius:8px;color:inherit;padding:9px 13px;cursor:pointer}
    #walk-map kbd{margin-left:9px;color:#aab2bf;font:11px monospace}
    #walk-map .map-panel{clear:both;padding:14px;background:#111820f5;border:1px solid #3b4857;border-radius:10px;box-shadow:0 12px 40px #0008;max-width:calc(100vw - 60px)}
    #walk-map strong{display:block;margin-bottom:10px;font-weight:550}
    #walk-map canvas{display:block;width:320px;max-width:100%;height:auto;border-radius:5px;cursor:crosshair;touch-action:none}
    #walk-map .map-legend{display:flex;gap:14px;font-size:11px;margin-top:10px;color:#a8b4c3}
    #walk-map .map-legend span:first-child{color:#f1c777}#walk-map .map-legend span:nth-child(2){color:#80bded}
    #walk-map p{font-size:12px;margin:9px 0 0;max-width:300px;min-height:30px;color:#c6d0dc}`;
  document.head.append(style);
  container.append(panel);
  const toggle = panel.querySelector('button')!,
    body = panel.querySelector<HTMLDivElement>('.map-panel')!;
  const canvas = panel.querySelector('canvas')!,
    ctx = canvas.getContext('2d')!,
    status = panel.querySelector('p')!;
  const g = grid,
    padding = 14;
  let minI = g.width,
    maxI = 0,
    minJ = g.height,
    maxJ = 0;
  for (let k = 0; k < g.floor.length; k++)
    if (g.inside[k] && Number.isFinite(g.floor[k])) {
      const i = k % g.width,
        j = Math.floor(k / g.width);
      minI = Math.min(minI, i);
      maxI = Math.max(maxI, i);
      minJ = Math.min(minJ, j);
      maxJ = Math.max(maxJ, j);
    }
  if (minI > maxI) {
    panel.remove();
    style.remove();
    return null;
  }
  const pixel = Math.min(
    (320 - 2 * padding) / (maxI - minI + 1),
    (240 - 2 * padding) / (maxJ - minJ + 1),
  );
  const left = (320 - (maxI - minI + 1) * pixel) / 2,
    top = (240 - (maxJ - minJ + 1) * pixel) / 2;
  const project = (x: number, z: number): [number, number] => [
    left + (x / g.cell - g.ox - minI) * pixel,
    top + (z / g.cell - g.oz - minJ) * pixel,
  ];
  const unproject = (x: number, y: number): MapPoint => ({
    x: ((x - left) / pixel + g.ox + minI) * g.cell,
    z: ((y - top) / pixel + g.oz + minJ) * g.cell,
  });
  const base = document.createElement('canvas');
  base.width = 320;
  base.height = 240;
  const b = base.getContext('2d')!;
  b.fillStyle = '#0a1017';
  b.fillRect(0, 0, 320, 240);
  for (let j = minJ; j <= maxJ; j++)
    for (let i = minI; i <= maxI; i++) {
      const k = j * g.width + i,
        x = (g.ox + i + 0.5) * g.cell,
        z = (g.oz + j + 0.5) * g.cell;
      if (!g.inside[k]) continue;
      b.fillStyle = !Number.isFinite(g.floor[k])
        ? '#202b36'
        : canStand(x, z)
          ? '#527364'
          : '#333f4b';
      b.fillRect(
        left + (i - minI) * pixel,
        top + (j - minJ) * pixel,
        Math.max(1, pixel),
        Math.max(1, pixel),
      );
    }
  if (path.length > 1) {
    b.strokeStyle = '#99b3ca';
    b.setLineDash([3, 3]);
    b.beginPath();
    path.forEach((p, i) => {
      const [x, y] = project(p.x, p.z);
      i ? b.lineTo(x, y) : b.moveTo(x, y);
    });
    b.stroke();
  }
  let cursor: MapPoint | null = null,
    open = false;
  function setOpen(value: boolean) {
    if (value && (!container.isConnected || container.hidden)) return;
    open = value;
    body.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    if (open) {
      document.exitPointerLock?.();
      cursor = { x: position().x, z: position().z };
      canvas.focus();
      draw();
    }
  }
  function draw() {
    if (!open || !container.isConnected || container.hidden) return;
    ctx.drawImage(base, 0, 0);
    for (const p of people()) {
      const [x, y] = project(p.x, p.z);
      ctx.fillStyle = '#80bded';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    const p = position(),
      [x, y] = project(p.x, p.z);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-yaw());
    ctx.fillStyle = '#f1c777';
    ctx.strokeStyle = '#171b20';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, -8);
    ctx.lineTo(5, 6);
    ctx.lineTo(0, 3);
    ctx.lineTo(-5, 6);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    if (cursor) {
      const [cx, cy] = project(cursor.x, cursor.z);
      ctx.strokeStyle = canStand(cursor.x, cursor.z) ? '#f1c777' : '#e6a096';
      ctx.lineWidth = 1;
      ctx.strokeRect(cx - 5, cy - 5, 10, 10);
    }
  }
  function choose(p: MapPoint) {
    cursor = p;
    const ok = stand(p.x, p.z);
    status.textContent = ok
      ? 'Moved. Keep walking, or choose another spot.'
      : 'Choose a lit floor tile clear of walls.';
    draw();
    return ok;
  }
  toggle.addEventListener('click', () => setOpen(!open));
  panel.addEventListener('pointerdown', (e) => e.stopPropagation());
  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    cursor = unproject(
      ((e.clientX - r.left) * 320) / r.width,
      ((e.clientY - r.top) * 240) / r.height,
    );
    draw();
  });
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    choose(
      unproject(((e.clientX - r.left) * 320) / r.width, ((e.clientY - r.top) * 240) / r.height),
    );
  });
  canvas.addEventListener('keydown', (e) => {
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Enter', ' '].includes(e.key)) {
      e.preventDefault();
      e.stopPropagation();
      cursor ||= { ...position() };
      if (e.key === 'ArrowLeft') cursor.x -= g.cell;
      if (e.key === 'ArrowRight') cursor.x += g.cell;
      if (e.key === 'ArrowUp') cursor.z -= g.cell;
      if (e.key === 'ArrowDown') cursor.z += g.cell;
      if (e.key === 'Enter' || e.key === ' ') choose(cursor);
      draw();
    }
  });
  const keys = (e: KeyboardEvent) => {
    if (!container.isConnected || container.hidden) return;
    if (e.code === 'KeyM' && !e.repeat) {
      e.preventDefault();
      setOpen(!open);
    } else if (e.code === 'Escape' && open) setOpen(false);
  };
  window.addEventListener('keydown', keys);
  const timer = setInterval(draw, 100);
  return {
    project,
    choose,
    setOpen,
    get open() {
      return open;
    },
    destroy() {
      clearInterval(timer);
      window.removeEventListener('keydown', keys);
      panel.remove();
      style.remove();
    },
  };
}
