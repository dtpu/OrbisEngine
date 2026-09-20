/** Ephemeral, single-eye spectator images. No audio or recording. */
export function mountQuestViewPanel({
  stage,
  toggle,
  isLocalPresenting = () => false,
}: {
  stage: HTMLElement;
  toggle: HTMLButtonElement;
  isLocalPresenting?: () => boolean;
}): () => void {
  const viewer = crypto.randomUUID();
  const panel = document.createElement('section');
  panel.id = `quest-view-${viewer}`;
  panel.className = 'quest-view-panel';
  panel.setAttribute('aria-label', 'Quest view');
  panel.innerHTML = `<header><strong>Quest view</strong><span role="status" aria-live="polite"></span><span class="quest-view-rate" data-role="rate" aria-label="Displayed frame rate" hidden></span><span class="quest-view-actions"><button type="button" data-role="minimize" aria-label="Close Quest view">×</button></span></header><div class="quest-view-picture"><img alt="Live view from the Quest" hidden><p>Enter VR on the Quest to share its view.</p></div><span class="quest-view-grip" role="slider" tabindex="0" aria-label="Resize Quest view" aria-valuemin="220" aria-valuemax="1200" aria-valuenow="380" title="Drag to resize"></span>`;
  const style = document.createElement('style');
  style.textContent = `
    .quest-view-panel { position:absolute; z-index:12; right:16px; bottom:88px; width:min(380px,calc(100% - 32px)); max-height:calc(100% - 104px); overflow:hidden; color:var(--ink,#1e1d22); background:var(--bg,#f0f0f2); border:1px solid var(--line-2,#d3d2d8); border-radius:14px; font:12px/1.4 var(--sans,system-ui,sans-serif); }
    .quest-view-panel[hidden], .quest-view-panel [hidden] { display:none !important; }
    .quest-view-panel header { display:flex; align-items:center; gap:8px; height:46px; min-height:46px; box-sizing:border-box; padding:8px 8px 8px 14px; border-bottom:1px solid var(--line,#dcdbe0); }
    .quest-view-panel strong { flex-shrink:0; font-size:11.5px; font-weight:600; letter-spacing:0.02em; color:var(--mute,#8b8996); white-space:nowrap; }
    .quest-view-panel [role=status] { min-width:0; margin-left:auto; overflow:hidden; color:var(--mute,#8b8996); font-size:11px; text-overflow:ellipsis; white-space:nowrap; }
    .quest-view-rate { flex-shrink:0; color:var(--dim,#b3b1bb); font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .quest-view-actions { display:flex; flex-shrink:0; gap:4px; }
    .quest-view-panel button { flex-shrink:0; height:28px; min-width:28px; padding:0 8px; border:1px solid var(--line-2,#d3d2d8); border-radius:999px; background:transparent; color:var(--ink-2,#4b4a52); font:500 12px/1 var(--sans,system-ui,sans-serif); cursor:pointer; transition:background 0.15s; }
    .quest-view-panel button[data-role="minimize"] { padding:0; font-size:17px; }
    .quest-view-panel button:hover { background:var(--surface,#e8e8eb); color:var(--ink,#1e1d22); }
    .quest-view-panel button:focus-visible { outline:2px solid var(--ink,#1e1d22); outline-offset:2px; }
    .quest-view-picture { position:relative; width:100%; height:auto; aspect-ratio:1 / 1; min-height:0; background:var(--surface-2,#e0e0e4); display:grid; place-items:center; overflow:hidden; }
    .quest-view-picture img { width:100%; height:100%; object-fit:contain; position:absolute; inset:0; background:#000; }
    .quest-view-picture p { color:var(--mute,#8b8996); max-width:220px; padding:16px; margin:0; text-align:center; font-size:13px; }
    .quest-view-grip { position:absolute; left:0; bottom:0; width:24px; height:24px; cursor:nesw-resize; touch-action:none; border-radius:0 0 0 14px; background:linear-gradient(45deg, transparent 9px, var(--line-2,#d3d2d8) 9px, var(--line-2,#d3d2d8) 10.5px, transparent 10.5px, transparent 14px, var(--line-2,#d3d2d8) 14px, var(--line-2,#d3d2d8) 15.5px, transparent 15.5px); }
    .quest-view-grip:hover, .quest-view-grip:focus-visible { background-color:var(--surface,#e8e8eb); outline:none; }
    @media(max-width:520px) { .quest-view-panel { right:10px; width:min(380px,calc(100% - 20px)); } }
  `;
  stage.append(style, panel);
  const status = panel.querySelector<HTMLElement>('[role=status]')!;
  const rate = panel.querySelector<HTMLElement>('[data-role="rate"]')!;
  const header = panel.querySelector<HTMLElement>('header')!;
  const picture = panel.querySelector<HTMLElement>('.quest-view-picture')!;
  const img = panel.querySelector('img')!;
  const hint = panel.querySelector('p')!;
  const grip = panel.querySelector<HTMLElement>('.quest-view-grip')!;
  const minimize = panel.querySelector<HTMLButtonElement>('[data-role="minimize"]')!;
  toggle.setAttribute('aria-controls', panel.id);
  toggle.setAttribute('aria-expanded', 'false');
  panel.hidden = true;
  let open = false;
  let preferred = 380;
  let disposed = false;
  let generation = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let request: AbortController | undefined;
  let imageURL: string | undefined;
  let identity = '';
  let sequence = -1;
  let imageRatio = 1;
  let sourceAvailable = false;
  let rateWindowStart = 0;
  let rateWindowFrames = 0;
  let rateLastUpdate = 0;

  function setStatus(value: string) {
    if (status.textContent !== value) status.textContent = value;
  }
  function resetRate() {
    rate.hidden = true;
    rate.textContent = '';
    rateWindowStart = 0;
    rateWindowFrames = 0;
    rateLastUpdate = 0;
  }
  function noteAppliedFrame() {
    const now = performance.now();
    if (!rateWindowStart) rateWindowStart = now;
    rateWindowFrames++;
    if (now - rateWindowStart >= 1000 && now - rateLastUpdate >= 1000) {
      const elapsed = now - rateWindowStart;
      rate.textContent = `${Math.round((rateWindowFrames * 1000) / elapsed)} fps`;
      rate.hidden = false;
      rateLastUpdate = now;
      rateWindowStart = now;
      rateWindowFrames = 0;
    }
  }
  function resizePanel() {
    if (disposed || !open) return;
    const stageRect = stage.getBoundingClientRect();
    const controls = stage.querySelector<HTMLElement>('#bar');
    const controlsRect = controls?.getBoundingClientRect();
    const margin = stageRect.width <= 520 ? 10 : 16;
    const bottomInset =
      controlsRect && controlsRect.height > 0 && controlsRect.top < stageRect.bottom
        ? Math.max(8, stageRect.bottom - controlsRect.top + 12)
        : margin;
    const availableHeight = Math.max(70, stageRect.height - bottomInset - 8);
    const headerHeight = header.getBoundingClientRect().height || 46;
    const maxPictureHeight = Math.max(24, availableHeight - headerHeight - 2);
    const maxWidth = Math.max(0, stageRect.width - margin * 2);
    const minWidth = Math.min(220, maxWidth);
    const width = Math.max(minWidth, Math.min(preferred, maxWidth));
    grip.setAttribute('aria-valuenow', String(Math.round(width)));
    panel.style.width = `${width}px`;
    panel.style.right = `${margin}px`;
    panel.style.bottom = `${bottomInset}px`;
    panel.style.maxHeight = `${availableHeight}px`;
    picture.style.height = `${Math.max(24, Math.min(width / imageRatio, maxPictureHeight))}px`;
  }
  const resizeObserver = new ResizeObserver(resizePanel);
  resizeObserver.observe(stage);
  window.addEventListener('resize', resizePanel);
  function setImageRatio(width: number, height: number) {
    if (width <= 0 || height <= 0) return;
    const nextRatio = width / height;
    if (Math.abs(nextRatio - imageRatio) < 0.0001) return;
    imageRatio = nextRatio;
    picture.style.aspectRatio = `${width} / ${height}`;
    resizePanel();
  }
  function clearImage(message = 'Enter VR on the Quest to share its view.') {
    img.hidden = true;
    img.removeAttribute('src');
    if (imageURL) URL.revokeObjectURL(imageURL);
    imageURL = undefined;
    sequence = -1;
    resetRate();
    setImageRatio(1, 1);
    hint.textContent = message;
    hint.hidden = false;
  }
  function invalidate() {
    generation++;
    request?.abort();
    clearImage();
  }
  function eligible() {
    return !disposed && open && !document.hidden && !isLocalPresenting();
  }
  async function poll() {
    timer = undefined;
    if (disposed) return;
    const started = performance.now();
    sourceAvailable = false;
    if (!eligible()) {
      clearImage(isLocalPresenting() ? 'Your view is being shared.' : undefined);
      setStatus(isLocalPresenting() ? 'Viewing on this Quest' : 'Waiting for Quest');
      timer = setTimeout(poll, 250);
      return;
    }
    const token = generation;
    const controller = new AbortController();
    request = controller;
    const valid = () => token === generation && eligible() && !controller.signal.aborted;
    const deadline = setTimeout(() => {
      controller.abort();
      if (token === generation) {
        clearImage();
        setStatus('Connection lost');
      }
    }, 2000);
    let candidateURL: string | undefined;
    try {
      const response = await fetch(`/api/quest-view?viewer=${encodeURIComponent(viewer)}`, {
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!response.ok) throw new Error('Unavailable');
      const state: {
        active: boolean;
        sessionId: string | null;
        clip: string;
        frame: number;
        updatedAt: number | null;
      } = await response.json();
      if (!valid()) return;
      const nextIdentity = `${state.sessionId}:${state.clip}`;
      if (identity !== nextIdentity) {
        clearImage();
        identity = nextIdentity;
      }
      if (!state.active || !state.sessionId || !state.updatedAt) {
        clearImage();
        setStatus('Waiting for Quest');
      } else if (Date.now() - state.updatedAt > 2500) {
        clearImage();
        setStatus('Connection lost');
      } else {
        sourceAvailable = true;
        if (state.frame !== sequence) {
          const frame = await fetch(
            `/api/quest-view/frame?session=${encodeURIComponent(state.sessionId)}&seq=${state.frame}`,
            { cache: 'no-store', signal: controller.signal },
          );
          if (!valid()) return;
          if (frame.status === 204) {
            clearImage();
            setStatus('Waiting for Quest');
          } else {
            if (!frame.ok) throw new Error('Unavailable');
            const blob = await frame.blob();
            if (!valid()) return;
            candidateURL = URL.createObjectURL(blob);
            const decoded = new Image();
            decoded.src = candidateURL;
            await Promise.race([
              decoded.decode(),
              new Promise<never>((_, reject) =>
                controller.signal.addEventListener('abort', () => reject(new Error('Cancelled')), {
                  once: true,
                }),
              ),
            ]);
            if (!valid()) return;
            if (Date.now() - state.updatedAt > 2500) {
              clearImage();
              setStatus('Connection lost');
              return;
            }
            const previousURL = imageURL;
            imageURL = candidateURL;
            candidateURL = undefined;
            img.src = imageURL;
            img.hidden = false;
            hint.hidden = true;
            setImageRatio(decoded.naturalWidth, decoded.naturalHeight);
            if (previousURL) URL.revokeObjectURL(previousURL);
            sequence = Number(frame.headers.get('X-Quest-Frame') ?? state.frame);
            noteAppliedFrame();
            setStatus('Live');
          }
        }
      }
    } catch {
      if (token === generation && eligible()) {
        clearImage();
        setStatus('Connection lost');
      }
    } finally {
      clearTimeout(deadline);
      if (candidateURL) URL.revokeObjectURL(candidateURL);
      request = undefined;
      if (!disposed) {
        const target = sourceAvailable ? 1000 / 60 : 250;
        timer = setTimeout(poll, Math.max(0, target - (performance.now() - started)));
      }
    }
  }
  function setOpen(value: boolean) {
    open = value;
    panel.hidden = !value;
    toggle.setAttribute('aria-expanded', String(value));
    invalidate();
    setStatus('Waiting for Quest');
    resizePanel();
  }
  const onToggle = () => setOpen(!open);
  const onMinimize = () => {
    setOpen(false);
    toggle.focus();
  };
  // The panel is anchored to the stage's bottom-right, so dragging its lower-left corner outward
  // (left or down) makes it larger; height follows the headset image's aspect ratio.
  let drag: { pointer: number; x: number; width: number } | undefined;
  const onGripDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    drag = { pointer: e.pointerId, x: e.clientX, width: panel.getBoundingClientRect().width };
    grip.setPointerCapture(e.pointerId);
  };
  const onGripMove = (e: PointerEvent) => {
    if (!drag || e.pointerId !== drag.pointer) return;
    preferred = Math.max(220, drag.width + (drag.x - e.clientX));
    resizePanel();
  };
  const onGripUp = (e: PointerEvent) => {
    if (!drag || e.pointerId !== drag.pointer) return;
    drag = undefined;
    grip.releasePointerCapture(e.pointerId);
  };
  const onGripKey = (e: KeyboardEvent) => {
    const step =
      e.key === 'ArrowLeft' || e.key === 'ArrowUp'
        ? 24
        : e.key === 'ArrowRight' || e.key === 'ArrowDown'
          ? -24
          : 0;
    if (!step) return;
    e.preventDefault();
    preferred = Math.max(220, panel.getBoundingClientRect().width + step);
    resizePanel();
  };
  const onVisibility = () => {
    invalidate();
    setStatus('Waiting for Quest');
  };
  toggle.addEventListener('click', onToggle);
  const syncToggle = () => toggle.classList.toggle('on', open);
  const observer = new MutationObserver(syncToggle);
  observer.observe(toggle, { attributes: true, attributeFilter: ['aria-expanded'] });
  minimize.addEventListener('click', onMinimize);
  grip.addEventListener('pointerdown', onGripDown);
  grip.addEventListener('pointermove', onGripMove);
  grip.addEventListener('pointerup', onGripUp);
  grip.addEventListener('pointercancel', onGripUp);
  grip.addEventListener('keydown', onGripKey);
  document.addEventListener('visibilitychange', onVisibility);
  function dispose() {
    if (disposed) return;
    disposed = true;
    invalidate();
    clearTimeout(timer);
    resizeObserver.disconnect();
    observer.disconnect();
    toggle.classList.remove('on');
    panel.remove();
    style.remove();
    window.removeEventListener('resize', resizePanel);
    toggle.removeEventListener('click', onToggle);
    minimize.removeEventListener('click', onMinimize);
    grip.removeEventListener('pointerdown', onGripDown);
    grip.removeEventListener('pointermove', onGripMove);
    grip.removeEventListener('pointerup', onGripUp);
    grip.removeEventListener('pointercancel', onGripUp);
    grip.removeEventListener('keydown', onGripKey);
    document.removeEventListener('visibilitychange', onVisibility);
    window.removeEventListener('pagehide', dispose);
    toggle.setAttribute('aria-expanded', 'false');
    toggle.removeAttribute('aria-controls');
  }
  window.addEventListener('pagehide', dispose);
  setStatus('Waiting for Quest');
  resizePanel();
  void poll();
  return dispose;
}
