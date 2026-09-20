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
  panel.innerHTML = `<header><strong>Quest view</strong><span role="status" aria-live="polite"></span><span class="quest-view-rate" data-role="rate" aria-label="Displayed frame rate" hidden></span><span class="quest-view-actions"><button type="button" data-role="expand" aria-label="Expand Quest view">＋</button><button type="button" data-role="minimize" aria-label="Minimize Quest view">−</button></span></header><div class="quest-view-picture"><img alt="Live view from the Quest" hidden><p>Enter VR on the Quest to share its view.</p></div>`;
  const style = document.createElement('style');
  style.textContent = `
    .quest-view-panel { position:absolute; z-index:12; right:16px; bottom:88px; width:min(380px,calc(100% - 32px)); max-height:calc(100% - 104px); overflow:hidden; color:#ece7dc; background:#121416; border:1px solid #766343; border-radius:12px; box-shadow:0 10px 32px #0008; font:12px/1.4 system-ui,sans-serif; }
    .quest-view-panel[hidden], .quest-view-panel [hidden] { display:none !important; }
    .quest-view-panel header { display:flex; align-items:center; gap:8px; height:46px; min-height:46px; box-sizing:border-box; padding:9px 10px 9px 13px; }
    .quest-view-panel strong { flex-shrink:0; font-size:13px; white-space:nowrap; }
    .quest-view-panel [role=status] { min-width:0; margin-left:auto; overflow:hidden; color:#c9b787; font-size:10px; text-overflow:ellipsis; white-space:nowrap; }
    .quest-view-rate { flex-shrink:0; color:#a7aaa8; font-size:10px; white-space:nowrap; }
    .quest-view-actions { display:flex; flex-shrink:0; gap:5px; }
    .quest-view-panel button { flex-shrink:0; width:28px; height:28px; padding:0; border:1px solid #645840; border-radius:6px; background:transparent; color:#ece7dc; font:18px/1 system-ui; cursor:pointer; }
    .quest-view-panel button:focus-visible { outline:2px solid #e5c47b; outline-offset:2px; }
    .quest-view-picture { position:relative; width:100%; height:auto; aspect-ratio:1 / 1; min-height:0; background:#080a0b; display:grid; place-items:center; overflow:hidden; }
    .quest-view-picture img { width:100%; height:100%; object-fit:contain; position:absolute; inset:0; }
    .quest-view-picture p { color:#a7aaa8; max-width:220px; padding:16px; margin:0; text-align:center; }
    .quest-view-panel[data-expanded="true"] { width:min(540px,calc(100% - 32px)); }
    @media(max-width:520px) { .quest-view-panel { right:10px; width:min(380px,calc(100% - 20px)); } .quest-view-panel[data-expanded="true"] { width:min(540px,calc(100% - 20px)); } }
  `;
  stage.append(style, panel);
  const status = panel.querySelector<HTMLElement>('[role=status]')!;
  const rate = panel.querySelector<HTMLElement>('[data-role="rate"]')!;
  const header = panel.querySelector<HTMLElement>('header')!;
  const picture = panel.querySelector<HTMLElement>('.quest-view-picture')!;
  const img = panel.querySelector('img')!;
  const hint = panel.querySelector('p')!;
  const expand = panel.querySelector<HTMLButtonElement>('[data-role="expand"]')!;
  const minimize = panel.querySelector<HTMLButtonElement>('[data-role="minimize"]')!;
  toggle.setAttribute('aria-controls', panel.id);
  toggle.setAttribute('aria-expanded', 'true');
  let open = true;
  let expanded = false;
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
        : stageRect.height <= 500
          ? 12
          : 88;
    const availableHeight = Math.max(70, stageRect.height - bottomInset - 8);
    const headerHeight = header.getBoundingClientRect().height || 46;
    const maxPictureHeight = Math.max(24, availableHeight - headerHeight - 2);
    const preferred = expanded ? 540 : 380;
    const maxWidth = Math.max(0, stageRect.width - margin * 2);
    const minWidth = Math.min(220, maxWidth);
    const width = Math.max(minWidth, Math.min(preferred, maxWidth));
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
  const onExpand = () => {
    expanded = !expanded;
    panel.dataset.expanded = String(expanded);
    expand.setAttribute('aria-label', expanded ? 'Reduce Quest view' : 'Expand Quest view');
    expand.textContent = expanded ? '−' : '＋';
    resizePanel();
  };
  const onVisibility = () => {
    invalidate();
    setStatus('Waiting for Quest');
  };
  toggle.addEventListener('click', onToggle);
  minimize.addEventListener('click', onMinimize);
  expand.addEventListener('click', onExpand);
  document.addEventListener('visibilitychange', onVisibility);
  function dispose() {
    if (disposed) return;
    disposed = true;
    invalidate();
    clearTimeout(timer);
    resizeObserver.disconnect();
    panel.remove();
    style.remove();
    window.removeEventListener('resize', resizePanel);
    toggle.removeEventListener('click', onToggle);
    minimize.removeEventListener('click', onMinimize);
    expand.removeEventListener('click', onExpand);
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
