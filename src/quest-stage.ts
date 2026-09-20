/** Ephemeral, single-eye spectator images shown in place of the desktop viewer. No audio or recording. */
export function mountQuestStage({
  stage,
  isLocalPresenting = () => false,
}: {
  stage: HTMLElement;
  isLocalPresenting?: () => boolean;
}): () => void {
  const viewer = crypto.randomUUID();
  const view = document.createElement('section');
  view.id = `quest-stage-${viewer}`;
  view.className = 'quest-stage';
  view.setAttribute('aria-label', 'Quest view');
  view.hidden = true;
  view.innerHTML = `<img alt="Live view from the Quest"><div class="quest-stage-badge"><strong>Quest view</strong><span data-role="rate" aria-label="Displayed frame rate" hidden></span></div><span role="status" aria-live="polite" class="quest-stage-status"></span>`;
  const style = document.createElement('style');
  style.textContent = `
    .quest-stage { position:absolute; inset:0; z-index:12; background:#000; font:12px/1.4 var(--sans,system-ui,sans-serif); }
    .quest-stage[hidden], .quest-stage [hidden] { display:none !important; }
    .quest-stage img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
    .quest-stage-badge { position:absolute; left:16px; top:16px; display:flex; align-items:center; gap:8px; height:30px; padding:0 12px; border-radius:999px; background:var(--bg,#f0f0f2); color:var(--ink-2,#4b4a52); border:1px solid var(--line-2,#d3d2d8); }
    .quest-stage-badge strong { font-size:11.5px; font-weight:600; letter-spacing:0.02em; }
    .quest-stage-badge span { color:var(--mute,#8b8996); font-size:11px; font-variant-numeric:tabular-nums; }
    .quest-stage-status { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
  `;
  stage.append(style, view);
  const status = view.querySelector<HTMLElement>('[role=status]')!;
  const rate = view.querySelector<HTMLElement>('[data-role="rate"]')!;
  const img = view.querySelector('img')!;
  let disposed = false;
  let generation = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let request: AbortController | undefined;
  let imageURL: string | undefined;
  let identity = '';
  let sequence = -1;
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
  // Without a live image the desktop viewer underneath is the demo, so the whole surface hides.
  function clearImage() {
    img.removeAttribute('src');
    if (imageURL) URL.revokeObjectURL(imageURL);
    imageURL = undefined;
    sequence = -1;
    resetRate();
    view.hidden = true;
  }
  function invalidate() {
    generation++;
    request?.abort();
    clearImage();
  }
  function eligible() {
    return !disposed && !document.hidden && !isLocalPresenting();
  }
  async function poll() {
    timer = undefined;
    if (disposed) return;
    const started = performance.now();
    sourceAvailable = false;
    if (!eligible()) {
      clearImage();
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
            view.hidden = false;
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
  const onVisibility = () => {
    invalidate();
    setStatus('Waiting for Quest');
  };
  document.addEventListener('visibilitychange', onVisibility);
  function dispose() {
    if (disposed) return;
    disposed = true;
    invalidate();
    clearTimeout(timer);
    view.remove();
    style.remove();
    document.removeEventListener('visibilitychange', onVisibility);
    window.removeEventListener('pagehide', dispose);
  }
  window.addEventListener('pagehide', dispose);
  setStatus('Waiting for Quest');
  void poll();
  return dispose;
}
