/** Clip transport never starts an authored camera tour or changes the user's camera. */
export function setClipPlaying(api, playing) {
  if (api.reel) api.showreel(false);
  if (playing) void api.unlockAudio?.();
  api.play(playing);
}

export function seekClip(api, seconds) {
  if (!Number.isFinite(seconds) || !Number.isFinite(api.dur) || api.dur <= 0) return;
  setClipPlaying(api, false);
  // The runtime loops at duration. Scrubbing to the right edge must hold the final pose.
  api.setTime(Math.max(0, Math.min(seconds, Math.max(0, api.dur - 0.001))));
}

export function clockLabel(seconds) {
  const tenths = Math.floor(Math.max(0, seconds) * 10);
  return `${Math.floor(tenths / 600)}:${String(Math.floor(tenths / 10) % 60).padStart(2, '0')}.${tenths % 10}`;
}

export function bindClipControls(root, api, scope) {
  const controls = root.querySelector('#clip-controls');
  if (!controls) return () => {};
  const toggle = controls.querySelector('[data-play]');
  const replay = controls.querySelector('[data-replay]');
  const timeline = controls.querySelector('input');
  const time = controls.querySelector('output');
  const loading = controls.querySelector('[data-loading]');
  scope.listenActive(toggle, 'click', () => setClipPlaying(api, !api.playing));
  scope.listenActive(replay, 'click', () => {
    seekClip(api, 0);
    setClipPlaying(api, true);
  });
  scope.listenActive(timeline, 'input', () => seekClip(api, Number(timeline.value)));
  return () => {
    const ready = scope.active && api.ready && Number.isFinite(api.dur) && api.dur > 0;
    for (const control of [toggle, replay, timeline]) control.disabled = !ready;
    loading.hidden = ready;
    if (!ready) return;
    const label = api.playing ? 'Pause clip' : 'Play clip';
    if (toggle.textContent !== label) toggle.textContent = label;
    timeline.max = String(api.dur);
    timeline.value = String(api.t);
    const current = clockLabel(api.t);
    timeline.setAttribute('aria-valuetext', `${current} of ${clockLabel(api.dur)}`);
    const text = `${current} / ${clockLabel(api.dur)}`;
    if (time.textContent !== text) time.textContent = text;
  };
}
