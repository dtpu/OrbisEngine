/** Unconfigured scenes must not inherit another room's camera tour. */
export function cameraReelMode(query) {
  if (query.get('reelmode')) return query.get('reelmode');
  if (query.get('reelkeys')) return 'keys';
  return query.get('reel') ? 'behind' : 'stationary';
}

export function playbackState(api) {
  const camera = api?.cameraReel !== false;
  const running = camera ? !!api?.reel : !!api?.playing;
  return {
    camera,
    running,
    label: camera
      ? running
        ? 'Playing — press any key to stop'
        : 'Play camera move'
      : running
        ? 'Pause clip'
        : 'Play clip',
  };
}

export function togglePlayback(api) {
  if (api.cameraReel === false) api.play(!api.playing);
  else api.showreel(!api.reel);
}
