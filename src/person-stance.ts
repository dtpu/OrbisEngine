type StanceOffset = [number, number];
type ReadFeet = (frame: number) => readonly number[] | undefined;

/** Derive stance from a complete prefix, independently of seek and frame arrival order. */
export function createStanceResolver(readFeet: ReadFeet) {
  let offsets: StanceOffset[] = [[0, 0]];
  let cachedScale: number | undefined;
  let cachedUpm: number | undefined;
  return (frame: number, scale: number, upm: number): StanceOffset => {
    if (scale !== cachedScale || upm !== cachedUpm) {
      offsets = [[0, 0]];
      cachedScale = scale;
      cachedUpm = upm;
    }
    for (let i = offsets.length; i <= frame; i++) {
      const previous = readFeet(i - 1);
      const current = readFeet(i);
      // A missing predecessor is temporary. Never cache a neutral placeholder or
      // corrections derived from it; resume this prefix after its frames arrive.
      if (!previous || !current) return [0, 0];
      const p = offsets[i - 1];
      const dx = current[1] - previous[1];
      const dz = current[2] - previous[2];
      const planted = (Math.hypot(dx, dz) * scale) / upm < 0.03;
      let ox = planted ? p[0] - dx : p[0] * 0.6;
      let oz = planted ? p[1] - dz : p[1] * 0.6;
      const limit = (0.1 * upm) / scale;
      const magnitude = Math.hypot(ox, oz);
      if (magnitude > limit) {
        ox *= limit / magnitude;
        oz *= limit / magnitude;
      }
      offsets.push([ox, oz]);
    }
    return offsets[frame];
  };
}
