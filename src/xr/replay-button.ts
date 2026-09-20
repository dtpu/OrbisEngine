type ReplaySource = Pick<XRInputSource, 'handedness' | 'gamepad'>;

/** xr-standard button 4 is X on the left Touch controller. Require release before rearming. */
export class ReplayButton {
  private source: ReplaySource | null = null;
  private armed = false;

  reset() {
    this.source = null;
    this.armed = false;
  }

  update(sources: Iterable<ReplaySource>): boolean {
    const source =
      [...sources].find(
        (candidate) =>
          candidate.handedness === 'left' && candidate.gamepad?.mapping === 'xr-standard',
      ) ?? null;
    if (source !== this.source) {
      this.source = source;
      this.armed = false;
    }
    const button = source?.gamepad?.buttons[4];
    if (!button) return false;
    if (!button.pressed && button.value < 0.55) {
      this.armed = true;
      return false;
    }
    if (!this.armed) return false;
    this.armed = false;
    return true;
  }
}
