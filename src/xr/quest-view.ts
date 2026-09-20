import type { WebGLRenderer, Vector4 } from 'three';

const ENDPOINT = '/api/quest-view';
const FRAME_INTERVAL = 1000 / 60;
const MAX_EDGE = 512;

/** Copies one already-rendered eye. The relay holds only the latest JPEG, never a recording. */
export function createQuestView(renderer: WebGLRenderer, clip: string) {
  const state = {
    status: 'idle',
    viewers: 0,
    frames: 0,
    width: 0,
    height: 0,
    error: '',
    sourceGlError: 0,
    captureMs: 0,
    uploadMs: 0,
  };
  const gl = renderer.getContext() as WebGL2RenderingContext;
  let framebuffer: WebGLFramebuffer | null = null;
  let color: WebGLRenderbuffer | null = null;
  let captureWidth = 0;
  let captureHeight = 0;
  let pixels = new Uint8Array(0);
  let encoder: Worker | null = null;
  let cancelEncoding: (() => void) | null = null;
  let sessionId = '';
  let generation = 0;
  let sequence = 0;
  let disposed = false;
  let busy = false;
  let uploading = false;
  let pendingFrame: { blob: Blob; id: string; epoch: number; capturedAt: number } | null = null;
  let nextCapture = 0;
  let acknowledgedAt = 0;
  let heartbeatTimer = 0;
  let heartbeatRequest: AbortController | null = null;
  let frameRequest: AbortController | null = null;

  async function heartbeat(id: string, epoch: number) {
    if (disposed || epoch !== generation) return;
    const request = new AbortController();
    heartbeatRequest = request;
    const timeout = window.setTimeout(() => request.abort(), 1800);
    try {
      const response = await fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: id, clip, active: true }),
        signal: request.signal,
      });
      if (!response.ok) throw new Error(`Quest view relay: ${response.status}`);
      const status = (await response.json()) as { viewers: number };
      if (epoch !== generation || disposed) return;
      state.viewers = Number.isFinite(status.viewers) ? Math.max(0, status.viewers) : 0;
      acknowledgedAt = performance.now();
      state.status = state.viewers ? (state.frames ? 'live' : 'connecting') : 'waiting';
    } catch (error) {
      if (epoch !== generation || disposed) return;
      state.viewers = 0;
      state.status = 'unavailable';
      state.error = error instanceof Error ? error.message : String(error);
    } finally {
      clearTimeout(timeout);
      if (heartbeatRequest === request) heartbeatRequest = null;
      if (epoch === generation && !disposed)
        heartbeatTimer = window.setTimeout(() => void heartbeat(id, epoch), 1000);
    }
  }

  function stop() {
    const id = sessionId;
    sessionId = '';
    generation++;
    clearTimeout(heartbeatTimer);
    heartbeatRequest?.abort();
    frameRequest?.abort();
    pendingFrame = null;
    cancelEncoding?.();
    encoder?.terminate();
    encoder = null;
    state.status = 'idle';
    state.viewers = 0;
    acknowledgedAt = 0;
    if (id)
      void fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: id, clip, active: false }),
        keepalive: true,
      }).catch(() => {});
  }

  function start() {
    if (disposed) return;
    stop();
    sessionId = crypto.randomUUID();
    state.frames = 0;
    sequence = 0;
    state.error = '';
    state.status = 'connecting';
    nextCapture = 0;
    void heartbeat(sessionId, generation);
  }

  async function copyEye(viewport: Vector4, epoch: number): Promise<Blob> {
    const scale = Math.min(1, MAX_EDGE / Math.max(viewport.z, viewport.w));
    const width = Math.max(1, Math.round(viewport.z * scale));
    const height = Math.max(1, Math.round(viewport.w * scale));
    // Keep prior renderer errors visible in diagnostics, separate from errors in this copy.
    // Quest can report an earlier XR texture update here even when both framebuffers are valid.
    state.sourceGlError = gl.getError();
    if (state.sourceGlError) for (let i = 0; i < 8 && gl.getError() !== gl.NO_ERROR; i++) {}
    // Raw GL changes are restored synchronously before yielding, preserving Three's state cache.
    const read = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const draw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const renderbuffer = gl.getParameter(gl.RENDERBUFFER_BINDING) as WebGLRenderbuffer | null;
    const pack = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING) as WebGLBuffer | null;
    const packParameters = [
      gl.PACK_ALIGNMENT,
      gl.PACK_ROW_LENGTH,
      gl.PACK_SKIP_ROWS,
      gl.PACK_SKIP_PIXELS,
    ];
    const packValues = packParameters.map((parameter) => gl.getParameter(parameter) as number);
    const scissor = gl.isEnabled(gl.SCISSOR_TEST);
    const buffer = gl.createBuffer();
    let fence: WebGLSync | null = null;
    try {
      try {
        if (!buffer) throw new Error('Quest view readback unavailable');
        if (!framebuffer || captureWidth !== width || captureHeight !== height) {
          if (framebuffer) gl.deleteFramebuffer(framebuffer);
          if (color) gl.deleteRenderbuffer(color);
          framebuffer = gl.createFramebuffer();
          color = gl.createRenderbuffer();
          if (!framebuffer || !color) throw new Error('Quest view framebuffer unavailable');
          gl.bindRenderbuffer(gl.RENDERBUFFER, color);
          gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, width, height);
          gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
          gl.framebufferRenderbuffer(
            gl.DRAW_FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.RENDERBUFFER,
            color,
          );
          if (gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
            throw new Error('Quest view framebuffer incomplete');
          captureWidth = width;
          captureHeight = height;
        }
        if (pixels.byteLength !== width * height * 4) pixels = new Uint8Array(width * height * 4);
        const target = renderer.getRenderTarget();
        // Three's resolved framebuffer covers both XRProjectionLayer and XRWebGLLayer.
        // It is accessible only inside this XR render callback; null is the fake-XR backbuffer.
        const properties = target
          ? (renderer.properties.get(target) as {
              __webglFramebuffer?: WebGLFramebuffer | WebGLFramebuffer[] | null;
            })
          : null;
        const source = properties ? properties.__webglFramebuffer : draw;
        if (Array.isArray(source)) throw new Error('Unsupported Quest view framebuffer');
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, source ?? null);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
        gl.disable(gl.SCISSOR_TEST);
        gl.blitFramebuffer(
          viewport.x,
          viewport.y,
          viewport.x + viewport.z,
          viewport.y + viewport.w,
          0,
          0,
          width,
          height,
          gl.COLOR_BUFFER_BIT,
          gl.LINEAR,
        );
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, pixels.byteLength, gl.STREAM_READ);
        packParameters.forEach((parameter, i) => gl.pixelStorei(parameter, i === 0 ? 1 : 0));
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, 0);
        if (gl.getError() !== gl.NO_ERROR) throw new Error('Quest view framebuffer copy failed');
        fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
        if (!fence) throw new Error('Quest view GPU synchronization unavailable');
      } finally {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, read);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, draw);
        gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pack);
        packParameters.forEach((parameter, i) => gl.pixelStorei(parameter, packValues[i]));
        if (scissor) gl.enable(gl.SCISSOR_TEST);
      }
      const deadline = performance.now() + 1000;
      while (true) {
        if (disposed || epoch !== generation || gl.isContextLost())
          throw new Error('Quest view ended');
        const status = gl.clientWaitSync(fence, 0, 0);
        if (status === gl.WAIT_FAILED || performance.now() > deadline)
          throw new Error('Quest view readback timed out');
        if (status !== gl.TIMEOUT_EXPIRED) break;
        await new Promise((resolve) => window.setTimeout(resolve, 4));
      }
      const currentPack = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING) as WebGLBuffer | null;
      try {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, pixels);
      } finally {
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, currentPack);
      }
      state.width = width;
      state.height = height;
      encoder ??= new Worker(new URL('./quest-view-encoder.ts', import.meta.url), {
        type: 'module',
      });
      const worker = encoder;
      return await new Promise<Blob>((resolve, reject) => {
        const finish = (error?: Error, blob?: Blob) => {
          clearTimeout(timeout);
          cancelEncoding = null;
          worker.onmessage = null;
          worker.onerror = null;
          if (error) {
            worker.terminate();
            if (encoder === worker) encoder = null;
            reject(error);
          } else resolve(blob!);
        };
        const timeout = window.setTimeout(
          () => finish(new Error('Quest view encoding timed out')),
          1800,
        );
        cancelEncoding = () => finish(new Error('Quest view ended'));
        worker.onerror = () => finish(new Error('Quest view encoder unavailable'));
        worker.onmessage = ({
          data,
        }: MessageEvent<{ blob?: Blob; pixels?: Uint8Array<ArrayBuffer>; error?: string }>) => {
          if (epoch !== generation || disposed) finish(new Error('Quest view ended'));
          else if (data.blob instanceof Blob) {
            // Return the readback buffer from the encoder instead of allocating it each frame.
            if (data.pixels instanceof Uint8Array && data.pixels.byteLength === width * height * 4)
              pixels = data.pixels;
            finish(undefined, data.blob);
          } else finish(new Error(data.error || 'Quest view encoding failed'));
        };
        worker.postMessage({ pixels, width, height }, [pixels.buffer]);
      });
    } finally {
      if (fence) gl.deleteSync(fence);
      if (buffer) gl.deleteBuffer(buffer);
    }
  }

  async function captureFrame(viewport: Vector4, id: string, epoch: number) {
    busy = true;
    const started = performance.now();
    try {
      const blob = await copyEye(viewport, epoch);
      if (epoch !== generation || disposed || !state.viewers) return;
      state.captureMs = performance.now() - started;
      pendingFrame = { blob, id, epoch, capturedAt: started };
      flushUpload();
    } catch (error) {
      if (epoch === generation && !disposed) {
        state.error = error instanceof Error ? error.message : String(error);
        state.status = 'unavailable';
        nextCapture = performance.now() + 1000;
      }
    } finally {
      busy = false;
    }
  }

  function flushUpload() {
    if (uploading || !pendingFrame || disposed) return;
    const frame = pendingFrame;
    pendingFrame = null;
    // Only one upload and one following capture may exist. Never drain an old frame queue.
    if (frame.epoch !== generation || !state.viewers || performance.now() - frame.capturedAt > 250)
      return;
    uploading = true;
    void uploadFrame(frame);
  }

  async function uploadFrame({ blob, id, epoch }: NonNullable<typeof pendingFrame>) {
    const request = new AbortController();
    frameRequest = request;
    const timeout = window.setTimeout(() => request.abort(), 2000);
    try {
      // A timed-out request may already have reached the relay. Never reuse its sequence.
      const seq = ++sequence;
      const uploadStarted = performance.now();
      const response = await fetch(
        `${ENDPOINT}/frame?session=${encodeURIComponent(id)}&seq=${seq}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'image/jpeg' },
          body: blob,
          signal: request.signal,
        },
      );
      if (!response.ok) throw new Error(`Quest view upload: ${response.status}`);
      if (epoch !== generation || disposed) return;
      state.uploadMs = performance.now() - uploadStarted;
      state.frames++;
      state.status = 'live';
      state.error = '';
    } catch (error) {
      if (epoch === generation && !disposed) {
        state.error = error instanceof Error ? error.message : String(error);
        state.status = 'unavailable';
        nextCapture = performance.now() + 1000;
      }
    } finally {
      clearTimeout(timeout);
      if (frameRequest === request) frameRequest = null;
      uploading = false;
      flushUpload();
    }
  }

  renderer.xr.addEventListener('sessionstart', start);
  renderer.xr.addEventListener('sessionend', stop);
  window.addEventListener('pagehide', stop);
  return {
    state,
    afterRender() {
      const now = performance.now();
      if (
        disposed ||
        !sessionId ||
        !renderer.xr.isPresenting ||
        !state.viewers ||
        busy ||
        pendingFrame !== null ||
        now + 0.75 < nextCapture ||
        now - acknowledgedAt > 2500
      )
        return;
      const viewport = (renderer.xr.getCamera().cameras[0] as { viewport?: Vector4 } | undefined)
        ?.viewport;
      if (!viewport || viewport.z <= 0 || viewport.w <= 0) return;
      // Keep a stable 60 Hz deadline. Scheduling from "now" can accidentally halve the
      // rate when a display frame arrives just before the previous deadline.
      const periods = Math.max(1, Math.floor((now - nextCapture) / FRAME_INTERVAL) + 1);
      nextCapture = nextCapture > 0 ? nextCapture + periods * FRAME_INTERVAL : now + FRAME_INTERVAL;
      void captureFrame(viewport, sessionId, generation);
    },
    dispose() {
      stop();
      disposed = true;
      renderer.xr.removeEventListener('sessionstart', start);
      renderer.xr.removeEventListener('sessionend', stop);
      window.removeEventListener('pagehide', stop);
      if (framebuffer) gl.deleteFramebuffer(framebuffer);
      if (color) gl.deleteRenderbuffer(color);
    },
  };
}
