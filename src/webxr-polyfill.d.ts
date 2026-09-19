declare module 'webxr-polyfill' {
  export default class WebXRPolyfill {
    constructor(config?: {
      cardboard?: boolean;
      allowCardboardOnDesktop?: boolean;
      global?: any;
      webvr?: boolean;
    });
  }
}
