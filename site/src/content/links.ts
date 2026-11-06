// App and viewers share the gateway origin by default.
export const APP_URL: string = import.meta.env.VITE_APP_URL || '/';
export const DEMO_URL: string = import.meta.env.VITE_DEMO_URL || '/demo.html';
export const REPO_URL = 'https://github.com/dtpu/OrbisEngine';
export function demoSceneUrl(id: string, demoUrl = DEMO_URL): string {
  const url = new URL(demoUrl, 'https://orbis.invalid');
  url.searchParams.set('clip', id);
  return /^https?:\/\//.test(demoUrl) ? url.href : `${url.pathname}${url.search}${url.hash}`;
}
