// Shared link targets. The demo runs on the viewer's own Vite server today; set VITE_DEMO_URL at
// build time once a hosted demo exists (see SPEC open question 2).
export const DEMO_URL: string = import.meta.env.VITE_DEMO_URL || '/demo.html';
export const REPO_URL = 'https://github.com/dtpu/htn2026';

// demo.html selects a scene from its `clip` query parameter.
export const demoSceneUrl = (id: string) => `${DEMO_URL}?clip=${encodeURIComponent(id)}`;
