# Wander landing page

Static Vite + React + TypeScript site; builds to `site/dist/`.

## Commands

- `bun install` — install dependencies.
- `bun run dev` — dev server at `http://127.0.0.1:5400`.
- `bun run build` — `tsc --noEmit` then `vite build`.
- `bun run format:check` — Prettier check.
- `bun run test:smoke` — Playwright smoke test against the dev server. It looks for Chrome at
  `CHROME_PATH`, then common install paths, then Playwright's own download; point `CHROME_PATH` at a
  Chromium binary if none of those exist.
- `bun run scripts/pull-media.ts` — pull media into `public/media/` from `media-manifest.json`.

Media files are pulled at build time, never committed (`public/` is gitignored).

`/?dev=1` renders a component demo page.

Fonts (Figtree, Gilda Display) are self-hosted from `@fontsource/*`; the page makes no third-party
requests. Set `VITE_DEMO_URL` at build time to point the demo buttons at a hosted viewer (default
`/demo.html`).
