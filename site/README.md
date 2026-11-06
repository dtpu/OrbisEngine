# Orbis Engine landing page

Static Vite + React + TypeScript site; builds to `site/dist/`. The unified app gateway serves it
at `/about/`, with upload and job management at `/` and viewers at `/demo.html` and `/fourd.html`.
Run `bun run app` from the repository root after building the site for that complete flow.

## Commands

- `bun install --frozen-lockfile` — install dependencies.
- `bun run dev` — standalone server at `http://127.0.0.1:5399/about/`; app links need the gateway.
- `bun run build` — `tsc --noEmit` then `vite build`.
- `bun run format:check` — Prettier check.
- `bun run test:smoke` — Playwright smoke. Reuses the standalone server or starts one on port
  5399 and stops only its own process. Set `SITE_SMOKE_URL=http://127.0.0.1:5399/about/` to test
  the running gateway; external URLs are never started or stopped by the test.
- `bun run scripts/pull-media.ts` — pull optional media into `public/media/` from the manifest.

Smoke honors `CHROME_PATH` or `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`, then common local Chrome
paths, then Playwright's Chromium install. Set an executable path if no browser is found.
Default smoke verifies the fresh-checkout empty-media state; after installing media, use
`SITE_SMOKE_EMPTY_MEDIA=0` to retain section, content, link and failed-request checks with assets.

Media files are optional and never committed (`public/` is gitignored). The checked-in download
manifest is empty. At build or dev-server startup, Vite inventories actual `public/media/` files.
Missing previews render labeled unavailable states without requesting absent URLs. The hero uses
the tracked project still from `docs/images/hero.jpg`, explicitly labeled as a still. Rebuild or
restart Vite after installing media. Missing previews do not imply successful reconstructions.

`/about/?dev=1` renders a component demo page. Fonts are self-hosted; no third-party requests.
Build-time configuration:

- `VITE_SITE_BASE` — site mount path, default `/about/` (use `/` for standalone root hosting).
- `VITE_APP_URL` — upload and jobs app URL, default `/`.
- `VITE_DEMO_URL` — viewer URL, default `/demo.html`; scene links preserve query options and
  fragments while setting the `clip` parameter.
