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
- `bun run scripts/capture-media.ts` — render the media only the viewer can make (see below).

## Media

Media files are pulled at build time, never committed (`public/` is gitignored).
`media-manifest.json` is the one inventory of what the page expects under `public/media/`. Each
entry is either a file in the private shared bucket (`asset`, the pinned viewer snapshot that
`bun run demo` serves, or `archive`, the author archive), a public `url`, a still or trimmed loop
derived with ffmpeg from an earlier entry (`frame`, `clip`, `ffmpeg`), or a `capture` that
`scripts/capture-media.ts` renders from the running viewer (the hero loop and the three compare
renders; everything else comes from footage already in the bucket).

1. From the repository root, `bun install --frozen-lockfile` and save the teammate credentials as
   `.env.local` (see `docs/shared-assets.md`); the pull uses the same credential rules as
   `bun run assets:pull`. Entries marked `optional` need author AWS credentials and are skipped
   without them.
2. `ffmpeg` must be on `PATH`, or set `FFMPEG=/path/to/ffmpeg`.
3. `bun run scripts/pull-media.ts` downloads and derives everything it can.
4. Start the viewer at the root (`bun run demo`) and run `bun run scripts/capture-media.ts`. It
   drives `fourd.html` through `window.wander`, reads frames straight from the WebGL canvas, and
   prints the viewpoint offsets in body-heights that the compare captions quote. It needs Chrome
   (`CHROME_PATH`, the usual install paths, or Playwright's own download). On a machine with a
   GPU it takes a few minutes; without one, set `LOD_COUNT=150000` and expect about an hour for
   the hero loop. Set `VIEWER_URL` for another viewer port.
5. Run `bun run scripts/pull-media.ts` again for the posters taken from the captured loops.

Existing files are kept; delete a file to rebuild it, or pass `--force` to the capture script.

`/?dev=1` renders a component demo page.

Fonts (Figtree, Gilda Display) are self-hosted from `@fontsource/*`; the page makes no third-party
requests. Set `VITE_DEMO_URL` at build time to point the demo buttons at a hosted viewer (default
`/demo.html`).
