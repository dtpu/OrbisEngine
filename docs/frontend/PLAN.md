# Wander landing page — PLAN

Companion to `SPEC.md` (same directory). Two phases: a small serial foundation, then fully parallel section work
where no two units touch the same file.

## Phase 0 — Foundation (serial, one owner, ~1 session)

Everything else depends on this; nothing else starts until it is merged.

```
site/
  package.json            vite, react, react-dom, typescript, @types/react*, playwright-core
  tsconfig.json
  vite.config.ts
  index.html              <div id="root">, Inter preload, meta/OG tags
  src/
    main.tsx              mounts <App/>
    App.tsx               imports every section in order; each section is a placeholder stub
    styles/
      tokens.css          --bg, --fg, --muted, --rule, --max-w, --max-w-wide, spacing + type scale
      base.css            reset, body type, [data-theme="dark"] inversion, focus styles
    components/
      Section.tsx         <Section id theme="light|dark" wide?> — padding, max-width, hairline rule
      Container.tsx
      Button.tsx          primary (filled) / secondary (outlined), one size
      Heading.tsx         Eyebrow + H1/H2 typographic variants
      VideoLoop.tsx       muted/loop/playsInline, poster required, IntersectionObserver play/pause
      Card.tsx            media + title + caption + optional label pill
      CompareSlider.tsx   two <img>/<video> with a draggable divider (pointer events, keyboard)
    content/
      scenes.ts           typed list of the 9 scenes {id, title, caption, label?, media paths}
      steps.ts            the six pipeline steps
      limits.ts           limits bullets
    sections/             one file per section, exported stubs: Hero, Gallery, Compare,
                          HowItWorks, Viewer, Limits, Team, FinalCta
  public/media/.gitkeep   loops/posters land here; gitignored
  scripts/pull-media.ts   fetches loops/posters from the private store into public/media
  tests/smoke.spec.ts     page loads, no console errors, every <video> has poster+src
```

Definition of done: `bun run dev` renders all eight stubbed sections with correct theme
inversion; `tsc` and Prettier pass; smoke test passes against the stubs; `VideoLoop`,
`CompareSlider`, and `Card` have a Storybook-free demo route (`?dev=1` renders them with a sample
asset) so section owners can see them.

Also in Phase 0, but can run alongside foundation code since it touches no `site/` files:

- **Media export** — for each of the 9 scenes, produce `poster.jpg`, `source.mp4`, `wander.mp4`
  (≤5 s, ≤2 MB), plus 3–4 matched-time compare pairs and one artifact image per pipeline step.
  Publish to the asset store; record the manifest `scripts/pull-media` reads. This is the long
  pole and needs someone with author credentials.

## Phase 1 — Parallel section units

Each unit owns exactly one file in `src/sections/` (plus an optional `*.module.css` beside it) and
may only read from `components/` and `content/`. No unit edits `App.tsx`, `tokens.css`, or any
shared component; if a shared component needs a change, file it back to the foundation owner
rather than editing it in place.

| Unit | Owns | Uses | Notes |
| --- | --- | --- | --- |
| A. Hero | `sections/Hero.tsx` | Section(dark), Heading, Button, VideoLoop | Split source/Wander loop; two CTAs. |
| B. Gallery | `sections/Gallery.tsx` | Section(wide), Card, VideoLoop, `content/scenes` | 3-col grid; hover-to-play; observed/generated toggle; label pills. |
| C. Compare | `sections/Compare.tsx` | Section, CompareSlider | 3–4 pairs; caption with body-height offset. |
| D. How it works | `sections/HowItWorks.tsx` | Section, `content/steps` | Six-step strip, horizontal scroll on mobile. |
| E. Viewer features | `sections/Viewer.tsx` | Section, VideoLoop | Three columns + headset line. |
| F. Limits + Team + Final CTA | `sections/Limits.tsx`, `Team.tsx`, `FinalCta.tsx` | Section, Heading, Button, `content/limits` | Small; bundled as one unit. |

Six units → up to six agents/people in parallel. Each unit's done check: its section renders with
real media from `public/media`, passes `tsc`/Prettier, adds one assertion to `tests/smoke.spec.ts`
**in its own describe block** (append-only; blocks are independent so merges don't conflict), and
has been eyeballed at 375px, 768px, and 1280px widths.

## Phase 2 — Integration (serial, short)

- Replace stubs in `App.tsx` with the real sections (already exported under the same names, so this
  is a no-op if stubs were exported correctly).
- Lighthouse pass: LCP < 2.5 s on the hero poster, total media on first load < 3 MB.
- Copy edit against SPEC.md; fill Team placeholders; confirm "Try the demo" target (open question 2).
- Deploy static `dist/` to the chosen host; add the URL to `README.md`.

## Risks

- Media export (Phase 0) gates every visual unit; start it first. Units can be built against a single
  placeholder loop and swapped when real media lands.
- `CompareSlider` is the only non-trivial component; it needs keyboard support and touch handling
  and should be reviewed before Unit C starts.
- Public publication of teammate footage needs consent (open question 3).
