# Landing page — progress

Branch: `devin/landing` (integration PR into `main`: https://github.com/dtpu/htn2026/pull/2). Section work
landed as stacked PRs (#4–#8, #11, #12) into `devin/landing`; the demo-shell redesign (#9) merged to `main`
separately, with its review follow-ups in #13 (stacked on `devin/landing`). See `SPEC.md` and `PLAN.md` in this directory.

| Unit                         | Owns                                                             | Status                       | PR                                             |
| ---------------------------- | ---------------------------------------------------------------- | ---------------------------- | ---------------------------------------------- |
| Phase 0 — Foundation         | `site/` scaffold, tokens, shared components, content, smoke test | merged                       | [#2](https://github.com/dtpu/htn2026/pull/2)   |
| A. Hero                      | `site/src/sections/Hero.tsx`                                     | merged                       | [#8](https://github.com/dtpu/htn2026/pull/8)   |
| B. Gallery                   | `site/src/sections/Gallery.tsx`                                  | merged                       | [#5](https://github.com/dtpu/htn2026/pull/5)   |
| C. Compare                   | `site/src/sections/Compare.tsx`                                  | merged                       | [#4](https://github.com/dtpu/htn2026/pull/4)   |
| D. How it works              | `site/src/sections/HowItWorks.tsx`                               | merged                       | [#7](https://github.com/dtpu/htn2026/pull/7)   |
| E. Viewer features           | `site/src/sections/Viewer.tsx`                                   | merged                       | [#6](https://github.com/dtpu/htn2026/pull/6)   |
| F. Limits + Team + Final CTA | `site/src/sections/{Limits,Team,FinalCta}.tsx`                   | merged                       | [#12](https://github.com/dtpu/htn2026/pull/12) |
| Fonts                        | `site/index.html`, `tokens.css`, `base.css`                      | merged                       | [#11](https://github.com/dtpu/htn2026/pull/11) |
| Phase 2 — Integration        | `App.tsx` wiring, full-page visual pass                          | done except media and deploy |                                                |

Statuses: pending → in progress → PR open → merged → blocked.

## Integration pass (Phase 2)

Done:

- All eight sections wired in `App.tsx`; `bun run build`, `bun run format:check`, and
  `bun run test:smoke` (12 checks) pass; no horizontal overflow at 375, 768, or 1280 px.
- Fonts (Figtree, Gilda Display) are self-hosted from `@fontsource/*` instead of Google Fonts, so
  the page has no third-party requests and the smoke test stays clean offline.
- Gallery cards link to `/demo.html?clip=<id>`, the parameter `demo.html` actually reads.
- Demo and repository links live in `site/src/content/links.ts`; set `VITE_DEMO_URL` at build time
  once a hosted demo exists (SPEC open question 2).
- Team section lists the four repository collaborators by name and GitHub login. Role lines are
  omitted until the team agrees on wording.
- Pipeline steps render as a 3 × 2 grid on desktop (2 × 3 on tablet, one column on phones) so all
  six are visible without a hidden horizontal scroll.
- Compare captions say the body-height offset is pending measurement instead of quoting numbers
  that were never measured.

Media (20 Sep):

- `site/media-manifest.json` now lists every file the page expects, sourced from the private
  shared bucket: source clips from `/clips/`, the reviewed viewer playback loops from
  `reviews/austin-overnight/playback-final/`, the accepted gym preview and the current kitchen
  repair preview from their review folders, and run intermediates from the author archive for the
  pipeline strip. `scripts/pull-media.ts` reads the bucket with the same credential rules as
  `bun run assets:pull` and derives posters, compare frames and trimmed loops with ffmpeg.
- `scripts/capture-media.ts` renders the hero loop and the three compare renders from the real
  viewer (`fourd.html` driven through `window.wander`) and prints the viewpoint offsets in
  body-heights that the compare captions quote. The viewer-feature loops and the last three
  pipeline stills are cut from reviewed footage already in the bucket instead: rendering them
  without a GPU took about half an hour per loop, which is not worth it for a landing page.
- The Orbis Engine logo and header photo live in `site/src/brand/`; the black logo sits in a
  sticky top bar and the colour header is the hero masthead.
- The gallery has seven cards: the scenes with a clean reviewed loop. Review captures with the
  debug HUD burned in (`authored/*.mp4`, `walk/*.mp4`) and copyrighted film clips other than
  Tears of Steel were left out. Cards link to the demo only for ids the picker has
  (`elevator`, `lobby`, `tos31`, `gym-accepted`, `kitchen-repair`).

Remaining:

- Publish the captured renders to the shared bucket so a build needs only `pull-media.ts`; until
  then a deploy runs both scripts (see `site/README.md`).
- Hosting and the `VITE_DEMO_URL` target; Lighthouse pass after real media is in place.
- Consent for publishing teammate-filmed clips (SPEC open question 3).
