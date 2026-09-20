# Landing page — progress

Branch: `devin/landing` (integration PR into `main`). Section work lands as PRs into `devin/landing`.
See `SPEC.md` and `PLAN.md` in this directory.

| Unit | Owns | Status | PR | Session |
| --- | --- | --- | --- | --- |
| Phase 0 — Foundation | `site/` scaffold, tokens, shared components, content, smoke test | done | (on `devin/landing`) | parent |
| A. Hero | `site/src/sections/Hero.tsx` | pending | | |
| B. Gallery | `site/src/sections/Gallery.tsx` | pending | | |
| C. Compare | `site/src/sections/Compare.tsx` | pending | | |
| D. How it works | `site/src/sections/HowItWorks.tsx` | pending | | |
| E. Viewer features | `site/src/sections/Viewer.tsx` | pending | | |
| F. Limits + Team + Final CTA | `site/src/sections/{Limits,Team,FinalCta}.tsx` | pending | | |
| Phase 2 — Integration | `App.tsx` wiring, full-page visual pass | pending | | |

Statuses: pending → in progress → PR open → merged → blocked.

## Notes

- Media export (posters/loops under `site/public/media/`) is not done; sections render against
  placeholders until it lands.
