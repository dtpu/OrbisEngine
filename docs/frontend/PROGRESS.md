# Landing page — progress

Branch: `devin/landing` (integration PR into `main`: https://github.com/dtpu/htn2026/pull/2). Section work lands as PRs into `devin/landing`.
See `SPEC.md` and `PLAN.md` in this directory.

| Unit | Owns | Status | PR | Session |
| --- | --- | --- | --- | --- |
| Phase 0 — Foundation | `site/` scaffold, tokens, shared components, content, smoke test | done | (on `devin/landing`) | parent |
| A. Hero | `site/src/sections/Hero.tsx` | in progress | | [session](https://app.devin.ai/sessions/0ac4ab8757064a3bbec814ba1cecd7de) |
| B. Gallery | `site/src/sections/Gallery.tsx` | in progress | | [session](https://app.devin.ai/sessions/c34112b0c4d54709af87b619ec048643) |
| C. Compare | `site/src/sections/Compare.tsx` | in progress | | [session](https://app.devin.ai/sessions/678f4613fc0f48a2b8b4c16bcb79f5dd) |
| D. How it works | `site/src/sections/HowItWorks.tsx` | in progress | | [session](https://app.devin.ai/sessions/0a2bf8410aee4a7da8745d02e590ae8e) |
| E. Viewer features | `site/src/sections/Viewer.tsx` | in progress | | [session](https://app.devin.ai/sessions/d8f8d27efce54556aa41b2bb605e9855) |
| F. Limits + Team + Final CTA | `site/src/sections/{Limits,Team,FinalCta}.tsx` | in progress | | [session](https://app.devin.ai/sessions/c865d9141022438084fdb523de188f0c) |
| Phase 2 — Integration | `App.tsx` wiring, full-page visual pass | pending | | |

Statuses: pending → in progress → PR open → merged → blocked.

## Notes

- Media export (posters/loops under `site/public/media/`) is not done; sections render against
  placeholders until it lands.
