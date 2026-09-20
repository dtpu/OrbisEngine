# Wander landing page — SPEC

Status: draft for review. Nothing is built yet.

## Pitch

**Wander turns a single video into a place you can walk around in.**

One handheld clip goes in. Out comes a navigable 3D scene: the room reconstructed and filled where
the camera never looked, the people moving through it as they did on film, objects placed where they
were, and the original audio in sync. You stand inside the moment and choose your own viewpoint;
the recording is the timeline and you are an observer.

Secondary line, for people who want the honest version: new viewpoints can expose generated
content, and Wander labels what was observed versus inferred rather than hiding it.

## Audience

1. **Hackathon judges and technical reviewers** (primary, near term). They have ~2 minutes, want to
   see it working immediately, and reward honest, well-evidenced results over hype.
2. **3D / graphics / ML practitioners** (secondary). They want to understand the pipeline stages and
   what is actually novel. They will look for a "how it works" section and limits.
3. **Curious visitors** who land from a shared link. They need the one-line idea and a video that
   proves it in the first screen.

Not an audience yet: paying customers. No pricing section (see assumptions).

## Tone and voice

- Plain, declarative, first-person-plural sparingly ("we"), mostly impersonal statements of fact.
- Show, then explain. Every claim sits next to a real example clip or comparison.
- No superlatives, no "revolutionary", no exclamation marks. Say "reconstructs" not "magically".
- Honest about limits: the page has a visible Limits section. This is a differentiator, not a
  disclaimer buried in a footer.
- Short sentences. Copy blocks under 40 words.

## Page structure and draft copy

The page is one long scroll, examples-dense, in the spirit of a research project page (many
side-by-side results, minimal chrome) but not a copy of any particular one.

### 1. Hero

- Eyebrow: `WANDER`
- H1: **Step inside a video.**
- Sub: One clip becomes a room you can walk through, with its people, objects, and sound replaying
  exactly as recorded.
- Primary CTA: **Try the demo** → `/demo.html` (existing viewer, runs locally today; see assumptions)
- Secondary CTA: **See how it works** → anchor to §4
- Visual: full-width autoplaying muted loop, split view: source footage left, Wander viewpoint
  moving through the same moment right. Use the elevator clip (two people, one handheld take).

### 2. Examples gallery ("Walk through these")

The core of the page. A grid of example cards, each a hover/tap-to-play loop with a caption and a
"Open in viewer" link. Aim for 8–9 cards using the scenes that already exist in the demo picker:

| Scene | Card title | One-line caption |
| --- | --- | --- |
| elevator | Two people, one handheld take | Lift doors, railings and floor give the reconstruction landmarks. |
| lobby | A lobby with a waving subject | Wide space, one subject, clear motion. |
| stairs2 | A short stair ascent | Vertical motion; the stairs must support walking. |
| atrium | Night atrium, partial view | Dark, glass, and a moving camera on a short clip. |
| tos31 | Tears of Steel · shot 31 | Cinematic footage with a reflective floor. |
| gym | Gym — the hardest pose | Body partly occluded by equipment. |
| hp-walk | Person and place, separated | Shows the person and room branches side by side. |
| tos | Tears of Steel — the older shot | An earlier attempt kept for comparison. |
| living | Living room | Furniture occlusion and a sharper alternate reconstruction. |

Each card has three states: source frame, reconstruction, and a small "observed / generated" toggle
overlay where evidence exists. Cards with known flaws keep an honest one-word label
("lighting mismatch", "lower body inferred") instead of being hidden.

### 3. Compare ("Same second, new angle")

Three to four before/after sliders: a source frame at time *t* next to the Wander render from a
viewpoint the camera never had, at the same *t*. Caption states the offset in body-heights, not
metres. Copy:

> Drag to compare. Left is the recorded frame. Right is the same instant from a place the camera
> never stood.

### 4. How it works (six steps)

Horizontal step strip, each step a real intermediate artifact from one run (gym or elevator):

1. **Original recording** — the clip and the interval used.
2. **Understand motion** — camera path, point cloud, tracked person.
3. **Separate people and room** — mask and cleaned video; removed regions get generated fill.
4. **Build room and people** — generated environment beside the inferred body.
5. **Fit and animate** — placement, object tracks, remaining contact errors shown.
6. **Explore the replay** — packaged scene, original audio, walk controls.

Footer line: "Completed processing is not accepted quality. Every scene above passed a manual
visual review against its source."

### 5. What you can do in the viewer

Three short columns with a 3–5 s loop each:

- **Walk** — W/A/S/D and mouse look; floors and stairs support you, walls stop you.
- **Rewind the moment** — the source clip owns the timeline; scrub and the whole scene follows.
- **Hear it where it happened** — original audio in sync; spatial tracks where reviewed.

Plus one line: **Works in a headset** — Quest via WebXR, teleport or smooth locomotion.

### 6. Limits

A plain bulleted list, straight from `docs/known-limits.md`, lightly reworded:

- One moving camera can't recover smoke, fire, or water.
- Faces smaller than ~64 px stay unreadable.
- Too little parallax or too little light and the reconstruction fails.
- Unseen geometry is generated, and labelled as such.

Copy above the list: "What Wander can't do yet. We'd rather you find out here than in the viewer."

### 7. Team / credits

Names, roles, one line each, GitHub links. Built at Hack the North 2026. Third-party components
credited (Three.js, Spark, Marble, model names actually used).

### 8. Final CTA

- H2: **Go walk around.**
- Buttons: **Try the demo** · **Read the code** (GitHub repo)

### Omitted on purpose

Pricing, testimonials/social proof, newsletter signup, FAQ. None fit a hackathon research demo;
a short Limits section does the FAQ's job better.

## Design direction

- **Palette:** pure black `#000` and white `#fff` only, plus one grey (`#888`) for captions and
  hairline borders (`#222` on dark / `#e5e5e5` on light). No accent color. Default theme is white
  background; hero and final CTA sections invert to black so the video loops sit on black.
- **Type:** one clean sans, system stack first (`Inter, ui-sans-serif, system-ui`), self-hosted
  Inter as the single web font. Headings tight (`letter-spacing -0.02em`), weights 400/600 only.
  Body 17px / 1.6. Mono (`ui-monospace`) for the eyebrow and any command snippets.
- **Layout:** max content width 1200px; gallery breaks out to 1400px. Generous vertical rhythm
  (section padding 96–128px). Grid: 1 col mobile, 2 tablet, 3 desktop for cards. Hairline rules
  between sections instead of background color changes.
- **Media:** all examples are short muted MP4/WebM loops (≤5 s, ≤2 MB each), poster images for
  first paint, lazy-loaded below the fold. Autoplay on intersection, pause off-screen.
- **Motion:** none beyond video playback and a 150ms opacity/transform on hover. No scroll-jacking,
  no parallax.
- **Feel:** a well-typeset research page. Whitespace does the work. No gradients, shadows, glass,
  or rounded blobs. 4px radius max on buttons and cards.

## Tech stack

- **Vite + React 19 + TypeScript**, in a new `site/` directory inside this repo with its own
  `package.json`, so it shares nothing with the Three.js viewer build and can't break it.
  Reasoning: you asked for clean React; the repo already uses Vite/Bun/Prettier so tooling is
  familiar; no framework features (routing, SSR, data fetching) are needed for a single page.
- **Styling:** plain CSS modules + a single `tokens.css` of custom properties. No Tailwind, no
  CSS-in-JS. Black/white/one grey does not justify a utility framework.
- **No component library, no animation library, no analytics.**
- **Output:** static `dist/`, deployable anywhere (GitHub Pages / Cloudflare Pages / S3). Media
  assets are not committed; they are pulled from the private asset store into `site/public/media/`
  by a script at build time, or referenced by public URLs once published.
- **Quality gates:** `tsc --noEmit`, Prettier (repo config), and a Playwright smoke test that loads
  the page and asserts every `<video>` has a poster and a reachable src.

## Open questions / assumptions

1. **Product = Wander.** Your message left the product placeholder unfilled; I assumed this repo's
   own project. If it's something else, the structure holds but every word of copy changes.
2. **"Try the demo" target.** The viewer currently runs on `127.0.0.1:5399` with private S3
   credentials. I assumed a hosted read-only demo will exist (or the button links to the README's
   run instructions until then). Needs a decision before launch.
3. **Media publication.** Example loops must be exported from existing accepted scenes and hosted
   publicly. I assumed we may publish short clips of the existing 9 scenes; Tears of Steel is CC-BY
   and fine, but any teammate-filmed footage needs the people in it to be OK with it.
4. **No pricing / signup.** Assumed this is a demo/research page, not a product launch.
5. **Team section content** (names, roles, links) is unknown to me; left as placeholders.
6. **Repo location.** Assumed `site/` inside `htn2026` rather than a separate repo, to keep the
   examples next to the code that made them. Easy to move later.
7. **Dark hero, light body.** You said black-and-white; I chose white-dominant with inverted
   hero/CTA. Flip to black-dominant if you prefer — it's one token change.
8. **Domain / hosting** not chosen. Static export keeps every option open.
