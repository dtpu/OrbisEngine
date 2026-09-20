# Wander admin

Operator console for pipeline runs: start a run from a clip, watch stages in dependency order,
look at what each attempt produced, and approve, retry, pause or cancel with a recorded reason.

## Run it

```sh
cd admin
bun install --frozen-lockfile
WANDER_PIPELINE_API=http://127.0.0.1:8000 WANDER_API_TOKEN=… bun run dev -- --port 5400 --hostname 127.0.0.1
```

The browser only ever talks to this server. `app/api/pipeline/[[...path]]/route.ts` proxies to the
orchestrator API and attaches the bearer token server-side, so the token never reaches the page.

- `WANDER_PIPELINE_API` — orchestrator base URL (default `http://127.0.0.1:8000`).
- `WANDER_API_TOKEN` — the API's bearer token; without it every request answers 503.

## What the pages show

- **Runs** (`/`): every run with a stage strip (one cell per stage, dependency order, coloured by
  status), plus a banner listing gates waiting on you. Shot runs nest under their parent.
- **Run** (`/runs/<id>`): stage ladder on the left, grouped by dependency depth; the inspector on
  the right shows the selected stage's attempts, an artifact viewer (video, image, audio, JSON,
  text inline; anything else as a download), and the decision panel. `?stage=<id>` opens a stage
  directly; otherwise the page lands on whatever needs attention first.

The viewer reads bytes from `GET /api/pipeline/runs/{run}/artifacts/{artifact}` on the orchestrator
(`orchestrator/api/artifacts.py`). Only image, video, audio, JSON and plain-text types render
inline; everything else is served as an octet-stream download.

## Design notes

Neutral review-suite grey, no tint, so footage is the only saturated thing on the page. Colour is
reserved for status and always accompanies a word. IBM Plex Sans for prose and controls, IBM Plex
Mono for identifiers, hashes, sizes and times. Tokens live at the top of `app/globals.css`.
