# Orbis admin

Operator console for pipeline runs: start a run from a clip, watch stages in dependency order,
look at what each attempt produced, and approve, retry, pause or cancel with a recorded reason.

Pause and cancel prevent new stages and agent turns. Provider operations already submitted may
still finish; their results and recovery receipts are retained. Cancellation stays in effect
until an operator explicitly resumes or retries the run.

## Run it

```sh
cd admin
bun install --frozen-lockfile
WANDER_PIPELINE_API=http://127.0.0.1:8000 WANDER_API_TOKEN=… bun run dev -- --port 5401 --hostname 127.0.0.1
```

The browser only ever talks to this server. `app/api/pipeline/[[...path]]/route.ts` proxies to the
orchestrator API and attaches the bearer token server-side, so the token never reaches the page.

- `WANDER_PIPELINE_API` — orchestrator base URL (default `http://127.0.0.1:8000`).
- `WANDER_API_TOKEN` — the API's bearer token; without it every request answers 503.

## What the pages show

- **Runs** (`/`): every run with a stage strip (one cell per stage, dependency order, coloured by
  status), plus a banner listing gates waiting on you. Shot runs nest under their parent. The
  upload form loads supported options and defaults from the API: choose a world mode, people
  limit and tracking behavior, moving objects, and fine-tuning. A setup check runs before
  submission and shows actionable local checks. Unverified provider access permits queuing;
  blocked local checks offer **Queue for later** while the worker is being set up.
- **Run** (`/runs/<id>`): stage ladder on the left, grouped by dependency depth; the inspector on
  the right shows the selected stage's attempts, an artifact viewer (video, image, audio, JSON,
  text inline; anything else as a download), and the decision panel. `?stage=<id>` opens a stage
  directly; otherwise the page lands on whatever needs attention first. The progress panel
  reads the run directory to show running stages, why processing stopped, and the next action.
  Source and cleaned clips, review images, and output reports appear as files are produced.

The viewer reads bytes from `GET /api/pipeline/runs/{run}/artifacts/{artifact}` on the orchestrator
(`orchestrator/api/artifacts.py`). Only image, video, audio, JSON and plain-text types render
inline; everything else is served as an octet-stream download.

Run `bun run test` for the form/schema contract checks and `bun run build` to check TypeScript
and the production bundle. Setup checks do not contact paid providers or approve a review.

For a live browser smoke test, start the API factory (`uv run --locked python -m uvicorn
orchestrator.api.service:create_service --factory --host 127.0.0.1 --port 8000`) and this admin
against an isolated database, `WANDER_RUNS_ROOT` and local `WANDER_ARTIFACT_ROOT`. Keep the
supervisor stopped: the test uploads a clip, queues it, previews it, then pauses, resumes and
cancels its own run. It also checks the missing-run error page. From the repository root, with
Chrome installed and root dependencies installed:

```sh
bun admin/scripts/live-smoke.ts /absolute/path/to/source.mp4
```

`ADMIN_SMOKE_URL` overrides the admin URL (default `http://127.0.0.1:5401`), and
`ADMIN_SMOKE_OUTPUT` overrides the ignored evidence directory (default
`.context/evidence/admin-live-smoke`). The script checks schema defaults, the real setup doctor,
source video decoding and persisted operator state; it never starts a worker. The Python
regressions also rebuild the service to verify queued sources and operator state survive restart:
`uv run --locked --group dev pytest scripts/tests/test_admin_live_commands.py -q`.

## Design notes

Neutral review-suite grey, no tint, so footage is the only saturated thing on the page. Colour is
reserved for status and always accompanies a word. IBM Plex Sans for prose and controls, IBM Plex
Mono for identifiers, hashes, sizes and times. Tokens live at the top of `app/globals.css`.
