# Shared demo assets

Run the viewer locally; scenes and video load from **private S3**. No AWS CLI is needed to view.
The shared migration is complete; the active demo loads published scenes from a pinned snapshot.

## Teammates

1. Clone `main` with `git clone --no-tags --single-branch --branch main https://github.com/StockerMC/wander.git`,
   enter the checkout, and run `bun install --frozen-lockfile` (Bun 1.2.21+).
2. Ask Aayan privately for the teammate environment file. Save it as `.env.local` at the repo root.
   It contains `WANDER_ASSET_ACCESS_KEY_ID` and `WANDER_ASSET_SECRET_ACCESS_KEY`. These credentials
   only read `viewer/*` in the shared bucket; they cannot upload, delete, or read intermediate archives.
3. Run `bun run demo` and open <http://127.0.0.1:5399/demo.html>.
   Enter starts/pauses playback. Use the clip picker for the demo presets.

The first visit downloads that scene's files into `.context/shared-assets/blobs/`; later visits use
this cache. Video seeking and byte ranges work. No complete 8 GB download is required.

Reviewed picker entries can name a private `sceneManifest` asset. Its
`wander.viewer-scene/1` schema contains a `params` object of viewer query defaults; explicit
URL options take precedence. The manifest cannot change the scene ID or select another
manifest. These generated configurations are published with the media, outside Git. The
gym and kitchen selections retain their existing reviewed assets through these configurations.
Restart the local server to pick up a newly published snapshot. A running server pins its snapshot
so a publish cannot mix an old scene manifest with new frames. `/api/shared-assets` shows the active
snapshot, timestamp, and file count without exposing credentials.

Do not give these keys a `VITE_` prefix: credentials belong in the local server, never the browser.
`.env.local` and the cache are gitignored. Keep the server on localhost (the default demo command).
This is a local development viewer, not an authenticated public hosting service.

## Prepared world cache

HTTP caching saves downloads, but Spark normally decodes a world and builds its level-of-detail
tree again for every fresh scene runtime. Prepare that data on the Mac to skip this CPU work on
the Quest. With the local viewer running and Google Chrome installed:

```sh
bun run prepare:worlds --url http://127.0.0.1:5399 /marble-lobby-clean.spz /marble-elevator-clean.spz
```

Pass the world asset paths used by the scenes you want to accelerate. Preparation uses the
installed Spark decoder and preserves its packed arrays, colour encoding, and detail tree exactly;
it does not reduce scene detail. Verified outputs stay under `.context/prepared-worlds/` and are
never uploaded. The source SHA-256 and a build identity covering installed Spark, the preparation client, and the
prepared-world codec identify each cache entry. Updated assets, dependencies, preparation policy,
or codec code cannot silently reuse stale data. Restart Vite and run the command again after code
or dependency changes. `--force` rebuilds an existing entry by atomic replacement. Prepared HTTP
responses revalidate with a SHA-256 ETag of the actual file bytes, so force replacements at the same
URL invalidate browser copies. The server caches this digest by file identity (device, inode, size,
and nanosecond modification/change times); unchanged files need no repeated hashing or download.

Vite serves these optional files through `/api/prepared-world/v1/`. The viewer uses the original
world if a prepared entry is missing, incompatible, or corrupt. Add `xrworldcache=0` to compare
the original loading path. Prepared files are larger than compressed SPZ sources (typically about
three times larger for the current scenes), trading initial transfer size for less headset CPU
work; subsequent visits can reuse the browser's disk cache. This shortcut requires the strong
SHA-256 ETag supplied by the shared-assets server. Servers without that identity use normal loading.
It does not bypass people, object, video, GPU-upload, or sorting work. The separate in-memory scene
cache still keeps only the active scene and one previous scene.

Validate the binary and HTTP behavior with
`bun test ./scripts/test-prepared-world.ts ./scripts/test-prepared-world-server.ts`.
After preparing the lobby, `bun scripts/test-prepared-world-browser.ts` compares its original and
prepared packed data and real rendering, including fallback from a corrupt entry. Set
`PREPARED_WORLD_TEST_URL` when testing another local port. Browser timings are not headset timings.

## Author credentials

Give a teammate access to the provider workspace/project where the work runs, then share any
required project secrets through the team's chosen private channel, such as a private group chat
or password-manager item. Prefer separate credentials per teammate when practical.
For an existing shared automation credential, send only the required values;
do not copy whole `~/.modal.toml`, `~/.aws`, shell profiles, or SSH directories.

| Service             | What this repository reads                                                                                                   | Teammate setup                                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Modal               | `MODAL_PROFILE=dtpu`, or `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`                                                             | Invite the teammate to the workspace owning the model volumes; create their own token there.                                               |
| Marble / World Labs | `WLT_API_KEY`                                                                                                                | Supply a developer API key for the intended billed account; the code sends it as `WLT-Api-Key`.                                            |
| OpenAI              | `OPENAI_API_KEY`                                                                                                             | Add the teammate to the intended API project and use their own project key, or a project service-account credential for shared automation. |
| S3 publishing       | An AWS author profile or standard `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN` for temporary access) | Grant the author identity access to the configured bucket's viewer/archive publishing paths.                                               |
| Viewer only         | `WANDER_ASSET_ACCESS_KEY_ID` + `WANDER_ASSET_SECRET_ACCESS_KEY`                                                              | Keep the existing read-only teammate credentials in `.env.local`; they cannot publish.                                                     |

Modal membership and tokens are workspace-specific. After accepting the
[workspace invite](https://modal.com/docs/guide/workspaces), create a token interactively,
selecting the shared workspace in the browser:

```sh
uv run --locked modal token new --profile dtpu
# Alternative: save an existing token using interactive prompts, without putting it in shell history.
uv run --locked modal token set --profile dtpu
```

For saved-profile authentication, choose one of those setup commands. If the team supplies both
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env.author`, skip token setup: the environment
pair works without a stored profile. Keep `MODAL_PROFILE=dtpu` for the pipeline convention.
`dtpu` is the local profile label used by this pipeline;
it does not itself grant access to a workspace or its cached models. Exported `MODAL_TOKEN_ID`
and `MODAL_TOKEN_SECRET` override the profile's token, so leave them unset when using the saved
profile. See [Modal configuration precedence](https://modal.com/docs/sdk/py/latest/config).
Confirm the teammate can access the existing model volumes before starting GPU work.

Use the [World Labs developer API](https://docs.worldlabs.ai/api) key, and an OpenAI API-project
key stored through environment variables or a secret manager, as described in
[OpenAI's production guidance](https://developers.openai.com/api/docs/guides/production-best-practices).
The AWS publisher uses the [SDK credential chain](https://docs.aws.amazon.com/sdk-for-javascript/v3/developer-guide/setting-credentials-node.html),
not the `WANDER_ASSET_*` pair. Choose either an author profile or standard AWS environment keys;
avoid mixing credential sources. A temporary AWS session must last for the run and its final publish.

Create `.env.author` privately from the author block in `.env.example`, uncomment the settings
needed for your chosen authentication routes, and fill them locally. Keep only viewer credentials
in `.env.local`. Python does not load either dotenv file automatically:

```sh
chmod 600 .env.author
set -a
source .env.author
set +a
uv run --locked --group inference scripts/run_clip.py --help
```

The help command checks the local CLI only; it does not authenticate or launch work. Source this
file in the shell that will start the run. Both private files are gitignored; never use `VITE_`
prefixes for secrets. Agree on the overnight task and spend limit. Interactive runs retain the
cleaned-frame review gate; the explicit unattended policy in [unattended runs](unattended-runs.md)
uses `--no-gate`. Both retain the one-generation-per-new-clip Marble rule. The pipeline has no
single cross-provider hard budget cap; credentials alone do not bound spending. Revoke temporary
shared access afterward. See [unattended runs](unattended-runs.md) before paid work.

## Recover input clips and runs

The private viewer snapshot includes trimmed/transcoded input clips, including elevator, lobby,
stairs2, atrium, HP, and Tears of Steel. This does not guarantee that the full phone originals or
full-length films are archived. Generated cinematic videos are outputs, not reconstruction inputs.

```sh
bun run assets:pull --out .context/inputs --list
bun run assets:pull --out .context/inputs --path /clips/elevator.mp4
# Explicit bulk selection, excluding cinematic outputs unless opted in:
bun run assets:pull --out .context/inputs --prefix /clips/
# Authors can inspect and recover intermediates with their normal AWS credentials:
bun run assets:pull --out .context/recovered --archive --list
bun run assets:pull --out .context/recovered --archive --prefix runs/elevator/
```

Paths are preserved below `--out`, for example `.context/inputs/clips/elevator.mp4`. Downloads
verify SHA-256 and size before becoming visible, and verified local files are reused. Different
local files require an explicit `--overwrite`; symlink destinations are rejected. Nothing is
written into `public/` or uploaded by this command. Outputs must be inside this checkout's `.context`.

The first list/download pins a snapshot in `.wander-pull.json`. Reuse that output directory to
resume consistently, or choose a new one for a newer snapshot. `--snapshot viewer/snapshots/<id>.json`
selects history; use an `archive/snapshots/...` key with `--archive`. Repeat `--path`/`--prefix` for
several selections. Broad selections omit `/clips/cinematic/` unless `--include-cinematic` is set;
an exact cinematic path/prefix also opts in. List output is an inventory, not a provenance verdict.
Viewer downloads use the private `WANDER_ASSET_*` pair when present, otherwise normal AWS access;
archive downloads always require normal author AWS access.

## Authors: publish runs

Publishing uses Node.js 22.18+ on the 22.x line, or Node.js 24+ (see `package.json`).
Keep invoking it with `bun run runs:publish`; that command selects Node for streaming S3 uploads.
Austin's source investigation measured a Bun 1.3.9 transport failure that removed the signed
`Content-Length` on a large streamed body and produced `SignatureDoesNotMatch`; the Node upload
retained the header and passed its SHA-256 check. Native TypeScript execution uses
[Node's type stripping](https://nodejs.org/api/typescript.html#type-stripping).
`bun run test:publish-transport` checks the production streaming upload helper against a local
HTTP fixture with fake credentials, including byte integrity, signed headers, and error propagation.
It does not contact S3 or publish a snapshot.

After the [author credential setup](#author-credentials), use your configured AWS author profile,
separately from the teammate environment file:

```sh
AWS_PROFILE=default bun run runs:publish
# Also preserve an external evidence/screenshots directory:
AWS_PROFILE=default bun run runs:publish --evidence-dir /path/to/share
```

The commands above select the `default` profile; replace it with your author profile. If using
standard AWS environment credentials instead, omit the `AWS_PROFILE=default` prefix and leave
`AWS_PROFILE` unset.

`scripts/run_clip.py` automatically archives `public/` and `.context/run/` at the end, including
failed/gated runs, using `--archive-only`. Public outputs are stored privately under `public/` in
the author archive; automatic completion does not promote them to the viewer. Review the local
result before explicitly running the publish command above to update the viewer.
`WANDER_EVIDENCE_DIR=/path/to/share` adds evidence to that automatic archive.
A publish failure returns an error and leaves local results intact; retry an automatic archive
with `bun run runs:publish --archive-only` (and the same evidence directory when supplied).
`--no-publish` is the explicit offline opt-out. Standalone experimental scripts outside `run_clip.py`
need `bun run runs:publish` after finishing. Hard-killed processes cannot run a completion hook;
resume them or publish their partial outputs manually.

`--archive-only` preserves existing archive paths and conditionally updates only the archive
reference in an existing shared pointer, retaining its viewer snapshot. It creates no viewer
blobs or snapshot. If no shared pointer exists, it saves the immutable archive without creating
a viewer pointer; recover it using the archive key in `.context/shared-storage/last-publish.json`
and `bun run assets:pull --archive --snapshot archive/snapshots/<id>.json --out .context/recovered --list`.
It cannot be combined with `--assets-only`.

To inspect unpublished local work, start Vite with `WANDER_ASSETS_MODE=local bun run demo`.
Remote mode never silently falls back to files on disk. The standard install downloads no local model assets.
If someone else already runs Vite, coordinate with them instead of killing their process.

The publisher hashes files, deduplicates them, verifies upload checksums, and retries interrupted
transfers. Snapshots are immutable. Only after every file succeeds does it conditionally update
`viewer/latest.json`. Concurrent publication or changing input files cause a failure; retry after
other work finishes. Existing snapshots and absent local paths are preserved, not deleted.
Never publish while a stage is still writing its assets. The local upload cache assumes mtime and
size identify unchanged files; remove `.context/shared-storage/upload-cache.json` to force rehashing.

## Bucket layout and recovery

Bucket and region are nonsecret configuration in `wander-storage.json`:

- `exports/`: 13 preserved demo/archive tar bundles and their `MANIFEST.json` (14 objects),
  using date-free object names and manifest references.
- `viewer/blobs/<sha256>`: viewer assets, read-only to teammates.
- `viewer/snapshots/<id>.json`: logical `/clips/...`, `/worlds/...` and other public asset paths.
- `viewer/latest.json`: pointer to the current complete snapshot and its author archive.
- `archive/blobs/<sha256>` and `archive/snapshots/<id>.json`: run intermediates and optional evidence,
  accessible to authors only. Archive manifests map `runs/...` and `evidence/...` to object keys.

S3 versioning is enabled. Previous snapshots remain available; no lifecycle deletion is configured.
Download a snapshot with the author's AWS CLI, then copy a file's `key` to recover it. For example:

```sh
aws s3 cp s3://wander-shared-797639045717/viewer/latest.json latest.json
# Use the snapshot/archive key in latest.json, then the desired file's key in that manifest.
aws s3 cp s3://wander-shared-797639045717/<object-key> ./recovered-file
```

Credential-like files and text containing detected secrets are excluded. The publisher records
exclusions in `.context/shared-storage/excluded.json` and the archive manifest. Stage directory
symlinks within `.context` are followed; unrelated directory symlinks are excluded. Review exclusions
when archiving a new source of outputs. Media and secrets are never committed to git.

The author shares the teammate environment file directly and privately, never through Git or a public
bucket. It belongs to IAM user `wander-viewer-readonly`; revoke or
rotate that user's access key in IAM to invalidate access. A new key requires updating local files.
Previously downloaded copies remain on each teammate's machine.
