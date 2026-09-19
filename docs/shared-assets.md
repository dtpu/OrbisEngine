# Shared demo assets

Run the viewer locally; scenes and video load from **private S3**. No AWS CLI is needed to view.
The shared migration is complete; the active demo loads published scenes from a pinned snapshot.

## Teammates

1. Clone `main` with `git clone --no-tags --single-branch --branch main https://github.com/StockerMC/wander.git`,
   enter the checkout, and run `npm ci` (Node 22.12+ or Node 24).
2. Ask Aayan privately for the teammate environment file. Save it as `.env.local` at the repo root.
   It contains `WANDER_ASSET_ACCESS_KEY_ID` and `WANDER_ASSET_SECRET_ACCESS_KEY`. These credentials
   only read `viewer/*` in the shared bucket; they cannot upload, delete, or read intermediate archives.
3. Run `npm run demo` and open <http://127.0.0.1:5399/demo.html>.
   Enter starts/pauses playback. Use the clip picker for the demo presets.

The first visit downloads that scene's files into `.context/shared-assets/blobs/`; later visits use
this cache. Video seeking and byte ranges work. No complete 8 GB download is required.
Restart the local server to pick up a newly published snapshot. A running server pins its snapshot
so a publish cannot mix an old scene manifest with new frames. `/api/shared-assets` shows the active
snapshot, timestamp, and file count without exposing credentials.

Do not give these keys a `VITE_` prefix: credentials belong in the Node server, never the browser.
`.env.local` and the cache are gitignored. Keep the server on localhost (the default demo command).
This is a local development viewer, not an authenticated public hosting service.

## Authors: publish runs

Use your normal AWS author profile, separately from the teammate environment file:

```sh
AWS_PROFILE=default npm run runs:publish
# Also preserve an external evidence/screenshots directory:
AWS_PROFILE=default npm run runs:publish -- --evidence-dir /path/to/share
```

`scripts/run_clip.py` automatically publishes `public/` and `.context/run/` at the end, including
failed/gated runs. `WANDER_EVIDENCE_DIR=/path/to/share` adds evidence to that automatic publish.
A publish failure returns an error and leaves local results intact; retry the command above.
`--no-publish` is the explicit offline opt-out. Standalone experimental scripts outside `run_clip.py`
need `npm run runs:publish` after finishing. Hard-killed processes cannot run a completion hook;
resume them or publish their partial outputs manually.

To inspect unpublished local work, start Vite with `WANDER_ASSETS_MODE=local npm run demo`.
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
