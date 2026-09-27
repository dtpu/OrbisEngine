# Contributing

Setup is in the [README](README.md) and [DEVELOPING](docs/DEVELOPING.md). You need Bun, and uv for
Python.

Before opening a PR:

```sh
bun run format:check
bun run lint:python
bun run test
```

If you touch walking, XR, audio or the interactive scene, run the matching browser check from
DEVELOPING too.

- One thing per PR.
- No footage, generated scenes, weights, run outputs, screenshots or credentials in Git. Media goes
  in shared storage (`docs/shared-assets.md`).
- No per-clip constants in runtime code; use scene manifests.
- Don't weaken or delete tests to get green.
- Label invented geometry and appearance. Check visual changes in the real viewer.
- New models or libraries get a row in [docs/licences.md](docs/licences.md).

Contributions are licensed under Apache-2.0.
