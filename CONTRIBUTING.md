# Contributing

## Setup

Viewer: [Bun](https://bun.sh) 1.2.21+ and Chrome. Pipeline and Python checks: [uv](https://docs.astral.sh/uv/).
See [DEVELOPING](docs/DEVELOPING.md) for the code map and [README](README.md) for running a scene.

## Before opening a PR

```sh
bun run format:check
bun run lint:python
bun run test
```

CI runs these plus type checks in `site/` and `admin/`. If you touch walking, XR, audio or the
interactive scene, also run the matching browser check listed in DEVELOPING.

## Rules

- Keep PRs small and about one thing. Say what changed and why.
- Never commit footage, generated scenes, weights, run outputs, screenshots or credentials. Media
  lives in shared storage (`docs/shared-assets.md`).
- No per-clip constants in runtime code; describe scenes with manifests.
- Don't weaken or delete a test to get to green.
- Label invented geometry and appearance as invented. Don't claim a visual result you haven't
  checked in the real viewer.
- A new model or library needs a row in [docs/licences.md](docs/licences.md).

By contributing you agree your contribution is licensed under Apache-2.0 (see `LICENSE`).
