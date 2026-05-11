# Claude Ready Handoff Package

This folder is the handoff package to give directly to Claude Code for the
current EvoRig mainline.

Use this package as the single packaging/docs entrypoint. It is intentionally
strict and assumes:

- Windows host
- `conda` environment `mygs`
- official runs on CUDA only
- one mesh at a time

Read in this order:

1. [ACTIVE_WORKLINE.md](./ACTIVE_WORKLINE.md)
2. [IMPORT_PATHS.md](./IMPORT_PATHS.md)
3. [ENVIRONMENT_AND_NO_FALLBACK.md](./ENVIRONMENT_AND_NO_FALLBACK.md)
4. [COMMANDS.md](./COMMANDS.md)
5. [DOUBLE_KNIFE_STATUS.md](./DOUBLE_KNIFE_STATUS.md)
6. [SERVER_UPLOAD.md](./SERVER_UPLOAD.md)
7. [PACKAGE_MANIFEST.json](./PACKAGE_MANIFEST.json)

Bundled reference snapshots are under [references](./references/). Those are
copies of the current repo docs at package time so the zip is self-contained.

Current package purpose:

- lock the active EvoRig workline and default configs
- distinguish UniRig dynamic imports from ActionMesh/real-GLB imports
- give exact environment and command lines
- forbid silent fallback paths
- document the current `double_knife` sample/run status and blockers
- provide a safe handoff for extra experiments without touching training code
