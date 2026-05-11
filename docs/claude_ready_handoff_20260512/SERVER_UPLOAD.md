# Server Upload Use

This zip is a Claude Code handoff package, not a full repository snapshot.

Expected server state:

- the current EvoRig repo is already checked out
- the repo contains `src/evorig_next`, `scripts`, `configs/frozen`, and `docs`
- the `mygs` conda environment exists
- CUDA and Blender are available as described in
  `ENVIRONMENT_AND_NO_FALLBACK.md`

How to use:

1. Upload and extract this folder inside the repo, for example:
   - `docs/claude_ready_handoff_20260512`
2. Start Claude Code from the repo root.
3. Read `README.md` first.
4. Run the preflight command in `COMMANDS.md`.
5. Follow the one-mesh-at-a-time command sequence.

If the server does not already have the repo, upload the repo separately. This
handoff package intentionally does not include large result folders, mesh data,
or source-code copies.
