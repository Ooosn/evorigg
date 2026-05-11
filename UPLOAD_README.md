# EvoRig Claude Code Full Upload Package

This package is a runnable repo subset for Claude Code. It includes current source code, active scripts, frozen configs, and handoff docs. It intentionally excludes large data, results, papers, and local temporary debug folders.

Start here:

1. Read `AGENTS.md`.
2. Read `docs/claude_ready_handoff_20260512/README.md`.
3. Run `conda activate mygs`.
4. Run `python scripts/check_evorig_environment.py`.
5. Follow `docs/claude_ready_handoff_20260512/COMMANDS.md`.

Data is not bundled. Put new UniRig samples under `E:\evorig_unirig\<mesh>\dynamic_mesh.glb` and follow the import protocol.

Import paths:

- UniRig dynamic default: `scripts/import_unirig_dynamic_glb_sample.py`.
- ActionMesh/real-GLB separate path: `scripts/import_real_glb_sample.py`.

Official runs must use CUDA. Do not fall back to CPU or missing optional modules.
