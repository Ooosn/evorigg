# EvoRig Agent Notes

当前主线是 EvoRig。实现包名暂时仍为 `src/evorig_next`；这是唯一实现源。
不要再把旧 `evorig`、`evorig2`、`evorig3` 或旧 accepted-line 配置当作默认路线。

Before making research or training changes, read:

1. [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md)
2. [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md)
3. [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md)
4. [readme.md](d:/Evorig/readme.md)

## Current Protocol

- Implementation: [src/evorig_next](d:/Evorig/src/evorig_next)
- Base init config: [configs/frozen/evorig_next_base_init_default.yaml](d:/Evorig/configs/frozen/evorig_next_base_init_default.yaml)
- Phase1A default config: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml)
- Phase1B default config: [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml)
- Phase1C default config: [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml)
- Phase2 default config: [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml)
- Phase3 default config: [configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml)
- Phase1 default losses: `loss_illegal_support=0.20`, `loss_gaussian_illegal_coverage=0.05`
- Phase1 current mainline: A800 fixed-rest -> B200 rest-refine -> C100 rest-refine + acceleration smooth
- Phase2 default topology edits: `seed_joint_repair center_capB`, branch, split
- Phase3 default: rest joints locked; `--unfreeze-rest-joints` is rejected

Do not reintroduce legacy implementation paths, old sweep scripts, or old frozen configs into the default path. If old history is needed, recover it from git history before the cleanup commit.

## Environment

- Use `conda activate mygs`.
- Prefer CUDA runs: `--device cuda`.
- Do not change installed `torch` or `cuda` packages unless explicitly approved.
- Do not run multiple training jobs in parallel on Windows.

## Output Locations

- `mygs/demo_data`
- `mygs/outputs`
- `mygs/results`
- `mygs/visuals`

## UniRig / Dynamic Mesh Protocol

The only source of truth for a new UniRig comparison sample is:

- `E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb`

Process exactly one mesh at a time. Finish or explicitly skip the current mesh before starting the next one.

Required order:

1. Bake rest and selected target frames from `dynamic_mesh.glb`.
2. Use the rest frame as the normalization reference and apply the same transform to all frames.
3. Check topology, vertex order, rest-vs-GT0 RMS/max, bbox/scale, NaN/Inf, and mesh quality.
4. Run UniRig on the normalized rest `mesh.glb`.
5. Align UniRig joints and skin output back to the normalized dynamic-rest frame.
6. Build the EvoRig sample from the cleaned UniRig skeleton.
7. Use UniRig weights only for the fixed-LBS baseline.
8. Run EvoRig with the current Phase1/Phase2/Phase3 protocol.

Training is forbidden until alignment and normalization gates pass.

## Working Rules

- Keep [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md) as the concise source of truth.
- Keep [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md) short and active-only.
- Follow [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md) before tuning.
- After each completed work chunk, create a dedicated git commit and append a short entry to [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md).
