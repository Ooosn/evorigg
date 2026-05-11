# Exact Commands

## 1. Preflight

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\check_evorig_environment.py
```

## 2. Prepare one UniRig asset window

Authoritative source for a new mesh:

- `E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb`

This section is only for the current UniRig dynamic protocol. For old
ActionMesh/real-GLB assets, use `scripts/import_real_glb_sample.py` instead;
do not mix the two import paths.

Example for `double_knife` current active window:

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\prepare_unirig_window_asset.py `
  --asset-dir E:\evorig_unirig\double_knife `
  --frame-start 31 `
  --frame-end 60 `
  --blender-path "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

This writes:

- `E:\evorig_unirig\double_knife\mesh.glb`
- `E:\evorig_unirig\double_knife\window_prepare_report.json`
- `E:\evorig_unirig\double_knife\run_skeleton.sh`
- `E:\evorig_unirig\double_knife\run_skin.sh`
- `E:\evorig_unirig\double_knife\run_merge.sh`

## 3. UniRig stage

Run the generated scripts in the UniRig environment. Exact execution depends on
your shell, but the generated scripts are the authoritative commands:

- `E:\evorig_unirig\double_knife\run_skeleton.sh`
- `E:\evorig_unirig\double_knife\run_skin.sh`
- `E:\evorig_unirig\double_knife\run_merge.sh`

Required outputs before EvoRig import:

- `E:\evorig_unirig\double_knife\skeleton.fbx`
- `E:\evorig_unirig\double_knife\skin.fbx`
- `E:\evorig_unirig\double_knife\rigged.glb`

## 4. Build EvoRig sample from UniRig rigged.glb plus skeleton.fbx connectivity

This is the authoritative import command for the current connected sample. It uses
`rigged.glb` for aligned joint positions and skin weights, then overlays
`skeleton.fbx` Blender `use_connect` into `connected_to_parent`.

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\import_unirig_dynamic_glb_sample.py `
  --asset double_knife `
  --source-dir E:\evorig_unirig\double_knife `
  --output-dir mygs/demo_data/evorig_unirig_windows_round1/double_knife_f31_60_rigged_connected_v1 `
  --frame-start 31 `
  --frame-end 60 `
  --max-frames 30 `
  --uniform-fraction 1.0 `
  --alignment-frame-index 31 `
  --blender-path "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

## 5. Phase1ABC

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase1abc.py `
  --data-dir mygs/demo_data/evorig_unirig_windows_round1/double_knife_f31_60_rigged_connected_v1 `
  --output-dir mygs/results/double_knife_phase1_rigged_connected_v1_round1 `
  --name-prefix phase1 `
  --phase1a-config configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml `
  --phase1b-config configs/frozen/evorig_next_phase1b_restrefine_default.yaml `
  --phase1c-config configs/frozen/evorig_next_phase1c_smooth_default.yaml `
  --phase1a-steps 800 `
  --device cuda
```

## 6. Phase2

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase2_round1.py `
  --data-dir mygs/demo_data/evorig_unirig_windows_round1/double_knife_f31_60_rigged_connected_v1 `
  --phase1-state mygs/results/double_knife_phase1_rigged_connected_v1_round1/phase1c_smooth/phase1_state.pt `
  --output-dir mygs/results/double_knife_phase2_rigged_connected_v1_round1 `
  --run-name phase2_default `
  --phase2-config configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml `
  --device cuda
```

## 7. Phase3

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase3_round1.py `
  --data-dir mygs/demo_data/evorig_unirig_windows_round1/double_knife_f31_60_rigged_connected_v1 `
  --resume-phase2-checkpoint mygs/results/double_knife_phase2_rigged_connected_v1_round1/phase2_default/phase2_checkpoint.pt `
  --output-root mygs/results/double_knife_phase3_rigged_connected_v1_round1 `
  --name phase3_locked_default `
  --phase3-config configs/frozen/evorig_next_phase3_locked_default.yaml `
  --device cuda
```

## 8. Viewer regeneration

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\view_run_interactive.py `
  --run-dir mygs/results/double_knife_phase2_rigged_connected_v1_round1/phase2_default `
  --kind both
```

## 9. Planned JLG sweep for current user request

Run only after the rigged-connected sample is rebuilt and visually verified:

- `loss_illegal_support = 0.10`
- `loss_illegal_support = 0.20`
- `loss_illegal_support = 0.50`

Do this on `Phase1 A+B` first, not on an old sample and not on `Phase2` first.
