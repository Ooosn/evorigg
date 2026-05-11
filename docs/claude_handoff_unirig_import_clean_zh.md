# Claude Handoff: UniRig Dynamic Import

这份文档是给 Claude Code 的最小上下文。不要先读旧实验日志，不要从旧
`evorig/evorig2/evorig3` 路线恢复代码。当前唯一主线是 `src/evorig_next`。

## 任务边界

目标是把艺术资产目录

```text
E:\evorig_unirig\<asset_name>\
```

转换成 EvoRigNext 训练样本。目录必须包含：

```text
dynamic_mesh.glb
rigged.glb
```

当前 importer 不使用 `skeleton.fbx` 来读骨架。骨架、rest mesh、skin weights
全部从 `rigged.glb` 通过 Blender 读取。

## 当前读取逻辑

入口：

```text
scripts/import_unirig_dynamic_glb_sample.py
src/evorig_next/io/unirig_dynamic.py
scripts/blender_bake_dynamic_glb_frames.py
```

`dynamic_mesh.glb`：

- 用 Blender 打开。
- 对每一帧取 evaluated mesh。
- 输出对齐后的动态顶点序列 `gt_anim_vertices.npy`。
- 训练阶段只看到这个 `(T,V,3)` 顶点序列，不直接优化源 GLB 动画。

`rigged.glb`：

- 用 Blender 打开同一个 scene。
- 取 evaluated rest mesh 和 faces。
- 取 mesh vertex groups 作为 skin weights。
- 取 `arm.data.bones` 作为骨架真值来源。
- 每根 Blender bone 读取 `name / parent / use_connect / head_local / tail_local`。

骨架连接规则：

- `bone.use_connect == True`：child bone head 复用 parent bone tail，这是物理连接。
- `bone.use_connect == False` 且有 parent：这是 Blender 虚线 hierarchy link。
- 虚线 link 只保留 FK 层级关系，不初始化 Gaussian，不参与 physical bone support。
- 不用 head-tail 距离阈值猜 connected。
- 不依赖 `bone_数字` 命名。

后处理规则：

- 读取后会执行 joint projection / real-rig cleanup / degenerate bone collapse。
- 这些步骤可以移动或合并 joint，但不能把 Blender 的虚线语义改成实线。
- 如果 viewer 中虚线/实线和 Blender 明显不一致，停止训练，先修 importer。

skin 权重规则：

- skin columns 按 Blender bone name 映射到该 bone 的 head joint。
- `sample_meta.json` 中 `unirig_skin_weights.mapping.missing_skin_joint_count`
  必须为 0；否则不要训练。
- UniRig skin weights 只用于 fixed-LBS baseline，不是 EvoRig 优化目标。

## 必跑命令

环境：

```powershell
conda activate mygs
python scripts/check_evorig_environment.py
```

导入一个样本：

```powershell
python scripts/import_unirig_dynamic_glb_sample.py `
  --asset double_knife `
  --source-dir E:\evorig_unirig\double_knife `
  --output-dir mygs\demo_data\evorig_unirig_windows_round1\double_knife_f31_60 `
  --frame-start 31 `
  --frame-end 60 `
  --alignment-frame-index 31 `
  --max-frames 30 `
  --uniform-fraction 1.0 `
  --blender-path "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

导入后必须检查：

```text
<sample>/sample_meta.json
<sample>/wrong_init_rig.json
<sample>/rest_mesh.obj
<sample>/gt_anim_vertices.npy
<sample>/unirig_skin_weights.npy
```

关键 gate：

- `dynamic_alignment.p95_relative <= 0.03`
- `dynamic_alignment.max_relative <= 0.08`
- `unirig_skin_weights.mapping.missing_skin_joint_count == 0`
- `joint_projection.projected_count` 不能异常大；大规模 projection 通常说明坐标或骨架读取错。
- viewer 里骨架必须贴合 mesh，虚线必须显示为 dashed hierarchy。

## 训练顺序

训练前必须先看导入 viewer。不要批量跑。

Phase1：

```powershell
python scripts/run_evorig_next_phase1.py `
  --data-dir <sample_dir> `
  --output-dir <result_root> `
  --run-name phase1a `
  --phase1-config configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml `
  --steps 800 `
  --device cuda
```

Phase1B 从 Phase1A `phase1_state.pt` resume：

```powershell
python scripts/run_evorig_next_phase1.py `
  --data-dir <sample_dir> `
  --output-dir <result_root> `
  --run-name phase1b `
  --phase1-config configs/frozen/evorig_next_phase1b_restrefine_default.yaml `
  --resume-phase1-state <phase1a_run>/phase1_state.pt `
  --steps 200 `
  --device cuda
```

Phase1C 从 Phase1B resume：

```powershell
python scripts/run_evorig_next_phase1.py `
  --data-dir <sample_dir> `
  --output-dir <result_root> `
  --run-name phase1c `
  --phase1-config configs/frozen/evorig_next_phase1c_smooth_default.yaml `
  --resume-phase1-state <phase1b_run>/phase1_state.pt `
  --steps 100 `
  --device cuda
```

Phase2/Phase3 只在 Phase1 viewer 正常后运行：

```powershell
python scripts/run_evorig_next_phase2_round1.py `
  --data-dir <sample_dir> `
  --phase1-run <phase1c_run> `
  --output-dir <result_root> `
  --run-name phase2 `
  --phase2-config configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml `
  --device cuda
```

```powershell
python scripts/run_evorig_next_phase3_round1.py `
  --data-dir <sample_dir> `
  --phase2-run <phase2_run> `
  --output-dir <result_root> `
  --run-name phase3 `
  --phase3-config configs/frozen/evorig_next_phase3_locked_default.yaml `
  --device cuda
```

生成 viewer：

```powershell
python scripts/view_run_interactive.py --run-dir <run_dir> --kind both
```

## 禁止事项

- 不要并行跑多个训练。
- 不要在 alignment gate 没过时训练。
- 不要用 `skeleton.fbx` 覆盖 `rigged.glb` 的 armature 读取。
- 不要用距离阈值重新猜 `connected_to_parent`。
- 不要把虚线 hierarchy link 当 physical bone 初始化 Gaussian。
- 不要把旧实验配置复制成默认路线。
- 不要在发现 viewer 错误后继续训练。

## 包内文件

最小代码包应包含：

```text
src/evorig_next/
scripts/import_unirig_dynamic_glb_sample.py
scripts/blender_bake_dynamic_glb_frames.py
scripts/check_evorig_environment.py
scripts/run_evorig_next_phase1.py
scripts/run_evorig_next_phase2_round1.py
scripts/run_evorig_next_phase3_round1.py
scripts/view_run_interactive.py
configs/default.yaml
configs/frozen/*.yaml
pyproject.toml
AGENTS.md
readme.md
docs/claude_handoff_unirig_import_clean_zh.md
```

不需要打包 `mygs/results`、`mygs/demo_data`、`paper`、历史 docs、旧 sweep
脚本或临时 debug 目录。
