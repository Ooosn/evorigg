# EvoRigNext 严格复现日志

这份日志用于把当前 EvoRigNext 主线按同一套环境、同一套配置跑出来。目标是避免别人因为缺依赖、用错 Python、用旧配置或 viewer fallback 得到不同结果。

## 0. 必须先做环境检查

在 `D:\Evorig` 下运行：

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\check_evorig_environment.py
```

要求：

- `status` 必须是 `ok`。
- `python` 必须指向 `D:\Users\namew\miniconda3\envs\mygs\python.exe` 或同一个 `mygs` 环境。
- `cuda.available` 必须是 `true`。正式实验不要用 `--no-cuda-required`。
- `open3d.t.geometry.RaycastingScene`、`trimesh`、`scipy`、`plotly`、`torch` 都必须存在。

如果 preflight 失败，停止实验，修环境。不要改代码绕过依赖，也不要允许 CPU 或缺包 fallback 继续训练。

## 1. 当前唯一主线

实现目录只使用：

```text
src/evorig_next
```

默认配置只使用：

```text
configs/frozen/evorig_next_base_init_default.yaml
configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml
configs/frozen/evorig_next_phase1b_restrefine_default.yaml
configs/frozen/evorig_next_phase1c_smooth_default.yaml
configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml
configs/frozen/evorig_next_phase3_locked_default.yaml
```

不要用 `configs/default.yaml`、`configs/probes/*`、结果目录里临时生成的 config，除非是在复现旧消融并明确写入日志。

## 2. 新 UniRig 样本入口

每个 mesh 的唯一原始入口是：

```text
E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb
```

训练前必须完成：

1. bake rest 和选定 target frames。
2. 用 rest frame 做 normalization reference，对所有帧应用同一个 transform。
3. 检查 topology、vertex order、rest-vs-GT0 RMS/max、bbox/scale、NaN/Inf、mesh quality。
4. 在 normalized rest `mesh.glb` 上跑 UniRig。
5. 把 UniRig joints 和 skin output 对齐回 normalized dynamic-rest 坐标。
6. 用清理后的 UniRig skeleton 构建 EvoRigNext sample。
7. UniRig weights 只用于 fixed-LBS baseline。

alignment / normalization gate 没过，禁止训练。

## 3. Phase1 标准命令

推荐使用 A/B/C 单进程入口，它会共享同一份 mesh cache：

```powershell
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase1abc.py `
  --data-dir <sample_dir> `
  --output-dir <result_dir>\phase1abc `
  --name-prefix phase1 `
  --phase1a-config configs\frozen\evorig_next_phase1_final500_supportloss_default.yaml `
  --phase1b-config configs\frozen\evorig_next_phase1b_restrefine_default.yaml `
  --phase1c-config configs\frozen\evorig_next_phase1c_smooth_default.yaml `
  --phase1a-steps 800 `
  --device cuda
```

当前 Phase1 语义：

- A: fixed rest joints, `steps=800`, `frame_batch_size=32`。
- B: 从 A 继续，rest refine, `steps=200`。
- C: 从 B 继续，继续 rest refine 并启用二阶平滑, `steps=100`。
- JLG support loss 全程开启：`loss_illegal_support=0.20`，`loss_gaussian_illegal_coverage=0.0`，`illegal_support_tau=0.0`。
- `loss_bone_vertex_recon_topk` 已删除，不再是 Phase1 目标。
- `bone_scale_length_cap_loss` 已删除，不再直接通过拉长骨段控制高斯尺度。

生成 viewer：

```powershell
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\view_run_interactive.py `
  --run-dir <phase1c_run_dir> `
  --kind both
```

viewer 必须能从 run dir 读取 `phase1_config.json` 或 checkpoint 内的 `phase1_config`。如果缺失应该失败，不允许猜旧默认值。

## 4. Phase2 标准命令

```powershell
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase2_round1.py `
  --data-dir <sample_dir> `
  --phase1-state <phase1c_run_dir>\phase1_state.pt `
  --output-dir <result_dir>\phase2 `
  --run-name phase2_default `
  --phase2-config configs\frozen\evorig_next_phase2_lineage_sibling_fast_default.yaml `
  --device cuda
```

当前 Phase2 默认：

- topology edits: `seed_joint_repair center_capB + branch + split`。
- branch 来自 combined `wrong OR uncovered` fault component。
- branch path 使用 voxel parent-to-tip route 加 route-internal curvature/long-segment sampling。
- segment inside fraction 是诊断，不是默认 hard gate。
- 每次 topology edit 后必须看 viewer 和日志，不要批量跳过结构检查。

## 5. Phase3 标准命令

```powershell
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase3_round1.py `
  --data-dir <sample_dir> `
  --phase2-state <phase2_run_dir>\phase2_state.pt `
  --output-dir <result_dir>\phase3 `
  --run-name phase3_locked_default `
  --phase3-config configs\frozen\evorig_next_phase3_locked_default.yaml `
  --device cuda
```

Phase3 默认锁定 rest joints。不要传 `--unfreeze-rest-joints`。

## 6. Phase1B 本次复验

本次用当前代码从 double_knife 的 A state 重新跑 Phase1B：

```text
run_dir:
D:\Evorig\mygs\results\joint5_loss_debug_round1\default_after_remove_topk_200

resume:
E:\evorig_unirig\double_knife\evorig_result\phase1abc\doubleknifea_800_fixedrest\phase1_state.pt
```

结果：

```text
final_error_raw      = 0.014664668589830399
final_error_raw_all  = 0.014952857978641987
zero_weight_row_count = 0
final_outside_active_gaussian_count = 2
```

对比旧 200-step B：

```text
old_with_topk_200: mean drift 0.0235678, max 0.1076411, joint5 0.1076411
new_no_topk_200:  mean drift 0.0153404, max 0.0610279, joint5 0.0174781
```

viewer：

```text
D:\Evorig\mygs\results\joint5_loss_debug_round1\default_after_remove_topk_200\visuals\interactive\interactive_motion.html
D:\Evorig\mygs\results\joint5_loss_debug_round1\default_after_remove_topk_200\visuals\interactive\interactive_final_topology.html
```

结论：当前 Phase1B 可以继续作为默认线的一部分；删除 `loss_bone_vertex_recon_topk` 后，double_knife 的 joint5 drift 明显下降。

## 7. 失败处理

如果出现下面任一情况，停止当前 mesh，不要继续 Phase2/3：

- preflight 失败。
- alignment / normalization gate 失败。
- Phase1A viewer 显示初始 skeleton 坐标系明显不对。
- Phase1B/C 后大面积 joint 飞出 mesh，且不是明确的输入 rig 问题。
- Phase2 第一次 topology edit 后骨架结构明显错误。

记录失败原因和路径后再换下一个 mesh。不要把失败结果标记为可用。
