# Claude Experiment Handoff

这份文档只服务于额外实验接力。目标不是介绍整个仓库，而是让 Claude Code
直接接上当前 EvoRig 主线，避免再走旧路线、旧 checkpoint、旧 smoke 结果。

## 1. 当前唯一有效主线

- 实现源：`src/evorig_next`
- 当前主线名称：`EvoRig`
- 不要把旧 `evorig`、`evorig2`、`evorig3`、旧 accepted-line 配置、旧 sweep
  脚本当作默认线

必须先读：

1. [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md)
2. [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md)
3. [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md)
4. [docs/claude_code_repo_guide_zh.md](d:/Evorig/docs/claude_code_repo_guide_zh.md)
5. [readme.md](d:/Evorig/readme.md)

如果这几份文档和任何旧日志/旧脚本冲突，以这几份为准。

## 2. 环境与执行规则

- `conda activate mygs`
- 推荐 Python：
  `D:\Users\namew\miniconda3\envs\mygs\python.exe`
- 训练前先跑：

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\check_evorig_environment.py
```

必须满足：

- `status == ok`
- `cuda.available == true`
- `open3d.t.geometry.RaycastingScene` 可用

硬规则：

- Windows 上不准并行训练多个任务
- 一次只处理一个 mesh
- 当前 mesh 没完成或没明确 `skip`，不准开下一个
- 不准在没有通过 alignment / normalization gate 的情况下训练

## 3. 新样本固定入口

新样本唯一入口：

- `E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb`

固定流程：

1. bake rest 和选定目标帧
2. 用 rest 做 normalization reference，并把同一 transform 应用到全部帧
3. 检查 topology、vertex order、rest-vs-GT0 RMS/max、bbox/scale、NaN/Inf、mesh quality
4. 在 normalized rest `mesh.glb` 上跑 UniRig
5. 把 UniRig joints / skin 对齐回 normalized dynamic-rest 坐标
6. 用清理后的 UniRig skeleton 构建 EvoRig sample
7. UniRig weights 只用于 fixed-LBS baseline
8. gate 全通过后再进 EvoRig Phase1/2/3

## 4. 当前默认命令

默认配置：

- Base init:
  [configs/frozen/evorig_next_base_init_default.yaml](d:/Evorig/configs/frozen/evorig_next_base_init_default.yaml)
- Phase1A:
  [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml)
- Phase1B:
  [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml)
- Phase1C:
  [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml)
- Phase2:
  [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml)
- Phase3:
  [configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml)

Phase1 当前主线：

- A800 fixed-rest
- B200 rest-refine
- C100 rest-refine + acceleration smooth

默认 Phase1ABC：

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase1abc.py `
  --data-dir mygs/demo_data/<sample_name> `
  --output-dir mygs/results/<asset>_phase1_round1 `
  --name-prefix phase1 `
  --phase1a-config configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml `
  --phase1b-config configs/frozen/evorig_next_phase1b_restrefine_default.yaml `
  --phase1c-config configs/frozen/evorig_next_phase1c_smooth_default.yaml `
  --phase1a-steps 800 `
  --device cuda
```

默认 Phase2：

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase2_round1.py `
  --data-dir mygs/demo_data/<sample_name> `
  --phase1-state mygs/results/<asset>_phase1_round1/phase1c_smooth/phase1_state.pt `
  --output-dir mygs/results/<asset>_phase2_round1 `
  --run-name phase2_default `
  --phase2-config configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml `
  --device cuda
```

默认 Phase3：

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\run_evorig_next_phase3_round1.py `
  --data-dir mygs/demo_data/<sample_name> `
  --resume-phase2-checkpoint mygs/results/<asset>_phase2_round1/phase2_default/phase2_checkpoint.pt `
  --output-root mygs/results/<asset>_phase3_round1 `
  --name phase3_locked_default `
  --phase3-config configs/frozen/evorig_next_phase3_locked_default.yaml `
  --device cuda
```

## 5. 当前默认语义

- Phase1/2/3 默认都开 JLG training loss：
  `loss_illegal_support=0.20`
  `loss_gaussian_illegal_coverage=0.0`
  `illegal_support_tau=0.0`
- `connected_to_parent=false` 表示 Blender 式虚线 hierarchy link
- 虚线 parent link 只保留层级，不是 physical support bone
- 虚线 link 不初始化高斯，不参与 PCJS / cross-section / posed-bone-inside
- Phase2 branch 语义是：
  `parent -> branch_root` 虚线，`branch_root -> tip` 连续 physical bones
- Phase3 默认锁定 `rest_joints`，拒绝 `--unfreeze-rest-joints`

## 6. 当前已知 blocker

这些不是“以后再看”，而是当前额外实验必须先尊重的边界：

1. `double_knife` 旧结果不能当真值  
   旧 run 混用了引入 `connected_to_parent` 之前的 Phase1 checkpoint；这会把本应是
   虚线的 parent link 当作 physical bone，并在其上遗留旧高斯。

2. `double_knife` 必须从头重跑 Phase1  
   不能直接沿旧 `phase1_state.pt` 接 Phase2/3。

3. `double_knife` 的 topology smoke 结果不能当质量结果  
   某些 run 只是为了验证 topology edit 结构，不代表动态已经重新收敛。

4. `double_knife` 仍有 branch / scale 诊断点需要盯  
   当前重点不是全局扫 mesh，而是看重新 Phase1 后：
   - `joint45` 的 branch scale 为什么会过大
   - `joint36` / `joint45` 的末端位置是否合法
   - 这些问题是不是来自 branch 初始化 patch，而不是后续训练

5. `split_fish` 现在不要启动  
   先把 `double_knife` 这条线清干净，再扩到新 mesh。

## 7. Claude 下一步应该继续什么

Claude 只需要继续下面这条实验接力，不要自己发散：

1. 以 `double_knife` 为当前唯一 mesh
2. 重新做一遍完整 Phase1ABC
   - 必须使用当前默认 frozen configs
   - 必须让 `connected_to_parent` 和 outside-mesh Gaussian pruning 生效
3. 重新检查静态和动态 viewer
   - UniRig 自带的非 connected bone 必须画虚线
   - 非 connected bone 上不应残留高斯
4. 再做 Phase2
   - 专盯 `joint45`、`joint36`
   - 看 branch 初始高斯 patch、scale、inside、viewer
5. 只有当 `double_knife` 结构和动态都过关后，才允许切下一个 mesh

## 8. 明确禁止

- 不要开并行训练
- 不要跳过 Phase1 直接从旧 checkpoint 接 Phase2
- 不要把 topology smoke 当作最终质量 run
- 不要在没有看 viewer 的情况下仅凭 loss 判定成功
- 不要把 `split_fish` 当成当前实验对象
- 不要修改训练代码作为这份 handoff 的一部分

## 9. 交接输出要求

Claude 完成额外实验时，至少要留下：

- 当前 mesh 名
- 使用的 frame 选择策略
- Phase1/2/3 结果目录
- viewer 路径
- 是否通过
- 如果失败，失败点是在导入 / Phase1 / Phase2 / Phase3 哪一层

并且每个独立工作块单独提交 commit，再补一条
[docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)。
