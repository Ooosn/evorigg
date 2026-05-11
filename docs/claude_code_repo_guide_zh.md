# Claude Code Repo Guide

这份文档只写给另一个 coding agent 使用。目标是让它在当前 EvoRig 主线上按正确协议工作，而不是自己猜默认配置、默认脚本或旧路线。

## 1. 先读这些文档

开始任何研究、导入、训练、调参之前，必须先读：

1. [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md)
2. [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md)
3. [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md)
4. [readme.md](d:/Evorig/readme.md)

不要把旧 `evorig`、`evorig2`、`evorig3`、旧 accepted-line 配置、旧 sweep 脚本当成默认线。
当前唯一实现源是 [src/evorig_next](d:/Evorig/src/evorig_next)。

## 2. 环境

- Conda 环境：`mygs`
- 推荐 Python：`D:\Users\namew\miniconda3\envs\mygs\python.exe`
- 正式训练前先跑：

```powershell
conda activate mygs
D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\check_evorig_environment.py
```

必须满足：

- `status == ok`
- `cuda.available == true`
- `open3d.t.geometry.RaycastingScene` 可用
- 不要在 Windows 上并行跑多个训练任务
- 不要改已安装的 `torch` / `cuda` 包，除非用户明确批准

## 3. 一次只处理一个 mesh

新的 UniRig/EvoRig 对比样本唯一入口是：

`E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb`

规则只有一条：一次只做一个 mesh。当前 mesh 没完成或没明确标记 skip，不准开始下一个。

## 4. UniRig / Dynamic Mesh 固定流程

训练前必须按这个顺序走完：

1. 从 `dynamic_mesh.glb` bake rest frame 和选中的目标帧。
2. 用 rest frame 做 normalization reference，并把同一个 transform 应用到所有帧。
3. 检查 topology、vertex order、rest-vs-GT0 RMS/max、bbox/scale、NaN/Inf、mesh quality。
4. 在 normalized rest `mesh.glb` 上运行 UniRig。
5. 把 UniRig joints 和 skin 输出对齐回 normalized dynamic-rest 坐标。
6. 用清理后的 UniRig skeleton 构建 EvoRig sample。
7. UniRig weights 只用于 fixed-LBS baseline，不作为 EvoRig 训练目标。
8. 只有 alignment / normalization gates 全通过后，才允许进入 EvoRig Phase1/2/3。

如果第 3 步或第 5 步不通过，停止。不要带着坏对齐去训练。

## 5. 当前默认配置

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

当前默认训练语义：

- Phase1 主线：A800 fixed-rest -> B200 rest-refine -> C100 rest-refine + acceleration smooth
- Phase1/2/3 默认都开 JLG training loss：
  `loss_illegal_support=0.20`
  `loss_gaussian_illegal_coverage=0.0`
  `illegal_support_tau=0.0`
- Phase2 默认 topology edits：
  `seed_joint_repair center_capB`, `branch`, `split`
- Phase3 默认锁定 `rest_joints`，拒绝 `--unfreeze-rest-joints`

## 6. 默认命令

### 6.1 Phase1

默认用单进程 A/B/C 入口，不要自己拆成三次独立进程，除非是在做明确诊断。

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

### 6.2 Phase2

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

### 6.3 Phase3

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

## 7. 输出目录

只往这些目录写：

- `mygs/demo_data`
- `mygs/outputs`
- `mygs/results`
- `mygs/visuals`

推荐命名：

- `mygs/demo_data/<asset>_<frame_policy>_v<N>`
- `mygs/results/<asset>_phase1_round<N>/<variant>`
- `mygs/results/<asset>_phase2_round<N>/<variant>`
- `mygs/results/<asset>_phase3_round<N>/<variant>`

## 8. 关键 failure gates

出现下面任一情况，停止当前 mesh，先修问题，不准直接推进到后续阶段：

- `check_evorig_environment.py` 失败
- alignment / normalization gate 失败
- rest-vs-GT0 RMS/max 明显异常
- vertex order / topology 不一致
- 发现 NaN / Inf
- Phase1A viewer 里初始 skeleton 明显不对齐
- Phase1B / C 后大量 joints 跑出 mesh，且不是输入 rig 本身就错
- Phase2 第一次 topology edit 后 branch / split 结构明显错误

额外原则：

- 不要只看 final reconstruction loss
- 每个阶段都看 viewer、trace、topology events、JLG faults
- Phase2 每次 topology edit 后都要检查结构；不要批量跑完整个 schedule 再回头看

## 9. 明确禁止

- 不要在 Windows 上并行训练
- 不要跳过 alignment / normalization gate
- 不要直接用旧路线、旧 frozen config、旧 probe config 当默认线
- 不要在没有记录原因的情况下改默认配置
- 不要把 UniRig weights 当成 EvoRig 训练 supervision
- 不要用 silent fallback 去猜 viewer/config 默认值

## 10. 对 Claude Code 的工作要求

如果你是外部 coding agent：

- 先读第 1 节文档，再动代码或跑实验
- 先确认当前 mesh 的导入和对齐通过，再训练
- 训练时只处理一个 mesh
- 只使用当前 frozen configs 和当前 runners
- 每完成一个独立工作块，单独提交 commit，并补一条 [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)
