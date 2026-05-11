# EvoRig

当前主线是 EvoRig。实现包名暂时保留为 `src/evorig_next`，但它是唯一代码主线。
旧 `evorig`、`evorig2`、`evorig3` 已从默认工作区移除。

## 必读文档

- [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md)
- [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md)
- [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md)
- [docs/evorig_next_strict_runbook_zh.md](d:/Evorig/docs/evorig_next_strict_runbook_zh.md)
- [docs/multi_dataset_protocol_zh.md](d:/Evorig/docs/multi_dataset_protocol_zh.md)

## 默认入口

- Phase1: [scripts/run_evorig_next_phase1.py](d:/Evorig/scripts/run_evorig_next_phase1.py)
- Phase2: [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py)
- Phase3: [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py)

## 默认配置

- Base init: [configs/frozen/evorig_next_base_init_default.yaml](d:/Evorig/configs/frozen/evorig_next_base_init_default.yaml)
- Phase1A: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml)
- Phase1B: [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml)
- Phase1C: [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml)
- Phase2: [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml)
- Phase3: [configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml)
- JLG support loss is on for all phases by default: `loss_illegal_support=0.20`, `loss_gaussian_illegal_coverage=0.0`, `illegal_support_tau=0.0`.

## 环境

- `conda activate mygs`
- 正式实验前先跑：`python scripts/check_evorig_environment.py`
- 优先 `--device cuda`
- Windows 上不要并行跑多个训练任务

## 输出目录

- `mygs/demo_data`
- `mygs/results`
- `mygs/visuals`
- `mygs/outputs`
