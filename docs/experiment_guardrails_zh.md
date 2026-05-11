# 实验 Guardrails

这份文档约束默认实验，避免临时调参污染当前主线。

## 训练前必须检查

- 当前代码目录只能依赖 `src/evorig_next`。
- 新样本必须先完成 normalization / alignment gates。
- 新样本先跑 Phase1A fixed-rest 并检查 trace / viewer；通过后才能跑 Phase1B，再跑 Phase1C；不能直接进 Phase2/3。
- Windows 上不要并行跑多个训练任务。
- CUDA 优先：`--device cuda`。

## 默认线不可随意修改

- Phase1A 默认配置：[configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml)
- Phase1B 默认配置：[configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml)
- Phase1C 默认配置：[configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml)
- Phase1 默认包含 `separate_motion_root=true` 和 Phase1A `cross_section_inner_ring` scale 初始化；改动这两项必须先给 root 验证、scale viewer 和 A/B/C 对比。
- Phase2 默认配置：[configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml)
- Phase3 默认配置：[configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml)
- JLG support loss 是默认目标的一部分，Phase1/Phase2/Phase3 全程开启：`loss_illegal_support=0.20`，`loss_gaussian_illegal_coverage=0.0`，`illegal_support_tau=0.0`。JLG-off 只能作为诊断或消融，不能作为默认质量线。

如果要改默认配置，必须先有：

- 明确失败现象。
- 可复现实验目录。
- 对比指标或 viewer。
- 结论写入 [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)。

## 消融和临时实验

- 消融脚本可以保留，但不能把临时 run 当默认线。
- 结果目录命名必须包含 asset、phase、round、variant。
- 不要只看 final reconstruction loss；同时看 topology events、JLG faults、zero coverage、inside fraction、viewer。
- Phase2 branch 结果仍必须检查最终新增骨段 inside fraction，但该值默认只作为诊断记录，不作为 hard accept/reject gate。
- Phase2 component 调参必须保持默认语义：branch seed 来自 combined `wrong OR uncovered` fault mask，mixed wrong+uncovered component 不能被 purity gate 丢掉；默认仍按 component vertex count 做噪声过滤，但顶点阈值必须按 average vertex area 相对骆驼缩放。
- 任何跳过 viewer / final topology signals 的加速，都要说明它只是训练加速，不改变模型质量。

## 回退规则

- 不使用 `git reset --hard` 或 `git checkout --` 回退用户改动。
- 需要恢复旧逻辑时，从 git history 拿具体文件或片段，不把旧整套路线重新引入默认路径。
