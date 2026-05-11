# EvoRig 当前系统说明

这份文档是当前实现的简明真相。当前主线叫 EvoRig；实现目录暂时仍叫
`src/evorig_next`，但旧 `evorig`、`evorig2`、`evorig3` 已不属于默认路线。

## 默认代码与配置

- Implementation: [src/evorig_next](d:/Evorig/src/evorig_next)
- Base init: [configs/frozen/evorig_next_base_init_default.yaml](d:/Evorig/configs/frozen/evorig_next_base_init_default.yaml)
- Phase1A default: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml)
- Phase1B default: [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml)
- Phase1C default: [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml)
- Phase2 default: [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml)
- Phase3 default: [configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml)

## 默认入口

- Phase1: [scripts/run_evorig_next_phase1.py](d:/Evorig/scripts/run_evorig_next_phase1.py)
- Phase2: [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py)
- Phase3: [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py)

## Phase1 语义

- Phase1 分为 A/B/C 三段：Phase1A 固定全部 rest joints；Phase1B 从 Phase1A state 继续，只允许非 root anatomical rest joints 小幅 refine；Phase1C 从 Phase1B state 继续，rest joint refine 仍保持开启，并打开 vertex acceleration / stronger temporal smoothing。
- Phase1A 默认 `steps=800`，`frame_batch_size=32`（runner 内部自动 clamp 到帧数），`rest_joint_start_step=999999`，`lr_rest_joints=0`。
- Phase1B 默认 `steps=200`，`frame_batch_size=32`，不 densify，`rest_joint_start_step=1`，`lr_rest_joints=0.0006`，`loss_rest_joint_anchor=1.0`。
- Phase1C 默认 `steps=100`，`frame_batch_size=32`，不 densify，`rest_joint_start_step=1`，`lr_rest_joints=0.0006`，`loss_vertex_acceleration=1.0`，`loss_temporal_smoothness=0.10`。
- Phase1 默认从 mesh 顶点平均位移初始化 `root_trans`，并冻结 root rest joint；root motion 可以优化，root rest 不能优化。
- 默认 JLG training loss 全程开启：`loss_illegal_support=0.20`，`loss_gaussian_illegal_coverage=0.0`，`illegal_support_tau=0.0`。
- `loss_illegal_support` 使用归一化前的非法 joint-to-vertex support mass；wrong-coverage 诊断仍使用 invalid ratio。`loss_gaussian_illegal_coverage` 默认关闭，只保留为实验项。
- 默认 joint-boundary losses 也开启：`loss_posed_joint_surface_clearance=0.20`，`loss_rest_joint_inside=0.35`，`loss_rest_joint_surface_clearance=0.20`，对应 clearance ratio 为 `0.06`。
- `loss_rest_joint_anchor` 只在 Phase1B 防 rest joint 漂移，不是 joint correctness loss。
- `loss_vertex_acceleration` 默认只在 Phase1C 开启。它约束预测 mesh 与 GT mesh 的二阶时间差分一致，用来抑制逐帧 pose fitting 在大幅运动上产生的高频抖动；不要在 Phase1A 从头打开。

## Phase2 语义

- Phase2 从 Phase1 state / checkpoint 继续，按 schedule 周期性计算 topology signal。
- Phase2 继承 Phase1 的 joint-boundary loss 配置；topology interval 只额外确保 JLG support losses 不低于默认值。
- 默认 topology edits 是 `seed_joint_repair center_capB`、branch、split。
- branch 由 under/wrong coverage component 触发。
- branch proposal 记录最终 parent-to-path 全段 inside fraction 作为诊断，但默认不再用它作为 hard accept/reject gate；branch 是否产生由 combined wrong/uncovered component、score/error gates、overlap 和 lineage/sibling 逻辑决定。
- split 由 Gaussian bone-local residual profile 提议，并用 two-rigid validation 接受。
- `fault_guided_seed_joint_repair center_capB` 是袋鼠阶段加入的第三类 topology 修复：不新增 branch，而是把已存在但位置错的 seed joint 拉回 fault component 附近。
- 当前实现不硬编码 joint id：候选 joint 必须是已有 one-child seed joint，fault component 的 dominant joints 必须主要落在该 joint 的 parent/self/child 上，并且移动后相邻 segments 的 inside fraction 必须改善且过阈值。
- 默认 fast schedule 保留 branch lineage / sibling guard、split-only signal、voxel path cache、最终 signal reuse。
- Phase2 voxel path field 使用磁盘 cache，路径为 `mygs/outputs/phase2_voxel_path_cache`；cache key 包含 normalized rest vertices、faces 和 voxel 参数。Phase2 component 连通性复用 Phase1 rest-mesh adjacency 磁盘 cache。coverage / wrong ratio / residual / branch component 数值仍随当前 Gaussian field 和 topology 重算，不能磁盘复用。

## Phase3 语义

- Phase3 是 topology-fixed refinement：不 branch，不 split。
- 默认锁定 `rest_joints`，当前 runner 直接拒绝 `--unfreeze-rest-joints`。
- 默认训练 SH16 方向响应，并从 step 0 开启所有 Gaussian 的 rest-space offset；base params 解冻，rest joints 锁定。
- JLG support loss 全程保留：`loss_illegal_support=0.20`，`loss_gaussian_illegal_coverage=0.0`，`illegal_support_tau=0.0`。Phase3 的高容量 SH / offset 不能重新扩大非法覆盖。
- 因为 Phase3 锁定 rest joints，rest-joint-only losses 会被跳过；它不能修复 Phase1/Phase2 已经放错的 rest joint。
- 默认不重新导出 final topology signals；需要诊断时显式 `--save-final-topology-signals`。

## 当前边界

- 当前没有把 `src/evorig_next` 重命名为 `src/evorig`。原因是这会牵涉全部 import、脚本名、结果路径和已有报告引用，应该作为单独迁移 commit 做。
- 文档中“EvoRig 主线”指当前唯一实现主线；`evorig_next` 只是保留下来的包名。
- 旧实验细节只从 git history 或 [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md) 追溯，不重新污染当前默认路径。

## UniRig / Dynamic Mesh 对比协议

新 UniRig 对比样本的唯一源文件是：

- `E:\evorig_unirig\<mesh_name>\dynamic_mesh.glb`

处理顺序固定：

1. 从 `dynamic_mesh.glb` bake rest frame 和选中的 target frames。
2. 用 rest frame 作为 normalization reference，对所有帧应用同一个 transform。
3. 检查 topology、vertex order、rest-vs-GT0 RMS/max、bbox/scale、NaN/Inf、mesh quality。
4. 在 normalized rest `mesh.glb` 上跑 UniRig。
5. 把 UniRig joints / skin output 对齐回 normalized dynamic-rest 坐标。
6. 用清理后的 UniRig skeleton 构建 EvoRig sample。
7. UniRig weights 只用于 fixed-LBS baseline。
8. 通过全部 alignment / normalization gates 后，才运行当前 Phase1/Phase2/Phase3。

训练在 gate 通过前禁止启动。
## 2026-05-10 Final Experiment Defaults

- Phase1: A800 -> B200 -> C100 using `evorig_next_phase1_final500_supportloss_default.yaml`, `evorig_next_phase1b_restrefine_default.yaml`, and `evorig_next_phase1c_smooth_default.yaml`; JLG training loss uses pre-normalized illegal support mass with `0.20/0.0/0.0` (`illegal_support` / `gaussian_illegal_coverage` / `tau`); joint-boundary losses `posed_clearance=0.20`, `rest_inside=0.35`, `rest_clearance=0.20`.
- Phase1 uses a separate virtual motion root: `root_trans` is initialized from mesh centroid motion and remains the global translation controller. The anatomical root rest joint stays hard-frozen whenever `freeze_root_rest_joint=true`, including `separate_motion_root=true`; only `root_trans` may carry global translation.
- Phase1A Gaussian scale initialization uses `phase1_scale_formula=cross_section_inner_ring`: it first cuts the full rest mesh by the Gaussian center plane, selects the smallest closed contour containing the Gaussian center as the inner cross-section ring, then uses the farthest ring direction and its perpendicular as the two radial axes. Each radial scale is the corresponding signed extent divided by `radial_sigma_divisor=3`; the local connected patch is only a fallback when no valid contour exists.
- Phase1A/B/C keep `fallback_low_support_to_bone_endpoint=true`, so numerically zero-support vertices receive nearest bone-endpoint fallback weights instead of being exported as static rest vertices.
- `Phase1Config()` inner defaults are aligned with the current support semantics (`endpoint_cut`, fallback on, kernel cutoff `0.0`, JLG `0.20/0.0/0.0`, and `cross_section_inner_ring`) so direct config construction does not silently return to the old legacy support field.
- Phase2: `evorig_next_phase2_lineage_sibling_fast_default.yaml`; scheduled `seed_joint_repair center_capB + branch + split`, `max_updates=4`, `noop_stop_patience=2`; JLG support loss stays on at `0.20/0.0/0.0`; branch segment inside fraction is diagnostic by default, not a hard acceptance gate. The old lineage/sibling relaxation is removed from the default code path.
- Phase2 branch components are built from the combined `wrong OR uncovered` fault mask, then nearby fault components can be merged across short mesh-adjacency gaps. Mixed wrong+uncovered vertices are retained and relax the global error gate dynamically as `0.03 * (1 - dual_fault_fraction)`. Component noise filtering is vertex-count based, but the reference counts are scaled by average vertex area: `scaled_min_vertices = ceil(base_min_vertices / ((area_current / vertex_count_current) / (area_camel / vertex_count_camel)))`, using camel reference area `3.9673382939` and vertex count `19869`. There is no component-surface-area gate in the default path. Branch path sampling follows the voxel parent-to-tip polyline with `min_path_points=1` and `max_intermediate_points=4`; pure curvature extrema are capped at three.
- Phase3: `evorig_next_phase3_locked_default.yaml`; rest joints locked, base params unfrozen, Gaussian offset enabled for all Gaussians from step 0, `SH16`, JLG support loss `0.20/0.0/0.0`.
- Selection rule: do not select by JLG-off reconstruction loss in any phase. JLG stays on because legality is part of the rigging objective, not a phase-specific diagnostic.
- Default propagation: every Phase run directory must carry the active `phase1_config.json`. Phase1 writes it directly; Phase2/Phase3 inherit it from the resumed Phase1/Phase2/Phase3 checkpoint when available, otherwise from the active Phase1 config object. Interactive viewers read `phase1_config.json`, then `phase1_state.pt`; if neither contains Phase1 config, viewer generation fails instead of silently guessing defaults. Viewers must not fall back to legacy `sigmoid` ownership or `1-sigma` Gaussian ranges.
- Scale regularization: `loss_bone_scale_consistency` is not a same-bone equalization loss. It is a per-bone outlier cap: per-axis Gaussian scales on the same bone can differ within `[0.5x, 2x]` of the per-axis mean without penalty; only larger deviations are penalized.

## 2026-05-11 Phase2 Branch Path Default

- Phase2 branch path is now defined by the voxel parent-to-tip route itself. The default branch control points are sampled from that route; the old component-root insertion, component-root bridge, medialization, axis alignment, post-medial axis alignment, root-tangent smoothing, and seed-leaf parent alignment code paths have been removed from the default implementation.
- Voxel routing keeps the same parent-selection semantics, but after a parent is selected the route is recomputed with a clearance-weighted cost (`branch_path_clearance_weight=2.0`, `branch_path_clearance_power=1.0`). This favors interior voxels without adding off-route postprocessing points.
- Branch path inside fraction remains diagnostic. It is logged in each accepted branch proposal, but the default line does not use the old hard inside-fraction gate to reject topology edits.
- Branch surface tips are used only to locate the fault region. The actual branch endpoint is a nearby mesh-inside voxel center with high clearance, and outside sampled path points are snapped back to mesh-inside centers on the same voxel route. Later updates reject branch candidates that repeat an already accepted branch by component overlap or near-duplicate tip target.
- Skeleton parent links now preserve Blender-style connection semantics through `connected_to_parent`. A parent relation with `connected_to_parent=false` is a dashed hierarchy link, not a physical support bone: it is used for forward kinematics hierarchy but skipped by Gaussian initialization, PCJS/cross-section descriptors, and posed-bone inside losses.
- Phase2 branch insertion creates a disconnected hierarchy link from the selected parent to the branch root, then connected physical bones from branch root to branch tip. The dashed parent-to-root link may leave the mesh and receives no Gaussians; only the connected root-to-tip chain is used for Gaussian support.
- Physical branch segments are refined along the same voxel route when their straight chord cuts through the mesh. This route subdivision is separate from curvature sampling: `branch_max_intermediate_points` controls the compact branch shape, while `branch_segment_refine_max_points=10` is an inside-validity subdivision cap.
- Validation runs for the new default:
  - `mygs/results/branch_path_fix_verify_round1/double_knife_phase2_pathfix_branchonly`: route-only branch path; first branch segment-inside improved from the old corrupted `0.476` path to `1.0`.
  - `mygs/results/branch_path_fix_verify_round1/kangaroo_phase2_pathfix_branchonly`: 2 branches accepted; route-only branch paths; branch inside fractions `0.714` and `1.0`.
  - `mygs/results/branch_path_fix_verify_round1/camel_phase2_pathfix_branchonly`: 2 branches accepted; route-only branch paths; branch inside fractions `1.0` and `0.952`.

## 2026-05-12 Import Path Clarification

- Current UniRig dynamic samples start from `E:\evorig_unirig\<mesh>\dynamic_mesh.glb`. The active importer is `scripts/import_unirig_dynamic_glb_sample.py`.
- For UniRig dynamic samples, `rigged.glb` is the aligned source for rest mesh, joint positions, and skin weights. `skeleton.fbx` is used only to overlay Blender `bone.use_connect` into `connected_to_parent`.
- The old skeleton-only UniRig importer has been removed from the active scripts. Do not rebuild official samples from `raw_data.npz` or skeleton-only joint reads.
- ActionMesh/real-GLB assets remain a separate supported path through `scripts/import_real_glb_sample.py`. That path expects explicit animated and rigged GLB inputs and is not the default for `E:\evorig_unirig` experiments.
- If an imported sample does not preserve disconnected Blender hierarchy links as `connected_to_parent=false`, stop and rebuild the sample before training.
