# 当前工作线

只记录当前 EvoRig 主线。实现目录暂时仍叫 `src/evorig_next`，但它已经是唯一主线。

## 默认入口

- Implementation: [src/evorig_next](d:/Evorig/src/evorig_next)
- Phase1 runner: [scripts/run_evorig_next_phase1.py](d:/Evorig/scripts/run_evorig_next_phase1.py)
- Phase2 runner: [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py)
- Phase3 runner: [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py)

## 默认配置

- Base init: [configs/frozen/evorig_next_base_init_default.yaml](d:/Evorig/configs/frozen/evorig_next_base_init_default.yaml)
- Phase1A: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml)
- Phase1B: [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml)
- Phase1C: [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml)
- Phase2: [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml)
- Phase3: [configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml)

## 当前协议

- Phase1/Phase2/Phase3 默认全程启用 JLG training loss: `loss_illegal_support=0.20`, `loss_gaussian_illegal_coverage=0.0`, `illegal_support_tau=0.0`。
- Phase1 当前主线是 A800 -> B200 -> C100。
- Phase1A 固定 rest joints，`steps=800`，`frame_batch_size=32`，使用 mesh 平均位移初始化 `root_trans`，并冻结 root rest joint。
- Phase1B 从 Phase1A state 继续，`steps=200`，不 densify，只以小学习率 refine 非 root rest joints。
- Phase1C 从 Phase1B state 继续，`steps=100`，rest-joint refine 继续开启，同时打开 `loss_vertex_acceleration=1.0` 和 `loss_temporal_smoothness=0.10`。
- `loss_vertex_acceleration` 是 Phase1C 的默认稳定项；不要在 Phase1A 从头打开。
- Phase1 默认启用 joint-boundary losses: `loss_posed_joint_surface_clearance=0.20`, `loss_rest_joint_inside=0.35`, `loss_rest_joint_surface_clearance=0.20`。
- Phase2 默认 topology edits 是 `seed_joint_repair center_capB`、branch、split。
- Phase2 默认使用 lineage-sibling fast schedule。
- Phase2 branch 默认记录最终 parent-to-path segment inside fraction 作为诊断；不再用该值作为 hard accept/reject gate。
- Phase2 voxel path field 默认磁盘复用；component 连通性复用 Phase1 rest-mesh adjacency 磁盘 cache；动态 coverage/wrong/residual/component 数值仍每次重算。
- Phase3 默认锁定 `rest_joints`；当前 protocol 拒绝 `--unfreeze-rest-joints`，但 base params 解冻，所有 Gaussian offset 从 step 0 开启，使用 `SH16`。
- 新数据必须先通过 alignment / normalization gates，再进入训练。

## 清理边界

- 默认代码只保留 `src/evorig_next`。
- 旧 `src/evorig`、`src/evorig2`、`src/evorig3` 不保留在工作树。
- 如需旧历史，只从 git history 追溯，不重新引入到默认路径。
## 2026-05-10 Final Experiment Defaults

- Phase1 default: A800 -> B200 -> C100; JLG training uses pre-normalized illegal support mass with `loss_illegal_support=0.20`, `loss_gaussian_illegal_coverage=0.0`, `illegal_support_tau=0.0`; joint-boundary losses stay on; `separate_motion_root=true` makes `root_trans` the global translation controller while the anatomical root rest joint remains hard-frozen; Phase1A uses `cross_section_inner_ring` Gaussian scale initialization from the full-mesh inner section contour containing each Gaussian center, with radial axes defined by the farthest contour direction and its perpendicular.
- Phase1A/B/C keep low-support endpoint fallback enabled to avoid static zero-weight vertices during staged resume.
- Phase2 default: scheduled `seed_joint_repair center_capB + branch + split`; JLG support losses stay on. Branch components use combined `wrong OR uncovered` faults, merge nearby fault fragments, keep mixed components, relax the global error gate by `0.03 * (1 - dual_fault_fraction)`, use vertex-count component filtering scaled by average vertex area relative to camel, and samples voxel-route branch paths with `min_path_points=1`, `max_intermediate_points=4`, and pure curvature extrema capped at three. The old lineage/sibling relaxation is removed.
- Phase3 default: rest joints locked, base params unfrozen, Gaussian offsets enabled on all Gaussians from step 0, `SH16`, JLG `0.20/0.0/0.0`.
- JLG-off runs are diagnostic only and must not be used as the final quality line.
- Viewer/eval consistency: Phase1/2/3 run directories preserve the inherited `phase1_config.json`; interactive viewers use that config or the checkpoint payload and fail if neither exists, so missing config cannot silently revert to another support interpretation. Benchmark metrics consume exported weights/signals and do not define a separate Gaussian-range or ownership fallback.
- Scale regularization: `loss_bone_scale_consistency` now caps same-bone scale outliers outside a `[0.5x, 2x]` per-axis mean-scale band. It must not pull all Gaussians on a bone toward the same scale.

## 2026-05-11 Active Phase2 Branch Path

- Branch paths are sampled from the voxel parent-to-tip route only. The old component-root insertion/bridge, medialization, axis alignment, root-tangent smoothing, and seed-leaf parent alignment code paths have been removed from the default implementation.
- Branch endpoints use a mesh-inside high-clearance voxel target near the surface fault tip, and historical branch tip/component overlap suppresses repeated branches in later updates.
- Parent selection is unchanged; after the parent is selected, the route is recomputed with clearance-weighted voxel cost (`path_clearance_weight=2.0`, `path_clearance_power=1.0`) to avoid thin-boundary chords.
- `connected_to_parent=false` means Blender-style dashed hierarchy link: hierarchy is kept, but the edge is not a physical support bone and receives no Gaussians or cross-section/PCJS bone descriptors.
- Phase2 branch now inserts parent -> branch-root as a dashed hierarchy link and branch-root -> tip as connected physical bones. Physical branch chords are subdivided on the same voxel route when they fail the inside-fraction diagnostic; the subdivision cap is `branch_segment_refine_max_points=10`.
- Verified short regressions:
  - double_knife branch-only: old corrupted first branch inside fraction `0.476` -> new path `1.0`.
  - kangaroo branch-only: 2 branches, inside fractions `0.714` and `1.0`.
  - camel branch-only: 2 branches, inside fractions `1.0` and `0.952`.

## 2026-05-12 Active Import Paths

- UniRig dynamic default: `E:\evorig_unirig\<mesh>\dynamic_mesh.glb` -> `prepare_unirig_window_asset.py` -> UniRig `skeleton.fbx` / `skin.fbx` / `rigged.glb` -> `import_unirig_dynamic_glb_sample.py`.
- In this path, `rigged.glb` provides aligned joints and skin weights. `skeleton.fbx` only provides Blender `use_connect` connectivity. The skeleton-only UniRig importer is removed.
- ActionMesh/real-GLB support is separate: use `scripts/import_real_glb_sample.py` only for assets that already provide explicit animated and rigged GLB files. Do not use it for the current UniRig dynamic mesh experiments.
