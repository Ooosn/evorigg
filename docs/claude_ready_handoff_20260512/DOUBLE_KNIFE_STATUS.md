# Current Double Knife Status

## Authoritative sample

Current authoritative sample path:

- `mygs/demo_data/evorig_unirig_windows_round1/double_knife_f31_60_rigged_connected_v1`

Reason:

- it is built from `E:\evorig_unirig\double_knife\rigged.glb` for aligned joint positions and skin weights
- it overlays Blender `use_connect` from `E:\evorig_unirig\double_knife\skeleton.fbx` into `connected_to_parent`
- older skeleton-only variants `double_knife_f31_60_connected_v4` and earlier are not authoritative

Current sample facts from `sample_meta.json`:

- `variant = evorig_unirig_dynamic_keyframes`
- `frame_count = 30`
- selected frames = `31..60`
- `vertex_count = 10678`
- `joint_count = 34`
- `dynamic_correspondence.p95_relative = 0.00357089`
- `dynamic_correspondence.max_relative = 0.00357089`
- `joint_projection.projected_count = 2`
- `fbx_connectivity_override.connected_true_count = 21`
- `fbx_connectivity_override.connected_false_count = 13`

## Current blocker

The latest clean `double_knife` training artifacts are still not authoritative,
because they were run on the old sample `double_knife_f31_60_v1`.

Evidence:

- `mygs/results/double_knife_phase1_clean_connected_round3/phase1abc_summary.json`
  - `data_dir = mygs\\demo_data\\evorig_unirig_windows_round1\\double_knife_f31_60_v1`
- `mygs/results/double_knife_phase2_clean_connected_round3/phase2_from_clean3_c_oneupdate/phase2_entry_summary.json`
  - `data_dir = mygs\\demo_data\\evorig_unirig_windows_round1\\double_knife_f31_60_v1`

That means:

- those runs do not prove the corrected `connected_to_parent` pipeline end-to-end
- they must not be used as the final current baseline for `double_knife`

## Current code-level fixes already landed

The repo state already contains these relevant fixes:

- `connected_to_parent` is preserved through rig load/export and viewer drawing
- dashed/non-connected parent links are hierarchy-only, not physical support bones
- branch insertion creates `parent -> branch_root` dashed and `branch_root -> tip` connected
- outside-mesh initialized Gaussians are pruned on new seed creation
- branch-end `cross_section_inner_ring` override is guarded so `lambda > 1` / outside centers do not inherit wide full-mesh rings
- current JLG training uses the active `illegal_support_margin = 0.99`

## Known result-level issues still open

1. The currently quoted clean runs still target the old sample `v1`, not `v4`.
2. `double_knife` must be rerun from scratch on `double_knife_f31_60_rigged_connected_v1` starting at Phase1A.
3. Current accepted branch geometry still needs stricter visual inspection for
   medial placement, not only inside-validity.
4. The next requested experiment is a Phase1 A+B sweep on JLG support weight:
   `0.10 / 0.20 / 0.50`.

## Required next step

Do exactly this next:

1. Rebuild or re-verify `double_knife_f31_60_rigged_connected_v1`
2. Run a fresh Phase1ABC on that sample
3. Inspect viewers
4. Only then run Phase2 on `v4`
5. Only then do the JLG weight sweep on A+B if the base run is structurally valid

Do not move to `split_fish` before this is clean.
