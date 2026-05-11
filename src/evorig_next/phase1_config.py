from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Phase1DensifyStage:
    warm_steps: int
    settle_steps: int
    max_bones: int
    seeds_per_bone: int = 1


@dataclass
class Phase1Config:
    steps: int = 800
    frame_batch_size: int = 32
    seed_count_scale: float = 5.0
    initial_lambda_policy: str = "freeze_forever"
    lambda_min: float = -0.2
    lambda_max: float = 1.2
    phase1_force_global_lambda_bounds: bool = True
    ownership_mode: str = "endpoint_cut"
    ownership_midpoint: float = 0.8
    ownership_slope: float = 0.08
    child_support_gate_start: float = 0.75
    child_support_gate_end: float = 0.95
    gaussian_kernel_mahal_cutoff_sq: float = 0.0
    legality_surface_tol: float = 3.0e-3
    legality_origin_epsilon: float = 1.0e-6
    legality_vertex_chunk_size: int = 20_000
    legality_propagation_mode: str = "off"
    legality_propagation_radius: int = 1
    legality_propagation_rounds: int = 1
    legality_propagation_target: str = "all"
    legality_propagation_sparse_joint_max_count: int = 0
    legality_propagation_majority_threshold: float = 0.5
    support_mass_eps: float = 1.0e-8
    fallback_low_support_to_bone_endpoint: bool = True
    fallback_support_mass_threshold: float = 1.0e-8
    illegal_support_tau: float = 0.0
    illegal_support_margin: float = 0.99
    no_joint_rest_displacement_tol: float = 1.0e-6
    hard_legal_support_mask: bool = False
    loss_vertex_recon: float = 1.0
    loss_vertex_acceleration: float = 0.0
    loss_temporal_smoothness: float = 0.05
    loss_illegal_support: float = 0.20
    loss_pcjs: float = 0.10
    loss_posed_joint_inside: float = 0.05
    loss_posed_bone_inside_mesh: float = 0.05
    loss_rest_joint_inside: float = 0.35
    loss_posed_joint_surface_clearance: float = 0.20
    loss_rest_joint_surface_clearance: float = 0.20
    loss_scale_anchor: float = 0.08
    loss_bone_scale_consistency: float = 0.08
    loss_gaussian_illegal_coverage: float = 0.0
    loss_bone_cov_offdiag: float = 0.08
    loss_bone_radial_symmetry: float = 0.04
    loss_bone_scale_band: float = 0.08
    loss_mesh_edge_length_floor: float = 4.0
    mesh_edge_length_floor_ratio: float = 0.93
    loss_bone_radial_distance_shrink: float = 0.12
    bone_radial_distance_shrink_ratio: float = 0.95
    posed_joint_inside_root_weight: float = 0.25
    posed_bone_inside_samples: int = 4
    pcjs_surface_tol: float = 3.0e-3
    pcjs_direction_count: int = 12
    pcjs_section_lambda: float = 0.1
    posed_joint_surface_clearance_ratio: float = 0.06
    posed_joint_surface_clearance_margin: float = 0.0
    rest_joint_surface_clearance_ratio: float = 0.06
    rest_joint_surface_clearance_margin: float = 0.0
    initial_seed_prune_outside_mesh: bool = True
    densify_seed_prune_outside_mesh: bool = True
    final_gaussian_prune_outside_mesh: bool = False
    seed_inside_surface_tol: float = 3.0e-3
    project_rest_joints_inside_after_step: bool = True
    rest_joint_projection_padding: float = 3.0e-3
    bone_scale_band_max_axial_log_span: float = 0.55
    bone_scale_band_max_radial_log_span: float = 0.35
    phase1_decouple_axial_from_radial_init: bool = False
    phase1_scale_init_divisor: float = 1.0
    phase1_scale_formula: str = "cross_section_inner_ring"
    phase1_radial_sigma_divisor: float = 3.0
    phase1_radial_extent_quantile: float = 1.0
    phase1_radial_patch_connected_component: bool = True
    phase1_cross_section_angle_bins: int = 64
    phase1_cross_section_min_bins: int = 10
    phase1_axial_three_center_divisor: float = 3.0
    phase1_axial_min_radial_ratio: float = 0.5
    phase1_formula_log_scale_min: float = -8.0
    phase1_formula_log_scale_max: float = 0.5
    init_log_opacity: float = -2.0
    lr_pose: float = 0.005
    lr_root: float = 0.001
    lr_field: float = 0.004
    lr_lambda_initial: float = 0.0
    lr_lambda_initial_thawed: float = 0.004
    lr_lambda_densified: float = 0.004
    lr_rot: float = 0.004
    lr_scale: float = 0.004
    lr_opacity: float = 0.004
    lr_value: float = 0.004
    lr_offset: float = 0.0
    lr_rest_joints: float = 0.0
    lr_sh: float = 0.004
    root_trans_init_mode: str = "centroid"
    freeze_root_rest_joint: bool = True
    frozen_rest_joint_ids: tuple[int, ...] = ()
    separate_motion_root: bool = True
    trace_interval_steps: int = 50
    grad_ema_decay: float = 0.9
    gaussian_offset_start_step: int = -1
    gaussian_offset_target: str = "all"
    parent_child_mix_start_step: int = -1
    lambda_thaw_start_step: int = 301
    lambda_thaw_target: str = "densified"
    rest_joint_start_step: int = 999999
    loss_rest_joint_anchor: float = 0.0
    loss_gaussian_offset_anchor: float = 0.0
    loss_gaussian_sh_reg: float = 0.001
    sh_start_step: int = 301
    sh_coeff_count: int = 4
    watch_joint_ids: tuple[int, ...] = (12, 13, 15, 16)
    densify_stages: list[Phase1DensifyStage] = field(
        default_factory=lambda: [
            Phase1DensifyStage(warm_steps=60, settle_steps=40, max_bones=12, seeds_per_bone=4),
            Phase1DensifyStage(warm_steps=60, settle_steps=40, max_bones=12, seeds_per_bone=4),
            Phase1DensifyStage(warm_steps=60, settle_steps=40, max_bones=12, seeds_per_bone=4),
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["densify_stages"] = [asdict(item) for item in self.densify_stages]
        return data
