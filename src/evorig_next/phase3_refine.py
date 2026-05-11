from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from evorig_next.io.outputs import save_outputs
from evorig_next.phase1_trainer import (
    audit_dominant_connectivity,
    dominant_joint_assignment,
)
from evorig_next.phase1_losses import vertex_recon_loss
from evorig_next.phase2_topology import (
    Phase2TopologyConfig,
    _mesh_segment_inside_fraction,
    save_phase2_topology_signals,
)

EPS = 1.0e-8


@dataclass
class Phase3RefineConfig:
    steps: int = 200
    sh_coeff_count: int = 16
    lr_sh: float = 0.002
    loss_gaussian_sh_reg: float = 5.0e-4
    freeze_base_params: bool = False
    enable_gaussian_offset: bool = True
    offset_start_step: int = 0
    gaussian_offset_target: str = "all"
    lr_offset: float = 5.0e-4
    loss_gaussian_offset_anchor: float = 0.05
    freeze_rest_joints: bool = True
    lr_rest_joints: float | None = None
    loss_pcjs: float | None = None
    loss_rest_joint_anchor: float | None = None
    loss_rest_joint_inside: float | None = None
    loss_posed_joint_surface_clearance: float | None = None
    posed_joint_surface_clearance_ratio: float | None = None
    posed_joint_surface_clearance_margin: float | None = None
    loss_rest_joint_surface_clearance: float | None = None
    rest_joint_surface_clearance_ratio: float | None = None
    rest_joint_surface_clearance_margin: float | None = None
    loss_illegal_support: float | None = 0.20
    loss_gaussian_illegal_coverage: float | None = 0.0
    illegal_support_tau: float | None = 0.0
    skip_frozen_only_losses: bool = True
    save_final_topology_signals: bool = False
    trace_interval: int = 25
    audit_interval: int = 50
    joint_displacement_warn_ratio: float = 0.05
    bone_inside_warn_fraction: float = 0.70
    bone_direction_warn_cos: float = 0.50


def _evaluate_metrics(trainer: Any) -> tuple[Any, dict[str, Any]]:
    cache = trainer.evaluate_full()
    recon_mask = trainer.legal_vertex_mask.unsqueeze(0).expand(int(trainer.gt_vertices.shape[0]), -1).to(
        dtype=cache.pred_vertices.dtype,
        device=cache.pred_vertices.device,
    )
    return cache, {
        "final_error_raw": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices, mask=recon_mask).item()),
        "final_error_raw_all": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices).item()),
        "zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
    }


def audit_phase3_state(
    trainer: Any,
    *,
    start_rest_joints: torch.Tensor,
    start_bone_parent_idx: torch.Tensor,
    start_bone_child_idx: torch.Tensor,
    cfg: Phase3RefineConfig,
) -> dict[str, Any]:
    current = trainer.skeleton.rest_joints.detach()
    start = start_rest_joints.to(device=current.device, dtype=current.dtype)
    common_joint_count = min(int(current.shape[0]), int(start.shape[0]))
    if common_joint_count > 0:
        displacement = torch.linalg.norm(current[:common_joint_count] - start[:common_joint_count], dim=-1)
    else:
        displacement = torch.zeros(0, dtype=current.dtype, device=current.device)
    radius = max(float(trainer.sample_radius), EPS)
    disp_norm = displacement / radius
    topk_count = min(12, int(disp_norm.numel()))
    top_displaced: list[dict[str, Any]] = []
    if topk_count > 0:
        values, ids = torch.topk(disp_norm, k=topk_count, largest=True)
        birth_modes = list(getattr(trainer.skeleton, "birth_modes", []))
        inserted = list(getattr(trainer.skeleton, "is_inserted", []))
        for value, joint_id in zip(values.detach().cpu().tolist(), ids.detach().cpu().tolist()):
            jid = int(joint_id)
            top_displaced.append(
                {
                    "joint": jid,
                    "displacement": float(displacement[jid].item()),
                    "displacement_ratio": float(value),
                    "birth_mode": str(birth_modes[jid]) if jid < len(birth_modes) else "unknown",
                    "inserted": bool(inserted[jid]) if jid < len(inserted) else False,
                    "warn": bool(float(value) > float(cfg.joint_displacement_warn_ratio)),
                }
            )

    current_parent = trainer.skeleton.bone_parent_idx.detach()
    current_child = trainer.skeleton.bone_child_idx.detach()
    start_parent = start_bone_parent_idx.to(device=current_parent.device, dtype=torch.long)
    start_child = start_bone_child_idx.to(device=current_child.device, dtype=torch.long)
    common_bone_count = min(int(current_parent.numel()), int(start_parent.numel()))
    bone_rows: list[dict[str, Any]] = []
    flagged_bones: list[dict[str, Any]] = []
    for bone_index in range(common_bone_count):
        p = int(current_parent[bone_index].item())
        c = int(current_child[bone_index].item())
        sp = int(start_parent[bone_index].item())
        sc = int(start_child[bone_index].item())
        if p >= int(current.shape[0]) or c >= int(current.shape[0]) or sp >= int(start.shape[0]) or sc >= int(start.shape[0]):
            continue
        cur_vec = current[c] - current[p]
        start_vec = start[sc] - start[sp]
        cur_len = float(cur_vec.norm().item())
        start_len = float(start_vec.norm().item())
        denom = max(cur_len * start_len, EPS)
        direction_cos = float(torch.abs((cur_vec * start_vec).sum() / denom).item())
        inside_fraction = _mesh_segment_inside_fraction(trainer, current[p], current[c], sample_count=21)
        warn = (
            inside_fraction < float(cfg.bone_inside_warn_fraction)
            or direction_cos < float(cfg.bone_direction_warn_cos)
        )
        row = {
            "bone_index": int(bone_index),
            "parent_joint": int(p),
            "child_joint": int(c),
            "length": float(cur_len),
            "start_length": float(start_len),
            "length_ratio": float(cur_len / max(start_len, EPS)),
            "direction_abs_cos_to_phase2_start": float(direction_cos),
            "rest_segment_inside_fraction": float(inside_fraction),
            "warn": bool(warn),
        }
        bone_rows.append(row)
        if warn:
            flagged_bones.append(row)

    return {
        "joint_count": int(trainer.skeleton.joint_count),
        "bone_count": int(trainer.skeleton.bone_count),
        "joint_displacement_warn_ratio": float(cfg.joint_displacement_warn_ratio),
        "bone_inside_warn_fraction": float(cfg.bone_inside_warn_fraction),
        "bone_direction_warn_cos": float(cfg.bone_direction_warn_cos),
        "max_joint_displacement_ratio": float(disp_norm.max().item()) if int(disp_norm.numel()) > 0 else 0.0,
        "mean_joint_displacement_ratio": float(disp_norm.mean().item()) if int(disp_norm.numel()) > 0 else 0.0,
        "top_displaced_joints": top_displaced,
        "flagged_bone_count": int(len(flagged_bones)),
        "flagged_bones": flagged_bones,
        "bone_rows": bone_rows,
    }


def save_phase3_checkpoint(
    trainer: Any,
    output_dir: Path,
    *,
    phase3_config: Phase3RefineConfig,
    source_phase2_checkpoint: str | None,
    phase2_topology_events: list[dict[str, Any]],
    phase3_summary: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "evorig_next_phase3_checkpoint_v1",
        "current_step": int(trainer.current_step),
        "source_phase2_checkpoint": source_phase2_checkpoint,
        "phase3_config": asdict(phase3_config),
        "phase2_topology_events": phase2_topology_events,
        "trainer_state": trainer._phase1_state_payload(),
        "phase3_summary": phase3_summary,
    }
    path = output_dir / "phase3_checkpoint.pt"
    tmp_path = output_dir / "phase3_checkpoint.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def load_phase3_checkpoint(
    trainer: Any,
    path: str | Path,
    *,
    restore_optimizer: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=trainer.device)
    if str(payload.get("format", "")) != "evorig_next_phase3_checkpoint_v1":
        raise ValueError(f"unsupported phase3 checkpoint format: {payload.get('format')}")
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise ValueError("phase3 checkpoint missing trainer_state")
    trainer.load_phase1_payload(
        trainer_state,
        restore_optimizer=restore_optimizer,
        restore_rng=restore_rng,
    )
    return payload


def _apply_phase3_frozen_only_loss_skips(trainer: Any, cfg: Phase3RefineConfig) -> dict[str, float]:
    if not bool(cfg.skip_frozen_only_losses):
        return {}
    skipped: dict[str, float] = {}

    def skip(name: str) -> None:
        value = float(getattr(trainer.cfg, name, 0.0))
        if abs(value) > 0.0:
            skipped[name] = value
            setattr(trainer.cfg, name, 0.0)

    if bool(cfg.freeze_base_params):
        for name in (
            "loss_temporal_smoothness",
            "loss_pcjs",
            "loss_posed_joint_inside",
            "loss_posed_joint_surface_clearance",
            "loss_scale_anchor",
            "loss_bone_scale_consistency",
            "loss_bone_cov_offdiag",
            "loss_bone_radial_symmetry",
            "loss_bone_scale_band",
        ):
            skip(name)
    if bool(cfg.freeze_rest_joints):
        for name in (
            "loss_rest_joint_anchor",
            "loss_rest_joint_inside",
            "loss_rest_joint_surface_clearance",
        ):
            skip(name)
    return skipped


def run_phase3_refine(
    trainer: Any,
    output_dir: Path,
    *,
    cfg: Phase3RefineConfig,
    source_phase2_checkpoint: str | None = None,
    phase2_topology_events: list[dict[str, Any]] | None = None,
    phase2_signal_config: Phase2TopologyConfig | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = list(phase2_topology_events or [])
    phase2_signal_cfg = phase2_signal_config or Phase2TopologyConfig()

    start_step = int(trainer.current_step)
    start_rest_joints = trainer.skeleton.rest_joints.detach().clone()
    start_bone_parent_idx = trainer.skeleton.bone_parent_idx.detach().clone()
    start_bone_child_idx = trainer.skeleton.bone_child_idx.detach().clone()
    _start_cache, start_metrics = _evaluate_metrics(trainer)

    trainer.cfg.steps = int(start_step + max(int(cfg.steps), 0))
    trainer.cfg.sh_coeff_count = int(cfg.sh_coeff_count)
    trainer.cfg.lr_sh = float(cfg.lr_sh)
    trainer.cfg.loss_gaussian_sh_reg = float(cfg.loss_gaussian_sh_reg)
    if bool(cfg.freeze_base_params):
        trainer.cfg.lr_pose = 0.0
        trainer.cfg.lr_root = 0.0
        trainer.cfg.lr_rot = 0.0
        trainer.cfg.lr_scale = 0.0
        trainer.cfg.lr_opacity = 0.0
        trainer.cfg.lr_value = 0.0
        trainer.cfg.lr_rest_joints = 0.0
        trainer.cfg.lr_lambda_initial = 0.0
        trainer.cfg.lr_lambda_initial_thawed = 0.0
        trainer.cfg.lr_lambda_densified = 0.0
    trainer.cfg.sh_start_step = 0
    if bool(cfg.enable_gaussian_offset):
        trainer.cfg.lr_offset = float(cfg.lr_offset)
        trainer.cfg.gaussian_offset_start_step = int(start_step + max(int(cfg.offset_start_step), 0))
        trainer.cfg.gaussian_offset_target = str(cfg.gaussian_offset_target)
        trainer.cfg.loss_gaussian_offset_anchor = float(cfg.loss_gaussian_offset_anchor)
    else:
        trainer.cfg.lr_offset = 0.0
        trainer.cfg.gaussian_offset_start_step = -1
        trainer.cfg.loss_gaussian_offset_anchor = 0.0
    if bool(cfg.freeze_rest_joints):
        trainer.cfg.rest_joint_start_step = 0
        trainer.rest_joint_train_mask = torch.zeros(
            int(trainer.skeleton.joint_count),
            dtype=torch.bool,
            device=trainer.device,
        )
    else:
        trainer.cfg.rest_joint_start_step = 0
        trainer.rest_joint_train_mask = None
        if cfg.lr_rest_joints is not None:
            trainer.cfg.lr_rest_joints = float(cfg.lr_rest_joints)
    if cfg.loss_pcjs is not None:
        trainer.cfg.loss_pcjs = float(cfg.loss_pcjs)
    if cfg.loss_rest_joint_anchor is not None:
        trainer.cfg.loss_rest_joint_anchor = float(cfg.loss_rest_joint_anchor)
    if cfg.loss_rest_joint_inside is not None:
        trainer.cfg.loss_rest_joint_inside = float(cfg.loss_rest_joint_inside)
    if cfg.loss_posed_joint_surface_clearance is not None:
        trainer.cfg.loss_posed_joint_surface_clearance = float(cfg.loss_posed_joint_surface_clearance)
    if cfg.posed_joint_surface_clearance_ratio is not None:
        trainer.cfg.posed_joint_surface_clearance_ratio = float(cfg.posed_joint_surface_clearance_ratio)
    if cfg.posed_joint_surface_clearance_margin is not None:
        trainer.cfg.posed_joint_surface_clearance_margin = float(cfg.posed_joint_surface_clearance_margin)
    if cfg.loss_rest_joint_surface_clearance is not None:
        trainer.cfg.loss_rest_joint_surface_clearance = float(cfg.loss_rest_joint_surface_clearance)
    if cfg.rest_joint_surface_clearance_ratio is not None:
        trainer.cfg.rest_joint_surface_clearance_ratio = float(cfg.rest_joint_surface_clearance_ratio)
    if cfg.rest_joint_surface_clearance_margin is not None:
        trainer.cfg.rest_joint_surface_clearance_margin = float(cfg.rest_joint_surface_clearance_margin)
    if cfg.loss_illegal_support is not None:
        trainer.cfg.loss_illegal_support = float(cfg.loss_illegal_support)
    if cfg.loss_gaussian_illegal_coverage is not None:
        trainer.cfg.loss_gaussian_illegal_coverage = float(cfg.loss_gaussian_illegal_coverage)
    if cfg.illegal_support_tau is not None:
        trainer.cfg.illegal_support_tau = float(cfg.illegal_support_tau)
    frozen_only_loss_skips = _apply_phase3_frozen_only_loss_skips(trainer, cfg)

    trainer.field.ensure_sh_coeffs(int(cfg.sh_coeff_count), preserve_unit_density=False)
    trainer.field.use_sh_response = True
    trainer._sh_initialized = True
    trainer.optimizer = trainer._build_optimizer()

    train_trace: list[dict[str, Any]] = []
    audit_trace: list[dict[str, Any]] = []
    for local_step in range(1, max(int(cfg.steps), 0) + 1):
        step = start_step + local_step
        metrics = trainer.train_step(step)
        if local_step == 1 or local_step == int(cfg.steps) or local_step % max(int(cfg.trace_interval), 1) == 0:
            cache, eval_metrics = _evaluate_metrics(trainer)
            row = {
                "step": int(step),
                "local_step": int(local_step),
                **{key: float(value) if isinstance(value, float) else value for key, value in metrics.items()},
                **eval_metrics,
            }
            train_trace.append(row)
            del cache
        if local_step == int(cfg.steps) or local_step % max(int(cfg.audit_interval), 1) == 0:
            audit_trace.append(
                {
                    "step": int(step),
                    "local_step": int(local_step),
                    "audit": audit_phase3_state(
                        trainer,
                        start_rest_joints=start_rest_joints,
                        start_bone_parent_idx=start_bone_parent_idx,
                        start_bone_child_idx=start_bone_child_idx,
                        cfg=cfg,
                    ),
                }
            )

    cache, final_metrics = _evaluate_metrics(trainer)
    pred_joint_positions = cache.global_transforms[..., :3, 3]
    pred_joint_rotations = cache.global_transforms[..., :3, :3]
    save_outputs(
        output_dir=output_dir,
        skeleton=trainer.skeleton,
        field=trainer.field,
        pred_vertices=cache.pred_vertices,
        pred_joint_positions=pred_joint_positions,
        pred_joint_rotations=pred_joint_rotations,
        weights=cache.weights,
        events=events,
        topology_diagnostics=[],
    )
    final_signal_summary = None
    if bool(cfg.save_final_topology_signals):
        final_signal_summary = save_phase2_topology_signals(output_dir, trainer, cache, config=phase2_signal_cfg)
    structure_audit = audit_dominant_connectivity(
        trainer.mesh_faces,
        int(trainer.rest_vertices.shape[0]),
        int(trainer.skeleton.joint_count),
        dominant_joint_assignment(cache.weights, cache.legal_support_mass, eps=EPS),
    )
    final_joint_audit = audit_phase3_state(
        trainer,
        start_rest_joints=start_rest_joints,
        start_bone_parent_idx=start_bone_parent_idx,
        start_bone_child_idx=start_bone_child_idx,
        cfg=cfg,
    )
    phase1_state_path = trainer._save_phase1_state(output_dir)
    summary = {
        "format": "evorig_next_phase3_summary_v1",
        "source_phase2_checkpoint": source_phase2_checkpoint,
        "phase3_config": asdict(cfg),
        "start_step": int(start_step),
        "steps": int(cfg.steps),
        "final_step": int(trainer.current_step),
        "start_metrics": start_metrics,
        "final_error_raw": float(final_metrics["final_error_raw"]),
        "final_error_raw_all": float(final_metrics["final_error_raw_all"]),
        "zero_weight_row_count": int(final_metrics["zero_weight_row_count"]),
        "joint_count": int(trainer.skeleton.joint_count),
        "bone_count": int(trainer.skeleton.bone_count),
        "gaussian_count": int(trainer.field.gaussian_count),
        "sh_coeff_count": int(trainer.field.sh_coeff_count),
        "offset_norm_mean": float(trainer.field.offset_local.detach().norm(dim=-1).mean().item()),
        "offset_norm_max": float(trainer.field.offset_local.detach().norm(dim=-1).max().item()),
        "structure_audit": structure_audit,
        "joint_audit": final_joint_audit,
        "audit_trace": audit_trace,
        "train_trace": train_trace,
        "frozen_only_loss_skips": frozen_only_loss_skips,
        "phase1_state_path": str(phase1_state_path),
        "phase3_checkpoint_path": str(output_dir / "phase3_checkpoint.pt"),
        "phase2_topology_signals_path": (
            str(output_dir / "phase2_topology_signals.npz") if final_signal_summary is not None else None
        ),
        "phase2_topology_signal_summary_path": (
            str(output_dir / "phase2_topology_signal_summary.json") if final_signal_summary is not None else None
        ),
        "phase2_topology_signal_summary": (
            {
                "branch_seed_vertex_count": int(final_signal_summary["branch_seed_vertex_count"]),
                "branch_component_count": int(final_signal_summary["branch_component_count"]),
                "split_candidate_count": int(final_signal_summary["split_candidate_count"]),
            }
            if final_signal_summary is not None
            else None
        ),
    }
    save_phase3_checkpoint(
        trainer,
        output_dir,
        phase3_config=cfg,
        source_phase2_checkpoint=source_phase2_checkpoint,
        phase2_topology_events=events,
        phase3_summary=summary,
    )
    (output_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "phase3_joint_audit.json").write_text(json.dumps(final_joint_audit, indent=2), encoding="utf-8")
    return summary
