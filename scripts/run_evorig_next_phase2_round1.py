from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.data import load_sample
from evorig_next.interactive_viewer import build_final_topology_figure, build_motion_figure, save_figure_html
from evorig_next.phase1_config import Phase1Config, Phase1DensifyStage
from evorig_next.phase1_losses import vertex_recon_loss
from evorig_next.phase1_trainer import Phase1Trainer
from evorig_next.phase2_topology import (
    Phase2TopologyConfig,
    load_phase2_checkpoint,
    run_phase2_topology_scheduled_refine,
    save_phase2_checkpoint,
    save_phase2_topology_signals,
)


BASE_CONFIG = ROOT / "configs" / "frozen" / "evorig_next_base_init_default.yaml"
DEFAULT_PHASE1_CONFIG = ROOT / "configs" / "frozen" / "evorig_next_phase1_final500_supportloss_default.yaml"
DATA_DIR = ROOT / "mygs" / "demo_data" / "real_glb_preprocess_restore_check"
OUTPUT_ROOT = ROOT / "mygs" / "results" / "evorig_next_phase2_round1"


def _phase1_config_from_json(path: Path, *, steps_override: int | None = None) -> Phase1Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    has_densify_stages = "densify_stages" in payload
    stages = [
        Phase1DensifyStage(
            warm_steps=int(item["warm_steps"]),
            settle_steps=int(item["settle_steps"]),
            max_bones=int(item["max_bones"]),
            seeds_per_bone=int(item.get("seeds_per_bone", 1)),
        )
        for item in payload.pop("densify_stages", [])
    ]
    if steps_override is not None:
        payload["steps"] = int(steps_override)
    if has_densify_stages:
        payload["densify_stages"] = stages
    valid_keys = {item.name for item in fields(Phase1Config)}
    payload = {key: value for key, value in payload.items() if key in valid_keys}
    return Phase1Config(**payload)


def _phase1_config_from_yaml(path: Path, *, steps_override: int | None = None) -> Phase1Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    phase_payload = dict(payload.get("phase1", {}))
    has_densify_stages = "densify_stages" in phase_payload
    stages = [
        Phase1DensifyStage(
            warm_steps=int(item["warm_steps"]),
            settle_steps=int(item["settle_steps"]),
            max_bones=int(item["max_bones"]),
            seeds_per_bone=int(item.get("seeds_per_bone", 1)),
        )
        for item in phase_payload.pop("densify_stages", [])
    ]
    if steps_override is not None:
        phase_payload["steps"] = int(steps_override)
    if has_densify_stages:
        phase_payload["densify_stages"] = stages
    valid_keys = {item.name for item in fields(Phase1Config)}
    phase_payload = {key: value for key, value in phase_payload.items() if key in valid_keys}
    return Phase1Config(**phase_payload)


def _phase1_config_from_path(path: Path, *, steps_override: int | None = None) -> Phase1Config:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _phase1_config_from_json(path, steps_override=steps_override)
    if suffix in {".yaml", ".yml"}:
        return _phase1_config_from_yaml(path, steps_override=steps_override)
    raise ValueError(f"unsupported phase1 config format: {path}")


def _base_config_from_phase1_source(path: Path) -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    if path.suffix.lower() in {".yaml", ".yml"}:
        phase1_payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        init_overrides = dict(phase1_payload.get("init_overrides", {}))
    else:
        init_overrides = {}
    merged = deepcopy(base_config)
    merged["init"] = dict(merged.get("init", {}))
    merged["init"].update(init_overrides)
    return merged


def _phase1_config_payload_from_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if str(payload.get("format", "")) == "evorig_next_phase1_state_v1":
        raw = payload.get("phase1_config")
    elif str(payload.get("format", "")) in {"evorig_next_phase2_checkpoint_v1", "evorig_next_phase3_checkpoint_v1"}:
        trainer_state = payload.get("trainer_state")
        raw = trainer_state.get("phase1_config") if isinstance(trainer_state, dict) else None
    else:
        raw = None
    return dict(raw) if isinstance(raw, dict) else None


def _phase1_config_payload_from_checkpoint(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    try:
        payload = torch.load(Path(path), map_location="cpu")
    except Exception:
        return None
    return _phase1_config_payload_from_checkpoint_payload(payload) if isinstance(payload, dict) else None


def _write_phase1_config_for_run(
    run_dir: Path,
    cfg: Phase1Config,
    *,
    resume_phase1_state: Path | None = None,
    checkpoint_payload: dict[str, Any] | None = None,
) -> None:
    payload = None
    if isinstance(checkpoint_payload, dict):
        payload = _phase1_config_payload_from_checkpoint_payload(checkpoint_payload)
    if payload is None and resume_phase1_state is not None:
        payload = _phase1_config_payload_from_checkpoint(Path(resume_phase1_state))
    if payload is None:
        payload = cfg.to_dict()
    (run_dir / "phase1_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_viewers(run_dir: Path, data_dir: Path) -> dict[str, str]:
    viewer_errors: dict[str, str] = {}
    (run_dir / "viewer_summary.json").write_text(
        json.dumps({"data_dir": str(data_dir), "output_dir": str(run_dir)}, indent=2),
        encoding="utf-8",
    )
    interactive_dir = run_dir / "visuals" / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    try:
        save_figure_html(build_motion_figure(run_dir), interactive_dir / "interactive_motion.html")
    except Exception as exc:
        viewer_errors["interactive_motion"] = repr(exc)
    try:
        save_figure_html(build_final_topology_figure(run_dir), interactive_dir / "interactive_final_topology.html")
    except Exception as exc:
        viewer_errors["interactive_final_topology"] = repr(exc)
    if viewer_errors:
        (run_dir / "viewer_errors.json").write_text(json.dumps(viewer_errors, indent=2), encoding="utf-8")
    return viewer_errors


def _phase2_topology_config_from_args(args: argparse.Namespace) -> Phase2TopologyConfig:
    return Phase2TopologyConfig(
        vertex_error_quantile=float(args.phase2_vertex_error_quantile),
        wrong_coverage_error_quantile=float(args.phase2_wrong_coverage_error_quantile),
        coverage_quantile=float(args.phase2_coverage_quantile),
        coverage_abs_threshold=float(args.phase2_coverage_abs_threshold),
        topology_support_sigma=float(args.phase2_topology_support_sigma),
        wrong_coverage_ratio=float(args.phase2_wrong_coverage_ratio),
        component_min_vertices=int(args.phase2_component_min_vertices),
        component_min_vertices_reference_area=float(args.phase2_component_min_vertices_reference_area),
        component_min_vertices_reference_vertex_count=int(args.phase2_component_min_vertices_reference_vertex_count),
        component_min_vertices_min=int(args.phase2_component_min_vertices_min),
        component_merge_hops=int(args.phase2_component_merge_hops),
        max_branch_components=int(args.phase2_max_branch_components),
        branch_min_global_error_mass_fraction=float(args.phase2_branch_min_global_error_mass_fraction),
        branch_min_wrong_fraction=float(args.phase2_branch_min_wrong_fraction),
        branch_min_uncovered_fraction=float(args.phase2_branch_min_uncovered_fraction),
        branch_min_score_fraction_of_best=float(args.phase2_branch_min_score_fraction_of_best),
        branch_component_overlap_reject_fraction=float(args.phase2_branch_component_overlap_reject_fraction),
        branch_accept_max_post_seed_fraction=float(args.phase2_branch_accept_max_post_seed_fraction),
        branch_accept_max_post_fault_fraction=float(args.phase2_branch_accept_max_post_fault_fraction),
        branch_max_intermediate_points=int(args.phase2_branch_max_intermediate_points),
        branch_min_path_points=int(args.phase2_branch_min_path_points),
        branch_segment_refine_inside_fraction=float(args.phase2_branch_segment_refine_inside_fraction),
        branch_segment_refine_max_points=int(args.phase2_branch_segment_refine_max_points),
        branch_tip_target_query_k=int(args.phase2_branch_tip_target_query_k),
        branch_tip_target_radius_voxels=float(args.phase2_branch_tip_target_radius_voxels),
        branch_tip_target_radius_ratio=float(args.phase2_branch_tip_target_radius_ratio),
        branch_tip_target_distance_weight=float(args.phase2_branch_tip_target_distance_weight),
        branch_long_segment_refine=not bool(args.phase2_disable_branch_long_segment_refine),
        branch_long_segment_max_arc_fraction=float(args.phase2_branch_long_segment_max_arc_fraction),
        branch_path_clearance_weight=float(args.phase2_branch_path_clearance_weight),
        branch_path_clearance_power=float(args.phase2_branch_path_clearance_power),
        split_inside_min_fraction=float(args.phase2_split_inside_min_fraction),
        topology_update_interval_steps=int(args.phase2_topology_interval_steps),
        topology_max_branch_per_update=int(args.phase2_max_branch_per_update),
        topology_max_split_per_update=int(args.phase2_max_split_per_update),
        topology_noop_stop_patience=int(args.phase2_topology_noop_stop_patience),
        seed_joint_repair_enabled=not bool(args.phase2_disable_seed_joint_repair),
        seed_joint_repair_variant=str(args.phase2_seed_joint_repair_variant),
        seed_joint_repair_max_per_update=int(args.phase2_seed_joint_repair_max_per_update),
        seed_joint_repair_max_components=int(args.phase2_seed_joint_repair_max_components),
        seed_joint_repair_min_vertices=int(args.phase2_seed_joint_repair_min_vertices),
        seed_joint_repair_min_vertices_reference_area=float(args.phase2_seed_joint_repair_min_vertices_reference_area),
        seed_joint_repair_min_vertices_reference_vertex_count=int(
            args.phase2_seed_joint_repair_min_vertices_reference_vertex_count
        ),
        seed_joint_repair_min_vertices_min=int(args.phase2_seed_joint_repair_min_vertices_min),
        seed_joint_repair_min_neighbor_fraction=float(args.phase2_seed_joint_repair_min_neighbor_fraction),
        seed_joint_repair_min_fault_fraction=float(args.phase2_seed_joint_repair_min_fault_fraction),
        seed_joint_repair_cap_sample_radius_ratio=float(args.phase2_seed_joint_repair_cap_sample_radius_ratio),
        seed_joint_repair_inside_min_fraction=float(args.phase2_seed_joint_repair_inside_min_fraction),
        seed_joint_repair_min_inside_improvement=float(args.phase2_seed_joint_repair_min_inside_improvement),
        phase2_loss_illegal_support=float(args.phase2_loss_illegal_support),
        phase2_loss_gaussian_illegal_coverage=float(args.phase2_loss_gaussian_illegal_coverage),
        phase2_illegal_support_tau=float(args.phase2_illegal_support_tau),
        phase2_illegal_support_margin=float(args.phase2_illegal_support_margin),
    )


def _phase2_signal_brief(summary: dict[str, Any]) -> dict[str, int]:
    return {
        "branch_seed_vertex_count": int(summary["branch_seed_vertex_count"]),
        "branch_component_count": int(summary["branch_component_count"]),
        "seed_joint_repair_candidate_count": int(summary.get("seed_joint_repair_candidate_count", 0)),
        "split_candidate_count": int(summary["split_candidate_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed evorig_next phase1 line and export phase2 topology signals."
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--name", type=str, default="phase1_fixed_signals")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Override phase1 steps for smoke tests only.")
    parser.add_argument(
        "--phase1-config",
        type=Path,
        default=DEFAULT_PHASE1_CONFIG,
        help="Phase1 JSON/YAML config. Default enables JLG support losses.",
    )
    parser.add_argument(
        "--resume-phase1-state",
        type=Path,
        default=None,
        help="Load a saved phase1_state.pt and skip rerunning phase1.",
    )
    parser.add_argument(
        "--resume-phase2-checkpoint",
        type=Path,
        default=None,
        help="Load a saved phase2_checkpoint.pt and continue from the phase2 topology state.",
    )
    parser.add_argument(
        "--skip-viewers",
        dest="skip_viewers",
        action="store_true",
        default=True,
        help="Skip HTML viewer export. This is the default for the current Phase2 training protocol.",
    )
    parser.add_argument(
        "--write-viewers",
        dest="skip_viewers",
        action="store_false",
        help="Write HTML viewers after the run.",
    )
    parser.add_argument("--phase2-new-seeds-per-bone", type=int, default=8)
    parser.add_argument(
        "--phase2-schedule",
        dest="phase2_schedule",
        action="store_true",
        default=True,
        help="Run the scheduled phase2 topology loop. This is the default current protocol.",
    )
    parser.add_argument(
        "--no-phase2-schedule",
        dest="phase2_schedule",
        action="store_false",
        help="Only export Phase2 topology signals; use for diagnostics, not the default training route.",
    )
    parser.add_argument("--phase2-schedule-max-updates", type=int, default=4)
    parser.add_argument("--phase2-topology-interval-steps", type=int, default=200)
    parser.add_argument("--phase2-max-branch-per-update", type=int, default=3)
    parser.add_argument("--phase2-max-split-per-update", type=int, default=3)
    parser.add_argument("--phase2-topology-noop-stop-patience", type=int, default=2)
    parser.add_argument("--phase2-disable-seed-joint-repair", action="store_true")
    parser.add_argument("--phase2-seed-joint-repair-variant", type=str, default="center_capB")
    parser.add_argument("--phase2-seed-joint-repair-max-per-update", type=int, default=2)
    parser.add_argument("--phase2-seed-joint-repair-max-components", type=int, default=16)
    parser.add_argument("--phase2-seed-joint-repair-min-vertices", type=int, default=24)
    parser.add_argument("--phase2-seed-joint-repair-min-vertices-reference-area", type=float, default=3.9673382939115642)
    parser.add_argument("--phase2-seed-joint-repair-min-vertices-reference-vertex-count", type=int, default=19869)
    parser.add_argument("--phase2-seed-joint-repair-min-vertices-min", type=int, default=8)
    parser.add_argument("--phase2-seed-joint-repair-min-neighbor-fraction", type=float, default=0.80)
    parser.add_argument("--phase2-seed-joint-repair-min-fault-fraction", type=float, default=0.50)
    parser.add_argument("--phase2-seed-joint-repair-cap-sample-radius-ratio", type=float, default=0.16)
    parser.add_argument("--phase2-seed-joint-repair-inside-min-fraction", type=float, default=0.70)
    parser.add_argument("--phase2-seed-joint-repair-min-inside-improvement", type=float, default=0.05)
    parser.add_argument("--phase2-loss-illegal-support", type=float, default=0.20)
    parser.add_argument("--phase2-loss-gaussian-illegal-coverage", type=float, default=0.0)
    parser.add_argument("--phase2-illegal-support-tau", type=float, default=0.0)
    parser.add_argument("--phase2-illegal-support-margin", type=float, default=0.99)
    parser.add_argument("--phase2-topology-support-sigma", type=float, default=5.0)
    parser.add_argument("--phase2-vertex-error-quantile", type=float, default=0.80)
    parser.add_argument("--phase2-wrong-coverage-error-quantile", type=float, default=0.50)
    parser.add_argument("--phase2-coverage-quantile", type=float, default=0.05)
    parser.add_argument("--phase2-coverage-abs-threshold", type=float, default=0.0)
    parser.add_argument("--phase2-wrong-coverage-ratio", type=float, default=0.35)
    parser.add_argument("--phase2-component-min-vertices", type=int, default=10)
    parser.add_argument("--phase2-component-min-vertices-reference-area", type=float, default=3.9673382939115642)
    parser.add_argument("--phase2-component-min-vertices-reference-vertex-count", type=int, default=19869)
    parser.add_argument("--phase2-component-min-vertices-min", type=int, default=4)
    parser.add_argument("--phase2-component-merge-hops", type=int, default=2)
    parser.add_argument("--phase2-max-branch-components", type=int, default=32)
    parser.add_argument("--phase2-branch-min-global-error-mass-fraction", type=float, default=0.03)
    parser.add_argument("--phase2-branch-min-wrong-fraction", type=float, default=0.70)
    parser.add_argument("--phase2-branch-min-uncovered-fraction", type=float, default=0.80)
    parser.add_argument("--phase2-branch-min-score-fraction-of-best", type=float, default=0.08)
    parser.add_argument("--phase2-split-inside-min-fraction", type=float, default=0.75)
    parser.add_argument("--phase2-branch-component-overlap-reject-fraction", type=float, default=0.25)
    parser.add_argument("--phase2-branch-accept-max-post-seed-fraction", type=float, default=0.50)
    parser.add_argument("--phase2-branch-accept-max-post-fault-fraction", type=float, default=0.50)
    parser.add_argument("--phase2-branch-max-intermediate-points", type=int, default=4)
    parser.add_argument("--phase2-branch-min-path-points", type=int, default=2)
    parser.add_argument("--phase2-branch-segment-refine-inside-fraction", type=float, default=0.75)
    parser.add_argument("--phase2-branch-segment-refine-max-points", type=int, default=10)
    parser.add_argument("--phase2-branch-tip-target-query-k", type=int, default=192)
    parser.add_argument("--phase2-branch-tip-target-radius-voxels", type=float, default=3.0)
    parser.add_argument("--phase2-branch-tip-target-radius-ratio", type=float, default=0.06)
    parser.add_argument("--phase2-branch-tip-target-distance-weight", type=float, default=1.0)
    parser.add_argument("--phase2-disable-branch-long-segment-refine", action="store_true")
    parser.add_argument("--phase2-branch-long-segment-max-arc-fraction", type=float, default=0.48)
    parser.add_argument("--phase2-branch-path-clearance-weight", type=float, default=2.0)
    parser.add_argument("--phase2-branch-path-clearance-power", type=float, default=1.0)
    args = parser.parse_args()

    phase1_cfg_path = Path(args.phase1_config)
    cfg = _phase1_config_from_path(phase1_cfg_path, steps_override=args.steps)
    phase2_cfg = _phase2_topology_config_from_args(args)
    data_dir = Path(args.data_dir)
    sample = load_sample(data_dir, device=args.device)
    device = sample["rest_vertices"].device
    trainer = Phase1Trainer(
        sample,
        base_config=_base_config_from_phase1_source(phase1_cfg_path),
        phase1_config=cfg,
        device=device,
    )
    run_dir = Path(args.output_root) / str(args.name)
    started = time.perf_counter()
    phase2_checkpoint_events: list[dict[str, Any]] = []
    schedule_summary: dict[str, Any] | None = None
    if args.resume_phase1_state is not None and args.resume_phase2_checkpoint is not None:
        raise ValueError("Use only one of --resume-phase1-state or --resume-phase2-checkpoint.")
    if args.resume_phase2_checkpoint is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = load_phase2_checkpoint(trainer, args.resume_phase2_checkpoint)
        _write_phase1_config_for_run(run_dir, cfg, checkpoint_payload=checkpoint_payload)
        phase2_checkpoint_events = list(checkpoint_payload.get("topology_events", []))
        cache = trainer.evaluate_full()
        recon_mask = trainer.legal_vertex_mask.unsqueeze(0).expand(int(trainer.gt_vertices.shape[0]), -1).to(
            dtype=cache.pred_vertices.dtype,
            device=cache.pred_vertices.device,
        )
        if bool(args.phase2_schedule):
            summary = {
                "mode": "evorig_next_phase2_checkpoint_resume_scheduled",
                "steps": int(trainer.current_step),
                "gaussian_count": int(trainer.field.gaussian_count),
                "joint_count": int(trainer.skeleton.joint_count),
                "bone_count": int(trainer.skeleton.bone_count),
                "initial_error_raw": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices, mask=recon_mask).item()),
                "initial_error_raw_all": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices).item()),
                "initial_zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
                "initial_phase2_signal_export_skipped": True,
                "resume_phase2_checkpoint": str(args.resume_phase2_checkpoint),
                "resume_phase2_checkpoint_format": str(checkpoint_payload.get("format", "")),
            }
        else:
            phase2_signal_summary = save_phase2_topology_signals(run_dir, trainer, cache, config=phase2_cfg)
            legacy_state_path = trainer._save_phase1_state(run_dir)
            summary = {
                "mode": "evorig_next_phase2_checkpoint_resume_signal_export",
                "steps": int(trainer.current_step),
                "gaussian_count": int(trainer.field.gaussian_count),
                "joint_count": int(trainer.skeleton.joint_count),
                "bone_count": int(trainer.skeleton.bone_count),
                "final_error_raw": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices, mask=recon_mask).item()),
                "final_error_raw_all": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices).item()),
                "zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
                "phase1_state_path": str(legacy_state_path),
                "phase2_checkpoint_path": str(run_dir / "phase2_checkpoint.pt"),
                "phase2_topology_signals_path": str(run_dir / "phase2_topology_signals.npz"),
                "phase2_topology_signal_summary_path": str(run_dir / "phase2_topology_signal_summary.json"),
                "phase2_topology_signal_summary": _phase2_signal_brief(phase2_signal_summary),
                "resume_phase2_checkpoint": str(args.resume_phase2_checkpoint),
                "resume_phase2_checkpoint_format": str(checkpoint_payload.get("format", "")),
            }
            save_phase2_checkpoint(
                trainer,
                run_dir,
                topology_config=phase2_cfg,
                phase2_summary=summary,
                topology_events=phase2_checkpoint_events,
                topology_signal_summary=phase2_signal_summary,
            )
    elif args.resume_phase1_state is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer.load_phase1_state(args.resume_phase1_state)
        _write_phase1_config_for_run(run_dir, cfg, resume_phase1_state=args.resume_phase1_state)
        cache = trainer.evaluate_full()
        recon_mask = trainer.legal_vertex_mask.unsqueeze(0).expand(int(trainer.gt_vertices.shape[0]), -1).to(
            dtype=cache.pred_vertices.dtype,
            device=cache.pred_vertices.device,
        )
        if bool(args.phase2_schedule):
            summary = {
                "mode": "evorig_next_phase2_resume_scheduled",
                "steps": int(trainer.current_step),
                "gaussian_count": int(trainer.field.gaussian_count),
                "joint_count": int(trainer.skeleton.joint_count),
                "bone_count": int(trainer.skeleton.bone_count),
                "initial_error_raw": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices, mask=recon_mask).item()),
                "initial_error_raw_all": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices).item()),
                "initial_zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
                "initial_phase2_signal_export_skipped": True,
            }
        else:
            phase2_signal_summary = save_phase2_topology_signals(run_dir, trainer, cache, config=phase2_cfg)
            trainer._save_phase1_state(run_dir)
            summary = {
                "mode": "evorig_next_phase2_resume_signal_export",
                "steps": int(trainer.current_step),
                "gaussian_count": int(trainer.field.gaussian_count),
                "joint_count": int(trainer.skeleton.joint_count),
                "final_error_raw": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices, mask=recon_mask).item()),
                "final_error_raw_all": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices).item()),
                "zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
                "phase1_state_path": str(run_dir / "phase1_state.pt"),
                "phase2_topology_signals_path": str(run_dir / "phase2_topology_signals.npz"),
                "phase2_topology_signal_summary_path": str(run_dir / "phase2_topology_signal_summary.json"),
                "phase2_topology_signal_summary": _phase2_signal_brief(phase2_signal_summary),
            }
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_phase1_config_for_run(run_dir, cfg)
        summary = trainer.run(run_dir)
        summary["mode"] = "evorig_next_phase2_phase1_signal_export"
    summary["source_phase1_config"] = str(phase1_cfg_path)
    summary["phase1_jlg_losses"] = {
        "loss_illegal_support": float(cfg.loss_illegal_support),
        "loss_gaussian_illegal_coverage": float(cfg.loss_gaussian_illegal_coverage),
        "illegal_support_tau": float(cfg.illegal_support_tau),
    }
    summary["data_dir"] = str(data_dir)
    if args.resume_phase1_state is not None:
        summary["resume_phase1_state"] = str(args.resume_phase1_state)
    summary["wall_time_sec"] = float(time.perf_counter() - started)
    if bool(args.phase2_schedule):
        schedule_summary = run_phase2_topology_scheduled_refine(
            trainer,
            run_dir,
            topology_config=phase2_cfg,
            max_updates=int(args.phase2_schedule_max_updates),
            seeds_per_new_bone=int(args.phase2_new_seeds_per_bone),
        )
        summary["phase2_schedule_summary_path"] = str(run_dir / "phase2_schedule_summary.json")
        summary["phase2_schedule_summary"] = {
            "completed_update_count": int(schedule_summary["completed_update_count"]),
            "accepted_event_count": int(schedule_summary["accepted_event_count"]),
            "final_error_raw": float(schedule_summary["final_error_raw"]),
            "zero_weight_row_count": int(schedule_summary["zero_weight_row_count"]),
        }
        phase2_checkpoint_events = list(schedule_summary.get("topology_events", []))
        summary["final_error_raw"] = float(schedule_summary["final_error_raw"])
        summary["final_error_raw_all"] = float(schedule_summary["final_error_raw_all"])
        summary["zero_weight_row_count"] = int(schedule_summary["zero_weight_row_count"])
        summary["joint_count"] = int(schedule_summary["joint_count"])
        summary["gaussian_count"] = int(schedule_summary["gaussian_count"])
    summary["wall_time_sec"] = float(time.perf_counter() - started)
    if schedule_summary is not None:
        summary["phase1_state_path"] = str(run_dir / "phase1_state.pt")
        summary["phase2_checkpoint_path"] = str(run_dir / "phase2_checkpoint.pt")
        summary["phase2_topology_signals_path"] = str(run_dir / "phase2_topology_signals.npz")
        summary["phase2_topology_signal_summary_path"] = str(run_dir / "phase2_topology_signal_summary.json")
        summary["phase2_topology_signal_summary"] = dict(schedule_summary["post_phase2_topology_signal_summary"])
        summary["steps"] = int(trainer.current_step)
        summary["gaussian_count"] = int(trainer.field.gaussian_count)
        summary["joint_count"] = int(trainer.skeleton.joint_count)
        summary["bone_count"] = int(trainer.skeleton.bone_count)
        summary["final_error_raw"] = float(schedule_summary["final_error_raw"])
        summary["final_error_raw_all"] = float(schedule_summary["final_error_raw_all"])
        summary["zero_weight_row_count"] = int(schedule_summary["zero_weight_row_count"])
    else:
        final_cache = trainer.evaluate_full()
        final_signal_summary = save_phase2_topology_signals(run_dir, trainer, final_cache, config=phase2_cfg)
        final_state_path = trainer._save_phase1_state(run_dir)
        final_recon_mask = trainer.legal_vertex_mask.unsqueeze(0).expand(int(trainer.gt_vertices.shape[0]), -1).to(
            dtype=final_cache.pred_vertices.dtype,
            device=final_cache.pred_vertices.device,
        )
        summary["phase1_state_path"] = str(final_state_path)
        summary["phase2_checkpoint_path"] = str(run_dir / "phase2_checkpoint.pt")
        summary["phase2_topology_signals_path"] = str(run_dir / "phase2_topology_signals.npz")
        summary["phase2_topology_signal_summary_path"] = str(run_dir / "phase2_topology_signal_summary.json")
        summary["phase2_topology_signal_summary"] = _phase2_signal_brief(final_signal_summary)
        summary["steps"] = int(trainer.current_step)
        summary["gaussian_count"] = int(trainer.field.gaussian_count)
        summary["joint_count"] = int(trainer.skeleton.joint_count)
        summary["bone_count"] = int(trainer.skeleton.bone_count)
        summary["final_error_raw"] = float(
            vertex_recon_loss(final_cache.pred_vertices, trainer.gt_vertices, mask=final_recon_mask).item()
        )
        summary["final_error_raw_all"] = float(vertex_recon_loss(final_cache.pred_vertices, trainer.gt_vertices).item())
        summary["zero_weight_row_count"] = int(final_cache.zero_weight_mask.sum().item())
        save_phase2_checkpoint(
            trainer,
            run_dir,
            topology_config=phase2_cfg,
            phase2_summary=summary,
            topology_events=phase2_checkpoint_events,
            topology_signal_summary=final_signal_summary,
        )
    if not bool(args.skip_viewers):
        summary["viewer_errors"] = _write_viewers(run_dir, data_dir)
    (run_dir / "phase2_entry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_dir": str(run_dir),
        "final_error_raw": float(summary.get("final_error_raw", 0.0)),
        "phase2_signals": summary.get("phase2_topology_signal_summary", {}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
