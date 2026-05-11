from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.data import load_sample
from evorig_next.interactive_viewer import build_final_topology_figure, build_motion_figure, save_figure_html
from evorig_next.phase1_trainer import Phase1Trainer
from evorig_next.phase2_topology import Phase2TopologyConfig, load_phase2_checkpoint
from evorig_next.phase3_refine import (
    Phase3RefineConfig,
    load_phase3_checkpoint,
    run_phase3_refine,
)
from run_evorig_next_phase2_round1 import (
    DATA_DIR,
    DEFAULT_PHASE1_CONFIG,
    OUTPUT_ROOT as PHASE2_OUTPUT_ROOT,
    _base_config_from_phase1_source,
    _phase1_config_from_path,
    _write_phase1_config_for_run,
)

OUTPUT_ROOT = ROOT / "mygs" / "results" / "evorig_next_phase3_round1"
DEFAULT_PHASE2_CHECKPOINT = (
    PHASE2_OUTPUT_ROOT
    / "fixed500_phase2_adaptive_max10_noopstop_round1"
    / "phase2_checkpoint.pt"
)


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


def _build_trainer(device: str, data_dir: Path) -> Phase1Trainer:
    sample = load_sample(data_dir, device=device)
    cfg = _phase1_config_from_path(DEFAULT_PHASE1_CONFIG)
    return Phase1Trainer(
        sample,
        base_config=_base_config_from_phase1_source(DEFAULT_PHASE1_CONFIG),
        phase1_config=cfg,
        device=sample["rest_vertices"].device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evorig_next phase3 topology-fixed refine.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--name", type=str, default="phase3_refine")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume-phase2-checkpoint", type=Path, default=DEFAULT_PHASE2_CHECKPOINT)
    parser.add_argument("--resume-phase3-checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sh-coeff-count", type=int, default=16)
    parser.add_argument("--lr-sh", type=float, default=0.002)
    parser.add_argument("--loss-gaussian-sh-reg", type=float, default=5.0e-4)
    parser.add_argument(
        "--unfreeze-base-params",
        dest="unfreeze_base_params",
        action="store_true",
        default=True,
        help="Do not zero the existing phase1/phase2 pose/root/Gaussian/lambda learning rates in phase3. This is the default.",
    )
    parser.add_argument(
        "--freeze-base-params",
        dest="unfreeze_base_params",
        action="store_false",
        help="Freeze phase1/phase2 base parameters for diagnostic equivalence tests only.",
    )
    parser.add_argument(
        "--enable-gaussian-offset",
        dest="enable_gaussian_offset",
        action="store_true",
        default=True,
        help="Enable rest-space Gaussian support offsets. This is the default Phase3 refinement path.",
    )
    parser.add_argument(
        "--disable-gaussian-offset",
        dest="enable_gaussian_offset",
        action="store_false",
        help="Disable Gaussian offsets for ablation or speed diagnostics.",
    )
    parser.add_argument("--offset-start-step", type=int, default=0)
    parser.add_argument("--gaussian-offset-target", type=str, default="all")
    parser.add_argument("--lr-offset", type=float, default=5.0e-4)
    parser.add_argument("--loss-gaussian-offset-anchor", type=float, default=0.05)
    parser.add_argument("--unfreeze-rest-joints", action="store_true")
    parser.add_argument("--lr-rest-joints", type=float, default=None)
    parser.add_argument("--loss-pcjs", type=float, default=None)
    parser.add_argument("--loss-rest-joint-anchor", type=float, default=None)
    parser.add_argument("--loss-rest-joint-inside", type=float, default=None)
    parser.add_argument("--loss-posed-joint-surface-clearance", type=float, default=None)
    parser.add_argument("--posed-joint-surface-clearance-ratio", type=float, default=None)
    parser.add_argument("--posed-joint-surface-clearance-margin", type=float, default=None)
    parser.add_argument("--loss-rest-joint-surface-clearance", type=float, default=None)
    parser.add_argument("--rest-joint-surface-clearance-ratio", type=float, default=None)
    parser.add_argument("--rest-joint-surface-clearance-margin", type=float, default=None)
    parser.add_argument("--loss-illegal-support", type=float, default=0.20)
    parser.add_argument("--loss-gaussian-illegal-coverage", type=float, default=0.0)
    parser.add_argument("--illegal-support-tau", type=float, default=0.0)
    parser.add_argument(
        "--disable-frozen-loss-skips",
        action="store_true",
        help="Keep computing losses whose parameters are frozen; for equivalence debugging only.",
    )
    parser.add_argument(
        "--save-final-topology-signals",
        action="store_true",
        help="Recompute and save final Phase3 topology diagnostics. Slower; training quality is unchanged.",
    )
    parser.add_argument("--trace-interval", type=int, default=25)
    parser.add_argument("--audit-interval", type=int, default=50)
    parser.add_argument("--joint-displacement-warn-ratio", type=float, default=0.05)
    parser.add_argument("--bone-inside-warn-fraction", type=float, default=0.70)
    parser.add_argument("--bone-direction-warn-cos", type=float, default=0.50)
    parser.add_argument(
        "--skip-viewers",
        dest="skip_viewers",
        action="store_true",
        default=True,
        help="Skip HTML viewer export. This is the default for the current Phase3 training protocol.",
    )
    parser.add_argument(
        "--write-viewers",
        dest="skip_viewers",
        action="store_false",
        help="Write HTML viewers after the run.",
    )
    args = parser.parse_args()
    if args.unfreeze_rest_joints:
        parser.error("--unfreeze-rest-joints is not part of the current default evorig_next Phase3 protocol.")

    run_dir = Path(args.output_root) / str(args.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    trainer = _build_trainer(str(args.device), data_dir)
    phase2_topology_events: list[dict] = []
    source_phase2_checkpoint: str | None = None
    if args.resume_phase3_checkpoint is not None:
        phase3_payload = load_phase3_checkpoint(trainer, args.resume_phase3_checkpoint)
        _write_phase1_config_for_run(run_dir, trainer.cfg, checkpoint_payload=phase3_payload)
        phase2_topology_events = list(phase3_payload.get("phase2_topology_events", []))
        source_phase2_checkpoint = str(phase3_payload.get("source_phase2_checkpoint", ""))
    else:
        phase2_payload = load_phase2_checkpoint(trainer, args.resume_phase2_checkpoint)
        _write_phase1_config_for_run(run_dir, trainer.cfg, checkpoint_payload=phase2_payload)
        phase2_topology_events = list(phase2_payload.get("topology_events", []))
        source_phase2_checkpoint = str(args.resume_phase2_checkpoint)

    cfg = Phase3RefineConfig(
        steps=int(args.steps),
        sh_coeff_count=int(args.sh_coeff_count),
        lr_sh=float(args.lr_sh),
        loss_gaussian_sh_reg=float(args.loss_gaussian_sh_reg),
        freeze_base_params=not bool(args.unfreeze_base_params),
        enable_gaussian_offset=bool(args.enable_gaussian_offset),
        offset_start_step=int(args.offset_start_step),
        gaussian_offset_target=str(args.gaussian_offset_target),
        lr_offset=float(args.lr_offset),
        loss_gaussian_offset_anchor=float(args.loss_gaussian_offset_anchor),
        freeze_rest_joints=not bool(args.unfreeze_rest_joints),
        lr_rest_joints=None if args.lr_rest_joints is None else float(args.lr_rest_joints),
        loss_pcjs=None if args.loss_pcjs is None else float(args.loss_pcjs),
        loss_rest_joint_anchor=None if args.loss_rest_joint_anchor is None else float(args.loss_rest_joint_anchor),
        loss_rest_joint_inside=None if args.loss_rest_joint_inside is None else float(args.loss_rest_joint_inside),
        loss_posed_joint_surface_clearance=(
            None if args.loss_posed_joint_surface_clearance is None else float(args.loss_posed_joint_surface_clearance)
        ),
        posed_joint_surface_clearance_ratio=(
            None if args.posed_joint_surface_clearance_ratio is None else float(args.posed_joint_surface_clearance_ratio)
        ),
        posed_joint_surface_clearance_margin=(
            None if args.posed_joint_surface_clearance_margin is None else float(args.posed_joint_surface_clearance_margin)
        ),
        loss_rest_joint_surface_clearance=(
            None if args.loss_rest_joint_surface_clearance is None else float(args.loss_rest_joint_surface_clearance)
        ),
        rest_joint_surface_clearance_ratio=(
            None if args.rest_joint_surface_clearance_ratio is None else float(args.rest_joint_surface_clearance_ratio)
        ),
        rest_joint_surface_clearance_margin=(
            None if args.rest_joint_surface_clearance_margin is None else float(args.rest_joint_surface_clearance_margin)
        ),
        loss_illegal_support=None if args.loss_illegal_support is None else float(args.loss_illegal_support),
        loss_gaussian_illegal_coverage=(
            None if args.loss_gaussian_illegal_coverage is None else float(args.loss_gaussian_illegal_coverage)
        ),
        illegal_support_tau=None if args.illegal_support_tau is None else float(args.illegal_support_tau),
        skip_frozen_only_losses=not bool(args.disable_frozen_loss_skips),
        save_final_topology_signals=bool(args.save_final_topology_signals),
        trace_interval=int(args.trace_interval),
        audit_interval=int(args.audit_interval),
        joint_displacement_warn_ratio=float(args.joint_displacement_warn_ratio),
        bone_inside_warn_fraction=float(args.bone_inside_warn_fraction),
        bone_direction_warn_cos=float(args.bone_direction_warn_cos),
    )
    summary = run_phase3_refine(
        trainer,
        run_dir,
        cfg=cfg,
        source_phase2_checkpoint=source_phase2_checkpoint,
        phase2_topology_events=phase2_topology_events,
        phase2_signal_config=Phase2TopologyConfig(),
    )
    summary["wall_time_sec"] = float(time.perf_counter() - started)
    summary["data_dir"] = str(data_dir)
    summary["viewer_errors"] = {} if bool(args.skip_viewers) else _write_viewers(run_dir, data_dir)
    (run_dir / "phase3_entry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "final_error_raw": float(summary["final_error_raw"]),
                "final_error_raw_all": float(summary["final_error_raw_all"]),
                "zero_weight_row_count": int(summary["zero_weight_row_count"]),
                "joint_count": int(summary["joint_count"]),
                "bone_count": int(summary["bone_count"]),
                "gaussian_count": int(summary["gaussian_count"]),
                "sh_coeff_count": int(summary["sh_coeff_count"]),
                "flagged_bone_count": int(summary["joint_audit"]["flagged_bone_count"]),
                "max_joint_displacement_ratio": float(summary["joint_audit"]["max_joint_displacement_ratio"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
