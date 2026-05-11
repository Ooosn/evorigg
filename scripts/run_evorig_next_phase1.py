from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.data import load_sample
from evorig_next.phase1_trainer import Phase1Trainer
from run_evorig_next_phase2_round1 import (
    DEFAULT_PHASE1_CONFIG,
    _base_config_from_phase1_source,
    _phase1_config_from_path,
)


DEFAULT_DATA_DIR = ROOT / "mygs" / "demo_data" / "real_glb_preprocess_restore_check"
DEFAULT_OUTPUT_ROOT = ROOT / "mygs" / "results" / "evorig_next_phase1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the new isolated evorig_next phase1-only trainer.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="phase1_baseline_v1")
    parser.add_argument("--phase1-config", type=Path, default=DEFAULT_PHASE1_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=None)
    parser.add_argument("--resume-phase1-state", type=Path, default=None)
    parser.add_argument(
        "--skip-topology-signals",
        action="store_true",
        help="Skip exporting Phase2 topology signals at the end of this Phase1 run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    phase1_config_path = Path(args.phase1_config)
    base_config = _base_config_from_phase1_source(phase1_config_path)
    sample = load_sample(args.data_dir, device=args.device)
    cfg = _phase1_config_from_path(phase1_config_path, steps_override=args.steps)
    if args.frame_batch_size is not None:
        cfg.frame_batch_size = int(args.frame_batch_size)
    output_dir = Path(args.output_dir) / str(args.run_name)
    trainer = Phase1Trainer(
        sample,
        base_config=base_config,
        phase1_config=cfg,
        device=sample["rest_vertices"].device,
    )
    if args.resume_phase1_state is not None:
        trainer.load_phase1_state(
            args.resume_phase1_state,
            restore_optimizer=False,
            restore_rng=False,
        )
    summary = trainer.run(output_dir, export_topology_signals=not bool(args.skip_topology_signals))
    (output_dir / "phase1_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
