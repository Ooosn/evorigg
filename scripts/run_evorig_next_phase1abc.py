from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.data import load_sample
from evorig_next.phase1_config import Phase1Config, Phase1DensifyStage
from evorig_next.phase1_trainer import Phase1Trainer


BASE_CONFIG = ROOT / "configs" / "frozen" / "evorig_next_base_init_default.yaml"
DEFAULT_A_CONFIG = ROOT / "configs" / "frozen" / "evorig_next_phase1_final500_supportloss_default.yaml"
DEFAULT_B_CONFIG = ROOT / "configs" / "frozen" / "evorig_next_phase1b_restrefine_default.yaml"
DEFAULT_C_CONFIG = ROOT / "configs" / "frozen" / "evorig_next_phase1c_smooth_default.yaml"


def _phase1_config_from_json(path: Path, *, steps_override: int | None = None) -> Phase1Config:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    phase_payload = dict(payload.get("phase1", payload))
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


def _base_config_from_phase1_source(path: Path) -> dict:
    base_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8-sig"))
    init_overrides = {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        phase1_payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        init_overrides = dict(phase1_payload.get("init_overrides", {}))
    merged = deepcopy(base_config)
    merged["init"] = dict(merged.get("init", {}))
    merged["init"].update(init_overrides)
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase1 A/B/C in one process with shared mesh caches.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name-prefix", default="phase1")
    parser.add_argument("--phase1a-config", type=Path, default=DEFAULT_A_CONFIG)
    parser.add_argument("--phase1b-config", type=Path, default=DEFAULT_B_CONFIG)
    parser.add_argument("--phase1c-config", type=Path, default=DEFAULT_C_CONFIG)
    parser.add_argument("--phase1a-steps", type=int, default=800)
    parser.add_argument("--phase1b-steps", type=int, default=None)
    parser.add_argument("--phase1c-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--export-intermediate-topology-signals",
        action="store_true",
        help="Export Phase2 topology signals for A and B as diagnostics. Default skips them.",
    )
    return parser.parse_args()


def _stage_config(path: Path, steps: int | None):
    return _phase1_config_from_path(path, steps_override=steps)


def _write_stage_config(output_dir: Path, cfg) -> None:
    (output_dir / "phase1_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    sample = load_sample(args.data_dir, device=args.device)
    base_config = _base_config_from_phase1_source(Path(args.phase1a_config))
    trainer = Phase1Trainer(
        sample,
        base_config=base_config,
        phase1_config=_stage_config(Path(args.phase1a_config), int(args.phase1a_steps)),
        device=sample["rest_vertices"].device,
    )
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stages = [
        (
            f"{args.name_prefix}a_{int(args.phase1a_steps)}_fixedrest",
            _stage_config(Path(args.phase1a_config), int(args.phase1a_steps)),
            bool(args.export_intermediate_topology_signals),
        ),
        (
            f"{args.name_prefix}b_restrefine",
            _stage_config(Path(args.phase1b_config), args.phase1b_steps),
            bool(args.export_intermediate_topology_signals),
        ),
        (
            f"{args.name_prefix}c_smooth",
            _stage_config(Path(args.phase1c_config), args.phase1c_steps),
            True,
        ),
    ]
    summaries: list[dict] = []
    for run_name, cfg, export_signals in stages:
        trainer.cfg = cfg
        trainer.optimizer = trainer._build_optimizer()
        trainer._init_lambda_optimizer_state()
        run_dir = out_root / run_name
        summary = trainer.run(run_dir, export_topology_signals=export_signals)
        _write_stage_config(run_dir, cfg)
        summaries.append({"run_name": run_name, "run_dir": str(run_dir), "summary": summary})

    entry_summary = {
        "format": "evorig_next_phase1abc_v1",
        "data_dir": str(args.data_dir),
        "stages": summaries,
    }
    (out_root / "phase1abc_summary.json").write_text(json.dumps(entry_summary, indent=2), encoding="utf-8")
    print(json.dumps(entry_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
