# Active Workline

Current active workline name: `EvoRig`

Implementation source of truth:

- `src/evorig_next`

Do not use as default:

- old `src/evorig`
- old `src/evorig2`
- old `src/evorig3`
- old accepted-line configs
- old sweep/probe configs

Current active runners:

- `scripts/run_evorig_next_phase1abc.py`
- `scripts/run_evorig_next_phase2_round1.py`
- `scripts/run_evorig_next_phase3_round1.py`
- `scripts/import_unirig_dynamic_glb_sample.py`
- `scripts/prepare_unirig_window_asset.py`
- `scripts/check_evorig_environment.py`

Supported but not default for UniRig dynamic experiments:

- `scripts/import_real_glb_sample.py` for old ActionMesh/real-GLB style assets

Current frozen configs:

- `configs/frozen/evorig_next_base_init_default.yaml`
- `configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml`
- `configs/frozen/evorig_next_phase1b_restrefine_default.yaml`
- `configs/frozen/evorig_next_phase1c_smooth_default.yaml`
- `configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml`
- `configs/frozen/evorig_next_phase3_locked_default.yaml`

Current required docs:

- `docs/current_system_zh.md`
- `docs/current_workline_zh.md`
- `docs/experiment_guardrails_zh.md`
- `docs/evorig_next_strict_runbook_zh.md`
- `docs/claude_code_repo_guide_zh.md`
- `docs/claude_experiment_handoff_zh.md`
- `readme.md`

Current default training semantics:

- Phase1 mainline: `A800 fixed-rest -> B200 rest-refine -> C100 rest-refine + acceleration smooth`
- Phase1/2/3 keep JLG training enabled by default
- current `loss_illegal_support = 0.20`
- current `illegal_support_margin = 0.99`
- current `loss_gaussian_illegal_coverage = 0.0`
- Phase3 keeps rest joints locked; `--unfreeze-rest-joints` is not part of the active protocol

One-mesh rule:

- Process exactly one mesh at a time.
- Do not start a second mesh before the current mesh is explicitly finished or
  explicitly skipped with a recorded reason.
