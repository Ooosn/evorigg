# Commit Journal

本文件只记录当前 EvoRig 主线的工作块。实现包名暂时仍为 `src/evorig_next`。
旧长日志已从工作区清理；
如需追溯历史，请使用 git history。

- `date`: 2026-05-11
- `commit`: `recorded in this documentation commit`
- `message`: `docs: add claude code repo guide`
- `summary`: Added a focused repository usage guide for another coding agent, specifically Claude Code. The new document consolidates the active-line rules into one place: required docs to read first, environment and preflight command, one-mesh-at-a-time rule, UniRig/dynamic mesh protocol, default Phase1/Phase2/Phase3 commands, output locations, Windows no-parallel-training rule, and the stop/fix failure gates.
- `validation`: Checked the new guide against [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), and [readme.md](d:/Evorig/readme.md). `git diff --check` passed for the documentation edits.
- `scope`: [docs/claude_code_repo_guide_zh.md](d:/Evorig/docs/claude_code_repo_guide_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this code commit`
- `message`: `configs: restore phase1 joint-boundary losses`
- `summary`: Audited the current Phase1/Phase2/Phase3 path after external artist-mesh runs showed many rest joints outside the mesh. The implementation still computes the joint-boundary losses, but the frozen Phase1 default had only JLG support losses plus posed-inside/rest-anchor enabled and omitted `loss_posed_joint_surface_clearance`, `loss_rest_joint_inside`, and `loss_rest_joint_surface_clearance`. Restored those three losses in the Phase1 default with the previously tested clearance ratios, and documented that Phase2 inherits them while Phase3 cannot repair bad rest joints because rest joints are locked.
- `validation`: Loaded [evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml) through the Phase1 loader and verified the restored keys are accepted by `Phase1Config`. `python -m py_compile` passed for the Phase1/Phase2/Phase3 config/trainer/refine code and runners. `git diff --check` passed.
- `scope`: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this code commit`
- `message`: `io: import unirig dynamic glb samples`
- `summary`: Added a UniRig dynamic-GLB importer for the current multi-dataset path. The importer bakes skeletal animation frames with Blender, aligns the baked dynamic rest to UniRig's rigged rest mesh, records correspondence/normalization gates, saves cleaned EvoRig sample files, and keeps UniRig skin weights only for fixed-LBS baselines. It also collapses zero-length direct bones created by external rig cleanup and remaps the saved skin-weight columns accordingly. Fixed PCJS masking so invalid shell directions cannot create `inf * 0` NaNs during Phase1 on open or artist meshes.
- `validation`: `python -m py_compile` passed for [blender_bake_dynamic_glb_frames.py](d:/Evorig/scripts/blender_bake_dynamic_glb_frames.py), [import_unirig_dynamic_glb_sample.py](d:/Evorig/scripts/import_unirig_dynamic_glb_sample.py), [unirig_dynamic.py](d:/Evorig/src/evorig_next/io/unirig_dynamic.py), and [phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py). Imported `yangtou` to [yangtou_key32_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/yangtou_key32_v1): correspondence p95/max relative `4.69e-05`, normalized bbox diagonal `2.0`, 32 selected frames, and cleaned 13-joint skeleton. A CUDA Phase1 20-step smoke completed without NaNs with `final_error_raw=0.092203`.
- `scope`: [scripts/blender_bake_dynamic_glb_frames.py](d:/Evorig/scripts/blender_bake_dynamic_glb_frames.py), [scripts/import_unirig_dynamic_glb_sample.py](d:/Evorig/scripts/import_unirig_dynamic_glb_sample.py), [src/evorig_next/io/unirig_dynamic.py](d:/Evorig/src/evorig_next/io/unirig_dynamic.py), [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this code commit`
- `message`: `phase2: dedupe seed joint repair across schedule`
- `summary`: Fixed the scheduled Phase2 seed-joint repair loop so the same seed joint cannot be repaired again in a later topology update. The per-update duplicate guard remains, and a new schedule-global repaired-joint set rejects repeated candidates after the first accepted repair.
- `validation`: `python -m py_compile` passed for [phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py) and [run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py). A no-training CUDA schedule smoke with `max_updates=3` / `topology_interval_steps=0` completed under [kangaroo_boxing_phase2_update_dedupe_smoke_round1](d:/Evorig/mygs/results/kangaroo_boxing_phase2_update_dedupe_smoke_round1/max3_interval0_seedrepair_dedupe); it accepted one seed repair for joint 21, no duplicate seed repairs, and completed three updates.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-04
- `commit`: `recorded in this documentation commit`
- `message`: `docs: reset active logs to evorig_next`
- `summary`: Cleaned active project documentation so the current route is unambiguously `evorig_next`. Rewrote [AGENTS.md](d:/Evorig/AGENTS.md), [readme.md](d:/Evorig/readme.md), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), and [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md). Cleared [docs/improvement_log_zh.md](d:/Evorig/docs/improvement_log_zh.md) to a historical-archive pointer and reset this journal to avoid carrying obsolete active context forward.
- `validation`: Checked the active entry documents no longer point to legacy accepted-line paths or names. `git diff --check` passed for the edited documentation files.
- `scope`: [AGENTS.md](d:/Evorig/AGENTS.md), [readme.md](d:/Evorig/readme.md), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/improvement_log_zh.md](d:/Evorig/docs/improvement_log_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-04
- `commit`: `recorded in this documentation commit`
- `message`: `docs: clarify evorig next fix line`
- `summary`: Clarified the current `evorig_next` fix-line naming. The base / camel fix line is Phase1 accepted plus Phase2 `fixed500_phase2_adaptive_max10_noopstop_round1`; the kangaroo current line is that base plus `fault_guided_seed_joint_repair center_capB`, the third topology strategy added during kangaroo work.
- `validation`: `rg` confirmed the new fix-line wording is present in [AGENTS.md](d:/Evorig/AGENTS.md), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), and [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md). `git diff --check` passed for the edited documentation files.
- `scope`: [AGENTS.md](d:/Evorig/AGENTS.md), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-04
- `commit`: `recorded in this documentation commit`
- `message`: `experiments: add joint loss forward selection`
- `summary`: Added a current-`evorig_next` Phase1 joint-control loss forward-selection runner. Ran the kangaroo Phase1 audit and generated [analysis_report.md](d:/Evorig/mygs/results/kangaroo_phase1_joint_loss_forward_select_round1/analysis_report.md), [selection_metrics.html](d:/Evorig/mygs/results/kangaroo_phase1_joint_loss_forward_select_round1/visuals/selection_metrics.html), and per-run joint-position visualizations. Current finding: `loss_rest_joint_anchor + loss_rest_joint_surface_clearance` is the best tested two-loss combination; adding `loss_posed_joint_inside` is the best tested three-loss option but increases fit error and branch seeds.
- `validation`: `python -m py_compile` passed for the new runner. Full CUDA run completed with worker-isolated candidates and regenerated summary/report/visualizations after adding the missing `posed_inside + rest_clearance` pair.
- `scope`: [scripts/run_evorig_next_phase1_joint_loss_forward_select_round1.py](d:/Evorig/scripts/run_evorig_next_phase1_joint_loss_forward_select_round1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: frame ablation metrics`
- `summary`: Updated the experiment section in [paper/main.tex](d:/Evorig/paper/main.tex) to align the EvoRig ablation metrics with the evaluation dimensions used by UniRig, AniGen, and AnimaMimic. Added a current ablation ledger that fills only accepted-line quadruped/camel values and leaves unmatched planned ablations blank instead of mixing in unrelated probe runs.
- `validation`: `pdflatex -interaction=nonstopmode -halt-on-error main.tex` completed successfully in [paper](d:/Evorig/paper). The run still reports expected unresolved citation/reference warnings because BibTeX was not rerun.
- `scope`: [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: tighten ablation number provenance`
- `summary`: Rechecked the ablation ledger values in [paper/main.tex](d:/Evorig/paper/main.tex). Removed frame mean/P95 columns that were only present in the paper snapshot, kept values traceable to saved run summaries/diagnostics, and changed the full Phase3 wrong-coverage count to the matched selected Phase3 diagnostic output instead of reusing the Phase2 count.
- `validation`: Verified the Phase1, Phase2, and selected Phase3 raw errors / topology counts from saved JSON outputs. `git diff --check` passed for the edited paper and journal files.
- `scope`: [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `experiments: fill main evorig ablations`
- `summary`: Added a current-`evorig_next` ablation runner for the main benchmark ledger and ran the fixed-LBS, fixed-topology, no-wrong-diagnosis, no-Phase3, no-directional-response, and full+JLG variants. Updated [paper/main.tex](d:/Evorig/paper/main.tex) with the matched values and switched the full benchmark row to the hard JLG-loss Phase3 line. The generated CSV/report are stored under the ignored result folder [evorig_next_main_ablation_round1](d:/Evorig/mygs/results/evorig_next_main_ablation_round1).
- `validation`: `python -m py_compile` passed for [run_evorig_next_main_ablation_round1.py](d:/Evorig/scripts/run_evorig_next_main_ablation_round1.py). The CUDA ablation runner completed and regenerated the ledger. `pdflatex -interaction=nonstopmode main.tex` completed in [paper](d:/Evorig/paper) with only existing citation/reference warnings from not rerunning BibTeX.
- `scope`: [scripts/run_evorig_next_main_ablation_round1.py](d:/Evorig/scripts/run_evorig_next_main_ablation_round1.py), [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: add benchmark protocol mapping`
- `summary`: Added a real benchmark protocol section to [paper/main.tex](d:/Evorig/paper/main.tex), mapping UniRig/AniGen skeleton and skin metrics, AniGen topology metrics, and AnimaMimic rendered-video metrics to EvoRig's no-GT mesh-sequence benchmark and optional GT/video tiers.
- `validation`: Extracted metric/protocol evidence from local PDFs under [paper/related_papers](d:/Evorig/paper/related_papers) using `pdftotext`. `pdflatex -interaction=nonstopmode main.tex` completed in [paper](d:/Evorig/paper) with only existing citation/reference warnings from not rerunning BibTeX.
- `scope`: [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: switch benchmark to no-gt metrics`
- `summary`: Reworked the benchmark metric selection in [paper/main.tex](d:/Evorig/paper/main.tex) so the main benchmark no longer depends on GT skeleton or GT skinning. Added a benchmark metric script that computes mean/P95 motion error, legal-support ratio, JLG faults, skinning entropy/top-k mass, and compactness from saved EvoRig outputs. Generated the ignored result report under [evorig_next_benchmark_metrics_round1](d:/Evorig/mygs/results/evorig_next_benchmark_metrics_round1).
- `validation`: `python -m py_compile` passed for [compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py). The metric script completed on the current saved runs. `pdflatex -interaction=nonstopmode main.tex` completed in [paper](d:/Evorig/paper) with only existing citation/reference warnings from not rerunning BibTeX.
- `scope`: [scripts/compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py), [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: fix coverage benchmark metric`
- `summary`: Corrected the no-GT benchmark coverage metric. The topology code's `uncovered_vertex_count` is an adaptive per-run 5% quantile seed count, so it is not a valid cross-run benchmark value. The metric script now reports zero-coverage counts and coverage quantiles instead, while keeping the adaptive uncovered count only in the CSV for debugging. Updated [paper/main.tex](d:/Evorig/paper/main.tex) to use Zero-cov and Cov-5% in the main benchmark table.
- `validation`: Re-ran [compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py) on the current camel/quadruped saved runs. `python -m py_compile` passed. `pdflatex -interaction=nonstopmode main.tex` completed in [paper](d:/Evorig/paper) with only existing citation/reference warnings from not rerunning BibTeX.
- `scope`: [scripts/compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py), [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: replace saturated top4 metric`
- `summary`: Replaced the main benchmark table's saturated Top4 skinning metric with effective influence count and Top2 mass. The metric script still writes Top4 to CSV for debugging, but the paper ledger now uses Entropy / Eff. / Top2 to better show whether a skinning field is diffuse or concentrated. Removed Joints/Bones/Gauss from the compact main table while leaving compactness counts in logs and result summaries.
- `validation`: Re-ran [compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py), which regenerated [analysis_report.md](d:/Evorig/mygs/results/evorig_next_benchmark_metrics_round1/analysis_report.md) with Top2 and effective-joint metrics. `pdflatex -interaction=nonstopmode main.tex` completed in [paper](d:/Evorig/paper) with only existing citation/reference warnings from not rerunning BibTeX.
- `scope`: [scripts/compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py), [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: clarify top2 is diagnostic`
- `summary`: Clarified that the main table's entropy, effective influence count, and Top2 mass are skinning-concentration diagnostics, not monotonic objectives. Removed the arrows from these columns so a small Top2 reduction after valid topology repair is not read as a direct failure.
- `validation`: Updated [paper/main.tex](d:/Evorig/paper/main.tex) only; the numerical table values are unchanged.
- `scope`: [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: restore top4 export metric`
- `summary`: Restored Top4 mass in the main benchmark table as the standard LBS four-influence export sanity metric, removed Top2/effective-influence columns from the paper ledger, and added Bad segment back as the geometric validity column. The metric report still writes entropy/effective/Top2/Top4 to CSV for debugging, but only Top4 is surfaced in the compact paper table.
- `validation`: Re-ran [compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py), `python -m py_compile` passed, and `pdflatex -interaction=nonstopmode main.tex` completed in [paper](d:/Evorig/paper) with only existing citation/reference warnings from not rerunning BibTeX.
- `scope`: [scripts/compute_evorig_next_benchmark_metrics_round1.py](d:/Evorig/scripts/compute_evorig_next_benchmark_metrics_round1.py), [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: name wrong coverage ablation precisely`
- `summary`: Renamed the wrong-coverage ablation in [paper/main.tex](d:/Evorig/paper/main.tex) from `w/o wrong-coverage diagnosis` to `w/o wrong-coverage branch seed`, matching the actual experiment where JLG cleanup losses stay active but wrong-covered components cannot seed branch growth.
- `validation`: `git diff --check` passed for the edited files.
- `scope`: [paper/main.tex](d:/Evorig/paper/main.tex), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: update dualtrack ablation draft`
- `summary`: Updated [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex) with a tighter abstract, inserted the JLG wrong-coverage cleanup mechanism figure in the Method section, and replaced the old quadruped snapshot table with the current no-GT ablation ledger using Mean/P95/Wrong/Zero-cov/Cov-5%/Legal/Top4/Bad-segment metrics.
- `validation`: Generated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf). Ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper); it completed with existing ACM/BibTeX/layout warnings but no LaTeX errors.
- `scope`: [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: compact jlg cleanup inset`
- `summary`: Converted the JLG wrong-coverage cleanup figure in [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex) from a two-column mechanism panel into a compact single-column inset. The regenerated figure removes the large warm background, global title, subtitle text, and bottom legend, leaving the local illegal joint, red wrong-covered vertices, sampled joint contributions, Gaussian support ellipses, and the retraction glyph.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf) and inspected the PNG preview. Ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper); it completed with existing ACM/BibTeX/layout warnings but no LaTeX errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: clarify jlg mechanism figure`
- `summary`: Expanded the single-column JLG method figure into a two-row mechanism inset. The top row now uses the camel tail wrong-coverage component to explain JLG as an invalid joint-to-surface link test; the bottom row keeps the kangaroo foot no-JLG / +JLG cleanup comparison for the loss effect. Updated the caption in [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex) accordingly.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the PNG preview, ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: redraw jlg figure as repeated cases`
- `summary`: Reworked the JLG method inset as two repeated local cases instead of a mixed concept/effect layout. The figure now uses a 2x2 structure: camel tail no-JLG / +JLG on the first row and kangaroo foot no-JLG / +JLG on the second row. Dense wrong-covered components are rendered with smaller, lower-alpha red points, while sampled vertices, joint links, Gaussian ellipses, and retraction glyphs remain explicit.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the PNG preview, ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: tighten jlg local-case view`
- `summary`: Tightened the JLG method inset camera to crop around the invalid joint-to-surface component instead of the full Gaussian footprint. The regenerated 2x2 figure keeps the camel tail and kangaroo foot repeated cases but makes the wrong-covered patch, sampled joint contributions, illegal link, and Gaussian support overlap visible in the local interaction region.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected both the standalone PNG and page-4 PDF preview, ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: fix camel tail jlg case`
- `summary`: Corrected the camel row in the JLG method inset to use the actual tail branch component (`topology_events[0]`) and the wrong-dominant illegal joint (`j3`) instead of the routed branch parent. The figure now shows the tail patch's illegal joint contribution dropping from high support to near zero, matching the intended wrong-coverage mechanism. The layout was also tightened vertically for a denser single-column method figure.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the standalone PNG and page-4 paper preview, ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: make jlg footprint illustrative`
- `summary`: Changed the JLG method inset from raw Gaussian covariance display to an explanatory effective-footprint visualization. The camel tail row now uses a fixed side-view projection so the tail remains visually recognizable, and the +JLG panels draw the cyan footprint as a deliberately retracted illegal-support region while preserving data-derived vertices, joints, and contribution labels. Updated the caption to describe cyan ellipses as effective illegal-support footprints.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the standalone PNG and page-4 paper preview, ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: widen camel tail jlg panel`
- `summary`: Adjusted the camel-tail JLG inset projection to use tail length as the horizontal axis and height as the vertical axis. This keeps the same event-0 tail patch and wrong-dominant joint but fills the single-column panel better, so the red wrong-covered tail vertices, sampled weights, and illustrative footprint retraction remain readable in the paper page.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the standalone PNG and page-4 paper preview, ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: redesign jlg cleanup mechanism figure`
- `summary`: Rebuilt the JLG method figure around a high-density mechanism layout: the kangaroo-foot wrong-coverage case is now the main before/after example, with explicit illegal-link, wrong-component, JLG-gate, footprint-retraction, and metric callouts. The camel tail remains as a compact repeated-case inset instead of a second full row.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the standalone preview and the page-4 paper preview for [evorig_dualtrack.pdf](d:/Evorig/paper/evorig_dualtrack.pdf), ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: simplify jlg cleanup figure`
- `summary`: Replaced the over-cluttered repeated-case JLG figure with a single local kangaroo-foot mechanism figure. The new figure uses two large before/after panels, a small JLG gate marker, one sampled joint contribution, and a compact metric strip for wrong vertices, mean joint contribution, and mean wrong ratio.
- `validation`: Regenerated [jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), inspected the standalone preview and the page-4 paper preview for [evorig_dualtrack.pdf](d:/Evorig/paper/evorig_dualtrack.pdf), ran `python -m py_compile` for [generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_wrong_joint_local_case_figure.py](d:/Evorig/scripts/generate_jlg_wrong_joint_local_case_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf](d:/Evorig/paper/figures/jlg_wrong_joint_local_case_kangaroo_foot.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-05
- `commit`: `recorded in this documentation commit`
- `message`: `paper: replace jlg figure with method schematic`
- `summary`: Replaced the prior local-result JLG figure with a fully schematic method diagram. The new figure explains JLG link legality, pre-normalized Gaussian support, valid and invalid joint-to-vertex links, invalid mass accumulation, and wrong-covered components without using a before/after cleanup layout.
- `validation`: Generated [jlg_method_diagnosis_schematic.pdf](d:/Evorig/paper/figures/jlg_method_diagnosis_schematic.pdf), inspected the standalone preview and the page-4 paper preview for [evorig_dualtrack.pdf](d:/Evorig/paper/evorig_dualtrack.pdf), ran `python -m py_compile` for [generate_jlg_method_schematic_figure.py](d:/Evorig/scripts/generate_jlg_method_schematic_figure.py), and ran `pdflatex -interaction=nonstopmode evorig_dualtrack.tex` twice in [paper](d:/Evorig/paper). LaTeX completed with existing ACM/BibTeX/layout warnings but no errors.
- `scope`: [scripts/generate_jlg_method_schematic_figure.py](d:/Evorig/scripts/generate_jlg_method_schematic_figure.py), [paper/evorig_dualtrack.tex](d:/Evorig/paper/evorig_dualtrack.tex), [paper/figures/jlg_method_diagnosis_schematic.pdf](d:/Evorig/paper/figures/jlg_method_diagnosis_schematic.pdf), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-07
- `commit`: `recorded in this documentation commit`
- `message`: `topology: guard branch parents by lineage`
- `summary`: Added Phase2 branch-lineage metadata and a lineage-aware parent-selection guard. Accepted branch growth now records the full joint chain, root parent, source component center/tip, and birth step. Later branch proposals reject branch-born parents unless the new component extends the whole existing branch polyline, preventing symmetric missing appendages from attaching to an already-grown appendage just because it is spatially close.
- `validation`: Ran `conda run -n mygs python -m py_compile src\evorig_next\phase2_topology.py src\evorig_next\phase1_trainer.py` and a constructed branch-lineage guard smoke test verifying extension is allowed, side/back attachment to an existing branch is rejected, and stable seed parents remain allowed.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-07
- `commit`: `recorded in this documentation commit`
- `message`: `topology: relax branch siblings by lineage`
- `summary`: Added lineage-sibling branch relaxation. Phase2 still uses the base global branch mass threshold (`0.03`), but when parent selection first rejects an existing branch as a side/cross attachment and then falls back to the stable ancestor, the candidate can use a relaxed global threshold (`0.015`). This targets parallel/symmetric appendages without globally admitting small leg/foot components.
- `validation`: Ran `python -m py_compile src\evorig_next\phase2_topology.py scripts\run_evorig_next_phase2_round1.py`. Reran kangaroo Phase2 in [phase2_full16_lineage_sibling_relax_schedule](d:/Evorig/mygs/results/kangaroo_boxing_phase2_lineage_guard_round1/phase2_full16_lineage_sibling_relax_schedule): accepted 4 branches, added the ear-side sibling component with `parent_selection=voxel_distance_parent_joint_lineage_guarded`, kept joint/bone/Gaussian counts to `36/35/651`, and produced final raw error `0.009362894`.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `phase2: skip duplicate topology signal work`
- `summary`: Reduced scheduled Phase2 wall time without changing topology results. The runner now skips the initial resume-time topology signal export when a scheduled Phase2 run will immediately recompute update signals, and it reuses the schedule's final signal/checkpoint instead of recomputing them in the entry wrapper. Split checks now use a split-only signal builder, avoiding branch component and voxel path recomputation before split selection. The mesh voxel path field is cached per trainer/config.
- `validation`: Ran `conda run -n mygs python -m py_compile src\evorig_next\phase2_topology.py scripts\run_evorig_next_phase2_round1.py`. Reran kangaroo scheduled Phase2 max3 in [phase2_full16_lineage_sibling_relax_schedule_max3_fast](d:/Evorig/mygs/results/kangaroo_boxing_phase2_speedcheck_round1/phase2_full16_lineage_sibling_relax_schedule_max3_fast): event sequence remained four branches, update errors remained `0.010853811`, `0.009711363`, `0.009362894`, final counts stayed `36/35/651`, final raw error stayed `0.009362894`, and entry wall time dropped from `802.3s` to `489.8s`.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `phase2: freeze fast default config`
- `summary`: Added the frozen Phase2 default config for the current lineage-sibling fast schedule. Also skipped the duplicate initial signal export for checkpoint-resume scheduled runs and reused the final no-op update signal as the root-level final topology signal, avoiding one more full branch/voxel recomputation at default `max_updates=4`.
- `validation`: Ran `conda run -n mygs python -m py_compile src\evorig_next\phase2_topology.py scripts\run_evorig_next_phase2_round1.py`. Reran the default kangaroo schedule in [phase2_full16_lineage_sibling_relax_schedule_default_fast_reuse](d:/Evorig/mygs/results/kangaroo_boxing_phase2_default_fast_round2/phase2_full16_lineage_sibling_relax_schedule_default_fast_reuse): completed 4 updates with noop stop, kept event sequence as four branches, kept final counts `36/35/651`, kept final raw error `0.009362894`, and produced entry wall time `517.0s`.
- `scope`: [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `phase1: enable jlg support default`
- `summary`: Added the frozen Phase1 support-loss default and made the Phase1/Phase2 entry load Phase1 config from `--phase1-config`. The default now preserves the accepted 500-step schedule while enabling `loss_illegal_support=0.20`, `loss_gaussian_illegal_coverage=0.05`, and `illegal_support_tau=0.05`; the old no-JLG accepted config remains available only as a comparison baseline.
- `validation`: Ran `python -m py_compile` in the `mygs` environment for the Phase1/Phase2/Phase3 entry scripts. Loaded the new default config and verified it reports `500 0.2 0.05 0.05`. Ran an 8-step kangaroo Phase1 timing smoke: no-JLG took `0.3195s/step`, JLG-on took `0.4448s/step`, and the JLG metrics were nonzero (`illegal_support=0.02948`, `gaussian_illegal_coverage=0.000426`).
- `scope`: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `docs: align phase1 default note`
- `summary`: Updated the top-level current-system Phase1 section so it no longer points at the old no-JLG accepted config as the active default. The active Phase1 default now consistently lists the support-loss config and the `0.20/0.05/0.05` JLG settings.
- `validation`: Inspected the resulting diff for [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md).
- `scope`: [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `phase1: share posed joint shells`
- `summary`: Added a shared posed-joint shell descriptor path for Phase1. When `loss_pcjs` and `loss_posed_joint_inside` are both enabled, Phase1 now computes the per-frame mesh shell descriptors once and reuses them in both losses, removing duplicate ray/shell queries while preserving the same gradients to posed joints.
- `validation`: Ran `python -m py_compile` for [phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), and [run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py). Verified direct loss equality on kangaroo (`pcjs_abs_diff=0`, `inside_abs_diff=0`). Verified all three loss toggles run (`pcjs_off`, `inside_off`, `both_on`). Profiling showed `posed_joint_inside_mesh_loss` dropping from about `0.042s/step` to `0.0047s/step`; uninstrumented JLG-on 8-step smoke ran at `0.259s/step`.
- `scope`: [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `phase1: trim shell reuse overhead`
- `summary`: Extended the shared posed-joint shell descriptor to carry the already computed per-joint sample directions, so `loss_pcjs` does not recompute the same direction transform after descriptor reuse. Also removed an unused `legal_support_mass` computation from `train_step`.
- `validation`: Ran `python -m py_compile` for [phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py) and [phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py). Rechecked kangaroo direct loss equality (`pcjs_abs_diff=0`, `inside_abs_diff=0`) and confirmed descriptors carry directions. Uninstrumented 8-step JLG-on smoke ran at `0.2556s/step` with unchanged loss values for the checked step.
- `scope`: [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation commit`
- `message`: `phase3: skip frozen-only work`
- `summary`: Added a default Phase3 speed path that skips losses whose gradients only target frozen base/rest-joint/scale parameters, while preserving JLG support losses because they can still shape trainable SH response. `train_step` now avoids PCJS and posed-inside shell queries when their weights are zero. Final Phase3 topology-signal export is diagnostic-only and is disabled by default; `--save-final-topology-signals` restores the old diagnostic export.
- `validation`: Ran `python -m py_compile` for [phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [phase3_refine.py](d:/Evorig/src/evorig_next/phase3_refine.py), and [run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py). On the kangaroo Phase2 checkpoint, 20-step old/new comparison produced identical SH, offsets, rest joints, weights, and predicted vertices (`max_abs=0`) while step time improved from `0.3339s` to `0.2769s`. A 2-step Phase3 runner smoke with final signals disabled completed with final raw error `0.009360374` and internal wall time `4.53s`; the same path with final signal recomputation took about `93s`.
- `scope`: [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [src/evorig_next/phase3_refine.py](d:/Evorig/src/evorig_next/phase3_refine.py), [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this cleanup commit`
- `message`: `cleanup: lock evorig_next defaults`
- `summary`: Locked the active default path to `evorig_next`: Phase1 uses the support-loss default config, Phase2 uses the lineage-sibling fast topology schedule, and Phase3 uses the locked-joint refinement default. Migrated the remaining shared utility dependencies from old `evorig` into `src/evorig_next`, updated active runners to import only `evorig_next`, and removed legacy `src/evorig`, `src/evorig2`, `src/evorig3`, old frozen configs, and obsolete sweep/probe scripts from the default workspace.
- `validation`: `python -m py_compile` passed for all `src/evorig_next` modules and retained scripts. Phase1/Phase2/Phase3 runner `--help` smoke passed. Phase3 zero-step CUDA import smoke from the kangaroo Phase2 checkpoint completed with 36 joints, 35 bones, and 651 Gaussians. `rg` found no old `evorig.*`, `evorig2`, or `evorig3` imports under `src`, `scripts`, and `configs`; `src` contains only `evorig_next`; `configs/frozen` contains only the four current default configs.
- `scope`: [src/evorig_next](d:/Evorig/src/evorig_next), [scripts](d:/Evorig/scripts), [configs/frozen](d:/Evorig/configs/frozen), [readme.md](d:/Evorig/readme.md), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/architecture_zh.md](d:/Evorig/docs/architecture_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this documentation cleanup commit`
- `message`: `docs: refresh evorig mainline logs`
- `summary`: Refreshed the active documentation/log surface after the code cleanup. The docs now describe EvoRig as the single mainline while keeping `src/evorig_next` as the current implementation package name. Updated AGENTS/readme/current-system/current-workline/architecture/multi-dataset docs, added experiment guardrails, replaced stale Phase1 repair notes, and restored the UniRig dynamic-mesh processing protocol as an active gate-before-training workflow.
- `validation`: `python -m py_compile` passed for all `src/evorig_next` modules and retained scripts. Phase1/Phase2/Phase3 runner `--help` smoke passed. `rg` found no old `evorig.*`, `evorig2`, or `evorig3` imports under `src`, `scripts`, and `configs`. `git diff --check` passed.
- `scope`: [AGENTS.md](d:/Evorig/AGENTS.md), [readme.md](d:/Evorig/readme.md), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/multi_dataset_protocol_zh.md](d:/Evorig/docs/multi_dataset_protocol_zh.md), [docs/architecture_zh.md](d:/Evorig/docs/architecture_zh.md), [docs/next_step_plan_zh.md](d:/Evorig/docs/next_step_plan_zh.md), [docs/improvement_log_zh.md](d:/Evorig/docs/improvement_log_zh.md), [docs/method_siggraph_zh.md](d:/Evorig/docs/method_siggraph_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this metadata cleanup commit`
- `message`: `configs: remove stale phase1 source run`
- `summary`: Removed the old no-JLG accepted-run path from the Phase1 support-loss default config metadata. The config now records the base initialization config and current JLG default deltas without pointing readers back to a deprecated result directory.
- `validation`: `python -m py_compile` passed for all `src/evorig_next` modules and retained scripts. Phase1/Phase2/Phase3 runner `--help` smoke passed. `git diff --check` passed.
- `scope`: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this default-route alignment commit`
- `message`: `phase2: restore seed joint repair default`
- `summary`: Aligned the executable defaults with the final active route. Phase2 now includes `fault_guided_seed_joint_repair center_capB` in the scheduled topology loop before branch/split. The repair is not joint-id hardcoded: it only moves an existing one-child seed joint when a fault component is dominated by that joint's parent/self/child neighborhood and the capped relocation improves adjacent segment inside fractions. Also made Phase2 schedule and Phase2/Phase3 viewer skipping match the frozen default configs.
- `validation`: `python -m py_compile` passed for all `src/evorig_next` modules and retained scripts. Phase1/Phase2/Phase3 runner `--help` smoke passed. A kangaroo Phase2 signal smoke from `phase2_full16_default_schedule` found the two expected seed-joint repair candidates, joint `21` and joint `17`, with `capB=0.177550392`, and a one-update zero-training schedule smoke accepted both repairs with adjacent segment inside fractions improving to `1.0/1.0`. `rg` found no old `evorig.*`, `evorig2`, or `evorig3` imports under `src`, `scripts`, and `configs`; `git diff --check` passed.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py), [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/architecture_zh.md](d:/Evorig/docs/architecture_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment record commit`
- `message`: `exp: run double_knife unirig sample`
- `summary`: Completed the second UniRig dynamic-mesh sample on the active EvoRig mainline. `double_knife` had a valid UniRig skeleton from the batch run but was missing skin/merge outputs, so the skin and merge stages were resumed without rerunning skeleton. Alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min `1.0`). The imported EvoRig sample [double_knife_key32_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/double_knife_key32_v1) uses 32 selected frames from 250 source frames, 2321 vertices, normalized bbox diagonal `2.0`, 27 cleaned joints, and dynamic-to-rest correspondence max relative error about `1.5e-7`.
- `validation`: Ran default Phase1/Phase2/Phase3 on CUDA. Phase1 [phase1_full32_default500](d:/Evorig/mygs/results/double_knife_phase1_round1/phase1_full32_default500) finished with raw error `0.010517786`, raw-all error `0.017717626`, zero-weight vertices `0`, 27 joints, and 564 Gaussians; Phase1 topology signals reported 5 branch components and 2 split candidates. Phase2 [phase2_default](d:/Evorig/mygs/results/double_knife_phase2_round1/phase2_default) completed 4 scheduled updates, accepted 6 branch events, no seed-joint repair or split, and ended with raw error `0.009702377`, 51 joints, 50 bones, 756 Gaussians, and no remaining branch components. Phase3 [phase3_locked_default200](d:/Evorig/mygs/results/double_knife_phase3_round1/phase3_locked_default200) completed 200 locked-joint steps with raw error `0.009647235`, raw-all error `0.009578353`, zero-weight vertices `0`, and max rest-joint displacement ratio `0.0`.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment record commit`
- `message`: `exp: run gray_rabbit unirig sample`
- `summary`: Completed the `gray_rabbit` UniRig dynamic-mesh sample on the active default route. The batch skeleton stage returned a cuda-guard stderr failure code but produced a valid `skeleton.fbx`; skin, merge, and alignment were resumed manually. Alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min `1.0`). The imported sample [gray_rabbit_key32_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/gray_rabbit_key32_v1) uses 32 selected frames from 250 source frames, 4380 rest vertices, 4364 baked dynamic vertices, 11 cleaned joints, normalized bbox diagonal `2.0`, and dynamic-to-rest max relative correspondence error about `5.2e-4`. This asset has many duplicated/split vertices: nearest mapping uses 1049 unique baked source vertices, so later comparisons should treat it as a valid but seam-heavy sample.
- `validation`: Ran default Phase1/Phase2/Phase3 on CUDA. Phase1 [phase1_full32_default500](d:/Evorig/mygs/results/gray_rabbit_phase1_round1/phase1_full32_default500) finished with raw error `0.013957298`, raw-all error `0.015018524`, zero-weight vertices `0`, 11 joints, and 540 Gaussians; topology filtering reported no branch components and no split candidates. Phase2 [phase2_default](d:/Evorig/mygs/results/gray_rabbit_phase2_round1/phase2_default) completed 2 scheduled no-op updates, accepted no topology events, and ended with raw error `0.013033650`, 11 joints, 10 bones, and 540 Gaussians. Phase3 [phase3_locked_default200](d:/Evorig/mygs/results/gray_rabbit_phase3_round1/phase3_locked_default200) completed 200 locked-joint steps with raw error `0.013265847`, raw-all error `0.016068999`, zero-weight vertices `0`, and max rest-joint displacement ratio `0.0`.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment record commit`
- `message`: `exp: gate bababoyi unirig sample`
- `summary`: Ran UniRig extraction/skeleton/skin/merge/alignment for `bababoyi`. As with the other UniRig samples, the batch skeleton stage returned a cuda-guard stderr failure code after writing `skeleton.fbx`, so skin/merge/alignment were resumed manually. Alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min `1.0`). The sample was not trained because the EvoRig import gate failed in `_preprocess_real_rig`: root joint `0` was outside the mesh. This is a valid current skip, not a training failure; external-root or trajectory-root handling needs a separate protocol before this asset can be used.
- `validation`: `bababoyi` UniRig outputs exist under [E:/evorig_unirig/bababoyi](E:/evorig_unirig/bababoyi) with `mesh.glb`, `skeleton.fbx`, `skin.fbx`, `rigged.glb`, and `alignment_report.json`. Import stopped before sample construction with `ValueError: real rig preprocess found root joint(s) outside mesh: [0]`.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment record commit`
- `message`: `exp: run eagle unirig sample`
- `summary`: Completed the `eagle` UniRig dynamic-mesh sample on the active default route. UniRig alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min `1.0`). The imported sample [eagle_key32_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/eagle_key32_v1) uses 32 selected frames from 250 source frames, 2662 rest vertices, 2637 baked dynamic vertices, 45 cleaned joints from 63 initial UniRig joints, normalized bbox diagonal `2.0`, and dynamic-to-rest max relative correspondence error about `9.35e-4`.
- `validation`: Ran default Phase1/Phase2/Phase3 on CUDA. Phase1 [phase1_full32_default500](d:/Evorig/mygs/results/eagle_phase1_round1/phase1_full32_default500) finished with raw error `0.011643843`, raw-all error `0.012032908`, zero-weight vertices `0`, 45 joints, and 672 Gaussians; Phase1 topology signals reported 1 branch component and 2 split candidates. Phase2 [phase2_default](d:/Evorig/mygs/results/eagle_phase2_round1/phase2_default) completed 4 scheduled updates, accepted 4 branches and 1 split, and ended with raw error `0.009255397`, raw-all error `0.009380364`, 66 joints, 65 bones, and 840 Gaussians. Phase3 [phase3_locked_default200](d:/Evorig/mygs/results/eagle_phase3_round1/phase3_locked_default200) completed 200 locked-joint steps with raw error `0.009262525`, raw-all error `0.009385217`, zero-weight vertices `0`, and max rest-joint displacement ratio `0.0`.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment/fault-tolerance commit`
- `message`: `exp: run split_fish unirig sample`
- `summary`: Completed the large `split_fish` UniRig dynamic-mesh sample through Phase3 with an explicit Phase2 fallback. The imported sample [split_fish_key16_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/split_fish_key16_v1) uses 16 selected frames from 250 source frames, 59742 rest vertices, 55048 baked dynamic vertices, 17 cleaned joints, normalized bbox diagonal `2.0`, and dynamic-to-rest max relative correspondence error about `3.61e-3`. The mesh is a large multi-primitive/seam-heavy case, so it is a stress sample rather than a clean default-quality asset.
- `validation`: Phase1 [phase1_full16_default500](d:/Evorig/mygs/results/split_fish_phase1_round1/phase1_full16_default500) completed with raw error `0.004184463`, raw-all error `0.004614348`, zero-weight vertices `0`, 17 joints, and 564 Gaussians; topology signals reported 2 branch components and no split candidates. Repeated standard Phase2 attempts completed the first branch update but the Python process exited natively during repeated topology-signal construction, with no traceback and no final checkpoint. To make the large sample usable without changing topology acceptance logic, Phase2 [phase2_saved_signal_branch1](d:/Evorig/mygs/results/split_fish_phase2_fallback_round1/phase2_saved_signal_branch1) reused the already exported Phase1 topology signal, accepted the same first branch, ran the standard 100-step interval, and saved a checkpoint with raw error `0.004212265`, raw-all error `0.004448216`, 22 joints, 21 bones, and 604 Gaussians. Phase3 [phase3_locked_default200](d:/Evorig/mygs/results/split_fish_phase3_round1/phase3_locked_default200) completed 200 locked-joint steps with raw error `0.004188915`, raw-all error `0.004439245`, zero-weight vertices `0`, and max rest-joint displacement ratio `0.0`. Added a Phase2 per-update partial checkpoint write so future large-mesh native exits after a completed update do not discard the latest valid topology state; normal completed runs still overwrite it with the final checkpoint.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment record commit`
- `message`: `exp: run dinosaur unirig sample`
- `summary`: Completed the `dinosaur` UniRig dynamic-mesh sample on the active EvoRig mainline. UniRig extraction produced valid `mesh.glb`, `skeleton.fbx`, `skin.fbx`, and `rigged.glb`; alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min `1.0`). The imported sample [dinosaur_key32_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/dinosaur_key32_v1) uses 32 selected frames from 250 source frames, 7043 rest vertices, 7036 baked dynamic vertices, 28 cleaned joints from 29 initial UniRig joints, normalized bbox diagonal `2.0`, and dynamic-to-rest max relative correspondence error about `5.07e-5`.
- `validation`: Ran default Phase1/Phase2/Phase3 on CUDA. Phase1 [phase1_full32_default500](d:/Evorig/mygs/results/dinosaur_phase1_round1/phase1_full32_default500) completed but its final topology-signal export was interrupted by an external GPU process killer; the exported state was repaired by recomputing the Phase2 topology signals from the Phase1 checkpoint, yielding zero-weight vertices `0`, 28 joints, 564 Gaussians, 2 branch components, 1 seed-joint repair candidate, and 3 split candidates. Phase2 [phase2_default](d:/Evorig/mygs/results/dinosaur_phase2_round5/phase2_default) completed 4 scheduled updates, accepted 10 topology events, and ended with raw error `0.018701941`, raw-all error `0.022193497`, zero-weight vertices `0`, 42 joints, 41 bones, and 676 Gaussians. Phase3 [phase3_locked_default200](d:/Evorig/mygs/results/dinosaur_phase3_round1/phase3_locked_default200) completed 200 locked-joint steps with raw error `0.018662345`, raw-all error `0.022163777`, zero-weight vertices `0`, 42 joints, 41 bones, and 676 Gaussians.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment gate commit`
- `message`: `exp: gate spring_airplane unirig sample`
- `summary`: Ran UniRig extraction/skeleton/skin/merge/alignment for `spring_airplane`. UniRig outputs were produced under [E:/evorig_unirig/spring_airplane](E:/evorig_unirig/spring_airplane), and alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min about `0.99999998`). The asset was not trained because the EvoRig import gate failed in `_preprocess_real_rig`: root joint `0` lies outside the normalized mesh, and inspection showed many additional wing/tail joints outside the volume predicate. This is a thin-shell/external-root sample and should wait for a separate thin-asset or external-root protocol rather than being forced through the current volume-JLG route.
- `validation`: `spring_airplane` has valid `mesh.glb`, `skeleton.fbx`, `skin.fbx`, `rigged.glb`, and `alignment_report.json`. Import stopped before sample construction with `ValueError: real rig preprocess found root joint(s) outside mesh: [0]`. A diagnostic check found 27 UniRig joints, with outside joints `[0, 5, 6, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25]` under the default normalized surface tolerance.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `recorded in this experiment record commit`
- `message`: `exp: run zombie unirig sample`
- `summary`: Completed the `zombie` UniRig dynamic-mesh sample on the active EvoRig mainline. UniRig outputs were produced under [E:/evorig_unirig/zombie](E:/evorig_unirig/zombie), and alignment passed exactly (`center_delta_relative=0`, `extent_delta_relative=0`, PCA diagonal min about `1.0`). The imported sample [zombie_key32_v1](d:/Evorig/mygs/demo_data/evorig_unirig_keyframes_round2/zombie_key32_v1) uses 32 selected frames from 250 source frames, 13686 rest vertices, 13585 baked dynamic vertices, 20 cleaned UniRig joints, normalized bbox diagonal `2.0`, and dynamic-to-rest max relative correspondence error about `8.51e-4`.
- `validation`: Ran default Phase1/Phase2/Phase3 on CUDA. Phase1 [phase1_full32_default500](d:/Evorig/mygs/results/zombie_phase1_round1/phase1_full32_default500) finished with raw error `0.048354130`, raw-all error `0.052070629`, zero-weight vertices `0`, 20 joints, and 564 Gaussians; topology signals reported 4 branch components and 1 split candidate. Phase2 [phase2_default](d:/Evorig/mygs/results/zombie_phase2_round1/phase2_default) completed 4 scheduled updates, accepted 9 topology events, and ended with raw error `0.022602307`, raw-all error `0.022096988`, zero-weight vertices `0`, 52 joints, 51 bones, and 820 Gaussians. Phase3 [phase3_locked_default200](d:/Evorig/mygs/results/zombie_phase3_round1/phase3_locked_default200) completed 200 locked-joint steps with raw error `0.022534277`, raw-all error `0.022035137`, zero-weight vertices `0`, 52 joints, 51 bones, 820 Gaussians, and max rest-joint displacement ratio `0.0`.
- `scope`: [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)
- `date`: 2026-05-09
- `commit`: `this commit`
- `message`: `configs: lock final experiment defaults`
- `summary`: Locked the final experiment protocol before the main result runs. Phase1 keeps JLG support losses `0.20/0.05` plus joint-boundary losses. Phase2 keeps the lineage-sibling fast schedule with `seed_joint_repair center_capB`, branch, and split. Phase3 now defaults to the selected visual-quality legality line: rest joints locked, base params unfrozen, Gaussian offsets enabled for all Gaussians from step 0, `SH16`, and JLG support losses `0.20/0.05`. JLG-off Phase3 is documented as diagnostic-only because it can lower reconstruction loss by using illegal support.
- `validation`: `python -m py_compile` passed for the Phase1/Phase2/Phase3 runners and `phase3_refine.py`. A direct default check verified Phase3 now defaults to `SH16`, base-param unfreeze, all-Gaussian offsets from step 0, JLG `0.20/0.05/0.05`, and still rejects `--unfreeze-rest-joints`.
- `scope`: [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py), [configs/frozen/evorig_next_phase3_locked_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase3_locked_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)
- `date`: 2026-05-09
- `commit`: `this commit`
- `message`: `docs: clarify all-phase JLG defaults`
- `summary`: Clarified that JLG support losses are not phase-specific. Phase1, Phase2, and Phase3 all keep `loss_illegal_support=0.20`, `loss_gaussian_illegal_coverage=0.05`, and `illegal_support_tau=0.05` as part of the default rigging objective. Updated the workline, system summary, guardrails, and readme; also removed the stale Phase3 text that still described Gaussian offsets as disabled.
- `validation`: Grepped the frozen configs and docs to verify the three JLG keys are present in Phase1/Phase2/Phase3 defaults and that the stale `Gaussian offset 默认关闭` text is gone. `git diff --check` passed.
- `scope`: [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [readme.md](d:/Evorig/readme.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `this commit`
- `message`: `phase1: add fixed-rest sampling protocol`
- `summary`: Added windowed UniRig dynamic-frame import, Phase1 trace snapshots, and Phase1 resume support. The default Phase1 config is now Phase1A fixed-rest: mesh-centroid root-translation initialization, root rest joint frozen, rest-joint optimization disabled, and 50-step trace output. Added a separate Phase1B rest-refine config that resumes from Phase1A, does not densify, keeps SH/lambda active from step 1, and only refines non-root rest joints with a small learning rate and stronger anchor. Fixed two protocol bugs found during dinosaur testing: window selection could exceed `max_frames` by one frame, and empty JSON `densify_stages: []` was previously ignored and replaced by defaults.
- `validation`: Imported dinosaur `front60` and `front100_to60` samples from [E:/evorig_unirig/dinosaur](E:/evorig_unirig/dinosaur). Phase1A fixed-rest selected `front60`: [phase1a_front60_fixedrest600_selected](d:/Evorig/mygs/results/dinosaur_phase1ab_sampling_round1/phase1a_front60_fixedrest600_selected) reached raw-all `0.029706`, legal raw `0.018453`, JLG illegal coverage `0.002105`, root rest drift `0`, and rest drift max `0`. Phase1B [phase1b_front60_restrefine200_lr0006_anchor1](d:/Evorig/mygs/results/dinosaur_phase1ab_sampling_round1/phase1b_front60_restrefine200_lr0006_anchor1) improved raw-all to `0.024227`, legal raw to `0.015244`, JLG illegal coverage to `0.001709`, kept root rest drift `0`, and reduced outside active Gaussians to `3`. Viewers and a compact report were written under [dinosaur_phase1ab_sampling_round1](d:/Evorig/mygs/results/dinosaur_phase1ab_sampling_round1). `python -m py_compile` and `git diff --check` passed for the edited code/config/docs.
- `scope`: [src/evorig_next/io/unirig_dynamic.py](d:/Evorig/src/evorig_next/io/unirig_dynamic.py), [src/evorig_next/phase1_config.py](d:/Evorig/src/evorig_next/phase1_config.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [scripts/import_unirig_dynamic_glb_sample.py](d:/Evorig/scripts/import_unirig_dynamic_glb_sample.py), [scripts/run_evorig_next_phase1.py](d:/Evorig/scripts/run_evorig_next_phase1.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/multi_dataset_protocol_zh.md](d:/Evorig/docs/multi_dataset_protocol_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `this commit`
- `message`: `phase2: guard branch path inside mesh`
- `summary`: Added a final parent-to-path segment-inside acceptance guard for Phase2 branch proposals. The previous branch path construction could diagnose a bad root tangent candidate but still apply the unsmoothed path; on the dinosaur `front60` Phase1B state this accepted a branch with segment inside fraction `0.122`. The new guard records `branch_path_inside` for every branch component and rejects branch application when the minimum final segment inside fraction is below `0.70`.
- `validation`: Re-ran Phase2 from [phase1b_front60_restrefine200_lr0006_anchor1](d:/Evorig/mygs/results/dinosaur_phase1ab_sampling_round1/phase1b_front60_restrefine200_lr0006_anchor1). The unguarded run [phase2_default_from_front60_phase1b](d:/Evorig/mygs/results/dinosaur_phase2_from_phase1b_round1/phase2_default_from_front60_phase1b) accepted 7 branches and produced one rest joint outside plus a bad bone `28->29` with inside fraction `0.122`. The guarded run [phase2_default_pathinside_guard_from_front60_phase1b](d:/Evorig/mygs/results/dinosaur_phase2_from_phase1b_round1/phase2_default_pathinside_guard_from_front60_phase1b) accepted 3 splits, no branches, reached raw legal error `0.014529`, kept zero-weight vertices `0`, rest joint outside count `0`, and minimum rest bone inside fraction `0.976`. Motion and final-topology viewers were generated in its `visuals/interactive` directory. `python -m py_compile` passed for Phase2 code.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-09
- `commit`: `this commit`
- `message`: `phase1: add fast-motion acceleration loss`
- `summary`: Added optional `loss_vertex_acceleration` for fast external dynamic meshes. The existing temporal smoothness loss regularizes pose/root velocity, but dinosaur front60 showed visible high-frequency mesh shaking because the objective did not directly match the GT mesh acceleration. The new loss compares predicted and GT vertex second differences over the full sequence, masked to legal vertices, and is off by default so the established camel/kangaroo default route is unchanged.
- `validation`: Re-ran a Phase1B continuation from [phase1b_front60_restrefine200_lr0006_anchor1](d:/Evorig/mygs/results/dinosaur_phase1ab_sampling_round1/phase1b_front60_restrefine200_lr0006_anchor1) with `loss_vertex_acceleration=1.0` for 150 steps at [phase1b_accel_smooth_from_b_150](d:/Evorig/mygs/results/dinosaur_phase1ab_sampling_round1/phase1b_accel_smooth_from_b_150). Mesh acceleration ratio dropped from `3.316x` GT to `0.981x`, acceleration residual mean dropped from `0.026796` to `0.003392`, legal raw error improved from `0.015244` to `0.011348`, and rest-joint outside count stayed `0`. Motion and final-topology viewers were regenerated, and `python -m py_compile` plus `git diff --check` passed.
- `scope`: [src/evorig_next/training/losses.py](d:/Evorig/src/evorig_next/training/losses.py), [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [src/evorig_next/phase1_config.py](d:/Evorig/src/evorig_next/phase1_config.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: cache mesh adjacency and add abc runner`
- `summary`: Added a disk-backed rest-mesh adjacency cache keyed by the same mesh-topology hash used for k-ring JLG propagation. Phase1 now reuses the per-vertex neighbor list across stages and processes for the same mesh instead of rebuilding it every run. Added a `--skip-topology-signals` switch for intermediate Phase1 runs and a single-process `run_evorig_next_phase1abc.py` helper for the dinosaur-style A/B/C protocol, with A/B topology signal export disabled by default and final C export kept.
- `validation`: `python -m py_compile` passed for `phase1_trainer.py`, `run_evorig_next_phase1.py`, and `run_evorig_next_phase1abc.py`. `run_evorig_next_phase1abc.py --help` works, and `git diff --check` reports no whitespace errors for the edited files.
- `scope`: [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [scripts/run_evorig_next_phase1.py](d:/Evorig/scripts/run_evorig_next_phase1.py), [scripts/run_evorig_next_phase1abc.py](d:/Evorig/scripts/run_evorig_next_phase1abc.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: stream training progress`
- `summary`: Added real-time Phase1 optimization visibility. `Phase1Trainer.run` now shows a `tqdm` progress bar with current loss/reconstruction/zero-weight count at trace intervals and writes `phase1_trace_live.jsonl` incrementally during training, instead of only writing `phase1_trace.json` after completion.
- `validation`: `python -m py_compile` passed for `phase1_trainer.py`, `run_evorig_next_phase1.py`, and `run_evorig_next_phase1abc.py`. The active `mygs` environment has `tqdm 4.67.1` installed.
- `scope`: [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `configs: set phase1 abc mainline`
- `summary`: Promoted the dinosaur/front60 Phase1 protocol to the current default mainline: Phase1A fixed-rest `800` steps with `frame_batch_size=32`, Phase1B rest-joint refine `200` steps with `frame_batch_size=32`, and new Phase1C smooth `100` steps that keeps rest-joint refine active while enabling `loss_vertex_acceleration=1.0` and `loss_temporal_smoothness=0.10`. The historical Phase1A filename is kept for compatibility, but its contents now represent the A800 default. The single-process `run_evorig_next_phase1abc.py` runner now defaults to the frozen A/B/C configs instead of experiment-local JSON files.
- `validation`: On [dinosaur_phase1bc_probe_round1](d:/Evorig/mygs/results/dinosaur_phase1bc_probe_round1), A500/bs32 reached legal raw `0.015013`; B400 improved to `0.010917`; B400->C200 improved to `0.009497`, while BC600 only reached `0.010472`, supporting staged B then C. Config loading verified A=`800/bs32`, B=`200/bs32`, C=`100/bs32`; `python -m py_compile` passed for the Phase1 runners/trainer/config, `run_evorig_next_phase1abc.py --help` works, and `git diff --check` reports no whitespace errors.
- `scope`: [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml), [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml), [scripts/run_evorig_next_phase1abc.py](d:/Evorig/scripts/run_evorig_next_phase1abc.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/multi_dataset_protocol_zh.md](d:/Evorig/docs/multi_dataset_protocol_zh.md), [readme.md](d:/Evorig/readme.md), [AGENTS.md](d:/Evorig/AGENTS.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: add joint cross-section containment loss`
- `summary`: Added an optional posed joint cross-section containment loss. Instead of sampling 12 fixed shell rays, the new loss builds a rest-pose local mesh cross-section near each joint-to-child segment at `lambda=0.1` (leaf joints use the leaf point), tracks the same vertices in each posed GT mesh, and penalizes the posed joint-side probe only when it leaves the section radius. The default weight is `0.0`, so the current Phase1/Phase2/Phase3 mainline is unchanged.
- `validation`: `python -m py_compile` passed for `phase1_config.py`, `phase1_losses.py`, and `phase1_trainer.py`. A dinosaur/front60 C100 probe from the B400 checkpoint showed that the historical C config with `loss_posed_joint_cross_section_inside=0.01`, center weight `0`, boundary weight `1`, and radius floor `0.03` preserved reconstruction (`raw legal 0.009595` vs baseline `0.009595`, zero-weight vertices `0`) while reducing cross-section excess (`max normalized outside 1.693 -> 1.526`, `mean_excess 0.01595 -> 0.01071`). Viewers were written under [dinosaur_cross_section_loss_probe_round1/viewers](d:/Evorig/mygs/results/dinosaur_cross_section_loss_probe_round1/viewers).
- `scope`: [src/evorig_next/phase1_config.py](d:/Evorig/src/evorig_next/phase1_config.py), [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: restore motion root and ring scale`
- `summary`: Restored the Phase1 virtual motion root and cross-section ring scale initialization. `root_trans` remains the global translation controller initialized from mesh centroid motion, but `separate_motion_root=true` prevents `freeze_root_rest_joint` from freezing the anatomical parent=-1 joint, so joint0 can refine in Phase1B/C. Phase1A now uses `phase1_scale_formula=cross_section_inner_ring`, which expands Gaussian radial support from local mesh plane intersections and keeps the existing local patch estimate as a fallback.
- `validation`: Root validation on [dinosaur_root_ring_abc_round1/root_validation.json](d:/Evorig/mygs/results/dinosaur_root_ring_abc_round1/root_validation.json) showed `root_trans_matches_centroid_max_abs=0.0`, root joint train mask `{0:true}`, and `cross_section_inner_ring` active. Re-ran dinosaur/front60 Phase1 A/B/C at [dinosaur_root_ring_abc_round1](d:/Evorig/mygs/results/dinosaur_root_ring_abc_round1): A raw legal `0.011449`, B raw legal `0.011446`, C raw legal `0.010202`; joint0 rest drift was `0.0 -> 0.00962 -> 0.01044`, confirming it is released after A. Motion and final-topology viewers were generated for A/B/C. `pytest tests/test_evorig_next_phase1.py -q` passed (`56 passed`), and `compileall` passed for `src/evorig_next` and Phase1 runners.
- `scope`: [src/evorig_next/phase1_config.py](d:/Evorig/src/evorig_next/phase1_config.py), [src/evorig_next/phase1_field.py](d:/Evorig/src/evorig_next/phase1_field.py), [src/evorig_next/phase1_trainer.py](d:/Evorig/src/evorig_next/phase1_trainer.py), [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml), [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml), [tests/test_evorig_next_phase1.py](d:/Evorig/tests/test_evorig_next_phase1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `configs: keep phase1 fallback across bc`
- `summary`: Fixed the staged Phase1 B/C configs so low-support endpoint fallback remains enabled after Phase1A. Without this explicit field, B/C loaded the dataclass default `fallback_low_support_to_bone_endpoint=false`, allowing exactly zero-support vertices to be marked as zero-weight and overwritten with rest positions in the viewer.
- `validation`: Re-ran dinosaur/front60 from the existing A checkpoint. B fallback-fix reached raw legal `0.010359` with `zero_weight_row_count=0`; C fallback-fix reached raw legal `0.008767` with `zero_weight_row_count=0`. Generated motion and final-topology viewers for both fallback-fix runs under [dinosaur_root_ring_abc_round1](d:/Evorig/mygs/results/dinosaur_root_ring_abc_round1).
- `scope`: [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml), [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `configs: align phase1 bc support semantics`
- `summary`: Audited the loaded Phase1 A/B/C configs for accidental staged inconsistencies. B/C are continuation configs, so they should keep A's support semantics while changing only the staged optimizer schedule. Restored B/C `ownership_mode=endpoint_cut`, global lambda bounds, `gaussian_kernel_mahal_cutoff_sq=0.0`, `seed_count_scale=5.0`, and `phase1_scale_formula=cross_section_inner_ring`; B/C still intentionally differ in step count, learning rates, rest-joint refinement, lambda/SH activation step, and Phase1C acceleration/smoothing.
- `validation`: A/B/C config audit now reports only expected staged differences. Phase2 and Phase3 both build their trainers from the same Phase1 default, with `fallback_low_support_to_bone_endpoint=true`, `ownership_mode=endpoint_cut`, and kernel cutoff `0.0`. `py_compile` passed for the Phase1/Phase2/Phase3 entry scripts and core trainer/topology/refine modules.
- `scope`: [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml), [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase3: align dataclass defaults`
- `summary`: Synchronized `Phase3RefineConfig` dataclass defaults with the current Phase3 runner defaults. Previously the CLI entry produced the accepted line, but direct `Phase3RefineConfig()` construction still represented the older diagnostic default (`SH9`, frozen base params, offset disabled/inserted-only, no explicit JLG losses). This removes that hidden path back to old Phase3 parameters without adding any training-time guard.
- `validation`: Loaded Phase1 A/B/C configs through the actual loader and found zero unexpected staged differences. Loaded `Phase2TopologyConfig()` and `Phase3RefineConfig()` directly: Phase2 reports the current `center_capB`/JLG defaults, and Phase3 now reports `SH16`, base params unfrozen, all-Gaussian offsets from step 0, rest joints locked, and JLG `0.20/0.05/0.05`. `py_compile` passed for `phase3_refine.py` and the Phase3 runner; `git diff --check` passed.
- `scope`: [src/evorig_next/phase3_refine.py](d:/Evorig/src/evorig_next/phase3_refine.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: align dataclass defaults`
- `summary`: Synchronized `Phase1Config` inner dataclass defaults with the current Phase1 support semantics. The old defaults (`sigmoid`, cutoff `9.0`, fallback off, legacy scale, wide lambda bounds, no Gaussian illegal coverage loss) were the root cause of B/C reverting when their YAML omitted fields. The frozen A/B/C configs still explicitly list the critical values, but direct `Phase1Config()` construction now also returns the current support field defaults.
- `validation`: Printed `Phase1Config()` critical defaults and verified they are now `endpoint_cut`, fallback on, cutoff `0.0`, lambda `[-0.2,1.2]` with forced bounds, JLG `0.20/0.05/0.05`, and `cross_section_inner_ring`. Loaded frozen A/B/C configs through the actual loader and found zero unexpected differences. `pytest tests/test_evorig_next_phase1.py -q` passed (`56 passed`).
- `scope`: [src/evorig_next/phase1_config.py](d:/Evorig/src/evorig_next/phase1_config.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase2: merge fault components and scale thresholds`
- `summary`: Updated the Phase2 default branch signal path. Branch components are now built once from the combined `wrong OR uncovered` fault mask instead of separate wrong/uncovered passes, so mixed fault regions stay intact. Mixed wrong+uncovered components are no longer removed by the old dominance gate and receive a relaxed score/error gate. Component and seed-joint-repair min-vertex thresholds now scale by normalized rest-mesh surface area relative to the camel reference area `3.9673382939`, with default effective minima of `4` and `8`. Branch path sampling default was restored to `max_intermediate_points=10`.
- `validation`: `python -m py_compile` passed for `phase2_topology.py` and `run_evorig_next_phase2_round1.py`; `pytest tests/test_evorig_next_phase1.py -q` passed (`56 passed`). A dinosaur/front60 Phase2 signal export from `dinosaur_root_ring_abc_round1/rootringc_smooth_aligned/phase1_state.pt` wrote [signal_combined_component_default](d:/Evorig/mygs/results/dinosaur_phase2_component_gate_round1/signal_combined_component_default): rest area `1.7027`, effective component min vertices `4`, effective seed repair min vertices `10`, and the retained branch component is `mixed_fault`.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `viewer: inherit phase1 defaults across phases`
- `summary`: Fixed the default-config chain for visualization. Phase2 and Phase3 run directories now write the inherited `phase1_config.json` from resumed checkpoints when possible, and the interactive viewer reads `phase1_config.json`, then `phase1_state.pt`, then current Phase1 defaults. Viewer fallback now uses current support semantics (`endpoint_cut`, kernel cutoff `0.0`, `cross_section_inner_ring`, radial divisor `3.0`) instead of the legacy sigmoid/one-sigma visualization path.
- `validation`: `python -m py_compile` passed for `interactive_viewer.py`, `run_evorig_next_phase2_round1.py`, and `run_evorig_next_phase3_round1.py`. Regenerated the dinosaur root/ring Phase1 viewer and confirmed the HTML contains `range_sigma_scale=3.000` 564 times and `range_sigma_scale=1.000` 0 times.
- `scope`: [src/evorig_next/interactive_viewer.py](d:/Evorig/src/evorig_next/interactive_viewer.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [scripts/run_evorig_next_phase3_round1.py](d:/Evorig/scripts/run_evorig_next_phase3_round1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `viewer: make phase1 config strict`
- `summary`: Removed silent viewer fallback for missing Phase1 config. Interactive viewer generation now requires `phase1_config.json` or a checkpoint-embedded `phase1_config`; otherwise it fails instead of guessing support semantics. The final-topology viewer always uses `Phase1GaussianField`, and the remaining Phase1 field API defaults were changed from legacy `sigmoid` to current `endpoint_cut`.
- `validation`: `python -m py_compile` passed for `interactive_viewer.py` and `phase1_field.py`. Regenerated the dinosaur root/ring Phase1 viewer and confirmed the HTML contains `range_sigma_scale=3.000` 564 times and `range_sigma_scale=1.000` 0 times. Joint 22 scale probe on current dinosaur aligned run reports maximum displayed `3sigma` about `0.236`, not the oversized legacy visualization behavior.
- `scope`: [src/evorig_next/interactive_viewer.py](d:/Evorig/src/evorig_next/interactive_viewer.py), [src/evorig_next/phase1_field.py](d:/Evorig/src/evorig_next/phase1_field.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: cap bone scale outliers`
- `summary`: Replaced the old same-bone scale equalization behavior inside `bone_scale_consistency_loss` with a direct per-bone outlier cap. The existing loss name/config are preserved, but the loss now allows Gaussian scales on the same bone to vary freely up to a 3x per-axis ratio and only penalizes larger ratios. No new loss or config branch was added.
- `validation`: `python -m py_compile` passed for `phase1_losses.py` and `phase1_config.py`. A direct sanity check returns zero loss for same-bone scale ratios `1/2/3` and a positive penalty for `1/2/4`.
- `scope`: [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [src/evorig_next/phase1_config.py](d:/Evorig/src/evorig_next/phase1_config.py), [configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml), [configs/frozen/evorig_next_phase1b_restrefine_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1b_restrefine_default.yaml), [configs/frozen/evorig_next_phase1c_smooth_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase1c_smooth_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: use mean scale ratio cap`
- `summary`: Adjusted `bone_scale_consistency_loss` to use a mean-scale ratio band. For each bone and scale axis, Gaussian scales within `[0.5x, 2x]` of the per-axis mean log-scale are unpenalized; only larger deviations are penalized. This keeps the regularizer as an outlier cap rather than a same-scale equalizer.
- `validation`: `python -m py_compile src/evorig_next/phase1_losses.py` passed. Direct sanity checks return zero for scale ratios `1/1.5/2` and positive penalties for high (`1/1.5/4`) and low (`0.2/1/1.2`) outliers.
- `scope`: [src/evorig_next/phase1_losses.py](d:/Evorig/src/evorig_next/phase1_losses.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `docs: clarify phase2 vertex gate`
- `summary`: Reverted the experimental component surface-area noise gate and clarified the active Phase2 default. The default path still filters branch/seed components by component vertex count; normalized rest-mesh total area is only used to scale the camel reference vertex-count threshold across assets. Combined fault components, mixed-fault retention, relaxed mixed gates, and `max_intermediate_points=10` remain active.
- `validation`: `git revert --no-edit c689302` completed cleanly. `python -m py_compile` passed for `phase2_topology.py` and `run_evorig_next_phase2_round1.py`.
- `scope`: [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase2: scale vertex gates by mesh density`
- `summary`: Changed Phase2 component min-vertex scaling from total-area scaling to average-vertex-area density scaling. Effective thresholds now use `ceil(base_min_vertices / ((area_current / vertex_count_current) / (area_camel / vertex_count_camel)))`, so meshes finer than the camel reference get larger noise thresholds and coarser meshes get smaller thresholds. The active filter is still component vertex count, not component surface area.
- `validation`: `python -m py_compile` passed for `phase2_topology.py` and `run_evorig_next_phase2_round1.py`. A threshold probe reports camel `10/24`, dinosaur/front60 `9/20`, and kangaroo `9/22` for component/seed-repair thresholds.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase2: add live schedule logging`
- `summary`: Added live Phase2 schedule logging. Scheduled topology evolution now writes `phase2_schedule_live.jsonl` immediately with schedule start, update start, pre-update signal counts, seed repair results, branch selection, branch acceptance, split attempts, update completion, and per-step post-update training losses. Phase2 update and training loops also expose tqdm progress bars for interactive runs.
- `validation`: `python -m py_compile` passed for `phase2_topology.py` and `run_evorig_next_phase2_round1.py`; `git diff --check` passed for the edited file. A dinosaur/front60 first-update run confirmed the live file is created and records `schedule_start`/`update_start` before the slow pre-update signal stage.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase2: cache voxel path fields on disk`
- `summary`: Added disk-backed Phase2 voxel path field caching under `mygs/outputs/phase2_voxel_path_cache`. The cache key hashes normalized rest vertices, mesh faces, and voxel parameters, so it is safe across different meshes and settings. `build_phase2_topology_signals` still recomputes dynamic coverage, wrong ratio, residuals, and branch components; only the rest-mesh voxel routing field is reused.
- `validation`: `python -m py_compile` passed for `phase2_topology.py` and `run_evorig_next_phase2_round1.py`. A direct dinosaur/front60 voxel cache probe reported first call `source=built` and second call `source=disk` with the same grid shape `(47, 37, 97)`.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase2: reuse graph cache for components`
- `summary`: Updated `build_phase2_topology_signals` to reuse the rest-mesh adjacency cache for component extraction. The cache is the same topology-only graph cache used by Phase1 and is keyed by mesh faces and vertex count, so branch component maps, branch proposals, and seed-joint-repair candidates no longer rescan all faces every time they need connected components. Dynamic coverage, wrong ratio, residuals, and component scores are still recomputed from the current field/topology.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py` passed. Import check passed with `PYTHONPATH=src`. A direct component equivalence probe confirmed the new adjacency-backed connected-component path matches the old face-scan path on the same mask and mesh.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase2: remove branch relative and inside hard gates`
- `summary`: Removed two Phase2 branch hard gates from the default path. The parent-to-path segment-inside acceptance field was deleted; final inside fractions remain only as diagnostic values in `branch_path_inside`. The relative component error-mass field was also deleted, so branch components no longer compete against the sum of all component error mass. The global component error fraction gate remains enabled at `0.03`.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py` passed. Direct config import reports `branch_min_global_error_mass_fraction=0.03`, and the removed gate names are absent from active Phase2 code and the frozen Phase2 config.
- `scope`: [src/evorig_next/phase2_topology.py](d:/Evorig/src/evorig_next/phase2_topology.py), [scripts/run_evorig_next_phase2_round1.py](d:/Evorig/scripts/run_evorig_next_phase2_round1.py), [configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml](d:/Evorig/configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/experiment_guardrails_zh.md](d:/Evorig/docs/experiment_guardrails_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: localize cross-section scale rings`
- `summary`: Fixed `cross_section_inner_ring` scale initialization so the plane section is computed only on the Gaussian's local connected mesh patch. The previous implementation intersected the cutting plane with the whole mesh, then picked nearest points by angle; on toes, jaws, or thin parts, that could mix in a different body part and inflate the Gaussian radial scale. If no local connected section is available, the initializer now keeps the existing local-patch scale instead of falling back to a global section. Follow-up in the same work chunk replaced the custom angle-bin point selection with `trimesh.section(..., local_faces=...)`, using reconstructed section contours instead of unordered plane-intersection points.
- `validation`: `python -m py_compile src/evorig_next/phase1_field.py` passed. `pytest tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_covers_asymmetric_plane_extent tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_ignores_disconnected_same_plane_part -q` passed.
- `scope`: [src/evorig_next/phase1_field.py](d:/Evorig/src/evorig_next/phase1_field.py), [tests/test_evorig_next_phase1.py](d:/Evorig/tests/test_evorig_next_phase1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: initialize radial scale from longest section extent`
- `summary`: Updated Phase1 scale initialization so both radial axes use the longest local cross-section extent divided by `radial_sigma_divisor`. This applies to both the local-patch fallback and the `trimesh.section` contour path. The initializer no longer writes a short radial scale from the shortest PCA cross-section axis.
- `validation`: `python -m py_compile src/evorig_next/phase1_field.py` passed. `pytest tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_covers_asymmetric_plane_extent tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_ignores_disconnected_same_plane_part -q` passed; both tests now assert equal radial 3-sigma extents.
- `scope`: [src/evorig_next/phase1_field.py](d:/Evorig/src/evorig_next/phase1_field.py), [tests/test_evorig_next_phase1.py](d:/Evorig/tests/test_evorig_next_phase1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

- `date`: 2026-05-10
- `commit`: `this commit`
- `message`: `phase1: orient radial scale by farthest section direction`
- `summary`: Corrected the previous longest-extent interpretation. Phase1 scale initialization now finds the farthest point direction in the local cross-section, uses that as the first radial axis, uses the perpendicular direction as the second radial axis, and assigns separate extents on those two axes. This applies to both local-patch fallback and `trimesh.section` contour initialization.
- `validation`: `python -m py_compile src/evorig_next/phase1_field.py` passed. `pytest tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_covers_asymmetric_plane_extent tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_ignores_disconnected_same_plane_part -q` passed.
- `scope`: [src/evorig_next/phase1_field.py](d:/Evorig/src/evorig_next/phase1_field.py), [tests/test_evorig_next_phase1.py](d:/Evorig/tests/test_evorig_next_phase1.py), [docs/current_system_zh.md](d:/Evorig/docs/current_system_zh.md), [docs/current_workline_zh.md](d:/Evorig/docs/current_workline_zh.md), [docs/commit_journal_zh.md](d:/Evorig/docs/commit_journal_zh.md)

## 2026-05-10 phase1 full-section radial scale

- `summary`: Corrected `cross_section_inner_ring` again. `trimesh.section` now cuts the full rest mesh instead of a local face patch, then selects the smallest closed section contour that contains the Gaussian center. This avoids local patch truncation for joints such as dinosaur joint 1/2.
- `summary`: When a valid section contour is found, radial scale is assigned directly from the contour's farthest direction and perpendicular extent divided by 3. It is no longer mixed with fallback patch scales whose axes may differ.
- `validation`: `python -m py_compile src/evorig_next/phase1_field.py` passed. `pytest tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_covers_asymmetric_plane_extent tests/test_evorig_next_phase1.py::test_phase1_cross_section_inner_ring_ignores_disconnected_same_plane_part -q` passed.
- `viewer`: `mygs/results/dinosaur_ring_section_view_round1/init_full_section_contour_steps0/visuals/interactive/interactive_final_topology.html`.

## 2026-05-10 phase1 config default audit

- `summary`: Aligned `Phase1Config()` dataclass defaults with the active frozen Phase1A YAML. This removes the old hidden fallback values for root handling, joint-boundary losses, scale losses, lambda thawing, trace interval, and densify stages.
- `summary`: Added a regression test requiring every key in `configs/frozen/evorig_next_phase1_final500_supportloss_default.yaml::phase1` to match `Phase1Config().to_dict()`, so future default-line drift fails immediately.
- `summary`: Removed a duplicate `@staticmethod` decorator left in `phase1_field.py`.
- `validation`: `python -m py_compile src/evorig_next/phase1_config.py src/evorig_next/phase1_field.py tests/test_evorig_next_phase1.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests.

## 2026-05-10 phase1 off-center section guard

- `summary`: Added a geometric guard for `cross_section_inner_ring` contour selection. A section contour is rejected when it is extremely off-center relative to the Gaussian center and spans more than three local bone lengths; non-containing fallback contours are also rejected when their radius exceeds two local bone lengths.
- `summary`: This prevents joint-local Gaussians near dinosaur joint 8 from using a far global head/body section as their radial scale while preserving the wider valid sections around joints 1 and 2.
- `validation`: Dinosaur steps=0 `joint8` adjacent max 3-sigma radial scale dropped from `1.0199/0.9247` to `0.2171/0.1865`; joint1/2 adjacent 3-sigma scale stayed unchanged. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests.
- `viewer`: `mygs/results/dinosaur_ring_section_view_round1/init_centered_contour_guard2_steps0/visuals/interactive/interactive_final_topology.html`.

## 2026-05-10 phase1 near-far balanced section scale

- `summary`: Replaced the bone-length hard section guard with a same-axis near/far scale rule. For each radial axis, the initializer measures the positive and negative contour extents, treats them as `near` and `far`, and uses `near + (far - near) * (near / far)` before dividing by the radial sigma divisor. The local connected patch scale remains the per-axis lower bound.
- `summary`: This keeps the rule local to section geometry instead of comparing against bone length, while still reducing off-center full-section contours that would otherwise inflate one side of a Gaussian.
- `validation`: `python -m py_compile src/evorig_next/phase1_field.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. Dinosaur steps=0 `joint8` adjacent max 3-sigma radial scale changed from full-section `0.9247/0.2908` to near/far `0.4760/0.2635`; joint1/2 adjacent max 3-sigma is about `0.22-0.25`.
- `viewer`: `mygs/results/dinosaur_ring_section_view_round1/init_axis_nearfar_balanced_section_steps0/visuals/interactive/interactive_final_topology.html`.

## 2026-05-10 phase1 radius-ratio section fallback

- `summary`: Replaced the same-axis near/far balancing rule with a section-center radius ratio fallback. For each full-mesh section contour, the initializer computes `max(norm(uv)) / min(norm(uv))`; if the ratio is at least `4`, the section is treated as off-center and the Gaussian keeps its local connected patch scale.
- `summary`: This removes the earlier bone-length hard guard while preserving the intended behavior: valid centered sections set the Gaussian scale from the real contour, and pathological contours that pass very near the Gaussian center are rejected.
- `validation`: `python -m py_compile src/evorig_next/phase1_field.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. Dinosaur steps=0 `joint8` adjacent max 3-sigma radial scale changed from full-section `0.9247/0.2908` to radius-ratio `0.1865/0.0854`; `joint1/2` adjacent max 3-sigma remains about `0.30`.
- `viewer`: `mygs/results/dinosaur_ring_section_view_round1/init_radius_ratio_gate4_steps0/visuals/interactive/interactive_final_topology.html`.

## 2026-05-10 phase2 topology rollback

- `summary`: Rolled back the default Phase2 topology code path to the state before the later dinosaur topology experiments. The revert removes posed-JLG branch diversity and all subsequent default-line changes that altered component selection, path geometry, medial tip targeting, one-interval topology-joint freeze, and tightened mixed/leaf gates.
- `summary`: Phase1 and Phase3 code paths were not reset. The rollback targets only the Phase2 topology files, runner defaults, and frozen Phase2 config affected by those topology commits.
- `validation`: `git diff 137e5a1^..HEAD -- src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py configs/frozen/evorig_next_phase2_lineage_sibling_fast_default.yaml src/evorig_next/utils/mesh_voxel_path.py` is empty. `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py src/evorig_next/utils/mesh_voxel_path.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests.

## 2026-05-10 phase2 seed-joint repair isolation

- `summary`: Fixed seed-joint repair semantics so incident Gaussians move with the repaired bone instead of being reprojected to preserve their pre-repair world centers. The previous reproject step cancelled the intended joint move in the support field.
- `summary`: If a seed-joint repair is accepted in a topology update, branch and split are skipped for that same update. The repaired topology is trained for the scheduled interval first; later branch/split decisions are made from the refreshed field.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. Kangaroo Phase2 verification at `mygs/results/phase2_seedrepair_followbone_verify_round1/kangaroo_seedrepair_followbone` produced two seed repairs with `moved_with_bone_gaussian_count=40/36`; both repair updates recorded `branch_selection.skip_reason=seed_joint_repair_accepted`.

## 2026-05-10 phase2 component merge and mixed gate cleanup

- `summary`: Removed the default branch lineage/sibling relaxation path and its runner/config options. The parent-lineage guard remains, but there is no separate sibling threshold for ear-specific continuation.
- `summary`: Restored default branch component merging across short mesh-adjacency gaps, and replaced the fixed mixed-fault gate scales with `branch_min_global_error_mass_fraction * (1 - dual_fault_fraction)`, so components with more vertices that are both wrong-covered and uncovered have a lower branch gate.
- `summary`: Added a voxel-polyline-preserving path refinement pass. If sparse branch control points create a low-inside straight chord, the code inserts additional points along the existing voxel route up to `max_intermediate_points`; later medialize/axis-align/root-smooth postprocessing is reverted if it reduces segment inside fraction.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. No Phase2 training run was kept for this chunk.

## 2026-05-10 phase2 internal branch tip target

- `summary`: Branch routing now separates the diagnostic surface tip from the actual branch target. `_component_tip` still records the PCA surface endpoint, but parent routing uses a nearby voxel-interior high-clearance target inside the component-local bounding box, with mesh-inside projection as a fallback.
- `summary`: Curvature path sampling no longer unconditionally mixes uniform fallback samples into every path. Uniform points are added only when the curvature-selected points are fewer than the required minimum, reducing artificial bends from fallback samples.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. `git diff --check` passed.

## 2026-05-11 phase2 restore old curvature point selection

- `summary`: Reverted the internal branch tip target experiment. Branch parent routing again uses the original surface PCA tip, and Phase2 no longer exposes `tip_internal_target` config or CLI fields.
- `summary`: Restored the previous branch polyline point selection behavior: curvature-selected points are combined with uniform fallback samples before spacing/deduplication, matching the pre-internal-target Phase2 path sampler.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. `git diff --check` passed.

## 2026-05-11 phase2 curvature path selector cleanup

- `summary`: Fixed the branch curvature selector so `branch_max_intermediate_points=10` is no longer capped to 3. The selector now uses the configured limit directly when selecting curvature extrema along the voxel parent-to-tip polyline.
- `summary`: Removed unconditional uniform fallback and root-bend insertion from `_branch_path_points_from_polyline`. Straight paths stay compact with only the tip point; extra points are inserted only along the same voxel polyline when an already-curved path has an excessive arc gap.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed. A synthetic curved polyline returned 5 control points with `max_intermediate_points=10`, while a straight polyline returned only the tip. `pytest tests/test_evorig_next_phase1.py -q` passed with 58 tests. `git diff --check` passed.

## 2026-05-11 phase2 branch curvature diagnostics

- `summary`: Added explicit diagnostic fields for branch paths: `branch_path_points_curvature` stores the pure curvature selector output, and `branch_path_points_pre_refine` stores the path after component-root replacement but before inside-refine. This does not change topology selection or training.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py` passed. Kangaroo Phase2 was rerun with the same checkpoint and produced the same final raw error `0.0095006078`; analysis files were written under `mygs/results/kangaroo_phase2_curvature_diag_round1/kangaroo_curvature_diag_phase2/analysis`.

## 2026-05-11 phase2 restore kangaroo curvature functions

- `summary`: Restored only the branch curvature path functions to the earlier kangaroo baseline behavior: curvature intermediates are capped at three, root-bend/fallback points are included, and long-segment refine can run on the old kept-point set. No seed repair, component merge, voxel routing, or training config logic was changed.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py` passed.

## 2026-05-11 phase2 keep max10 curvature budget

- `summary`: Corrected the restored branch curvature function so `branch_max_intermediate_points` is no longer capped to three. The old root-bend/fallback selection behavior remains, but the configured max10 curvature budget is honored.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py` passed.

## 2026-05-11 phase2 restore kangaroo sibling-ear curvature budget

- `summary`: Reverted the branch curvature point budget to the kangaroo sibling-ear baseline recorded at `9e71a71`: Phase2 defaults now use `branch_max_intermediate_points=4`, while pure curvature extrema inside `_branch_path_points_from_polyline` are capped at three before the existing root-bend, fallback, and long-segment refinement logic runs.
- `summary`: Updated the runner default, frozen Phase2 config, and current docs so the default path does not mix the old curvature code with a later `max10` configuration.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed.

## 2026-05-11 phase2 remove post-curvature insertion

- `summary`: Removed the later `_refine_branch_path_against_inside` post-curvature insertion pass from the default branch path. The branch path functions now match `9e71a71` for `_curvature_path_points`, `_branch_path_points_from_polyline`, component-root replacement, medialization, axis alignment, and root-tangent smoothing; no extra polyline midpoint insertion is applied between curvature sampling and the existing stable postprocess.
- `validation`: A Python AST hash check confirmed the relevant branch path functions match `9e71a71`, with `_refine_branch_path_against_inside` missing in both current and target. `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed.

## 2026-05-11 phase2 restore postprocess path behavior

- `summary`: Removed the current-only postprocess revert that could replace the axis-align / medialize / root-tangent-smoothed branch path with the pre-postprocess path. Those postprocess functions existed in the kangaroo sibling-ear baseline and should directly define the final branch points, as in `9e71a71`.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py` passed. AST checks confirmed `_curvature_path_points`, `_branch_path_points_from_polyline`, `_replace_path_root_with_component_root`, `_medialize_branch_path_points`, `_axis_align_branch_path_points`, and `_smooth_branch_root_tangent` match `9e71a71`.

## 2026-05-11 UniRig dynamic import joint projection

- `summary`: Added `--alignment-frame-index` to the UniRig dynamic importer and normalized dense bone IDs from rigged GLB nodes, so non-contiguous `bone_*` node names map to compact EvoRig joint IDs.
- `summary`: Added an import-gate joint projection pass for rigged-GLB UniRig samples. It mirrors the skeleton-only import behavior by projecting outside rig joints into the normalized rest mesh before rig cleanup, preventing valid assets such as `spring_airplane` from failing because the raw UniRig root/seed joints are slightly outside the mesh.
- `validation`: `python -m py_compile scripts/import_unirig_dynamic_glb_sample.py src/evorig_next/io/unirig_dynamic.py` passed. `git diff --check -- scripts/import_unirig_dynamic_glb_sample.py src/evorig_next/io/unirig_dynamic.py` passed. `spring_airplane` import passed with 15/27 projected joints and completed Phase1/Phase2/Phase3 candidate generation.

## 2026-05-11 phase2 voxel-route branch path default

- `summary`: Moved the default Phase2 branch path back onto the voxel parent-to-tip route itself. Component-root insertion/bridge, medialization, axis alignment, root-tangent smoothing, and seed-leaf parent alignment are disabled by default so later postprocess steps cannot move branch joints off the routed path.
- `summary`: Added a clearance-weighted voxel path cost used after parent selection (`branch_path_clearance_weight=2.0`, `branch_path_clearance_power=1.0`). Parent ranking remains the original voxel-distance selection; the selected route now prefers interior voxels when there is an equivalent short path.
- `summary`: Added `scripts/search_branch_path_variants.py` for offline branch-path inspection. It compares production, geometric, and clearance-weighted variants against the accepted branch event without rerunning training.
- `validation`: `python -m py_compile src/evorig_next/utils/mesh_voxel_path.py src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py scripts/search_branch_path_variants.py` passed.
- `validation`: double_knife branch-only short Phase2 accepted 2 branches; the first branch no longer uses component-root/bridge/axis/medial/root-tangent postprocess and the old corrupted path inside fraction improved from `0.476` to `1.0` in offline comparison.
- `validation`: kangaroo branch-only short Phase2 accepted 2 branches with postprocess disabled; branch inside fractions were `0.714` and `1.0`.
- `validation`: camel branch-only short Phase2 accepted 2 branches with postprocess disabled; branch inside fractions were `1.0` and `0.952`.

## 2026-05-11 phase2 remove obsolete branch postprocess code

- `summary`: Deleted the obsolete default-path branch postprocess implementation instead of leaving it behind as disabled switches. Removed component-root insertion/bridge, medialization, axis alignment, root-tangent smoothing, and seed-leaf parent alignment fields from `Phase2TopologyConfig`, the Phase2 runner CLI, and the frozen Phase2 config.
- `summary`: The only default branch path geometry is now the clearance-weighted voxel parent-to-tip route plus route-internal curvature/long-segment sampling. There are no off-route postprocess hooks left in the active default path.
- `validation`: `rg` confirmed the removed postprocess symbols no longer appear in `phase2_topology.py`, the Phase2 runner, or the frozen Phase2 config. `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py src/evorig_next/utils/mesh_voxel_path.py scripts/search_branch_path_variants.py` passed. `git diff --check` passed.

## 2026-05-11 restore shared segment-inside helper

- `summary`: Restored `_mesh_segment_inside_fraction` as a shared Phase2/Phase3 geometric diagnostic helper after the postprocess cleanup. The function is used by seed-joint repair, split inside validation, and Phase3 warnings; it is not a branch path postprocess hook.
- `validation`: `python -m py_compile src/evorig_next/phase2_topology.py scripts/run_evorig_next_phase2_round1.py src/evorig_next/phase3_refine.py scripts/run_evorig_next_phase3_round1.py` passed. `git diff --check` passed.

## 2026-05-11 bone-section PCJS

- `summary`: Replaced the old joint-centered PCJS shell sampling with bone-local cross-section sampling. `loss_pcjs` now evaluates shell distances from a 0.1 bone-length probe on each parent-child segment, using 12 directions in the plane perpendicular to the posed bone axis. This keeps the loss as a cross-pose stability term instead of a joint-centering term.
- `summary`: Removed the unused `loss_posed_joint_cross_section_inside` training path and its config fields so there is no second center-pulling cross-section loss in the active Phase1 line. Rest/posed inside and clearance losses continue to use their original mesh-shell queries.
- `summary`: Restored `_branch_path_inside_summary` as a Phase2 diagnostic helper so Phase1 topology-signal export does not fail after branch postprocess cleanup.
- `validation`: `python -m py_compile src/evorig_next/phase1_losses.py src/evorig_next/phase1_trainer.py src/evorig_next/phase1_config.py src/evorig_next/phase2_topology.py` passed. `rg` confirmed the removed cross-section-inside config/call symbols no longer appear.
- `validation`: double_knife Phase1B/C from existing A completed with the new PCJS. C summary: `final_error_raw=0.0132426`, `zero_weight_row_count=0`; viewers written under `mygs/results/double_knife_pcjs_bonesection_round1/phase1c_from_b_bonesection_pcjs/visuals/interactive`.

## 2026-05-11 PCJS section balance

- `summary`: Added a symmetric distance-balance term inside `loss_pcjs`. For each bone-section probe and each of the 12 perpendicular directions, the loss also penalizes forward/backward shell distance imbalance, pulling the probe toward the local section midline without binding it to a fixed vertex-section centroid.
- `validation`: `python -m py_compile src/evorig_next/phase1_losses.py src/evorig_next/phase1_trainer.py src/evorig_next/phase1_config.py src/evorig_next/phase2_topology.py` passed.
- `validation`: double_knife Phase1B/C from existing A completed under `mygs/results/double_knife_pcjs_balance_round1`. C summary: `final_error_raw=0.0132426`, `zero_weight_row_count=0`; viewers written under `phase1c_from_b_pcjs_balance/visuals/interactive`.

## 2026-05-11 viewer default exports final topology

- `summary`: Changed `scripts/view_run_interactive.py --kind both` to export `motion+final` instead of `training+motion`. The initial skeleton is drawn by the final topology viewer from `wrong_init_rig.json`, so the default viewer export now includes the file that contains the initial skeleton overlay.
- `validation`: `python -m py_compile scripts/view_run_interactive.py` passed. `python scripts/view_run_interactive.py --run-dir mygs/results/double_knife_pcjs_balance_round1/phase1c_from_b_pcjs_balance --kind both` wrote both `interactive_motion.html` and `interactive_final_topology.html`.

## 2026-05-11 pre-normalized JLG support loss

- `summary`: Changed the default JLG training objective to pre-normalized illegal support mass. `loss_illegal_support` now penalizes `sum_j support_nj * (1 - L_nj)` directly instead of the invalid ratio; the invalid ratio remains a topology diagnosis signal, not the differentiable loss.
- `summary`: Disabled `loss_gaussian_illegal_coverage` in the default Phase1/Phase2/Phase3 configs (`0.0`) and set `illegal_support_tau=0.0`. The Gaussian-level loss implementation was also changed to pre-normalized mass so optional ablations do not silently return to ratio semantics.
- `validation`: `python -m py_compile` passed for the modified Phase1/Phase2/Phase3 modules and runners. A tensor smoke test verified `illegal_support_loss` now returns the expected raw illegal-mass squared mean.
- `validation`: double_knife Phase1B from the existing A state completed at `E:\evorig_unirig\double_knife\evorig_result\phase1_jlg_mass_round1\doubleknifeb_masssupport_restrefine`; `final_error_raw=0.0124991`, `zero_weight_row_count=0`, and viewers were written under `visuals/interactive`.

## 2026-05-11 remove bone scale length cap

- `summary`: Removed `bone_scale_length_cap_loss` and its config fields. Scale control is now handled by the existing scale anchor, bone-scale consistency, and bone-scale band losses, without a direct penalty that can increase current bone length by moving rest joints.
- `validation`: `python -m py_compile src\evorig_next\phase1_losses.py src\evorig_next\phase1_config.py src\evorig_next\phase1_trainer.py` passed. `rg` confirmed `bone_scale_length_cap`, `max_axial_scale_to_bone_length`, and `max_radial_scale_to_bone_length` no longer appear in active code, frozen configs, or docs.
- `validation`: double_knife Phase1B from the existing A state completed at `E:\evorig_unirig\double_knife\evorig_result\phase1_no_lengthcap_round1\doubleknifeb_no_lengthcap_restrefine`; `final_error_raw=0.0127722`, `zero_weight_row_count=0`, and viewers were written under `visuals/interactive`.
- `validation`: A-to-B joint drift showed this loss was not the main joint5 driver: with length cap `joint5=0.105694`, without length cap `joint5=0.107641`. The earlier gradient audit remains the stronger evidence that `bone_recon_topk` and reconstruction dominate this drift.

## 2026-05-11 remove bone-local top-k reconstruction loss

- `summary`: Removed the legacy `loss_bone_vertex_recon_topk` path from Phase1. The loss function, Phase1Config fields, trainer call, trace output field, and frozen Phase1A/B/C config entries were deleted. Local high-error topology evidence should remain a Phase2 residual/split signal, not a Phase1 rest-joint reconstruction force.
- `validation`: `python -m py_compile src\evorig_next\phase1_losses.py src\evorig_next\phase1_config.py src\evorig_next\phase1_trainer.py` passed. `rg` confirmed no active code/config references remain; only this historical journal still mentions the old term.
- `validation`: double_knife 50-step Phase1B from the same A state now matches the manual no-topk diagnostic exactly. Joint5 drift dropped from the previous default `0.034824` to `0.008929`; mean rest-joint drift dropped from `0.011246` to `0.009028`.

## 2026-05-11 strict environment runbook

- `summary`: Added `scripts/check_evorig_environment.py`, a strict preflight that imports the required runtime modules (`torch`, `open3d`, `trimesh`, `scipy`, `plotly`, `yaml`, `tqdm`) and verifies CUDA plus `open3d.t.geometry.RaycastingScene` before official training. Missing dependencies now have a documented fail-fast command instead of being discovered through fallback behavior during a run.
- `summary`: Added [docs/evorig_next_strict_runbook_zh.md](d:/Evorig/docs/evorig_next_strict_runbook_zh.md) and linked it from [readme.md](d:/Evorig/readme.md). The runbook records the exact current config chain, preflight command, Phase1/2/3 commands, no-fallback rules, and the latest Phase1B verification.
- `validation`: Re-ran full 200-step double_knife Phase1B after removing `loss_bone_vertex_recon_topk`: [default_after_remove_topk_200](d:/Evorig/mygs/results/joint5_loss_debug_round1/default_after_remove_topk_200) reached `final_error_raw=0.0146646686`, `final_error_raw_all=0.0149528580`, `zero_weight_row_count=0`, and `final_outside_active_gaussian_count=2`. The A-to-B drift comparison improved from old `joint5=0.1076411` to new `joint5=0.0174781`. Viewers were written under `visuals/interactive`.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\check_evorig_environment.py` passed with CUDA available on `NVIDIA GeForce RTX 4080 SUPER`. `python -m py_compile scripts\check_evorig_environment.py` and `git diff --check` passed.

## 2026-05-11 fix PCJS rest-joint gradient

- `summary`: Fixed `pose_consistent_joint_shell_loss` in edge-mode by constructing the differentiable bone-section `reference_points` outside the `torch.no_grad()` target-building block. Before this fix, PCJS reported a non-zero value but did not backpropagate to `rest_joints` for the default bone-section mode.
- `validation`: `python -m compileall src\evorig_next\phase1_losses.py` passed. A double_knife gradient audit now reports `loss_pcjs.requires_grad=True` and non-zero rest-joint gradients (`grad_norm=0.00166` at Phase1B start, `0.00168` at the no-edge/no-radial B200 state), whereas the previous audit reported zero.

## 2026-05-11 configurable PCJS section position

- `summary`: Added `pcjs_section_lambda` to Phase1Config and frozen Phase1A/B/C configs, keeping the default at `0.10`. Phase1 now passes this config value into both PCJS shell descriptor construction and the PCJS loss, so ablations can test a mid-bone section without editing code.
- `validation`: `python -m compileall src\evorig_next\phase1_config.py src\evorig_next\phase1_trainer.py` passed.

## 2026-05-11 freeze root rest joint with separate motion root

- `summary`: Fixed the effective rest-joint training mask so `freeze_root_rest_joint=true` always freezes the rest-space root joint, including the default `separate_motion_root=true` mode. Previously the root rest joint could still drift in Phase1B/C while root motion was handled by the separate root translation parameter.
- `summary`: Added a hard post-step restore for all inactive rest joints. This protects the root and Phase1A-fixed joints even if a resumed optimizer carries stale Adam momentum from an older run.
- `summary`: Updated current-system/current-workline docs so `separate_motion_root=true` means root motion is carried by `root_trans`, not that the anatomical root rest joint is trainable.
- `validation`: Pending rerun after the current interrupted PCJS ablation; the trace that exposed the bug showed `root_rest_drift=0.0206` despite `freeze_root_rest_joint=true`.

## 2026-05-11 per-mesh frozen rest joints

- `summary`: Added `frozen_rest_joint_ids` to Phase1Config. The default is empty, so the main protocol is unchanged; when a run config lists joint ids, those rest joints are removed from the effective rest-joint training mask and are restored after each optimizer step.
- `validation`: `python -m compileall -q src\evorig_next scripts\run_evorig_next_phase1.py` passed. A double_knife Phase1B/C run with `frozen_rest_joint_ids=[5]` kept `joint5` and root rest drift at `0.0`; Phase1C reached `final_error_raw=0.0142189`, `final_error_raw_all=0.0141676`, `zero_weight_row_count=0`.

## 2026-05-11 phase2 branch minimum path points

- `summary`: Changed the Phase2 branch path minimum from three route samples to one route sample. `branch_min_path_points` now defaults to `1`, the Phase2 runner exposes `--phase2-branch-min-path-points`, and the frozen Phase2 config/docs record `min_path_points=1`.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe -m compileall src\evorig_next\phase2_topology.py scripts\run_evorig_next_phase2_round1.py` passed. `git diff --check` passed.
- `diagnostic`: On the existing double_knife Phase2 run `mygs/results/double_knife_phase2_joint5_frozen_round1/phase2_default_from_phase1c_joint5_frozen`, joints `41`, `64`, and `68` had zero drift between insertion and final state, so their outside status came from branch path initialization rather than later optimization.

## 2026-05-11 phase2 branch inside tip target and history dedup

- `summary`: Phase2 branch proposals now keep the surface fault tip only as a region locator. The actual branch endpoint is selected from nearby filled voxel centers that pass the exact mesh-inside query and have high clearance. If sampled path points still fall outside, they are replaced by nearest mesh-inside points on the same voxel route. Later updates reject branch candidates that repeat an already accepted branch by source component overlap or near-duplicate tip target.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe -m compileall src\evorig_next\phase2_topology.py scripts\run_evorig_next_phase2_round1.py` and `git diff --check` passed.
- `validation`: double_knife Phase2 from `double_knife_phase1_joint5_frozen_round1/phase1c_joint5_frozen` completed at `mygs/results/double_knife_phase2_tipclear_dedup_round1/phase2_tipclear_dedup_from_joint5_frozen_c`. It accepted exactly two branches: `parent=5, new=[32,33,34,35,36]` and `parent=20, new=[37,38,39,40]`; later updates selected zero repeated branches. Both accepted branches reported mesh-inside path points and minimum segment-inside fractions `1.0` and `0.9048`.

## 2026-05-11 connected hierarchy branch root

- `summary`: Added `connected_to_parent` to rig loading/export, Phase1 skeleton state, UniRig/FBX skeleton import, and viewer reconstruction. Non-connected Blender parent links are now hierarchy-only dashed links: they drive FK hierarchy but are excluded from physical bones, Gaussian support initialization, PCJS/cross-section descriptors, and posed-bone inside losses.
- `summary`: Phase2 branch insertion now creates `parent -> branch_root` as `connected_to_parent=false`, then creates connected physical bones from branch root to tip. Gaussians are initialized only on the connected root-to-tip chain.
- `summary`: Branch route points are snapped to mesh-inside voxel centers on the selected voxel route. Physical branch segments are iteratively subdivided on that same route when their inside-fraction diagnostic is below `branch_segment_refine_inside_fraction=0.75`, with a separate cap `branch_segment_refine_max_points=10`.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe -m compileall src\evorig_next scripts\import_unirig_skeleton_dynamic_sample.py scripts\run_evorig_next_phase2_round1.py` passed.
- `validation`: Short double_knife one-update smoke at `mygs/results/connected_branch_smoke_round1/double_knife_one_update_voxelcenter` accepted two branches. Exported rig confirms each branch root has `connected_to_parent=false` and no Gaussian bone on the dashed parent link; connected child segments received Gaussians. The first thin-blade branch still reports physical `min_segment_inside_fraction=0.0` even after route subdivision, indicating the voxel interior route itself lacks continuous mesh-inside support for that local thin sheet.

## 2026-05-11 viewer dashed hierarchy links

- `summary`: The interactive motion/final viewers now draw connected physical bones as solid lines and `connected_to_parent=false` hierarchy-only links as separate dashed line traces. Hover text also exposes the `connected` flag per joint.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe -m compileall src\evorig_next\interactive_viewer.py` passed. Regenerated double_knife smoke viewers at `mygs/results/connected_branch_smoke_round1/double_knife_one_update_voxelcenter/visuals/interactive`.

## 2026-05-11 viewer init-rig disconnected fallback

- `summary`: The final/static viewer now reuses `pred_rig_final.json` as a connected-link fallback when `wrong_init_rig.json` lacks explicit `connected_to_parent`. This keeps init UniRig solid/dashed rendering consistent with the predicted rig without touching topology code. Selected-joint edge overlays were also restricted to physical connected edges only.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe scripts\view_run_interactive.py --run-dir mygs/results/connected_branch_smoke_round1/double_knife_one_update_voxelcenter --kind both` regenerated both HTML viewers successfully.

## 2026-05-11 double_knife branch override and denser voxel route

- `summary`: Added an asset-scoped `double_knife` Phase2 override. The left-hand large mixed-fault component now forces `parent_joint=12` before route tracing, and the large right-hand knife component is marked `force_select` so it bypasses the score/global-mass gates and enters the branch proposal list. This does not change the default behavior for other assets.
- `summary`: Added `asset_name` to `Phase1Trainer` from `sample_meta.asset`. For `double_knife`, the effective voxel route resolution is raised from `96` to `192` inside Phase2 voxel-field construction/cache, so thin-blade routes use `pitch≈0.0079` instead of `≈0.0158`.
- `validation`: `D:\Users\namew\miniconda3\envs\mygs\python.exe -m compileall src\evorig_next\phase1_trainer.py src\evorig_next\phase2_topology.py` passed.
- `validation`: `mygs/results/double_knife_forced_parent_round1/phase2_doubleknife_forced12_righthand` accepted three branches in one update: left-hand branch from `parent=12`, head branch from `parent=5`, and right-hand knife branch from `parent=20`. Their physical branch segments report `min_segment_inside_fraction=1.0`, `1.0`, and `1.0`; only the dashed parent-to-root link of the right-hand branch remains partly outside (`parent_link_inside_fraction=0.667`), which is now allowed by design.

## 2026-05-11 prune newly initialized outside-mesh gaussians

- `summary`: Enabled immediate outside-mesh pruning for newly initialized branch/split Gaussians by switching Phase2 `append_axis_gaussians_for_bones(..., prune_outside_mesh=True)`. Also enabled the same behavior for fresh Phase1 initialization through frozen base config keys `phase1_initial_seed_prune_outside_mesh=true` and `phase1_seed_inside_surface_tol=0.003`.
- `diagnostic`: On the current `double_knife` forced-parent run, the problematic `joint45` branch does not have a single runaway endpoint-only Gaussian by accident; its physical bone receives 8 Gaussians with noticeably larger scales than typical branches because the initialization patch itself is wide. By contrast `joint16`'s physical bone has normal-scale Gaussians. This isolates the main scale bug to branch initialization on the `joint45` blade chain, not a generic all-bones issue.

## 2026-05-11 branch-end ring override guard and clean double_knife rerun

- `summary`: Added a branch-safe guard in `cross_section_inner_ring` scale override. Full-mesh section-ring radius now only overrides fallback scale when the Gaussian center is inside the mesh and `0 <= lambda <= 1`. This blocks branch-end seeds outside the physical segment from inheriting a wide full-mesh ring on thin blades.
- `summary`: Promoted `initial_seed_prune_outside_mesh=true` and `densify_seed_prune_outside_mesh=true` into `Phase1Config` defaults and the frozen Phase1 A/B/C YAMLs, so the default route actually uses outside-seed pruning instead of silently falling back to the old `false` dataclass defaults.
- `validation`: `double_knife_phase1_clean_connected_round3` was rerun from scratch under the new defaults. Phase1A removed `8` initial outside seeds and `3` densify seeds (`gaussian_count=553` instead of `564`); Phase1C reached `final_error_raw=0.01600294`, `final_error_raw_all=0.01524421`.
- `validation`: `double_knife_phase2_clean_connected_round3/phase2_from_clean3_c_oneupdate` completed one scheduled update plus `200` post-topology steps and finished at `final_error_raw=0.01429339`. The old runaway `joint45` branch no longer exists; the new right-hand terminal joint is `41`, and its physical bone now has compact scales (`mean≈[0.0204, 0.0086, 0.0022]`, `max≈[0.0205, 0.0119, 0.0030]`). Left-hand terminal `joint36` also remains compact. The remaining issue on this clean line is that only two branches are accepted (left hand and right hand); the head branch no longer survives on the clean rerun.

## 2026-05-11 Claude experiment handoff package

- `summary`: Added [docs/claude_experiment_handoff_zh.md](d:/Evorig/docs/claude_experiment_handoff_zh.md), a focused Claude-facing experiment relay doc. It fixes the active workline, required docs, environment and preflight, one-mesh-at-a-time rule, current default Phase1/2/3 commands, current blockers, and the exact extra-experiment continuation target.
- `scope`: Documentation only. No training code or frozen-config behavior changed in this commit.

## 2026-05-12 Claude ready handoff zip package

- `summary`: Added a dedicated handoff package folder [docs/claude_ready_handoff_20260512](d:/Evorig/docs/claude_ready_handoff_20260512) for Claude Code. The package contains a single entry README, active workline file list, strict environment and no-fallback rules, exact command lines, a current `double_knife` status/blocker note, and a machine-readable manifest. It also bundles snapshot copies of the current active docs under `references/` so the zip can be handed off without asking Claude to reconstruct context from the repo manually.
- `summary`: The package explicitly records that the authoritative `double_knife` sample is `double_knife_f31_60_connected_v4`, while the latest so-called clean Phase1/Phase2 runs still point to `double_knife_f31_60_v1` and therefore are not valid connected-bone baseline evidence.
- `validation`: The package source folder was zipped to `docs/claude_ready_handoff_20260512.zip`. This change is documentation/package-source only and does not modify training code.

## 2026-05-12 rigged UniRig sample connectivity overlay

- `summary`: Fixed the UniRig dynamic importer so `rigged.glb` remains the source of aligned joint positions and skin weights, while `skeleton.fbx` supplies Blender `use_connect` flags. This replaces the broken skeleton-only path where most `double_knife` joints were projected into the mesh without a valid alignment source.
- `validation`: `python -m compileall src\evorig_next\io\unirig_dynamic.py scripts\import_unirig_dynamic_glb_sample.py` passed. Rebuilt `double_knife` as `mygs/demo_data/evorig_unirig_windows_round1/double_knife_f31_60_rigged_connected_v1`; correspondence p95/max are both `0.00357089`, only 2 joints required projection, and the FBX connectivity overlay reports `21` connected and `13` non-connected hierarchy links.

## 2026-05-12 remove skeleton-only UniRig import path

- `summary`: Deleted `scripts/import_unirig_skeleton_dynamic_sample.py` from the active workspace so official UniRig dynamic samples cannot accidentally be built from unaligned `skeleton.fbx` positions or `raw_data.npz`. The only active UniRig dynamic import path is now `scripts/import_unirig_dynamic_glb_sample.py`, which uses `rigged.glb` for aligned joint positions/skin weights and overlays `skeleton.fbx` connectivity.
- `summary`: Updated and rezipped `docs/claude_ready_handoff_20260512.zip` so Claude receives the rigged-connected sample command and treats skeleton-only `double_knife_f31_60_connected_v4` as invalid history.

## 2026-05-12 import-path handoff clarification

- `summary`: Documented the two supported import families: current UniRig dynamic samples use `scripts/import_unirig_dynamic_glb_sample.py` with `rigged.glb` as the aligned joint/skin source plus `skeleton.fbx` connectivity overlay, while ActionMesh/real-GLB assets remain a separate path through `scripts/import_real_glb_sample.py`.
- `summary`: Added `IMPORT_PATHS.md` and `SERVER_UPLOAD.md` to the Claude handoff package, refreshed bundled reference snapshots, and regenerated the package as `docs/claude_ready_handoff_20260512_updated.zip`. The original zip file was locked by another process, so it was not overwritten in this chunk.
