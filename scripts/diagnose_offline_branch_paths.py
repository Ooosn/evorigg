from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import trimesh
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.data import load_sample
from evorig_next.phase1_config import Phase1Config, Phase1DensifyStage
from evorig_next.phase1_trainer import Phase1Trainer
from evorig_next.phase2_topology import (  # type: ignore
    Phase2TopologyConfig,
    _axis_align_branch_path_points,
    _branch_path_points_from_polyline,
    _build_voxel_field,
    _medialize_branch_path_points,
)
from evorig_next.utils.mesh_voxel_path import trace_voxel_parent_paths


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(path_text: str | None, *, base: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_mesh_vertices_faces(run_dir: Path, repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    summary_path = run_dir / "phase2_entry_summary.json"
    data_dir: Path | None = None
    if summary_path.exists():
        summary = _load_json(summary_path)
        data_dir = _resolve_path(summary.get("data_dir"), base=repo_root)

    candidates: list[Path] = []
    if data_dir is not None:
        candidates.extend([data_dir / "rest_mesh.obj", data_dir / "mesh.obj", data_dir / "mesh.glb"])
    candidates.extend([run_dir / "rest_mesh.obj", run_dir / "mesh.obj"])
    for candidate in candidates:
        if not candidate.exists():
            continue
        mesh = trimesh.load(candidate, process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                continue
            mesh = trimesh.util.concatenate(meshes)
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)
    raise FileNotFoundError(f"Could not find rest mesh for {run_dir}")


def _event_list(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("topology_events"), list):
        return [item for item in data["topology_events"] if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        return [item for item in data["events"] if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _points(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 3:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return None
    return arr


def _phase1_config_from_payload(payload: dict[str, Any]) -> Phase1Config:
    payload = dict(payload)
    if isinstance(payload.get("phase1"), dict):
        payload = dict(payload["phase1"])
    stages = []
    for item in payload.pop("densify_stages", []) or []:
        stages.append(
            Phase1DensifyStage(
                warm_steps=int(item["warm_steps"]),
                settle_steps=int(item["settle_steps"]),
                max_bones=int(item["max_bones"]),
                seeds_per_bone=int(item.get("seeds_per_bone", 1)),
            )
        )
    if stages:
        payload["densify_stages"] = stages
    valid = set(Phase1Config.__dataclass_fields__.keys())
    return Phase1Config(**{key: value for key, value in payload.items() if key in valid})


def _phase1_config_for_run(run_dir: Path, repo_root: Path) -> Phase1Config:
    candidates = [run_dir / "phase1_config.json"]
    summary_path = run_dir / "phase2_entry_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        source = _resolve_path(summary.get("source_phase1_config"), base=repo_root)
        if source is not None:
            candidates.append(source)
    candidates.append(repo_root / "configs" / "frozen" / "evorig_next_phase1_final500_supportloss_default.yaml")
    import yaml

    for path in candidates:
        if path is None or not path.exists():
            continue
        if path.suffix.lower() == ".json":
            return _phase1_config_from_payload(_load_json(path))
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return _phase1_config_from_payload(payload)
    return Phase1Config()


def _trainer_for_run(run_dir: Path, repo_root: Path) -> Phase1Trainer | None:
    summary_path = run_dir / "phase2_entry_summary.json"
    if not summary_path.exists():
        return None
    summary = _load_json(summary_path)
    data_dir = _resolve_path(summary.get("data_dir"), base=repo_root)
    state_path = _resolve_path(summary.get("resume_phase1_state"), base=repo_root)
    if data_dir is None or state_path is None or not data_dir.exists() or not state_path.exists():
        return None
    sample = load_sample(data_dir, device="cpu")
    cfg = _phase1_config_for_run(run_dir, repo_root)
    base_config_path = repo_root / "configs" / "frozen" / "evorig_next_base_init_default.yaml"
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) if base_config_path.exists() else {}
    trainer = Phase1Trainer(sample, base_config=base_config, phase1_config=cfg, device=torch.device("cpu"))
    trainer.load_phase1_state(state_path)
    return trainer


def _recompute_no_root_medial_path(
    trainer: Phase1Trainer | None,
    event: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if trainer is None:
        return None, {"enabled": False, "reason": "no_trainer"}
    tip = _points(payload.get("tip"))
    vertex_ids = payload.get("vertex_ids")
    if tip is None or vertex_ids is None:
        return None, {"enabled": False, "reason": "missing_tip_or_vertex_ids"}
    cfg = Phase2TopologyConfig()
    # Diagnostic path: keep voxel route + curvature + medial; remove component-root insertion.
    cfg.branch_component_root_point = False
    cfg.branch_component_root_bridge = False
    cfg.branch_axis_align_path_points = False
    cfg.branch_axis_align_after_medialization = False
    cfg.branch_align_seed_leaf_parent = False
    parent_joint = int(event.get("parent_joint", payload.get("parent_joint", -1)))
    voxel_field = _build_voxel_field(trainer, cfg)
    if voxel_field is None:
        return None, {"enabled": False, "reason": "no_voxel_field"}
    rest_joints = trainer.skeleton.rest_joints.detach()
    if not (0 <= parent_joint < int(rest_joints.shape[0])):
        return None, {"enabled": False, "reason": "invalid_parent"}
    # Reuse recorded parent to avoid parent-selection differences; trace only this parent.
    ranking = trace_voxel_parent_paths(
        query_point=torch.tensor(tip[0], dtype=rest_joints.dtype),
        joint_positions=rest_joints,
        field=voxel_field,
        candidate_joint_ids=torch.tensor([parent_joint], dtype=torch.long),
    )
    path_info = ranking[0] if ranking else None
    if not isinstance(path_info, dict) or not isinstance(path_info.get("polyline"), torch.Tensor):
        return None, {"enabled": False, "reason": "no_polyline"}
    polyline = path_info["polyline"].to(dtype=rest_joints.dtype)
    path = _branch_path_points_from_polyline(polyline, cfg)
    vertex_tensor = torch.tensor(vertex_ids, dtype=torch.long)
    path_medial, medial_info = _medialize_branch_path_points(
        path,
        vertex_tensor,
        trainer,
        voxel_field,
        cfg,
        locked_prefix_count=0,
    )
    return path_medial.detach().cpu().numpy().astype(np.float32), {
        "enabled": True,
        "parent_joint": parent_joint,
        "polyline_points": int(polyline.reshape(-1, 3).shape[0]),
        "curvature_points": int(path.reshape(-1, 3).shape[0]),
        "medial_points": int(path_medial.reshape(-1, 3).shape[0]),
        "path_length": float(path_info.get("path_length", 0.0)),
        "mean_clearance": float(path_info.get("mean_clearance", 0.0)),
        "medialization": medial_info,
    }


def _angle_stats(points: np.ndarray | None) -> dict[str, float]:
    if points is None or points.shape[0] < 3:
        return {"max_turn_deg": 0.0, "mean_turn_deg": 0.0}
    turns: list[float] = []
    for idx in range(1, points.shape[0] - 1):
        a = points[idx] - points[idx - 1]
        b = points[idx + 1] - points[idx]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1.0e-12:
            continue
        cosv = float(np.dot(a, b) / denom)
        cosv = max(-1.0, min(1.0, cosv))
        turns.append(float(np.degrees(np.arccos(cosv))))
    if not turns:
        return {"max_turn_deg": 0.0, "mean_turn_deg": 0.0}
    return {"max_turn_deg": float(max(turns)), "mean_turn_deg": float(np.mean(turns))}


def _line_trace(points: np.ndarray | None, *, name: str, color: str, width: int = 8) -> go.Scatter3d | None:
    if points is None:
        return None
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines+markers",
        name=name,
        line={"color": color, "width": width},
        marker={"size": 5, "color": color},
    )


def _point_trace(points: np.ndarray | None, *, name: str, color: str, size: int = 8, symbol: str = "diamond") -> go.Scatter3d | None:
    if points is None:
        return None
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers+text",
        name=name,
        marker={"size": size, "color": color, "symbol": symbol},
        text=[name] * points.shape[0],
        textposition="top center",
    )


def _mesh_trace(vertices: np.ndarray, faces: np.ndarray) -> go.Mesh3d:
    max_faces = 9000
    if faces.shape[0] > max_faces:
        step = int(np.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="rgba(210,220,230,0.28)",
        opacity=0.24,
        name="rest mesh",
        showscale=False,
        flatshading=True,
    )


def _branch_payload(event: dict[str, Any]) -> dict[str, Any]:
    proposal = event.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    return event


def build_figure(run_dir: Path, repo_root: Path, title: str) -> tuple[go.Figure, list[dict[str, Any]]]:
    vertices, faces = _load_mesh_vertices_faces(run_dir, repo_root)
    trainer = _trainer_for_run(run_dir, repo_root)
    events = [event for event in _event_list(run_dir / "topology_events.json") if event.get("type") == "branch"]
    traces: list[Any] = [_mesh_trace(vertices, faces)]
    rows: list[dict[str, Any]] = []

    palette = [
        ("#2563eb", "#ef4444", "#f59e0b"),
        ("#0ea5e9", "#dc2626", "#f97316"),
        ("#22c55e", "#be123c", "#eab308"),
        ("#8b5cf6", "#f43f5e", "#fb923c"),
    ]
    for idx, event in enumerate(events):
        payload = _branch_payload(event)
        final_path = _points(payload.get("branch_path_points"))
        curvature_path = _points(payload.get("branch_path_points_curvature"))
        raw_path = _points(payload.get("branch_path_points_raw"))
        pre_refine = _points(payload.get("branch_path_points_pre_refine"))
        no_root_medial, no_root_info = _recompute_no_root_medial_path(trainer, event, payload)
        tip = _points(payload.get("tip"))
        center = _points(payload.get("center"))

        parent_pos = None
        parent_joint = event.get("parent_joint", payload.get("parent_joint"))
        root_info = payload.get("branch_component_root")
        component_root = None
        if isinstance(root_info, dict):
            component_root = _points(root_info.get("point"))

        colors = palette[idx % len(palette)]
        for trace in [
            _line_trace(raw_path, name=f"B{idx} raw/voxel", color="#9ca3af", width=4),
            _line_trace(pre_refine, name=f"B{idx} pre-refine", color="#a855f7", width=5),
            _line_trace(curvature_path, name=f"B{idx} no-root curvature", color=colors[0], width=9),
            _line_trace(no_root_medial, name=f"B{idx} no-root medial", color="#06b6d4", width=10),
            _line_trace(final_path, name=f"B{idx} current final", color=colors[1], width=7),
            _point_trace(component_root, name=f"B{idx} component_root", color=colors[2], size=9, symbol="diamond"),
            _point_trace(tip, name=f"B{idx} tip", color="#111827", size=8, symbol="circle"),
            _point_trace(center, name=f"B{idx} component_center", color="#64748b", size=5, symbol="circle"),
        ]:
            if trace is not None:
                traces.append(trace)

        inside = payload.get("branch_path_inside") if isinstance(payload.get("branch_path_inside"), dict) else {}
        row = {
            "branch_index": idx,
            "step": event.get("step"),
            "parent_joint": parent_joint,
            "new_joints": event.get("new_joints"),
            "component_index": payload.get("component_index"),
            "vertex_count": payload.get("vertex_count"),
            "wrong_fraction": payload.get("wrong_fraction"),
            "uncovered_fraction": payload.get("uncovered_fraction"),
            "final_points": int(final_path.shape[0]) if final_path is not None else 0,
            "curvature_points": int(curvature_path.shape[0]) if curvature_path is not None else 0,
            "final_max_turn_deg": _angle_stats(final_path)["max_turn_deg"],
            "curvature_max_turn_deg": _angle_stats(curvature_path)["max_turn_deg"],
            "no_root_medial_points": int(no_root_medial.shape[0]) if no_root_medial is not None else 0,
            "no_root_medial_max_turn_deg": _angle_stats(no_root_medial)["max_turn_deg"],
            "no_root_medial_info": no_root_info,
            "min_segment_inside_fraction": inside.get("min_segment_inside_fraction"),
            "component_root_enabled": bool(isinstance(root_info, dict) and root_info.get("enabled")),
            "component_root_bridge": payload.get("branch_component_root_bridge"),
        }
        rows.append(row)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    return fig, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline branch path diagnostics from existing Phase2 topology events.")
    parser.add_argument("--run-dir", action="append", required=True, help="Phase2 run directory containing topology_events.json.")
    parser.add_argument("--name", action="append", default=[], help="Display name for each run-dir.")
    parser.add_argument("--out-dir", default="mygs/visuals/offline_branch_path_diagnostics")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for idx, run_text in enumerate(args.run_dir):
        run_dir = Path(run_text).resolve()
        name = args.name[idx] if idx < len(args.name) else run_dir.name
        fig, rows = build_figure(run_dir, repo_root, name)
        html_path = out_dir / f"{name}_branch_paths.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        for row in rows:
            row["name"] = name
            row["run_dir"] = str(run_dir)
            row["html"] = str(html_path)
        all_rows.extend(rows)
        print(f"[ok] {name}: {html_path} branches={len(rows)}")

    summary_path = out_dir / "branch_path_summary.json"
    summary_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
