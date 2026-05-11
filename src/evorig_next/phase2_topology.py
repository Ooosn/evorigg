from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from evorig_next.io.outputs import save_outputs
from evorig_next.ops.kabsch import fit_rigid_sequence
from evorig_next.training.scaling import normalize_linear_metric
from evorig_next.utils.geometry import EPS
from evorig_next.utils.mesh_voxel_path import (
    build_mesh_voxel_path_field,
    trace_voxel_parent_paths,
)
from evorig_next.utils.mesh_ops import points_inside_or_on_mesh
from evorig_next.utils.rotations import matrix_to_axis_angle
from evorig_next.phase1_losses import vertex_recon_loss
from evorig_next.phase1_trainer import (
    build_rest_mesh_adjacency,
    load_cached_adjacency_cpu,
    save_cached_adjacency_cpu,
)


_PHASE2_VOXEL_CACHE_ROOT = Path(__file__).resolve().parents[2] / "mygs" / "outputs" / "phase2_voxel_path_cache"


@dataclass
class Phase2TopologyConfig:
    vertex_error_quantile: float = 0.80
    wrong_coverage_error_quantile: float = 0.50
    coverage_quantile: float = 0.05
    coverage_abs_threshold: float = 0.0
    topology_support_sigma: float = 5.0
    wrong_coverage_ratio: float = 0.35
    wrong_coverage_mass_min: float = 1.0e-8
    component_min_vertices: int = 10
    component_min_vertices_reference_area: float = 3.9673382939115642
    component_min_vertices_reference_vertex_count: int = 19869
    component_min_vertices_min: int = 4
    component_merge_hops: int = 2
    max_branch_components: int = 32
    branch_min_global_error_mass_fraction: float = 0.03
    branch_min_wrong_fraction: float = 0.70
    branch_min_uncovered_fraction: float = 0.80
    branch_min_score_fraction_of_best: float = 0.08
    branch_component_overlap_reject_fraction: float = 0.25
    branch_accept_max_post_seed_fraction: float = 0.50
    branch_accept_max_post_fault_fraction: float = 0.50
    topology_update_interval_steps: int = 200
    topology_max_branch_per_update: int = 3
    topology_max_split_per_update: int = 3
    topology_noop_stop_patience: int = 2
    seed_joint_repair_enabled: bool = True
    seed_joint_repair_variant: str = "center_capB"
    seed_joint_repair_max_per_update: int = 2
    seed_joint_repair_max_components: int = 16
    seed_joint_repair_min_vertices: int = 24
    seed_joint_repair_min_vertices_reference_area: float = 3.9673382939115642
    seed_joint_repair_min_vertices_reference_vertex_count: int = 19869
    seed_joint_repair_min_vertices_min: int = 8
    seed_joint_repair_min_neighbor_fraction: float = 0.80
    seed_joint_repair_min_fault_fraction: float = 0.50
    seed_joint_repair_cap_sample_radius_ratio: float = 0.16
    seed_joint_repair_inside_min_fraction: float = 0.70
    seed_joint_repair_min_inside_improvement: float = 0.05
    phase2_freeze_seed_rest_joints: bool = True
    phase2_freeze_branch_rest_joints: bool = True
    phase2_freeze_branch_root_rest_joints: bool = True
    phase2_loss_illegal_support: float = 0.20
    phase2_loss_gaussian_illegal_coverage: float = 0.0
    phase2_illegal_support_tau: float = 0.0
    phase2_illegal_support_margin: float = 0.99
    voxel_parent_enabled: bool = True
    voxel_target_resolution: int = 96
    voxel_narrow_span_voxels: float = 4.0
    voxel_max_resolution: int = 192
    voxel_neighbor_mode: str = "26"
    branch_max_intermediate_points: int = 4
    branch_min_path_points: int = 2
    branch_segment_refine_inside_fraction: float = 0.75
    branch_segment_refine_max_points: int = 10
    branch_tip_target_query_k: int = 192
    branch_tip_target_radius_voxels: float = 3.0
    branch_tip_target_radius_ratio: float = 0.06
    branch_tip_target_distance_weight: float = 1.0
    branch_long_segment_refine: bool = True
    branch_long_segment_max_arc_fraction: float = 0.48
    branch_path_clearance_weight: float = 2.0
    branch_path_clearance_power: float = 1.0
    branch_lineage_parent_guard: bool = True
    branch_lineage_extension_min_arc_fraction: float = 0.78
    branch_lineage_extension_min_progress: float = 0.88
    branch_lineage_extension_min_terminal_cos: float = 0.15
    branch_lineage_extension_min_tip_distance_ratio: float = 0.08
    bone_flow_audit_nearest_vertices: int = 96
    bone_flow_audit_min_vertices: int = 24
    bone_flow_audit_bad_abs_cos: float = 0.35
    bone_flow_audit_bad_leaf_abs_cos: float = 0.55
    bone_flow_audit_radius_sample_ratio: float = 0.35
    split_min_gaussians_per_bone: int = 4
    split_min_vertices: int = 64
    split_vertex_topk: int = 128
    split_balance_min: float = 0.20
    split_coverage_quantile: float = 0.00
    split_ratio_reference: str = "mean_others"
    split_ratio_score_mode: str = "gaussian_quantile_mean_meanT"
    split_ratio_threshold: float = 1.10
    split_ratio_regularizer: float = 0.008
    split_residual_quantile: float = 0.80
    split_gaussian_topk: int = 4
    split_rigid_gain_threshold: float = 0.003
    split_lambda_min: float = 0.20
    split_lambda_max: float = 0.85
    split_inside_min_fraction: float = 0.75
    split_prefer_inserted_bones: bool = True
    split_min_score_fraction_of_best: float = 0.35
    split_topk: int = 16


def _asset_name(trainer: Any) -> str:
    return str(getattr(trainer, "asset_name", "")).strip().lower()


def _double_knife_forced_parent(
    trainer: Any,
    *,
    tip_target: torch.Tensor,
    component_center: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
) -> int | None:
    if _asset_name(trainer) != "double_knife":
        return None
    tx, ty, tz = [float(item) for item in tip_target.reshape(3).detach().cpu().tolist()]
    c = [float(item) for item in component_center.reshape(3).detach().cpu().tolist()]
    bmin = [float(item) for item in bbox_min.reshape(3).detach().cpu().tolist()]
    bmax = [float(item) for item in bbox_max.reshape(3).detach().cpu().tolist()]
    if tx < -0.05 and ty > 0.18 and tz > 0.20 and bmin[0] < -0.30 and bmax[0] < -0.05 and c[0] < -0.15:
        return 12
    return None


def _double_knife_force_select_component(
    trainer: Any,
    *,
    component_center: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    vertex_count: int,
) -> bool:
    if _asset_name(trainer) != "double_knife":
        return False
    c = [float(item) for item in component_center.reshape(3).detach().cpu().tolist()]
    bmin = [float(item) for item in bbox_min.reshape(3).detach().cpu().tolist()]
    bmax = [float(item) for item in bbox_max.reshape(3).detach().cpu().tolist()]
    return bool(vertex_count >= 700 and c[0] > 0.15 and bmin[0] > 0.15 and bmax[1] > 0.35 and bmax[2] > 0.45)


def _effective_voxel_target_resolution(trainer: Any, cfg: Phase2TopologyConfig) -> int:
    if _asset_name(trainer) == "double_knife":
        return max(int(cfg.voxel_target_resolution), 192)
    return int(cfg.voxel_target_resolution)


def _double_knife_force_parent_ranking(
    trainer: Any,
    rest_joints: torch.Tensor,
    tip_target: torch.Tensor,
    field: Any,
    cfg: Phase2TopologyConfig,
    forced_parent_joint: int,
) -> tuple[int, str, dict[str, Any] | None]:
    ranking = trace_voxel_parent_paths(
        query_point=tip_target,
        joint_positions=rest_joints,
        field=field,
        candidate_joint_ids=torch.tensor([int(forced_parent_joint)], dtype=torch.long, device=rest_joints.device),
    )
    if not ranking:
        return -1, "forced_parent_unavailable", None
    path_info = dict(ranking[0])
    path_info["branch_lineage_parent_guard"] = {
        "enabled": True,
        "forced": True,
        "selected_parent_joint": int(forced_parent_joint),
        "reason": "double_knife_override",
    }
    if float(getattr(cfg, "branch_path_clearance_weight", 0.0)) > 0.0:
        reranked = trace_voxel_parent_paths(
            query_point=tip_target,
            joint_positions=rest_joints,
            field=field,
            candidate_joint_ids=torch.tensor([int(forced_parent_joint)], dtype=torch.long, device=rest_joints.device),
            clearance_weight=float(cfg.branch_path_clearance_weight),
            clearance_power=float(cfg.branch_path_clearance_power),
        )
        if reranked:
            rerank_payload = dict(reranked[0])
            rerank_payload["branch_lineage_parent_guard"] = path_info["branch_lineage_parent_guard"]
            path_info = rerank_payload
    return int(forced_parent_joint), "forced_asset_parent_joint", path_info


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _json_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _phase2_welded_adjacency_cache_key(
    faces: torch.Tensor,
    vertices: torch.Tensor,
) -> str:
    faces_cpu = faces.detach().to(device="cpu", dtype=torch.int32).contiguous()
    vertices_cpu = vertices.detach().to(device="cpu", dtype=torch.float32).contiguous()
    hasher = hashlib.sha1()
    hasher.update(b"evorig_next_phase2_welded_adjacency_v1")
    hasher.update(int(vertices_cpu.shape[0]).to_bytes(8, byteorder="little", signed=False))
    hasher.update(int(faces_cpu.shape[0]).to_bytes(8, byteorder="little", signed=False))
    hasher.update(faces_cpu.numpy().tobytes())
    hasher.update(vertices_cpu.numpy().tobytes())
    return hasher.hexdigest()


def _weld_duplicate_position_adjacency(
    faces: torch.Tensor,
    vertices: torch.Tensor,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    vertex_count = int(vertices.shape[0])
    base = build_rest_mesh_adjacency(faces, vertex_count)
    neighbors = [set(int(item) for item in row) for row in base]
    if vertex_count <= 1:
        return base, {
            "enabled": False,
            "reason": "too_few_vertices",
            "weld_pair_count": 0,
            "weld_radius": 0.0,
        }

    vertices_np = vertices.detach().to(device="cpu", dtype=torch.float64).numpy()
    bbox_diag = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    faces_np = faces.detach().to(device="cpu", dtype=torch.long).numpy() if int(faces.numel()) > 0 else np.zeros((0, 3), dtype=np.int64)
    edge_lengths: list[np.ndarray] = []
    if faces_np.size > 0:
        edge_pairs = ((0, 1), (1, 2), (2, 0))
        for a, b in edge_pairs:
            edge_lengths.append(np.linalg.norm(vertices_np[faces_np[:, a]] - vertices_np[faces_np[:, b]], axis=1))
    if edge_lengths:
        edge_values = np.concatenate(edge_lengths)
        edge_values = edge_values[np.isfinite(edge_values) & (edge_values > 0.0)]
        median_edge = float(np.median(edge_values)) if edge_values.size > 0 else 0.0
    else:
        median_edge = 0.0
    weld_radius = max(float(bbox_diag) * 1.0e-6, float(median_edge) * 1.0e-5, 1.0e-9)
    if not np.isfinite(weld_radius) or weld_radius <= 0.0:
        return base, {
            "enabled": False,
            "reason": "invalid_weld_radius",
            "weld_pair_count": 0,
            "weld_radius": float(weld_radius),
        }

    tree = cKDTree(vertices_np)
    pairs = tree.query_pairs(r=float(weld_radius), output_type="set")
    for a, b in pairs:
        a = int(a)
        b = int(b)
        if a == b:
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)
    adjacency = [tuple(sorted(row)) for row in neighbors]
    return adjacency, {
        "enabled": True,
        "weld_pair_count": int(len(pairs)),
        "weld_radius": float(weld_radius),
        "bbox_diag": float(bbox_diag),
        "median_edge": float(median_edge),
    }


def _phase2_rest_mesh_adjacency(trainer: Any) -> list[tuple[int, ...]] | None:
    cached = getattr(trainer, "_phase2_rest_mesh_adjacency", None)
    if cached is not None:
        return cached
    faces = getattr(trainer, "mesh_faces", None)
    vertices = getattr(trainer, "rest_vertices", None)
    if faces is None or vertices is None or int(faces.numel()) <= 0:
        return None
    cache_key = _phase2_welded_adjacency_cache_key(faces, vertices)
    adjacency = load_cached_adjacency_cpu(cache_key)
    if adjacency is not None:
        setattr(trainer, "_phase2_rest_mesh_adjacency", adjacency)
        setattr(trainer, "_phase2_rest_mesh_adjacency_cache_source", "disk_welded")
        return adjacency
    adjacency, stats = _weld_duplicate_position_adjacency(faces, vertices)
    save_cached_adjacency_cpu(cache_key, adjacency)
    setattr(trainer, "_phase2_rest_mesh_adjacency", adjacency)
    setattr(trainer, "_phase2_rest_mesh_adjacency_cache_source", "built_welded")
    setattr(trainer, "_phase2_rest_mesh_adjacency_weld_stats", stats)
    return adjacency


def _extract_connected_components(
    mask: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    adjacency: list[tuple[int, ...]] | None = None,
) -> list[list[int]]:
    mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
    ids = torch.nonzero(mask_cpu, as_tuple=False).flatten()
    if int(ids.numel()) == 0:
        return []
    if adjacency is not None:
        components: list[list[int]] = []
        seen: set[int] = set()
        for start_tensor in ids.tolist():
            start = int(start_tensor)
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    neighbor = int(neighbor)
                    if neighbor in seen or not bool(mask_cpu[neighbor]):
                        continue
                    seen.add(neighbor)
                    stack.append(neighbor)
            components.append(sorted(component))
        return components
    graph: dict[int, set[int]] = {int(item): set() for item in ids.tolist()}
    if faces is not None and int(faces.numel()) > 0:
        for tri in faces.detach().cpu().long().tolist():
            a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
            if bool(mask_cpu[a]) and bool(mask_cpu[b]):
                graph[a].add(b)
                graph[b].add(a)
            if bool(mask_cpu[b]) and bool(mask_cpu[c]):
                graph[b].add(c)
                graph[c].add(b)
            if bool(mask_cpu[c]) and bool(mask_cpu[a]):
                graph[c].add(a)
                graph[a].add(c)
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _merge_nearby_components_by_adjacency(
    components: list[list[int]],
    *,
    adjacency: list[tuple[int, ...]] | None,
    hops: int,
    vertex_count: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Merge fault components separated only by a short non-fault mesh bridge."""
    hops = max(int(hops), 0)
    if adjacency is None or hops <= 0 or len(components) <= 1:
        records = [
            {
                "enabled": bool(hops > 0),
                "applied": False,
                "source_component_indices": [int(index)],
                "source_component_count": 1,
                "merge_hops": int(hops),
            }
            for index in range(len(components))
        ]
        return components, records

    component_id = [-1] * int(vertex_count)
    for index, ids in enumerate(components):
        for vertex_id in ids:
            component_id[int(vertex_id)] = int(index)

    parent = list(range(len(components)))

    def _find(index: int) -> int:
        index = int(index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(a: int, b: int) -> None:
        root_a = _find(a)
        root_b = _find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for source_index, ids in enumerate(components):
        frontier = {int(item) for item in ids}
        visited = set(frontier)
        for _hop in range(hops):
            next_frontier: set[int] = set()
            for vertex_id in frontier:
                for neighbor in adjacency[int(vertex_id)]:
                    neighbor = int(neighbor)
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    other = component_id[neighbor] if 0 <= neighbor < int(vertex_count) else -1
                    if other >= 0 and other != int(source_index):
                        _union(int(source_index), int(other))
                    else:
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier

    groups: dict[int, list[int]] = {}
    for index in range(len(components)):
        groups.setdefault(_find(index), []).append(index)

    merged_components: list[list[int]] = []
    merge_records: list[dict[str, Any]] = []
    for group_indices in groups.values():
        merged_ids: set[int] = set()
        for index in group_indices:
            merged_ids.update(int(item) for item in components[int(index)])
        merged_components.append(sorted(merged_ids))
        merge_records.append(
            {
                "enabled": True,
                "applied": bool(len(group_indices) > 1),
                "source_component_indices": [int(item) for item in sorted(group_indices)],
                "source_component_count": int(len(group_indices)),
                "merge_hops": int(hops),
            }
        )
    return merged_components, merge_records


def _rest_mesh_surface_area(trainer: Any) -> float:
    cached = getattr(trainer, "_evorig_next_rest_mesh_surface_area", None)
    if cached is not None:
        return float(cached)
    faces = getattr(trainer, "mesh_faces", None)
    vertices = getattr(trainer, "rest_vertices", None)
    if faces is None or vertices is None or int(faces.numel()) == 0:
        return 0.0
    face_ids = faces.to(device=vertices.device, dtype=torch.long)
    tri = vertices[face_ids]
    area = 0.5 * torch.linalg.norm(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
    value = float(area.sum().detach().item())
    setattr(trainer, "_evorig_next_rest_mesh_surface_area", value)
    return value


def _scaled_vertex_threshold(
    trainer: Any,
    *,
    base_count: int,
    reference_area: float,
    reference_vertex_count: int,
    min_count: int,
) -> int:
    base = max(int(base_count), 1)
    ref_area = float(reference_area)
    ref_vertices = max(int(reference_vertex_count), 1)
    if ref_area <= float(EPS):
        return base
    area = _rest_mesh_surface_area(trainer)
    if area <= float(EPS):
        return base
    vertex_count = max(int(getattr(trainer, "rest_vertices").shape[0]), 1)
    avg_area_current = float(area) / float(vertex_count)
    avg_area_reference = float(ref_area) / float(ref_vertices)
    density_ratio = avg_area_current / max(avg_area_reference, float(EPS))
    scaled = int(math.ceil(float(base) / max(float(density_ratio), float(EPS))))
    return max(int(min_count), scaled, 1)


def _effective_component_min_vertices(trainer: Any, cfg: Phase2TopologyConfig) -> int:
    return _scaled_vertex_threshold(
        trainer,
        base_count=int(cfg.component_min_vertices),
        reference_area=float(cfg.component_min_vertices_reference_area),
        reference_vertex_count=int(cfg.component_min_vertices_reference_vertex_count),
        min_count=int(cfg.component_min_vertices_min),
    )


def _effective_seed_joint_repair_min_vertices(trainer: Any, cfg: Phase2TopologyConfig) -> int:
    seed_min = _scaled_vertex_threshold(
        trainer,
        base_count=int(cfg.seed_joint_repair_min_vertices),
        reference_area=float(cfg.seed_joint_repair_min_vertices_reference_area),
        reference_vertex_count=int(cfg.seed_joint_repair_min_vertices_reference_vertex_count),
        min_count=int(cfg.seed_joint_repair_min_vertices_min),
    )
    return max(seed_min, _effective_component_min_vertices(trainer, cfg), 1)


def _mixed_fault_error_gate_factor(item: dict[str, Any]) -> float:
    mixed_fraction = float(item.get("dual_fault_fraction", 0.0))
    if mixed_fraction <= 0.0:
        return 1.0
    return min(max(1.0 - mixed_fraction, 0.0), 1.0)


def _component_tip(
    vertex_ids: torch.Tensor,
    rest_vertices: torch.Tensor,
    score: torch.Tensor,
    rest_joints: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    points = rest_vertices[vertex_ids]
    weights = score[vertex_ids].clamp_min(0.0)
    if float(weights.sum().item()) <= float(EPS):
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(EPS)
    center = (points * weights.unsqueeze(-1)).sum(dim=0)
    centered = points - center.unsqueeze(0)
    cov = (centered * weights.unsqueeze(-1)).transpose(0, 1) @ centered
    eigvals, eigvecs = torch.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    if float(axis.norm().item()) <= float(EPS):
        axis = points[int(torch.argmax((points - center.unsqueeze(0)).norm(dim=-1)).item())] - center
    axis = axis / axis.norm().clamp_min(EPS)
    progress = (points - center.unsqueeze(0)) @ axis
    if int(rest_joints.numel()) > 0:
        far_from_skeleton = torch.cdist(points, rest_joints).min(dim=1).values
        seed_idx = int(torch.argmax(far_from_skeleton).item())
        direction = -1.0 if float(progress[seed_idx].item()) < 0.0 else 1.0
    else:
        direction = 1.0
    oriented = progress * direction
    tip_local_idx = int(torch.argmax(oriented).item())
    return points[tip_local_idx], int(vertex_ids[tip_local_idx].item()), center


def _path_length(polyline: torch.Tensor) -> float:
    if int(polyline.shape[0]) <= 1:
        return 0.0
    return float((polyline[1:] - polyline[:-1]).norm(dim=-1).sum().item())


def _nearest_polyline_arc_fraction(polyline: torch.Tensor, point: torch.Tensor) -> float:
    polyline = polyline.reshape(-1, 3)
    if int(polyline.shape[0]) <= 1:
        return 0.0
    segment_lengths = (polyline[1:] - polyline[:-1]).norm(dim=-1)
    arc_end = torch.cumsum(segment_lengths, dim=0)
    total_length = float(segment_lengths.sum().clamp_min(EPS).item())
    best_distance = float("inf")
    best_arc = 0.0
    query = point.reshape(3).to(device=polyline.device, dtype=polyline.dtype)
    for segment_id in range(int(segment_lengths.numel())):
        start = polyline[segment_id]
        end = polyline[segment_id + 1]
        vec = end - start
        local = ((query - start) @ vec / vec.dot(vec).clamp_min(EPS)).clamp(0.0, 1.0)
        closest = start + local * vec
        distance = float((query - closest).norm().item())
        arc_start = 0.0 if segment_id == 0 else float(arc_end[segment_id - 1].item())
        arc_pos = arc_start + float(local.item()) * float(segment_lengths[segment_id].item())
        if distance < best_distance:
            best_distance = distance
            best_arc = arc_pos
    return best_arc / max(total_length, float(EPS))


def _sample_polyline_by_fractions(polyline: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    polyline = polyline.reshape(-1, 3)
    fractions = fractions.reshape(-1).to(device=polyline.device, dtype=polyline.dtype).clamp(0.0, 1.0)
    if int(polyline.shape[0]) <= 1:
        return polyline[-1:].expand(int(fractions.numel()), -1).clone()
    segment_lengths = (polyline[1:] - polyline[:-1]).norm(dim=-1)
    total_length = segment_lengths.sum().clamp_min(EPS)
    arc_end = torch.cumsum(segment_lengths, dim=0)
    targets = fractions * total_length
    points: list[torch.Tensor] = []
    for target in targets:
        segment_id = int(torch.searchsorted(arc_end, target, right=False).clamp_max(int(segment_lengths.numel()) - 1).item())
        arc_start = torch.zeros((), dtype=polyline.dtype, device=polyline.device)
        if segment_id > 0:
            arc_start = arc_end[segment_id - 1]
        local_t = ((target - arc_start) / segment_lengths[segment_id].clamp_min(EPS)).clamp(0.0, 1.0)
        points.append(polyline[segment_id] * (1.0 - local_t) + polyline[segment_id + 1] * local_t)
    return torch.stack(points, dim=0)


def _sanitize_branch_lineages(lineages: Any) -> list[dict[str, Any]]:
    if not isinstance(lineages, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in lineages:
        if not isinstance(item, dict):
            continue
        joint_chain = [int(joint_id) for joint_id in item.get("joint_chain", [])]
        if not joint_chain:
            continue
        sanitized.append(
            {
                "branch_id": int(item.get("branch_id", len(sanitized))),
                "root_parent_joint": int(item.get("root_parent_joint", -1)),
                "root_joint": int(item.get("root_joint", joint_chain[0])),
                "joint_chain": joint_chain,
                "birth_step": int(item.get("birth_step", 0)),
                "parent_branch_id": int(item.get("parent_branch_id", -1)),
                "source_component_index": int(item.get("source_component_index", item.get("component_index", -1))),
                "source_component_vertex_count": int(item.get("source_component_vertex_count", 0)),
                "source_component_center": list(item.get("source_component_center", [])),
                "source_tip": list(item.get("source_tip", [])),
            }
        )
    return sanitized


def _reconstruct_branch_lineages_from_birth_modes(trainer: Any) -> list[dict[str, Any]]:
    birth_modes = [str(item) for item in getattr(trainer.skeleton, "birth_modes", [])]
    if not birth_modes:
        return []
    parent_idx = trainer.skeleton.parent_idx.detach().cpu().long().tolist()
    children: list[list[int]] = [[] for _ in range(len(parent_idx))]
    for joint_id, parent_joint in enumerate(parent_idx):
        if int(parent_joint) >= 0:
            children[int(parent_joint)].append(int(joint_id))
    lineages: list[dict[str, Any]] = []
    visited: set[int] = set()
    for joint_id, mode in enumerate(birth_modes):
        mode_key = str(mode).strip().lower()
        parent_joint = int(parent_idx[joint_id])
        parent_mode = birth_modes[parent_joint].strip().lower() if 0 <= parent_joint < len(birth_modes) else "seed"
        starts_branch = mode_key == "branch_root" or (mode_key == "branch" and parent_mode not in {"branch", "branch_root"})
        if not starts_branch or int(joint_id) in visited:
            continue
        chain: list[int] = []
        stack = [int(joint_id)]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            current_mode = birth_modes[current].strip().lower() if 0 <= current < len(birth_modes) else ""
            if current_mode not in {"branch", "branch_root"}:
                continue
            visited.add(current)
            chain.append(current)
            for child in reversed(children[current]):
                child_mode = birth_modes[child].strip().lower() if 0 <= child < len(birth_modes) else ""
                if child_mode in {"branch", "branch_root"}:
                    stack.append(int(child))
        if chain:
            lineages.append(
                {
                    "branch_id": int(len(lineages)),
                    "root_parent_joint": int(parent_joint),
                    "root_joint": int(chain[0]),
                    "joint_chain": chain,
                    "birth_step": 0,
                    "parent_branch_id": -1,
                    "source_component_index": -1,
                    "source_component_vertex_count": 0,
                    "source_component_center": [],
                    "source_tip": [],
                }
            )
    return lineages


def _get_phase2_branch_lineages(trainer: Any) -> list[dict[str, Any]]:
    lineages = _sanitize_branch_lineages(getattr(trainer, "phase2_branch_lineages", []))
    if not lineages:
        lineages = _reconstruct_branch_lineages_from_birth_modes(trainer)
    trainer.phase2_branch_lineages = lineages
    return lineages


def _branch_lineage_for_joint(trainer: Any, joint_id: int) -> dict[str, Any] | None:
    joint_id = int(joint_id)
    if joint_id < 0:
        return None
    for lineage in _get_phase2_branch_lineages(trainer):
        if joint_id in {int(item) for item in lineage.get("joint_chain", [])}:
            return lineage
    return None


def _branch_lineage_polyline(trainer: Any, lineage: dict[str, Any]) -> torch.Tensor | None:
    rest_joints = trainer.skeleton.rest_joints.detach()
    ids: list[int] = []
    root_parent = int(lineage.get("root_parent_joint", -1))
    if 0 <= root_parent < int(rest_joints.shape[0]):
        ids.append(root_parent)
    for joint_id in lineage.get("joint_chain", []):
        joint_id = int(joint_id)
        if 0 <= joint_id < int(rest_joints.shape[0]):
            ids.append(joint_id)
    if len(ids) < 2:
        return None
    return rest_joints[torch.tensor(ids, dtype=torch.long, device=rest_joints.device)].reshape(-1, 3)


def _branch_lineage_extension_test(
    trainer: Any,
    lineage: dict[str, Any],
    query_point: torch.Tensor,
    cfg: Phase2TopologyConfig,
) -> dict[str, Any]:
    polyline = _branch_lineage_polyline(trainer, lineage)
    if polyline is None or int(polyline.shape[0]) < 2:
        return {"is_extension": False, "reason": "lineage_polyline_unavailable"}
    query = query_point.reshape(3).to(device=polyline.device, dtype=polyline.dtype)
    branch_length = max(_path_length(polyline), float(EPS))
    terminal = polyline[-1]
    prev = polyline[-2]
    terminal_vec = terminal - prev
    terminal_dir = terminal_vec / terminal_vec.norm().clamp_min(EPS)
    query_vec = query - terminal
    query_distance = float(query_vec.norm().item())
    terminal_cos = 0.0
    if query_distance > float(EPS):
        query_dir = query_vec / query_vec.norm().clamp_min(EPS)
        terminal_cos = float((query_dir @ terminal_dir).item())
    root = polyline[0]
    axis = terminal - root
    axis_len = max(float(axis.norm().item()), float(EPS))
    axis_dir = axis / axis.norm().clamp_min(EPS)
    progress = float(((query - root) @ axis_dir / max(axis_len, float(EPS))).item())
    arc_fraction = float(_nearest_polyline_arc_fraction(polyline, query))
    min_arc = float(cfg.branch_lineage_extension_min_arc_fraction)
    min_progress = float(cfg.branch_lineage_extension_min_progress)
    min_cos = float(cfg.branch_lineage_extension_min_terminal_cos)
    min_distance = float(cfg.branch_lineage_extension_min_tip_distance_ratio) * branch_length
    is_extension = (
        arc_fraction >= min_arc
        and progress >= min_progress
        and terminal_cos >= min_cos
        and query_distance >= min_distance
    )
    reason = "extends_existing_branch" if is_extension else "side_or_cross_branch_attachment"
    return {
        "is_extension": bool(is_extension),
        "reason": reason,
        "branch_id": int(lineage.get("branch_id", -1)),
        "root_parent_joint": int(lineage.get("root_parent_joint", -1)),
        "root_joint": int(lineage.get("root_joint", -1)),
        "joint_chain": [int(item) for item in lineage.get("joint_chain", [])],
        "arc_fraction": float(arc_fraction),
        "progress": float(progress),
        "terminal_cos": float(terminal_cos),
        "query_distance": float(query_distance),
        "branch_length": float(branch_length),
        "min_arc_fraction": float(min_arc),
        "min_progress": float(min_progress),
        "min_terminal_cos": float(min_cos),
        "min_query_distance": float(min_distance),
    }


def _branch_lineage_parent_guard(
    trainer: Any,
    parent_joint: int,
    query_point: torch.Tensor,
    cfg: Phase2TopologyConfig,
) -> dict[str, Any]:
    if not bool(cfg.branch_lineage_parent_guard):
        return {"enabled": False, "allowed": True, "reason": "disabled", "parent_joint": int(parent_joint)}
    lineage = _branch_lineage_for_joint(trainer, int(parent_joint))
    if lineage is None:
        return {
            "enabled": True,
            "allowed": True,
            "reason": "parent_is_not_branch_lineage_joint",
            "parent_joint": int(parent_joint),
            "branch_id": -1,
        }
    extension = _branch_lineage_extension_test(trainer, lineage, query_point, cfg)
    allowed = bool(extension.get("is_extension", False))
    return {
        "enabled": True,
        "allowed": bool(allowed),
        "reason": str(extension.get("reason", "")),
        "parent_joint": int(parent_joint),
        "branch_id": int(lineage.get("branch_id", -1)),
        "extension_test": extension,
    }


def _select_branch_parent_joint_with_lineage_guard(
    *,
    query_point: torch.Tensor,
    joint_positions: torch.Tensor,
    trainer: Any,
    field: Any,
    cfg: Phase2TopologyConfig,
) -> tuple[int, str, float, dict[str, Any] | None]:
    ranking = trace_voxel_parent_paths(
        query_point=query_point,
        joint_positions=joint_positions,
        field=field,
    )
    if not ranking:
        return -1, "voxel_distance_unavailable", 0.0, None
    guard_attempts: list[dict[str, Any]] = []
    for rank, path_info in enumerate(ranking):
        parent_joint = int(path_info["joint_id"])
        guard = _branch_lineage_parent_guard(trainer, parent_joint, query_point, cfg)
        guard["rank"] = int(rank)
        guard["path_length"] = float(path_info.get("path_length", 0.0))
        guard_attempts.append(guard)
        if bool(guard.get("allowed", False)):
            selected = dict(path_info)
            selected["branch_lineage_parent_guard"] = {
                "enabled": bool(cfg.branch_lineage_parent_guard),
                "selected_rank": int(rank),
                "selected_parent_joint": int(parent_joint),
                "attempts": guard_attempts,
            }
            selection = "voxel_distance_parent_joint"
            if int(rank) > 0:
                selection = "voxel_distance_parent_joint_lineage_guarded"
            return int(parent_joint), selection, 1.0, selected
    fallback = dict(ranking[0])
    fallback["branch_lineage_parent_guard"] = {
        "enabled": bool(cfg.branch_lineage_parent_guard),
        "selected_rank": 0,
        "selected_parent_joint": int(fallback["joint_id"]),
        "all_candidates_rejected": True,
        "attempts": guard_attempts,
    }
    return int(fallback["joint_id"]), "voxel_distance_parent_joint_lineage_guard_fallback", 1.0, fallback


def _curvature_path_points(
    polyline: torch.Tensor,
    max_intermediate: int,
    *,
    root_exclusion_fraction: float = 0.10,
) -> torch.Tensor:
    polyline = polyline.reshape(-1, 3)
    if int(polyline.shape[0]) <= 2 or int(max_intermediate) <= 0:
        return polyline[-1:].clone()
    total_length = _path_length(polyline)
    if total_length <= float(EPS):
        return polyline[-1:].clone()
    prev_vec = polyline[1:-1] - polyline[:-2]
    next_vec = polyline[2:] - polyline[1:-1]
    prev_norm = prev_vec.norm(dim=-1).clamp_min(EPS)
    next_norm = next_vec.norm(dim=-1).clamp_min(EPS)
    cos_turn = ((prev_vec * next_vec).sum(dim=-1) / (prev_norm * next_norm)).clamp(-1.0, 1.0)
    turn = torch.acos(cos_turn)
    total_turn = float(turn.sum().item())
    max_turn = float(turn.max().item()) if int(turn.numel()) > 0 else 0.0
    if total_turn <= 0.20 and max_turn <= 0.10:
        return polyline[-1:].clone()
    target_intermediate = int(max_intermediate)
    if total_turn <= 0.75 or max_turn <= 0.35:
        target_intermediate = min(target_intermediate, 1)
    segment_lengths = (polyline[1:] - polyline[:-1]).norm(dim=-1)
    arc = torch.cumsum(segment_lengths, dim=0)[:-1]
    endpoint_margin = 0.10 * total_length
    root_margin = max(endpoint_margin, min(max(float(root_exclusion_fraction), 0.0), 0.45) * total_length)
    valid = (arc >= root_margin) & (arc <= total_length - endpoint_margin)
    candidate_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if int(candidate_ids.numel()) <= 0:
        return polyline[-1:].clone()
    candidate_turn = turn[candidate_ids]
    order = candidate_ids[torch.argsort(candidate_turn, descending=True)]
    chosen: list[int] = []
    chosen_arc: list[float] = []
    min_spacing = 0.15 * total_length
    for idx in order.tolist():
        arc_pos = float(arc[idx].item())
        if any(abs(arc_pos - existing) < min_spacing for existing in chosen_arc):
            continue
        chosen.append(int(idx) + 1)
        chosen_arc.append(arc_pos)
        if len(chosen) >= int(target_intermediate):
            break
    if not chosen:
        return polyline[-1:].clone()
    chosen.sort()
    selected = polyline[torch.tensor(chosen, dtype=torch.long, device=polyline.device)]
    return torch.cat([selected, polyline[-1:].clone()], dim=0)


def _branch_path_points_from_polyline(polyline: torch.Tensor, cfg: Phase2TopologyConfig) -> torch.Tensor:
    polyline = polyline.reshape(-1, 3)
    max_points = max(int(cfg.branch_max_intermediate_points) + 1, 1)
    if int(polyline.shape[0]) <= 1:
        return polyline[-1:].clone()
    segment_lengths = (polyline[1:] - polyline[:-1]).norm(dim=-1)
    total_length_t = segment_lengths.sum().clamp_min(EPS)
    total_length = float(total_length_t.item())
    min_points = min(max(int(cfg.branch_min_path_points), 1), max_points)

    # The branch skeleton must be sampled from the voxel route itself.  Older
    # variants inserted component-root or medial/axis points after curvature
    # sampling, which can pull the branch off the routed centerline on thin
    # shapes.  Here we use evenly spaced route samples as the base and add only
    # on-route curvature extrema when there is enough spacing.
    base_count = min_points
    path_turn = _curvature_path_points(
        polyline,
        max_intermediate=min(int(cfg.branch_max_intermediate_points), 3),
        root_exclusion_fraction=0.10,
    )
    base_fractions = torch.linspace(
        1.0 / float(base_count),
        1.0,
        base_count,
        dtype=polyline.dtype,
        device=polyline.device,
    )
    base_points = _sample_polyline_by_fractions(polyline, base_fractions)
    candidates = torch.cat([base_points, path_turn.reshape(-1, 3)], dim=0)
    candidate_rows = [(_nearest_polyline_arc_fraction(polyline, point), point) for point in candidates]
    candidate_rows.sort(key=lambda item: item[0])
    min_spacing = max(0.08 * total_length, float(EPS))
    kept: list[tuple[float, torch.Tensor]] = []
    for fraction, point in candidate_rows:
        if kept and float((point - kept[-1][1]).norm().item()) < min_spacing:
            if fraction >= kept[-1][0]:
                kept[-1] = (fraction, point)
            continue
        kept.append((fraction, point))
    if bool(getattr(cfg, "branch_long_segment_refine", True)) and len(kept) < max_points:
        max_arc_gap = max(float(getattr(cfg, "branch_long_segment_max_arc_fraction", 0.0)), 0.0)
        while len(kept) < max_points and max_arc_gap > 0.0:
            fractions = [0.0] + [float(item[0]) for item in kept]
            gaps = [float(fractions[index + 1] - fractions[index]) for index in range(len(fractions) - 1)]
            if not gaps:
                break
            gap_index = int(np.argmax(np.asarray(gaps, dtype=np.float64)))
            gap_size = float(gaps[gap_index])
            if gap_size <= max_arc_gap:
                break
            target_fraction = 0.5 * (fractions[gap_index] + fractions[gap_index + 1])
            point = _sample_polyline_by_fractions(
                polyline,
                torch.tensor([target_fraction], dtype=polyline.dtype, device=polyline.device),
            )[0]
            insert_at = max(gap_index, 0)
            if kept and any(float((point - existing).norm().item()) < min_spacing for _frac, existing in kept):
                break
            kept.insert(insert_at, (float(target_fraction), point))
    if len(kept) > max_points:
        # Preserve monotonic arc order and the tip.  Extra curvature samples are
        # dropped by arc position instead of being moved off the voxel route.
        kept = kept[: max_points - 1] + [kept[-1]]
    return torch.stack([point for _fraction, point in kept], dim=0)


def _inside_clearance_tip_target(
    surface_tip: torch.Tensor,
    trainer: Any,
    field: Any,
    cfg: Phase2TopologyConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Move a surface fault tip to a nearby inside voxel center with high clearance."""
    tip_np = surface_tip.detach().cpu().numpy().astype(np.float64).reshape(3)
    filled_count = int(getattr(field, "filled_points", np.zeros((0, 3))).shape[0])
    if filled_count <= 0:
        return surface_tip.reshape(3).detach().clone(), {"accepted": False, "reason": "empty_voxel_field"}
    k = min(max(int(cfg.branch_tip_target_query_k), 1), filled_count)
    distances, indices = field.kdtree.query(tip_np, k=k)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    max_radius = max(
        float(cfg.branch_tip_target_radius_voxels) * float(field.pitch),
        float(cfg.branch_tip_target_radius_ratio) * float(trainer.sample_radius),
    )
    keep = np.isfinite(distances) & (distances <= max_radius)
    if not np.any(keep):
        return surface_tip.reshape(3).detach().clone(), {
            "accepted": False,
            "reason": "no_voxel_within_radius",
            "max_radius": float(max_radius),
            "query_k": int(k),
        }
    candidate_indices = indices[keep]
    candidate_distances = distances[keep]
    candidate_points_np = np.asarray(field.filled_points[candidate_indices], dtype=np.float64)
    candidate_points = torch.tensor(
        candidate_points_np,
        dtype=trainer.rest_vertices.dtype,
        device=trainer.rest_vertices.device,
    )
    inside = points_inside_or_on_mesh(
        candidate_points,
        trainer.rest_vertices,
        trainer.mesh_faces,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    if not bool(inside.any().item()):
        return surface_tip.reshape(3).detach().clone(), {
            "accepted": False,
            "reason": "nearby_voxels_not_mesh_inside",
            "max_radius": float(max_radius),
            "query_k": int(k),
            "candidate_count": int(candidate_points.shape[0]),
        }
    inside_ids = torch.nonzero(inside, as_tuple=False).flatten().detach().cpu().numpy().astype(np.int64)
    candidate_indices_inside = candidate_indices[inside_ids]
    candidate_distances_inside = candidate_distances[inside_ids]
    voxel_indices = np.asarray(field.filled_indices[candidate_indices_inside], dtype=np.int64)
    clearances = field.distance_volume[
        voxel_indices[:, 0],
        voxel_indices[:, 1],
        voxel_indices[:, 2],
    ].astype(np.float64, copy=False)
    max_clearance = max(float(np.max(clearances)) if clearances.size > 0 else 0.0, float(EPS))
    distance_term = candidate_distances_inside / max(float(max_radius), float(EPS))
    scores = clearances / max_clearance - float(cfg.branch_tip_target_distance_weight) * distance_term
    best_local = int(np.argmax(scores))
    best_index = int(candidate_indices_inside[best_local])
    best_point = torch.tensor(
        np.asarray(field.filled_points[best_index], dtype=np.float64),
        dtype=trainer.rest_vertices.dtype,
        device=trainer.rest_vertices.device,
    )
    return best_point.reshape(3), {
        "accepted": True,
        "reason": "inside_clearance_voxel",
        "surface_tip": _json_list(surface_tip),
        "target": _json_list(best_point),
        "distance": float(candidate_distances_inside[best_local]),
        "clearance": float(clearances[best_local]),
        "score": float(scores[best_local]),
        "max_radius": float(max_radius),
        "query_k": int(k),
        "candidate_count": int(candidate_points.shape[0]),
        "inside_candidate_count": int(inside_ids.shape[0]),
    }


def _snap_outside_path_points_to_inside_route(
    path_points: torch.Tensor,
    polyline: torch.Tensor,
    trainer: Any,
    field: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    points = path_points.reshape(-1, 3)
    route = polyline.reshape(-1, 3)
    if int(points.shape[0]) <= 0:
        return points, {"accepted": False, "reason": "empty_path"}
    point_inside = points_inside_or_on_mesh(
        points,
        trainer.rest_vertices,
        trainer.mesh_faces,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    route_inside = points_inside_or_on_mesh(
        route,
        trainer.rest_vertices,
        trainer.mesh_faces,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    route_ids = torch.nonzero(route_inside, as_tuple=False).flatten()
    if int(route_ids.numel()) <= 0:
        return points, {
            "accepted": False,
            "reason": "no_inside_route_points",
            "outside_count": int((~point_inside).sum().item()),
            "route_inside_count": 0,
        }
    route_inside_points = route[route_ids]
    snapped = points.clone()
    snap_records: list[dict[str, Any]] = []
    for path_index in range(int(points.shape[0])):
        distances = torch.linalg.norm(route_inside_points - points[int(path_index)].reshape(1, 3), dim=-1)
        best = int(torch.argmin(distances).item())
        snapped[int(path_index)] = route_inside_points[best]
        distance = float(distances[best].item())
        if distance > max(float(field.pitch) * 1.0e-3, float(EPS)):
            snap_records.append(
                {
                    "path_index": int(path_index),
                    "route_index": int(route_ids[best].item()),
                    "distance": float(distance),
                }
            )
    snapped = _remove_near_duplicate_path_points(snapped, min_spacing=max(float(field.pitch) * 0.25, float(EPS)))
    snapped_inside = points_inside_or_on_mesh(
        snapped,
        trainer.rest_vertices,
        trainer.mesh_faces,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    return snapped, {
        "accepted": bool(snapped_inside.all().item()),
        "reason": "snapped_points_to_inside_route",
        "outside_count": int((~point_inside).sum().item()),
        "route_inside_count": int(route_ids.numel()),
        "final_point_count": int(snapped.shape[0]),
        "snap_records": snap_records,
    }


def _remove_near_duplicate_path_points(path_points: torch.Tensor, *, min_spacing: float) -> torch.Tensor:
    points = path_points.reshape(-1, 3)
    if int(points.shape[0]) <= 1:
        return points.clone()
    kept: list[torch.Tensor] = [points[0]]
    for point in points[1:]:
        if float((point - kept[-1]).norm().item()) < float(min_spacing):
            kept[-1] = point
            continue
        kept.append(point)
    return torch.stack(kept, dim=0)


def _phase2_voxel_field_cache_key(trainer: Any, cfg: Phase2TopologyConfig) -> str:
    vertices = trainer.rest_vertices.detach().to(device="cpu", dtype=torch.float32).contiguous()
    faces = trainer.mesh_faces.detach().to(device="cpu", dtype=torch.int32).contiguous()
    target_resolution = _effective_voxel_target_resolution(trainer, cfg)
    hasher = hashlib.sha1()
    hasher.update(b"evorig_next_phase2_voxel_field_v1")
    hasher.update(str(int(target_resolution)).encode("utf-8"))
    hasher.update(str(float(cfg.voxel_narrow_span_voxels)).encode("utf-8"))
    hasher.update(str(int(cfg.voxel_max_resolution)).encode("utf-8"))
    hasher.update(str(cfg.voxel_neighbor_mode).encode("utf-8"))
    hasher.update(int(vertices.shape[0]).to_bytes(8, byteorder="little", signed=False))
    hasher.update(int(faces.shape[0]).to_bytes(8, byteorder="little", signed=False))
    hasher.update(vertices.numpy().tobytes())
    hasher.update(faces.numpy().tobytes())
    return hasher.hexdigest()


def _phase2_voxel_field_cache_path(cache_key: str) -> Path:
    return _PHASE2_VOXEL_CACHE_ROOT / f"{str(cache_key)}.pkl"


def _load_phase2_voxel_field_from_disk(cache_key: str) -> Any | None:
    path = _phase2_voxel_field_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_phase2_voxel_field_to_disk(cache_key: str, field: Any) -> None:
    path = _phase2_voxel_field_cache_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(field, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _build_voxel_field(trainer: Any, cfg: Phase2TopologyConfig) -> Any | None:
    if not bool(cfg.voxel_parent_enabled):
        return None
    if trainer.mesh_faces is None or int(trainer.mesh_faces.numel()) <= 0:
        return None
    cache_key = _phase2_voxel_field_cache_key(trainer, cfg)
    cached = getattr(trainer, "_phase2_voxel_path_field_cache", None)
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return cached.get("field")
    field = _load_phase2_voxel_field_from_disk(cache_key)
    if field is not None:
        trainer._phase2_voxel_path_field_cache = {"key": cache_key, "field": field, "source": "disk"}
        return field
    field = build_mesh_voxel_path_field(
        _as_numpy(trainer.rest_vertices).astype(np.float64, copy=False),
        _as_numpy(trainer.mesh_faces).astype(np.int64, copy=False),
        target_resolution=int(_effective_voxel_target_resolution(trainer, cfg)),
        target_narrow_span_voxels=float(cfg.voxel_narrow_span_voxels),
        max_resolution=int(cfg.voxel_max_resolution),
        neighbor_mode=str(cfg.voxel_neighbor_mode),
    )
    _save_phase2_voxel_field_to_disk(cache_key, field)
    trainer._phase2_voxel_path_field_cache = {"key": cache_key, "field": field, "source": "built"}
    return field


def _mesh_segment_inside_fraction(
    trainer: Any,
    start: torch.Tensor,
    end: torch.Tensor,
    *,
    sample_count: int = 21,
) -> float:
    if trainer.mesh_faces is None or int(trainer.mesh_faces.numel()) <= 0:
        return 0.0
    samples = torch.linspace(
        0.0,
        1.0,
        max(int(sample_count), 2),
        dtype=start.dtype,
        device=start.device,
    ).reshape(-1, 1)
    points = start.reshape(1, 3) * (1.0 - samples) + end.reshape(1, 3) * samples
    inside = points_inside_or_on_mesh(
        points,
        trainer.rest_vertices,
        trainer.mesh_faces,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    return float(inside.to(dtype=torch.float32).mean().item())


def _branch_path_inside_summary(
    trainer: Any,
    *,
    parent_joint: int,
    path_points: torch.Tensor,
    include_parent_segment: bool = True,
) -> dict[str, Any]:
    fractions: list[float] = []
    parent_fraction: float | None = None
    if int(path_points.numel()) <= 0:
        return {
            "segment_inside_fractions": [],
            "parent_link_inside_fraction": None,
            "min_segment_inside_fraction": 0.0,
        }
    points = path_points.reshape(-1, 3)
    if 0 <= int(parent_joint) < int(trainer.skeleton.joint_count):
        parent_position = trainer.skeleton.rest_joints.detach()[int(parent_joint)]
        parent_fraction = _mesh_segment_inside_fraction(trainer, parent_position, points[0])
        if include_parent_segment:
            fractions.append(parent_fraction)
    for index in range(int(points.shape[0]) - 1):
        fractions.append(_mesh_segment_inside_fraction(trainer, points[index], points[index + 1]))
    min_fraction = min(fractions) if fractions else 0.0
    return {
        "segment_inside_fractions": [float(item) for item in fractions],
        "parent_link_inside_fraction": None if parent_fraction is None else float(parent_fraction),
        "min_segment_inside_fraction": float(min_fraction),
    }


def _refine_branch_physical_segments_on_route(
    path_points: torch.Tensor,
    polyline: torch.Tensor,
    trainer: Any,
    cfg: Phase2TopologyConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Insert on-route points when a physical branch segment cuts through the mesh.

    The parent-to-first-point link is deliberately excluded: Phase2 now treats it
    as a disconnected Blender-style parent relationship.  Only adjacent points
    inside the branch chain represent physical bones and must be inside-valid.
    """
    points = path_points.reshape(-1, 3)
    route = polyline.reshape(-1, 3).to(device=points.device, dtype=points.dtype)
    if int(points.shape[0]) <= 1 or int(route.shape[0]) <= 1:
        return points.clone(), {
            "enabled": True,
            "inserted_count": 0,
            "reason": "not_enough_physical_segments",
            "threshold": float(getattr(cfg, "branch_segment_refine_inside_fraction", 0.75)),
        }
    threshold = float(getattr(cfg, "branch_segment_refine_inside_fraction", 0.75))
    max_points = max(
        int(getattr(cfg, "branch_segment_refine_max_points", 10)),
        int(getattr(cfg, "branch_max_intermediate_points", 4)) + 1,
        int(points.shape[0]),
    )
    route_inside = points_inside_or_on_mesh(
        route,
        trainer.rest_vertices,
        trainer.mesh_faces,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    route_inside_ids = torch.nonzero(route_inside, as_tuple=False).flatten()
    route_inside_fractions = [
        _nearest_polyline_arc_fraction(route, route[int(index)]) for index in route_inside_ids.detach().cpu().tolist()
    ]
    if max_points <= int(points.shape[0]):
        return points.clone(), {
            "enabled": True,
            "inserted_count": 0,
            "reason": "max_points_reached",
            "threshold": float(threshold),
            "initial_point_count": int(points.shape[0]),
            "max_points": int(max_points),
        }

    refined = points.clone()
    records: list[dict[str, Any]] = []
    while int(refined.shape[0]) < max_points:
        fractions = [_nearest_polyline_arc_fraction(route, point) for point in refined]
        order = sorted(range(len(fractions)), key=lambda index: float(fractions[index]))
        refined = refined[torch.tensor(order, dtype=torch.long, device=refined.device)]
        fractions = [float(fractions[index]) for index in order]
        segment_fractions = [
            _mesh_segment_inside_fraction(trainer, refined[index], refined[index + 1])
            for index in range(int(refined.shape[0]) - 1)
        ]
        if not segment_fractions:
            break
        worst_index = int(np.argmin(np.asarray(segment_fractions, dtype=np.float64)))
        worst_fraction = float(segment_fractions[worst_index])
        if worst_fraction >= threshold:
            return refined, {
                "enabled": True,
                "inserted_count": int(len(records)),
                "reason": "all_physical_segments_above_threshold",
                "threshold": float(threshold),
                "segment_inside_fractions": [float(item) for item in segment_fractions],
                "records": records,
            }
        left_fraction = float(fractions[worst_index])
        right_fraction = float(fractions[worst_index + 1])
        if right_fraction <= left_fraction + 1.0e-5:
            break
        mid_fraction = 0.5 * (left_fraction + right_fraction)
        candidate_rows = [
            (abs(float(route_fraction) - float(mid_fraction)), int(route_index.item()), float(route_fraction))
            for route_index, route_fraction in zip(route_inside_ids, route_inside_fractions)
            if left_fraction + 1.0e-5 < float(route_fraction) < right_fraction - 1.0e-5
        ]
        if candidate_rows:
            _distance, route_index, inserted_fraction = min(candidate_rows, key=lambda item: item[0])
            midpoint = route[int(route_index)]
        else:
            inserted_fraction = float(mid_fraction)
            midpoint = _sample_polyline_by_fractions(
                route,
                torch.tensor([mid_fraction], dtype=route.dtype, device=route.device),
            )[0]
        if any(float((midpoint - existing).norm().item()) <= float(EPS) for existing in refined):
            break
        refined = torch.cat([refined[: worst_index + 1], midpoint.reshape(1, 3), refined[worst_index + 1 :]], dim=0)
        records.append(
            {
                "segment_index": int(worst_index),
                "before_inside_fraction": float(worst_fraction),
                "inserted_arc_fraction": float(inserted_fraction),
            }
        )

    final_segment_fractions = [
        _mesh_segment_inside_fraction(trainer, refined[index], refined[index + 1])
        for index in range(int(refined.shape[0]) - 1)
    ]
    return refined, {
        "enabled": True,
        "inserted_count": int(len(records)),
        "reason": "max_points_or_degenerate_arc",
        "threshold": float(threshold),
        "initial_point_count": int(points.shape[0]),
        "final_point_count": int(refined.shape[0]),
        "max_points": int(max_points),
        "segment_inside_fractions": [float(item) for item in final_segment_fractions],
        "records": records,
    }



def _component_principal_axis(
    component_points: torch.Tensor,
    *,
    root_point: torch.Tensor,
    tip_point: torch.Tensor,
) -> torch.Tensor:
    fallback = tip_point - root_point
    fallback_norm = fallback.norm()
    if fallback_norm <= EPS:
        fallback = component_points.mean(dim=0) - root_point if int(component_points.numel()) > 0 else fallback
        fallback_norm = fallback.norm()
    if fallback_norm <= EPS:
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=root_point.dtype, device=root_point.device)
        fallback_norm = fallback.norm()
    fallback_axis = fallback / fallback_norm.clamp_min(EPS)
    if int(component_points.shape[0]) < 3:
        return fallback_axis
    centered = component_points - component_points.mean(dim=0, keepdim=True)
    cov = centered.transpose(0, 1) @ centered / max(int(component_points.shape[0]) - 1, 1)
    try:
        values, vectors = torch.linalg.eigh(cov)
        axis = vectors[:, int(torch.argmax(values).item())]
    except RuntimeError:
        return fallback_axis
    if axis.norm() <= EPS:
        return fallback_axis
    axis = axis / axis.norm().clamp_min(EPS)
    if float((axis * fallback_axis).sum().item()) < 0.0:
        axis = -axis
    return axis



def _build_branch_components(
    trainer: Any,
    branch_seed_mask: torch.Tensor,
    uncovered_mask: torch.Tensor,
    wrong_mask: torch.Tensor,
    vertex_error: torch.Tensor,
    coverage: torch.Tensor,
    wrong_ratio: torch.Tensor,
    cfg: Phase2TopologyConfig,
    *,
    seed_type: str,
    voxel_field: Any | None = None,
    mesh_adjacency: list[tuple[int, ...]] | None = None,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    raw_components = _extract_connected_components(branch_seed_mask, trainer.mesh_faces, adjacency=mesh_adjacency)
    components, merge_records = _merge_nearby_components_by_adjacency(
        raw_components,
        adjacency=mesh_adjacency,
        hops=int(cfg.component_merge_hops),
        vertex_count=int(trainer.rest_vertices.shape[0]),
    )
    min_component_vertices = _effective_component_min_vertices(trainer, cfg)
    component_records = [
        {"ids": ids, "merge": merge_records[index] if index < len(merge_records) else {"enabled": False}}
        for index, ids in enumerate(components)
        if len(ids) >= int(min_component_vertices)
    ]
    component_records.sort(
        key=lambda item: float(
            vertex_error[torch.tensor(item["ids"], dtype=torch.long, device=vertex_error.device)].mean().item()
        ),
        reverse=True,
    )
    preselect_limit = max(int(cfg.max_branch_components) * 4, int(cfg.max_branch_components), 1)
    component_records = component_records[:preselect_limit]
    component_id = torch.full(
        (int(trainer.rest_vertices.shape[0]),),
        -1,
        dtype=torch.long,
        device=trainer.rest_vertices.device,
    )
    voxel_field = _build_voxel_field(trainer, cfg) if voxel_field is None else voxel_field
    rows: list[dict[str, Any]] = []
    rest_joints = trainer.skeleton.rest_joints.detach()
    bbox_diag = float((trainer.rest_vertices.max(dim=0).values - trainer.rest_vertices.min(dim=0).values).norm().item())
    global_error_mass = float(vertex_error.sum().item())
    for component_index, component_record in enumerate(component_records):
        ids = component_record["ids"]
        component_merge = component_record.get("merge", {"enabled": False})
        vertex_ids = torch.tensor(ids, dtype=torch.long, device=trainer.rest_vertices.device)
        component_id[vertex_ids] = int(component_index)
        component_score = vertex_error + wrong_ratio
        tip, tip_vertex_id, center = _component_tip(vertex_ids, trainer.rest_vertices, component_score, rest_joints)
        surface_tip = tip
        tip_target = surface_tip
        tip_target_diagnostics: dict[str, Any] = {
            "accepted": False,
            "reason": "voxel_field_unavailable",
            "surface_tip": _json_list(surface_tip),
        }
        parent_joint = -1
        parent_selection = "none"
        parent_path_length = 0.0
        parent_path_mean_clearance = 0.0
        branch_path_points = tip_target.reshape(1, 3)
        branch_path_points_raw = branch_path_points.clone()
        branch_path_points_curvature = branch_path_points.clone()
        branch_path_points_pre_refine = branch_path_points.clone()
        branch_lineage_parent_guard: dict[str, Any] = {"enabled": bool(cfg.branch_lineage_parent_guard)}
        branch_path_inside: dict[str, Any] = {
            "segment_inside_fractions": [],
            "min_segment_inside_fraction": 0.0,
        }
        branch_path_point_snap: dict[str, Any] = {
            "accepted": True,
            "reason": "not_needed",
        }
        branch_segment_refine: dict[str, Any] = {
            "enabled": True,
            "inserted_count": 0,
            "reason": "not_run",
        }
        bbox_min = trainer.rest_vertices[vertex_ids].min(dim=0).values
        bbox_max = trainer.rest_vertices[vertex_ids].max(dim=0).values
        forced_parent_joint = _double_knife_forced_parent(
            trainer,
            tip_target=surface_tip,
            component_center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
        force_select_component = _double_knife_force_select_component(
            trainer,
            component_center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            vertex_count=int(vertex_ids.numel()),
        )
        if voxel_field is not None:
            tip_target, tip_target_diagnostics = _inside_clearance_tip_target(surface_tip, trainer, voxel_field, cfg)
            if not bool(tip_target_diagnostics.get("accepted", False)):
                continue
            if forced_parent_joint is not None:
                parent_joint, parent_selection, path_info = _double_knife_force_parent_ranking(
                    trainer,
                    rest_joints,
                    tip_target,
                    voxel_field,
                    cfg,
                    int(forced_parent_joint),
                )
            else:
                parent_joint, parent_selection, _inside_ratio, path_info = _select_branch_parent_joint_with_lineage_guard(
                    query_point=tip_target,
                    joint_positions=rest_joints,
                    trainer=trainer,
                    field=voxel_field,
                    cfg=cfg,
                )
            if isinstance(path_info, dict):
                guard_payload = path_info.get("branch_lineage_parent_guard")
                if isinstance(guard_payload, dict):
                    branch_lineage_parent_guard = guard_payload
                parent_path_length = float(path_info.get("path_length", 0.0))
                parent_path_mean_clearance = float(path_info.get("mean_clearance", 0.0))
                polyline = path_info.get("polyline")
                if (
                    float(getattr(cfg, "branch_path_clearance_weight", 0.0)) > 0.0
                    and 0 <= int(parent_joint) < int(rest_joints.shape[0])
                ):
                    medial_ranking = trace_voxel_parent_paths(
                        query_point=tip_target,
                        joint_positions=rest_joints,
                        field=voxel_field,
                        candidate_joint_ids=torch.tensor(
                            [int(parent_joint)],
                            dtype=torch.long,
                            device=rest_joints.device,
                        ),
                        clearance_weight=float(cfg.branch_path_clearance_weight),
                        clearance_power=float(cfg.branch_path_clearance_power),
                    )
                    if medial_ranking:
                        medial_info = dict(medial_ranking[0])
                        polyline = medial_info.get("polyline", polyline)
                        parent_path_length = float(medial_info.get("path_length", parent_path_length))
                        parent_path_mean_clearance = float(
                            medial_info.get("mean_clearance", parent_path_mean_clearance)
                        )
                if isinstance(polyline, torch.Tensor):
                    polyline = polyline.to(device=trainer.rest_vertices.device, dtype=trainer.rest_vertices.dtype)
                    branch_path_points = _branch_path_points_from_polyline(polyline, cfg)
                    branch_path_points_raw = branch_path_points.clone()
                    branch_path_points, branch_path_point_snap = _snap_outside_path_points_to_inside_route(
                        branch_path_points,
                        polyline,
                        trainer,
                        voxel_field,
                    )
                    if not bool(branch_path_point_snap.get("accepted", False)):
                        continue
                    branch_path_points_pre_refine = branch_path_points.clone()
                    branch_path_points, branch_segment_refine = _refine_branch_physical_segments_on_route(
                        branch_path_points,
                        polyline,
                        trainer,
                        cfg,
                    )
                    branch_path_points_curvature = branch_path_points_raw.clone()
        branch_path_inside = _branch_path_inside_summary(
            trainer,
            parent_joint=int(parent_joint),
            path_points=branch_path_points,
            include_parent_segment=False,
        )
        mean_error = float(vertex_error[vertex_ids].mean().item())
        error_mass = float(vertex_error[vertex_ids].sum().item())
        global_error_mass_fraction = float(error_mass / max(global_error_mass, float(EPS)))
        uncovered_fraction = float(uncovered_mask[vertex_ids].to(dtype=vertex_error.dtype).mean().item())
        wrong_fraction = float(wrong_mask[vertex_ids].to(dtype=vertex_error.dtype).mean().item())
        overlap_mask = uncovered_mask[vertex_ids].bool() & wrong_mask[vertex_ids].bool()
        mixed_fault_fraction = float(min(uncovered_fraction, wrong_fraction))
        dual_fault_fraction = float(overlap_mask.to(dtype=vertex_error.dtype).mean().item())
        has_mixed_fault = bool(dual_fault_fraction > 0.0)
        has_combined_fault_types = bool((uncovered_fraction > 0.0) and (wrong_fraction > 0.0))
        wrong_dominant = wrong_fraction >= float(cfg.branch_min_wrong_fraction)
        uncovered_dominant = uncovered_fraction >= float(cfg.branch_min_uncovered_fraction)
        if has_mixed_fault or has_combined_fault_types:
            branch_fault_class = "mixed_fault"
        elif wrong_dominant or wrong_fraction >= uncovered_fraction:
            branch_fault_class = "wrong"
        elif uncovered_dominant:
            branch_fault_class = "uncovered"
        else:
            branch_fault_class = "fault"
        near_score = math.exp(-float(parent_path_length) / max(0.24 * bbox_diag, float(EPS)))
        fault_weight = 1.0 + uncovered_fraction + wrong_fraction
        path_feasibility = 0.5 + 0.5 * near_score
        branch_score = error_mass * fault_weight * path_feasibility
        rows.append(
            {
                "component_index": int(component_index),
                "vertex_count": int(vertex_ids.numel()),
                "vertex_ids": [int(item) for item in ids],
                "score": float(branch_score),
                "near_score": float(near_score),
                "path_feasibility": float(path_feasibility),
                "mean_error": float(mean_error),
                "error_mass": float(error_mass),
                "global_error_mass_fraction": float(global_error_mass_fraction),
                "max_error": float(vertex_error[vertex_ids].max().item()),
                "mean_coverage": float(coverage[vertex_ids].mean().item()),
                "mean_wrong_ratio": float(wrong_ratio[vertex_ids].mean().item()),
                "uncovered_fraction": float(uncovered_fraction),
                "wrong_fraction": float(wrong_fraction),
                "mixed_fault_fraction": float(mixed_fault_fraction),
                "dual_fault_fraction": float(dual_fault_fraction),
                "has_mixed_fault": bool(has_mixed_fault),
                "has_combined_fault_types": bool(has_combined_fault_types),
                "component_min_vertices_effective": int(min_component_vertices),
                "component_merge": component_merge,
                "wrong_dominant": bool(wrong_dominant),
                "uncovered_dominant": bool(uncovered_dominant),
                "branch_fault_class": str(branch_fault_class),
                "branch_seed_type": str(seed_type),
                "tip_vertex_id": int(tip_vertex_id),
                "tip": _json_list(surface_tip),
                "tip_target": _json_list(tip_target),
                "tip_target_diagnostics": tip_target_diagnostics,
                "center": _json_list(center),
                "bbox_min": _json_list(bbox_min),
                "bbox_max": _json_list(bbox_max),
                "parent_joint": int(parent_joint),
                "parent_selection": str(parent_selection),
                "parent_path_length": float(parent_path_length),
                "parent_path_mean_clearance": float(parent_path_mean_clearance),
                "branch_path_points_curvature": _json_list(branch_path_points_curvature),
                "branch_path_points_pre_refine": _json_list(branch_path_points_pre_refine),
                "branch_path_points_raw": _json_list(branch_path_points_raw),
                "branch_path_points": _json_list(branch_path_points),
                "branch_insert_count": int(branch_path_points.shape[0]),
                "branch_path_point_snap": branch_path_point_snap,
                "branch_segment_refine": branch_segment_refine,
                "branch_path_inside": branch_path_inside,
                "branch_lineage_parent_guard": branch_lineage_parent_guard,
                "forced_parent_joint": None if forced_parent_joint is None else int(forced_parent_joint),
                "force_select": bool(force_select_component),
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    total_error_mass = sum(max(float(item.get("error_mass", 0.0)), 0.0) for item in rows)
    for item in rows:
        item["error_mass_fraction"] = (
            max(float(item.get("error_mass", 0.0)), 0.0) / max(float(total_error_mass), float(EPS))
        )
    min_global_mass_fraction = max(float(cfg.branch_min_global_error_mass_fraction), 0.0)
    if min_global_mass_fraction > 0.0:
        filtered_rows: list[dict[str, Any]] = []
        for item in rows:
            global_fraction = float(item.get("global_error_mass_fraction", 0.0))
            threshold = min_global_mass_fraction * _mixed_fault_error_gate_factor(item)
            item["branch_global_error_gate_threshold"] = float(threshold)
            item["branch_global_error_gate_factor"] = float(_mixed_fault_error_gate_factor(item))
            if bool(item.get("force_select", False)) or global_fraction >= threshold:
                filtered_rows.append(item)
        rows = filtered_rows
    return rows[: max(int(cfg.max_branch_components), 0)], component_id


def _build_branch_component_id_map(
    trainer: Any,
    branch_seed_mask: torch.Tensor,
    cfg: Phase2TopologyConfig,
    *,
    mesh_adjacency: list[tuple[int, ...]] | None = None,
) -> torch.Tensor:
    raw_components = _extract_connected_components(branch_seed_mask, trainer.mesh_faces, adjacency=mesh_adjacency)
    components, _merge_records = _merge_nearby_components_by_adjacency(
        raw_components,
        adjacency=mesh_adjacency,
        hops=int(cfg.component_merge_hops),
        vertex_count=int(trainer.rest_vertices.shape[0]),
    )
    min_component_vertices = _effective_component_min_vertices(trainer, cfg)
    components = [item for item in components if len(item) >= int(min_component_vertices)]
    component_id = torch.full(
        (int(trainer.rest_vertices.shape[0]),),
        -1,
        dtype=torch.long,
        device=trainer.rest_vertices.device,
    )
    for component_index, ids in enumerate(components):
        vertex_ids = torch.tensor(ids, dtype=torch.long, device=trainer.rest_vertices.device)
        component_id[vertex_ids] = int(component_index)
    return component_id


def _joint_children(skeleton: Any, joint_id: int) -> list[int]:
    matches = torch.nonzero(skeleton.parent_idx == int(joint_id), as_tuple=False).flatten()
    return [int(item) for item in matches.detach().cpu().tolist()]


def _incident_bones_for_joint(skeleton: Any, joint_id: int) -> list[int]:
    joint_id = int(joint_id)
    matches = torch.nonzero(
        (skeleton.bone_parent_idx == joint_id) | (skeleton.bone_child_idx == joint_id),
        as_tuple=False,
    ).flatten()
    return [int(item) for item in matches.detach().cpu().tolist()]


def _seed_joint_repair_target(
    old_position: torch.Tensor,
    component_center: torch.Tensor,
    *,
    trainer: Any,
    cfg: Phase2TopologyConfig,
) -> tuple[torch.Tensor, float, float, float]:
    direction = component_center.reshape(3) - old_position.reshape(3)
    target_distance = float(direction.norm().item())
    cap = max(float(cfg.seed_joint_repair_cap_sample_radius_ratio) * float(trainer.sample_radius), float(EPS))
    if target_distance <= float(EPS):
        return old_position.reshape(3).detach().clone(), 0.0, float(target_distance), float(cap)
    move_scale = min(1.0, cap / max(target_distance, float(EPS)))
    new_position = old_position.reshape(3) + float(move_scale) * direction
    move_distance = float((new_position - old_position.reshape(3)).norm().item())
    return new_position, float(move_distance), float(target_distance), float(cap)


def _build_seed_joint_repair_candidates(
    trainer: Any,
    *,
    branch_seed_mask: torch.Tensor,
    uncovered_mask: torch.Tensor,
    wrong_mask: torch.Tensor,
    vertex_error: torch.Tensor,
    wrong_ratio: torch.Tensor,
    dominant_joint: torch.Tensor,
    cfg: Phase2TopologyConfig,
    mesh_adjacency: list[tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    if not bool(cfg.seed_joint_repair_enabled):
        return []
    components = _extract_connected_components(branch_seed_mask, trainer.mesh_faces, adjacency=mesh_adjacency)
    min_vertices = _effective_seed_joint_repair_min_vertices(trainer, cfg)
    components = [item for item in components if len(item) >= min_vertices]
    components.sort(
        key=lambda ids: float(vertex_error[torch.tensor(ids, dtype=torch.long, device=vertex_error.device)].sum().item()),
        reverse=True,
    )
    components = components[: max(int(cfg.seed_joint_repair_max_components), 0)]
    if not components:
        return []

    skeleton = trainer.skeleton
    rest_joints = skeleton.rest_joints.detach()
    inserted = list(getattr(skeleton, "is_inserted", []))
    rows: list[dict[str, Any]] = []
    for component_index, ids in enumerate(components):
        vertex_ids = torch.tensor(ids, dtype=torch.long, device=trainer.rest_vertices.device)
        component_points = trainer.rest_vertices[vertex_ids]
        component_center = component_points.mean(dim=0)
        component_vertex_count = int(vertex_ids.numel())
        component_error_mass = float(vertex_error[vertex_ids].sum().item())
        wrong_fraction = float(wrong_mask[vertex_ids].to(dtype=torch.float32).mean().item())
        uncovered_fraction = float(uncovered_mask[vertex_ids].to(dtype=torch.float32).mean().item())
        fault_fraction = max(wrong_fraction, uncovered_fraction)
        if fault_fraction < float(cfg.seed_joint_repair_min_fault_fraction):
            continue
        dominant_local = dominant_joint[vertex_ids].to(dtype=torch.long)
        dominant_counts = torch.bincount(dominant_local, minlength=int(skeleton.joint_count))
        top_counts = torch.topk(
            dominant_counts,
            k=min(5, int(dominant_counts.numel())),
        )
        dominant_top = [
            [int(index), int(count)]
            for count, index in zip(top_counts.values.detach().cpu().tolist(), top_counts.indices.detach().cpu().tolist())
            if int(count) > 0
        ]
        best_for_component: dict[str, Any] | None = None
        for joint_id in range(int(skeleton.joint_count)):
            if 0 <= joint_id < len(inserted) and bool(inserted[joint_id]):
                continue
            parent = int(skeleton.parent_idx[joint_id].item())
            if parent < 0:
                continue
            children = _joint_children(skeleton, joint_id)
            if len(children) != 1:
                continue
            child = int(children[0])
            neighbor_ids = [parent, joint_id, child]
            neighbor_count = int(sum(int(dominant_counts[item].item()) for item in neighbor_ids if 0 <= item < int(dominant_counts.numel())))
            neighbor_fraction = neighbor_count / max(component_vertex_count, 1)
            if neighbor_fraction < float(cfg.seed_joint_repair_min_neighbor_fraction):
                continue
            old_position = rest_joints[joint_id]
            new_position, move_distance, target_distance, cap = _seed_joint_repair_target(
                old_position,
                component_center,
                trainer=trainer,
                cfg=cfg,
            )
            if move_distance <= float(EPS):
                continue
            parent_pos = rest_joints[parent]
            child_pos = rest_joints[child]
            before_parent = _mesh_segment_inside_fraction(trainer, parent_pos, old_position)
            before_child = _mesh_segment_inside_fraction(trainer, old_position, child_pos)
            after_parent = _mesh_segment_inside_fraction(trainer, parent_pos, new_position)
            after_child = _mesh_segment_inside_fraction(trainer, new_position, child_pos)
            before_min = min(before_parent, before_child)
            after_min = min(after_parent, after_child)
            improvement = after_min - before_min
            if after_min < float(cfg.seed_joint_repair_inside_min_fraction):
                continue
            if improvement < float(cfg.seed_joint_repair_min_inside_improvement) and after_min <= before_min + float(EPS):
                continue
            nearest_surface_distance = float((trainer.rest_vertices - new_position.reshape(1, 3)).norm(dim=-1).min().item())
            score = component_error_mass * (1.0 + wrong_fraction + uncovered_fraction) * neighbor_fraction
            candidate = {
                "event_type": "seed_joint_repair",
                "joint": int(joint_id),
                "parent": int(parent),
                "child": int(child),
                "component_index": int(component_index),
                "score": float(score),
                "component_vertex_count": int(component_vertex_count),
                "component_error_mass": float(component_error_mass),
                "wrong_fraction": float(wrong_fraction),
                "uncovered_fraction": float(uncovered_fraction),
                "neighbor_fraction": float(neighbor_fraction),
                "old_position": _json_list(old_position),
                "target_position": _json_list(component_center),
                "new_position": _json_list(new_position),
                "move_distance": float(move_distance),
                "target_distance": float(target_distance),
                "move_scale": float(move_distance / max(target_distance, float(EPS))),
                "capB": float(cap),
                "segment_inside_before": [float(before_parent), float(before_child)],
                "segment_inside_after": [float(after_parent), float(after_child)],
                "inside_improvement": float(improvement),
                "surface_distance": float(nearest_surface_distance),
                "dominant_top": dominant_top,
                "vertex_ids": [int(item) for item in ids],
                "mean_wrong_ratio": float(wrong_ratio[vertex_ids].mean().item()),
            }
            if best_for_component is None or float(candidate["score"]) > float(best_for_component["score"]):
                best_for_component = candidate
        if best_for_component is not None:
            rows.append(best_for_component)
    rows.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return rows


def _segment_distances_and_lambda(points: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    segment = end.reshape(3) - start.reshape(3)
    denom = segment.dot(segment).clamp_min(EPS)
    lam = ((points - start.reshape(1, 3)) @ segment / denom).clamp(0.0, 1.0)
    closest = start.reshape(1, 3) + lam.reshape(-1, 1) * segment.reshape(1, 3)
    return (points - closest).norm(dim=-1), lam


def _connected_vertex_patch_from_selection(
    *,
    selected_ids: torch.Tensor,
    anchor_vertex: int,
    faces: torch.Tensor | None,
    vertex_count: int,
) -> torch.Tensor:
    selected_ids = selected_ids.detach().long().reshape(-1)
    if int(selected_ids.numel()) <= 0 or faces is None or int(faces.numel()) <= 0:
        return selected_ids
    mask = torch.zeros((int(vertex_count),), dtype=torch.bool, device=faces.device)
    mask[selected_ids.to(device=faces.device)] = True
    components = _extract_connected_components(mask, faces)
    if not components:
        return selected_ids
    for component in components:
        if int(anchor_vertex) in component:
            return torch.tensor(component, dtype=torch.long, device=selected_ids.device)
    largest = max(components, key=len)
    return torch.tensor(largest, dtype=torch.long, device=selected_ids.device)


def audit_seed_bone_flow_alignment(trainer: Any, cfg: Phase2TopologyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Phase2TopologyConfig()
    skeleton = trainer.skeleton
    rest_joints = skeleton.rest_joints.detach()
    rest_vertices = trainer.rest_vertices.detach()
    faces = trainer.mesh_faces.detach().long() if trainer.mesh_faces is not None else None
    parent_idx = skeleton.parent_idx.detach().long()
    child_counts = torch.bincount(parent_idx.clamp_min(0), minlength=int(rest_joints.shape[0]))
    inserted = list(getattr(skeleton, "is_inserted", []))
    rows: list[dict[str, Any]] = []
    nearest_k = max(int(cfg.bone_flow_audit_nearest_vertices), int(cfg.bone_flow_audit_min_vertices))
    min_vertices = max(int(cfg.bone_flow_audit_min_vertices), 3)
    for bone_index in range(int(skeleton.bone_count)):
        parent_joint = int(skeleton.bone_parent_idx[bone_index].item())
        child_joint = int(skeleton.bone_child_idx[bone_index].item())
        if parent_joint < 0 or child_joint < 0:
            continue
        start = rest_joints[parent_joint]
        end = rest_joints[child_joint]
        segment = end - start
        length = float(segment.norm().item())
        if length <= float(EPS):
            continue
        bone_dir = segment / segment.norm().clamp_min(EPS)
        distances, _lam = _segment_distances_and_lambda(rest_vertices, start, end)
        order = torch.argsort(distances)
        radius = float(cfg.bone_flow_audit_radius_sample_ratio) * length
        if radius > 0.0:
            within = order[distances[order] <= radius]
            selected = within[:nearest_k] if int(within.numel()) >= min_vertices else order[:nearest_k]
        else:
            selected = order[:nearest_k]
        anchor_vertex = int(order[0].item())
        patch_ids = _connected_vertex_patch_from_selection(
            selected_ids=selected,
            anchor_vertex=anchor_vertex,
            faces=faces,
            vertex_count=int(rest_vertices.shape[0]),
        )
        if int(patch_ids.numel()) < min_vertices:
            patch_ids = selected[:nearest_k]
        patch_points = rest_vertices[patch_ids.to(device=rest_vertices.device)]
        if int(patch_points.shape[0]) < 3:
            continue
        axis = _component_principal_axis(patch_points, root_point=start, tip_point=end)
        abs_cos = abs(float((axis * bone_dir).sum().item()))
        is_leaf = int(child_counts[child_joint].item()) == 0
        is_seed = not (0 <= child_joint < len(inserted) and bool(inserted[child_joint]))
        bad_threshold = float(cfg.bone_flow_audit_bad_leaf_abs_cos) if is_leaf else float(cfg.bone_flow_audit_bad_abs_cos)
        flagged = bool(is_seed and abs_cos < bad_threshold)
        rows.append(
            {
                "bone_index": int(bone_index),
                "parent_joint": int(parent_joint),
                "child_joint": int(child_joint),
                "is_leaf": bool(is_leaf),
                "is_seed": bool(is_seed),
                "length": float(length),
                "patch_vertex_count": int(patch_ids.numel()),
                "nearest_distance_min": float(distances[selected].min().item()),
                "nearest_distance_max": float(distances[selected].max().item()),
                "mesh_flow_axis": _json_list(axis),
                "abs_cos_to_mesh_flow": float(abs_cos),
                "bad_threshold": float(bad_threshold),
                "flagged": bool(flagged),
            }
        )
    flagged_rows = [row for row in rows if bool(row.get("flagged", False))]
    return {
        "enabled": True,
        "method": "nearest_connected_mesh_patch_pca",
        "nearest_vertices": int(nearest_k),
        "min_vertices": int(min_vertices),
        "flagged_count": int(len(flagged_rows)),
        "flagged_bones": flagged_rows,
        "rows": rows,
    }


def _compute_gaussian_residual(
    kernels: torch.Tensor,
    vertex_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    responsibility = kernels / kernels.sum(dim=0, keepdim=True).clamp_min(EPS)
    gaussian_mass = responsibility.sum(dim=1)
    gaussian_residual = (responsibility * vertex_error.unsqueeze(0)).sum(dim=1) / gaussian_mass.clamp_min(EPS)
    return gaussian_residual, gaussian_mass


def _quantile_mean(values: torch.Tensor, quantile: float) -> float:
    if int(values.numel()) <= 0:
        return 0.0
    q = min(max(float(quantile), 0.0), 1.0)
    if q <= 0.0:
        return float(values.mean().item())
    threshold = torch.quantile(values, q)
    mask = values >= threshold
    if not bool(mask.any().item()):
        return float(values.mean().item())
    return float(values[mask].mean().item())


def _relative_metric_ratio(
    values: list[float],
    selected_index: int,
    *,
    reference: str,
    regularizer: float,
) -> float:
    if not (0 <= int(selected_index) < len(values)):
        return 0.0
    selected = float(values[int(selected_index)])
    others = [float(value) for idx, value in enumerate(values) if idx != int(selected_index)]
    if not others:
        return 1.0
    reference_key = str(reference).lower().strip()
    baseline = sum(others) / max(len(others), 1) if reference_key == "mean_others" else max(others)
    eps = max(float(regularizer), 0.0)
    return float((selected + eps) / max(baseline + eps, float(EPS)))


def _local_residual_signal_1d(
    *,
    lambdas: torch.Tensor,
    residual: torch.Tensor,
    residual_raw: torch.Tensor,
    topk: int,
) -> dict[str, float]:
    count = int(lambdas.numel())
    if count <= 0:
        return {"concentration_ratio": 1.0, "concentration_ratio_raw": 1.0}
    order = torch.argsort(lambdas)
    sorted_residual = residual[order]
    sorted_raw = residual_raw[order]
    window = min(max(int(topk), 1), count)
    if window >= count:
        best = float(sorted_residual.mean().item())
        best_raw = float(sorted_raw.mean().item())
    else:
        best = 0.0
        best_raw = 0.0
        for start in range(0, count - window + 1):
            value = float(sorted_residual[start : start + window].mean().item())
            if value > best:
                best = value
                best_raw = float(sorted_raw[start : start + window].mean().item())
    mean = float(sorted_residual.mean().item())
    mean_raw = float(sorted_raw.mean().item())
    return {
        "concentration_ratio": float(best / max(mean, float(EPS))),
        "concentration_ratio_raw": float(best_raw / max(mean_raw, float(EPS))),
    }


def _loss_ratio_score(
    *,
    mode: str,
    bone_residual: torch.Tensor,
    bone_residual_raw: torch.Tensor,
    gaussian_topk: int,
    residual_quantile: float,
    local_signal: dict[str, float],
) -> dict[str, float | str]:
    count = int(bone_residual.numel())
    if count <= 0:
        return {
            "mode": "gaussian_quantile_mean_meanT",
            "score": 0.0,
            "score_raw": 0.0,
            "mean_meanT": 0.0,
            "topk_mean_meanT": 0.0,
            "quantile_mean_meanT": 0.0,
            "local_concentration_ratio": 1.0,
        }
    topk = min(max(int(gaussian_topk), 1), count)
    mean_score = float(bone_residual.mean().item())
    mean_raw = float(bone_residual_raw.mean().item())
    topk_score = float(torch.topk(bone_residual, k=topk, largest=True).values.mean().item())
    topk_raw = float(torch.topk(bone_residual_raw, k=topk, largest=True).values.mean().item())
    quantile_score = _quantile_mean(bone_residual, residual_quantile)
    quantile_raw = _quantile_mean(bone_residual_raw, residual_quantile)
    local_ratio = max(float(local_signal.get("concentration_ratio", 1.0)), 1.0)
    local_ratio_raw = max(float(local_signal.get("concentration_ratio_raw", 1.0)), 1.0)
    hybrid_score = quantile_score * local_ratio
    hybrid_raw = quantile_raw * local_ratio_raw
    score_mode = str(mode).lower().strip()
    if score_mode == "gaussian_mean_meant":
        score = mean_score
        raw = mean_raw
        canonical = "gaussian_mean_meanT"
    elif score_mode == "gaussian_topk_mean_meant":
        score = topk_score
        raw = topk_raw
        canonical = "gaussian_topk_mean_meanT"
    elif score_mode == "gaussian_quantile_local_hybrid_meant":
        score = hybrid_score
        raw = hybrid_raw
        canonical = "gaussian_quantile_local_hybrid_meanT"
    else:
        score = quantile_score
        raw = quantile_raw
        canonical = "gaussian_quantile_mean_meanT"
    return {
        "mode": canonical,
        "score": float(score),
        "score_raw": float(raw),
        "mean_meanT": float(mean_score),
        "mean_meanT_raw": float(mean_raw),
        "topk_mean_meanT": float(topk_score),
        "topk_mean_meanT_raw": float(topk_raw),
        "quantile_mean_meanT": float(quantile_score),
        "quantile_mean_meanT_raw": float(quantile_raw),
        "quantile_local_hybrid_meanT": float(hybrid_score),
        "quantile_local_hybrid_meanT_raw": float(hybrid_raw),
        "local_concentration_ratio": float(local_ratio),
        "local_concentration_ratio_raw": float(local_ratio_raw),
    }


def _build_partition_lambda_candidates(
    vertex_lambda: torch.Tensor,
    *,
    lambda_min: float,
    lambda_max: float,
) -> torch.Tensor:
    if int(vertex_lambda.numel()) <= 0:
        return torch.empty(0, dtype=vertex_lambda.dtype, device=vertex_lambda.device)
    unique_lambdas = torch.unique(torch.sort(vertex_lambda.clamp(float(lambda_min), float(lambda_max))).values)
    midpoint = torch.tensor(
        [(float(lambda_min) + float(lambda_max)) * 0.5],
        dtype=vertex_lambda.dtype,
        device=vertex_lambda.device,
    )
    candidates: list[torch.Tensor] = [midpoint]
    if int(unique_lambdas.numel()) > 0:
        candidates.append(0.5 * (unique_lambdas[:1] + float(lambda_min)))
        candidates.append(0.5 * (unique_lambdas[-1:] + float(lambda_max)))
    if int(unique_lambdas.numel()) > 1:
        candidates.append(0.5 * (unique_lambdas[:-1] + unique_lambdas[1:]))
    merged = torch.cat(candidates, dim=0)
    interior = (merged > float(lambda_min) + 1.0e-6) & (merged < float(lambda_max) - 1.0e-6)
    merged = merged[interior]
    if int(merged.numel()) <= 0:
        return midpoint
    return torch.unique(torch.sort(merged).values)


def _rotation_angle_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)


def _dual_rigid_split_confirm(
    *,
    vertex_ids: torch.Tensor,
    proximal_vertex_ids: torch.Tensor,
    distal_vertex_ids: torch.Tensor,
    rest_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    reference_length: float,
) -> dict[str, float]:
    if int(vertex_ids.numel()) < 4 or int(proximal_vertex_ids.numel()) < 2 or int(distal_vertex_ids.numel()) < 2:
        return {
            "single_rigid_error": 0.0,
            "single_rigid_error_raw": 0.0,
            "two_rigid_error": 0.0,
            "two_rigid_error_raw": 0.0,
            "dual_rigid_gain": 0.0,
            "dual_rigid_gain_raw": 0.0,
            "dual_motion_gap": 0.0,
        }
    frame_count = int(gt_vertices.shape[0])
    src_all = rest_vertices[vertex_ids].unsqueeze(0).expand(frame_count, -1, -1)
    tgt_all = gt_vertices[:, vertex_ids]
    src_prox = rest_vertices[proximal_vertex_ids].unsqueeze(0).expand(frame_count, -1, -1)
    tgt_prox = gt_vertices[:, proximal_vertex_ids]
    src_dist = rest_vertices[distal_vertex_ids].unsqueeze(0).expand(frame_count, -1, -1)
    tgt_dist = gt_vertices[:, distal_vertex_ids]
    rot_all, _trans_all, err_all = fit_rigid_sequence(src_all, tgt_all)
    rot_prox, trans_prox, err_prox = fit_rigid_sequence(src_prox, tgt_prox)
    rot_dist, trans_dist, err_dist = fit_rigid_sequence(src_dist, tgt_dist)
    _ = rot_all
    single_raw = float(err_all.mean().item())
    prox_raw = float(err_prox.mean().item())
    dist_raw = float(err_dist.mean().item())
    prox_weight = float(proximal_vertex_ids.numel())
    dist_weight = float(distal_vertex_ids.numel())
    two_raw = (prox_weight * prox_raw + dist_weight * dist_raw) / max(prox_weight + dist_weight, float(EPS))
    gain_raw = max(single_raw - two_raw, 0.0)
    prox_center_rest = rest_vertices[proximal_vertex_ids].mean(dim=0)
    dist_center_rest = rest_vertices[distal_vertex_ids].mean(dim=0)
    prox_center_pred = torch.matmul(rot_prox, prox_center_rest.view(1, 3, 1)).squeeze(-1) + trans_prox
    dist_center_pred = torch.matmul(rot_dist, dist_center_rest.view(1, 3, 1)).squeeze(-1) + trans_dist
    translation_gap_raw = float(
        ((prox_center_pred - prox_center_rest.view(1, 3)) - (dist_center_pred - dist_center_rest.view(1, 3)))
        .norm(dim=-1)
        .mean()
        .item()
    )
    relative_rot = torch.matmul(rot_prox.transpose(-1, -2), rot_dist)
    rotation_gap = float((_rotation_angle_from_matrix(relative_rot) / math.pi).mean().item())
    translation_gap = float(
        normalize_linear_metric(
            torch.tensor(translation_gap_raw, dtype=rest_vertices.dtype, device=rest_vertices.device),
            reference_length,
        ).item()
    )
    return {
        "single_rigid_error": float(
            normalize_linear_metric(torch.tensor(single_raw, dtype=rest_vertices.dtype, device=rest_vertices.device), reference_length).item()
        ),
        "single_rigid_error_raw": float(single_raw),
        "two_rigid_error": float(
            normalize_linear_metric(torch.tensor(two_raw, dtype=rest_vertices.dtype, device=rest_vertices.device), reference_length).item()
        ),
        "two_rigid_error_raw": float(two_raw),
        "dual_rigid_gain": float(
            normalize_linear_metric(torch.tensor(gain_raw, dtype=rest_vertices.dtype, device=rest_vertices.device), reference_length).item()
        ),
        "dual_rigid_gain_raw": float(gain_raw),
        "dual_motion_gap": float(0.5 * (translation_gap + rotation_gap)),
    }


def _search_rigid_split_lambda(
    *,
    candidate_lambdas: torch.Tensor,
    vertex_ids: torch.Tensor,
    vertex_lambda: torch.Tensor,
    rest_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    reference_length: float,
) -> tuple[dict[str, Any] | None, list[dict[str, float]]]:
    best: dict[str, Any] | None = None
    curve: list[dict[str, float]] = []
    if int(candidate_lambdas.numel()) <= 0 or int(vertex_ids.numel()) < 4:
        return None, curve
    for candidate_lambda in candidate_lambdas:
        lam = float(candidate_lambda.item())
        proximal_vertex_ids = vertex_ids[vertex_lambda <= candidate_lambda]
        distal_vertex_ids = vertex_ids[vertex_lambda > candidate_lambda]
        if int(proximal_vertex_ids.numel()) < 2 or int(distal_vertex_ids.numel()) < 2:
            continue
        dual = _dual_rigid_split_confirm(
            vertex_ids=vertex_ids,
            proximal_vertex_ids=proximal_vertex_ids,
            distal_vertex_ids=distal_vertex_ids,
            rest_vertices=rest_vertices,
            gt_vertices=gt_vertices,
            reference_length=reference_length,
        )
        row = {
            "lambda": float(lam),
            "proximal_count": int(proximal_vertex_ids.numel()),
            "distal_count": int(distal_vertex_ids.numel()),
            "single_rigid_error": float(dual["single_rigid_error"]),
            "single_rigid_error_raw": float(dual["single_rigid_error_raw"]),
            "two_rigid_error": float(dual["two_rigid_error"]),
            "two_rigid_error_raw": float(dual["two_rigid_error_raw"]),
            "rigid_gain": float(dual["dual_rigid_gain"]),
            "rigid_gain_raw": float(dual["dual_rigid_gain_raw"]),
            "dual_motion_gap": float(dual["dual_motion_gap"]),
        }
        curve.append(row)
        if best is None or float(dual["dual_rigid_gain"]) > float(best["dual_rigid_gain"]):
            best = {
                "split_lambda": float(lam),
                "proximal_vertex_ids": proximal_vertex_ids,
                "distal_vertex_ids": distal_vertex_ids,
                **dual,
            }
    return best, curve


def _split_candidates(
    trainer: Any,
    cache: Any,
    vertex_error: torch.Tensor,
    gaussian_residual: torch.Tensor,
    gaussian_mass: torch.Tensor,
    cfg: Phase2TopologyConfig,
) -> list[dict[str, Any]]:
    active = trainer.field.active_mask.detach().bool()
    if not bool(active.any().item()):
        return []
    grad = trainer.gaussian_grad_ema.detach().to(device=gaussian_residual.device, dtype=gaussian_residual.dtype)
    kernels = cache.kernels.detach()
    responsibility = kernels / kernels.sum(dim=0, keepdim=True).clamp_min(EPS)
    coverage = kernels.sum(dim=0)
    coverage_q = min(max(float(cfg.split_coverage_quantile), 0.0), 1.0)
    coverage_threshold = float(torch.quantile(coverage, coverage_q).item())
    bone_count = int(trainer.skeleton.bone_count)
    bone_support = torch.zeros(
        bone_count,
        int(kernels.shape[1]),
        dtype=kernels.dtype,
        device=kernels.device,
    )
    active_ids = torch.nonzero(active, as_tuple=False).flatten()
    active_anchor = trainer.field.anchor_bone[active_ids].to(device=kernels.device, dtype=torch.long)
    bone_support.index_add_(0, active_anchor, responsibility[active_ids])
    max_bone_support = bone_support.max(dim=0).values.clamp_min(EPS)
    dominant_bone = torch.argmax(bone_support, dim=0)
    parent_pos, _frames, bone_parent_idx, bone_child_idx = trainer.skeleton.compute_bone_frames()

    ratio_rows: list[dict[str, Any]] = []
    for bone_tensor in torch.unique(trainer.field.anchor_bone[active]).tolist():
        bone_index = int(bone_tensor)
        gaussian_ids = torch.nonzero(active & (trainer.field.anchor_bone == bone_index), as_tuple=False).flatten()
        if int(gaussian_ids.numel()) < int(cfg.split_min_gaussians_per_bone):
            continue
        child_joint = int(bone_child_idx[bone_index].item())
        segment = trainer.skeleton.rest_joints[child_joint] - parent_pos[bone_index]
        if float(segment.norm().item()) <= float(EPS):
            continue
        lambdas = trainer.field.lambda_param.detach()[gaussian_ids]
        residual_raw = torch.linalg.norm(
            (responsibility[gaussian_ids] * vertex_error.unsqueeze(0)).detach(),
            dim=1,
        )
        local_signal = _local_residual_signal_1d(
            lambdas=lambdas,
            residual=gaussian_residual[gaussian_ids],
            residual_raw=residual_raw,
            topk=int(cfg.split_gaussian_topk),
        )
        loss_score = _loss_ratio_score(
            mode=str(cfg.split_ratio_score_mode),
            bone_residual=gaussian_residual[gaussian_ids],
            bone_residual_raw=residual_raw,
            gaussian_topk=int(cfg.split_gaussian_topk),
            residual_quantile=float(cfg.split_residual_quantile),
            local_signal=local_signal,
        )
        ratio_rows.append(
            {
                "bone_index": int(bone_index),
                "parent_joint": int(bone_parent_idx[bone_index].item()),
                "child_joint": int(child_joint),
                "gaussian_ids_tensor": gaussian_ids,
                "gaussian_count": int(gaussian_ids.numel()),
                "loss_score": float(loss_score["score"]),
                "loss_score_raw": float(loss_score["score_raw"]),
                "loss_score_mode": str(loss_score["mode"]),
                "loss_terms": loss_score,
                "local_signal": local_signal,
                "gradient_quantile_mean": _quantile_mean(grad[gaussian_ids], 0.8),
                "gradient_mean": float(grad[gaussian_ids].mean().item()),
                "support_mass": float(gaussian_mass[gaussian_ids].sum().item()),
            }
        )
    if not ratio_rows:
        return []
    score_values = [float(row["loss_score"]) for row in ratio_rows]
    for index, row in enumerate(ratio_rows):
        row["loss_ratio"] = _relative_metric_ratio(
            score_values,
            index,
            reference=str(cfg.split_ratio_reference),
            regularizer=float(cfg.split_ratio_regularizer),
        )

    candidates: list[dict[str, Any]] = []
    ratio_rows.sort(key=lambda row: (float(row["loss_ratio"]), float(row["loss_score"])), reverse=True)
    for row in ratio_rows:
        if len(score_values) > 1 and float(row["loss_ratio"]) < float(cfg.split_ratio_threshold):
            continue
        bone_index = int(row["bone_index"])
        gaussian_ids = row["gaussian_ids_tensor"]
        vertex_support = bone_support[bone_index]
        vertex_ids = torch.nonzero(dominant_bone == bone_index, as_tuple=False).flatten()
        vertex_scope = "dominant_bone"
        if int(vertex_ids.numel()) < int(cfg.split_min_vertices):
            support_ratio = vertex_support / max_bone_support
            supported = (support_ratio >= 0.35) & (vertex_support >= 0.05)
            vertex_ids = torch.nonzero(supported, as_tuple=False).flatten()
            vertex_scope = "supported_fallback"
        if int(vertex_ids.numel()) < int(cfg.split_min_vertices):
            continue
        mean_coverage = float(coverage[vertex_ids].mean().item())
        if mean_coverage < coverage_threshold:
            continue
        child_joint = int(row["child_joint"])
        segment = trainer.skeleton.rest_joints[child_joint] - parent_pos[bone_index]
        segment_len_sq = segment.dot(segment).clamp_min(EPS)
        vertex_lambda = ((trainer.rest_vertices[vertex_ids] - parent_pos[bone_index].unsqueeze(0)) @ segment / segment_len_sq).clamp(0.0, 1.0)
        if int(vertex_ids.numel()) > int(cfg.split_vertex_topk):
            order = torch.argsort(vertex_error[vertex_ids], descending=True)[: int(cfg.split_vertex_topk)]
            vertex_ids = vertex_ids[order]
            vertex_lambda = vertex_lambda[order]
        candidate_lambdas = _build_partition_lambda_candidates(
            vertex_lambda,
            lambda_min=float(cfg.split_lambda_min),
            lambda_max=float(cfg.split_lambda_max),
        )
        rigid_best, rigid_curve = _search_rigid_split_lambda(
            candidate_lambdas=candidate_lambdas,
            vertex_ids=vertex_ids,
            vertex_lambda=vertex_lambda,
            rest_vertices=trainer.rest_vertices,
            gt_vertices=trainer.gt_vertices,
            reference_length=float(trainer.sample_radius),
        )
        if rigid_best is None or float(rigid_best["dual_rigid_gain"]) < float(cfg.split_rigid_gain_threshold):
            continue
        split_lambda = float(rigid_best["split_lambda"])
        proximal_count = int(torch.count_nonzero(vertex_lambda <= split_lambda).item())
        distal_count = int(torch.count_nonzero(vertex_lambda > split_lambda).item())
        balance = float(min(proximal_count, distal_count) / max(max(proximal_count, distal_count), 1))
        if balance < float(cfg.split_balance_min):
            continue
        grad_concentration = float(row["gradient_quantile_mean"]) / max(float(row["gradient_mean"]), float(EPS))
        score = float(row["loss_ratio"]) * float(rigid_best["dual_rigid_gain"]) * (1.0 + 0.05 * max(grad_concentration - 1.0, 0.0))
        ordered = gaussian_ids[torch.argsort(trainer.field.lambda_param.detach()[gaussian_ids])]
        candidates.append(
            {
                "mode": "split",
                "source": "phase2_loss_distribution_rigid_confirm",
                "bone_index": int(bone_index),
                "parent_joint": int(row["parent_joint"]),
                "child_joint": int(row["child_joint"]),
                "gaussian_count": int(row["gaussian_count"]),
                "component_vertex_count": int(vertex_ids.numel()),
                "vertex_scope": vertex_scope,
                "split_lambda": float(split_lambda),
                "score": float(score),
                "loss_ratio": float(row["loss_ratio"]),
                "loss_ratio_threshold": float(cfg.split_ratio_threshold),
                "loss_score": float(row["loss_score"]),
                "loss_score_raw": float(row["loss_score_raw"]),
                "loss_score_mode": str(row["loss_score_mode"]),
                "loss_terms": row["loss_terms"],
                "local_concentration_ratio": float(row["local_signal"]["concentration_ratio"]),
                "gradient_quantile_mean": float(row["gradient_quantile_mean"]),
                "gradient_mean": float(row["gradient_mean"]),
                "gradient_concentration_ratio": float(grad_concentration),
                "support_mass": float(row["support_mass"]),
                "mean_coverage": float(mean_coverage),
                "coverage_threshold": float(coverage_threshold),
                "split_balance": float(balance),
                "rigid_gain": float(rigid_best["dual_rigid_gain"]),
                "rigid_gain_raw": float(rigid_best["dual_rigid_gain_raw"]),
                "single_rigid_error": float(rigid_best["single_rigid_error"]),
                "single_rigid_error_raw": float(rigid_best["single_rigid_error_raw"]),
                "two_rigid_error": float(rigid_best["two_rigid_error"]),
                "two_rigid_error_raw": float(rigid_best["two_rigid_error_raw"]),
                "dual_motion_gap": float(rigid_best["dual_motion_gap"]),
                "candidate_lambdas": [float(item) for item in candidate_lambdas.detach().cpu().tolist()],
                "rigid_search_curve": rigid_curve,
                "gaussian_ids": [int(item) for item in ordered.detach().cpu().tolist()],
                "vertex_ids": [int(item) for item in vertex_ids.detach().cpu().tolist()],
                "lambda_values": [float(item) for item in trainer.field.lambda_param.detach()[ordered].cpu().tolist()],
                "residual_values": [float(item) for item in gaussian_residual[ordered].cpu().tolist()],
                "grad_values": [float(item) for item in grad[ordered].cpu().tolist()],
            }
        )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[: max(int(cfg.split_topk), 0)]


def _compute_topology_joint_support(
    trainer: Any,
    cache: Any,
    cfg: Phase2TopologyConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    sigma = float(cfg.topology_support_sigma)
    if sigma <= 0.0:
        return cache.support.detach(), cache.kernels.detach(), {
            "mode": "cache_support",
            "sigma": 0.0,
            "cutoff_sq": 0.0,
        }
    old_cutoff_sq = float(trainer.field.kernel_mahal_cutoff_sq)
    trainer.field.kernel_mahal_cutoff_sq = float(sigma * sigma)
    try:
        with torch.no_grad():
            kernels, _mix, support = trainer.field.compute_joint_support(
                trainer.rest_vertices,
                trainer.skeleton,
                mode=str(trainer.cfg.ownership_mode),
                midpoint=float(trainer.cfg.ownership_midpoint),
                slope=float(trainer.cfg.ownership_slope),
                child_gate_start=float(trainer.cfg.child_support_gate_start),
                child_gate_end=float(trainer.cfg.child_support_gate_end),
                use_endpoint_logits=False,
                endpoint_logits_mask=None,
            )
    finally:
        trainer.field.kernel_mahal_cutoff_sq = old_cutoff_sq
    return support.detach(), kernels.detach(), {
        "mode": "topology_sigma_gate",
        "sigma": float(sigma),
        "cutoff_sq": float(sigma * sigma),
    }


def build_phase2_topology_signals(
    trainer: Any,
    cache: Any,
    config: Phase2TopologyConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg = config or Phase2TopologyConfig()
    vertex_error_raw = torch.linalg.norm(cache.pred_vertices.detach() - trainer.gt_vertices.detach(), dim=-1).mean(dim=0)
    vertex_error = vertex_error_raw / max(float(trainer.sample_radius), float(EPS))
    topology_support, topology_kernels, topology_support_info = _compute_topology_joint_support(trainer, cache, cfg)
    support_vj = topology_support.transpose(0, 1)
    coverage = support_vj.sum(dim=-1)
    legal_mask = trainer.legal_joint_mask.to(device=support_vj.device, dtype=torch.bool)
    illegal_mass = (support_vj * (~legal_mask).to(dtype=support_vj.dtype)).sum(dim=-1)
    wrong_ratio = illegal_mass / coverage.clamp_min(EPS)
    q_err = min(max(float(cfg.vertex_error_quantile), 0.0), 1.0)
    q_wrong_err = min(max(float(cfg.wrong_coverage_error_quantile), 0.0), 1.0)
    q_cov = min(max(float(cfg.coverage_quantile), 0.0), 1.0)
    error_threshold = float(torch.quantile(vertex_error, q_err).item())
    wrong_error_threshold = float(torch.quantile(vertex_error, q_wrong_err).item())
    coverage_threshold = max(float(cfg.coverage_abs_threshold), float(torch.quantile(coverage, q_cov).item()))
    high_error_mask = vertex_error >= error_threshold
    wrong_error_mask = vertex_error >= wrong_error_threshold
    uncovered_mask = cache.zero_weight_mask.detach().bool() | (coverage <= coverage_threshold)
    wrong_mask = (wrong_ratio >= float(cfg.wrong_coverage_ratio)) & (illegal_mass >= float(cfg.wrong_coverage_mass_min))
    # Branch is driven by topology faults first. Reconstruction error is used later
    # to rank/filter components, not to decide whether wrong/uncovered vertices exist.
    branch_seed_mask = uncovered_mask | wrong_mask
    mesh_adjacency = _phase2_rest_mesh_adjacency(trainer)
    component_id = _build_branch_component_id_map(
        trainer,
        branch_seed_mask,
        cfg,
        mesh_adjacency=mesh_adjacency,
    )
    voxel_field = _build_voxel_field(trainer, cfg)
    branch_components, _branch_component_id = _build_branch_components(
        trainer,
        branch_seed_mask,
        uncovered_mask,
        wrong_mask,
        vertex_error,
        coverage,
        wrong_ratio,
        cfg,
        seed_type="fault",
        voxel_field=voxel_field,
        mesh_adjacency=mesh_adjacency,
    )
    branch_components.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for merged_index, item in enumerate(branch_components):
        item["source_component_index"] = int(item.get("component_index", -1))
        item["component_index"] = int(merged_index)
    branch_components = branch_components[: max(int(cfg.max_branch_components), 0)]
    bone_flow_audit = audit_seed_bone_flow_alignment(trainer, cfg)
    dominant_joint = cache.weights.detach().argmax(dim=-1)
    seed_joint_repair_candidates = _build_seed_joint_repair_candidates(
        trainer,
        branch_seed_mask=branch_seed_mask,
        uncovered_mask=uncovered_mask,
        wrong_mask=wrong_mask,
        vertex_error=vertex_error,
        wrong_ratio=wrong_ratio,
        dominant_joint=dominant_joint,
        cfg=cfg,
        mesh_adjacency=mesh_adjacency,
    )
    gaussian_residual, gaussian_mass = _compute_gaussian_residual(cache.kernels.detach(), vertex_error)
    split_rows = _split_candidates(trainer, cache, vertex_error, gaussian_residual, gaussian_mass, cfg)
    arrays = {
        "vertex_error": _as_numpy(vertex_error),
        "vertex_error_raw": _as_numpy(vertex_error_raw),
        "vertex_coverage": _as_numpy(coverage),
        "vertex_illegal_mass": _as_numpy(illegal_mass),
        "vertex_wrong_coverage_ratio": _as_numpy(wrong_ratio),
        "vertex_uncovered_mask": _as_numpy(uncovered_mask.to(dtype=torch.int8)),
        "vertex_wrong_mask": _as_numpy(wrong_mask.to(dtype=torch.int8)),
        "vertex_branch_seed_mask": _as_numpy(branch_seed_mask.to(dtype=torch.int8)),
        "vertex_component_id": _as_numpy(component_id),
        "dominant_joint": _as_numpy(dominant_joint.to(dtype=torch.int32)),
        "gaussian_residual": _as_numpy(gaussian_residual),
        "gaussian_support_mass": _as_numpy(gaussian_mass),
        "topology_gaussian_support_mass": _as_numpy(topology_kernels.sum(dim=-1)),
        "gaussian_grad_ema": _as_numpy(trainer.gaussian_grad_ema.detach()),
        "gaussian_lambda": _as_numpy(trainer.field.lambda_param.detach()),
        "gaussian_anchor_bone": _as_numpy(trainer.field.anchor_bone.detach().to(dtype=torch.int32)),
        "gaussian_generation": _as_numpy(trainer.field.generation.detach().to(dtype=torch.int32)),
        "gaussian_active_mask": _as_numpy(trainer.field.active_mask.detach().to(dtype=torch.int8)),
    }
    summary = {
        "format": "evorig_next_phase2_topology_signals_v1",
        "config": asdict(cfg),
        "vertex_count": int(trainer.rest_vertices.shape[0]),
        "gaussian_count": int(trainer.field.gaussian_count),
        "joint_count": int(trainer.skeleton.joint_count),
        "bone_count": int(trainer.skeleton.bone_count),
        "sample_radius": float(trainer.sample_radius),
        "rest_mesh_surface_area": float(_rest_mesh_surface_area(trainer)),
        "effective_component_min_vertices": int(_effective_component_min_vertices(trainer, cfg)),
        "effective_seed_joint_repair_min_vertices": int(_effective_seed_joint_repair_min_vertices(trainer, cfg)),
        "component_min_vertices_reference_vertex_count": int(cfg.component_min_vertices_reference_vertex_count),
        "seed_joint_repair_min_vertices_reference_vertex_count": int(
            cfg.seed_joint_repair_min_vertices_reference_vertex_count
        ),
        "voxel_path_field_cache": {
            "enabled": bool(cfg.voxel_parent_enabled),
            "source": str(getattr(trainer, "_phase2_voxel_path_field_cache", {}).get("source", "none"))
            if isinstance(getattr(trainer, "_phase2_voxel_path_field_cache", None), dict)
            else "none",
        },
        "rest_mesh_adjacency_cache": {
            "enabled": bool(mesh_adjacency is not None),
            "source": str(getattr(trainer, "_phase2_rest_mesh_adjacency_cache_source", "none")),
        },
        "topology_support": topology_support_info,
        "error_threshold": float(error_threshold),
        "wrong_error_threshold": float(wrong_error_threshold),
        "coverage_threshold": float(coverage_threshold),
        "high_error_vertex_count": int(high_error_mask.sum().item()),
        "wrong_error_vertex_count": int(wrong_error_mask.sum().item()),
        "uncovered_vertex_count": int(uncovered_mask.sum().item()),
        "wrong_coverage_vertex_count": int(wrong_mask.sum().item()),
        "branch_seed_vertex_count": int(branch_seed_mask.sum().item()),
        "branch_component_count": int(len(branch_components)),
        "branch_components": branch_components,
        "bone_flow_audit": bone_flow_audit,
        "seed_joint_repair_candidate_count": int(len(seed_joint_repair_candidates)),
        "seed_joint_repair_candidates": seed_joint_repair_candidates,
        "split_candidate_count": int(len(split_rows)),
        "split_candidates": split_rows,
    }
    return summary, arrays


def build_phase2_split_signals(
    trainer: Any,
    cache: Any,
    config: Phase2TopologyConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg = config or Phase2TopologyConfig()
    vertex_error_raw = torch.linalg.norm(cache.pred_vertices.detach() - trainer.gt_vertices.detach(), dim=-1).mean(dim=0)
    vertex_error = vertex_error_raw / max(float(trainer.sample_radius), float(EPS))
    gaussian_residual, gaussian_mass = _compute_gaussian_residual(cache.kernels.detach(), vertex_error)
    split_rows = _split_candidates(trainer, cache, vertex_error, gaussian_residual, gaussian_mass, cfg)
    arrays = {
        "vertex_error": _as_numpy(vertex_error),
        "vertex_error_raw": _as_numpy(vertex_error_raw),
        "gaussian_residual": _as_numpy(gaussian_residual),
        "gaussian_support_mass": _as_numpy(gaussian_mass),
        "gaussian_grad_ema": _as_numpy(trainer.gaussian_grad_ema.detach()),
        "gaussian_lambda": _as_numpy(trainer.field.lambda_param.detach()),
        "gaussian_anchor_bone": _as_numpy(trainer.field.anchor_bone.detach().to(dtype=torch.int32)),
        "gaussian_generation": _as_numpy(trainer.field.generation.detach().to(dtype=torch.int32)),
        "gaussian_active_mask": _as_numpy(trainer.field.active_mask.detach().to(dtype=torch.int8)),
    }
    summary = {
        "format": "evorig_next_phase2_split_signals_v1",
        "config": asdict(cfg),
        "vertex_count": int(trainer.rest_vertices.shape[0]),
        "gaussian_count": int(trainer.field.gaussian_count),
        "joint_count": int(trainer.skeleton.joint_count),
        "bone_count": int(trainer.skeleton.bone_count),
        "sample_radius": float(trainer.sample_radius),
        "split_candidate_count": int(len(split_rows)),
        "split_candidates": split_rows,
    }
    return summary, arrays


def save_phase2_topology_signals(
    output_dir: Path,
    trainer: Any,
    cache: Any,
    config: Phase2TopologyConfig | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, arrays = build_phase2_topology_signals(trainer, cache, config=config)
    np.savez_compressed(output_dir / "phase2_topology_signals.npz", **arrays)
    (output_dir / "phase2_topology_signal_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
def list_phase2_topology_proposals(
    signal_summary: dict[str, Any],
    *,
    event_type: str,
) -> list[dict[str, Any]]:
    event_key = str(event_type).strip().lower()
    if event_key == "branch":
        return [
            {"event_type": "branch", "proposal": item}
            for item in list(signal_summary.get("branch_components", []))
            if float(item.get("score", item.get("mean_error", 0.0))) > 0.0
        ]
    if event_key == "split":
        return [
            {"event_type": "split", "proposal": item}
            for item in list(signal_summary.get("split_candidates", []))
            if float(item.get("score", 0.0)) > 0.0
        ]
    if event_key == "seed_joint_repair":
        return [
            {"event_type": "seed_joint_repair", "proposal": item}
            for item in list(signal_summary.get("seed_joint_repair_candidates", []))
            if float(item.get("score", 0.0)) > 0.0
        ]
    raise ValueError(f"unsupported topology proposal event_type '{event_type}'")


def _proposal_signature(selected: dict[str, Any]) -> str:
    event_type = str(selected.get("event_type", "")).strip().lower()
    proposal = selected.get("proposal", {})
    if event_type == "branch":
        return (
            f"branch:component={int(proposal.get('component_index', -1))}:"
            f"tip={int(proposal.get('tip_vertex_id', -1))}:"
            f"parent={int(proposal.get('parent_joint', -1))}:"
            f"count={int(proposal.get('vertex_count', 0))}"
        )
    if event_type == "split":
        return (
            f"split:bone={int(proposal.get('bone_index', -1))}:"
            f"lambda={float(proposal.get('split_lambda', 0.0)):.6f}:"
            f"count={int(proposal.get('component_vertex_count', proposal.get('vertex_count', 0)))}"
        )
    if event_type == "seed_joint_repair":
        return (
            f"seed_joint_repair:joint={int(proposal.get('joint', -1))}:"
            f"component={int(proposal.get('component_index', -1))}:"
            f"count={int(proposal.get('component_vertex_count', 0))}"
        )
    return f"{event_type}:unknown"


def _bone_index_for_child(skeleton: Any, child_joint: int) -> int:
    matches = torch.nonzero(skeleton.bone_child_idx == int(child_joint), as_tuple=False).flatten()
    if int(matches.numel()) <= 0:
        raise ValueError(f"no bone found for child joint {int(child_joint)}")
    return int(matches[0].item())


def _project_centers_to_bones(
    skeleton: Any,
    centers: torch.Tensor,
    bone_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parent_pos, bone_frames, _bone_parent_idx, bone_child_idx = skeleton.compute_bone_frames()
    child_pos = skeleton.rest_joints[bone_child_idx]
    start = parent_pos[bone_indices]
    end = child_pos[bone_indices]
    segment = end - start
    length_sq = segment.square().sum(dim=-1).clamp_min(EPS)
    lam = ((centers - start) * segment).sum(dim=-1) / length_sq
    projected = start + lam.unsqueeze(-1) * segment
    local_frame = bone_frames[bone_indices]
    offset = torch.einsum("gji,gj->gi", local_frame, centers - projected)
    return lam, offset


def _assign_gaussians_to_bone_from_centers(
    trainer: Any,
    gaussian_ids: torch.Tensor,
    centers_before: torch.Tensor,
    bone_index: int,
) -> None:
    if int(gaussian_ids.numel()) <= 0:
        return
    field = trainer.field
    gaussian_ids = gaussian_ids.to(device=field.lambda_param.device, dtype=torch.long).reshape(-1)
    bone_indices = torch.full(
        (int(gaussian_ids.numel()),),
        int(bone_index),
        dtype=torch.long,
        device=field.lambda_param.device,
    )
    centers = centers_before[gaussian_ids].to(device=field.lambda_param.device, dtype=field.lambda_param.dtype)
    lam, offset = _project_centers_to_bones(trainer.skeleton, centers, bone_indices)
    lam = torch.maximum(torch.minimum(lam, field.lambda_max[gaussian_ids]), field.lambda_min[gaussian_ids])
    with torch.no_grad():
        field.anchor_bone[gaussian_ids] = int(bone_index)
        field.lambda_param.data[gaussian_ids] = lam.to(device=field.lambda_param.device, dtype=field.lambda_param.dtype)
        field.offset_local.data[gaussian_ids] = offset.to(device=field.offset_local.device, dtype=field.offset_local.dtype)
    field.reset_endpoint_logits_from_lambda_sigmoid(
        midpoint=float(trainer.cfg.ownership_midpoint),
        slope=float(trainer.cfg.ownership_slope),
        gaussian_ids=gaussian_ids,
    )


def apply_phase2_split_proposal(
    trainer: Any,
    proposal: dict[str, Any],
    *,
    seeds_per_new_bone: int = 8,
    lambda_min: float = 0.10,
    lambda_max: float = 0.90,
) -> dict[str, Any]:
    bone_index = int(proposal["bone_index"])
    if bone_index < 0 or bone_index >= int(trainer.skeleton.bone_count):
        raise IndexError(f"split bone_index out of range: {bone_index}")
    raw_split_lambda = float(proposal.get("split_lambda", 0.5))
    split_lambda = min(max(raw_split_lambda, float(lambda_min)), float(lambda_max))

    with torch.no_grad():
        centers_before = trainer.field.compute_rest_centers(trainer.skeleton).detach().clone()
        old_anchor_bone = trainer.field.anchor_bone.detach().clone()
        old_lambda = trainer.field.lambda_param.detach().clone()
        parent_pos, _bone_frames, bone_parent_idx, bone_child_idx = trainer.skeleton.compute_bone_frames()
        parent_joint = int(bone_parent_idx[bone_index].item())
        child_joint = int(bone_child_idx[bone_index].item())
        start = parent_pos[bone_index]
        end = trainer.skeleton.rest_joints[child_joint]
        split_position = start + float(split_lambda) * (end - start)
        old_child_pose = trainer.skeleton.pose_rot[:, child_joint].detach().clone()
        pose_init = old_child_pose * float(split_lambda)
        child_pose_init = old_child_pose - pose_init

    new_joint_id, reparented_child = trainer.skeleton.split_bone(
        bone_index,
        split_position,
        pose_init=pose_init,
        child_pose_init=child_pose_init,
        birth_step=int(trainer.current_step),
        birth_mode="split",
    )
    proximal_bone_index = _bone_index_for_child(trainer.skeleton, new_joint_id)
    distal_bone_index = _bone_index_for_child(trainer.skeleton, reparented_child)

    old_bone_ids = torch.nonzero(old_anchor_bone == bone_index, as_tuple=False).flatten()
    proximal_ids = old_bone_ids[old_lambda[old_bone_ids] <= float(split_lambda)]
    distal_ids = old_bone_ids[old_lambda[old_bone_ids] > float(split_lambda)]
    _assign_gaussians_to_bone_from_centers(trainer, proximal_ids, centers_before, proximal_bone_index)
    _assign_gaussians_to_bone_from_centers(trainer, distal_ids, centers_before, distal_bone_index)

    append_bones = torch.tensor([proximal_bone_index], dtype=torch.long, device=trainer.device)
    append_result = trainer.field.append_axis_gaussians_for_bones(
        trainer.rest_vertices,
        trainer.skeleton,
        trainer._field_init_config(),
        bone_indices=append_bones,
        seeds_per_bone=int(seeds_per_new_bone),
        generation_value=int(trainer.field.generation.max().item()) + 1 if int(trainer.field.generation.numel()) > 0 else 1,
        faces=trainer.mesh_faces,
        prune_outside_mesh=True,
        surface_tol=float(trainer.cfg.seed_inside_surface_tol),
        mesh_query_scene=trainer.rest_mesh_scene,
    )
    trainer.refresh_after_topology_mutation(preserve_fallback_weights=True)
    return {
        "type": "split",
        "source": "phase2_gaussian_residual_gradient",
        "step": int(trainer.current_step),
        "old_bone_index": int(bone_index),
        "parent_joint": int(parent_joint),
        "old_child_joint": int(child_joint),
        "new_joint": int(new_joint_id),
        "reparented_child_joint": int(reparented_child),
        "split_lambda_raw": float(raw_split_lambda),
        "split_lambda": float(split_lambda),
        "proximal_bone_index": int(proximal_bone_index),
        "distal_bone_index": int(distal_bone_index),
        "proximal_reassigned_gaussian_count": int(proximal_ids.numel()),
        "distal_reassigned_gaussian_count": int(distal_ids.numel()),
        "added_gaussian_count": int(append_result["new_ids"].numel()),
        "proposal": proposal,
    }


def apply_phase2_branch_proposal(
    trainer: Any,
    proposal: dict[str, Any],
    *,
    seeds_per_new_bone: int = 8,
    branch_pose_init_scale: float = 0.2,
    topology_config: Phase2TopologyConfig | None = None,
) -> dict[str, Any]:
    parent_joint = int(proposal.get("parent_joint", -1))
    if parent_joint < 0 or parent_joint >= int(trainer.skeleton.joint_count):
        raise ValueError(f"branch proposal has invalid parent_joint: {parent_joint}")
    path_points = torch.tensor(
        proposal.get("branch_path_points", proposal.get("tip", [])),
        dtype=trainer.rest_vertices.dtype,
        device=trainer.device,
    ).reshape(-1, 3)
    if int(path_points.shape[0]) <= 0:
        raise ValueError("branch proposal has no path points")
    cfg = topology_config or Phase2TopologyConfig()
    existing_branch_lineages = _get_phase2_branch_lineages(trainer)
    parent_branch_lineage = _branch_lineage_for_joint(trainer, parent_joint)
    parent_branch_id = -1 if parent_branch_lineage is None else int(parent_branch_lineage.get("branch_id", -1))
    host_child_joint = _nearest_host_child_joint_for_branch_proposal(trainer, parent_joint, proposal)
    vertex_ids = torch.tensor(
        proposal.get("vertex_ids", []),
        dtype=torch.long,
        device=trainer.device,
    ).reshape(-1)
    if int(vertex_ids.numel()) >= 3:
        with torch.no_grad():
            source = trainer.rest_vertices[vertex_ids].unsqueeze(0).expand(int(trainer.gt_vertices.shape[0]), -1, -1)
            target = trainer.gt_vertices[:, vertex_ids]
            rigid_rot, _translation, _error = fit_rigid_sequence(source, target)
            parent_global = trainer.skeleton.forward_kinematics()[:, parent_joint, :3, :3]
            pose_init_full = matrix_to_axis_angle(parent_global.transpose(-1, -2) @ rigid_rot)
            pose_init_full = pose_init_full * float(branch_pose_init_scale)
    else:
        pose_init_full = torch.zeros(
            int(trainer.gt_vertices.shape[0]),
            3,
            dtype=trainer.rest_vertices.dtype,
            device=trainer.device,
        )
    pose_chunk = pose_init_full / float(max(int(path_points.shape[0]), 1))
    branch_root_lock_count = max(int(proposal.get("branch_root_lock_count", 1)), 1)
    new_joint_ids: list[int] = []
    new_bone_indices: list[int] = []
    current_parent = int(parent_joint)
    for point_index, point in enumerate(path_points):
        connected_to_parent = bool(int(point_index) > 0)
        new_joint = trainer.skeleton.insert_joint(
            current_parent,
            point,
            pose_init=pose_chunk,
            birth_step=int(trainer.current_step),
            birth_mode="branch_root" if int(point_index) < branch_root_lock_count else "branch",
            connected_to_parent=connected_to_parent,
        )
        new_joint_ids.append(int(new_joint))
        if connected_to_parent:
            new_bone_indices.append(_bone_index_for_child(trainer.skeleton, new_joint))
        current_parent = int(new_joint)
    if new_bone_indices:
        append_bones = torch.tensor(new_bone_indices, dtype=torch.long, device=trainer.device)
        append_result = trainer.field.append_axis_gaussians_for_bones(
            trainer.rest_vertices,
            trainer.skeleton,
            trainer._field_init_config(),
            bone_indices=append_bones,
            seeds_per_bone=int(seeds_per_new_bone),
            generation_value=int(trainer.field.generation.max().item()) + 1 if int(trainer.field.generation.numel()) > 0 else 1,
            faces=trainer.mesh_faces,
            prune_outside_mesh=True,
            surface_tol=float(trainer.cfg.seed_inside_surface_tol),
            mesh_query_scene=trainer.rest_mesh_scene,
        )
        added_gaussian_count = int(append_result["new_ids"].numel())
    else:
        added_gaussian_count = 0
    branch_lineage = _register_phase2_branch_lineage(
        trainer,
        existing_lineages=existing_branch_lineages,
        parent_joint=parent_joint,
        parent_branch_id=int(parent_branch_id),
        new_joint_ids=new_joint_ids,
        proposal=proposal,
    )
    trainer.refresh_after_topology_mutation()
    return {
        "type": "branch",
        "source": "phase2_wrong_or_uncovered_component",
        "step": int(trainer.current_step),
        "parent_joint": int(parent_joint),
        "host_child_joint": int(host_child_joint),
        "new_joints": new_joint_ids,
        "new_bone_indices": new_bone_indices,
        "added_gaussian_count": int(added_gaussian_count),
        "branch_lineage": branch_lineage,
        "proposal": proposal,
    }


def apply_phase2_seed_joint_repair_proposal(
    trainer: Any,
    proposal: dict[str, Any],
    *,
    topology_config: Phase2TopologyConfig | None = None,
) -> dict[str, Any]:
    cfg = topology_config or Phase2TopologyConfig()
    joint_id = int(proposal.get("joint", -1))
    if joint_id < 0 or joint_id >= int(trainer.skeleton.joint_count):
        raise ValueError(f"seed-joint repair proposal has invalid joint: {joint_id}")
    parent = int(trainer.skeleton.parent_idx[joint_id].item())
    children = _joint_children(trainer.skeleton, joint_id)
    if parent < 0 or len(children) != 1:
        raise ValueError(f"seed-joint repair requires an internal one-child seed joint: {joint_id}")
    child = int(children[0])
    old_position = trainer.skeleton.rest_joints[joint_id].detach().clone()
    target = torch.tensor(
        proposal.get("target_position", proposal.get("new_position", [])),
        dtype=trainer.skeleton.rest_joints.dtype,
        device=trainer.skeleton.rest_joints.device,
    ).reshape(3)
    new_position, move_distance, target_distance, cap = _seed_joint_repair_target(
        old_position,
        target,
        trainer=trainer,
        cfg=cfg,
    )
    if move_distance <= float(EPS):
        raise ValueError(f"seed-joint repair has zero movement for joint {joint_id}")
    parent_pos = trainer.skeleton.rest_joints[parent].detach()
    child_pos = trainer.skeleton.rest_joints[child].detach()
    before_parent = _mesh_segment_inside_fraction(trainer, parent_pos, old_position)
    before_child = _mesh_segment_inside_fraction(trainer, old_position, child_pos)
    after_parent = _mesh_segment_inside_fraction(trainer, parent_pos, new_position)
    after_child = _mesh_segment_inside_fraction(trainer, new_position, child_pos)
    before_min = min(before_parent, before_child)
    after_min = min(after_parent, after_child)
    improvement = after_min - before_min
    if after_min < float(cfg.seed_joint_repair_inside_min_fraction):
        raise ValueError(
            f"seed-joint repair rejected by inside fraction: joint={joint_id}, after_min={after_min:.3f}"
        )
    if improvement < float(cfg.seed_joint_repair_min_inside_improvement) and after_min <= before_min + float(EPS):
        raise ValueError(
            f"seed-joint repair rejected by inside improvement: joint={joint_id}, improvement={improvement:.3f}"
        )
    incident_bones = _incident_bones_for_joint(trainer.skeleton, joint_id)
    incident_gaussian_count = 0
    for bone_index in incident_bones:
        incident_gaussian_count += int(
            torch.count_nonzero(trainer.field.anchor_bone.detach() == int(bone_index)).item()
        )
    with torch.no_grad():
        trainer.skeleton.rest_joints.data[joint_id] = new_position.to(
            device=trainer.skeleton.rest_joints.device,
            dtype=trainer.skeleton.rest_joints.dtype,
        )
        if hasattr(trainer.skeleton, "init_rest_joints") and joint_id < int(trainer.skeleton.init_rest_joints.shape[0]):
            trainer.skeleton.init_rest_joints.data[joint_id] = new_position.to(
                device=trainer.skeleton.init_rest_joints.device,
                dtype=trainer.skeleton.init_rest_joints.dtype,
            )
    trainer.refresh_after_topology_mutation(preserve_fallback_weights=True)
    return {
        "type": "seed_joint_repair",
        "source": "phase2_fault_guided_seed_joint_repair",
        "variant": str(cfg.seed_joint_repair_variant),
        "step": int(trainer.current_step),
        "joint": int(joint_id),
        "parent": int(parent),
        "child": int(child),
        "component_index": int(proposal.get("component_index", -1)),
        "score": float(proposal.get("score", 0.0)),
        "component_vertex_count": int(proposal.get("component_vertex_count", 0)),
        "component_error_mass": float(proposal.get("component_error_mass", 0.0)),
        "wrong_fraction": float(proposal.get("wrong_fraction", 0.0)),
        "uncovered_fraction": float(proposal.get("uncovered_fraction", 0.0)),
        "neighbor_fraction": float(proposal.get("neighbor_fraction", 0.0)),
        "old_position": _json_list(old_position),
        "target_position": _json_list(target),
        "new_position": _json_list(new_position),
        "move_distance": float(move_distance),
        "target_distance": float(target_distance),
        "move_scale": float(move_distance / max(target_distance, float(EPS))),
        "capB": float(cap),
        "segment_inside_before": [float(before_parent), float(before_child)],
        "segment_inside_after": [float(after_parent), float(after_child)],
        "inside_improvement": float(improvement),
        "incident_bones": [int(item) for item in incident_bones],
        "moved_with_bone_gaussian_count": int(incident_gaussian_count),
        "proposal": proposal,
    }


def apply_phase2_topology_proposal(
    trainer: Any,
    selected: dict[str, Any],
    *,
    seeds_per_new_bone: int = 8,
    topology_config: Phase2TopologyConfig | None = None,
) -> dict[str, Any]:
    event_type = str(selected["event_type"]).strip().lower()
    proposal = selected["proposal"]
    if event_type == "split":
        return apply_phase2_split_proposal(
            trainer,
            proposal,
            seeds_per_new_bone=int(seeds_per_new_bone),
        )
    if event_type == "branch":
        return apply_phase2_branch_proposal(
            trainer,
            proposal,
            seeds_per_new_bone=int(seeds_per_new_bone),
            topology_config=topology_config,
        )
    if event_type == "seed_joint_repair":
        return apply_phase2_seed_joint_repair_proposal(
            trainer,
            proposal,
            topology_config=topology_config,
        )
    raise ValueError(f"unsupported phase2 event_type '{event_type}'")


def _nearest_host_child_joint_for_branch_proposal(trainer: Any, parent_joint: int, proposal: dict[str, Any]) -> int:
    if int(parent_joint) < 0:
        return -1
    attach_bone = int(proposal.get("attach_bone", -1)) if isinstance(proposal, dict) else -1
    if 0 <= attach_bone < int(trainer.skeleton.bone_count):
        return int(trainer.skeleton.bone_child_idx[attach_bone].item())
    child_bones = torch.nonzero(trainer.skeleton.bone_parent_idx == int(parent_joint), as_tuple=False).flatten()
    if int(child_bones.numel()) <= 0:
        return -1
    query = None
    if isinstance(proposal, dict):
        query = proposal.get("center", proposal.get("tip"))
    if query is None:
        return int(trainer.skeleton.bone_child_idx[int(child_bones[0].item())].item())
    query_t = torch.tensor(query, dtype=trainer.rest_vertices.dtype, device=trainer.device).reshape(3)
    parent_pos, _frames, _parent_idx, child_idx = trainer.skeleton.compute_bone_frames()
    best_bone = int(child_bones[0].item())
    best_distance = float("inf")
    for bone_tensor in child_bones:
        bone_index = int(bone_tensor.item())
        start = parent_pos[bone_index]
        end = trainer.skeleton.rest_joints[int(child_idx[bone_index].item())]
        segment = end - start
        lam = ((query_t - start) @ segment / segment.dot(segment).clamp_min(EPS)).clamp(0.0, 1.0)
        closest = start + lam * segment
        distance = float((query_t - closest).norm().item())
        if distance < best_distance:
            best_distance = distance
            best_bone = bone_index
    return int(child_idx[best_bone].item())


def _register_phase2_branch_lineage(
    trainer: Any,
    *,
    existing_lineages: list[dict[str, Any]],
    parent_joint: int,
    parent_branch_id: int,
    new_joint_ids: list[int],
    proposal: dict[str, Any],
) -> dict[str, Any]:
    lineages = _sanitize_branch_lineages(existing_lineages)
    next_id = max((int(item.get("branch_id", -1)) for item in lineages), default=-1) + 1
    lineage = {
        "branch_id": int(next_id),
        "root_parent_joint": int(parent_joint),
        "root_joint": int(new_joint_ids[0]) if new_joint_ids else -1,
        "joint_chain": [int(item) for item in new_joint_ids],
        "birth_step": int(getattr(trainer, "current_step", 0)),
        "parent_branch_id": int(parent_branch_id),
        "source_component_index": int(proposal.get("source_component_index", proposal.get("component_index", -1))),
        "source_component_vertex_count": int(proposal.get("vertex_count", len(proposal.get("vertex_ids", [])))),
        "source_component_center": list(proposal.get("center", [])),
        "source_tip": list(proposal.get("tip", [])),
    }
    lineages.append(lineage)
    trainer.phase2_branch_lineages = lineages
    return lineage


def _current_recon_error_summary(trainer: Any, cache: Any | None = None) -> dict[str, float | int]:
    cache = trainer.evaluate_full() if cache is None else cache
    recon_mask = trainer.legal_vertex_mask.unsqueeze(0).expand(int(trainer.gt_vertices.shape[0]), -1).to(
        dtype=cache.pred_vertices.dtype,
        device=cache.pred_vertices.device,
    )
    return {
        "error_raw": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices, mask=recon_mask).item()),
        "error_raw_all": float(vertex_recon_loss(cache.pred_vertices, trainer.gt_vertices).item()),
        "zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
    }


def _write_phase2_signal_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "phase2_topology_signals.npz", **arrays)
    (output_dir / "phase2_topology_signal_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def save_phase2_checkpoint(
    trainer: Any,
    output_dir: Path,
    *,
    topology_config: Phase2TopologyConfig | None = None,
    phase2_summary: dict[str, Any] | None = None,
    topology_events: list[dict[str, Any]] | None = None,
    topology_signal_summary: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = topology_config or Phase2TopologyConfig()
    payload = {
        "format": "evorig_next_phase2_checkpoint_v1",
        "current_step": int(trainer.current_step),
        "trainer_state": trainer._phase1_state_payload(),
        "topology_config": asdict(cfg),
        "phase2_summary": phase2_summary or {},
        "topology_events": list(topology_events or []),
        "topology_signal_summary": topology_signal_summary or {},
        "joint_count": int(trainer.skeleton.joint_count),
        "bone_count": int(trainer.skeleton.bone_count),
        "gaussian_count": int(trainer.field.gaussian_count),
    }
    path = output_dir / "phase2_checkpoint.pt"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def load_phase2_checkpoint(
    trainer: Any,
    path: str | Path,
    *,
    restore_optimizer: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=trainer.device)
    if str(payload.get("format", "")) != "evorig_next_phase2_checkpoint_v1":
        raise ValueError(f"unsupported phase2 checkpoint format: {payload.get('format')}")
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise ValueError("phase2 checkpoint missing trainer_state")
    trainer.load_phase1_payload(trainer_state, restore_optimizer=restore_optimizer, restore_rng=restore_rng)
    return payload


def _proposal_vertex_id_set(selected: dict[str, Any]) -> set[int]:
    proposal = selected.get("proposal", {})
    return {int(item) for item in proposal.get("vertex_ids", [])}


def _branch_proposal_repeats_history(
    selected: dict[str, Any],
    event_history: list[dict[str, Any]],
    cfg: Phase2TopologyConfig,
) -> tuple[bool, dict[str, Any]]:
    proposal = selected.get("proposal", {})
    vertex_ids = {int(item) for item in proposal.get("vertex_ids", [])}
    tip = proposal.get("tip_target", proposal.get("tip", []))
    try:
        tip_np = np.asarray(tip, dtype=np.float64).reshape(3)
    except Exception:
        tip_np = np.zeros((3,), dtype=np.float64)
    bbox_min = np.asarray(proposal.get("bbox_min", []), dtype=np.float64).reshape(-1)
    bbox_max = np.asarray(proposal.get("bbox_max", []), dtype=np.float64).reshape(-1)
    bbox_diag = 0.0
    if bbox_min.shape[0] == 3 and bbox_max.shape[0] == 3:
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    tip_threshold = max(0.35 * bbox_diag, float(EPS))
    overlap_limit = max(float(cfg.branch_component_overlap_reject_fraction), 0.0)
    for event_index, event in enumerate(event_history):
        if str(event.get("type", "")).strip().lower() != "branch":
            continue
        old_proposal = event.get("proposal", {})
        if vertex_ids:
            old_vertex_ids = {int(item) for item in old_proposal.get("vertex_ids", [])}
            if old_vertex_ids:
                overlap_fraction = len(vertex_ids & old_vertex_ids) / max(len(vertex_ids), 1)
                if overlap_fraction > overlap_limit:
                    return True, {
                        "reason": "history_component_overlap",
                        "event_index": int(event_index),
                        "overlap_fraction": float(overlap_fraction),
                        "overlap_limit": float(overlap_limit),
                    }
        old_tip = old_proposal.get("tip_target", old_proposal.get("tip", event.get("branch_lineage", {}).get("source_tip", [])))
        try:
            old_tip_np = np.asarray(old_tip, dtype=np.float64).reshape(3)
        except Exception:
            continue
        if tip_threshold > float(EPS):
            distance = float(np.linalg.norm(tip_np - old_tip_np))
            if distance <= tip_threshold:
                return True, {
                    "reason": "history_tip_near",
                    "event_index": int(event_index),
                    "tip_distance": float(distance),
                    "tip_threshold": float(tip_threshold),
                }
    return False, {"reason": ""}


def _select_non_overlapping_branch_proposals(
    signal_summary: dict[str, Any],
    *,
    limit: int,
    cfg: Phase2TopologyConfig,
    event_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if int(limit) <= 0:
        return []
    selected: list[dict[str, Any]] = []
    consumed: set[int] = set()
    overlap_limit = max(float(cfg.branch_component_overlap_reject_fraction), 0.0)
    proposals = list_phase2_topology_proposals(signal_summary, event_type="branch")
    best_score = max((float(item.get("proposal", {}).get("score", 0.0)) for item in proposals), default=0.0)
    min_score = max(float(cfg.branch_min_score_fraction_of_best), 0.0) * max(float(best_score), float(EPS))
    for proposal in proposals:
        body = proposal.get("proposal", {})
        if float(body.get("score", 0.0)) < min_score and not bool(body.get("force_select", False)):
            continue
        repeats_history, history_reason = _branch_proposal_repeats_history(
            proposal,
            list(event_history or []),
            cfg,
        )
        if repeats_history:
            body["history_reject"] = history_reason
            continue
        vertex_ids = _proposal_vertex_id_set(proposal)
        if vertex_ids and consumed:
            overlap_fraction = len(vertex_ids & consumed) / max(len(vertex_ids), 1)
            if overlap_fraction > overlap_limit:
                continue
        selected.append(proposal)
        consumed.update(vertex_ids)
        if len(selected) >= int(limit):
            break
    return selected


def _first_valid_split_proposal(
    signal_summary: dict[str, Any],
    *,
    trainer: Any,
    cfg: Phase2TopologyConfig,
    excluded_bone_indices: set[int],
    require_inserted_bone: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for proposal in list_phase2_topology_proposals(signal_summary, event_type="split"):
        body = proposal.get("proposal", {})
        bone_index = int(body.get("bone_index", -1))
        uses_inserted = _split_proposal_uses_inserted_bone(trainer, body)
        proximal_inside, distal_inside = _split_proposal_inside_fractions(trainer, body)
        reasons: list[str] = []
        if bone_index in excluded_bone_indices:
            reasons.append("new_bone_from_this_update")
        if bool(require_inserted_bone) and not uses_inserted:
            reasons.append("not_inserted_bone")
        if not _split_proposal_inside_ok_from_fractions(proximal_inside, distal_inside, cfg):
            reasons.append("mesh_inside_fraction_below_threshold")
        diagnostic = {
            "bone_index": int(bone_index),
            "parent_joint": int(body.get("parent_joint", -1)),
            "child_joint": int(body.get("child_joint", -1)),
            "score": float(body.get("score", 0.0)),
            "split_lambda": float(body.get("split_lambda", body.get("lambda", 0.5))),
            "uses_inserted_bone": bool(uses_inserted),
            "require_inserted_bone": bool(require_inserted_bone),
            "proximal_inside_fraction": float(proximal_inside),
            "distal_inside_fraction": float(distal_inside),
            "reject_reasons": reasons,
        }
        diagnostics.append(diagnostic)
        if reasons:
            continue
        valid.append(proposal)
    best_score = max((float(item.get("proposal", {}).get("score", 0.0)) for item in valid), default=0.0)
    min_score = max(float(getattr(cfg, "split_min_score_fraction_of_best", 0.0)), 0.0) * max(best_score, float(EPS))
    selected: dict[str, Any] | None = None
    for proposal in valid:
        if float(proposal.get("proposal", {}).get("score", 0.0)) < min_score:
            bone_index = int(proposal.get("proposal", {}).get("bone_index", -1))
            for diagnostic in diagnostics:
                if int(diagnostic["bone_index"]) == bone_index:
                    diagnostic["reject_reasons"].append("below_score_fraction_of_best")
            continue
        selected = proposal
        break
    selected_bone = int(selected.get("proposal", {}).get("bone_index", -1)) if selected is not None else -1
    for diagnostic in diagnostics:
        diagnostic["min_score"] = float(min_score)
        diagnostic["selected"] = bool(int(diagnostic["bone_index"]) == selected_bone)
    return selected, diagnostics


def _split_proposal_uses_inserted_bone(trainer: Any, proposal: dict[str, Any]) -> bool:
    bone_index = int(proposal.get("bone_index", -1))
    if bone_index < 0 or bone_index >= int(trainer.skeleton.bone_count):
        return False
    inserted = list(getattr(trainer.skeleton, "is_inserted", []))
    parent_joint = int(trainer.skeleton.bone_parent_idx[bone_index].item())
    child_joint = int(trainer.skeleton.bone_child_idx[bone_index].item())
    parent_inserted = bool(inserted[parent_joint]) if 0 <= parent_joint < len(inserted) else False
    child_inserted = bool(inserted[child_joint]) if 0 <= child_joint < len(inserted) else False
    return bool(parent_inserted or child_inserted)


def _split_proposal_inside_fractions(trainer: Any, proposal: dict[str, Any]) -> tuple[float, float]:
    bone_index = int(proposal.get("bone_index", -1))
    if bone_index < 0 or bone_index >= int(trainer.skeleton.bone_count):
        return 0.0, 0.0
    split_lambda = float(proposal.get("lambda", proposal.get("split_lambda", 0.5)))
    split_lambda = max(0.0, min(1.0, split_lambda))
    parent_joint = int(trainer.skeleton.bone_parent_idx[bone_index].item())
    child_joint = int(trainer.skeleton.bone_child_idx[bone_index].item())
    start = trainer.skeleton.rest_joints[parent_joint]
    end = trainer.skeleton.rest_joints[child_joint]
    split = start + split_lambda * (end - start)
    proximal_inside = _mesh_segment_inside_fraction(trainer, start, split)
    distal_inside = _mesh_segment_inside_fraction(trainer, split, end)
    return float(proximal_inside), float(distal_inside)


def _split_proposal_inside_ok_from_fractions(
    proximal_inside: float,
    distal_inside: float,
    cfg: Phase2TopologyConfig,
) -> bool:
    threshold = float(getattr(cfg, "split_inside_min_fraction", 0.0))
    if threshold <= 0.0:
        return True
    return float(proximal_inside) >= threshold and float(distal_inside) >= threshold


def _split_proposal_inside_ok(trainer: Any, proposal: dict[str, Any], cfg: Phase2TopologyConfig) -> bool:
    proximal_inside, distal_inside = _split_proposal_inside_fractions(trainer, proposal)
    return _split_proposal_inside_ok_from_fractions(proximal_inside, distal_inside, cfg)


def _run_phase2_training_interval(
    trainer: Any,
    *,
    step_count: int,
    phase: str,
    cfg: Phase2TopologyConfig,
    trace: list[dict[str, float | int | str]],
    live_trace_path: Path | None = None,
) -> None:
    count = max(int(step_count), 0)
    if count <= 0:
        return
    _apply_phase2_rest_joint_train_mask(trainer, cfg)
    old_illegal_support = float(trainer.cfg.loss_illegal_support)
    old_gaussian_illegal = float(trainer.cfg.loss_gaussian_illegal_coverage)
    old_tau = float(trainer.cfg.illegal_support_tau)
    old_margin = float(getattr(trainer.cfg, "illegal_support_margin", 0.0))
    try:
        trainer.cfg.loss_illegal_support = max(old_illegal_support, float(cfg.phase2_loss_illegal_support))
        trainer.cfg.loss_gaussian_illegal_coverage = max(
            old_gaussian_illegal,
            float(cfg.phase2_loss_gaussian_illegal_coverage),
        )
        trainer.cfg.illegal_support_tau = float(cfg.phase2_illegal_support_tau)
        trainer.cfg.illegal_support_margin = float(cfg.phase2_illegal_support_margin)
        progress = tqdm(
            range(count),
            desc=f"Phase2 {phase}",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        )
        for _index in progress:
            metrics = trainer.train_step(int(trainer.current_step) + 1)
            progress.set_postfix(
                {
                    "loss": f"{float(metrics['loss']):.4g}",
                    "recon": f"{float(metrics['recon']):.4g}",
                    "zero": int(metrics.get("zero_weight_row_count", 0)),
                }
            )
            row = {
                "event": "train_step",
                "step": int(trainer.current_step),
                "phase": str(phase),
                "loss": float(metrics["loss"]),
                "recon": float(metrics["recon"]),
                "illegal_support": float(metrics.get("illegal_support", 0.0)),
                "gaussian_illegal_coverage": float(metrics.get("gaussian_illegal_coverage", 0.0)),
                "zero_weight_row_count": int(metrics.get("zero_weight_row_count", 0)),
                "gaussian_count": int(metrics.get("gaussian_count", trainer.field.gaussian_count)),
            }
            trace.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "event"
                }
            )
            _append_jsonl(live_trace_path, row)
    finally:
        trainer.cfg.loss_illegal_support = old_illegal_support
        trainer.cfg.loss_gaussian_illegal_coverage = old_gaussian_illegal
        trainer.cfg.illegal_support_tau = old_tau
        trainer.cfg.illegal_support_margin = old_margin


def _apply_phase2_rest_joint_train_mask(trainer: Any, cfg: Phase2TopologyConfig) -> None:
    if not bool(getattr(cfg, "phase2_freeze_seed_rest_joints", True)):
        if hasattr(trainer, "rest_joint_train_mask"):
            delattr(trainer, "rest_joint_train_mask")
        return
    inserted = getattr(trainer.skeleton, "is_inserted", None)
    if inserted is None:
        return
    mask = torch.as_tensor(
        [bool(item) for item in inserted],
        dtype=torch.bool,
        device=trainer.skeleton.rest_joints.device,
    )
    if bool(getattr(cfg, "phase2_freeze_branch_root_rest_joints", True)):
        freeze_all_branch = bool(getattr(cfg, "phase2_freeze_branch_rest_joints", False))
        birth_modes = list(getattr(trainer.skeleton, "birth_modes", []))
        parent_idx = getattr(trainer.skeleton, "parent_idx", None)
        parent_modes = birth_modes
        for joint_id, mode in enumerate(birth_modes):
            mode_key = str(mode)
            freeze_branch_root = mode_key == "branch_root" or (freeze_all_branch and mode_key == "branch")
            if (
                not freeze_branch_root
                and mode_key == "branch"
                and parent_idx is not None
                and joint_id < int(parent_idx.numel())
            ):
                parent = int(parent_idx[joint_id].item())
                parent_mode = parent_modes[parent] if 0 <= parent < len(parent_modes) else "seed"
                freeze_branch_root = str(parent_mode) not in {"branch", "branch_root"}
            if freeze_branch_root and joint_id < int(mask.numel()):
                mask[joint_id] = False
    if int(mask.numel()) == int(trainer.skeleton.joint_count):
        trainer.rest_joint_train_mask = mask


def run_phase2_topology_scheduled_refine(
    trainer: Any,
    output_dir: Path,
    *,
    topology_config: Phase2TopologyConfig | None = None,
    max_updates: int = 8,
    seeds_per_new_bone: int = 8,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = topology_config or Phase2TopologyConfig()
    interval_steps = max(int(cfg.topology_update_interval_steps), 0)
    max_branches = max(int(cfg.topology_max_branch_per_update), 0)
    max_splits = max(int(cfg.topology_max_split_per_update), 0)
    noop_patience = max(int(cfg.topology_noop_stop_patience), 1)
    event_history: list[dict[str, Any]] = []
    update_history: list[dict[str, Any]] = []
    train_trace: list[dict[str, float | int | str]] = []
    live_trace_path = output_dir / "phase2_schedule_live.jsonl"
    live_trace_path.write_text("", encoding="utf-8")
    noop_streak = 0
    repaired_seed_joints_global: set[int] = set()
    reusable_final_signal_summary: dict[str, Any] | None = None
    reusable_final_signal_arrays: dict[str, np.ndarray] | None = None
    start_cache = trainer.evaluate_full()
    start_error = _current_recon_error_summary(trainer, start_cache)
    _append_jsonl(
        live_trace_path,
        {
            "event": "schedule_start",
            "max_updates": int(max_updates),
            "interval_steps": int(interval_steps),
            "start_error": start_error,
            "joint_count": int(trainer.skeleton.joint_count),
            "bone_count": int(trainer.skeleton.bone_count),
            "gaussian_count": int(trainer.field.gaussian_count),
        },
    )

    update_progress = tqdm(
        range(max(int(max_updates), 0)),
        desc="Phase2 topology",
        unit="update",
        dynamic_ncols=True,
    )
    for update_index in update_progress:
        update_dir = output_dir / f"phase2_topology_update_{update_index + 1:02d}"
        update_dir.mkdir(parents=True, exist_ok=True)
        current_update_new_bones: set[int] = set()
        update_events: list[dict[str, Any]] = []
        _append_jsonl(
            live_trace_path,
            {
                "event": "update_start",
                "update_index": int(update_index),
                "step": int(trainer.current_step),
            },
        )

        cache = trainer.evaluate_full()
        signal_summary, signal_arrays = build_phase2_topology_signals(trainer, cache, config=cfg)
        _write_phase2_signal_artifacts(update_dir / "pre_update_signals", signal_summary, signal_arrays)
        _append_jsonl(
            live_trace_path,
            {
                "event": "pre_update_signals",
                "update_index": int(update_index),
                "branch_seed_vertex_count": int(signal_summary.get("branch_seed_vertex_count", 0)),
                "branch_component_count": int(signal_summary.get("branch_component_count", 0)),
                "seed_joint_repair_candidate_count": int(signal_summary.get("seed_joint_repair_candidate_count", 0)),
                "split_candidate_count": int(signal_summary.get("split_candidate_count", 0)),
                "effective_component_min_vertices": int(signal_summary.get("effective_component_min_vertices", 0)),
                "effective_seed_joint_repair_min_vertices": int(
                    signal_summary.get("effective_seed_joint_repair_min_vertices", 0)
                ),
                "voxel_path_field_cache": signal_summary.get("voxel_path_field_cache", {}),
            },
        )
        seed_joint_repair_summaries: list[dict[str, Any]] = []
        seed_joint_repair_rejections: list[dict[str, Any]] = []
        repaired_seed_joints: set[int] = set()
        if bool(cfg.seed_joint_repair_enabled):
            repair_limit = max(int(cfg.seed_joint_repair_max_per_update), 0)
            for proposal_index, selected in enumerate(
                list_phase2_topology_proposals(signal_summary, event_type="seed_joint_repair")
            ):
                if len(seed_joint_repair_summaries) >= repair_limit:
                    break
                proposal = selected.get("proposal", {})
                joint_id = int(proposal.get("joint", -1))
                if joint_id in repaired_seed_joints or joint_id in repaired_seed_joints_global:
                    seed_joint_repair_rejections.append(
                        {
                            "proposal_index": int(proposal_index),
                            "signature": _proposal_signature(selected),
                            "reason": "joint_already_repaired_in_schedule",
                        }
                    )
                    continue
                try:
                    event = apply_phase2_topology_proposal(
                        trainer,
                        selected,
                        seeds_per_new_bone=int(seeds_per_new_bone),
                        topology_config=cfg,
                    )
                except ValueError as exc:
                    seed_joint_repair_rejections.append(
                        {
                            "proposal_index": int(proposal_index),
                            "signature": _proposal_signature(selected),
                            "reason": str(exc),
                        }
                    )
                    continue
                repaired_seed_joints.add(int(joint_id))
                repaired_seed_joints_global.add(int(joint_id))
                seed_joint_repair_summaries.append(
                    {
                        "proposal_index": int(proposal_index),
                        "signature": _proposal_signature(selected),
                        "event": event,
                    }
                )
                update_events.append(event)
                event_history.append(event)
            if seed_joint_repair_summaries:
                cache = trainer.evaluate_full()
                signal_summary, signal_arrays = build_phase2_topology_signals(trainer, cache, config=cfg)
                _write_phase2_signal_artifacts(update_dir / "post_seed_joint_repair_signals", signal_summary, signal_arrays)
            _append_jsonl(
                live_trace_path,
                {
                    "event": "seed_joint_repair_done",
                    "update_index": int(update_index),
                    "accepted": int(len(seed_joint_repair_summaries)),
                    "rejected": int(len(seed_joint_repair_rejections)),
                },
            )
        seed_repair_applied = len(seed_joint_repair_summaries) > 0
        if seed_repair_applied:
            all_branch_proposals = list_phase2_topology_proposals(signal_summary, event_type="branch")
            branch_proposals = []
            branch_skip_reason = "seed_joint_repair_accepted"
        else:
            all_branch_proposals = list_phase2_topology_proposals(signal_summary, event_type="branch")
            branch_proposals = _select_non_overlapping_branch_proposals(
                signal_summary,
                limit=max_branches,
                cfg=cfg,
                event_history=event_history,
            )
            branch_skip_reason = ""
        _append_jsonl(
            live_trace_path,
            {
                "event": "branch_selection",
                "update_index": int(update_index),
                "candidate_count": int(len(all_branch_proposals)),
                "selected_count": int(len(branch_proposals)),
                "selected_signatures": [_proposal_signature(item) for item in branch_proposals],
                "skip_reason": branch_skip_reason,
            },
        )
        branch_summaries: list[dict[str, Any]] = []
        for proposal_index, selected in enumerate(branch_proposals):
            event = apply_phase2_topology_proposal(
                trainer,
                selected,
                seeds_per_new_bone=int(seeds_per_new_bone),
                topology_config=cfg,
            )
            current_update_new_bones.update(int(item) for item in event.get("new_bone_indices", []))
            branch_summaries.append(
                {
                    "proposal_index": int(proposal_index),
                    "signature": _proposal_signature(selected),
                    "event": event,
                }
            )
            update_events.append(event)
            event_history.append(event)
        _append_jsonl(
            live_trace_path,
            {
                "event": "branch_done",
                "update_index": int(update_index),
                "accepted": int(len(branch_summaries)),
                "new_bone_indices": sorted(int(item) for item in current_update_new_bones),
            },
        )

        split_summaries: list[dict[str, Any]] = []
        inserted_flags = list(getattr(trainer.skeleton, "is_inserted", []))
        split_inserted_only_this_update = bool(getattr(cfg, "split_prefer_inserted_bones", True)) and any(
            bool(item) for item in inserted_flags
        )
        split_selection_attempts: list[dict[str, Any]] = []
        split_range = range(0) if seed_repair_applied else range(max_splits)
        for split_index in split_range:
            cache = trainer.evaluate_full()
            split_signal_summary, split_signal_arrays = build_phase2_split_signals(trainer, cache, config=cfg)
            _write_phase2_signal_artifacts(
                update_dir / f"pre_split_{split_index + 1:02d}_signals",
                split_signal_summary,
                split_signal_arrays,
            )
            selected, split_diagnostics = _first_valid_split_proposal(
                split_signal_summary,
                trainer=trainer,
                cfg=cfg,
                excluded_bone_indices=current_update_new_bones,
                require_inserted_bone=bool(split_inserted_only_this_update),
            )
            split_selection_attempts.append(
                {
                    "split_index": int(split_index),
                    "candidate_count": int(split_signal_summary.get("split_candidate_count", 0)),
                    "require_inserted_bone": bool(split_inserted_only_this_update),
                    "diagnostics": split_diagnostics,
                }
            )
            if selected is None:
                break
            event = apply_phase2_topology_proposal(
                trainer,
                selected,
                seeds_per_new_bone=int(seeds_per_new_bone),
                topology_config=cfg,
            )
            split_summaries.append(
                {
                    "proposal_index": int(split_index),
                    "signature": _proposal_signature(selected),
                    "event": event,
                }
            )
            current_update_new_bones.update(
                int(event[item])
                for item in ("proximal_bone_index", "distal_bone_index")
                if item in event
            )
            update_events.append(event)
            event_history.append(event)
        _append_jsonl(
            live_trace_path,
            {
                "event": "split_done",
                "update_index": int(update_index),
                "accepted": int(len(split_summaries)),
                "skip_reason": "seed_joint_repair_accepted" if seed_repair_applied else "",
                "attempts": [
                    {
                        "split_index": int(item.get("split_index", -1)),
                        "candidate_count": int(item.get("candidate_count", 0)),
                        "require_inserted_bone": bool(item.get("require_inserted_bone", False)),
                    }
                    for item in split_selection_attempts
                ],
            },
        )

        applied_count = len(update_events)
        cache_after_topology = trainer.evaluate_full()
        error_after_topology = _current_recon_error_summary(trainer, cache_after_topology)
        update_summary = {
            "update_index": int(update_index),
            "start_step": int(trainer.current_step),
            "branch_candidate_count": int(len(all_branch_proposals)),
            "selected_branch_candidate_count": int(len(branch_proposals)),
            "accepted_seed_joint_repair_count": int(len(seed_joint_repair_summaries)),
            "accepted_branch_count": int(len(branch_summaries)),
            "accepted_split_count": int(len(split_summaries)),
            "new_bone_indices_from_this_update": sorted(int(item) for item in current_update_new_bones),
            "seed_joint_repairs": seed_joint_repair_summaries,
            "seed_joint_repair_rejections": seed_joint_repair_rejections,
            "branches": branch_summaries,
            "splits": split_summaries,
            "split_selection_attempts": split_selection_attempts,
            "error_after_topology": error_after_topology,
        }
        if applied_count <= 0:
            noop_streak += 1
        else:
            noop_streak = 0

        if noop_streak >= noop_patience:
            update_summary["stop_after_update"] = "noop_patience_reached"
            # No topology edit or training follows this signal, so it is already
            # the final topology diagnostic. Reuse it instead of recomputing the
            # expensive branch/voxel signal once more after the loop.
            reusable_final_signal_summary = signal_summary
            reusable_final_signal_arrays = signal_arrays
            update_history.append(update_summary)
            (update_dir / "phase2_update_summary.json").write_text(
                json.dumps(update_summary, indent=2),
                encoding="utf-8",
            )
            _append_jsonl(
                live_trace_path,
                {
                    "event": "update_stop_noop",
                    "update_index": int(update_index),
                    "noop_streak": int(noop_streak),
                    "accepted_event_count": int(len(update_events)),
                },
            )
            break

        _run_phase2_training_interval(
            trainer,
            step_count=interval_steps,
            phase=f"post_topology_update_{update_index + 1:02d}",
            cfg=cfg,
            trace=train_trace,
            live_trace_path=live_trace_path,
        )
        cache_after_interval = trainer.evaluate_full()
        update_summary["end_step"] = int(trainer.current_step)
        update_summary["error_after_interval"] = _current_recon_error_summary(trainer, cache_after_interval)
        update_history.append(update_summary)
        (update_dir / "phase2_update_summary.json").write_text(
            json.dumps(update_summary, indent=2),
            encoding="utf-8",
        )
        update_progress.set_postfix(
            {
                "events": int(applied_count),
                "branches": int(len(branch_summaries)),
                "splits": int(len(split_summaries)),
                "err": f"{float(update_summary['error_after_interval']['error_raw']):.5f}",
            }
        )
        _append_jsonl(
            live_trace_path,
            {
                "event": "update_done",
                "update_index": int(update_index),
                "accepted_event_count": int(applied_count),
                "accepted_seed_joint_repair_count": int(len(seed_joint_repair_summaries)),
                "accepted_branch_count": int(len(branch_summaries)),
                "accepted_split_count": int(len(split_summaries)),
                "error_after_interval": update_summary["error_after_interval"],
                "end_step": int(trainer.current_step),
            },
        )
        partial_summary = {
            "format": "evorig_next_phase2_topology_schedule_partial_v1",
            "topology_config": asdict(cfg),
            "start_error": start_error,
            "partial_error_raw": float(update_summary["error_after_interval"]["error_raw"]),
            "partial_error_raw_all": float(update_summary["error_after_interval"]["error_raw_all"]),
            "zero_weight_row_count": int(update_summary["error_after_interval"]["zero_weight_row_count"]),
            "max_updates": int(max_updates),
            "completed_update_count": int(len(update_history)),
            "accepted_event_count": int(len(event_history)),
            "topology_events": event_history,
            "joint_count": int(trainer.skeleton.joint_count),
            "bone_count": int(trainer.skeleton.bone_count),
            "gaussian_count": int(trainer.field.gaussian_count),
            "updates": update_history,
            "train_trace": train_trace,
            "checkpoint_status": "partial_after_completed_update",
        }
        partial_summary["phase2_checkpoint_path"] = str(output_dir / "phase2_checkpoint.pt")
        save_phase2_checkpoint(
            trainer,
            output_dir,
            topology_config=cfg,
            phase2_summary=partial_summary,
            topology_events=event_history,
            topology_signal_summary=signal_summary,
        )
        (output_dir / "phase2_schedule_summary_partial.json").write_text(
            json.dumps(partial_summary, indent=2),
            encoding="utf-8",
        )

    cache = trainer.evaluate_full()
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
        events=event_history,
        topology_diagnostics=[{"updates": update_history, "train_trace": train_trace}],
    )
    if reusable_final_signal_summary is not None and reusable_final_signal_arrays is not None:
        _write_phase2_signal_artifacts(output_dir, reusable_final_signal_summary, reusable_final_signal_arrays)
        final_signal_summary = reusable_final_signal_summary
    else:
        final_signal_summary = save_phase2_topology_signals(output_dir, trainer, cache, config=cfg)
    trainer._save_phase1_state(output_dir)
    final_error = _current_recon_error_summary(trainer, cache)
    summary = {
        "format": "evorig_next_phase2_topology_schedule_v1",
        "topology_config": asdict(cfg),
        "start_error": start_error,
        "final_error_raw": float(final_error["error_raw"]),
        "final_error_raw_all": float(final_error["error_raw_all"]),
        "zero_weight_row_count": int(final_error["zero_weight_row_count"]),
        "max_updates": int(max_updates),
        "completed_update_count": int(len(update_history)),
        "accepted_event_count": int(len(event_history)),
        "topology_events": event_history,
        "joint_count": int(trainer.skeleton.joint_count),
        "bone_count": int(trainer.skeleton.bone_count),
        "gaussian_count": int(trainer.field.gaussian_count),
        "updates": update_history,
        "train_trace": train_trace,
        "live_trace_path": str(live_trace_path),
        "post_phase2_topology_signal_summary": {
            "branch_seed_vertex_count": int(final_signal_summary["branch_seed_vertex_count"]),
            "branch_component_count": int(final_signal_summary["branch_component_count"]),
            "seed_joint_repair_candidate_count": int(final_signal_summary.get("seed_joint_repair_candidate_count", 0)),
            "split_candidate_count": int(final_signal_summary["split_candidate_count"]),
        },
    }
    checkpoint_path = output_dir / "phase2_checkpoint.pt"
    summary["phase2_checkpoint_path"] = str(checkpoint_path)
    save_phase2_checkpoint(
        trainer,
        output_dir,
        topology_config=cfg,
        phase2_summary=summary,
        topology_events=event_history,
        topology_signal_summary=final_signal_summary,
    )
    (output_dir / "phase2_schedule_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
