from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from evorig_next.io.outputs import save_outputs
from evorig_next.models.gaussian_field import GaussianSupportField as LegacyGaussianSupportField
from evorig_next.models.lbs import lbs_deform
from evorig_next.training.losses import skeleton_anchor_loss
from evorig_next.utils.geometry import EPS, mesh_radius
from evorig_next.utils.mesh_ops import (
    MeshQueryScene,
    build_mesh_query_scene,
    project_points_inside_mesh,
    ray_mesh_first_hit_distance,
)

from evorig_next.phase1_config import Phase1Config, Phase1DensifyStage
from evorig_next.phase1_field import Phase1FieldState, Phase1GaussianField
from evorig_next.phase1_losses import (
    bone_cov_offdiag_loss,
    mesh_edge_length_floor_loss,
    bone_radial_distance_shrink_loss,
    bone_radial_symmetry_loss,
    bone_scale_band_loss,
    bone_scale_consistency_loss,
    gaussian_illegal_coverage_loss,
    gaussian_log_scale_anchor_loss,
    illegal_support_loss,
    joint_inside_mesh_loss,
    pose_consistent_joint_shell_loss,
    posed_bone_inside_mesh_loss,
    posed_joint_inside_mesh_loss,
    posed_joint_shell_descriptors,
    posed_joint_surface_clearance_loss,
    rest_joint_surface_clearance_loss,
    temporal_smoothness_loss,
    vertex_acceleration_loss,
    vertex_recon_loss,
)
from evorig_next.phase1_skeleton import Phase1Skeleton


_PHASE1_GRAPH_CACHE_ROOT = Path(__file__).resolve().parents[2] / "mygs" / "outputs" / "phase1_graph_cache"
_PHASE1_ADJACENCY_CACHE_CPU: dict[str, list[tuple[int, ...]]] = {}
_PHASE1_RING_CACHE_CPU: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


@dataclass
class Phase1EvalCache:
    weights: torch.Tensor
    kernels: torch.Tensor
    support: torch.Tensor
    legal_support_mass: torch.Tensor
    zero_weight_mask: torch.Tensor
    global_transforms: torch.Tensor
    pred_vertices: torch.Tensor


def build_rest_mesh_adjacency(
    faces: torch.Tensor,
    vertex_count: int,
) -> list[tuple[int, ...]]:
    neighbors: list[set[int]] = [set() for _ in range(int(vertex_count))]
    if faces.numel() <= 0:
        return [tuple() for _ in range(int(vertex_count))]
    faces_cpu = faces.detach().to(device="cpu", dtype=torch.long)
    for tri in faces_cpu.tolist():
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [tuple(sorted(item)) for item in neighbors]


def adjacency_to_edge_index(
    adjacency: list[tuple[int, ...]],
    *,
    device: torch.device,
    degree_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_list: list[int] = []
    dst_list: list[int] = []
    degree = torch.zeros(len(adjacency), dtype=degree_dtype, device=device)
    for vertex_idx, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        degree[vertex_idx] = float(len(neighbors))
        src_list.extend([int(vertex_idx)] * len(neighbors))
        dst_list.extend(int(item) for item in neighbors)
    if not src_list:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, degree
    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    return src, dst, degree


def build_k_ring_edge_index(
    adjacency: list[tuple[int, ...]],
    radius: int,
    *,
    device: torch.device,
    degree_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if radius <= 1:
        return adjacency_to_edge_index(adjacency, device=device, degree_dtype=degree_dtype)
    src_list: list[int] = []
    dst_list: list[int] = []
    degree = torch.zeros(len(adjacency), dtype=degree_dtype, device=device)
    for root_idx, root_neighbors in enumerate(adjacency):
        if not root_neighbors:
            continue
        seen = {int(root_idx)}
        frontier = [int(root_idx)]
        collected: list[int] = []
        for _ in range(int(radius)):
            next_frontier: list[int] = []
            for current in frontier:
                for neighbor in adjacency[current]:
                    neighbor = int(neighbor)
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
                    collected.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        if not collected:
            continue
        degree[root_idx] = float(len(collected))
        src_list.extend([int(root_idx)] * len(collected))
        dst_list.extend(collected)
    if not src_list:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, degree
    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    return src, dst, degree


def topology_cache_key(
    faces: torch.Tensor,
    vertex_count: int,
) -> str:
    faces_cpu = faces.detach().to(device="cpu", dtype=torch.int32).contiguous()
    hasher = hashlib.sha1()
    hasher.update(int(vertex_count).to_bytes(8, byteorder="little", signed=False))
    hasher.update(int(faces_cpu.shape[0]).to_bytes(8, byteorder="little", signed=False))
    hasher.update(faces_cpu.numpy().tobytes())
    return hasher.hexdigest()


def _phase1_ring_cache_path(cache_key: str, radius: int) -> Path:
    return _PHASE1_GRAPH_CACHE_ROOT / f"{cache_key}_ring{int(radius)}.pt"


def _phase1_adjacency_cache_path(cache_key: str) -> Path:
    return _PHASE1_GRAPH_CACHE_ROOT / f"{cache_key}_adjacency.pt"


def load_cached_adjacency_cpu(cache_key: str) -> list[tuple[int, ...]] | None:
    cache_key = str(cache_key)
    cached = _PHASE1_ADJACENCY_CACHE_CPU.get(cache_key)
    if cached is not None:
        return cached
    path = _phase1_adjacency_cache_path(cache_key)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    rows = payload.get("adjacency", payload)
    adjacency = [tuple(int(item) for item in row) for row in rows]
    _PHASE1_ADJACENCY_CACHE_CPU[cache_key] = adjacency
    return adjacency


def save_cached_adjacency_cpu(cache_key: str, adjacency: list[tuple[int, ...]]) -> None:
    cache_key = str(cache_key)
    normalized = [tuple(int(item) for item in row) for row in adjacency]
    _PHASE1_ADJACENCY_CACHE_CPU[cache_key] = normalized
    path = _phase1_adjacency_cache_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"adjacency": [list(row) for row in normalized]}, tmp_path)
    tmp_path.replace(path)


def load_cached_ring_edge_index_cpu(
    cache_key: str,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    cache_token = (str(cache_key), int(radius))
    cached = _PHASE1_RING_CACHE_CPU.get(cache_token)
    if cached is not None:
        return cached
    path = _phase1_ring_cache_path(cache_key, radius)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    cached = (
        torch.as_tensor(payload["src"], dtype=torch.long).contiguous(),
        torch.as_tensor(payload["dst"], dtype=torch.long).contiguous(),
        torch.as_tensor(payload["degree"], dtype=torch.float32).contiguous(),
    )
    _PHASE1_RING_CACHE_CPU[cache_token] = cached
    return cached


def save_cached_ring_edge_index_cpu(
    cache_key: str,
    radius: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
) -> None:
    cache_token = (str(cache_key), int(radius))
    cached = (
        src.detach().to(device="cpu", dtype=torch.long).contiguous(),
        dst.detach().to(device="cpu", dtype=torch.long).contiguous(),
        degree.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    )
    _PHASE1_RING_CACHE_CPU[cache_token] = cached
    path = _phase1_ring_cache_path(cache_key, radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "src": cached[0],
            "dst": cached[1],
            "degree": cached[2],
        },
        tmp_path,
    )
    tmp_path.replace(path)


def propagate_joint_legality_majority(
    mask: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    *,
    threshold: float,
    target_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [V, J]")
    if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
        raise ValueError("src and dst must share shape [E]")
    if degree.ndim != 1 or degree.shape[0] != mask.shape[0]:
        raise ValueError("degree must have shape [V]")
    if target_rows.ndim != 1 or target_rows.shape[0] != mask.shape[0]:
        raise ValueError("target_rows must have shape [V]")
    if src.numel() == 0:
        return mask, {
            "target_vertex_count": int(target_rows.sum().item()),
            "changed_vertex_count": 0,
            "added_joint_pair_count": 0,
        }
    mask_float = mask.to(dtype=degree.dtype, device=degree.device)
    votes = torch.zeros_like(mask_float)
    votes.index_add_(0, src, mask_float[dst])
    neighbor_fraction = votes / degree.unsqueeze(-1).clamp_min(1.0)
    reinforced = target_rows.unsqueeze(-1) & (neighbor_fraction >= float(threshold))
    updated = mask | reinforced
    changed = updated & (~mask)
    return updated, {
        "target_vertex_count": int(target_rows.sum().item()),
        "changed_vertex_count": int(changed.any(dim=-1).sum().item()),
        "added_joint_pair_count": int(changed.sum().item()),
    }


def dominant_joint_assignment(
    weights: torch.Tensor,
    legal_support_mass: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("weights must have shape [V, J]")
    if legal_support_mass.ndim != 1 or legal_support_mass.shape[0] != weights.shape[0]:
        raise ValueError("legal_support_mass must have shape [V]")
    dominant = torch.full((weights.shape[0],), -1, dtype=torch.long, device=weights.device)
    active = legal_support_mass > float(eps)
    if bool(active.any().item()):
        dominant[active] = weights[active].argmax(dim=-1)
    return dominant


def audit_dominant_connectivity(
    faces: torch.Tensor,
    vertex_count: int,
    joint_count: int,
    dominant_joint: torch.Tensor,
) -> dict[str, Any]:
    if dominant_joint.ndim != 1 or int(dominant_joint.shape[0]) != int(vertex_count):
        raise ValueError("dominant_joint must have shape [V]")
    faces_cpu = faces.detach().to(device="cpu", dtype=torch.long)
    dominant_cpu = dominant_joint.detach().to(device="cpu", dtype=torch.long)
    neighbors: list[set[int]] = [set() for _ in range(int(vertex_count))]
    for tri in faces_cpu.tolist():
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    component_counts: dict[str, int] = {}
    vertex_counts: dict[str, int] = {}
    disconnected_joint_ids: list[int] = []
    for joint_id in range(int(joint_count)):
        vertices = torch.nonzero(dominant_cpu == joint_id, as_tuple=False).flatten().tolist()
        vertex_counts[str(joint_id)] = int(len(vertices))
        if not vertices:
            component_counts[str(joint_id)] = 0
            continue
        allowed = set(int(item) for item in vertices)
        visited: set[int] = set()
        components = 0
        for start in vertices:
            start = int(start)
            if start in visited:
                continue
            components += 1
            stack = [start]
            visited.add(start)
            while stack:
                current = stack.pop()
                for nxt in neighbors[current]:
                    if nxt not in allowed or nxt in visited:
                        continue
                    visited.add(nxt)
                    stack.append(nxt)
        component_counts[str(joint_id)] = int(components)
        if components > 1:
            disconnected_joint_ids.append(int(joint_id))
    no_joint_mask = dominant_cpu < 0
    no_joint_vertex_count = int(no_joint_mask.sum().item())
    no_joint_vertex_fraction = float(no_joint_vertex_count / max(int(vertex_count), 1))
    return {
        "dominant_joint_component_counts": component_counts,
        "dominant_joint_vertex_counts": vertex_counts,
        "disconnected_joint_ids": disconnected_joint_ids,
        "no_joint_vertex_count": no_joint_vertex_count,
        "no_joint_vertex_fraction": no_joint_vertex_fraction,
    }


def summarize_joint_rotation_deltas(
    joint_rotations: torch.Tensor,
    watch_joint_ids: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    if joint_rotations.ndim != 4:
        raise ValueError("joint_rotations must have shape [T, J, 3, 3]")
    if joint_rotations.shape[0] <= 0:
        return {}
    base = joint_rotations[0]
    summary: dict[str, dict[str, float]] = {}
    for joint_id in watch_joint_ids:
        if joint_id < 0 or joint_id >= int(joint_rotations.shape[1]):
            continue
        relative = torch.matmul(base[joint_id].transpose(-1, -2), joint_rotations[:, joint_id])
        trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        delta = torch.arccos(cos_theta)
        summary[str(int(joint_id))] = {
            "rot_delta_max": float(delta.max().item()),
            "rot_delta_mean": float(delta.mean().item()),
        }
    return summary


def summarize_structure_acceptance(
    *,
    disconnected_joint_ids: list[int],
    no_joint_rest_displacement_max: float,
    no_joint_rest_displacement_tol: float,
) -> dict[str, Any]:
    strict_single_connectivity_pass = len(disconnected_joint_ids) == 0
    no_joint_static_pass = float(no_joint_rest_displacement_max) <= float(no_joint_rest_displacement_tol)
    return {
        "strict_single_connectivity_pass": bool(strict_single_connectivity_pass),
        "no_joint_static_pass": bool(no_joint_static_pass),
        "passes_structure_acceptance": bool(strict_single_connectivity_pass and no_joint_static_pass),
        "failure_reasons": [
            *([] if strict_single_connectivity_pass else ["dominant_joint_disconnected"]),
            *([] if no_joint_static_pass else ["no_joint_vertices_moved"]),
        ],
    }


class Phase1Trainer:
    def __init__(
        self,
        sample: dict[str, Any],
        *,
        base_config: dict[str, Any],
        phase1_config: Phase1Config,
        device: torch.device,
    ) -> None:
        self.sample = sample
        self.base_config = base_config
        self.cfg = phase1_config
        self.device = device
        sample_meta = sample.get("sample_meta") if isinstance(sample, dict) else None
        self.asset_name = str(sample_meta.get("asset", "")) if isinstance(sample_meta, dict) else ""
        self.rest_vertices = sample["rest_vertices"].to(device)
        self.mesh_faces = sample["faces"].to(device)
        self.gt_vertices = sample["gt_vertices"].to(device)
        self.sample_radius = max(float(mesh_radius(self.rest_vertices).item()), 1.0e-8)
        self.skeleton = Phase1Skeleton(
            parent_idx=sample["parent_idx"].to(device),
            rest_joints=sample["rest_joints"].to(device),
            frame_count=int(self.gt_vertices.shape[0]),
            init_pose=None if sample.get("init_pose") is None else sample["init_pose"].to(device),
            birth_steps=sample.get("birth_steps"),
            inserted=sample.get("inserted"),
            birth_modes=sample.get("birth_modes"),
            connected_to_parent=None if sample.get("connected_to_parent") is None else sample["connected_to_parent"].to(device),
        ).to(device)
        self._initialize_root_trans_from_mesh_motion()
        self.field = Phase1GaussianField.initialize_from_legacy(
            self.rest_vertices,
            self.skeleton,
            self._field_init_config(),
            lambda_min=self.cfg.lambda_min,
            lambda_max=self.cfg.lambda_max,
            seed_count_scale=self.cfg.seed_count_scale,
            kernel_mahal_cutoff_sq=float(self.cfg.gaussian_kernel_mahal_cutoff_sq),
            faces=self.mesh_faces,
        ).to(device)
        self.field.reset_endpoint_logits_from_lambda_sigmoid(
            midpoint=float(self.cfg.ownership_midpoint),
            slope=float(self.cfg.ownership_slope),
        )
        self._sh_initialized = False
        self.bind_transforms = self.skeleton.compute_bind_transforms()
        self.rest_mesh_scene = build_mesh_query_scene(self.rest_vertices, self.mesh_faces)
        self.gt_mesh_scene_cache: dict[int, MeshQueryScene | None] = {}
        self.legality_diagnostics: dict[str, Any] = {}
        self.legal_joint_mask = self._compute_vertex_joint_legality_mask()
        self.legal_vertex_mask = self.legal_joint_mask.any(dim=-1)
        self.bone_radial_distance_cache = self._build_bone_radial_distance_cache()
        self.optimizer = self._build_optimizer()
        self._init_lambda_optimizer_state()
        self.gaussian_grad_ema = torch.zeros(self.field.gaussian_count, dtype=self.rest_vertices.dtype, device=device)
        self.current_step = 0

    def _initialize_root_trans_from_mesh_motion(self) -> None:
        mode = str(getattr(self.cfg, "root_trans_init_mode", "none")).strip().lower()
        if mode in {"", "none", "zero", "zeros"}:
            return
        if int(self.gt_vertices.shape[0]) != int(self.skeleton.root_trans.shape[0]):
            return
        if mode in {"centroid", "mean"}:
            target = self.gt_vertices.mean(dim=1) - self.rest_vertices.mean(dim=0).unsqueeze(0)
        elif mode == "median":
            target = (self.gt_vertices - self.rest_vertices.unsqueeze(0)).median(dim=1).values
        elif mode in {"bbox", "bbox_center"}:
            rest_center = 0.5 * (self.rest_vertices.min(dim=0).values + self.rest_vertices.max(dim=0).values)
            frame_center = 0.5 * (self.gt_vertices.min(dim=1).values + self.gt_vertices.max(dim=1).values)
            target = frame_center - rest_center.unsqueeze(0)
        else:
            raise ValueError(f"unsupported root_trans_init_mode '{mode}'")
        target = target - target[:1]
        with torch.no_grad():
            self.skeleton.root_trans.copy_(target.to(dtype=self.skeleton.root_trans.dtype, device=self.device))

    def _rest_joint_effective_train_mask(self) -> torch.Tensor:
        joint_count = int(getattr(self.skeleton, "joint_count", self.skeleton.rest_joints.shape[0]))
        device = getattr(self, "device", self.skeleton.rest_joints.device)
        mask = torch.ones(joint_count, dtype=torch.bool, device=device)
        saved = getattr(self, "rest_joint_train_mask", None)
        if isinstance(saved, torch.Tensor) and int(saved.numel()) == int(mask.numel()):
            mask &= saved.to(device=device, dtype=torch.bool).reshape(-1)
        if bool(getattr(self.cfg, "freeze_root_rest_joint", False)):
            parent_idx = getattr(self.skeleton, "parent_idx", None)
            if isinstance(parent_idx, torch.Tensor) and int(parent_idx.numel()) == int(mask.numel()):
                root_mask = (parent_idx < 0).to(device=device, dtype=torch.bool)
                mask &= ~root_mask
        frozen_ids = getattr(self.cfg, "frozen_rest_joint_ids", ())
        if frozen_ids:
            for joint_id in frozen_ids:
                index = int(joint_id)
                if 0 <= index < int(mask.numel()):
                    mask[index] = False
        return mask

    def _field_init_config(self) -> dict[str, Any]:
        config = dict(self.base_config)
        config["phase1"] = {
            "initial_seed_prune_outside_mesh": bool(self.cfg.initial_seed_prune_outside_mesh),
            "force_global_lambda_bounds": bool(self.cfg.phase1_force_global_lambda_bounds),
            "global_lambda_min": float(self.cfg.lambda_min),
            "global_lambda_max": float(self.cfg.lambda_max),
            "decouple_axial_from_radial_init": bool(self.cfg.phase1_decouple_axial_from_radial_init),
            "scale_init_divisor": float(self.cfg.phase1_scale_init_divisor),
            "scale_formula": str(self.cfg.phase1_scale_formula),
            "radial_sigma_divisor": float(self.cfg.phase1_radial_sigma_divisor),
            "radial_extent_quantile": float(self.cfg.phase1_radial_extent_quantile),
            "radial_patch_connected_component": bool(self.cfg.phase1_radial_patch_connected_component),
            "cross_section_angle_bins": int(self.cfg.phase1_cross_section_angle_bins),
            "cross_section_min_bins": int(self.cfg.phase1_cross_section_min_bins),
            "axial_three_center_divisor": float(self.cfg.phase1_axial_three_center_divisor),
            "axial_min_radial_ratio": float(self.cfg.phase1_axial_min_radial_ratio),
            "formula_log_scale_min": float(self.cfg.phase1_formula_log_scale_min),
            "formula_log_scale_max": float(self.cfg.phase1_formula_log_scale_max),
        }
        config["phase1_initial_seed_prune_outside_mesh"] = bool(self.cfg.initial_seed_prune_outside_mesh)
        config["phase1_seed_inside_surface_tol"] = float(self.cfg.seed_inside_surface_tol)
        return config

    def _build_optimizer(self) -> Adam:
        param_groups = [
            {"params": [self.skeleton.rest_joints], "lr": float(self.cfg.lr_rest_joints)},
            {"params": [self.skeleton.pose_rot], "lr": float(self.cfg.lr_pose)},
            {"params": [self.skeleton.root_trans], "lr": float(self.cfg.lr_root)},
        ]
        param_groups.extend(
            [
                {"params": [self.field.rot_local], "lr": float(self.cfg.lr_rot)},
                {"params": [self.field.log_scale], "lr": float(self.cfg.lr_scale)},
                {"params": [self.field.log_opacity], "lr": float(self.cfg.lr_opacity)},
                {"params": [self.field.log_value], "lr": float(self.cfg.lr_value)},
                {"params": [self.field.sh_coeffs], "lr": float(self.cfg.lr_sh)},
            ]
        )
        if float(getattr(self.cfg, "lr_offset", 0.0)) > 0.0:
            param_groups.append({"params": [self.field.offset_local], "lr": float(self.cfg.lr_offset)})
        if not self._use_manual_lambda_optimizer():
            param_groups.append({"params": [self.field.lambda_param], "lr": float(self.cfg.lr_lambda_initial)})
        return Adam(param_groups)

    def _use_manual_lambda_optimizer(self) -> bool:
        return (
            str(getattr(self.cfg, "initial_lambda_policy", "learn_all")).lower() != "learn_all"
            or int(getattr(self.cfg, "lambda_thaw_start_step", -1)) >= 0
        )

    def _init_lambda_optimizer_state(self) -> None:
        self.lambda_adam_m = torch.zeros_like(self.field.lambda_param.detach())
        self.lambda_adam_v = torch.zeros_like(self.field.lambda_param.detach())
        self.lambda_adam_step = torch.zeros_like(self.field.lambda_param.detach(), dtype=torch.long)

    @staticmethod
    def _scheduled_param_active(start_step: int, step: int) -> bool:
        return int(start_step) >= 0 and int(step) >= int(start_step)

    @staticmethod
    def _scheduled_loss_active(weight: float, start_step: int, step: int) -> bool:
        return float(weight) > 0.0 and (int(start_step) < 0 or int(step) >= int(start_step))

    def _rest_joint_active(self, step: int | None = None) -> bool:
        current = int(getattr(self, "current_step", 0)) if step is None else int(step)
        return self._scheduled_param_active(int(self.cfg.rest_joint_start_step), current)

    def _sh_active(self, step: int | None = None) -> bool:
        current = int(getattr(self, "current_step", 0)) if step is None else int(step)
        return self._scheduled_param_active(int(self.cfg.sh_start_step), current)

    def _gaussian_offset_active(self, step: int | None = None) -> bool:
        current = int(getattr(self, "current_step", 0)) if step is None else int(step)
        return self._scheduled_param_active(int(getattr(self.cfg, "gaussian_offset_start_step", -1)), current)

    def _gaussian_offset_train_mask(self) -> torch.Tensor:
        target = str(getattr(self.cfg, "gaussian_offset_target", "all")).lower().strip()
        generation = getattr(self.field, "generation", None)
        if not isinstance(generation, torch.Tensor):
            return torch.ones(int(self.field.offset_local.shape[0]), dtype=torch.bool, device=self.field.offset_local.device)
        generation = generation.to(device=self.field.offset_local.device)
        if target in {"all", "*"}:
            return torch.ones(int(self.field.gaussian_count), dtype=torch.bool, device=self.field.offset_local.device)
        if target in {"initial", "initial_only"}:
            return generation <= 0
        if target in {"densified", "densified_only"}:
            return generation > 0
        if target in {"inserted", "inserted_bones", "new_bones"}:
            inserted = torch.tensor(
                [bool(item) for item in getattr(self.skeleton, "is_inserted", [])],
                dtype=torch.bool,
                device=self.field.offset_local.device,
            )
            if int(inserted.numel()) != int(self.skeleton.joint_count):
                return torch.zeros(int(self.field.gaussian_count), dtype=torch.bool, device=self.field.offset_local.device)
            child_joints = self.skeleton.bone_child_idx[self.field.anchor_bone].to(device=inserted.device)
            parent_joints = self.skeleton.bone_parent_idx[self.field.anchor_bone].to(device=inserted.device)
            return inserted[child_joints] | inserted[parent_joints]
        raise ValueError(f"unsupported gaussian_offset_target '{target}'")

    def _prepare_staged_params_for_step(self, step: int) -> None:
        if self._sh_active(step) and not self._sh_initialized:
            changed = self.field.ensure_sh_coeffs(int(self.cfg.sh_coeff_count), preserve_unit_density=True)
            self.field.use_sh_response = True
            self._sh_initialized = True
            if changed:
                self.optimizer = self._build_optimizer()
        else:
            self.field.use_sh_response = bool(self._sh_active(step))

    def _apply_staged_gradient_freezes(self, step: int) -> None:
        if self.skeleton.rest_joints.grad is not None:
            if not self._rest_joint_active(step):
                self.skeleton.rest_joints.grad.zero_()
            else:
                mask = self._rest_joint_effective_train_mask().to(device=self.skeleton.rest_joints.grad.device)
                inactive = ~mask
                if bool(inactive.any().item()):
                    self.skeleton.rest_joints.grad[inactive] = 0
        if (
            not self._sh_active(step)
            and hasattr(self.field, "sh_coeffs")
            and self.field.sh_coeffs.grad is not None
        ):
            self.field.sh_coeffs.grad.zero_()
        if self.field.offset_local.grad is not None:
            if not self._gaussian_offset_active(step):
                self.field.offset_local.grad.zero_()
            else:
                active = self._gaussian_offset_train_mask().to(device=self.field.offset_local.grad.device)
                if int(active.numel()) == int(self.field.offset_local.grad.shape[0]):
                    inactive = ~active
                    if bool(inactive.any().item()):
                        self.field.offset_local.grad[inactive] = 0
        endpoint_logits = getattr(self.field, "endpoint_logits", None)
        parent_child_mix_start = int(getattr(self.cfg, "parent_child_mix_start_step", -1))
        if (
            parent_child_mix_start >= 0
            and int(step) < parent_child_mix_start
            and endpoint_logits is not None
            and getattr(endpoint_logits, "grad", None) is not None
        ):
            endpoint_logits.grad.zero_()

    def _project_rest_joints_inside_after_step(self, step: int) -> None:
        if not bool(getattr(self.cfg, "project_rest_joints_inside_after_step", False)):
            return
        if not self._rest_joint_active(step):
            return
        if self.mesh_faces is None or int(self.mesh_faces.numel()) <= 0:
            return
        padding = float(getattr(self.cfg, "rest_joint_projection_padding", self.cfg.pcjs_surface_tol))
        padding = max(padding, float(self.cfg.pcjs_surface_tol))
        with torch.no_grad():
            train_mask = self._rest_joint_effective_train_mask()
            if not bool(train_mask.any().item()):
                return
            projected, _inside_mask, _outside_distance = project_points_inside_mesh(
                self.skeleton.rest_joints.detach(),
                self.rest_vertices,
                self.mesh_faces,
                padding=padding,
                mesh_query_scene=self.rest_mesh_scene,
            )
            updated = self.skeleton.rest_joints.detach().clone()
            updated[train_mask] = projected[train_mask]
            self.skeleton.rest_joints.copy_(updated)

    def _restore_inactive_rest_joints_after_step(self) -> None:
        train_mask = self._rest_joint_effective_train_mask()
        inactive = ~train_mask
        if not bool(inactive.any().item()):
            return
        with torch.no_grad():
            current = self.skeleton.rest_joints.detach().clone()
            current[inactive] = self.skeleton.init_rest_joints.detach()[inactive].to(
                device=current.device,
                dtype=current.dtype,
            )
            self.skeleton.rest_joints.copy_(current)

    def _first_densify_event_step(self) -> int:
        if not self.cfg.densify_stages:
            return int(self.cfg.steps) + 1
        return max(int(self.cfg.densify_stages[0].warm_steps), 0)

    def _lambda_learning_rates(self) -> torch.Tensor:
        policy = str(getattr(self.cfg, "initial_lambda_policy", "learn_all")).lower()
        generation = self.field.generation.to(device=self.field.lambda_param.device)
        initial_mask = generation <= 0
        densified_mask = generation > 0
        lr = torch.zeros_like(self.field.lambda_param.detach())
        lambda_thaw_start = int(getattr(self.cfg, "lambda_thaw_start_step", -1))
        if lambda_thaw_start >= 0:
            if int(self.current_step) >= lambda_thaw_start:
                target = str(getattr(self.cfg, "lambda_thaw_target", "all")).lower().strip()
                if target in {"all", "initial", "initial_only"}:
                    lr[initial_mask] = float(self.cfg.lr_lambda_initial_thawed)
                if target in {"all", "densified", "densified_only"}:
                    lr[densified_mask] = float(self.cfg.lr_lambda_densified)
                if target not in {"all", "initial", "initial_only", "densified", "densified_only"}:
                    raise ValueError(f"unsupported lambda_thaw_target '{target}'")
            return lr
        if policy == "learn_all":
            lr[initial_mask] = float(self.cfg.lr_lambda_initial)
        elif policy == "freeze_forever":
            pass
        elif policy == "thaw_after_stage0":
            if int(self.current_step) > self._first_densify_event_step():
                lr[initial_mask] = float(self.cfg.lr_lambda_initial_thawed)
        else:
            raise ValueError(f"unsupported initial_lambda_policy '{policy}'")
        lr[densified_mask] = float(self.cfg.lr_lambda_densified)
        return lr

    def _step_lambda_param(self) -> None:
        if not self._use_manual_lambda_optimizer():
            return
        grad = self.field.lambda_param.grad
        if grad is None:
            return
        lr = self._lambda_learning_rates()
        active = lr > 0.0
        if not bool(active.any().item()):
            return
        beta1 = 0.9
        beta2 = 0.999
        eps = float(EPS)
        with torch.no_grad():
            self.lambda_adam_m[active] = beta1 * self.lambda_adam_m[active] + (1.0 - beta1) * grad[active]
            self.lambda_adam_v[active] = beta2 * self.lambda_adam_v[active] + (1.0 - beta2) * grad[active].square()
            self.lambda_adam_step[active] = self.lambda_adam_step[active] + 1
            steps = self.lambda_adam_step[active].to(dtype=grad.dtype)
            beta1_t = torch.full_like(steps, beta1)
            beta2_t = torch.full_like(steps, beta2)
            bias_c1 = 1.0 - torch.pow(beta1_t, steps)
            bias_c2 = 1.0 - torch.pow(beta2_t, steps)
            m_hat = self.lambda_adam_m[active] / bias_c1.clamp_min(EPS)
            v_hat = self.lambda_adam_v[active] / bias_c2.clamp_min(EPS)
            self.field.lambda_param.data[active] = self.field.lambda_param.data[active] - lr[active] * m_hat / (v_hat.sqrt() + eps)

    def _sample_frames(self) -> torch.Tensor:
        frame_count = int(self.gt_vertices.shape[0])
        batch = min(int(self.cfg.frame_batch_size), frame_count)
        if batch >= frame_count:
            return torch.arange(frame_count, dtype=torch.long, device=self.device)
        return torch.randperm(frame_count, device=self.device)[:batch]

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def refresh_after_topology_mutation(self, *, preserve_fallback_weights: bool = False) -> None:
        old_fallback_weights = None
        if bool(preserve_fallback_weights) and hasattr(self, "_bone_endpoint_fallback_weights"):
            old_fallback_weights = self._bone_endpoint_fallback_weights.detach().clone()
        self.bind_transforms = self.skeleton.compute_bind_transforms()
        self.legal_joint_mask = self._compute_vertex_joint_legality_mask()
        self.legal_vertex_mask = self.legal_joint_mask.any(dim=-1)
        self.bone_radial_distance_cache = self._build_bone_radial_distance_cache()
        if hasattr(self, "_joint_cross_section_sections"):
            delattr(self, "_joint_cross_section_sections")
        if hasattr(self, "_bone_endpoint_fallback_weights"):
            delattr(self, "_bone_endpoint_fallback_weights")
        if old_fallback_weights is not None:
            fallback = old_fallback_weights.to(device=self.device, dtype=self.rest_vertices.dtype)
            joint_delta = int(self.skeleton.joint_count) - int(fallback.shape[1])
            if joint_delta > 0:
                fallback = torch.cat(
                    [
                        fallback,
                        torch.zeros(
                            int(fallback.shape[0]),
                            joint_delta,
                            dtype=fallback.dtype,
                            device=fallback.device,
                        ),
                    ],
                    dim=1,
                )
            elif joint_delta < 0:
                fallback = fallback[:, : int(self.skeleton.joint_count)]
            if int(fallback.shape[0]) == int(self.rest_vertices.shape[0]):
                self._bone_endpoint_fallback_weights = fallback
        if int(self.gaussian_grad_ema.shape[0]) != int(self.field.gaussian_count):
            old_count = int(self.gaussian_grad_ema.shape[0])
            new_count = int(self.field.gaussian_count)
            if new_count > old_count:
                self.gaussian_grad_ema = torch.cat(
                    [
                        self.gaussian_grad_ema,
                        torch.zeros(
                            new_count - old_count,
                            dtype=self.gaussian_grad_ema.dtype,
                            device=self.gaussian_grad_ema.device,
                        ),
                    ],
                    dim=0,
                )
            else:
                self.gaussian_grad_ema = self.gaussian_grad_ema[:new_count].detach().clone()
        self.optimizer = self._build_optimizer()
        if self._use_manual_lambda_optimizer():
            self._init_lambda_optimizer_state()

    def _get_rest_mesh_adjacency(self) -> list[tuple[int, ...]]:
        cached = getattr(self, "_rest_mesh_adjacency", None)
        if cached is not None:
            return cached
        cache_key = self._get_rest_mesh_topology_cache_key()
        cached_cpu = load_cached_adjacency_cpu(cache_key)
        if cached_cpu is not None:
            self._rest_mesh_adjacency = cached_cpu
            return cached_cpu
        adjacency = build_rest_mesh_adjacency(self.mesh_faces, int(self.rest_vertices.shape[0]))
        save_cached_adjacency_cpu(cache_key, adjacency)
        self._rest_mesh_adjacency = adjacency
        return adjacency

    def _get_rest_mesh_edge_index(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst, degree, _ = self._get_rest_mesh_k_ring_edge_index(radius=1)
        return src, dst, degree

    def _build_bone_radial_distance_cache(self) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            assignment = LegacyGaussianSupportField._assign_vertices_to_bones(self.rest_vertices.detach(), self.skeleton)
        bone_index = assignment["bone_index"].detach().to(device=self.device, dtype=torch.long)
        rest_radial = assignment["radial_distance"].detach().to(device=self.device, dtype=self.rest_vertices.dtype)
        safe_bone = bone_index.clamp_min(0)
        parent = self.skeleton.bone_parent_idx[safe_bone].to(device=self.device, dtype=torch.long)
        child = self.skeleton.bone_child_idx[safe_bone].to(device=self.device, dtype=torch.long)
        invalid = bone_index < 0
        if bool(invalid.any().item()):
            parent[invalid] = 0
            child[invalid] = 0
        return {
            "bone_index": bone_index,
            "parent_joint": parent,
            "child_joint": child,
            "lambda_value": assignment["lambda_value"].detach().to(device=self.device, dtype=self.rest_vertices.dtype),
            "rest_radial_distance": rest_radial,
        }

    def _get_bone_endpoint_fallback_weights(self) -> torch.Tensor:
        cached = getattr(self, "_bone_endpoint_fallback_weights", None)
        if cached is not None:
            return cached
        vertex_count = int(self.rest_vertices.shape[0])
        joint_count = int(self.skeleton.joint_count)
        fallback = torch.zeros(
            (vertex_count, joint_count),
            dtype=self.rest_vertices.dtype,
            device=self.device,
        )
        if not hasattr(self.skeleton, "bone_parent_idx") or not hasattr(self.skeleton, "bone_child_idx"):
            cache = getattr(self, "bone_radial_distance_cache", {})
            parent = cache.get("parent_joint")
            child = cache.get("child_joint")
            lam = cache.get("lambda_value")
            if isinstance(parent, torch.Tensor) and isinstance(child, torch.Tensor) and isinstance(lam, torch.Tensor):
                row_count = min(vertex_count, int(parent.numel()), int(child.numel()), int(lam.numel()))
                if row_count > 0:
                    rows = torch.arange(row_count, dtype=torch.long, device=self.device)
                    parent = parent[:row_count].to(device=self.device, dtype=torch.long)
                    child = child[:row_count].to(device=self.device, dtype=torch.long)
                    lam = lam[:row_count].to(device=self.device, dtype=self.rest_vertices.dtype).clamp(0.0, 1.0)
                    valid_parent = (parent >= 0) & (parent < joint_count)
                    valid_child = (child >= 0) & (child < joint_count)
                    if bool(valid_parent.any().item()):
                        fallback[rows[valid_parent], parent[valid_parent]] = 1.0 - lam[valid_parent]
                    if bool(valid_child.any().item()):
                        fallback[rows[valid_child], child[valid_child]] = torch.where(
                            valid_parent[valid_child],
                            lam[valid_child],
                            torch.ones_like(lam[valid_child]),
                        )
            row_sum = fallback.sum(dim=-1, keepdim=True)
            empty = row_sum <= EPS
            if bool(empty.any().item()):
                fallback[empty.squeeze(-1), 0] = 1.0
                row_sum = fallback.sum(dim=-1, keepdim=True)
            fallback = fallback / row_sum.clamp_min(EPS)
            self._bone_endpoint_fallback_weights = fallback.detach()
            return self._bone_endpoint_fallback_weights
        parent_all = self.skeleton.bone_parent_idx.to(device=self.device, dtype=torch.long)
        child_all = self.skeleton.bone_child_idx.to(device=self.device, dtype=torch.long)
        joint_positions = self.skeleton.init_rest_joints.detach().to(device=self.device, dtype=self.rest_vertices.dtype)
        if int(joint_positions.shape[0]) != joint_count:
            joint_positions = self.skeleton.rest_joints.detach()
        bone_count = int(child_all.numel())
        if bone_count > 0:
            starts = joint_positions[parent_all]
            ends = joint_positions[child_all]
            segments = ends - starts
            seg_len_sq = segments.square().sum(dim=-1).clamp_min(EPS)
            best_dist_sq = torch.full((vertex_count,), float("inf"), dtype=self.rest_vertices.dtype, device=self.device)
            best_bone = torch.full((vertex_count,), -1, dtype=torch.long, device=self.device)
            best_lam = torch.zeros(vertex_count, dtype=self.rest_vertices.dtype, device=self.device)
            chunk_size = max(int(getattr(self.cfg, "legality_vertex_chunk_size", 4096)), 1)
            for start_id in range(0, vertex_count, chunk_size):
                end_id = min(start_id + chunk_size, vertex_count)
                points = self.rest_vertices[start_id:end_id]
                delta = points[:, None, :] - starts[None, :, :]
                lam_all = (delta * segments[None, :, :]).sum(dim=-1) / seg_len_sq[None, :]
                lam_all = lam_all.clamp(0.0, 1.0)
                closest = starts[None, :, :] + lam_all[..., None] * segments[None, :, :]
                dist_sq = (points[:, None, :] - closest).square().sum(dim=-1)
                local_dist, local_bone = dist_sq.min(dim=1)
                row_slice = slice(start_id, end_id)
                best_dist_sq[row_slice] = local_dist
                best_bone[row_slice] = local_bone
                best_lam[row_slice] = lam_all.gather(1, local_bone.unsqueeze(1)).squeeze(1)
            valid = best_bone >= 0
        else:
            valid = torch.zeros(vertex_count, dtype=torch.bool, device=self.device)
        if bool(valid.any().item()):
            row_ids = torch.nonzero(valid, as_tuple=False).flatten()
            parent = parent_all[best_bone[row_ids]]
            child = child_all[best_bone[row_ids]]
            lam = best_lam[row_ids].clamp(0.0, 1.0)
            valid_parent = parent >= 0
            if bool(valid_parent.any().item()):
                fallback[row_ids[valid_parent], parent[valid_parent]] = 1.0 - lam[valid_parent]
            fallback[row_ids, child] = torch.where(valid_parent, lam, torch.ones_like(lam))
        row_sum = fallback.sum(dim=-1, keepdim=True)
        empty = row_sum <= EPS
        if bool(empty.any().item()):
            fallback[empty.squeeze(-1), 0] = 1.0
            row_sum = fallback.sum(dim=-1, keepdim=True)
        fallback = fallback / row_sum.clamp_min(EPS)
        self._bone_endpoint_fallback_weights = fallback.detach()
        return self._bone_endpoint_fallback_weights

    def _get_rest_mesh_topology_cache_key(self) -> str:
        cached = getattr(self, "_rest_mesh_topology_cache_key", None)
        if cached is not None:
            return cached
        cache_key = topology_cache_key(self.mesh_faces, int(self.rest_vertices.shape[0]))
        self._rest_mesh_topology_cache_key = cache_key
        return cache_key

    def _get_rest_mesh_k_ring_edge_index(self, radius: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        radius = max(int(radius), 1)
        cache = getattr(self, "_rest_mesh_k_ring_edge_index_cache", None)
        if cache is None:
            cache = {}
            self._rest_mesh_k_ring_edge_index_cache = cache
        cached = cache.get(radius)
        if cached is not None:
            src, dst, degree = cached
            return src, dst, degree, 0.0
        cache_key = self._get_rest_mesh_topology_cache_key()
        cached_cpu = load_cached_ring_edge_index_cpu(cache_key, radius)
        if cached_cpu is not None:
            src = cached_cpu[0].to(device=self.device, dtype=torch.long)
            dst = cached_cpu[1].to(device=self.device, dtype=torch.long)
            degree = cached_cpu[2].to(device=self.device, dtype=self.rest_vertices.dtype)
            cache[radius] = (src, dst, degree)
            return src, dst, degree, 0.0
        self._sync_device()
        started = time.perf_counter()
        adjacency = self._get_rest_mesh_adjacency()
        if radius <= 1:
            src, dst, degree = adjacency_to_edge_index(
                adjacency,
                device=self.device,
                degree_dtype=self.rest_vertices.dtype,
            )
        else:
            src, dst, degree = build_k_ring_edge_index(
                adjacency,
                radius=radius,
                device=self.device,
                degree_dtype=self.rest_vertices.dtype,
            )
        build_ms = float((time.perf_counter() - started) * 1000.0)
        save_cached_ring_edge_index_cpu(cache_key, radius, src, dst, degree)
        cache[radius] = (src, dst, degree)
        return src, dst, degree, build_ms

    def _compute_vertex_joint_visibility_mask(self) -> torch.Tensor:
        vertex_count = int(self.rest_vertices.shape[0])
        joint_positions = self.skeleton.rest_joints.detach()
        joint_count = int(joint_positions.shape[0])
        mask = torch.ones((vertex_count, joint_count), dtype=torch.bool, device=self.device)
        origin_eps = float(max(self.cfg.legality_origin_epsilon, 1.0e-8))
        chunk_size = max(int(self.cfg.legality_vertex_chunk_size), 1)
        for start in range(0, vertex_count, chunk_size):
            end = min(start + chunk_size, vertex_count)
            chunk = self.rest_vertices[start:end]
            chunk_count = int(chunk.shape[0])
            origins = chunk[:, None, :].expand(-1, joint_count, -1)
            targets = joint_positions[None, :, :].expand(chunk_count, -1, -1)
            direction = targets - origins
            lengths = direction.norm(dim=-1)
            safe_lengths = lengths.clamp_min(EPS)
            ray_dir = direction / safe_lengths.unsqueeze(-1)
            open_eps = torch.maximum(
                torch.full_like(safe_lengths, origin_eps),
                safe_lengths * 1.0e-6,
            )
            same_point = lengths <= open_eps
            flat_origins = origins.reshape(-1, 3)
            flat_dirs = ray_dir.reshape(-1, 3)
            flat_lengths = safe_lengths.reshape(-1)
            flat_open_eps = open_eps.reshape(-1)
            blocked_flat = torch.zeros(flat_lengths.shape[0], dtype=torch.bool, device=self.device)
            if self.rest_mesh_scene is not None and hasattr(self.rest_mesh_scene, "ray_mesh_intersections"):
                intersections = self.rest_mesh_scene.ray_mesh_intersections(flat_origins, flat_dirs)
                ray_ids = intersections["ray_ids"].to(device=self.device, dtype=torch.long)
                if int(ray_ids.numel()) > 0:
                    t_hit = intersections["t_hit"].to(device=self.device, dtype=flat_lengths.dtype)
                    primitive_ids = intersections["primitive_ids"].to(device=self.device, dtype=torch.long)
                    source_vertex_ids = torch.div(ray_ids, joint_count, rounding_mode="floor") + int(start)
                    hit_faces = self.mesh_faces[primitive_ids]
                    incident_hit = (hit_faces == source_vertex_ids.unsqueeze(-1)).any(dim=-1)
                    inside_open_segment = (t_hit > flat_open_eps[ray_ids]) & (t_hit < (flat_lengths[ray_ids] - flat_open_eps[ray_ids]))
                    valid_blocker = (~incident_hit) & inside_open_segment
                    if bool(valid_blocker.any().item()):
                        blocked_flat[ray_ids[valid_blocker]] = True
            else:
                shifted_origins = flat_origins + origin_eps * flat_dirs
                t_hit, hit = ray_mesh_first_hit_distance(
                    shifted_origins,
                    flat_dirs,
                    self.rest_vertices,
                    self.mesh_faces,
                    mesh_query_scene=self.rest_mesh_scene,
                )
                remaining_dist = (flat_lengths - origin_eps).clamp_min(0.0)
                blocked_flat = hit & (t_hit > flat_open_eps) & (t_hit < (remaining_dist - flat_open_eps))
            visible = ~blocked_flat
            visible = visible.view(chunk_count, joint_count)
            visible = visible | same_point
            mask[start:end, :] = visible
        return mask

    def _select_legality_propagation_rows(self, mask: torch.Tensor) -> torch.Tensor:
        target = str(getattr(self.cfg, "legality_propagation_target", "all")).lower()
        row_count = mask.sum(dim=-1)
        if target == "all":
            return torch.ones(mask.shape[0], dtype=torch.bool, device=mask.device)
        if target == "empty":
            return row_count <= 0
        if target == "sparse":
            max_count = max(int(getattr(self.cfg, "legality_propagation_sparse_joint_max_count", 0)), 0)
            return row_count <= int(max_count)
        raise ValueError(f"unsupported legality_propagation_target '{target}'")

    def _apply_legality_propagation(self, raw_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        mode = str(getattr(self.cfg, "legality_propagation_mode", "off")).lower()
        diagnostics: dict[str, Any] = {
            "mode": mode,
            "radius": int(getattr(self.cfg, "legality_propagation_radius", 1)),
            "rounds": int(getattr(self.cfg, "legality_propagation_rounds", 1)),
            "target": str(getattr(self.cfg, "legality_propagation_target", "all")),
            "majority_threshold": float(getattr(self.cfg, "legality_propagation_majority_threshold", 0.5)),
            "raw_zero_legal_joint_row_count": int((raw_mask.sum(dim=-1) <= 0).sum().item()),
            "raw_legal_joint_mean_per_vertex": float(raw_mask.sum(dim=-1).float().mean().item()),
            "neighbor_index_build_ms": 0.0,
            "propagation_ms": 0.0,
            "round_summaries": [],
        }
        if mode == "off":
            diagnostics["final_zero_legal_joint_row_count"] = diagnostics["raw_zero_legal_joint_row_count"]
            diagnostics["final_legal_joint_mean_per_vertex"] = diagnostics["raw_legal_joint_mean_per_vertex"]
            diagnostics["changed_vertex_count"] = 0
            diagnostics["added_joint_pair_count"] = 0
            return raw_mask, diagnostics

        started = time.perf_counter()
        threshold = float(getattr(self.cfg, "legality_propagation_majority_threshold", 0.5))
        propagated = raw_mask.clone()
        total_changed_vertex_ids = torch.zeros(raw_mask.shape[0], dtype=torch.bool, device=raw_mask.device)
        total_added_joint_pairs = 0
        if mode == "k_ring_once":
            radius = max(int(getattr(self.cfg, "legality_propagation_radius", 1)), 1)
            src, dst, degree, build_ms = self._get_rest_mesh_k_ring_edge_index(radius)
            diagnostics["neighbor_index_build_ms"] += float(build_ms)
            target_rows = self._select_legality_propagation_rows(propagated)
            propagated, round_stats = propagate_joint_legality_majority(
                propagated,
                src,
                dst,
                degree,
                threshold=threshold,
                target_rows=target_rows,
            )
            total_added_joint_pairs += int(round_stats["added_joint_pair_count"])
            if round_stats["changed_vertex_count"] > 0:
                total_changed_vertex_ids |= propagated.ne(raw_mask).any(dim=-1)
            diagnostics["round_summaries"].append(
                {
                    "round_index": 0,
                    "radius": int(radius),
                    **round_stats,
                }
            )
        elif mode in {"one_ring_iterative", "k_ring_iterative"}:
            rounds = max(int(getattr(self.cfg, "legality_propagation_rounds", 1)), 1)
            radius = max(int(getattr(self.cfg, "legality_propagation_radius", 1)), 1)
            if radius <= 1:
                src, dst, degree = self._get_rest_mesh_edge_index()
            else:
                src, dst, degree, build_ms = self._get_rest_mesh_k_ring_edge_index(radius)
                diagnostics["neighbor_index_build_ms"] += float(build_ms)
            for round_idx in range(rounds):
                target_rows = self._select_legality_propagation_rows(propagated)
                updated, round_stats = propagate_joint_legality_majority(
                    propagated,
                    src,
                    dst,
                    degree,
                    threshold=threshold,
                    target_rows=target_rows,
                )
                round_changed = updated.ne(propagated).any(dim=-1)
                total_changed_vertex_ids |= round_changed
                total_added_joint_pairs += int(round_stats["added_joint_pair_count"])
                propagated = updated
                diagnostics["round_summaries"].append(
                    {
                        "round_index": int(round_idx),
                        "radius": int(radius),
                        **round_stats,
                    }
                )
        else:
            raise ValueError(f"unsupported legality_propagation_mode '{mode}'")
        self._sync_device()
        diagnostics["propagation_ms"] = float((time.perf_counter() - started) * 1000.0)
        diagnostics["final_zero_legal_joint_row_count"] = int((propagated.sum(dim=-1) <= 0).sum().item())
        diagnostics["final_legal_joint_mean_per_vertex"] = float(propagated.sum(dim=-1).float().mean().item())
        diagnostics["changed_vertex_count"] = int(total_changed_vertex_ids.sum().item())
        diagnostics["added_joint_pair_count"] = int(total_added_joint_pairs)
        return propagated, diagnostics

    def _compute_vertex_joint_legality_mask(self) -> torch.Tensor:
        raw_mask = self._compute_vertex_joint_visibility_mask()
        propagated, diagnostics = self._apply_legality_propagation(raw_mask)
        self.legality_diagnostics = diagnostics
        return propagated

    def _compute_gaussian_legal_vertex_mask(self) -> torch.Tensor:
        parent_joint = self.skeleton.bone_parent_idx[self.field.anchor_bone]
        child_joint = self.skeleton.bone_child_idx[self.field.anchor_bone]
        parent_legal = self.legal_joint_mask[:, parent_joint].transpose(0, 1)
        child_legal = self.legal_joint_mask[:, child_joint].transpose(0, 1)
        return parent_legal | child_legal

    def _compute_field_control_diagnostics(self, kernels: torch.Tensor) -> dict[str, torch.Tensor]:
        _, bone_frames, _, _ = self.skeleton.compute_bone_frames()
        anchor_frames = bone_frames[self.field.anchor_bone]
        covariance = self.field.compute_covariance(self.skeleton)
        bone_local_covariance = torch.matmul(anchor_frames.transpose(-1, -2), torch.matmul(covariance, anchor_frames))
        gaussian_legal_vertex_mask = self._compute_gaussian_legal_vertex_mask()
        return {
            "gaussian_illegal_coverage": gaussian_illegal_coverage_loss(
                kernels,
                gaussian_legal_vertex_mask,
                active_mask=self.field.active_mask,
                tau=0.0,
            ),
            "bone_cov_offdiag": bone_cov_offdiag_loss(
                bone_local_covariance,
                active_mask=self.field.active_mask,
            ),
            "bone_radial_symmetry": bone_radial_symmetry_loss(
                self.field.log_scale,
                active_mask=self.field.active_mask,
            ),
            "bone_scale_band": bone_scale_band_loss(
                self.field.anchor_bone,
                self.field.log_scale,
                active_mask=self.field.active_mask,
                max_axial_log_span=float(self.cfg.bone_scale_band_max_axial_log_span),
                max_radial_log_span=float(self.cfg.bone_scale_band_max_radial_log_span),
            ),
        }

    def _count_active_gaussian_centers_outside_mesh(self) -> int:
        if self.field.gaussian_count <= 0 or self.mesh_faces is None or self.mesh_faces.numel() == 0:
            return 0
        from evorig_next.utils.mesh_ops import points_inside_or_on_mesh

        centers = self.field.compute_rest_centers(self.skeleton)
        inside = points_inside_or_on_mesh(
            centers,
            self.rest_vertices,
            self.mesh_faces,
            surface_tol=float(self.cfg.seed_inside_surface_tol),
            mesh_query_scene=self.rest_mesh_scene,
        )
        return int((self.field.active_mask & (~inside)).sum().item())

    def _count_rest_joints_outside_mesh(self) -> int:
        if self.mesh_faces is None or self.mesh_faces.numel() == 0:
            return 0
        from evorig_next.utils.mesh_ops import points_inside_or_on_mesh

        inside = points_inside_or_on_mesh(
            self.skeleton.rest_joints.detach(),
            self.rest_vertices,
            self.mesh_faces,
            surface_tol=float(self.cfg.pcjs_surface_tol),
            mesh_query_scene=self.rest_mesh_scene,
        )
        return int((~inside).sum().item())

    def _phase1_trace_snapshot(self, step: int, losses: dict[str, float]) -> dict[str, Any]:
        with torch.no_grad():
            cache = self.evaluate_full()
            raw_all = float(vertex_recon_loss(cache.pred_vertices, self.gt_vertices).item())
            recon_mask = self.legal_vertex_mask.unsqueeze(0).expand(int(self.gt_vertices.shape[0]), -1).to(
                dtype=cache.pred_vertices.dtype,
                device=cache.pred_vertices.device,
            )
            raw_legal = float(vertex_recon_loss(cache.pred_vertices, self.gt_vertices, mask=recon_mask).item())
            root_trans = self.skeleton.root_trans.detach()
            root_disp = torch.linalg.norm(root_trans - root_trans[:1], dim=-1)
            rest_drift = torch.linalg.norm(
                self.skeleton.rest_joints.detach() - self.skeleton.init_rest_joints.detach(),
                dim=-1,
            )
        return {
            "step": int(step),
            "loss": float(losses.get("loss", 0.0)),
            "recon": float(losses.get("recon", 0.0)),
            "raw_error_legal": raw_legal,
            "raw_error_all": raw_all,
            "root_trans_disp_mean": float(root_disp.mean().item()),
            "root_trans_disp_max": float(root_disp.max().item()),
            "rest_joint_drift_mean": float(rest_drift.mean().item()),
            "rest_joint_drift_max": float(rest_drift.max().item()),
            "root_rest_drift": float(rest_drift[self.skeleton.parent_idx < 0].max().item()),
            "rest_joint_outside_count": self._count_rest_joints_outside_mesh(),
            "active_gaussian_count": int(self.field.active_mask.sum().item()),
            "zero_weight_row_count": int(cache.zero_weight_mask.sum().item()),
        }

    def _get_gt_mesh_scenes(self, frame_idx: torch.Tensor) -> list[MeshQueryScene | None]:
        scenes: list[MeshQueryScene | None] = []
        for item in frame_idx.tolist():
            key = int(item)
            scene = self.gt_mesh_scene_cache.get(key)
            if scene is None:
                scene = build_mesh_query_scene(self.gt_vertices[key], self.mesh_faces)
                self.gt_mesh_scene_cache[key] = scene
            scenes.append(scene)
        return scenes

    def _compute_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernels, _mix, support = self.field.compute_joint_support(
            self.rest_vertices,
            self.skeleton,
            mode=str(self.cfg.ownership_mode),
            midpoint=float(self.cfg.ownership_midpoint),
            slope=float(self.cfg.ownership_slope),
            child_gate_start=float(self.cfg.child_support_gate_start),
            child_gate_end=float(self.cfg.child_support_gate_end),
            use_endpoint_logits=False,
            endpoint_logits_mask=None,
        )
        vertex_support = support.transpose(0, 1)
        support_mass = vertex_support.sum(dim=-1, keepdim=True)
        if bool(getattr(self.cfg, "fallback_low_support_to_bone_endpoint", False)):
            threshold = float(getattr(self.cfg, "fallback_support_mass_threshold", EPS))
            fallback_rows = support_mass.squeeze(-1) <= max(threshold, 0.0)
            if bool(fallback_rows.any().item()):
                fallback_weights = self._get_bone_endpoint_fallback_weights().to(
                    dtype=vertex_support.dtype,
                    device=vertex_support.device,
                )
                vertex_support = vertex_support.clone()
                vertex_support[fallback_rows] = fallback_weights[fallback_rows]
                support = vertex_support.transpose(0, 1)
                support_mass = vertex_support.sum(dim=-1, keepdim=True)
        if bool(getattr(self.cfg, "hard_legal_support_mask", False)):
            legal = self.legal_joint_mask.to(dtype=vertex_support.dtype, device=vertex_support.device)
            vertex_support = vertex_support * legal
            masked_mass = vertex_support.sum(dim=-1, keepdim=True)
            valid_masked = masked_mass > EPS
            vertex_support = torch.where(
                valid_masked,
                vertex_support / masked_mass.clamp_min(EPS),
                vertex_support,
            )
            support = vertex_support.transpose(0, 1)
            support_mass = vertex_support.sum(dim=-1, keepdim=True)
        eps = float(getattr(self.cfg, "support_mass_eps", EPS))
        if eps > 0.0:
            denom = support_mass.clamp_min(eps)
        else:
            denom = torch.where(support_mass > 0.0, support_mass, torch.ones_like(support_mass))
        weights = vertex_support / denom
        return weights, kernels, support

    def _effective_support_mass(self, support: torch.Tensor) -> torch.Tensor:
        return support.transpose(0, 1).sum(dim=-1)

    def _zero_weight_mask(self, support: torch.Tensor) -> torch.Tensor:
        mass = self._effective_support_mass(support)
        eps = float(getattr(self.cfg, "support_mass_eps", EPS))
        threshold = eps if eps > 0.0 else 0.0
        return mass <= threshold

    @staticmethod
    def _apply_static_rest_for_uncontrolled_vertices(
        pred_vertices: torch.Tensor,
        rest_vertices: torch.Tensor,
        zero_weight_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pred_vertices.ndim != 3:
            raise ValueError("pred_vertices must have shape [T, V, 3]")
        if zero_weight_mask.ndim != 1 or zero_weight_mask.shape[0] != pred_vertices.shape[1]:
            raise ValueError("zero_weight_mask must have shape [V]")
        if not bool(zero_weight_mask.any().item()):
            return pred_vertices
        corrected = pred_vertices.clone()
        corrected[:, zero_weight_mask] = rest_vertices[zero_weight_mask].unsqueeze(0)
        return corrected

    def evaluate_full(self) -> Phase1EvalCache:
        weights, kernels, support = self._compute_weights()
        legal_support_mass = self._effective_support_mass(support)
        zero_weight_mask = self._zero_weight_mask(support)
        global_transforms = self.skeleton.forward_kinematics()
        bind_transforms = self.skeleton.compute_bind_transforms()
        pred_vertices = lbs_deform(self.rest_vertices, weights, bind_transforms, global_transforms)
        pred_vertices = self._apply_static_rest_for_uncontrolled_vertices(
            pred_vertices,
            self.rest_vertices,
            zero_weight_mask,
        )
        return Phase1EvalCache(
            weights=weights,
            kernels=kernels,
            support=support,
            legal_support_mass=legal_support_mass,
            zero_weight_mask=zero_weight_mask,
            global_transforms=global_transforms,
            pred_vertices=pred_vertices,
        )

    def _update_gaussian_grad_ema(self) -> None:
        grad_terms = []
        for parameter in (
            self.field.lambda_param,
            self.field.offset_local,
            self.field.rot_local,
            self.field.log_scale,
            self.field.log_opacity,
            self.field.log_value,
            self.field.endpoint_logits,
            self.field.sh_coeffs,
        ):
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            grad_terms.append(grad.abs() if grad.ndim == 1 else grad.norm(dim=-1))
        if not grad_terms:
            return
        grad_norm = torch.stack(grad_terms, dim=0).mean(dim=0)
        decay = float(self.cfg.grad_ema_decay)
        self.gaussian_grad_ema = decay * self.gaussian_grad_ema + (1.0 - decay) * grad_norm

    def train_step(self, step: int) -> dict[str, float]:
        self.current_step = int(step)
        self._prepare_staged_params_for_step(step)
        frame_idx = self._sample_frames()
        if self._use_manual_lambda_optimizer() and self.field.lambda_param.grad is not None:
            self.field.lambda_param.grad = None
        self.optimizer.zero_grad(set_to_none=True)
        weights, kernels, support = self._compute_weights()
        zero_weight_mask = self._zero_weight_mask(support)
        global_transforms = self.skeleton.forward_kinematics(frame_idx=frame_idx)
        bind_transforms = self.skeleton.compute_bind_transforms()
        pred_vertices = lbs_deform(self.rest_vertices, weights, bind_transforms, global_transforms)
        pred_vertices = self._apply_static_rest_for_uncontrolled_vertices(
            pred_vertices,
            self.rest_vertices,
            zero_weight_mask,
        )
        gt_vertices = self.gt_vertices[frame_idx]
        posed_joints = global_transforms[..., :3, 3]
        joint_rotations = global_transforms[..., :3, :3]
        gt_mesh_scenes = self._get_gt_mesh_scenes(frame_idx)

        recon_mask = self.legal_vertex_mask.unsqueeze(0).expand(int(gt_vertices.shape[0]), -1).to(dtype=pred_vertices.dtype, device=pred_vertices.device)
        recon = vertex_recon_loss(
            pred_vertices,
            gt_vertices,
            mask=recon_mask,
            reference_length=self.sample_radius,
        ) * float(self.cfg.loss_vertex_recon)
        smooth = temporal_smoothness_loss(
            self.skeleton.pose_rot,
            self.skeleton.root_trans,
            root_reference_length=self.sample_radius,
        ) * float(self.cfg.loss_temporal_smoothness)
        acceleration_weight = float(getattr(self.cfg, "loss_vertex_acceleration", 0.0))
        if acceleration_weight > 0.0:
            full_global_transforms = self.skeleton.forward_kinematics()
            full_pred_vertices = lbs_deform(self.rest_vertices, weights, bind_transforms, full_global_transforms)
            full_pred_vertices = self._apply_static_rest_for_uncontrolled_vertices(
                full_pred_vertices,
                self.rest_vertices,
                zero_weight_mask,
            )
            vertex_acceleration = vertex_acceleration_loss(
                full_pred_vertices,
                self.gt_vertices,
                mask=self.legal_vertex_mask,
                reference_length=self.sample_radius,
            ) * acceleration_weight
        else:
            vertex_acceleration = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        illegal = illegal_support_loss(
            support,
            self.legal_joint_mask,
            tau=float(self.cfg.illegal_support_tau),
            margin=float(getattr(self.cfg, "illegal_support_margin", 0.0)),
        ) * float(self.cfg.loss_illegal_support)
        eligible_joint_mask = (self.skeleton.parent_idx >= 0).to(dtype=self.rest_vertices.dtype, device=self.device)
        all_joint_mask = torch.ones_like(eligible_joint_mask)
        pcjs_weight = float(self.cfg.loss_pcjs)
        posed_inside_weight_value = float(self.cfg.loss_posed_joint_inside)
        pcjs_section_lambda = float(getattr(self.cfg, "pcjs_section_lambda", 0.1))
        pcjs_shell_descriptors = None
        if pcjs_weight > 0.0:
            support_parent_idx = self.skeleton.support_parent_idx()
            pcjs_shell_descriptors = posed_joint_shell_descriptors(
                posed_joints,
                joint_rotations,
                gt_vertices,
                self.mesh_faces,
                surface_tol=float(self.cfg.pcjs_surface_tol),
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scenes=gt_mesh_scenes,
                parent_idx=support_parent_idx,
                section_lambda=pcjs_section_lambda,
            )
            pcjs = pose_consistent_joint_shell_loss(
                posed_joints,
                joint_rotations,
                gt_vertices,
                self.mesh_faces,
                joint_weight=eligible_joint_mask,
                surface_tol=float(self.cfg.pcjs_surface_tol),
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scenes=gt_mesh_scenes,
                shell_descriptors=pcjs_shell_descriptors,
                reference_length=self.sample_radius,
                parent_idx=support_parent_idx,
                section_lambda=pcjs_section_lambda,
            ) * pcjs_weight
        else:
            pcjs = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        posed_joint_weight = eligible_joint_mask.clone()
        root_mask = self.skeleton.parent_idx < 0
        if bool(root_mask.any().item()):
            posed_joint_weight[root_mask] = float(self.cfg.posed_joint_inside_root_weight)
        if posed_inside_weight_value > 0.0:
            posed_inside = posed_joint_inside_mesh_loss(
                posed_joints,
                gt_vertices,
                self.mesh_faces,
                joint_weight=posed_joint_weight,
                surface_tol=float(self.cfg.pcjs_surface_tol),
                joint_rotations=joint_rotations,
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scenes=gt_mesh_scenes,
                shell_descriptors=None,
                reference_length=self.sample_radius,
            ) * posed_inside_weight_value
        else:
            posed_inside = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        posed_bone_inside_weight = float(getattr(self.cfg, "loss_posed_bone_inside_mesh", 0.0))
        if posed_bone_inside_weight > 0.0:
            posed_bone_inside = posed_bone_inside_mesh_loss(
                posed_joints,
                self.skeleton.support_parent_idx(),
                gt_vertices,
                self.mesh_faces,
                samples_per_bone=int(getattr(self.cfg, "posed_bone_inside_samples", 4)),
                surface_tol=float(self.cfg.pcjs_surface_tol),
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scenes=gt_mesh_scenes,
                reference_length=self.sample_radius,
            ) * posed_bone_inside_weight
        else:
            posed_bone_inside = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        posed_clearance_weight = float(getattr(self.cfg, "loss_posed_joint_surface_clearance", 0.0))
        if posed_clearance_weight > 0.0:
            clearance_margin = float(getattr(self.cfg, "posed_joint_surface_clearance_margin", 0.0))
            if clearance_margin <= 0.0:
                clearance_margin = (
                    float(getattr(self.cfg, "posed_joint_surface_clearance_ratio", 0.02))
                    * float(self.sample_radius)
                )
            posed_clearance = posed_joint_surface_clearance_loss(
                posed_joints,
                gt_vertices,
                self.mesh_faces,
                min_clearance=clearance_margin,
                joint_weight=posed_joint_weight,
                surface_tol=float(self.cfg.pcjs_surface_tol),
                joint_rotations=joint_rotations,
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scenes=gt_mesh_scenes,
                reference_length=self.sample_radius,
            ) * posed_clearance_weight
        else:
            posed_clearance = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        scale_anchor = gaussian_log_scale_anchor_loss(
            self.field.log_scale,
            self.field.init_log_scale,
            active_mask=self.field.active_mask,
        ) * float(self.cfg.loss_scale_anchor)
        bone_scale_consistency = bone_scale_consistency_loss(
            self.field.anchor_bone,
            self.field.log_scale,
            active_mask=self.field.active_mask,
        ) * float(self.cfg.loss_bone_scale_consistency)
        rest_joint_anchor = skeleton_anchor_loss(
            self.skeleton.rest_joints,
            self.skeleton.init_rest_joints,
            reference_length=self.sample_radius,
        ) * float(self.cfg.loss_rest_joint_anchor)
        rest_inside_weight = float(getattr(self.cfg, "loss_rest_joint_inside", 0.0))
        if rest_inside_weight > 0.0:
            rest_joint_inside = joint_inside_mesh_loss(
                self.skeleton.rest_joints,
                self.rest_vertices,
                self.mesh_faces,
                joint_weight=all_joint_mask,
                surface_tol=float(self.cfg.pcjs_surface_tol),
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scene=self.rest_mesh_scene,
                reference_length=self.sample_radius,
            ) * rest_inside_weight
        else:
            rest_joint_inside = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        rest_clearance_weight = float(getattr(self.cfg, "loss_rest_joint_surface_clearance", 0.0))
        if rest_clearance_weight > 0.0:
            rest_clearance_margin = float(getattr(self.cfg, "rest_joint_surface_clearance_margin", 0.0))
            if rest_clearance_margin <= 0.0:
                rest_clearance_margin = (
                    float(getattr(self.cfg, "rest_joint_surface_clearance_ratio", 0.02))
                    * float(self.sample_radius)
                )
            rest_joint_clearance = rest_joint_surface_clearance_loss(
                self.skeleton.rest_joints,
                self.rest_vertices,
                self.mesh_faces,
                min_clearance=rest_clearance_margin,
                joint_weight=all_joint_mask,
                surface_tol=float(self.cfg.pcjs_surface_tol),
                direction_count=int(self.cfg.pcjs_direction_count),
                mesh_query_scene=self.rest_mesh_scene,
                reference_length=self.sample_radius,
            ) * rest_clearance_weight
        else:
            rest_joint_clearance = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        if float(getattr(self.cfg, "loss_bone_radial_distance_shrink", 0.0)) > 0.0:
            radial_cache = self.bone_radial_distance_cache
            bone_radial_shrink = bone_radial_distance_shrink_loss(
                pred_vertices,
                posed_joints,
                radial_cache["bone_index"],
                radial_cache["parent_joint"],
                radial_cache["child_joint"],
                radial_cache["rest_radial_distance"],
                min_ratio=float(getattr(self.cfg, "bone_radial_distance_shrink_ratio", 0.95)),
            ) * float(self.cfg.loss_bone_radial_distance_shrink)
        else:
            bone_radial_shrink = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        if float(getattr(self.cfg, "loss_mesh_edge_length_floor", 0.0)) > 0.0:
            edge_src, edge_dst, _degree = self._get_rest_mesh_edge_index()
            mesh_edge_length_floor = mesh_edge_length_floor_loss(
                pred_vertices,
                self.rest_vertices,
                edge_src,
                edge_dst,
                min_ratio=float(getattr(self.cfg, "mesh_edge_length_floor_ratio", 0.9)),
            ) * float(getattr(self.cfg, "loss_mesh_edge_length_floor", 0.0))
        else:
            mesh_edge_length_floor = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        if float(self.cfg.loss_gaussian_sh_reg) > 0.0 and int(self.field.sh_coeff_count) > 1:
            gaussian_sh_reg = self.field.sh_coeffs[:, 1:].square().mean() * float(self.cfg.loss_gaussian_sh_reg)
        else:
            gaussian_sh_reg = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        if float(getattr(self.cfg, "loss_gaussian_offset_anchor", 0.0)) > 0.0:
            offset_mask = self._gaussian_offset_train_mask().to(device=self.field.offset_local.device, dtype=torch.bool)
            if bool(offset_mask.any().item()):
                offset_norm = self.field.offset_local[offset_mask].square().sum(dim=-1)
                gaussian_offset_anchor = (
                    offset_norm / max(float(self.sample_radius) ** 2, float(EPS))
                ).mean() * float(self.cfg.loss_gaussian_offset_anchor)
            else:
                gaussian_offset_anchor = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        else:
            gaussian_offset_anchor = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
        if (
            float(self.cfg.loss_gaussian_illegal_coverage) > 0.0
            or float(self.cfg.loss_bone_cov_offdiag) > 0.0
            or float(self.cfg.loss_bone_radial_symmetry) > 0.0
            or float(self.cfg.loss_bone_scale_band) > 0.0
        ):
            control_diag = self._compute_field_control_diagnostics(kernels)
            gaussian_illegal_coverage = control_diag["gaussian_illegal_coverage"] * float(self.cfg.loss_gaussian_illegal_coverage)
            bone_cov_offdiag = control_diag["bone_cov_offdiag"] * float(self.cfg.loss_bone_cov_offdiag)
            bone_radial_symmetry = control_diag["bone_radial_symmetry"] * float(self.cfg.loss_bone_radial_symmetry)
            bone_scale_band = control_diag["bone_scale_band"] * float(self.cfg.loss_bone_scale_band)
        else:
            zero = torch.zeros((), dtype=self.rest_vertices.dtype, device=self.device)
            gaussian_illegal_coverage = zero
            bone_cov_offdiag = zero
            bone_radial_symmetry = zero
            bone_scale_band = zero
        total = (
            recon
            + smooth
            + vertex_acceleration
            + illegal
            + pcjs
            + posed_inside
            + posed_bone_inside
            + posed_clearance
            + scale_anchor
            + bone_scale_consistency
            + rest_joint_anchor
            + rest_joint_inside
            + rest_joint_clearance
            + bone_radial_shrink
            + mesh_edge_length_floor
            + gaussian_sh_reg
            + gaussian_offset_anchor
            + gaussian_illegal_coverage
            + bone_cov_offdiag
            + bone_radial_symmetry
            + bone_scale_band
        )
        total.backward()
        self._apply_staged_gradient_freezes(step)
        self._update_gaussian_grad_ema()
        self.optimizer.step()
        self._restore_inactive_rest_joints_after_step()
        self._project_rest_joints_inside_after_step(step)
        self._restore_inactive_rest_joints_after_step()
        self._step_lambda_param()
        self.field.clamp_lambda_param()
        if self._use_manual_lambda_optimizer():
            self.field.lambda_param.grad = None
        return {
            "loss": float(total.item()),
            "recon": float(recon.item()),
            "smooth": float(smooth.item()),
            "vertex_acceleration": float(vertex_acceleration.item()),
            "illegal_support": float(illegal.item()),
            "pcjs": float(pcjs.item()),
            "posed_joint_inside": float(posed_inside.item()),
            "posed_bone_inside": float(posed_bone_inside.item()),
            "posed_joint_clearance": float(posed_clearance.item()),
            "scale_anchor": float(scale_anchor.item()),
            "bone_scale_consistency": float(bone_scale_consistency.item()),
            "rest_joint_anchor": float(rest_joint_anchor.item()),
            "rest_joint_inside": float(rest_joint_inside.item()),
            "rest_joint_clearance": float(rest_joint_clearance.item()),
            "bone_radial_shrink": float(bone_radial_shrink.item()),
            "mesh_edge_length_floor": float(mesh_edge_length_floor.item()),
            "gaussian_sh_reg": float(gaussian_sh_reg.item()),
            "gaussian_offset_anchor": float(gaussian_offset_anchor.item()),
            "gaussian_illegal_coverage": float(gaussian_illegal_coverage.item()),
            "bone_cov_offdiag": float(bone_cov_offdiag.item()),
            "bone_radial_symmetry": float(bone_radial_symmetry.item()),
            "bone_scale_band": float(bone_scale_band.item()),
            "gaussian_count": int(self.field.gaussian_count),
            "zero_weight_row_count": int(zero_weight_mask.sum().item()),
        }

    def _score_bones_for_densify(self) -> torch.Tensor:
        with torch.no_grad():
            cache = self.evaluate_full()
            vertex_error = torch.linalg.norm(cache.pred_vertices - self.gt_vertices, dim=-1).mean(dim=0)
            vertex_error = vertex_error * self.legal_vertex_mask.to(dtype=vertex_error.dtype, device=vertex_error.device)
            responsibility = cache.kernels / cache.kernels.sum(dim=0, keepdim=True).clamp_min(EPS)
            gaussian_error = (responsibility * vertex_error.unsqueeze(0)).sum(dim=-1)
            gaussian_score = gaussian_error * self.gaussian_grad_ema.clamp_min(1.0e-8)
            bone_count = int(self.skeleton.bone_count)
            bone_score = torch.zeros(bone_count, dtype=self.rest_vertices.dtype, device=self.device)
            bone_score.index_add_(0, self.field.anchor_bone, gaussian_score)
            return bone_score

    def densify(self, stage: Phase1DensifyStage, generation_value: int) -> dict[str, Any]:
        bone_score = self._score_bones_for_densify()
        topk = min(int(stage.max_bones), int(bone_score.numel()))
        if topk <= 0:
            return {"added": 0, "selected_bones": []}
        values, indices = torch.topk(bone_score, k=topk, largest=True)
        keep = values > 0.0
        selected_bones = indices[keep]
        if selected_bones.numel() <= 0:
            return {"added": 0, "selected_bones": []}
        densify_result = self.field.append_axis_gaussians_for_bones(
            self.rest_vertices,
            self.skeleton,
            self._field_init_config(),
            bone_indices=selected_bones,
            seeds_per_bone=int(stage.seeds_per_bone),
            generation_value=int(generation_value),
            faces=self.mesh_faces,
            prune_outside_mesh=bool(self.cfg.densify_seed_prune_outside_mesh),
            surface_tol=float(self.cfg.seed_inside_surface_tol),
            mesh_query_scene=self.rest_mesh_scene,
        )
        new_ids = densify_result["new_ids"]
        if new_ids.numel() > 0:
            self.gaussian_grad_ema = torch.cat(
                [
                    self.gaussian_grad_ema,
                    torch.zeros(int(new_ids.numel()), dtype=self.gaussian_grad_ema.dtype, device=self.gaussian_grad_ema.device),
                ],
                dim=0,
            )
            self.optimizer = self._build_optimizer()
            if self._use_manual_lambda_optimizer():
                self._init_lambda_optimizer_state()
        return {
            "added": int(new_ids.numel()),
            "removed_outside_count": int(densify_result["removed_outside_count"]),
            "selected_bones": [int(item) for item in selected_bones.tolist()],
            "skipped_bones_outside_only": [int(item) for item in densify_result["skipped_bones_outside_only"]],
            "bone_scores": [float(item) for item in values[keep].tolist()],
        }

    def _phase1_state_payload(self) -> dict[str, Any]:
        numpy_random_state = np.random.get_state()
        runtime_state: dict[str, Any] = {
            "rng": {
                "python_random": random.getstate(),
                "numpy_random": {
                    "bit_generator": str(numpy_random_state[0]),
                    "state": torch.as_tensor(numpy_random_state[1].astype(np.int64), dtype=torch.long),
                    "pos": int(numpy_random_state[2]),
                    "has_gauss": int(numpy_random_state[3]),
                    "cached_gaussian": float(numpy_random_state[4]),
                },
                "torch_cpu": torch.get_rng_state().detach().cpu(),
                "torch_cuda_all": [
                    item.detach().cpu()
                    for item in torch.cuda.get_rng_state_all()
                ] if torch.cuda.is_available() else [],
            },
            "sh_initialized": bool(getattr(self, "_sh_initialized", False)),
        }
        rest_joint_train_mask = getattr(self, "rest_joint_train_mask", None)
        if isinstance(rest_joint_train_mask, torch.Tensor):
            runtime_state["rest_joint_train_mask"] = rest_joint_train_mask.detach().cpu()
        fallback_weights = getattr(self, "_bone_endpoint_fallback_weights", None)
        if isinstance(fallback_weights, torch.Tensor):
            runtime_state["bone_endpoint_fallback_weights"] = fallback_weights.detach().cpu()
        branch_lineages = getattr(self, "phase2_branch_lineages", None)
        if isinstance(branch_lineages, list):
            runtime_state["phase2_branch_lineages"] = branch_lineages
        return {
            "format": "evorig_next_phase1_state_v1",
            "current_step": int(self.current_step),
            "base_config": self.base_config,
            "phase1_config": self.cfg.to_dict(),
            "sample_radius": float(self.sample_radius),
            "sample_shapes": {
                "rest_vertices": list(self.rest_vertices.shape),
                "faces": list(self.mesh_faces.shape),
                "gt_vertices": list(self.gt_vertices.shape),
            },
            "skeleton": {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.skeleton.state_dict().items()
                },
                "birth_steps": [int(item) for item in self.skeleton.birth_steps],
                "is_inserted": [bool(item) for item in self.skeleton.is_inserted],
                "birth_modes": [str(item) for item in self.skeleton.birth_modes],
            },
            "field": {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.field.state_dict().items()
                },
                "kernel_mahal_cutoff_sq": float(self.field.kernel_mahal_cutoff_sq),
                "use_sh_response": bool(self.field.use_sh_response),
                "initial_seed_removed_outside_count": int(self.field.initial_seed_removed_outside_count),
                "densify_seed_removed_outside_count": int(self.field.densify_seed_removed_outside_count),
                "final_gaussian_pruned_outside_count": int(self.field.final_gaussian_pruned_outside_count),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "manual_lambda_optimizer": {
                "m": self.lambda_adam_m.detach().cpu(),
                "v": self.lambda_adam_v.detach().cpu(),
                "step": self.lambda_adam_step.detach().cpu(),
            },
            "gaussian_grad_ema": self.gaussian_grad_ema.detach().cpu(),
            "legal_joint_mask": self.legal_joint_mask.detach().cpu(),
            "legal_vertex_mask": self.legal_vertex_mask.detach().cpu(),
            "legality_diagnostics": self.legality_diagnostics,
            "runtime_state": runtime_state,
        }

    def _save_phase1_state(self, output_dir: Path) -> Path:
        path = Path(output_dir) / "phase1_state.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._phase1_state_payload(), tmp_path)
        tmp_path.replace(path)
        return path

    def _restore_rng_state(self, runtime_state: dict[str, Any]) -> None:
        rng_state = runtime_state.get("rng", {})
        if not isinstance(rng_state, dict):
            return
        python_state = rng_state.get("python_random")
        if python_state is not None:
            random.setstate(python_state)
        numpy_state = rng_state.get("numpy_random")
        if isinstance(numpy_state, dict):
            state_tensor = numpy_state.get("state")
            if isinstance(state_tensor, torch.Tensor):
                np.random.set_state(
                    (
                        str(numpy_state.get("bit_generator", "MT19937")),
                        state_tensor.detach().cpu().numpy().astype(np.uint32),
                        int(numpy_state.get("pos", 0)),
                        int(numpy_state.get("has_gauss", 0)),
                        float(numpy_state.get("cached_gaussian", 0.0)),
                    )
                )
        elif numpy_state is not None:
            np.random.set_state(numpy_state)
        torch_cpu = rng_state.get("torch_cpu")
        if isinstance(torch_cpu, torch.Tensor):
            torch.set_rng_state(torch_cpu.detach().to(device="cpu", dtype=torch.uint8))
        torch_cuda_all = rng_state.get("torch_cuda_all", [])
        if torch.cuda.is_available() and isinstance(torch_cuda_all, list) and len(torch_cuda_all) > 0:
            torch.cuda.set_rng_state_all(
                [item.detach().to(device="cpu", dtype=torch.uint8) for item in torch_cuda_all if isinstance(item, torch.Tensor)]
            )

    def load_phase1_payload(
        self,
        payload: dict[str, Any],
        *,
        restore_optimizer: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        if str(payload.get("format", "")) != "evorig_next_phase1_state_v1":
            raise ValueError(f"unsupported phase1 state format: {payload.get('format')}")
        skeleton_state = payload["skeleton"]["state_dict"]
        parent_idx = skeleton_state["parent_idx"].to(device=self.device, dtype=torch.long)
        if "connected_to_parent" not in skeleton_state:
            skeleton_state["connected_to_parent"] = (parent_idx >= 0).to(device=parent_idx.device, dtype=torch.bool)
        init_rest_joints = skeleton_state["init_rest_joints"].to(device=self.device, dtype=self.rest_vertices.dtype)
        pose_rot = skeleton_state["pose_rot"].to(device=self.device, dtype=self.rest_vertices.dtype)
        self.skeleton = Phase1Skeleton(
            parent_idx=parent_idx,
            rest_joints=init_rest_joints,
            frame_count=int(pose_rot.shape[0]),
            init_pose=pose_rot,
            birth_steps=payload["skeleton"].get("birth_steps"),
            inserted=payload["skeleton"].get("is_inserted"),
            birth_modes=payload["skeleton"].get("birth_modes"),
            connected_to_parent=skeleton_state["connected_to_parent"].to(device=self.device, dtype=torch.bool),
        ).to(self.device)
        self.skeleton.load_state_dict(
            {key: value.to(self.device) for key, value in skeleton_state.items()},
            strict=True,
        )
        self.skeleton._refresh_bones()
        field_state = payload["field"]["state_dict"]
        self.field = Phase1GaussianField(
            Phase1FieldState(
                anchor_bone=field_state["anchor_bone"].to(device=self.device, dtype=torch.long),
                lambda_param=field_state["lambda_param"].to(device=self.device, dtype=self.rest_vertices.dtype),
                lambda_min=field_state["lambda_min"].to(device=self.device, dtype=self.rest_vertices.dtype),
                lambda_max=field_state["lambda_max"].to(device=self.device, dtype=self.rest_vertices.dtype),
                offset_local=field_state["offset_local"].to(device=self.device, dtype=self.rest_vertices.dtype),
                rot_local=field_state["rot_local"].to(device=self.device, dtype=self.rest_vertices.dtype),
                log_scale=field_state["log_scale"].to(device=self.device, dtype=self.rest_vertices.dtype),
                init_log_scale=field_state["init_log_scale"].to(device=self.device, dtype=self.rest_vertices.dtype),
                log_opacity=field_state["log_opacity"].to(device=self.device, dtype=self.rest_vertices.dtype),
                log_value=field_state["log_value"].to(device=self.device, dtype=self.rest_vertices.dtype),
                kernel_mahal_cutoff_sq=float(payload["field"].get("kernel_mahal_cutoff_sq", self.cfg.gaussian_kernel_mahal_cutoff_sq)),
            )
        ).to(self.device)
        if "sh_coeffs" in field_state:
            self.field.ensure_sh_coeffs(
                int(field_state["sh_coeffs"].shape[1]),
                preserve_unit_density=False,
            )
        self.field.load_state_dict(
            {key: value.to(self.device) for key, value in field_state.items()},
            strict=True,
        )
        self.field.kernel_mahal_cutoff_sq = float(payload["field"].get("kernel_mahal_cutoff_sq", self.field.kernel_mahal_cutoff_sq))
        self.field.use_sh_response = bool(payload["field"].get("use_sh_response", self.field.use_sh_response))
        self.field.initial_seed_removed_outside_count = int(payload["field"].get("initial_seed_removed_outside_count", 0))
        self.field.densify_seed_removed_outside_count = int(payload["field"].get("densify_seed_removed_outside_count", 0))
        self.field.final_gaussian_pruned_outside_count = int(payload["field"].get("final_gaussian_pruned_outside_count", 0))
        self.gaussian_grad_ema = payload.get(
            "gaussian_grad_ema",
            torch.zeros(self.field.gaussian_count, dtype=self.rest_vertices.dtype),
        ).to(device=self.device, dtype=self.rest_vertices.dtype)
        self.current_step = int(payload.get("current_step", 0))
        runtime_state = payload.get("runtime_state", {})
        if not isinstance(runtime_state, dict):
            runtime_state = {}
        self._sh_initialized = bool(runtime_state.get("sh_initialized", self._sh_active()))
        self.refresh_after_topology_mutation()
        saved_legal_joint_mask = payload.get("legal_joint_mask")
        if isinstance(saved_legal_joint_mask, torch.Tensor):
            saved_legal_joint_mask = saved_legal_joint_mask.to(device=self.device, dtype=torch.bool)
            if tuple(saved_legal_joint_mask.shape) == tuple(self.legal_joint_mask.shape):
                self.legal_joint_mask = saved_legal_joint_mask
                self.legal_vertex_mask = self.legal_joint_mask.any(dim=-1)
        saved_train_mask = runtime_state.get("rest_joint_train_mask")
        if isinstance(saved_train_mask, torch.Tensor) and int(saved_train_mask.numel()) == int(self.skeleton.joint_count):
            self.rest_joint_train_mask = saved_train_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        elif hasattr(self, "rest_joint_train_mask"):
            delattr(self, "rest_joint_train_mask")
        saved_fallback = runtime_state.get("bone_endpoint_fallback_weights")
        if (
            isinstance(saved_fallback, torch.Tensor)
            and tuple(saved_fallback.shape) == (int(self.rest_vertices.shape[0]), int(self.skeleton.joint_count))
        ):
            self._bone_endpoint_fallback_weights = saved_fallback.to(device=self.device, dtype=self.rest_vertices.dtype)
        elif hasattr(self, "_bone_endpoint_fallback_weights"):
            delattr(self, "_bone_endpoint_fallback_weights")
        saved_branch_lineages = runtime_state.get("phase2_branch_lineages")
        if isinstance(saved_branch_lineages, list):
            self.phase2_branch_lineages = saved_branch_lineages
        elif hasattr(self, "phase2_branch_lineages"):
            delattr(self, "phase2_branch_lineages")
        if restore_optimizer and "optimizer_state_dict" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
                for state in self.optimizer.state.values():
                    for key, value in list(state.items()):
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self.device)
            except ValueError:
                self.optimizer = self._build_optimizer()
        manual = payload.get("manual_lambda_optimizer", {})
        if self._use_manual_lambda_optimizer() and manual:
            self.lambda_adam_m = manual.get("m", self.lambda_adam_m).to(device=self.device, dtype=self.field.lambda_param.dtype)
            self.lambda_adam_v = manual.get("v", self.lambda_adam_v).to(device=self.device, dtype=self.field.lambda_param.dtype)
            self.lambda_adam_step = manual.get("step", self.lambda_adam_step).to(device=self.device, dtype=torch.long)
        if restore_rng:
            self._restore_rng_state(runtime_state)
        return payload

    def load_phase1_state(
        self,
        path: str | Path,
        *,
        restore_optimizer: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device)
        return self.load_phase1_payload(payload, restore_optimizer=restore_optimizer, restore_rng=restore_rng)

    def run(self, output_dir: Path, *, export_topology_signals: bool = True) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        step = 0
        stage_summaries = []
        trace_interval = int(getattr(self.cfg, "trace_interval_steps", 0))
        trace: list[dict[str, Any]] = []
        live_trace_path = output_dir / "phase1_trace_live.jsonl"
        if trace_interval > 0:
            live_trace_path.write_text("", encoding="utf-8")

        def record_step(current_step: int, losses: dict[str, float]) -> None:
            if trace_interval <= 0:
                return
            if current_step % trace_interval != 0 and current_step != int(self.cfg.steps):
                return
            snapshot = self._phase1_trace_snapshot(current_step, losses)
            trace.append(snapshot)
            with live_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        def update_progress(progress: Any, losses: dict[str, float]) -> None:
            progress.update(1)
            if trace_interval > 0 and (step % trace_interval == 0 or step == int(self.cfg.steps)):
                progress.set_postfix(
                    loss=f"{float(losses.get('loss', 0.0)):.4g}",
                    recon=f"{float(losses.get('recon', 0.0)):.4g}",
                    zw=int(losses.get("zero_weight_row_count", 0)),
                )

        progress = tqdm(
            total=int(self.cfg.steps),
            desc="Phase1",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        try:
            for stage_index, stage in enumerate(self.cfg.densify_stages):
                for _ in range(int(stage.warm_steps)):
                    step += 1
                    if step > int(self.cfg.steps):
                        break
                    losses = self.train_step(step)
                    record_step(step, losses)
                    update_progress(progress, losses)
                if step > int(self.cfg.steps):
                    break
                densify_summary = self.densify(stage, generation_value=stage_index + 1)
                for _ in range(int(stage.settle_steps)):
                    step += 1
                    if step > int(self.cfg.steps):
                        break
                    losses = self.train_step(step)
                    record_step(step, losses)
                    update_progress(progress, losses)
                stage_summaries.append(
                    {
                        "stage_index": int(stage_index),
                        "warm_steps": int(stage.warm_steps),
                        "settle_steps": int(stage.settle_steps),
                        "densify": densify_summary,
                        "step_after_stage": int(step),
                    }
                )
                if step > int(self.cfg.steps):
                    break
            while step < int(self.cfg.steps):
                step += 1
                losses = self.train_step(step)
                record_step(step, losses)
                update_progress(progress, losses)
        finally:
            progress.close()

        if bool(self.cfg.final_gaussian_prune_outside_mesh):
            self.field.prune_active_outside_mesh(
                self.skeleton,
                self.rest_vertices,
                self.mesh_faces,
                surface_tol=float(self.cfg.seed_inside_surface_tol),
                mesh_query_scene=self.rest_mesh_scene,
            )
        cache = self.evaluate_full()
        pred_joint_positions = cache.global_transforms[..., :3, 3]
        pred_joint_rotations = cache.global_transforms[..., :3, :3]
        save_outputs(
            output_dir=output_dir,
            skeleton=self.skeleton,
            field=self.field,
            pred_vertices=cache.pred_vertices,
            pred_joint_positions=pred_joint_positions,
            pred_joint_rotations=pred_joint_rotations,
            weights=cache.weights,
            events=[],
            topology_diagnostics=[],
        )
        recon_mask = self.legal_vertex_mask.unsqueeze(0).expand(int(self.gt_vertices.shape[0]), -1).to(dtype=cache.pred_vertices.dtype, device=cache.pred_vertices.device)
        final_error_raw = float(vertex_recon_loss(cache.pred_vertices, self.gt_vertices, mask=recon_mask).item())
        final_error = float(
            vertex_recon_loss(
                cache.pred_vertices,
                self.gt_vertices,
                mask=recon_mask,
                reference_length=self.sample_radius,
            ).item()
        )
        final_error_raw_all = float(vertex_recon_loss(cache.pred_vertices, self.gt_vertices).item())
        structure_audit = audit_dominant_connectivity(
            self.mesh_faces,
            int(self.rest_vertices.shape[0]),
            int(self.skeleton.joint_count),
            dominant_joint_assignment(cache.weights, cache.legal_support_mass, eps=EPS),
        )
        no_joint_mask = cache.zero_weight_mask
        no_joint_disp = torch.linalg.norm(
            cache.pred_vertices - self.rest_vertices.unsqueeze(0),
            dim=-1,
        )
        if bool(no_joint_mask.any().item()):
            no_joint_mean = float(no_joint_disp[:, no_joint_mask].mean().item())
            no_joint_max = float(no_joint_disp[:, no_joint_mask].max().item())
        else:
            no_joint_mean = 0.0
            no_joint_max = 0.0
        watch_joint_rotation = summarize_joint_rotation_deltas(
            pred_joint_rotations,
            tuple(int(item) for item in self.cfg.watch_joint_ids),
        )
        final_control_diag = self._compute_field_control_diagnostics(cache.kernels)
        final_outside_active_gaussian_count = self._count_active_gaussian_centers_outside_mesh()
        acceptance = summarize_structure_acceptance(
            disconnected_joint_ids=structure_audit["disconnected_joint_ids"],
            no_joint_rest_displacement_max=no_joint_max,
            no_joint_rest_displacement_tol=float(self.cfg.no_joint_rest_displacement_tol),
        )
        summary = {
            "steps": int(self.cfg.steps),
            "gaussian_count": int(self.field.gaussian_count),
            "joint_count": int(self.skeleton.joint_count),
            "final_error": final_error,
            "final_error_raw": final_error_raw,
            "final_error_raw_all": final_error_raw_all,
            "legal_joint_mean_per_vertex": float(self.legal_joint_mask.sum(dim=-1).float().mean().item()),
            "legal_joint_min_per_vertex": int(self.legal_joint_mask.sum(dim=-1).min().item()),
            "legal_joint_max_per_vertex": int(self.legal_joint_mask.sum(dim=-1).max().item()),
            "zero_legal_joint_row_count": int((self.legal_joint_mask.sum(dim=-1) <= 0).sum().item()),
            "zero_weight_row_count": int((cache.zero_weight_mask).sum().item()),
            "legal_vertex_count": int(self.legal_vertex_mask.sum().item()),
            "legal_vertex_fraction": float(self.legal_vertex_mask.to(dtype=self.rest_vertices.dtype).mean().item()),
            "dominant_joint_component_counts": structure_audit["dominant_joint_component_counts"],
            "dominant_joint_vertex_counts": structure_audit["dominant_joint_vertex_counts"],
            "disconnected_joint_ids": structure_audit["disconnected_joint_ids"],
            "no_joint_vertex_count": structure_audit["no_joint_vertex_count"],
            "no_joint_vertex_fraction": structure_audit["no_joint_vertex_fraction"],
            "no_joint_rest_displacement_mean": no_joint_mean,
            "no_joint_rest_displacement_max": no_joint_max,
            "watch_joint_rotation": watch_joint_rotation,
            "gaussian_illegal_coverage": float(final_control_diag["gaussian_illegal_coverage"].item()),
            "bone_cov_offdiag": float(final_control_diag["bone_cov_offdiag"].item()),
            "bone_radial_symmetry": float(final_control_diag["bone_radial_symmetry"].item()),
            "bone_scale_band": float(final_control_diag["bone_scale_band"].item()),
            "initial_seed_removed_outside_count": int(self.field.initial_seed_removed_outside_count),
            "densify_seed_removed_outside_count": int(self.field.densify_seed_removed_outside_count),
            "final_gaussian_pruned_outside_count": int(self.field.final_gaussian_pruned_outside_count),
            "final_outside_active_gaussian_count": int(final_outside_active_gaussian_count),
            "active_gaussian_count": int(self.field.active_mask.sum().item()),
            "controllable_vertex_count": int((~cache.zero_weight_mask).sum().item()),
            "controllable_vertex_fraction": float((~cache.zero_weight_mask).to(dtype=self.rest_vertices.dtype).mean().item()),
            "acceptance": acceptance,
            "legality_propagation": self.legality_diagnostics,
            "densify_stages": stage_summaries,
            "staged_trainability": {
                "lambda_thaw_active": bool(
                    int(self.cfg.lambda_thaw_start_step) >= 0
                    and int(self.current_step) >= int(self.cfg.lambda_thaw_start_step)
                ),
                "lambda_thaw_target": str(self.cfg.lambda_thaw_target),
                "rest_joint_active": bool(self._rest_joint_active()),
                "sh_active": bool(self._sh_active()),
                "sh_initialized": bool(self._sh_initialized),
            },
            "trace_interval_steps": int(trace_interval),
            "trace_path": str(output_dir / "phase1_trace.json") if trace else None,
            "trace_live_path": str(live_trace_path) if trace_interval > 0 else None,
            "config": self.cfg.to_dict(),
        }
        if trace:
            (output_dir / "phase1_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        state_path = self._save_phase1_state(output_dir)
        summary["phase1_state_path"] = str(state_path)
        if bool(export_topology_signals):
            from evorig_next.phase2_topology import save_phase2_topology_signals

            phase2_signal_summary = save_phase2_topology_signals(output_dir, self, cache)
            summary["phase2_topology_signals_path"] = str(output_dir / "phase2_topology_signals.npz")
            summary["phase2_topology_signal_summary_path"] = str(output_dir / "phase2_topology_signal_summary.json")
            summary["phase2_topology_signal_summary"] = {
                "branch_seed_vertex_count": int(phase2_signal_summary["branch_seed_vertex_count"]),
                "branch_component_count": int(phase2_signal_summary["branch_component_count"]),
                "split_candidate_count": int(phase2_signal_summary["split_candidate_count"]),
            }
        else:
            summary["phase2_topology_signals_path"] = None
            summary["phase2_topology_signal_summary_path"] = None
            summary["phase2_topology_signal_summary"] = None
        (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
