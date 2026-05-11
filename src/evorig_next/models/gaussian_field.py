from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from evorig_next.utils.geometry import EPS, farthest_point_sampling, knn_indices, mesh_radius, pca_frame
from evorig_next.utils.mesh_ops import points_inside_or_on_mesh
from evorig_next.utils.rotations import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_between_vectors,
)


@dataclass
class SeedStats:
    centers: torch.Tensor
    principal_dirs: torch.Tensor
    log_scales: torch.Tensor


def _resolve_reference_length(rest_vertices: torch.Tensor) -> float:
    return max(float(mesh_radius(rest_vertices).item()), 1.0e-8)


def _resolve_length_floor(
    init_cfg: dict[str, Any],
    *,
    reference_length: float,
    ratio_key: str,
    fallback_ratio: float,
) -> float:
    if ratio_key in init_cfg:
        return max(reference_length * float(init_cfg[ratio_key]), 1.0e-8)
    return max(reference_length * fallback_ratio, 1.0e-8)


def _resolve_seed_log_opacity(init_cfg: dict[str, Any]) -> float:
    return float(init_cfg.get("seed_log_opacity", -2.0))


class GaussianSupportField(nn.Module):
    def __init__(
        self,
        anchor_bone: torch.Tensor,
        lambda_param: torch.Tensor,
        offset_local: torch.Tensor,
        rot_local: torch.Tensor,
        log_scale: torch.Tensor,
        log_alpha: torch.Tensor,
        q_logits: torch.Tensor,
        endpoint_logits: torch.Tensor | None = None,
        lambda_min: torch.Tensor | None = None,
        lambda_max: torch.Tensor | None = None,
        generation: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
        log_opacity: torch.Tensor | None = None,
        log_value: torch.Tensor | None = None,
        sh_coeffs: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if lambda_min is None:
            lambda_min = torch.zeros_like(lambda_param)
        if lambda_max is None:
            lambda_max = torch.ones_like(lambda_param)
        if log_value is None:
            log_value = log_alpha
        if log_opacity is None:
            log_opacity = torch.zeros_like(log_value)
        if endpoint_logits is None:
            endpoint_logits = self._initialize_endpoint_logits_from_lambda(lambda_param)
        if sh_coeffs is None:
            sh_coeffs = torch.ones((anchor_bone.shape[0], 1), dtype=log_value.dtype, device=log_value.device)
        rot_local = self._coerce_rotation_parameter(rot_local)
        self.register_buffer("anchor_bone", anchor_bone.long().clone())
        self.lambda_param = nn.Parameter(lambda_param.clone())
        self.register_buffer("lambda_min", lambda_min.detach().clone())
        self.register_buffer("lambda_max", lambda_max.detach().clone())
        self.offset_local = nn.Parameter(offset_local.clone())
        self.rot_local = nn.Parameter(rot_local.clone())
        self.log_scale = nn.Parameter(log_scale.clone())
        self.log_opacity = nn.Parameter(log_opacity.clone())
        self.log_value = nn.Parameter(log_value.clone())
        self.sh_coeffs = nn.Parameter(sh_coeffs.clone())
        self.q_logits = nn.Parameter(q_logits.clone())
        self.endpoint_logits = nn.Parameter(endpoint_logits.clone())
        gaussian_count = int(anchor_bone.shape[0])
        self.register_buffer("generation", generation.clone() if generation is not None else torch.zeros(gaussian_count, dtype=torch.long, device=anchor_bone.device))
        self.register_buffer("active_mask", active_mask.clone() if active_mask is not None else torch.ones(gaussian_count, dtype=torch.bool, device=anchor_bone.device))
        self.track_center_gradients: bool = False
        self.last_rest_centers: torch.Tensor | None = None

    @property
    def gaussian_count(self) -> int:
        return int(self.anchor_bone.shape[0])

    @property
    def log_alpha(self) -> torch.Tensor:
        return self.log_value

    def compute_gaussian_opacity(self) -> torch.Tensor:
        return torch.exp(self.log_opacity)

    def compute_gaussian_value(self) -> torch.Tensor:
        return torch.exp(self.log_value)

    def compute_gaussian_log_amplitude(self) -> torch.Tensor:
        return self.log_opacity + self.log_value

    def compute_gaussian_amplitude(self) -> torch.Tensor:
        return torch.exp(self.compute_gaussian_log_amplitude())

    @property
    def sh_coeff_count(self) -> int:
        return int(self.sh_coeffs.shape[1])

    def ensure_sh_coeffs(self, coeff_count: int, *, dc_value: float = 1.0) -> bool:
        coeff_count = max(int(coeff_count), 1)
        current = int(self.sh_coeffs.shape[1])
        if current >= coeff_count:
            return False
        expanded = torch.zeros(
            (self.gaussian_count, coeff_count),
            dtype=self.sh_coeffs.dtype,
            device=self.sh_coeffs.device,
        )
        expanded[:, :current] = self.sh_coeffs.detach()
        if current == 0:
            expanded[:, 0] = float(dc_value)
        elif current == 1:
            expanded[:, 0] = self.sh_coeffs.detach()[:, 0]
        self._replace_parameter("sh_coeffs", expanded)
        return True

    @staticmethod
    def _initialize_endpoint_logits_from_lambda(lambda_param: torch.Tensor) -> torch.Tensor:
        lambda_clamped = lambda_param.clamp(0.0, 1.0)
        parent_weight = (1.0 - lambda_clamped).clamp_min(EPS)
        child_weight = lambda_clamped.clamp_min(EPS)
        return torch.stack([torch.log(parent_weight), torch.log(child_weight)], dim=-1)

    @staticmethod
    def _coerce_rotation_parameter(rotation: torch.Tensor) -> torch.Tensor:
        if rotation.ndim != 2:
            raise ValueError(f"rotation parameter must have shape [G, C], got {tuple(rotation.shape)}")
        if rotation.shape[-1] == 4:
            return rotation
        if rotation.shape[-1] == 3:
            return matrix_to_quaternion(axis_angle_to_matrix(rotation))
        raise ValueError(f"unsupported rotation parameter width {rotation.shape[-1]}; expected 3 or 4")

    @classmethod
    def initialize_from_center_seeds(
        cls,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        config: dict[str, Any],
        faces: torch.Tensor | None = None,
    ) -> "GaussianSupportField":
        init_cfg = config["init"]
        seed_ratio = float(init_cfg.get("seed_vertex_ratio", 0.0))
        ratio_count = int(torch.ceil(torch.tensor(rest_vertices.shape[0] * seed_ratio)).item()) if seed_ratio > 0.0 else 0
        seed_count = min(max(int(init_cfg["seed_count"]), ratio_count), int(rest_vertices.shape[0]))
        layout_mode = str(init_cfg.get("layout_mode", "centerline_uniform"))
        if layout_mode == "mesh_medial_cloud":
            seed_stats = cls._estimate_medial_seed_stats(
                rest_vertices=rest_vertices,
                seed_count=seed_count,
                init_cfg=init_cfg,
            )
            anchor_bone, lambda_param, lambda_min, lambda_max, offset_local, rot_local = cls.attach_seeds_to_skeleton(
                seed_stats.centers,
                seed_stats.principal_dirs,
                skeleton,
                rest_vertices=rest_vertices,
                faces=faces,
                init_cfg=init_cfg,
            )
            log_scale = seed_stats.log_scales
        elif layout_mode == "centerline_uniform":
            anchor_bone, lambda_param, lambda_min, lambda_max, offset_local, rot_local, log_scale = cls._centerline_uniform_seed_layout(
                rest_vertices=rest_vertices,
                skeleton=skeleton,
                seed_count=seed_count,
                init_cfg=init_cfg,
            )
        else:
            raise ValueError(
                f"unsupported init.layout_mode '{layout_mode}'; supported modes are "
                "'centerline_uniform' and 'mesh_medial_cloud'"
            )
        logits_mode = str(config.get("support", {}).get("initial_joint_logits_mode", "bone_parent"))
        endpoint_split_lambda = float(config.get("support", {}).get("endpoint_split_lambda", 0.6))
        q_logits = cls._initialize_joint_logits(
            anchor_bone,
            skeleton,
            lambda_param=lambda_param,
            mode=logits_mode,
            endpoint_split_lambda=endpoint_split_lambda,
        )
        seed_log_opacity = _resolve_seed_log_opacity(init_cfg)
        field = cls(
            anchor_bone=anchor_bone,
            lambda_param=lambda_param,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            offset_local=offset_local,
            rot_local=rot_local,
            log_scale=log_scale,
            log_alpha=torch.zeros(anchor_bone.shape[0], dtype=rest_vertices.dtype, device=rest_vertices.device),
            q_logits=q_logits,
            endpoint_logits=cls._initialize_endpoint_logits_from_lambda(lambda_param),
            log_opacity=torch.full(
                (anchor_bone.shape[0],),
                seed_log_opacity,
                dtype=rest_vertices.dtype,
                device=rest_vertices.device,
            ),
        )
        field.repair_coverage(rest_vertices, skeleton, config, faces=faces)
        return field

    @staticmethod
    def _segment_stays_inside_mesh(
        start: torch.Tensor,
        end: torch.Tensor,
        rest_vertices: torch.Tensor,
        faces: torch.Tensor | None,
        sample_count: int,
        surface_tol: float,
    ) -> bool:
        if faces is None or faces.numel() == 0 or sample_count <= 0:
            return True
        ts = torch.linspace(0.0, 1.0, sample_count + 2, dtype=start.dtype, device=start.device)[1:-1]
        probe = start.unsqueeze(0) * (1.0 - ts.unsqueeze(-1)) + end.unsqueeze(0) * ts.unsqueeze(-1)
        return bool(points_inside_or_on_mesh(probe, rest_vertices, faces, surface_tol=surface_tol).all().item())

    @classmethod
    def _assign_centers_to_bones_by_internal_distance(
        cls,
        centers: torch.Tensor,
        skeleton: Any,
        rest_vertices: torch.Tensor | None,
        faces: torch.Tensor | None,
        init_cfg: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if (
            rest_vertices is None
            or faces is None
            or faces.numel() == 0
            or centers.numel() == 0
            or skeleton.bone_count == 0
        ):
            return None
        parent_pos, _, _, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        sample_count = max(int((init_cfg or {}).get("bone_assignment_samples_per_bone", 5)), 2)
        lambdas = torch.linspace(0.0, 1.0, sample_count, dtype=centers.dtype, device=centers.device)
        bone_points = parent_pos[:, None, :] * (1.0 - lambdas[None, :, None]) + child_pos[:, None, :] * lambdas[None, :, None]
        bone_points = bone_points.reshape(-1, 3)
        bone_ids = torch.arange(skeleton.bone_count, device=centers.device, dtype=torch.long).repeat_interleave(sample_count)
        nodes = torch.cat([centers, bone_points], dim=0)
        node_count = int(nodes.shape[0])
        if node_count <= 1:
            return None
        knn = min(max(int((init_cfg or {}).get("bone_assignment_internal_knn", 8)), 1), node_count - 1)
        edge_samples = int((init_cfg or {}).get("bone_assignment_edge_samples", 6))
        surface_tol = float((init_cfg or {}).get("bone_assignment_surface_tol", 3.0e-3))
        pairwise = torch.cdist(nodes, nodes)
        adjacency: list[dict[int, float]] = [dict() for _ in range(node_count)]
        for left_idx in range(node_count):
            order = torch.argsort(pairwise[left_idx], descending=False)
            added = 0
            for right_idx in order.tolist():
                if left_idx == right_idx:
                    continue
                if not cls._segment_stays_inside_mesh(
                    nodes[left_idx],
                    nodes[right_idx],
                    rest_vertices,
                    faces,
                    sample_count=edge_samples,
                    surface_tol=surface_tol,
                ):
                    continue
                weight = float(pairwise[left_idx, right_idx].item())
                previous = adjacency[left_idx].get(right_idx)
                if previous is None or weight < previous:
                    adjacency[left_idx][right_idx] = weight
                    adjacency[right_idx][left_idx] = weight
                added += 1
                if added >= knn:
                    break
        bone_node_offset = int(centers.shape[0])
        assignments = torch.full((centers.shape[0],), -1, dtype=torch.long, device=centers.device)
        import heapq

        for seed_idx in range(int(centers.shape[0])):
            distances = [float("inf")] * node_count
            distances[seed_idx] = 0.0
            heap: list[tuple[float, int]] = [(0.0, seed_idx)]
            best_bone = -1
            best_dist = float("inf")
            while heap:
                current_dist, node_idx = heapq.heappop(heap)
                if current_dist > distances[node_idx]:
                    continue
                if node_idx >= bone_node_offset:
                    best_bone = int(bone_ids[node_idx - bone_node_offset].item())
                    best_dist = current_dist
                    break
                for neighbor_idx, edge_weight in adjacency[node_idx].items():
                    new_dist = current_dist + edge_weight
                    if new_dist < distances[neighbor_idx]:
                        distances[neighbor_idx] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor_idx))
            if best_bone >= 0 and best_dist < float("inf"):
                assignments[seed_idx] = best_bone
        if bool((assignments >= 0).all().item()):
            return assignments
        return None

    @classmethod
    def _estimate_medial_seed_stats(
        cls,
        rest_vertices: torch.Tensor,
        seed_count: int,
        init_cfg: dict[str, Any],
        seed_vertex_ids: torch.Tensor | None = None,
    ) -> SeedStats:
        reference_length = _resolve_reference_length(rest_vertices)
        if seed_vertex_ids is None:
            oversample = max(int(init_cfg.get("medial_seed_oversample", 3)), 1)
            candidate_surface_count = min(int(rest_vertices.shape[0]), max(seed_count * oversample, seed_count))
            seed_vertex_ids = farthest_point_sampling(rest_vertices, candidate_surface_count)
        medial_knn_k = int(init_cfg.get("medial_knn_k", max(int(init_cfg["knn_k"]), 64)))
        axial_band_scale = float(init_cfg.get("medial_axial_band_scale", 1.0))
        axial_band_min = float(init_cfg.get("medial_axial_band_min", 0.015))
        partner_topk = max(int(init_cfg.get("medial_partner_topk", 4)), 1)
        radial_scale_factor = float(init_cfg.get("medial_radial_scale_factor", init_cfg.get("centerline_radial_scale_factor", 0.65)))
        axial_scale_factor = float(init_cfg.get("medial_axial_scale_factor", init_cfg.get("centerline_axial_scale_factor", 0.8)))
        ratio_min = float(init_cfg.get("longitudinal_scale_min_ratio", 1.0))
        min_radial_scale = _resolve_length_floor(
            init_cfg,
            reference_length=reference_length,
            ratio_key="medial_min_radial_scale_ratio",
            legacy_key="medial_min_radial_scale",
            fallback_ratio=float(init_cfg.get("centerline_min_radial_scale_ratio", 0.015)),
        )
        log_scale_min = float(init_cfg["log_scale_min"])
        log_scale_max = float(init_cfg["log_scale_max"])

        centers: list[torch.Tensor] = []
        principal_dirs: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []

        for seed_id in seed_vertex_ids.tolist():
            seed = rest_vertices[seed_id]
            neighborhood = knn_indices(rest_vertices, seed.unsqueeze(0), min(medial_knn_k, int(rest_vertices.shape[0])))[0]
            patch = rest_vertices[neighborhood]
            frame, eigvals = pca_frame(patch)
            patch_center = patch.mean(dim=0)
            local_patch = torch.matmul((patch - patch_center), frame)
            seed_local = torch.matmul(seed - patch_center, frame)
            seed_radial = seed_local[1:]
            axial_band = max(float(local_patch[:, 0].std(unbiased=False).item()) * axial_band_scale, axial_band_min)
            axial_delta = (local_patch[:, 0] - seed_local[0]).abs()
            radial_vectors = local_patch[:, 1:]
            radial_norm = radial_vectors.norm(dim=-1)
            valid_mask = axial_delta <= axial_band
            valid_mask = valid_mask & (neighborhood != seed_id)
            if int(valid_mask.sum().item()) == 0:
                valid_mask = neighborhood != seed_id
            candidate_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
            if int(candidate_ids.numel()) == 0:
                candidate_ids = torch.arange(patch.shape[0], device=patch.device)
            candidate_radial = radial_vectors[candidate_ids]
            candidate_norm = radial_norm[candidate_ids].clamp_min(EPS)
            candidate_axial = axial_delta[candidate_ids]
            if float(seed_radial.norm().item()) > EPS:
                seed_dir = seed_radial / seed_radial.norm().clamp_min(EPS)
                opposite_score = -(candidate_radial @ seed_dir) / candidate_norm
            else:
                opposite_score = candidate_norm / candidate_norm.max().clamp_min(EPS)
            score = opposite_score + candidate_norm / candidate_norm.max().clamp_min(EPS) - candidate_axial / max(axial_band, float(EPS))
            keep_k = min(partner_topk, int(candidate_ids.numel()))
            top_idx = torch.topk(score, k=keep_k, largest=True).indices
            partner_points = patch[candidate_ids[top_idx]]
            partner_scores = score[top_idx]
            partner_weight = torch.softmax(partner_scores, dim=0)
            center = (partner_weight.unsqueeze(-1) * (0.5 * (seed.unsqueeze(0) + partner_points))).sum(dim=0)
            partner_distance = (partner_points - seed.unsqueeze(0)).norm(dim=-1)
            radial_radius = torch.maximum(0.5 * partner_distance.mean(), torch.tensor(min_radial_scale, dtype=rest_vertices.dtype, device=rest_vertices.device))
            axial_std = local_patch[:, 0].std(unbiased=False).clamp_min(EPS)
            axial_scale = max(float(axial_std.item()) * axial_scale_factor, float(radial_radius.item()) * ratio_min)
            radial_scale = max(float(radial_radius.item()) * radial_scale_factor, min_radial_scale)
            centers.append(center)
            principal_dirs.append(frame[:, 0])
            log_scales.append(
                torch.log(
                    torch.tensor(
                        [axial_scale, radial_scale, radial_scale],
                        dtype=rest_vertices.dtype,
                        device=rest_vertices.device,
                    ).clamp_min(EPS)
                ).clamp(log_scale_min, log_scale_max)
            )

        stacked_centers = torch.stack(centers, dim=0)
        stacked_dirs = torch.stack(principal_dirs, dim=0)
        stacked_scales = torch.stack(log_scales, dim=0)
        if int(stacked_centers.shape[0]) > seed_count:
            keep = farthest_point_sampling(stacked_centers, seed_count)
            stacked_centers = stacked_centers[keep]
            stacked_dirs = stacked_dirs[keep]
            stacked_scales = stacked_scales[keep]
        return SeedStats(
            centers=stacked_centers,
            principal_dirs=stacked_dirs,
            log_scales=stacked_scales,
        )

    @staticmethod
    def _assign_vertices_to_bones(rest_vertices: torch.Tensor, skeleton: Any) -> dict[str, torch.Tensor]:
        parent_pos, bone_frames, _, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        segment = child_pos - parent_pos
        segment_len_sq = segment.square().sum(dim=-1).clamp_min(EPS)
        rel = rest_vertices.unsqueeze(0) - parent_pos.unsqueeze(1)
        lambda_raw = (rel * segment.unsqueeze(1)).sum(dim=-1) / segment_len_sq.unsqueeze(1)
        lambda_clamped = lambda_raw.clamp(0.0, 1.0)
        projected = parent_pos.unsqueeze(1) + lambda_clamped.unsqueeze(-1) * segment.unsqueeze(1)
        distances = (rest_vertices.unsqueeze(0) - projected).norm(dim=-1)
        best_bone = distances.argmin(dim=0)
        vertex_ids = torch.arange(rest_vertices.shape[0], device=rest_vertices.device)
        best_lambda = lambda_clamped[best_bone, vertex_ids]
        best_projected = projected[best_bone, vertex_ids]
        local_frame = bone_frames[best_bone]
        local_offset = torch.matmul(local_frame.transpose(-1, -2), (rest_vertices - best_projected).unsqueeze(-1)).squeeze(-1)
        radial_distance = local_offset[:, 1:].norm(dim=-1)
        return {
            "bone_index": best_bone,
            "lambda_value": best_lambda,
            "projected": best_projected,
            "local_offset": local_offset,
            "radial_distance": radial_distance,
        }

    @staticmethod
    def _compute_bone_lambda_bounds(
        skeleton: Any,
        init_cfg: dict[str, Any],
        reference_length: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if skeleton.bone_count == 0:
            empty = torch.zeros(0, dtype=skeleton.rest_joints.dtype, device=skeleton.rest_joints.device)
            return empty, empty
        parent_pos, _, bone_parent_idx, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        bone_lengths = (child_pos - parent_pos).norm(dim=-1).clamp_min(EPS)
        child_count = torch.bincount(
            skeleton.parent_idx[skeleton.parent_idx >= 0],
            minlength=skeleton.joint_count,
        )
        root_mask = skeleton.parent_idx < 0
        root_endpoint = root_mask & (child_count <= 1)
        leaf_mask = child_count == 0
        if reference_length is None:
            reference_length = max(float(mesh_radius(skeleton.rest_joints).item()), 1.0e-8)
        extension_ratio = float(init_cfg.get("endpoint_extension_ratio", 0.35))
        extension_min = _resolve_length_floor(
            init_cfg,
            reference_length=reference_length,
            ratio_key="endpoint_extension_min_ratio",
            fallback_ratio=0.08,
        )
        extension_max = float(init_cfg.get("endpoint_extension_max_ratio", 0.75))
        extension_abs = torch.maximum(
            bone_lengths * extension_ratio,
            torch.full_like(bone_lengths, extension_min),
        )
        extension_lambda = (extension_abs / bone_lengths).clamp_max(extension_max)
        start_extension = torch.where(root_endpoint[bone_parent_idx], extension_lambda, torch.zeros_like(extension_lambda))
        end_extension = torch.where(leaf_mask[bone_child_idx], extension_lambda, torch.zeros_like(extension_lambda))
        lambda_min = -start_extension
        lambda_max = 1.0 + end_extension
        global_lambda_min = init_cfg.get("global_lambda_min")
        global_lambda_max = init_cfg.get("global_lambda_max")
        if global_lambda_min is not None:
            lambda_min = torch.minimum(
                lambda_min,
                torch.full_like(lambda_min, float(global_lambda_min)),
            )
        if global_lambda_max is not None:
            lambda_max = torch.maximum(
                lambda_max,
                torch.full_like(lambda_max, float(global_lambda_max)),
            )
        return lambda_min, lambda_max

    @staticmethod
    def _allocate_centerline_seed_counts(
        bone_lengths: torch.Tensor,
        assigned_bone: torch.Tensor,
        seed_count: int,
        init_cfg: dict[str, Any],
    ) -> torch.Tensor:
        bone_count = int(bone_lengths.shape[0])
        min_per_bone = int(init_cfg.get("centerline_min_seeds_per_bone", 1))
        target_total = max(seed_count, bone_count * min_per_bone)
        if bone_count == 1:
            return torch.tensor([target_total], dtype=torch.long, device=bone_lengths.device)
        length_weight = float(init_cfg.get("centerline_length_weight", 0.6))
        support_weight = float(init_cfg.get("centerline_support_weight", 0.4))
        bone_vertex_count = torch.bincount(assigned_bone, minlength=bone_count).to(bone_lengths.dtype)
        normalized_lengths = bone_lengths / bone_lengths.sum().clamp_min(EPS)
        normalized_support = bone_vertex_count / bone_vertex_count.sum().clamp_min(EPS)
        combined = length_weight * normalized_lengths + support_weight * normalized_support
        combined = combined / combined.sum().clamp_min(EPS)
        extra = max(target_total - bone_count * min_per_bone, 0)
        counts = torch.full((bone_count,), min_per_bone, dtype=torch.long, device=bone_lengths.device)
        if extra <= 0:
            return counts
        float_extra = combined * extra
        counts = counts + torch.floor(float_extra).long()
        remainder = extra - int((counts.sum() - bone_count * min_per_bone).item())
        if remainder > 0:
            order = torch.argsort(float_extra - torch.floor(float_extra), descending=True)
            counts[order[:remainder]] += 1
        return counts

    @classmethod
    def _centerline_uniform_seed_layout(
        cls,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        seed_count: int,
        init_cfg: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if skeleton.bone_count == 0:
            raise ValueError("centerline_uniform initialization requires at least one bone")
        reference_length = _resolve_reference_length(rest_vertices)
        assignment = cls._assign_vertices_to_bones(rest_vertices, skeleton)
        parent_pos, bone_frames, _, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        bone_lengths = (child_pos - parent_pos).norm(dim=-1).clamp_min(EPS)
        bone_lambda_min, bone_lambda_max = cls._compute_bone_lambda_bounds(
            skeleton,
            init_cfg,
            reference_length=reference_length,
        )
        counts = cls._allocate_centerline_seed_counts(
            bone_lengths=bone_lengths,
            assigned_bone=assignment["bone_index"],
            seed_count=seed_count,
            init_cfg=init_cfg,
        )
        patch_min_vertices = int(init_cfg.get("centerline_patch_min_vertices", max(8, int(init_cfg["knn_k"]) // 4)))
        window_scale = float(init_cfg.get("centerline_lambda_window_scale", 1.25))
        axial_scale_factor = float(init_cfg.get("centerline_axial_scale_factor", 1.25))
        radial_scale_factor = float(init_cfg.get("centerline_radial_scale_factor", 0.5))
        min_radial_scale = _resolve_length_floor(
            init_cfg,
            reference_length=reference_length,
            ratio_key="centerline_min_radial_scale_ratio",
            fallback_ratio=0.015,
        )
        ratio_min = float(init_cfg.get("longitudinal_scale_min_ratio", 2.5))
        log_scale_min = float(init_cfg["log_scale_min"])
        log_scale_max = float(init_cfg["log_scale_max"])
        local_knn = int(init_cfg.get("centerline_knn_k", init_cfg["knn_k"]))
        anchor_bones: list[torch.Tensor] = []
        lambda_values: list[torch.Tensor] = []
        lambda_mins: list[torch.Tensor] = []
        lambda_maxs: list[torch.Tensor] = []
        offset_local: list[torch.Tensor] = []
        rot_local: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []

        for bone_index in range(skeleton.bone_count):
            count = int(counts[bone_index].item())
            if count <= 0:
                continue
            lower = float(bone_lambda_min[bone_index].item())
            upper = float(bone_lambda_max[bone_index].item())
            if count == 1:
                lambdas = torch.tensor([(lower + upper) * 0.5], dtype=rest_vertices.dtype, device=rest_vertices.device)
            else:
                lambdas = torch.linspace(
                    lower,
                    upper,
                    count,
                    dtype=rest_vertices.dtype,
                    device=rest_vertices.device,
                )
            bone_vertex_ids = torch.nonzero(assignment["bone_index"] == bone_index, as_tuple=False).flatten()
            bone_vertex_lambdas = assignment["lambda_value"][bone_vertex_ids] if bone_vertex_ids.numel() > 0 else None
            frame = bone_frames[bone_index]
            segment = child_pos[bone_index] - parent_pos[bone_index]
            lambda_window = max((upper - lower) * window_scale / max(count + 1, 2), 0.05)
            axial_spacing = float(bone_lengths[bone_index].item()) * (upper - lower) / max(count + 1, 1)
            for lambda_value in lambdas:
                projected_point = parent_pos[bone_index] + lambda_value * segment
                patch_ids = bone_vertex_ids
                if bone_vertex_ids.numel() > 0 and bone_vertex_lambdas is not None:
                    local_mask = (bone_vertex_lambdas - lambda_value.clamp(0.0, 1.0)).abs() <= lambda_window
                    candidate_ids = bone_vertex_ids[local_mask]
                    if candidate_ids.numel() >= patch_min_vertices:
                        patch_ids = candidate_ids
                if patch_ids.numel() < patch_min_vertices:
                    if bone_vertex_ids.numel() >= patch_min_vertices:
                        local_patch = rest_vertices[bone_vertex_ids]
                        nearest = knn_indices(local_patch, projected_point.unsqueeze(0), min(local_knn, local_patch.shape[0]))[0]
                        patch = local_patch[nearest]
                    else:
                        nearest = knn_indices(rest_vertices, projected_point.unsqueeze(0), min(local_knn, rest_vertices.shape[0]))[0]
                        patch = rest_vertices[nearest]
                else:
                    patch = rest_vertices[patch_ids]
                local_patch = torch.matmul(frame.transpose(0, 1), (patch - projected_point).T).T
                radial_distance = local_patch[:, 1:].norm(dim=-1)
                radial_radius = torch.maximum(radial_distance.median(), torch.tensor(min_radial_scale, dtype=rest_vertices.dtype, device=rest_vertices.device))
                axial_scale = max(axial_spacing * axial_scale_factor, float(radial_radius.item()) * ratio_min)
                radial_scale = max(float(radial_radius.item()) * radial_scale_factor, min_radial_scale)
                anchor_bones.append(torch.tensor(bone_index, dtype=torch.long, device=rest_vertices.device))
                lambda_values.append(lambda_value)
                lambda_mins.append(torch.tensor(lower, dtype=rest_vertices.dtype, device=rest_vertices.device))
                lambda_maxs.append(torch.tensor(upper, dtype=rest_vertices.dtype, device=rest_vertices.device))
                offset_local.append(torch.zeros(3, dtype=rest_vertices.dtype, device=rest_vertices.device))
                rot_local.append(matrix_to_quaternion(frame.unsqueeze(0))[0])
                log_scales.append(
                    torch.log(
                        torch.tensor(
                            [axial_scale, radial_scale, radial_scale],
                            dtype=rest_vertices.dtype,
                            device=rest_vertices.device,
                        ).clamp_min(EPS)
                    ).clamp(log_scale_min, log_scale_max)
                )
        return (
            torch.stack(anchor_bones, dim=0),
            torch.stack(lambda_values, dim=0),
            torch.stack(lambda_mins, dim=0),
            torch.stack(lambda_maxs, dim=0),
            torch.stack(offset_local, dim=0),
            torch.stack(rot_local, dim=0),
            torch.stack(log_scales, dim=0),
        )

    def prune_gaussians(
        self,
        support_mass: torch.Tensor,
        residual: torch.Tensor,
        config: dict[str, Any],
        step: int = 0,
    ) -> torch.Tensor:
        support_cfg = config["support"]
        progress_cfg = config.get("training", {}).get("progress_schedule", {})
        total_steps = max(int(config.get("training", {}).get("steps", 1)), 1)
        if "prune_start_progress" in support_cfg:
            prune_after_step = int(round(float(support_cfg["prune_start_progress"]) * total_steps))
        else:
            prune_after_step = int(round(float(progress_cfg["prune_start_progress"]) * total_steps))
        prune_support_threshold = float(support_cfg.get("prune_support_mass_threshold", 0.0))
        prune_residual_threshold = float(support_cfg.get("prune_residual_threshold", 0.0))
        prune_max_per_update = int(support_cfg.get("prune_max_per_update", 0))
        min_per_bone = int(support_cfg.get("prune_min_gaussians_per_bone", 0))
        if (
            step < prune_after_step
            or prune_support_threshold <= 0.0
            or prune_max_per_update <= 0
            or self.gaussian_count == 0
        ):
            return torch.empty(0, dtype=torch.long, device=self.anchor_bone.device)
        candidate_mask = (
            self.active_mask
            & (support_mass <= prune_support_threshold)
            & (residual <= prune_residual_threshold)
            & (self.generation > 0)
        )
        candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_ids.numel() == 0:
            return candidate_ids
        active_per_bone = torch.bincount(self.anchor_bone[self.active_mask], minlength=int(self.anchor_bone.max().item()) + 1)
        ordered = candidate_ids[torch.argsort(support_mass[candidate_ids], descending=False)]
        pruned: list[int] = []
        for gaussian_id in ordered.tolist():
            bone_index = int(self.anchor_bone[gaussian_id].item())
            if min_per_bone > 0 and int(active_per_bone[bone_index].item()) <= min_per_bone:
                continue
            self.active_mask[gaussian_id] = False
            active_per_bone[bone_index] -= 1
            pruned.append(gaussian_id)
            if len(pruned) >= prune_max_per_update:
                break
        if not pruned:
            return torch.empty(0, dtype=torch.long, device=self.anchor_bone.device)
        return torch.tensor(pruned, dtype=torch.long, device=self.anchor_bone.device)

    @staticmethod
    def _bone_parent_joint_ids(
        anchor_bone: torch.Tensor,
        skeleton: Any,
    ) -> torch.Tensor:
        child_joints = skeleton.bone_child_idx[anchor_bone]
        parent_joints = skeleton.parent_idx[child_joints]
        target_joints = parent_joints.clone()
        invalid_parent = target_joints < 0
        target_joints[invalid_parent] = child_joints[invalid_parent]
        return target_joints

    @classmethod
    def _bone_parent_joint_assignment(
        cls,
        anchor_bone: torch.Tensor,
        skeleton: Any,
    ) -> torch.Tensor:
        gaussian_count = int(anchor_bone.shape[0])
        assignment = torch.zeros(
            (gaussian_count, skeleton.joint_count),
            dtype=skeleton.rest_joints.dtype,
            device=skeleton.rest_joints.device,
        )
        if gaussian_count == 0:
            return assignment
        target_joints = cls._bone_parent_joint_ids(anchor_bone, skeleton)
        gaussian_ids = torch.arange(gaussian_count, device=assignment.device)
        assignment[gaussian_ids, target_joints] = 1.0
        return assignment

    @classmethod
    def _bone_endpoint_joint_assignment(
        cls,
        anchor_bone: torch.Tensor,
        skeleton: Any,
        lambda_param: torch.Tensor,
    ) -> torch.Tensor:
        gaussian_count = int(anchor_bone.shape[0])
        assignment = torch.zeros(
            (gaussian_count, skeleton.joint_count),
            dtype=skeleton.rest_joints.dtype,
            device=skeleton.rest_joints.device,
        )
        if gaussian_count == 0:
            return assignment
        child_joints = skeleton.bone_child_idx[anchor_bone]
        parent_joints = skeleton.parent_idx[child_joints]
        gaussian_ids = torch.arange(gaussian_count, device=assignment.device)
        lambda_clamped = lambda_param.clamp(0.0, 1.0)
        assignment[gaussian_ids, child_joints] = 1.0
        valid_parent = parent_joints >= 0
        if bool(valid_parent.any().item()):
            valid_ids = gaussian_ids[valid_parent]
            valid_child = child_joints[valid_parent]
            valid_parent_joints = parent_joints[valid_parent]
            valid_lambda = lambda_clamped[valid_parent]
            assignment[valid_ids, valid_parent_joints] = 1.0 - valid_lambda
            assignment[valid_ids, valid_child] = valid_lambda
        return assignment

    @classmethod
    def _bone_endpoint_cut_joint_assignment(
        cls,
        anchor_bone: torch.Tensor,
        skeleton: Any,
        lambda_param: torch.Tensor,
        *,
        split_lambda: float = 0.6,
    ) -> torch.Tensor:
        gaussian_count = int(anchor_bone.shape[0])
        assignment = torch.zeros(
            (gaussian_count, skeleton.joint_count),
            dtype=skeleton.rest_joints.dtype,
            device=skeleton.rest_joints.device,
        )
        if gaussian_count == 0:
            return assignment
        target_joints = cls._bone_endpoint_cut_joint_ids(
            anchor_bone,
            skeleton,
            lambda_param,
            split_lambda=split_lambda,
        )
        gaussian_ids = torch.arange(gaussian_count, device=assignment.device)
        assignment[gaussian_ids, target_joints] = 1.0
        return assignment

    @classmethod
    def _bone_endpoint_matrix_joint_assignment(
        cls,
        anchor_bone: torch.Tensor,
        skeleton: Any,
        endpoint_logits: torch.Tensor,
    ) -> torch.Tensor:
        gaussian_count = int(anchor_bone.shape[0])
        assignment = torch.zeros(
            (gaussian_count, skeleton.joint_count),
            dtype=skeleton.rest_joints.dtype,
            device=skeleton.rest_joints.device,
        )
        if gaussian_count == 0:
            return assignment
        child_joints = skeleton.bone_child_idx[anchor_bone]
        parent_joints = skeleton.parent_idx[child_joints]
        gaussian_ids = torch.arange(gaussian_count, device=assignment.device)
        endpoint_weights = torch.softmax(endpoint_logits, dim=-1)
        assignment[gaussian_ids, child_joints] = 1.0
        valid_parent = parent_joints >= 0
        if bool(valid_parent.any().item()):
            valid_ids = gaussian_ids[valid_parent]
            valid_child = child_joints[valid_parent]
            valid_parent_joints = parent_joints[valid_parent]
            valid_weights = endpoint_weights[valid_parent]
            assignment[valid_ids, valid_parent_joints] = valid_weights[:, 0]
            assignment[valid_ids, valid_child] = valid_weights[:, 1]
        return assignment

    @classmethod
    def _initialize_joint_logits(
        cls,
        anchor_bone: torch.Tensor,
        skeleton: Any,
        lambda_param: torch.Tensor | None = None,
        *,
        mode: str = "bone_parent",
        endpoint_split_lambda: float = 0.6,
    ) -> torch.Tensor:
        gaussian_count = int(anchor_bone.shape[0])
        logits = torch.full(
            (gaussian_count, skeleton.joint_count),
            -8.0,
            dtype=skeleton.rest_joints.dtype,
            device=skeleton.rest_joints.device,
        )
        if gaussian_count == 0:
            return logits
        if lambda_param is None:
            lambda_param = torch.full(
                (gaussian_count,),
                0.5,
                dtype=skeleton.rest_joints.dtype,
                device=skeleton.rest_joints.device,
            )
        if str(mode) == "bone_parent":
            target_joints = cls._bone_parent_joint_ids(anchor_bone, skeleton)
            gaussian_ids = torch.arange(gaussian_count, device=logits.device)
            logits[gaussian_ids, target_joints] = 0.0
            return logits
        if str(mode) == "bone_endpoint_cut":
            target_joints = cls._bone_endpoint_cut_joint_ids(
                anchor_bone,
                skeleton,
                lambda_param,
                split_lambda=float(endpoint_split_lambda),
            )
            gaussian_ids = torch.arange(gaussian_count, device=logits.device)
            logits[gaussian_ids, target_joints] = 0.0
            return logits
        if str(mode) not in {"bone_endpoint_blend", "bone_endpoint_matrix"}:
            raise ValueError(f"unsupported joint logits init mode '{mode}'")
        endpoint_assignment = cls._bone_endpoint_joint_assignment(anchor_bone, skeleton, lambda_param)
        valid = endpoint_assignment > 0.0
        logits[valid] = torch.log(endpoint_assignment[valid].clamp_min(EPS))
        return logits

    @staticmethod
    def _bone_endpoint_joint_ids(
        anchor_bone: torch.Tensor,
        skeleton: Any,
        lambda_param: torch.Tensor,
    ) -> torch.Tensor:
        child_joints = skeleton.bone_child_idx[anchor_bone]
        parent_joints = skeleton.parent_idx[child_joints]
        target_joints = child_joints.clone()
        valid_parent = parent_joints >= 0
        if bool(valid_parent.any().item()):
            choose_parent = valid_parent & (lambda_param.clamp(0.0, 1.0) < 0.5)
            target_joints[choose_parent] = parent_joints[choose_parent]
        return target_joints

    @staticmethod
    def _bone_endpoint_cut_joint_ids(
        anchor_bone: torch.Tensor,
        skeleton: Any,
        lambda_param: torch.Tensor,
        *,
        split_lambda: float = 0.6,
    ) -> torch.Tensor:
        child_joints = skeleton.bone_child_idx[anchor_bone]
        parent_joints = skeleton.parent_idx[child_joints]
        target_joints = child_joints.clone()
        valid_parent = parent_joints >= 0
        if bool(valid_parent.any().item()):
            choose_parent = valid_parent & (lambda_param <= float(split_lambda))
            target_joints[choose_parent] = parent_joints[choose_parent]
        return target_joints

    @staticmethod
    def _bone_endpoint_matrix_joint_ids(
        anchor_bone: torch.Tensor,
        skeleton: Any,
        endpoint_logits: torch.Tensor,
    ) -> torch.Tensor:
        child_joints = skeleton.bone_child_idx[anchor_bone]
        parent_joints = skeleton.parent_idx[child_joints]
        target_joints = child_joints.clone()
        valid_parent = parent_joints >= 0
        if bool(valid_parent.any().item()):
            endpoint_weights = torch.softmax(endpoint_logits, dim=-1)
            choose_parent = valid_parent.clone()
            choose_parent[valid_parent] = endpoint_weights[valid_parent, 0] > endpoint_weights[valid_parent, 1]
            target_joints[choose_parent] = parent_joints[choose_parent]
        return target_joints

    @classmethod
    def _initialize_hard_joint_logits(
        cls,
        anchor_bone: torch.Tensor,
        skeleton: Any,
        lambda_param: torch.Tensor | None = None,
        *,
        mode: str = "bone_endpoint",
        endpoint_split_lambda: float = 0.6,
        positive_logit: float = 8.0,
        negative_logit: float = -8.0,
    ) -> torch.Tensor:
        gaussian_count = int(anchor_bone.shape[0])
        logits = torch.full(
            (gaussian_count, skeleton.joint_count),
            float(negative_logit),
            dtype=skeleton.rest_joints.dtype,
            device=skeleton.rest_joints.device,
        )
        if gaussian_count == 0:
            return logits
        if lambda_param is None:
            lambda_param = torch.full(
                (gaussian_count,),
                0.5,
                dtype=skeleton.rest_joints.dtype,
                device=skeleton.rest_joints.device,
            )
        if str(mode) not in {"bone_endpoint", "bone_parent", "bone_endpoint_cut"}:
            raise ValueError(f"unsupported hard ownership refresh mode '{mode}'")
        if str(mode) == "bone_endpoint_cut":
            target_joints = cls._bone_endpoint_cut_joint_ids(
                anchor_bone,
                skeleton,
                lambda_param,
                split_lambda=float(endpoint_split_lambda),
            )
        else:
            target_joints = cls._bone_endpoint_joint_ids(anchor_bone, skeleton, lambda_param)
        gaussian_ids = torch.arange(gaussian_count, device=logits.device)
        logits[gaussian_ids, target_joints] = float(positive_logit)
        return logits

    @classmethod
    def attach_seeds_to_skeleton(
        cls,
        centers: torch.Tensor,
        principal_dirs: torch.Tensor,
        skeleton: Any,
        rest_vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
        init_cfg: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_pos, bone_frames, bone_parent_idx, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        segment = child_pos - parent_pos
        segment_len_sq = segment.square().sum(dim=-1).clamp_min(EPS)
        cfg = {} if init_cfg is None else init_cfg
        if rest_vertices is not None and rest_vertices.numel() > 0:
            reference_length = _resolve_reference_length(rest_vertices)
        else:
            reference_length = max(float(mesh_radius(skeleton.rest_joints).item()), 1.0e-8)
        bone_lambda_min, bone_lambda_max = cls._compute_bone_lambda_bounds(
            skeleton,
            cfg,
            reference_length=reference_length,
        )
        anchor_bones = []
        lambda_values = []
        lambda_mins = []
        lambda_maxs = []
        offset_local = []
        rot_local = []
        x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=centers.dtype, device=centers.device)
        internal_assignments = cls._assign_centers_to_bones_by_internal_distance(
            centers=centers,
            skeleton=skeleton,
            rest_vertices=rest_vertices,
            faces=faces,
            init_cfg=init_cfg,
        )
        for seed_idx, center in enumerate(centers):
            rel = center.unsqueeze(0) - parent_pos
            lambda_raw = (rel * segment).sum(dim=-1) / segment_len_sq
            lambda_clamped = lambda_raw.clamp(0.0, 1.0)
            projected = parent_pos + lambda_clamped.unsqueeze(-1) * segment
            if internal_assignments is not None and int(internal_assignments[seed_idx].item()) >= 0:
                best = int(internal_assignments[seed_idx].item())
            else:
                dists = (projected - center).norm(dim=-1)
                best = int(dists.argmin().item())
            anchor_bones.append(best)
            lambda_values.append(lambda_raw[best].clamp(bone_lambda_min[best], bone_lambda_max[best]))
            lambda_mins.append(bone_lambda_min[best])
            lambda_maxs.append(bone_lambda_max[best])
            frame = bone_frames[best]
            offset_local.append(torch.zeros(3, dtype=centers.dtype, device=centers.device))
            local_dir = frame.transpose(0, 1) @ principal_dirs[seed_idx]
            if local_dir.norm() < EPS:
                local_dir = x_axis
            rot_matrix = rotation_between_vectors(x_axis.unsqueeze(0), local_dir.unsqueeze(0))[0]
            world_rot = frame @ rot_matrix
            rot_local.append(matrix_to_quaternion(world_rot.unsqueeze(0))[0])
        return (
            torch.tensor(anchor_bones, dtype=torch.long, device=centers.device),
            torch.stack(lambda_values, dim=0).to(device=centers.device, dtype=centers.dtype),
            torch.stack(lambda_mins, dim=0).to(device=centers.device, dtype=centers.dtype),
            torch.stack(lambda_maxs, dim=0).to(device=centers.device, dtype=centers.dtype),
            torch.stack(offset_local, dim=0),
            torch.stack(rot_local, dim=0),
        )

    def _replace_parameter(self, name: str, value: torch.Tensor) -> None:
        setattr(self, name, nn.Parameter(value))

    def append_gaussians(
        self,
        anchor_bone: torch.Tensor,
        lambda_param: torch.Tensor,
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
        offset_local: torch.Tensor,
        rot_local: torch.Tensor,
        log_scale: torch.Tensor,
        log_alpha: torch.Tensor,
        q_logits: torch.Tensor,
        endpoint_logits: torch.Tensor | None,
        generation: torch.Tensor,
        log_opacity: torch.Tensor | None = None,
        log_value: torch.Tensor | None = None,
        sh_coeffs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        start = self.gaussian_count
        if log_value is None:
            log_value = log_alpha
        if log_opacity is None:
            log_opacity = torch.zeros_like(log_value)
        if endpoint_logits is None:
            endpoint_logits = self._initialize_endpoint_logits_from_lambda(lambda_param)
        if sh_coeffs is None:
            sh_coeffs = torch.zeros(
                (anchor_bone.shape[0], self.sh_coeff_count),
                dtype=self.sh_coeffs.dtype,
                device=self.sh_coeffs.device,
            )
            sh_coeffs[:, 0] = 1.0
        self.anchor_bone = torch.cat([self.anchor_bone, anchor_bone.long().to(self.anchor_bone.device)])
        self._replace_parameter("lambda_param", torch.cat([self.lambda_param.detach(), lambda_param.to(self.lambda_param.device)], dim=0))
        self.lambda_min = torch.cat([self.lambda_min, lambda_min.detach().to(self.lambda_min.device)], dim=0)
        self.lambda_max = torch.cat([self.lambda_max, lambda_max.detach().to(self.lambda_max.device)], dim=0)
        self._replace_parameter("offset_local", torch.cat([self.offset_local.detach(), offset_local.to(self.offset_local.device)], dim=0))
        self._replace_parameter("rot_local", torch.cat([self.rot_local.detach(), rot_local.to(self.rot_local.device)], dim=0))
        self._replace_parameter("log_scale", torch.cat([self.log_scale.detach(), log_scale.to(self.log_scale.device)], dim=0))
        self._replace_parameter("log_opacity", torch.cat([self.log_opacity.detach(), log_opacity.to(self.log_opacity.device)], dim=0))
        self._replace_parameter("log_value", torch.cat([self.log_value.detach(), log_value.to(self.log_value.device)], dim=0))
        self._replace_parameter("sh_coeffs", torch.cat([self.sh_coeffs.detach(), sh_coeffs.to(self.sh_coeffs.device)], dim=0))
        self._replace_parameter("q_logits", torch.cat([self.q_logits.detach(), q_logits.to(self.q_logits.device)], dim=0))
        self._replace_parameter("endpoint_logits", torch.cat([self.endpoint_logits.detach(), endpoint_logits.to(self.endpoint_logits.device)], dim=0))
        self.generation = torch.cat([self.generation, generation.long().to(self.generation.device)], dim=0)
        active = torch.ones(anchor_bone.shape[0], dtype=torch.bool, device=self.active_mask.device)
        self.active_mask = torch.cat([self.active_mask, active], dim=0)
        return torch.arange(start, self.gaussian_count, device=self.anchor_bone.device, dtype=torch.long)

    def compute_rest_centers(self, skeleton: Any) -> torch.Tensor:
        parent_pos, _, bone_parent_idx, bone_child_idx = skeleton.compute_bone_frames()
        del bone_parent_idx
        child_pos = skeleton.rest_joints[bone_child_idx]
        lambda_value = torch.maximum(torch.minimum(self.lambda_param, self.lambda_max), self.lambda_min)
        return parent_pos[self.anchor_bone] + lambda_value.unsqueeze(-1) * (child_pos[self.anchor_bone] - parent_pos[self.anchor_bone])

    def compute_covariance(self, skeleton: Any) -> torch.Tensor:
        orientation = quaternion_to_matrix(self.rot_local)
        scales = torch.exp(self.log_scale).clamp_min(EPS)
        diag = torch.diag_embed(scales.square())
        covariance = orientation @ diag @ orientation.transpose(-1, -2)
        inactive = ~self.active_mask
        if inactive.any():
            eye = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
            covariance[inactive] = eye
        return covariance

    def compute_principal_dirs(self, skeleton: Any) -> torch.Tensor:
        orientation = quaternion_to_matrix(self.rot_local)
        return orientation[..., 0]

    def reattach_existing_gaussians(
        self,
        centers: torch.Tensor,
        principal_dirs: torch.Tensor,
        skeleton: Any,
        rest_vertices: torch.Tensor,
        config: dict[str, Any],
        faces: torch.Tensor | None = None,
    ) -> None:
        anchor_bone, lambda_param, lambda_min, lambda_max, offset_local, rot_local = self.attach_seeds_to_skeleton(
            centers=centers.detach(),
            principal_dirs=principal_dirs.detach(),
            skeleton=skeleton,
            rest_vertices=rest_vertices,
            faces=faces,
            init_cfg=config["init"],
        )
        logits_mode = str(config.get("support", {}).get("initial_joint_logits_mode", "bone_parent"))
        endpoint_split_lambda = float(config.get("support", {}).get("endpoint_split_lambda", 0.6))
        q_logits = self._initialize_joint_logits(
            anchor_bone,
            skeleton,
            lambda_param=lambda_param,
            mode=logits_mode,
            endpoint_split_lambda=endpoint_split_lambda,
        )
        endpoint_logits = self._initialize_endpoint_logits_from_lambda(lambda_param)
        self.anchor_bone = anchor_bone.long().to(self.anchor_bone.device)
        self._replace_parameter("lambda_param", lambda_param.to(self.lambda_param.device))
        self.lambda_min = lambda_min.detach().to(self.lambda_min.device)
        self.lambda_max = lambda_max.detach().to(self.lambda_max.device)
        self._replace_parameter("offset_local", offset_local.to(self.offset_local.device))
        self._replace_parameter("rot_local", rot_local.to(self.rot_local.device))
        self._replace_parameter("q_logits", q_logits.to(self.q_logits.device))
        self._replace_parameter("endpoint_logits", endpoint_logits.to(self.endpoint_logits.device))

    def compute_gaussian_density(self, rest_vertices: torch.Tensor, skeleton: Any) -> torch.Tensor:
        centers = self.compute_rest_centers(skeleton)
        if self.track_center_gradients and centers.requires_grad:
            centers.retain_grad()
            self.last_rest_centers = centers
        else:
            self.last_rest_centers = None
        cov = self.compute_covariance(skeleton)
        inv_cov = torch.linalg.inv(cov)
        diff = rest_vertices.unsqueeze(0) - centers.unsqueeze(1)
        mahal = torch.einsum("gvi,gij,gvj->gv", diff, inv_cov, diff)
        opacity = self.compute_gaussian_opacity().unsqueeze(-1)
        density = opacity * torch.exp(-0.5 * mahal)
        if self.sh_coeff_count > 0:
            diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(EPS)
            d_hat = diff / diff_norm
            sh_response = self.sh_coeffs[:, 0].unsqueeze(-1)
            if self.sh_coeff_count >= 4:
                sh_response = sh_response + (
                    self.sh_coeffs[:, 1].unsqueeze(-1) * d_hat[..., 0]
                    + self.sh_coeffs[:, 2].unsqueeze(-1) * d_hat[..., 1]
                    + self.sh_coeffs[:, 3].unsqueeze(-1) * d_hat[..., 2]
                )
            if self.sh_coeff_count >= 9:
                x = d_hat[..., 0]
                y = d_hat[..., 1]
                z = d_hat[..., 2]
                sh_response = sh_response + (
                    self.sh_coeffs[:, 4].unsqueeze(-1) * (x * y)
                    + self.sh_coeffs[:, 5].unsqueeze(-1) * (y * z)
                    + self.sh_coeffs[:, 6].unsqueeze(-1) * (2.0 * z * z - x * x - y * y)
                    + self.sh_coeffs[:, 7].unsqueeze(-1) * (x * z)
                    + self.sh_coeffs[:, 8].unsqueeze(-1) * (x * x - y * y)
                )
            density = density * torch.nn.functional.softplus(sh_response)
        return density * self.active_mask.to(density.dtype).unsqueeze(-1)

    def compute_gaussian_kernels(self, rest_vertices: torch.Tensor, skeleton: Any) -> torch.Tensor:
        return self.compute_gaussian_density(rest_vertices, skeleton)

    def compute_responsibility(self, kernels: torch.Tensor) -> torch.Tensor:
        denom = kernels.sum(dim=0, keepdim=True).clamp_min(EPS)
        return kernels / denom

    def compute_joint_assignment(self, skeleton: Any, ownership_strictness: float = 0.0) -> torch.Tensor:
        return self.compute_joint_assignment_with_mode(
            skeleton,
            ownership_strictness=ownership_strictness,
            strict_mode="bone_parent",
        )

    def compute_joint_assignment_with_mode(
        self,
        skeleton: Any,
        *,
        ownership_strictness: float = 0.0,
        strict_mode: str = "bone_parent",
        endpoint_split_lambda: float = 0.6,
    ) -> torch.Tensor:
        logits = self.q_logits
        soft_assignment = torch.softmax(logits, dim=-1)
        if str(strict_mode) == "bone_parent":
            strict_assignment = self._bone_parent_joint_assignment(
                self.anchor_bone,
                skeleton,
            )
        elif str(strict_mode) == "bone_endpoint_blend":
            strict_assignment = self._bone_endpoint_joint_assignment(
                self.anchor_bone,
                skeleton,
                self.lambda_param.detach(),
            )
        elif str(strict_mode) == "bone_endpoint_cut":
            strict_assignment = self._bone_endpoint_cut_joint_assignment(
                self.anchor_bone,
                skeleton,
                self.lambda_param.detach(),
                split_lambda=float(endpoint_split_lambda),
            )
        elif str(strict_mode) == "bone_endpoint_matrix":
            strict_assignment = self._bone_endpoint_matrix_joint_assignment(
                self.anchor_bone,
                skeleton,
                self.endpoint_logits,
            )
        else:
            raise ValueError(f"unsupported strict assignment mode '{strict_mode}'")
        ownership_strictness = float(max(0.0, min(1.0, ownership_strictness)))
        assignment = (1.0 - ownership_strictness) * soft_assignment + ownership_strictness * strict_assignment
        return assignment * self.active_mask.to(logits.dtype).unsqueeze(-1)

    def compute_hard_joint_assignment(
        self,
        skeleton: Any,
        *,
        mode: str = "q_logits",
        endpoint_split_lambda: float = 0.6,
    ) -> torch.Tensor:
        assignment = torch.zeros(
            (self.gaussian_count, skeleton.joint_count),
            dtype=self.q_logits.dtype,
            device=self.q_logits.device,
        )
        if self.gaussian_count == 0:
            return assignment
        if str(mode) == "q_logits":
            target_joints = self.q_logits.argmax(dim=-1)
        elif str(mode) == "bone_endpoint":
            target_joints = self._bone_endpoint_joint_ids(self.anchor_bone, skeleton, self.lambda_param.detach())
        elif str(mode) == "bone_endpoint_cut":
            target_joints = self._bone_endpoint_cut_joint_ids(
                self.anchor_bone,
                skeleton,
                self.lambda_param.detach(),
                split_lambda=float(endpoint_split_lambda),
            )
        elif str(mode) == "bone_endpoint_matrix":
            target_joints = self._bone_endpoint_matrix_joint_ids(self.anchor_bone, skeleton, self.endpoint_logits.detach())
        else:
            raise ValueError(f"unsupported hard assignment mode '{mode}'")
        gaussian_ids = torch.arange(self.gaussian_count, device=assignment.device)
        assignment[gaussian_ids, target_joints] = 1.0
        return assignment * self.active_mask.to(assignment.dtype).unsqueeze(-1)

    def compute_joint_support(self, kernels: torch.Tensor, skeleton: Any, ownership_strictness: float = 0.0) -> torch.Tensor:
        if self.q_logits.shape[0] != kernels.shape[0]:
            raise ValueError("kernel and q_logits gaussian dimensions do not match")
        assignment = self.compute_joint_assignment(skeleton, ownership_strictness=ownership_strictness)
        return self.compute_joint_support_from_assignment(kernels, assignment)

    def compute_joint_support_from_assignment(
        self,
        kernels: torch.Tensor,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        if assignment.shape[0] != kernels.shape[0]:
            raise ValueError("kernel and assignment gaussian dimensions do not match")
        support_kernels = self.compute_gaussian_value().unsqueeze(-1) * kernels
        return torch.einsum("gj,gv->jv", assignment, support_kernels)

    def compute_skinning_weights(
        self,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        ownership_strictness: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernels = self.compute_gaussian_kernels(rest_vertices, skeleton)
        support = self.compute_joint_support(kernels, skeleton, ownership_strictness=ownership_strictness)
        weights = support.transpose(0, 1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)
        return weights, kernels, support

    def expand_joint_logits(
        self,
        new_joint_count: int,
        promoted_gaussian_ids: torch.Tensor | None,
        new_joint_logit: float,
        other_joint_logit: float,
    ) -> None:
        current = self.q_logits.shape[1]
        if new_joint_count <= current:
            return
        extra = new_joint_count - current
        append = torch.full((self.gaussian_count, extra), other_joint_logit, dtype=self.q_logits.dtype, device=self.q_logits.device)
        if promoted_gaussian_ids is not None and promoted_gaussian_ids.numel() > 0:
            append[promoted_gaussian_ids, -1] = new_joint_logit
        self._replace_parameter("q_logits", torch.cat([self.q_logits.detach(), append], dim=1))

    def soft_reset_support(
        self,
        gaussian_ids: torch.Tensor,
        *,
        alpha_mix: float = 0.0,
        q_mix: float = 0.0,
        target_log_opacity: float | None = None,
        skeleton: Any | None = None,
    ) -> None:
        if gaussian_ids.numel() == 0:
            return
        gaussian_ids = gaussian_ids.long().to(self.anchor_bone.device)
        alpha_mix = float(max(0.0, min(1.0, alpha_mix)))
        q_mix = float(max(0.0, min(1.0, q_mix)))
        if alpha_mix > 0.0:
            updated_opacity = self.log_opacity.detach().clone()
            target_value = 0.0 if target_log_opacity is None else float(target_log_opacity)
            updated_opacity[gaussian_ids] = (
                (1.0 - alpha_mix) * updated_opacity[gaussian_ids]
                + alpha_mix * target_value
            )
            self._replace_parameter("log_opacity", updated_opacity)
        if q_mix > 0.0 and skeleton is not None:
            target_logits = self._initialize_joint_logits(
                self.anchor_bone[gaussian_ids],
                skeleton,
                lambda_param=self.lambda_param.detach()[gaussian_ids],
                mode=str(config.get("support", {}).get("initial_joint_logits_mode", "bone_parent")),
                endpoint_split_lambda=float(config.get("support", {}).get("endpoint_split_lambda", 0.6)),
            )
            updated_q = self.q_logits.detach().clone()
            updated_q[gaussian_ids] = (1.0 - q_mix) * updated_q[gaussian_ids] + q_mix * target_logits
            self._replace_parameter("q_logits", updated_q)

    def refresh_joint_logits_from_geometry(
        self,
        gaussian_ids: torch.Tensor,
        skeleton: Any,
        *,
        mode: str = "bone_endpoint",
        positive_logit: float = 8.0,
        negative_logit: float = -8.0,
    ) -> None:
        if gaussian_ids.numel() == 0:
            return
        gaussian_ids = gaussian_ids.long().to(self.anchor_bone.device)
        target_logits = self._initialize_hard_joint_logits(
            self.anchor_bone[gaussian_ids],
            skeleton,
            lambda_param=self.lambda_param.detach()[gaussian_ids],
            mode=mode,
            positive_logit=positive_logit,
            negative_logit=negative_logit,
        )
        updated_q = self.q_logits.detach().clone()
        updated_q[gaussian_ids] = target_logits
        self._replace_parameter("q_logits", updated_q)

    def deactivate_gaussians(self, gaussian_ids: torch.Tensor) -> None:
        if gaussian_ids.numel() == 0:
            return
        gaussian_ids = gaussian_ids.long().to(self.active_mask.device)
        self.active_mask[gaussian_ids] = False

    def split_gaussians(
        self,
        candidate_ids: torch.Tensor,
        config: dict[str, Any],
    ) -> torch.Tensor:
        if candidate_ids.numel() == 0:
            return candidate_ids
        support_cfg = config["support"]
        keep = []
        for idx in candidate_ids.tolist():
            if not bool(self.active_mask[idx].item()):
                continue
            if int(self.generation[idx].item()) >= int(support_cfg["split_max_generation"]):
                continue
            keep.append(idx)
        if not keep:
            return torch.empty(0, dtype=torch.long, device=self.anchor_bone.device)
        keep_tensor = torch.tensor(keep, dtype=torch.long, device=self.anchor_bone.device)
        lambda_noise = (torch.rand_like(self.lambda_param[keep_tensor]) - 0.5) * 2.0 * float(support_cfg["split_lambda_perturb"])
        scale_noise = torch.randn_like(self.log_scale[keep_tensor]) * float(support_cfg["split_log_scale_perturb"])
        new_ids = self.append_gaussians(
            anchor_bone=self.anchor_bone[keep_tensor],
            lambda_param=torch.maximum(
                torch.minimum(self.lambda_param[keep_tensor] + lambda_noise, self.lambda_max[keep_tensor]),
                self.lambda_min[keep_tensor],
            ),
            lambda_min=self.lambda_min[keep_tensor],
            lambda_max=self.lambda_max[keep_tensor],
            offset_local=torch.zeros_like(self.offset_local[keep_tensor]),
            rot_local=self.rot_local[keep_tensor].detach(),
            log_scale=self.log_scale[keep_tensor] + scale_noise,
            log_alpha=self.log_alpha[keep_tensor].detach(),
            q_logits=self.q_logits[keep_tensor].detach(),
            endpoint_logits=self.endpoint_logits[keep_tensor].detach(),
            generation=self.generation[keep_tensor] + 1,
            log_opacity=self.log_opacity[keep_tensor].detach(),
        )
        return new_ids

    def clamp_lambda_param(self) -> None:
        with torch.no_grad():
            self.lambda_param.data.copy_(torch.maximum(torch.minimum(self.lambda_param.data, self.lambda_max), self.lambda_min))

    @classmethod
    def build_axis_gaussians_for_bones(
        cls,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        init_cfg: dict[str, Any],
        bone_indices: torch.Tensor,
        reference_vertex_ids: torch.Tensor | None = None,
        seed_count_override: int | None = None,
        lambda_window_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if bone_indices.numel() == 0:
            raise ValueError("bone_indices must not be empty")
        reference_length = _resolve_reference_length(rest_vertices)
        assignment = cls._assign_vertices_to_bones(rest_vertices, skeleton)
        parent_pos, bone_frames, _, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        bone_lengths = (child_pos - parent_pos).norm(dim=-1).clamp_min(EPS)
        bone_lambda_min, bone_lambda_max = cls._compute_bone_lambda_bounds(
            skeleton,
            init_cfg,
            reference_length=reference_length,
        )
        ratio_min = float(init_cfg.get("longitudinal_scale_min_ratio", 2.5))
        axial_scale_factor = float(init_cfg.get("centerline_axial_scale_factor", 1.25))
        radial_scale_factor = float(init_cfg.get("centerline_radial_scale_factor", 0.5))
        min_radial_scale = _resolve_length_floor(
            init_cfg,
            reference_length=reference_length,
            ratio_key="centerline_min_radial_scale_ratio",
            fallback_ratio=0.015,
        )
        log_scale_min = float(init_cfg["log_scale_min"])
        log_scale_max = float(init_cfg["log_scale_max"])
        local_knn = int(init_cfg.get("centerline_knn_k", init_cfg["knn_k"]))
        patch_min_vertices = int(init_cfg.get("centerline_patch_min_vertices", max(8, int(init_cfg["knn_k"]) // 4)))
        counts_default = int(init_cfg.get("centerline_min_seeds_per_bone", 8))

        if reference_vertex_ids is not None and reference_vertex_ids.numel() > 0:
            reference_vertices = rest_vertices[reference_vertex_ids]
        else:
            reference_vertices = rest_vertices

        anchor_bones: list[torch.Tensor] = []
        lambda_values: list[torch.Tensor] = []
        lambda_mins: list[torch.Tensor] = []
        lambda_maxs: list[torch.Tensor] = []
        offset_local: list[torch.Tensor] = []
        rot_local: list[torch.Tensor] = []
        log_scales: list[torch.Tensor] = []

        for bone_tensor in bone_indices.tolist():
            bone_index = int(bone_tensor)
            count = int(seed_count_override) if seed_count_override is not None else counts_default
            if lambda_window_overrides is not None and bone_index in lambda_window_overrides:
                lower, upper = lambda_window_overrides[bone_index]
            else:
                lower = float(bone_lambda_min[bone_index].item())
                upper = float(bone_lambda_max[bone_index].item())
            lambdas = torch.linspace(
                lower,
                upper,
                max(count, 1),
                dtype=rest_vertices.dtype,
                device=rest_vertices.device,
            )
            segment = child_pos[bone_index] - parent_pos[bone_index]
            frame = bone_frames[bone_index]
            bone_vertex_ids = torch.nonzero(assignment["bone_index"] == bone_index, as_tuple=False).flatten()
            if bone_vertex_ids.numel() > 0:
                candidate_vertices = rest_vertices[bone_vertex_ids]
                candidate_lambdas = assignment["lambda_value"][bone_vertex_ids]
            else:
                candidate_vertices = reference_vertices
                candidate_lambdas = None
            lambda_window = max((upper - lower) / max(count + 1, 2), 0.05)
            axial_spacing = float(bone_lengths[bone_index].item()) * (upper - lower) / max(count + 1, 1)
            for lambda_value in lambdas:
                axis_point = parent_pos[bone_index] + lambda_value * segment
                if candidate_vertices.shape[0] == 0:
                    patch = rest_vertices
                elif candidate_lambdas is not None:
                    local_mask = (candidate_lambdas - lambda_value.clamp(0.0, 1.0)).abs() <= lambda_window
                    patch = candidate_vertices[local_mask]
                    if patch.shape[0] < patch_min_vertices:
                        nearest = knn_indices(candidate_vertices, axis_point.unsqueeze(0), min(local_knn, candidate_vertices.shape[0]))[0]
                        patch = candidate_vertices[nearest]
                else:
                    nearest = knn_indices(candidate_vertices, axis_point.unsqueeze(0), min(local_knn, candidate_vertices.shape[0]))[0]
                    patch = candidate_vertices[nearest]
                local_patch = torch.matmul(frame.transpose(0, 1), (patch - axis_point).T).T
                radial_distance = local_patch[:, 1:].norm(dim=-1)
                radial_radius = torch.maximum(radial_distance.median(), torch.tensor(min_radial_scale, dtype=rest_vertices.dtype, device=rest_vertices.device))
                axial_scale = max(axial_spacing * axial_scale_factor, float(radial_radius.item()) * ratio_min)
                radial_scale = max(float(radial_radius.item()) * radial_scale_factor, min_radial_scale)
                anchor_bones.append(torch.tensor(bone_index, dtype=torch.long, device=rest_vertices.device))
                lambda_values.append(lambda_value)
                lambda_mins.append(torch.tensor(lower, dtype=rest_vertices.dtype, device=rest_vertices.device))
                lambda_maxs.append(torch.tensor(upper, dtype=rest_vertices.dtype, device=rest_vertices.device))
                offset_local.append(torch.zeros(3, dtype=rest_vertices.dtype, device=rest_vertices.device))
                rot_local.append(matrix_to_quaternion(frame.unsqueeze(0))[0])
                log_scales.append(
                    torch.log(
                        torch.tensor([axial_scale, radial_scale, radial_scale], dtype=rest_vertices.dtype, device=rest_vertices.device).clamp_min(EPS)
                    ).clamp(log_scale_min, log_scale_max)
                )
        return (
            torch.stack(anchor_bones, dim=0),
            torch.stack(lambda_values, dim=0),
            torch.stack(lambda_mins, dim=0),
            torch.stack(lambda_maxs, dim=0),
            torch.stack(offset_local, dim=0),
            torch.stack(rot_local, dim=0),
            torch.stack(log_scales, dim=0),
        )

    def append_axis_gaussians_for_bones(
        self,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        config: dict[str, Any],
        bone_indices: torch.Tensor,
        reference_vertex_ids: torch.Tensor | None = None,
        seed_count_override: int | None = None,
        generation_value: int = 0,
        log_alpha_value: float | None = None,
        log_opacity_value: float | None = None,
        lambda_window_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> torch.Tensor:
        params = self.build_axis_gaussians_for_bones(
            rest_vertices=rest_vertices,
            skeleton=skeleton,
            init_cfg=config["init"],
            bone_indices=bone_indices,
            reference_vertex_ids=reference_vertex_ids,
            seed_count_override=seed_count_override,
            lambda_window_overrides=lambda_window_overrides,
        )
        anchor_bone, lambda_param, lambda_min, lambda_max, offset_local, rot_local, log_scale = params
        logits_mode = str(config.get("support", {}).get("initial_joint_logits_mode", "bone_parent"))
        endpoint_split_lambda = float(config.get("support", {}).get("endpoint_split_lambda", 0.6))
        q_logits = self._initialize_joint_logits(
            anchor_bone,
            skeleton,
            lambda_param=lambda_param,
            mode=logits_mode,
            endpoint_split_lambda=endpoint_split_lambda,
        )
        endpoint_logits = self._initialize_endpoint_logits_from_lambda(lambda_param)
        generation = torch.full(
            (anchor_bone.shape[0],),
            int(generation_value),
            dtype=torch.long,
            device=self.generation.device,
        )
        if log_alpha_value is None:
            log_alpha = torch.zeros(anchor_bone.shape[0], dtype=self.log_alpha.dtype, device=self.log_alpha.device)
        else:
            log_alpha = torch.full(
                (anchor_bone.shape[0],),
                float(log_alpha_value),
                dtype=self.log_alpha.dtype,
                device=self.log_alpha.device,
            )
        if log_opacity_value is None:
            log_opacity = torch.full(
                (anchor_bone.shape[0],),
                _resolve_seed_log_opacity(config["init"]),
                dtype=self.log_opacity.dtype,
                device=self.log_opacity.device,
            )
        else:
            log_opacity = torch.full(
                (anchor_bone.shape[0],),
                float(log_opacity_value),
                dtype=self.log_opacity.dtype,
                device=self.log_opacity.device,
            )
        return self.append_gaussians(
            anchor_bone=anchor_bone,
            lambda_param=lambda_param,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            offset_local=offset_local,
            rot_local=rot_local,
            log_scale=log_scale,
            log_alpha=log_alpha,
            q_logits=q_logits,
            endpoint_logits=endpoint_logits,
            generation=generation,
            log_opacity=log_opacity,
        )

    def repair_coverage(
        self,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        config: dict[str, Any],
        faces: torch.Tensor | None = None,
    ) -> None:
        init_cfg = config["init"]
        reference_length = _resolve_reference_length(rest_vertices)
        coverage_min = float(init_cfg["coverage_min"])
        max_iters = int(init_cfg["coverage_repair_max_iters"])
        full_assignment = self._assign_vertices_to_bones(rest_vertices, skeleton)
        radial_distance = full_assignment["radial_distance"]
        bone_index = full_assignment["bone_index"]
        frontier_ratio = float(init_cfg.get("coverage_frontier_radial_ratio", 1.45))
        frontier_min = _resolve_length_floor(
            init_cfg,
            reference_length=reference_length,
            ratio_key="coverage_frontier_min_distance_ratio",
            fallback_ratio=0.04,
        )

        radial_threshold = torch.full(
            (skeleton.bone_count,),
            frontier_min,
            dtype=rest_vertices.dtype,
            device=rest_vertices.device,
        )
        for local_bone in range(skeleton.bone_count):
            bone_mask = bone_index == local_bone
            if not bool(bone_mask.any().item()):
                continue
            bone_median = float(radial_distance[bone_mask].median().item())
            radial_threshold[local_bone] = max(bone_median * frontier_ratio, frontier_min)

        def split_low_coverage(vertex_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if vertex_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=rest_vertices.device)
                return empty, empty
            local_bone = bone_index[vertex_ids]
            local_radial = radial_distance[vertex_ids]
            frontier_mask = local_radial > radial_threshold[local_bone]
            repair_ids = vertex_ids[~frontier_mask]
            frontier_ids = vertex_ids[frontier_mask]
            return repair_ids, frontier_ids

        for _ in range(max_iters):
            kernels = self.compute_gaussian_kernels(rest_vertices, skeleton)
            coverage = kernels.sum(dim=0)
            low_vertices = torch.nonzero(coverage < coverage_min, as_tuple=False).flatten()
            if low_vertices.numel() == 0:
                return
            repair_vertex_ids, frontier_vertex_ids = split_low_coverage(low_vertices)
            if repair_vertex_ids.numel() == 0 and frontier_vertex_ids.numel() > 0:
                return
            seed_vertex_ids = repair_vertex_ids[: min(8, repair_vertex_ids.numel())]
            layout_mode = str(init_cfg.get("layout_mode", "centerline_uniform"))
            if layout_mode == "centerline_uniform":
                low_assignment = self._assign_vertices_to_bones(rest_vertices[seed_vertex_ids], skeleton)
                selected_bones = torch.unique(low_assignment["bone_index"])
                anchor_bone, lambda_param, lambda_min, lambda_max, offset_local, rot_local, log_scale = self.build_axis_gaussians_for_bones(
                    rest_vertices=rest_vertices,
                    skeleton=skeleton,
                    init_cfg=init_cfg,
                    bone_indices=selected_bones,
                    reference_vertex_ids=seed_vertex_ids,
                    seed_count_override=min(int(init_cfg.get("coverage_add_per_bone", 6)), max(int(seed_vertex_ids.numel()), 1)),
                )
            elif layout_mode == "mesh_medial_cloud":
                seed_stats = self._estimate_medial_seed_stats(
                    rest_vertices=rest_vertices,
                    seed_count=int(seed_vertex_ids.shape[0]),
                    init_cfg=init_cfg,
                    seed_vertex_ids=seed_vertex_ids,
                )
                anchor_bone, lambda_param, lambda_min, lambda_max, offset_local, rot_local = self.attach_seeds_to_skeleton(
                    seed_stats.centers,
                    seed_stats.principal_dirs,
                    skeleton,
                    rest_vertices=rest_vertices,
                    faces=faces,
                    init_cfg=init_cfg,
                )
                log_scale = seed_stats.log_scales
            else:
                raise ValueError(
                    f"unsupported init.layout_mode '{layout_mode}'; supported modes are "
                    "'centerline_uniform' and 'mesh_medial_cloud'"
                )
            logits_mode = str(config.get("support", {}).get("initial_joint_logits_mode", "bone_parent"))
            endpoint_split_lambda = float(config.get("support", {}).get("endpoint_split_lambda", 0.6))
            q_logits = self._initialize_joint_logits(
                anchor_bone,
                skeleton,
                lambda_param=lambda_param,
                mode=logits_mode,
                endpoint_split_lambda=endpoint_split_lambda,
            )
            endpoint_logits = self._initialize_endpoint_logits_from_lambda(lambda_param)
            added_count = int(anchor_bone.shape[0])
            self.append_gaussians(
                anchor_bone=anchor_bone,
                lambda_param=lambda_param,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                offset_local=offset_local,
                rot_local=rot_local,
                log_scale=log_scale,
                log_alpha=torch.zeros(added_count, dtype=self.log_alpha.dtype, device=self.log_alpha.device),
                q_logits=q_logits,
                endpoint_logits=endpoint_logits,
                generation=torch.zeros(added_count, dtype=torch.long, device=self.generation.device),
                log_opacity=torch.full(
                    (added_count,),
                    _resolve_seed_log_opacity(init_cfg),
                    dtype=self.log_opacity.dtype,
                    device=self.log_opacity.device,
                ),
            )
        kernels = self.compute_gaussian_kernels(rest_vertices, skeleton)
        coverage = kernels.sum(dim=0)
        low_vertices = torch.nonzero(coverage < coverage_min, as_tuple=False).flatten()
        repair_vertex_ids, frontier_vertex_ids = split_low_coverage(low_vertices)
        if repair_vertex_ids.numel() == 0 and frontier_vertex_ids.numel() > 0:
            return
        if repair_vertex_ids.numel() > 0:
            raise RuntimeError("coverage repair failed to eliminate near-zero-support regions")
