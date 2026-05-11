from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evorig_next.models.gaussian_field import GaussianSupportField as LegacyGaussianSupportField
from evorig_next.utils.geometry import EPS, knn_indices
from evorig_next.utils.mesh_ops import MeshQueryScene, build_mesh_query_scene, points_inside_or_on_mesh
from evorig_next.utils.rotations import matrix_to_quaternion, quaternion_to_matrix


@dataclass
class Phase1FieldState:
    anchor_bone: torch.Tensor
    lambda_param: torch.Tensor
    lambda_min: torch.Tensor
    lambda_max: torch.Tensor
    offset_local: torch.Tensor
    rot_local: torch.Tensor
    log_scale: torch.Tensor
    init_log_scale: torch.Tensor
    log_opacity: torch.Tensor
    log_value: torch.Tensor
    kernel_mahal_cutoff_sq: float


class Phase1GaussianField(nn.Module):
    def __init__(self, state: Phase1FieldState) -> None:
        super().__init__()
        self.register_buffer("anchor_bone", state.anchor_bone.long().clone())
        self.register_buffer("lambda_min", state.lambda_min.detach().clone())
        self.register_buffer("lambda_max", state.lambda_max.detach().clone())
        self.lambda_param = nn.Parameter(state.lambda_param.clone())
        self.offset_local = nn.Parameter(state.offset_local.clone())
        self.rot_local = nn.Parameter(state.rot_local.clone())
        self.log_scale = nn.Parameter(state.log_scale.clone())
        self.register_buffer("init_log_scale", state.init_log_scale.detach().clone())
        self.log_opacity = nn.Parameter(state.log_opacity.clone())
        self.log_value = nn.Parameter(state.log_value.clone())
        self.kernel_mahal_cutoff_sq = float(state.kernel_mahal_cutoff_sq)
        gaussian_count = int(state.anchor_bone.shape[0])
        self.register_buffer("active_mask", torch.ones(gaussian_count, dtype=torch.bool, device=state.anchor_bone.device))
        self.register_buffer("generation", torch.zeros(gaussian_count, dtype=torch.long, device=state.anchor_bone.device))
        self.register_buffer("q_logits", torch.zeros(gaussian_count, 1, dtype=state.lambda_param.dtype, device=state.lambda_param.device))
        self.endpoint_logits = nn.Parameter(self._endpoint_logits_from_lambda(state.lambda_param))
        self.sh_coeffs = nn.Parameter(torch.ones(gaussian_count, 1, dtype=state.lambda_param.dtype, device=state.lambda_param.device))
        self.use_sh_response: bool = False
        self.initial_seed_removed_outside_count: int = 0
        self.densify_seed_removed_outside_count: int = 0
        self.final_gaussian_pruned_outside_count: int = 0

    @staticmethod
    def _endpoint_logits_from_lambda(lambda_param: torch.Tensor) -> torch.Tensor:
        lam = lambda_param.clamp(0.0, 1.0)
        parent = (1.0 - lam).clamp_min(EPS)
        child = lam.clamp_min(EPS)
        return torch.stack([torch.log(parent), torch.log(child)], dim=-1)

    @staticmethod
    def _endpoint_logits_from_lambda_sigmoid(
        lambda_param: torch.Tensor,
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
        *,
        midpoint: float,
        slope: float,
    ) -> torch.Tensor:
        lam = torch.maximum(torch.minimum(lambda_param, lambda_max), lambda_min).clamp(0.0, 1.0)
        child = torch.sigmoid((lam - float(midpoint)) / max(float(slope), 1.0e-6))
        parent = (1.0 - child).clamp_min(EPS)
        child = child.clamp_min(EPS)
        return torch.stack([torch.log(parent), torch.log(child)], dim=-1)

    def reset_endpoint_logits_from_lambda_sigmoid(
        self,
        *,
        midpoint: float,
        slope: float,
        gaussian_ids: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            logits = self._endpoint_logits_from_lambda_sigmoid(
                self.lambda_param.detach(),
                self.lambda_min.detach(),
                self.lambda_max.detach(),
                midpoint=float(midpoint),
                slope=float(slope),
            ).to(device=self.endpoint_logits.device, dtype=self.endpoint_logits.dtype)
            if gaussian_ids is None:
                self.endpoint_logits.data.copy_(logits)
                return
            ids = gaussian_ids.to(device=self.endpoint_logits.device, dtype=torch.long).reshape(-1)
            if int(ids.numel()) > 0:
                self.endpoint_logits.data[ids] = logits[ids]

    @property
    def gaussian_count(self) -> int:
        return int(self.anchor_bone.shape[0])

    @property
    def sh_coeff_count(self) -> int:
        return int(self.sh_coeffs.shape[1])

    @staticmethod
    def unit_softplus_logit(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.log(torch.expm1(torch.ones((), dtype=dtype, device=device)))

    def ensure_sh_coeffs(self, coeff_count: int, *, preserve_unit_density: bool = True) -> bool:
        coeff_count = max(int(coeff_count), 1)
        current = int(self.sh_coeffs.shape[1])
        if current >= coeff_count:
            if preserve_unit_density and current > 0:
                with torch.no_grad():
                    self.sh_coeffs.data[:, 0] = self.unit_softplus_logit(self.sh_coeffs.dtype, self.sh_coeffs.device)
            return False
        expanded = torch.zeros(
            (self.gaussian_count, coeff_count),
            dtype=self.sh_coeffs.dtype,
            device=self.sh_coeffs.device,
        )
        expanded[:, :current] = self.sh_coeffs.detach()
        if preserve_unit_density:
            expanded[:, 0] = self.unit_softplus_logit(expanded.dtype, expanded.device)
        elif current == 0:
            expanded[:, 0] = 1.0
        self.sh_coeffs = nn.Parameter(expanded)
        return True

    @staticmethod
    def _phase1_scale_formula_config(base_config: dict[str, Any]) -> dict[str, Any]:
        return dict(base_config.get("phase1", {}))

    @staticmethod
    def _maybe_force_global_lambda_bounds(
        phase1_cfg: dict[str, Any],
        lambda_param: torch.Tensor,
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(phase1_cfg.get("force_global_lambda_bounds", False)):
            return lambda_param, lambda_min, lambda_max
        lower = float(phase1_cfg.get("global_lambda_min", -0.5))
        upper = float(phase1_cfg.get("global_lambda_max", 1.5))
        if lower > upper:
            raise ValueError("phase1 global_lambda_min must be <= global_lambda_max")
        forced_min = torch.full_like(lambda_min, lower)
        forced_max = torch.full_like(lambda_max, upper)
        return lambda_param.clamp(lower, upper), forced_min, forced_max

    @staticmethod
    def _build_mesh_adjacency(vertex_count: int, faces: torch.Tensor | None) -> list[tuple[int, ...]] | None:
        if faces is None or int(faces.numel()) == 0:
            return None
        adjacency: list[set[int]] = [set() for _ in range(int(vertex_count))]
        for tri in faces.detach().cpu().long().tolist():
            a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
            adjacency[a].update((b, c))
            adjacency[b].update((a, c))
            adjacency[c].update((a, b))
        return [tuple(sorted(item)) for item in adjacency]

    @staticmethod
    def _nearest_connected_patch_ids(
        patch_ids: torch.Tensor,
        center: torch.Tensor,
        rest_vertices: torch.Tensor,
        adjacency: list[tuple[int, ...]] | None,
    ) -> torch.Tensor:
        if adjacency is None or int(patch_ids.numel()) <= 1:
            return patch_ids
        patch_ids = patch_ids.reshape(-1).long()
        distances = torch.linalg.norm(rest_vertices[patch_ids] - center.unsqueeze(0), dim=-1)
        start = int(patch_ids[int(torch.argmin(distances).item())].item())
        allowed = set(int(item) for item in patch_ids.detach().cpu().tolist())
        seen = {start}
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in allowed and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if not component:
            return patch_ids
        component.sort()
        return torch.tensor(component, dtype=torch.long, device=patch_ids.device)

    @staticmethod
    def _polygon_contains_origin(points: np.ndarray) -> bool:
        if points.shape[0] < 3:
            return False
        x = points[:, 0]
        y = points[:, 1]
        inside = False
        j = points.shape[0] - 1
        for i in range(points.shape[0]):
            yi = y[i]
            yj = y[j]
            crosses = (yi > 0.0) != (yj > 0.0)
            if crosses:
                x_intersect = (x[j] - x[i]) * (-yi) / (yj - yi + 1.0e-20) + x[i]
                if x_intersect > 0.0:
                    inside = not inside
            j = i
        return inside

    @classmethod
    def _trimesh_section_ring_uv(
        cls,
        mesh: Any,
        *,
        center: torch.Tensor,
        frame: torch.Tensor,
        min_vertices: int,
        local_faces: np.ndarray | None = None,
    ) -> torch.Tensor | None:
        center_np = center.detach().cpu().numpy().astype(np.float64)
        frame_np = frame.detach().cpu().numpy().astype(np.float64)
        try:
            section_kwargs: dict[str, Any] = {}
            if local_faces is not None and int(local_faces.size) > 0:
                section_kwargs["local_faces"] = local_faces
            section = mesh.section(plane_normal=frame_np[:, 0], plane_origin=center_np, **section_kwargs)
        except (ValueError, IndexError, RuntimeError):
            return None
        if section is None:
            return None
        candidates: list[tuple[int, float, float, np.ndarray]] = []
        for contour in section.discrete:
            contour = np.asarray(contour, dtype=np.float64)
            if contour.ndim != 2 or contour.shape[0] < max(3, int(min_vertices)):
                continue
            uv = (contour - center_np.reshape(1, 3)) @ frame_np[:, 1:3]
            closed = bool(uv.shape[0] >= 2 and np.linalg.norm(uv[0] - uv[-1]) <= 1.0e-6)
            if closed:
                uv = uv[:-1]
            if uv.shape[0] < max(3, int(min_vertices)):
                continue
            radii = np.linalg.norm(uv, axis=1)
            if not np.isfinite(radii).all() or float(radii.max()) <= float(EPS):
                continue
            contains = closed and cls._polygon_contains_origin(uv)
            area = 0.5 * abs(float(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1))))
            if contains:
                candidates.append((0, area, float(radii.max()), uv))
            else:
                candidates.append((1, float(radii.min()), float(radii.max()), uv))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        selected = candidates[0][3]
        return torch.as_tensor(selected, dtype=center.dtype, device=center.device)

    @staticmethod
    def _farthest_radial_axes_2d(uv: torch.Tensor) -> torch.Tensor:
        if int(uv.numel()) == 0:
            return torch.eye(2, dtype=uv.dtype, device=uv.device)
        radii = torch.linalg.norm(uv, dim=-1)
        valid = radii > float(EPS)
        if not bool(valid.any()):
            return torch.eye(2, dtype=uv.dtype, device=uv.device)
        valid_uv = uv[valid]
        valid_radii = radii[valid]
        axis0 = valid_uv[int(torch.argmax(valid_radii).item())] / valid_radii.max().clamp_min(EPS)
        axis1 = torch.stack([-axis0[1], axis0[0]])
        return torch.stack([axis0, axis1], dim=-1)

    @staticmethod
    def _axis_extent(values: torch.Tensor, extent_fn: Any) -> torch.Tensor:
        positive = values[values >= 0.0]
        negative = -values[values <= 0.0]
        return torch.maximum(extent_fn(positive), extent_fn(negative))

    @classmethod
    def _compute_phase1_formula_log_scale(
        cls,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        init_cfg: dict[str, Any],
        phase1_cfg: dict[str, Any],
        *,
        anchor_bone: torch.Tensor,
        lambda_param: torch.Tensor,
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
        fallback_log_scale: torch.Tensor,
        fallback_rot_local: torch.Tensor,
        faces: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        formula = str(phase1_cfg.get("scale_formula", "legacy")).lower()
        if formula in {"", "legacy", "none"}:
            return fallback_log_scale, fallback_rot_local
        if formula not in {"local_radius_three_center", "cross_section_inner_ring"}:
            raise ValueError(f"unsupported phase1 scale_formula '{formula}'")
        if int(anchor_bone.numel()) == 0:
            return fallback_log_scale, fallback_rot_local

        radial_divisor = max(float(phase1_cfg.get("radial_sigma_divisor", 3.0)), EPS)
        radial_extent_quantile = min(max(float(phase1_cfg.get("radial_extent_quantile", 1.0)), 0.0), 1.0)
        axial_divisor = max(float(phase1_cfg.get("axial_three_center_divisor", 3.0)), EPS)
        axial_min_radial_ratio = max(float(phase1_cfg.get("axial_min_radial_ratio", 0.0)), 0.0)
        log_scale_min = float(phase1_cfg.get("formula_log_scale_min", -8.0))
        log_scale_max = float(phase1_cfg.get("formula_log_scale_max", 0.5))
        if log_scale_min > log_scale_max:
            raise ValueError("phase1 formula_log_scale_min must be <= formula_log_scale_max")

        def extent_1d(values: torch.Tensor) -> torch.Tensor:
            if int(values.numel()) == 0:
                return torch.zeros((), dtype=rest_vertices.dtype, device=rest_vertices.device)
            if radial_extent_quantile >= 1.0:
                return values.max()
            return torch.quantile(values, radial_extent_quantile)

        parent_pos, bone_frames, _, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        bone_lengths = (child_pos - parent_pos).norm(dim=-1).clamp_min(EPS)
        centers = cls.compute_rest_centers_from_params(
            anchor_bone,
            lambda_param,
            lambda_min,
            lambda_max,
            skeleton,
        )

        axial_sigma = torch.exp(fallback_log_scale[:, 0]).clone()
        for bone_tensor in torch.unique(anchor_bone).tolist():
            bone_index = int(bone_tensor)
            gaussian_ids = torch.nonzero(anchor_bone == bone_index, as_tuple=False).flatten()
            if int(gaussian_ids.numel()) <= 1:
                axial_sigma[gaussian_ids] = bone_lengths[bone_index] / axial_divisor
                continue
            ordered = gaussian_ids[torch.argsort(lambda_param[gaussian_ids])]
            ordered_centers = centers[ordered]
            for local_idx, gaussian_id in enumerate(ordered.tolist()):
                if local_idx == 0:
                    span = 2.0 * torch.linalg.norm(ordered_centers[1] - ordered_centers[0])
                elif local_idx == int(ordered.numel()) - 1:
                    span = 2.0 * torch.linalg.norm(ordered_centers[-1] - ordered_centers[-2])
                else:
                    span = torch.linalg.norm(ordered_centers[local_idx + 1] - ordered_centers[local_idx - 1])
                axial_sigma[gaussian_id] = span / axial_divisor

        assignment = LegacyGaussianSupportField._assign_vertices_to_bones(rest_vertices, skeleton)
        assigned_bone = assignment["bone_index"]
        assigned_lambda = assignment["lambda_value"]
        patch_min_vertices = int(init_cfg.get("centerline_patch_min_vertices", max(8, int(init_cfg.get("knn_k", 48)) // 4)))
        local_knn = int(init_cfg.get("centerline_knn_k", init_cfg.get("knn_k", 48)))
        window_scale = float(init_cfg.get("centerline_lambda_window_scale", 1.25))
        mesh_adjacency = (
            cls._build_mesh_adjacency(int(rest_vertices.shape[0]), faces)
            if bool(phase1_cfg.get("radial_patch_connected_component", True))
            else None
        )
        radial_sigma = torch.empty(anchor_bone.shape[0], 2, dtype=fallback_log_scale.dtype, device=fallback_log_scale.device)
        rot_local = fallback_rot_local.detach().clone()
        for bone_tensor in torch.unique(anchor_bone).tolist():
            bone_index = int(bone_tensor)
            gaussian_ids = torch.nonzero(anchor_bone == bone_index, as_tuple=False).flatten()
            count = max(int(gaussian_ids.numel()), 1)
            lower = float(lambda_min[gaussian_ids].min().item())
            upper = float(lambda_max[gaussian_ids].max().item())
            lambda_window = max((upper - lower) * window_scale / max(count + 1, 2), 0.05)
            bone_vertex_ids = torch.nonzero(assigned_bone == bone_index, as_tuple=False).flatten()
            frame = bone_frames[bone_index]
            for gaussian_id in gaussian_ids.tolist():
                center = centers[gaussian_id]
                patch_ids = bone_vertex_ids
                if int(bone_vertex_ids.numel()) > 0:
                    local_mask = (assigned_lambda[bone_vertex_ids] - lambda_param[gaussian_id].clamp(0.0, 1.0)).abs() <= lambda_window
                    candidate_ids = bone_vertex_ids[local_mask]
                    if int(candidate_ids.numel()) >= patch_min_vertices:
                        patch_ids = candidate_ids
                if int(patch_ids.numel()) < patch_min_vertices:
                    if int(bone_vertex_ids.numel()) >= patch_min_vertices:
                        local_patch = rest_vertices[bone_vertex_ids]
                        nearest = knn_indices(local_patch, center.unsqueeze(0), min(local_knn, int(local_patch.shape[0])))[0]
                        patch_ids = bone_vertex_ids[nearest]
                    else:
                        nearest = knn_indices(rest_vertices, center.unsqueeze(0), min(local_knn, int(rest_vertices.shape[0])))[0]
                        patch_ids = nearest
                patch_ids = cls._nearest_connected_patch_ids(
                    patch_ids,
                    center,
                    rest_vertices,
                    mesh_adjacency,
                )
                patch = rest_vertices[patch_ids]
                local_patch = torch.matmul(frame.transpose(0, 1), (patch - center).T).T
                uv = local_patch[:, 1:3]
                axes_2d = cls._farthest_radial_axes_2d(uv)
                projected = uv @ axes_2d
                axis_extents = []
                for axis_index in range(2):
                    axis_extents.append(cls._axis_extent(projected[:, axis_index], extent_1d))
                radial_sigma[gaussian_id] = torch.stack(axis_extents, dim=0) / radial_divisor
                yz_frame = frame[:, 1:3]
                radial_y = yz_frame @ axes_2d[:, 0]
                radial_z = yz_frame @ axes_2d[:, 1]
                pca_frame = torch.stack([frame[:, 0], radial_y, radial_z], dim=-1)
                rot_local[gaussian_id] = matrix_to_quaternion(pca_frame.unsqueeze(0))[0]

        if formula == "cross_section_inner_ring" and faces is not None and int(faces.numel()) > 0:
            import trimesh

            face_indices = faces.to(device=rest_vertices.device, dtype=torch.long)
            min_bins = int(phase1_cfg.get("cross_section_min_bins", 10))
            section_surface_tol = float(init_cfg.get("bone_assignment_surface_tol", 3.0e-3))
            section_mesh = trimesh.Trimesh(
                vertices=rest_vertices.detach().cpu().numpy(),
                faces=face_indices.detach().cpu().numpy(),
                process=False,
            )
            for gaussian_id in range(int(anchor_bone.shape[0])):
                bone_index = int(anchor_bone[gaussian_id].item())
                if bone_index < 0 or bone_index >= int(bone_frames.shape[0]):
                    continue
                lam_value = float(lambda_param[gaussian_id].item())
                if lam_value < 0.0 or lam_value > 1.0:
                    continue
                frame = bone_frames[bone_index]
                center = centers[gaussian_id]
                center_inside = points_inside_or_on_mesh(
                    center.reshape(1, 3),
                    rest_vertices,
                    face_indices,
                    surface_tol=section_surface_tol,
                )
                if not bool(center_inside.item()):
                    continue
                ring_uv = cls._trimesh_section_ring_uv(
                    section_mesh,
                    center=center,
                    frame=frame,
                    min_vertices=min_bins,
                )
                if ring_uv is None or int(ring_uv.shape[0]) < 2:
                    continue
                axes_2d = cls._farthest_radial_axes_2d(ring_uv)
                projected = ring_uv @ axes_2d
                radii = torch.linalg.norm(ring_uv, dim=-1)
                radius_ratio = radii.max() / radii.min().clamp_min(EPS)
                if bool((radius_ratio >= 4.0).item()):
                    continue
                axis_extents = []
                for axis_index in range(2):
                    axis_extents.append(cls._axis_extent(projected[:, axis_index], extent_1d))
                radial_sigma[gaussian_id] = torch.stack(axis_extents, dim=0) / radial_divisor
                yz_frame = frame[:, 1:3]
                radial_y = yz_frame @ axes_2d[:, 0]
                radial_z = yz_frame @ axes_2d[:, 1]
                pca_frame = torch.stack([frame[:, 0], radial_y, radial_z], dim=-1)
                rot_local[gaussian_id] = matrix_to_quaternion(pca_frame.unsqueeze(0))[0]

        if axial_min_radial_ratio > 0.0:
            axial_sigma = torch.maximum(axial_sigma, radial_sigma.max(dim=-1).values * axial_min_radial_ratio)
        log_scale = torch.log(torch.cat([axial_sigma.unsqueeze(-1), radial_sigma], dim=-1).clamp_min(EPS))
        return log_scale.clamp(log_scale_min, log_scale_max), rot_local

    @classmethod
    def initialize_from_legacy(
        cls,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        base_config: dict[str, Any],
        *,
        lambda_min: float,
        lambda_max: float,
        seed_count_scale: float,
        kernel_mahal_cutoff_sq: float,
        faces: torch.Tensor | None = None,
    ) -> "Phase1GaussianField":
        cfg = {
            **base_config,
            "init": dict(base_config.get("init", {})),
            "support": dict(base_config.get("support", {})),
        }
        init_cfg = cfg["init"]
        init_cfg["seed_count"] = max(int(round(int(init_cfg.get("seed_count", 1)) * float(seed_count_scale))), 1)
        init_cfg["branch_new_bone_seed_count"] = max(
            int(round(int(init_cfg.get("branch_new_bone_seed_count", 1)) * float(seed_count_scale))),
            1,
        )
        init_cfg["split_new_bone_seed_count"] = max(
            int(round(int(init_cfg.get("split_new_bone_seed_count", 1)) * float(seed_count_scale))),
            1,
        )
        init_cfg["global_lambda_min"] = float(lambda_min)
        init_cfg["global_lambda_max"] = float(lambda_max)
        if bool(base_config.get("phase1", {}).get("decouple_axial_from_radial_init", False)):
            init_cfg["longitudinal_scale_min_ratio"] = 0.0
        scale_init_divisor = float(base_config.get("phase1", {}).get("scale_init_divisor", 1.0))
        if scale_init_divisor > 0.0 and abs(scale_init_divisor - 1.0) > 1.0e-12:
            init_cfg["centerline_axial_scale_factor"] = float(init_cfg.get("centerline_axial_scale_factor", 1.25)) / scale_init_divisor
            init_cfg["centerline_radial_scale_factor"] = float(init_cfg.get("centerline_radial_scale_factor", 0.5)) / scale_init_divisor
            init_cfg["centerline_min_radial_scale_ratio"] = float(init_cfg.get("centerline_min_radial_scale_ratio", 0.015)) / scale_init_divisor
            init_cfg["endpoint_extension_min_ratio"] = float(init_cfg.get("endpoint_extension_min_ratio", 0.08)) / scale_init_divisor
        cfg["support"]["initial_joint_logits_mode"] = "bone_endpoint_cut"
        cfg["support"]["endpoint_split_lambda"] = 0.75
        legacy = LegacyGaussianSupportField.initialize_from_center_seeds(
            rest_vertices,
            skeleton,
            cfg,
            faces=faces,
        )
        phase1_cfg = cls._phase1_scale_formula_config(base_config)
        lambda_param, lambda_min_buf, lambda_max_buf = cls._maybe_force_global_lambda_bounds(
            phase1_cfg,
            legacy.lambda_param.detach(),
            legacy.lambda_min.detach(),
            legacy.lambda_max.detach(),
        )
        log_scale, rot_local = cls._compute_phase1_formula_log_scale(
            rest_vertices,
            skeleton,
            cfg["init"],
            phase1_cfg,
            anchor_bone=legacy.anchor_bone.detach(),
            lambda_param=lambda_param,
            lambda_min=lambda_min_buf,
            lambda_max=lambda_max_buf,
            fallback_log_scale=legacy.log_scale.detach(),
            fallback_rot_local=legacy.rot_local.detach(),
            faces=faces,
        )
        state = Phase1FieldState(
            anchor_bone=legacy.anchor_bone.detach(),
            lambda_param=lambda_param.detach(),
            lambda_min=lambda_min_buf.detach(),
            lambda_max=lambda_max_buf.detach(),
            offset_local=legacy.offset_local.detach(),
            rot_local=rot_local.detach(),
            log_scale=log_scale.detach(),
            init_log_scale=log_scale.detach(),
            log_opacity=legacy.log_opacity.detach(),
            log_value=legacy.log_value.detach(),
            kernel_mahal_cutoff_sq=float(kernel_mahal_cutoff_sq),
        )
        removed_count = 0
        prune_flag = bool(base_config.get("phase1", {}).get("initial_seed_prune_outside_mesh", False))
        if not prune_flag:
            prune_flag = bool(base_config.get("phase1_initial_seed_prune_outside_mesh", False))
        if faces is not None and faces.numel() > 0 and prune_flag:
            mesh_query_scene = build_mesh_query_scene(rest_vertices, faces)
            state, removed_count, _ = cls._prune_state_outside_mesh(
                state,
                skeleton,
                rest_vertices,
                faces,
                surface_tol=float(base_config.get("phase1_seed_inside_surface_tol", 3.0e-3)),
                mesh_query_scene=mesh_query_scene,
            )
        field = cls(state)
        field.initial_seed_removed_outside_count = int(removed_count)
        return field

    @staticmethod
    def _clamped_lambda(
        lambda_param: torch.Tensor,
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
    ) -> torch.Tensor:
        return torch.maximum(torch.minimum(lambda_param, lambda_max), lambda_min)

    @classmethod
    def compute_rest_centers_from_params(
        cls,
        anchor_bone: torch.Tensor,
        lambda_param: torch.Tensor,
        lambda_min: torch.Tensor,
        lambda_max: torch.Tensor,
        skeleton: Any,
        offset_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parent_pos, bone_frames, _, bone_child_idx = skeleton.compute_bone_frames()
        child_pos = skeleton.rest_joints[bone_child_idx]
        lam = cls._clamped_lambda(lambda_param, lambda_min, lambda_max)
        centers = parent_pos[anchor_bone] + lam.unsqueeze(-1) * (child_pos[anchor_bone] - parent_pos[anchor_bone])
        if offset_local is not None and int(offset_local.numel()) > 0:
            local_frame = bone_frames[anchor_bone]
            centers = centers + torch.einsum(
                "gij,gj->gi",
                local_frame,
                offset_local.to(device=centers.device, dtype=centers.dtype),
            )
        return centers

    @classmethod
    def _prune_state_outside_mesh(
        cls,
        state: Phase1FieldState,
        skeleton: Any,
        rest_vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        surface_tol: float,
        mesh_query_scene: MeshQueryScene | None,
    ) -> tuple[Phase1FieldState, int, list[int]]:
        if int(state.anchor_bone.numel()) == 0:
            return state, 0, []
        centers = cls.compute_rest_centers_from_params(
            state.anchor_bone,
            state.lambda_param,
            state.lambda_min,
            state.lambda_max,
            skeleton,
            state.offset_local,
        )
        keep_mask = points_inside_or_on_mesh(
            centers,
            rest_vertices,
            faces,
            surface_tol=float(surface_tol),
            mesh_query_scene=mesh_query_scene,
        )
        removed_count = int((~keep_mask).sum().item())
        if removed_count <= 0:
            return state, 0, []
        raw_counts = torch.bincount(state.anchor_bone, minlength=max(int(skeleton.bone_count), 1))
        kept_anchor = state.anchor_bone[keep_mask]
        kept_counts = torch.bincount(kept_anchor, minlength=max(int(skeleton.bone_count), 1)) if kept_anchor.numel() > 0 else torch.zeros_like(raw_counts)
        skipped_bones = torch.nonzero((raw_counts > 0) & (kept_counts <= 0), as_tuple=False).flatten().tolist()
        if kept_anchor.numel() <= 0:
            empty = torch.zeros(0, dtype=state.anchor_bone.dtype, device=state.anchor_bone.device)
            empty_vec = torch.zeros(0, 3, dtype=state.lambda_param.dtype, device=state.lambda_param.device)
            empty_rot = torch.zeros(0, state.rot_local.shape[-1], dtype=state.rot_local.dtype, device=state.rot_local.device)
            empty_scalar = torch.zeros(0, dtype=state.log_opacity.dtype, device=state.log_opacity.device)
            state = Phase1FieldState(
                anchor_bone=empty.long(),
                lambda_param=empty_scalar,
                lambda_min=empty_scalar,
                lambda_max=empty_scalar,
                offset_local=empty_vec,
                rot_local=empty_rot,
                log_scale=empty_vec,
                init_log_scale=empty_vec,
                log_opacity=empty_scalar,
                log_value=empty_scalar,
                kernel_mahal_cutoff_sq=state.kernel_mahal_cutoff_sq,
            )
            return state, removed_count, [int(item) for item in skipped_bones]
        state = Phase1FieldState(
            anchor_bone=state.anchor_bone[keep_mask],
            lambda_param=state.lambda_param[keep_mask],
            lambda_min=state.lambda_min[keep_mask],
            lambda_max=state.lambda_max[keep_mask],
            offset_local=state.offset_local[keep_mask],
            rot_local=state.rot_local[keep_mask],
            log_scale=state.log_scale[keep_mask],
            init_log_scale=state.init_log_scale[keep_mask],
            log_opacity=state.log_opacity[keep_mask],
            log_value=state.log_value[keep_mask],
            kernel_mahal_cutoff_sq=state.kernel_mahal_cutoff_sq,
        )
        return state, removed_count, [int(item) for item in skipped_bones]

    def clamp_lambda_param(self) -> None:
        with torch.no_grad():
            self.lambda_param.data.copy_(torch.maximum(torch.minimum(self.lambda_param.data, self.lambda_max), self.lambda_min))

    def compute_rest_centers(self, skeleton: Any) -> torch.Tensor:
        return self.compute_rest_centers_from_params(
            self.anchor_bone,
            self.lambda_param,
            self.lambda_min,
            self.lambda_max,
            skeleton,
            self.offset_local,
        )

    def prune_active_outside_mesh(
        self,
        skeleton: Any,
        rest_vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        surface_tol: float,
        mesh_query_scene: MeshQueryScene | None = None,
    ) -> int:
        if self.gaussian_count <= 0 or faces is None or faces.numel() == 0:
            return 0
        centers = self.compute_rest_centers(skeleton)
        inside_mask = points_inside_or_on_mesh(
            centers,
            rest_vertices,
            faces,
            surface_tol=float(surface_tol),
            mesh_query_scene=mesh_query_scene if mesh_query_scene is not None else build_mesh_query_scene(rest_vertices, faces),
        )
        prune_mask = self.active_mask & (~inside_mask)
        pruned_count = int(prune_mask.sum().item())
        if pruned_count > 0:
            self.active_mask[prune_mask] = False
            self.final_gaussian_pruned_outside_count += pruned_count
        return pruned_count

    def compute_covariance(self, skeleton: Any) -> torch.Tensor:
        del skeleton
        orientation = quaternion_to_matrix(self.rot_local)
        scales = torch.exp(self.log_scale).clamp_min(EPS)
        diag = torch.diag_embed(scales.square())
        cov = orientation @ diag @ orientation.transpose(-1, -2)
        inactive = ~self.active_mask
        if bool(inactive.any().item()):
            eye = torch.eye(3, dtype=cov.dtype, device=cov.device)
            cov[inactive] = eye
        return cov

    def compute_gaussian_opacity(self) -> torch.Tensor:
        return torch.exp(self.log_opacity)

    def compute_gaussian_value(self) -> torch.Tensor:
        return torch.exp(self.log_value)

    def compute_gaussian_kernels(self, rest_vertices: torch.Tensor, skeleton: Any) -> torch.Tensor:
        centers = self.compute_rest_centers(skeleton)
        cov = self.compute_covariance(skeleton)
        inv_cov = torch.linalg.inv(cov)
        diff = rest_vertices.unsqueeze(0) - centers.unsqueeze(1)
        mahal = torch.einsum("gvi,gij,gvj->gv", diff, inv_cov, diff)
        cutoff_sq = float(self.kernel_mahal_cutoff_sq)
        density = self.compute_gaussian_opacity().unsqueeze(-1) * torch.exp(-0.5 * mahal)
        if cutoff_sq > 0.0:
            support_mask = (mahal <= float(cutoff_sq)).to(dtype=rest_vertices.dtype, device=rest_vertices.device)
            density = density * support_mask
        if bool(self.use_sh_response) and self.sh_coeff_count > 0:
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
            density = density * F.softplus(sh_response)
        return density * self.active_mask.to(density.dtype).unsqueeze(-1)

    def compute_joint_mix(
        self,
        skeleton: Any,
        *,
        midpoint: float,
        slope: float,
        mode: str = "endpoint_cut",
        use_endpoint_logits: bool = False,
        endpoint_logits_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joint_count = int(skeleton.joint_count)
        mix = torch.zeros(self.gaussian_count, joint_count, dtype=self.lambda_param.dtype, device=self.lambda_param.device)
        child_joints = skeleton.bone_child_idx[self.anchor_bone]
        parent_joints = skeleton.bone_parent_idx[self.anchor_bone]
        lam = torch.maximum(torch.minimum(self.lambda_param, self.lambda_max), self.lambda_min).clamp(0.0, 1.0)
        ownership_mode = str(mode).strip().lower()
        if ownership_mode in {"sigmoid", "soft", "soft_sigmoid"}:
            base_child_weight = torch.sigmoid((lam - float(midpoint)) / max(float(slope), 1.0e-6))
        elif ownership_mode in {"endpoint_cut", "hard_cut", "vertex_endpoint_cut", "child_support_gate"}:
            base_child_weight = (lam >= float(midpoint)).to(dtype=lam.dtype)
        elif ownership_mode in {"count_endpoint_cut", "rank_endpoint_cut"}:
            base_child_weight = torch.zeros_like(lam)
            for bone_tensor in torch.unique(self.anchor_bone).tolist():
                bone_index = int(bone_tensor)
                gaussian_ids = torch.nonzero(
                    (self.anchor_bone == bone_index) & self.active_mask,
                    as_tuple=False,
                ).flatten()
                count = int(gaussian_ids.numel())
                if count <= 1:
                    continue
                child_count = max(1, int(round(count * max(1.0 - float(midpoint), 0.0))))
                child_count = min(child_count, count - 1)
                ordered = gaussian_ids[torch.argsort(lam[gaussian_ids])]
                base_child_weight[ordered[-child_count:]] = 1.0
        else:
            raise ValueError(f"unsupported phase1 ownership_mode '{mode}'")
        base_parent_weight = 1.0 - base_child_weight

        if bool(use_endpoint_logits):
            endpoint_weight = torch.softmax(self.endpoint_logits, dim=-1)
            if endpoint_logits_mask is not None:
                mask = endpoint_logits_mask.to(device=self.lambda_param.device, dtype=torch.bool).reshape(-1)
                if int(mask.shape[0]) != self.gaussian_count:
                    raise ValueError("endpoint_logits_mask must have shape [G]")
                parent_weight = torch.where(mask, endpoint_weight[:, 0], base_parent_weight)
                child_weight = torch.where(mask, endpoint_weight[:, 1], base_child_weight)
            else:
                parent_weight = endpoint_weight[:, 0]
                child_weight = endpoint_weight[:, 1]
        else:
            parent_weight = base_parent_weight
            child_weight = base_child_weight
        gaussian_ids = torch.arange(self.gaussian_count, dtype=torch.long, device=self.lambda_param.device)
        mix[gaussian_ids, child_joints] = child_weight
        valid_parent = parent_joints >= 0
        if bool(valid_parent.any().item()):
            mix[gaussian_ids[valid_parent], parent_joints[valid_parent]] = parent_weight[valid_parent]
        return mix * self.active_mask.to(mix.dtype).unsqueeze(-1)

    def compute_joint_support(
        self,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        *,
        midpoint: float,
        slope: float,
        mode: str = "endpoint_cut",
        child_gate_start: float = 0.75,
        child_gate_end: float = 0.95,
        use_endpoint_logits: bool = False,
        endpoint_logits_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernels = self.compute_gaussian_kernels(rest_vertices, skeleton)
        ownership_mode = str(mode).strip().lower()
        if ownership_mode in {"vertex_endpoint_cut", "child_support_gate"} and not bool(use_endpoint_logits):
            mix = self.compute_joint_mix(
                skeleton,
                midpoint=midpoint,
                slope=slope,
                mode="endpoint_cut",
                use_endpoint_logits=False,
            )
            support_kernels = self.compute_gaussian_value().unsqueeze(-1) * kernels
            support = torch.zeros(
                int(skeleton.joint_count),
                int(rest_vertices.shape[0]),
                dtype=support_kernels.dtype,
                device=support_kernels.device,
            )
            parent_pos, _, bone_parent_idx, bone_child_idx = skeleton.compute_bone_frames()
            child_pos = skeleton.rest_joints[bone_child_idx]
            for bone_tensor in torch.unique(self.anchor_bone).tolist():
                bone_index = int(bone_tensor)
                if bone_index < 0 or bone_index >= int(bone_child_idx.shape[0]):
                    continue
                gaussian_mask = (self.anchor_bone == bone_index) & self.active_mask
                if not bool(gaussian_mask.any().item()):
                    continue
                bone_support = support_kernels[gaussian_mask].sum(dim=0)
                parent_joint = int(bone_parent_idx[bone_index].item())
                child_joint = int(bone_child_idx[bone_index].item())
                start = parent_pos[bone_index]
                end = child_pos[bone_index]
                segment = end - start
                segment_len_sq = segment.square().sum().clamp_min(EPS)
                vertex_lambda = ((rest_vertices - start.unsqueeze(0)) * segment.unsqueeze(0)).sum(dim=-1) / segment_len_sq
                vertex_lambda = vertex_lambda.clamp(0.0, 1.0)
                if ownership_mode == "vertex_endpoint_cut":
                    child_mask = (vertex_lambda >= float(midpoint)).to(dtype=bone_support.dtype)
                else:
                    gate_start = float(child_gate_start)
                    gate_end = max(float(child_gate_end), gate_start + 1.0e-6)
                    gate_t = ((vertex_lambda - gate_start) / (gate_end - gate_start)).clamp(0.0, 1.0)
                    child_mask = gate_t * gate_t * (3.0 - 2.0 * gate_t)
                if parent_joint >= 0:
                    support[parent_joint] = support[parent_joint] + bone_support * (1.0 - child_mask)
                else:
                    child_mask = torch.ones_like(child_mask)
                support[child_joint] = support[child_joint] + bone_support * child_mask
            return kernels, mix, support
        mix = self.compute_joint_mix(
            skeleton,
            mode=str(mode),
            midpoint=midpoint,
            slope=slope,
            use_endpoint_logits=bool(use_endpoint_logits),
            endpoint_logits_mask=endpoint_logits_mask,
        )
        support_kernels = self.compute_gaussian_value().unsqueeze(-1) * kernels
        support = torch.einsum("gj,gv->jv", mix, support_kernels)
        return kernels, mix, support

    def append_axis_gaussians_for_bones(
        self,
        rest_vertices: torch.Tensor,
        skeleton: Any,
        base_config: dict[str, Any],
        *,
        bone_indices: torch.Tensor,
        seeds_per_bone: int,
        generation_value: int,
        faces: torch.Tensor | None = None,
        prune_outside_mesh: bool = False,
        surface_tol: float = 3.0e-3,
        mesh_query_scene: MeshQueryScene | None = None,
    ) -> dict[str, Any]:
        (
            anchor_bone,
            lambda_param,
            lambda_min,
            lambda_max,
            offset_local,
            rot_local,
            log_scale,
        ) = LegacyGaussianSupportField.build_axis_gaussians_for_bones(
            rest_vertices=rest_vertices,
            skeleton=skeleton,
            init_cfg=self._phase1_densify_init_cfg(base_config["init"], base_config),
            bone_indices=bone_indices,
            seed_count_override=int(seeds_per_bone),
        )
        phase1_cfg = self._phase1_scale_formula_config(base_config)
        lambda_param, lambda_min, lambda_max = self._maybe_force_global_lambda_bounds(
            phase1_cfg,
            lambda_param,
            lambda_min,
            lambda_max,
        )
        removed_outside_count = 0
        skipped_bones_outside_only: list[int] = []
        if prune_outside_mesh and faces is not None and faces.numel() > 0 and int(anchor_bone.numel()) > 0:
            state, removed_outside_count, skipped_bones_outside_only = self._prune_state_outside_mesh(
                Phase1FieldState(
                    anchor_bone=anchor_bone,
                    lambda_param=lambda_param,
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                    offset_local=offset_local,
                    rot_local=rot_local,
                    log_scale=log_scale,
                    init_log_scale=log_scale,
                    log_opacity=torch.full((anchor_bone.shape[0],), -2.0, dtype=log_scale.dtype, device=log_scale.device),
                    log_value=torch.zeros(anchor_bone.shape[0], dtype=log_scale.dtype, device=log_scale.device),
                    kernel_mahal_cutoff_sq=self.kernel_mahal_cutoff_sq,
                ),
                skeleton,
                rest_vertices,
                faces,
                surface_tol=float(surface_tol),
                mesh_query_scene=mesh_query_scene if mesh_query_scene is not None else build_mesh_query_scene(rest_vertices, faces),
            )
            anchor_bone = state.anchor_bone
            lambda_param = state.lambda_param
            lambda_min = state.lambda_min
            lambda_max = state.lambda_max
            offset_local = state.offset_local
            rot_local = state.rot_local
            log_scale = state.log_scale
        log_scale, rot_local = self._compute_phase1_formula_log_scale(
            rest_vertices,
            skeleton,
            self._phase1_densify_init_cfg(base_config["init"], base_config),
            phase1_cfg,
            anchor_bone=anchor_bone,
            lambda_param=lambda_param,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            fallback_log_scale=log_scale,
            fallback_rot_local=rot_local,
            faces=faces,
        )
        anchor_bone = anchor_bone.detach()
        lambda_param = lambda_param.detach()
        lambda_min = lambda_min.detach()
        lambda_max = lambda_max.detach()
        offset_local = offset_local.detach()
        rot_local = rot_local.detach()
        log_scale = log_scale.detach()
        new_count = int(anchor_bone.shape[0])
        if new_count <= 0:
            self.densify_seed_removed_outside_count += int(removed_outside_count)
            return {
                "new_ids": torch.zeros(0, dtype=torch.long, device=self.anchor_bone.device),
                "removed_outside_count": int(removed_outside_count),
                "skipped_bones_outside_only": [int(item) for item in skipped_bones_outside_only],
            }
        start = self.gaussian_count
        new_ids = torch.arange(start, start + new_count, dtype=torch.long, device=self.anchor_bone.device)
        self.anchor_bone = torch.cat([self.anchor_bone.detach(), anchor_bone.to(self.anchor_bone.device)], dim=0)
        self.lambda_min = torch.cat([self.lambda_min.detach(), lambda_min.to(self.lambda_min.device)], dim=0)
        self.lambda_max = torch.cat([self.lambda_max.detach(), lambda_max.to(self.lambda_max.device)], dim=0)
        self.lambda_param = nn.Parameter(torch.cat([self.lambda_param.detach(), lambda_param.to(self.lambda_param.device)], dim=0))
        self.offset_local = nn.Parameter(torch.cat([self.offset_local.detach(), offset_local.to(self.offset_local.device)], dim=0))
        self.rot_local = nn.Parameter(torch.cat([self.rot_local.detach(), rot_local.to(self.rot_local.device)], dim=0))
        self.log_scale = nn.Parameter(torch.cat([self.log_scale.detach(), log_scale.to(self.log_scale.device)], dim=0))
        self.init_log_scale = torch.cat([self.init_log_scale.detach(), log_scale.to(self.init_log_scale.device)], dim=0)
        mean_opacity = float(self.log_opacity.detach().mean().item()) if self.log_opacity.numel() > 0 else -2.0
        mean_value = float(self.log_value.detach().mean().item()) if self.log_value.numel() > 0 else 0.0
        self.log_opacity = nn.Parameter(
            torch.cat(
                [self.log_opacity.detach(), torch.full((new_count,), mean_opacity, dtype=self.log_opacity.dtype, device=self.log_opacity.device)],
                dim=0,
            )
        )
        self.log_value = nn.Parameter(
            torch.cat(
                [self.log_value.detach(), torch.full((new_count,), mean_value, dtype=self.log_value.dtype, device=self.log_value.device)],
                dim=0,
            )
        )
        self.active_mask = torch.cat(
            [self.active_mask.detach(), torch.ones(new_count, dtype=torch.bool, device=self.active_mask.device)],
            dim=0,
        )
        self.generation = torch.cat(
            [self.generation.detach(), torch.full((new_count,), int(generation_value), dtype=torch.long, device=self.generation.device)],
            dim=0,
        )
        self.q_logits = torch.cat(
            [self.q_logits.detach(), torch.zeros(new_count, 1, dtype=self.q_logits.dtype, device=self.q_logits.device)],
            dim=0,
        )
        self.endpoint_logits = nn.Parameter(
            torch.cat(
                [self.endpoint_logits.detach(), self._endpoint_logits_from_lambda(lambda_param.to(self.lambda_param.device))],
                dim=0,
            )
        )
        new_sh = torch.zeros(new_count, self.sh_coeff_count, dtype=self.sh_coeffs.dtype, device=self.sh_coeffs.device)
        if self.sh_coeff_count > 0:
            if bool(self.use_sh_response):
                new_sh[:, 0] = self.unit_softplus_logit(new_sh.dtype, new_sh.device)
            else:
                new_sh[:, 0] = 1.0
        self.sh_coeffs = nn.Parameter(
            torch.cat(
                [self.sh_coeffs.detach(), new_sh],
                dim=0,
            )
        )
        self.densify_seed_removed_outside_count += int(removed_outside_count)
        return {
            "new_ids": new_ids,
            "removed_outside_count": int(removed_outside_count),
            "skipped_bones_outside_only": [int(item) for item in skipped_bones_outside_only],
        }

    @staticmethod
    def _phase1_densify_init_cfg(init_cfg: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(init_cfg)
        if bool(base_config.get("phase1", {}).get("decouple_axial_from_radial_init", False)):
            cfg["longitudinal_scale_min_ratio"] = 0.0
        scale_init_divisor = float(base_config.get("phase1", {}).get("scale_init_divisor", 1.0))
        if scale_init_divisor > 0.0 and abs(scale_init_divisor - 1.0) > 1.0e-12:
            cfg["centerline_axial_scale_factor"] = float(cfg.get("centerline_axial_scale_factor", 1.25)) / scale_init_divisor
            cfg["centerline_radial_scale_factor"] = float(cfg.get("centerline_radial_scale_factor", 0.5)) / scale_init_divisor
            cfg["centerline_min_radial_scale_ratio"] = float(cfg.get("centerline_min_radial_scale_ratio", 0.015)) / scale_init_divisor
            cfg["endpoint_extension_min_ratio"] = float(cfg.get("endpoint_extension_min_ratio", 0.08)) / scale_init_divisor
        return cfg
