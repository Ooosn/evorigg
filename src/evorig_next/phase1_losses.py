from __future__ import annotations

import math

import torch

from evorig_next.phase1_rotations import compute_bone_frames
from evorig_next.training.losses import (
    joint_inside_mesh_loss,
    posed_bone_inside_mesh_loss,
    posed_joint_inside_mesh_loss,
    temporal_smoothness_loss,
    vertex_acceleration_loss,
    vertex_recon_loss,
)
from evorig_next.training.scaling import normalize_linear_metric, normalize_squared_metric
from evorig_next.utils.geometry import EPS
from evorig_next.utils.mesh_ops import MeshQueryScene, compute_inside_shell_descriptor, default_inside_sample_directions


def illegal_support_loss(
    support: torch.Tensor,
    legal_joint_mask: torch.Tensor,
    *,
    tau: float = 0.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize illegal support by making legal support beat the illegal one.

    Wrong-coverage diagnosis uses the invalid ratio I / C.  The differentiable
    JLG loss must not use that ratio: when C is already tiny, the ratio can stay
    large even though the actual illegal influence is negligible.

    For vertices that have at least one legal joint candidate, this becomes a
    competitive loss on the strongest illegal vs strongest legal pre-normalized
    support:

        relu(max_illegal + margin - max_legal)

    When a vertex has no legal candidate at all, it falls back to the old
    illegal-mass penalty so branch growth can still receive a signal.
    """
    if support.ndim != 2:
        raise ValueError("support must have shape [J, V]")
    if legal_joint_mask.ndim != 2:
        raise ValueError("legal_joint_mask must have shape [V, J]")
    if support.shape[0] != legal_joint_mask.shape[1] or support.shape[1] != legal_joint_mask.shape[0]:
        raise ValueError("support and legal_joint_mask shapes do not match")
    support_vj = support.transpose(0, 1)
    legal_mask = legal_joint_mask.to(dtype=torch.bool, device=support_vj.device)
    illegal_mask = ~legal_mask
    illegal_mass = (support_vj * illegal_mask.to(dtype=support_vj.dtype)).sum(dim=-1)
    illegal_max = (support_vj * illegal_mask.to(dtype=support_vj.dtype)).max(dim=-1).values
    legal_max = support_vj.masked_fill(~legal_mask, float("-inf")).max(dim=-1).values
    has_legal = legal_mask.any(dim=-1)
    threshold = float(max(tau, 0.0))
    competitive = torch.relu(illegal_max + float(max(margin, 0.0)) - legal_max)
    competitive = torch.where(has_legal & (illegal_max > threshold), competitive, torch.zeros_like(competitive))
    fallback = torch.relu(illegal_mass - threshold)
    value = torch.where(has_legal, competitive, fallback)
    return value.square().mean()


def vertex_recon_topk_loss(
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    fraction: float = 0.05,
    min_count: int = 128,
    reference_length: float | None = None,
) -> torch.Tensor:
    if pred_vertices.shape != gt_vertices.shape or pred_vertices.ndim != 3:
        raise ValueError("pred_vertices and gt_vertices must share shape [T, V, 3]")
    diff = (pred_vertices - gt_vertices).norm(dim=-1)
    if mask is not None:
        if mask.shape != diff.shape:
            raise ValueError("mask must have shape [T, V]")
        values = diff[mask.to(device=diff.device, dtype=torch.bool)]
    else:
        values = diff.reshape(-1)
    if int(values.numel()) <= 0:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    requested = int(math.ceil(float(values.numel()) * max(float(fraction), 0.0)))
    k = min(int(values.numel()), max(int(min_count), requested, 1))
    value = torch.topk(values, k=k, largest=True).values.mean()
    return normalize_linear_metric(value, reference_length)


def gaussian_log_scale_anchor_loss(
    log_scale: torch.Tensor,
    init_log_scale: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if log_scale.shape != init_log_scale.shape:
        raise ValueError("log_scale and init_log_scale must share shape [G, 3]")
    if active_mask is None:
        active = torch.ones(log_scale.shape[0], dtype=torch.bool, device=log_scale.device)
    else:
        active = active_mask.to(device=log_scale.device, dtype=torch.bool)
    if not bool(active.any().item()):
        return torch.zeros((), dtype=log_scale.dtype, device=log_scale.device)
    diff = log_scale[active] - init_log_scale[active].to(device=log_scale.device, dtype=log_scale.dtype)
    return diff.square().mean()


def bone_scale_consistency_loss(
    anchor_bone: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Limit per-bone scale outliers without forcing all Gaussians equal.

    Scales on one bone are compared to the per-axis mean log-scale. Values
    within [0.5x, 2x] of that mean scale are free; only larger deviations are
    penalized.
    """
    if anchor_bone.ndim != 1 or log_scale.ndim != 2 or int(anchor_bone.shape[0]) != int(log_scale.shape[0]):
        raise ValueError("anchor_bone must be [G] and log_scale must be [G, 3]")
    if active_mask is None:
        active = torch.ones(anchor_bone.shape[0], dtype=torch.bool, device=anchor_bone.device)
    else:
        active = active_mask.to(device=anchor_bone.device, dtype=torch.bool)
    penalties = []
    log_ratio = math.log(2.0)
    for bone_id in torch.unique(anchor_bone[active]).tolist():
        idx = torch.nonzero(active & (anchor_bone == int(bone_id)), as_tuple=False).flatten()
        if int(idx.numel()) <= 1:
            continue
        values = log_scale[idx]
        axis_mean = values.mean(dim=0, keepdim=True)
        deviation = (values - axis_mean).abs()
        penalties.append(torch.relu(deviation - log_ratio).square().mean())
    if not penalties:
        return torch.zeros((), dtype=log_scale.dtype, device=log_scale.device)
    return torch.stack(penalties).mean()


def gaussian_illegal_coverage_loss(
    kernels: torch.Tensor,
    gaussian_legal_vertex_mask: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    tau: float = 0.0,
) -> torch.Tensor:
    if kernels.ndim != 2:
        raise ValueError("kernels must have shape [G, V]")
    if gaussian_legal_vertex_mask.shape != kernels.shape:
        raise ValueError("gaussian_legal_vertex_mask must have shape [G, V]")
    if active_mask is None:
        active = torch.ones(kernels.shape[0], dtype=torch.bool, device=kernels.device)
    else:
        if active_mask.shape != (kernels.shape[0],):
            raise ValueError("active_mask must have shape [G]")
        active = active_mask.to(device=kernels.device, dtype=torch.bool)
    total_mass = kernels.sum(dim=-1)
    active = active & (total_mass > float(EPS))
    if not bool(active.any().item()):
        return torch.zeros((), dtype=kernels.dtype, device=kernels.device)
    legal_mask = gaussian_legal_vertex_mask.to(device=kernels.device, dtype=torch.bool)
    illegal_mass = (kernels * (~legal_mask).to(dtype=kernels.dtype)).mean(dim=-1)
    return torch.relu(illegal_mass[active] - float(max(tau, 0.0))).square().mean()


def bone_cov_offdiag_loss(
    bone_local_covariance: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if bone_local_covariance.ndim != 3 or bone_local_covariance.shape[-2:] != (3, 3):
        raise ValueError("bone_local_covariance must have shape [G, 3, 3]")
    if active_mask is None:
        active = torch.ones(bone_local_covariance.shape[0], dtype=torch.bool, device=bone_local_covariance.device)
    else:
        if active_mask.shape != (bone_local_covariance.shape[0],):
            raise ValueError("active_mask must have shape [G]")
        active = active_mask.to(device=bone_local_covariance.device, dtype=torch.bool)
    if not bool(active.any().item()):
        return torch.zeros((), dtype=bone_local_covariance.dtype, device=bone_local_covariance.device)
    active_cov = bone_local_covariance[active]
    offdiag = active_cov - torch.diag_embed(torch.diagonal(active_cov, dim1=-2, dim2=-1))
    return offdiag.square().mean()


def bone_radial_symmetry_loss(
    log_scale: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if log_scale.ndim != 2 or log_scale.shape[-1] != 3:
        raise ValueError("log_scale must have shape [G, 3]")
    if active_mask is None:
        active = torch.ones(log_scale.shape[0], dtype=torch.bool, device=log_scale.device)
    else:
        if active_mask.shape != (log_scale.shape[0],):
            raise ValueError("active_mask must have shape [G]")
        active = active_mask.to(device=log_scale.device, dtype=torch.bool)
    if not bool(active.any().item()):
        return torch.zeros((), dtype=log_scale.dtype, device=log_scale.device)
    diff = log_scale[active, 1] - log_scale[active, 2]
    return diff.square().mean()


def bone_scale_band_loss(
    anchor_bone: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    max_axial_log_span: float,
    max_radial_log_span: float,
) -> torch.Tensor:
    if anchor_bone.ndim != 1 or log_scale.ndim != 2 or int(anchor_bone.shape[0]) != int(log_scale.shape[0]):
        raise ValueError("anchor_bone must be [G] and log_scale must be [G, 3]")
    if active_mask is None:
        active = torch.ones(anchor_bone.shape[0], dtype=torch.bool, device=anchor_bone.device)
    else:
        if active_mask.shape != (anchor_bone.shape[0],):
            raise ValueError("active_mask must have shape [G]")
        active = active_mask.to(device=anchor_bone.device, dtype=torch.bool)
    penalties = []
    axial_limit = float(max_axial_log_span)
    radial_limit = float(max_radial_log_span)
    for bone_id in torch.unique(anchor_bone[active]).tolist():
        idx = torch.nonzero(active & (anchor_bone == int(bone_id)), as_tuple=False).flatten()
        if int(idx.numel()) <= 1:
            continue
        axial = log_scale[idx, 0]
        radial = log_scale[idx, 1:3].mean(dim=-1)
        axial_span = axial.max() - axial.min()
        radial_span = radial.max() - radial.min()
        penalties.append(torch.relu(axial_span - axial_limit).square() + torch.relu(radial_span - radial_limit).square())
    if not penalties:
        return torch.zeros((), dtype=log_scale.dtype, device=log_scale.device)
    return torch.stack(penalties).mean()


def bone_radial_distance_shrink_loss(
    pred_vertices: torch.Tensor,
    posed_joints: torch.Tensor,
    vertex_bone_index: torch.Tensor,
    vertex_parent_joint: torch.Tensor,
    vertex_child_joint: torch.Tensor,
    rest_radial_distance: torch.Tensor,
    *,
    min_ratio: float,
) -> torch.Tensor:
    if pred_vertices.ndim != 3 or posed_joints.ndim != 3:
        raise ValueError("pred_vertices must be [T,V,3] and posed_joints must be [T,J,3]")
    if vertex_bone_index.ndim != 1:
        raise ValueError("vertex_bone_index must be [V]")
    valid = (vertex_bone_index >= 0) & (rest_radial_distance > EPS)
    if not bool(valid.any().item()):
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    ids = torch.nonzero(valid.to(device=pred_vertices.device), as_tuple=False).flatten()
    parent = vertex_parent_joint.to(device=pred_vertices.device, dtype=torch.long)[ids]
    child = vertex_child_joint.to(device=pred_vertices.device, dtype=torch.long)[ids]
    rest_radial = rest_radial_distance.to(device=pred_vertices.device, dtype=pred_vertices.dtype)[ids].clamp_min(EPS)
    points = pred_vertices[:, ids]
    start = posed_joints[:, parent]
    end = posed_joints[:, child]
    segment = end - start
    denom = segment.square().sum(dim=-1).clamp_min(EPS)
    lam = ((points - start) * segment).sum(dim=-1) / denom
    closest = start + lam.clamp(0.0, 1.0).unsqueeze(-1) * segment
    radial = torch.linalg.norm(points - closest, dim=-1)
    ratio = radial / rest_radial.unsqueeze(0)
    return torch.relu(float(min_ratio) - ratio).square().mean()


def mesh_edge_length_floor_loss(
    pred_vertices: torch.Tensor,
    rest_vertices: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    *,
    min_ratio: float,
) -> torch.Tensor:
    if pred_vertices.ndim != 3:
        raise ValueError("pred_vertices must have shape [T, V, 3]")
    if rest_vertices.ndim != 2:
        raise ValueError("rest_vertices must have shape [V, 3]")
    if edge_src.ndim != 1 or edge_dst.ndim != 1 or edge_src.shape != edge_dst.shape:
        raise ValueError("edge_src and edge_dst must share shape [E]")
    if edge_src.numel() == 0:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    src = edge_src.to(device=pred_vertices.device, dtype=torch.long)
    dst = edge_dst.to(device=pred_vertices.device, dtype=torch.long)
    rest = rest_vertices.to(device=pred_vertices.device, dtype=pred_vertices.dtype)
    rest_len = torch.linalg.norm(rest[src] - rest[dst], dim=-1)
    valid = rest_len > EPS
    if not bool(valid.any().item()):
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    src = src[valid]
    dst = dst[valid]
    rest_len = rest_len[valid]
    pred_len = torch.linalg.norm(pred_vertices[:, src] - pred_vertices[:, dst], dim=-1)
    ratio = pred_len / rest_len.unsqueeze(0).clamp_min(EPS)
    return torch.relu(float(min_ratio) - ratio).square().mean()


def rest_bone_length_anchor_loss(
    rest_joints: torch.Tensor,
    init_rest_joints: torch.Tensor,
    parent_idx: torch.Tensor,
) -> torch.Tensor:
    if rest_joints.ndim != 2 or init_rest_joints.shape != rest_joints.shape:
        raise ValueError("rest_joints and init_rest_joints must have shape [J, 3]")
    if parent_idx.ndim != 1 or parent_idx.shape[0] != rest_joints.shape[0]:
        raise ValueError("parent_idx must have shape [J]")
    child = torch.nonzero(parent_idx >= 0, as_tuple=False).flatten()
    if child.numel() == 0:
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    parent = parent_idx[child].to(device=rest_joints.device, dtype=torch.long)
    init = init_rest_joints.to(device=rest_joints.device, dtype=rest_joints.dtype)
    current_len = torch.linalg.norm(rest_joints[child] - rest_joints[parent], dim=-1)
    init_len = torch.linalg.norm(init[child] - init[parent], dim=-1).clamp_min(EPS)
    ratio = current_len / init_len
    return (ratio - 1.0).square().mean()


def joint_shell_anchor_consistency_loss(
    pred_vertices: torch.Tensor,
    posed_joints: torch.Tensor,
    joint_rotations: torch.Tensor,
    anchor_vertex_ids: torch.Tensor,
    anchor_vertex_weights: torch.Tensor,
    rest_local_vectors: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    joint_weight: torch.Tensor | None = None,
    distance_weight: float = 1.0,
    orthogonal_weight: float = 0.1,
    reference_length: float | None = None,
) -> torch.Tensor:
    if pred_vertices.ndim != 3:
        raise ValueError("pred_vertices must have shape [T, V, 3]")
    if posed_joints.ndim != 3 or joint_rotations.ndim != 4:
        raise ValueError("posed_joints must be [T, J, 3] and joint_rotations must be [T, J, 3, 3]")
    if anchor_vertex_ids.ndim != 3 or anchor_vertex_weights.shape != anchor_vertex_ids.shape:
        raise ValueError("anchor_vertex_ids and anchor_vertex_weights must have shape [J, K, H]")
    if rest_local_vectors.ndim != 3 or rest_local_vectors.shape[:2] != anchor_vertex_ids.shape[:2]:
        raise ValueError("rest_local_vectors must have shape [J, K, 3]")
    if valid_mask.shape != anchor_vertex_ids.shape[:2]:
        raise ValueError("valid_mask must have shape [J, K]")
    if not bool(valid_mask.any().item()):
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    safe_ids = anchor_vertex_ids.to(device=pred_vertices.device, dtype=torch.long).clamp_min(0)
    weights = anchor_vertex_weights.to(device=pred_vertices.device, dtype=pred_vertices.dtype)
    anchor_points = (pred_vertices[:, safe_ids] * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)
    relative = anchor_points - posed_joints.unsqueeze(2)
    local_relative = torch.einsum(
        "tjab,tjkb->tjka",
        joint_rotations.transpose(-1, -2),
        relative,
    )
    rest_local = rest_local_vectors.to(device=pred_vertices.device, dtype=pred_vertices.dtype)
    rest_distance = torch.linalg.norm(rest_local, dim=-1).clamp_min(EPS)
    rest_direction = rest_local / rest_distance.unsqueeze(-1)
    current_distance = (local_relative * rest_direction.unsqueeze(0)).sum(dim=-1)
    orthogonal = local_relative - current_distance.unsqueeze(-1) * rest_direction.unsqueeze(0)
    diff = (
        float(distance_weight) * (current_distance - rest_distance.unsqueeze(0)).square()
        + float(orthogonal_weight) * orthogonal.square().sum(dim=-1)
    )
    weight = valid_mask.to(device=pred_vertices.device, dtype=pred_vertices.dtype).unsqueeze(0)
    if joint_weight is not None:
        joint_scale = joint_weight.to(device=pred_vertices.device, dtype=pred_vertices.dtype).view(1, -1, 1)
        weight = weight * joint_scale
    if not bool((weight > 0).any().item()):
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    value = (diff * weight).sum() / weight.sum().clamp_min(EPS)
    return normalize_squared_metric(value, reference_length)


def bone_cross_section_consistency_loss(
    pred_vertices: torch.Tensor,
    posed_joints: torch.Tensor,
    sections: list[dict[str, torch.Tensor | float | int]],
    *,
    perimeter_weight: float,
    radius_weight: float,
) -> torch.Tensor:
    if pred_vertices.ndim != 3:
        raise ValueError("pred_vertices must have shape [T, V, 3]")
    if posed_joints.ndim != 3:
        raise ValueError("posed_joints must have shape [T, J, 3]")
    if not sections:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    bone_parent_idx = torch.tensor(
        [int(section["parent_joint"]) for section in sections],
        dtype=torch.long,
        device=pred_vertices.device,
    )
    bone_child_idx = torch.tensor(
        [int(section["child_joint"]) for section in sections],
        dtype=torch.long,
        device=pred_vertices.device,
    )
    frame_stack: list[torch.Tensor] = []
    for frame_idx in range(int(posed_joints.shape[0])):
        _parent_pos, frame = compute_bone_frames(posed_joints[frame_idx], bone_parent_idx, bone_child_idx)
        frame_stack.append(frame)
    frames = torch.stack(frame_stack, dim=0)
    penalties: list[torch.Tensor] = []
    per_w = float(perimeter_weight)
    rad_w = float(radius_weight)
    for section_idx, section in enumerate(sections):
        parent_joint = int(section["parent_joint"])
        child_joint = int(section["child_joint"])
        lambda_value = float(section["lambda_value"])
        start = posed_joints[:, parent_joint]
        end = posed_joints[:, child_joint]
        center = start + lambda_value * (end - start)
        if "node_edge_src" in section:
            node_edge_src = torch.as_tensor(section["node_edge_src"], dtype=torch.long, device=pred_vertices.device)
            node_edge_dst = torch.as_tensor(section["node_edge_dst"], dtype=torch.long, device=pred_vertices.device)
            node_edge_t = torch.as_tensor(section["node_edge_t"], dtype=pred_vertices.dtype, device=pred_vertices.device)
            if int(node_edge_src.numel()) <= 0:
                continue
            section_points = (
                (1.0 - node_edge_t).view(1, -1, 1) * pred_vertices[:, node_edge_src]
                + node_edge_t.view(1, -1, 1) * pred_vertices[:, node_edge_dst]
            )
        else:
            vertex_ids = torch.as_tensor(section["vertex_ids"], dtype=torch.long, device=pred_vertices.device)
            if int(vertex_ids.numel()) <= 0:
                continue
            section_points = pred_vertices[:, vertex_ids]
        local = torch.einsum(
            "tij,tnj->tni",
            frames[:, section_idx].transpose(-1, -2),
            section_points - center.unsqueeze(1),
        )
        radius = torch.linalg.norm(local[..., 1:3], dim=-1)
        section_penalty = torch.zeros(pred_vertices.shape[0], dtype=pred_vertices.dtype, device=pred_vertices.device)
        if rad_w > 0.0:
            rest_p50 = max(float(section["rest_radius_p50"]), float(EPS))
            rest_p90 = max(float(section["rest_radius_p90"]), float(EPS))
            radius_p50 = torch.quantile(radius, 0.5, dim=-1)
            radius_p90 = torch.quantile(radius, 0.9, dim=-1)
            section_penalty = section_penalty + rad_w * (
                (radius_p50 / rest_p50 - 1.0).square() + (radius_p90 / rest_p90 - 1.0).square()
            )
        edge_src = torch.as_tensor(
            section["segment_src"] if "segment_src" in section else section["edge_src"],
            dtype=torch.long,
            device=pred_vertices.device,
        )
        edge_dst = torch.as_tensor(
            section["segment_dst"] if "segment_dst" in section else section["edge_dst"],
            dtype=torch.long,
            device=pred_vertices.device,
        )
        if per_w > 0.0 and int(edge_src.numel()) > 0 and float(section["rest_perimeter"]) > float(EPS):
            perimeter = torch.linalg.norm(section_points[:, edge_src] - section_points[:, edge_dst], dim=-1).sum(dim=-1)
            section_penalty = section_penalty + per_w * (perimeter / float(section["rest_perimeter"]) - 1.0).square()
        penalties.append(section_penalty.mean())
    if not penalties:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    return torch.stack(penalties).mean()


def cross_pose_section_consistency_loss(
    pred_vertices: torch.Tensor,
    posed_joints: torch.Tensor,
    joint_rotations: torch.Tensor,
    sections: list[dict[str, torch.Tensor | float | int]],
    *,
    perimeter_weight: float,
    joint_distance_weight: float,
    joint_direction_weight: float,
    joint_lambdas: tuple[float, ...] = (0.2, 0.8),
    joint_lambda_tol: float = 1.0e-4,
    reference_vertices: torch.Tensor | None = None,
    reference_joints: torch.Tensor | None = None,
    reference_joint_rotations: torch.Tensor | None = None,
    balance_reference: bool = False,
    topk_fraction: float = 1.0,
) -> torch.Tensor:
    if pred_vertices.ndim != 3:
        raise ValueError("pred_vertices must have shape [T, V, 3]")
    if posed_joints.ndim != 3 or joint_rotations.ndim != 4:
        raise ValueError("posed_joints must be [T, J, 3] and joint_rotations must be [T, J, 3, 3]")
    if not sections:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    if reference_vertices is not None:
        if reference_vertices.ndim == 2:
            reference_vertices_batch = reference_vertices.unsqueeze(0)
        elif reference_vertices.ndim == 3:
            reference_vertices_batch = reference_vertices
        else:
            raise ValueError("reference_vertices must have shape [V, 3] or [T_ref, V, 3]")
        if bool(balance_reference) and int(reference_vertices_batch.shape[0]) == 1 and int(pred_vertices.shape[0]) > 1:
            reference_vertices_batch = reference_vertices_batch.expand(int(pred_vertices.shape[0]), -1, -1)
        pred_vertices_all = torch.cat(
            [reference_vertices_batch.to(device=pred_vertices.device, dtype=pred_vertices.dtype), pred_vertices],
            dim=0,
        )
        if reference_joints is None or reference_joint_rotations is None:
            raise ValueError("reference_joints and reference_joint_rotations are required with reference_vertices")
        if reference_joints.ndim != 3 or reference_joint_rotations.ndim != 4:
            raise ValueError("reference_joints must be [T_ref, J, 3] and reference_joint_rotations must be [T_ref, J, 3, 3]")
        if bool(balance_reference) and int(reference_joints.shape[0]) == 1 and int(pred_vertices.shape[0]) > 1:
            reference_joints = reference_joints.expand(int(pred_vertices.shape[0]), -1, -1)
            reference_joint_rotations = reference_joint_rotations.expand(int(pred_vertices.shape[0]), -1, -1, -1)
        posed_joints_all = torch.cat(
            [reference_joints.to(device=posed_joints.device, dtype=posed_joints.dtype), posed_joints],
            dim=0,
        )
        joint_rotations_all = torch.cat(
            [
                reference_joint_rotations.to(device=joint_rotations.device, dtype=joint_rotations.dtype),
                joint_rotations,
            ],
            dim=0,
        )
    else:
        pred_vertices_all = pred_vertices
        posed_joints_all = posed_joints
        joint_rotations_all = joint_rotations
    if pred_vertices_all.shape[0] <= 1:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    section_penalties: list[torch.Tensor] = []
    per_w = float(perimeter_weight)
    dist_w = float(joint_distance_weight)
    dir_w = float(joint_direction_weight)
    joint_lambda_values = tuple(float(item) for item in joint_lambdas)
    lambda_tol = max(float(joint_lambda_tol), 0.0)
    for section in sections:
        terms: list[torch.Tensor] = []
        if "node_edge_src" in section:
            node_edge_src = torch.as_tensor(section["node_edge_src"], dtype=torch.long, device=pred_vertices.device)
            node_edge_dst = torch.as_tensor(section["node_edge_dst"], dtype=torch.long, device=pred_vertices.device)
            node_edge_t = torch.as_tensor(section["node_edge_t"], dtype=pred_vertices.dtype, device=pred_vertices.device)
            if int(node_edge_src.numel()) <= 0:
                continue
            section_points = (
                (1.0 - node_edge_t).view(1, -1, 1) * pred_vertices_all[:, node_edge_src]
                + node_edge_t.view(1, -1, 1) * pred_vertices_all[:, node_edge_dst]
            )
        else:
            vertex_ids = torch.as_tensor(section["vertex_ids"], dtype=torch.long, device=pred_vertices.device)
            if int(vertex_ids.numel()) <= 0:
                continue
            section_points = pred_vertices_all[:, vertex_ids]
        if per_w > 0.0:
            edge_src = torch.as_tensor(
                section["segment_src"] if "segment_src" in section else section["edge_src"],
                dtype=torch.long,
                device=pred_vertices.device,
            )
            edge_dst = torch.as_tensor(
                section["segment_dst"] if "segment_dst" in section else section["edge_dst"],
                dtype=torch.long,
                device=pred_vertices.device,
            )
            if int(edge_src.numel()) > 0:
                perimeter = torch.linalg.norm(section_points[:, edge_src] - section_points[:, edge_dst], dim=-1).sum(dim=-1)
                perimeter_mean = perimeter.mean().detach().clamp_min(EPS)
                terms.append(per_w * (perimeter / perimeter_mean - 1.0).square().mean())
        lambda_value = float(section["lambda_value"])
        use_joint_section = any(abs(lambda_value - item) <= lambda_tol for item in joint_lambda_values)
        if use_joint_section and (dist_w > 0.0 or dir_w > 0.0):
            anchor_joint = int(section["anchor_joint"])
            relative = section_points - posed_joints_all[:, anchor_joint].unsqueeze(1)
            local_relative = torch.einsum(
                "tab,tnb->tna",
                joint_rotations_all[:, anchor_joint].transpose(-1, -2),
                relative,
            )
            distance = torch.linalg.norm(local_relative, dim=-1).clamp_min(EPS)
            if dist_w > 0.0:
                distance_mean = distance.mean(dim=0, keepdim=True).detach().clamp_min(EPS)
                terms.append(dist_w * (distance / distance_mean - 1.0).square().mean())
            if dir_w > 0.0:
                direction = local_relative / distance.unsqueeze(-1)
                mean_direction = direction.mean(dim=0, keepdim=True)
                mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True).clamp_min(EPS)
                cosine = (direction * mean_direction.detach()).sum(dim=-1).clamp(-1.0, 1.0)
                terms.append(dir_w * (1.0 - cosine).mean())
        if terms:
            section_penalties.append(torch.stack(terms).sum())
    if not section_penalties:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    values = torch.stack(section_penalties)
    fraction = float(topk_fraction)
    if 0.0 < fraction < 1.0 and int(values.numel()) > 1:
        k = max(1, int(math.ceil(float(values.numel()) * fraction)))
        return torch.topk(values, k=k, largest=True).values.mean()
    return values.mean()


def joint_side_section_consistency_loss(
    pred_vertices: torch.Tensor,
    posed_joints: torch.Tensor,
    joint_rotations: torch.Tensor,
    sections: list[dict[str, torch.Tensor | float | int]],
    *,
    lambda_threshold: float = 0.15,
    centroid_weight: float,
    distance_weight: float,
    reference_length: float | None = None,
) -> torch.Tensor:
    if pred_vertices.ndim != 3:
        raise ValueError("pred_vertices must have shape [T, V, 3]")
    if posed_joints.ndim != 3 or joint_rotations.ndim != 4:
        raise ValueError("posed_joints must be [T, J, 3] and joint_rotations must be [T, J, 3, 3]")
    if not sections:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    penalties: list[torch.Tensor] = []
    threshold = max(0.0, min(float(lambda_threshold), 0.5))
    centroid_w = float(centroid_weight)
    distance_w = float(distance_weight)
    for section in sections:
        lambda_value = float(section["lambda_value"])
        if lambda_value > threshold and lambda_value < 1.0 - threshold:
            continue
        node_edge_src = torch.as_tensor(section["node_edge_src"], dtype=torch.long, device=pred_vertices.device)
        node_edge_dst = torch.as_tensor(section["node_edge_dst"], dtype=torch.long, device=pred_vertices.device)
        node_edge_t = torch.as_tensor(section["node_edge_t"], dtype=pred_vertices.dtype, device=pred_vertices.device)
        if int(node_edge_src.numel()) <= 0:
            continue
        anchor_joint = int(section["anchor_joint"])
        section_points = (
            (1.0 - node_edge_t).view(1, -1, 1) * pred_vertices[:, node_edge_src]
            + node_edge_t.view(1, -1, 1) * pred_vertices[:, node_edge_dst]
        )
        relative = section_points - posed_joints[:, anchor_joint].unsqueeze(1)
        local_relative = torch.einsum(
            "tab,tnb->tna",
            joint_rotations[:, anchor_joint].transpose(-1, -2),
            relative,
        )
        section_penalty = torch.zeros(pred_vertices.shape[0], dtype=pred_vertices.dtype, device=pred_vertices.device)
        if centroid_w > 0.0:
            rest_centroid = torch.as_tensor(section["rest_joint_centroid_local"], dtype=pred_vertices.dtype, device=pred_vertices.device)
            centroid = local_relative.mean(dim=1)
            section_penalty = section_penalty + centroid_w * (centroid - rest_centroid.unsqueeze(0)).square().sum(dim=-1)
        if distance_w > 0.0:
            distance = torch.linalg.norm(local_relative, dim=-1)
            rest_p50 = max(float(section["rest_joint_distance_p50"]), float(EPS))
            rest_p90 = max(float(section["rest_joint_distance_p90"]), float(EPS))
            dist_p50 = torch.quantile(distance, 0.5, dim=-1)
            dist_p90 = torch.quantile(distance, 0.9, dim=-1)
            section_penalty = section_penalty + distance_w * (
                (dist_p50 / rest_p50 - 1.0).square() + (dist_p90 / rest_p90 - 1.0).square()
            )
        penalties.append(section_penalty.mean())
    if not penalties:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    return normalize_squared_metric(torch.stack(penalties).mean(), reference_length)


def _bone_section_edges(
    parent_idx: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    parents = torch.as_tensor(parent_idx, dtype=torch.long, device=device).reshape(-1)
    child_ids = torch.nonzero(parents >= 0, as_tuple=False).reshape(-1)
    if int(child_ids.numel()) <= 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    anchor_ids = parents[child_ids]
    return anchor_ids, child_ids


def _bone_section_points_and_directions(
    posed_joints_frame: torch.Tensor,
    joint_rotations_frame: torch.Tensor,
    anchor_ids: torch.Tensor,
    child_ids: torch.Tensor,
    *,
    section_lambda: float,
    direction_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(anchor_ids.numel()) <= 0 or int(direction_count) <= 0:
        empty_points = torch.zeros(0, 3, dtype=posed_joints_frame.dtype, device=posed_joints_frame.device)
        empty_dirs = torch.zeros(
            0,
            max(int(direction_count), 0),
            3,
            dtype=posed_joints_frame.dtype,
            device=posed_joints_frame.device,
        )
        return empty_points, empty_dirs

    lam = max(0.0, min(float(section_lambda), 1.0))
    anchors = posed_joints_frame[anchor_ids]
    children = posed_joints_frame[child_ids]
    axis = children - anchors
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(EPS)
    probe_points = anchors + lam * (children - anchors)

    anchor_rot = joint_rotations_frame[anchor_ids]
    ref = anchor_rot[:, :, 0]
    ref = ref - axis * (ref * axis).sum(dim=-1, keepdim=True)
    ref_norm = ref.norm(dim=-1, keepdim=True)
    alt = anchor_rot[:, :, 1]
    alt = alt - axis * (alt * axis).sum(dim=-1, keepdim=True)
    alt_norm = alt.norm(dim=-1, keepdim=True)
    ref = torch.where(ref_norm > 1.0e-6, ref, alt)
    ref_norm = torch.where(ref_norm > 1.0e-6, ref_norm, alt_norm)

    fallback = torch.zeros_like(ref)
    fallback[:, 0] = 1.0
    fallback = fallback - axis * (fallback * axis).sum(dim=-1, keepdim=True)
    fallback_norm = fallback.norm(dim=-1, keepdim=True)
    fallback_y = torch.zeros_like(ref)
    fallback_y[:, 1] = 1.0
    fallback_y = fallback_y - axis * (fallback_y * axis).sum(dim=-1, keepdim=True)
    fallback_y_norm = fallback_y.norm(dim=-1, keepdim=True)
    fallback = torch.where(fallback_norm > 1.0e-6, fallback, fallback_y)
    fallback_norm = torch.where(fallback_norm > 1.0e-6, fallback_norm, fallback_y_norm)
    ref = torch.where(ref_norm > 1.0e-6, ref, fallback)
    ref_norm = torch.where(ref_norm > 1.0e-6, ref_norm, fallback_norm)
    ref = ref / ref_norm.clamp_min(EPS)
    tangent = torch.cross(axis, ref, dim=-1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(EPS)

    angles = torch.arange(int(direction_count), dtype=posed_joints_frame.dtype, device=posed_joints_frame.device)
    angles = angles * (2.0 * math.pi / float(max(int(direction_count), 1)))
    directions = (
        torch.cos(angles).view(1, -1, 1) * ref.unsqueeze(1)
        + torch.sin(angles).view(1, -1, 1) * tangent.unsqueeze(1)
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return probe_points, directions


def pose_consistent_joint_shell_loss(
    posed_joints: torch.Tensor,
    joint_rotations: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    shell_descriptors: list[dict[str, torch.Tensor]] | None = None,
    reference_length: float | None = None,
    parent_idx: torch.Tensor | None = None,
    section_lambda: float = 0.1,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if posed_joints.ndim != 3 or joint_rotations.ndim != 4:
        raise ValueError("posed_joints must be [T, J, 3] and joint_rotations must be [T, J, 3, 3]")
    if posed_vertices.ndim != 3 or posed_vertices.shape[0] != posed_joints.shape[0]:
        raise ValueError("posed_vertices must be [T, V, 3] and share frame count")
    if joint_rotations.shape[:2] != posed_joints.shape[:2]:
        raise ValueError("joint_rotations must match posed_joints frames and joints")
    if posed_joints.shape[0] <= 1:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)

    if shell_descriptors is not None and len(shell_descriptors) != int(posed_joints.shape[0]):
        raise ValueError("shell_descriptors must match the posed frame count")
    edge_anchor_ids: torch.Tensor | None = None
    edge_child_ids: torch.Tensor | None = None
    if parent_idx is not None:
        edge_anchor_ids, edge_child_ids = _bone_section_edges(parent_idx, device=posed_joints.device)
    elif shell_descriptors is not None and len(shell_descriptors) > 0:
        first = shell_descriptors[0]
        if "anchor_joint_ids" in first and "child_joint_ids" in first:
            edge_anchor_ids = torch.as_tensor(first["anchor_joint_ids"], dtype=torch.long, device=posed_joints.device)
            edge_child_ids = torch.as_tensor(first["child_joint_ids"], dtype=torch.long, device=posed_joints.device)

    edge_mode = edge_anchor_ids is not None and edge_child_ids is not None and int(edge_anchor_ids.numel()) > 0
    local_dirs = None
    point_stack: list[torch.Tensor] = []
    margin_stack: list[torch.Tensor] = []
    center_delta_stack: list[torch.Tensor] = []
    valid_stack: list[torch.Tensor] = []
    signed_dir_stack: list[torch.Tensor] = []
    frame_dir_stack: list[torch.Tensor] = []
    for frame_idx in range(int(posed_joints.shape[0])):
        frame_dirs = None
        if edge_mode:
            assert edge_anchor_ids is not None and edge_child_ids is not None
            sample_points, frame_dirs = _bone_section_points_and_directions(
                posed_joints[frame_idx],
                joint_rotations[frame_idx],
                edge_anchor_ids,
                edge_child_ids,
                section_lambda=section_lambda,
                direction_count=direction_count,
            )
            if shell_descriptors is None:
                frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
                with torch.no_grad():
                    shell = compute_inside_shell_descriptor(
                        sample_points.detach(),
                        posed_vertices[frame_idx],
                        faces,
                        inward_hint=posed_vertices[frame_idx].mean(dim=0, keepdim=True).expand(sample_points.shape[0], -1),
                        padding=surface_tol,
                        surface_tol=surface_tol,
                        direction_count=direction_count,
                        directions=frame_dirs,
                        mesh_query_scene=frame_scene,
                    )
            else:
                shell = shell_descriptors[frame_idx]
                maybe_dirs = shell.get("directions")
                if isinstance(maybe_dirs, torch.Tensor) and maybe_dirs.shape == (
                    sample_points.shape[0],
                    direction_count,
                    3,
                ):
                    frame_dirs = maybe_dirs.to(device=posed_joints.device, dtype=posed_joints.dtype)
        elif shell_descriptors is None:
            if local_dirs is None:
                local_dirs = default_inside_sample_directions(
                    direction_count,
                    dtype=posed_joints.dtype,
                    device=posed_joints.device,
                )
            frame_dirs = torch.einsum("jab,kb->jka", joint_rotations[frame_idx], local_dirs)
            frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
            with torch.no_grad():
                shell = compute_inside_shell_descriptor(
                    posed_joints[frame_idx].detach(),
                    posed_vertices[frame_idx],
                    faces,
                    inward_hint=posed_vertices[frame_idx].mean(dim=0, keepdim=True).expand(posed_joints.shape[1], -1),
                    padding=surface_tol,
                    surface_tol=surface_tol,
                    direction_count=direction_count,
                    directions=frame_dirs,
                    mesh_query_scene=frame_scene,
                )
        else:
            shell = shell_descriptors[frame_idx]
            maybe_dirs = shell.get("directions")
            if isinstance(maybe_dirs, torch.Tensor) and maybe_dirs.shape == (
                posed_joints.shape[1],
                direction_count,
                3,
            ):
                frame_dirs = maybe_dirs.to(device=posed_joints.device, dtype=posed_joints.dtype)
        if frame_dirs is None:
            if local_dirs is None:
                local_dirs = default_inside_sample_directions(
                    direction_count,
                    dtype=posed_joints.dtype,
                    device=posed_joints.device,
                )
            frame_dirs = torch.einsum("jab,kb->jka", joint_rotations[frame_idx], local_dirs)
        with torch.no_grad():
            forward_is_closer = shell["forward_distance"] <= shell["backward_distance"]
            signed_surface_dir = torch.where(
                forward_is_closer.unsqueeze(-1),
                frame_dirs,
                -frame_dirs,
            )
            center_delta = 0.5 * (shell["forward_distance"] - shell["backward_distance"])
        if edge_mode:
            point_stack.append(sample_points)
        margin_stack.append(shell["margin"])
        center_delta_stack.append(center_delta)
        valid_stack.append(shell["valid_pairs"] & shell["inside_mask"].unsqueeze(-1))
        signed_dir_stack.append(signed_surface_dir)
        frame_dir_stack.append(frame_dirs)

    margins = torch.stack(margin_stack, dim=0)
    valid = torch.stack(valid_stack, dim=0)
    if not bool(valid.any().item()):
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    valid_f = valid.to(margins.dtype)
    safe_margins = torch.where(valid, margins, torch.zeros_like(margins))
    mean_margin = safe_margins.sum(dim=0) / valid_f.sum(dim=0).clamp_min(1.0)
    if edge_mode:
        reference_points = torch.stack(point_stack, dim=0)
    else:
        reference_points = posed_joints
    with torch.no_grad():
        signed_dirs = torch.stack(signed_dir_stack, dim=0)
        frame_dirs_all = torch.stack(frame_dir_stack, dim=0)
        center_delta_all = torch.stack(center_delta_stack, dim=0)
        delta = torch.where(valid, margins - mean_margin.unsqueeze(0), torch.zeros_like(margins))
        target_points = reference_points.detach().unsqueeze(2) + signed_dirs * delta.unsqueeze(-1)
        target_points = torch.where(valid.unsqueeze(-1), target_points, reference_points.detach().unsqueeze(2))
        center_delta_all = torch.where(valid, center_delta_all, torch.zeros_like(center_delta_all))
        center_targets = reference_points.detach().unsqueeze(2) + frame_dirs_all * center_delta_all.unsqueeze(-1)
        center_targets = torch.where(valid.unsqueeze(-1), center_targets, reference_points.detach().unsqueeze(2))
    deviation = torch.where(
        valid,
        (reference_points.unsqueeze(2) - target_points.detach()).square().sum(dim=-1),
        torch.zeros_like(margins),
    )
    center_deviation = torch.where(
        valid,
        (reference_points.unsqueeze(2) - center_targets.detach()).square().sum(dim=-1),
        torch.zeros_like(margins),
    )
    per_item = (deviation + center_deviation).sum(dim=(0, 2)) / valid_f.sum(dim=(0, 2)).clamp_min(1.0)
    if joint_weight is not None:
        weight = joint_weight.to(device=per_item.device, dtype=per_item.dtype)
        if edge_mode:
            assert edge_anchor_ids is not None and edge_child_ids is not None
            weight = torch.maximum(weight[edge_anchor_ids], weight[edge_child_ids])
        value = (per_item * weight).sum() / weight.sum().clamp_min(EPS)
    else:
        value = per_item.mean()
    return normalize_squared_metric(value, reference_length)


def posed_joint_shell_descriptors(
    posed_joints: torch.Tensor,
    joint_rotations: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    parent_idx: torch.Tensor | None = None,
    section_lambda: float = 0.1,
) -> list[dict[str, torch.Tensor]]:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0:
        return []
    if posed_joints.ndim != 3 or joint_rotations.ndim != 4:
        raise ValueError("posed_joints must be [T, J, 3] and joint_rotations must be [T, J, 3, 3]")
    if posed_vertices.ndim != 3 or posed_vertices.shape[0] != posed_joints.shape[0]:
        raise ValueError("posed_vertices must be [T, V, 3] and share frame count")
    if joint_rotations.shape[:2] != posed_joints.shape[:2]:
        raise ValueError("joint_rotations must match posed_joints frames and joints")
    edge_anchor_ids = None
    edge_child_ids = None
    if parent_idx is not None:
        edge_anchor_ids, edge_child_ids = _bone_section_edges(parent_idx, device=posed_joints.device)
    edge_mode = edge_anchor_ids is not None and edge_child_ids is not None and int(edge_anchor_ids.numel()) > 0
    local_dirs = None
    if not edge_mode:
        local_dirs = default_inside_sample_directions(
            direction_count,
            dtype=posed_joints.dtype,
            device=posed_joints.device,
        )
    descriptors: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for frame_idx in range(int(posed_joints.shape[0])):
            if edge_mode:
                assert edge_anchor_ids is not None and edge_child_ids is not None
                sample_points, frame_dirs = _bone_section_points_and_directions(
                    posed_joints[frame_idx],
                    joint_rotations[frame_idx],
                    edge_anchor_ids,
                    edge_child_ids,
                    section_lambda=section_lambda,
                    direction_count=direction_count,
                )
            else:
                assert local_dirs is not None
                frame_dirs = torch.einsum("jab,kb->jka", joint_rotations[frame_idx], local_dirs)
                sample_points = posed_joints[frame_idx].detach()
            frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
            descriptor = compute_inside_shell_descriptor(
                sample_points.detach(),
                posed_vertices[frame_idx],
                faces,
                inward_hint=posed_vertices[frame_idx].mean(dim=0, keepdim=True).expand(sample_points.shape[0], -1),
                padding=surface_tol,
                surface_tol=surface_tol,
                direction_count=direction_count,
                directions=frame_dirs,
                mesh_query_scene=frame_scene,
            )
            descriptor = dict(descriptor)
            descriptor["directions"] = frame_dirs
            if edge_mode:
                assert edge_anchor_ids is not None and edge_child_ids is not None
                descriptor["anchor_joint_ids"] = edge_anchor_ids
                descriptor["child_joint_ids"] = edge_child_ids
            descriptors.append(descriptor)
    return descriptors


def posed_joint_surface_clearance_loss(
    posed_joints: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    min_clearance: float,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    joint_rotations: torch.Tensor | None = None,
    direction_count: int = 12,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0 or float(min_clearance) <= 0.0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if posed_joints.ndim != 3:
        raise ValueError("posed_joints must have shape [T, J, 3]")
    if posed_vertices.ndim != 3 or posed_vertices.shape[0] != posed_joints.shape[0]:
        raise ValueError("posed_vertices must be [T, V, 3] and share frame count")
    if joint_rotations is not None:
        if joint_rotations.ndim != 4 or joint_rotations.shape[:2] != posed_joints.shape[:2] or joint_rotations.shape[-2:] != (3, 3):
            raise ValueError("joint_rotations must have shape [T, J, 3, 3] matching posed_joints")

    local_dirs = default_inside_sample_directions(
        direction_count,
        dtype=posed_joints.dtype,
        device=posed_joints.device,
    )
    penalties: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for frame_idx in range(int(posed_joints.shape[0])):
        with torch.no_grad():
            if joint_rotations is None:
                frame_dirs = local_dirs.unsqueeze(0).expand(posed_joints.shape[1], -1, -1)
            else:
                frame_dirs = torch.einsum("jab,kb->jka", joint_rotations[frame_idx], local_dirs)
            frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
            shell = compute_inside_shell_descriptor(
                posed_joints[frame_idx].detach(),
                posed_vertices[frame_idx],
                faces,
                inward_hint=posed_vertices[frame_idx].mean(dim=0, keepdim=True).expand(posed_joints.shape[1], -1),
                padding=max(float(surface_tol), float(min_clearance)),
                surface_tol=surface_tol,
                direction_count=direction_count,
                directions=frame_dirs,
                mesh_query_scene=frame_scene,
            )
            outside_mask = ~shell["inside_mask"]
            valid = shell["valid_pairs"]
            margin = shell["margin"]
            active = valid & (margin < float(min_clearance))
            if not bool((active.any(dim=-1) | outside_mask).any().item()):
                continue
            forward_is_closer = shell["forward_distance"] <= shell["backward_distance"]
            signed_surface_dir = torch.where(
                forward_is_closer.unsqueeze(-1),
                frame_dirs,
                -frame_dirs,
            )
            surface_distance = torch.where(
                forward_is_closer,
                shell["forward_distance"],
                shell["backward_distance"],
            )
            surface_points = shell["sample_points"].unsqueeze(1) + signed_surface_dir * surface_distance.unsqueeze(-1)
            target_points = surface_points - signed_surface_dir * float(min_clearance)
            inf_margin = torch.full_like(margin, float("inf"))
            closest_margin = torch.where(active, margin, inf_margin)
            closest_idx = closest_margin.argmin(dim=-1)
            point_active = torch.isfinite(closest_margin.min(dim=-1).values)
            gather_idx = closest_idx.view(-1, 1, 1).expand(-1, 1, 3)
            target_points = torch.gather(target_points, dim=1, index=gather_idx).squeeze(1)
            target_points = torch.where(outside_mask.unsqueeze(-1), shell["sample_points"], target_points)
            point_active = point_active | outside_mask

        diff_sq = (posed_joints[frame_idx] - target_points.detach()).square().sum(dim=-1)
        active_weight = point_active.to(dtype=posed_joints.dtype, device=posed_joints.device)
        if joint_weight is not None:
            active_weight = active_weight * joint_weight.to(device=posed_joints.device, dtype=posed_joints.dtype)
        penalties.append(diff_sq)
        weights.append(active_weight)

    if not penalties:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    penalty = torch.cat([item.reshape(-1) for item in penalties], dim=0)
    weight = torch.cat([item.reshape(-1) for item in weights], dim=0)
    valid_weight = weight > 1.0e-8
    if not bool(valid_weight.any().item()):
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    value = (penalty[valid_weight] * weight[valid_weight]).sum() / weight[valid_weight].sum().clamp_min(EPS)
    return normalize_squared_metric(value, reference_length)


def rest_joint_surface_clearance_loss(
    rest_joints: torch.Tensor,
    rest_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    min_clearance: float,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    mesh_query_scene: MeshQueryScene | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or rest_joints.numel() == 0 or float(min_clearance) <= 0.0:
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    if rest_joints.ndim != 2 or rest_joints.shape[-1] != 3:
        raise ValueError("rest_joints must have shape [J, 3]")
    with torch.no_grad():
        shell = compute_inside_shell_descriptor(
            rest_joints.detach(),
            rest_vertices,
            faces,
            inward_hint=rest_vertices.mean(dim=0, keepdim=True).expand(rest_joints.shape[0], -1),
            padding=max(float(surface_tol), float(min_clearance)),
            surface_tol=surface_tol,
            direction_count=direction_count,
            mesh_query_scene=mesh_query_scene,
        )
        outside_mask = ~shell["inside_mask"]
        valid = shell["valid_pairs"]
        margin = shell["margin"]
        active = valid & (margin < float(min_clearance))
        if not bool((active.any(dim=-1) | outside_mask).any().item()):
            return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
        dirs = default_inside_sample_directions(
            direction_count,
            dtype=rest_joints.dtype,
            device=rest_joints.device,
        ).unsqueeze(0).expand(rest_joints.shape[0], -1, -1)
        forward_is_closer = shell["forward_distance"] <= shell["backward_distance"]
        signed_surface_dir = torch.where(forward_is_closer.unsqueeze(-1), dirs, -dirs)
        surface_distance = torch.where(forward_is_closer, shell["forward_distance"], shell["backward_distance"])
        surface_points = shell["sample_points"].unsqueeze(1) + signed_surface_dir * surface_distance.unsqueeze(-1)
        target_points = surface_points - signed_surface_dir * float(min_clearance)
        inf_margin = torch.full_like(margin, float("inf"))
        closest_margin = torch.where(active, margin, inf_margin)
        closest_idx = closest_margin.argmin(dim=-1)
        point_active = torch.isfinite(closest_margin.min(dim=-1).values)
        gather_idx = closest_idx.view(-1, 1, 1).expand(-1, 1, 3)
        target_points = torch.gather(target_points, dim=1, index=gather_idx).squeeze(1)
        target_points = torch.where(outside_mask.unsqueeze(-1), shell["sample_points"], target_points)
        point_active = point_active | outside_mask

    diff_sq = (rest_joints - target_points.detach()).square().sum(dim=-1)
    weight = point_active.to(dtype=rest_joints.dtype, device=rest_joints.device)
    if joint_weight is not None:
        weight = weight * joint_weight.to(device=rest_joints.device, dtype=rest_joints.dtype)
    valid_weight = weight > 1.0e-8
    if not bool(valid_weight.any().item()):
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    value = (diff_sq[valid_weight] * weight[valid_weight]).sum() / weight[valid_weight].sum().clamp_min(EPS)
    return normalize_squared_metric(value, reference_length)


__all__ = [
    "bone_cov_offdiag_loss",
    "bone_radial_symmetry_loss",
    "bone_scale_band_loss",
    "bone_scale_consistency_loss",
    "bone_radial_distance_shrink_loss",
    "gaussian_illegal_coverage_loss",
    "gaussian_log_scale_anchor_loss",
    "illegal_support_loss",
    "cross_pose_section_consistency_loss",
    "joint_side_section_consistency_loss",
    "joint_shell_anchor_consistency_loss",
    "joint_inside_mesh_loss",
    "pose_consistent_joint_shell_loss",
    "posed_joint_shell_descriptors",
    "rest_joint_surface_clearance_loss",
    "bone_cross_section_consistency_loss",
    "posed_bone_inside_mesh_loss",
    "posed_joint_inside_mesh_loss",
    "posed_joint_surface_clearance_loss",
    "temporal_smoothness_loss",
    "vertex_acceleration_loss",
    "vertex_recon_topk_loss",
    "vertex_recon_loss",
]
