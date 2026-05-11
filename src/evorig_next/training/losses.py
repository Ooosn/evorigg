from __future__ import annotations

import torch

from evorig_next.training.scaling import normalize_linear_metric, normalize_squared_metric
from evorig_next.utils.geometry import EPS
from evorig_next.utils.mesh_ops import (
    MeshQueryScene,
    closest_point_on_mesh,
    compute_inside_shell_descriptor,
    default_inside_sample_directions,
    points_inside_or_on_mesh,
    project_points_inside_mesh,
)


def _weighted_positive_violation_mean(
    violation: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight is None:
        valid = violation > 1.0e-12
        if not torch.any(valid):
            return torch.zeros((), dtype=violation.dtype, device=violation.device)
        return violation[valid].mean()
    weights = weight.to(violation.device, violation.dtype)
    valid = (weights > 1.0e-8) & (violation > 1.0e-12)
    if not torch.any(valid):
        return torch.zeros((), dtype=violation.dtype, device=violation.device)
    normalized = weights[valid] / weights[valid].sum().clamp_min(1.0e-8)
    return (violation[valid] * normalized).sum()


def _inside_margin_violation(
    margin: torch.Tensor,
    valid_pairs: torch.Tensor,
    inside_mask: torch.Tensor,
    target_margin: float,
) -> torch.Tensor:
    if margin.ndim != 2 or valid_pairs.shape != margin.shape:
        raise ValueError("margin and valid_pairs must have shape [N, K]")
    shortfall = (float(target_margin) - margin).clamp_min(0.0)
    shortfall = torch.where(valid_pairs, shortfall, torch.zeros_like(shortfall))
    per_point = shortfall.sum(dim=-1) / valid_pairs.sum(dim=-1).clamp_min(1)
    return per_point * inside_mask.to(per_point.dtype)


def vertex_recon_loss(
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reference_length: float | None = None,
) -> torch.Tensor:
    diff = (pred_vertices - gt_vertices).norm(dim=-1)
    if mask is not None:
        diff = diff * mask
        value = diff.sum() / mask.sum().clamp_min(1.0)
    else:
        value = diff.mean()
    return normalize_linear_metric(value, reference_length)


def temporal_smoothness_loss(
    pose_rot: torch.Tensor,
    root_trans: torch.Tensor,
    *,
    root_reference_length: float | None = None,
) -> torch.Tensor:
    if pose_rot.shape[0] <= 1:
        return torch.zeros((), dtype=pose_rot.dtype, device=pose_rot.device)
    pose_term = (pose_rot[1:] - pose_rot[:-1]).square().mean()
    root_term = (root_trans[1:] - root_trans[:-1]).square().mean()
    return pose_term + normalize_squared_metric(root_term, root_reference_length)


def vertex_acceleration_loss(
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reference_length: float | None = None,
) -> torch.Tensor:
    if pred_vertices.shape[0] <= 2:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    pred_acc = pred_vertices[2:] - 2.0 * pred_vertices[1:-1] + pred_vertices[:-2]
    gt_acc = gt_vertices[2:] - 2.0 * gt_vertices[1:-1] + gt_vertices[:-2]
    diff = (pred_acc - gt_acc).norm(dim=-1)
    if mask is not None:
        if mask.ndim == 1:
            active = mask.reshape(1, -1).expand(diff.shape[0], -1)
        else:
            active = mask[1:-1]
        active = active.to(device=diff.device, dtype=diff.dtype)
        value = (diff * active).sum() / active.sum().clamp_min(1.0)
    else:
        value = diff.mean()
    return normalize_linear_metric(value, reference_length)


def gaussian_ownership_anchor_loss(
    q_logits: torch.Tensor,
    target_assignment: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q_logits.shape != target_assignment.shape:
        raise ValueError("q_logits and target_assignment must share shape [G, J]")
    if q_logits.ndim != 2:
        raise ValueError("q_logits and target_assignment must be rank-2 tensors")
    target = target_assignment.to(dtype=q_logits.dtype, device=q_logits.device)
    if active_mask is None:
        active = torch.ones(q_logits.shape[0], dtype=torch.bool, device=q_logits.device)
    else:
        if active_mask.shape != (q_logits.shape[0],):
            raise ValueError("active_mask must have shape [G]")
        active = active_mask.to(device=q_logits.device, dtype=torch.bool)
    active = active & (target.sum(dim=-1) > 1.0e-8)
    if not bool(active.any().item()):
        return torch.zeros((), dtype=q_logits.dtype, device=q_logits.device)
    target = target[active]
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(EPS)
    log_soft = torch.log_softmax(q_logits[active], dim=-1)
    log_target = torch.log(target.clamp_min(EPS))
    return (target * (log_target - log_soft)).sum(dim=-1).mean()


def joint_incident_control_loss(
    weights: torch.Tensor,
    assigned_bone_index: torch.Tensor,
    bone_parent_idx: torch.Tensor,
    bone_child_idx: torch.Tensor,
    *,
    target_mean_weight: float = 0.25,
    eligible_joint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("weights must have shape [V, J]")
    if assigned_bone_index.ndim != 1 or assigned_bone_index.shape[0] != weights.shape[0]:
        raise ValueError("assigned_bone_index must have shape [V]")
    if bone_parent_idx.shape != bone_child_idx.shape or bone_parent_idx.ndim != 1:
        raise ValueError("bone_parent_idx and bone_child_idx must share shape [B]")
    joint_count = int(weights.shape[1])
    if eligible_joint_mask is None:
        eligible = torch.ones(joint_count, dtype=torch.bool, device=weights.device)
    else:
        if eligible_joint_mask.shape != (joint_count,):
            raise ValueError("eligible_joint_mask must have shape [J]")
        eligible = eligible_joint_mask.to(device=weights.device, dtype=torch.bool)
    target_value = float(max(target_mean_weight, 0.0))
    penalties: list[torch.Tensor] = []
    for joint_id in range(joint_count):
        if not bool(eligible[joint_id].item()):
            continue
        incident_bones = (bone_parent_idx == joint_id) | (bone_child_idx == joint_id)
        if not bool(incident_bones.any().item()):
            continue
        incident_vertex_mask = incident_bones[assigned_bone_index]
        if not bool(incident_vertex_mask.any().item()):
            continue
        local_mean = weights[incident_vertex_mask, joint_id].mean()
        penalties.append(torch.relu(local_mean.new_tensor(target_value) - local_mean).square())
    if not penalties:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    return torch.stack(penalties).mean()


def vertex_joint_spill_loss(
    weights: torch.Tensor,
    allowed_joint_mask: torch.Tensor,
    *,
    illegal_joint_penalty: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("weights must have shape [V, J]")
    if allowed_joint_mask.shape != weights.shape:
        raise ValueError("allowed_joint_mask must have shape [V, J]")
    allowed = allowed_joint_mask.to(device=weights.device, dtype=torch.bool)
    illegal_mass = weights * (~allowed).to(weights.dtype)
    if illegal_joint_penalty is not None:
        if illegal_joint_penalty.shape != weights.shape:
            raise ValueError("illegal_joint_penalty must have shape [V, J]")
        penalty = illegal_joint_penalty.to(device=weights.device, dtype=weights.dtype).clamp_min(0.0)
        illegal_mass = illegal_mass * penalty
    per_vertex = illegal_mass.sum(dim=-1)
    if not bool((per_vertex > 1.0e-12).any().item()):
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    return per_vertex.square().mean()


def gaussian_bone_lambda_span_loss(
    lambda_param: torch.Tensor,
    anchor_bone: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    bone_count: int,
    lambda_min: torch.Tensor | None = None,
    lambda_max: torch.Tensor | None = None,
    target_fraction: float = 0.6,
    eligible_bone_mask: torch.Tensor | None = None,
    min_active_gaussians: int = 1,
) -> torch.Tensor:
    if lambda_param.ndim != 1 or anchor_bone.ndim != 1 or active_mask.ndim != 1:
        raise ValueError("lambda_param, anchor_bone, and active_mask must have shape [G]")
    if lambda_param.shape != anchor_bone.shape or lambda_param.shape != active_mask.shape:
        raise ValueError("lambda_param, anchor_bone, and active_mask must share shape [G]")
    if lambda_param.numel() == 0 or int(bone_count) <= 0:
        return torch.zeros((), dtype=lambda_param.dtype, device=lambda_param.device)
    if eligible_bone_mask is None:
        eligible = torch.ones(int(bone_count), dtype=torch.bool, device=lambda_param.device)
    else:
        if eligible_bone_mask.shape != (int(bone_count),):
            raise ValueError("eligible_bone_mask must have shape [B]")
        eligible = eligible_bone_mask.to(device=lambda_param.device, dtype=torch.bool)
    if lambda_min is None:
        lambda_min = torch.zeros_like(lambda_param)
    if lambda_max is None:
        lambda_max = torch.ones_like(lambda_param)
    target_fraction = float(max(target_fraction, 0.0))
    min_active_gaussians = max(int(min_active_gaussians), 1)
    penalties: list[torch.Tensor] = []
    for bone_idx in range(int(bone_count)):
        if not bool(eligible[bone_idx].item()):
            continue
        bone_mask = active_mask & (anchor_bone == bone_idx)
        bone_count_active = int(bone_mask.sum().item())
        if bone_count_active < min_active_gaussians:
            continue
        bone_lambda = lambda_param[bone_mask]
        available_span = (lambda_max[bone_mask].max() - lambda_min[bone_mask].min()).clamp_min(EPS)
        current_span = (bone_lambda.max() - bone_lambda.min()).clamp_min(0.0)
        target_span = available_span * target_fraction
        penalties.append(((target_span - current_span).clamp_min(0.0) / available_span).square())
    if not penalties:
        return torch.zeros((), dtype=lambda_param.dtype, device=lambda_param.device)
    return torch.stack(penalties).mean()


def skeleton_anchor_loss(
    rest_joints: torch.Tensor,
    init_rest_joints: torch.Tensor,
    joint_weight: torch.Tensor | None = None,
    *,
    reference_length: float | None = None,
) -> torch.Tensor:
    distance_sq = (rest_joints - init_rest_joints).square().sum(dim=-1)
    if joint_weight is None:
        value = distance_sq.mean()
        return normalize_squared_metric(value, reference_length)
    weighted = joint_weight.to(distance_sq.device, distance_sq.dtype)
    valid = weighted > 1.0e-8
    if not torch.any(valid):
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    weights = weighted[valid] / weighted[valid].sum().clamp_min(1.0e-8)
    value = (distance_sq[valid] * weights).sum()
    return normalize_squared_metric(value, reference_length)


def joint_inside_mesh_loss(
    rest_joints: torch.Tensor,
    rest_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    margin_target: float | None = None,
    mesh_query_scene: MeshQueryScene | None = None,
    *,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or rest_joints.numel() == 0:
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    if margin_target is None:
        margin_target = float(surface_tol)
    with torch.no_grad():
        shell = compute_inside_shell_descriptor(
            rest_joints.detach(),
            rest_vertices,
            faces,
            inward_hint=rest_vertices.mean(dim=0, keepdim=True).expand(rest_joints.shape[0], -1),
            padding=surface_tol,
            surface_tol=surface_tol,
            direction_count=direction_count,
            mesh_query_scene=mesh_query_scene,
        )
    outside_violation = (rest_joints - shell["sample_points"]).square().sum(dim=-1)
    margin_violation = _inside_margin_violation(
        shell["margin"],
        shell["valid_pairs"],
        shell["inside_mask"],
        float(margin_target),
    ).square()
    violation = outside_violation + margin_violation
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    value = _weighted_positive_violation_mean(violation, joint_weight)
    return normalize_squared_metric(value, reference_length)


def joint_surface_clearance_loss(
    rest_joints: torch.Tensor,
    rest_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    min_clearance: float,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    mesh_query_scene: MeshQueryScene | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or rest_joints.numel() == 0 or float(min_clearance) <= 0.0:
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    with torch.no_grad():
        inside_mask = points_inside_or_on_mesh(
            rest_joints.detach(),
            rest_vertices,
            faces,
            surface_tol=surface_tol,
            mesh_query_scene=mesh_query_scene,
        )
    if not bool(inside_mask.any().item()):
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    _, dist_sq = closest_point_on_mesh(rest_joints, rest_vertices, faces, mesh_query_scene=mesh_query_scene)
    distance = dist_sq.sqrt()
    violation = (float(min_clearance) - distance).clamp_min(0.0).square()
    effective_weight = inside_mask.to(rest_joints.dtype)
    if joint_weight is not None:
        effective_weight = effective_weight * joint_weight.to(rest_joints.dtype)
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=rest_joints.dtype, device=rest_joints.device)
    value = _weighted_positive_violation_mean(violation, effective_weight)
    return normalize_squared_metric(value, reference_length)


def posed_joint_inside_mesh_loss(
    posed_joints: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    joint_rotations: torch.Tensor | None = None,
    direction_count: int = 12,
    margin_target: float | None = None,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    shell_descriptors: list[dict[str, torch.Tensor]] | None = None,
    *,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if posed_joints.ndim != 3:
        raise ValueError("posed_joints must have shape [T, J, 3]")
    if posed_vertices.ndim != 3:
        raise ValueError("posed_vertices must have shape [T, V, 3]")
    if posed_joints.shape[0] != posed_vertices.shape[0]:
        raise ValueError("posed_joints and posed_vertices must share the same frame count")
    if joint_rotations is not None:
        if joint_rotations.ndim != 4 or joint_rotations.shape[:2] != posed_joints.shape[:2] or joint_rotations.shape[-2:] != (3, 3):
            raise ValueError("joint_rotations must have shape [T, J, 3, 3] matching posed_joints")
    if margin_target is None:
        margin_target = float(surface_tol)

    frame_violations: list[torch.Tensor] = []
    local_dirs = default_inside_sample_directions(direction_count, dtype=posed_joints.dtype, device=posed_joints.device)
    for frame_idx in range(int(posed_joints.shape[0])):
        frame_mesh = posed_vertices[frame_idx]
        frame_joints = posed_joints[frame_idx]
        if shell_descriptors is None:
            frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
            with torch.no_grad():
                frame_dirs = None
                if joint_rotations is not None and int(local_dirs.shape[0]) > 0:
                    frame_dirs = torch.einsum("jab,kb->jka", joint_rotations[frame_idx], local_dirs)
                shell = compute_inside_shell_descriptor(
                    frame_joints.detach(),
                    frame_mesh,
                    faces,
                    inward_hint=frame_mesh.mean(dim=0, keepdim=True).expand(frame_joints.shape[0], -1),
                    padding=surface_tol,
                    surface_tol=surface_tol,
                    direction_count=direction_count,
                    directions=frame_dirs,
                    mesh_query_scene=frame_scene,
                )
        else:
            shell = shell_descriptors[frame_idx]
        outside_violation = (frame_joints - shell["sample_points"]).norm(dim=-1)
        margin_violation = _inside_margin_violation(
            shell["margin"],
            shell["valid_pairs"],
            shell["inside_mask"],
            float(margin_target),
        )
        frame_violations.append(outside_violation + margin_violation)
    violation = torch.stack(frame_violations, dim=0)
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    expanded_weight = None
    if joint_weight is not None:
        expanded_weight = joint_weight.reshape(1, -1).expand_as(violation)
    value = _weighted_positive_violation_mean(violation.reshape(-1), None if expanded_weight is None else expanded_weight.reshape(-1))
    return normalize_linear_metric(value, reference_length)


def posed_joint_surface_clearance_loss(
    posed_joints: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    min_clearance: float,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0 or float(min_clearance) <= 0.0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if posed_joints.ndim != 3:
        raise ValueError("posed_joints must have shape [T, J, 3]")
    if posed_vertices.ndim != 3:
        raise ValueError("posed_vertices must have shape [T, V, 3]")
    if posed_joints.shape[0] != posed_vertices.shape[0]:
        raise ValueError("posed_joints and posed_vertices must share the same frame count")

    frame_violations: list[torch.Tensor] = []
    frame_weights: list[torch.Tensor] = []
    for frame_idx in range(int(posed_joints.shape[0])):
        frame_mesh = posed_vertices[frame_idx]
        frame_joints = posed_joints[frame_idx]
        frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
        with torch.no_grad():
            inside_mask = points_inside_or_on_mesh(
                frame_joints.detach(),
                frame_mesh,
                faces,
                surface_tol=surface_tol,
                mesh_query_scene=frame_scene,
            )
        if not bool(inside_mask.any().item()):
            frame_violations.append(torch.zeros(frame_joints.shape[0], dtype=posed_joints.dtype, device=posed_joints.device))
            frame_weights.append(torch.zeros(frame_joints.shape[0], dtype=posed_joints.dtype, device=posed_joints.device))
            continue
        _, dist_sq = closest_point_on_mesh(frame_joints, frame_mesh, faces, mesh_query_scene=frame_scene)
        distance = dist_sq.sqrt()
        frame_violations.append((float(min_clearance) - distance).clamp_min(0.0).square())
        base_weight = inside_mask.to(posed_joints.dtype)
        if joint_weight is not None:
            base_weight = base_weight * joint_weight.to(posed_joints.dtype)
        frame_weights.append(base_weight)
    violation = torch.stack(frame_violations, dim=0)
    weight = torch.stack(frame_weights, dim=0)
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    value = _weighted_positive_violation_mean(violation.reshape(-1), weight.reshape(-1))
    return normalize_squared_metric(value, reference_length)


def posed_joint_interior_consistency_loss(
    posed_joints: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    joint_rotations: torch.Tensor | None = None,
    joint_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    shell_descriptors: list[dict[str, torch.Tensor]] | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if posed_joints.ndim != 3:
        raise ValueError("posed_joints must have shape [T, J, 3]")
    if posed_vertices.ndim != 3:
        raise ValueError("posed_vertices must have shape [T, V, 3]")
    if posed_joints.shape[0] != posed_vertices.shape[0]:
        raise ValueError("posed_joints and posed_vertices must share the same frame count")
    if posed_joints.shape[0] <= 1:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if joint_rotations is not None:
        if joint_rotations.ndim != 4 or joint_rotations.shape[:2] != posed_joints.shape[:2] or joint_rotations.shape[-2:] != (3, 3):
            raise ValueError("joint_rotations must have shape [T, J, 3, 3] matching posed_joints")

    local_dirs = default_inside_sample_directions(direction_count, dtype=posed_joints.dtype, device=posed_joints.device)
    frame_balance: list[torch.Tensor] = []
    frame_valid: list[torch.Tensor] = []
    for frame_idx in range(int(posed_joints.shape[0])):
        frame_mesh = posed_vertices[frame_idx]
        frame_joints = posed_joints[frame_idx]
        if shell_descriptors is None:
            frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
            with torch.no_grad():
                frame_dirs = None
                if int(local_dirs.shape[0]) > 0:
                    if joint_rotations is not None:
                        frame_dirs = torch.einsum("jab,kb->jka", joint_rotations[frame_idx], local_dirs)
                    else:
                        frame_dirs = local_dirs.unsqueeze(0).expand(frame_joints.shape[0], -1, -1)
                shell = compute_inside_shell_descriptor(
                    frame_joints.detach(),
                    frame_mesh,
                    faces,
                    inward_hint=frame_mesh.mean(dim=0, keepdim=True).expand(frame_joints.shape[0], -1),
                    padding=surface_tol,
                    surface_tol=surface_tol,
                    direction_count=direction_count,
                    directions=frame_dirs,
                    mesh_query_scene=frame_scene,
                )
        else:
            shell = shell_descriptors[frame_idx]
        frame_balance.append(shell["balance"])
        frame_valid.append(shell["valid_pairs"])
    balance = torch.stack(frame_balance, dim=0)
    valid = torch.stack(frame_valid, dim=0)
    weights = valid.to(balance.dtype)
    mean_balance = (balance * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
    deviation_sq = (balance - mean_balance.unsqueeze(0)).square() * weights
    deviation_sq = deviation_sq.sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    if bool((deviation_sq <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    expanded_weight = None
    if joint_weight is not None:
        expanded_weight = joint_weight.reshape(1, -1).expand_as(deviation_sq)
    value = _weighted_positive_violation_mean(
        deviation_sq.reshape(-1),
        None if expanded_weight is None else expanded_weight.reshape(-1),
    )
    return normalize_squared_metric(value, reference_length)


def posed_bone_inside_mesh_loss(
    posed_joints: torch.Tensor,
    parent_idx: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    samples_per_bone: int = 4,
    bone_weight: torch.Tensor | None = None,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    margin_target: float | None = None,
    mesh_query_scenes: list[MeshQueryScene | None] | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or posed_joints.numel() == 0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if posed_joints.ndim != 3:
        raise ValueError("posed_joints must have shape [T, J, 3]")
    if posed_vertices.ndim != 3:
        raise ValueError("posed_vertices must have shape [T, V, 3]")
    if posed_joints.shape[0] != posed_vertices.shape[0]:
        raise ValueError("posed_joints and posed_vertices must share the same frame count")
    child_ids = torch.nonzero(parent_idx >= 0, as_tuple=False).flatten()
    if child_ids.numel() == 0 or samples_per_bone <= 0:
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    if margin_target is None:
        margin_target = float(surface_tol)
    parent_ids = parent_idx[child_ids]
    ts = torch.linspace(
        0.0,
        1.0,
        int(samples_per_bone) + 2,
        dtype=posed_joints.dtype,
        device=posed_joints.device,
    )[1:-1]

    frame_violations: list[torch.Tensor] = []
    for frame_idx in range(int(posed_joints.shape[0])):
        frame_mesh = posed_vertices[frame_idx]
        frame_joints = posed_joints[frame_idx]
        frame_scene = None if mesh_query_scenes is None else mesh_query_scenes[frame_idx]
        start = frame_joints[parent_ids]
        end = frame_joints[child_ids]
        probes = start.unsqueeze(1) * (1.0 - ts.view(1, -1, 1)) + end.unsqueeze(1) * ts.view(1, -1, 1)
        probes = probes.reshape(-1, 3)
        with torch.no_grad():
            shell = compute_inside_shell_descriptor(
                probes.detach(),
                frame_mesh,
                faces,
                inward_hint=frame_mesh.mean(dim=0, keepdim=True).expand(probes.shape[0], -1),
                padding=surface_tol,
                surface_tol=surface_tol,
                direction_count=direction_count,
                mesh_query_scene=frame_scene,
            )
        outside_violation = (probes - shell["sample_points"]).norm(dim=-1)
        margin_violation = _inside_margin_violation(
            shell["margin"],
            shell["valid_pairs"],
            shell["inside_mask"],
            float(margin_target),
        )
        frame_violations.append((outside_violation + margin_violation).reshape(child_ids.shape[0], int(samples_per_bone)))
    violation = torch.stack(frame_violations, dim=0)
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=posed_joints.dtype, device=posed_joints.device)
    expanded_weight = None
    if bone_weight is not None:
        expanded_weight = bone_weight.reshape(1, -1, 1).expand_as(violation)
    value = _weighted_positive_violation_mean(violation.reshape(-1), None if expanded_weight is None else expanded_weight.reshape(-1))
    return normalize_linear_metric(value, reference_length)


def bone_direction_consistency_loss(
    rest_joints: torch.Tensor,
    parent_idx: torch.Tensor,
    rest_vertices: torch.Tensor,
    weights: torch.Tensor,
    *,
    bone_weight: torch.Tensor | None = None,
    anisotropy_threshold: float = 0.1,
) -> torch.Tensor:
    if rest_joints.numel() == 0 or rest_vertices.numel() == 0 or weights.numel() == 0:
        return torch.zeros((), dtype=rest_vertices.dtype, device=rest_vertices.device)
    if weights.ndim != 2 or int(weights.shape[0]) != int(rest_vertices.shape[0]) or int(weights.shape[1]) != int(rest_joints.shape[0]):
        raise ValueError("weights must have shape [V, J] matching rest_vertices and rest_joints")
    child_ids = torch.nonzero(parent_idx >= 0, as_tuple=False).flatten()
    if child_ids.numel() == 0:
        return torch.zeros((), dtype=rest_vertices.dtype, device=rest_vertices.device)
    parent_ids = parent_idx[child_ids]
    bone_dir = rest_joints[child_ids] - rest_joints[parent_ids]
    bone_dir = bone_dir / bone_dir.norm(dim=-1, keepdim=True).clamp_min(EPS)
    penalties: list[torch.Tensor] = []
    penalty_weights: list[torch.Tensor] = []
    for bone_idx, child_joint in enumerate(child_ids.tolist()):
        vertex_weight = weights[:, child_joint].clamp_min(0.0)
        total_weight = vertex_weight.sum()
        if float(total_weight.item()) <= 1.0e-6:
            continue
        centroid = (vertex_weight.unsqueeze(-1) * rest_vertices).sum(dim=0) / total_weight.clamp_min(EPS)
        centered = rest_vertices - centroid
        weighted_centered = centered * vertex_weight.unsqueeze(-1).sqrt()
        if int(weighted_centered.shape[0]) < 2:
            continue
        cov = centered.transpose(0, 1) @ (centered * vertex_weight.unsqueeze(-1)) / total_weight.clamp_min(EPS)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        principal_dir = bone_dir[bone_idx]
        for _ in range(6):
            principal_dir = cov @ principal_dir
            principal_dir = principal_dir / principal_dir.norm().clamp_min(EPS)
        max_eig = (principal_dir * (cov @ principal_dir)).sum().clamp_min(EPS)
        trace = torch.diagonal(cov).sum().clamp_min(EPS)
        mean_other = ((trace - max_eig) / 2.0).clamp_min(0.0)
        anisotropy = ((max_eig - mean_other) / max_eig).clamp_min(0.0)
        if float(anisotropy.item()) <= float(anisotropy_threshold):
            continue
        align_penalty = 1.0 - (bone_dir[bone_idx] * principal_dir).sum().abs().clamp(0.0, 1.0)
        penalties.append(align_penalty)
        weight_value = anisotropy
        if bone_weight is not None:
            weight_value = weight_value * bone_weight[bone_idx].to(weight_value.dtype)
        penalty_weights.append(weight_value)
    if not penalties:
        return torch.zeros((), dtype=rest_vertices.dtype, device=rest_vertices.device)
    penalty_tensor = torch.stack(penalties)
    weight_tensor = torch.stack(penalty_weights).clamp_min(0.0)
    valid = weight_tensor > 1.0e-8
    if not bool(valid.any().item()):
        return torch.zeros((), dtype=rest_vertices.dtype, device=rest_vertices.device)
    normalized = weight_tensor[valid] / weight_tensor[valid].sum().clamp_min(EPS)
    return (penalty_tensor[valid] * normalized).sum()


def gaussian_topk_isolation_loss(
    weights: torch.Tensor,
    *,
    keep_topk: int = 2,
    max_extra_mass: float = 0.05,
    vertex_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights.numel() == 0:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    if weights.ndim != 2:
        raise ValueError("weights must have shape [V, J]")
    keep_topk = max(int(keep_topk), 1)
    if int(weights.shape[1]) <= keep_topk:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    topk_mass = torch.topk(weights, k=keep_topk, dim=-1).values.sum(dim=-1)
    extra_mass = (weights.sum(dim=-1) - topk_mass).clamp_min(0.0)
    violation = (extra_mass - float(max_extra_mass)).clamp_min(0.0)
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    return _weighted_positive_violation_mean(violation, vertex_weight)


def gaussian_top2_isolation_loss(
    weights: torch.Tensor,
    *,
    max_third_mass: float = 0.05,
    vertex_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return gaussian_topk_isolation_loss(
        weights,
        keep_topk=2,
        max_extra_mass=max_third_mass,
        vertex_weight=vertex_weight,
    )


def gaussian_connectivity_loss(
    weights: torch.Tensor,
    edge_index: torch.Tensor | None,
    *,
    activation_threshold: float = 0.05,
) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0 or weights.numel() == 0:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    if weights.ndim != 2:
        raise ValueError("weights must have shape [V, J]")
    if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
        raise ValueError("edge_index must have shape [E, 2]")
    vertex_count, joint_count = int(weights.shape[0]), int(weights.shape[1])
    if vertex_count == 0 or joint_count == 0:
        return torch.zeros((), dtype=weights.dtype, device=weights.device)

    def _extract_components_from_active_mask(
        active_mask_cpu: torch.Tensor,
        edge_index_cpu: torch.Tensor,
    ) -> list[torch.Tensor]:
        active_vertex_ids = torch.nonzero(active_mask_cpu, as_tuple=False).flatten()
        if int(active_vertex_ids.numel()) == 0:
            return []
        if int(active_vertex_ids.numel()) == 1:
            return [active_vertex_ids]
        local_index = {int(vertex_id): local_id for local_id, vertex_id in enumerate(active_vertex_ids.tolist())}
        parent = list(range(int(active_vertex_ids.numel())))

        def _find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def _union(left_node: int, right_node: int) -> None:
            left_root = _find(left_node)
            right_root = _find(right_node)
            if left_root != right_root:
                parent[right_root] = left_root

        active_edge_mask = active_mask_cpu[edge_index_cpu[:, 0]] & active_mask_cpu[edge_index_cpu[:, 1]]
        active_edges = edge_index_cpu[active_edge_mask]
        for left_vertex, right_vertex in active_edges.tolist():
            _union(local_index[int(left_vertex)], local_index[int(right_vertex)])

        grouped: dict[int, list[int]] = {}
        for local_id, vertex_id in enumerate(active_vertex_ids.tolist()):
            root = _find(local_id)
            grouped.setdefault(root, []).append(int(vertex_id))
        return [
            torch.tensor(sorted(component_vertex_ids), dtype=torch.long)
            for component_vertex_ids in grouped.values()
        ]

    edge_index_cpu = edge_index.detach().cpu()
    weights_cpu = weights.detach().cpu().clamp_min(0.0)
    peak_per_joint = weights_cpu.max(dim=0).values.clamp_min(EPS)
    total_penalty = torch.zeros((), dtype=weights.dtype, device=weights.device)
    total_mass = torch.zeros((), dtype=weights.dtype, device=weights.device)

    for joint_idx in range(joint_count):
        joint_weight = weights[:, joint_idx].clamp_min(0.0)
        joint_total_mass = joint_weight.sum()
        if float(joint_total_mass.item()) <= float(EPS):
            continue
        peak_value = float(peak_per_joint[joint_idx].item())
        with torch.no_grad():
            active_mask_cpu = weights_cpu[:, joint_idx] >= (float(activation_threshold) * peak_value)
            components_cpu = _extract_components_from_active_mask(active_mask_cpu, edge_index_cpu)
        if len(components_cpu) <= 1:
            continue
        component_masses = [
            joint_weight.index_select(0, component_vertex_ids.to(device=weights.device)).sum()
            for component_vertex_ids in components_cpu
        ]
        component_mass_tensor = torch.stack(component_masses)
        main_component_mass = component_mass_tensor.max()
        stray_mass = (component_mass_tensor.sum() - main_component_mass).clamp_min(0.0)
        if float(stray_mass.item()) <= float(EPS):
            continue
        total_penalty = total_penalty + stray_mass
        total_mass = total_mass + joint_total_mass

    if float(total_mass.item()) <= float(EPS):
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    return total_penalty / total_mass.clamp_min(EPS)


def joint_incident_centering_loss(
    pred_vertices: torch.Tensor,
    gt_vertices: torch.Tensor,
    assigned_bone_index: torch.Tensor,
    bone_parent_idx: torch.Tensor,
    bone_child_idx: torch.Tensor,
    *,
    joint_count: int,
    eligible_joint_mask: torch.Tensor | None = None,
    min_incident_vertices: int = 8,
) -> torch.Tensor:
    """Penalize the mean displacement of each joint's incident-bone region.

    For each eligible joint j, computes mean(pred_v - gt_v) over the vertices
    assigned to j's incident bones across all frames, then penalises
    ||mean_disp||^2.  Because we average *before* taking the norm, random
    per-vertex LBS noise cancels out and only systematic pose-level offsets
    (the whole region moving in one direction) contribute to the loss.
    Gradients flow back through LBS/FK to the joint's pose rotation parameters.
    """
    if pred_vertices.ndim != 3:
        raise ValueError("pred_vertices must have shape [F, V, 3]")
    if gt_vertices.shape != pred_vertices.shape:
        raise ValueError("gt_vertices must match pred_vertices shape [F, V, 3]")
    if assigned_bone_index.ndim != 1 or assigned_bone_index.shape[0] != pred_vertices.shape[1]:
        raise ValueError("assigned_bone_index must have shape [V]")
    if bone_parent_idx.shape != bone_child_idx.shape or bone_parent_idx.ndim != 1:
        raise ValueError("bone_parent_idx and bone_child_idx must share shape [B]")
    joint_count_val = int(joint_count)
    if joint_count_val <= 0:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    if eligible_joint_mask is None:
        eligible = torch.ones(joint_count_val, dtype=torch.bool, device=pred_vertices.device)
    else:
        if eligible_joint_mask.shape != (joint_count_val,):
            raise ValueError("eligible_joint_mask must have shape [J]")
        eligible = eligible_joint_mask.to(device=pred_vertices.device, dtype=torch.bool)
    diff = pred_vertices - gt_vertices  # [F, V, 3]
    penalties: list[torch.Tensor] = []
    for joint_id in range(joint_count_val):
        if not bool(eligible[joint_id].item()):
            continue
        incident_bones = (bone_parent_idx == joint_id) | (bone_child_idx == joint_id)
        if not bool(incident_bones.any().item()):
            continue
        incident_vertex_mask = incident_bones[assigned_bone_index]
        incident_count = int(incident_vertex_mask.sum().item())
        if incident_count < int(min_incident_vertices):
            continue
        # mean displacement over incident vertices per frame, then penalise magnitude
        mean_disp = diff[:, incident_vertex_mask, :].mean(dim=1)  # [F, 3]
        penalties.append((mean_disp ** 2).sum(dim=-1).mean())     # scalar
    if not penalties:
        return torch.zeros((), dtype=pred_vertices.dtype, device=pred_vertices.device)
    return torch.stack(penalties).mean()


def gaussian_inside_mesh_loss(
    centers: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    active_mask: torch.Tensor,
    rest_vertices: torch.Tensor,
    faces: torch.Tensor | None,
    *,
    sigma_radius: float = 1.0,
    surface_tol: float = 3.0e-3,
    direction_count: int = 12,
    center_weight: float = 0.25,
    mesh_query_scene: MeshQueryScene | None = None,
    reference_length: float | None = None,
) -> torch.Tensor:
    if faces is None or faces.numel() == 0 or centers.numel() == 0:
        return torch.zeros((), dtype=centers.dtype, device=centers.device)
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("centers must have shape [G, 3]")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [G, 3, 3]")
    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError("scales must have shape [G, 3]")
    if active_mask.ndim != 1 or int(active_mask.shape[0]) != int(centers.shape[0]):
        raise ValueError("active_mask must have shape [G]")
    active_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return torch.zeros((), dtype=centers.dtype, device=centers.device)

    active_centers = centers[active_ids]
    active_rotations = rotations[active_ids]
    active_scales = scales[active_ids].clamp_min(EPS)
    directions = default_inside_sample_directions(
        int(direction_count),
        dtype=centers.dtype,
        device=centers.device,
    )
    axis_directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=centers.dtype,
        device=centers.device,
    )
    if int(directions.shape[0]) > 0:
        directions = torch.cat([directions, axis_directions], dim=0)
    else:
        directions = axis_directions
    if int(directions.shape[0]) == 0:
        return torch.zeros((), dtype=centers.dtype, device=centers.device)

    shell_local = directions.unsqueeze(0) * active_scales.unsqueeze(1) * float(sigma_radius)
    shell_world = torch.einsum("gij,gkj->gki", active_rotations, shell_local) + active_centers.unsqueeze(1)
    flat_shell = shell_world.reshape(-1, 3)
    shell_hint = active_centers[:, None, :].expand(-1, directions.shape[0], -1).reshape(-1, 3)
    center_hint = rest_vertices.mean(dim=0, keepdim=True).expand(active_centers.shape[0], -1)

    with torch.no_grad():
        projected_centers, _, _ = project_points_inside_mesh(
            active_centers.detach(),
            rest_vertices,
            faces,
            inward_hint=center_hint,
            padding=surface_tol,
            mesh_query_scene=mesh_query_scene,
        )
        projected_shell, _, _ = project_points_inside_mesh(
            flat_shell.detach(),
            rest_vertices,
            faces,
            inward_hint=shell_hint.detach(),
            padding=surface_tol,
            mesh_query_scene=mesh_query_scene,
        )

    center_violation = (active_centers - projected_centers).norm(dim=-1)
    shell_violation = (
        (flat_shell - projected_shell)
        .norm(dim=-1)
        .reshape(active_centers.shape[0], directions.shape[0])
        .amax(dim=-1)
    )
    center_weight = min(max(float(center_weight), 0.0), 1.0)
    violation = center_weight * center_violation + (1.0 - center_weight) * shell_violation
    if bool((violation <= 1.0e-12).all().item()):
        return torch.zeros((), dtype=centers.dtype, device=centers.device)
    value = violation.mean()
    return normalize_linear_metric(value, reference_length)
