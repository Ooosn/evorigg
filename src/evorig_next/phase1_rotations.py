from __future__ import annotations

import torch

from evorig_next.utils.geometry import EPS, safe_normalize
from evorig_next.utils.rotations import invert_transform, make_transform, stable_up_vector


def skew_symmetric(vectors: torch.Tensor) -> torch.Tensor:
    x, y, z = vectors.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def axis_angle_to_matrix_stable(axis_angle: torch.Tensor) -> torch.Tensor:
    theta2 = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(EPS))
    axis = axis_angle / theta.clamp_min(EPS)
    k = skew_symmetric(axis)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(axis_angle.shape[:-1] + (3, 3))

    small = theta2 <= 1.0e-8
    sin_over_theta = torch.sin(theta) / theta.clamp_min(EPS)
    one_minus_cos_over_theta2 = (1.0 - torch.cos(theta)) / theta2.clamp_min(EPS)

    # First stable terms of Rodrigues around zero.
    sin_over_theta = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, sin_over_theta)
    one_minus_cos_over_theta2 = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        one_minus_cos_over_theta2,
    )

    omega = skew_symmetric(axis_angle)
    return eye + sin_over_theta[..., None] * omega + one_minus_cos_over_theta2[..., None] * (omega @ omega)


def compute_bone_frames(rest_joints: torch.Tensor, bone_parent_idx: torch.Tensor, bone_child_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    parent_pos = rest_joints[bone_parent_idx]
    child_pos = rest_joints[bone_child_idx]
    direction = child_pos - parent_pos
    length = direction.norm(dim=-1, keepdim=True).clamp_min(EPS)
    x_axis = direction / length
    up = stable_up_vector(x_axis)
    z_axis = safe_normalize(torch.cross(x_axis, up, dim=-1))
    y_axis = safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
    frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    fallback = torch.eye(3, dtype=frame.dtype, device=frame.device).expand_as(frame)
    degenerate = (length.squeeze(-1) <= EPS).unsqueeze(-1).unsqueeze(-1)
    frame = torch.where(degenerate, fallback, frame)
    return parent_pos, frame


__all__ = [
    "axis_angle_to_matrix_stable",
    "compute_bone_frames",
    "invert_transform",
    "make_transform",
]
