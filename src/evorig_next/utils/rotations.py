from __future__ import annotations

import math

import torch

from .geometry import EPS, safe_normalize


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


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(EPS)
    k = skew_symmetric(axis)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(axis_angle.shape[:-1] + (3, 3))
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * k + (1.0 - cos) * (k @ k)


def quaternion_normalize(quaternion: torch.Tensor) -> torch.Tensor:
    return safe_normalize(quaternion, dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion_normalize(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix_to_quaternion expects [..., 3, 3], got {tuple(matrix.shape)}")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    qw = 0.5 * q_abs[..., 0]
    qx = 0.5 * q_abs[..., 1] * torch.sign(matrix[..., 2, 1] - matrix[..., 1, 2] + EPS)
    qy = 0.5 * q_abs[..., 2] * torch.sign(matrix[..., 0, 2] - matrix[..., 2, 0] + EPS)
    qz = 0.5 * q_abs[..., 3] * torch.sign(matrix[..., 1, 0] - matrix[..., 0, 1] + EPS)
    return quaternion_normalize(torch.stack([qw, qx, qy, qz], dim=-1))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    small = angle.abs() < 1.0e-5
    axis = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    axis = safe_normalize(axis)
    result = axis * angle.unsqueeze(-1)
    if small.any():
        result = torch.where(small.unsqueeze(-1), torch.zeros_like(result), result)
    return result


def rotation_between_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = safe_normalize(source)
    target = safe_normalize(target)
    v = torch.cross(source, target, dim=-1)
    c = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    s = v.norm(dim=-1, keepdim=True)
    k = skew_symmetric(v / s.clamp_min(EPS))
    eye = torch.eye(3, dtype=source.dtype, device=source.device).expand(source.shape[:-1] + (3, 3))
    rotation = eye + k * s[..., None] + (k @ k) * (1.0 - c)[..., None]
    opposite = c.squeeze(-1) < -0.999
    if opposite.any():
        alt_axis = torch.tensor([1.0, 0.0, 0.0], dtype=source.dtype, device=source.device).expand_as(source)
        alt_axis = torch.where((source[..., 0].abs() > 0.9).unsqueeze(-1), torch.tensor([0.0, 1.0, 0.0], dtype=source.dtype, device=source.device).expand_as(source), alt_axis)
        ortho = safe_normalize(torch.cross(source, alt_axis, dim=-1))
        pi_rot = axis_angle_to_matrix(ortho * math.pi)
        rotation = torch.where(opposite[..., None, None], pi_rot, rotation)
    return rotation


def make_transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    batch_shape = rotation.shape[:-2]
    transform = torch.eye(4, dtype=rotation.dtype, device=rotation.device).expand(batch_shape + (4, 4)).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    return transform


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inv_rotation = rotation.transpose(-1, -2)
    inv_translation = -(inv_rotation @ translation.unsqueeze(-1)).squeeze(-1)
    return make_transform(inv_rotation, inv_translation)


def stable_up_vector(direction: torch.Tensor) -> torch.Tensor:
    up = torch.tensor([0.0, 1.0, 0.0], dtype=direction.dtype, device=direction.device).expand_as(direction)
    alt = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device).expand_as(direction)
    use_alt = direction[..., 1].abs() > 0.9
    return torch.where(use_alt.unsqueeze(-1), alt, up)
