from __future__ import annotations

import torch

from evorig_next.utils.geometry import EPS, assert_shape


def fit_rigid_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert_shape(source, (None, 3), "source")
    assert_shape(target, (source.shape[0], 3), "target")
    if weights is None:
        weights = torch.ones(source.shape[0], dtype=source.dtype, device=source.device)
    weights = weights / weights.sum().clamp_min(EPS)
    src_center = (source * weights.unsqueeze(-1)).sum(dim=0)
    tgt_center = (target * weights.unsqueeze(-1)).sum(dim=0)
    src_zero = source - src_center
    tgt_zero = target - tgt_center
    cov = src_zero.transpose(0, 1) @ (tgt_zero * weights.unsqueeze(-1))
    u, _, vh = torch.linalg.svd(cov)
    rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    if torch.linalg.det(rotation) < 0:
        vh[-1] = -vh[-1]
        rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
    translation = tgt_center - rotation @ src_center
    aligned = (rotation @ source.transpose(0, 1)).transpose(0, 1) + translation
    error = (aligned - target).norm(dim=-1).mean()
    return rotation, translation, error


def fit_rigid_sequence(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert_shape(source, (None, None, 3), "source")
    assert_shape(target, (source.shape[0], source.shape[1], 3), "target")
    rotations = []
    translations = []
    errors = []
    for frame_idx in range(source.shape[0]):
        rotation, translation, error = fit_rigid_kabsch(source[frame_idx], target[frame_idx], weights)
        rotations.append(rotation)
        translations.append(translation)
        errors.append(error)
    return torch.stack(rotations, dim=0), torch.stack(translations, dim=0), torch.stack(errors, dim=0)


def rigid_distance(
    source: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    aligned = torch.matmul(rotation, source.transpose(-1, -2)).transpose(-1, -2) + translation.unsqueeze(-2)
    return (aligned - target).norm(dim=-1).mean(dim=-1)
