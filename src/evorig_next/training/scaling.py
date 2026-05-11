from __future__ import annotations

from typing import TypeVar

import torch

T = TypeVar("T", float, torch.Tensor)


def clamp_reference_length(reference_length: float | torch.Tensor | None) -> float:
    if reference_length is None:
        return 1.0
    if isinstance(reference_length, torch.Tensor):
        if reference_length.numel() == 0:
            return 1.0
        return max(float(reference_length.detach().reshape(-1)[0].item()), 1.0e-8)
    return max(float(reference_length), 1.0e-8)


def normalize_linear_metric(value: T, reference_length: float | torch.Tensor | None) -> T:
    scale = clamp_reference_length(reference_length)
    return value / scale


def normalize_squared_metric(value: T, reference_length: float | torch.Tensor | None) -> T:
    scale = clamp_reference_length(reference_length)
    return value / (scale * scale)
