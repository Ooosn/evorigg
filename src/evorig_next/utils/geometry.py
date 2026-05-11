from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


EPS = 1.0e-8


def assert_shape(tensor: torch.Tensor, expected: Iterable[int | None], name: str) -> None:
    if tensor.ndim != len(tuple(expected)):
        raise ValueError(f"{name} expected rank {len(tuple(expected))}, got {tensor.shape}")
    for index, (dim, exp) in enumerate(zip(tensor.shape, expected)):
        if exp is not None and dim != exp:
            raise ValueError(f"{name} dim {index} expected {exp}, got {dim}")


def safe_normalize(vectors: torch.Tensor, dim: int = -1) -> torch.Tensor:
    denom = vectors.norm(dim=dim, keepdim=True).clamp_min(EPS)
    return vectors / denom


def pairwise_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    return diff.square().sum(dim=-1).sqrt()


def mesh_radius(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return torch.tensor(1.0, dtype=points.dtype if points.numel() > 0 else torch.float32, device=points.device)
    center = points.mean(dim=0, keepdim=True)
    return (points - center).norm(dim=-1).max().clamp_min(EPS)


def farthest_point_sampling(points: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    if points.shape[0] <= count:
        return torch.arange(points.shape[0], device=points.device)
    indices = [0]
    distances = torch.full((points.shape[0],), float("inf"), device=points.device)
    for _ in range(1, count):
        last = points[indices[-1]]
        distances = torch.minimum(distances, (points - last).square().sum(dim=-1))
        indices.append(int(distances.argmax().item()))
    return torch.tensor(indices, dtype=torch.long, device=points.device)


def knn_indices(points: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    dists = pairwise_distances(centers, points)
    k = min(k, points.shape[0])
    return torch.topk(dists, k=k, largest=False, dim=1).indices


def pca_frame(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = points - points.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if torch.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] = -eigvecs[:, -1]
    return eigvecs, eigvals.clamp_min(EPS)


def chunk_indices(length: int, chunk_size: int | None) -> list[torch.Tensor]:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= length:
        return [torch.arange(length, dtype=torch.long)]
    chunks: list[torch.Tensor] = []
    start = 0
    while start < length:
        stop = min(length, start + chunk_size)
        chunks.append(torch.arange(start, stop, dtype=torch.long))
        start = stop
    return chunks


@dataclass
class BoneProjection:
    bone_index: int
    lambda_value: float
    projected_point: torch.Tensor
    distance: float
