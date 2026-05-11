from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch
import trimesh


_VOXEL_NEIGHBORS_26 = np.asarray(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ],
    dtype=np.int16,
)
_VOXEL_NEIGHBORS_6 = np.asarray(
    [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ],
    dtype=np.int16,
)


@dataclass
class MeshVoxelPathField:
    pitch: float
    grid_shape: tuple[int, int, int]
    target_resolution: int
    target_narrow_span_voxels: float
    max_resolution: int
    voxel_grid: Any
    filled_volume: np.ndarray
    distance_volume: np.ndarray
    filled_indices: np.ndarray
    filled_points: np.ndarray
    kdtree: cKDTree
    neighbor_offsets: np.ndarray
    neighbor_costs: np.ndarray
    flat_strides: tuple[int, int, int]


def _select_neighbor_offsets(mode: str) -> np.ndarray:
    normalized = str(mode).strip().lower()
    if normalized == "6":
        return _VOXEL_NEIGHBORS_6
    return _VOXEL_NEIGHBORS_26


def _estimate_voxel_pitch(
    extent: np.ndarray,
    *,
    target_resolution: int,
    target_narrow_span_voxels: float,
    max_resolution: int,
) -> float:
    positive_extent = extent[extent > 0.0]
    if positive_extent.size == 0:
        raise ValueError("mesh extent must be positive")
    max_extent = float(positive_extent.max())
    min_extent = float(positive_extent.min())
    pitch_from_resolution = max_extent / float(max(int(target_resolution), 8))
    pitch_from_narrow_span = min_extent / max(float(target_narrow_span_voxels), 1.0)
    pitch_floor = max_extent / float(max(int(max_resolution), int(target_resolution), 8))
    return max(min(pitch_from_resolution, pitch_from_narrow_span), pitch_floor, 1.0e-6)


def build_mesh_voxel_path_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_resolution: int = 72,
    target_narrow_span_voxels: float = 3.0,
    max_resolution: int = 192,
    neighbor_mode: str = "26",
) -> MeshVoxelPathField:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    extent = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    pitch = _estimate_voxel_pitch(
        extent,
        target_resolution=target_resolution,
        target_narrow_span_voxels=target_narrow_span_voxels,
        max_resolution=max_resolution,
    )
    voxel_grid = mesh.voxelized(pitch=pitch).fill()
    filled_volume = np.asarray(voxel_grid.matrix, dtype=bool)
    filled_indices = np.argwhere(filled_volume)
    if filled_indices.size == 0:
        raise ValueError("voxelized mesh produced no filled voxels")
    filled_points = np.asarray(voxel_grid.indices_to_points(filled_indices), dtype=np.float64)
    distance_volume = ndimage.distance_transform_edt(filled_volume).astype(np.float32) * float(pitch)
    neighbor_offsets = _select_neighbor_offsets(neighbor_mode)
    neighbor_costs = np.linalg.norm(neighbor_offsets.astype(np.float64), axis=1) * float(pitch)
    shape = tuple(int(item) for item in filled_volume.shape)
    return MeshVoxelPathField(
        pitch=float(pitch),
        grid_shape=shape,
        target_resolution=int(target_resolution),
        target_narrow_span_voxels=float(target_narrow_span_voxels),
        max_resolution=int(max_resolution),
        voxel_grid=voxel_grid,
        filled_volume=filled_volume,
        distance_volume=distance_volume,
        filled_indices=filled_indices.astype(np.int32, copy=False),
        filled_points=filled_points,
        kdtree=cKDTree(filled_points),
        neighbor_offsets=neighbor_offsets,
        neighbor_costs=neighbor_costs.astype(np.float32, copy=False),
        flat_strides=(shape[1] * shape[2], shape[2], 1),
    )


def _index_in_bounds(index: np.ndarray, shape: tuple[int, int, int]) -> bool:
    return bool(
        0 <= int(index[0]) < int(shape[0])
        and 0 <= int(index[1]) < int(shape[1])
        and 0 <= int(index[2]) < int(shape[2])
    )


def _flat_index(index: np.ndarray, strides: tuple[int, int, int]) -> int:
    return int(index[0]) * int(strides[0]) + int(index[1]) * int(strides[1]) + int(index[2])


def _unflatten_index(flat_index: int, shape: tuple[int, int, int], strides: tuple[int, int, int]) -> np.ndarray:
    x = int(flat_index) // int(strides[0])
    remainder = int(flat_index) % int(strides[0])
    y = remainder // int(strides[1])
    z = remainder % int(strides[1])
    return np.asarray([x, y, z], dtype=np.int32)


def _point_to_filled_index(field: MeshVoxelPathField, point: torch.Tensor | np.ndarray) -> tuple[np.ndarray, float]:
    point_np = np.asarray(point, dtype=np.float64).reshape(1, 3)
    try:
        candidate = np.asarray(field.voxel_grid.points_to_indices(point_np), dtype=np.int32).reshape(-1, 3)[0]
    except Exception:
        candidate = None
    if candidate is not None and _index_in_bounds(candidate, field.grid_shape):
        if bool(field.filled_volume[tuple(candidate.tolist())]):
            center = field.filled_points[field.kdtree.query(point_np[0], k=1)[1]]
            return candidate, float(np.linalg.norm(center - point_np[0]))
    distance, nearest = field.kdtree.query(point_np[0], k=1)
    nearest_index = np.asarray(field.filled_indices[int(nearest)], dtype=np.int32)
    return nearest_index, float(distance)


def _dijkstra_to_targets(
    field: MeshVoxelPathField,
    start_index: np.ndarray,
    target_flat_indices: set[int],
    *,
    clearance_weight: float = 0.0,
    clearance_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    flat_size = int(np.prod(np.asarray(field.grid_shape, dtype=np.int64)))
    distances = np.full((flat_size,), np.inf, dtype=np.float64)
    previous = np.full((flat_size,), -1, dtype=np.int64)
    visited = np.zeros((flat_size,), dtype=bool)
    start_flat = _flat_index(start_index, field.flat_strides)
    distances[start_flat] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_flat)]
    pending_targets = set(int(item) for item in target_flat_indices)
    max_clearance = 1.0
    if float(clearance_weight) > 0.0 and np.any(field.filled_volume):
        max_clearance = max(float(field.distance_volume[field.filled_volume].max()), 1.0e-9)
    while heap and pending_targets:
        current_cost, current_flat = heapq.heappop(heap)
        if visited[current_flat]:
            continue
        visited[current_flat] = True
        if current_flat in pending_targets:
            pending_targets.remove(current_flat)
            if not pending_targets:
                break
        current_index = _unflatten_index(current_flat, field.grid_shape, field.flat_strides)
        for neighbor_offset, neighbor_cost in zip(field.neighbor_offsets, field.neighbor_costs.tolist(), strict=True):
            neighbor_index = current_index + neighbor_offset
            if not _index_in_bounds(neighbor_index, field.grid_shape):
                continue
            if not bool(field.filled_volume[tuple(neighbor_index.tolist())]):
                continue
            neighbor_flat = _flat_index(neighbor_index, field.flat_strides)
            if visited[neighbor_flat]:
                continue
            step_cost = float(neighbor_cost)
            if float(clearance_weight) > 0.0:
                clearance = float(field.distance_volume[tuple(neighbor_index.tolist())])
                low_clearance = max(0.0, 1.0 - clearance / max_clearance)
                step_cost *= 1.0 + float(clearance_weight) * (low_clearance ** max(float(clearance_power), 0.0))
            candidate_cost = float(current_cost) + step_cost
            if candidate_cost < float(distances[neighbor_flat]):
                distances[neighbor_flat] = candidate_cost
                previous[neighbor_flat] = int(current_flat)
                heapq.heappush(heap, (candidate_cost, neighbor_flat))
    return distances, previous


def _reconstruct_voxel_path(
    field: MeshVoxelPathField,
    previous: np.ndarray,
    target_flat: int,
) -> np.ndarray:
    path_flat: list[int] = []
    current = int(target_flat)
    while current >= 0:
        path_flat.append(current)
        current = int(previous[current])
    path_flat.reverse()
    if not path_flat:
        return np.zeros((0, 3), dtype=np.float64)
    indices = np.stack(
        [_unflatten_index(flat_idx, field.grid_shape, field.flat_strides) for flat_idx in path_flat],
        axis=0,
    )
    return np.asarray(field.voxel_grid.indices_to_points(indices), dtype=np.float64)


def trace_voxel_parent_paths(
    query_point: torch.Tensor,
    joint_positions: torch.Tensor,
    field: MeshVoxelPathField,
    *,
    candidate_joint_ids: torch.Tensor | None = None,
    clearance_weight: float = 0.0,
    clearance_power: float = 1.0,
) -> list[dict[str, Any]]:
    if int(joint_positions.numel()) == 0:
        return []
    query = query_point.detach().cpu().numpy().astype(np.float64).reshape(3)
    joints = joint_positions.detach().cpu().numpy().astype(np.float64).reshape(-1, 3)
    if candidate_joint_ids is None:
        candidate_joint_ids = torch.arange(joint_positions.shape[0], dtype=torch.long, device=joint_positions.device)
    else:
        candidate_joint_ids = candidate_joint_ids.to(device=joint_positions.device, dtype=torch.long).reshape(-1)
    if int(candidate_joint_ids.numel()) == 0:
        return []
    start_index, start_link_cost = _point_to_filled_index(field, query)
    target_payloads: list[dict[str, Any]] = []
    target_flat_indices: set[int] = set()
    for joint_id in candidate_joint_ids.tolist():
        joint_pos = joints[int(joint_id)]
        target_index, target_link_cost = _point_to_filled_index(field, joint_pos)
        target_flat = _flat_index(target_index, field.flat_strides)
        target_payloads.append(
            {
                "joint_id": int(joint_id),
                "joint_position": joint_pos,
                "target_index": target_index,
                "target_flat": target_flat,
                "target_link_cost": float(target_link_cost),
            }
        )
        target_flat_indices.add(int(target_flat))
    distances, previous = _dijkstra_to_targets(
        field,
        start_index,
        target_flat_indices,
        clearance_weight=float(clearance_weight),
        clearance_power=float(clearance_power),
    )
    ranked_paths: list[dict[str, Any]] = []
    for payload in target_payloads:
        target_flat = int(payload["target_flat"])
        graph_cost = float(distances[target_flat])
        if not math.isfinite(graph_cost):
            continue
        voxel_polyline = _reconstruct_voxel_path(field, previous, target_flat)
        if voxel_polyline.shape[0] == 0:
            continue
        reversed_polyline = voxel_polyline[::-1]
        points: list[np.ndarray] = [payload["joint_position"].reshape(1, 3)]
        if np.linalg.norm(reversed_polyline[0] - payload["joint_position"]) > float(field.pitch) * 0.5:
            points.append(reversed_polyline[0:1])
            tail = reversed_polyline[1:]
        else:
            tail = reversed_polyline[1:]
        if tail.shape[0] > 0:
            points.append(tail)
        if np.linalg.norm(reversed_polyline[-1] - query) > float(field.pitch) * 0.5:
            points.append(query.reshape(1, 3))
        polyline = np.concatenate(points, axis=0)
        path_length = graph_cost + float(start_link_cost) + float(payload["target_link_cost"])
        voxel_indices = np.stack(
            [_unflatten_index(flat_idx, field.grid_shape, field.flat_strides) for flat_idx in np.unique(np.asarray([
                _flat_index(np.asarray(index, dtype=np.int32), field.flat_strides)
                for index in np.asarray(field.voxel_grid.points_to_indices(voxel_polyline), dtype=np.int32).reshape(-1, 3)
            ], dtype=np.int64))],
            axis=0,
        )
        clearances = field.distance_volume[
            voxel_indices[:, 0],
            voxel_indices[:, 1],
            voxel_indices[:, 2],
        ]
        ranked_paths.append(
            {
                "joint_id": int(payload["joint_id"]),
                "path_length": float(path_length),
                "graph_length": float(graph_cost),
                "mean_clearance": float(clearances.mean()) if clearances.size > 0 else 0.0,
                "min_clearance": float(clearances.min()) if clearances.size > 0 else 0.0,
                "max_clearance": float(clearances.max()) if clearances.size > 0 else 0.0,
                "polyline": torch.tensor(polyline, dtype=joint_positions.dtype, device=joint_positions.device),
                "target_link_cost": float(payload["target_link_cost"]),
                "start_link_cost": float(start_link_cost),
            }
        )
    ranked_paths.sort(
        key=lambda item: (
            float(item["path_length"]),
            float(-item["mean_clearance"]),
            int(item["joint_id"]),
        )
    )
    return ranked_paths


def select_parent_joint_by_voxel_distance(
    query_point: torch.Tensor,
    joint_positions: torch.Tensor,
    field: MeshVoxelPathField,
    *,
    candidate_joint_ids: torch.Tensor | None = None,
) -> tuple[int, str, float, dict[str, Any] | None]:
    ranking = trace_voxel_parent_paths(
        query_point=query_point,
        joint_positions=joint_positions,
        field=field,
        candidate_joint_ids=candidate_joint_ids,
    )
    if not ranking:
        return -1, "voxel_distance_unavailable", 0.0, None
    best = ranking[0]
    return (
        int(best["joint_id"]),
        "voxel_distance_parent_joint",
        1.0,
        best,
    )
