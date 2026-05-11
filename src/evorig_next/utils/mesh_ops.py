from __future__ import annotations

import math
import numpy as np

import torch

from evorig_next.utils.geometry import EPS

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency fallback
    o3d = None


def _to_cpu_contiguous(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target = tensor.detach()
    if dtype is not None:
        target = target.to(dtype=dtype)
    return target.to(device="cpu").contiguous()


class MeshQueryScene:
    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> None:
        if o3d is None:
            raise RuntimeError("open3d is not available")
        verts_cpu = _to_cpu_contiguous(vertices, dtype=torch.float32)
        faces_cpu = _to_cpu_contiguous(faces, dtype=torch.int32)
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor.from_numpy(verts_cpu.numpy()),
            o3d.core.Tensor.from_numpy(faces_cpu.numpy()),
        )
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(mesh)
        self._ray_dir = torch.tensor([1.0, 0.173, 0.071], dtype=torch.float32)
        self._ray_dir = self._ray_dir / self._ray_dir.norm().clamp_min(float(EPS))

    def closest_point_on_mesh(
        self,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(points.shape[0]) == 0:
            empty = torch.zeros(0, 3, dtype=points.dtype, device=points.device)
            empty_dist = torch.zeros(0, dtype=points.dtype, device=points.device)
            return empty, empty_dist
        points_cpu = _to_cpu_contiguous(points, dtype=torch.float32)
        closest = self._scene.compute_closest_points(
            o3d.core.Tensor.from_numpy(points_cpu.numpy())
        )["points"].numpy()
        closest_cpu = torch.from_numpy(closest)
        dist_sq_cpu = (closest_cpu - points_cpu).square().sum(dim=-1)
        return closest_cpu.to(device=points.device, dtype=points.dtype), dist_sq_cpu.to(device=points.device, dtype=points.dtype)

    def points_inside_mesh(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        if int(points.shape[0]) == 0:
            return torch.zeros(0, dtype=torch.bool, device=points.device)
        points_cpu = _to_cpu_contiguous(points, dtype=torch.float32)
        ray_dir = self._ray_dir.to(dtype=points_cpu.dtype).view(1, 3).expand(points_cpu.shape[0], -1)
        rays = torch.cat([points_cpu, ray_dir], dim=-1)
        counts = self._scene.count_intersections(
            o3d.core.Tensor.from_numpy(rays.numpy())
        ).numpy()
        inside_cpu = torch.from_numpy(counts % 2 == 1)
        return inside_cpu.to(device=points.device)

    def ray_mesh_first_hit_distance(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if origins.shape != directions.shape:
            raise ValueError("origins and directions must have the same shape")
        if origins.ndim != 2 or origins.shape[-1] != 3:
            raise ValueError("origins and directions must have shape [N, 3]")
        if int(origins.shape[0]) == 0:
            empty = torch.zeros(0, dtype=origins.dtype, device=origins.device)
            return empty, empty.bool()
        origins_cpu = _to_cpu_contiguous(origins, dtype=torch.float32)
        directions_cpu = _to_cpu_contiguous(directions, dtype=torch.float32)
        ray_norm = directions_cpu.norm(dim=-1, keepdim=True).clamp_min(float(EPS))
        safe_directions = directions_cpu / ray_norm
        rays = torch.cat([origins_cpu, safe_directions], dim=-1)
        cast = self._scene.cast_rays(o3d.core.Tensor.from_numpy(rays.numpy()))
        t_hit_cpu = torch.from_numpy(cast["t_hit"].numpy())
        hit_mask_cpu = torch.isfinite(t_hit_cpu)
        distances_cpu = torch.where(hit_mask_cpu, t_hit_cpu, torch.zeros_like(t_hit_cpu))
        return distances_cpu.to(device=origins.device, dtype=origins.dtype), hit_mask_cpu.to(device=origins.device)

    def ray_mesh_intersections(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if origins.shape != directions.shape:
            raise ValueError("origins and directions must have the same shape")
        if origins.ndim != 2 or origins.shape[-1] != 3:
            raise ValueError("origins and directions must have shape [N, 3]")
        if int(origins.shape[0]) == 0:
            empty_float = torch.zeros(0, dtype=origins.dtype, device=origins.device)
            empty_long = torch.zeros(0, dtype=torch.long, device=origins.device)
            return {
                "t_hit": empty_float,
                "ray_ids": empty_long,
                "ray_splits": torch.zeros(1, dtype=torch.long, device=origins.device),
                "primitive_ids": empty_long,
            }
        origins_cpu = _to_cpu_contiguous(origins, dtype=torch.float32)
        directions_cpu = _to_cpu_contiguous(directions, dtype=torch.float32)
        ray_norm = directions_cpu.norm(dim=-1, keepdim=True).clamp_min(float(EPS))
        safe_directions = directions_cpu / ray_norm
        rays = torch.cat([origins_cpu, safe_directions], dim=-1)
        intersections = self._scene.list_intersections(
            o3d.core.Tensor.from_numpy(rays.numpy())
        )
        return {
            "t_hit": torch.from_numpy(intersections["t_hit"].numpy()).to(device=origins.device, dtype=origins.dtype),
            "ray_ids": torch.from_numpy(intersections["ray_ids"].numpy().astype(np.int64, copy=False)).to(device=origins.device),
            "ray_splits": torch.from_numpy(intersections["ray_splits"].numpy().astype(np.int64, copy=False)).to(device=origins.device),
            "primitive_ids": torch.from_numpy(intersections["primitive_ids"].numpy().astype(np.int64, copy=False)).to(device=origins.device),
        }


def build_mesh_query_scene(
    vertices: torch.Tensor,
    faces: torch.Tensor | None,
) -> MeshQueryScene | None:
    if o3d is None or faces is None or faces.numel() == 0:
        return None
    return MeshQueryScene(vertices, faces)


def _resolve_point_chunk_size(points: torch.Tensor, faces: torch.Tensor, *, point_face_budget: int = 300_000) -> int:
    point_count = int(points.shape[0]) if points.ndim > 0 else 0
    face_count = int(faces.shape[0]) if faces.ndim > 0 else 0
    if point_count <= 0:
        return 1
    if face_count <= 0:
        return point_count
    chunk = max(int(point_face_budget // max(face_count, 1)), 1)
    return max(1, min(point_count, chunk))


def _closest_point_on_segment(points: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    segment = end - start
    denom = segment.square().sum(dim=-1).clamp_min(EPS)
    rel = points - start
    lam = (rel * segment).sum(dim=-1) / denom
    lam = lam.clamp(0.0, 1.0)
    return start + lam.unsqueeze(-1) * segment


def _closest_point_on_mesh_chunk(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    triangles = vertices[faces]
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    normal = torch.cross(ab, ac, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(EPS)

    points_exp = points[:, None, :]
    a_exp = a[None, :, :]
    ab_exp = ab[None, :, :]
    ac_exp = ac[None, :, :]
    normal_exp = normal[None, :, :]
    ap = points_exp - a_exp
    plane_offset = (ap * normal_exp).sum(dim=-1, keepdim=True)
    plane_proj = points_exp - plane_offset * normal_exp

    proj_rel = plane_proj - a_exp
    d00 = (ab_exp * ab_exp).sum(dim=-1)
    d01 = (ab_exp * ac_exp).sum(dim=-1)
    d11 = (ac_exp * ac_exp).sum(dim=-1)
    d20 = (proj_rel * ab_exp).sum(dim=-1)
    d21 = (proj_rel * ac_exp).sum(dim=-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(EPS)
    bary_v = (d11 * d20 - d01 * d21) / denom
    bary_w = (d00 * d21 - d01 * d20) / denom
    bary_u = 1.0 - bary_v - bary_w
    inside_face = (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_w >= 0.0)

    closest_ab = _closest_point_on_segment(points_exp, a_exp, b[None, :, :])
    closest_bc = _closest_point_on_segment(points_exp, b[None, :, :], c[None, :, :])
    closest_ca = _closest_point_on_segment(points_exp, c[None, :, :], a_exp)
    edge_candidates = torch.stack([closest_ab, closest_bc, closest_ca], dim=2)
    edge_dist_sq = (edge_candidates - points_exp.unsqueeze(2)).square().sum(dim=-1)
    edge_best_idx = edge_dist_sq.argmin(dim=2)
    gather_idx = edge_best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    best_edge_point = torch.gather(edge_candidates, dim=2, index=gather_idx).squeeze(2)
    best_edge_dist_sq = edge_dist_sq.min(dim=2).values

    plane_dist_sq = (plane_proj - points_exp).square().sum(dim=-1)
    candidate_point = torch.where(inside_face.unsqueeze(-1), plane_proj, best_edge_point)
    candidate_dist_sq = torch.where(inside_face, plane_dist_sq, best_edge_dist_sq)

    best_face_idx = candidate_dist_sq.argmin(dim=1)
    best_point = candidate_point[torch.arange(points.shape[0], device=points.device), best_face_idx]
    best_dist_sq = candidate_dist_sq[torch.arange(points.shape[0], device=points.device), best_face_idx]
    return best_point, best_dist_sq


def closest_point_on_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mesh_query_scene: MeshQueryScene | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mesh_query_scene is not None:
        return mesh_query_scene.closest_point_on_mesh(points)
    if points.shape[0] <= 1:
        return _closest_point_on_mesh_chunk(points, vertices, faces)
    chunk_size = _resolve_point_chunk_size(points, faces)
    best_points: list[torch.Tensor] = []
    best_dist_sq: list[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), chunk_size):
        chunk = points[start : start + chunk_size]
        chunk_best_point, chunk_best_dist_sq = _closest_point_on_mesh_chunk(chunk, vertices, faces)
        best_points.append(chunk_best_point)
        best_dist_sq.append(chunk_best_dist_sq)
    return torch.cat(best_points, dim=0), torch.cat(best_dist_sq, dim=0)


def _points_inside_mesh_chunk(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    triangles = vertices[faces]
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    ray_dir = torch.tensor([1.0, 0.173, 0.071], dtype=points.dtype, device=points.device)
    ray_dir = ray_dir / ray_dir.norm().clamp_min(EPS)

    origin = points[:, None, :]
    edge1 = (v1 - v0)[None, :, :]
    edge2 = (v2 - v0)[None, :, :]
    pvec = torch.cross(ray_dir.view(1, 1, 3).expand_as(edge2), edge2, dim=-1)
    det = (edge1 * pvec).sum(dim=-1)
    det_mask = det.abs() > 1.0e-8
    inv_det = torch.zeros_like(det)
    inv_det[det_mask] = 1.0 / det[det_mask]
    tvec = origin - v0[None, :, :]
    u = (tvec * pvec).sum(dim=-1) * inv_det
    qvec = torch.cross(tvec, edge1, dim=-1)
    v = (ray_dir.view(1, 1, 3) * qvec).sum(dim=-1) * inv_det
    t = (edge2 * qvec).sum(dim=-1) * inv_det
    hits = det_mask & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1.0e-6)
    return (hits.sum(dim=1) % 2) == 1


def points_inside_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mesh_query_scene: MeshQueryScene | None = None,
) -> torch.Tensor:
    if mesh_query_scene is not None:
        return mesh_query_scene.points_inside_mesh(points)
    if points.shape[0] <= 1:
        return _points_inside_mesh_chunk(points, vertices, faces)
    chunk_size = _resolve_point_chunk_size(points, faces)
    inside_chunks: list[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), chunk_size):
        chunk = points[start : start + chunk_size]
        inside_chunks.append(_points_inside_mesh_chunk(chunk, vertices, faces))
    return torch.cat(inside_chunks, dim=0)


def points_inside_or_on_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    surface_tol: float = 1.0e-4,
    mesh_query_scene: MeshQueryScene | None = None,
) -> torch.Tensor:
    inside = points_inside_mesh(points, vertices, faces, mesh_query_scene=mesh_query_scene)
    _, dist_sq = closest_point_on_mesh(points, vertices, faces, mesh_query_scene=mesh_query_scene)
    return inside | (dist_sq <= float(surface_tol) ** 2)


def project_points_inside_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    inward_hint: torch.Tensor | None = None,
    padding: float = 1.0e-3,
    mesh_query_scene: MeshQueryScene | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tol = max(float(padding), 1.0e-4)
    closest_points_all, dist_sq_all = closest_point_on_mesh(points, vertices, faces, mesh_query_scene=mesh_query_scene)
    inside_mask = points_inside_mesh(points, vertices, faces, mesh_query_scene=mesh_query_scene) | (dist_sq_all <= tol**2)
    projected = points.clone()
    if bool(inside_mask.all().item()):
        return projected, inside_mask, torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    outside_ids = torch.nonzero(~inside_mask, as_tuple=False).flatten()
    outside_points = points[outside_ids]
    closest_points = closest_points_all[outside_ids]
    dist_sq = dist_sq_all[outside_ids]
    if inward_hint is None:
        hint = vertices.mean(dim=0, keepdim=True).expand_as(outside_points)
    else:
        hint = inward_hint[outside_ids]
    direction = hint - closest_points
    direction_norm = direction.norm(dim=-1, keepdim=True)
    fallback = vertices.mean(dim=0, keepdim=True) - closest_points
    fallback_norm = fallback.norm(dim=-1, keepdim=True).clamp_min(EPS)
    safe_direction = torch.where(direction_norm > 1.0e-8, direction / direction_norm.clamp_min(EPS), fallback / fallback_norm)
    projected[outside_ids] = closest_points + float(padding) * safe_direction
    outside_distance = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    outside_distance[outside_ids] = dist_sq.sqrt()
    return projected, inside_mask, outside_distance


def _ray_mesh_first_hit_chunk(
    origins: torch.Tensor,
    directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    triangles = vertices[faces]
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    origin = origins[:, None, :]
    direction = directions[:, None, :]
    edge1 = (v1 - v0)[None, :, :]
    edge2 = (v2 - v0)[None, :, :]
    pvec = torch.cross(direction, edge2, dim=-1)
    det = (edge1 * pvec).sum(dim=-1)
    det_mask = det.abs() > 1.0e-8
    inv_det = torch.zeros_like(det)
    inv_det[det_mask] = 1.0 / det[det_mask]
    tvec = origin - v0[None, :, :]
    u = (tvec * pvec).sum(dim=-1) * inv_det
    qvec = torch.cross(tvec, edge1, dim=-1)
    v = (direction * qvec).sum(dim=-1) * inv_det
    t = (edge2 * qvec).sum(dim=-1) * inv_det
    hits = det_mask & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > 1.0e-6)
    inf = torch.full_like(t, float("inf"))
    first_hit = torch.where(hits, t, inf).min(dim=1).values
    return first_hit, torch.isfinite(first_hit)


def ray_mesh_first_hit_distance(
    origins: torch.Tensor,
    directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mesh_query_scene: MeshQueryScene | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mesh_query_scene is not None:
        return mesh_query_scene.ray_mesh_first_hit_distance(origins, directions)
    if origins.shape != directions.shape:
        raise ValueError("origins and directions must have the same shape")
    if origins.ndim != 2 or origins.shape[-1] != 3:
        raise ValueError("origins and directions must have shape [N, 3]")
    if int(origins.shape[0]) == 0:
        empty = torch.zeros(0, dtype=origins.dtype, device=origins.device)
        return empty, empty.bool()
    if int(origins.shape[0]) <= 1:
        return _ray_mesh_first_hit_chunk(origins, directions, vertices, faces)
    # Directional inside queries run many small ray batches inside the training loop.
    # On Windows/WDDM, a chunk that is still modest in memory can nevertheless hold a
    # single CUDA kernel long enough to trigger an "unknown error" / driver reset.
    # Use a more conservative point-face budget here than the generic mesh queries.
    chunk_size = _resolve_point_chunk_size(origins, faces, point_face_budget=25_000)
    distances: list[torch.Tensor] = []
    hit_masks: list[torch.Tensor] = []
    for start in range(0, int(origins.shape[0]), chunk_size):
        stop = start + chunk_size
        chunk_distance, chunk_hit = _ray_mesh_first_hit_chunk(
            origins[start:stop],
            directions[start:stop],
            vertices,
            faces,
        )
        distances.append(chunk_distance)
        hit_masks.append(chunk_hit)
    return torch.cat(distances, dim=0), torch.cat(hit_masks, dim=0)


def default_inside_sample_directions(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    count = int(count)
    if count <= 0:
        return torch.zeros(0, 3, dtype=dtype, device=device)
    canonical = torch.eye(3, dtype=dtype, device=device)
    if count <= 3:
        return canonical[:count]
    idx = torch.arange(count, dtype=dtype, device=device)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (idx + 0.5) / float(count)
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    theta = idx * golden_angle
    directions = torch.stack(
        [
            radius * torch.cos(theta),
            radius * torch.sin(theta),
            z,
        ],
        dim=-1,
    )
    directions[: min(3, count)] = canonical[: min(3, count)]
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(EPS)


def _expand_inside_sample_directions(
    point_count: int,
    *,
    directions: torch.Tensor | None,
    direction_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if directions is None:
        base = default_inside_sample_directions(direction_count, dtype=dtype, device=device)
        return base.unsqueeze(0).expand(point_count, -1, -1)
    if directions.ndim == 2:
        base = directions.to(device=device, dtype=dtype)
        return base.unsqueeze(0).expand(point_count, -1, -1)
    if directions.ndim == 3 and int(directions.shape[0]) == point_count:
        return directions.to(device=device, dtype=dtype)
    raise ValueError("directions must have shape [K, 3] or [N, K, 3]")


def compute_inside_shell_descriptor(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    inward_hint: torch.Tensor | None = None,
    padding: float = 1.0e-3,
    surface_tol: float = 1.0e-4,
    direction_count: int = 12,
    directions: torch.Tensor | None = None,
    mesh_query_scene: MeshQueryScene | None = None,
) -> dict[str, torch.Tensor]:
    return _compute_inside_shell_descriptor_impl(
        points,
        vertices,
        faces,
        inward_hint=inward_hint,
        padding=padding,
        surface_tol=surface_tol,
        direction_count=direction_count,
        directions=directions,
        allow_cpu_fallback=True,
        mesh_query_scene=mesh_query_scene,
    )


def _compute_inside_shell_descriptor_impl(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    inward_hint: torch.Tensor | None,
    padding: float,
    surface_tol: float,
    direction_count: int,
    directions: torch.Tensor | None,
    allow_cpu_fallback: bool,
    mesh_query_scene: MeshQueryScene | None,
) -> dict[str, torch.Tensor]:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if int(points.shape[0]) == 0:
        empty = torch.zeros(0, 3, dtype=points.dtype, device=points.device)
        empty_scalar = torch.zeros(0, dtype=points.dtype, device=points.device)
        return {
            "sample_points": empty,
            "inside_mask": empty_scalar.bool(),
            "outside_distance": empty_scalar,
            "forward_distance": empty_scalar.view(0, 0),
            "backward_distance": empty_scalar.view(0, 0),
            "margin": empty_scalar.view(0, 0),
            "balance": empty_scalar.view(0, 0),
            "valid_pairs": empty_scalar.bool().view(0, 0),
        }
    sample_points, inside_mask, outside_distance = project_points_inside_mesh(
        points,
        vertices,
        faces,
        inward_hint=inward_hint,
        padding=max(float(padding), float(surface_tol), 1.0e-4),
        mesh_query_scene=mesh_query_scene,
    )
    sample_dirs = _expand_inside_sample_directions(
        int(points.shape[0]),
        directions=directions,
        direction_count=direction_count,
        dtype=points.dtype,
        device=points.device,
    )
    if int(sample_dirs.shape[1]) == 0:
        empty = torch.zeros(points.shape[0], 0, dtype=points.dtype, device=points.device)
        return {
            "sample_points": sample_points,
            "inside_mask": inside_mask,
            "outside_distance": outside_distance,
            "forward_distance": empty,
            "backward_distance": empty,
            "margin": empty,
            "balance": empty,
            "valid_pairs": torch.zeros(points.shape[0], 0, dtype=torch.bool, device=points.device),
        }
    try:
        flat_origins = sample_points.unsqueeze(1).expand_as(sample_dirs).reshape(-1, 3)
        flat_dirs = sample_dirs.reshape(-1, 3)
        forward_distance, forward_hit = ray_mesh_first_hit_distance(
            flat_origins,
            flat_dirs,
            vertices,
            faces,
            mesh_query_scene=mesh_query_scene,
        )
        backward_distance, backward_hit = ray_mesh_first_hit_distance(
            flat_origins,
            -flat_dirs,
            vertices,
            faces,
            mesh_query_scene=mesh_query_scene,
        )
        forward_distance = forward_distance.view(points.shape[0], -1)
        backward_distance = backward_distance.view(points.shape[0], -1)
        valid_pairs = forward_hit.view(points.shape[0], -1) & backward_hit.view(points.shape[0], -1)
        margin = torch.minimum(forward_distance, backward_distance)
        balance = torch.zeros_like(margin)
        pair_sum = (forward_distance + backward_distance).clamp_min(EPS)
        balance[valid_pairs] = ((forward_distance - backward_distance) / pair_sum)[valid_pairs]
        return {
            "sample_points": sample_points,
            "inside_mask": inside_mask,
            "outside_distance": outside_distance,
            "forward_distance": forward_distance,
            "backward_distance": backward_distance,
            "margin": margin,
            "balance": balance,
            "valid_pairs": valid_pairs,
        }
    except RuntimeError:
        if not allow_cpu_fallback or not points.is_cuda:
            raise
        cpu_result = _compute_inside_shell_descriptor_impl(
            points.detach().cpu(),
            vertices.detach().cpu(),
            faces.detach().cpu(),
            inward_hint=None if inward_hint is None else inward_hint.detach().cpu(),
            padding=padding,
            surface_tol=surface_tol,
            direction_count=direction_count,
            directions=None if directions is None else directions.detach().cpu(),
            allow_cpu_fallback=False,
            mesh_query_scene=None,
        )
        result: dict[str, torch.Tensor] = {}
        for key, value in cpu_result.items():
            result[key] = value.to(device=points.device)
        return result
