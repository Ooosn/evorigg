from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from pygltflib import GLTF2

from evorig_next.config import merge_with_default
from evorig_next.models.skeleton import Skeleton
from evorig_next.utils.mesh_ops import points_inside_or_on_mesh


BONE_NAME_RE = re.compile(r"^bone_(\d+)$")


def _read_accessor_dense(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    accessor = gltf.accessors[accessor_index]
    component_dtype = {
        5121: np.uint8,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }[accessor.componentType]
    elem_count = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }[accessor.type]
    dtype = np.dtype(component_dtype)
    values = np.zeros((accessor.count, elem_count), dtype=dtype)

    if accessor.bufferView is not None:
        buffer_view = gltf.bufferViews[accessor.bufferView]
        blob = gltf.binary_blob()
        offset = int(buffer_view.byteOffset or 0) + int(accessor.byteOffset or 0)
        flat = np.frombuffer(blob, dtype=dtype, count=accessor.count * elem_count, offset=offset)
        values = flat.reshape(accessor.count, elem_count).copy()

    if accessor.sparse is not None and accessor.sparse.count > 0:
        sparse = accessor.sparse
        blob = gltf.binary_blob()
        sparse_index_dtype = {
            5121: np.uint8,
            5123: np.uint16,
            5125: np.uint32,
        }[sparse.indices.componentType]
        index_view = gltf.bufferViews[sparse.indices.bufferView]
        sparse_indices = np.frombuffer(
            blob,
            dtype=np.dtype(sparse_index_dtype),
            count=sparse.count,
            offset=int(index_view.byteOffset or 0) + int(sparse.indices.byteOffset or 0),
        )
        value_view = gltf.bufferViews[sparse.values.bufferView]
        sparse_values = np.frombuffer(
            blob,
            dtype=dtype,
            count=sparse.count * elem_count,
            offset=int(value_view.byteOffset or 0) + int(sparse.values.byteOffset or 0),
        ).reshape(sparse.count, elem_count)
        values[sparse_indices] = sparse_values

    if elem_count == 1:
        return values[:, 0]
    return values


def _quat_to_matrix_xyzw(quat: list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray:
    if quat is None:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = [float(v) for v in quat]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _node_local_matrix(node: Any) -> np.ndarray:
    if node.matrix is not None and len(node.matrix) == 16:
        return np.asarray(node.matrix, dtype=np.float32).reshape(4, 4).T.copy()
    translation = np.asarray(node.translation if node.translation is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.asarray(node.scale if node.scale is not None else [1.0, 1.0, 1.0], dtype=np.float32)
    rotation = _quat_to_matrix_xyzw(node.rotation)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _compute_node_world_matrices(gltf: GLTF2) -> tuple[list[np.ndarray], dict[int, int | None]]:
    node_count = len(gltf.nodes or [])
    parent_of: dict[int, int | None] = {index: None for index in range(node_count)}
    for parent_index, node in enumerate(gltf.nodes or []):
        for child_index in node.children or []:
            parent_of[int(child_index)] = int(parent_index)

    local_mats = [_node_local_matrix(node) for node in gltf.nodes or []]
    world_mats: list[np.ndarray | None] = [None] * node_count

    def solve(node_index: int) -> np.ndarray:
        cached = world_mats[node_index]
        if cached is not None:
            return cached
        parent_index = parent_of[node_index]
        if parent_index is None:
            world = local_mats[node_index]
        else:
            world = solve(parent_index) @ local_mats[node_index]
        world_mats[node_index] = world
        return world

    solved = [solve(index) for index in range(node_count)]
    return solved, parent_of


def _extract_bone_rig(rigged_glb_path: str | Path) -> dict[str, Any]:
    gltf = GLTF2().load_binary(str(rigged_glb_path))
    world_mats, parent_of = _compute_node_world_matrices(gltf)

    numbered_nodes: list[tuple[int, int]] = []
    for node_index, node in enumerate(gltf.nodes or []):
        if not node.name:
            continue
        match = BONE_NAME_RE.match(str(node.name))
        if match is None:
            continue
        numbered_nodes.append((int(match.group(1)), node_index))
    if not numbered_nodes:
        raise ValueError(f"no bone_* nodes found in {rigged_glb_path}")

    numbered_nodes.sort(key=lambda item: item[0])
    bone_ids = [bone_id for bone_id, _ in numbered_nodes]
    if bone_ids != list(range(len(bone_ids))):
        raise ValueError(f"bone ids are not contiguous 0..J-1 in {rigged_glb_path}: {bone_ids}")

    node_to_bone = {node_index: bone_id for bone_id, node_index in numbered_nodes}
    joints: list[dict[str, Any]] = []
    for bone_id, node_index in numbered_nodes:
        parent_node = parent_of[node_index]
        parent_bone = -1
        while parent_node is not None:
            if parent_node in node_to_bone:
                parent_bone = int(node_to_bone[parent_node])
                break
            parent_node = parent_of[parent_node]
        position = world_mats[node_index][:3, 3].astype(np.float32)
        joints.append(
            {
                "id": int(bone_id),
                "parent_id": int(parent_bone),
                "connected_to_parent": bool(parent_bone >= 0),
                "rest_position": position.tolist(),
                "birth_step": 0,
                "is_inserted": False,
                "birth_mode": "seed",
                "name": str(gltf.nodes[node_index].name),
                "source_node_id": int(node_index),
            }
        )
    return {"joints": joints}


def _joint_children_from_parent(parent_idx: torch.Tensor) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(int(parent_idx.shape[0]))]
    for joint_id, parent_joint in enumerate(parent_idx.detach().cpu().tolist()):
        parent_joint = int(parent_joint)
        if parent_joint >= 0:
            children[parent_joint].append(int(joint_id))
    return children


def _joint_depths_from_parent(parent_idx: torch.Tensor) -> list[int]:
    depths = [0] * int(parent_idx.shape[0])
    children = _joint_children_from_parent(parent_idx)
    roots = [joint_id for joint_id, parent_joint in enumerate(parent_idx.detach().cpu().tolist()) if int(parent_joint) < 0]
    stack = [(int(root_joint), 0) for root_joint in roots]
    while stack:
        joint_id, depth = stack.pop()
        depths[joint_id] = depth
        for child_joint in children[joint_id]:
            stack.append((int(child_joint), depth + 1))
    return depths


def _segment_inside_or_on_mesh(
    start: torch.Tensor,
    end: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    surface_tol: float,
    sample_count: int,
) -> tuple[bool, int]:
    if int(sample_count) <= 0:
        return True, 0
    ts = torch.linspace(0.0, 1.0, int(sample_count) + 2, dtype=start.dtype, device=start.device)[1:-1]
    probes = start.unsqueeze(0) * (1.0 - ts.unsqueeze(-1)) + end.unsqueeze(0) * ts.unsqueeze(-1)
    inside = points_inside_or_on_mesh(probes, vertices, faces, surface_tol=surface_tol)
    outside_count = int((~inside).sum().item())
    return bool(inside.all().item()), outside_count


def _serialize_rig_from_skeleton(
    skeleton: Skeleton,
    extras: list[dict[str, Any]],
) -> dict[str, Any]:
    joints: list[dict[str, Any]] = []
    rest_positions = skeleton.rest_joints.detach().cpu().tolist()
    parent_idx = skeleton.parent_idx.detach().cpu().tolist()
    connected_to_parent = getattr(skeleton, "connected_to_parent", None)
    if isinstance(connected_to_parent, torch.Tensor):
        connected_values = connected_to_parent.detach().cpu().bool().tolist()
    else:
        connected_values = [int(parent_id) >= 0 for parent_id in parent_idx]
    for joint_id in range(skeleton.joint_count):
        payload = {
            "id": int(joint_id),
            "parent_id": int(parent_idx[joint_id]),
            "connected_to_parent": bool(connected_values[joint_id]) and int(parent_idx[joint_id]) >= 0,
            "rest_position": rest_positions[joint_id],
            "birth_step": int(skeleton.birth_steps[joint_id]),
            "is_inserted": bool(skeleton.is_inserted[joint_id]),
            "birth_mode": str(skeleton.birth_modes[joint_id]),
        }
        payload.update(extras[joint_id])
        joints.append(payload)
    return {"joints": joints}


def _collapse_joint_and_pop_extra(
    skeleton: Skeleton,
    extras: list[dict[str, Any]],
    *,
    removed_joint: int,
    target_joint: int,
) -> dict[str, Any]:
    record = skeleton.collapse_joint_into_target(int(removed_joint), int(target_joint))
    extras.pop(int(record["removed_joint_id"]))
    return record


def _load_real_preprocess_config(config_template_path: str | Path | None) -> dict[str, Any]:
    merged = merge_with_default(config_template_path)
    cfg = dict(merged.get("real_preprocess", {}))
    cfg.setdefault("enabled", True)
    cfg.setdefault("surface_tol", 3.0e-3)
    cfg.setdefault("mean_bone_gate_multiplier", 1.0)
    cfg.setdefault("incident_length_multiplier", 0.5)
    cfg.setdefault("segment_sample_count_min", 8)
    cfg.setdefault("segment_sample_length_ratio", 0.1)
    return cfg


def _preprocess_real_rig(
    rig_json: dict[str, Any],
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    preprocess_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    joints = list(rig_json.get("joints", []))
    if not joints:
        raise ValueError("rig_json must contain a non-empty joints list")
    device = torch.device("cpu")
    dtype = torch.float32
    rest_vertices_t = torch.tensor(rest_vertices, dtype=dtype, device=device)
    faces_t = torch.tensor(faces, dtype=torch.long, device=device)
    parent_idx = torch.tensor([int(joint["parent_id"]) for joint in joints], dtype=torch.long, device=device)
    rest_joints = torch.tensor([joint["rest_position"] for joint in joints], dtype=dtype, device=device)
    skeleton = Skeleton(
        parent_idx=parent_idx,
        rest_joints=rest_joints,
        frame_count=1,
        birth_steps=[int(joint.get("birth_step", 0)) for joint in joints],
        inserted=[bool(joint.get("is_inserted", False)) for joint in joints],
        birth_modes=[str(joint.get("birth_mode", "seed")) for joint in joints],
        connected_to_parent=[
            bool(joint.get("connected_to_parent", int(joint.get("parent_id", -1)) >= 0))
            and int(joint.get("parent_id", -1)) >= 0
            for joint in joints
        ],
    ).to(device)
    extras: list[dict[str, Any]] = []
    for joint in joints:
        extras.append(
            {
                key: value
                for key, value in joint.items()
                if key not in {"id", "parent_id", "rest_position", "birth_step", "is_inserted", "birth_mode"}
            }
        )
    summary: dict[str, Any] = {
        "enabled": bool(preprocess_cfg.get("enabled", True)),
        "initial_joint_count": int(skeleton.joint_count),
        "outside_joint_removals": [],
        "near_pair_merges": [],
        "config": {
            "surface_tol": float(preprocess_cfg.get("surface_tol", 3.0e-3)),
            "mean_bone_gate_multiplier": float(preprocess_cfg.get("mean_bone_gate_multiplier", 1.0)),
            "incident_length_multiplier": float(preprocess_cfg.get("incident_length_multiplier", 0.5)),
            "segment_sample_count_min": int(preprocess_cfg.get("segment_sample_count_min", 8)),
            "segment_sample_length_ratio": float(preprocess_cfg.get("segment_sample_length_ratio", 0.1)),
        },
    }
    if not summary["enabled"]:
        summary["final_joint_count"] = int(skeleton.joint_count)
        return _serialize_rig_from_skeleton(skeleton, extras), summary

    surface_tol = float(preprocess_cfg.get("surface_tol", 3.0e-3))
    mean_gate_multiplier = float(preprocess_cfg.get("mean_bone_gate_multiplier", 1.0))
    incident_gate_multiplier = float(preprocess_cfg.get("incident_length_multiplier", 0.5))
    segment_sample_count_min = max(int(preprocess_cfg.get("segment_sample_count_min", 8)), 2)
    segment_sample_length_ratio = max(float(preprocess_cfg.get("segment_sample_length_ratio", 0.1)), 1.0e-4)

    while True:
        inside = points_inside_or_on_mesh(
            skeleton.rest_joints.detach(),
            rest_vertices_t,
            faces_t,
            surface_tol=surface_tol,
        )
        outside_joint_ids = torch.nonzero(~inside, as_tuple=False).flatten()
        if outside_joint_ids.numel() == 0:
            break
        root_ids = [int(joint_id) for joint_id in outside_joint_ids.tolist() if int(skeleton.parent_idx[int(joint_id)].item()) < 0]
        if root_ids:
            raise ValueError(f"real rig preprocess found root joint(s) outside mesh: {root_ids}")
        depths = _joint_depths_from_parent(skeleton.parent_idx.detach())
        remove_joint = max(
            (int(joint_id) for joint_id in outside_joint_ids.tolist()),
            key=lambda joint_id: (int(depths[joint_id]), int(joint_id)),
        )
        parent_joint = int(skeleton.parent_idx[remove_joint].item())
        position = skeleton.rest_joints.detach()[remove_joint].cpu().tolist()
        record = _collapse_joint_and_pop_extra(
            skeleton,
            extras,
            removed_joint=remove_joint,
            target_joint=parent_joint,
        )
        summary["outside_joint_removals"].append(
            {
                "removed_joint_id": int(record["removed_joint_id"]),
                "kept_parent_id": int(record["kept_joint_id"]),
                "target_joint_id": int(record["target_joint_id"]),
                "reparented_children": [int(child_id) for child_id in record.get("reparented_children", [])],
                "rest_position": position,
            }
        )

    while True:
        if int(skeleton.bone_count) == 0:
            break
        positions = skeleton.rest_joints.detach()
        parent_idx = skeleton.parent_idx.detach()
        mean_bone_length = float(
            (positions[skeleton.bone_child_idx] - positions[skeleton.bone_parent_idx]).norm(dim=-1).mean().item()
        )
        mean_gate = float(mean_bone_length * mean_gate_multiplier)
        incident_lengths: dict[int, list[float]] = {joint_id: [] for joint_id in range(skeleton.joint_count)}
        direct_edges = set()
        for child_joint, parent_joint in enumerate(parent_idx.cpu().tolist()):
            parent_joint = int(parent_joint)
            if parent_joint < 0:
                continue
            bone_length = float((positions[child_joint] - positions[parent_joint]).norm().item())
            incident_lengths[int(child_joint)].append(bone_length)
            incident_lengths[parent_joint].append(bone_length)
            direct_edges.add(tuple(sorted((int(child_joint), parent_joint))))
        children = _joint_children_from_parent(parent_idx)
        depths = _joint_depths_from_parent(parent_idx)
        candidate_pairs: list[dict[str, Any]] = []
        for left_joint in range(skeleton.joint_count):
            for right_joint in range(left_joint + 1, skeleton.joint_count):
                if (left_joint, right_joint) in direct_edges:
                    continue
                pair_distance = float((positions[left_joint] - positions[right_joint]).norm().item())
                if pair_distance >= mean_gate:
                    continue
                local_lengths = incident_lengths[left_joint] + incident_lengths[right_joint]
                if not local_lengths:
                    continue
                min_incident_length = float(min(local_lengths))
                incident_gate = float(min_incident_length * incident_gate_multiplier)
                if pair_distance >= incident_gate:
                    continue
                sample_count = max(
                    segment_sample_count_min,
                    int(math.ceil(pair_distance / max(mean_bone_length * segment_sample_length_ratio, 1.0e-4))),
                )
                segment_inside, outside_probe_count = _segment_inside_or_on_mesh(
                    positions[left_joint],
                    positions[right_joint],
                    rest_vertices_t,
                    faces_t,
                    surface_tol=surface_tol,
                    sample_count=sample_count,
                )
                if not segment_inside:
                    continue
                left_priority = (
                    0 if int(parent_idx[left_joint].item()) < 0 else 1,
                    0 if not bool(skeleton.is_inserted[left_joint]) else 1,
                    int(depths[left_joint]),
                    int(skeleton.birth_steps[left_joint]),
                    int(left_joint),
                )
                right_priority = (
                    0 if int(parent_idx[right_joint].item()) < 0 else 1,
                    0 if not bool(skeleton.is_inserted[right_joint]) else 1,
                    int(depths[right_joint]),
                    int(skeleton.birth_steps[right_joint]),
                    int(right_joint),
                )
                candidate_pairs.append(
                    {
                        "left_joint": int(left_joint),
                        "right_joint": int(right_joint),
                        "pair_distance": float(pair_distance),
                        "min_incident_length": float(min_incident_length),
                        "mean_bone_length": float(mean_bone_length),
                        "outside_probe_count": int(outside_probe_count),
                        "left_priority": left_priority,
                        "right_priority": right_priority,
                    }
                )
        if not candidate_pairs:
            break
        candidate_pairs.sort(
            key=lambda item: (
                float(item["pair_distance"]),
                item["left_priority"],
                item["right_priority"],
            )
        )
        merged = False
        for candidate in candidate_pairs:
            left_joint = int(candidate["left_joint"])
            right_joint = int(candidate["right_joint"])
            if left_joint >= skeleton.joint_count or right_joint >= skeleton.joint_count:
                continue
            if candidate["left_priority"] <= candidate["right_priority"]:
                keep_joint, remove_joint = left_joint, right_joint
            else:
                keep_joint, remove_joint = right_joint, left_joint
            merge_direction = "priority"
            try:
                record = _collapse_joint_and_pop_extra(
                    skeleton,
                    extras,
                    removed_joint=remove_joint,
                    target_joint=keep_joint,
                )
            except ValueError:
                keep_joint, remove_joint = remove_joint, keep_joint
                merge_direction = "fallback_reverse"
                try:
                    record = _collapse_joint_and_pop_extra(
                        skeleton,
                        extras,
                        removed_joint=remove_joint,
                        target_joint=keep_joint,
                    )
                except ValueError:
                    continue
            summary["near_pair_merges"].append(
                {
                    "kept_joint_id": int(record["kept_joint_id"]),
                    "removed_joint_id": int(record["removed_joint_id"]),
                    "target_joint_id": int(record["target_joint_id"]),
                    "pair_distance": float(candidate["pair_distance"]),
                    "min_incident_length": float(candidate["min_incident_length"]),
                    "mean_bone_length": float(candidate["mean_bone_length"]),
                    "segment_outside_probe_count": int(candidate["outside_probe_count"]),
                    "merge_direction": merge_direction,
                    "reparented_children": [int(child_id) for child_id in record.get("reparented_children", [])],
                }
            )
            merged = True
            break
        if not merged:
            break

    summary["final_joint_count"] = int(skeleton.joint_count)
    summary["outside_joint_removal_count"] = int(len(summary["outside_joint_removals"]))
    summary["near_pair_merge_count"] = int(len(summary["near_pair_merges"]))
    return _serialize_rig_from_skeleton(skeleton, extras), summary


def _load_merged_rest_mesh(rigged_glb_path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(rigged_glb_path, process=False)
    if not hasattr(loaded, "geometry") or len(loaded.geometry) != 1:
        raise ValueError(f"expected exactly one mesh in {rigged_glb_path}")
    mesh = list(loaded.geometry.values())[0].copy()
    mesh.merge_vertices()
    return mesh


def _reconstruct_gt_anim_vertices(animated_glb_path: str | Path, base_vertices: np.ndarray) -> np.ndarray:
    gltf = GLTF2().load_binary(str(animated_glb_path))
    if not gltf.meshes or len(gltf.meshes) != 1:
        raise ValueError(f"expected exactly one mesh in {animated_glb_path}")
    mesh = gltf.meshes[0]
    primitive = mesh.primitives[0]
    if not primitive.targets:
        raise ValueError(f"mesh in {animated_glb_path} has no morph targets")

    target_deltas = np.stack(
        [_read_accessor_dense(gltf, int(target["POSITION"])) for target in primitive.targets],
        axis=0,
    ).astype(np.float32)
    if target_deltas.shape[1:] != base_vertices.shape:
        raise ValueError(
            f"morph target vertex shape {target_deltas.shape[1:]} does not match base vertices {base_vertices.shape}"
        )

    if gltf.animations:
        animation = gltf.animations[0]
        weight_channel = None
        for channel in animation.channels or []:
            if str(channel.target.path) == "weights":
                weight_channel = channel
                break
        if weight_channel is None:
            raise ValueError(f"no weights animation channel found in {animated_glb_path}")
        sampler = animation.samplers[weight_channel.sampler]
        frame_times = _read_accessor_dense(gltf, int(sampler.input)).astype(np.float32)
        frame_weights = _read_accessor_dense(gltf, int(sampler.output)).astype(np.float32).reshape(frame_times.shape[0], -1)
    else:
        default_weights = np.asarray(mesh.weights if mesh.weights is not None else [1.0] + [0.0] * (target_deltas.shape[0] - 1), dtype=np.float32)
        frame_times = np.asarray([0.0], dtype=np.float32)
        frame_weights = default_weights.reshape(1, -1)

    if frame_weights.shape[1] != target_deltas.shape[0]:
        raise ValueError(
            f"frame weight width {frame_weights.shape[1]} does not match target count {target_deltas.shape[0]} in {animated_glb_path}"
        )

    frames = base_vertices[None, :, :] + np.einsum("ft,tvc->fvc", frame_weights, target_deltas, optimize=True)
    return frames.astype(np.float32), frame_times.astype(np.float32), frame_weights.astype(np.float32)


def import_real_glb_sample(
    *,
    rigged_glb_path: str | Path,
    animated_glb_path: str | Path,
    output_dir: str | Path,
    config_template_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rest_mesh = _load_merged_rest_mesh(rigged_glb_path)
    rig_json = _extract_bone_rig(rigged_glb_path)
    preprocess_cfg = _load_real_preprocess_config(config_template_path)
    base_vertices = np.asarray(rest_mesh.vertices, dtype=np.float32)
    rig_json, rig_cleanup_summary = _preprocess_real_rig(
        rig_json=rig_json,
        rest_vertices=base_vertices,
        faces=np.asarray(rest_mesh.faces, dtype=np.int64),
        preprocess_cfg=preprocess_cfg,
    )
    gt_anim_vertices, frame_times, frame_weights = _reconstruct_gt_anim_vertices(animated_glb_path, base_vertices)

    rest_mesh.export(output_dir / "rest_mesh.obj")
    with (output_dir / "wrong_init_rig.json").open("w", encoding="utf-8") as handle:
        json.dump(rig_json, handle, indent=2)
    np.save(output_dir / "gt_anim_vertices.npy", gt_anim_vertices)
    np.save(output_dir / "frame_times.npy", frame_times)
    np.save(output_dir / "animated_mesh_weights.npy", frame_weights)

    sample_meta = {
        "variant": "real_glb_asset",
        "source": {
            "rigged_glb": str(Path(rigged_glb_path).resolve()),
            "animated_glb": str(Path(animated_glb_path).resolve()),
        },
        "frame_count": int(gt_anim_vertices.shape[0]),
        "joint_count": int(len(rig_json["joints"])),
        "vertex_count": int(gt_anim_vertices.shape[1]),
        "note": "rest mesh comes from merged rigged.glb; animated frames come from animated_mesh.glb morph targets",
        "rig_cleanup": rig_cleanup_summary,
    }
    with (output_dir / "sample_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, indent=2)

    if config_template_path is not None:
        config_template_path = Path(config_template_path)
        if config_template_path.exists():
            with config_template_path.open("r", encoding="utf-8") as src, (output_dir / "config.yaml").open(
                "w", encoding="utf-8"
            ) as dst:
                dst.write(src.read())

    return {
        "output_dir": str(output_dir.resolve()),
        "frame_count": int(gt_anim_vertices.shape[0]),
        "vertex_count": int(gt_anim_vertices.shape[1]),
        "joint_count": int(len(rig_json["joints"])),
        "rig_cleanup": rig_cleanup_summary,
    }
