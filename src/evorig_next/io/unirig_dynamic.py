from __future__ import annotations

import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import torch
from pygltflib import GLTF2
from scipy.spatial import cKDTree

from evorig_next.io.real_glb import (
    BONE_NAME_RE,
    _compute_node_world_matrices,
    _load_real_preprocess_config,
    _preprocess_real_rig,
    _read_accessor_dense,
)
from evorig_next.utils.mesh_ops import points_inside_or_on_mesh, project_points_inside_mesh


def _resolve_blender_path(blender_path: str | Path | None) -> str:
    if blender_path is not None and str(blender_path):
        path = Path(blender_path)
        if not path.exists():
            raise FileNotFoundError(f"Blender executable not found: {path}")
        return str(path)
    default = Path(r"D:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
    if default.exists():
        return str(default)
    resolved = shutil.which("blender")
    if resolved:
        return resolved
    raise FileNotFoundError("Blender executable not found; pass --blender-path")


def _read_weight_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    values = _read_accessor_dense(gltf, accessor_index).astype(np.float32)
    accessor = gltf.accessors[accessor_index]
    if accessor.componentType == 5121 and (accessor.normalized or float(values.max(initial=0.0)) > 1.0):
        values = values / 255.0
    elif accessor.componentType == 5123 and (accessor.normalized or float(values.max(initial=0.0)) > 1.0):
        values = values / 65535.0
    return values


def _bone_node_to_id(gltf: GLTF2) -> dict[int, int]:
    numbered_nodes: list[tuple[int, int]] = []
    for node_index, node in enumerate(gltf.nodes or []):
        if not node.name:
            continue
        match = BONE_NAME_RE.match(str(node.name))
        if match is not None:
            numbered_nodes.append((int(match.group(1)), int(node_index)))
    numbered_nodes.sort(key=lambda item: item[0])
    return {node_index: dense_id for dense_id, (_, node_index) in enumerate(numbered_nodes)}


def _extract_dense_bone_rig(rigged_glb_path: str | Path) -> dict[str, Any]:
    gltf = GLTF2().load_binary(str(rigged_glb_path))
    world_mats, parent_of = _compute_node_world_matrices(gltf)
    numbered_nodes: list[tuple[int, int]] = []
    for node_index, node in enumerate(gltf.nodes or []):
        if not node.name:
            continue
        match = BONE_NAME_RE.match(str(node.name))
        if match is not None:
            numbered_nodes.append((int(match.group(1)), int(node_index)))
    if not numbered_nodes:
        raise ValueError(f"no bone_* nodes found in {rigged_glb_path}")
    numbered_nodes.sort(key=lambda item: item[0])
    node_to_dense = {node_index: dense_id for dense_id, (_, node_index) in enumerate(numbered_nodes)}
    joints: list[dict[str, Any]] = []
    for dense_id, (source_bone_id, node_index) in enumerate(numbered_nodes):
        parent_node = parent_of[node_index]
        parent_bone = -1
        while parent_node is not None:
            if parent_node in node_to_dense:
                parent_bone = int(node_to_dense[parent_node])
                break
            parent_node = parent_of[parent_node]
        position = world_mats[node_index][:3, 3].astype(np.float32)
        joints.append(
            {
                "id": int(dense_id),
                "parent_id": int(parent_bone),
                "connected_to_parent": bool(parent_bone >= 0),
                "rest_position": position.tolist(),
                "birth_step": 0,
                "is_inserted": False,
                "birth_mode": "seed",
                "name": str(gltf.nodes[node_index].name),
                "source_node_id": int(node_index),
                "source_bone_id": int(source_bone_id),
            }
        )
    return {"joints": joints}


def _extract_fbx_connectivity(
    skeleton_fbx_path: str | Path,
    *,
    blender_path: str | Path | None = None,
) -> dict[int, bool]:
    skeleton_fbx_path = Path(skeleton_fbx_path)
    if not skeleton_fbx_path.exists():
        return {}
    blender = _resolve_blender_path(blender_path)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        script_path = Path(handle.name)
        handle.write(
            r'''
import json
import re
import sys
from pathlib import Path

import bpy

fbx = Path(sys.argv[-2])
out = Path(sys.argv[-1])
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=str(fbx))
arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
if not arms:
    raise SystemExit(f"no armature found in {fbx}")
arm = max(arms, key=lambda obj: len(obj.data.bones))
bone_name_re = re.compile(r"^bone[_\.\-\s]*(\d+)$", re.IGNORECASE)
records = []
for bone in arm.data.bones:
    match = bone_name_re.match(str(bone.name))
    if match is None:
        continue
    source_bone_id = int(match.group(1))
    parent = bone.parent
    parent_source_bone_id = -1
    if parent is not None:
        parent_match = bone_name_re.match(str(parent.name))
        if parent_match is not None:
            parent_source_bone_id = int(parent_match.group(1))
    records.append({
        "source_bone_id": source_bone_id,
        "parent_source_bone_id": parent_source_bone_id,
        "connected_to_parent": bool(bone.use_connect and parent_source_bone_id >= 0),
        "name": str(bone.name),
    })
out.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
'''
        )
    output_json = skeleton_fbx_path.with_suffix(".connectivity.tmp.json")
    try:
        cmd = [str(blender), "--background", "--python", str(script_path), "--", str(skeleton_fbx_path), str(output_json)]
        subprocess.run(cmd, check=True)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    finally:
        try:
            script_path.unlink()
        except OSError:
            pass
        try:
            output_json.unlink()
        except OSError:
            pass
    return {
        int(record["source_bone_id"]): bool(record["connected_to_parent"])
        for record in payload.get("records", [])
    }


def _apply_fbx_connectivity_override(
    rig_json: dict[str, Any],
    connectivity_by_source_bone: dict[int, bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not connectivity_by_source_bone:
        return rig_json, {"enabled": False, "reason": "missing_fbx_connectivity"}
    updated = {key: value for key, value in rig_json.items() if key != "joints"}
    joints = [dict(joint) for joint in rig_json.get("joints", [])]
    changed = 0
    missing = 0
    false_count = 0
    true_count = 0
    for joint in joints:
        source_id = int(joint.get("source_bone_id", joint.get("id", -1)))
        parent_id = int(joint.get("parent_id", -1))
        if source_id not in connectivity_by_source_bone:
            missing += 1
            joint["connected_to_parent"] = bool(joint.get("connected_to_parent", parent_id >= 0)) and parent_id >= 0
            continue
        new_value = bool(connectivity_by_source_bone[source_id]) and parent_id >= 0
        old_value = bool(joint.get("connected_to_parent", parent_id >= 0)) and parent_id >= 0
        if new_value != old_value:
            changed += 1
        joint["connected_to_parent"] = new_value
        true_count += int(new_value)
        false_count += int(not new_value)
    updated["joints"] = joints
    return updated, {
        "enabled": True,
        "source": "skeleton.fbx use_connect",
        "changed_edge_count": int(changed),
        "missing_source_bone_count": int(missing),
        "connected_true_count": int(true_count),
        "connected_false_count": int(false_count),
    }


def _load_rigged_mesh_and_skin(rigged_glb_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    gltf = GLTF2().load_binary(str(rigged_glb_path))
    world_mats, _ = _compute_node_world_matrices(gltf)
    node_to_bone = _bone_node_to_id(gltf)
    if not node_to_bone:
        raise ValueError(f"no bone_* nodes found in {rigged_glb_path}")
    joint_count = max(node_to_bone.values()) + 1

    vertices_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    weights_chunks: list[np.ndarray] = []
    primitive_reports: list[dict[str, Any]] = []
    vertex_offset = 0

    for node_index, node in enumerate(gltf.nodes or []):
        if node.mesh is None:
            continue
        mesh = gltf.meshes[int(node.mesh)]
        node_world = world_mats[node_index].astype(np.float32)
        skin = gltf.skins[int(node.skin)] if node.skin is not None and gltf.skins else None
        skin_joints = [int(item) for item in (skin.joints if skin is not None else [])]

        for primitive_index, primitive in enumerate(mesh.primitives or []):
            attrs = primitive.attributes
            if attrs is None or attrs.POSITION is None:
                continue
            positions = _read_accessor_dense(gltf, int(attrs.POSITION)).astype(np.float32)
            homogeneous = np.concatenate([positions, np.ones((positions.shape[0], 1), dtype=np.float32)], axis=1)
            positions_world = (homogeneous @ node_world.T)[:, :3].astype(np.float32)

            if primitive.indices is None:
                if positions.shape[0] % 3 != 0:
                    raise ValueError(f"primitive {primitive_index} has no indices and non-triangle vertex count")
                faces = np.arange(positions.shape[0], dtype=np.int64).reshape(-1, 3)
            else:
                indices = _read_accessor_dense(gltf, int(primitive.indices)).astype(np.int64).reshape(-1)
                if indices.shape[0] % 3 != 0:
                    raise ValueError(f"primitive {primitive_index} index count is not divisible by 3")
                faces = indices.reshape(-1, 3)
            faces_chunks.append(faces + vertex_offset)
            vertices_chunks.append(positions_world)

            dense_weights = np.zeros((positions.shape[0], joint_count), dtype=np.float32)
            joints_accessor = getattr(attrs, "JOINTS_0", None)
            weights_accessor = getattr(attrs, "WEIGHTS_0", None)
            if joints_accessor is not None and weights_accessor is not None and skin_joints:
                local_joint_ids = _read_accessor_dense(gltf, int(joints_accessor)).astype(np.int64)
                local_weights = _read_weight_accessor(gltf, int(weights_accessor)).astype(np.float32)
                if local_joint_ids.shape != local_weights.shape:
                    raise ValueError(f"JOINTS_0 shape {local_joint_ids.shape} != WEIGHTS_0 shape {local_weights.shape}")
                for influence_index in range(local_joint_ids.shape[1]):
                    local_ids = local_joint_ids[:, influence_index]
                    influence_weights = local_weights[:, influence_index]
                    for skin_joint_local in np.unique(local_ids):
                        skin_joint_local = int(skin_joint_local)
                        if skin_joint_local < 0 or skin_joint_local >= len(skin_joints):
                            continue
                        bone_node = skin_joints[skin_joint_local]
                        bone_id = node_to_bone.get(int(bone_node))
                        if bone_id is None:
                            continue
                        mask = local_ids == skin_joint_local
                        dense_weights[mask, int(bone_id)] += influence_weights[mask]
                row_sum = dense_weights.sum(axis=1, keepdims=True)
                active = row_sum[:, 0] > 1.0e-8
                dense_weights[active] = dense_weights[active] / row_sum[active]
            weights_chunks.append(dense_weights)
            primitive_reports.append(
                {
                    "node_index": int(node_index),
                    "node_name": str(node.name or ""),
                    "mesh_index": int(node.mesh),
                    "primitive_index": int(primitive_index),
                    "vertex_count": int(positions_world.shape[0]),
                    "face_count": int(faces.shape[0]),
                    "has_skin_weights": bool(dense_weights.sum() > 0.0),
                }
            )
            vertex_offset += int(positions_world.shape[0])

    if not vertices_chunks:
        raise ValueError(f"no mesh primitives found in {rigged_glb_path}")
    vertices = np.concatenate(vertices_chunks, axis=0).astype(np.float32)
    faces = np.concatenate(faces_chunks, axis=0).astype(np.int64)
    weights = np.concatenate(weights_chunks, axis=0).astype(np.float32)
    report = {
        "primitive_count": int(len(primitive_reports)),
        "primitives": primitive_reports,
        "joint_count": int(joint_count),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "skinned_vertex_count": int((weights.sum(axis=1) > 1.0e-8).sum()),
    }
    return vertices, faces, weights, report


def _run_blender_bake(
    *,
    dynamic_glb_path: str | Path,
    blender_path: str | Path | None,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_npz = cache_dir / "baked_dynamic_frames_raw_blender.npz"
    report_json = cache_dir / "baked_dynamic_frames_report.json"
    if not frames_npz.exists():
        cmd = [
            _resolve_blender_path(blender_path),
            "--background",
            "--python",
            str(Path(__file__).resolve().parents[3] / "scripts" / "blender_bake_dynamic_glb_frames.py"),
            "--",
            str(dynamic_glb_path),
            str(frames_npz),
            str(report_json),
        ]
        subprocess.run(cmd, check=True)
    data = np.load(frames_npz, allow_pickle=True)
    frames = np.asarray(data["frames"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)
    source_times = np.asarray(data["source_times"], dtype=np.float32)
    report = json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else {}
    return frames, faces, source_times, report


def _best_axis_alignment(dynamic_rest: np.ndarray, rest_vertices: np.ndarray) -> dict[str, Any]:
    rest_center = rest_vertices.mean(axis=0)
    best: dict[str, Any] | None = None
    for perm in itertools.permutations(range(3)):
        permuted = dynamic_rest[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            signs_arr = np.asarray(signs, dtype=np.float32)
            oriented = permuted * signs_arr[None, :]
            translation = rest_center - oriented.mean(axis=0)
            aligned = oriented + translation[None, :]
            tree = cKDTree(aligned)
            distances, indices = tree.query(rest_vertices, k=1)
            score = float(np.percentile(distances, 95))
            mean = float(distances.mean())
            max_value = float(distances.max(initial=0.0))
            candidate = {
                "perm": [int(item) for item in perm],
                "signs": [int(item) for item in signs],
                "translation": translation.astype(float).tolist(),
                "mean": mean,
                "p95": score,
                "max": max_value,
                "indices": indices.astype(np.int64),
                "distances": distances.astype(np.float32),
            }
            if best is None or (candidate["p95"], candidate["mean"], candidate["max"]) < (
                best["p95"],
                best["mean"],
                best["max"],
            ):
                best = candidate
    if best is None:
        raise RuntimeError("axis alignment search produced no candidates")
    return best


def _apply_axis_alignment(frames: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    perm = [int(item) for item in alignment["perm"]]
    signs = np.asarray(alignment["signs"], dtype=np.float32)
    translation = np.asarray(alignment["translation"], dtype=np.float32)
    return frames[:, :, perm] * signs[None, None, :] + translation[None, None, :]


def _motion_stats(frames: np.ndarray) -> dict[str, np.ndarray | dict[str, float]]:
    if frames.shape[0] <= 1:
        zeros = np.zeros((frames.shape[0],), dtype=np.float32)
        return {
            "step_motion": zeros,
            "rest_motion": zeros,
            "acceleration_motion": zeros,
            "selection_score": zeros,
            "summary": {},
        }
    step_motion = np.zeros((frames.shape[0],), dtype=np.float32)
    step_motion[1:] = np.linalg.norm(frames[1:] - frames[:-1], axis=-1).mean(axis=1)
    rest_motion = np.linalg.norm(frames - frames[:1], axis=-1).mean(axis=1).astype(np.float32)
    acceleration = np.zeros((frames.shape[0],), dtype=np.float32)
    if frames.shape[0] > 2:
        acceleration[2:] = np.linalg.norm(frames[2:] - 2.0 * frames[1:-1] + frames[:-2], axis=-1).mean(axis=1)

    def normalize(values: np.ndarray) -> np.ndarray:
        v_min = float(values.min(initial=0.0))
        v_max = float(values.max(initial=0.0))
        if v_max <= v_min + 1.0e-12:
            return np.zeros_like(values)
        return (values - v_min) / (v_max - v_min)

    selection_score = (normalize(step_motion) + normalize(rest_motion) + normalize(acceleration)) / 3.0
    summary = {}
    for name, values in (
        ("step_motion", step_motion),
        ("rest_motion", rest_motion),
        ("acceleration_motion", acceleration),
        ("selection_score", selection_score),
    ):
        summary[name] = {
            "mean": float(values.mean()),
            "max": float(values.max(initial=0.0)),
            "p95": float(np.percentile(values, 95)),
        }
    return {
        "step_motion": step_motion,
        "rest_motion": rest_motion,
        "acceleration_motion": acceleration,
        "selection_score": selection_score.astype(np.float32),
        "summary": summary,
    }


def _top_frames(values: np.ndarray, *, count: int = 12) -> list[dict[str, float]]:
    if values.size == 0:
        return []
    order = np.argsort(-values)[: int(count)]
    return [{"frame": int(frame_id), "value": float(values[frame_id])} for frame_id in order.tolist()]


def _select_frames(
    frames: np.ndarray,
    *,
    max_frames: int,
    uniform_fraction: float,
    include_last: bool,
    motion_min_gap: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray | dict[str, float]]]:
    frame_count = int(frames.shape[0])
    stats = _motion_stats(frames)
    if frame_count <= int(max_frames):
        selected = np.arange(frame_count, dtype=np.int64)
    else:
        uniform_count = max(2, int(math.ceil(float(max_frames) * float(uniform_fraction))))
        uniform = np.linspace(0, frame_count - 1, uniform_count).round().astype(np.int64)
        selected_set = {int(item) for item in uniform.tolist()}
        selected_set.add(0)
        if include_last:
            selected_set.add(frame_count - 1)
        score = np.asarray(stats["selection_score"], dtype=np.float32)
        for frame_id in np.argsort(-score).tolist():
            if len(selected_set) >= int(max_frames):
                break
            frame_id = int(frame_id)
            if frame_id in selected_set:
                continue
            if int(motion_min_gap) > 0 and any(abs(frame_id - existing) < int(motion_min_gap) for existing in selected_set):
                continue
            selected_set.add(frame_id)
            if len(selected_set) >= int(max_frames):
                break
        selected = np.asarray(sorted(selected_set), dtype=np.int64)
    policy = {
        "frame_count": frame_count,
        "selected_frame_count": int(selected.shape[0]),
        "selected_frame_indices": [int(item) for item in selected.tolist()],
        "max_frames": int(max_frames),
        "uniform_fraction": float(uniform_fraction),
        "include_last": bool(include_last),
        "motion_min_gap": int(motion_min_gap),
        "selection_policy": "rest_plus_uniform_plus_motion_energy",
        "motion_metric": stats["summary"],
        "top_step_motion_frames": _top_frames(np.asarray(stats["step_motion"]), count=12),
        "top_rest_motion_frames": _top_frames(np.asarray(stats["rest_motion"]), count=12),
        "top_selection_score_frames": _top_frames(np.asarray(stats["selection_score"]), count=12),
    }
    return selected, policy, stats


def _normalization_transform(rest_vertices: np.ndarray, target_diag: float | None) -> dict[str, Any]:
    bounds_min = rest_vertices.min(axis=0)
    bounds_max = rest_vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    diag = float(np.linalg.norm(bounds_max - bounds_min))
    if target_diag is None:
        scale = 1.0
    else:
        scale = float(target_diag) / max(diag, 1.0e-8)
    return {
        "mode": "bbox_center_diag",
        "target_diag": None if target_diag is None else float(target_diag),
        "source_bounds_min": bounds_min.astype(float).tolist(),
        "source_bounds_max": bounds_max.astype(float).tolist(),
        "source_diag": diag,
        "center": center.astype(float).tolist(),
        "scale": float(scale),
    }


def _apply_normalization(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    center = np.asarray(transform["center"], dtype=np.float32)
    scale = float(transform["scale"])
    return ((points - center.reshape((1,) * (points.ndim - 1) + (3,))) * scale).astype(np.float32)


def _load_prepared_window_normalization(source_dir: Path) -> dict[str, Any] | None:
    report_path = source_dir / "window_prepare_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    normalization = report.get("normalization")
    return dict(normalization) if isinstance(normalization, dict) else None


def _quality_checks(vertices: np.ndarray, faces: np.ndarray, frames: np.ndarray) -> dict[str, Any]:
    face_vertices = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]),
        axis=1,
    )
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    diag = float(np.linalg.norm(bounds_max - bounds_min))
    return {
        "has_nan_or_inf": bool((~np.isfinite(vertices)).any() or (~np.isfinite(frames)).any()),
        "degenerate_face_count": int((double_area <= max(diag * diag, 1.0e-12) * 1.0e-12).sum()),
        "bounds_min": bounds_min.astype(float).tolist(),
        "bounds_max": bounds_max.astype(float).tolist(),
        "bbox_diag": diag,
        "frame_count": int(frames.shape[0]),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
    }


def _collapse_weight_columns(weights: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    cleaned = np.asarray(weights, dtype=np.float32).copy()
    for record in records:
        removed_joint = int(record["removed_joint_id"])
        target_joint = int(record["target_joint_id"])
        if removed_joint < 0 or removed_joint >= cleaned.shape[1]:
            raise ValueError(f"cleanup record removed joint {removed_joint} outside weights shape {cleaned.shape}")
        if target_joint < 0 or target_joint >= cleaned.shape[1]:
            raise ValueError(f"cleanup record target joint {target_joint} outside weights shape {cleaned.shape}")
        if target_joint != removed_joint:
            cleaned[:, target_joint] += cleaned[:, removed_joint]
        cleaned = np.delete(cleaned, removed_joint, axis=1)
    row_sum = cleaned.sum(axis=1, keepdims=True)
    active = row_sum[:, 0] > 1.0e-8
    cleaned[active] = cleaned[active] / row_sum[active]
    return cleaned.astype(np.float32)


def _remap_skin_weights_after_cleanup(weights: np.ndarray, cleanup_summary: dict[str, Any]) -> np.ndarray:
    records: list[dict[str, Any]] = []
    records.extend(list(cleanup_summary.get("outside_joint_removals", [])))
    records.extend(list(cleanup_summary.get("near_pair_merges", [])))
    return _collapse_weight_columns(weights, records)


def _collapse_joint_in_rig_json(rig_json: dict[str, Any], *, removed_joint: int, target_joint: int) -> dict[str, Any]:
    joints = sorted(rig_json["joints"], key=lambda item: int(item["id"]))
    joint_count = len(joints)
    if removed_joint <= 0 or removed_joint >= joint_count:
        raise ValueError(f"cannot remove joint {removed_joint} from rig with {joint_count} joints")
    if target_joint < 0 or target_joint >= joint_count or target_joint == removed_joint:
        raise ValueError(f"invalid target joint {target_joint} for removed joint {removed_joint}")
    old_to_new = {
        old_id: new_id
        for new_id, old_id in enumerate(old_id for old_id in range(joint_count) if old_id != int(removed_joint))
    }
    new_joints: list[dict[str, Any]] = []
    for joint in joints:
        old_id = int(joint["id"])
        if old_id == int(removed_joint):
            continue
        new_joint = dict(joint)
        parent_id = int(new_joint["parent_id"])
        if parent_id == int(removed_joint):
            parent_id = int(target_joint)
        new_joint["id"] = int(old_to_new[old_id])
        new_joint["parent_id"] = -1 if parent_id < 0 else int(old_to_new[parent_id])
        new_joint["connected_to_parent"] = bool(new_joint.get("connected_to_parent", parent_id >= 0)) and parent_id >= 0
        new_joints.append(new_joint)
    new_joints.sort(key=lambda item: int(item["id"]))
    return {"joints": new_joints}


def _collapse_degenerate_bones(
    rig_json: dict[str, Any],
    weights: np.ndarray,
    *,
    min_bone_length: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    cleaned_rig = {"joints": [dict(joint) for joint in sorted(rig_json["joints"], key=lambda item: int(item["id"]))]}
    cleaned_weights = np.asarray(weights, dtype=np.float32).copy()
    records: list[dict[str, Any]] = []
    while True:
        joints = sorted(cleaned_rig["joints"], key=lambda item: int(item["id"]))
        positions = np.asarray([joint["rest_position"] for joint in joints], dtype=np.float32)
        parents = np.asarray([int(joint["parent_id"]) for joint in joints], dtype=np.int64)
        candidate: tuple[int, int, float] | None = None
        for joint_id, parent_id in enumerate(parents.tolist()):
            if joint_id == 0 or parent_id < 0:
                continue
            bone_length = float(np.linalg.norm(positions[joint_id] - positions[parent_id]))
            if bone_length <= float(min_bone_length):
                candidate = (int(joint_id), int(parent_id), bone_length)
                break
        if candidate is None:
            break
        removed_joint, target_joint, bone_length = candidate
        records.append(
            {
                "removed_joint_id": int(removed_joint),
                "target_joint_id": int(target_joint),
                "bone_length": float(bone_length),
                "min_bone_length": float(min_bone_length),
                "rest_position": positions[removed_joint].astype(float).tolist(),
            }
        )
        cleaned_weights = _collapse_weight_columns(
            cleaned_weights,
            [{"removed_joint_id": int(removed_joint), "target_joint_id": int(target_joint)}],
        )
        cleaned_rig = _collapse_joint_in_rig_json(
            cleaned_rig,
            removed_joint=int(removed_joint),
            target_joint=int(target_joint),
        )
    summary = {
        "enabled": True,
        "min_bone_length": float(min_bone_length),
        "collapse_count": int(len(records)),
        "collapses": records,
    }
    return cleaned_rig, cleaned_weights, summary


def _project_rig_joints_inside_mesh(
    rig_json: dict[str, Any],
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    surface_tol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    joints = list(rig_json.get("joints", []))
    if not joints:
        return rig_json, {"enabled": False, "reason": "empty_rig"}

    vertices_t = torch.tensor(rest_vertices, dtype=torch.float32)
    faces_t = torch.tensor(faces, dtype=torch.long)
    joints_t = torch.tensor([joint["rest_position"] for joint in joints], dtype=torch.float32)
    tol = max(float(surface_tol), 3.0e-3)
    inside_before = points_inside_or_on_mesh(joints_t, vertices_t, faces_t, surface_tol=tol)
    if bool(inside_before.all().item()):
        return rig_json, {
            "enabled": True,
            "projected_count": 0,
            "surface_tol": float(tol),
            "max_projection_distance": 0.0,
        }

    padding = max(float(tol) * 2.0, 6.0e-3)
    projected, _inside_mask, outside_distance = project_points_inside_mesh(
        joints_t,
        vertices_t,
        faces_t,
        inward_hint=vertices_t.mean(dim=0, keepdim=True).expand(joints_t.shape[0], -1),
        padding=padding,
    )
    inside_after = points_inside_or_on_mesh(projected, vertices_t, faces_t, surface_tol=max(tol, padding))
    updated = {key: value for key, value in rig_json.items() if key != "joints"}
    updated["joints"] = [dict(joint) for joint in joints]
    for joint, position in zip(updated["joints"], projected.detach().cpu().numpy()):
        joint["rest_position"] = position.astype(float).tolist()
    outside_ids = torch.nonzero(~inside_before, as_tuple=False).flatten().detach().cpu().tolist()
    return updated, {
        "enabled": True,
        "projected_count": int(len(outside_ids)),
        "projected_joint_ids": [int(item) for item in outside_ids],
        "surface_tol": float(tol),
        "projection_padding": float(padding),
        "inside_before_count": int(inside_before.sum().item()),
        "inside_after_count": int(inside_after.sum().item()),
        "max_projection_distance": float(outside_distance.max().item()) if int(outside_distance.numel()) else 0.0,
        "projection_distance_by_joint": [float(item) for item in outside_distance.detach().cpu().tolist()],
    }


def _write_rig_json(path: Path, rig_json: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rig_json, handle, indent=2)


def import_unirig_dynamic_sample(
    *,
    asset: str,
    source_dir: str | Path,
    output_dir: str | Path,
    blender_path: str | Path | None = None,
    config_template_path: str | Path | None = None,
    max_frames: int = 32,
    uniform_fraction: float = 0.625,
    include_last: bool = True,
    motion_min_gap: int = 0,
    frame_start: int = 0,
    frame_end: int | None = None,
    alignment_frame_index: int | None = None,
    normalize_target_diag: float | None = 2.0,
    max_correspondence_p95_relative: float = 0.03,
    max_correspondence_max_relative: float = 0.08,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    dynamic_glb_path = source_dir / "dynamic_mesh.glb"
    rigged_glb_path = source_dir / "rigged.glb"
    if not dynamic_glb_path.exists():
        raise FileNotFoundError(dynamic_glb_path)
    if not rigged_glb_path.exists():
        raise FileNotFoundError(rigged_glb_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_import_cache"
    rest_vertices_raw, faces, unirig_weights, mesh_report = _load_rigged_mesh_and_skin(rigged_glb_path)
    dynamic_frames_raw, dynamic_faces_raw, source_times, bake_report = _run_blender_bake(
        dynamic_glb_path=dynamic_glb_path,
        blender_path=blender_path,
        cache_dir=cache_dir,
    )
    prepared_normalization = _load_prepared_window_normalization(source_dir)
    if prepared_normalization is not None:
        dynamic_frames_raw = _apply_normalization(dynamic_frames_raw, prepared_normalization)

    source_frame_count_raw = int(dynamic_frames_raw.shape[0])
    alignment_frame_index = int(frame_start) if alignment_frame_index is None else int(alignment_frame_index)
    alignment_frame_index = max(0, min(alignment_frame_index, source_frame_count_raw - 1))
    alignment = _best_axis_alignment(dynamic_frames_raw[alignment_frame_index], rest_vertices_raw)
    aligned_dynamic_frames = _apply_axis_alignment(dynamic_frames_raw, alignment)
    mapping = np.asarray(alignment["indices"], dtype=np.int64)
    distances = np.asarray(alignment["distances"], dtype=np.float32)
    rest_diag_raw = float(
        np.linalg.norm(rest_vertices_raw.max(axis=0) - rest_vertices_raw.min(axis=0))
    )
    correspondence = {
        "mean": float(distances.mean()),
        "p95": float(np.percentile(distances, 95)),
        "max": float(distances.max(initial=0.0)),
        "mean_relative": float(distances.mean() / max(rest_diag_raw, 1.0e-8)),
        "p95_relative": float(np.percentile(distances, 95) / max(rest_diag_raw, 1.0e-8)),
        "max_relative": float(distances.max(initial=0.0) / max(rest_diag_raw, 1.0e-8)),
        "unique_dynamic_source_vertices": int(np.unique(mapping).shape[0]),
        "rest_vertex_count": int(rest_vertices_raw.shape[0]),
        "dynamic_vertex_count": int(dynamic_frames_raw.shape[1]),
        "note": "nearest dynamic-frame0 vertex per UniRig-rest vertex; duplicate seam vertices are allowed",
    }
    if correspondence["p95_relative"] > float(max_correspondence_p95_relative):
        raise ValueError(
            f"dynamic/rest correspondence p95_relative={correspondence['p95_relative']:.6f} "
            f"> {max_correspondence_p95_relative:.6f}"
        )
    if correspondence["max_relative"] > float(max_correspondence_max_relative):
        raise ValueError(
            f"dynamic/rest correspondence max_relative={correspondence['max_relative']:.6f} "
            f"> {max_correspondence_max_relative:.6f}"
        )

    mapped_frames_raw = aligned_dynamic_frames[:, mapping, :].astype(np.float32)
    source_frame_count = int(mapped_frames_raw.shape[0])
    frame_start = max(0, int(frame_start))
    frame_end = source_frame_count - 1 if frame_end is None else min(int(frame_end), source_frame_count - 1)
    if frame_start > frame_end:
        raise ValueError(f"invalid frame window [{frame_start}, {frame_end}] for {source_frame_count} frames")
    window_indices = np.arange(frame_start, frame_end + 1, dtype=np.int64)
    window_frames_raw = mapped_frames_raw[window_indices]
    selected_local_indices, frame_policy, motion_stats = _select_frames(
        window_frames_raw,
        max_frames=int(max_frames),
        uniform_fraction=float(uniform_fraction),
        include_last=bool(include_last),
        motion_min_gap=int(motion_min_gap),
    )
    selected_indices = window_indices[selected_local_indices]
    frame_policy["frame_window_start"] = int(frame_start)
    frame_policy["frame_window_end"] = int(frame_end)
    frame_policy["frame_window_count"] = int(window_indices.shape[0])
    frame_policy["selected_local_indices"] = [int(item) for item in selected_local_indices.tolist()]
    frame_policy["selected_frame_indices"] = [int(item) for item in selected_indices.tolist()]
    selected_frames_raw = mapped_frames_raw[selected_indices]
    selected_times = source_times[selected_indices]

    normalization = _normalization_transform(rest_vertices_raw, normalize_target_diag)
    rest_vertices = _apply_normalization(rest_vertices_raw, normalization)
    selected_frames = _apply_normalization(selected_frames_raw, normalization)

    rig_json_raw = _extract_dense_bone_rig(rigged_glb_path)
    fbx_connectivity_summary: dict[str, Any] = {"enabled": False, "reason": "missing_skeleton_fbx"}
    skeleton_fbx_path = source_dir / "skeleton.fbx"
    if skeleton_fbx_path.exists():
        fbx_connectivity = _extract_fbx_connectivity(skeleton_fbx_path, blender_path=blender_path)
        rig_json_raw, fbx_connectivity_summary = _apply_fbx_connectivity_override(
            rig_json_raw,
            fbx_connectivity,
        )
    for joint in rig_json_raw["joints"]:
        joint["rest_position"] = _apply_normalization(
            np.asarray(joint["rest_position"], dtype=np.float32).reshape(1, 3),
            normalization,
        ).reshape(3).astype(float).tolist()
    preprocess_cfg = _load_real_preprocess_config(config_template_path)
    surface_tol = float(preprocess_cfg.get("surface_tol", 3.0e-3))
    rig_json_projected, joint_projection_summary = _project_rig_joints_inside_mesh(
        rig_json_raw,
        rest_vertices,
        faces,
        surface_tol=surface_tol,
    )
    preprocess_cfg["surface_tol"] = max(surface_tol, float(joint_projection_summary.get("projection_padding", surface_tol)))
    rig_json, rig_cleanup_summary = _preprocess_real_rig(
        rig_json=rig_json_projected,
        rest_vertices=rest_vertices,
        faces=faces,
        preprocess_cfg=preprocess_cfg,
    )
    cleaned_unirig_weights = _remap_skin_weights_after_cleanup(unirig_weights, rig_cleanup_summary)
    normalized_diag = float(np.linalg.norm(rest_vertices.max(axis=0) - rest_vertices.min(axis=0)))
    rig_json, cleaned_unirig_weights, degenerate_bone_summary = _collapse_degenerate_bones(
        rig_json,
        cleaned_unirig_weights,
        min_bone_length=max(normalized_diag * 1.0e-5, 1.0e-7),
    )
    if int(cleaned_unirig_weights.shape[1]) != int(len(rig_json["joints"])):
        raise ValueError(
            f"cleaned UniRig weights have {cleaned_unirig_weights.shape[1]} joints, "
            f"but cleaned rig has {len(rig_json['joints'])}"
        )

    rest_mesh = trimesh.Trimesh(vertices=rest_vertices, faces=faces, process=False)
    rest_mesh.export(output_dir / "rest_mesh.obj")
    _write_rig_json(output_dir / "wrong_init_rig.json", rig_json)
    np.save(output_dir / "gt_anim_vertices.npy", selected_frames.astype(np.float32))
    np.save(output_dir / "frame_indices.npy", selected_indices.astype(np.int64))
    np.save(output_dir / "frame_times.npy", selected_times.astype(np.float32))
    np.save(output_dir / "source_frame_times.npy", source_times.astype(np.float32))
    np.save(output_dir / "dynamic_to_rest_vertex_indices.npy", mapping.astype(np.int64))
    np.save(output_dir / "dynamic_to_rest_vertex_distance.npy", distances.astype(np.float32))
    np.save(output_dir / "unirig_skin_weights.npy", cleaned_unirig_weights.astype(np.float32))
    np.savez_compressed(
        output_dir / "motion_stats.npz",
        step_motion=np.asarray(motion_stats["step_motion"], dtype=np.float32),
        rest_motion=np.asarray(motion_stats["rest_motion"], dtype=np.float32),
        acceleration_motion=np.asarray(motion_stats["acceleration_motion"], dtype=np.float32),
        selection_score=np.asarray(motion_stats["selection_score"], dtype=np.float32),
    )
    with (output_dir / "motion_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(frame_policy["motion_metric"], handle, indent=2)

    quality = _quality_checks(rest_vertices, faces, selected_frames)
    sample_meta = {
        "variant": "evorig_unirig_dynamic_keyframes",
        "asset": str(asset),
        "source": {
            "source_dir": str(source_dir.resolve()),
            "dynamic_mesh_glb": str(dynamic_glb_path.resolve()),
            "rigged_glb": str(rigged_glb_path.resolve()),
        },
        "frame_count": int(selected_frames.shape[0]),
        "source_frame_count": int(dynamic_frames_raw.shape[0]),
        "vertex_count": int(rest_vertices.shape[0]),
        "dynamic_baked_vertex_count": int(dynamic_frames_raw.shape[1]),
        "face_count": int(faces.shape[0]),
        "joint_count": int(len(rig_json["joints"])),
        "initial_unirig_joint_count": int(mesh_report["joint_count"]),
        "frame_policy": frame_policy,
        "dynamic_alignment": {
            "align_frame_index": int(alignment_frame_index),
            "perm": alignment["perm"],
            "signs": alignment["signs"],
            "translation": alignment["translation"],
            "mean_relative": correspondence["mean_relative"],
            "p95_relative": correspondence["p95_relative"],
            "max_relative": correspondence["max_relative"],
            "prepared_dynamic_normalization_applied": prepared_normalization is not None,
        },
        "dynamic_correspondence": correspondence,
        "normalization": normalization,
        "quality": quality,
        "rigged_mesh": mesh_report,
        "fbx_connectivity_override": fbx_connectivity_summary,
        "unirig_skin_weights": {
            "initial_shape": [int(unirig_weights.shape[0]), int(unirig_weights.shape[1])],
            "cleaned_shape": [int(cleaned_unirig_weights.shape[0]), int(cleaned_unirig_weights.shape[1])],
            "cleanup_policy": "removed joint columns are merged into target_joint_id and then deleted",
        },
        "blender_bake": bake_report,
        "rig_cleanup": rig_cleanup_summary,
        "joint_projection": joint_projection_summary,
        "degenerate_bone_cleanup": degenerate_bone_summary,
        "note": "EvoRig uses only the cleaned UniRig initial skeleton. UniRig skin weights are saved only for fixed-LBS baseline.",
    }
    with (output_dir / "sample_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, indent=2)

    if config_template_path is not None:
        config_template_path = Path(config_template_path)
        if config_template_path.exists():
            shutil.copyfile(config_template_path, output_dir / "config.yaml")

    return {
        "output_dir": str(output_dir.resolve()),
        "frame_count": int(selected_frames.shape[0]),
        "source_frame_count": int(dynamic_frames_raw.shape[0]),
        "vertex_count": int(rest_vertices.shape[0]),
        "dynamic_baked_vertex_count": int(dynamic_frames_raw.shape[1]),
        "joint_count": int(len(rig_json["joints"])),
        "initial_unirig_joint_count": int(mesh_report["joint_count"]),
        "correspondence_p95_relative": correspondence["p95_relative"],
        "correspondence_max_relative": correspondence["max_relative"],
        "normalization_scale": float(normalization["scale"]),
        "selected_frame_indices": frame_policy["selected_frame_indices"],
    }
