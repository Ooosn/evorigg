from __future__ import annotations

import itertools
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree

from evorig_next.io.real_glb import (
    _load_real_preprocess_config,
    _preprocess_real_rig,
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


def _build_rig_from_blender_armature_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("Blender armature payload has no bones")
    record_by_name = {str(record["name"]): record for record in records}
    joints: list[dict[str, Any]] = []
    head_joint_by_bone: dict[str, int] = {}
    tail_joint_by_bone: dict[str, int] = {}
    visiting: set[str] = set()

    def add_joint(
        position: Any,
        *,
        parent_id: int,
        connected_to_parent: bool,
        name: str,
        bone_name: str,
        endpoint: str,
    ) -> int:
        joint_id = len(joints)
        joints.append(
            {
                "id": int(joint_id),
                "parent_id": int(parent_id),
                "connected_to_parent": bool(connected_to_parent and parent_id >= 0),
                "rest_position": np.asarray(position, dtype=np.float32).astype(float).tolist(),
                "birth_step": 0,
                "is_inserted": False,
                "birth_mode": "seed",
                "name": str(name),
                "source_bone_name": str(bone_name),
                "source_endpoint": str(endpoint),
            }
        )
        return joint_id

    def ensure_bone(bone_name: str) -> None:
        if bone_name in tail_joint_by_bone:
            return
        if bone_name in visiting:
            raise ValueError(f"cycle detected in armature bone hierarchy at {bone_name}")
        if bone_name not in record_by_name:
            raise KeyError(bone_name)
        visiting.add(bone_name)
        record = record_by_name[bone_name]
        parent_name = record.get("parent_name")
        if parent_name is not None and str(parent_name) in record_by_name:
            ensure_bone(str(parent_name))
            parent_tail = tail_joint_by_bone[str(parent_name)]
            if bool(record.get("use_connect", False)):
                head_joint = parent_tail
            else:
                head_joint = add_joint(
                    record["head"],
                    parent_id=parent_tail,
                    connected_to_parent=False,
                    name=f"{bone_name}_head",
                    bone_name=bone_name,
                    endpoint="head",
                )
        else:
            head_joint = add_joint(
                record["head"],
                parent_id=-1,
                connected_to_parent=False,
                name=f"{bone_name}_head",
                bone_name=bone_name,
                endpoint="head",
            )
        tail_joint = add_joint(
            record["tail"],
            parent_id=head_joint,
            connected_to_parent=True,
            name=f"{bone_name}_tail",
            bone_name=bone_name,
            endpoint="tail",
        )
        head_joint_by_bone[bone_name] = int(head_joint)
        tail_joint_by_bone[bone_name] = int(tail_joint)
        visiting.remove(bone_name)

    for record in records:
        ensure_bone(str(record["name"]))

    connected_edge_count = sum(
        bool(joint.get("connected_to_parent", False)) and int(joint.get("parent_id", -1)) >= 0
        for joint in joints
    )
    dashed_edge_count = sum(
        (not bool(joint.get("connected_to_parent", False))) and int(joint.get("parent_id", -1)) >= 0
        for joint in joints
    )
    summary = {
        "enabled": True,
        "source": payload.get("source", ""),
        "armature_name": payload.get("armature_name", ""),
        "bone_count": int(len(records)),
        "joint_count": int(len(joints)),
        "connected_edge_count": int(connected_edge_count),
        "dashed_hierarchy_edge_count": int(dashed_edge_count),
        "bone_head_joint_by_name": {key: int(value) for key, value in head_joint_by_bone.items()},
        "bone_tail_joint_by_name": {key: int(value) for key, value in tail_joint_by_bone.items()},
        "records": records,
    }
    return {"joints": joints}, summary


def _map_skin_weights_to_rig_joints(
    weights: np.ndarray,
    skin_joint_names: list[str],
    rig_json: dict[str, Any],
    armature_summary: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    joint_count = len(rig_json.get("joints", []))
    mapped = np.zeros((int(weights.shape[0]), int(joint_count)), dtype=np.float32)
    head_by_name = {str(key): int(value) for key, value in armature_summary.get("bone_head_joint_by_name", {}).items()}
    matched = 0
    missing: list[str] = []
    for source_id, name in enumerate(skin_joint_names):
        if source_id >= int(weights.shape[1]):
            break
        target = head_by_name.get(str(name))
        if target is None or target < 0 or target >= joint_count:
            missing.append(str(name))
            continue
        mapped[:, target] += weights[:, source_id]
        matched += 1
    return mapped, {
        "source": "skin joint columns mapped to Blender bone head joints by bone name",
        "input_shape": [int(weights.shape[0]), int(weights.shape[1])],
        "output_shape": [int(mapped.shape[0]), int(mapped.shape[1])],
        "skin_joint_count": int(len(skin_joint_names)),
        "matched_skin_joint_count": int(matched),
        "missing_skin_joint_count": int(len(missing)),
        "missing_skin_joint_names": missing[:64],
    }


def _run_blender_extract_rigged_scene(
    *,
    rigged_glb_path: str | Path,
    blender_path: str | Path | None,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scene_npz = cache_dir / "rigged_rest_scene_blender.npz"
    report_json = cache_dir / "rigged_rest_scene_report.json"
    if not scene_npz.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            script_path = Path(handle.name)
            handle.write(
                r'''
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.actions):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def _mesh_bounds(objects):
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return None
    mins = [float(min(p[i] for p in points)) for i in range(3)]
    maxs = [float(max(p[i] for p in points)) for i in range(3)]
    return {"min": mins, "max": maxs, "extent": [float(maxs[i] - mins[i]) for i in range(3)]}


def _is_unwanted_helper_mesh(obj):
    if obj.type != "MESH":
        return False
    if obj.parent is not None or obj.modifiers:
        return False
    if len(obj.data.materials) != 0:
        return False
    if not obj.name.lower().startswith("icosphere"):
        return False
    if len(obj.data.polygons) != 80:
        return False
    bounds = _mesh_bounds([obj])
    if bounds is None:
        return False
    extent = bounds["extent"]
    center = [(bounds["min"][i] + bounds["max"][i]) * 0.5 for i in range(3)]
    return all(abs(value) < 1e-4 for value in center) and all(1.8 <= value <= 2.1 for value in extent)


def _evaluated_mesh_arrays(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    try:
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
    except TypeError:
        mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    mesh.transform(evaluated.matrix_world)
    mesh.update()
    vertices = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float32)
    faces = np.asarray([polygon.vertices[:] for polygon in mesh.polygons], dtype=np.int64)
    bpy.data.meshes.remove(mesh)
    if faces.ndim != 2 or (faces.size and faces.shape[1] != 3):
        raise RuntimeError(f"non-triangular evaluated mesh in object {obj.name}")
    return vertices, faces


def main():
    src = Path(sys.argv[-3])
    out_npz = Path(sys.argv[-2])
    out_json = Path(sys.argv[-1])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(src))
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not arms:
        raise SystemExit(f"no armature found in {src}")
    arm = max(arms, key=lambda obj: len(obj.data.bones))
    arm.data.pose_position = "REST"
    bpy.context.scene.frame_set(int(bpy.context.scene.frame_start))
    bpy.context.view_layer.update()

    bone_names = [str(bone.name) for bone in arm.data.bones]
    bone_to_col = {name: idx for idx, name in enumerate(bone_names)}
    records = []
    for bone_index, bone in enumerate(arm.data.bones):
        parent = bone.parent
        head_world = arm.matrix_world @ bone.head_local
        tail_world = arm.matrix_world @ bone.tail_local
        records.append({
            "bone_index": int(bone_index),
            "name": str(bone.name),
            "parent_name": str(parent.name) if parent is not None else None,
            "use_connect": bool(bone.use_connect),
            "head": [float(value) for value in head_world],
            "tail": [float(value) for value in tail_world],
        })

    all_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    skipped_meshes = [obj for obj in all_meshes if _is_unwanted_helper_mesh(obj)]
    mesh_objects = [obj for obj in all_meshes if obj not in skipped_meshes]
    mesh_objects.sort(key=lambda obj: obj.name)
    if not mesh_objects:
        raise SystemExit(f"no mesh objects found in {src}")

    vertices_chunks = []
    faces_chunks = []
    weights_chunks = []
    reports = []
    vertex_offset = 0
    for obj in mesh_objects:
        vertices, faces = _evaluated_mesh_arrays(obj)
        if int(len(obj.data.vertices)) != int(vertices.shape[0]):
            raise RuntimeError(
                f"evaluated vertex count changed for {obj.name}: source={len(obj.data.vertices)}, evaluated={vertices.shape[0]}"
            )
        dense_weights = np.zeros((int(vertices.shape[0]), len(bone_names)), dtype=np.float32)
        vertex_groups = {int(group.index): str(group.name) for group in obj.vertex_groups}
        for vertex in obj.data.vertices:
            row = int(vertex.index)
            for group_ref in vertex.groups:
                group_name = vertex_groups.get(int(group_ref.group))
                col = bone_to_col.get(str(group_name))
                if col is not None:
                    dense_weights[row, int(col)] += float(group_ref.weight)
        row_sum = dense_weights.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > 1.0e-8
        dense_weights[active] = dense_weights[active] / row_sum[active]
        vertices_chunks.append(vertices)
        faces_chunks.append(faces + vertex_offset)
        weights_chunks.append(dense_weights)
        reports.append({
            "name": str(obj.name),
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "vertex_group_count": int(len(obj.vertex_groups)),
            "skinned_vertex_count": int(active.sum()),
        })
        vertex_offset += int(vertices.shape[0])

    vertices = np.concatenate(vertices_chunks, axis=0).astype(np.float32)
    faces = np.concatenate(faces_chunks, axis=0).astype(np.int64)
    weights = np.concatenate(weights_chunks, axis=0).astype(np.float32)
    np.savez_compressed(
        out_npz,
        vertices=vertices,
        faces=faces,
        weights=weights,
        bone_names=np.asarray(bone_names),
    )
    report = {
        "source": str(src),
        "armature_name": str(arm.name),
        "bone_count": int(len(bone_names)),
        "skin_joint_names": bone_names,
        "joint_count": int(len(bone_names)),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "skinned_vertex_count": int((weights.sum(axis=1) > 1.0e-8).sum()),
        "mesh_objects": reports,
        "source_mesh_objects": int(len(all_meshes)),
        "baked_mesh_objects": int(len(mesh_objects)),
        "skipped_mesh_objects": [str(obj.name) for obj in skipped_meshes],
        "bounds": _mesh_bounds(mesh_objects),
        "armature_payload": {
            "source": str(src),
            "armature_name": str(arm.name),
            "records": records,
        },
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''
            )
        try:
            cmd = [
                _resolve_blender_path(blender_path),
                "--background",
                "--python",
                str(script_path),
                "--",
                str(rigged_glb_path),
                str(scene_npz),
                str(report_json),
            ]
            subprocess.run(cmd, check=True)
        finally:
            try:
                script_path.unlink()
            except OSError:
                pass
    data = np.load(scene_npz, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)
    weights = np.asarray(data["weights"], dtype=np.float32)
    report = json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else {}
    report["skin_joint_names"] = [str(item) for item in data["bone_names"].tolist()]
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
    rest_vertices_raw, faces, unirig_weights, mesh_report = _run_blender_extract_rigged_scene(
        rigged_glb_path=rigged_glb_path,
        blender_path=blender_path,
        cache_dir=cache_dir,
    )
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

    rig_json_raw, armature_import_summary = _build_rig_from_blender_armature_payload(
        dict(mesh_report.get("armature_payload", {}))
    )
    armature_import_summary["selected_source"] = "rigged.glb_blender_scene"
    for joint in rig_json_raw["joints"]:
        joint["rest_position"] = _apply_normalization(
            np.asarray(joint["rest_position"], dtype=np.float32).reshape(1, 3),
            normalization,
        ).reshape(3).astype(float).tolist()
    unirig_weights_for_rig, skin_mapping_summary = _map_skin_weights_to_rig_joints(
        unirig_weights,
        [str(item) for item in mesh_report.get("skin_joint_names", [])],
        rig_json_raw,
        armature_import_summary,
    )
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
    cleaned_unirig_weights = _remap_skin_weights_after_cleanup(unirig_weights_for_rig, rig_cleanup_summary)
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
        "armature_import": armature_import_summary,
        "unirig_skin_weights": {
            "initial_shape": [int(unirig_weights.shape[0]), int(unirig_weights.shape[1])],
            "mapped_shape": [int(unirig_weights_for_rig.shape[0]), int(unirig_weights_for_rig.shape[1])],
            "cleaned_shape": [int(cleaned_unirig_weights.shape[0]), int(cleaned_unirig_weights.shape[1])],
            "mapping": skin_mapping_summary,
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
