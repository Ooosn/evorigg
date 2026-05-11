from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _parse_args() -> tuple[Path, Path, Path | None]:
    if "--" not in sys.argv:
        raise SystemExit(
            "Usage: blender --background --python blender_bake_dynamic_glb_frames.py -- "
            "<input.glb> <output.npz> [report.json]"
        )
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit(
            "Usage: blender --background --python blender_bake_dynamic_glb_frames.py -- "
            "<input.glb> <output.npz> [report.json]"
        )
    report_path = Path(args[2]) if len(args) > 2 else None
    return Path(args[0]), Path(args[1]), report_path


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.actions):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def _mesh_bounds(objects: list[bpy.types.Object]) -> dict[str, list[float]] | None:
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return None
    mins = [float(min(p[i] for p in points)) for i in range(3)]
    maxs = [float(max(p[i] for p in points)) for i in range(3)]
    return {
        "min": mins,
        "max": maxs,
        "extent": [float(maxs[i] - mins[i]) for i in range(3)],
    }


def _is_unwanted_helper_mesh(obj: bpy.types.Object) -> bool:
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


def _evaluated_mesh_arrays(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        try:
            mesh = bpy.data.meshes.new_from_object(
                evaluated,
                preserve_all_data_layers=True,
                depsgraph=depsgraph,
            )
        except TypeError:
            mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
        mesh.transform(evaluated.matrix_world)
        mesh.update()
        obj_vertices = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float32)
        obj_faces = np.asarray([polygon.vertices[:] for polygon in mesh.polygons], dtype=np.int64)
        if obj_faces.ndim != 2 or (obj_faces.size and obj_faces.shape[1] != 3):
            bpy.data.meshes.remove(mesh)
            raise RuntimeError(f"non-triangular evaluated mesh in object {obj.name}")
        vertices.append(obj_vertices)
        if obj_faces.size:
            faces.append(obj_faces + vertex_offset)
        vertex_offset += int(obj_vertices.shape[0])
        bpy.data.meshes.remove(mesh)
    merged_vertices = np.concatenate(vertices, axis=0) if vertices else np.zeros((0, 3), dtype=np.float32)
    merged_faces = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64)
    return merged_vertices, merged_faces


def main() -> None:
    input_path, output_path, report_path = _parse_args()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(input_path))

    all_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    skipped_meshes = [obj for obj in all_meshes if _is_unwanted_helper_mesh(obj)]
    imported_meshes = [obj for obj in all_meshes if obj not in skipped_meshes]
    imported_meshes.sort(key=lambda obj: obj.name)
    if not imported_meshes:
        raise SystemExit(f"No mesh objects found in {input_path}")

    frame_start = int(bpy.context.scene.frame_start)
    frame_end = int(bpy.context.scene.frame_end)
    frame_numbers = np.arange(frame_start, frame_end + 1, dtype=np.int32)

    frames: list[np.ndarray] = []
    reference_faces: np.ndarray | None = None
    vertex_count: int | None = None
    face_count: int | None = None
    topology_errors: list[dict[str, int]] = []
    for frame in frame_numbers.tolist():
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        vertices, faces = _evaluated_mesh_arrays(imported_meshes)
        if vertex_count is None:
            vertex_count = int(vertices.shape[0])
            face_count = int(faces.shape[0])
            reference_faces = faces.copy()
        if int(vertices.shape[0]) != vertex_count or int(faces.shape[0]) != face_count:
            topology_errors.append(
                {
                    "frame": int(frame),
                    "vertex_count": int(vertices.shape[0]),
                    "face_count": int(faces.shape[0]),
                }
            )
        frames.append(vertices)

    if topology_errors:
        raise RuntimeError(f"evaluated topology changed across frames: {topology_errors[:5]}")

    frame_array = np.stack(frames, axis=0).astype(np.float32)
    source_times = frame_numbers.astype(np.float32)
    np.savez_compressed(
        output_path,
        frames=frame_array,
        faces=reference_faces.astype(np.int64) if reference_faces is not None else np.zeros((0, 3), dtype=np.int64),
        frame_numbers=frame_numbers,
        source_times=source_times,
        object_names=np.asarray([obj.name for obj in imported_meshes]),
    )

    if report_path is not None:
        skipped = [
            {
                "name": obj.name,
                "vertices": int(len(obj.data.vertices)),
                "faces": int(len(obj.data.polygons)),
                "reason": "materialless origin Icosphere helper",
            }
            for obj in skipped_meshes
        ]
        report = {
            "source": str(input_path),
            "output": str(output_path),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": int(frame_array.shape[0]),
            "vertex_count": int(frame_array.shape[1]),
            "face_count": int(reference_faces.shape[0]) if reference_faces is not None else 0,
            "source_mesh_objects": int(len(all_meshes)),
            "baked_mesh_objects": int(len(imported_meshes)),
            "skipped_mesh_objects": skipped,
            "source_armatures": int(sum(1 for obj in bpy.context.scene.objects if obj.type == "ARMATURE")),
            "bounds": _mesh_bounds(imported_meshes),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
