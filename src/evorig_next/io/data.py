from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from evorig_next.utils.device import resolve_device
from evorig_next.utils.geometry import assert_shape


REQUIRED_FILES = ("rest_mesh.obj", "wrong_init_rig.json", "gt_anim_vertices.npy")


def _load_optional_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def validate_tree(parent_idx: torch.Tensor) -> None:
    if parent_idx.ndim != 1:
        raise ValueError("parent_idx must be rank-1")
    joint_count = int(parent_idx.shape[0])
    visited = [0] * joint_count
    stack = [0] * joint_count

    def dfs(node: int) -> None:
        visited[node] = 1
        stack[node] = 1
        children = torch.nonzero(parent_idx == node, as_tuple=False).flatten().tolist()
        for child in children:
            if not visited[child]:
                dfs(child)
            elif stack[child]:
                raise ValueError("rig hierarchy contains a cycle")
        stack[node] = 0

    roots = torch.nonzero(parent_idx == -1, as_tuple=False).flatten()
    if roots.numel() != 1:
        raise ValueError("rig must have exactly one root")
    if int(roots[0].item()) != 0:
        raise ValueError("root joint id must be 0")
    dfs(0)
    if not all(visited):
        raise ValueError("rig contains disconnected joints")


def load_rig_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "joints" in raw:
        joints = raw["joints"]
    else:
        joints = raw
    if not isinstance(joints, list) or not joints:
        raise ValueError("wrong_init_rig.json must contain a non-empty joint list")
    joints = sorted(joints, key=lambda item: item["id"])
    ids = [joint["id"] for joint in joints]
    if ids != list(range(len(joints))):
        raise ValueError("joint ids must be contiguous 0..J-1")
    parent_idx = torch.tensor([joint["parent_id"] for joint in joints], dtype=torch.long)
    rest_joints = torch.tensor([joint["rest_position"] for joint in joints], dtype=torch.float32)
    connected_to_parent = torch.tensor(
        [
            bool(joint.get("connected_to_parent", joint.get("connected", int(joint["parent_id"]) >= 0)))
            and int(joint["parent_id"]) >= 0
            for joint in joints
        ],
        dtype=torch.bool,
    )
    validate_tree(parent_idx)
    assert_shape(rest_joints, (len(joints), 3), "rest_joints")
    bind_transforms = None
    if all("bind_transform" in joint for joint in joints):
        bind_transforms = torch.tensor([joint["bind_transform"] for joint in joints], dtype=torch.float32)
    birth_steps = [int(joint.get("birth_step", 0)) for joint in joints]
    inserted = [bool(joint.get("is_inserted", False)) for joint in joints]
    birth_modes = [str(joint.get("birth_mode", "seed" if not bool(joint.get("is_inserted", False)) else "inserted")) for joint in joints]
    return {
        "parent_idx": parent_idx,
        "rest_joints": rest_joints,
        "connected_to_parent": connected_to_parent,
        "bind_transforms": bind_transforms,
        "birth_steps": birth_steps,
        "inserted": inserted,
        "birth_modes": birth_modes,
    }


def load_sample(data_dir: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    resolved_device = resolve_device(device)
    base = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    mesh = trimesh.load_mesh(base / "rest_mesh.obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("rest_mesh.obj must load as a single Trimesh")
    rest_vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.long)

    gt_vertices = torch.tensor(np.load(base / "gt_anim_vertices.npy"), dtype=torch.float32)
    assert_shape(rest_vertices, (None, 3), "rest_vertices")
    assert_shape(gt_vertices, (None, rest_vertices.shape[0], 3), "gt_vertices")

    rig = load_rig_json(base / "wrong_init_rig.json")
    init_pose = _load_optional_array(base / "init_pose.npy")
    gt_masks = None
    for optional_name in ("gt_masks.npy", "gt_masks"):
        candidate = base / optional_name
        if candidate.exists():
            gt_masks = np.load(candidate, allow_pickle=True)
            break
    sample_meta = None
    if (base / "sample_meta.json").exists():
        with (base / "sample_meta.json").open("r", encoding="utf-8") as handle:
            sample_meta = json.load(handle)
    gt_rig = None
    if (base / "gt_rig.json").exists():
        gt_rig = load_rig_json(base / "gt_rig.json")

    sample: dict[str, Any] = {
        "rest_vertices": rest_vertices.to(resolved_device),
        "faces": faces.to(resolved_device),
        "gt_vertices": gt_vertices.to(resolved_device),
        "parent_idx": rig["parent_idx"].to(resolved_device),
        "rest_joints": rig["rest_joints"].to(resolved_device),
        "connected_to_parent": rig["connected_to_parent"].to(resolved_device),
        "bind_transforms": None if rig["bind_transforms"] is None else rig["bind_transforms"].to(resolved_device),
        "birth_steps": rig["birth_steps"],
        "inserted": rig["inserted"],
        "birth_modes": rig["birth_modes"],
        "init_pose": None if init_pose is None else torch.tensor(init_pose, dtype=torch.float32, device=resolved_device),
        "gt_masks": None if gt_masks is None else torch.tensor(gt_masks, dtype=torch.float32, device=resolved_device),
        "sample_meta": sample_meta,
        "mesh": mesh,
        "gt_parent_idx": None if gt_rig is None else gt_rig["parent_idx"].to(resolved_device),
        "gt_rest_joints": None if gt_rig is None else gt_rig["rest_joints"].to(resolved_device),
        "gt_connected_to_parent": None if gt_rig is None else gt_rig["connected_to_parent"].to(resolved_device),
    }
    return sample
