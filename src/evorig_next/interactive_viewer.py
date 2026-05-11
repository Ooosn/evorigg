from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh

from evorig_next.phase1_skeleton import Phase1Skeleton


def _require_plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("interactive viewer requires plotly; install it in the active environment") from exc
    return go, go.Figure


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_or_reconstruct_gt_joint_motion(data_dir: str | Path) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    raw = _load_json(data_dir / "gt_rig.json")
    joints = raw.get("joints", raw) if isinstance(raw, dict) else raw
    joints = sorted(joints, key=lambda item: int(item["id"]))
    parent_idx = np.asarray([int(joint["parent_id"]) for joint in joints], dtype=np.int32)
    rest_joints = np.asarray([joint["rest_position"] for joint in joints], dtype=np.float32)
    positions_path = data_dir / "gt_joint_positions.npy"
    rotations_path = data_dir / "gt_joint_rotations.npy"
    if positions_path.exists() and rotations_path.exists():
        joint_positions = np.load(positions_path).astype(np.float32)
        joint_rotations = np.load(rotations_path).astype(np.float32)
    else:
        frame_count = int(np.load(data_dir / "gt_anim_vertices.npy", mmap_mode="r").shape[0])
        joint_positions = np.repeat(rest_joints[None, ...], frame_count, axis=0).astype(np.float32)
        identity = np.eye(3, dtype=np.float32)
        joint_rotations = np.repeat(identity[None, None, ...], frame_count, axis=0)
        joint_rotations = np.repeat(joint_rotations, int(rest_joints.shape[0]), axis=1)
    return {
        "joint_positions": joint_positions,
        "joint_rotations": joint_rotations,
        "parent_idx": parent_idx,
        "rest_joints": rest_joints,
    }


def _resolve_run_dirs(run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    output_dir = run_dir / "output"
    if data_dir.exists() and output_dir.exists():
        return data_dir, output_dir
    summary_candidates = sorted(run_dir.glob("*summary.json"))
    for summary_path in summary_candidates:
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue
        data_value = summary.get("data_dir")
        output_value = summary.get("output_dir")
        if data_value and output_value:
            return Path(str(data_value)), Path(str(output_value))
    return run_dir, run_dir


def _load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing training trace: {path}")
    with np.load(path, allow_pickle=True) as data:
        snapshot_count = int(data["step"].shape[0])
        trace: list[dict[str, Any]] = []
        for index in range(snapshot_count):
            trace.append(
                {
                    "step": int(data["step"][index]),
                    "progress": float(data["progress"][index]) if "progress" in data else float("nan"),
                    "label": str(data["label"][index]),
                    "loss": float(data["loss"][index]),
                    "recon": float(data["recon"][index]),
                    "recon_raw": float(data["recon_raw"][index]) if "recon_raw" in data else float("nan"),
                    "joint_count": int(data["joint_count"][index]) if "joint_count" in data else int(data["rest_joints"][index].shape[0]),
                    "gaussian_count": int(data["gaussian_count"][index]) if "gaussian_count" in data else 0,
                    "active_gaussian_count": int(data["active_gaussian_count"][index]) if "active_gaussian_count" in data else 0,
                    "gaussian_strictness": float(data["gaussian_strictness"][index]) if "gaussian_strictness" in data else float("nan"),
                    "rest_joints": np.asarray(data["rest_joints"][index], dtype=np.float32),
                    "parent_idx": np.asarray(data["parent_idx"][index], dtype=np.int32),
                    "pred_vertices": np.asarray(data["pred_vertices"][index], dtype=np.float32) if "pred_vertices" in data else None,
                    "joint_positions": np.asarray(data["joint_positions"][index], dtype=np.float32) if "joint_positions" in data else None,
                    "gaussian_centers": np.asarray(data["gaussian_centers"][index], dtype=np.float32) if "gaussian_centers" in data else None,
                    "gaussian_lambda": np.asarray(data["gaussian_lambda"][index], dtype=np.float32) if "gaussian_lambda" in data else None,
                    "gaussian_log_opacity": np.asarray(data["gaussian_log_opacity"][index], dtype=np.float32) if "gaussian_log_opacity" in data else None,
                    "gaussian_log_value": np.asarray(data["gaussian_log_value"][index], dtype=np.float32) if "gaussian_log_value" in data else None,
                    "gaussian_log_alpha": np.asarray(data["gaussian_log_alpha"][index], dtype=np.float32) if "gaussian_log_alpha" in data else None,
                    "gaussian_generation": np.asarray(data["gaussian_generation"][index], dtype=np.int32) if "gaussian_generation" in data else None,
                    "gaussian_active_mask": np.asarray(data["gaussian_active_mask"][index], dtype=bool) if "gaussian_active_mask" in data else None,
                    "support_mass": np.asarray(data["support_mass"][index], dtype=np.float32) if "support_mass" in data else None,
                    "residual": np.asarray(data["residual"][index], dtype=np.float32) if "residual" in data else None,
                    "anchor_bone": np.asarray(data["anchor_bone"][index], dtype=np.int32) if "anchor_bone" in data else None,
                    "event_count": int(data["event_count"][index]),
                }
            )
    return trace


def _load_inserted_joint_meta(pred_rig_path: Path) -> dict[int, dict[str, Any]]:
    raw = _load_json(pred_rig_path)
    joints = raw.get("joints", raw)
    meta: dict[int, dict[str, Any]] = {}
    for joint in joints:
        meta[int(joint["id"])] = {
            "birth_step": int(joint.get("birth_step", 0)),
            "is_inserted": bool(joint.get("is_inserted", False)),
            "birth_mode": str(joint.get("birth_mode", "seed")),
            "connected_to_parent": bool(joint.get("connected_to_parent", int(joint.get("parent_id", -1)) >= 0))
            and int(joint.get("parent_id", -1)) >= 0,
        }
    return meta


def _load_connected_lookup(rig_path: Path) -> dict[int, bool]:
    raw = _load_json(rig_path)
    joints = raw.get("joints", raw)
    lookup: dict[int, bool] = {}
    for joint in joints:
        joint_id = int(joint["id"])
        parent_id = int(joint.get("parent_id", -1))
        lookup[joint_id] = bool(joint.get("connected_to_parent", joint.get("connected", parent_id >= 0))) and parent_id >= 0
    return lookup


def _load_rest_skeleton(
    pred_rig_path: Path,
    *,
    connected_fallback: dict[int, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _load_json(pred_rig_path)
    joints = raw.get("joints", raw)
    parent_idx = np.asarray([int(joint["parent_id"]) for joint in joints], dtype=np.int32)
    rest_joints = np.asarray([joint["rest_position"] for joint in joints], dtype=np.float32)
    connected_to_parent = np.asarray(
        [
            bool(
                joint.get(
                    "connected_to_parent",
                    joint.get(
                        "connected",
                        connected_fallback.get(int(joint["id"]), int(joint["parent_id"]) >= 0)
                        if connected_fallback is not None
                        else int(joint["parent_id"]) >= 0,
                    ),
                )
            )
            and int(joint["parent_id"]) >= 0
            for joint in joints
        ],
        dtype=bool,
    )
    return parent_idx, rest_joints, connected_to_parent


def _phase1_view_config_from_raw(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "ownership_mode": str(raw.get("ownership_mode", "endpoint_cut")),
        "ownership_midpoint": float(raw.get("ownership_midpoint", 0.8)),
        "ownership_slope": float(raw.get("ownership_slope", 0.08)),
        "child_support_gate_start": float(raw.get("child_support_gate_start", 0.75)),
        "child_support_gate_end": float(raw.get("child_support_gate_end", 0.95)),
        "gaussian_kernel_mahal_cutoff_sq": float(raw.get("gaussian_kernel_mahal_cutoff_sq", 0.0)),
        "phase1_scale_formula": str(raw.get("phase1_scale_formula", "cross_section_inner_ring")),
        "phase1_radial_sigma_divisor": float(raw.get("phase1_radial_sigma_divisor", 3.0)),
        "parent_child_mix_start_step": float(raw.get("parent_child_mix_start_step", -1)),
        "steps": float(raw.get("steps", 0)),
    }


def _load_phase1_view_config(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "phase1_config.json"
    if path.exists():
        try:
            return _phase1_view_config_from_raw(_load_json(path))
        except Exception as exc:
            raise ValueError(f"failed to read phase1_config.json from {output_dir}") from exc
    state_path = output_dir / "phase1_state.pt"
    if state_path.exists():
        try:
            payload = torch.load(state_path, map_location="cpu")
            raw = payload.get("phase1_config", {}) if isinstance(payload, dict) else {}
            if isinstance(raw, dict) and raw:
                return _phase1_view_config_from_raw(raw)
        except Exception as exc:
            raise ValueError(f"failed to read phase1_config from phase1_state.pt in {output_dir}") from exc
    raise FileNotFoundError(
        f"missing phase1_config for viewer in {output_dir}; regenerate this run or copy the inherited phase1_config.json"
    )


def _phase1_lambda_joint_mix(
    anchor_bone: np.ndarray,
    lambda_param: np.ndarray,
    lambda_min: np.ndarray,
    lambda_max: np.ndarray,
    parent_idx: np.ndarray,
    connected_to_parent: np.ndarray | None = None,
    *,
    midpoint: float,
    slope: float,
) -> np.ndarray:
    connected = (parent_idx >= 0) if connected_to_parent is None else (np.asarray(connected_to_parent, dtype=bool) & (parent_idx >= 0))
    bone_child_idx = np.nonzero(connected)[0].astype(np.int64, copy=False)
    joint_count = int(parent_idx.shape[0])
    mix = np.zeros((int(anchor_bone.shape[0]), joint_count), dtype=np.float32)
    if int(anchor_bone.shape[0]) == 0 or int(bone_child_idx.shape[0]) == 0:
        return mix
    valid_anchor = (anchor_bone >= 0) & (anchor_bone < bone_child_idx.shape[0])
    if not bool(valid_anchor.any()):
        return mix
    gaussian_ids = np.nonzero(valid_anchor)[0].astype(np.int64, copy=False)
    child_joints = bone_child_idx[anchor_bone[valid_anchor].astype(np.int64, copy=False)]
    parent_joints = parent_idx[child_joints].astype(np.int64, copy=False)
    lam = np.minimum(np.maximum(lambda_param[valid_anchor], lambda_min[valid_anchor]), lambda_max[valid_anchor])
    lam = np.clip(lam, 0.0, 1.0)
    safe_slope = max(float(slope), 1.0e-6)
    child_weight = 1.0 / (1.0 + np.exp(-((lam - float(midpoint)) / safe_slope)))
    parent_weight = 1.0 - child_weight
    mix[gaussian_ids, child_joints] = child_weight.astype(np.float32, copy=False)
    valid_parent = parent_joints >= 0
    if bool(valid_parent.any()):
        mix[gaussian_ids[valid_parent], parent_joints[valid_parent]] = parent_weight[valid_parent].astype(np.float32, copy=False)
    return mix


def _load_gaussian_rest_centers(
    output_dir: Path,
    rest_joints: np.ndarray,
    parent_idx: np.ndarray,
    connected_to_parent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = output_dir / "gaussians_final.npz"
    if not path.exists():
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    with np.load(path, allow_pickle=False) as data:
        active = np.asarray(data["active_mask"], dtype=bool)
        anchor_bone = np.asarray(data["anchor_bone"], dtype=np.int32)[active]
        lam = np.asarray(data["lambda"], dtype=np.float32)[active]
        lambda_min = np.asarray(data["lambda_min"], dtype=np.float32)[active] if "lambda_min" in data else np.full_like(lam, -np.inf)
        lambda_max = np.asarray(data["lambda_max"], dtype=np.float32)[active] if "lambda_max" in data else np.full_like(lam, np.inf)
        offset_local = (
            np.asarray(data["offset_local"], dtype=np.float32)[active]
            if "offset_local" in data
            else np.zeros((int(anchor_bone.shape[0]), 3), dtype=np.float32)
        )
        generation = np.asarray(data["generation"], dtype=np.int32)[active]
        log_alpha = np.asarray(data["log_alpha"], dtype=np.float32)[active]
        log_opacity = np.asarray(data["log_opacity"], dtype=np.float32)[active] if "log_opacity" in data else np.zeros_like(log_alpha)
        log_value = np.asarray(data["log_value"], dtype=np.float32)[active] if "log_value" in data else log_alpha.copy()
    connected = (parent_idx >= 0) if connected_to_parent is None else (np.asarray(connected_to_parent, dtype=bool) & (parent_idx >= 0))
    bone_child_idx = np.nonzero(connected)[0].astype(np.int32, copy=False)
    skeleton = Phase1Skeleton(
        parent_idx=torch.as_tensor(parent_idx, dtype=torch.long),
        rest_joints=torch.as_tensor(rest_joints, dtype=torch.float32),
        frame_count=1,
        connected_to_parent=torch.as_tensor(connected, dtype=torch.bool),
    )
    with torch.no_grad():
        _parent_pos, bone_frames_t, _bone_parent_idx, _bone_child_idx = skeleton.compute_bone_frames()
    bone_frames = bone_frames_t.cpu().numpy().astype(np.float32, copy=False)
    centers: list[np.ndarray] = []
    valid_anchor: list[int] = []
    valid_generation: list[int] = []
    valid_alpha: list[float] = []
    valid_opacity: list[float] = []
    valid_value: list[float] = []
    for bone_index, lambda_value, lambda_min_value, lambda_max_value, offset_value, generation_value, alpha_value, opacity_value, value_value in zip(
        anchor_bone.tolist(),
        lam.tolist(),
        lambda_min.tolist(),
        lambda_max.tolist(),
        offset_local.tolist(),
        generation.tolist(),
        log_alpha.tolist(),
        log_opacity.tolist(),
        log_value.tolist(),
    ):
        if bone_index < 0 or bone_index >= bone_child_idx.shape[0]:
            continue
        child_joint = int(bone_child_idx[bone_index])
        parent_joint = int(parent_idx[child_joint])
        if parent_joint < 0:
            continue
        start = rest_joints[parent_joint]
        end = rest_joints[child_joint]
        clamped_lambda = min(max(float(lambda_value), float(lambda_min_value)), float(lambda_max_value))
        center = start + clamped_lambda * (end - start)
        if 0 <= bone_index < bone_frames.shape[0]:
            center = center + bone_frames[bone_index] @ np.asarray(offset_value, dtype=np.float32)
        centers.append(center)
        valid_anchor.append(int(bone_index))
        valid_generation.append(int(generation_value))
        valid_alpha.append(float(alpha_value))
        valid_opacity.append(float(opacity_value))
        valid_value.append(float(value_value))
    if not centers:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(valid_alpha, dtype=np.float32),
        np.asarray(valid_opacity, dtype=np.float32),
        np.asarray(valid_value, dtype=np.float32),
        np.asarray(valid_generation, dtype=np.int32),
        np.asarray(valid_anchor, dtype=np.int32),
    )


def _quantize_unit_interval(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def _selected_joint_segments(
    points: np.ndarray,
    parent_idx: np.ndarray,
    joint_id: int,
    connected_to_parent: np.ndarray | None = None,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    points = np.asarray(points, dtype=np.float32)
    parent_idx = np.asarray(parent_idx, dtype=np.int32)
    connected = (
        np.asarray(parent_idx, dtype=np.int32) >= 0
        if connected_to_parent is None
        else (np.asarray(connected_to_parent, dtype=bool) & (np.asarray(parent_idx, dtype=np.int32) >= 0))
    )
    if joint_id < 0 or joint_id >= points.shape[0]:
        return xs, ys, zs
    center = points[int(joint_id)]
    neighbors: list[int] = []
    parent_joint = int(parent_idx[int(joint_id)])
    if parent_joint >= 0 and bool(connected[int(joint_id)]):
        neighbors.append(parent_joint)
    children = np.nonzero((parent_idx == int(joint_id)) & connected)[0].astype(np.int32, copy=False).tolist()
    neighbors.extend(int(child_joint) for child_joint in children)
    seen: set[int] = set()
    for neighbor in neighbors:
        if neighbor in seen or neighbor < 0 or neighbor >= points.shape[0]:
            continue
        seen.add(neighbor)
        other = points[int(neighbor)]
        xs.extend([float(center[0]), float(other[0]), None])
        ys.extend([float(center[1]), float(other[1]), None])
        zs.extend([float(center[2]), float(other[2]), None])
    return xs, ys, zs


def _build_gaussian_covariance_wireframe(
    centers: np.ndarray,
    covariances: np.ndarray,
    *,
    sigma_scale: float = 1.0,
    ring_samples: int = 18,
) -> tuple[list[float], list[float], list[float], np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    covariances = np.asarray(covariances, dtype=np.float32).reshape(-1, 3, 3)
    if centers.shape[0] == 0 or covariances.shape[0] == 0:
        return xs, ys, zs, np.zeros((0, 3), dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, int(max(ring_samples, 4)) + 1, dtype=np.float32)
    zeros = np.zeros_like(theta)
    circle_templates = (
        np.stack([np.cos(theta), np.sin(theta), zeros], axis=1),
        np.stack([np.cos(theta), zeros, np.sin(theta)], axis=1),
        np.stack([zeros, np.cos(theta), np.sin(theta)], axis=1),
    )
    sigma_axes = np.zeros((centers.shape[0], 3), dtype=np.float32)
    for gaussian_id, (center, covariance) in enumerate(zip(centers, covariances)):
        eigvals, eigvecs = np.linalg.eigh(covariance.astype(np.float64, copy=False))
        eigvals = np.clip(eigvals, 1.0e-12, None)
        order = np.argsort(eigvals)[::-1]
        basis = eigvecs[:, order].astype(np.float32, copy=False)
        sigma = (np.sqrt(eigvals[order]).astype(np.float32, copy=False) * float(sigma_scale)).reshape(1, 3)
        sigma_axes[gaussian_id] = sigma[0]
        for template in circle_templates:
            local = template * sigma
            world = local @ basis.T + center.reshape(1, 3)
            xs.extend(world[:, 0].astype(np.float32, copy=False).tolist())
            ys.extend(world[:, 1].astype(np.float32, copy=False).tolist())
            zs.extend(world[:, 2].astype(np.float32, copy=False).tolist())
            xs.append(None)
            ys.append(None)
            zs.append(None)
    return xs, ys, zs, sigma_axes


def _load_final_joint_viewer_payload(
    output_dir: Path,
    rest_vertices: np.ndarray,
    pred_parent_idx: np.ndarray,
    pred_rest_joints: np.ndarray,
    pred_connected_to_parent: np.ndarray,
    joint_meta: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    gauss_path = output_dir / "gaussians_final.npz"
    pred_weights_path = output_dir / "pred_weights.npy"
    pred_connected = np.asarray(pred_connected_to_parent, dtype=bool) & (pred_parent_idx >= 0)
    bone_child_idx = np.nonzero(pred_connected)[0].astype(np.int32, copy=False)
    bone_parent_idx = pred_parent_idx[bone_child_idx].astype(np.int32, copy=False) if bone_child_idx.size else np.zeros((0,), dtype=np.int32)
    empty = {
        "joint_id": -1,
        "label": "All joints",
        "parent_joint": -1,
        "gaussian_count": 0,
        "gaussian_x": [],
        "gaussian_y": [],
        "gaussian_z": [],
        "gaussian_text": [],
        "gaussian_color": [],
        "gaussian_cmax": 1.0,
        "cov_x": [],
        "cov_y": [],
        "cov_z": [],
        "line_x": [],
        "line_y": [],
        "line_z": [],
        "joint_text": [],
        "joint_x": [],
        "joint_y": [],
        "joint_z": [],
        "weight_u8": [],
        "weight_cmax": 1.0,
    }
    if not gauss_path.exists():
        return {"joints": [], "empty": empty}

    with np.load(gauss_path, allow_pickle=False) as data:
        anchor_bone = np.asarray(data["anchor_bone"], dtype=np.int64)
        lambda_param = np.asarray(data["lambda"], dtype=np.float32)
        lambda_min = np.asarray(data["lambda_min"], dtype=np.float32)
        lambda_max = np.asarray(data["lambda_max"], dtype=np.float32)
        offset_local = np.asarray(data["offset_local"], dtype=np.float32)
        if "rot_q" in data:
            rot_local = np.asarray(data["rot_q"], dtype=np.float32)
        else:
            rot_local = np.asarray(data["rot_local"], dtype=np.float32)
        log_scale = np.asarray(data["log_scale"], dtype=np.float32)
        log_opacity = np.asarray(data["log_opacity"], dtype=np.float32) if "log_opacity" in data else np.zeros_like(anchor_bone, dtype=np.float32)
        log_value = np.asarray(data["log_value"], dtype=np.float32) if "log_value" in data else np.asarray(data["log_alpha"], dtype=np.float32)
        log_alpha = np.asarray(data["log_alpha"], dtype=np.float32)
        q_logits = np.asarray(data["q_logits"], dtype=np.float32)
        endpoint_logits = np.asarray(data["endpoint_logits"], dtype=np.float32) if "endpoint_logits" in data else None
        generation = np.asarray(data["generation"], dtype=np.int64)
        active_mask = np.asarray(data["active_mask"], dtype=bool)

    if pred_weights_path.exists():
        pred_weights = np.asarray(np.load(pred_weights_path), dtype=np.float32)
    else:
        pred_weights = np.zeros((rest_vertices.shape[0], pred_rest_joints.shape[0]), dtype=np.float32)

    if anchor_bone.size == 0 or bone_child_idx.size == 0:
        return {"joints": [], "empty": empty}
    phase1_view_config = _load_phase1_view_config(output_dir)
    range_sigma_scale = 1.0
    if phase1_view_config is not None:
        cutoff_sq = float(phase1_view_config["gaussian_kernel_mahal_cutoff_sq"])
        if cutoff_sq > 0.0:
            range_sigma_scale = math.sqrt(cutoff_sq)
        elif str(phase1_view_config.get("phase1_scale_formula", "")).lower() == "cross_section_inner_ring":
            range_sigma_scale = max(float(phase1_view_config.get("phase1_radial_sigma_divisor", 3.0)), 1.0)
        else:
            range_sigma_scale = 1.0

    skeleton = Phase1Skeleton(
        parent_idx=torch.as_tensor(pred_parent_idx, dtype=torch.long),
        rest_joints=torch.as_tensor(pred_rest_joints, dtype=torch.float32),
        frame_count=1,
        connected_to_parent=torch.as_tensor(pred_connected, dtype=torch.bool),
    )
    from evorig_next.phase1_field import Phase1FieldState, Phase1GaussianField

    field = Phase1GaussianField(
        Phase1FieldState(
            anchor_bone=torch.as_tensor(anchor_bone, dtype=torch.long),
            lambda_param=torch.as_tensor(lambda_param, dtype=torch.float32),
            lambda_min=torch.as_tensor(lambda_min, dtype=torch.float32),
            lambda_max=torch.as_tensor(lambda_max, dtype=torch.float32),
            offset_local=torch.as_tensor(offset_local, dtype=torch.float32),
            rot_local=torch.as_tensor(rot_local, dtype=torch.float32),
            log_scale=torch.as_tensor(log_scale, dtype=torch.float32),
            init_log_scale=torch.as_tensor(log_scale, dtype=torch.float32),
            log_opacity=torch.as_tensor(log_opacity, dtype=torch.float32),
            log_value=torch.as_tensor(log_value, dtype=torch.float32),
            kernel_mahal_cutoff_sq=float(phase1_view_config["gaussian_kernel_mahal_cutoff_sq"]),
        )
    )
    field.generation = torch.as_tensor(generation, dtype=torch.long)
    field.active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
    if endpoint_logits is not None:
        field.endpoint_logits = torch.nn.Parameter(torch.as_tensor(endpoint_logits, dtype=torch.float32))
    with torch.no_grad():
        centers_t = field.compute_rest_centers(skeleton)
        covariances_t = field.compute_covariance(skeleton)
    centers_all = centers_t.cpu().numpy().astype(np.float32)
    covariances_all = covariances_t.cpu().numpy().astype(np.float32)
    use_endpoint_mix = (
        float(phase1_view_config.get("parent_child_mix_start_step", -1)) >= 0
        and float(phase1_view_config.get("steps", 0)) >= float(phase1_view_config.get("parent_child_mix_start_step", -1))
        and endpoint_logits is not None
    )
    with torch.no_grad():
        assignment_t = field.compute_joint_mix(
            skeleton,
            mode=str(phase1_view_config.get("ownership_mode", "endpoint_cut")),
            midpoint=float(phase1_view_config["ownership_midpoint"]),
            slope=float(phase1_view_config["ownership_slope"]),
            use_endpoint_logits=bool(use_endpoint_mix),
        )
    assignment = assignment_t.cpu().numpy().astype(np.float32)
    active_indices = np.nonzero(active_mask)[0].astype(np.int32, copy=False)
    centers = centers_all[active_indices] if active_indices.size else np.zeros((0, 3), dtype=np.float32)
    covariances = covariances_all[active_indices] if active_indices.size else np.zeros((0, 3, 3), dtype=np.float32)
    active_log_alpha = log_alpha[active_indices] if active_indices.size else np.zeros((0,), dtype=np.float32)
    active_log_opacity = log_opacity[active_indices] if active_indices.size else np.zeros((0,), dtype=np.float32)
    active_log_value = log_value[active_indices] if active_indices.size else np.zeros((0,), dtype=np.float32)
    active_generation = generation[active_indices] if active_indices.size else np.zeros((0,), dtype=np.int32)
    active_anchor_bone = anchor_bone[active_indices] if active_indices.size else np.zeros((0,), dtype=np.int32)
    active_assignment = assignment[active_indices] if active_indices.size else np.zeros((0, pred_rest_joints.shape[0]), dtype=np.float32)
    dominant_joint = active_assignment.argmax(axis=1) if active_assignment.size else np.zeros((0,), dtype=np.int32)
    assignment_threshold = 0.10
    active_text_template = [
        (
            int(gaussian_id),
            int(bone_id),
            int(bone_parent_idx[bone_id]) if 0 <= int(bone_id) < bone_parent_idx.shape[0] else -1,
            int(bone_child_idx[bone_id]) if 0 <= int(bone_id) < bone_child_idx.shape[0] else -1,
            float(alpha_value),
            float(active_log_opacity[idx]),
            float(active_log_value[idx]),
            int(generation_value),
        )
        for idx, (gaussian_id, bone_id, alpha_value, generation_value) in enumerate(zip(
            active_indices.tolist(),
            active_anchor_bone.tolist(),
            active_log_alpha.tolist(),
            active_generation.tolist(),
        ))
    ]

    joints: list[dict[str, Any]] = []
    for joint_id in range(int(pred_rest_joints.shape[0])):
        parent_joint = int(pred_parent_idx[joint_id])
        gaussian_mask = active_assignment[:, joint_id] >= assignment_threshold if active_assignment.size else np.zeros((0,), dtype=bool)
        dominant_gaussian_mask = (dominant_joint == joint_id) & gaussian_mask if active_assignment.size else np.zeros((0,), dtype=bool)
        if gaussian_mask.any():
            gaussian_indices = np.nonzero(gaussian_mask)[0].astype(np.int32, copy=False)
            cov_x, cov_y, cov_z, sigma_axes = _build_gaussian_covariance_wireframe(
                centers[gaussian_indices],
                covariances[gaussian_indices],
                sigma_scale=float(range_sigma_scale),
            )
            sigma_max = sigma_axes.max(axis=1)
            gaussian_text = []
            for local_idx, active_idx in enumerate(gaussian_indices.tolist()):
                sx, sy, sz = sigma_axes[local_idx].tolist()
                gaussian_text.append(
                    f"g{active_text_template[active_idx][0]}"
                    f"<br>joint={joint_id}"
                    f"<br>joint_assign={float(active_assignment[active_idx, joint_id]):.3f}"
                    f"<br>dominant_joint={int(dominant_joint[active_idx])}"
                    f"<br>anchor_bone={active_text_template[active_idx][1]} "
                    f"({active_text_template[active_idx][2]}->{active_text_template[active_idx][3]})"
                    f"<br>range_axes=({sx:.3f}, {sy:.3f}, {sz:.3f})"
                    f"<br>range_sigma_scale={float(range_sigma_scale):.3f}"
                    f"<br>log_alpha={active_text_template[active_idx][4]:.3f}"
                    f"<br>log_opacity={active_text_template[active_idx][5]:.3f}"
                    f"<br>log_value={active_text_template[active_idx][6]:.3f}"
                    f"<br>generation={active_text_template[active_idx][7]}"
                )
            gaussian_color = sigma_max.astype(np.float32).tolist()
            gaussian_cmax = float(max(float(sigma_max.max()), 1.0e-3))
        else:
            cov_x, cov_y, cov_z = [], [], []
            gaussian_text = []
            gaussian_color = []
            gaussian_cmax = 1.0e-3
        weight_values = (
            np.clip(pred_weights[:, joint_id], 0.0, 1.0)
            if pred_weights.ndim == 2 and joint_id < pred_weights.shape[1]
            else np.zeros((rest_vertices.shape[0],), dtype=np.float32)
        )
        line_x, line_y, line_z = _selected_joint_segments(
            pred_rest_joints,
            pred_parent_idx,
            joint_id,
            pred_connected_to_parent,
        )
        meta = joint_meta.get(int(joint_id), {})
        joint_text = [
            f"selected joint {joint_id}"
            f"<br>parent={parent_joint}"
            f"<br>birth_step={int(meta.get('birth_step', 0))}"
            f"<br>inserted={bool(meta.get('is_inserted', False))}"
            f"<br>mode={meta.get('birth_mode', 'seed')}"
            f"<br>pos={_format_xyz(pred_rest_joints[joint_id])}"
        ]
        joints.append(
            {
                "joint_id": joint_id,
                "label": f"j{joint_id} | parent={parent_joint} | g={int(gaussian_mask.sum())} | dominant_g={int(dominant_gaussian_mask.sum())}",
                "parent_joint": parent_joint,
                "gaussian_count": int(gaussian_mask.sum()),
                "dominant_gaussian_count": int(dominant_gaussian_mask.sum()),
                "gaussian_x": centers[gaussian_mask, 0].tolist() if gaussian_mask.any() else [],
                "gaussian_y": centers[gaussian_mask, 1].tolist() if gaussian_mask.any() else [],
                "gaussian_z": centers[gaussian_mask, 2].tolist() if gaussian_mask.any() else [],
                "gaussian_text": gaussian_text,
                "gaussian_color": gaussian_color,
                "gaussian_cmax": gaussian_cmax,
                "cov_x": cov_x,
                "cov_y": cov_y,
                "cov_z": cov_z,
                "line_x": line_x,
                "line_y": line_y,
                "line_z": line_z,
                "joint_text": joint_text,
                "joint_x": [float(pred_rest_joints[joint_id, 0])],
                "joint_y": [float(pred_rest_joints[joint_id, 1])],
                "joint_z": [float(pred_rest_joints[joint_id, 2])],
                "weight_u8": _quantize_unit_interval(weight_values).astype(np.int32).tolist(),
                "weight_cmax": float(max(float(weight_values.max()), 1.0e-3)),
            }
        )
    empty["weight_u8"] = [0] * int(rest_vertices.shape[0])
    return {"joints": joints, "empty": empty}


def _make_limits(arrays: Iterable[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    valid = [np.asarray(arr, dtype=np.float32).reshape(-1, 3) for arr in arrays if arr is not None and np.asarray(arr).size > 0]
    stacked = np.concatenate(valid, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.58 * float((maxs - mins).max() + 1.0e-6)
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(center[1] - radius), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def _scene_layout(limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> dict[str, Any]:
    return {
        "xaxis": {"range": list(limits[0]), "visible": False},
        "yaxis": {"range": list(limits[1]), "visible": False},
        "zaxis": {"range": list(limits[2]), "visible": False},
        "aspectmode": "cube",
        "uirevision": "keep_camera",
        "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 0.95}},
    }


def _line_segments(
    points: np.ndarray,
    parent_idx: np.ndarray,
    connected_to_parent: np.ndarray | None = None,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    connected = (
        np.asarray(parent_idx, dtype=np.int32) >= 0
        if connected_to_parent is None
        else (np.asarray(connected_to_parent, dtype=bool) & (np.asarray(parent_idx, dtype=np.int32) >= 0))
    )
    for joint_id, parent_id in enumerate(parent_idx.tolist()):
        if parent_id < 0 or not bool(connected[int(joint_id)]):
            continue
        start = points[parent_id]
        end = points[joint_id]
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return xs, ys, zs


def _dashed_line_segments(
    points: np.ndarray,
    parent_idx: np.ndarray,
    connected_to_parent: np.ndarray,
    *,
    dash_count: int = 8,
    visible_fraction: float = 0.55,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    connected = np.asarray(connected_to_parent, dtype=bool) & (np.asarray(parent_idx, dtype=np.int32) >= 0)
    dash_count = max(int(dash_count), 1)
    visible_fraction = min(max(float(visible_fraction), 0.05), 0.95)
    for joint_id, parent_id in enumerate(parent_idx.tolist()):
        if parent_id < 0 or bool(connected[int(joint_id)]):
            continue
        start = points[parent_id].astype(np.float32, copy=False)
        end = points[joint_id].astype(np.float32, copy=False)
        delta = end - start
        for dash_index in range(dash_count):
            t0 = float(dash_index) / float(dash_count)
            t1 = min(t0 + visible_fraction / float(dash_count), 1.0)
            seg_start = start + t0 * delta
            seg_end = start + t1 * delta
            xs.extend([float(seg_start[0]), float(seg_end[0]), None])
            ys.extend([float(seg_start[1]), float(seg_end[1]), None])
            zs.extend([float(seg_start[2]), float(seg_end[2]), None])
    return xs, ys, zs


def _joint_partition_ids(points: np.ndarray, step: int, joint_meta: dict[int, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    old_ids: list[int] = []
    inserted_ids: list[int] = []
    for joint_id in range(points.shape[0]):
        meta = joint_meta.get(joint_id, {})
        is_inserted = bool(meta.get("is_inserted", False))
        birth_step = int(meta.get("birth_step", 0))
        if is_inserted and birth_step <= int(step):
            inserted_ids.append(joint_id)
        else:
            old_ids.append(joint_id)
    return np.asarray(old_ids, dtype=np.int32), np.asarray(inserted_ids, dtype=np.int32)


def _format_xyz(point: np.ndarray) -> str:
    return f"({float(point[0]):.3f}, {float(point[1]):.3f}, {float(point[2]):.3f})"


def _joint_hover_strings(
    joint_ids: np.ndarray,
    points: np.ndarray,
    parent_idx: np.ndarray,
    joint_meta: dict[int, dict[str, Any]],
    *,
    fallback_mode: str = "seed",
    prefix: str = "joint",
) -> list[str]:
    hover: list[str] = []
    for joint_id in joint_ids.tolist():
        meta = joint_meta.get(int(joint_id), {})
        parent_id = int(parent_idx[int(joint_id)]) if int(joint_id) < int(parent_idx.shape[0]) else -1
        point = points[int(joint_id)]
        hover.append(
            f"{prefix} {int(joint_id)}"
            f"<br>parent={parent_id}"
            f"<br>connected={bool(meta.get('connected_to_parent', parent_id >= 0)) and parent_id >= 0}"
            f"<br>birth_step={int(meta.get('birth_step', 0))}"
            f"<br>inserted={bool(meta.get('is_inserted', False))}"
            f"<br>mode={meta.get('birth_mode', fallback_mode)}"
            f"<br>pos={_format_xyz(point)}"
        )
    return hover


def _mesh_toggle_menu(mesh_trace_indices: list[int], *, x: float = 0.56, y: float = 1.12) -> dict[str, Any]:
    return {
        "type": "buttons",
        "direction": "left",
        "x": x,
        "y": y,
        "showactive": False,
        "buttons": [
            {
                "label": "Hide Mesh",
                "method": "restyle",
                "args": [{"visible": "legendonly"}, mesh_trace_indices],
            },
            {
                "label": "Show Mesh",
                "method": "restyle",
                "args": [{"visible": True}, mesh_trace_indices],
            },
        ],
    }


def _training_status_annotation(snapshot: dict[str, Any]) -> dict[str, Any]:
    loss = float(snapshot.get("loss", float("nan")))
    recon = float(snapshot.get("recon", float("nan")))
    recon_raw = float(snapshot.get("recon_raw", float("nan")))
    return {
        "xref": "paper",
        "yref": "paper",
        "x": 0.01,
        "y": 0.99,
        "xanchor": "left",
        "yanchor": "top",
        "align": "left",
        "showarrow": False,
        "bgcolor": "rgba(255,255,255,0.82)",
        "bordercolor": "rgba(148,163,184,0.9)",
        "borderwidth": 1,
        "font": {"size": 12, "color": "#0f172a"},
        "text": (
            f"step={int(snapshot['step'])}  label={snapshot['label']}"
            f"<br>loss={loss:.6f}  recon={recon:.6f}  recon_raw={recon_raw:.6f}"
            f"<br>joints={int(snapshot.get('joint_count', 0))}  gaussians={int(snapshot.get('gaussian_count', 0))}"
            f"  active_gaussians={int(snapshot.get('active_gaussian_count', 0))}"
            f"<br>events={int(snapshot.get('event_count', 0))}  strictness={float(snapshot.get('gaussian_strictness', float('nan'))):.3f}"
        ),
    }


def _masked_trace_values(values: Any, active: np.ndarray, *, dtype: Any, width: int | None = None) -> np.ndarray:
    active_count = int(active.sum())
    if values is None:
        if width is None:
            return np.zeros((active_count,), dtype=dtype)
        return np.zeros((active_count, width), dtype=dtype)
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        if width is None:
            return np.zeros((active_count,), dtype=dtype)
        return np.zeros((active_count, width), dtype=dtype)
    if width is None:
        if arr.ndim != 1 or arr.shape[0] != active.shape[0]:
            return np.zeros((active_count,), dtype=dtype)
        return np.asarray(arr[active], dtype=dtype)
    if arr.ndim != 2 or arr.shape[0] != active.shape[0] or arr.shape[1] != width:
        return np.zeros((active_count, width), dtype=dtype)
    return np.asarray(arr[active], dtype=dtype)


def _metric_values(snapshot: dict[str, Any], color_by: str) -> tuple[np.ndarray, str]:
    active_mask = snapshot.get("gaussian_active_mask")
    if active_mask is None:
        return np.zeros((0,), dtype=np.float32), "none"
    active = active_mask.astype(bool)
    if color_by == "support":
        return _masked_trace_values(snapshot.get("support_mass"), active, dtype=np.float32), "support_mass"
    if color_by == "generation":
        return _masked_trace_values(snapshot.get("gaussian_generation"), active, dtype=np.float32), "generation"
    if color_by == "alpha":
        return _masked_trace_values(snapshot.get("gaussian_log_alpha"), active, dtype=np.float32), "log_alpha"
    return _masked_trace_values(snapshot.get("residual"), active, dtype=np.float32), "residual"


def build_training_figure(
    run_dir: str | Path,
    *,
    trace_stride: int = 1,
    color_by: str = "residual",
) -> Any:
    go, Figure = _require_plotly()
    run_dir = Path(run_dir)
    data_dir, output_dir = _resolve_run_dirs(run_dir)
    trace = _load_trace(output_dir / "training_trace.npz")
    mesh = trimesh.load_mesh(data_dir / "rest_mesh.obj", process=False)
    rest_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    joint_meta = _load_inserted_joint_meta(output_dir / "pred_rig_final.json")

    trace_stride = max(int(trace_stride), 1)
    selected = trace[::trace_stride]
    if selected[-1]["step"] != trace[-1]["step"]:
        selected.append(trace[-1])

    limits = _make_limits(
        [
            rest_vertices,
            *(snapshot["rest_joints"] for snapshot in selected),
            *(snapshot["gaussian_centers"] for snapshot in selected if snapshot.get("gaussian_centers") is not None),
        ]
    )

    def build_snapshot_traces(snapshot: dict[str, Any]) -> list[Any]:
        step = int(snapshot["step"])
        parent_idx = snapshot["parent_idx"]
        rest_joints = snapshot["rest_joints"]
        line_x, line_y, line_z = _line_segments(rest_joints, parent_idx)
        old_ids, inserted_ids = _joint_partition_ids(rest_joints, step, joint_meta)
        old_points = rest_joints[old_ids] if old_ids.size else np.zeros((0, 3), dtype=np.float32)
        inserted_points = rest_joints[inserted_ids] if inserted_ids.size else np.zeros((0, 3), dtype=np.float32)
        old_text = _joint_hover_strings(old_ids, rest_joints, parent_idx, joint_meta)
        inserted_text = _joint_hover_strings(inserted_ids, rest_joints, parent_idx, joint_meta)

        active_mask = snapshot.get("gaussian_active_mask")
        active = active_mask.astype(bool) if active_mask is not None else np.zeros((0,), dtype=bool)
        centers = _masked_trace_values(snapshot.get("gaussian_centers"), active, dtype=np.float32, width=3)
        metric, metric_label = _metric_values(snapshot, color_by)
        anchor_bone = _masked_trace_values(snapshot.get("anchor_bone"), active, dtype=np.int32)
        support_mass = _masked_trace_values(snapshot.get("support_mass"), active, dtype=np.float32)
        generation = _masked_trace_values(snapshot.get("gaussian_generation"), active, dtype=np.int32)
        lambda_values = _masked_trace_values(snapshot.get("gaussian_lambda"), active, dtype=np.float32)
        if lambda_values.shape[0] != metric.shape[0]:
            lambda_values = np.full(metric.shape, np.nan, dtype=np.float32)
        hover = [
            f"g{idx}<br>{metric_label}={float(metric[idx]):.5f}<br>support={float(support_mass[idx]):.5f}<br>bone={int(anchor_bone[idx])}<br>generation={int(generation[idx])}<br>lambda={float(lambda_values[idx]):.3f}"
            for idx in range(centers.shape[0])
        ]

        return [
            go.Mesh3d(
                x=rest_vertices[:, 0],
                y=rest_vertices[:, 1],
                z=rest_vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#cbd5e1",
                opacity=0.18,
                name="rest mesh",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": "#2563eb", "width": 6},
                name="skeleton",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=old_points[:, 0] if old_points.size else [],
                y=old_points[:, 1] if old_points.size else [],
                z=old_points[:, 2] if old_points.size else [],
                mode="markers",
                marker={"size": 4, "color": "#1d4ed8"},
                text=old_text,
                hovertemplate="%{text}<extra></extra>",
                name="old joints",
            ),
            go.Scatter3d(
                x=inserted_points[:, 0] if inserted_points.size else [],
                y=inserted_points[:, 1] if inserted_points.size else [],
                z=inserted_points[:, 2] if inserted_points.size else [],
                mode="markers",
                marker={"size": 6, "color": "#f97316", "symbol": "diamond"},
                text=inserted_text,
                hovertemplate="%{text}<extra></extra>",
                name="inserted joints",
            ),
            go.Scatter3d(
                x=centers[:, 0] if centers.size else [],
                y=centers[:, 1] if centers.size else [],
                z=centers[:, 2] if centers.size else [],
                mode="markers",
                marker={
                    "size": 4,
                    "color": metric.tolist() if metric.size else [],
                    "colorscale": "Inferno",
                    "opacity": 0.95,
                    "colorbar": {"title": metric_label},
                },
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name="gaussians",
            ),
        ]

    initial_traces = build_snapshot_traces(selected[0])
    frames = [
        go.Frame(
            name=str(snapshot["step"]),
            data=build_snapshot_traces(snapshot),
            traces=list(range(len(initial_traces))),
            layout={"annotations": [_training_status_annotation(snapshot)]},
        )
        for snapshot in selected
    ]
    slider_steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f"s{snapshot['step']}:{snapshot['label']}",
            "method": "animate",
        }
        for frame, snapshot in zip(frames, selected)
    ]

    fig = Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title=f"Rest-Space Training Viewer | color_by={color_by}",
        scene=_scene_layout(limits),
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        annotations=[_training_status_annotation(selected[0])],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 36},
                "currentvalue": {"prefix": "snapshot: "},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 140, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            },
            _mesh_toggle_menu([0], x=0.56, y=1.12),
        ],
    )
    return fig


def build_motion_figure(run_dir: str | Path) -> Any:
    go, Figure = _require_plotly()
    run_dir = Path(run_dir)
    data_dir, output_dir = _resolve_run_dirs(run_dir)
    pred_vertices = np.load(output_dir / "pred_anim_vertices.npy")
    gt_vertices = np.load(data_dir / "gt_anim_vertices.npy")
    mesh = trimesh.load_mesh(data_dir / "rest_mesh.obj", process=False)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    joint_meta = _load_inserted_joint_meta(output_dir / "pred_rig_final.json")
    pred_rig_path = output_dir / "pred_rig_final.json"
    pred_parent_idx, pred_rest_joints, pred_connected_to_parent = _load_rest_skeleton(pred_rig_path)

    trace_path = output_dir / "training_trace.npz"
    if trace_path.exists():
        trace = _load_trace(trace_path)
        final_snapshot = trace[-1]
        pred_joint_positions = np.asarray(final_snapshot["joint_positions"], dtype=np.float32)
        final_step = int(trace[-1]["step"])
    else:
        pred_joint_positions_path = output_dir / "pred_joint_positions.npy"
        if pred_joint_positions_path.exists():
            pred_joint_positions = np.load(pred_joint_positions_path).astype(np.float32)
        else:
            frame_count = int(pred_vertices.shape[0])
            pred_joint_positions = np.repeat(pred_rest_joints[None, ...], frame_count, axis=0)
        final_step = int(max((meta.get("birth_step", 0) for meta in joint_meta.values()), default=0))

    gt_joint_positions = None
    gt_parent_idx = None
    gt_rig_path = data_dir / "gt_rig.json"
    if gt_rig_path.exists():
        gt_motion = load_or_reconstruct_gt_joint_motion(data_dir)
        gt_joint_positions = np.asarray(gt_motion["joint_positions"], dtype=np.float32)
        gt_parent_idx = np.asarray(gt_motion["parent_idx"], dtype=np.int32)

    limit_arrays: list[np.ndarray] = [pred_vertices, gt_vertices, pred_joint_positions]
    if gt_joint_positions is not None:
        limit_arrays.append(gt_joint_positions)
    limits = _make_limits(limit_arrays)

    error_max = max(float(np.linalg.norm(pred_vertices - gt_vertices, axis=-1).max()), 1.0e-5)

    def build_frame(frame_index: int) -> list[Any]:
        pred_frame = np.asarray(pred_vertices[frame_index], dtype=np.float32)
        gt_frame = np.asarray(gt_vertices[frame_index], dtype=np.float32)
        pred_joints = np.asarray(pred_joint_positions[frame_index], dtype=np.float32)
        errors = np.linalg.norm(pred_frame - gt_frame, axis=-1)
        pred_line_x, pred_line_y, pred_line_z = _line_segments(pred_joints, pred_parent_idx, pred_connected_to_parent)
        pred_dash_x, pred_dash_y, pred_dash_z = _dashed_line_segments(pred_joints, pred_parent_idx, pred_connected_to_parent)
        old_ids, inserted_ids = _joint_partition_ids(pred_joints, final_step, joint_meta)
        old_points = pred_joints[old_ids] if old_ids.size else np.zeros((0, 3), dtype=np.float32)
        inserted_points = pred_joints[inserted_ids] if inserted_ids.size else np.zeros((0, 3), dtype=np.float32)
        old_text = _joint_hover_strings(old_ids, pred_joints, pred_parent_idx, joint_meta)
        inserted_text = _joint_hover_strings(inserted_ids, pred_joints, pred_parent_idx, joint_meta)

        traces = [
            go.Mesh3d(
                x=gt_frame[:, 0],
                y=gt_frame[:, 1],
                z=gt_frame[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#22c55e",
                opacity=0.16,
                name="gt mesh",
                hoverinfo="skip",
            ),
            go.Mesh3d(
                x=pred_frame[:, 0],
                y=pred_frame[:, 1],
                z=pred_frame[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=errors,
                colorscale="Magma",
                cmin=0.0,
                cmax=error_max,
                opacity=0.60,
                colorbar={"title": "vertex error"},
                name="pred mesh",
                hovertemplate="err=%{intensity:.5f}<extra></extra>",
            ),
            go.Scatter3d(
                x=pred_line_x,
                y=pred_line_y,
                z=pred_line_z,
                mode="lines",
                line={"color": "#2563eb", "width": 6},
                name="pred skeleton",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=pred_dash_x,
                y=pred_dash_y,
                z=pred_dash_z,
                mode="lines",
                line={"color": "rgba(37,99,235,0.55)", "width": 4},
                name="pred hierarchy link",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=old_points[:, 0] if old_points.size else [],
                y=old_points[:, 1] if old_points.size else [],
                z=old_points[:, 2] if old_points.size else [],
                mode="markers",
                marker={"size": 4, "color": "#1d4ed8"},
                text=old_text,
                hovertemplate="%{text}<extra></extra>",
                name="pred old joints",
            ),
            go.Scatter3d(
                x=inserted_points[:, 0] if inserted_points.size else [],
                y=inserted_points[:, 1] if inserted_points.size else [],
                z=inserted_points[:, 2] if inserted_points.size else [],
                mode="markers",
                marker={"size": 6, "color": "#f97316", "symbol": "diamond"},
                text=inserted_text,
                hovertemplate="%{text}<extra></extra>",
                name="pred inserted joints",
            ),
        ]
        if gt_joint_positions is not None and gt_parent_idx is not None:
            gt_joints = np.asarray(gt_joint_positions[frame_index], dtype=np.float32)
            gt_line_x, gt_line_y, gt_line_z = _line_segments(gt_joints, gt_parent_idx)
            traces.insert(
                2,
                go.Scatter3d(
                    x=gt_line_x,
                    y=gt_line_y,
                    z=gt_line_z,
                    mode="lines",
                    line={"color": "#16a34a", "width": 5},
                    name="gt skeleton",
                    hoverinfo="skip",
                ),
            )
            traces.insert(
                3,
                go.Scatter3d(
                    x=gt_joints[:, 0],
                    y=gt_joints[:, 1],
                    z=gt_joints[:, 2],
                    mode="markers",
                    marker={"size": 4, "color": "#16a34a"},
                    name="gt joints",
                    text=[
                        f"gt joint {joint_id}<br>parent={int(gt_parent_idx[joint_id])}<br>pos={_format_xyz(gt_joints[joint_id])}"
                        for joint_id in range(gt_joints.shape[0])
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ),
            )
        return traces

    frame_count = int(pred_vertices.shape[0])
    initial_traces = build_frame(0)
    frames = [
        go.Frame(name=str(frame_index), data=build_frame(frame_index), traces=list(range(len(initial_traces))))
        for frame_index in range(frame_count)
    ]
    slider_steps = [
        {
            "args": [[str(frame_index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(frame_index),
            "method": "animate",
        }
        for frame_index in range(frame_count)
    ]

    fig = Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title="Motion Viewer | GT vs Pred",
        scene=_scene_layout(limits),
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        sliders=[
            {
                "active": 0,
                "pad": {"t": 36},
                "currentvalue": {"prefix": "frame: "},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 160, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            },
            _mesh_toggle_menu([0, 1], x=0.56, y=1.12),
        ],
    )
    return fig


def build_final_topology_figure(run_dir: str | Path) -> Any:
    go, Figure = _require_plotly()
    run_dir = Path(run_dir)
    data_dir, output_dir = _resolve_run_dirs(run_dir)

    mesh = trimesh.load_mesh(data_dir / "rest_mesh.obj", process=False)
    rest_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pred_rig_path = output_dir / "pred_rig_final.json"
    pred_connected_lookup = _load_connected_lookup(pred_rig_path)
    pred_parent_idx, pred_rest_joints, pred_connected_to_parent = _load_rest_skeleton(pred_rig_path)
    init_rig_path = data_dir / "wrong_init_rig.json"
    init_parent_idx = None
    init_rest_joints = None
    init_connected_to_parent = None
    if init_rig_path.exists():
        init_parent_idx, init_rest_joints, init_connected_to_parent = _load_rest_skeleton(
            init_rig_path,
            connected_fallback=pred_connected_lookup,
        )
    joint_meta = _load_inserted_joint_meta(output_dir / "pred_rig_final.json")
    gaussian_centers, gaussian_log_alpha, gaussian_log_opacity, gaussian_log_value, gaussian_generation, gaussian_anchor_bone = _load_gaussian_rest_centers(
        output_dir,
        pred_rest_joints,
        pred_parent_idx,
        pred_connected_to_parent,
    )
    joint_view_payload = _load_final_joint_viewer_payload(
        output_dir,
        rest_vertices,
        pred_parent_idx,
        pred_rest_joints,
        pred_connected_to_parent,
        joint_meta,
    )
    limit_arrays: list[np.ndarray] = [rest_vertices, pred_rest_joints, gaussian_centers]
    if init_rest_joints is not None:
        limit_arrays.append(init_rest_joints)
    limits = _make_limits(limit_arrays)
    old_ids, inserted_ids = _joint_partition_ids(pred_rest_joints, 10**9, joint_meta)
    old_points = pred_rest_joints[old_ids] if old_ids.size else np.zeros((0, 3), dtype=np.float32)
    inserted_points = pred_rest_joints[inserted_ids] if inserted_ids.size else np.zeros((0, 3), dtype=np.float32)
    old_text = _joint_hover_strings(old_ids, pred_rest_joints, pred_parent_idx, joint_meta)
    inserted_text = _joint_hover_strings(inserted_ids, pred_rest_joints, pred_parent_idx, joint_meta)
    line_x, line_y, line_z = _line_segments(pred_rest_joints, pred_parent_idx, pred_connected_to_parent)
    dash_x, dash_y, dash_z = _dashed_line_segments(pred_rest_joints, pred_parent_idx, pred_connected_to_parent)
    init_line_x: list[float] = []
    init_line_y: list[float] = []
    init_line_z: list[float] = []
    init_dash_x: list[float] = []
    init_dash_y: list[float] = []
    init_dash_z: list[float] = []
    init_joint_hover: list[str] = []
    if init_rest_joints is not None and init_parent_idx is not None:
        init_line_x, init_line_y, init_line_z = _line_segments(init_rest_joints, init_parent_idx, init_connected_to_parent)
        init_dash_x, init_dash_y, init_dash_z = _dashed_line_segments(init_rest_joints, init_parent_idx, init_connected_to_parent)
        init_joint_hover = [
            f"init joint {joint_id}"
            f"<br>parent={int(init_parent_idx[joint_id])}"
            f"<br>connected={bool(init_connected_to_parent[joint_id]) if init_connected_to_parent is not None else int(init_parent_idx[joint_id]) >= 0}"
            f"<br>pos={_format_xyz(init_rest_joints[joint_id])}"
            for joint_id in range(init_rest_joints.shape[0])
        ]
    gaussian_hover = [
        f"g{idx}"
        f"<br>opacity={float(np.exp(gaussian_log_opacity[idx])):.3f}"
        f"<br>value={float(np.exp(gaussian_log_value[idx])):.3f}"
        f"<br>strength={float(np.exp(gaussian_log_opacity[idx] + gaussian_log_value[idx])):.3f}"
        f"<br>generation={int(gaussian_generation[idx])}<br>bone={int(gaussian_anchor_bone[idx])}"
        for idx in range(gaussian_centers.shape[0])
    ]
    gaussian_strength = np.exp(gaussian_log_opacity + gaussian_log_value).astype(np.float32, copy=False) if gaussian_log_opacity.size else np.zeros((0,), dtype=np.float32)
    traces: list[Any] = []
    trace_indices: dict[str, int] = {}

    trace_indices["rest_mesh"] = len(traces)
    traces.append(
        go.Mesh3d(
            x=rest_vertices[:, 0],
            y=rest_vertices[:, 1],
            z=rest_vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#cbd5e1",
            opacity=0.14,
            name="rest mesh",
            hoverinfo="skip",
        )
    )
    trace_indices["covariance_overlay"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line={"color": "rgba(14,165,233,0.75)", "width": 2},
            visible=False,
            name="selected joint covariance",
            hoverinfo="skip",
        )
    )
    trace_indices["weight_overlay"] = len(traces)
    traces.append(
        go.Mesh3d(
            x=rest_vertices[:, 0],
            y=rest_vertices[:, 1],
            z=rest_vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=np.zeros((rest_vertices.shape[0],), dtype=np.float32),
            colorscale="Plasma",
            cmin=0.0,
            cmax=1.0,
            opacity=0.46,
            visible=False,
            colorbar={"title": "skin weight", "x": 1.02, "len": 0.42, "y": 0.28},
            name="selected joint weight",
            hovertemplate="weight=%{intensity:.3f}<extra></extra>",
        )
    )
    if init_rest_joints is not None and init_parent_idx is not None:
        trace_indices["init_skeleton"] = len(traces)
        traces.extend(
            [
                go.Scatter3d(
                    x=init_line_x,
                    y=init_line_y,
                    z=init_line_z,
                    mode="lines",
                    line={"color": "#94a3b8", "width": 4},
                    name="init skeleton",
                    hoverinfo="skip",
                ),
                go.Scatter3d(
                    x=init_dash_x,
                    y=init_dash_y,
                    z=init_dash_z,
                    mode="lines",
                    line={"color": "rgba(148,163,184,0.55)", "width": 3},
                    name="init hierarchy link",
                    hoverinfo="skip",
                ),
                go.Scatter3d(
                    x=init_rest_joints[:, 0],
                    y=init_rest_joints[:, 1],
                    z=init_rest_joints[:, 2],
                    mode="markers",
                    marker={"size": 3, "color": "#94a3b8"},
                    text=init_joint_hover,
                    hovertemplate="%{text}<extra></extra>",
                    name="init joints",
                ),
            ]
        )
        trace_indices["init_joints"] = len(traces) - 1
    trace_indices["pred_skeleton"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": "#2563eb", "width": 6},
            name="pred skeleton",
            hoverinfo="skip",
        )
    )
    trace_indices["pred_hierarchy_links"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=dash_x,
            y=dash_y,
            z=dash_z,
            mode="lines",
            line={"color": "rgba(37,99,235,0.55)", "width": 4},
            name="pred hierarchy link",
            hoverinfo="skip",
        )
    )
    trace_indices["old_joints"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=old_points[:, 0] if old_points.size else [],
            y=old_points[:, 1] if old_points.size else [],
            z=old_points[:, 2] if old_points.size else [],
            mode="markers",
            marker={"size": 4, "color": "#1d4ed8"},
            text=old_text,
            hovertemplate="%{text}<extra></extra>",
            name="old joints",
        )
    )
    trace_indices["inserted_joints"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=inserted_points[:, 0] if inserted_points.size else [],
            y=inserted_points[:, 1] if inserted_points.size else [],
            z=inserted_points[:, 2] if inserted_points.size else [],
            mode="markers",
            marker={"size": 6, "color": "#f97316", "symbol": "diamond"},
            text=inserted_text,
            hovertemplate="%{text}<extra></extra>",
            name="inserted joints",
        )
    )
    trace_indices["gaussians_all"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=gaussian_centers[:, 0] if gaussian_centers.size else [],
            y=gaussian_centers[:, 1] if gaussian_centers.size else [],
            z=gaussian_centers[:, 2] if gaussian_centers.size else [],
            mode="markers",
            marker={
                "size": 4,
                "color": gaussian_strength.tolist() if gaussian_strength.size else [],
                "colorscale": "Viridis",
                "opacity": 0.92,
                "colorbar": {"title": "gaussian strength", "x": 1.02, "len": 0.42, "y": 0.76},
            },
            text=gaussian_hover,
            hovertemplate="%{text}<extra></extra>",
            name="gaussians",
        )
    )
    trace_indices["selected_joint_lines"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="lines",
            line={"color": "#ef4444", "width": 10},
            visible=False,
            name="selected joint edges",
            hoverinfo="skip",
        )
    )
    trace_indices["selected_joint_point"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker={"size": 9, "color": "#ef4444", "symbol": "circle-open"},
            text=[],
            visible=False,
            hovertemplate="%{text}<extra></extra>",
            name="selected joint",
        )
    )
    trace_indices["selected_joint_gaussians"] = len(traces)
    traces.append(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            marker={
                "size": 7,
                "color": [],
                "colorscale": "Turbo",
                "colorbar": {"title": "range max", "x": 1.12, "len": 0.42, "y": 0.76},
                "opacity": 1.0,
                "line": {"color": "#111827", "width": 1},
            },
            text=[],
            visible=False,
            hovertemplate="%{text}<extra></extra>",
            name="selected joint gaussians",
        )
    )

    fig = Figure(data=traces)
    fig.update_layout(
        meta={
            "evorig_viewer_type": "final_topology",
            "evorig_final_payload": {
                "trace_indices": trace_indices,
                "joints": joint_view_payload["joints"],
                "empty": joint_view_payload["empty"],
            },
        }
    )
    fig.update_layout(
        title="Final Topology Viewer | rest mesh + final skeleton",
        scene=_scene_layout(limits),
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        updatemenus=[_mesh_toggle_menu([trace_indices["rest_mesh"]], x=0.56, y=1.12)],
    )
    return fig


_FINAL_VIEWER_POST_SCRIPT = r"""
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !gd.layout || !gd.layout.meta || gd.layout.meta.evorig_viewer_type !== 'final_topology') {
    return;
  }
  var payload = gd.layout.meta.evorig_final_payload || {};
  var traceIdx = payload.trace_indices || {};
  var joints = payload.joints || [];
  var empty = payload.empty || {};
  var host = gd.parentNode;
  if (!host || host.dataset.evorigFinalControlsMounted === '1') {
    return;
  }
  host.dataset.evorigFinalControlsMounted = '1';

  function dequantize(values) {
    var arr = values || [];
    var out = new Array(arr.length);
    for (var i = 0; i < arr.length; i++) {
      out[i] = Number(arr[i]) / 255.0;
    }
    return out;
  }

  function jointData(index) {
    if (index < 0 || index >= joints.length) {
      return empty;
    }
    return joints[index];
  }

  function makeLabel(text) {
    var label = document.createElement('label');
    label.style.display = 'inline-flex';
    label.style.alignItems = 'center';
    label.style.gap = '6px';
    label.style.fontSize = '12px';
    label.style.color = '#0f172a';
    label.textContent = text;
    return label;
  }

  var panel = document.createElement('div');
  panel.style.display = 'flex';
  panel.style.flexWrap = 'wrap';
  panel.style.gap = '10px 14px';
  panel.style.alignItems = 'center';
  panel.style.margin = '4px 0 10px 0';
  panel.style.padding = '10px 12px';
  panel.style.border = '1px solid rgba(148,163,184,0.5)';
  panel.style.borderRadius = '10px';
  panel.style.background = 'rgba(248,250,252,0.92)';

  var jointLabel = makeLabel('Joint');
  var jointSelect = document.createElement('select');
  jointSelect.style.minWidth = '260px';
  jointSelect.style.padding = '4px 6px';
  var allOption = document.createElement('option');
  allOption.value = '-1';
  allOption.textContent = 'All joints';
  jointSelect.appendChild(allOption);
  joints.forEach(function(joint, index) {
    var option = document.createElement('option');
    option.value = String(index);
    option.textContent = joint.label;
    jointSelect.appendChild(option);
  });
  jointLabel.appendChild(jointSelect);
  panel.appendChild(jointLabel);

  var rangeLabel = makeLabel('Show Gaussian Covariance');
  var rangeCheck = document.createElement('input');
  rangeCheck.type = 'checkbox';
  rangeLabel.insertBefore(rangeCheck, rangeLabel.firstChild);
  panel.appendChild(rangeLabel);

  var weightLabel = makeLabel('Show Skin Weight');
  var weightCheck = document.createElement('input');
  weightCheck.type = 'checkbox';
  weightLabel.insertBefore(weightCheck, weightLabel.firstChild);
  panel.appendChild(weightLabel);

  var status = document.createElement('div');
  status.style.width = '100%';
  status.style.fontSize = '12px';
  status.style.color = '#0f172a';
  status.style.lineHeight = '1.5';
  panel.appendChild(status);
  host.insertBefore(panel, gd);

  function applyJointSelection(index) {
    var joint = jointData(index);
    var selected = index >= 0;
    Plotly.restyle(gd, {
      x: [joint.line_x || []],
      y: [joint.line_y || []],
      z: [joint.line_z || []],
      visible: [selected]
    }, [traceIdx.selected_joint_lines]);
    Plotly.restyle(gd, {
      x: [joint.joint_x || []],
      y: [joint.joint_y || []],
      z: [joint.joint_z || []],
      text: [joint.joint_text || []],
      visible: [selected]
    }, [traceIdx.selected_joint_point]);
    Plotly.restyle(gd, {
      x: [joint.gaussian_x || []],
      y: [joint.gaussian_y || []],
      z: [joint.gaussian_z || []],
      text: [joint.gaussian_text || []],
      'marker.color': [joint.gaussian_color || []],
      'marker.cmin': [0.0],
      'marker.cmax': [joint.gaussian_cmax || 1.0],
      visible: [selected && (joint.gaussian_x || []).length > 0]
    }, [traceIdx.selected_joint_gaussians]);
    Plotly.restyle(gd, {
      x: [joint.cov_x || []],
      y: [joint.cov_y || []],
      z: [joint.cov_z || []],
      visible: [selected && rangeCheck.checked && (joint.cov_x || []).length > 0]
    }, [traceIdx.covariance_overlay]);
    Plotly.restyle(gd, {
      intensity: [dequantize(joint.weight_u8 || [])],
      cmin: [0.0],
      cmax: [joint.weight_cmax || 1.0],
      visible: [selected && weightCheck.checked]
    }, [traceIdx.weight_overlay]);
    if (!selected) {
      status.textContent = 'Selected joint: none. Choose one joint to inspect its gaussians, covariance ellipsoids, and final skin weight.';
      return;
    }
    status.innerHTML =
      'Selected joint: <b>' + joint.label + '</b>' +
      ' | parent=' + joint.parent_joint +
      ' | supporting gaussians=' + joint.gaussian_count +
      ' | dominant gaussians=' + (joint.dominant_gaussian_count || 0) +
      ' | sigma max=' + Number(joint.gaussian_cmax || 0).toFixed(3) +
      ' | skin-weight max=' + Number(joint.weight_cmax || 0).toFixed(3);
  }

  jointSelect.addEventListener('change', function() {
    applyJointSelection(Number(jointSelect.value));
  });
  rangeCheck.addEventListener('change', function() {
    applyJointSelection(Number(jointSelect.value));
  });
  weightCheck.addEventListener('change', function() {
    applyJointSelection(Number(jointSelect.value));
  });

  applyJointSelection(-1);
})();
"""


def save_figure_html(fig: Any, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        post_script=_FINAL_VIEWER_POST_SCRIPT,
    )
    return str(path)
