from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evorig_next.utils.rotations import quaternion_to_matrix, matrix_to_axis_angle


def save_numpy(path: Path, tensor: torch.Tensor) -> None:
    np.save(path, tensor.detach().cpu().numpy())


def save_rig_json(path: Path, skeleton: Any) -> None:
    joints = []
    bind_transforms = skeleton.compute_bind_transforms().detach().cpu().tolist()
    rest_joints = skeleton.rest_joints.detach().cpu().tolist()
    parent_idx = skeleton.parent_idx.detach().cpu().tolist()
    connected_to_parent = getattr(skeleton, "connected_to_parent", None)
    if isinstance(connected_to_parent, torch.Tensor):
        connected_values = connected_to_parent.detach().cpu().bool().tolist()
    else:
        connected_values = [int(parent_id) >= 0 for parent_id in parent_idx]
    for joint_id, parent_id in enumerate(parent_idx):
        joints.append(
            {
                "id": joint_id,
                "parent_id": parent_id,
                "connected_to_parent": bool(connected_values[joint_id]) and int(parent_id) >= 0,
                "rest_position": rest_joints[joint_id],
                "bind_transform": bind_transforms[joint_id],
                "birth_step": skeleton.birth_steps[joint_id],
                "is_inserted": skeleton.is_inserted[joint_id],
                "birth_mode": skeleton.birth_modes[joint_id],
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"joints": joints}, handle, indent=2)


def save_gaussians(path: Path, field: Any) -> None:
    rot_q = field.rot_local.detach().cpu()
    rot_axis_angle = matrix_to_axis_angle(quaternion_to_matrix(rot_q)).cpu().numpy()
    log_opacity = field.log_opacity.detach().cpu().numpy()
    log_value = field.log_value.detach().cpu().numpy()
    sh_coeffs = field.sh_coeffs.detach().cpu().numpy()
    np.savez(
        path,
        **{
            "anchor_bone": field.anchor_bone.detach().cpu().numpy(),
            "lambda": field.lambda_param.detach().cpu().numpy(),
            "lambda_min": field.lambda_min.detach().cpu().numpy(),
            "lambda_max": field.lambda_max.detach().cpu().numpy(),
            "offset_local": field.offset_local.detach().cpu().numpy(),
            "rot_q": rot_q.numpy(),
            "rot_local": rot_axis_angle,
            "log_scale": field.log_scale.detach().cpu().numpy(),
            "log_opacity": log_opacity,
            "log_value": log_value,
            "log_alpha": log_opacity + log_value,
            "sh_coeffs": sh_coeffs,
            "q_logits": field.q_logits.detach().cpu().numpy(),
            "endpoint_logits": field.endpoint_logits.detach().cpu().numpy(),
            "generation": field.generation.detach().cpu().numpy(),
            "active_mask": field.active_mask.detach().cpu().numpy(),
        },
    )


def save_topology_events(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(events, handle, indent=2)


def save_topology_diagnostics(path: Path, diagnostics: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)


def save_training_trace_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_training_trace_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    if not isinstance(snapshot, dict):
        raise ValueError(f"trace snapshot must be a dict: {path}")
    return snapshot


def _materialize_training_trace(trace: list[dict[str, Any]] | Path) -> list[dict[str, Any]]:
    if isinstance(trace, Path):
        if not trace.exists() or not trace.is_dir():
            return []
        snapshots: list[dict[str, Any]] = []
        for snapshot_path in sorted(trace.glob("*.pkl")):
            snapshots.append(_load_training_trace_snapshot(snapshot_path))
        return snapshots
    return trace


def save_training_trace(path: Path, trace: list[dict[str, Any]] | Path) -> None:
    trace_dir = trace if isinstance(trace, Path) else None
    snapshots = _materialize_training_trace(trace)
    if not snapshots:
        return

    def object_stack(values: list[Any]) -> np.ndarray:
        result = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            result[index] = value
        return result

    payload = {
        "step": np.asarray([item["step"] for item in snapshots], dtype=np.int32),
        "progress": np.asarray([item.get("progress", np.nan) for item in snapshots], dtype=np.float32),
        "label": np.asarray([item["label"] for item in snapshots], dtype=object),
        "loss": np.asarray([item["loss"] for item in snapshots], dtype=np.float32),
        "recon": np.asarray([item["recon"] for item in snapshots], dtype=np.float32),
        "recon_raw": np.asarray([item.get("recon_raw", np.nan) for item in snapshots], dtype=np.float32),
        "sample_radius": np.asarray([item.get("sample_radius", np.nan) for item in snapshots], dtype=np.float32),
        "joint_count": np.asarray([item["joint_count"] for item in snapshots], dtype=np.int32),
        "gaussian_count": np.asarray([item["gaussian_count"] for item in snapshots], dtype=np.int32),
        "gaussian_strictness": np.asarray([item.get("gaussian_strictness", np.nan) for item in snapshots], dtype=np.float32),
        "skeleton_anchor_scale": np.asarray([item.get("skeleton_anchor_scale", np.nan) for item in snapshots], dtype=np.float32),
        "pred_vertices": object_stack([item.get("pred_vertices") for item in snapshots]),
        "joint_positions": object_stack([item["joint_positions"] for item in snapshots]),
        "joint_rotations": object_stack([item.get("joint_rotations") for item in snapshots]),
        "pose_rot_local": object_stack([item.get("pose_rot_local") for item in snapshots]),
        "rest_joints": object_stack([item["rest_joints"] for item in snapshots]),
        "parent_idx": object_stack([item["parent_idx"] for item in snapshots]),
        "gaussian_centers": object_stack([item["gaussian_centers"] for item in snapshots]),
        "gaussian_lambda": object_stack([item.get("gaussian_lambda") for item in snapshots]),
        "gaussian_log_opacity": object_stack([item.get("gaussian_log_opacity") for item in snapshots]),
        "gaussian_log_value": object_stack([item.get("gaussian_log_value") for item in snapshots]),
        "gaussian_log_alpha": object_stack([item.get("gaussian_log_alpha") for item in snapshots]),
        "gaussian_generation": object_stack([item["gaussian_generation"] for item in snapshots]),
        "gaussian_active_mask": object_stack([item["gaussian_active_mask"] for item in snapshots]),
        "support_mass": object_stack([item.get("support_mass") for item in snapshots]),
        "residual": object_stack([item.get("residual") for item in snapshots]),
        "residual_raw": object_stack([item.get("residual_raw") for item in snapshots]),
        "joint_gradient": object_stack([item.get("joint_gradient") for item in snapshots]),
        "bone_gradient": object_stack([item.get("bone_gradient") for item in snapshots]),
        "gaussian_gradient": object_stack([item.get("gaussian_gradient") for item in snapshots]),
        "anchor_bone": object_stack([item["anchor_bone"] for item in snapshots]),
        "gaussian_dominant_joint": object_stack([item.get("gaussian_dominant_joint") for item in snapshots]),
        "gaussian_dominant_weight": object_stack([item.get("gaussian_dominant_weight") for item in snapshots]),
        "event_count": np.asarray([item["event_count"] for item in snapshots], dtype=np.int32),
        "active_event_profile_state": object_stack([item.get("active_event_profile_state", []) for item in snapshots]),
    }
    np.savez_compressed(path, **payload)
    if trace_dir is not None and trace_dir.exists():
        shutil.rmtree(trace_dir, ignore_errors=True)


def save_parent_assignment_history(path: Path, history: dict[int, list[dict[str, Any]]]) -> None:
    serializable = {str(int(joint_id)): entries for joint_id, entries in history.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def save_parent_candidate_history(path: Path, history: dict[int, list[dict[str, Any]]]) -> None:
    serializable = {str(int(joint_id)): entries for joint_id, entries in history.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def save_parent_selection_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_postprocess_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_outputs(
    output_dir: str | Path,
    skeleton: Any,
    field: Any,
    pred_vertices: torch.Tensor,
    pred_joint_positions: torch.Tensor | None,
    pred_joint_rotations: torch.Tensor | None,
    weights: torch.Tensor,
    events: list[dict[str, Any]],
    topology_diagnostics: list[dict[str, Any]] | None = None,
    training_trace: list[dict[str, Any]] | Path | None = None,
    parent_assignment_history: dict[int, list[dict[str, Any]]] | None = None,
    parent_candidate_history: dict[int, list[dict[str, Any]]] | None = None,
    parent_selection_summary: dict[str, Any] | None = None,
    postprocess_summary: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rig_json(output_dir / "pred_rig_final.json", skeleton)
    save_numpy(output_dir / "pred_anim_vertices.npy", pred_vertices)
    if pred_joint_positions is not None:
        save_numpy(output_dir / "pred_joint_positions.npy", pred_joint_positions)
    if pred_joint_rotations is not None:
        save_numpy(output_dir / "pred_joint_rotations.npy", pred_joint_rotations)
    save_numpy(output_dir / "pred_weights.npy", weights)
    save_gaussians(output_dir / "gaussians_final.npz", field)
    save_topology_events(output_dir / "topology_events.json", events)
    if topology_diagnostics is not None:
        save_topology_diagnostics(output_dir / "topology_diagnostics.json", topology_diagnostics)
    if training_trace is not None:
        save_training_trace(output_dir / "training_trace.npz", training_trace)
    if parent_assignment_history is not None:
        save_parent_assignment_history(output_dir / "parent_assignment_history.json", parent_assignment_history)
    if parent_candidate_history is not None:
        save_parent_candidate_history(output_dir / "parent_candidate_history.json", parent_candidate_history)
    if parent_selection_summary is not None:
        save_parent_selection_summary(output_dir / "parent_selection_summary.json", parent_selection_summary)
    if postprocess_summary is not None:
        save_postprocess_summary(output_dir / "postprocess_summary.json", postprocess_summary)
