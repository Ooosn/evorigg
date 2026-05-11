from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def _resolve_blender_path(path: str | None) -> str:
    if path:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return str(candidate)
    default = Path(r"D:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
    if default.exists():
        return str(default)
    raise FileNotFoundError("Blender executable not found; pass --blender-path")


def _run_bake(dynamic_glb: Path, cache_dir: Path, blender_path: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_npz = cache_dir / "baked_dynamic_frames_raw_blender.npz"
    report_json = cache_dir / "baked_dynamic_frames_report.json"
    if not frames_npz.exists():
        cmd = [
            _resolve_blender_path(blender_path),
            "--background",
            "--python",
            str(ROOT / "scripts" / "blender_bake_dynamic_glb_frames.py"),
            "--",
            str(dynamic_glb),
            str(frames_npz),
            str(report_json),
        ]
        subprocess.run(cmd, check=True)
    data = np.load(frames_npz, allow_pickle=True)
    report = json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else {}
    return (
        np.asarray(data["frames"], dtype=np.float32),
        np.asarray(data["faces"], dtype=np.int64),
        np.asarray(data["source_times"], dtype=np.float32),
        report,
    )


def _motion_scores(frames: np.ndarray) -> dict[str, np.ndarray]:
    frame_count = int(frames.shape[0])
    step_motion = np.zeros(frame_count, dtype=np.float32)
    if frame_count > 1:
        step_motion[1:] = np.linalg.norm(frames[1:] - frames[:-1], axis=-1).mean(axis=1)
    rest_motion = np.linalg.norm(frames - frames[:1], axis=-1).mean(axis=1).astype(np.float32)
    accel = np.zeros(frame_count, dtype=np.float32)
    if frame_count > 2:
        accel[2:] = np.linalg.norm(frames[2:] - 2.0 * frames[1:-1] + frames[:-2], axis=-1).mean(axis=1)
    return {"step_motion": step_motion, "rest_motion": rest_motion, "acceleration_motion": accel}


def _pick_window(frames: np.ndarray, *, length: int, max_step_jump_ratio: float) -> tuple[int, int, dict]:
    frame_count = int(frames.shape[0])
    if frame_count <= 0:
        raise ValueError("empty dynamic mesh sequence")
    length = min(max(int(length), 2), frame_count)
    stats = _motion_scores(frames)
    step = stats["step_motion"]
    rest_from_start = np.zeros(frame_count, dtype=np.float32)
    best: tuple[float, int, float, float] | None = None
    global_p95_step = float(np.percentile(step[1:] if frame_count > 1 else step, 95))
    jump_limit = max(global_p95_step * float(max_step_jump_ratio), 1.0e-8)
    for start in range(0, frame_count - length + 1):
        end = start + length - 1
        window = frames[start : end + 1]
        local_step = np.zeros(length, dtype=np.float32)
        local_step[1:] = np.linalg.norm(window[1:] - window[:-1], axis=-1).mean(axis=1)
        local_rest = np.linalg.norm(window - window[:1], axis=-1).mean(axis=1).astype(np.float32)
        local_accel = np.zeros(length, dtype=np.float32)
        if length > 2:
            local_accel[2:] = np.linalg.norm(window[2:] - 2.0 * window[1:-1] + window[:-2], axis=-1).mean(axis=1)
        max_step = float(local_step.max(initial=0.0))
        if max_step > jump_limit:
            continue
        # Prefer continuous windows with large deformation and sustained motion, not isolated jumps.
        score = float(local_rest.max(initial=0.0) + 0.5 * local_step.mean() + 0.25 * local_accel.mean())
        candidate = (score, int(start), float(local_rest.max(initial=0.0)), float(local_step.mean()))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        # Fallback: still choose the strongest window, but record that jump filtering failed.
        for start in range(0, frame_count - length + 1):
            end = start + length - 1
            window = frames[start : end + 1]
            local_step = np.zeros(length, dtype=np.float32)
            local_step[1:] = np.linalg.norm(window[1:] - window[:-1], axis=-1).mean(axis=1)
            local_rest = np.linalg.norm(window - window[:1], axis=-1).mean(axis=1).astype(np.float32)
            score = float(local_rest.max(initial=0.0) + 0.5 * local_step.mean())
            candidate = (score, int(start), float(local_rest.max(initial=0.0)), float(local_step.mean()))
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    start = int(best[1])
    end = int(start + length - 1)
    report = {
        "selection_policy": "best_contiguous_motion_window",
        "frame_start": start,
        "frame_end": end,
        "frame_count": length,
        "score": float(best[0]),
        "window_max_rest_motion": float(best[2]),
        "window_mean_step_motion": float(best[3]),
        "global_step_p95": global_p95_step,
        "jump_limit": float(jump_limit),
        "jump_filter": "passed" if float(np.max(step[start : end + 1], initial=0.0)) <= jump_limit else "fallback",
        "top_step_motion_frames": [
            {"frame": int(i), "value": float(step[i])} for i in np.argsort(-step)[:12].tolist()
        ],
    }
    return start, end, report


def _normalization_transform(vertices: np.ndarray, target_diag: float) -> dict:
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    diag = float(np.linalg.norm(bounds_max - bounds_min))
    scale = float(target_diag) / max(diag, 1.0e-8)
    return {
        "mode": "bbox_center_diag",
        "target_diag": float(target_diag),
        "source_bounds_min": bounds_min.astype(float).tolist(),
        "source_bounds_max": bounds_max.astype(float).tolist(),
        "source_diag": diag,
        "center": center.astype(float).tolist(),
        "scale": scale,
    }


def _apply_normalization(points: np.ndarray, transform: dict) -> np.ndarray:
    center = np.asarray(transform["center"], dtype=np.float32)
    scale = float(transform["scale"])
    return ((points - center.reshape((1,) * (points.ndim - 1) + (3,))) * scale).astype(np.float32)


def _posix_e_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved)[3:].replace("\\", "/")
    return f"/{drive}/{tail}"


def _write_unirig_scripts(asset_dir: Path) -> None:
    env_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd '/d/git/UniRig'",
        "export PATH='/d/Users/namew/miniconda3/envs/UniRig:/d/Users/namew/miniconda3/envs/UniRig/Scripts:/d/Users/namew/miniconda3/envs/UniRig/Library/bin':$PATH",
        "export HF_HOME='/e/huggingface'",
        "export HUGGINGFACE_HUB_CACHE='/e/huggingface/hub'",
        "export HF_HUB_CACHE='/e/huggingface/hub'",
        "export TORCH_HOME='/e/torch'",
        "export HF_HUB_DISABLE_XET='1'",
        "export HF_HUB_DISABLE_SYMLINKS_WARNING='1'",
        "export PYTHONUTF8='1'",
        "export PYTHONIOENCODING='utf-8'",
        "export LANG='C.UTF-8'",
        "export LC_ALL='C.UTF-8'",
        "export TERM='dumb'",
        "export NO_COLOR='1'",
        "export RICH_FORCE_TERMINAL='0'",
        "export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'",
    ]
    mesh = _posix_e_path(asset_dir / "mesh.glb")
    skeleton = _posix_e_path(asset_dir / "skeleton.fbx")
    skin = _posix_e_path(asset_dir / "skin.fbx")
    rigged = _posix_e_path(asset_dir / "rigged.glb")
    scripts = {
        "run_skeleton.sh": f"bash launch/inference/generate_skeleton.sh --force_override true --input '{mesh}' --output '{skeleton}'",
        "run_skin.sh": f"bash launch/inference/generate_skin.sh --force_override true --input '{skeleton}' --output '{skin}'",
        "run_merge.sh": f"bash launch/inference/merge.sh --source '{skin}' --target '{mesh}' --output '{rigged}'",
    }
    for name, command in scripts.items():
        (asset_dir / name).write_text("\n".join(env_lines + [command, ""]) , encoding="utf-8", newline="\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a normalized rest mesh for UniRig from a dynamic GLB window.")
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--normalize-target-diag", type=float, default=2.0)
    parser.add_argument("--max-step-jump-ratio", type=float, default=3.0)
    parser.add_argument("--blender-path", default=r"D:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    asset_dir = Path(args.asset_dir)
    dynamic_glb = asset_dir / "dynamic_mesh.glb"
    if not dynamic_glb.exists():
        raise FileNotFoundError(dynamic_glb)
    frames, faces, source_times, bake_report = _run_bake(dynamic_glb, asset_dir / "_bake_cache", args.blender_path)
    if args.frame_start is None:
        frame_start, frame_end, selection_report = _pick_window(
            frames,
            length=int(args.window_length),
            max_step_jump_ratio=float(args.max_step_jump_ratio),
        )
    else:
        frame_start = int(args.frame_start)
        frame_end = int(args.frame_end) if args.frame_end is not None else min(frame_start + int(args.window_length) - 1, int(frames.shape[0]) - 1)
        if frame_start < 0 or frame_end >= int(frames.shape[0]) or frame_start > frame_end:
            raise ValueError(f"invalid requested frame window [{frame_start}, {frame_end}] for {frames.shape[0]} frames")
        selection_report = {
            "selection_policy": "manual_contiguous_window",
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": int(frame_end - frame_start + 1),
        }
    rest_raw = frames[frame_start]
    normalization = _normalization_transform(rest_raw, float(args.normalize_target_diag))
    rest_vertices = _apply_normalization(rest_raw, normalization)
    mesh = trimesh.Trimesh(vertices=rest_vertices, faces=faces, process=False)
    mesh.export(asset_dir / "mesh.glb")
    _write_unirig_scripts(asset_dir)

    quality = {
        "source_frame_count": int(frames.shape[0]),
        "vertex_count": int(frames.shape[1]),
        "face_count": int(faces.shape[0]),
        "has_nan_or_inf": bool((~np.isfinite(frames)).any()),
        "rest_bbox_diag": float(np.linalg.norm(rest_vertices.max(axis=0) - rest_vertices.min(axis=0))),
        "rest_vs_gt0_rms": 0.0,
        "rest_vs_gt0_max": 0.0,
    }
    report = {
        "asset_dir": str(asset_dir.resolve()),
        "dynamic_mesh_glb": str(dynamic_glb.resolve()),
        "mesh_glb": str((asset_dir / "mesh.glb").resolve()),
        "frame_window": selection_report,
        "normalization": normalization,
        "quality": quality,
        "bake_report": bake_report,
        "next_commands": [
            f"bash {asset_dir / 'run_skeleton.sh'}",
            f"bash {asset_dir / 'run_skin.sh'}",
            f"bash {asset_dir / 'run_merge.sh'}",
        ],
    }
    (asset_dir / "window_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
