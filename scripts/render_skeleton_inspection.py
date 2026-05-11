from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _load_obj_vertices(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not rows:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(rows, dtype=np.float32)


def render_inspection(sample_dir: Path, run_dir: Path, output: Path) -> Path:
    vertices = _load_obj_vertices(sample_dir / "rest_mesh.obj")
    rig = json.loads((run_dir / "pred_rig_final.json").read_text(encoding="utf-8"))
    joints = np.asarray([item["rest_position"] for item in rig["joints"]], dtype=np.float32)
    parents = np.asarray([item["parent_id"] for item in rig["joints"]], dtype=np.int64)
    inserted = np.asarray([bool(item.get("is_inserted", False)) for item in rig["joints"]], dtype=bool)

    segments: list[np.ndarray] = []
    colors: list[str] = []
    for joint_id, parent_id in enumerate(parents):
        if int(parent_id) < 0:
            continue
        segments.append(np.stack([joints[int(parent_id)], joints[int(joint_id)]], axis=0))
        colors.append("#f97316" if bool(inserted[int(joint_id)]) else "#2563eb")

    rng = np.random.default_rng(0)
    draw_ids = np.arange(int(vertices.shape[0]))
    if int(draw_ids.shape[0]) > 7000:
        draw_ids = rng.choice(draw_ids, size=7000, replace=False)
    draw_vertices = vertices[draw_ids]

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    span = float((bounds_max - bounds_min).max()) * 0.58
    limits = [(float(coord - span), float(coord + span)) for coord in center]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 9), dpi=160)
    views = [("front", (8, -82)), ("side", (8, 8)), ("top", (85, -90)), ("iso", (20, -45))]
    for plot_index, (name, view) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, plot_index, projection="3d")
        ax.scatter(
            draw_vertices[:, 0],
            draw_vertices[:, 1],
            draw_vertices[:, 2],
            s=0.25,
            c="#d1d5db",
            alpha=0.18,
            depthshade=False,
        )
        if segments:
            ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=2.3, alpha=0.95))
        if np.any(~inserted):
            ax.scatter(
                joints[~inserted, 0],
                joints[~inserted, 1],
                joints[~inserted, 2],
                s=24,
                c="#1d4ed8",
                depthshade=False,
            )
        if np.any(inserted):
            ax.scatter(
                joints[inserted, 0],
                joints[inserted, 1],
                joints[inserted, 2],
                s=30,
                c="#f97316",
                marker="D",
                depthshade=False,
            )
        for joint_id, point in enumerate(joints):
            ax.text(float(point[0]), float(point[1]), float(point[2]), str(joint_id), fontsize=5, color="#111827")
        ax.view_init(elev=float(view[0]), azim=float(view[1]))
        ax.set_title(name, fontsize=10)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Render mesh+skeleton inspection views for an EvoRig run.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render_inspection(args.sample_dir, args.run_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
