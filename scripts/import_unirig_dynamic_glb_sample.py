from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.unirig_dynamic import import_unirig_dynamic_sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a UniRig dynamic GLB output folder into EvoRig sample format.")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--source-dir", required=True, help="Folder containing dynamic_mesh.glb and rigged.glb")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blender-path", default=r"D:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
    parser.add_argument("--config-template", default=str(ROOT / "configs" / "frozen" / "evorig_next_base_init_default.yaml"))
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--uniform-fraction", type=float, default=0.625)
    parser.add_argument("--motion-min-gap", type=int, default=0)
    parser.add_argument("--no-include-last", action="store_true")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument(
        "--alignment-frame-index",
        type=int,
        default=None,
        help="Dynamic source frame that corresponds to rigged.glb rest. Defaults to --frame-start.",
    )
    parser.add_argument("--normalize-target-diag", type=float, default=2.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-correspondence-p95-relative", type=float, default=0.03)
    parser.add_argument("--max-correspondence-max-relative", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = import_unirig_dynamic_sample(
        asset=args.asset,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        blender_path=args.blender_path,
        config_template_path=args.config_template,
        max_frames=args.max_frames,
        uniform_fraction=args.uniform_fraction,
        include_last=not bool(args.no_include_last),
        motion_min_gap=args.motion_min_gap,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        alignment_frame_index=args.frame_start if args.alignment_frame_index is None else args.alignment_frame_index,
        normalize_target_diag=None if bool(args.no_normalize) else float(args.normalize_target_diag),
        max_correspondence_p95_relative=args.max_correspondence_p95_relative,
        max_correspondence_max_relative=args.max_correspondence_max_relative,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
