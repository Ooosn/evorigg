from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.interactive_viewer import build_final_topology_figure, build_motion_figure, build_training_figure, save_figure_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interactive 3D HTML viewers for an EvoRig run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing data/ and output/")
    parser.add_argument(
        "--kind",
        choices=("training", "motion", "final", "both", "all"),
        default="both",
        help="Which interactive viewer(s) to export. both exports motion+final; all also exports training.",
    )
    parser.add_argument(
        "--color-by",
        choices=("residual", "support", "generation", "alpha"),
        default="residual",
        help="Gaussian coloring metric for the training viewer.",
    )
    parser.add_argument("--trace-stride", type=int, default=1, help="Keep every Nth trace snapshot in the training viewer.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated HTML. Defaults to <run-dir>/visuals/interactive",
    )
    parser.add_argument("--open-browser", action="store_true", help="Open generated HTML files in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "visuals" / "interactive")
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []

    if args.kind in ("training", "all"):
        try:
            training_fig = build_training_figure(
                run_dir,
                trace_stride=args.trace_stride,
                color_by=args.color_by,
            )
        except FileNotFoundError:
            if args.kind == "training":
                raise
            print(f"[viewer] skip training viewer for {run_dir}: missing training trace", file=sys.stderr)
        else:
            training_path = output_dir / f"interactive_training_{args.color_by}.html"
            save_figure_html(training_fig, training_path)
            written_paths.append(training_path)

    if args.kind in ("motion", "both", "all"):
        motion_fig = build_motion_figure(run_dir)
        motion_path = output_dir / "interactive_motion.html"
        save_figure_html(motion_fig, motion_path)
        written_paths.append(motion_path)

    if args.kind in ("final", "both", "all"):
        final_fig = build_final_topology_figure(run_dir)
        final_path = output_dir / "interactive_final_topology.html"
        save_figure_html(final_fig, final_path)
        written_paths.append(final_path)

    for path in written_paths:
        print(path)
        if args.open_browser:
            webbrowser.open_new_tab(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
