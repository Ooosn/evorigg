from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evorig_next.io.real_glb import import_real_glb_sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a real GLB rigged mesh + animated mesh pair into EvoRig sample format.")
    parser.add_argument("--rigged-glb", default=str(ROOT / "assets" / "rigged.glb"))
    parser.add_argument("--animated-glb", default=str(ROOT / "assets" / "animated_mesh.glb"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-template", default=str(ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = import_real_glb_sample(
        rigged_glb_path=args.rigged_glb,
        animated_glb_path=args.animated_glb,
        output_dir=args.output_dir,
        config_template_path=args.config_template,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
