from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("torch", "torch"),
    ("tqdm", "tqdm"),
    ("trimesh", "trimesh"),
    ("scipy", "scipy"),
    ("scipy.ndimage", "scipy"),
    ("scipy.spatial", "scipy"),
    ("open3d", "open3d"),
    ("plotly.graph_objects", "plotly"),
]


def _module_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_required_modules() -> tuple[dict[str, Any], list[str]]:
    modules: dict[str, Any] = {}
    errors: list[str] = []
    for module_name, distribution in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{module_name}: import failed: {exc}")
            continue
        modules[module_name] = {
            "distribution": distribution,
            "version": _module_version(distribution),
        }
    return modules, errors


def _check_open3d() -> list[str]:
    errors: list[str] = []
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"open3d: import failed: {exc}"]
    if not hasattr(o3d, "t") or not hasattr(o3d.t, "geometry"):
        errors.append("open3d: missing tensor geometry namespace")
    elif not hasattr(o3d.t.geometry, "RaycastingScene"):
        errors.append("open3d: missing t.geometry.RaycastingScene")
    return errors


def _check_cuda(*, require_cuda: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False}, [f"torch: import failed before CUDA check: {exc}"]
    available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "available": available,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "devices": [],
    }
    if available:
        for idx in range(int(torch.cuda.device_count())):
            info["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                }
            )
    elif require_cuda:
        errors.append("CUDA is required for official EvoRig runs, but torch.cuda.is_available() is false")
    return info, errors


def _check_repo_imports() -> list[str]:
    errors: list[str] = []
    required = [
        "evorig_next.io.data",
        "evorig_next.phase1_config",
        "evorig_next.phase1_trainer",
        "evorig_next.phase2_topology",
        "evorig_next.phase3_refine",
        "evorig_next.interactive_viewer",
    ]
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: import failed: {exc}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict EvoRigNext environment preflight. Fails instead of allowing missing-module fallbacks."
    )
    parser.add_argument(
        "--no-cuda-required",
        action="store_true",
        help="Allow CPU-only diagnostics. Do not use this for official experiments.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors: list[str] = []
    modules, module_errors = _import_required_modules()
    errors.extend(module_errors)
    errors.extend(_check_open3d())
    cuda_info, cuda_errors = _check_cuda(require_cuda=not bool(args.no_cuda_required))
    errors.extend(cuda_errors)
    errors.extend(_check_repo_imports())

    report = {
        "status": "ok" if not errors else "failed",
        "python": sys.executable,
        "repo_root": str(ROOT),
        "modules": modules,
        "cuda": cuda_info,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
