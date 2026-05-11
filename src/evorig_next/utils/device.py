from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        requested = str(device)
    elif device is None:
        requested = "auto"
    else:
        requested = str(device).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment")
    return torch.device(requested)
