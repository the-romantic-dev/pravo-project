from __future__ import annotations

from src import config


def get_model_device() -> str:
    configured = str(getattr(config, "model_device", "auto")).strip().lower()
    if configured and configured != "auto":
        return configured

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_transformers_device() -> int:
    return 0 if get_model_device() == "cuda" else -1
