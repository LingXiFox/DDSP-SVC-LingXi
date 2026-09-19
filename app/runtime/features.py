"""Feature extraction orchestration (reuses repo extractors, no new math)."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FeatureBundle:
    units: torch.Tensor  # [1, T, units] on target device
    f0: torch.Tensor  # [1, T, 1] on target device
    volume: torch.Tensor  # [1, T, 1] on target device
    frame_count: int


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values in {name}.")
    return tensor


def align_frames(units, f0, volume):
    """Trim all streams to the shared frame count (repo inference semantics)."""
    frames = min(units.size(1), f0.size(1), volume.size(1))
    if frames <= 0:
        raise ValueError("Feature extraction produced zero frames.")
    return units[:, :frames], f0[:, :frames], volume[:, :frames], frames


def to_device_float32(tensor, device):
    out = tensor.to(device=device, dtype=torch.float32)
    return check_finite("features", out)
