from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


_EPS = 1e-5


def _require_3d_control(name: str, value: torch.Tensor) -> None:
    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [B, T, 1]")


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return kernel_size


def _masked_local_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size)
    pad = kernel_size // 2
    value_ch = value.transpose(1, 2)
    mask_ch = mask.to(value.dtype).transpose(1, 2)
    numerator = F.avg_pool1d(
        value_ch * mask_ch, kernel_size, stride=1, padding=pad
    )
    denominator = F.avg_pool1d(
        mask_ch, kernel_size, stride=1, padding=pad
    )
    return (numerator / denominator.clamp_min(_EPS)).transpose(1, 2)


def _local_mean(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size)
    pad = kernel_size // 2
    value_ch = value.transpose(1, 2)
    if pad > 0:
        value_ch = F.pad(value_ch, (pad, pad), mode="replicate")
    return F.avg_pool1d(
        value_ch, kernel_size, stride=1, padding=0
    ).transpose(1, 2)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(value.dtype)
    return (value * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(
        pred, target, reduction="none", beta=beta
    )
    return _masked_mean(loss, mask)


def extract_realism_features(
    f0: torch.Tensor,
    volume: torch.Tensor,
    trend_kernel: int = 31,
) -> Dict[str, torch.Tensor]:
    """Extract singer-weak control features from frame-level F0 and loudness.

    Absolute pitch register and absolute loudness are removed with local trends.
    The returned tensors describe performance dynamics rather than timbre.
    """
    _require_3d_control("f0", f0)
    _require_3d_control("volume", volume)
    if f0.shape[:2] != volume.shape[:2]:
        raise ValueError("f0 and volume must share batch/frame dimensions")

    trend_kernel = _odd_kernel(trend_kernel)
    voiced = f0 > 0

    log2_f0 = torch.where(
        voiced,
        torch.log2(torch.clamp(f0, min=_EPS)),
        torch.zeros_like(f0),
    )
    pitch_trend = _masked_local_mean(log2_f0, voiced, trend_kernel)
    pitch_residual_cents = torch.where(
        voiced,
        (log2_f0 - pitch_trend) * 1200.0,
        torch.zeros_like(f0),
    )

    pitch_velocity = (
        pitch_residual_cents[:, 1:] - pitch_residual_cents[:, :-1]
    )
    pitch_velocity_mask = voiced[:, 1:] & voiced[:, :-1]
    pitch_acceleration = pitch_velocity[:, 1:] - pitch_velocity[:, :-1]
    pitch_acceleration_mask = (
        pitch_velocity_mask[:, 1:] & pitch_velocity_mask[:, :-1]
    )

    volume_db = 20.0 * torch.log10(torch.clamp(volume, min=_EPS))
    volume_trend = _local_mean(volume_db, trend_kernel)
    energy_residual_db = volume_db - volume_trend
    energy_velocity = (
        energy_residual_db[:, 1:] - energy_residual_db[:, :-1]
    )

    return {
        "voiced": voiced,
        "pitch_residual_cents": pitch_residual_cents,
        "pitch_velocity_cents": pitch_velocity,
        "pitch_velocity_mask": pitch_velocity_mask,
        "pitch_acceleration_cents": pitch_acceleration,
        "pitch_acceleration_mask": pitch_acceleration_mask,
        "energy_residual_db": energy_residual_db,
        "energy_velocity_db": energy_velocity,
    }


def robotize_controls(
    f0: torch.Tensor,
    volume: torch.Tensor,
    smoothing_kernel: int = 9,
    pitch_quantization_cents: float = 0.0,
    volume_quantization_db: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove micro-dynamics while keeping pitch register, loudness and voicing.

    This creates an identity-safe training input for the public realism prior:
    the adapter sees an over-smoothed trajectory and learns to restore human
    performance dynamics from speaker-weak targets.
    """
    _require_3d_control("f0", f0)
    _require_3d_control("volume", volume)
    if f0.shape[:2] != volume.shape[:2]:
        raise ValueError("f0 and volume must share batch/frame dimensions")
    if pitch_quantization_cents < 0 or volume_quantization_db < 0:
        raise ValueError("quantization steps must be non-negative")

    smoothing_kernel = _odd_kernel(smoothing_kernel)
    voiced = f0 > 0
    log2_f0 = torch.where(
        voiced,
        torch.log2(torch.clamp(f0, min=_EPS)),
        torch.zeros_like(f0),
    )
    smooth_log2_f0 = _masked_local_mean(
        log2_f0, voiced, smoothing_kernel
    )
    if pitch_quantization_cents > 0:
        cents = smooth_log2_f0 * 1200.0
        cents = (
            torch.round(cents / pitch_quantization_cents)
            * pitch_quantization_cents
        )
        smooth_log2_f0 = cents / 1200.0
    f0_robot = torch.where(
        voiced,
        torch.pow(2.0, smooth_log2_f0),
        torch.zeros_like(f0),
    )

    volume_db = 20.0 * torch.log10(torch.clamp(volume, min=_EPS))
    smooth_volume_db = _local_mean(volume_db, smoothing_kernel)
    if volume_quantization_db > 0:
        smooth_volume_db = (
            torch.round(smooth_volume_db / volume_quantization_db)
            * volume_quantization_db
        )
    volume_robot = torch.pow(
        10.0, smooth_volume_db / 20.0
    ).clamp_min(0.0)
    return f0_robot.to(f0.dtype), volume_robot.to(volume.dtype)


@dataclass(frozen=True)
class RealismLossWeights:
    pitch_contour: float = 1.0
    pitch_velocity: float = 0.35
    pitch_acceleration: float = 0.15
    energy_contour: float = 0.5
    energy_velocity: float = 0.2


def realism_control_loss(
    pred_f0: torch.Tensor,
    pred_volume: torch.Tensor,
    target_f0: torch.Tensor,
    target_volume: torch.Tensor,
    trend_kernel: int = 31,
    weights: RealismLossWeights = RealismLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable singer-weak humanization loss.

    The loss never sees waveform, mel, speaker embeddings, formants, or a free
    spectral residual. It compares only locally normalized control dynamics.
    """
    pred = extract_realism_features(
        pred_f0, pred_volume, trend_kernel=trend_kernel
    )
    target = extract_realism_features(
        target_f0, target_volume, trend_kernel=trend_kernel
    )

    voiced = pred["voiced"] & target["voiced"]
    pitch_velocity_mask = (
        pred["pitch_velocity_mask"] & target["pitch_velocity_mask"]
    )
    pitch_acceleration_mask = (
        pred["pitch_acceleration_mask"]
        & target["pitch_acceleration_mask"]
    )

    pitch_contour = _masked_smooth_l1(
        pred["pitch_residual_cents"],
        target["pitch_residual_cents"],
        voiced,
        beta=5.0,
    )
    pitch_velocity = _masked_smooth_l1(
        pred["pitch_velocity_cents"],
        target["pitch_velocity_cents"],
        pitch_velocity_mask,
        beta=5.0,
    )
    pitch_acceleration = _masked_smooth_l1(
        pred["pitch_acceleration_cents"],
        target["pitch_acceleration_cents"],
        pitch_acceleration_mask,
        beta=5.0,
    )
    energy_contour = F.smooth_l1_loss(
        pred["energy_residual_db"],
        target["energy_residual_db"],
        beta=0.5,
    )
    energy_velocity = F.smooth_l1_loss(
        pred["energy_velocity_db"],
        target["energy_velocity_db"],
        beta=0.5,
    )

    total = (
        weights.pitch_contour * pitch_contour
        + weights.pitch_velocity * pitch_velocity
        + weights.pitch_acceleration * pitch_acceleration
        + weights.energy_contour * energy_contour
        + weights.energy_velocity * energy_velocity
    )
    parts = {
        "pitch_contour": pitch_contour,
        "pitch_velocity": pitch_velocity,
        "pitch_acceleration": pitch_acceleration,
        "energy_contour": energy_contour,
        "energy_velocity": energy_velocity,
        "total": total,
    }
    return total, parts
