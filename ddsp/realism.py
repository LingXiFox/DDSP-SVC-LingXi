from typing import Dict, Tuple

import torch
import torch.nn as nn


class VocalRealismAdapter(nn.Module):
    """Small residual controller for non-timbre vocal realism cues.

    R0 deliberately limits the adapter to bounded F0 and loudness residuals.
    The final projection is zero-initialized, so enabling the module at step 0 is
    an exact identity transform. Future realism controls (vibrato, aperiodicity,
    attack/release, timing) can be added without changing the public API.
    """

    def __init__(
        self,
        n_unit: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        max_f0_cents: float = 35.0,
        max_volume_db: float = 3.0,
    ):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if max_f0_cents < 0 or max_volume_db < 0:
            raise ValueError(
                "realism residual bounds must be non-negative"
            )

        self.max_f0_cents = float(max_f0_cents)
        self.max_volume_db = float(max_volume_db)

        self.unit_norm = nn.LayerNorm(n_unit)
        self.input_proj = nn.Linear(
            n_unit + 2,
            hidden_channels,
        )
        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(
            hidden_channels,
            2,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        units: torch.Tensor,
        f0: torch.Tensor,
        volume: torch.Tensor,
        strength: float = 1.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        if (
            units.ndim != 3
            or f0.ndim != 3
            or volume.ndim != 3
        ):
            raise ValueError(
                "units, f0 and volume must be "
                "[B, T, C] tensors"
            )
        if (
            units.shape[:2] != f0.shape[:2]
            or units.shape[:2] != volume.shape[:2]
        ):
            raise ValueError(
                "units, f0 and volume must share "
                "batch/frame dimensions"
            )
        if (
            f0.shape[-1] != 1
            or volume.shape[-1] != 1
        ):
            raise ValueError(
                "f0 and volume must have a "
                "singleton feature dimension"
            )

        # Speaker-weak control inputs. Remove per-utterance pitch register and
        # loudness level before predicting residual performance dynamics.
        voiced = f0 > 0
        log_f0 = torch.where(
            voiced,
            torch.log2(
                torch.clamp(
                    f0,
                    min=1e-5,
                )
            ),
            torch.zeros_like(f0),
        )
        voiced_f = voiced.to(log_f0.dtype)
        f0_center = (
            (log_f0 * voiced_f)
            .sum(
                dim=1,
                keepdim=True,
            )
            / voiced_f
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp_min(1.0)
        )
        relative_log_f0 = torch.where(
            voiced,
            log_f0 - f0_center,
            torch.zeros_like(log_f0),
        )

        log_volume = torch.log1p(
            torch.clamp(
                volume,
                min=0.0,
            )
        )
        relative_log_volume = (
            log_volume
            - log_volume.mean(
                dim=1,
                keepdim=True,
            )
        )
        x = torch.cat(
            (
                self.unit_norm(units),
                relative_log_f0,
                relative_log_volume,
            ),
            dim=-1,
        )

        h = torch.nn.functional.silu(
            self.input_proj(x)
        ).transpose(1, 2)
        for block in self.temporal:
            h = h + block(h)
        raw = self.output_proj(
            h.transpose(1, 2)
        )

        strength_t = torch.as_tensor(
            strength,
            dtype=raw.dtype,
            device=raw.device,
        )
        delta_cents = (
            torch.tanh(
                raw[..., 0:1]
            )
            * self.max_f0_cents
            * strength_t
        )
        delta_volume_db = (
            torch.tanh(
                raw[..., 1:2]
            )
            * self.max_volume_db
            * strength_t
        )

        pitch_ratio = torch.pow(
            2.0,
            delta_cents / 1200.0,
        )
        f0_out = torch.where(
            voiced,
            f0 * pitch_ratio,
            f0,
        )
        volume_gain = torch.pow(
            10.0,
            delta_volume_db / 20.0,
        )
        volume_out = torch.clamp(
            volume * volume_gain,
            min=0.0,
        )

        diagnostics = {
            "delta_f0_cents": delta_cents,
            "delta_volume_db": delta_volume_db,
        }
        return (
            f0_out,
            volume_out,
            diagnostics,
        )
