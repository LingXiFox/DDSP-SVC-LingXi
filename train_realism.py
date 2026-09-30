import argparse
import os
from pathlib import Path

import torch
import yaml

from ddsp.realism import VocalRealismAdapter
from ddsp.realism_features import (
    RealismLossWeights,
    realism_control_loss,
    robotize_controls,
)
from logger.utils import DotDict
from reflow.realism_data_loaders import get_realism_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the speaker-weak Vocal Realism prior without "
            "waveform/mel supervision."
        )
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    config_path,
    best_val=None,
    best_step=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config_path,
            "format": "lingxi-vocal-realism-v1",
            "best_val": best_val,
            "best_step": best_step,
        },
        path,
    )


def _load_checkpoint(path, model, optimizer, device):
    if not path or not os.path.exists(path):
        return 0, 0, None, None
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("global_step", 0)),
        ckpt.get("best_val"),
        ckpt.get("best_step"),
    )


def _weights(cfg):
    loss_cfg = cfg.loss
    return RealismLossWeights(
        pitch_contour=float(loss_cfg.pitch_contour),
        pitch_velocity=float(loss_cfg.pitch_velocity),
        pitch_acceleration=float(loss_cfg.pitch_acceleration),
        energy_contour=float(loss_cfg.energy_contour),
        energy_velocity=float(loss_cfg.energy_velocity),
    )


def _step(model, batch, cfg, device):
    units = batch["units"].to(device, non_blocking=True)
    target_f0 = batch["f0"].to(device, non_blocking=True)
    target_volume = batch["volume"].to(device, non_blocking=True)

    input_f0, input_volume = robotize_controls(
        target_f0,
        target_volume,
        smoothing_kernel=int(cfg.robotize.smoothing_kernel),
        pitch_quantization_cents=float(
            cfg.robotize.pitch_quantization_cents
        ),
        volume_quantization_db=float(
            cfg.robotize.volume_quantization_db
        ),
    )
    pred_f0, pred_volume, diagnostics = model(
        units,
        input_f0,
        input_volume,
        strength=1.0,
    )
    loss, parts = realism_control_loss(
        pred_f0,
        pred_volume,
        target_f0,
        target_volume,
        trend_kernel=int(cfg.trend_kernel),
        weights=_weights(cfg),
    )
    return loss, parts, diagnostics


def _validate(model, loader, cfg, device):
    model.eval()
    totals = {}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            _, parts, _ = _step(model, batch, cfg, device)
            for key, value in parts.items():
                totals[key] = (
                    totals.get(key, 0.0)
                    + float(value.detach().cpu())
                )
            batches += 1
    model.train()
    return {
        key: value / max(batches, 1)
        for key, value in totals.items()
    }


def main():
    cli = parse_args()
    with open(cli.config, "r") as f:
        args = DotDict(yaml.safe_load(f))
    cfg = args.realism_train
    device = cli.device or cfg.device

    model = VocalRealismAdapter(
        n_unit=int(args.data.encoder_out_channels),
        hidden_channels=int(args.model.realism.hidden_channels),
        num_layers=int(args.model.realism.num_layers),
        max_f0_cents=float(args.model.realism.max_f0_cents),
        max_volume_db=float(args.model.realism.max_volume_db),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    train_loader, valid_loader = get_realism_data_loaders(args)

    save_dir = Path(cfg.expdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_path = save_dir / "realism_latest.pt"
    best_path = save_dir / "realism_best.pt"
    start_epoch, global_step, best_val, best_step = _load_checkpoint(
        str(latest_path) if bool(cfg.resume) else "",
        model,
        optimizer,
        device,
    )
    if best_val is None:
        best_val, best_step = float("inf"), 0

    print("[Vocal Realism] control-only public prior training")
    print(
        f"  device={device} "
        f"train_items={len(train_loader.dataset)} "
        f"valid_items={len(valid_loader.dataset)}"
    )
    print("  safety=NO waveform/mel/timbre supervision")

    model.train()
    for epoch in range(start_epoch, int(cfg.epochs)):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts, diagnostics = _step(
                model, batch, cfg, device
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite realism loss at step "
                    f"{global_step}: {loss}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg.grad_clip_norm),
            )
            optimizer.step()
            global_step += 1

            if global_step % int(cfg.log_every) == 0:
                delta_f0 = (
                    diagnostics["delta_f0_cents"]
                    .detach()
                    .abs()
                    .mean()
                    .item()
                )
                delta_vol = (
                    diagnostics["delta_volume_db"]
                    .detach()
                    .abs()
                    .mean()
                    .item()
                )
                print(
                    f"step={global_step} "
                    f"loss={loss.item():.5f} "
                    f"pitch={parts['pitch_contour'].item():.5f} "
                    f"energy={parts['energy_contour'].item():.5f} "
                    f"|df0|={delta_f0:.3f}c "
                    f"|dvol|={delta_vol:.3f}dB "
                    f"grad={float(grad_norm):.3f}"
                )

            if global_step % int(cfg.validate_every) == 0:
                metrics = _validate(
                    model, valid_loader, cfg, device
                )
                print(
                    "validation",
                    " ".join(
                        f"{k}={v:.5f}"
                        for k, v in metrics.items()
                    ),
                )
                val_total = float(metrics["total"])
                if val_total < best_val:
                    best_val, best_step = val_total, global_step
                    _save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        epoch,
                        global_step,
                        cli.config,
                        best_val,
                        best_step,
                    )
                    print(
                        f"new best realism prior: "
                        f"step={global_step} val={val_total:.5f}"
                    )

            if global_step % int(cfg.save_every) == 0:
                _save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    cli.config,
                    best_val,
                    best_step,
                )

        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch + 1,
            global_step,
            cli.config,
            best_val,
            best_step,
        )

    final_path = save_dir / "realism_final.pt"
    _save_checkpoint(
        final_path,
        model,
        optimizer,
        int(cfg.epochs),
        global_step,
        cli.config,
        best_val,
        best_step,
    )
    print(f"saved {final_path}")
    print(f"best realism prior: step={best_step} val={best_val}")


if __name__ == "__main__":
    main()
