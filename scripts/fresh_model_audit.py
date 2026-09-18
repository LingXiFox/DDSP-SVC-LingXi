"""Fresh-model end-to-end audit (no historical checkpoint needed).

Validates on a randomly initialized full Unit2Wav + real NSF-HiFiGAN vocoder:
  init / infer forward / train forward+backward / optimizer step /
  full checkpoint save+strict reload / simulated-old-checkpoint compat /
  R2 stage freeze counts / joint LR groups / peak VRAM.

Usage:
  PYTHONPATH=<repo> python scripts/fresh_model_audit.py -c <reflow.yaml> \
      --sample-dir <preprocessed train dir> --device cuda
"""

import argparse
import os
import tempfile

import numpy as np
import torch
import yaml

from ddsp.realism_stages import configure_training_stage, split_realism_parameters
from logger.utils import DotDict
from reflow.vocoder import Unit2Wav, Vocoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_sample(sample_dir, device, frames=172):
    names = sorted(
        f for f in os.listdir(os.path.join(sample_dir, "audio"))
        if f.endswith(".wav")
    )
    if not names:
        raise RuntimeError(f"no wav files under {sample_dir}/audio")
    stem = os.path.splitext(names[0])[0]
    units = np.load(os.path.join(sample_dir, "units", stem + ".wav.npy"))
    f0 = np.load(os.path.join(sample_dir, "f0", stem + ".wav.npy"))
    volume = np.load(os.path.join(sample_dir, "volume", stem + ".wav.npy"))
    mel = np.load(os.path.join(sample_dir, "mel", stem + ".wav.npy"))
    units_t = torch.from_numpy(units[:frames]).float().unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0[:frames]).float().reshape(1, -1, 1).to(device)
    vol_t = torch.from_numpy(volume[:frames]).float().reshape(1, -1, 1).to(device)
    mel_t = torch.from_numpy(mel[:frames]).float().unsqueeze(0).to(device)
    spk = torch.zeros(1, 1, dtype=torch.long, device=device)
    return units_t, f0_t, vol_t, mel_t, spk


def _build_model(args, device, realism_config):
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    model = Unit2Wav(
        args.data.sampling_rate,
        args.data.block_size,
        args.model.win_length,
        args.data.encoder_out_channels,
        args.model.n_spk,
        args.model.use_norm,
        args.model.use_attention,
        args.model.use_pitch_aug,
        vocoder.dimension,
        args.model.n_aux_layers,
        args.model.n_aux_chans,
        args.model.n_layers,
        args.model.n_chans,
        realism_config=realism_config,
    ).to(device)
    return model, vocoder


def main():
    cli = parse_args()
    device = cli.device
    with open(cli.config) as f:
        args = DotDict(yaml.safe_load(f))

    torch.manual_seed(20260918)
    realism_on = {
        "enabled": True, "strength": 1.0, "hidden_channels": 128,
        "num_layers": 2, "max_f0_cents": 35.0, "max_volume_db": 3.0,
        "checkpoint": None,
    }
    model, vocoder = _build_model(args, device, realism_on)
    assert model.ddsp_model.realism is not None
    units, f0, volume, mel, spk = _load_sample(cli.sample_dir, device)

    # 1. infer forward with real vocoder (validates vocoder asset load)
    model.eval()
    with torch.no_grad():
        out = model(units, f0, volume, spk_id=spk, vocoder=vocoder,
                    infer=True, return_wav=True, infer_step=2)
    assert torch.isfinite(out).all(), "non-finite inference wav"
    print(f"infer-ok wav_shape={tuple(out.shape)}")

    # 2. train forward + backward + personalize optimizer step
    info = configure_training_stage(model, "realism_personalize")
    assert info["backbone_trainable"] == 0 and info["realism_trainable"] > 0
    print(f"stage personalize {info}")
    model.train()
    _, realism_params = split_realism_parameters(model)
    optimizer = torch.optim.AdamW(
        [p for p in realism_params if p.requires_grad], lr=3e-4)
    before = model.ddsp_model.realism.output_proj.weight.detach().clone()
    assert torch.count_nonzero(before) == 0, "output_proj not zero-init"
    ddsp_loss, reflow_loss = model(units, f0, volume, spk_id=spk,
                                   vocoder=vocoder, gt_spec=mel, infer=False,
                                   t_start=0.0)
    total = ddsp_loss + reflow_loss
    assert torch.isfinite(total), f"non-finite train loss {total}"
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    assert torch.isfinite(torch.as_tensor(float(grad_norm)))
    optimizer.step()
    after = model.ddsp_model.realism.output_proj.weight.detach()
    assert torch.count_nonzero(after) > 0, "output_proj did not update"
    # backbone must not move
    print(f"train-ok loss={float(total):.5f} grad={float(grad_norm):.3f}")

    # 3. full checkpoint save + strict reload round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "full_smoke.pt")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": 1}, path)
        torch.manual_seed(20260918)
        model2, vocoder2 = _build_model(args, device, realism_on)
        ckpt = torch.load(path, map_location=device)
        missing, unexpected = model2.load_state_dict(ckpt["model"], strict=True), None
        del missing
        optimizer2 = torch.optim.AdamW(
            [p for p in split_realism_parameters(model2)[1] if p.requires_grad],
            lr=3e-4,
        )
        optimizer2.load_state_dict(ckpt["optimizer"])
        assert any("ddsp_model.realism." in k for k in ckpt["model"]), \
            "full checkpoint lacks realism params"
        print("ckpt-strict-reload-ok "
              f"realism_keys={sum(1 for k in ckpt['model'] if 'realism' in k)}")

        # 4. simulated old checkpoint (no realism keys) -> strict=False, no crash
        stripped = {k: v for k, v in ckpt["model"].items()
                    if "ddsp_model.realism." not in k}
        torch.manual_seed(20260918)
        model3, _ = _build_model(args, device, realism_on)
        incompatible = model3.load_state_dict(stripped, strict=False)
        assert len(incompatible.unexpected_keys) == 0, \
            f"unexpected keys: {incompatible.unexpected_keys}"
        assert all("realism" in k for k in incompatible.missing_keys), \
            f"non-realism missing keys: {incompatible.missing_keys}"
        model3.eval()
        with torch.no_grad():
            f0_h, vol_h, diag = model3.ddsp_model.realism(units, f0, volume)
        assert torch.count_nonzero(diag["delta_f0_cents"]) == 0
        assert torch.count_nonzero(diag["delta_volume_db"]) == 0
        assert torch.equal(f0_h, f0) and torch.equal(vol_h, volume)
        print(f"old-ckpt-sim-ok missing_realism_keys={len(incompatible.missing_keys)}")

    # 5. R2 stage matrix on fresh model
    torch.manual_seed(20260918)
    model4, _ = _build_model(args, device, realism_on)
    timbre = configure_training_stage(model4, "timbre")
    assert timbre["backbone_trainable"] > 0 and timbre["realism_trainable"] == 0
    pers = configure_training_stage(model4, "realism_personalize")
    assert pers["backbone_trainable"] == 0 and pers["realism_trainable"] > 0
    joint = configure_training_stage(model4, "joint")
    assert joint["backbone_trainable"] > 0 and joint["realism_trainable"] > 0
    bb, rp = split_realism_parameters(model4)
    opt = torch.optim.AdamW([
        {"params": [p for p in bb if p.requires_grad],
         "lr": float(args.train.backbone_lr)},
        {"params": [p for p in rp if p.requires_grad],
         "lr": float(args.train.realism_lr)},
    ])
    lrs = [g["lr"] for g in opt.param_groups]
    assert len(lrs) == 2 and lrs[0] < lrs[1], lrs
    print(f"stages-ok timbre={timbre} personalize={pers} joint={joint} lrs={lrs}")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else 0.0
    print(f"FRESH MODEL AUDIT PASS device={device} peak_cuda_mb={peak_mb:.1f}")


if __name__ == "__main__":
    main()
