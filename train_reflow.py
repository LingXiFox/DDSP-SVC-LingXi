import os
import argparse
import torch
import numpy as np
import random
from torch.optim import lr_scheduler
from optimizer.muon import Muon_AdamW
from logger import utils
from reflow.data_loaders import get_data_loaders
from reflow.vocoder import Vocoder, Unit2Wav
from ddsp.realism_stages import (
    configure_training_stage,
    split_realism_parameters,
)


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


def apply_freeze_reflow(model):
    """Freeze every reflow_model parameter (train.freeze_reflow, plan 16).

    Must be called AFTER configure_training_stage, which re-enables
    requires_grad on all backbone params (reflow included) for stage 'timbre'.
    Returns (frozen_tensors, frozen_params).
    """
    frozen_tensors = 0
    frozen_params = 0
    for p in model.reflow_model.parameters():
        p.requires_grad = False
        frozen_tensors += 1
        frozen_params += p.numel()
    return frozen_tensors, frozen_params


def build_muon_adamw_optimizer(model, args, freeze_reflow=False):
    """Default (timbre stage) Muon+AdamW optimizer.

    With freeze_reflow, params whose requires_grad=False are kept out of the
    optimizer entirely (plan 16). Without it, params=None preserves the
    legacy layout exactly (all model parameters enter the optimizer).
    """
    return Muon_AdamW(
        model,
        lr=args.train.lr,
        muon_args={
            'weight_decay': args.train.weight_decay
        },
        adamw_args={'weight_decay': 0},
        params=(
            [p for p in model.parameters() if p.requires_grad]
            if freeze_reflow else None
        ),
    )


if __name__ == '__main__':
    cmd = parse_args()

    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    if args.train.get('seed') is not None:
        seed = int(args.train.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(' > training seed:', seed)

    vocoder = Vocoder(
        args.vocoder.type,
        args.vocoder.ckpt,
        device=args.device,
    )

    if args.model.type == 'RectifiedFlow':
        from reflow.solver import train
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
                    realism_config=args.model.realism)

    else:
        raise ValueError(
            f" [x] Unknown Model: {args.model.type}"
        )

    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)

    # R2 staged optimization. Public prior training intentionally uses
    # train_realism.py so public data cannot touch waveform/mel timbre losses.
    stage = args.train.get('realism_stage', 'timbre')
    stage_info = configure_training_stage(model, stage)
    print(' > realism stage:', stage, stage_info)

    # freeze_reflow (timbre-blend Stage 2): freeze all RectifiedFlow weights.
    # Must run AFTER configure_training_stage, which re-enables requires_grad
    # on every backbone param (reflow included) for stage 'timbre'.
    # Semantics: forward still runs and the reflow loss still backprops INTO
    # the DDSP backbone (keeping ddsp_mel compatible with the frozen reflow),
    # but no gradient accumulates on reflow params and they are kept out of
    # the optimizer, so no reflow weight update can occur.
    freeze_reflow = bool(args.train.get('freeze_reflow', False))
    if freeze_reflow:
        frozen_tensors, frozen_params = apply_freeze_reflow(model)
        print(f' > freeze_reflow: {frozen_tensors} reflow tensors '
              f'({frozen_params} params) set to requires_grad=False')

    if stage == 'joint':
        backbone_params, realism_params = split_realism_parameters(model)
        optimizer = torch.optim.AdamW([
            {
                'params': [
                    p for p in backbone_params if p.requires_grad
                ],
                'lr': float(
                    args.train.get(
                        'backbone_lr',
                        args.train.lr,
                    )
                ),
                'weight_decay': args.train.weight_decay,
            },
            {
                'params': [
                    p for p in realism_params if p.requires_grad
                ],
                'lr': float(
                    args.train.get(
                        'realism_lr',
                        args.train.lr,
                    )
                ),
                'weight_decay': 0.0,
            },
        ])
    elif stage == 'realism_personalize':
        _, realism_params = split_realism_parameters(model)
        optimizer = torch.optim.AdamW(
            [p for p in realism_params if p.requires_grad],
            lr=float(
                args.train.get(
                    'realism_lr',
                    args.train.lr,
                )
            ),
            weight_decay=0.0,
        )
    else:
        optimizer = build_muon_adamw_optimizer(
            model, args, freeze_reflow=freeze_reflow)

    resume_optimizer = bool(
        args.train.get('resume_optimizer', True)
    )
    initial_global_step, model, optimizer = utils.load_model(
        args.env.expdir,
        model,
        optimizer,
        device=args.device,
        load_optimizer=resume_optimizer,
    )

    decay_power = max(
        (initial_global_step - 2)
        // args.train.decay_step,
        0,
    )
    for param_group in optimizer.param_groups:
        base_lr = param_group.get(
            'initial_lr',
            param_group['lr'],
        )
        param_group['initial_lr'] = base_lr
        param_group['lr'] = (
            base_lr
            * args.train.gamma ** decay_power
        )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=args.train.decay_step,
        gamma=args.train.gamma,
        last_epoch=initial_global_step-2,
    )

    loader_train, loader_valid = get_data_loaders(
        args,
        whole_audio=False,
    )

    if args.train.get('virtual_spk_id') is not None:
        from torch.utils.data import DataLoader, Subset
        target = str(int(args.train.virtual_spk_id))
        paths = loader_valid.dataset.paths
        is_virtual = [p.split(os.sep, 1)[0].split('_', 1)[0] == target
                      for p in paths]
        indices = {
            'public': [i for i, flag in enumerate(is_virtual) if not flag],
            'virtual': [i for i, flag in enumerate(is_virtual) if flag],
        }
        if not all(indices.values()):
            raise ValueError('both public and virtual validation sets are required')
        loader_valid = {
            key: DataLoader(Subset(loader_valid.dataset, idx), batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=False)
            for key, idx in indices.items()
        }
        print(' > validation slices:', {key: len(idx) for key, idx in indices.items()})

    train(
        args,
        initial_global_step,
        model,
        optimizer,
        scheduler,
        vocoder,
        loader_train,
        loader_valid,
    )
