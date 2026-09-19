import os
import argparse
import torch
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


if __name__ == '__main__':
    cmd = parse_args()

    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

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
        optimizer = Muon_AdamW(
            model,
            lr=args.train.lr,
            muon_args={
                'weight_decay': args.train.weight_decay
            },
            adamw_args={'weight_decay': 0},
        )

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
