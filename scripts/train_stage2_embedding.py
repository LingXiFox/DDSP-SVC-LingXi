"""Stage 2 C: train only the virtual speaker embedding row."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from logger import utils
from logger.saver import Saver
from reflow.data_loaders import AudioDataset
from reflow.solver import build_reflow_mask, test
from reflow.vocoder import Unit2Wav, Vocoder

STAGE1_MD5 = 'e026eb6e60b7e2e8ba4c579ad8a0ceff'
KEY = utils.SPK_EMBED_KEY
FEATURES = ('audio', 'mel', 'f0', 'units', 'volume', 'aug_mel', 'aug_vol')


def digest(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(chunk)
    return md5.hexdigest()


def sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def virtual_tree(source, target):
    """Link only virtual speaker features; do not duplicate private audio."""
    source, target = Path(source), Path(target)
    assert not target.exists(), f'refusing existing virtual tree {target}'
    songs = set()
    for audio in sorted((source / 'audio' / '13_lingxi').glob('*.wav')):
        songs.add(audio.name.split('_')[0] + '_' + audio.name.split('_')[1])
        for kind in FEATURES:
            original = source / kind / '13_lingxi' / (audio.name + ('' if kind == 'audio' else '.npy'))
            dest = target / kind / '13_lingxi' / original.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(original, dest)
    assert len(songs) == 9 and len(list((target / 'audio' / '13_lingxi').glob('*.wav'))) == 138
    os.link(source / 'pitch_aug_dict.npy', target / 'pitch_aug_dict.npy')
    return target


def dataset(root, args, whole_audio=False, use_aug=False):
    return AudioDataset(str(root), waveform_sec=args.data.duration,
                        hop_size=args.data.block_size, sample_rate=args.data.sampling_rate,
                        load_all_data=True, whole_audio=whole_audio,
                        extensions=args.data.extensions, n_spk=args.model.n_spk,
                        device='cpu', fp16=not whole_audio and args.train.cache_fp16,
                        use_aug=use_aug)


def loaders(args, root):
    train_data = dataset(root, args, use_aug=True)
    assert len(train_data) == 138
    train = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True,
                       num_workers=args.train.num_workers, pin_memory=False)
    train_whole = dataset(root, args, whole_audio=True)
    chosen = sorted(random.Random(int(args.train.seed)).sample(range(len(train_whole)), 34))
    train_probe = DataLoader(Subset(train_whole, chosen), batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)
    val_data = dataset(args.data.valid_path, args, whole_audio=True)
    indices = {'public': [], 'virtual': []}
    for i, path in enumerate(val_data.paths):
        indices['virtual' if path.startswith('13_lingxi/') else 'public'].append(i)
    assert len(indices['virtual']) == 34 and len(indices['public']) == 558
    val = {group: DataLoader(Subset(val_data, ids), batch_size=1,
                             shuffle=False, num_workers=0, pin_memory=False)
           for group, ids in indices.items()}
    return train, train_probe, val, [train_whole.paths[i] for i in chosen]


def frozen_digests(model, row):
    return {key: sha(tensor if key != KEY else tensor[:row])
            for key, tensor in model.state_dict().items()}


def assert_frozen(model, expected, row):
    current = frozen_digests(model, row)
    changed = [key for key, value in current.items() if value != expected[key]]
    if changed:
        raise RuntimeError('FROZEN_WEIGHT_CHANGED: ' + ', '.join(changed))
    return current[KEY]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/timbre_blend_stage2_embedding.yaml')
    parser.add_argument('--smoke', action='store_true', help='one training step in separate scratch expdir')
    opt = parser.parse_args()
    args = utils.load_config(opt.config)
    assert args.model.n_spk == 13 and args.train.virtual_spk_id == 13
    assert args.train.reflow_exclude_spk == [13]
    assert args.train.lr == 1e-3 and args.train.interval_val == 200
    assert args.train.weight_decay == 0 and args.train.virtual_rise_threshold == 0.02
    assert args.train.max_steps == 3000 and args.train.val_seed is not None
    if opt.smoke:
        args['env']['expdir'] = '.tmp/stage2_c_smoke_exp'
        assert not Path(args.env.expdir).exists(), 'smoke expdir already exists'
    else:
        assert not Path(args.env.expdir).exists(), 'refusing to overwrite existing C experiment'
    assert digest(args.train.init_checkpoint) == STAGE1_MD5, 'Stage 1 MD5 mismatch'
    random.seed(args.train.seed)
    np.random.seed(args.train.seed)
    torch.manual_seed(args.train.seed)
    torch.cuda.set_device(args.env.gpu_id)
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    model = Unit2Wav(args.data.sampling_rate, args.data.block_size, args.model.win_length,
                     args.data.encoder_out_channels, args.model.n_spk,
                     args.model.use_norm, args.model.use_attention, args.model.use_pitch_aug,
                     vocoder.dimension, args.model.n_aux_layers, args.model.n_aux_chans,
                     args.model.n_layers, args.model.n_chans,
                     realism_config=args.model.realism).to(args.device)
    ckpt = torch.load(args.train.init_checkpoint, map_location='cpu', weights_only=True)
    assert ckpt['global_step'] == 20000 and ckpt['model'][KEY].shape[0] == 12
    model.load_state_dict(utils.expand_spk_embed_state(ckpt['model'], model), strict=True)
    weight = model.ddsp_model.unit2ctrl.spk_embed.weight
    row = int(args.train.virtual_spk_id) - 1
    with torch.no_grad():
        weight[row].copy_(weight[:row].mean(dim=0))
    print('EMBED_INIT=stage1_public_mean row=12 public_rows=12 stage1_md5=' + STAGE1_MD5, flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    weight.requires_grad_(True)
    mask = torch.zeros_like(weight)
    mask[row] = 1
    weight.register_hook(lambda grad: grad * mask)
    assert [name for name, p in model.named_parameters() if p.requires_grad] == [KEY]
    optimizer = torch.optim.AdamW([weight], lr=args.train.lr, weight_decay=0)
    assert optimizer.param_groups[0]['weight_decay'] == 0
    # Eval mode prevents frozen modules' running-stat buffers from changing.
    model.eval()
    expected = frozen_digests(model, row)
    assert_frozen(model, expected, row)
    virtual_root = virtual_tree(args.data.train_path, Path('.tmp/stage2_c_virtual_train_smoke' if opt.smoke else '.tmp/stage2_c_virtual_train'))
    loader, train_probe, val, selected = loaders(args, virtual_root)
    saver = Saver(args, initial_global_step=0)
    baseline = {group: test(args, model, vocoder, dl, saver, 'validation/' + group, True)
                for group, dl in {'train_probe': train_probe, **val}.items()}
    public_ref = json.loads(Path('exp/timbre_blend_stage2_masked/baseline.json').read_text())['public']
    assert baseline['public'] == public_ref, 'PUBLIC_BASELINE_MISMATCH'
    assert_frozen(model, expected, row)
    exp = Path(args.env.expdir)
    (exp / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
    (exp / 'train_probe_files.json').write_text(json.dumps(selected, indent=2) + '\n')
    saver.save_model(model, None, postfix='0')
    prev = baseline['virtual']['ddsp_loss']
    best_step, best_loss, rises = 0, prev, 0
    limit = 1 if opt.smoke else args.train.max_steps
    while saver.global_step < limit:
        for data in loader:
            optimizer.zero_grad(set_to_none=True)
            for key in data:
                if not key.startswith('name'):
                    data[key] = data[key].to(args.device)
            assert bool((data['spk_id'] == 13).all()), 'non-virtual training sample'
            reflow_mask = build_reflow_mask(data['spk_id'], args.train.reflow_exclude_spk)
            assert not bool(reflow_mask.any()), 'unexpected reflow inclusion'
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                ddsp_loss, reflow_loss = model(
                    data['units'], data['f0'], data['volume'], data['spk_id'],
                    aug_shift=data['aug_shift'], vocoder=vocoder,
                    gt_spec=data['mel'].float(), infer=False,
                    t_start=args.model.t_start, reflow_mask=reflow_mask)
            assert torch.isfinite(ddsp_loss).item() and reflow_loss.item() == 0.0
            ddsp_loss.backward()
            assert weight.grad is not None and torch.count_nonzero(weight.grad[:row]) == 0
            optimizer.step()
            saver.global_step_increment()
            step = saver.global_step
            if step % args.train.interval_log == 0:
                saver.log_info(f'C_TRAIN step={step} ddsp={ddsp_loss.item():.8f} lr={optimizer.param_groups[0]["lr"]:.7f}')
            if step % args.train.interval_val == 0 or opt.smoke:
                # Fail before writing the checkpoint if any non-target state moved.
                other_rows_sha = assert_frozen(model, expected, row)
                saver.save_model(model, None, postfix=str(step))
                metrics = {group: test(args, model, vocoder, dl, saver, 'validation/' + group, True)
                           for group, dl in {'train_probe': train_probe, **val}.items()}
                assert_frozen(model, expected, row)
                assert metrics['public'] == public_ref, 'PUBLIC_FORGETTING_OR_NONDETERMINISM'
                virt = metrics['virtual']['ddsp_loss']
                gap = virt - metrics['train_probe']['ddsp_loss']
                rises = rises + 1 if virt > prev * 1.02 else 0
                if virt < best_loss:
                    best_step, best_loss = step, virt
                row_data = {'step': step, **metrics, 'virtual_val_minus_train': gap,
                            'other_speaker_rows_sha256': other_rows_sha,
                            'consecutive_rises_gt_2pct': rises}
                with (exp / 'validation_history.jsonl').open('a') as f:
                    f.write(json.dumps(row_data) + '\n')
                saver.log_info('C_VALIDATION ' + json.dumps(row_data))
                prev = virt
                if rises >= 2:
                    saver.log_info(f'C_STOP=VIRTUAL_TWO_RISES best_step={best_step} best_loss={best_loss:.8f}')
                    return
            if step >= limit:
                saver.log_info(f'C_STOP={"SMOKE" if opt.smoke else "MAX_STEPS"} best_step={best_step} best_loss={best_loss:.8f}')
                return


if __name__ == '__main__':
    main()
