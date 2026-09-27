import os
import time
import json
import random
import numpy as np
import torch
import librosa
from logger.saver import Saver
from logger import utils
from torch.amp import autocast, GradScaler

def build_reflow_mask(spk_id, reflow_exclude_spk):
    '''
    Build the [B] bool reflow participation mask for a batch.

    True  = sample takes part in the reflow loss
    False = speaker id listed in reflow_exclude_spk (1-based ids)

    Returns None when reflow_exclude_spk is empty so callers take the
    legacy unmasked code path unchanged (backward compatibility).
    spk_id may be [B] or [B, 1] (data loader layout).
    '''
    if not reflow_exclude_spk:
        return None
    excluded = torch.tensor(
        sorted({int(s) for s in reflow_exclude_spk}),
        device=spk_id.device,
        dtype=spk_id.dtype)
    return ~torch.isin(spk_id.reshape(-1), excluded)

def calculate_mel_snr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SNR
    snr = 10 * torch.log10(mean_square_reference / variance_error)
    return snr


def calculate_mel_si_snr(gt_mel, pred_mel):
    # 将测试图像按比例调整以最小化误差
    scale = torch.sum(gt_mel * pred_mel) / torch.sum(gt_mel ** 2)
    test_image_scaled = scale * gt_mel
    # 计算误差图像
    error_image = pred_mel - test_image_scaled 
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SI-SNR
    si_snr = 10 * torch.log10(mean_square_reference / variance_error)
    return si_snr


def calculate_mel_psnr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算误差图像的均方误差
    mse = torch.mean(error_image ** 2)
    # 计算参考图像的最大可能功率
    max_power = torch.max(gt_mel) ** 2
    # 计算并返回PSNR
    psnr = 10 * torch.log10(max_power / mse)
    return psnr

def _test_impl(args, model, vocoder, loader_test, saver, metric_prefix):
    print(f' [*] testing {metric_prefix}...')
    model.eval()

    # losses
    test_ddsp_loss = 0.
    test_reflow_loss = 0.

    # reflow accounting under train.reflow_exclude_spk:
    #   included  - official masked metric, same rule as training; equals the
    #               legacy validation/reflow_loss meaning when nothing is excluded
    #   excluded  - diagnostic only, computed under no_grad on excluded-speaker
    #               batches; never part of any optimization objective
    reflow_exclude_spk = list(args.train.get('reflow_exclude_spk') or [])
    test_reflow_loss_excluded = 0.
    num_reflow_included_batches = 0
    num_reflow_excluded_batches = 0

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    # intialization
    num_batches = len(loader_test)
    # validation cost controls (optional train-config keys; absent keeps the
    # legacy full-pass behavior):
    #   val_max_batches: process only the first N validation batches
    #     (deterministic subset; the val loader is shuffle=False).
    #   val_log_samples: cap the heavy per-sample tensorboard media
    #     (spectrogram figure + gt/pred audio via librosa) to the first K.
    val_max_batches = int(args.train.get('val_max_batches') or 0)
    val_log_samples = args.train.get('val_log_samples')
    if val_log_samples is not None:
        val_log_samples = int(val_log_samples)
    processed_batches = 0
    rtf_all = []
    spec_min = -6
    spec_max = 6
    spec_range = 12
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            if val_max_batches > 0 and bidx >= val_max_batches:
                break
            processed_batches += 1
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method,
                    t_start=args.model.t_start)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss (same mask rule as training; see build_reflow_mask)
            reflow_mask = build_reflow_mask(data['spk_id'], reflow_exclude_spk)
            ddsp_loss, reflow_loss = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False,
                t_start=args.model.t_start,
                reflow_mask=reflow_mask)
            test_ddsp_loss += ddsp_loss.item()
            if reflow_mask is None or bool(reflow_mask.any()):
                test_reflow_loss += reflow_loss.item()
                num_reflow_included_batches += 1
            if reflow_mask is not None and bool((~reflow_mask).any()):
                # diagnostic reflow loss on the excluded samples of this batch,
                # under the enclosing torch.no_grad(); never optimized/backprop'd
                _, reflow_loss_excluded = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    gt_spec=data['mel'],
                    infer=False,
                    t_start=args.model.t_start,
                    reflow_mask=~reflow_mask)
                test_reflow_loss_excluded += reflow_loss_excluded.item()
                num_reflow_excluded_batches += 1
            
            # log mel + audio (heavy: matplotlib figure, librosa decode,
            # tensorboard media) - capped by train.val_log_samples
            if val_log_samples is None or bidx < val_log_samples:
                saver.log_spec(metric_prefix + '/' + data['name'][0], data['mel'], mel)
                path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
                audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
                saver.log_audio({metric_prefix+'/'+fn+'/gt.wav': audio, metric_prefix+'/'+fn+'/pred.wav': signal})

            # 计算指标
            mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
            gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
            gt_mel_norm = gt_mel_norm / spec_range + spec_min
            pre_mel_norm = torch.clip(mel, spec_min, spec_max)
            pre_mel_norm = pre_mel_norm / spec_range + spec_min
            mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
            mel_val_mse_all_num += 1
            
    # report
    test_ddsp_loss /= processed_batches
    if num_reflow_included_batches > 0:
        test_reflow_loss /= num_reflow_included_batches
    else:
        # every validation batch was excluded from the reflow loss
        test_reflow_loss = 0.
        print(' [!] reflow_exclude_spk covers all validation batches; '
              'validation/reflow_loss reported as 0.0')
    if num_reflow_excluded_batches > 0:
        test_reflow_loss_excluded /= num_reflow_excluded_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    # check
    print(' [test_ddsp_loss] test_ddsp_loss:', test_ddsp_loss)
    print(' [test_reflow_loss] test_reflow_loss:', test_reflow_loss)
    if reflow_exclude_spk:
        # split metrics (plan 15.1): included is the official masked metric;
        # excluded is diagnostic-only and must never enter the objective
        print(' [test_reflow_loss_included]', test_reflow_loss)
        saver.log_value({
            f'{metric_prefix}/reflow_loss_included': test_reflow_loss
        })
        if num_reflow_excluded_batches > 0:
            print(' [test_reflow_loss_excluded_diagnostic]', test_reflow_loss_excluded)
            saver.log_value({
                f'{metric_prefix}/reflow_loss_excluded_diagnostic': test_reflow_loss_excluded
            })
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        f'{metric_prefix}/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        f'{metric_prefix}/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        f'{metric_prefix}/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        f'{metric_prefix}/mel_val_sisnr': mel_val_sisnr_all
    })
    return {
        'ddsp_loss': test_ddsp_loss,
        'reflow_loss': test_reflow_loss,
        'reflow_loss_excluded_diagnostic': test_reflow_loss_excluded,
        'mel_val_mse': float(mel_val_mse_all),
        'mel_val_snr': float(mel_val_snr_all),
        'mel_val_psnr': float(mel_val_psnr_all),
        'mel_val_sisnr': float(mel_val_sisnr_all),
    }


def test(args, model, vocoder, loader_test, saver,
         metric_prefix='validation', return_metrics=False):
    seed = args.train.get('val_seed')
    if seed is None:
        metrics = _test_impl(args, model, vocoder, loader_test, saver, metric_prefix)
    else:
        # Fixed validation RNG; restore training RNG so testing does not alter
        # the sampler / augmentation stream. Each curve gets the same seed.
        py_state, np_state = random.getstate(), np.random.get_state()
        devices = [torch.cuda.current_device()] if args.device == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(int(seed))
                np.random.seed(int(seed))
                torch.manual_seed(int(seed))
                metrics = _test_impl(args, model, vocoder, loader_test, saver,
                                     metric_prefix)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
    if return_metrics:
        return metrics
    return metrics['ddsp_loss'], metrics['reflow_loss']


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # reflow exclusion config (plan 15); empty -> legacy unmasked path (None mask)
    reflow_exclude_spk = list(args.train.get('reflow_exclude_spk') or [])
    if reflow_exclude_spk:
        saver.log_info(
            f' > reflow_exclude_spk: {reflow_exclude_spk} '
            '(masked out of the reflow loss in both training and validation)')

    # Stage-2 validation: fixed, separately logged public/virtual curves.
    # On resume, reuse the original baseline and history, never recalibrate
    # a threshold after seeing training results.
    stage2 = isinstance(loader_test, dict)
    if stage2:
        baseline_path = os.path.join(args.env.expdir, 'baseline.json')
        history_path = os.path.join(args.env.expdir, 'validation_history.jsonl')
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            saver.log_info(' [validation] baseline restored from ' + baseline_path)
        else:
            baseline = {
                group: test(args, model, vocoder, loader, saver,
                            f'validation/{group}', return_metrics=True)
                for group, loader in loader_test.items()
            }
            with open(baseline_path, 'w') as f:
                json.dump(baseline, f, indent=2)
            saver.log_info(' [validation] step-zero baseline: ' + json.dumps(baseline))
            for group, metrics in baseline.items():
                saver.log_value({f'validation/{group}/{k}': v
                                 for k, v in metrics.items()})
            model.train()
        history = []
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = [json.loads(line) for line in f if line.strip()]
        virtual_base = baseline['virtual']['ddsp_loss']
        virtual_best = min([virtual_base] + [r['virtual']['ddsp_loss'] for r in history])
        best_step = next((r['step'] for r in history
                          if r['virtual']['ddsp_loss'] == virtual_best), 0)
        prev_virtual = history[-1]['virtual']['ddsp_loss'] if history else virtual_base
        # Count only the trailing consecutive upward comparisons.
        values = [virtual_base] + [r['virtual']['ddsp_loss'] for r in history]
        rises = 0
        for before, after in zip(values[-2::-1], values[:0:-1]):
            if after <= before:
                break
            rises += 1
        virtual_seen_drop = virtual_best < virtual_base
        n_virtual, n_seen = 0, 0
        saver.log_info(f' [validation] public forgetting threshold: '
                       f"{baseline['public']['ddsp_loss'] * 1.1:.8f} "
                       '(step-zero public ddsp x 1.10)')
        if args.train.get('baseline_only'):
            saver.log_info('STAGE2_BASELINE_ONLY_DONE')
            return

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            if stage2:
                n_virtual += int((data['spk_id'] == int(args.train.virtual_spk_id)).sum())
                n_seen += data['spk_id'].numel()

            # forward
            reflow_mask = build_reflow_mask(data['spk_id'], reflow_exclude_spk)
            if dtype == torch.float32:
                ddsp_loss, reflow_loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start,
                                reflow_mask=reflow_mask)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    ddsp_loss, reflow_loss=model(data['units'], data['f0'], data['volume'], data['spk_id'], 
                                    aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False, t_start=args.model.t_start,
                                    reflow_mask=reflow_mask)
            
            # handle nan loss
            if torch.isnan(ddsp_loss):
                print(' [x] nan ddsp_loss ')
                optimizer.zero_grad()
                del ddsp_loss
                del reflow_loss
                continue
            elif torch.isnan(reflow_loss):
                raise ValueError(' [x] nan reflow_loss ')
            else:
                loss = args.train.lambda_ddsp * ddsp_loss + reflow_loss
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/ddsp_loss': ddsp_loss.item(),
                    'train/reflow_loss': reflow_loss.item(),
                    'train/lr': current_lr
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                if stage2:
                    metrics = {
                        group: test(args, model, vocoder, loader, saver,
                                    f'validation/{group}', return_metrics=True)
                        for group, loader in loader_test.items()
                    }
                    ratio = n_virtual / n_seen if n_seen else 0
                    row = {'step': saver.global_step, **metrics,
                           'actual_virtual_fraction': ratio}
                    with open(history_path, 'a') as f:
                        f.write(json.dumps(row) + '\n')
                    saver.log_info(' --- <validation> --- ' + json.dumps(row))
                    saver.log_value({
                        f'validation/{group}/{key}': val
                        for group, result in metrics.items()
                        for key, val in result.items()
                    } | {'train/actual_virtual_fraction': ratio})
                    n_virtual, n_seen = 0, 0
                    pub = metrics['public']['ddsp_loss']
                    virt = metrics['virtual']['ddsp_loss']
                    if virt < virtual_best:
                        virtual_best, best_step = virt, saver.global_step
                    virtual_seen_drop |= virt < virtual_base
                    rises = rises + 1 if virt > prev_virtual else 0
                    prev_virtual = virt
                    saver.log_info(f' [validation] virtual best: step={best_step} '
                                   f'loss={virtual_best:.8f}; consecutive rises={rises}')
                    if pub > baseline['public']['ddsp_loss'] * 1.1:
                        saver.log_info('STAGE2_STOP_REASON=PUBLIC_FORGETTING '
                                       f'public_ddsp={pub:.8f}; '
                                       f"baseline={baseline['public']['ddsp_loss']:.8f}")
                        return
                    if virtual_seen_drop and rises >= 2:
                        saver.log_info('STAGE2_STOP_REASON=VIRTUAL_TWO_RISES '
                                       f'best_step={best_step} best_loss={virtual_best:.8f}')
                        return
                    if saver.global_step >= int(args.train.get('max_steps') or 10000):
                        saver.log_info('STAGE2_STOP_REASON=MAX_STEPS '
                                       f'best_step={best_step} best_loss={virtual_best:.8f}')
                        return
                else:
                    test_ddsp_loss, test_reflow_loss = test(args, model, vocoder, loader_test, saver)
                    test_loss = args.train.lambda_ddsp * test_ddsp_loss + test_reflow_loss
                    saver.log_info(
                        ' --- <validation> --- \nloss: {:.3f}. '.format(test_loss)
                    )
                    saver.log_value({
                        'validation/loss': test_loss,
                        'validation/ddsp_loss': test_ddsp_loss,
                        'validation/reflow_loss': test_reflow_loss
                    })
                model.train()
