"""Offline ECAPA checks for the new Stage-3 partners, at 3- and 11-clip gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from scripts.eval_stage2_similarity import MODEL_FILES, select

SCREEN = Path('samples/stage3/new_partner_screen')
HOLDOUT = Path('samples/stage3/new_partner_11')
MARGIN = .01


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(phase):
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    root = SCREEN if phase == 'screen' else HOLDOUT
    out = Path(f'reports/timbre_blend_stage3_new_partner_{phase if phase == "screen" else "11"}.json')
    assert not out.exists()
    plan = json.loads((root / 'plan.json').read_text())
    corrected = json.loads(Path('reports/timbre_blend_stage3_corrected_standard.json').read_text())
    diagnostic = json.loads(Path('reports/timbre_blend_stage3_diagnostics.json').read_text())
    assert plan['checkpoint_md5'] == '733d9c6d44071d91a1c0b0f190db44db'
    assert plan['protocol']['vocoder_cudnn'] is False
    assert len(plan['inputs']) == (3 if phase == 'screen' else 11)
    assert len(plan['recipes']) >= 1 and plan['recipes'][0]['id'] == 'v100_t000'
    assert phase != 'screen' or tuple(plan['selected_public_spk_ids']) == (4, 8, 11)
    original = select(Path('data/timbre_blend_stage2/train/audio'))
    public = {name: songs for name, songs in original.items() if name != '13_lingxi'}
    names = {i: next(name for name in public if name.startswith(f'{i}_')) for i in range(1, 13)}
    assert len(public) == len(set(names.values())) == 12
    private = select(Path('data/timbre_blend_stage2/val/audio'))['13_lingxi']
    assert len(private) == 2 and all(len(files) == 2 for _, files in private)
    root_model = Path.home() / 'work/eval-models/ecapa-voxceleb-0f99f2d'
    for name, digest in MODEL_FILES.items():
        assert sha(root_model / name) == digest
    torch.set_num_threads(2)
    classifier = SpeakerRecognition.from_hparams(source=str(root_model),
        hparams_file='hyperparams_eval.yaml', savedir=str(root_model / 'runtime'))

    @torch.inference_mode()
    def embed(path):
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        a = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        a = torchaudio.functional.resample(a, sr, 16000)
        return torch.nn.functional.normalize(classifier.encode_batch(a).reshape(-1), dim=0)

    refs = {i: [embed(wav) for _, files in public[names[i]] for wav in files]
            for i in range(1, 13)}
    virtual_refs = [embed(wav) for _, files in private for wav in files]
    assert all(len(v) == 4 for v in refs.values()) and len(virtual_refs) == 4
    floor = corrected['standard']['floor']
    inflation = corrected['calibration']['signed_model_inflation']
    native = diagnostic['native_virtual']['by_public_spk']
    assert floor == .4137 and len(native) == 12
    results = []
    for recipe in plan['recipes']:
        embedded = []
        hashes = {}
        for row in plan['inputs']:
            name = row['band'] + '.wav' if phase == 'screen' else row['input']
            path = root / recipe['id'] / name
            info = sf.info(path)
            expected_duration = row['duration_seconds'] if phase == 'screen' else row['seconds']
            assert info.samplerate == 44100 and info.channels == 1
            assert abs(info.duration - expected_duration) < .05
            hashes[name] = sha(path)
            embedded.append(embed(path))
        virtual_per_input = {row['input']: float(np.mean([float(torch.dot(a, ref)) for ref in virtual_refs]))
                             for row, a in zip(plan['inputs'], embedded)}
        virtual_mean = float(np.mean(list(virtual_per_input.values())))
        scores, failures_a, failures_b = {}, [], []
        for i in range(1, 13):
            per_input = {row['input']: float(np.mean([float(torch.dot(a, ref)) for ref in refs[i]]))
                         for row, a in zip(plan['inputs'], embedded)}
            score = float(np.mean(list(per_input.values())))
            ceiling = max(native[str(i)]['mean'], floor) + inflation
            gap = ceiling - score
            a_passes = gap >= MARGIN
            b_passes = virtual_mean > score
            scores[str(i)] = {'opensinger_singer': int(names[i].split('singer')[1]),
                              'mean': score, 'per_input': per_input,
                              'a_ceiling_inclusive': ceiling, 'a_gap': gap,
                              'a_passes': a_passes, 'b_passes': b_passes}
            if not a_passes:
                failures_a.append(i)
            if not b_passes:
                failures_b.append(i)
        results.append({'recipe': recipe, 'output_sha256': hashes, 'virtual_mean': virtual_mean,
                        'virtual_per_input': virtual_per_input, 'by_public_spk': scores,
                        'minimum_a_gap': min(s['a_gap'] for s in scores.values()),
                        'maximum_public_mean': max(s['mean'] for s in scores.values()),
                        'a_failed_ids': failures_a, 'b_failed_ids': failures_b,
                        'passes': not failures_a and not failures_b})
        print('EVAL',phase,recipe['id'],'PASS' if results[-1]['passes'] else 'FAIL',
              'min_A_gap',round(results[-1]['minimum_a_gap'],6),
              'virtual',round(virtual_mean,6),'A',failures_a,'B',failures_b,flush=True)
    report = {'stage': '3_new_partner_'+phase, 'checkpoint_md5': plan['checkpoint_md5'],
              'selected_public_spk_ids': [4, 8, 11], 'inputs': plan['inputs'],
              'comparison_count_per_public_singer': len(plan['inputs'])*4,
              'threshold': {'floor': floor, 'signed_model_inflation': inflation,
                            'condition_a_safety_margin_inclusive': MARGIN,
                            'condition_b': 'virtual mean strictly greater than each of all twelve public means'},
              'reference_protocol': 'Same Stage-3 corrected reference choice, offline CPU ECAPA, four clips per singer. Every output clip gets equal weight.',
              'reference_model_file_sha256': MODEL_FILES,
              'results': results,
              'passing_recipe_ids': [row['recipe']['id'] for row in results if row['passes']],
              'failed_recipe_ids': [row['recipe']['id'] for row in results if not row['passes']],
              'caveat': 'Speech-trained ECAPA on singing is a similarity proxy, not a listening or identity guarantee.'}
    if phase == 'screen':
        baseline = next(row for row in corrected['results'] if row['recipe_id'] == 'v100_t000')
        actual = results[0]
        assert abs(actual['virtual_mean'] - baseline['virtual_mean']) < 1e-5
        assert all(abs(actual['by_public_spk'][str(i)]['mean'] - baseline['by_public_spk'][str(i)]['public_mean']) < 1e-5
                   for i in range(1, 13))
    else:
        screening = json.loads(Path('reports/timbre_blend_stage3_new_partner_screen.json').read_text())
        assert {r['recipe']['id'] for r in results} == set(screening['passing_recipe_ids'])
        report['three_clip_screen_passing_recipe_ids'] = screening['passing_recipe_ids']
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('EVAL_PASS', phase, 'screened_in', report['passing_recipe_ids'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('screen', 'holdout'))
    main(ap.parse_args().phase)
