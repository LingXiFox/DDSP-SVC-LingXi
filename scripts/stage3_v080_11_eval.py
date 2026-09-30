"""Offline 11-clip check of candidate v080_t000 against 12 public and virtual refs."""
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

ROOT = Path('samples/stage3/v080_11')
OUT = Path('reports/timbre_blend_stage3_v080_11.json')
MARGIN = 0.01


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    plan = json.loads((ROOT / 'plan.json').read_text())
    expected = json.loads((ROOT / 'output_sha256.json').read_text())
    corrected = json.loads(Path('reports/timbre_blend_stage3_corrected_standard.json').read_text())
    diagnostics = json.loads(Path('reports/timbre_blend_stage3_diagnostics.json').read_text())
    assert len(plan['inputs']) == len(expected) == 11
    assert plan['md5'] == '733d9c6d44071d91a1c0b0f190db44db'
    assert plan['t_start'] == 0 and plan['mix'] == {'13': .8, '1': .05, '8': .05, '9': .05, '12': .05}
    assert plan['protocol']['vocoder_cudnn'] is False
    original = select(Path('data/timbre_blend_stage2/train/audio'))
    public = {k: v for k, v in original.items() if k != '13_lingxi'}
    names = {i: next(k for k in public if k.startswith(f'{i}_')) for i in range(1, 13)}
    assert len(public) == len(set(names.values())) == 12
    virtual = select(Path('data/timbre_blend_stage2/val/audio'))['13_lingxi']
    assert len(virtual) == 2 and all(len(files) == 2 for _, files in virtual)
    model = Path.home() / 'work/eval-models/ecapa-voxceleb-0f99f2d'
    for name, digest in MODEL_FILES.items():
        assert sha(model / name) == digest
    torch.set_num_threads(2)
    classifier = SpeakerRecognition.from_hparams(source=str(model),
        hparams_file='hyperparams_eval.yaml', savedir=str(model / 'runtime'))

    @torch.inference_mode()
    def embed(path):
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        a = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        a = torchaudio.functional.resample(a, sr, 16000)
        return torch.nn.functional.normalize(classifier.encode_batch(a).reshape(-1), dim=0)

    public_refs = {i: [embed(wav) for _, files in public[names[i]] for wav in files]
                   for i in range(1, 13)}
    virtual_refs = [embed(wav) for _, files in virtual for wav in files]
    assert all(len(v) == 4 for v in public_refs.values()) and len(virtual_refs) == 4
    embedded = []
    for row in plan['inputs']:
        input_path = Path('samples/stage2/expanded_c_blind/inputs') / row['input']
        output_path = ROOT / row['input']
        assert sha(input_path) == row['source_sha256']
        assert sha(output_path) == expected[row['input']]
        info = sf.info(output_path)
        assert info.samplerate == 44100 and info.channels == 1
        assert abs(info.duration - row['duration_seconds']) < .05
        embedded.append(embed(output_path))
    inflation = corrected['calibration']['signed_model_inflation']
    floor = corrected['standard']['floor']
    assert floor == .4137 and MARGIN == .01
    native = diagnostics['native_virtual']['by_public_spk']
    virtual_per_input = {row['input']: float(np.mean([float(torch.dot(a, ref)) for ref in virtual_refs]))
                         for row, a in zip(plan['inputs'], embedded)}
    virtual_mean = float(np.mean(list(virtual_per_input.values())))
    singers = {}
    failed_a, failed_b = [], []
    for i in range(1, 13):
        per_input = {row['input']: float(np.mean([float(torch.dot(a, ref)) for ref in public_refs[i]]))
                     for row, a in zip(plan['inputs'], embedded)}
        score = float(np.mean(list(per_input.values())))
        ceiling = max(native[str(i)]['mean'], floor) + inflation
        gap = ceiling - score
        a_pass = gap >= MARGIN
        b_pass = virtual_mean > score
        if not a_pass:
            failed_a.append(i)
        if not b_pass:
            failed_b.append(i)
        singers[str(i)] = {'opensinger_singer': int(names[i].split('singer')[1]),
                           'native_mean': native[str(i)]['mean'], 'mean_11x4': score,
                           'per_input_mean_of_4': per_input,
                           'condition_a_ceiling_inclusive': ceiling, 'safety_gap': gap,
                           'condition_a_with_0_01_margin_passes': a_pass,
                           'condition_b_strict_passes': b_pass}
        print('V080_11_SINGER', i, 'mean', round(score, 6), 'A_gap', round(gap, 6),
              'A', a_pass, 'B', b_pass, flush=True)
    previous = next(x for x in corrected['results'] if x['recipe_id'] == 'v080_t000')
    report = {'status': '11_CLIP_DIAGNOSTIC_ONLY_NOT_PREAPPROVED_BACKUP',
              'recipe': 'v080_t000', 'checkpoint_md5': plan['md5'], 'mix': plan['mix'],
              'inputs': plan['inputs'], 'output_sha256': expected,
              'protocol': {'model': 'speechbrain/spkrec-ecapa-voxceleb',
                           'reference_selection': corrected['calibration']['source'],
                           'aggregation': 'For each public Y or virtual: mean of 11 outputs x four reference clips, equally weighted; not the old three-clip mean.',
                           'inference': plan['protocol'],
                           'reference_model_file_sha256': MODEL_FILES},
              'calibration': {'real_different_mean': corrected['calibration']['real_different_singer_mean'],
                              'model_different_mean': corrected['calibration']['model_different_singer_grouped_mean'],
                              'signed_inflation': inflation, 'floor': floor, 'safety_margin_inclusive': MARGIN},
              'virtual_mean_11x4': virtual_mean, 'virtual_per_input_mean_of_4': virtual_per_input,
              'by_public_spk': singers, 'condition_a_failed_spk_ids': failed_a,
              'condition_b_failed_spk_ids': failed_b, 'passes_11_clip_gate': not failed_a and not failed_b,
              'previous_three_clip': {'condition_a_min_gap': min(
                   d['condition_a_ceiling_inclusive']-d['public_mean'] for d in previous['by_public_spk'].values()),
                   'passes_0_01_safety_margin': all(
                   d['condition_a_ceiling_inclusive']-d['public_mean'] >= MARGIN
                   for d in previous['by_public_spk'].values()) and not previous['condition_b_failures']},
              'caveat': 'A pass on eleven clips cannot retroactively turn the separately specified three-clip safety-margin gate into a pass. Speech ECAPA remains a singing-domain proxy; do not publish or bake a model from this diagnostic.'}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('V080_11_GATE', 'PASS' if report['passes_11_clip_gate'] else 'FAIL',
          'A', failed_a, 'B', failed_b, 'virtual', virtual_mean,
          'original_three_safety', report['previous_three_clip'], flush=True)


if __name__ == '__main__':
    main()
