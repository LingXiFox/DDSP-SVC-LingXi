"""Stage-3 gate 1: recalibrate real public cross-singer mean and reassess saved recipes."""
import hashlib
import json
import os
from itertools import combinations, product
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from scripts.eval_stage2_similarity import MODEL_FILES, select

OUT = Path('reports/timbre_blend_stage3_corrected_standard.json')
FLOOR = 0.4137  # Explicit user-specified grouped-different p95 rounded to four decimals.


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists(), f'refusing existing report: {OUT}'
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    diagnostics = json.loads(Path('reports/timbre_blend_stage3_diagnostics.json').read_text())
    grid = json.loads(Path('reports/timbre_blend_stage3_recipe_screen.json').read_text())
    assert len(grid['results']) == 12 and len(diagnostics['native_virtual']['by_public_spk']) == 12
    assert len(grid['plan']['inputs']) == 3
    converted = diagnostics['pure_public_conversion']
    assert len(converted['outputs']) == 12
    converted_different = [row['by_public_spk_mean'][str(y)]
                           for row in converted['outputs']
                           for y in range(1, 13) if y != row['target_spk_id']]
    assert len(converted_different) == 132
    model_other_mean = float(np.mean(converted_different))
    assert abs(model_other_mean - converted['different_singer_grouped']['mean']) < 1e-12
    assert abs(converted['different_singer_grouped']['p95'] - FLOOR) < .00005

    original = select(Path('data/timbre_blend_stage2/train/audio'))
    public = {name: songs for name, songs in original.items() if name != '13_lingxi'}
    assert len(public) == 12 and '13_lingxi' in original
    names = {i: next(name for name in public if name.startswith(f'{i}_')) for i in range(1, 13)}
    assert len(set(names.values())) == 12 and names[9] == '9_singer33'
    root = Path.home() / 'work/eval-models/ecapa-voxceleb-0f99f2d'
    for name, digest in MODEL_FILES.items():
        assert sha(root / name) == digest, name
    torch.set_num_threads(2)
    classifier = SpeakerRecognition.from_hparams(source=str(root),
        hparams_file='hyperparams_eval.yaml', savedir=str(root / 'runtime'))

    @torch.inference_mode()
    def embed(path):
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
        waveform = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return torch.nn.functional.normalize(classifier.encode_batch(waveform).reshape(-1), dim=0)

    refs = {i: [embed(wav) for _, files in public[names[i]] for wav in files]
            for i in range(1, 13)}
    assert all(len(v) == 4 for v in refs.values())
    pair_means = [float(np.mean([float(torch.dot(a, b)) for a, b in product(refs[i], refs[j])]))
                  for i, j in combinations(range(1, 13), 2)]
    assert len(pair_means) == 66
    real_other_mean = float(np.mean(pair_means))
    individual = [float(torch.dot(a, b)) for i, j in combinations(range(1, 13), 2)
                  for a, b in product(refs[i], refs[j])]
    assert len(individual) == 1056
    old_corrected_p95 = grid['actual_12_public_different_speaker_p95']
    assert abs(float(np.percentile(individual, 95)) - old_corrected_p95) < 1e-6
    assert abs(real_other_mean - float(np.mean(individual))) < 1e-12
    inflation = model_other_mean - real_other_mean  # Signed; never truncate negative values.
    print('CALIBRATION', 'real_mean', real_other_mean, 'model_mean', model_other_mean,
          'signed_inflation', inflation, 'pairs', len(pair_means), flush=True)
    native = diagnostics['native_virtual']['by_public_spk']
    results = []
    for row in grid['results']:
        virtual = row['virtual_mean']
        assert len(row['public_scores']) == 12
        details = {}
        failures_a, failures_b = [], []
        for i in range(1, 13):
            key = str(i)
            native_mean = native[key]['mean']
            public_mean = row['public_scores'][key]['mean']
            ceiling = max(native_mean, FLOOR) + inflation
            fails_a = public_mean > ceiling
            fails_b = virtual <= public_mean
            details[key] = {'opensinger_singer': int(names[i].split('singer')[1]),
                            'native_mean': native_mean, 'public_mean': public_mean,
                            'condition_a_ceiling_inclusive': ceiling,
                            'condition_a_passes': not fails_a, 'condition_b_passes': not fails_b}
            if fails_a:
                failures_a.append({'spk_id': i, 'public_mean': public_mean,
                                   'condition_a_ceiling_inclusive': ceiling})
            if fails_b:
                failures_b.append({'spk_id': i, 'public_mean': public_mean,
                                   'virtual_mean': virtual})
        results.append({'recipe_id': row['recipe']['id'], 'virtual_weight': row['recipe']['virtual_weight'],
                        't_start': row['recipe']['t_start'], 'virtual_mean': virtual,
                        'maximum_public_mean': max(v['public_mean'] for v in details.values()),
                        'condition_a_failures': failures_a, 'condition_b_failures': failures_b,
                        'passes': not failures_a and not failures_b, 'by_public_spk': details})
        print('REASSESS', row['recipe']['id'], 'PASS' if results[-1]['passes'] else 'FAIL',
              'A', [(x['spk_id'], round(x['public_mean'], 6), round(x['condition_a_ceiling_inclusive'], 6))
                   for x in failures_a], 'B', [x['spk_id'] for x in failures_b], flush=True)
    output = {'status': 'STEP_1_COMPLETE_AWAITING_USER_CONFIRMATION',
              'calibration': {'real_different_singer_mean': real_other_mean,
                              'real_public_singer_pair_count': len(pair_means),
                              'real_individual_cosine_count': len(individual),
                              'corrected_real_different_singer_p95_check': old_corrected_p95,
                              'model_different_singer_grouped_mean': model_other_mean,
                              'model_grouped_count': len(converted_different),
                              'signed_model_inflation': inflation,
                              'source': '2 songs x 2 clips per public singer, all 12 public IDs 1–12 (spk13 excluded, spk09 included). Each unordered pair averages 4x4=16 cosines; 66 pair means are averaged with equal weight. Model reference is the mean of 132 directed X-to-other-Y groups, each a 3x4 mean.'},
              'standard': {'status': 'USER_PROPOSED_NOT_YET_CONFIRMED', 'floor': FLOOR,
                           'condition_a': 'For each public Y: recipe_mean_Y <= max(native_virtual_mean_Y, 0.4137) + signed_model_inflation.',
                           'condition_b': 'Recipe mean to four virtual validation references across each of three inputs must be strictly greater than recipe mean to each public Y (same 3x4 aggregation).'},
              'source_reports': ['reports/timbre_blend_stage3_recipe_screen.json',
                                 'reports/timbre_blend_stage3_diagnostics.json'],
              'results': results,
              'passing_recipe_ids': [r['recipe_id'] for r in results if r['passes']],
              'failed_recipe_ids': [r['recipe_id'] for r in results if not r['passes']],
              'caveat': 'Read-only rescoring of existing 12 recipes, no new inference. ECAPA in singing is a proxy, not a perceptual or legal identity guarantee. No standard has been approved yet; step 2/3 not started.'}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print('REASSESS_REPORT', OUT, 'PASS', output['passing_recipe_ids'],
          'FAIL', output['failed_recipe_ids'], flush=True)


if __name__ == '__main__':
    main()
