"""Offline 12-public-speaker hard gate for Stage-3 recipes, using Stage-2 ECAPA references."""
import hashlib
import json
import os
from pathlib import Path
from itertools import combinations, product

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from scripts.eval_stage2_similarity import MODEL_FILES, select

ROOT = Path('samples/stage3/recipe_screen')
OUT = Path('reports/timbre_blend_stage3_recipe_screen.json')
THRESHOLD = 0.456


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists(), f'refusing existing report: {OUT}'
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    calibration = json.loads(Path('reports/timbre_blend_stage2_c_resume_similarity.json').read_text())
    assert calibration['model'] == 'speechbrain/spkrec-ecapa-voxceleb'
    assert abs(calibration['public_reference']['different_speaker']['p95'] - .45596396923065186) < 1e-10
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert len(plan['recipes']) == 12 and len(plan['inputs']) == 3
    all_train = select(Path('data/timbre_blend_stage2/train/audio'))
    public = {name: songs for name, songs in all_train.items() if name != '13_lingxi'}
    assert len(public) == 12 and all(len(songs) == 2 and all(len(wavs) == 2 for _, wavs in songs)
                                    for songs in public.values())
    expected = {i: next(s for s in public if s.startswith(f'{i}_')) for i in range(1, 13)}
    assert len(set(expected.values())) == 12
    virtual = select(Path('data/timbre_blend_stage2/val/audio'))['13_lingxi']
    assert len(virtual) == 2 and all(len(wavs) == 2 for _, wavs in virtual)
    model_root = Path.home() / 'work/eval-models/ecapa-voxceleb-0f99f2d'
    for filename, digest in MODEL_FILES.items():
        assert sha(model_root / filename) == digest, filename
    torch.set_num_threads(2)
    classifier = SpeakerRecognition.from_hparams(source=str(model_root),
        hparams_file='hyperparams_eval.yaml', savedir=str(model_root / 'runtime'))

    @torch.inference_mode()
    def embed(path):
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        audio = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, sr, 16000)
        return torch.nn.functional.normalize(classifier.encode_batch(audio).reshape(-1), dim=0)

    refs = {i: [embed(wav) for _, wavs in public[name] for wav in wavs]
            for i, name in expected.items()}
    virtual_refs = [embed(wav) for _, wavs in virtual for wav in wavs]
    assert all(len(refs[i]) == 4 for i in refs) and len(virtual_refs) == 4
    different = [float(torch.dot(a, b)) for i, j in combinations(range(1, 13), 2)
                 for a, b in product(refs[i], refs[j])]
    assert len(different) == 1056
    true_p95 = float(np.percentile(different, 95))
    gate_limit = min(THRESHOLD, true_p95)
    print('CALIBRATION_PUBLIC_ONLY', 'p95', true_p95, 'fixed_ceiling', THRESHOLD,
          'effective_limit', gate_limit, flush=True)
    results = []
    for recipe in plan['recipes']:
        output_embeds = []
        output_hashes = {}
        for clip in plan['inputs']:
            path = ROOT / recipe['id'] / f"{clip['band']}.wav"
            audio, sr = sf.read(path)
            assert sr == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
            assert np.max(np.abs(audio)) > 0 and abs(len(audio) / sr - clip['duration_seconds']) < .05
            output_hashes[clip['band']] = sha(path)
            output_embeds.append(embed(path))
        public_scores = {}
        failed = []
        for i in range(1, 13):
            by_input = {clip['band']: float(np.mean([float(torch.dot(output_embeds[j], ref)) for ref in refs[i]]))
                        for j, clip in enumerate(plan['inputs'])}
            value = float(np.mean(list(by_input.values())))
            public_scores[str(i)] = {'mean': value, 'by_input': by_input}
            if value >= gate_limit:
                failed.append({'spk_id': i, 'opensinger_singer': int(expected[i].split('singer')[1]),
                               'mean_similarity': value})
        virtual_by_input = {clip['band']: float(np.mean([float(torch.dot(output_embeds[j], ref)) for ref in virtual_refs]))
                            for j, clip in enumerate(plan['inputs'])}
        result = {'recipe': recipe, 'outputs_sha256': output_hashes,
                  'public_scores': public_scores, 'failed_public_speakers': failed,
                  'max_public_mean': max(v['mean'] for v in public_scores.values()),
                  'virtual_mean': float(np.mean(list(virtual_by_input.values()))),
                  'virtual_by_input': virtual_by_input, 'passes': not failed}
        results.append(result)
        print('GATE', recipe['id'], 'PASS' if not failed else 'FAIL',
              'max_public', round(result['max_public_mean'], 6),
              'virtual', round(result['virtual_mean'], 6),
              'violations', [(row['spk_id'], round(row['mean_similarity'], 6)) for row in failed], flush=True)
    report = {'status': 'GATE_COMPLETE', 'threshold_strictly_below': gate_limit,
              'requested_ceiling': THRESHOLD, 'actual_12_public_different_speaker_p95': true_p95,
              'different_public_score_count': len(different),
              'old_calibration_included_virtual_and_omitted_spk09': True,
              'old_calibration_p95_including_virtual': calibration['public_reference']['different_speaker']['p95'],
              'calibration_source': 'reports/timbre_blend_stage2_c_resume_similarity.json',
              'model': calibration['model'], 'model_revision': calibration['revision'],
              'model_file_sha256': MODEL_FILES,
              'reference_protocol': 'Stage-2 per-singer selection (two songs, two longest clips per song), corrected to all twelve public IDs 1–12, excluding virtual ID 13. Cross-singer 12 choose 2 x 16 = 1056 cosines for p95. Four virtual validation clips across two songs. For each output, cosine to each of four clips; equal mean across three output clips.',
              'caveat': 'Speech-trained ECAPA on sung audio is an offline similarity proxy, not a listener judgment or a legal guarantee. A three-clip screen does not prove the remaining eight clips meet the gate.',
              'plan': plan, 'results': results,
              'passing_recipe_ids': [r['recipe']['id'] for r in results if r['passes']]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('GATE_REPORT', OUT, 'passing', report['passing_recipe_ids'], flush=True)


if __name__ == '__main__':
    main()
