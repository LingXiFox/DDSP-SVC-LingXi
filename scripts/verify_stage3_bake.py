"""Check the selected mix, baked row, and local one-row model on four clips."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from logger.utils import SPK_EMBED_KEY

SOURCE = Path('exp/timbre_blend_stage2_embedding_resume_4600/model_7800.pt')
FULL = Path('exp/timbre_blend_stage3_baked_full/model_baked.pt')
SLIM = Path('exp/timbre_blend_stage3_final_slim/model_slim.pt')
ROOT = Path('samples/stage3/baked_compare')
REPORT = Path('reports/timbre_blend_stage3_baked_model.json')
ARMS = (
    ('original_mix', SOURCE, 13, '{13: 0.4, 4: 0.2, 8: 0.2, 11: 0.2}', 'mix'),
    ('full_baked', FULL, 14, None, 'full'),
    ('slim_baked', SLIM, 1, None, 'slim'),
)


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def structural_checks():
    assert digest(SOURCE, 'md5') == '733d9c6d44071d91a1c0b0f190db44db'
    source = torch.load(SOURCE, map_location='cpu', weights_only=True)['model']
    original = source[SPK_EMBED_KEY]
    assert original.shape == (13, 512)
    expected = torch.zeros_like(original[0])
    for speaker, weight in ((13, .4), (4, .2), (8, .2), (11, .2)):
        expected = expected + weight * original[speaker - 1]
    for path, count in ((FULL, 14), (SLIM, 1)):
        cfg = yaml.safe_load((path.parent / 'config.yaml').read_text())
        assert cfg['model']['n_spk'] == count
        assert 'train_path' not in cfg['data'] and 'valid_path' not in cfg['data']
        state = torch.load(path, map_location='cpu', weights_only=True)['model']
        assert set(state) == set(source)
        assert state[SPK_EMBED_KEY].shape == (count, 512)
        assert torch.equal(state[SPK_EMBED_KEY][-1], expected)
        assert all(torch.equal(v, state[k]) for k, v in source.items() if k != SPK_EMBED_KEY)
        if count == 14:
            assert torch.equal(state[SPK_EMBED_KEY][:13], original)
        del state
    return {'source_md5': digest(SOURCE, 'md5'), 'full_sha256': digest(FULL),
            'slim_sha256': digest(SLIM), 'source_speakers': 13,
            'full_speakers': 14, 'slim_speakers': 1,
            'all_non_embedding_tensors_bitwise_equal': True,
            'old_thirteen_rows_preserved_in_full': True,
            'slim_only_embedding_is_weighted_sum': True}


def run(clip, name, model, speaker, mix, prefix, seed):
    out = ROOT / (prefix + '__' + name)
    if not out.exists():
        args = ['.venv/bin/python', '.tmp/stage2_seeded_infer.py', str(seed),
                '-m', str(model), '-i', str(clip), '-o', str(out), '-id', str(speaker),
                '-ts', '0.0', '-step', '50', '-method', 'euler', '-k', '0', '-f', '0',
                '-pe', 'rmvpe', '-th', '-60', '-fmin', '50', '-fmax', '1100', '-d', 'cuda',
                '--disable-vocoder-cudnn', '--seed-after-load', str(seed)]
        if mix is not None:
            args += ['-mix', mix]
        print('COMPARE_START', name, prefix, flush=True)
        with (ROOT / (prefix + '__' + name + '.log')).open('x') as stream:
            subprocess.run(args, check=True, env={**os.environ, 'PYTHONHASHSEED': '0',
                           'PYTHONPATH': '.', 'PYTHONFAULTHANDLER': '1'},
                           stdout=stream, stderr=subprocess.STDOUT, timeout=360)
    audio, sr = sf.read(out)
    assert sr == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
    assert .001 < float(np.sqrt(np.mean(audio**2))) < 1
    assert abs(len(audio)/sr - sf.info(clip).duration) < .05
    return audio, digest(out)


def main():
    assert not REPORT.exists() and ROOT.is_dir()
    assert sorted(p.name for p in ROOT.iterdir()) == sorted(
        f'{arm}__10_侧脸_2.wav{suffix}'
        for arm in ('mix', 'full', 'slim') for suffix in ('', '.log'))
    structure = structural_checks()
    cfg = json.loads(Path('configs/timbre_blend_stage3_listening_set.json').read_text())
    selection = json.loads(Path(cfg['source_selection']).read_text())
    assert len(cfg['clips']) == 4
    comparisons = []
    for item in cfg['clips']:
        name = item['input']
        clip = Path('samples/stage2/expanded_c_blind/inputs') / name
        assert digest(clip) == item['sha256'] == next(
            r['sha256'] for r in selection['inputs'] if Path(r['file']).name == name)
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
        outputs = {}
        hashes = {}
        for label, model, speaker, mix, prefix in ARMS:
            outputs[label], hashes[label] = run(clip, name, model, speaker, mix, prefix, seed)
        a, b, c = (outputs[arm] for arm in ('original_mix', 'full_baked', 'slim_baked'))
        assert len(a) == len(b) == len(c)
        diff = a - b
        relative_rmse = float(np.sqrt(np.mean(diff**2) / np.mean(a**2)))
        correlation = float(np.corrcoef(a, b)[0, 1])
        max_abs_diff = float(np.max(np.abs(diff)))
        exact = bool(np.array_equal(b, c))
        assert exact, f'{name}: full and slim output not sample-identical'
        assert relative_rmse < .001 and correlation > .99999, (name, relative_rmse, correlation)
        result = {'input': name, 'role': item['role'], 'seed': seed,
                  'output_sha256': hashes, 'duration_seconds': len(a)/44100,
                  'mix_vs_full_relative_rmse': relative_rmse,
                  'mix_vs_full_correlation': correlation,
                  'mix_vs_full_max_abs_diff': max_abs_diff,
                  'mix_vs_full_sample_exact': bool(np.array_equal(a, b)),
                  'full_vs_slim_sample_exact': exact}
        comparisons.append(result)
        print('COMPARE_PASS', name, 'rel_rmse', relative_rmse, 'corr', correlation,
              'max_abs', max_abs_diff, 'full_slim_exact', exact, flush=True)
    report = {'selected_recipe': 'v040_t000', 'mix': {'13': .4, '4': .2, '8': .2, '11': .2},
              'checkpoint_structure': structure,
              'comparison': {'inputs': len(comparisons), 'seed_rule': 'Stage 2 SHA-256 input basename first 4 bytes; reset torch CPU/CUDA RNG after model and encoder initialization.',
                             'infer_step': 50, 'method': 'euler', 't_start': 0,
                             'vocoder_cudnn': False, 'relative_rmse_limit_exclusive': .001,
                             'correlation_min_exclusive': .99999,
                             'note': 'The original 4-row additive mix and one baked row differ slightly due to float32 addition order / PCM quantization. Full baked vs one-row slim must be sample-exact.'},
              'clips': comparisons,
              'all_full_and_slim_sample_exact': all(x['full_vs_slim_sample_exact'] for x in comparisons),
              'all_original_mix_near_baked': all(x['mix_vs_full_relative_rmse'] < .001 and x['mix_vs_full_correlation'] > .99999 for x in comparisons),
              'model_location': 'WSL local exp/ paths only; no transfer or upload'}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('BAKE_VERIFICATION_PASS', REPORT, flush=True)


if __name__ == '__main__':
    main()
