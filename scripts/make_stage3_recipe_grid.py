"""Render the 12 fixed Stage-3 recipes on three selected, held-out clips."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('samples/stage3/recipe_screen')
PREVIEW = Path('samples/stage3/public_singer_preview')
MODEL = Path('exp/timbre_blend_stage2_embedding_resume_4600/model_7800.pt')
SELECTED = (1, 8, 9, 12)
STARTS = (0.0, 0.25, 0.5)
VIRTUAL = (1.0, 0.8, 0.6, 0.4)


def sha(path, kind='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, kind).hexdigest()


def recipes():
    for virtual in VIRTUAL:
        public_weight = round((1 - virtual) / len(SELECTED), 8)
        assert public_weight <= 0.2
        weights = {13: virtual, **({spk: public_weight for spk in SELECTED} if public_weight else {})}
        assert abs(sum(weights.values()) - 1) < 1e-8
        for start in STARTS:
            yield {'id': f'v{int(virtual * 100):03d}_t{int(start * 100):03d}',
                   'virtual_weight': virtual, 'public_weight_each': public_weight,
                   'mix': weights, 't_start': start}


def check(path, reference):
    audio, sr = sf.read(path)
    assert sr == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
    assert 0.001 < float(np.sqrt(np.mean(audio ** 2))) < 1
    assert float(np.max(np.abs(audio))) <= 1
    assert abs(len(audio) / sr - reference['duration_seconds']) < 0.05
    return sha(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    opt = ap.parse_args()
    assert sha(MODEL, 'md5') == '733d9c6d44071d91a1c0b0f190db44db'
    selection = json.loads((PREVIEW / 'manifest.json').read_text())
    inputs = selection['inputs']
    assert len(inputs) == 3 and {r['band'] for r in inputs} == {'low', 'mid', 'high'}
    assert {r['spk_id'] for r in selection['speakers'] if r['spk_id'] in SELECTED} == set(SELECTED)
    for clip in inputs:
        src = PREVIEW / 'inputs' / (clip['band'] + '__' + clip['input'])
        assert sha(src) == clip['source_sha256']
        assert clip['seed'] == int.from_bytes(hashlib.sha256(clip['input'].encode()).digest()[:4], 'big')
    grid = list(recipes())
    assert len(grid) == 12 and len({r['id'] for r in grid}) == 12
    if opt.check_only:
        print('GRID_PREFLIGHT_PASS', [(r['id'], r['mix']) for r in grid], flush=True)
        return
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert not ROOT.exists(), f'refusing existing screening directory: {ROOT}'
    ROOT.mkdir(parents=True)
    (ROOT / 'plan.json').write_text(json.dumps({'checkpoint': str(MODEL), 'checkpoint_md5': sha(MODEL, 'md5'),
        'selected_public_spk_ids': SELECTED, 'inputs': inputs, 'recipes': grid,
        'inference': {'infer_step': 50, 'method': 'euler', 'key': 0, 'formant_shift': 0,
            'pitch_extractor': 'rmvpe', 'threshold_db': -60, 'f0_min': 50,
            'f0_max': 1100, 'vocoder_cudnn': False,
            'seed_rule': 'Stage 2 SHA-256 basename first four bytes'}}, ensure_ascii=False, indent=2) + '\n')
    env = {**os.environ, 'PYTHONFAULTHANDLER': '1', 'PYTHONHASHSEED': '0', 'PYTHONPATH': '.'}
    for recipe in grid:
        dest = ROOT / recipe['id']
        dest.mkdir()
        for clip in inputs:
            src = PREVIEW / 'inputs' / (clip['band'] + '__' + clip['input'])
            out = dest / (clip['band'] + '.wav')
            args = ['.venv/bin/python', '.tmp/stage2_seeded_infer.py', str(clip['seed']),
                    '-m', str(MODEL), '-i', str(src), '-o', str(out), '-id', '13',
                    '-mix', str(recipe['mix']), '-ts', str(recipe['t_start']),
                    '-step', '50', '-method', 'euler', '-k', '0', '-f', '0',
                    '-pe', 'rmvpe', '-th', '-60', '-fmin', '50', '-fmax', '1100',
                    '-d', 'cuda', '--disable-vocoder-cudnn']
            log = dest / (clip['band'] + '.log')
            print('GRID_START', recipe['id'], clip['band'], flush=True)
            with log.open('x') as stream:
                subprocess.run(args, check=True, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=360)
            print('GRID_OUTPUT_OK', recipe['id'], clip['band'], check(out, clip), flush=True)
    print('GRID_RENDER_PASS', len(grid) * len(inputs), flush=True)


if __name__ == '__main__':
    main()
