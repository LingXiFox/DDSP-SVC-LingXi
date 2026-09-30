"""Render the three-clip grid, then only screened-in recipes on eleven holdouts."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = Path('exp/timbre_blend_stage2_embedding_resume_4600/model_7800.pt')
PREVIEW = Path('samples/stage3/public_singer_preview')
SCREEN = Path('samples/stage3/new_partner_screen')
HOLDOUT = Path('samples/stage3/new_partner_11')
SELECTED = (4, 8, 11)


def sha(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def recipes():
    rows = []
    for virtual in (1.0, .8, .6, .4):
        public = round((1 - virtual) / len(SELECTED), 12)
        assert public <= .2 and abs(virtual + len(SELECTED)*public - 1) < 1e-9
        rows.append({'id': f'v{round(100*virtual):03d}_t000', 'virtual_weight': virtual,
                     'public_weight_each': public,
                     'mix': {13: virtual, **({i: public for i in SELECTED} if public else {})},
                     't_start': 0.0})
    assert len(rows) == 4
    return rows


def check(path, seconds):
    audio, sr = sf.read(path)
    assert sr == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
    assert .001 < float(np.sqrt(np.mean(audio**2))) < 1
    assert float(np.max(np.abs(audio))) <= 1
    assert abs(len(audio)/sr - seconds) < .05
    return sha(path)


def infer(src, dst, recipe, seed, log):
    assert not dst.exists()
    mix = {int(k): float(v) for k, v in recipe['mix'].items()}
    assert all(1 <= k <= 13 for k in mix) and abs(sum(mix.values()) - 1) < 1e-9
    env = {**os.environ, 'PYTHONFAULTHANDLER': '1', 'PYTHONHASHSEED': '0', 'PYTHONPATH': '.'}
    args = ['.venv/bin/python', '.tmp/stage2_seeded_infer.py', str(seed),
            '-m', str(MODEL), '-i', str(src), '-o', str(dst), '-id', '13',
            '-mix', str(mix), '-ts', '0.0', '-step', '50', '-method', 'euler',
            '-k', '0', '-f', '0', '-pe', 'rmvpe', '-th', '-60', '-fmin', '50',
            '-fmax', '1100', '-d', 'cuda', '--disable-vocoder-cudnn']
    if log.exists():
        log = log.with_name(log.name + '.retry1')  # Keep the failed first log intact.
    with log.open('x') as stream:
        subprocess.run(args, env=env, check=True, stdout=stream,
                       stderr=subprocess.STDOUT, timeout=360)


def screen():
    assert not SCREEN.exists()
    assert sha(MODEL, 'md5') == '733d9c6d44071d91a1c0b0f190db44db'
    preview = json.loads((PREVIEW / 'manifest.json').read_text())
    old = json.loads(Path('samples/stage3/recipe_screen/plan.json').read_text())
    assert preview['checkpoint_md5'] == old['checkpoint_md5'] == sha(MODEL, 'md5')
    assert old['inference']['vocoder_cudnn'] is False
    assert [s['spk_id'] for s in preview['speakers'] if s['spk_id'] in SELECTED] == list(SELECTED)
    inputs = preview['inputs']
    assert [r['band'] for r in inputs] == ['low', 'mid', 'high']
    for clip in inputs:
        assert sha(PREVIEW / 'inputs' / f"{clip['band']}__{clip['input']}") == clip['source_sha256']
    SCREEN.mkdir(parents=True)
    plan = {'checkpoint': str(MODEL), 'checkpoint_md5': sha(MODEL, 'md5'),
            'selected_public_spk_ids': SELECTED, 'recipes': recipes(), 'inputs': inputs,
            'protocol': old['inference'],
            'baseline_source': 'samples/stage3/recipe_screen/v100_t000/ (byte-for-byte reused, same model/input/seed/parameters)'}
    (SCREEN / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    for recipe in plan['recipes']:
        dest = SCREEN / recipe['id']
        dest.mkdir()
        for clip in inputs:
            src = PREVIEW / 'inputs' / f"{clip['band']}__{clip['input']}"
            out = dest / f"{clip['band']}.wav"
            if recipe['virtual_weight'] == 1.0:
                prior = Path('samples/stage3/recipe_screen/v100_t000') / f"{clip['band']}.wav"
                assert check(prior, clip['duration_seconds'])
                shutil.copyfile(prior, out)
                print('GRID_BASELINE_REUSED', clip['band'], flush=True)
            else:
                print('GRID_JOB_START', recipe['id'], clip['band'], flush=True)
                infer(src, out, recipe, clip['seed'], dest / f"{clip['band']}.log")
            print('GRID_OUTPUT_OK', recipe['id'], clip['band'], check(out, clip['duration_seconds']), flush=True)
    print('GRID_SCREEN_RENDER_PASS', len(plan['recipes']) * len(inputs), flush=True)


def holdout():
    assert sha(MODEL, 'md5') == '733d9c6d44071d91a1c0b0f190db44db'
    screened = json.loads(Path('reports/timbre_blend_stage3_new_partner_screen.json').read_text())
    plan = json.loads((SCREEN / 'plan.json').read_text())
    assert screened['checkpoint_md5'] == plan['checkpoint_md5'] == sha(MODEL, 'md5')
    selection = json.loads(Path('samples/stage2/expanded_c_blind/input_selection.json').read_text())
    input_rows = []
    for row in selection['inputs']:
        name = Path(row['file']).name
        source = Path('samples/stage2/expanded_c_blind/inputs') / name
        assert sha(source) == row['sha256']
        input_rows.append({'input': name, 'singer': row['singer'], 'seconds': sf.info(source).duration,
                           'sha256': row['sha256'], 'seed': int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')})
    assert len(input_rows) == 11
    for clip in plan['inputs']:
        row = next(r for r in input_rows if r['input'] == clip['input'])
        assert row['sha256'] == clip['source_sha256'] and row['seed'] == clip['seed']
    chosen = [r for r in plan['recipes'] if r['id'] in screened['passing_recipe_ids']]
    assert {r['id'] for r in chosen} == set(screened['passing_recipe_ids'])
    assert chosen and chosen[0]['id'] == 'v100_t000'
    holdout_plan = {'checkpoint': str(MODEL), 'checkpoint_md5': sha(MODEL, 'md5'),
        'inputs': input_rows, 'recipes': chosen, 'protocol': plan['protocol'],
        'note': 'Only recipes passing the three-clip 0.01-margin screen; v100 is the separate pure-virtual control.'}
    if HOLDOUT.exists():
        assert json.loads((HOLDOUT / 'plan.json').read_text()) == holdout_plan
    else:
        HOLDOUT.mkdir(parents=True)
        (HOLDOUT / 'plan.json').write_text(json.dumps(holdout_plan, ensure_ascii=False, indent=2) + '\n')
    for recipe in chosen:
        dest = HOLDOUT / recipe['id']
        dest.mkdir(exist_ok=True)
        for row in input_rows:
            src = Path('samples/stage2/expanded_c_blind/inputs') / row['input']
            out = dest / row['input']
            if out.exists():
                assert check(out, row['seconds'])
                print('HOLDOUT_OUTPUT_REUSED', recipe['id'], row['input'], flush=True)
            else:
                print('HOLDOUT_JOB_START', recipe['id'], row['input'], flush=True)
                infer(src, out, recipe, row['seed'], dest / (row['input'] + '.log'))
            assert check(out, row['seconds'])
            overlap = next((clip for clip in plan['inputs'] if clip['input'] == row['input']), None)
            if overlap:
                reference, sr = sf.read(SCREEN / recipe['id'] / f"{overlap['band']}.wav")
                audio, out_sr = sf.read(out)
                assert sr == out_sr and len(audio) == len(reference)
                delta = float(np.sqrt(np.mean((audio-reference)**2) / np.mean(reference**2)))
                assert delta < .01, (recipe['id'], row['input'], delta)
                print('HOLDOUT_PRIOR_MATCH', recipe['id'], row['input'], delta, flush=True)
            print('HOLDOUT_OUTPUT_OK', recipe['id'], row['input'], sha(out), flush=True)
    print('GRID_HOLDOUT_RENDER_PASS', len(chosen)*len(input_rows), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('screen', 'holdout'))
    opt = ap.parse_args()
    assert os.environ.get('PYTHONHASHSEED') == '0'
    (screen if opt.phase == 'screen' else holdout)()
