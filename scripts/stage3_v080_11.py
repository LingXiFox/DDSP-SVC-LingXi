"""Render the specifically requested v080_t000 candidate on all eleven held-out WAVs."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('samples/stage3/v080_11')
SOURCE = Path('samples/stage2/expanded_c_blind/inputs')
MODEL = Path('exp/timbre_blend_stage2_embedding_resume_4600/model_7800.pt')
MIX = {13: 0.8, 1: 0.05, 8: 0.05, 9: 0.05, 12: 0.05}


def sha(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    assert not ROOT.exists(), f'refusing existing output directory: {ROOT}'
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert sha(MODEL, 'md5') == '733d9c6d44071d91a1c0b0f190db44db'
    selection = json.loads(Path('samples/stage2/expanded_c_blind/input_selection.json').read_text())
    grid = json.loads(Path('samples/stage3/recipe_screen/plan.json').read_text())
    recipe = next(r for r in grid['recipes'] if r['id'] == 'v080_t000')
    assert recipe['mix'] == {str(k): v for k, v in MIX.items()}
    assert recipe['t_start'] == 0 and grid['checkpoint_md5'] == sha(MODEL, 'md5')
    rows = selection['inputs']
    assert len(rows) == 11 and selection['total_seconds'] >= 110
    plan = []
    for row in rows:
        name = Path(row['file']).name
        source = SOURCE / name
        assert sha(source) == row['sha256']
        plan.append({'input': name, 'singer': row['singer'], 'source_sha256': row['sha256'],
                     'seed': int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big'),
                     'duration_seconds': sf.info(source).duration})
    assert len({x['input'] for x in plan}) == 11
    for clip in grid['inputs']:
        row = next(x for x in plan if x['input'] == clip['input'])
        assert row['seed'] == clip['seed'] and row['source_sha256'] == clip['source_sha256']
    ROOT.mkdir(parents=True)
    (ROOT / 'plan.json').write_text(json.dumps({'checkpoint': str(MODEL), 'md5': sha(MODEL, 'md5'),
        'mix': MIX, 't_start': 0.0, 'inputs': plan,
        'protocol': grid['inference'],
        'note': '11 new inferences requested; v080_t000 is a diagnostic candidate, not certified by the three-clip 0.01 margin.'},
        ensure_ascii=False, indent=2) + '\n')
    env = {**os.environ, 'PYTHONFAULTHANDLER': '1', 'PYTHONHASHSEED': '0', 'PYTHONPATH': '.'}
    for index, item in enumerate(plan, 1):
        source = SOURCE / item['input']
        out = ROOT / item['input']
        assert not out.exists()
        args = ['.venv/bin/python', '.tmp/stage2_seeded_infer.py', str(item['seed']),
                '-m', str(MODEL), '-i', str(source), '-o', str(out), '-id', '13',
                '-mix', str(MIX), '-ts', '0.0', '-step', '50', '-method', 'euler',
                '-k', '0', '-f', '0', '-pe', 'rmvpe', '-th', '-60', '-fmin', '50',
                '-fmax', '1100', '-d', 'cuda', '--disable-vocoder-cudnn']
        print('V080_START', index, item['input'], flush=True)
        with (ROOT / f'{index:02d}.log').open('x') as stream:
            subprocess.run(args, check=True, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=360)
        audio, sr = sf.read(out)
        assert sr == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        assert .001 < rms < 1 and float(np.max(np.abs(audio))) <= 1
        assert abs(len(audio) / sr - item['duration_seconds']) < .05
        match = next((clip for clip in grid['inputs'] if clip['input'] == item['input']), None)
        if match:
            previous, previous_sr = sf.read(Path('samples/stage3/recipe_screen/v080_t000') / f"{match['band']}.wav")
            assert previous_sr == sr and len(previous) == len(audio)
            delta = float(np.sqrt(np.mean((audio - previous)**2) / np.mean(previous**2)))
            assert delta < .01, (item['input'], delta)
            print('V080_PRIOR_MATCH', item['input'], delta, flush=True)
        print('V080_OUTPUT_OK', index, item['input'], sha(out), 'rms', rms, flush=True)
    outputs = {r['input']: sha(ROOT / r['input']) for r in plan}
    (ROOT / 'output_sha256.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + '\n')
    print('V080_RENDER_PASS', len(outputs), flush=True)


if __name__ == '__main__':
    main()
