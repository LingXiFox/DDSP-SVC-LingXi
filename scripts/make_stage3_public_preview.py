"""Stage 3 gate 1: render three public holdout clips for each of 12 Stage-1 voices."""
import argparse
import gc
import hashlib
import json
import os
import random
import runpy
import shutil
import sys
from pathlib import Path

print("STAGE3_PROCESS_STARTED", flush=True)
import numpy as np
import soundfile as sf
import torch
print("STAGE3_IMPORTS_READY", flush=True)

ROOT = Path('samples/stage3/public_singer_preview')
SOURCE = Path('samples/stage2/expanded_c_blind/inputs')
MODEL = Path('exp/timbre_blend_stage2_embedding_resume_4600/model_7800.pt')
MD5 = '733d9c6d44071d91a1c0b0f190db44db'
CLIPS = (
    ('low', '10_侧脸_0.wav', 220),
    ('mid', '47_秋酿_19.wav', 298),
    ('high', '29_月光_13.wav', 361),
)


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def preflight():
    assert digest(MODEL, 'md5') == MD5
    mapping = json.loads(Path('reports/timbre_blend_speakers.json').read_text())['mapping']
    speakers = mapping[:12]
    assert [row['spk_id'] for row in speakers] == list(range(1, 13))
    expected = {Path(row['source_relative']).name: row for row in json.loads(
        Path('reports/timbre_blend_stage2_c_resume_holdout_inputs.json').read_text())['inputs']}
    selection = []
    for band, name, median_approx in CLIPS:
        source = SOURCE / name
        assert digest(source) == expected[name]['sha256']
        duration = sf.info(source).duration
        assert duration >= 5
        hop = 512 * sf.info(source).samplerate / 44100
        cache = Path('cache') / f'rmvpe_{hop}_50_1100_{digest(source, "md5")}.npy'
        f0 = np.load(cache, allow_pickle=False)
        positive = f0[np.isfinite(f0) & (f0 >= 65)]
        median = float(np.median(positive))
        assert abs(median - median_approx) < 5, (name, median)
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
        selection.append({'band': band, 'input': name, 'duration_seconds': duration,
                          'source_sha256': expected[name]['sha256'], 'seed': seed,
                          'interpolated_rmvpe_median_hz': round(median, 1),
                          'rmvpe_b4_voiced_fraction': expected[name]['rmvpe_b4_voiced_fraction']})
    return speakers, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    opt = parser.parse_args()
    speakers, selection = preflight()
    if opt.check_only:
        print('STAGE3_PREFLIGHT_PASS', json.dumps(selection, ensure_ascii=False), flush=True)
        return
    assert os.environ.get('PYTHONHASHSEED') == '0'
    assert torch.cuda.is_available()
    assert not ROOT.exists(), f'refusing existing Stage 3 preview: {ROOT}'
    (ROOT / 'inputs').mkdir(parents=True)
    for clip in selection:
        source = SOURCE / clip['input']
        target = ROOT / 'inputs' / (clip['band'] + '__' + clip['input'])
        shutil.copyfile(source, target)
        assert digest(target) == clip['source_sha256']
    for speaker in speakers:
        spk_id = speaker['spk_id']
        folder = ROOT / f'spk{spk_id:02d}_singer{speaker["opensinger_singer"]:02d}'
        folder.mkdir()
        for clip in selection:
            target = folder / f'{clip["band"]}.wav'
            assert not target.exists()
            print("STAGE3_JOB_START", f"spk{spk_id:02d}", clip["band"], flush=True)
            seed = clip['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            old_argv = sys.argv
            sys.argv = ['main_reflow.py', '-m', str(MODEL),
                        '-i', str(ROOT / 'inputs' / (clip['band'] + '__' + clip['input'])),
                        '-o', str(target), '-id', str(spk_id), '-ts', '0.0',
                        '-step', '50', '-method', 'euler', '-k', '0', '-f', '0',
                        '-pe', 'rmvpe', '-th', '-60', '-fmin', '50', '-fmax', '1100',
                        '-d', 'cuda', '--disable-vocoder-cudnn']
            try:
                runpy.run_path('main_reflow.py', run_name='__main__')
            finally:
                sys.argv = old_argv
            gc.collect()
            torch.cuda.empty_cache()
            audio, sr = sf.read(target)
            assert sr == 44100 and audio.ndim == 1 and np.isfinite(audio).all()
            assert np.max(np.abs(audio)) > 0
            assert abs(len(audio) / sr - clip['duration_seconds']) < .05
            if spk_id == 1 and clip["band"] == "low":
                reference, reference_sr = sf.read(".tmp/stage3_public_spk01_low_diagnostic.wav")
                assert reference_sr == sr and len(reference) == len(audio)
                relative_rms = np.sqrt(np.mean((audio - reference) ** 2) / np.mean(reference ** 2))
                assert relative_rms < .05, f"smoke output mismatch: {relative_rms}"
                print("STAGE3_SMOKE_COMPARISON_PASS", float(relative_rms), flush=True)
            print('STAGE3_OUTPUT_OK', f'spk{spk_id:02d}', clip['band'],
                  digest(target), flush=True)
    manifest = {'checkpoint': str(MODEL), 'checkpoint_md5': MD5,
                'protocol': {'t_start': 0.0, 'infer_step': 50, 'method': 'euler',
                             'key': 0, 'formant_shift': 0, 'pitch_extractor': 'rmvpe',
                             'threshold_db': -60, 'f0_min': 50, 'f0_max': 1100,
                             'vocoder_cudnn': False, 'seed_rule': 'Stage 2 SHA-256 basename first four bytes'},
                'inputs': selection, 'speakers': speakers,
                'outputs': {f'spk{spk["spk_id"]:02d}_singer{spk["opensinger_singer"]:02d}/{clip["band"]}.wav':
                            digest(ROOT / f'spk{spk["spk_id"]:02d}_singer{spk["opensinger_singer"]:02d}' /
                                   f'{clip["band"]}.wav')
                            for spk in speakers for clip in selection}}
    assert len(manifest['outputs']) == 36
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('STAGE3_PUBLIC_PREVIEW_PASS', len(manifest['outputs']), flush=True)


if __name__ == '__main__':
    main()
