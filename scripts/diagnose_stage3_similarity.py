"""Read-only Stage-3 diagnosis: native virtual recordings and existing public conversions."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from scripts.eval_stage2_similarity import MODEL_FILES, describe, select

OUT = Path('reports/timbre_blend_stage3_diagnostics.json')
SONGS = {'train': ('00', '01'), 'val': ('05', '08')}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists(), f'refusing existing report: {OUT}'
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    gate = json.loads(Path('reports/timbre_blend_stage3_recipe_screen.json').read_text())
    preview_root = Path('samples/stage3/public_singer_preview')
    preview = json.loads((preview_root / 'manifest.json').read_text())
    assert gate['status'] == 'GATE_COMPLETE' and len(gate['results']) == 12
    assert gate['plan']['checkpoint_md5'] == preview['checkpoint_md5'] == '733d9c6d44071d91a1c0b0f190db44db'
    assert [row['band'] for row in preview['inputs']] == ['low', 'mid', 'high']
    assert [row['spk_id'] for row in preview['speakers']] == list(range(1, 13))
    for clip in preview['inputs']:
        assert next(x for x in gate['plan']['inputs'] if x['band'] == clip['band']) == clip
    raw_rows = []
    for split, song_ids in SONGS.items():
        source = Path('data/timbre_blend_stage2') / split / 'audio/13_lingxi'
        for song in song_ids:
            candidates = sorted((p for p in source.glob(f'r13_{song}_*.wav')
                                 if sf.info(p).duration >= 3),
                                key=lambda p: (-sf.info(p).duration, p.name))
            chosen = candidates[:3]
            assert len(chosen) == 3 and all(re.fullmatch(r'r13_\d{2}_.+\.wav', p.name) for p in chosen)
            for p in chosen:
                originals = [q for base in ('train', 'val')
                             for q in (Path('datasets/private-timbre/sliced-v2') / base / 'audio' / p.name,)
                             if q.is_file() and sha(q) == sha(p)]
                assert len(originals) == 1, f'original source not byte-identical: {p.name}'
                raw_rows.append({'song_id': song, 'stage2_split': split, 'source': str(originals[0]),
                                 'sha256': sha(p), 'duration_seconds': sf.info(p).duration})
    assert len(raw_rows) == 12 and len({x['song_id'] for x in raw_rows}) == 4
    all_train = select(Path('data/timbre_blend_stage2/train/audio'))
    public = {k: v for k, v in all_train.items() if k != '13_lingxi'}
    assert len(public) == 12
    names = {i: next(k for k in public if k.startswith(f'{i}_')) for i in range(1, 13)}
    model_root = Path.home() / 'work/eval-models/ecapa-voxceleb-0f99f2d'
    for name, expected in MODEL_FILES.items():
        assert sha(model_root / name) == expected, name
    torch.set_num_threads(2)
    classifier = SpeakerRecognition.from_hparams(source=str(model_root),
        hparams_file='hyperparams_eval.yaml', savedir=str(model_root / 'runtime'))

    @torch.inference_mode()
    def embed(path):
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
        audio = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, sr, 16000)
        return torch.nn.functional.normalize(classifier.encode_batch(audio).reshape(-1), dim=0)

    refs = {i: [embed(wav) for _, wavs in public[names[i]] for wav in wavs]
            for i in range(1, 13)}
    assert all(len(x) == 4 for x in refs.values())
    raw_embeds = [embed(row['source']) for row in raw_rows]
    native_scores = {str(i): {
        'mean': float(np.mean([float(torch.dot(a, b)) for a in raw_embeds for b in refs[i]])),
        'by_song': {song: float(np.mean([float(torch.dot(raw_embeds[j], b))
                       for j, row in enumerate(raw_rows) if row['song_id'] == song for b in refs[i]]))
                    for song in ('00', '01', '05', '08')},
        'opensinger_singer': int(names[i].split('singer')[1])}
        for i in range(1, 13)}
    pair_same, pair_different = [], []
    grouped_same, grouped_different = [], []
    converted_rows = []
    for sp in preview['speakers']:
        i = sp['spk_id']
        wavs = [preview_root / f"spk{i:02d}_singer{sp['opensinger_singer']:02d}" / f"{band}.wav"
                for band in ('low', 'mid', 'high')]
        embeds = []
        for wav in wavs:
            rel = wav.relative_to(preview_root).as_posix()
            assert sha(wav) == preview['outputs'][rel], rel
            audio_info = sf.info(wav)
            source = next(x for x in preview['inputs'] if x['band'] == wav.stem)
            assert audio_info.samplerate == 44100 and abs(audio_info.duration - source['duration_seconds']) < .05
            embeds.append(embed(wav))
        by_public = {}
        for j in range(1, 13):
            values = [float(torch.dot(a, b)) for a in embeds for b in refs[j]]
            assert len(values) == 12
            mean = float(np.mean(values))
            by_public[str(j)] = mean
            (pair_same if i == j else pair_different).extend(values)
            (grouped_same if i == j else grouped_different).append(mean)
        converted_rows.append({'target_spk_id': i, 'opensinger_singer': sp['opensinger_singer'],
                               'output_sha256': {p.stem: sha(p) for p in wavs},
                               'by_public_spk_mean': by_public})
        print('CONVERSION_DIAG', i, 'self', round(by_public[str(i)], 6),
              'other_max', round(max(v for key, v in by_public.items() if key != str(i)), 6), flush=True)
    assert len(pair_same) == 144 and len(pair_different) == 1584
    assert len(grouped_same) == 12 and len(grouped_different) == 132
    report = {'protocol': {'model': gate['model'], 'revision': gate['model_revision'],
               'model_file_sha256': MODEL_FILES, 'public_reference': gate['reference_protocol'],
               'native_selection': 'Two Stage-2 training songs 00,01 and two validation songs 05,08; each three longest >=3s unchanged original sliced-v2 human-vocal WAVs (filename tie break). All original source SHA-256 match Stage-2 copies. No model outputs used for native baseline.',
               'native_aggregation': 'For each public Y, equal mean of 12 original source segments x 4 reference segments = 48 cosine scores.',
               'converted_aggregation': 'For each target X and reference singer Y, equal mean of three outputs x four references = 12 cosine scores. Pairwise distribution counts each cosine; grouped distribution counts each X,Y mean once.',
               'stage3_source': 'Existing C@7800 Stage-3 public preview WAVs; no new inference.'},
              'native_virtual': {'segments': raw_rows, 'by_public_spk': native_scores},
              'pure_public_conversion': {'outputs': converted_rows,
                   'same_singer_pairwise': describe(pair_same),
                   'different_singer_pairwise': describe(pair_different),
                   'same_singer_grouped': describe(grouped_same),
                   'different_singer_grouped': describe(grouped_different)},
              'existing_grid_report': 'reports/timbre_blend_stage3_recipe_screen.json',
              'caveat': 'ECAPA is a speech-domain proxy on singing; original singer recordings and model conversions differ in source-content distribution. No threshold is approved by this diagnostic.'}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('DIAG_COMPLETE', OUT, 'native', len(raw_rows), 'converted', len(converted_rows),
          'same', report['pure_public_conversion']['same_singer_pairwise'],
          'different', report['pure_public_conversion']['different_singer_pairwise'],
          'grouped_different', report['pure_public_conversion']['different_singer_grouped'], flush=True)


if __name__ == '__main__':
    main()
