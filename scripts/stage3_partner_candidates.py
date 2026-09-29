"""Stage 3 gate 2: real public-singer matrix and six existing C@7800 previews."""
import hashlib
import json
import os
import shutil
from itertools import combinations, product
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from scripts.eval_stage2_similarity import MODEL_FILES, select

REPORT = Path('reports/timbre_blend_stage3_public_matrix.json')
PACK = Path('samples/stage3/partner_candidate_preview')
PREVIEW = Path('samples/stage3/public_singer_preview')


def sha(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    assert not REPORT.exists() and not PACK.exists()
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    old = json.loads((PREVIEW / 'manifest.json').read_text())
    corrected = json.loads(Path('reports/timbre_blend_stage3_corrected_standard.json').read_text())
    assert sha(Path(old['checkpoint']), 'md5') == old['checkpoint_md5'] == '733d9c6d44071d91a1c0b0f190db44db'
    assert old['protocol']['vocoder_cudnn'] is False and old['protocol']['t_start'] == 0
    assert [r['spk_id'] for r in old['speakers']] == list(range(1, 13))
    assert [r['band'] for r in old['inputs']] == ['low', 'mid', 'high']
    all_train = select(Path('data/timbre_blend_stage2/train/audio'))
    public = {k: v for k, v in all_train.items() if k != '13_lingxi'}
    names = {i: next(k for k in public if k.startswith(f'{i}_')) for i in range(1, 13)}
    assert len(public) == len(set(names.values())) == 12 and names[9] == '9_singer33'
    model = Path.home() / 'work/eval-models/ecapa-voxceleb-0f99f2d'
    for name, digest in MODEL_FILES.items():
        assert sha(model / name) == digest
    torch.set_num_threads(2)
    classifier = SpeakerRecognition.from_hparams(source=str(model),
        hparams_file='hyperparams_eval.yaml', savedir=str(model / 'runtime'))

    @torch.inference_mode()
    def embed(path):
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        wave = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        wave = torchaudio.functional.resample(wave, sr, 16000)
        return torch.nn.functional.normalize(classifier.encode_batch(wave).reshape(-1), dim=0)

    refs = {i: [[embed(wav) for wav in files] for _, files in public[names[i]]]
            for i in range(1, 13)}
    assert all(len(songs) == 2 and all(len(files) == 2 for files in songs) for songs in refs.values())
    matrix = [[0.0] * 12 for _ in range(12)]
    for i in range(1, 13):
        a, b = refs[i]
        matrix[i - 1][i - 1] = float(np.mean([float(torch.dot(u, v)) for u, v in product(a, b)]))
    for i, j in combinations(range(1, 13), 2):
        values = [float(torch.dot(a, b)) for a, b in product(
            [v for song in refs[i] for v in song], [v for song in refs[j] for v in song])]
        matrix[i - 1][j - 1] = matrix[j - 1][i - 1] = float(np.mean(values))
    assert len(matrix) == 12 and all(len(row) == 12 for row in matrix)
    real_mean = float(np.mean([matrix[i - 1][j - 1] for i, j in combinations(range(1, 13), 2)]))
    assert abs(real_mean - corrected['calibration']['real_different_singer_mean']) < 1e-9
    # Protect against proximity to EITHER high-risk singer, not only their average.
    candidate_ids = sorted((i for i in range(1, 13) if i not in (10, 12)),
                           key=lambda i: (max(matrix[i - 1][9], matrix[i - 1][11]),
                                          (matrix[i - 1][9] + matrix[i - 1][11]) / 2, i))
    rows = [{'rank': rank, 'spk_id': i,
             'opensinger_singer': next(s['opensinger_singer'] for s in old['speakers'] if s['spk_id'] == i),
             'to_spk10': matrix[i - 1][9], 'to_spk12': matrix[i - 1][11],
             'worst_of_two': max(matrix[i - 1][9], matrix[i - 1][11])}
            for rank, i in enumerate(candidate_ids, 1)]
    chosen = rows[:6]
    # Verify every source WAV before creating the new listening package.
    files = {}
    for clip in old['inputs']:
        src = PREVIEW / 'inputs' / (clip['band'] + '__' + clip['input'])
        assert sha(src) == clip['source_sha256']
        files[f"inputs/{src.name}"] = src
    for row in chosen:
        folder = f"spk{row['spk_id']:02d}_singer{row['opensinger_singer']:02d}"
        for clip in old['inputs']:
            rel = f'{folder}/{clip["band"]}.wav'
            src = PREVIEW / rel
            assert sha(src) == old['outputs'][rel]
            info = sf.info(src)
            assert info.samplerate == 44100 and info.channels == 1
            assert abs(info.duration - clip['duration_seconds']) < .05
            files[rel] = src
    assert len(files) == 21
    report = {'stage': '3_step_2', 'status': 'WAITING_FOR_USER_3_TO_4_SINGERS',
              'public_spk_ids': list(range(1, 13)),
              'opensinger_mapping': {str(row['spk_id']): row['opensinger_singer'] for row in old['speakers']},
              'protocol': 'Same offline CPU ECAPA model and Stage-3 corrected public reference selection: each singer 2 songs x 2 clips. Off diagonal: mean of 4x4 = 16 cross-singer cosines. Diagonal: mean of 2x2 cross-song same-singer cosines, never self-file comparisons.',
              'model_file_sha256': MODEL_FILES,
              'matrix_row_column_order': list(range(1, 13)), 'matrix': matrix,
              'real_different_singer_mean_check': real_mean,
              'candidate_ranking': 'Exclude spk10 and spk12. Sort by max(similarity to spk10, similarity to spk12) ascending; tie break by average of the two then spk_id. This minimizes the worse of the two risks.',
              'all_eligible_ranked': rows, 'top_six': chosen,
              'preview': {'checkpoint': old['checkpoint'], 'md5': old['checkpoint_md5'],
                          'protocol': old['protocol'], 'inputs': old['inputs'],
                          'note': 'Copied previously rendered C@7800 pure-singer outputs byte-for-byte; no new inference or mixing.',
                          'output_sha256': {rel: sha(src) for rel, src in files.items()}},
              'caveat': 'Speech-trained ECAPA on singing is a proxy for human listening, not an identity verdict. Rank is based on raw public references, not conversion scores.'}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    PACK.mkdir(parents=True)
    for rel, src in files.items():
        dest = PACK / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        assert sha(dest) == report['preview']['output_sha256'][rel]
    (PACK / 'manifest.json').write_text(json.dumps({k: report[k] for k in
        ('stage', 'status', 'candidate_ranking', 'top_six', 'preview')}, ensure_ascii=False, indent=2) + '\n')
    instructions = ['# 六位候选公开歌手 · 试听', '',
        '根据与 spk10、spk12 的真人参考片相似度中**较高**的一项由低到高排序；筛除 spk10、spk12。每位有同样三条原片的 low/mid/high 纯音色转换。请从六位中挑选 **3～4 位**，回复 spk 编号；本狐在此停下。', '',
        '| 名次 | 编号 | OpenSinger | 对 spk10 | 对 spk12 | 两者较高值 |',
        '|---:|---|---:|---:|---:|---:|']
    for row in chosen:
        instructions.append(f"| {row['rank']} | spk{row['spk_id']:02d} | {row['opensinger_singer']} | {row['to_spk10']:.6f} | {row['to_spk12']:.6f} | {row['worst_of_two']:.6f} |")
    instructions += ['', '这里直接复用 C@7800 已经生成、核验 SHA-256 的纯音色输出（不是新融合配方）。三条原片在 `inputs/`，参数及每条输出哈希在 `manifest.json`；没有私有音频或模型权重。完整 12×12 真人相似度矩阵见独立报告。ECAPA 只是歌唱音频的代理分数，选音色仍请以试听为主。']
    (PACK / 'README.md').write_text('\n'.join(instructions) + '\n')
    print('MATRIX_PASS', real_mean, flush=True)
    for row in chosen:
        print('CANDIDATE', row['rank'], row['spk_id'], row['opensinger_singer'],
              row['to_spk10'], row['to_spk12'], row['worst_of_two'], flush=True)
    print('PREVIEW_PASS', len(files), 'audio_files', flush=True)


if __name__ == '__main__':
    main()
