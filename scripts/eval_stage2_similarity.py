"""Offline ECAPA cosine proxy, calibrated with cross-song OpenSinger examples."""
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

MODEL_FILES = {
    'hyperparams.yaml': '6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41',
    'embedding_model.ckpt': '0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2',
    'mean_var_norm_emb.ckpt': 'cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33',
    'classifier.ckpt': 'fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535',
    'label_encoder.txt': 'e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9',
}

def song(path):
    stem = re.sub(r'_*s\d+$', '', path.stem)
    return re.sub(r'_\d+$', '', stem)

def choose_two_audio(files):
    available = [p for p in files if sf.info(str(p)).duration >= 3]
    return sorted(available, key=lambda p: (-sf.info(str(p)).duration, str(p)))[:2]

def select(root, limit_speakers=None, limit_songs=2):
    selected = {}
    for speaker in sorted(p for p in root.iterdir() if p.is_dir()):
        by_song = defaultdict(list)
        for wav in speaker.glob('*.wav'):
            by_song[song(wav)].append(wav)
        choices = {key: choose_two_audio(wavs) for key, wavs in sorted(by_song.items())}
        songs = [(key, files) for key, files in choices.items() if len(files) == 2][:limit_songs]
        if len(songs) == limit_songs:
            selected[speaker.name] = songs
        if limit_speakers and len(selected) >= limit_speakers:
            break
    return selected

def describe(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return {'count': int(len(a)), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'std': float(a.std()), 'p05': float(np.percentile(a,5)),
            'p25': float(np.percentile(a,25)), 'p75': float(np.percentile(a,75)),
            'p95': float(np.percentile(a,95))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--c-dir', type=Path)
    opt=ap.parse_args()
    assert os.environ.get('HF_HUB_OFFLINE')=='1'
    assert os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')=='1'
    root=Path.home()/'work/eval-models/ecapa-voxceleb-0f99f2d'
    for filename, expected in MODEL_FILES.items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected, filename
    torch.set_num_threads(2)
    classifier=SpeakerRecognition.from_hparams(source=str(root),
                    hparams_file='hyperparams_eval.yaml', savedir=str(root/'runtime'))
    @torch.inference_mode()
    def embed(path):
        audio,sr=sf.read(str(path),dtype='float32',always_2d=True)
        a=torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        a=torchaudio.functional.resample(a,sr,16000)
        a=classifier.encode_batch(a).reshape(-1)
        return torch.nn.functional.normalize(a,dim=0)
    cache={}
    def embedding(path):
        if path not in cache:cache[path]=embed(path)
        return cache[path]
    def cosine(a,b):
        return float(torch.dot(embedding(a),embedding(b)))
    public=select(Path('data/timbre_blend_stage2/train/audio'),limit_speakers=12)
    assert len(public)==12 and all(len(songs)==2 for songs in public.values())
    same=[];different=[]
    for singer,songs in public.items():
        same.extend(cosine(a,b) for a,b in product(songs[0][1],songs[1][1]))
    for (singer_a,songs_a),(singer_b,songs_b) in combinations(public.items(),2):
        # Cross-speaker, cross-song (each singer: two distinct songs, two clips/song).
        different.extend(cosine(a,b) for a,b in product(
            [p for _,files in songs_a for p in files],
            [p for _,files in songs_b for p in files]))
    refs=select(Path('data/timbre_blend_stage2/val/audio'),limit_speakers=None)
    assert '13_lingxi' in refs and len(refs['13_lingxi'])==2
    private=[wav for _,files in refs['13_lingxi'] for wav in files]
    def group_outputs(name,dir,filenames=None):
        rows=[]
        for singer in (10,47):
            path=dir/(filenames[singer] if filenames else f'singer{singer}__to_spk13.wav')
            assert path.is_file(),path
            values=[cosine(path,ref) for ref in private]
            rows.append({'input_singer':singer,'similarity_to_virtual_val':describe(values)})
        return {'name':name,'per_input':rows,
                'all_output_to_virtual_val':describe([
                  cosine(dir/(filenames[singer] if filenames else f'singer{singer}__to_spk13.wav'),ref)
                  for singer in (10,47) for ref in private])}
    if opt.c_dir:
        key=json.loads(Path('.tmp/stage2_c_vs_b1k_blind_key.json').read_text())
        groups=[]
        for name,label in (('B_1k','B_1k'),('C_best','C_3000')):
            filenames={int(item['input_singer']):filename
                       for filename,item in key['files'].items() if item['group']==label}
            assert set(filenames)=={10,47}
            groups.append(group_outputs(name,opt.c_dir,filenames))
    else:
        groups=[group_outputs('B_1k',Path('samples/stage2/probe_1k'))]
    result={'model':'speechbrain/spkrec-ecapa-voxceleb',
            'revision':'0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',
            'license':'Apache-2.0','speechbrain_version':'1.1.1',
            'embedding':'ECAPA 192-dimensional, resampled to 16kHz, cosine',
            'public_reference':{'speakers':len(public),'songs_per_speaker':2,
                                'clips_per_song':2,'same_speaker_different_song':describe(same),
                                'different_speaker':describe(different)},
            'virtual_reference':{'songs':2,'clips':len(private),'split':'validation only'},
            'groups':groups,
            'caveat':'Speech-trained speaker model on sung audio: proxy only; no speech threshold reused.'}
    out=Path('.tmp/stage2_similarity_preliminary.json' if not opt.c_dir else
             'reports/timbre_blend_stage2_similarity.json')
    assert not out.exists(),f'refusing to overwrite {out}'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('SIMILARITY_REPORT',out,'PUBLIC',result['public_reference'])
    for group in groups:print('SIMILARITY_GROUP',group)

if __name__=='__main__':
    main()
