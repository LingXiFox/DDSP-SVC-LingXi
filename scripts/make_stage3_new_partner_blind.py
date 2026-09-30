"""Package only 11-clip gated recipes with a separate random-name decoding key."""
import hashlib
import json
import secrets
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('samples/stage3/new_partner_blind')
SOURCE = Path('samples/stage2/expanded_c_blind/inputs')
AUDIO = Path('samples/stage3/new_partner_11')
KEY = Path('.tmp/stage3_new_partner_blind_key.json')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not ROOT.exists() and not KEY.exists()
    gate = json.loads(Path('reports/timbre_blend_stage3_new_partner_11.json').read_text())
    screening = json.loads(Path('reports/timbre_blend_stage3_new_partner_screen.json').read_text())
    assert gate['checkpoint_md5'] == screening['checkpoint_md5'] == '733d9c6d44071d91a1c0b0f190db44db'
    assert set(gate['passing_recipe_ids']) <= set(screening['passing_recipe_ids'])
    assert gate['passing_recipe_ids'] == ['v100_t000', 'v060_t000', 'v040_t000']
    assert len(gate['inputs']) == 11 and all(r['passes'] for r in gate['results'])
    # The pure virtual recipe is a reference, not a third blend option.
    blend_ids = [i for i in gate['passing_recipe_ids'] if i != 'v100_t000']
    assert len(blend_ids) == 2
    mapping = []
    taken = set()
    for row in gate['inputs']:
        original = SOURCE / row['input']
        assert sha(original) == row['sha256']
        for arm in gate['passing_recipe_ids']:
            evaluated = next(x for x in gate['results'] if x['recipe']['id'] == arm)
            src = AUDIO / arm / row['input']
            assert sha(src) == evaluated['output_sha256'][row['input']]
            info = sf.info(src)
            assert info.samplerate == 44100 and info.channels == 1
            assert abs(info.duration - row['seconds']) < .05
            random_name = secrets.token_hex(12) + '.wav'
            assert random_name not in taken
            taken.add(random_name)
            mapping.append({'input': row['input'], 'output': random_name, 'arm': arm,
                            'sha256': evaluated['output_sha256'][row['input']]})
    assert len(mapping) == len(taken) == 33
    ROOT.mkdir(parents=True)
    (ROOT / 'inputs').mkdir()
    (ROOT / 'audio').mkdir()
    for row in gate['inputs']:
        source = SOURCE / row['input']
        dest = ROOT / 'inputs' / row['input']
        shutil.copyfile(source, dest)
        assert sha(dest) == row['sha256']
    for item in mapping:
        dest = ROOT / 'audio' / item['output']
        shutil.copyfile(AUDIO / item['arm'] / item['input'], dest)
        assert sha(dest) == item['sha256']
    rng = secrets.SystemRandom()
    pairs = {}
    for row in gate['inputs']:
        labels = [item['output'] for item in mapping if item['input'] == row['input']]
        rng.shuffle(labels)
        assert len(labels) == 3 and len(set(labels)) == 3
        pairs[row['input']] = labels
    (ROOT / 'pairs.json').write_text(json.dumps({'pairs': pairs,
        'instruction': 'For each original input compare its three randomly named outputs. Mapping to blend/control is stored separately, outside this package.'},
        ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'README.md').write_text('# 阶段 3 · 新伙伴 11 段盲听\n\n'
        '对每条 `inputs/` 中的原片，在 `pairs.json` 找到三个 `audio/` 随机名文件进行比较。'
        '其中两组为通过三段初筛和 11 段复核的融合配方，一组为纯虚拟歌手对照。'
        '音频文件名和此包不揭晓配方；解盲表单独保存，请先凭耳朵选择。'
        '所有输入、输出都是公开原片转换，不含私有训练音频或模型权重。\n')
    with KEY.open('x') as stream:
        stream.write(json.dumps({'checkpoint_md5': gate['checkpoint_md5'],
            'blends': blend_ids, 'control': 'v100_t000',
            'mapping': mapping, 'input_count': len(gate['inputs'])}, ensure_ascii=False, indent=2) + '\n')
    print('BLIND_PREPARED',len(pairs),'inputs',len(mapping),'audio',
          'key_outside_pack',KEY,flush=True)


if __name__ == '__main__':
    main()
