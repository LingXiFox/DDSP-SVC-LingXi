"""Build the reusable four-input, offline listening console from gated outputs."""
import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path

import soundfile as sf

CONFIG = Path('configs/timbre_blend_stage3_listening_set.json')
GATE = Path('reports/timbre_blend_stage3_new_partner_11.json')
OUTPUTS = Path('samples/stage3/new_partner_11')
INPUTS = Path('samples/stage2/expanded_c_blind/inputs')
ROOT = Path('samples/stage3/new_partner_listening_4')
TEMPLATE = Path('scripts/stage3_listening_index.html')
KEY = Path('.tmp/stage3_listening_4_key.json')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_private(path, contents):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(contents)


def main():
    assert not ROOT.exists() and not KEY.exists()
    config = json.loads(CONFIG.read_text())
    gate = json.loads(GATE.read_text())
    source_selection = json.loads(Path(config['source_selection']).read_text())
    assert config['id'] == 'stage3_listening_v1' and len(config['clips']) == 4
    assert gate['checkpoint_md5'] == '733d9c6d44071d91a1c0b0f190db44db'
    assert len(gate['inputs']) == 11 and gate['comparison_count_per_public_singer'] == 44
    assert gate['passing_recipe_ids'] == ['v100_t000', 'v060_t000', 'v040_t000']
    control = 'v100_t000'
    blends = [name for name in gate['passing_recipe_ids'] if name != control]
    assert len(blends) == 2
    assert all(row['passes'] and not row['a_failed_ids'] and not row['b_failed_ids'] for row in gate['results'])
    arms = blends + [control]
    secrets.SystemRandom().shuffle(arms)
    label_to_arm = dict(zip('ABC', arms))
    pack_id = 'stage3-four-' + secrets.token_hex(6)
    rows, private_rows, public_hashes = [], [], {}
    for clip in config['clips']:
        name = clip['input']
        src = INPUTS / name
        original = next(row for row in source_selection['inputs'] if Path(row['file']).name == name)
        check = next(row for row in gate['inputs'] if row['input'] == name)
        assert clip['sha256'] == original['sha256'] == check['sha256'] == sha(src)
        assert sf.info(src).duration > 4
        public_hashes['inputs/' + name] = clip['sha256']
        outputs = {}
        for label, arm in label_to_arm.items():
            evaluated = next(row for row in gate['results'] if row['recipe']['id'] == arm)
            audio = OUTPUTS / arm / name
            assert sha(audio) == evaluated['output_sha256'][name]
            info = sf.info(audio)
            assert info.samplerate == 44100 and info.channels == 1
            assert abs(info.duration - check['seconds']) < .05
            alias = secrets.token_hex(12) + '.wav'
            relative = 'audio/' + alias
            assert relative not in public_hashes
            public_hashes[relative] = sha(audio)
            outputs[label] = {'relative': relative, 'source': audio, 'arm': arm}
        rows.append({'role': clip['role'], 'label': clip['label'], 'input': name,
                     'audio': {'O': 'inputs/' + name, **{label: x['relative'] for label, x in outputs.items()}}})
        private_rows.append({'input': name, 'roles': clip['role'],
                             'labels': {label: {'arm': x['arm'], 'filename': x['relative'],
                                                'sha256': public_hashes[x['relative']]}
                                        for label, x in outputs.items()}})
    assert len({x['input'] for x in rows}) == 4 and len(public_hashes) == 16
    assert len(set(public_hashes)) == 16
    ROOT.mkdir(parents=True)
    (ROOT / 'audio').mkdir()
    (ROOT / 'inputs').mkdir()
    for row in rows:
        dst = ROOT / row['audio']['O']
        shutil.copyfile(INPUTS / row['input'], dst)
        assert sha(dst) == public_hashes[row['audio']['O']]
    for item in private_rows:
        for data in item['labels'].values():
            dst = ROOT / data['filename']
            shutil.copyfile(OUTPUTS / data['arm'] / item['input'], dst)
            assert sha(dst) == data['sha256']
    payload = {'pack_id': pack_id, 'listening_set_id': config['id'], 'rows': rows}
    template = TEMPLATE.read_text()
    assert template.count('__EMBEDDED_JSON__') == 1
    serialized = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    (ROOT / 'index.html').write_text(template.replace('__EMBEDDED_JSON__', serialized))
    (ROOT / 'manifest.json').write_text(json.dumps({
        'pack_id': pack_id, 'listening_set_id': config['id'],
        'checkpoint_md5': gate['checkpoint_md5'],
        'input_sha256': {name: digest for name, digest in public_hashes.items() if name.startswith('inputs/')},
        'audio_count': 12,
        'note': 'Only three objectively gated groups on four fixed listening inputs; label-to-arm mapping is stored outside this package.'
    }, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'README.md').write_text('# 四段听感 · 离线盲听包\n\n'
        '双击 `index.html`；不要改动 `audio/` 和 `inputs/` 的相对位置。'
        '每行点击原声或 A/B/C 可在相同播放位置切换，默认循环；投票与备注后点击“导出投票 JSON”。'
        '导出文件只保存在你的电脑上，之后交给本狐解盲统计。'
        '四段听感测试集固定在项目配置中；11 段全集已用于客观相似度复核，不会全部放进盲听包。'
        '气声与长音均为测量候选标签，未经人耳事先确证。'
        '没有外部字体、脚本、服务器或网络请求；此包不含解盲表。\n')
    key = {'pack_id': pack_id, 'listening_set_id': config['id'],
           'label_to_arm': label_to_arm, 'inputs': private_rows,
           'source_gate': str(GATE), 'source_checkpoint_md5': gate['checkpoint_md5']}
    write_private(KEY, json.dumps(key, ensure_ascii=False, indent=2) + '\n')
    print('LISTENING_PACK_READY',pack_id,'rows',len(rows),'anonymous_audio',12,
          'key_outside_pack',KEY,flush=True)


if __name__ == '__main__':
    main()
