"""Bake the selected Stage-3 mix as a new row and a one-row local-only model."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from logger.utils import SPK_EMBED_KEY

SOURCE = Path('exp/timbre_blend_stage2_embedding_resume_4600/model_7800.pt')
FULL_DIR = Path('exp/timbre_blend_stage3_baked_full')
SLIM_DIR = Path('exp/timbre_blend_stage3_final_slim')
RECIPE = 'v040_t000'
MIX = ((13, .4), (4, .2), (8, .2), (11, .2))


def sha(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    assert not FULL_DIR.exists() and not SLIM_DIR.exists()
    assert sha(SOURCE, 'md5') == '733d9c6d44071d91a1c0b0f190db44db'
    passed = json.loads(Path('reports/timbre_blend_stage3_new_partner_11.json').read_text())
    votes = json.loads(Path('reports/timbre_blend_stage3_listening_4_votes.json').read_text())
    assert RECIPE in passed['passing_recipe_ids']
    assert votes['wins_by_arm'][RECIPE] == 2
    row = next(r['recipe'] for r in passed['results'] if r['recipe']['id'] == RECIPE)
    assert row['t_start'] == 0 and row['mix'] == {str(i): weight for i, weight in MIX}
    ckpt = torch.load(SOURCE, map_location='cpu', weights_only=True)
    source_state = ckpt['model']
    embed = source_state[SPK_EMBED_KEY]
    assert embed.shape == (13, 512) and embed.dtype == torch.float32
    assert [key for key in source_state if 'spk_embed' in key] == [SPK_EMBED_KEY]
    baked = torch.zeros_like(embed[0])
    for i, weight in MIX:
        baked = baked + weight * embed[i - 1]
    assert torch.isfinite(baked).all() and baked.shape == (512,)
    full_state = dict(source_state)
    full_state[SPK_EMBED_KEY] = torch.cat((embed, baked.unsqueeze(0)))
    slim_state = dict(source_state)
    slim_state[SPK_EMBED_KEY] = baked.unsqueeze(0).clone()
    assert torch.equal(full_state[SPK_EMBED_KEY][:13], embed)
    assert torch.equal(full_state[SPK_EMBED_KEY][-1], slim_state[SPK_EMBED_KEY][0])
    assert all(torch.equal(v, slim_state[k]) for k, v in source_state.items() if k != SPK_EMBED_KEY)
    assert sum(weight for _, weight in MIX) == 1.0
    original_config = yaml.safe_load((SOURCE.parent / 'config.yaml').read_text())
    assert original_config['model']['n_spk'] == 13
    config = {k: copy.deepcopy(original_config[k]) for k in ('data', 'infer', 'model', 'vocoder')}
    config['data'].pop('train_path', None)
    config['data'].pop('valid_path', None)
    FULL_DIR.mkdir(parents=True)
    SLIM_DIR.mkdir(parents=True)
    for dirname, n_spk, state, filename in (
        (FULL_DIR, 14, full_state, 'model_baked.pt'),
        (SLIM_DIR, 1, slim_state, 'model_slim.pt'),
    ):
        local_config = copy.deepcopy(config)
        local_config['model']['n_spk'] = n_spk
        (dirname / 'config.yaml').write_text(yaml.safe_dump(local_config, allow_unicode=True, sort_keys=False))
        torch.save({'global_step': ckpt['global_step'], 'model': state}, dirname / filename)
        loaded = torch.load(dirname / filename, map_location='cpu', weights_only=True)['model']
        assert set(loaded) == set(source_state)
        assert loaded[SPK_EMBED_KEY].shape == (n_spk, 512)
        assert torch.equal(loaded[SPK_EMBED_KEY][-1], baked)
        assert all(torch.equal(value, loaded[key]) for key, value in source_state.items() if key != SPK_EMBED_KEY)
        print('BAKED_LOCAL_MODEL',dirname / filename,'n_spk',n_spk,
              'sha256',sha(dirname / filename),flush=True)
    print('BAKE_PASS', RECIPE, 'baked_row', 14, 'slim_row', 1, flush=True)


if __name__ == '__main__':
    main()
