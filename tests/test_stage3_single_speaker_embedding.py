"""Keep legacy single-voice behavior while loading a baked one-row embedding."""

import torch
from torch import nn

from ddsp.unit2control import Unit2Control


def test_single_voice_uses_baked_row_only_when_present():
    torch.manual_seed(7)
    model = Unit2Control(4, 2, 1, {'voice': 1}, num_layers=0, dim_model=16)
    model.decoder = nn.Identity()
    model.norm = nn.Identity()
    model.dense_out = nn.Linear(16, 1, bias=False)
    with torch.no_grad():
        model.dense_out.weight.zero_()
        model.dense_out.weight[0, 0] = 1
    units = torch.randn(1, 5, 4)
    source = torch.randn(1, 5, 2)
    noise = torch.randn(1, 5, 2)
    volume = torch.randn(1, 5, 1)
    speaker = torch.ones(1, 1, dtype=torch.long)

    assert not hasattr(model, 'spk_embed')  # Legacy one-speaker checkpoints.
    old_controls, old_hidden = model(units, source, noise, volume, spk_id=speaker)
    model.spk_embed = nn.Embedding(1, 16)
    with torch.no_grad():
        model.spk_embed.weight.zero_()
        model.spk_embed.weight[0, 0] = .75
    new_controls, new_hidden = model(units, source, noise, volume, spk_id=speaker)

    assert torch.allclose(new_hidden[..., 0] - old_hidden[..., 0], torch.full((1, 5), .75))
    assert torch.equal(new_hidden[..., 1:], old_hidden[..., 1:])
    assert torch.allclose(new_controls['voice'] - old_controls['voice'], torch.full((1, 5, 1), .75))
