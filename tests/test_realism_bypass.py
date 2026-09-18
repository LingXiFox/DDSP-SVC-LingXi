"""Disabled-path structural bypass tests (no checkpoint needed).

Proves CombSubSuperFastRealism with realism disabled is a pure
pass-through to the upstream CombSubSuperFast code path:
no adapter instance, no extra params, bit-identical forward outputs.
"""

import torch

from ddsp.realism_vocoder import CombSubSuperFastRealism
from ddsp.vocoder import CombSubSuperFast

_ARGS = dict(
    sampling_rate=44100,
    block_size=512,
    win_length=2048,
    n_unit=32,
    n_spk=1,
    num_layers=1,
    dim_model=32,
)


def _make_inputs(device="cpu"):
    torch.manual_seed(7)
    units = torch.randn(1, 8, _ARGS["n_unit"], device=device)
    f0 = torch.full((1, 8, 1), 220.0, device=device)
    f0[:, :2] = 0.0
    volume = torch.full((1, 8, 1), 0.5, device=device)
    spk_id = torch.zeros(1, 1, dtype=torch.long, device=device)
    return units, f0, volume, spk_id


def _fresh_pair():
    torch.manual_seed(1234)
    base = CombSubSuperFast(**_ARGS)
    torch.manual_seed(1234)
    wrapped = CombSubSuperFastRealism(
        **_ARGS, realism_config={"enabled": False}
    )
    return base, wrapped


def test_disabled_wrapper_instantiates_no_adapter():
    _, wrapped = _fresh_pair()
    assert wrapped.realism_enabled is False
    assert wrapped.realism is None


def test_disabled_wrapper_adds_no_parameters():
    base, wrapped = _fresh_pair()
    base_state = base.state_dict()
    wrapped_state = wrapped.state_dict()
    assert set(wrapped_state) == set(base_state)
    for key in base_state:
        assert torch.equal(wrapped_state[key], base_state[key])


def _forward(model, inputs, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available() and inputs[0].is_cuda:
        torch.cuda.manual_seed_all(seed)
    model.eval()
    with torch.no_grad():
        return model(*inputs)


def test_disabled_forward_matches_upstream():
    base, wrapped = _fresh_pair()
    inputs = _make_inputs()
    sig_base, hidden_base = _forward(base, inputs, seed=99)
    sig_wrap, hidden_wrap = _forward(wrapped, inputs, seed=99)
    assert torch.equal(sig_base, sig_wrap)
    assert torch.equal(hidden_base, hidden_wrap)


def test_enabled_zero_init_matches_disabled_output():
    _, wrapped_disabled = _fresh_pair()
    torch.manual_seed(1234)
    wrapped_enabled = CombSubSuperFastRealism(
        **_ARGS, realism_config={"enabled": True, "checkpoint": None}
    )
    inputs = _make_inputs()
    sig_dis, hidden_dis = _forward(wrapped_disabled, inputs, seed=99)
    sig_en, hidden_en = _forward(wrapped_enabled, inputs, seed=99)
    assert torch.equal(sig_dis, sig_en)
    assert torch.equal(hidden_dis, hidden_en)
