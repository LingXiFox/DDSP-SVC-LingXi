"""Disabled-path structural bypass tests (no checkpoint needed).

Proves CombSubSuperFastRealism with realism disabled is a pure
pass-through to the upstream CombSubSuperFast code path:
no adapter instance, no extra params, bit-identical forward outputs.
"""

import torch
import torch.nn as nn

from ddsp.realism_vocoder import CombSubSuperFastRealism
from ddsp.vocoder import CombSubSuperFast
from reflow.vocoder import Unit2Wav

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


def test_wrapper_exposes_the_controls_used_by_ddsp():
    _, wrapped = _fresh_pair()
    inputs = _make_inputs()
    sig, hidden, adapted_f0, adapted_volume = (
        wrapped.forward_with_adapted_controls(*inputs)
    )
    assert sig is not None
    assert hidden is not None
    assert torch.equal(adapted_f0, inputs[1])
    assert torch.equal(adapted_volume, inputs[2])


def test_enabled_wrapper_exposes_adapted_f0():
    torch.manual_seed(1234)
    wrapped = CombSubSuperFastRealism(
        **_ARGS, realism_config={"enabled": True, "checkpoint": None}
    )
    assert wrapped.realism is not None
    with torch.no_grad():
        wrapped.realism.output_proj.bias[0] = 1.0
    inputs = _make_inputs()
    _, _, adapted_f0, _ = wrapped.forward_with_adapted_controls(
        *inputs
    )
    voiced = inputs[1] > 0
    assert torch.all(adapted_f0[voiced] > inputs[1][voiced])
    assert torch.equal(adapted_f0[~voiced], inputs[1][~voiced])


class _FakeDdsp(nn.Module):
    def forward_with_adapted_controls(
        self, units, f0, volume, **kwargs
    ):
        adapted_f0 = f0 + 17.0
        wav = torch.zeros(f0.shape[0], 1, f0.shape[1] * 2)
        return wav, torch.zeros_like(units), adapted_f0, volume


class _FakeReflow(nn.Module):
    def forward(self, ddsp_mel, **kwargs):
        return ddsp_mel


class _RecordingVocoder:
    def __init__(self):
        self.f0 = None

    def extract(self, audio):
        return torch.zeros(audio.shape[0], audio.shape[-1] // 2, 4)

    def infer(self, mel, f0):
        self.f0 = f0
        return torch.zeros(mel.shape[0], 1, mel.shape[1] * 2)


def test_adapted_f0_reaches_final_vocoder():
    model = Unit2Wav.__new__(Unit2Wav)
    nn.Module.__init__(model)
    model.sampling_rate = 2
    model.block_size = 2
    model.ddsp_model = _FakeDdsp()
    model.reflow_model = _FakeReflow()
    vocoder = _RecordingVocoder()
    units, f0, volume, spk_id = _make_inputs()

    model(
        units,
        f0,
        volume,
        spk_id=spk_id,
        vocoder=vocoder,
        return_wav=True,
        use_tqdm=False,
    )

    assert torch.equal(vocoder.f0, f0 + 17.0)
