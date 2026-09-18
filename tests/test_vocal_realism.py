import torch

from ddsp.realism import VocalRealismAdapter


def _inputs(batch=2, frames=24, n_unit=16):
    torch.manual_seed(7)
    units = torch.randn(batch, frames, n_unit)
    f0 = torch.rand(batch, frames, 1) * 300 + 100
    f0[:, :3] = 0
    volume = torch.rand(batch, frames, 1)
    return units, f0, volume


def test_zero_init_is_exact_identity():
    units, f0, volume = _inputs()
    model = VocalRealismAdapter(n_unit=units.shape[-1])
    f0_out, volume_out, diag = model(units, f0, volume)
    assert torch.equal(f0_out, f0)
    assert torch.equal(volume_out, volume)
    assert torch.count_nonzero(diag["delta_f0_cents"]) == 0
    assert torch.count_nonzero(diag["delta_volume_db"]) == 0


def test_residuals_are_bounded_and_unvoiced_stays_zero():
    units, f0, volume = _inputs()
    model = VocalRealismAdapter(
        n_unit=units.shape[-1], max_f0_cents=20.0, max_volume_db=2.0)
    with torch.no_grad():
        model.output_proj.bias[:] = torch.tensor([10.0, -10.0])
    f0_out, _, diag = model(units, f0, volume)
    assert torch.max(torch.abs(diag["delta_f0_cents"])) <= 20.0 + 1e-5
    assert torch.max(torch.abs(diag["delta_volume_db"])) <= 2.0 + 1e-5
    assert torch.equal(f0_out[f0 == 0], f0[f0 == 0])


def test_backward_reaches_adapter_parameters():
    units, f0, volume = _inputs()
    model = VocalRealismAdapter(n_unit=units.shape[-1])
    f0_out, volume_out, _ = model(units, f0, volume)
    loss = f0_out.mean() + volume_out.mean()
    loss.backward()
    assert model.output_proj.weight.grad is not None
    assert torch.isfinite(model.output_proj.weight.grad).all()
    assert model.output_proj.weight.grad.abs().sum() > 0
