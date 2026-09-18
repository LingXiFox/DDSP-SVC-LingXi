import torch

from ddsp.realism_features import (
    extract_realism_features,
    realism_control_loss,
    robotize_controls,
)


def _controls(batch=2, frames=64):
    t = torch.linspace(0, 1, frames).reshape(1, frames, 1)
    vibrato = 8.0 * torch.sin(2 * torch.pi * 6 * t)
    f0 = 220.0 * torch.pow(2.0, vibrato / 1200.0)
    f0 = f0.repeat(batch, 1, 1)
    f0[:, :3] = 0
    volume = (
        0.4 + 0.1 * torch.sin(2 * torch.pi * 3 * t)
    ).repeat(batch, 1, 1)
    return f0, volume


def test_feature_extractor_is_register_invariant():
    f0, volume = _controls()
    shifted = f0 * 2.0
    base = extract_realism_features(
        f0, volume, trend_kernel=15
    )
    octave = extract_realism_features(
        shifted, volume, trend_kernel=15
    )
    mask = base["voiced"] & octave["voiced"]
    assert torch.allclose(
        base["pitch_residual_cents"][mask],
        octave["pitch_residual_cents"][mask],
        atol=2e-3,
        rtol=1e-4,
    )


def test_robotizer_preserves_voicing_and_reduces_micro_pitch_motion():
    f0, volume = _controls()
    robot_f0, robot_volume = robotize_controls(
        f0, volume, smoothing_kernel=9
    )
    assert torch.equal(robot_f0 == 0, f0 == 0)
    assert torch.isfinite(robot_volume).all()

    original = extract_realism_features(
        f0, volume, trend_kernel=15
    )
    robot = extract_realism_features(
        robot_f0, robot_volume, trend_kernel=15
    )
    original_energy = (
        original["pitch_velocity_cents"].abs().mean()
    )
    robot_energy = (
        robot["pitch_velocity_cents"].abs().mean()
    )
    assert robot_energy < original_energy


def test_realism_loss_is_zero_for_identical_controls_and_backward_safe():
    f0, volume = _controls()
    pred_f0 = f0.clone().requires_grad_(True)
    pred_volume = volume.clone().requires_grad_(True)
    loss, parts = realism_control_loss(
        pred_f0,
        pred_volume,
        f0,
        volume,
        trend_kernel=15,
    )
    assert loss.item() < 1e-8
    loss.backward()
    assert pred_f0.grad is not None
    assert pred_volume.grad is not None
    assert all(
        torch.isfinite(value)
        for value in parts.values()
    )


def test_loss_ignores_global_octave_shift_but_detects_shape_change():
    f0, volume = _controls()
    shifted = f0 * 2.0
    invariant_loss, _ = realism_control_loss(
        shifted,
        volume,
        f0,
        volume,
        trend_kernel=15,
    )

    warped = f0.clone()
    warped[:, 20:30] *= 1.03
    warped_loss, _ = realism_control_loss(
        warped,
        volume,
        f0,
        volume,
        trend_kernel=15,
    )

    assert invariant_loss.item() < 1e-3
    assert (
        warped_loss.item()
        > invariant_loss.item() + 1e-2
    )


def test_constant_volume_is_not_distorted_at_sequence_edges():
    f0, _ = _controls(batch=1, frames=32)
    volume = torch.full_like(f0, 0.5)
    _, robot_volume = robotize_controls(
        f0,
        volume,
        smoothing_kernel=9,
    )
    assert torch.allclose(
        robot_volume,
        volume,
        atol=1e-6,
        rtol=1e-6,
    )
