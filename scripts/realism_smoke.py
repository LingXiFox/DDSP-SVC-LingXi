import argparse
import tempfile
from pathlib import Path

import torch

from ddsp.realism import VocalRealismAdapter
from ddsp.realism_features import (
    realism_control_loss,
    robotize_controls,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic CPU/CUDA smoke test for "
            "the Vocal Realism adapter."
        )
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--n-unit", type=int, default=768)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if (
        device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false"
        )

    torch.manual_seed(20260918)
    units = torch.randn(
        2,
        args.frames,
        args.n_unit,
        device=device,
    )
    t = torch.linspace(
        0,
        1,
        args.frames,
        device=device,
    ).reshape(1, -1, 1)

    target_f0 = 220.0 * torch.pow(
        2.0,
        (
            10.0
            * torch.sin(
                2 * torch.pi * 5.5 * t
            )
        )
        / 1200.0,
    )
    target_f0 = target_f0.repeat(2, 1, 1)
    target_f0[:, :5] = 0
    target_volume = (
        0.45
        + 0.08
        * torch.sin(
            2 * torch.pi * 2.5 * t
        )
    ).repeat(2, 1, 1)

    input_f0, input_volume = robotize_controls(
        target_f0,
        target_volume,
        smoothing_kernel=9,
    )
    model = VocalRealismAdapter(
        n_unit=args.n_unit
    ).to(device)

    with torch.no_grad():
        zero_f0, zero_volume, diagnostics = model(
            units,
            input_f0,
            input_volume,
        )
        if (
            torch.count_nonzero(
                diagnostics["delta_f0_cents"]
            )
            != 0
        ):
            raise AssertionError(
                "zero-init F0 residual is not zero"
            )
        if (
            torch.count_nonzero(
                diagnostics["delta_volume_db"]
            )
            != 0
        ):
            raise AssertionError(
                "zero-init volume residual is not zero"
            )
        if not torch.equal(zero_f0, input_f0):
            raise AssertionError(
                "zero-init F0 path is not identity"
            )
        if not torch.equal(
            zero_volume,
            input_volume,
        ):
            raise AssertionError(
                "zero-init volume path is not identity"
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )
    initial_loss = None
    final_loss = None

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        pred_f0, pred_volume, _ = model(
            units,
            input_f0,
            input_volume,
        )
        loss, _ = realism_control_loss(
            pred_f0,
            pred_volume,
            target_f0,
            target_volume,
            trend_kernel=31,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {step}: {loss}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )
        if not torch.isfinite(
            torch.as_tensor(grad_norm)
        ):
            raise FloatingPointError(
                f"non-finite grad norm at step {step}"
            )
        optimizer.step()

        if initial_loss is None:
            initial_loss = float(
                loss.detach().cpu()
            )
        final_loss = float(
            loss.detach().cpu()
        )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "realism_smoke.pt"
        torch.save(
            {"model": model.state_dict()},
            path,
        )
        clone = VocalRealismAdapter(
            n_unit=args.n_unit
        ).to(device)
        state = torch.load(
            path,
            map_location=device,
        )["model"]
        clone.load_state_dict(
            state,
            strict=True,
        )

    peak_mb = 0.0
    if device.startswith("cuda"):
        peak_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    print(
        "Vocal Realism smoke PASS "
        f"device={device} "
        f"steps={args.steps} "
        f"initial_loss={initial_loss:.6f} "
        f"final_loss={final_loss:.6f} "
        f"peak_cuda_mb={peak_mb:.1f}"
    )


if __name__ == "__main__":
    main()
