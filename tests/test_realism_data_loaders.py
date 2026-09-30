from pathlib import Path

import numpy as np

from reflow.realism_data_loaders import RealismControlDataset


def test_control_dataset_never_requires_mel_or_augmented_targets(
    tmp_path: Path,
):
    root = tmp_path / "public"
    for sub in ("audio", "units", "f0", "volume"):
        (root / sub).mkdir(
            parents=True,
            exist_ok=True,
        )

    name = "sample.wav"
    (root / "audio" / name).write_bytes(b"")
    frames = 32
    np.save(
        root / "units" / f"{name}.npy",
        np.random.randn(frames, 8).astype("float32"),
    )
    np.save(
        root / "f0" / f"{name}.npy",
        np.linspace(200, 220, frames).astype("float32"),
    )
    np.save(
        root / "volume" / f"{name}.npy",
        np.linspace(0.3, 0.6, frames).astype("float32"),
    )

    ds = RealismControlDataset(
        str(root),
        waveform_sec=0.2,
        hop_size=100,
        sample_rate=1000,
        extensions=("wav",),
    )
    item = ds[0]

    assert set(item) == {
        "units",
        "f0",
        "volume",
        "name_ext",
        "name",
    }
    assert item["units"].shape == (2, 8)
    assert item["f0"].shape == (2, 1)
    assert item["volume"].shape == (2, 1)
