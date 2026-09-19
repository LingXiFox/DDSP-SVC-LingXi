import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from reflow.data_loaders import traverse_dir


class RealismControlDataset(Dataset):
    """Control-only dataset for the public Vocal Realism prior.

    It intentionally never reads waveform, mel, speaker ID, or timbre targets.
    Only preprocessed content units, F0 and frame loudness are exposed.
    """

    def __init__(
        self,
        path_root: str,
        waveform_sec: float,
        hop_size: int,
        sample_rate: int,
        extensions=("wav",),
        whole_audio: bool = False,
    ):
        super().__init__()
        self.path_root = path_root
        self.crop_len = int(waveform_sec * sample_rate / hop_size)
        self.whole_audio = whole_audio
        candidates = traverse_dir(
            os.path.join(path_root, "audio"),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True,
        )
        self.items: List[Dict[str, object]] = []
        for name_ext in candidates:
            unit_path = os.path.join(path_root, "units", name_ext) + ".npy"
            f0_path = os.path.join(path_root, "f0", name_ext) + ".npy"
            volume_path = os.path.join(path_root, "volume", name_ext) + ".npy"
            if not all(
                os.path.exists(p)
                for p in (unit_path, f0_path, volume_path)
            ):
                continue
            units_len = np.load(unit_path, mmap_mode="r").shape[0]
            f0_len = np.load(f0_path, mmap_mode="r").shape[0]
            volume_len = np.load(volume_path, mmap_mode="r").shape[0]
            frame_len = min(units_len, f0_len, volume_len)
            if not whole_audio and frame_len < self.crop_len:
                continue
            self.items.append(
                {
                    "name_ext": name_ext,
                    "unit_path": unit_path,
                    "f0_path": f0_path,
                    "volume_path": volume_path,
                    "frame_len": frame_len,
                }
            )
        if not self.items:
            raise RuntimeError(
                f"No usable control triplets found under {path_root!r}; "
                "run the normal DDSP-SVC preprocessing first."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        frame_len = int(item["frame_len"])
        length = frame_len if self.whole_audio else self.crop_len
        start = (
            0
            if self.whole_audio
            else random.randint(0, frame_len - length)
        )
        end = start + length

        units = np.load(
            item["unit_path"], mmap_mode="r"
        )[start:end].copy()
        f0 = np.load(
            item["f0_path"], mmap_mode="r"
        )[start:end].copy()
        volume = np.load(
            item["volume_path"], mmap_mode="r"
        )[start:end].copy()

        units = torch.from_numpy(units).float()
        f0 = torch.from_numpy(f0).float().reshape(-1, 1)
        volume = torch.from_numpy(volume).float().reshape(-1, 1)
        name_ext = str(item["name_ext"])
        return {
            "units": units,
            "f0": f0,
            "volume": volume,
            "name_ext": name_ext,
            "name": os.path.splitext(name_ext)[0],
        }


def get_realism_data_loaders(args):
    cfg = args.realism_train
    train_data = RealismControlDataset(
        cfg.train_path,
        waveform_sec=cfg.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        extensions=args.data.extensions,
        whole_audio=False,
    )
    valid_data = RealismControlDataset(
        cfg.valid_path,
        waveform_sec=cfg.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        extensions=args.data.extensions,
        whole_audio=False,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader
