"""Stage-2 replay balance and fixed validation RNG without changing legacy config."""
import random
from unittest.mock import patch

import numpy as np
import torch

from logger.utils import DotDict
from reflow import data_loaders, solver


def test_fixed_validation_seed_restores_training_rng():
    args = DotDict({'device': 'cpu', 'train': {'val_seed': 123}})

    def metric(*_):
        return {'ddsp_loss': torch.rand(()).item() + np.random.rand() + random.random()}

    torch.manual_seed(9)
    np.random.seed(9)
    random.seed(9)
    expected = (torch.rand(()).item(), np.random.rand(), random.random())
    torch.manual_seed(9)
    np.random.seed(9)
    random.seed(9)
    with patch.object(solver, '_test_impl', side_effect=metric):
        first = solver.test(args, None, None, None, None, return_metrics=True)
        second = solver.test(args, None, None, None, None, return_metrics=True)
    actual = (torch.rand(()).item(), np.random.rand(), random.random())
    assert first == second
    assert actual == expected


def test_weighted_replay_balances_virtual_to_half():
    class FakeDataset:
        def __init__(self, root, **_):
            self.paths = ([f'1_public/file{i}' for i in range(1900)]
                          + [f'13_lingxi/file{i}' for i in range(100)]) if root == 'train' else ['1_public/val']

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.paths[i]

    args = DotDict({'data': {'train_path': 'train', 'valid_path': 'valid',
                             'duration': 2, 'block_size': 512, 'sampling_rate': 44100,
                             'extensions': ['wav']},
                    'model': {'n_spk': 13},
                    'train': {'batch_size': 48, 'cache_all_data': True,
                              'cache_device': 'cpu', 'cache_fp16': False,
                              'num_workers': 0, 'virtual_replay_fraction': 0.5,
                              'virtual_spk_id': 13, 'seed': 20260927}})
    with patch.object(data_loaders, 'AudioDataset', FakeDataset):
        loader, _ = data_loaders.get_data_loaders(args)
        selected = list(loader.sampler)
        virtual_count = sum(i >= 1900 for i in selected)
        assert 900 <= virtual_count <= 1100
