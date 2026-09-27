"""Stage 2a timbre-blend tests: reflow_mask, reflow_exclude_spk, freeze_reflow,
checkpoint/optimizer compatibility and n_spk expansion migration.

All tests run on CPU with a tiny Unit2Wav and a deterministic stub vocoder
(no pretrain downloads needed). Covers plan sections 14-18 and 21:

  reflow_mask=None            -> legacy path, bit-for-bit unchanged
  reflow_mask=all True        -> equals None within float tolerance (bitwise here)
  reflow_mask=all False       -> zero scalar, right device/dtype, reflow_model
                                 never called, no reflow grads, ddsp loss intact
  reflow_mask=partial         -> loss equals subset-only loss (reduction over
                                 actual participants), excluded samples leak
                                 nothing into the reflow loss or grads
  build_reflow_mask           -> solver-side mask generation from spk_id
  freeze_reflow=false/true    -> optimizer layout + weights unchanged after step
  missing config keys         -> default to legacy behavior
  old ckpt -> frozen training -> weights restored, optimizer state explicitly
                                 discarded with a loud log, global_step kept
  n_spk N -> N+1              -> old embedding rows verbatim, new row fresh
"""

import os
import sys

import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger.utils import DotDict  # noqa: E402
from logger import utils  # noqa: E402
from reflow.solver import build_reflow_mask  # noqa: E402
from reflow.vocoder import Unit2Wav  # noqa: E402
from train_reflow import apply_freeze_reflow, build_muon_adamw_optimizer  # noqa: E402

T_START = 0.5
N_FRAMES = 8
BLOCK_SIZE = 512
MEL_BINS = 128


class _StubVocoder:
    """Deterministic differentiable vocoder stub (no NSF-HiFiGAN needed).

    extract() folds a waveform into [B, T, 128] mel-like frames so gradients
    flow from the losses back into the DDSP backbone, mirroring the real
    training graph. infer() returns silence of the right length.
    """

    dimension = MEL_BINS

    def __init__(self, block_size=BLOCK_SIZE):
        self.block_size = block_size

    def extract(self, wav):
        b, n = wav.shape
        t = n // self.block_size
        x = wav[:, : t * self.block_size].reshape(b, t, self.block_size)
        proj = (torch.arange(1, self.block_size + 1, dtype=wav.dtype,
                             device=wav.device) ** 0.5)
        x = x * proj
        return x.reshape(b, t, 4, self.block_size // 4).mean(dim=2)

    def infer(self, mel, f0):
        return torch.zeros(mel.shape[0], mel.shape[1] * self.block_size)


def _make_model(n_spk=2, seed=1234):
    torch.manual_seed(seed)
    return Unit2Wav(
        44100, BLOCK_SIZE, 2048,
        n_unit=32, n_spk=n_spk,
        use_norm=False, use_attention=False, use_pitch_aug=False,
        out_dims=MEL_BINS, n_aux_layers=1, n_aux_chans=32,
        n_layers=1, n_chans=32,
        realism_config={'enabled': False},
    )


def _make_batch(b=4, seed=7, spk_pattern=(1, 2)):
    torch.manual_seed(seed)
    units = torch.randn(b, N_FRAMES, 32)
    f0 = torch.full((b, N_FRAMES, 1), 220.0)
    volume = torch.full((b, N_FRAMES, 1), 0.5)
    spk_id = torch.tensor(
        [[spk_pattern[i % len(spk_pattern)]] for i in range(b)],
        dtype=torch.long)
    gt_spec = torch.randn(b, N_FRAMES, MEL_BINS) * 0.5
    return units, f0, volume, spk_id, gt_spec


def _train_forward(model, batch, reflow_mask=None, seed=99, vocoder=None):
    units, f0, volume, spk_id, gt_spec = batch
    torch.manual_seed(seed)
    return model(
        units, f0, volume, spk_id,
        vocoder=vocoder if vocoder is not None else _StubVocoder(),
        gt_spec=gt_spec, infer=False, t_start=T_START,
        reflow_mask=reflow_mask)


def _args_for_optimizer():
    return DotDict({'train': {'lr': 5e-4, 'weight_decay': 0.1}})


# ---------------------------------------------------------------------------
# 14.1 reflow_mask semantics
# ---------------------------------------------------------------------------

def test_mask_none_is_default_and_deterministic():
    model = _make_model()
    batch = _make_batch()
    ddsp_a, rl_a = _train_forward(model, batch, reflow_mask=None, seed=99)
    ddsp_b, rl_b = _train_forward(model, batch, seed=99)  # kwarg omitted
    assert torch.equal(ddsp_a, ddsp_b)
    assert torch.equal(rl_a, rl_b)
    assert rl_a.dim() == 0 and rl_a.item() > 0


def test_mask_all_true_equals_none():
    model = _make_model()
    batch = _make_batch()
    ddsp_none, rl_none = _train_forward(model, batch, reflow_mask=None)
    mask = torch.ones(4, dtype=torch.bool)
    ddsp_all, rl_all = _train_forward(model, batch, reflow_mask=mask)
    # same RNG consumption and identical tensors -> bitwise equality expected
    assert torch.equal(rl_none, rl_all)
    assert torch.equal(ddsp_none, ddsp_all)


def test_mask_all_false_zero_no_reflow_call_no_grads():
    model = _make_model()
    batch = _make_batch()

    # reference ddsp loss (mask must not change it)
    ddsp_ref, _ = _train_forward(model, batch, reflow_mask=None)

    # reflow_model.forward must never be invoked for an all-False mask
    def _boom(*a, **k):
        raise AssertionError('reflow_model must not be called for all-False mask')

    original_forward = model.reflow_model.forward
    model.reflow_model.forward = _boom
    try:
        ddsp_loss, rl = _train_forward(
            model, batch, reflow_mask=torch.zeros(4, dtype=torch.bool))
    finally:
        model.reflow_model.forward = original_forward

    assert rl.dim() == 0
    assert rl.item() == 0.0
    assert rl.device == ddsp_loss.device
    assert rl.dtype == ddsp_loss.dtype  # compatible scalar on batch device
    assert torch.equal(ddsp_loss, ddsp_ref)

    (ddsp_loss + rl).backward()
    for p in model.reflow_model.parameters():
        assert p.grad is None, 'reflow params must not receive gradients'
    ddsp_grads = [
        p.grad for n, p in model.ddsp_model.named_parameters()
        if p.grad is not None]
    assert len(ddsp_grads) > 0, 'ddsp backbone must still receive gradients'


def test_masked_reflow_loss_direct_subset_equivalence():
    """masked_reflow_loss == reflow_model on the subset, bitwise (same RNG).

    Verifies the reduction contract: mean over ACTUAL participants, never a
    full-batch mean scaled by the mask.
    """
    model = _make_model()
    torch.manual_seed(5)
    ddsp_mel = torch.randn(4, N_FRAMES, MEL_BINS)
    gt_spec = torch.randn(4, N_FRAMES, MEL_BINS)
    mask = torch.tensor([True, False, True, False])

    torch.manual_seed(99)
    rl_masked = model.masked_reflow_loss(ddsp_mel, gt_spec, T_START, mask)
    torch.manual_seed(99)
    rl_subset = model.reflow_model(
        ddsp_mel[mask], gt_spec=gt_spec[mask], t_start=T_START, infer=False)
    assert torch.equal(rl_masked, rl_subset)

    # sanity: the mask actually changes the loss vs the full batch
    torch.manual_seed(99)
    rl_full = model.reflow_model(
        ddsp_mel, gt_spec=gt_spec, t_start=T_START, infer=False)
    assert not torch.allclose(rl_masked, rl_full)


def test_mask_partial_full_forward_excludes_leaks():
    """Full-forward leak checks, robust to the DDSP synthesizer's internal
    randn_like noise draw (same batch shape + same seed => same RNG stream).

    Note: comparing a b=4 masked forward against a separate b=2 subset
    forward is NOT valid here because CombSubSuperFast.forward consumes
    batch-size-dependent RNG (noise exciter) before the reflow call.
    """
    model = _make_model()
    units, f0, volume, spk_id, gt_spec = _make_batch(b=4)
    mask = torch.tensor([True, False, True, False])

    _, rl_clean = _train_forward(
        model, (units, f0, volume, spk_id, gt_spec), reflow_mask=mask)

    # poison the EXCLUDED samples' gt: reflow loss must not change AT ALL
    gt_poison = gt_spec.clone()
    gt_poison[1] *= 1000.0
    gt_poison[3] += 500.0
    ddsp_poison, rl_poison = _train_forward(
        model, (units, f0, volume, spk_id, gt_poison), reflow_mask=mask)
    assert torch.equal(rl_poison, rl_clean)
    # sanity: the poison is real (ddsp loss sees gt)
    assert ddsp_poison.item() > 100.0

    # poison an INCLUDED sample: reflow loss MUST change
    gt_poison_incl = gt_spec.clone()
    gt_poison_incl[0] *= 100.0
    _, rl_poison_incl = _train_forward(
        model, (units, f0, volume, spk_id, gt_poison_incl), reflow_mask=mask)
    assert not torch.allclose(rl_poison_incl, rl_clean)

    # gradients: excluded samples contribute nothing to reflow param grads
    model.zero_grad()
    _, rl_g = _train_forward(
        model, (units, f0, volume, spk_id, gt_poison), reflow_mask=mask)
    rl_g.backward()
    grads_poison = {
        n: p.grad.clone() for n, p in model.reflow_model.named_parameters()
        if p.grad is not None}
    assert len(grads_poison) > 0

    model.zero_grad()
    _, rl_c = _train_forward(
        model, (units, f0, volume, spk_id, gt_spec), reflow_mask=mask)
    rl_c.backward()
    for n, g in grads_poison.items():
        ref = dict(model.reflow_model.named_parameters())[n].grad
        assert torch.equal(g, ref), f'reflow grad {n} affected by excluded samples'


def test_mask_validation_errors():
    model = _make_model()
    batch = _make_batch()
    with pytest.raises(ValueError):
        _train_forward(model, batch, reflow_mask=torch.ones(4))  # not bool
    with pytest.raises(ValueError):
        _train_forward(
            model, batch,
            reflow_mask=torch.ones(4, 1, dtype=torch.bool))  # wrong shape
    with pytest.raises(ValueError):
        _train_forward(
            model, batch,
            reflow_mask=torch.ones(3, dtype=torch.bool))  # wrong batch size


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required for device checks')
def test_mask_all_false_device_dtype_on_cuda():
    """Plan 14.1: the all-False zero loss must live on the batch device with a
    compatible dtype, and backward must not raise cross-device errors."""
    model = _make_model().cuda()
    batch = tuple(t.cuda() for t in _make_batch(b=2))
    units, f0, volume, spk_id, gt_spec = batch
    stub = _StubVocoder()
    mask = torch.zeros(2, dtype=torch.bool, device='cuda')
    torch.manual_seed(99)
    ddsp_loss, rl = model(
        units, f0, volume, spk_id, vocoder=stub, gt_spec=gt_spec,
        infer=False, t_start=T_START, reflow_mask=mask)
    assert rl.dim() == 0 and rl.item() == 0.0
    assert rl.device == ddsp_loss.device and rl.device.type == 'cuda'
    assert rl.dtype == ddsp_loss.dtype
    (ddsp_loss + rl).backward()  # no cross-device error
    for p in model.reflow_model.parameters():
        assert p.grad is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required for autocast checks')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_masked_reflow_under_autocast_on_cuda(dtype):
    """Stage 2 production config: partial mask + AMP autocast on GPU.

    Losses must stay finite, reflow grads must exist for included samples,
    and the all-False zero must keep device/dtype compatibility.
    """
    from torch.amp import autocast
    model = _make_model(n_spk=3).cuda()
    units, f0, volume, spk_id, gt_spec = _make_batch(b=4)
    units, f0, volume, spk_id, gt_spec = (
        t.cuda() for t in (units, f0, volume, spk_id, gt_spec))
    spk_id = torch.tensor([[1], [2], [3], [2]], device='cuda')
    stub = _StubVocoder()
    mask = torch.tensor([True, False, True, False], device='cuda')

    model.zero_grad()
    torch.manual_seed(99)
    with autocast(device_type='cuda', dtype=dtype):
        ddsp_loss, rl = model(
            units, f0, volume, spk_id, vocoder=stub, gt_spec=gt_spec,
            infer=False, t_start=T_START, reflow_mask=mask)
    assert torch.isfinite(ddsp_loss) and torch.isfinite(rl)
    (ddsp_loss + rl).backward()
    gsum = sum(
        p.grad.abs().sum().item()
        for p in model.reflow_model.parameters() if p.grad is not None)
    assert gsum > 0, 'included samples must produce reflow grads under autocast'

    with autocast(device_type='cuda', dtype=dtype):
        _, rl0 = model(
            units, f0, volume, spk_id, vocoder=stub, gt_spec=gt_spec,
            infer=False, t_start=T_START,
            reflow_mask=torch.zeros(4, dtype=torch.bool, device='cuda'))
    assert rl0.item() == 0.0 and rl0.device.type == 'cuda'


# ---------------------------------------------------------------------------
# 15. solver-side mask generation
# ---------------------------------------------------------------------------

def test_build_reflow_mask():
    spk = torch.tensor([[1], [2], [3], [2]], dtype=torch.long)
    assert build_reflow_mask(spk, []) is None
    assert build_reflow_mask(spk, None) is None
    mask = build_reflow_mask(spk, [2])
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True, False]
    # 1-D spk_id layout also supported
    mask1d = build_reflow_mask(spk.reshape(-1), [2, 3])
    assert mask1d.tolist() == [True, False, False, False]
    # device preserved
    assert build_reflow_mask(spk, [2]).device == spk.device


# ---------------------------------------------------------------------------
# 16. freeze_reflow
# ---------------------------------------------------------------------------

def _optimizer_param_ids(opt):
    return {id(p) for g in opt.param_groups for p in g['params']}


def test_freeze_reflow_false_keeps_legacy_optimizer():
    from ddsp.realism_stages import configure_training_stage
    model = _make_model()
    configure_training_stage(model, 'timbre')
    opt = build_muon_adamw_optimizer(model, _args_for_optimizer(), freeze_reflow=False)
    ids = _optimizer_param_ids(opt)
    reflow_ids = {id(p) for p in model.reflow_model.parameters()}
    assert reflow_ids <= ids, 'legacy layout: reflow params must be in optimizer'
    assert ids == {id(p) for p in model.parameters()}


def test_freeze_reflow_true_excludes_and_never_updates():
    from ddsp.realism_stages import configure_training_stage
    model = _make_model()
    configure_training_stage(model, 'timbre')  # production order: stage first...
    n_tensors, n_params = apply_freeze_reflow(model)  # ...then freeze
    assert n_tensors > 0 and n_params > 0
    assert all(not p.requires_grad for p in model.reflow_model.parameters())

    opt = build_muon_adamw_optimizer(model, _args_for_optimizer(), freeze_reflow=True)
    ids = _optimizer_param_ids(opt)
    reflow_ids = {id(p) for p in model.reflow_model.parameters()}
    assert not (ids & reflow_ids), 'frozen reflow params must not enter optimizer'
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert ids == trainable_ids

    frozen_before = {
        n: p.detach().clone() for n, p in model.reflow_model.named_parameters()}
    backbone_before = {
        n: p.detach().clone() for n, p in model.ddsp_model.named_parameters()
        if p.requires_grad}

    # a real step with the reflow loss ACTIVE (mask None): grads flow through
    # the frozen net into the backbone, but reflow weights must not move at all
    batch = _make_batch(b=2)
    ddsp_loss, rl = _train_forward(model, batch)
    (ddsp_loss + rl).backward()
    for p in model.reflow_model.parameters():
        assert p.grad is None
    opt.step()

    for n, p in model.reflow_model.named_parameters():
        assert torch.equal(p.detach(), frozen_before[n]), \
            f'reflow param {n} changed despite freeze_reflow'
    changed = sum(
        1 for n, p in model.ddsp_model.named_parameters()
        if p.requires_grad and not torch.equal(p.detach(), backbone_before[n]))
    assert changed > 0, 'non-reflow trainable params must keep training'


# ---------------------------------------------------------------------------
# config defaults (plan 18: missing keys must fall back to old behavior)
# ---------------------------------------------------------------------------

def test_missing_config_keys_default_to_legacy():
    args = DotDict({'train': {}})
    exclude = list(args.train.get('reflow_exclude_spk') or [])
    freeze = bool(args.train.get('freeze_reflow', False))
    assert exclude == []
    assert freeze is False
    spk = torch.tensor([[1], [2]], dtype=torch.long)
    assert build_reflow_mask(spk, exclude) is None


def test_reflow_yaml_declares_legacy_defaults():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'reflow.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg['train']['reflow_exclude_spk'] == []
    assert cfg['train']['freeze_reflow'] is False
    assert cfg['model']['realism']['enabled'] is False


# ---------------------------------------------------------------------------
# 17. old checkpoint -> freeze_reflow training path
# ---------------------------------------------------------------------------

def _save_stage1_ckpt(tmp_path, with_optimizer=True, n_spk=2, step=42):
    """Train-model-and-save exactly like a Stage 1 run with save_opt=true."""
    from ddsp.realism_stages import configure_training_stage
    model = _make_model(n_spk=n_spk, seed=1234)
    configure_training_stage(model, 'timbre')
    opt = build_muon_adamw_optimizer(model, _args_for_optimizer(), freeze_reflow=False)
    batch = _make_batch(b=2)
    ddsp_loss, rl = _train_forward(model, batch)
    (ddsp_loss + rl).backward()
    opt.step()  # populate optimizer state (momentum buffers)
    ckpt = {'global_step': step, 'model': model.state_dict()}
    if with_optimizer:
        ckpt['optimizer'] = opt.state_dict()
    torch.save(ckpt, os.path.join(str(tmp_path), f'model_{step}.pt'))
    return model, opt


def test_old_ckpt_into_frozen_model_discards_optimizer_loudly(tmp_path, capsys):
    from ddsp.realism_stages import configure_training_stage
    stage1_model, stage1_opt = _save_stage1_ckpt(tmp_path)

    # Stage 2: fresh model, freeze_reflow=true -> different optimizer layout
    model2 = _make_model(seed=999)
    configure_training_stage(model2, 'timbre')
    apply_freeze_reflow(model2)
    opt2 = build_muon_adamw_optimizer(model2, _args_for_optimizer(), freeze_reflow=True)

    step, model2, opt2 = utils.load_model(
        str(tmp_path), model2, opt2, device='cpu', load_optimizer=True)
    out = capsys.readouterr().out

    assert step == 42, 'global_step must be preserved'
    # 1-3: every weight restored exactly (embeddings, reflow, ddsp)
    s1 = stage1_model.state_dict()
    s2 = model2.state_dict()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f'weight {k} not restored'
    # 4: frozen stays frozen
    assert all(not p.requires_grad for p in model2.reflow_model.parameters())
    # 7-9: optimizer state explicitly discarded, loudly, never pretended
    assert 'NOT restored' in out
    assert 'fresh optimizer' in out
    sd = opt2.state_dict()
    for sub in sd['optimizers']:
        assert len(sub['state']) == 0, 'discarded optimizer must stay empty'
    # 5: training can continue
    batch = _make_batch(b=2)
    ddsp_loss, rl = _train_forward(model2, batch)
    (ddsp_loss + rl).backward()
    opt2.step()


def test_matching_layout_optimizer_fully_restored(tmp_path, capsys):
    from ddsp.realism_stages import configure_training_stage
    stage1_model, stage1_opt = _save_stage1_ckpt(tmp_path)

    model2 = _make_model(seed=999)
    configure_training_stage(model2, 'timbre')
    opt2 = build_muon_adamw_optimizer(model2, _args_for_optimizer(), freeze_reflow=False)
    step, model2, opt2 = utils.load_model(
        str(tmp_path), model2, opt2, device='cpu', load_optimizer=True)
    out = capsys.readouterr().out
    assert step == 42
    assert 'fully restored' in out
    ref = stage1_opt.state_dict()
    got = opt2.state_dict()
    for sub_ref, sub_got in zip(ref['optimizers'], got['optimizers']):
        assert len(sub_got['state']) == len(sub_ref['state'])
        assert len(sub_got['state']) > 0
        for k in sub_ref['state']:
            for field, tensor in sub_ref['state'][k].items():
                if torch.is_tensor(tensor):
                    assert torch.equal(sub_got['state'][k][field], tensor)


def test_ckpt_without_optimizer_starts_fresh(tmp_path, capsys):
    from ddsp.realism_stages import configure_training_stage
    _save_stage1_ckpt(tmp_path, with_optimizer=False)
    model2 = _make_model(seed=999)
    configure_training_stage(model2, 'timbre')
    opt2 = build_muon_adamw_optimizer(model2, _args_for_optimizer(), freeze_reflow=False)
    step, _, _ = utils.load_model(
        str(tmp_path), model2, opt2, device='cpu', load_optimizer=True)
    out = capsys.readouterr().out
    assert step == 42
    assert 'none in checkpoint' in out


# ---------------------------------------------------------------------------
# 21. n_spk expansion N -> N+1
# ---------------------------------------------------------------------------

def test_spk_embed_expansion_preserves_old_rows(tmp_path):
    stage1_model, _ = _save_stage1_ckpt(tmp_path, with_optimizer=False, n_spk=2)
    old_embed = stage1_model.ddsp_model.unit2ctrl.spk_embed.weight.detach().clone()
    old_state = {k: v.clone() for k, v in stage1_model.state_dict().items()}

    model3 = _make_model(n_spk=3, seed=999)
    fresh_row = model3.ddsp_model.unit2ctrl.spk_embed.weight.detach().clone()[2]

    step, model3, _ = utils.load_model(
        str(tmp_path), model3, None, device='cpu', load_optimizer=False)
    assert step == 42

    new_embed = model3.ddsp_model.unit2ctrl.spk_embed.weight.detach()
    # old rows verbatim (exact equality), new row keeps its fresh init
    assert torch.equal(new_embed[:2], old_embed)
    assert torch.equal(new_embed[2], fresh_row)
    # every other tensor restored exactly from the Stage 1 checkpoint
    new_state = model3.state_dict()
    for k, v in old_state.items():
        if k.endswith('spk_embed.weight'):
            continue
        assert torch.equal(new_state[k], v), f'{k} not restored'


def test_spk_embed_shrink_raises(tmp_path):
    _save_stage1_ckpt(tmp_path, with_optimizer=False, n_spk=3, step=7)
    model2 = _make_model(n_spk=2, seed=999)
    with pytest.raises(RuntimeError, match='shrinking'):
        utils.load_model(
            str(tmp_path), model2, None, device='cpu', load_optimizer=False)


def test_other_shape_mismatch_still_raises(tmp_path):
    """Expansion must not mask unrelated incompatibilities."""
    _save_stage1_ckpt(tmp_path, with_optimizer=False, n_spk=2)
    ckpt_path = os.path.join(str(tmp_path), 'model_42.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    key = 'reflow_model.velocity_fn.net.0.0.weight'  # any non-embedding tensor
    found = None
    for k in ckpt['model']:
        if k.startswith('reflow_model.') and ckpt['model'][k].dim() >= 1:
            found = k
            ckpt['model'][k] = torch.randn(3)
            break
    assert found is not None, 'test needs at least one reflow tensor in ckpt'
    torch.save(ckpt, ckpt_path)
    model2 = _make_model(n_spk=2, seed=999)
    with pytest.raises(RuntimeError):
        utils.load_model(
            str(tmp_path), model2, None, device='cpu', load_optimizer=False)


# ---------------------------------------------------------------------------
# 15.1 validation metric split through solver.test()
# ---------------------------------------------------------------------------

class _StubSaver:
    def __init__(self):
        self.values = []

    def log_value(self, d):
        self.values.append(dict(d))

    def log_spec(self, *a, **k):
        pass

    def log_audio(self, *a, **k):
        pass


def _make_val_loader(tmp_path, exclude_target_spk=2):
    """Two val batches (batch_size=1 like the real loader): spk1 and spk2."""
    import soundfile as sf
    os.makedirs(os.path.join(str(tmp_path), 'audio'), exist_ok=True)
    loader = []
    for i, spk in enumerate((1, 2)):
        units, f0, volume, spk_id, gt_spec = _make_batch(b=1, seed=11 + i, spk_pattern=(spk,))
        name = f'val{i}'
        sf.write(
            os.path.join(str(tmp_path), 'audio', f'{name}.wav'),
            torch.zeros(N_FRAMES * BLOCK_SIZE).numpy(), 44100)
        loader.append({
            'name': [name],
            'name_ext': [f'{name}.wav'],
            'units': units, 'f0': f0, 'volume': volume,
            'spk_id': spk_id, 'mel': gt_spec,
        })
    return loader


def _solver_args(tmp_path, exclude):
    return DotDict({
        'device': 'cpu',
        'data': {'sampling_rate': 44100, 'valid_path': str(tmp_path)},
        'model': {'t_start': T_START},
        'infer': {'infer_step': 2, 'method': 'euler'},
        'train': {'reflow_exclude_spk': exclude},
    })


def test_solver_test_splits_included_and_diagnostic(tmp_path):
    from reflow.solver import test as solver_test
    model = _make_model()
    loader = _make_val_loader(tmp_path)
    saver = _StubSaver()
    ddsp_loss, reflow_loss = solver_test(
        _solver_args(tmp_path, [2]), model, _StubVocoder(), loader, saver)

    logged = {}
    for d in saver.values:
        logged.update(d)
    assert 'validation/reflow_loss_included' in logged
    assert 'validation/reflow_loss_excluded_diagnostic' in logged
    # the returned (optimization-facing) metric is the INCLUDED one
    assert reflow_loss == logged['validation/reflow_loss_included']
    # included averages only the spk1 batch; diagnostic only the spk2 batch;
    # with one batch each they must both be finite and generally different
    assert reflow_loss > 0
    assert logged['validation/reflow_loss_excluded_diagnostic'] > 0
    assert isinstance(ddsp_loss, float) and ddsp_loss > 0


def test_solver_test_legacy_without_exclude(tmp_path):
    from reflow.solver import test as solver_test
    model = _make_model()
    loader = _make_val_loader(tmp_path)
    saver = _StubSaver()
    _, reflow_loss = solver_test(
        _solver_args(tmp_path, []), model, _StubVocoder(), loader, saver)
    logged = {}
    for d in saver.values:
        logged.update(d)
    # legacy behavior: no split keys, reflow loss over all batches
    assert 'validation/reflow_loss_included' not in logged
    assert 'validation/reflow_loss_excluded_diagnostic' not in logged
    assert reflow_loss > 0


def test_solver_test_all_excluded_reports_zero(tmp_path):
    from reflow.solver import test as solver_test
    model = _make_model()
    loader = _make_val_loader(tmp_path)
    saver = _StubSaver()
    _, reflow_loss = solver_test(
        _solver_args(tmp_path, [1, 2]), model, _StubVocoder(), loader, saver)
    logged = {}
    for d in saver.values:
        logged.update(d)
    assert reflow_loss == 0.0
    assert logged['validation/reflow_loss_included'] == 0.0
    assert logged['validation/reflow_loss_excluded_diagnostic'] > 0
