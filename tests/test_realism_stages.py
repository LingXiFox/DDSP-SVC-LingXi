import torch.nn as nn

from ddsp.realism_stages import (
    configure_training_stage,
    split_realism_parameters,
)


class DummyDDSP(nn.Module):
    def __init__(self, with_realism=True):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.realism = (
            nn.Linear(4, 2)
            if with_realism
            else None
        )


class DummyModel(nn.Module):
    def __init__(self, with_realism=True):
        super().__init__()
        self.ddsp_model = DummyDDSP(
            with_realism=with_realism
        )
        self.reflow_model = nn.Linear(4, 4)


def test_timbre_stage_freezes_only_realism():
    model = DummyModel()
    configure_training_stage(model, "timbre")
    backbone, realism = split_realism_parameters(model)
    assert all(p.requires_grad for p in backbone)
    assert all(not p.requires_grad for p in realism)


def test_personalize_stage_freezes_backbone():
    model = DummyModel()
    configure_training_stage(
        model,
        "realism_personalize",
    )
    backbone, realism = split_realism_parameters(model)
    assert all(not p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in realism)


def test_joint_stage_enables_both_groups():
    model = DummyModel()
    configure_training_stage(model, "joint")
    backbone, realism = split_realism_parameters(model)
    assert all(p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in realism)


def test_realism_stage_requires_enabled_adapter():
    model = DummyModel(with_realism=False)
    try:
        configure_training_stage(
            model,
            "realism_personalize",
        )
    except RuntimeError as exc:
        assert "enabled=true" in str(exc)
    else:
        raise AssertionError(
            "expected missing realism adapter to fail"
        )
