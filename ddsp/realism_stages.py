from typing import Dict, List, Tuple

import torch.nn as nn


VALID_REALISM_STAGES = ("timbre", "realism_personalize", "joint")


def get_realism_module(model: nn.Module):
    ddsp_model = getattr(model, "ddsp_model", None)
    return (
        getattr(ddsp_model, "realism", None)
        if ddsp_model is not None
        else None
    )


def split_realism_parameters(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    realism = get_realism_module(model)
    realism_ids = (
        set()
        if realism is None
        else {id(p) for p in realism.parameters()}
    )
    backbone = []
    realism_params = []
    for param in model.parameters():
        if id(param) in realism_ids:
            realism_params.append(param)
        else:
            backbone.append(param)
    return backbone, realism_params


def configure_training_stage(
    model: nn.Module,
    stage: str,
) -> Dict[str, int]:
    """Apply the R2 parameter-freezing contract to a full Unit2Wav model.

    - timbre: train normal backbone, freeze realism if it exists.
    - realism_personalize: freeze backbone, train realism on private data.
    - joint: train both; caller should keep backbone LR below realism LR.

    Public realism prior training is intentionally absent here and must use
    train_realism.py, whose dataset cannot expose waveform/mel timbre targets.
    """
    if stage not in VALID_REALISM_STAGES:
        raise ValueError(
            f"unknown realism stage {stage!r}; "
            f"expected one of {VALID_REALISM_STAGES}"
        )

    backbone, realism_params = split_realism_parameters(model)
    if stage != "timbre" and not realism_params:
        raise RuntimeError(
            f"stage {stage!r} requires model.realism.enabled=true"
        )

    if stage == "timbre":
        for param in backbone:
            param.requires_grad_(True)
        for param in realism_params:
            param.requires_grad_(False)
    elif stage == "realism_personalize":
        for param in backbone:
            param.requires_grad_(False)
        for param in realism_params:
            param.requires_grad_(True)
    else:
        for param in backbone:
            param.requires_grad_(True)
        for param in realism_params:
            param.requires_grad_(True)

    return {
        "backbone_total": sum(p.numel() for p in backbone),
        "backbone_trainable": sum(
            p.numel() for p in backbone if p.requires_grad
        ),
        "realism_total": sum(p.numel() for p in realism_params),
        "realism_trainable": sum(
            p.numel() for p in realism_params if p.requires_grad
        ),
    }
