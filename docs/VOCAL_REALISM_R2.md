# Vocal Realism R2: staged optimization

R2 formalizes which parameters may move at each training phase.

## Stages

`train_reflow.py` accepts `train.realism_stage`:

- `timbre`: trains the normal DDSP/Reflow backbone and freezes the realism adapter. This is the private target-voice warm-up stage.
- `realism_personalize`: freezes the full backbone and updates only the realism adapter on private target data.
- `joint`: enables both groups and uses separate AdamW learning rates. The backbone learning rate should remain substantially lower than the realism learning rate.

The public singer-independent prior is intentionally not a `train_reflow.py` stage. It continues to use `train_realism.py`, whose dataset exposes only units/F0/volume and cannot apply waveform/mel timbre losses.

## Suggested sequence

1. Private target data, `realism_stage: timbre`, with `model.realism.enabled: false` or enabled-but-frozen.
2. Public singing data through `train_realism.py` to produce `realism_final.pt`.
3. Set `model.realism.enabled: true` and `model.realism.checkpoint` to the standalone prior, then run private target data with `realism_stage: realism_personalize`.
4. Run a short private-data convergence stage with `realism_stage: joint`, low `backbone_lr`, and higher `realism_lr`.
5. Save the normal full-model checkpoint. At that point the realism parameters live inside the same model state as the timbre backbone and Reflow model.

## Optimizer-state boundary

Changing stage changes the trainable parameter set and, for `joint`, the optimizer type/group layout. Set `train.resume_optimizer: false` when crossing a stage boundary unless the new stage uses the exact same optimizer layout. Model weights still resume from the normal checkpoint.

## Invariants

- Public prior training never sees waveform/mel targets.
- Public data never updates the timbre backbone.
- `realism_personalize` never updates the backbone.
- `joint` is intended to be short and private-data-only.
- `model.realism.enabled: false` preserves the upstream synthesis path.
