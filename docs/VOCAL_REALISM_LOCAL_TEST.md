# Vocal Realism local test handoff

Run these checks on `feature/vocal-realism` before marking PR #1 ready.

## 1. Static and unit checks

```bash
python -m compileall ddsp reflow logger train_reflow.py train_realism.py
pytest -q   tests/test_vocal_realism.py   tests/test_realism_features.py   tests/test_realism_data_loaders.py   tests/test_realism_stages.py
```

Expected: all tests pass with no import-cycle, device, shape, NaN, or dtype errors.

## 2. Baseline compatibility

With `model.realism.enabled: false`:

- load a known-good 6.3 checkpoint;
- run the same input through branch `6.3` and `feature/vocal-realism`;
- compare model output shape, dtype, device, and numerical output;
- verify old checkpoint load does not report fatal missing/unexpected keys;
- verify no `realism.*` parameters are instantiated when disabled.

Target: disabled path should be behaviorally identical to 6.3 within deterministic/noise limitations of the existing synthesizer.

## 3. Zero-init enabled compatibility

Set:

```yaml
model:
  realism:
    enabled: true
    checkpoint: null
```

Before any optimizer step:

- verify adapter diagnostics are exactly zero;
- verify F0 and volume entering the upstream synthesizer are unchanged;
- verify unvoiced F0 stays exactly zero;
- run forward and backward on CPU and CUDA if available.

## 4. Standalone public-prior trainer

Prepare a tiny preprocessed dataset containing only the normal `audio/`, `units/`, `f0/`, and `volume/` paths. Do not create mel files for this test.

```bash
python train_realism.py -c configs/realism.yaml
```

For a smoke run, temporarily reduce epochs/save intervals and point the config at the tiny dataset.

Verify:

- the loader never requests `mel/`, `aug_mel/`, `aug_vol/`, or speaker IDs;
- loss is finite;
- `output_proj` leaves zero initialization after the first optimizer step;
- gradient norm is finite;
- `realism_latest.pt` is written and resumes;
- the checkpoint declares format `lingxi-vocal-realism-v1`.

## 5. Prior injection

Set `model.realism.checkpoint` to the standalone checkpoint and instantiate the full Reflow model.

Verify:

- the prior loads strictly into the adapter;
- a normal full-model checkpoint save contains `ddsp_model.realism.*` keys;
- after that full checkpoint is saved, it can be reloaded without requiring the standalone prior path if the config is adjusted appropriately for the test.

## 6. R2 stage contract

Check trainable parameter counts printed by `train_reflow.py`.

`timbre`:
- backbone trainable > 0;
- realism trainable = 0.

`realism_personalize`:
- backbone trainable = 0;
- realism trainable > 0.

`joint`:
- both trainable > 0;
- optimizer has separate backbone and realism learning rates.

When switching optimizer layout between stages, set:

```yaml
train:
  resume_optimizer: false
```

Model weights should still resume.

## 7. CUDA smoke test

On the local 4090 or another CUDA GPU:

- one forward/backward step for each R2 stage;
- 100-step `train_realism.py` run;
- 100-step private `realism_personalize` run;
- short `joint` run;
- record peak VRAM, step time, loss curve, gradient norm, NaN/Inf incidence.

Failure conditions:

- public-prior code touches waveform/mel targets;
- public-prior code can update the timbre backbone;
- disabled realism changes the normal inference path;
- unvoiced frames receive non-zero F0;
- stage transition silently restores an incompatible optimizer state;
- any non-finite loss/gradient appears under normal input.

## 8. Audio A/B sanity check

Only after the above passes:

- render baseline 6.3;
- render realism enabled at zero-init;
- render after a short prior/personalization run;
- compare pitch stability, vibrato dynamics, attacks/releases, loudness motion, and speaker identity.

Do not judge the architecture from one subjective clip. Keep numerical controls and identity checks alongside listening tests.
