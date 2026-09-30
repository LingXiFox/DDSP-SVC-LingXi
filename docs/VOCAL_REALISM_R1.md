# Vocal Realism R1: speaker-weak control prior

R1 adds a trainable human-performance prior without allowing public singers to become a timbre source.

## Data boundary

The public-data trainer reads only three products from the normal DDSP-SVC preprocessing layout:

- `units/` content features;
- `f0/` frame pitch;
- `volume/` frame loudness.

It does not read waveform, mel, augmented mel, speaker IDs, formants, or target spectral envelopes. Consequently, the R1 objective cannot directly update a waveform/mel timbre backbone.

## Training task

`robotize_controls()` deliberately removes micro-dynamics with masked smoothing while preserving voiced/unvoiced structure and the broad pitch/loudness trajectory. The zero-initialized `VocalRealismAdapter` receives the robotized controls and learns to restore locally normalized human dynamics.

The feature loss is intentionally register/level weak. It removes local trends before comparing:

- relative pitch contour in cents;
- pitch velocity;
- pitch acceleration;
- relative energy contour in dB;
- energy velocity.

A global octave shift should therefore have near-zero feature loss while a local contour deformation should not.

## Standalone prior training

Use `configs/realism.yaml` with:

```bash
python train_realism.py -c configs/realism.yaml
```

The output checkpoint format is `lingxi-vocal-realism-v1` and contains only the adapter state plus optimizer/training metadata. It can be injected into the full DDSP model through `model.realism.checkpoint`, after which the normal full-model checkpoint will contain the adapter together with the rest of the model.

## Safety contract

Public-data R1 training must remain control-only. Any future change that introduces waveform/mel reconstruction, speaker embedding prediction, formant targets, or free spectral residuals into this stage breaks the architecture contract and must be reviewed separately.

## Next phase

R2 will formalize staged optimization and parameter freezing for private timbre warm-up, public realism prior, private realism personalization, and low-learning-rate joint convergence. R3 remains native Mid/Side stereo.
