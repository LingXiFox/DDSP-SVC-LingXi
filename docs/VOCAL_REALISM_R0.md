# Vocal Realism R0

This branch introduces the first architecture hook for non-timbre vocal realism while preserving DDSP-SVC 6.3 behavior by default.

## Scope

R0 adds a small residual controller before the DDSP source generator and Unit2Control. It can adjust only:

- relative F0 in bounded cents;
- frame loudness in bounded dB.

The adapter does **not** predict waveform, mel, speaker embeddings, formants, or a free spectral residual. This is intentional: public singing datasets must teach singer-independent human performance dynamics without becoming an alternate source of target timbre.

## Compatibility contract

- `model.realism.enabled: false` keeps the upstream 6.3 path unchanged.
- Enabling the module starts as an exact identity transform because the final projection is zero-initialized.
- Existing checkpoints remain loadable with non-strict full-model restore.
- Existing call sites remain compatible because the new constructor argument is optional and appended at the end.
- A standalone R1 prior may be injected through `model.realism.checkpoint` before private personalization/joint convergence.

## R0 controls

`VocalRealismAdapter` consumes content units plus speaker-weak normalized F0/loudness controls and predicts two bounded residuals:

- `delta_f0_cents` in `[-max_f0_cents, +max_f0_cents]`;
- `delta_volume_db` in `[-max_volume_db, +max_volume_db]`.

Unvoiced F0 remains exactly zero.

## Implemented follow-up phases

R1 is implemented in `VOCAL_REALISM_R1.md`: control-only public prior training, robotization corruption, and locally normalized singer-weak dynamics losses.

R2 is implemented in `VOCAL_REALISM_R2.md`: explicit timbre, private realism-personalization, and low-LR joint stages.

The full local/GPU acceptance checklist is in `VOCAL_REALISM_LOCAL_TEST.md`.

R3 native stereo using Mid/Side remains deliberately deferred until R0-R2 pass full repository and CUDA integration tests.
