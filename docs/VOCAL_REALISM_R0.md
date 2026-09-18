# Vocal Realism R0

This branch introduces the first architecture hook for non-timbre vocal realism while preserving DDSP-SVC 6.3 behavior by default.

## Scope

R0 adds a small residual controller before the DDSP source generator and Unit2Control. It can adjust only:

- relative F0 in bounded cents;
- frame loudness in bounded dB.

The adapter does **not** predict waveform, mel, speaker embeddings, formants, or a free spectral residual. This is intentional: future public singing datasets must teach singer-independent human performance dynamics without becoming an alternate source of target timbre.

## Compatibility contract

- `model.realism.enabled: false` keeps the upstream 6.3 path unchanged.
- Enabling the module starts as an exact identity transform because the final projection is zero-initialized.
- Existing checkpoints remain loadable because the project already restores model state with `strict=False`.
- Existing call sites remain compatible because the new constructor argument is optional and appended at the end.

## R0 controls

`VocalRealismAdapter` consumes content units plus compressed F0/loudness controls and predicts two bounded residuals:

- `delta_f0_cents` in `[-max_f0_cents, +max_f0_cents]`;
- `delta_volume_db` in `[-max_volume_db, +max_volume_db]`.

Unvoiced F0 remains exactly zero.

## Next phases

R1 will add a singer-weak human-feature pipeline and validation metrics for F0 dynamics, vibrato, timing, voiced/unvoiced transitions, energy envelopes, and aperiodicity. Public data must never apply waveform/mel reconstruction loss to the timbre backbone.

R2 will add staged training (private timbre warm-up -> public realism prior -> private personalization -> low-LR joint convergence).

R3 will add native stereo using Mid/Side rather than independent left/right conversion.
