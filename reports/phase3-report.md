# Phase 3 Report — LingXi SVC Runtime Pipeline

Date: 2026-09-19. Python 3.14.6, torch 2.14.0, torchaudio 2.11.0, MPS.

## Compatibility patch (committed)

- `MAC_COMPAT_PATCH_COMMIT=ff2cc92` — `fix(macOS): make pretrained checkpoints device-portable`
- 4 files, 6 one-line changes, all `torch.load(..., map_location="cpu")`
  (downstream `.to(device)` unchanged). Zero math change.
- Pre-commit validation: `tests/test_realism_bypass.py` exit 0;
  Phase 2 reference inference already PASS on both CPU and MPS.
- `DEPENDENCY_SPEC_UPDATED=PASS` — `app/pyproject.toml` pins the exact
  validated set (torch 2.14.0 / torchaudio 2.11.0 kept as-is, no re-versioning);
  `environment/.venv` was built from it.

## Runtime

- `PIPELINE_API=PASS` — `LingXiSVCPipeline.convert(input, output,
  realism_strength, transpose, seed)` in `app/runtime/`.
- `BUNDLE_AUTO_DISCOVERY=PASS` — default bundle resolved from project root,
  required files + optional SHA256 verification.
- `MODEL_LOAD_ONCE=PASS` — one pipeline instance served all test converts.
- `AUDIO_LOAD=PASS` / `FEATURE_EXTRACTION=PASS` (finite/shape/dtype/align
  assertions, no disk cache).
- `HUMANIZER_CHAIN=PASS` — adapted F0/volume verified into `vocoder.infer`;
  strength>0 asserts change, strength==0 asserts exact identity.
- `FINAL_SYNTHESIS=PASS` — DDSP/Reflow/HiFiGAN via reused repo code.
- `TRANSPOSE_ORDER=transpose -> humanizer` (audited from main_reflow.py,
  preserved: F0 key-shift applied before Humanizer, adapted F0 to vocoder).
- `SEED_CONTROL=PASS` — same-seed repeat maxabsdiff exactly 0.0
  (torch + numpy + MPS generators seeded per convert; single initial
  randn per Reflow forward is the only stochastic source).

## Regression (bundle reference input, MPS, seed 1234)

- `REFERENCE_REGRESSION=PASS` — 18.4599 s, peak 0.7253, finite, no clipping.
- `PHASE2_VS_PHASE3_OUTPUT_DIFF=` RMS 0.1100 vs 0.1094 (~0.5%; Phase 2 run
  was unseeded, Phase 3 seeded — sampling noise only, same pipeline math).
- `F0_RESIDUAL_ON=13.48c` (Phase 2 probe on first 2 s: 14.68c — same magnitude)
- `VOLUME_RESIDUAL_ON=1.755dB` (Phase 2: 1.82dB — same magnitude)
- `OFF_IDENTITY=PASS`
- `OUTPUT_FINITE=PASS`, `OUTPUT_CLIPPING=NO`

## Workspace (single root, no ~/LingXiSVC anymore)

- `LINGXISVC_ROOT=/Volumes/Development/Projects/projects/DDSP-SVC-LingXi`
- `PRODUCTION_PATH=<root>/production/DDSP-SVC-LingXi-Production-v1`
- `REPO_PATH=<root>` (the repo itself; no duplicate checkout)
- `VENV_PATH=<root>/environment/.venv`
- `OUTPUT_PATH=<root>/outputs`, `CACHE_PATH=<root>/cache`
- `OLD_DUPLICATES=` ~/LingXiSVC removed (stale venv only; outputs/logs migrated);
  ~/Downloads bundle dir moved; ~/Downloads tarball retained as archive backup.
- `PATH_MIGRATION=PASS`
- `PRODUCTION_SHA256_AFTER_MOVE=PASS` (12/12)
- `REFERENCE_INFERENCE_AFTER_MOVE=PASS` (this report's test ran post-move)
- Runtime contains no hardcoded `~/Downloads`, `~/LingXiSVC`, `/root/autodl-`
  paths (grep verified; only identifier substrings remain).

## Phase verdict

`PHASE_3=PASS`. Stopped here for user confirmation before Phase 4 CLI.
