# LingXi SVC — Final System Report (封存状态)

Date: 2026-09-19. Root: `/Volumes/Development/Projects/projects/DDSP-SVC-LingXi`
(single workspace; no second checkout).

## Components ([1]–[19])

- [1] Production v1: `production/DDSP-SVC-LingXi-Production-v1/model/production_v1.pt`
  (Candidate P, SHA256 `b628470c…f307`, 12/12 bundle hashes PASS after cleanup)
- [2] Config: `config/production.yaml` (+ loader copy `model/config.yaml`),
  realism enabled, strength 1.0
- [3–5] Assets: ContentVec / RMVPE / NSF-HiFiGAN exact training-time versions
- [6] `LingXiSVCPipeline` (`app/runtime/`): audio → features → Humanizer →
  DDSP/Reflow → vocoder; transpose-before-humanizer preserved; seed-controlled
- [7–8] macOS MPS compat (`ff2cc92`, load-to-CPU then `.to(device)`, zero math
  change); `--device auto` prefers MPS, else CPU; per-module devices reported
- [9] CLI `lingxi-svc` (pure wrapper, bit-identical to API: max_abs_diff 0.0)
- [10] Isolated env `environment/.venv` (Python 3.14.6, torch 2.14.0)
- [11] Reference cold test: bundle `test/` + `outputs/reference/` baselines
- [12] SHA256: bundle `metadata/SHA256SUMS` + `lingxi-selfcheck`
- [13] Docs: `app/README.md` (install/use/uninstall)
- [14] Bootstrap: `app/install.sh` (idempotent, no system pollution)
- [15–16] Diagnostics via `ConversionResult`; health via `lingxi-selfcheck`
  (last run: SELFCHECK=PASS)
- [17] Uninstall/cleanup: see `app/README.md`
- [18] `.gitignore` covers production/environment/outputs/inputs/logs/reports/cache/archive
- [19] This report + `reports/phase3-report.md` + `reports/phase4-report.md`

Deferred (not required): real-song tests, feature cache data, --compare,
GUI, realtime, plugins.

## Kept test assets (outputs: 6.2 MB)

- `outputs/reference/mac_reference_output.wav` (CPU baseline, Phase 2)
- `outputs/reference/mac_reference_output_mps.wav` (MPS baseline, Phase 2)
- `outputs/cli_reference.wav` (CLI ON proof, bit-identical to API)
- `outputs/cli_reference_off.wav` (OFF proof)
- `inputs/cli_reference_input.wav` (CLI test input copy; bundle original untouched)

## Cleanup performed

- Deleted: 9 redundant wavs (API duplicate, repeats, transpose probes,
  default-path probes, Phase 3 trio superseded by CLI proofs),
  stale `cache/rmvpe_*.npy`, all `__pycache__`.
- Moved to `archive/` (15 MB, gitignored): `personalize-abc-step8460/`,
  `personalize-robotized-abc-step405/`, `tmp/` (training-era experiment scripts).
- Kept: `~/Downloads/DDSP-SVC-LingXi-Production-v1.tar.zst` (archive backup).

## Disk

- production 980M / environment 1.4G / app 140K / outputs 6.2M /
  inputs 1.6M / logs 48K / reports 8K / cache 0B / archive 15M
- Total LingXi footprint ≈ 3.2 GB (adds `archive/` 822 MB incl. the
  production tarball backup); volume free 663 GB.
- Local archive backup now lives at
  `archive/DDSP-SVC-LingXi-Production-v1.tar.zst` (+ `.sha256`, re-verified
  OK after the move); nothing LingXi-related remains in `~/Downloads`.

## Seal status

System is sealed and idle: no training, no background jobs, Production
weights untouched since freeze. Next use starts with `lingxi-selfcheck`.
