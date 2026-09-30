# Phase 4 Report — lingxi-svc CLI

Date: 2026-09-19. venv `environment/.venv`, torch 2.14.0, MPS.

## Install

- `CLI_INSTALL=PASS` — `pip install -e ./app` with
  `[project.scripts] lingxi-svc = "runtime.cli:main"`.
- Editable install keeps `runtime.__file__` inside the workspace, so the
  project root (repo + production + runtime) resolves from any CWD.
  `LINGXI_PROJECT_ROOT` env override exists as fallback.
- `CLI_HELP=PASS`, `CLI_ANY_DIRECTORY=PASS` (verified from /tmp).

## Behavior

- `DEFAULT_OUTPUT_PATH=PASS` — `vocal.wav` -> `vocal_lingxi.wav` beside input.
- `CUSTOM_OUTPUT_PATH=PASS` — `-o` respected exactly.
- `OVERWRITE_PROTECTION=PASS` — existing output refused without `--force`
  (exit 2, one-line message, no traceback).
- Error UX: short `Error:` lines, exit 2 for usage errors; `--verbose`
  prints full tracebacks.
- Summary printed from Phase 3 `ConversionResult` (no recomputed stats).

## Arguments

- `REALISM_ARGUMENT=PASS` — 0.0 and 1.0 full runs; out-of-range rejected pre-load.
- `TRANSPOSE_ARGUMENT=PASS` — transpose 0 vs 2: duration identical,
  output voiced-median F0 ratio 1.1225 == 2^(2/12) exactly.
  Order preserved: transpose -> humanizer (pipeline unchanged).
- `DEVICE_ARGUMENT=PASS` — auto resolves MPS; explicit `--device cpu`
  full run PASS.
- `SEED_ARGUMENT=PASS` — fixed seeds everywhere; same-seed repeat
  max_abs_diff 0.0 (Phase 3, MPS).

## Regression (bundle reference input, seed 1234)

- `REFERENCE_CLI_INFERENCE=PASS` — 18.46 s, MPS, RTF 0.18, finite, no clipping.
- `CLI_VS_API_MAX_ABS_DIFF=0.0` — CLI output bit-identical to direct
  `LingXiSVCPipeline.convert()` with same params. CLI is a pure wrapper.
- `REALISM_OFF_IDENTITY=PASS` — asserted inside convert during the CLI run.
- `MPS_CLI=PASS`, `OUTPUT_FINITE=PASS`, `OUTPUT_CLIPPING=NO`.

## Phase verdict

`PHASE_4=PASS`. Stopped here. Phase 5 (feature cache, --compare) not started.
