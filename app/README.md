# LingXi SVC — Production v1 Local Inference

Offline singing-voice conversion on macOS. One dry vocal in, one converted
WAV out. No training, no network, no AutoDL dependency.

Production model: Candidate P (Private Timbre + Personalized Humanizer
best@405). J50/J100 are experimental and are never used here.

## Install

```bash
./app/install.sh
source environment/.venv/bin/activate
```

This creates `environment/.venv` (Python 3.14) and installs the pinned
runtime (`app/pyproject.toml`: torch 2.14.0, MPS enabled). Nothing is
installed into system Python.

## Use

```bash
lingxi-svc vocal.wav
lingxi-svc vocal.wav -o converted.wav --realism 0.85 --transpose 0
lingxi-svc vocal.wav --compare  # (Phase 5, not yet implemented)
```

Full options: `lingxi-svc --help`.

Defaults: output `<input>_lingxi.wav` next to the input; realism from
`production.yaml` (1.0); transpose 0 semitones; device auto (MPS, else CPU);
seed 1234. Existing outputs are never overwritten without `--force`.

## Self-check

```bash
lingxi-selfcheck
```

Verifies bundle files + SHA256, torch/MPS status, and a full model load.
Ends with `SELFCHECK=PASS` or a concrete failure.

## Layout (workspace root = this repo)

- `production/DDSP-SVC-LingXi-Production-v1/` — immutable model bundle
  (never edit; verify with `metadata/SHA256SUMS`)
- `environment/.venv/` — isolated Python (never commit)
- `app/runtime/` — orchestration only; model math lives in the repo
  (`ddsp/`, `reflow/`, `encoder/`, `nsf_hifigan/`, `slicer.py`)
- `inputs/` — your test vocals; `outputs/` — conversion results
- `cache/` — future feature cache (not used yet)
- `logs/`, `reports/` — run evidence and phase reports
- `archive/` — superseded local artifacts, kept out of the way

## Uninstall / cleanup

```bash
deactivate 2>/dev/null
rm -rf environment/.venv          # regenerable via ./app/install.sh
rm -rf outputs/* cache/*          # generated audio and caches
```

Keep `production/` (the model) and `~/Downloads/DDSP-SVC-LingXi-Production-v1.tar.zst`
(archive backup). To remove everything LingXi-related, delete this repo
checkout too — no files are installed outside of it.

## Notes

- Strength is applied at inference time only; the checkpoint is never
  rewritten, quantized, or retrained.
- Reference input/output for sanity checks live inside the bundle
  (`production/.../test/`).
- macOS device policy: `--device auto` prefers MPS, falls back to CPU.
  Per-module fallbacks are printed, never silent.
