# LingXi SVC — Production v1 Local Inference

Offline singing-voice conversion on macOS. One dry vocal in, one converted
WAV out. No training, no network, no AutoDL dependency.

Production model: Candidate P (Private Timbre + Personalized Humanizer
best@405). J50/J100 are experimental and are never used here.

## Install

```bash
./app/install.sh
```

This creates `environment/.venv` (Python 3.14), installs the pinned
runtime (`app/pyproject.toml`: torch 2.14.0, MPS enabled), and places
`lingxi-svc` / `lingxi-selfcheck` launchers in `~/.local/bin` (added to
PATH once). Nothing is installed into system Python. Open a new shell
after install.

## Use

Self-check:

```bash
lingxi-selfcheck
```

Convert:

```bash
lingxi-svc vocal.wav
lingxi-svc vocal.wav -o vocal_lingxi.wav
lingxi-svc vocal.wav --realism 0.85 --transpose 0
```

Full options: `lingxi-svc --help`. No `cd`, no `source activate` needed;
both commands work from any directory.

Defaults: output `<input>_lingxi.wav` next to the input; realism from
`production.yaml` (1.0); transpose 0 semitones; device auto (MPS, else CPU);
seed 1234. Existing outputs are never overwritten without `--force`.

## Development / debugging

```bash
source environment/.venv/bin/activate
lingxi-svc --help
environment/.venv/bin/python app/tests/test_phase3_reference.py
```

Normal users never need this; the launchers already target the venv.
Direct `main_reflow.py` / `batch_infer.py` entry points remain for
low-level work but are not part of the supported UX.

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
