# Timbre Blend Experiment Log

Continuous log for the multi-singer pretrain -> virtual-singer finetune -> timbre blend
pipeline. Append-only; newest entries at the bottom. No secrets allowed in this file.

---

## 2026-09-27 10:20-10:40 | Stage 0: reconnaissance, baseline, resource prep

- Git: branch `feature/timbre-blend` created from `feature/vocal-realism` @ `95f5bfd`
  (working tree was clean before branch creation).
- Note: the macOS-side clone of this repo has diverged (different HEAD + uncommitted
  changes). The WSL2 remote repo is the single source of truth for this task.

### Environment (verified by actual commands)

- Host: WSL2 Ubuntu, kernel 6.18.33.2-microsoft-standard-WSL2, via `ssh -p 2222 hands@10.0.0.150`.
- Python: `.venv` (uv-managed), Python 3.11.16, torch 2.11.0+cu128, `torch.cuda.is_available() == True`.
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU, 15.99 GiB, compute capability (8, 9) -> bf16 supported.
  Idle: 54 C, ~33 W, 1033 MiB used.
- `nvidia-smi` is NOT on PATH. Working binary:
  `/usr/lib/wsl/drivers/nvami.inf_amd64_b259952749ea1a39/nvidia-smi`
- Disk: `/dev/sdd` 1007G total, 935G available (3% used). Repo 7.3G (mostly `.venv`).
  No `~/datasets` yet, no HF cache yet.

### Baseline static checks and tests (before any code change)

- `python -m compileall -q ddsp reflow logger train_reflow.py train_realism.py` -> OK (exit 0).
- `pytest -q tests/` -> **17 passed, 0 failed, 0 skipped** in 6.54s (5 warnings:
  pkg_resources deprecation, weight_norm deprecation). No baseline failures.

### Repository reconnaissance findings (all verified against code @ 95f5bfd)

1. Multi-speaker layout: `data/{train,val}/audio/<spk_id>_<name>/*.wav`.
   `reflow/data_loaders.py` parses spk_id as `re.split(r"_|\-", dirname, 2)[0]`, must be
   an int in `[1, n_spk]`, else raises. Features live in sibling dirs
   (`units/ f0/ volume/ mel/ aug_mel/ aug_vol/`) mirroring the audio relative path.
2. Speaker IDs are 1-based; embedding lookup uses `spk_id - 1`.
3. `model.n_spk`: if `> 1`, `Unit2Control` creates `nn.Embedding(n_spk, dim_model)`
   (`ddsp/unit2control.py`). `n_spk == 1` -> no spk embedding at all.
4. Speaker embedding checkpoint key: `ddsp_model.unit2ctrl.spk_embed.weight`
   (Unit2Wav.ddsp_model = CombSubSuperFastRealism -> CombSubSuperFast.unit2ctrl).
5. `spk_mix_dict`: implemented in `Unit2Control.forward` - adds `v * spk_embed(k - 1)`
   per entry. Exposed at inference via `main_reflow.py -mix` (literal_eval of a dict str).
6. `t_start`: training samples `t ~ U(t_start, 1)`; reflow loss skipped (cpu
   `torch.tensor(0)`) when `t_start >= 1.0`. Inference: ODE init
   `x = t_start * norm(ddsp_mel) + (1 - t_start) * noise`, `dt = (1 - t_start)/infer_step`.
   `main_reflow.py` clamps user `-ts` to be >= config `model.t_start` (config value 0.0).
7. Losses: `reflow/vocoder.py Unit2Wav.forward(infer=False)` computes
   `ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)` and calls
   `RectifiedFlow.forward -> reflow_loss` (l2_lognorm weighted MSE on velocity).
   Combined in `reflow/solver.py train()`: `loss = lambda_ddsp * ddsp_loss + reflow_loss`.
   NaN ddsp_loss -> skip batch; NaN reflow_loss -> raise.
8. Optimizer/ckpt: default realism stage `timbre` -> `Muon_AdamW(model)` (Muon for
   requires_grad params with ndim>=2 except nn.Embedding; AdamW for the rest).
   `logger/utils.load_model` scans expdir for `model_<step>.pt`, loads max step with
   `model.load_state_dict(..., strict=False)` (NOTE: size mismatches still raise even
   with strict=False), restores optimizer only if present in ckpt and `resume_optimizer`.
   `Saver.save_model` stores optimizer only when `train.save_opt: true` (config default
   false) -> Stage 1 ckpts will NOT contain optimizer state by default.
   `global_step` is restored from ckpt and drives the StepLR decay reconstruction in
   `train_reflow.py`.
9. `preprocess.py` generates per audio file: `units/f0/volume/mel/aug_mel/aug_vol` .npy
   + `pitch_aug_dict.npy`. IMPORTANT: on F0-extraction failure it **moves the source
   audio into `<path>/skip/`** (in-place mutation of the input tree) - another reason
   all processing must run on fresh working copies, never on source data.
   Preprocessing exceptions per file are caught and printed (not silently swallowed
   wholesale, but a file that raises is skipped with only a console message).
10. Pretrain deps (all MISSING at recon time, download started, see below):
    - ContentVec `pretrain/contentvec/pytorch_model.bin`
      (HF `lengyue233/content-vec-best`, public, via mirror).
    - RMVPE `pretrain/rmvpe/model.pt`
      (GitHub release `yxlllc/RMVPE` 230917, zip contains rmvpe.pt -> rename).
    - NSF-HiFiGAN `pretrain/nsf_hifigan/model` + `config.json`
      (GitHub release `openvpi/vocoders` pc-nsf-hifigan-44.1k-hop512-128bin-2025.02).
11. Realism disabled path: `CombSubSuperFastRealism` with `realism=None` delegates
    straight to `CombSubSuperFast.forward`; covered by `tests/test_realism_bypass.py`.
    Config default `model.realism.enabled: false`. This task keeps it false everywhere.
12. Tests: 17 tests, all realism-focused. No existing tests for reflow solver /
    data loaders / vocoder masking. New Stage 2a tests will be added under `tests/`.

### Config baseline (`configs/reflow.yaml`, verified)

- sr 44100, block_size 512, win 2048, train duration 2s, f0 rmvpe [65, 800],
  encoder contentvec768l12tta2x (768ch), use_pitch_aug true, n_spk 1,
  batch_size 48, lr 5e-4 (Muon_AdamW), amp_dtype fp16, cache_all_data true (cpu, fp16),
  interval_val 2000, interval_force_save 10000, decay_step 4000, gamma 0.9,
  save_opt false, resume_optimizer true, expdir exp/reflow-test.

### Hugging Face status - BLOCKER (human action required)

- `HF_ENDPOINT=https://hf-mirror.com` set and echoed before every HF access.
- Not logged in (`whoami` -> LocalTokenNotFoundError; no env token, no saved token).
- Anonymous `repo_info` on `LingXiFox/ddsp-svc-lingxi` and `LingXiFox/opensinger-womanraw`
  -> **401 RepositoryNotFoundError** for both `model` and `dataset` repo types.
- Per plan section 3.2: stopped and reported to user. Waiting for user to complete
  `hf auth login` (or provide access) on the WSL2 machine. No token was printed,
  searched for, or copied from anywhere.

### Actions in flight

- tmux session `pretrain_dl`: downloading ContentVec (HF mirror), NSF-HiFiGAN and
  RMVPE (GitHub releases) via `.tmp/download_pretrain.sh`; log at
  `.tmp/pretrain_download.log`.
- `.gitignore`: added `.tmp/`, `logs/`, `samples/`, `data/timbre_blend_*/`
  (working data/artifacts must never be committed; `reports/` stays tracked).

### Deviations from plan

- None so far. (nvidia-smi path differs from a plain `nvidia-smi` call; recorded above.)

### Pretrain dependency download + functional verification (10:34-10:45)

- tmux `pretrain_dl` ran `.tmp/download_pretrain.sh`; all downloads finished 10:36.
- ContentVec: `hf download lengyue233/content-vec-best pytorch_model.bin` via
  HF_ENDPOINT=https://hf-mirror.com -> `pretrain/contentvec/pytorch_model.bin` (378 MB).
- RMVPE: GitHub release zip -> `pretrain/rmvpe/model.pt` (368 MB).
- NSF-HiFiGAN: GitHub release zip contains `model.ckpt` (script initially missed the
  extension; fixed manually) -> `pretrain/nsf_hifigan/model` (56.7 MB) + `config.json`.
- Functional check `.tmp/check_pretrain.py` (13.3s, cuda) — ALL PASSED:
  - Vocoder init: sr 44100, hop 512, 128 mel dims; mel of 2s sine = [1,172,128];
    vocoded wav [1,1,88064], rms 0.209.
  - RMVPE on 440 Hz sine: 173 frames, all voiced, median f0 = 439.11 Hz.
  - ContentVec (contentvec768l12tta2x): units [1,173,768].
- Git identity: repo-local `user.name=LingXiFox`,
  `user.email=198294627+LingXiFox@users.noreply.github.com` (same as all existing
  commits on this repo; global config untouched).
- First commit on branch: `8bbc397 docs(timbre-blend): add stage 0 recon log and
  ignore task working dirs`.
- Cleaned `.tmp` download zips/extract dirs after install.

### BLOCKED - waiting for user (human gate: HF login)

Stage 0 items 10-12 (bucket recon, locating virtual-singer data) and Stage 1a
(OpenSinger download) require authenticated access to the private repos
`LingXiFox/ddsp-svc-lingxi` and `LingXiFox/opensinger-womanraw`.
Anonymous access via mirror returns 401 for both (model and dataset types).
Stopped per plan section 3.2. No token handling was attempted.

## 2026-09-27 10:42-10:55 | HF login done; bucket recon; OpenSinger download started

- User completed `hf auth login` on WSL2. Verified via mirror: whoami -> LingXiFox
  (token never printed). All HF commands carry `HF_ENDPOINT=https://hf-mirror.com`
  (user does not export it globally).
- `LingXiFox/opensinger-womanraw` [dataset]: ACCESSIBLE. 79,866 files, 9.44 GiB.
  Layout: `<spk>_<song>/<spk>_<song>_<seg>.{wav,lab,txt}` (OpenSinger naming, first
  number = singer id).
- `LingXiFox/ddsp-svc-lingxi`: **404 for both model and dataset types while logged
  in**. `list_models/list_datasets/list_spaces(author=LingXiFox)` shows only
  `opensinger-womanraw`. The ~14GB bucket named in the plan does not exist under
  this account.
- Local filesystem search found `/mnt/d/AI/voice-lab` (40 GiB) - a previous virtual
  singer workspace for 泠溪小狐狸:
  - `dataset_final/`: 75 wav slices, 111 MB, **21.8 min total**, manifest.csv +
    speaker_report.csv (per-file cosine/cluster). Sources: 8 cover songs
    (圣贤书 solo; 7 duets credited 泠溪小狐狸+若溪（虚拟歌手）).
  - `separated_vocals/`: 16 wav, 1.1 GiB (full separated vocal tracks).
  - `validation_results/`: 185 wav incl. `target_svc_44k/` (19), blind/, blind_map.csv
    (previous blind-listening artifacts).
  - `test_sources/`: 8 wav, 535 MB (能伴此梦无 / 让我做你的眼睛).
  - `dereverb_vocals/`, `references/`: empty. `raw_mix*`: mixes, not clean vocals.
  - Also contains seed-vc / separator project checkouts (not DDSP-SVC).
- OpenSinger download started 10:45 in tmux `opensinger_dl`:
  `hf download LingXiFox/opensinger-womanraw --repo-type dataset
  --local-dir ~/datasets/opensinger-womanraw` (resumable), log `.tmp/opensinger_download.log`.

### STOPPED - asked user (plan 10.5 / 20.1: multiple candidates, bucket missing)

Questions: (1) where is the real ddsp-svc-lingxi bucket / is it gone;
(2) is `dataset_final` (21.8 min) the authoritative target-singer training set,
noting duet sources may contain a second voice (若溪) and one low-cosine outlier
(track01_005.wav, 0.147).

## 2026-09-27 10:50-11:00 | Bucket found (new HF "buckets" type); user decisions; Stage 1a tooling

### User decisions (chat)

- `LingXiFox/ddsp-svc-lingxi` is a **HF Storage Bucket** (new repo type, web path
  `/buckets/...`), not a model/dataset repo - that is why repo_info 404-ed.
  Accessible via mirror with `hf buckets` CLI / bucket_info (huggingface_hub 1.33.0).
- 泠溪小狐狸 and 若溪（虚拟歌手） are the SAME voice (泠溪 is the project/brand name).
  Duets in the target data are NOT speaker contamination. No filtering needed for
  "second singer" risk.

### Bucket recon (metadata only, nothing downloaded yet)

- `LingXiFox/ddsp-svc-lingxi`: private, 13.97 GB, 81,580 files, created 2026-09-18.
- Top level: datasets/ experiments/ logs/ models/ outputs/ pretrain/ runs/.
- `datasets/private-timbre/` (target virtual singer 泠溪小狐狸/若溪):
  - `raw/`: 12 full vocal tracks (~714 MB: LX, lingxi, 圣贤书/风的告别/赴春寰/归途的光/
    眉南边/莫愁乡/若神明偏爱/无邪 covers...).
  - `sliced-v2/{train,val}/audio`: **138 train + 34 val slices** (Sep 18, current).
  - `processed/{train,val}`: full DDSP-SVC feature trees (units/f0/mel/aug_*/volume +
    pitch_aug_dict.npy) for those 138+34 (Sep 19, newest).
  - `train/`,`val/`: older smaller preprocess (17+4 files, Sep 18) - superseded.
- `datasets/public-realism/`: prior public-data realism experiment (not for this task;
  realism stays disabled here).
- `datasets/test/`: test01-03.wav (~2s each, smoke inputs).
- `experiments/`: old checkpoints - `private-joint/model_90.pt` (661.6 MB full model),
  `private-personalize/model_300.pt` (222.1 MB), `full-smoke/full_with_prior.pt`,
  realism adapters (2.4 MB each) + configs/logs.
- `models/`: realism adapter checkpoints only.
- Decision: download only `datasets/private-timbre/sliced-v2` (+ raw if re-slicing
  needed) at Stage 2b; whether to reuse `processed/` features or re-preprocess with
  Stage 1 settings will be decided (and logged) at Stage 2b after comparing configs.

### OpenSinger download

- Restarted 10:49 with `--include "*.wav" --max-workers 16` (labels .lab/.txt are not
  consumed by the DDSP-SVC pipeline; wav-only cuts 79,866 -> 26,621 files). Resumable;
  already-fetched files kept. Log: `.tmp/opensinger_download.log`. tmux `opensinger_dl`.
- Format verified on samples: 44.1 kHz mono PCM_16, 2-9 s slices, folder layout
  `<singer>_<song>/<singer>_<song>_<seg>.wav` -> singer id = leading integer.
- Rate fluctuates 1.5-13 files/s via mirror; ETA a few hours.

### Stage 1a tool: scripts/select_opensinger.py (commit 36b2c1b)

- Per-file: duration, clip_ratio, SNR proxy (p95-p20 frame-energy dB, documented as
  relative indicator), RMVPE voiced_ratio, octave jumps (>=1100 cents with
  stable neighbours <200 cents, per voiced minute), f0 quarter-tone histogram
  (96 bins, pooled per singer for p5/median/p95), usable flag.
- USABLE GATES FIXED A PRIORI (before seeing any aggregate data):
  duration>=2.0s, voiced_ratio>=0.25, clip_ratio<=0.01, snr_db>=15,
  oct_jumps/voiced_min<=30.
- Score weights (rank-normalized, fixed a priori): snr_median .25, oct_jumps -.20,
  usable_min_capped60 .20, voiced_ratio .15, clip_ratio -.10, range_fit .10
  (trapezoid on f0 median over [100,150,350,450] Hz).
- Proposal eligibility (fixed a priori): usable_min >= 10, failed_ratio <= 0.2.
  Proposes 12 train + 3 holdout by default (CLI 8-15 / 2-3). Seed 20260927 recorded.
- Resumable JSONL cache `reports/opensinger_file_metrics.jsonl` (gitignored, >1MiB).
- Smoke-tested on 10 downloaded files: metrics sane (snr 12-25 dB, voiced ~0.8,
  f0 hist peak ~300 Hz), cache + CSV + proposal all written. ~10 files/s on GPU.
- tmux `stage1a`: watcher runs the full analysis automatically when the download
  finishes -> `reports/opensinger_singers.csv` + `reports/opensinger_proposal.json`,
  log `.tmp/analysis.log`. Expected analysis time ~45 min for 26.6k files.

### Next

- Wait for download + analysis, then present Gate 1 (singer selection) to user.

## 2026-09-27 11:05-11:40 | Stage 2a code complete (parallel with OpenSinger download)

User approved parallelizing the pure-code Stage 2a work (plan 14-19 + 21) while the
OpenSinger download runs. No GPU training started; download untouched.

### Implementation (4 commits)

- `9c3e2cf` feat(reflow): 
  - `Unit2Wav.forward(..., reflow_mask=None)` + `Unit2Wav.masked_reflow_loss()`.
    None -> legacy path bit-for-bit. [B] bool mask validated (dtype/shape/device,
    raises ValueError). Subsets ddsp_mel/gt_spec BEFORE RectifiedFlow, so the
    reduction is the mean over actual participants (never full-batch mean x mask).
    All-False -> `ddsp_mel.new_zeros(())`: correct device/dtype, reflow_model not
    called at all (no params, no grads, no RNG consumed).
  - `reflow/solver.py`: `build_reflow_mask(spk_id, exclude)` (returns None for
    empty exclude = legacy); train() and test() use the SAME rule.
  - Validation split (15.1): returned/optimized `validation/reflow_loss` is the
    INCLUDED masked loss; when exclude is set also logs
    `validation/reflow_loss_included` and (under the existing no_grad)
    `validation/reflow_loss_excluded_diagnostic` - diagnostic never enters the
    objective. All-excluded edge case reports 0.0 with a loud warning.
  - `configs/reflow.yaml`: `train.reflow_exclude_spk: []`, `train.freeze_reflow: false`.
- `e70608d` feat(training): `train.freeze_reflow` via importable helpers
  `apply_freeze_reflow(model)` + `build_muon_adamw_optimizer(model, args, freeze_reflow)`
  in train_reflow.py. Freeze runs AFTER configure_training_stage (which would
  re-enable requires_grad on the whole backbone for stage timbre - order matters).
  Frozen params excluded from Muon_AdamW via new optional `params=` kwarg
  (None = legacy layout exactly). Semantics documented: reflow forward still runs
  and its loss still backprops INTO the DDSP backbone (keeps ddsp_mel compatible
  with the frozen reflow), but reflow params get no grad and no optimizer update.
- `c9538f5` feat(ckpt): `logger/utils.load_model` hardened:
  - `optimizer_state_mismatch()`: structural check (param_group counts/sizes,
    recursing into ChainedOptimizer sub-states). Match -> restore + log
    "fully restored". Mismatch -> explicit discard + loud multi-line log
    (reason, likely cause, what WAS restored); training continues with fresh
    optimizer. Never silent, never pretends. `resume_optimizer` mechanism reused.
  - `expand_spk_embed_state()` (plan 21): n_spk N->N+K migration for
    `ddsp_model.unit2ctrl.spk_embed.weight`. Old rows copied verbatim (exact
    equality), new rows keep the model fresh init. Shrink -> RuntimeError.
    Any OTHER shape mismatch untouched (still raises in load_state_dict -
    no silent masking). Triggers only where the old code crashed anyway.
- `d069595` test(reflow): `tests/test_reflow_masking.py`, 21 tests, CPU-only tiny
  Unit2Wav + deterministic differentiable stub vocoder (+1 conditional CUDA test
  for the all-False device/dtype contract). Covers every case listed in plan 18:
  None/all-True(bitwise == None)/all-False(zero, no reflow call, no reflow grads,
  ddsp loss bitwise unchanged)/partial(bitwise subset equality via direct
  masked_reflow_loss call + full-forward poison leak checks incl. gradients),
  mask validation errors, build_reflow_mask, freeze false/true (optimizer layout,
  weights bitwise unchanged after a real step, backbone keeps training), missing
  config keys, yaml defaults, old-ckpt->frozen (weights exact, optimizer discarded
  loudly, step continues), matching-layout restore (state tensors bitwise equal),
  ckpt-without-optimizer, spk expansion (old rows exact + fresh row kept + all
  other tensors exact), shrink raises, unrelated mismatch still raises, and
  solver.test() validation split (included/diagnostic/all-excluded/legacy).

### Verification

- `python -m compileall -q ddsp reflow logger train_reflow.py train_realism.py optimizer scripts` OK.
- `.venv/bin/python -m pytest -q tests/` -> **38 passed, 0 failed** (17 baseline + 21 new).
- NOTE: must run `python -m pytest` (not bare `pytest`): pre-existing realism tests
  have no sys.path bootstrap and rely on CWD injection by `python -m`. Baseline
  Stage 0 run used the same invocation.

### Findings worth remembering

- `CombSubSuperFast.forward` unconditionally draws `noise = torch.randn_like(combtooth)`
  (noise exciter) - RNG consumption is batch-size dependent even with no dropout.
  Consequence for tests: a b=4 masked forward and a separate b=2 subset forward
  diverge in RNG stream; equivalence must be tested either at the
  masked_reflow_loss level (seeded right before) or via poison-invariance within
  identical batch shapes. This is also a (pre-existing, upstream) source of
  nondeterminism in inference.
- First partial-mask test draft failed for exactly this reason (0.166 vs 1.858,
  l2_lognorm weights amplify different t draws). Implementation was correct;
  test comparison method was wrong. Fixed by redesigning the comparison, NOT by
  loosening tolerances.

### Status

- OpenSinger download: ~10% (2.7k/26.6k files), ETA ~3.5h, tmux `opensinger_dl`.
- tmux `stage1a` watcher will auto-run the full singer analysis on completion.
- Next: Gate 1 (singer selection) after analysis; then Stage 1b data prep.
