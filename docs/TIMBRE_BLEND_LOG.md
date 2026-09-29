# Timbre Blend Experiment Log

Continuous log for the multi-singer pretrain -> virtual-singer finetune -> timbre blend
pipeline. Append-only; newest entries at the bottom. No secrets allowed in this file.

---

**Current access (2026-09-29):** WSL2 SSH `hands@10.0.0.244:2222` (verified
against the previously trusted SSH host key). Earlier `.150` addresses in the
append-only entries below describe historical access, not the current host.
TensorBoard policy is **127.0.0.1:6006 only**; the existing `.tmp/run_tb.py`
launcher was changed from `0.0.0.0` to loopback. It is currently stopped
(no port 6006 listener). Stage 2 validation curves are written to JSONL.

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

### 2026-09-27 11:55 | Stage 2a addendum

- Autocast smoke (fp16 + bf16, partial mask + all-False, CUDA) passed as a .tmp
  probe, then promoted into the permanent suite as
  `test_masked_reflow_under_autocast_on_cuda` (skipif no CUDA) - this is the
  exact Stage 2 production config (AMP + masked reflow). Probe script deleted.
- Note: the ComplexHalf UserWarning under fp16 autocast comes from the upstream
  DDSP harmonic synthesizer (`ddsp/vocoder.py`), pre-existing, unrelated to the
  masking changes; losses stay finite and fp32.
- Full suite now: **40 passed, 0 failed** (17 baseline + 23 new).

## 2026-09-27 Stage 1a 数据源切换：HF 私仓 → 官方 Google Drive 归档（用户批准，验证通过）

### 背景：HF 私仓下载过慢
- `hf download LingXiFox/opensinger-womanraw`（hf-mirror）实测：16 workers ~2.2 文件/s（~0.82 MB/s）；32 workers ~2.30 文件/s（~0.89 MB/s），并发翻倍仅 +5%。
- 诊断：镜像 connect 59ms / TLS 120ms / TTFB 0.39s 正常；同一镜像匿名下载公开仓库对照跑到 6.8 MB/s → 瓶颈在 hf-mirror 对私有仓库的认证中继路径（推测回源 huggingface.co 被限速），非客户端并发、非文件系统（已是 WSL2 原生 ext4）。
- aria2+HF token 方案按用户指令放弃（需显式传递凭据，且对私有中继限额效果不确定）。

### 官方公开源调查（仅元数据，用户指令）
- 官方唯一分发：Multi-Singer 官网（multi-singer.github.io）Google Drive 共享文件 `OpenSinger.tar.gz`（file id 1EofoZxvalgMjZqzUEuEdleHIZ6SHtNuK），Last-Modified 2021-11-26，大小 14,046,666,684 B（13.08 GiB）。
- 公开匿名可下、零凭据零 token；远端机可直达 Google（本机透明代理，无 proxy 环境变量）。
- 测速：单连接 4.13 MB/s；4 路并行聚合 ~11-13 MB/s。HF 镜像上无公开 raw 版本（仅 Codec-SUPERB/CodecSR 等 parquet 衍生品）。
- 用户确认后切换。

### 执行记录
- 顺序：先 kill watcher（防 EXIT= 误触发分析）→ SIGINT 优雅暂停 HF 下载（日志 Aborted!/EXIT=1）→ 已下载 7,124 个 wav（~2.7GB）全部保留（规则 12：暂不清理，等用户处置）。
- aria2 安装：sudo 需密码不可用 → 免 sudo 用户态安装：`apt-get download` + `dpkg -x` 解包 Ubuntu 官方签名 deb（aria2 1.37.0、libaria2-0、libcares2；注意 Ubuntu 26.04 中 libc-ares2 是过渡空包，真实包名为 libcares2），置于 `~/.local/aria2`，ldd 依赖全解析。未使用任何来路不明二进制。
- aria2 8 连接（-x8 -s8 --continue=true --max-tries=0 --file-allocation=falloc）：13 GiB 用时 8.5 分钟，平均 25 MiB/s（约为 HF 私有中继的 28 倍）。

### 文件数量矛盾排查（用户规则 7）
- HF 私仓 79,866 文件 = 26,621 wav + 26,621 lab + 26,621 txt + 3 个非数据文件：`.DS_Store`（6,148B）、`._.DS_Store`（4,096B）为 macOS 垃圾，`.gitattributes`（2,504B）为 HF LFS 配置。数据本体 = 26,621 × 3 = 79,863。
- 官方 tar 的 WomanRaw 根目录同样含 `.DS_Store`/`._.DS_Store` 且字节大小与私仓完全一致 → 私仓即由该官方 tar 解出内容直接上传。

### 验证结果（规则 5/8，全部通过）
1. 精确大小 14,046,666,684 B 一致；`gzip -t` 通过；`tar -tzvf` 列出 130,381 条目（owner huangrongjie = 论文一作 Huang Rongjie，时间戳 2021-11-25，与论文时间线吻合）。
2. 只解压 WomanRaw + LICENSE + README.md，未解出 ManRaw（规则 6）。
3. 相对路径+字节大小全量 diff：79,863 vs 79,863 完全一致（对称排除 .DS_Store 类垃圾；唯一差异 = HF 侧特有 .gitattributes，属 HF 工件）。
4. HF 已下载 7,124 个 wav 与 tar 解出对应文件 md5 全量对比：全部一致。
5. lab/txt 各随机抽样 200（seed=42）从 HF 私仓按需拉取原文与 tar 版本 hash 对比：400/400 一致。
6. 结论：官方 tar 的 WomanRaw 与 `LingXiFox/opensinger-womanraw` 为同一份数据，字节级一致。验证脚本与中间产物：`.tmp/verify_official.sh`、`.tmp/verify_extract_diff.sh`、`.tmp/sample_labtxt_verify.py`、`.tmp/official_tar_list.txt`、`.tmp/hf_repo_listing.tsv`、`.tmp/hf_wavs.md5`、`.tmp/official_wavs.md5`。

### 许可证（以 tar 内权威文本为准，非第三方描述）
- `OpenSinger/LICENSE`（20,842 B）= **CC BY-NC-SA 4.0 International** 全文。
- `OpenSinger/README.md`（1,363 B）：所有使用者必须遵循 CC BY-NC-SA；官方引用为 ACM MM 2021 Multi-Singer 论文（Huang, Chen, Ren, Liu, Cui, Zhao）；数据申请渠道 help_multisinger@163.com。
- 对本项目含义：仅限非商业目的；衍生作品须以相同许可证共享；须署名。本项目为私人非商业研究实验，在许可范围内；若未来发布衍生模型/数据需注意 ShareAlike 与 NonCommercial 条款。

### 数据根目录切换（规则 9/11）
- Stage 1 数据根目录正式切换为：`~/datasets/opensinger-official/OpenSinger/WomanRaw`（79,863 数据文件，wav 共 9.43 GiB，48 位女歌手，711 个 singer_song 目录）。
- HF 部分下载 `~/datasets/opensinger-womanraw`（7,124 wav）保留作验证证据与备份，等用户验收后决定清理。
- Stage 1a 完整分析已启动（tmux `stage1a`，`.tmp/run_stage1a.sh`，日志 `.tmp/analysis.log`）：扫描确认 26,621 wav / 48 歌手，GPU（cuda）~30-40 it/s，ETA ~15 分钟。完成后进入【人工关卡 1：歌手选择】。

### Stage 1a 分析完成，进入【人工关卡 1：歌手选择】（等待用户决定）
- 分析范围：官方 WomanRaw 根目录，26,621 wav / 48 歌手，RMVPE F0，cuda，13m42s，0 个文件分析失败（所有歌手 failed_ratio=0）。
- 产物：`reports/opensinger_singers.csv`（48 行全量统计）、`reports/opensinger_proposal.json`（seed=20260927，含全部阈值与权重快照）。
- 阈值与权重为 2026-09-27 预先固定（见本日志前文与脚本 docstring），本次未做任何事后调整。
- 提案 TRAIN 12 人（按分数降序）：36, 14, 27, 28, 32, 42, 19, 41, 21, 33, 9, 43；HOLDOUT（未见过测试歌手）3 人：29, 47, 10。
- 硬门槛淘汰 5 人（usable < 10 min）：2 (8.8), 5 (9.1), 31 (9.7), 35 (6.3), 44 (6.2)；其余 28 人分数低于第 12 名落选。
- 训练池合计（60 min/人封顶后）447.1 min ≈ 7.45 h；最大单人占比 13.4%（spk 14 触顶 60 min）。
- 两个如实呈报的观察（不影响固定规则的执行，供关卡决策参考）：
  1. spk 41 的 f0 p5/med/p95 = 127/202/314 Hz，显著低于其他女歌手（中位数普遍 288-353 Hz）。oct_jumps 仅 0.62/min、range_fit=1.0，指标不排除「系统性低八度」与「真女中音」两种解释。按固定规则入选 TRAIN 第 8。
  2. spk 43 仅 155 文件 / usable 10.55 min，刚过 10 min 门槛，为 TRAIN 中最小样本。

### 【人工关卡 1】通过（含一项人工调整，2026-09-27）
- 用户批准最终选择：**TRAIN = [36,14,27,28,32,42,19,41,21,33,9,30]（12 人），HOLDOUT = [29,47,10]（3 人）**。
- 人工调整（用户决定，非自动规则产物）：spk 43 → spk 30。理由：spk 43 仅 10.6 min / 155 文件，训练数据量过小；spk 30 有 27.5 min，虽综合评分略低（0.277 vs 0.295），但更适合作为稳定的多说话人训练成员。
- spk 41 保留：当前证据不足以证明其 F0 为系统性低八度，且数据量、浊音比例、octave-jump 指标均较好；定位为低音域/女中音方向训练歌手。
- **待办（Stage 1c 预处理后强制检查项）**：单独复核 spk 41 的 F0 分布与 octave-error 特征；只有出现实际证据证明系统性低八度时才停下报告，不得提前人为修正。
- 预先固定的自动筛选阈值与权重保持不变（用户明确要求）。
- 最终选择已固化到 `reports/opensinger_selection_final.json`；后续 Stage 1b/2/3 以该文件为准，speaker ID 不得随意重编号（§12.6）。

### Stage 1b 启动：构建脚本与处理参数（全量运行前固定，2026-09-27）
- 新脚本 `scripts/build_stage1_dataset.py`：复用 `slicer.py` 的 Slicer（§12.3）与 `select_opensinger.py` 的 analyze_file/阈值常量（§12.4），不重复实现逻辑。源数据只读；全部产物写入 `data/timbre_blend_stage1/`（gitignored）。
- 响度归一化（§12.2，全 Stage 1 统一参数）：ffmpeg 两遍 loudnorm，`I=-23 LUFS, TP=-1.5 dBTP, LRA=11, linear=true`（纯静态增益，保留动态范围，真峰限制不引入 clipping）；输出 44100 Hz 单声道 pcm_s16le。实测：归一化文件积分响度 -23.01 LUFS；静态增益对 SNR 代理完全不变（18.81 → 18.81 dB）。非峰值拉满。
- 采样率：源数据即 44100 Hz，与仓库 `configs/reflow.yaml sampling_rate: 44100` 一致，无需重采样（ffmpeg `-ar 44100` 仅兜底）。
- 切片（§12.3）：repo `Slicer(threshold=-40dB, min_length=2000ms, min_interval=300ms, hop=20ms, max_sil_kept=500ms)`。min_length=2000ms 对应模型最短训练时长（`data.duration: 2`）；max_sil_kept=500 偏离 repo 默认 5000（该默认是推理分块参数），训练用途裁掉边缘长静音。仅保留 slice==False 且 ≥2.0s 的 chunk；不覆盖任何原始音频。
- 切片级质量复检（§12.4，GPU RMVPE，复用 Stage 1a analyze_file）：duration≥2s、voiced_ratio≥0.25、clip_ratio≤0.01、oct_jumps≤30/min——阈值数值与 Stage 1a 完全一致，未做任何调整。
- **方法学决定（在任何全量运行/训练之前做出，附实证）**：SNR 代理（p95−p20 帧能量差）不作为切片级剔除判据，改由 Stage 1a 源文件级预过滤强制执行（≥15 dB，入池文件已全部通过）。理由：该代理度量录音本底属性，被静音裁剪系统性扭曲——实测干净样本整文件 18.81 dB，仅裁掉 0.3s 边缘静音后跌至 13.74 dB（裁剪移除最低能量帧使 p20 抬升）；冒烟测试中该判据 4 切片误杀 1 个干净切片（若全量保持该判据将带偏倚地丢弃大量干净数据）。五个阈值数值全部未变；每个切片的 snr_db 仍完整记录于 `data/timbre_blend_stage1/slice_metrics.jsonl`，两种判据均可离线复盘。
- 数据组织（§12.5/12.6/§11）：pool = Stage 1a usable 源文件按 seed=20260927 洗牌后累计至 60 min/人封顶；**源文件级**（原始演唱片段级）95/5 train/val 划分，稳定可复现；speaker 映射 = 歌手号升序 → spk_id 1..12（`reflow/data_loaders.py` 取 audio 子目录名首个 `_`/`-` token，1-based），目录名 `<spk_id>_singer<NN>`；映射固化 `reports/timbre_blend_speakers.json`，下游 Stage 2/3 不得重编号（Stage 2 虚拟歌手将追加为 spk 13，走已实现的 N→N+1 扩展）。HOLDOUT [29,47,10] 不进训练树，源音频留在官方 WomanRaw 根作 unseen test。
- 冒烟测试（--singers 36 --limit-per-singer 4，独立 smoke 目录）：loudnorm 4/4 ok、切片 4、train 3 + val 1、短切片 0、质检误杀 0、输出 44100Hz 单声道 PCM_16、目录结构与 data loader 解析规则吻合。
- 全量构建启动：tmux `stage1b`，日志 `.tmp/build_stage1.log`，预计 ~20-30 min（loudnorm 8 并发 + GPU 复检）。完成后汇报每歌手统计，再进入 §12.7 配置与 preprocess。

### Stage 1b 全量构建完成（2026-09-27）
- tmux `stage1b`，BUILD_EXIT=0，12:25 启动约 25 min 完成。12 歌手全部构建：pool 6041 源文件（464.0 min），loudnorm 6041/6041 成功（0 measure/apply 失败）。
- 产出：**train 6664 slices / 394.85 min，val 345 slices / 20.3 min**。每歌手 train 23.0-51.7 min（仅 singer 14 触发 60 min 池上限：926→826 文件）；拒绝统计：过短丢弃共 1223、质检剔除共 48（约占切片总数 0.7%，明细 `data/timbre_blend_stage1/rejected.jsonl`）。
- 产物：`data/timbre_blend_stage1/{train,val}/audio`、`normalized/`、`manifest.json`、`slice_metrics.jsonl`、`rejected.jsonl`；`reports/timbre_blend_speakers.json` 与 `reports/timbre_blend_stage1_build.json` 已提交。smoke 树 `data/timbre_blend_stage1_smoke` 已删除（可再生）。

### 用户补充验收要求（2026-09-27）与落实
- **要求 1：preprocess 不得以 exit 0 判定成功**（worker 异常路径打印后继续）。工具 `scripts/audit_preprocess_features.py`：对 {train,val}/audio 每个文件核对 6 特征 npy（units/f0/volume/mel/aug_mel/aug_vol）**存在且完整可读**（np.load 全量读取，可抓截断/损坏）、形状（f0/volume/aug_vol 1-D；mel/aug_mel/units 2-D 且 axis0=帧，与 loader get_npy_shape[0] 约定一致）、六特征帧数极差 ≤2（loader 取 min() 容忍 ±1，更大即损坏判 FAIL）、`pitch_aug_dict.npy` 完整（loader 按 dict[rel] 取值，缺键即训练崩溃；train ∈ [-5,5]，val 全 0）、`skip/` 必须为空（builder 已 RMVPE 预检每个切片，任何 skip 都是异常）、孤儿 npy（WARN）、与 build report 切片数交叉核对。任一 FAIL 级问题 → 列出全部路径、exit 1、判 preprocess 未通过。
- 要求 1 自测（合成 fixture，`.tmp/make_audit_fixture.py`）：坏树 7 类缺陷全部抓获（缺特征/损坏 npy/帧差 7/形状违规/pitch_aug 缺键+鬼键+越界/skip 文件/孤儿 npy）exit 1，且同 run 内干净 val split 判 PASS；干净树整体 PASS exit 0。无假阴性。
- **要求 2：linear=true 不等于实际线性，须汇总第二遍 loudnorm 的 normalization_type；存在 dynamic → 停下报告，不得直接进 preprocess**。本次全量构建时脚本丢弃了 pass-2 stderr，故用事后补测：`scripts/verify_loudnorm.py`（loudnorm 两遍均确定性，已实证复现 pass-1 测量值逐位一致）。每文件：复现 pass-2 到 null 解析 Normalization Type + 磁盘实证（帧能量代理 p95−p20 不变性、活跃帧增益残差 robust std、LRA 不变性、实测响度 vs 按源码推导的 expected_i）。预登记判据（全量验证运行前固定）：type==Linear；|dproxy|≤0.30 dB；|dlra|≤0.50 LU；|i_norm−expected_i|≤1.0 LUFS。
- builder 同步补丁（未来运行原生满足要求 2）：apply_loudnorm 解析 pass-2 Normalization Type，per-singer stats 计入 norm_type_*，任何非 Linear 回退 → 记录 rejected.jsonl 并立即中止构建。
- **机制发现（ffmpeg 8.0.1 af_loudnorm.c 源码 + 受控实验双重确认）**：
  1. 整文件 <3s：filter_frame() 无条件转 LINEAR_MODE（短文件规则，增益现场计算、TP 超限时压到 TP 上限而非转 dynamic）。
  2. ≥3s：init() 授予 linear 需 measured_tp≠99 ∧ measured_thresh≠−70 ∧ **measured_lra≠0** ∧ measured_i≠0 ∧ TP 余量 ∧ lra≤target。**实测 LRA 恰为 0.00（短且响度均匀文件的统计简并）与「未提供测量值」哨兵碰撞 → 回退 Dynamic**，音频本身无任何问题。铁证：smoke 文件 36_一笑倾城_12（3.09s，LRA 0.00）按实测值复现 = Dynamic，改喂 measured_LRA=2.5 = Linear。
  3. LINEAR_MODE 下 init() 用 target_I−measured_I 覆盖 s->offset，**用户 offset 参数被丢弃**；expected_i 公式据此修正（smoke di_err max 0.45→0.08 LU）。
- smoke 树验证结果：3 Linear（含 1 短文件规则）+ 1 Dynamic（lra_zero_sentinel）；该 Dynamic 文件磁盘实证：增益恒定（帧增益残差 std 0.0000 dB、dproxy 0.0001 dB、dlra 0.0、限幅器未触发 tp_norm −11.2 ≪ −1.5）。
- 全量验证（6041 文件）：tmux `verifyln`，日志 `.tmp/verify_loudnorm.log`，报告 `reports/timbre_blend_stage1_loudnorm_verification.json` + 逐文件 `data/timbre_blend_stage1/loudnorm_verification.jsonl`。按用户指令：如存在 Dynamic → 携每文件机制分类 + 磁盘实证停下报告，等待决定（候选项：按实证接受 / 对受影响文件以显式静态增益重归一化并重建其切片 / 剔除），不得自行进入 preprocess。

### Stage 1b 验收检查 2 结果：loudnorm 全量验证 FAIL → 管线停在关卡（2026-09-27）
- `scripts/verify_loudnorm.py` 全量 6041 文件（tmux verifyln，约 2.6 min，39 文件/s）：**5991 Linear + 50 Dynamic**，VERDICT FAIL（exit 1）。汇总报告 `reports/timbre_blend_stage1_loudnorm_verification.json`（已提交），逐文件证据 `data/timbre_blend_stage1/loudnorm_verification.jsonl`（gitignored）。
- Linear 文件零失败：全树 dproxy p99=0.0069 dB、rstd p99=0.0002、dlra p99=0.1、di p99=0.14 —— 对 99.2% 的文件，「纯静态增益、动态保留、响度到位」成立。机制分布：init_linear 4955、short_file_rule 1036（<3s 短文件规则，源码确认为无条件 LINEAR_MODE）。
- 50 个 Dynamic 的机制分解（ffmpeg 8.0.1 af_loudnorm.c 源码 + 受控实验 + 时长相关性三重确认）：
  - **35 个 `lra_above_target`**（时长 5.5-13.4s，源实测 LRA > 目标 LRA=11）：init() 条件 `measured_lra <= target_lra` 不满足 → 按 loudnorm 设计执行真动态压缩。磁盘实证动态确被改变：dproxy −3.14~+1.04 dB、rstd 最大 2.17、dlra −5.1~+1.1 LU；28 个超 |dproxy|>0.30、27 个超 |dlra|>0.50、18 个超响度偏差（有重叠），其余 3 个压缩轻微仍在阈值内。**该类违背「只改整体增益、不动原录音动态」的目标，不建议按原样接受**。
  - **15 个 `lra_zero_sentinel`**（时长 3.02-3.69s，源 LRA 实测恰为 0.00，与 init() 的「未提供测量值」哨兵 `measured_lra != 0` 碰撞）：磁盘实证增益完全恒定（dproxy ≤ 0.0032 dB、rstd ≤ 0.0002、dlra ≤ 0.1、di ≤ 0.2、限幅器未触发）—— Dynamic 标签属哨兵误报，实际效果与纯静态增益无异。铁证：同文件改喂 measured_LRA=2.5 复现即为 Linear。
- 修复可行性已验证：50 个文件全部可用**显式静态增益**（gain = −23 − measured_I，范围 −8.55~+8.66 dB，i_src 范围 −31.66~−14.45）归一到 −23 LUFS 且真峰保持 ≤ −1.5 dBTP（50/50 TP-safe，无需封顶、不会引入削波）。
- 影响面：66 个切片（train 64 + val 2）/ 4.60 min，占全树 415.2 min 的 **1.11%**；lra_zero 类切片仅 0.73 min。每歌手最多 spk2 singer14（16/918 切片），其余 ≤ 11。
- 按用户指令（2026-09-27）：**管线停在 preprocess 之前**，已报告决策选项（A 修复全部 50 / B 剔除 50 源切片 / C 按原样接受 / D 混合：15 按实证接受 + 35 修复或剔除），推荐 A。等待用户决定，决定后记录并继续。

### 【人工关卡·补充验收】用户决策：A+ 全量修复（2026-09-27）
- 用户选择选项 A 并升级为生产流水线标准 A+：**50 个 Dynamic 源文件全部修复，不删除，也不接受 Dynamic 处理结果**。
- 决策理由（用户给出）：
  1. 35 个 lra_above_target 文件恰含较强真实演唱动态，删除会产生数据选择偏差；
  2. 接受 Dynamic 违背「响度处理只允许整体静态增益、不改变原始动态」的数据原则；
  3. 15 个 sentinel 文件虽磁盘实证等价静态增益，但生产流水线不应依赖 FFmpeg 的 sentinel/fallback 特例。
- 用户指令要点（10 条）：①已验证的 5991 个 Linear 文件保留，不全量重建；②50 源重新生成 normalized，不再用 loudnorm 第二遍做实际归一化；③新生产归一化 = 仅测量（integrated loudness + true peak）→ gain_loudness = target_I − measured_I，gain_peak = target_TP − measured_TP，gain = min(两者) → ffmpeg volume=<gain>dB 单一恒定增益；禁止 compressor/limiter/dynamic loudnorm；target_I = −23 LUFS，target_TP = −1.5 dBTP；④TP 受限时 TP 安全优先，允许最终响度低于 −23，明确记录，不得用动态处理强行达标；⑤本轮 50 个预计全部 TP-safe，磁盘输出后仍需实测确认；⑥重跑 Slicer、删除/替换旧 Dynamic 版本切片、重跑切片级 GPU 质检、保持原 train/val 源划分不得重新随机；⑦build_stage1_dataset.py 今后统一新方案（loudnorm 仅可用于测量，不得执行动态归一化）；⑧verify_loudnorm.py 升级为 static-normalization verifier（帧增益残差/动态代理/积分响度/真峰/无动态压缩）；⑨修复后按清单报告（50/50 成功数、前后切片数与时长、最终 train/val 总时长、响度误差分布、TP 最大值、动态代理 p99/max、TP-limited 清单、动态残留）；⑩全部通过后再进入 preprocess。另补 duration waterfall：stage-1a 入选 → 60min cap → 归一化 → 切片 → QC → 最终 train+val，解释 464.0 → 415.15 min 去向。
- **预登记：修复与复验判据（运行前固定，不得事后调整）**
  - 静态增益：gain_db = min(−23 − measured_I, −1.5 − measured_TP)，测量值来自 loudnorm pass-1（确定性，已实证复现一致）；施加：ffmpeg `volume=<gain>dB` + `-ar 44100 -ac 1 -c:a pcm_s16le`（纯恒定乘法，无任何动态环节）。
  - 修复后磁盘复测（逐文件）：tp_norm ≤ −1.2 dBTP；|i_norm − (i_src + gain)| ≤ 1.0 LU；任一不过 → 该文件记修复失败并列出。
  - 全树复验（verify_loudnorm.py v2，6041 文件全部通过才 PASS）：① 帧增益残差 rstd ≤ 0.05 dB（恒定增益 ⇒ 无动态压缩）；② |dproxy| ≤ 0.30 dB（动态代理不变）；③ |dlra| ≤ 0.50 LU（响度范围不变）；④ |i_norm − (i_src + expected_gain)| ≤ 1.00 LUFS，expected_gain = min(−23 − i_src, −1.5 − tp_src)（历史 short-file-rule 文件的 loudnorm 现场 global 测量对 pass-1 有 ~0.3-0.7 LU 抖动，由 1.0 容差吸收，v1 全量实测 Linear 文件 di p99=0.14 佐证）；⑤ tp_norm ≤ −1.20 dBTP（上限 −1.5 + 0.3 测量余量）；⑥ 树完整性（pool = 磁盘 = 验证数，orphans/missing = 0）。
  - 切片质检：与构建时完全相同的固定阈值（duration ≥ 2s、voiced ≥ 0.25、clip ≤ 0.01、oct ≤ 30/min，GPU rmvpe）。
- 执行记录与结果见后续条目。

---

## 2026-09-27 A+ 修复执行：50 文件静态增益重制 + 全树 v2 复验 PASS + 时长瀑布

### 修复执行（scripts/remediate_static_gain.py，13:10:49–13:11:08，REMEDIATE_EXIT=0）
- 输入：v1 复验 jsonl 中 50 个 Dynamic 文件（35 lra_above_target + 15 lra_zero_sentinel）；所有测量新鲜重取，不信任旧值
- 方法（指令③⑦）：gain = min(−23 − measured_I, −1.5 − measured_TP)，ffmpeg volume=<gain>dB 单次常数乘法覆写 normalized；全程无 compressor / limiter / 动态 loudnorm
- 结果（指令⑨）：
  - 50/50 成功；写后磁盘复检 0 失败（tp_norm ≤ −1.2 dBTP ∧ |i_err| ≤ 1.0 LU）；census mismatch 0
  - 增益范围 −8.55 ~ +8.66 dB；tp_limited = 0（指令④未触发：全部达到 −23 LUFS 且 TP 有余量）
  - 响度误差（50 文件）：median 0.01 / p99 0.12 / max 0.13 LU
  - 修复后 tp_norm（50 文件）：max −4.18 dBTP
  - 切片（指令⑥）：旧 66（4.60 min）→ 重写 67（<2s 丢弃 11，GPU QC 移除 0）→ 最终 67（4.66 min）；净 +1 片 / +0.06 min
  - 3 个文件切片数变化（去除动态压缩后 −40dB 切点位置改变）：14_你把我灌醉_23 1→2，19_给我一个理由忘记_39 2→1，30_你把我灌醉_23 1→2
  - slice_metrics.jsonl 清除 66 条陈旧记录（防 __s{k} 同名缓存碰撞），保留 6991
  - train/val 源文件划分保持原样（manifest 权威，未重新随机）
- 最终全树磁盘普查（reports/timbre_blend_stage1_remediation.json 为权威，supersede 构建报告计数）：train 6665 切片 / 394.90 min，val 345 切片 / 20.30 min（修复前 6664 / 345）

### 静态归一化 v2 全树复验（scripts/verify_loudnorm.py v2，13:11:08–13:13:20，VERIFY_EXIT=0）
- 6041/6041 全部通过，orphans 0 / missing 0，fail reasons: none → VERDICT: PASS
- 预登记判据（daa5627 已登记，未做任何事后调整）实测：
  - ① rstd ≤ 0.05 dB：median 0.0001 / max 0.004
  - ② |dproxy| ≤ 0.30 dB：p99 0.0037 / max 0.0599
  - ③ |dlra| ≤ 0.50 LU：max 0.1
  - ④ |di_err| ≤ 1.00 LUFS：p99 0.096 / max 0.22
  - ⑤ tp_norm ≤ −1.20 dBTP：全树 max −1.7
  - ⑥ 树完整性：pool = disk = verified = 6041
- gain_err（磁盘实际帧增益 vs 公式增益）max 0.005 dB → 全树零动态残留（指令⑨动态残留项）
- len_diff_samples 全树 max 0 → 归一化时长不变（瀑布 C 步证据）
- 判据有效性实证：修复前对 50 个 Dynamic 文件跑 v2 → 33 异常（rstd 32 / dproxy 28 / dlra 27 / di 18）；legacy Linear 前 20 文件正样本 → 0 异常
- 报告：reports/timbre_blend_stage1_static_norm_verification.json（逐文件 jsonl 证据在 data/ 下，不入库）

### 时长瀑布（464 min → 415.2 min 去向；只读探针，全部对账通过）
| 步骤 | 文件/切片 | 分钟 | 损失与说明 |
|---|---|---|---|
| A stage-1a 可用（12 歌手） | 6141 | 471.74 | — |
| B 60-min cap 后池 | 6041 | 464.06 | −7.68：cap（仅 singer 14：67.7→60.0） |
| C 归一化后 | 6041 | 464.06 | 0：静态增益时长不变（len_diff=0） |
| D 切片后（≥2s，QC 前） | 7058 | 417.34 | −46.72：Slicer 静音修剪 + 1223 个 <2s 短块丢弃（合计口径；逐源时长未记录） |
| E 切片 QC 后（最终树） | 7010 | 415.20 | −2.14：QC 移除 48 片（全部来自原构建，修复批 0） |
| F train | 6665 | 394.90 | — |
| F val | 345 | 20.30 | — |
- 对账：E 计数 = D − QC = 7010 ✓；E 分钟差 0.009（舍入）✓
- 决策说明：C→D 损失无法逐源拆分（Slicer 未记录静音/短块时长），按聚合口径报告；此为构建期记录粒度限制，非数据丢失

### 环境异常记录（系统级瞬时不稳定，非仓库/venv 问题）
- 13:14 preprocess 首启（-j 4，tmux）3 秒内 SIGSEGV（exit 139）；-j 1 前台复跑出现 scipy.signal 导入期不可能异常（AttributeError：int 对象无 _l 属性——正常 Python 语义无法构造）
- dmesg 证据：约 3.6h 内多个无关进程共 6 次不可能故障——LingXiAgentPack invalid opcode ×4、python 执行数据地址（ip==fault addr）×1、python 空指针调用 ×1 → 判定宿主/WSL2 瞬时内存损坏窗口
- 复测全绿：scipy.signal 单独导入 6/6、preprocess 完整导入序列 3/3、真实 preprocess -j 1 三次 45s 存活并正常处理（~10–15 it/s）
- 处置：继续 -j 4 正式运行。安全网：preprocess 无跳过逻辑（全量覆写，探针残留必被覆盖）；audit 关卡 eager np.load 兜底捕获任何损坏；源数据只读
- 建议主人留意宿主机稳定性（内存压力 / 温度 / LingXiAgentPack 崩溃史）

### 指令⑩：进入 preprocess
- 13:23:10 tmux ppchain 启动：preprocess（-d cuda -j 4）→（exit 0 门）→ audit_preprocess_features（--build-report = 修复报告，期望 train 6665 / val 345）
- GPU 每 120s 记录 → logs/gpu_preprocess.csv
- 通过后：spk-41 F0 复核（关卡 1 遗留义务，判据已预登记：harvest/rmvpe 中位比值多数 ≥1.5 → 系统性低八度 → STOP；[0.75,1.33] 且 ≥1.5 占比 <20% → 真实低音区 → 继续）→ Stage 1c

### 环境异常记录——更正（用户澄清，2026-09-27）
- 用户澄清：LingXiAgent（dmesg 中的 LingXiAgentPack）是同机 Windows/WSL 上并行运行、由另一 agent 管理的测试项目，其崩溃记录（invalid opcode ×4）属该项目自身测试内容，不作为宿主机不稳定证据
- 前条「多个无关进程共 6 次不可能故障」更正为：与本任务相关的仅 2 次 python 故障（13:14 窗口内 SIGSEGV + scipy 导入期不可能 AttributeError），原因未定，属瞬时事件；其后 12 次复测全绿（含 3 次真实 preprocess 45s 存活）
- 撤回前条中「留意宿主机稳定性 / LingXiAgentPack 崩溃史」的建议
- 运行提示：同机存在并行测试项目，若发生 GPU/内存争用，症状会是 CUDA OOM 或显著变慢——pp 链若失败先查此项；正常运行则无需任何动作

### Stage 1b preprocess + 特征 audit + spk-41 F0 复核 全部通过（13:23–13:31）

#### preprocess（13:23:10–13:27:33，PREPROC_EXIT=0）
- `preprocess.py -c configs/timbre_blend_stage1.yaml -d cuda -j 4`，7010 切片耗时 4.4 min
- `[Error]` 0 条（无 worker 异常被吞）；pitch_aug_dict.npy train/val 双 split 落盘
- GPU 日志：logs/gpu_preprocess.csv（120s 粒度）
- 备注：正式启动前有 3 次 -j 1 探针（45s timeout，诊断环境异常用）；preprocess 无跳过逻辑、全量覆写，探针残留无影响

#### 特征 audit（验收要求 1：exit 0 不充分，逐文件特征完整性审计）
- AUDIT_EXIT=0，**AUDIT VERDICT: PASS**（reports/timbre_blend_stage1_preprocess_audit.json）
- train 6665 / val 345：missing 0、unreadable 0、skip/ 空、orphans 0
- pitch_aug：keys 全配齐（6665/345），missing 0、extra 0、bad_vals 0
- 六特征（mel/f0/volume/units/aug_mel/aug_vol）eager np.load 全部可读，帧数对齐通过
- 按 spk F0 中位（train）：284–344 Hz，唯一例外 11_singer41 = 203.4 Hz → 触发 spk-41 复核

#### spk-41 F0 复核（关卡 1 遗留义务，预登记判据，scripts/spk41_f0_review.py）
- 方法：最终训练切片上逐文件对照 harvest（pyworld，独立第二提取器）voiced 中位 / rmvpe（训练实际使用的 f0 特征 npy）voiced 中位；spk-41 全量 544 片 + 对照歌手各 150 片
- 结果：

| spk | n | ratio_med | p10–p90 | ≥1.5 占比 | rmvpe Hz | harvest Hz |
|---|---|---|---|---|---|---|
| 11_singer41 | 544 | 1.0033 | 0.98–1.03 | 0.0% | 204.2 | 203.8 |
| 10_singer36（对照） | 150 | 0.9986 | 0.97–1.02 | 0.0% | 305.6 | 308.1 |
| 2_singer14（对照） | 150 | 1.0005 | 0.98–1.02 | 0.0% | 320.4 | 319.5 |

- **判定：CONTINUE——真实低音区，无八度病理**。预登记 CONTINUE 判据（ratio_med ∈ [0.75,1.33] 且 ≥1.5 占比 <20%）满足，实测 ≥1.5 占比为 0%；STOP 判据（多数 ≥1.5）完全未触发
- 结论：singer 41 的 ~204 Hz 为真实女低音/中音音区，RMVPE 跟踪正确；关卡 1 决定（保留为低音区训练歌手）维持，未改动任何既定阈值
- 证据：reports/timbre_blend_stage1_spk41_f0_review.json（844 片逐文件）

#### 下一步
- Stage 1c：200-step bf16 smoke（计划 §13.2 清单）→ 正式训练 →【人工关卡 2】

---

## 2026-09-27 · Stage 1c smoke 测试：一次死锁 → 根因定位与修复 → 二跑全绿（4398 步）

### Smoke #1（13:35–14:42）——FAIL：首次 validation 冻结

- 训练本体健康：100 步/13.7s（8.4 batch/s），loss 10.25→1.73，0 NaN，model_100.pt（219MB）正常落盘，VRAM 峰值 4527MiB/16376MiB，49°C，54.9M 参数全可训练，realism 参数 0
- 第一次 validation（当时为全量 345 样本）在样本 1 处 66 分钟无任何输出，GPU 利用率 0% → 判定冻结；kill tmux 会话（SIGHUP），追加 SMOKE_FAILED 标记。证据：.tmp/smoke_run_hang1.log
- 取证（py-spy/gdb/strace 均未安装，全程 /proc 取证）：
  - 主线程 wchan = futex_do_wait（阻塞等待）
  - pin_memory 线程（TID 1037666，与 fork worker PID 相邻而识别）100% 用户态空转：6 秒内 606 jiffies，State R
  - 独立复现脚本（同 ckpt、同样本、完整推理链：euler-50 推理 0.52s @420it/s + NSF vocoder 0.31s）秒级通过 → 算法与推理路径无罪
  - 日志探针（matplotlib pcolor 0.12s、librosa.load 1.04s、TB add_audio）全部快速 → 排除日志慢
- **根因**：DataLoader pin_memory 线程的 cudaHostAlloc 与主线程在 test() 中的 CUDA 调用在 WSL2 半虚拟化 GPU 驱动下互相死锁

### 修复集（3 文件；新增键缺省 = 旧行为，向后兼容）

1. `reflow/data_loaders.py`：两个 loader 均 `pin_memory=False`。数据本就全量驻内存（cache_all_data），pinned 拷贝收益 ≈ 0
2. `reflow/solver.py` `test()`：
   - 新增 `val_max_batches` 确定性子集（val loader shuffle=False，子集跨轮可比），分母改为 `processed_batches`
   - 重 TB 媒体日志（谱图 + librosa 解码 + 音频）仅记录前 `val_log_samples` 个样本
3. `configs/timbre_blend_stage1.yaml`：`interval_val` 2000→10000（与 interval_force_save 对齐，每个 val ckpt 均保留）；新增 `val_max_batches: 96`、`val_log_samples: 3`
- 成本论证：全量 345 样本 validation（euler-50 + NSF + librosa + matplotlib + TB 音频）> 10 分钟/轮，对比每 2000 步训练约 4 分钟 → 全量验证经济上不可行；96 样本子集实测 30–60s/轮
- 回归：pytest 40 passed（17 realism + 23 stage 2a；reflow_exclude_spk 掩码测试不受影响，聚合语义保持）

### Smoke #2（14:44–15:09）——PASS：4398 步 + 43 次 validation 零死锁

- **运维事故与教训**：wrapper 的 `kill -INT` 停机失效——非交互 bash 脚本内 `&` 启动的后台进程继承 SIGINT/SIGQUIT 的 SIG_IGN（实测 /proc SigIgn=0x1301006），CPython 保留继承的 SIG_IGN 不装 KeyboardInterrupt handler，停机信号静默无效，wrapper 在 wait 处死等 16 分钟（期间训练照常推进到 4398 步、43 个 ckpt 全保留）。人工 `kill -TERM` 一次成功（exit 143）。**规约：停训练进程一律 SIGTERM；启动一律 `PYTHONUNBUFFERED=1` 保证重定向 stdout 实时可 tail**
- §13.2 清单逐项：
  - 步数真实推进：4398 步 / 20:37（计划要求 200），139 batch/epoch 正常轮转
  - 两 loss 有限：43 次 validation 均打印 test_ddsp_loss / test_reflow_loss，全部有限；val ddsp 1.0749（step100）→ 0.4441（step4300），val reflow 0.1351 → 0.0449
  - 无 NaN/Inf：`[x] nan` 0 次（训练循环对两分量分别 isnan 检查）；log_info 全文 inf/nan 0 命中
  - backward/optimizer 正常：train loss 前 20 步均值 8.52 → 末 20 步均值 0.369；val 组合 loss 1.210 → 0.829 → 0.715 → …（窗口内持续下降，非单点判断）
  - VRAM 余量：4522–4527MiB / 16376MiB（28%）稳定无爬升
  - bf16 真跑通：amp_dtype=bf16 全程 4398 步稳定，无回退
  - ckpt 保存 + 重载：43 个 ckpt 正常落盘；重载另证（下）
  - 日志正常：log_info / stdout / logs/gpu_smoke.csv 齐全；重媒体日志 cap 生效（每轮 4 条 Mel Val = val_log_samples 截断，而非 96 条）
- 吞吐与温度：7.4–8.5 batch/s；validation RTF 0.02–0.19；GPU 温度峰值 85°C（validation 推理 burst 期间出现，csv 全程记录，未见持续降频迹象）
- **重载/续训证明**（15:08:59–15:09:30，.tmp/run_resume_test.sh，输出 .tmp/resume_test.log）：同 expdir 重启 → `[*] restoring model from exp/timbre_blend_stage1_smoke/model_4300.pt`；`optimizer state: none in checkpoint (save_opt=false?)`（符合配置预期）；步数自 4301 续进至 4320（阈值 4308）；0 NaN；SIGTERM 停机 OK
- 清理：smoke expdir（43 ckpts ≈ 9.5GB，可再生 runtime state）证据固化后删除，磁盘占用 71G→62G。证据保留：.tmp/smoke2_log_info.txt、.tmp/smoke_run.log、.tmp/smoke_run_hang1.log、.tmp/resume_test.log、logs/gpu_smoke.csv

### 结论与下一步

- §13.2 smoke 清单全绿 → 进入 §13.3 正式训练：exp/timbre_blend_stage1 全新目录，interval_val=10000（约 20 分钟训练 + 30–60s validation/轮），总步数由收敛情况决定（validation loss 平台期判停），不机械刷 epoch
- 正式训练完成 + §13.4 样本推理后 →【人工关卡 2】

## 2026-09-27 Stage 1 正式训练 + §13.4 推理样本（→ 人工关卡 2）

### 正式训练（OpenSinger 12 女声多说话人预训练）
- 时间窗：15:12:05 → 16:40:39（**1h28m34s**），tmux `strain1`，SIGTERM 干净停机（TRAIN_EXIT=143，TRAIN_DONE 标记齐全）
- 最终 step **40644**（epoch 292 / 139 batches），速度 7.6–8.3 batch/s，**0 NaN/inf**
- ckpt：model_10000/20000/30000/40000.pt 全部保留（interval_val = interval_force_save = 10000，无删除）
- GPU：峰值 VRAM **4542 MiB / 16376**，峰值温度 **84°C**，clocks 2175–2310 MHz 无降频（logs/gpu_stage1.csv，600s 采样）
- lr 阶梯按配置执行（5e-4 × 0.9^(step/4000)），停机时 lr = 1.74e-4

### 验证轨迹（固定 96 batch 子集，shuffle=False）
| step | val ddsp | 每万步改善 | val reflow | 组合 | train loss(窗口均值) |
|---|---|---|---|---|---|
| 10k | 0.43804 | — | 0.0426 | 0.481 | 0.252→ |
| 20k | 0.42546 | 2.87% | 0.0449 | **0.470（最低）** | 0.252 |
| 30k | 0.42484 | 0.15% | 0.0504 | 0.475 | 0.227 |
| 40k | 0.42935 | **−1.06%（回升）** | **0.0711（暴涨 +41%）** | 0.501 | 0.214 |

### 停训判定（预登记规则的执行记录）
预登记规则：「连续 ≥3 个 val 点 ddsp 每万步改善 <1% 且 reflow 无下行趋势 → 停」。实际在 30k 拿到第一票平台票（0.15%）后，**40k 出现规则未覆盖的情形：val 回升**——ddsp 转升 +1.06%、reflow 四点单调上升（0.0426→0.0449→0.0504→0.0711）冲穿 smoke 噪声带（0.041–0.048）、组合 val 0.501 比 10k 还差、train/val gap 拉大到 2.34。按 30k 时预先公布的双分支处置（「40k 不回落 → 停，并记录为对三票规则的提前触发，依据是过拟合而非平台」）执行停机。**不属于事后挪门槛：分支判据在 30k 当轮已原文公布。**
- **最优 ckpt = model_20000.pt**（md5 e026eb6e60b7e2e8ba4c579ad8a0ceff）：组合 val 最低 0.470；ddsp 与 30k 差 0.15% 属噪声级，而 reflow 明显更优
- 过拟合形态备注：train loss 全程健康下降（0.252→0.214），val 20k 后掉头——12 说话人 × 395 分钟数据在 batch 48 下 20k 步 ≈ 144 epoch，记忆化起点与数据规模自洽

### §13.4 推理样本（9 主样本 + 3 对比样本）
- 输入全部取自 **val 集**（未参与训练），覆盖 3 位已训歌手 spk2 / spk10 / spk11
- 参数：`-ts 0.0 -step 50 -method euler -k 0 -f 0 -pe rmvpe -th -60 -fmin 50 -fmax 1100`
- 矩阵：6 条同说话人重建（每位歌手 ×2）+ 3 条跨说话人转换（41_一次就好→spk2、14_从前慢→spk11、36_云烟成雨→spk2）
- 附加 `compare_40k/`：model_40000.pt（过拟合 ckpt）对 3 条相同输入的输出，供关卡 2 A/B 试听验证 ckpt 选择
- 路径：远端 `samples/stage1/`（records.tsv 13 列 × 9 行 + manifest.json）；Mac `~/Downloads/timbre_blend_stage1_samples/`（含 inputs/ 原始切片供对照）
- 清单归档（git 追踪）：`reports/timbre_blend_stage1_samples_manifest.json`

### 异常记录（均为瞬时宿主不稳，非代码问题，重试即愈）
1. 样本 spk11__14_从前慢_22__s2 首跑 **SIGSEGV**；同参数原样重试成功
2. 40k 对比 spk2 首跑**挂起**（主线程 R 态 100% CPU 空转 12 分钟、54 线程 futex 等待、零输出、GPU 0%）；kill 后带 `timeout -k 10 120` 护栏分离重试，**11 秒完成**
3. 两起均落在本日已记录的 WSL2 宿主不稳窗口；同二进制同输入重试成功 → 判定环境抖动，未改任何代码

### 断点恢复检查
- 恢复机制已于 15:08 实测验证（smoke ckpt model_4300.pt 重载、step 4301→4320 续跑、TERM_KILL_OK）
- 正式训练如需续跑：重launch `.tmp/run_stage1_train.sh` 即从 model_40000.pt 恢复（save_opt=false → 优化器重建 + lr 按 step 重算，smoke 已验证该路径无异常）
- 但基于过拟合判定，**不建议续跑**；20k ckpt 为交付候选

### TensorBoard 服务（长期方案落地）
- Windows 防火墙新增入站规则 `TensorBoard WSL 6006`：TCP 6006 / 仅 Private 网络类别 / 源限定 10.0.0.0/24（最小暴露，重启不丢）；SSH 隧道方案退役
- LAN 直连 `http://10.0.0.150:6006` 实测 HTTP 200；SCALARS / IMAGES（前 3 个 val 样本谱图）/ AUDIO（10k–40k 四代 ckpt 试听）齐全
- WSL2 三个坑的修复保留在 `.tmp/run_tb.py`：Rust data-server 静默死亡（--load_fast=false）、未绑定回环端口 connect_ex 黑洞（1s 超时包装）、镜像模式 LAN 入站需防火墙规则

→ **【人工关卡 2】**：训练与样本交付完毕，等待主人试听验收后进入 Stage 2（虚拟歌手 spk13 微调）。

## 2026-09-27 Stage 2 启动前修正（关卡 2 后）

- **纠错**：先前「Stage 1 歌级 train/val 零交集」的判断错误，检查器当时只去掉 `__sN`，漏掉 OpenSinger 同一首歌不同 `_段号`。重新按 `(歌手, 整首歌)` 检查：Stage 1 训练 **230** 首 / 验证 **162** 首，交集 **162**；虚拟歌手原 `sliced-v2` 训练/验证各 **11** 首，交集 **11**。这两个旧验证集均有泄漏，Stage 1 `model_20000.pt` 是在泄漏验证集上选的，真实最佳点可能早于 20k。0.4255 **仅作历史参考**，不得用于新集遗忘阈值。已征得主人同意：保留该 ckpt，在固定新集上重测基线。
- 脚本：`scripts/check_song_split_leakage.py`（缺失目录也判 FAIL），测试 `tests/test_song_key.py`。纠错前报告：`reports/split_leakage_before_stage1.json` 和 `reports/split_leakage_before_slicedv2.json`。
- 新建 **独立** `data/timbre_blend_stage2/`（原始 Stage 1/私有音频不动），歌曲级确定性划分；train **220 首/6590 切片**，val **21 首/592 切片**（公开 19 首/558、虚拟 2 首/34），歌曲交集 **0**。19 首公开验证歌从 Stage 2 replay 中全部剔除（脚本断言 PASS）；独立报告 `reports/timbre_blend_stage2_split.json`、`reports/split_leakage_after_stage2.json`。Stage 2 预处理完全沿用 Stage 1 特征参数，7182 个切片逐文件审计 **PASS / 0 缺失 / 0 损坏**（`.tmp/stage2_feature_audit.json`）。
- 重要限制：19 首新的公开 val 歌**全部曾进入 Stage 1 训练**，因此此公开曲线只用于同一固定数据上的**遗忘检测**，不代表未见歌泛化。Stage 2 虚拟歌手对 `model_20000.pt` 为新数据，整歌划分后的虚拟 val 将用于主要 checkpoint 选择。
- 公开歌手 #41（spk11）在 Stage 1 **参与训练**（训练树有 519 切片），原关卡 2 所谓「跨说话人测试」并非未见歌手测试。按原 holdout 名单 [10,29,47] 准备三位未训练歌手的测试输入；原始音频保持只读，推理待基线结束后执行。
- 私有 `LingXiFox/ddsp-svc-lingxi` 是 HF **Storage Bucket**，不是 dataset repo；只以 `HF_ENDPOINT=https://huggingface.co` 访问官方端点。官方 Bucket 直连枚举 172 个 `sliced-v2` WAV，**本机 172 个文件大小逐一匹配**，未向镜像发送私有请求；`datasets/private-timbre/` 已加入 `.gitignore`。仅能证明当前直接元数据和本机文件匹配，先前已存在文件的下载链路未观察，不声称有下载过程审计证据。公开模型仍可使用镜像。
- TensorBoard 现只绑定 **127.0.0.1:6006**（`ss -tln` 实测），不再接受局域网直连；旧 Windows 入站规则虽仍在，但无对外监听。
- 起点 checkpoint **MD5=e026eb6e60b7e2e8ba4c579ad8a0ceff** 与指定值逐字匹配；独立迁移校验：12/12 旧 speaker embedding 行逐字节相等，其他全部可比权重 exact equality；spk13 是新增行，fresh optimizer，Stage 2 step 从 0 计。A/B 两组初始权重相同，独立 expdir。配置均 lr=5e-5、batch=48、虚拟回放目标 50%、每 1000 步验证并全留 ckpt、最多计划 10000 步、realism=false；B freeze_reflow=true + exclude_spk=[13]，A 两项均关闭。固定验证随机种子且恢复训练 RNG，分别记录公开/虚拟全量验证集的 ddsp/reflow/mel 曲线。单测（mask + split + stage2 RNG / replay）28 PASS。
- 新公开 ddsp 初始基线以及全部 `mel_val_*` 将由不训练的 step-0 验证写入 `exp/timbre_blend_stage2_masked/baseline.json`；公开遗忘警戒线是**该新基线 ×1.10**。`mel_val_*` 的每次变化也将完整归档并汇报，不以旧泄漏数值作对照。虚拟歌手曲线先降后连续两升停训，挑拐点附近 checkpoint；遗忘触发则立即停训并先报告。

### Stage 2 零步基线（B 配置，固定全量新验证集）
- `BASELINE_START=18:40:11`，`BASELINE_END=18:42:31`，`EXIT=0`，零次参数更新；模型从 20k 权重迁移至 13 spk，运行前 MD5 通过。
- 公开（558 切片、19 首歌，**只用于遗忘检测、非未见歌泛化**）：ddsp_loss **0.25046291546795957**、reflow_loss **0.03105112985524542**、mel_val_mse **0.321774833999227**、mel_val_snr **45.57452837626139**、mel_val_psnr **44.75988653996512**、mel_val_sisnr **45.570797294698735**；遗忘暂停线 **0.27550920701475555**（新 ddsp 基线 ×1.10，严格“大于”才触发）。
- 虚拟歌手（34 切片、2 首新歌）：ddsp_loss **6.497708446839276**、reflow_loss **0**（按 B 掩码排除）、reflow_loss_excluded_diagnostic **3.3281879800620335**（不参与优化）、mel_val_mse **6.95990044930402**、mel_val_snr **32.53868439618279**、mel_val_psnr **31.358286072226132**、mel_val_sisnr **32.57868643367992**。
- 权威基线 JSON：`exp/timbre_blend_stage2_masked/baseline.json`（可再生运行产物；关卡 3 固化到报告）。
- 补齐**真正未见歌手**输入 [10,29,47]：`~/Downloads/timbre_blend_stage1_samples/holdout_unseen_singers/` 含 3 输入 + 3 输出 + records.tsv。旧 #41 是已训练歌手，其旧跨说话人样本只测了已见歌手间转换，不能替代这 3 条未见歌手测试。

### Stage 2 B（Masked/Frozen）实跑：已按虚拟曲线提前停机
- 18:43:58 → 18:53:25（约 9m27s），3k steps，EXIT 0；计划上限 10k，但预先规定的「虚拟 ddsp 先降后连续两次上升」优先触发。1k / 2k / 3k checkpoint 全留，`model_0.pt` 起点全留。
- 虚拟 ddsp：**6.497708 (step0) → 0.862014 (1k，最低) → 0.863978 (2k，回升 #1) → 0.866100 (3k，回升 #2)**；停机日志 `STAGE2_STOP_REASON=VIRTUAL_TWO_RISES best_step=1000`。B 候选 `exp/timbre_blend_stage2_masked/model_1000.pt`；虚拟 mel MSE 1k=0.916251、3k=0.915953（轻微改善，听感仍需盲听）。
- 公开 ddsp：基线 0.250463 → 1k 0.257112 (+2.655%) → 2k 0.258037 (+3.024%) → 3k 0.260430 (+3.979%)，**均未触发 0.275509 遗忘线**。公开 mel MSE 0.321775→0.328231（+2.007%）；SNR 45.5745→45.4695（-0.105 dB）、PSNR 44.7599→44.6679（-0.092 dB）、SI-SNR 45.5708→45.4663（-0.104 dB），这些退化均已单独记录。
- 实际虚拟采样比例：1k 50.18%、2k 49.78%、3k 50.10%；完全落在约 50% 的预定范围。
- 冻结真实性：`model_0.pt` 与 `model_3000.pt` **全部 72 个 Reflow 张量逐个 exact equality**；DDSP 权重确实变化，冻结不是仅靠梯度标志声称。
- 与原计划第 24 节「至少 5 个有意义中间 checkpoint」不同：用户新增明确停训规则在 3k 抢先触发，仅产出 3 个训练中间点。**不为凑数量继续训练已转升的虚拟验证损失**。对照 A 设相同实际总步数 3k（配置中的计划上限由 10k 改为 3k），其余数据/seed/初始权重一致；163 个起始模型张量全等。A 从 18:54:03 开始，运行中。

### 验证 RNG 路径不一致：A/B 旧结果暂停作为结论（19:01）
- A 的 step-0 公开指标与 B **完全一致**，虚拟 ddsp 却不同（B=6.497708，A=6.471634；虚拟 mel MSE 也不同）。虽然 163 个模型起始权重全等、train/val seed 一致，但两条路径在每条虚拟验证样本之间执行不同次数的随机 Reflow forward（B 为排除后的额外诊断，A 为正常 included 计算），推进了下一条样本的 RNG 状态。**A/B 旧虚拟验证曲线不可直接比较，B 的 3k 早停/1k 最优结论也待重验。**
- 19:01:21 用 SIGTERM 暂停 A（保留 step0 + 1k ckpt，EXIT=143），无任何并行 GPU 训练。已在 `test()` 的验证循环中固定每条样本 RNG = val_seed + index，顶层 RNG fork 仍在验证结束时还原训练 RNG，避免验证分支改变训练数据流；若同权重/同输入真实 A/B step-0 探针全部 ddsp/mel 完全一致，再从 step0 重跑两组并保留旧 exp 目录作审计，不复用有争议的旧早停判定。
- 修正期间一次 pytest 在 `transformers` 模块导入时触发 WSL2 宿主段错误（尚未执行测试体），不当作通过；需探针、重试单测后才重启训练。

### Stage 2 验证随机流修复确认（19:25）
- 改为每条验证样本独立 seed = val_seed + 样本序号（验证结束仍恢复训练 RNG），解决 A/B 前向分支消耗不同随机数导致的后续样本不公平。
- 两组初始 163 权重张量全等，read-only/无 TensorBoard 探针在实际 GPU、全部 558 公开 + 34 虚拟切片上验证：A/B **公开和虚拟 ddsp_loss、mel_val_mse、mel_val_snr、mel_val_psnr、mel_val_sisnr 10/10 数值 exact equality**。Reflow 不需一致（A 正式 included，B excluded 仅诊断）。探针日志 `.tmp/stage2_rng_probe2.log`（PASS）。
- **修正后 step-0 基线**：公开 ddsp_loss 0.25033053218997936，reflow_loss 0.029853311794676315，mel MSE/SNR/PSNR/SI-SNR = 0.32234610595797125 / 45.57268249347646 / 44.754996644980594 / 45.568840860893225；遗忘阈值 = 0.2753635854089773。虚拟 ddsp_loss 6.515204850365134，mel MSE/SNR/PSNR/SI-SNR = 7.006810258416569 / 32.496908748851105 / 31.32902431488037 / 32.536703278036676。旧 step0 基线及旧 B 3k 停止结论**作废**，不得作为后续阈值或候选选择。
- 一次含 CUDA 的全套 pytest 挂住 180 秒（GPU 0%，孤儿进程 100% CPU），TERM 终止；CPU 相关测试 25 PASS / 3 CUDA deselected，真实 GPU A/B 探针通过。先前两个 probe/pytest 段错误分别在 TensorBoard writer CRC 和 Transformers 导入期；为绕开宿主不稳，Stage 2 正式训练禁用 TensorBoard 事件及媒体写入（`train.disable_tensorboard=true`, `val_log_samples=0`），**仍完整写 `validation_history.jsonl` 的两条全量曲线**。Stage 1 TB 服务保持原状。
- 旧 B/A exp 目录均**移动保留**到 `exp/timbre_blend_stage2_{masked,baseline}_invalid_valrng_20260927/`（含 `INVALID_VALIDATION_RNG.txt`），不覆盖/删除原始证据；新运行从 MD5 已核验的 20k 权重 step0 重新启动。A 步数上限先恢复 10k，B 实际停机后再设同实际步数。

### 2026-09-27 19:42-19:45 | 主人修订停训规则 + model_2000 虚拟训练/验证对照
- 主人将虚拟 ddsp 停训投票改为：相对前一次验证点**上升严格超过 2%**才计一次；±2% 以内持平，不计票（且打断「连续两次」）。`reflow/solver.py` 与 A/B 配置 `train.virtual_rise_threshold: 0.02` 已调整，缺省为旧行为。当前 B 进程是在热更新前启动，磁盘改动不会改变内存代码；它在 3k 验证保存后按照**旧**规则自行停机（`STAGE2_B2_EXIT=0`），不是新规则触发。修正后 B 虚拟 ddsp：step0 6.515205 → 1k 0.860010 → 2k 0.864297（较 1k **+0.499%**）→ 3k 0.865626（较 2k **+0.154%**）。按新规则 2k/3k **均为持平，0 张上升票**；公开 3k ddsp 0.260397，低于 0.275364 遗忘线。禁止把旧日志的 VIRTUAL_TWO_RISES 当成新规则决定。
- 按主人指示暂停续训决策，专用 GPU（无并发训练）用 **model_2000.pt** 从 138 条虚拟歌手 Stage 2 train 切片中以 seed=20260927 无放回随机抽取 **34** 条，与虚拟 val **34** 条运行同一个 `reflow.solver.test()`、相同 `val_seed+bidx` 和相同推理设置；关闭媒体/TB 写入，不更新参数。验证侧复算 ddsp **0.864297463613398** 与正式 2k 曲线逐位一致，方法核对 PASS。
- **训练抽样 ddsp 0.22834584730512955 vs 验证 ddsp 0.864297463613398（验证/训练=3.79x）**。训练侧 mel MSE 0.330954 / SNR 44.2271 / PSNR 43.3864 / SI-SNR 44.2240；验证侧 mel MSE 0.921573 / SNR 40.3183 / PSNR 39.3349 / SI-SNR 40.3253。判定 **泛化差距**，不是「两侧都 >0.8」的欠拟合；提高新 speaker embedding LR 5e-4 不针对当前问题，可能加剧过拟合，暂不实施。
- 整歌划分后虚拟歌手 **9 首 25.20 分钟 / 138 切片 train**，**2 首 5.75 分钟 / 34 切片 val**；约 50% 目标采样 × batch 48 × 2k steps，虚拟样本抽取约 48,000 次，即平均每训练切片约 348 次（训练时有随机裁剪，因此并非完全同一 crop）。目标歌曲仅 9 首，多样性不足，当前训练-验证差距较大，建议由主人决定补数据、调整过采样/正则化，或使用已保存的早期 checkpoint 继续 A/B 听感对照。
- 完整选中切片列表与数值留在未追踪 `.tmp/stage2_virtual_train2000_diagnostic.json`；原始数据、切片及各 ckpt 均未删除或覆盖。**此处暂停，不自行重训/续训 A，也不作关卡 3 通过宣告。**

### 2026-09-27 19:45-19:51 | 关卡 2 后追加检查（主人指定，Stage 2 A/B 暂停）
- **训练状态**：A 未重跑、B 在 3k 被旧内存规则停住；按已提交的新规则（单点上升 >2% 才计票）2k +0.499%、3k +0.154% 均不计票。停止在此等待主人补充虚拟歌手歌曲，**不续跑、不改现有 50% 配置、不删除 B@1k / B@2k / B@3k 及原起点模型**。补歌后按整歌划分，新验证集至少 3–4 首，目标采样约 30%，仍从校验过 MD5 的 `model_20000.pt` 重跑 B/A，其余设置不变（包含 >2% 新规则）。
- **F0 分布审计**：读取 `data/timbre_blend_stage2/{train,val}/f0/13_lingxi/*.wav.npy`，只统计有限且 >0 Hz 的预处理 F0 帧。训练 **9 首/138 切片/130306 帧**，p05/中位/p95 = **212.42 / 328.64 / 490.94 Hz**，观测最小—最大 34.25–675.69 Hz；验证 **2 首/34 切片/29731 帧**，p05/中位/p95 = **233.32 / 332.32 / 488.42 Hz**，最小—最大 173.22–591.76 Hz。验证帧中低于训练 p05 占 2.92%，高于训练 p95 占 4.78%，训练实际最小—最大以外 **0%**；r13_08 自身 p95=523 Hz，但训练 r13_12 p95=549 Hz，故目前**无验证歌曲整体超出训练音域的证据**。抽取/插值后的 F0 帧全为正，不能据此声称原音频全部有声。详见 `reports/timbre_blend_stage2_f0_diagnostic.json`（含每首统计）。
- **B@1k 非训练歌手推理探针**：用官方 OpenSinger 训练保留歌手 **10_侧脸_26.wav** 和 **47_秋酿_16.wav** 作输入，均未参加 Stage 1/2 训练；目标 spk13，`-ts 0.0 -step 50 -method euler -k 0 -f 0 -pe rmvpe -th -60 -fmin 50 -fmax 1100` 与关卡 2 推理一致，`PYTHONFAULTHANDLER=1`。模型 `exp/timbre_blend_stage2_masked/model_1000.pt` md5=`18270a6fa8dbe26d37a5ba3b9d164662`。两条输出完整（44.1 kHz 单声道，3.77s / 8.02s，均有非零波形）：远端 `samples/stage2/probe_1k/`，Mac `~/Downloads/timbre_blend_stage2_probe_1k/`（含 inputs/ 和 records.tsv 供 A/B）；追踪清单 `reports/timbre_blend_stage2_probe_1k.json`，wav 不入 git。输出质量仍须主人试听确认，自动数值不能代替听感。
- 后续等待主人新增歌曲及试听反馈；没有声明【人工关卡 3】完成。

## 2026-09-29 Stage 2 C：只训练虚拟说话人嵌入（进行中）
- 主人因补歌受阻批准独立 C 实验：A/B 继续暂停，B@1k/@2k 保留。远端 WSL2 SSH 地址从 `10.0.0.150:2222` 变为 **`10.0.0.244:2222`**；新地址 SSH 已用旧地址保存的主机密钥身份比对后连通（不是关闭密钥校验）。旧日志中的 `.150` 是当时真实历史记录，未篡改。TB 新 IP `http://10.0.0.244:6006/` 当前超时，WSL2 上无 6006 listener，不得声称新地址上的 TB 可用；目前 Stage 2 验证完整曲线仍写 JSONL。
- Stage 1 原 checkpoint `exp/timbre_blend_stage1/model_20000.pt` 校验 MD5 **`e026eb6e60b7e2e8ba4c579ad8a0ceff`**。原模型的 `nn.Embedding` 有 **12 行**：公开号 1–12 对应下标 0–11，新增 spk13 对应下标 **12**，旧迁移路径给新增行的是 PyTorch Embedding 随机初始化。C 在加载后**将全部 12 个 Stage 1 公开嵌入的逐维均值写到第 12 行**，原 12 行不变；不与 A/B 随机新增行混淆。
- C 独立配置 `configs/timbre_blend_stage2_embedding.yaml`，独立入口 `scripts/train_stage2_embedding.py`；只读入 9 首/138 个目标歌手训练切片，采用 `.tmp` 中的硬链接树，公开音频完全不进入训练。除整张 embedding 权重参数外，所有模型参数 `requires_grad=False`；embedding hook 只放行第 12 行梯度，AdamW 仅优化该张表且 `weight_decay=0`，Reflow spk13 排除，优化目标**只有 DDSP loss**。模型保持 eval 状态进行参数训练，以防冻结网络的运行统计缓冲区变化；这与原 B 的 train 状态不相同，是本消融实验的明示差异。每次验证对所有其余参数/缓冲区逐张量 SHA-256 和前 12 行 SHA-256 作严格一致性断言，公开验证全部七项指标与修正后 B step-0 基线做逐项严格相等断言；任一不等即抛错停机。
- 训练参数：batch=48、lr=1e-3、无公开回放、200 步一次三组验证与 checkpoint、最多 3000 步、严格 >2% 连续两次计票早停、`PYTHONFAULTHANDLER=1`、不写 TB 事件。train 曲线固定无放回抽取目标训练切片 34 个，与目标整歌验证 34 个以相同 `test()` 和相同逐样本固定 RNG 评测；公开 558 个切片作冻结断言；每点记录差距、完整指标、其他行哈希。mel 仅留作记录，**不按虚拟 mel 选优**。
- 首次 1-step 烟测发现 `DotDict` 的嵌套属性赋值不回写，结果误写到预定正式目录，**未启动正式 C**；立刻将全部误写产物完整移动保留在 `exp/timbre_blend_stage2_embedding_smoke_backup_20260929/`，原硬链接树也移动到 `.tmp/stage2_c_virtual_train_smoke_backup_20260929/`。修正为底层 dict 写法后第二次烟测在 `.tmp/stage2_c_smoke_exp/` 通过，正式 expdir 确认仍不存在；公开七项严格相等与全模型冻结校验 PASS。备份不删除，等待主人验收。C 正式运行结果在结束后补充。
- 主人明确许可新增相似度评测依赖，但**不得污染训练 `.venv`**。训练 Torch 安装前后分别确认 `2.11.0+cu128`。先前使用系统 Python 3.14 创建的评测环境完整保留于 `~/work/eval-venv-system314-backup-20260929/`；按主人最新要求，现评测环境 `~/work/eval-venv` 用经 PyPI wheel SHA-256 校验的 **uv 0.12.20**（wheel hash `d0837cb0ec80f830198e4a0fe86385e2a8ba8846f066acd243a1c15f4d52ca74`）管理 **CPython 3.12.14**，隔离安装 `speechbrain==1.1.1`、`torch==2.11.0+cpu`、`torchaudio==2.11.0+cpu`、`huggingface_hub==0.36.0`；CPU `torch.cuda.is_available()` 为 False，原训练 Torch 保持不变。
- 公开相似度模型：`speechbrain/spkrec-ecapa-voxceleb`，官方 `huggingface.co` revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`，HF 模型卡与 SpeechBrain PyPI 元数据标注 **Apache-2.0**。仅下载公开文件、请求不带 HF token；原始 model 文件 SHA-256：`hyperparams.yaml` = `6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41`，`embedding_model.ckpt` = `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`，`mean_var_norm_emb.ckpt` = `cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33`，`classifier.ckpt` = `fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535`，`label_encoder.txt` = `e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9`。三个权重文件通过 `torch.load(weights_only=True)` 安全加载审查；本地评测强制 `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` 与 `HF_HUB_OFFLINE=1`。审核过官方 YAML 中的类均为已安装 SpeechBrain 内置组件；本地副本只把 pretrained_path 改为已核验的绝对本机目录。使用已装的 soundfile 读取 WAV、torchaudio functional 重采样至 16k，不增加 TorchCodec；首条公开歌手切片在 CPU 离线提取 192 维 embedding PASS。**该语音模型在歌唱域只是 proxy，不以论文阈值代替唱歌域校准或主人盲听**。

### 2026-09-29 12:08–关卡 3 | C 最终结果、B@1k 公平复核与盲听交付
- C 起点 step0 checkpoint 写出约 11:38，step3000 约 12:08（含每 200 步全量双验证，约 30 分钟），`C_STOP=MAX_STEPS best_step=3000 best_loss=1.02055079`、`C_TRAIN_EXIT=0`。虚拟整歌验证在 200→3000 全部 15 次持续下降，**未达到连续两次 >2% 上升**；按主人指示先在 3000 停下，不自动续训。保留 `model_0.pt` 和每 200 步 checkpoint，共 16 个，最佳 `exp/timbre_blend_stage2_embedding/model_3000.pt`：MD5 `ad187f802e17d4bd04d424ce0174d97e`、SHA-256 `7917dfc56894b8b68fd00d7f70261da7a9ea0fab3c78bff1183aaff9a208a617`。完整 15 点三组验证曲线同步到 `reports/timbre_blend_stage2_c_validation_history.jsonl`；原训练运行 log/模型保留于 `exp/timbre_blend_stage2_embedding/`。
- C@3000 虚拟固定 34 条训练切片 ddsp **0.8636362132**、整歌验证 34 条 ddsp **1.0205507927**，验证–训练差 **0.1569145795**、比值 **1.1817x**；模型选点只依虚拟验证 ddsp，虚拟 mel 指标虽记录但**不用于结论**。B@1k 用同一批 34 条训练切片和相同验证算法/RNG 独占 GPU 重新计算：训练 ddsp **0.2891307894**、整歌验证 **0.8600097211**（与原 B 曲线逐位相同），差 **0.5708789317**、比值 **2.9745x**。两个 34 条训练样本集合逐项一致。C 差距变小，但训练/验证绝对损失都比 B 高（C 虚拟 val 高 **0.16054**）；**不能把小差距等同于更优音色**。
- 冻结复核：C `model_0.pt`→`model_3000.pt` 全 state_dict 仅 `ddsp_model.unit2ctrl.spk_embed.weight` 一张表发生变化，该表只有最后一行 spk13 变化，旧 12 行逐元素完全相等；起点和所有 15 次验证的旧行 SHA-256 均为 `dc1a76612a3531bf18fea47e61b6af3383a101cf1f6b0ce43ef2ba2539244bb1`。公开 558 切片的 ddsp/reflow/四项 mel 等七指标在 C 的**全部 15 次**验证逐项严格等于修正后的新零步公开基线（公开 ddsp **0.2503305322**）。B@1k 公开 ddsp **0.2570199005**，比零步 **+2.6722%**；其余公开指标和 C/基线详见 `reports/timbre_blend_stage2_c_gate3.json`。公开曲线只测已见歌曲上的遗忘，不能代表未见歌曲泛化。
- 盲听两组新推理均使用相同的 OpenSinger 官方未训练歌手 **10_侧脸_26.wav**、**47_秋酿_16.wav**；两组同输入各固定 seed=20260929+歌手 ID、spk13、`-ts 0.0 -step 50 -method euler -k 0 -f 0 -pe rmvpe -th -60 -fmin 50 -fmax 1100 -d cuda`，`PYTHONFAULTHANDLER=1`。随机十六进制文件名四条输出与两输入位于 `samples/stage2/blind_c_vs_b1k/`；`pairing.json` 仅按输入给出两个随机文件名、不揭露模型。解盲对照表**单独**保存在未跟踪 `.tmp/stage2_c_vs_b1k_blind_key.json`。Mac 已复制至 `~/Downloads/timbre_blend_stage2_C_vs_B_blind/` 与独立的 `~/Downloads/timbre_blend_stage2_C_vs_B_blind_key.json`，4 个 WAV 的 SHA-256 全部核对；checkpoint 与曲线另存 `~/Downloads/timbre_blend_stage2_C_model_3000.pt` 和 `~/Downloads/timbre_blend_stage2_C_validation_history.jsonl`，Mac 模型 MD5 一致。**试听前勿打开对照表。** 原 `samples/stage2/probe_1k/` 及其 Mac 包完全保留。
- 相似度评测用固定 HF 官方 revision 的 SpeechBrain ECAPA，私有参考仅从本地虚拟验证 **2 首 × 2 段**提 embedding、只输出汇总分数，`HF_HUB_OFFLINE=1`，未上传任何音频。OpenSinger 公共歌唱参照：12 位歌手、每位 2 首、每歌 2 片；同歌手**不同歌曲**余弦 48 对 mean/median **0.5575/0.5881**、异歌手 1056 对 mean/median **0.2642/0.2590**；p05/p25/p75/p95、std 均在 `reports/timbre_blend_stage2_similarity.json`。用**同一份固定种子盲听包的实际四条输出**做对比：B@1k 对目标原声 mean **0.65830**（10号 0.58692，47号 0.72968），C@3k mean **0.67860**（10号 0.66217，47号 0.69502）。C 总体仅 **+0.02030**，两个输入方向相反，且 8 对输出–原声比较不是 8 个独立歌手；语音模型跨到歌唱的数值仅为 proxy，不能声称机械感改善。首次相似度脚本曾把旧**未固定种子** B 探针与新固定种子 C 混比，旧报告已完整移至 `.tmp/stage2_similarity_mismatched_b_preseed_20260929.json` 作废保留；现最终报告只评分盲听包同批固定种子四条输出。
- 对比表见 `reports/timbre_blend_stage2_c_gate3.json`；B@1k 与 C@3k **不同训练步数/新行初始化/训练数据和冻结策略**，只能描述整体方案表现，不能单因素归因。A/B 训练仍暂停，B@1k/@2k 保留、不推送；备份待主人验收后再议。**【人工关卡 3】：交付试听，停下等待主人盲听选择，不自动续训 C。**

### 2026-09-29 | 关卡 3 通过后的 C 续训前检查（进行中）
- 主人盲听 10、47 两位歌手输入均选择 C@3000；确认 C 作为主线，B 不再训练。B@1k / B@2k checkpoint MD5 `18270a6fa8dbe26d37a5ba3b9d164662` / `4031af7eef33f83e8714f0e7f917786a`，均保留不删。重新核验 Stage 1 起点 MD5 `e026eb6e60b7e2e8ba4c579ad8a0ceff`、C@3000 MD5 `ad187f802e17d4bd04d424ce0174d97e`。
- **续训冻结先决条件已通过**：C `model_0.pt` 与 `model_3000.pt` 的全 state_dict 只有 embedding 表一张张量不同，表的旧 12 行逐元素 exact equality；两者的旧行 SHA-256 与此前 15 次正式验证完全一致：`dc1a76612a3531bf18fea47e61b6af3383a101cf1f6b0ce43ef2ba2539244bb1`。原 ckpt 无 optimizer 字段，因此后续会从相同权重/步数重建 AdamW 动量；数据打乱 RNG 也重新按同一 seed 起步，不能冒称不中断连续训练。
- 新入口仍是 `scripts/train_stage2_embedding.py`，仅加从已校验 C@3000 checkpoint 恢复的路径；`configs/timbre_blend_stage2_embedding_resume.yaml` 仅将 `max_steps` 设为 10000、使用独立 expdir 和恢复路径/MD5，仍保持 lr=1e-3、仅目标行、无回放、weight_decay=0、Reflow spk13 排除、每 200 步留 ckpt/三组全量验证和 >2% 两次上升停训。完整拷贝原 15 点曲线作新目录历史，从 3200 接续；每次仍逐项核对模型其余所有 state/旧行 SHA、公开验证七项 exact equality。隔离烟测在新 `.tmp/stage2_c_resume_smoke_exp/` 全部通过：未更新时重新计算 C@3000 的三组指标和历史末点逐项严格相等、公开零步不变，step3001 再验证其余 12 行 SHA/公开七项全相等。正式训练尚未启动时，旧 C expdir 只读、旧 checkpoint 未覆盖。
- 盲听原片限制经主人决策：不拼接；三位 Stage 1 预留未训练歌手 [10,29,47] 各取 3–4 条官方 OpenSinger 连续原片，每条 ≥5s、总时长 ≥60s。当前筛选 **11 段/113.593s**，只复制原片而不编辑。候选明细 `.tmp/stage2_holdout_selection.json`（含逐片音高/HNR/SNR）；先前 Praat 的一些高音是高八度误判，以已有 RMVPE 原片 F0 直方图交叉核对，实际 B4 以上候选为 29 号《月光》#5/#13/#15（有声帧比例约 5.2–5.9%），10/47 暂无足够证据宣称高音。长音仅是稳定音高候选；47 号 #15/#25 低谐噪比为**气声候选、未人工听辨确认**。若不能确认气声，最终报告应列缺口，绝不按 HNR 冒充听感事实。
- 当前访问为 `.244:2222`；日志历史中的 `.150` 保留原日期证据、页首注明旧址。按主人要求 `.tmp/run_tb.py` 已由对外 `0.0.0.0` 改为仅 `127.0.0.1`，原脚本完整备份在 `.tmp/run_tb_before_loopback_20260929.py`；当前 `ss -ltn` 显示 6006 未启动，不声称已监听，也不在续训期间启动 TensorBoard。

### 2026-09-29 12:40–12:42 | C 续训在公开验证期段错误，暂停待主人决策
- 进程 `C_RESUME_TRAIN_EXIT=139`，`PYTHONFAULTHANDLER=1` 的堆栈显示主线程在 `reflow.solver._test_impl()` 的 `vocoder.infer(mel, f0)`，进入 NSF-HiFiGAN `models.py` 的 PyTorch `Conv1d` 原生调用时 segfault；发生在 step3400 验证公开第 49/558 条附近。前一次 step3200 的三组验证全部写完，虚拟 ddsp **1.019117048558067**（比 C@3000 的 1.0205507927081163 小约 0.001434），公开七项仍严格等于起点，其他嵌入行 SHA-256 仍是 `dc1a76612a3531bf18fea47e61b6af3383a101cf1f6b0ce43ef2ba2539244bb1`，>2% 上升票 0。
- 保存发生在验证之前：`model_3400.pt` 文件完整存在，**但 3400 验证未完成**，不得用其选最优。CPU 安全加载 `model_3200.pt` 与 `model_3400.pt` 成功；两者对 C@3000 只改变 spk13 embedding 一行，其余 state_dict 逐张量 exact equality、原 12 行 SHA 均与起点一致；MD5 依次 `bd0d7cf040c90ca270f496ffec0c69ef`、`e9c92ca2c8393069c4249b60d678b3f4`。完整曲线停在 16 条（原 15 条加 3200）；在这一时点最佳**已验证**点为 3200。无训练/推理 GPU 进程、显卡空闲。仅由 Python 原生堆栈不能断言是模型算法还是 WSL2/CUDA 驱动；不可将退出 139 当成停训规则。
- 严守先前约定：**不擅自重启 WSL、不自动续训或推理、不推送部分结果**。保留完整原始 stderr、3200/3400 checkpoint 与曲线，在主人决定恢复方案后再继续关卡 3；当前盲听包只有 11 条原声输入，**没有三组输出**，不得宣称新关卡完成。

### 2026-09-29 | Fatal signal 11 只读诊断（主人选择先停并诊断）
- `journalctl -k` 在 **12:40:56** 记录训练 Python PID 19651 `potentially unexpected fatal signal 11`，同时间窗口**未见 Xid/NVRM/GPU OOM/dxg 新错误**；不能从堆栈断定模型算法、cuDNN、Windows 驱动三者谁是根因。WSL2 内核 `6.18.33.2-microsoft-standard-WSL2`、Windows NVIDIA 驱动由 WSL nvidia-smi 显示 `617.14`、训练 Torch `2.11.0+cu128`。内核启动 **11:12:11** 曾有 `dxgvmb_send_wait_sync_object_gpu` 的 `memcpy: detected field-spanning write` WARNING 与 ioctl `-22`，内核现标为 `Tainted: G W`；这一较早告警提示 WSL GPU 桥接值得调查，但**与 12:40 段错误间无已证实因果链**。NVIDIA WSL 官方文档提醒 CUDA 依赖 Windows 驱动映射，**不可在 WSL 内安装 Linux NVIDIA 驱动**：https://docs.nvidia.com/cuda/wsl-user-guide/ 。无重启/驱动/包升级操作。
- 静态代码核查：`reflow.solver._test_impl` 每条验证样本都在 DDSP/reflow/mel loss **之前**执行 `vocoder.infer(mel,f0)`，本次 segfault 的 `Conv1d` 来自 NSF-HiFiGAN 生成**波形**；当前 `train.val_log_samples=0`、TensorBoard 关闭，波形只用于 RTF 时间统计而**不用于 ddsp/reflow/mel 损失或公开冻结判断**。不过 NSF source 代码里有 `torch.rand` 和 `torch.randn_like`，直接跳过该前向会改变后续损失计算的 RNG 流；先前 A/B 旧验证比较正因随机流差异作废过。任何 metrics-only 新路径须另建可比基线、复算原 checkpoint，不可把新旧数字拼接或悄然跳过 RNG 比较。并且独立盲听推理仍必须生成波形，单纯改变验证不保证解决环境风险。
- 官方 PyTorch 2.11 文档说明 `CUDA_LAUNCH_BLOCKING=1` 只用于定位异步 CUDA 调用，**不是修复**：https://docs.pytorch.org/docs/2.11/cuda_environment_variables.html 。没有在诊断阶段运行 GPU 重现、切换 cuDNN、改训练/验证代码、继续到 10000、制作 33 条推理或推送分支。建议先请主人选择：① 单独批准采用 metrics-only 新验证协议、重新校准/复算全部相关基线与候选，随后隔离 GPU 探针再决定是否续训；或 ② 停训于已完整验证的 **3200**、单独批准用现有波形推理制作三组盲听（仍存在 GPU 原生崩溃风险）。两条都**不是**自动恢复授权。原模型、完整 stdout/stderr、16 点曲线与 3400 未验证 ckpt 继续保留。

### 2026-09-29 | C@3400 原生崩溃诊断性续跑（主人授权）
- 保留原 `exp/timbre_blend_stage2_embedding_resume/`（含 C@3200、未完成验证的 C@3400、16 点历史）和原始 stdout 不覆盖；新配置 `configs/timbre_blend_stage2_embedding_resume_3400.yaml` 使用独立 `exp/timbre_blend_stage2_embedding_resume_3400/`，从 MD5 `e9c92ca2c8393069c4249b60d678b3f4` 的 3400 权重起步。只复制已完整验证的 16 点历史，**3400 补验成功才入历史**；若崩溃则旧 3400 仍未验证，最佳保持 3200。原检查点无 AdamW 状态，3400 后再次重建动量，不能与不中断训练等同；其余 lr、冻结行、验证周期、停训阈值及上限 10000 不变。
- 新运行开启 `CUDA_LAUNCH_BLOCKING=1` 使 CUDA 调用同步定位（降低速度，不保证修复）、`PYTHONFAULTHANDLER=1`，独立 `faulthandler.log`，完整 unbuffered stdout/stderr；按样本在 `model_infer_start`、`vocoder_start`、`vocoder_done`、`metrics_done` 每阶段将步数/分组/样本及时间写入独立 JSONL 并 fsync。诊断仅在设置 `STAGE2_VALIDATION_TRACE` 时启用；旧验证数值计算不改。脚本在初始补验前再次核对历史公开指标、旧行 SHA 与模型权重，若不通过即停止。未自动选择未经验证检查点；此前原始日志、脚本和文档已备份在 `.tmp/stage2_c_backups/`。GPU 运行结果待补。

### 2026-09-29 | WSL 意外退出后的 4200 历史恢复与高速续训（主人授权）
- 前次 WSL boot 于 13:21:31 后停止，13:38:43 重启。无 C_STOP、无进程退出码、无新 Python faulthandler 堆栈；末次 stdout 训练记录 step4351，实际终止原因未确定，不把它视为正式停训。`exp/timbre_blend_stage2_embedding_resume_3400/validation_history.jsonl` 末尾有 789 个 NUL 字节，完整 JSON 到 4000 为止；另有 stdout 和 `log_info.txt` 末尾 NUL。**所有原始文件原样保留**，原曲线与 stdout 的逐字节备份在 `.tmp/stage2_c_backups/`。原日志中含 3400、3600、3800、4000、4200 五条完整 `C_VALIDATION` 行，前四条与原曲线逐条严格相等；由日志第 4200 条补全的新 21 点历史为 `reports/timbre_blend_stage2_c_history_recovered_4200.jsonl`（SHA-256 `6a469f83df9bbd5e0ea000c6be4acd99270560a6c085ef769d5c2d7a1c8b8f6a`）。4200 虚拟 ddsp **1.0077299826285417**、>2% 连涨票 0；公开七项等于原基线。
- `model_4200.pt` 219764276 字节、MD5 `b3d4842f8c538f69e72dd80538130c5f`。CPU 权重加载成功，相对 C@3000 除 spk13 embedding 行之外均逐张量相等；其余 12 行 SHA-256 `dc1a76612a3531bf18fea47e61b6af3383a101cf1f6b0ce43ef2ba2539244bb1`。原 3400、3600、3800、4000、4200 checkpoint 不覆盖，B 停训保留。
- 新 `configs/timbre_blend_stage2_embedding_resume_4200.yaml` 和独立 `exp/timbre_blend_stage2_embedding_resume_4200/` 从 4200 权重和恢复的历史续训，最多 10000 步，每 200 步仍对训练抽样、558 条公开及 34 条虚拟片段做同协议全量验证、核查其余权重及旧行 SHA；规则仍是 >2% 连续两次上升即停止。启动时先从 4200 权重重新跑三组验证并与恢复的 4200 指标逐项 strict equality；不一致即拒绝续训。检查点从来不保存 AdamW 状态，因此 4200 后再次重置优化器动量，**不等于不中断续训**。无 TensorBoard，也不并发 GPU 推理。
- 性能设置：不再设置 `STAGE2_VALIDATION_TRACE` 和 `CUDA_LAUNCH_BLOCKING=1`；新配置只降低训练 stdout 频率为每 50 步一次，并关闭单样本 stdout/进度条；保留全部验证数值和 checkpoint 频率。为减少 WSL 再次异常退出的影响，**仅每 200 步对已保存 checkpoint 和追加的完整验证 JSONL 做 fsync**，不对每条样本做 fsync。原始诊断程序与日志未删。运行结果待补。

### 2026-09-29 13:51 | 高速续训于 C@4600 公开验证再次段错误，停止自动重试
- 新独立运行 `C_RESUME_4200_EXIT=139`；启动阶段 C@4200 的三组指标逐项严格复算一致，`C_RESUME_VERIFY=PASS`。C@4400 完整验证记录已写入 JSONL（22 点均可解析，公开七项/非目标行 SHA 全程一致），虚拟 ddsp **1.0084749635528116**，>2% 连涨票 0；最佳已验证点仍为 **4200**（1.0077299826285417）。C@4600 checkpoint 于公开验证前落盘，CPU 安全加载并确认除 spk13 行外所有权重逐张量未变，MD5 `f3bbb354c3e049c4aea5e73be7e03ec7`；但 4600 的公开验证未完成，不能选优。C@4400 MD5 `4e8f39984fa2c334b1a8ddeef573cf84`。不运行推理、不推送。
- 独立运行禁用了每样本 trace/fsync 与 `CUDA_LAUNCH_BLOCKING=1`、关闭验证 stdout/进度条、训练日志每 50 步一次。尽管如此，13:51:13 再现 **Python signal 11**：主线程在 `reflow.solver._test_impl -> vocoder.infer -> nsf_hifigan/models.py -> torch.nn.Conv1d`，与首次 3400 公开验证段错误同一调用路径；内核同时记录 `python3.11 ... fatal signal 11`，未见同时间窗口 Xid/OOM 证据。两次不同检查点公开验证都在 GPU vocoder 原生卷积中退出，不能继续视为已证实的单次偶发事件；也不能仅凭堆栈判定代码、cuDNN、驱动何者根因。**在明确验证/波形合成路径的隔离方案并获主人确认前不再重启训练**。保留全部 ckpt、stdout/stderr、原始及恢复的曲线和已存在的备份，不覆盖 B。

### 2026-09-29 | CUDA Conv1d 崩溃的版本与社区调查（只读）
- 实机：Windows `10.0.26200.9550`、WSL `2.7.14.0`、内核 `6.18.33.2-2`、WSLg `1.0.73.2`、NVIDIA Windows KMD `617.14`（WSL nvidia-smi 工具显示 `615.78.02`、CUDA UMD `13.4`）；训练 `.venv` CPython `3.11.16`、torch/torchaudio `2.11.0+cu128`、CUDA `12.8`、cuDNN `9.19.0.56`、triton `3.6.0`。`LD_LIBRARY_PATH` / `LD_PRELOAD` 为空，未发现系统 cuDNN 包或 linker 候选；只读 `/proc/self/maps` 显示当前 torch 导入的 cuDNN/cuBLAS 来自同一个 `.venv`，CUDA 驱动 stub 来自 `/usr/lib/wsl/`；不能据此排除实际卷积时动态加载问题。NVIDIA 文档确认新驱动运行旧 CUDA runtime 属于向后兼容，13.4/12.8 的版本数字差异本身并非错误：https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html 。
- 官方 Microsoft `2.7.14` 是目前最新稳定 WSL，发行说明未列此 dxgkrnl 问题的修复；`2.9.13` 为预发布版本，也未查到针对该告警的明确修复说明：https://github.com/microsoft/WSL/releases/tag/2.7.14 、https://github.com/microsoft/WSL/releases/tag/2.9.13 。上游独立 WSL kernel `6.18.40.1` 有 dxgkrnl 内存泄漏修复，但不是已证明针对 `dxgvmbus.c:3095` 的修复，且非本机稳定版默认内核；不安装自编译/未审计补丁：https://github.com/microsoft/WSL2-Linux-Kernel/releases/tag/linux-msft-wsl-6.18.40.1 。
- 与本机字符串完全相同的 `current_pos` field-spanning warning 有两种社区现象：#40580 指出 **Xwayland 启动时**警告但 GPU 仍可运行；#41017 报持续 GPU 负载下挂起/设备错误/其他人的 WSL 退出，该 issue 被策略机器人关闭，**并非官方确认修复**。本机 13:38:47 的 warning 线程 `Comm: Xwayland`，13:51:13 Python 原生 signal 11；没有同时 Xid/OOM，不能仅凭启动告警推导出后来的 PyTorch 崩溃因果链：https://github.com/microsoft/WSL/issues/40580 、https://github.com/microsoft/WSL/issues/41017 。社区 #41093 是未合并的提案，不采用：https://github.com/microsoft/WSL/issues/41093 。
- PyTorch 社区曾报告 GPU `Conv1d` 进程 SIGSEGV 且调用栈相似，但旧案例（torch2.2/CUDA12.1）还伴随本地 CUDA 库覆盖，不能视为本环境同一根因：https://github.com/pytorch/pytorch/issues/121222 。官方 PyTorch `2.14.0` 于 2026-09-02 发布（本机为 2.11），升级包含 CUDA/cuDNN 依赖更换、cuDNN 9.24 和其他卷积修复，**未见本次 Ada/WSL NSF-HiFiGAN Conv1d 崩溃的对症修复承诺**；升级需主人明确许可、另建隔离环境，不直接修改训练 `.venv` 或拼接新旧验证曲线：https://github.com/pytorch/pytorch/releases/tag/v2.14.0 。Windows 驱动 `617.14` 是 NVIDIA 官方 2026-09-22 版本，调查时未确认有更高版本或此故障的对症补丁：https://www.nvidia.com/download/driverResults.aspx/279803/en-us/ 。NVIDIA 官方明确 WSL 仅安装 Windows 驱动，**不能在 WSL 中安装 Linux NVIDIA 显示驱动**：https://docs.nvidia.com/cuda/wsl-user-guide/index.html 。
- 建议**先隔离验证，不盲升或续训**：在现有环境使用公开片段/模型做小范围 vocoder 卷积复现，分别试默认 cuDNN 与临时禁用 cuDNN（PyTorch 2.11 提供 `torch.backends.cudnn.flags(enabled=False)`），并核对实际动态库、Windows 主机事件日志；任何新数值协议先重算可比基线。上述 GPU 测试/安装/重启均未执行，等待主人授权。

### 2026-09-29 | 仅验证波形合成禁用 cuDNN 的隔离测试（无训练）
- 以 C@4200（MD5 `b3d4842f8c538f69e72dd80538130c5f`）为只读输入，在 `.tmp/stage2_c_cudnn_vocoder_probe.py` 独立进程中仅以 `torch.backends.cudnn.flags(enabled=False)` 包裹验证 `vocoder.infer(mel, f0)`；原验证的模型推理、损失、固定随机种子、逐样本顺序均不动。没有修改正式训练源码或 checkpoint；探针结果保存在 `.tmp/stage2_c_cudnn_vocoder_probe/`。
- 小样本 A/B：公开 3 条、虚拟 3 条；默认 cuDNN 与局部禁用两次运行均正常结束，七项指标逐项完全相同。C@4200 全量局部禁用：公开 558 条、虚拟 34 条，均正常结束；七项指标与 `reports/timbre_blend_stage2_c_history_recovered_4200.jsonl` 的对应 4200 记录严格相等（公开 ddsp `0.25033053218997936`、虚拟 ddsp `1.0077299826285417`）。未复算训练探针 34 条，因此不能说“三组指标”均已复算。该结果证明**这一次**的替代卷积路径没有改变所比较的指标，不证明能避免偶发段错误。
- 初版临时探针对嵌套 `DotDict` 属性赋值没有生效，导致 `Saver` 误写正式 C@4200 续训实验的 `config.yaml`。已核对内容逐字等于原配置经 `yaml.dump` 的确定性 Saver 输出；仅该文件修改时间改变，正式曲线/权重未改。修复为字典索引赋值后临时输出目录与配置校验通过；脚本修复前版本已备份至 `.tmp/stage2_c_backups/`。
- 对此前失败的 C@4600（MD5 `f3bbb354c3e049c4aea5e73be7e03ec7`）启动同类全量隔离探针后，远端命令通道中断；探针仍存活约 15 分钟、单核 CPU 100%、输出文件为空且 GPU 无活跃计算，未走到验证开始标记。已仅对该探针发送 SIGTERM，未自动重试；其现象不能判断 cuDNN 开关能否规避 4600 的原生崩溃。4600 依旧未完整验证，不选优、不续训。

### 2026-09-29 | C@4600 禁用 cuDNN 的独立完整复核（不续训）
- 加启动阶段标记与硬超时后，C@4600（MD5 `f3bbb354c3e049c4aea5e73be7e03ec7`）局部禁用 cuDNN 的公开/虚拟各 3 条探针成功；随后同协议完成公开 **558**、虚拟 **34**、训练探针 **34** 条，没有段错误。完整探针目录 `.tmp/stage2_c_cudnn_vocoder_probe/4600_no-cudnn_558/` 及 `4600_no-cudnn_34_train_probe/`；不包含训练或优化器修改。
- 4600 公开七项指标与既有公开基线逐项严格相等；训练探针七项指标与原 4600 崩溃前已记录的训练探针一致（ddsp `0.8565065781859791`）。虚拟 ddsp **`1.0032078080317552`**（4200 为 `1.0077299826285417`，4400 为 `1.0084749635528116`），是截至目前本协议已完整评估 checkpoint 中最低的一点。CPU 校验 4600 vs 4200：除 spk13 行以外所有模型参数/状态与前 12 位嵌入行相等，仅 spk13 行发生变化。
- “完整评估”是**独立临时探针**产出的三组指标，还未补入正式续训历史；先前会话的 4600 验证仍以段错误结束，不能隐去事故或宣称原因已经确定。局部禁用 cuDNN 路径在 4200 对旧曲线逐项严格复算通过、4600 一次全量通过，说明具备有条件继续训练的依据，**不保证此后长时间运行不再因 WSL/GPU 层崩溃**。前次 CPU 空转没有重现，也无足够证据归因；正式续训须单独获准并保留备份/冻结核对、每 200 步验证与虚拟损失停训规则。

### 2026-09-29 | C@4600 正式续训配置及一步 smoke
- 新增 `configs/timbre_blend_stage2_embedding_resume_4600.yaml`，新实验目录 `exp/timbre_blend_stage2_embedding_resume_4600`；从旧实验 C@4600（MD5 `f3bbb354c3e049c4aea5e73be7e03ec7`）加载，沿用旧实验 22 个完整历史点（末点 4400），以独立三组指标 `reports/timbre_blend_stage2_c4600_isolated_metrics.json` 严格校验重算 4600，才写入新实验历史并推进训练。旧日志/checkpoint 不改；新配置 `val_disable_cudnn_vocoder: true` 只作用于验证 `vocoder.infer` 调用，所有其他验证路径照旧；每 200 步验证与两次 >2% 上升停训、10000 步上限不变。优化器状态不存在，恢复时 AdamW 动量重置，不能称为未中断训练。
- `tests/test_stage2_validation.py` 与 `tests/test_reflow_masking.py`：25 passed。独立 smoke 从 4600 重算三组 `ISOLATED_4600_METRICS` 严格匹配，`C_RESUME_VERIFY=PASS`；更新一步后 C@4601 完整验证、冻结/公开基线检查通过，`C_STOP=SMOKE`，虚拟 ddsp `1.00175196`（临时 smoke 点不纳入正式选优）。第一次调用漏传仓库导入路径导致 `ModuleNotFoundError: logger`，无模型/实验输出；随后使用 `PYTHONPATH=.` 正常完成。smoke 产物只保留在 `.tmp/`，正式实验仍须另行启动。

### 2026-09-29 | Stage 2 C 完成 10000 步；扩大盲听 32/33 暂停
- 正式 `resume_4600` 进程正常退出 `0`，`C_STOP=MAX_STEPS`；`validation_history.jsonl` 含 200..10000 每 200 步连续 **50** 个完整验证点，公开三组指标记录中公开组严格保持原基线，所有点的旧 12 行 SHA-256 与先前一致，未触发连续两次 >2% 上升规则。虚拟 ddsp 最低在 **C@7800 = 0.9929537212147432**（MD5 `733d9c6d44071d91a1c0b0f190db44db`）；C@10000 为 `0.9967395785976859`（MD5 `f2b62856e11612406be4ddfb9bf93fbc`），因此选优不能用末点。CPU 加载 7800/10000 与旧 4600 比对：所有非目标状态及前 12 位嵌入行逐元素相等，仅 spk13 行变化。全部 checkpoint 保留，无训练进程。
- `.tmp/make_stage2_expanded_blind.py` 固定三组 B@1k、C@3000、C@7800 与相同 11 段原片（113.593 秒），随机输出名和单独 `.tmp/stage2_expanded_blind_key.json` 已建立。首次成功 22/33；作业 23 在加载 `transformers → sklearn → scipy` 时抛出 `UnboundLocalError: start`，未进入模型推理；SciPy 源文件有先定义 `start`，新进程 `import scipy.stats` 成功。保留错误原件 `.tmp/stage2_c_backups/stage2_expanded_infer_23_failed_import_20260929.log`，核对 22 个音频格式、非静音、有限值和长度后只补未完成输出。补跑成功至 32/33；作业 33（47 号 `47_秋酿_25.wav` 的 C@7800 输出）在 Python 标准库 `importlib.metadata`/`email.feedparser` 加载阶段发生原生 signal 11，`journalctl -k` 17:36:52 也记录 fatal signal 11；仍未进入 GPU 模型推理。原 `.tmp/stage2_expanded_infer_33.log` 和两次汇总日志保留；音频目录仅 32 个随机命名文件、缺失的随机输出名没有文件、`pairs.json` 未创建。未自动进一步重试、未发布不完整包。根因未定，不能把导入阶段故障直接归因于 C@7800、cuDNN 或 WSLg；CUDA 训练正常结束也不能证明 Python/WSL 运行环境完全稳定。
- 独立相似度全量报告、随机盲听包交付、关卡 3、分支推送均仍待完成。未读取私有凭据，未把私有音频上传第三方或改动原有三个未跟踪文件。

### 2026-09-29 | 扩大盲听导入阶段崩溃的只读排查（未补跑音频）
- 重启/会话恢复后复核：C@10000 训练退出码 0、最佳 7800；三组盲听 32/33、`pairs.json` 尚不存在、未见训练/推理进程。失败第 23 个原日志已备份；第 33 个原生 SIGSEGV 栈在 `importlib.metadata -> email.feedparser`、内核同时间记录 Python signal 11，均未进入模型推理。
- 当前 venv CPython 3.11.16、NumPy 1.26.4、PyTorch 2.11.0+cu128、SciPy 1.17.1、scikit-learn 1.9.1、Transformers 5.17.0。SciPy `_docscrape.py`、sklearn `_array_api.py`、Transformers `import_utils.py`、NumPy `__init__.py` 均与各自安装 RECORD 哈希匹配。SciPy `_docscrape.pyc` 与源码重新编译的 `Reader.read_to_condition` 字节码/常量相符，且 `start` 在源码中无条件赋值，无法以静态源码错误解释首次 `UnboundLocalError`。当前资源空闲、无 OOM/MCE/Xid 内核记录；Windows System 日志最近 12 小时 WHEA-Logger 查询返回“未找到匹配事件”，不能由此排除主机硬件/虚拟化问题。
- 限时 CPU-only、`CUDA_VISIBLE_DEVICES=-1 PYTHONMALLOC=debug python -X dev -X faulthandler`、各次独立进程导入：标准库/元数据 **30/30** 成功；NumPy＋torch＋元数据 **25/25** 成功；`numpy, torch; from ddsp.vocoder ...; import scipy.stats` 第 **20/20** 次失败，在 Transformers 的 `isinstance(v, frozenset)` 报“arg 2 must be a type”；直接 `numpy, torch; from transformers import HubertModel...; import scipy.stats` 第 **3/25** 次失败，在标准库 `inspect.cleandoc` 的普通 `for i in range(...)` 行报 `range_iterator does not support the context manager protocol`。这表明不经音频/GPU 合成也可触发不相称的运行时异常，关联重导入链，但**尚不能判明**是 CPython 构建、某原生扩展内存破坏、WSL/主机内存还是其他原因；控制组通过不等于证明无问题。
- 原始错误和三组诊断日志位于 `.tmp/stage2_*import*diag_20260929.log`；未升级包、新建环境、重启 WSL、运行 GPU、覆盖 32 个输出或接触私有凭据。正式盲听包和评测保持暂停，下一步应由主人决定是否进一步隔离运行环境或批准在严格校验/保留日志条件下只补缺失的 1 条。

### 2026-09-29 | 扩大盲听补齐及关卡 3 比较结果（待人工盲听）
- 主人批准继续后，只对缺失的第 33 条 `47_秋酿_25.wav` / C@7800 按同原片、checkpoint（MD5 `733d9c6d44071d91a1c0b0f190db44db`）、推理参数、seed `1277298111` 限时补跑 **一次**，退出码 0；失败前原始 33 号日志先保存为 `.tmp/stage2_c_backups/stage2_expanded_infer_33_failed_import_20260929.log`。原 32 文件未覆盖。全 33 条及 11 原片校验输入哈希、输出 44.1 kHz、单声道、非静音、有限值、长度相符，同输入的三组随机种子严格相同；盲配对表 `samples/stage2/expanded_c_blind/pairs.json` 完成，映射密钥仍独立保存在 `.tmp/stage2_expanded_blind_key.json`。
- 独立 `/home/hands/work/eval-venv` CPU 环境 `HF_HUB_OFFLINE=1` 评测完整跑通，固定模型 revision/文件哈希，原始汇总 `reports/timbre_blend_stage2_c_resume_similarity.json`。对照报告 `reports/timbre_blend_stage2_c_resume_gate3.md` 给出公开同歌手跨歌/异歌手、B@1k/C@3000/C@7800 各 44 个比较的均值与 p05/p50/p95 和分歌手明细；B@1k 在本组输入的 ECAPA 代理均值高于两组 C，C@7800 相对 C@3000 仅略升，不能以此代替机械感盲听。歌手 29 有 RMVPE B4 以上帧，歌手 47 低 HNR 仅是未经听辨确认的气声候选。Python 导入阶段故障根因依旧未解，单次补跑成功不能视作修复。
- 接下来仅需将不含解盲表的音频包与单独密钥安全交付、核验包内容、按先前授权推送提交，停在【人工关卡 3】等待主人盲听；不可自行宣称关卡通过。

### 2026-09-29 | 扩大盲听包完整性交付及【人工关卡 3】暂停
- `.tmp/timbre_blend_stage2_three_arm_blind.zip` 含 `README.md`、11 条未拼接 `inputs/`、33 条匿名 `audio/` 与不含模型身份的 `pairs.json`，共 46 个 ZIP 成员；`ZipFile.testzip()` 全部通过且未包含 `key` 或 selection 源路径。ZIP SHA-256 `42d20efc4b0c3ea7d3ecb53bc65b67cdbbeec2801937ab3e5c1befcc7cc7e672`。主人本机 `~/Downloads/timbre_blend_stage2_C7800_C3000_B1k_blind.zip` 拉取后哈希相同，`unzip -t` 通过。
- **单独**解盲表 `~/Downloads/timbre_blend_stage2_C7800_C3000_B1k_blind_key.json`（远端 `.tmp/stage2_expanded_blind_key.json`，SHA-256 `398fd6f7031bebe801f18d3e914e3af708f79f53dff8f3bd9dd741ba707ab587`，本机权限 0600），应在盲听决策锁定后才打开；不在 ZIP 内。对照报告 `~/Downloads/timbre_blend_stage2_C7800_C3000_B1k_comparison.md`，SHA-256 `e05f0f273c1464f298a86113a63fb099940f6b7769145bee5cfdec1b125d8064`，原版在 `reports/timbre_blend_stage2_c_resume_gate3.md`。独立评测明细在 `reports/timbre_blend_stage2_c_resume_similarity.json`。
- B@1k、B@2k、C@3000 等旧 checkpoint 和全部 C 新点均保留。Python 导入阶段偶发运行异常仍未定因，补跑成功**不**消除运行环境风险；关卡 3 仅进入“等待主人试听”，不得写成验收通过。

### 2026-09-29 | 【人工关卡 3】主人选定 C@7800
- 主人反馈扩大盲听整体表现不错、各版本相较前轮差异缩小；特别点名 `47_秋酿_19.wav` / C@3000、`29_月光_13.wav` / C@7800、`47_秋酿_15.wav` / C@3000 音色饱满。这三段**来自不同原片**，不能当作同输入 2:1 的版本投票。
- 随后在 `10_侧脸_3.wav` 同原片/seed/推理设置上，只比较 C@3000 与 C@7800 的随机 1/2 盲听：主人表示难以听出区别、频谱几乎一致；该两条 PCM 波形相关系数 `0.996234`，不是逐采样相同。盲听单片段压缩包与不含映射的 1/2 音频已放在主人本机 `~/Downloads/timbre_blend_C3000_vs_C7800_one_clip/`，密钥独立存于远端 `.tmp/timbre_blend_c3000_vs_c7800_one_clip_key.json`（未在对话中揭晓标签映射）。
- 主人明确接受依原定硬指标打平决胜：**选 C@7800 为主 checkpoint**（MD5 `733d9c6d44071d91a1c0b0f190db44db`）。虚拟验证 ddsp C@3000 `1.0205507927081163` → C@7800 `0.9929537212147432`（约降 2.7%），公开指标不变；扩大盲听 ECAPA 均值 `0.612739` → `0.616574`（很小的代理改善）。**不可宣称有显著可听提升**，C@3000 保留作为备选，B/其他 C checkpoint 全部保留，不再续训。
- `reports/timbre_blend_stage2_c_resume_gate3.md` 已标关卡 3 人工选型结论；分支提交仍仅在远端 WSL 本地，GitHub HTTPS `git push` 因无非交互认证失败，未读取/更改凭据，待主人配置认证或指定授权推送方式。Python 导入阶段偶发异常根因未定，属于独立环境风险，不能将单次成功补跑当作已修复。

### 2026-09-29 | 阶段 3 第一步：12 位公开歌手试听包，等待主人选 3～5 位
- 【人工关卡 3】通过后以 C@7800（MD5 `733d9c6d44071d91a1c0b0f190db44db`）为唯一基准。本步**未进行任何音色融合、配方网格、ECAPA 硬筛选或模型导出**。CPU 加载 Stage 1 MD5 `e026eb6e60b7e2e8ba4c579ad8a0ceff` 与 C@7800：前 12 位公开歌手嵌入行以及所有非嵌入权重逐元素相同。
- 从 Stage 1 预留歌手原片选 3 段、各自不裁剪/拼接：低 `10_侧脸_0.wav` 6.85s（RMVPE 插值后 F0 中位 `219.6 Hz`），中 `47_秋酿_19.wav` 11.05s（`298.3 Hz`），高 `29_月光_13.wav` 18.76s（`361.0 Hz`，独立已确认的 RMVPE 发声帧约 5.88% 超过 B4）。三个中位数来自插值曲线，不能误写为逐帧真实发声或绝对音域分类；选择按相对音域覆盖，规避歌手 47 号的 Praat 高频与 RMVPE 不一致。原片 SHA-256 与 Stage 2 记录一致。
- `main_reflow.py` 仅新增可选 `--disable-vocoder-cudnn`，其作用域仅包裹 `vocoder.infer` 波形合成，与 Stage 2 验证设置一致；默认旧 CLI 行为不变。`scripts/make_stage3_public_preview.py` 沿用 Stage 2 Euler/50 步/`t_start=0`、RMVPE、f0 50–1100、阈值 −60 dB、无升降调，以及相同原片的 SHA-256 派生 seed；12 位歌手各推理 low/mid/high，共 36 条。首次独立 smoke 在输出前超时且日志为空，保留 `.tmp/stage3_public_spk01_low_smoke.log`，没有被当作成功；随后分阶段 CPU 导入探针通过、带标记的单次 GPU 诊断成功，正式批量首条与其音频逐样本相同（相对 RMS 差 0），后续 36/36 正常完成。此前 Python 导入不稳定的根因仍未确定，不能把单次完成说成已修复。
- 所有输出经 SHA-256、44.1 kHz 单声道、有限值、非静音、与原片时长误差 <0.05s 校验；每段的 12 位输出哈希互异，RMS 最小 0.0188/最大 0.1131，峰值最大 0.6332。显式映射、参数、每个输出哈希见 `reports/timbre_blend_stage3_public_selection.json`；远端 WAV 与原片仅留在 `samples/stage3/public_singer_preview/`（gitignored，不入 Git）。主人本机 `~/Downloads/timbre_blend_stage3_public_singer_preview/` 和同名 `.zip` 含 12 歌手 × 3 条、3 原片、`README.md` 与清单；ZIP 41 个成员，SHA-256 `0061987ec6468bc726985b3291087ba3022a39c77f00f54453723f7633c18e2c`，本机解压及 ZIP CRC 验证通过。私有训练音频与模型权重均未传输。
- **停在阶段 3 第一步**：等待主人从 `spk01`…`spk12` 选 3～5 位喜欢的公开歌手，再讨论/执行第二步。未提前触碰第二～五步。

### 2026-09-29 | 阶段 3 配方网格与硬筛选：全数失败，停止在第四步之前
- 主人选择 `spk01 / spk09 / spk12 / spk08`，固定四位均分虚拟以外权重。`spk13=1.0/0.8/0.6/0.4` 分别对应每位公开歌手 `0/0.05/0.10/0.15`；与 `t_start=0/0.25/0.5` 交叉，共 12 配方。沿用 C@7800、阶段 2 相同原片 seed／推理参数，仅在声码器合成中禁用 cuDNN。低中高三条共 36 条输出，均通过 44.1kHz、非静音、有限值、时长检查；只留在 WSL gitignored `samples/stage3/recipe_screen/`，没有向第三方传送。
- 阶段 2 的 ECAPA 校准脚本有口径错误：`select(..., limit_speakers=12)` 对目录名作字符串排序，12 目录实际包含训练中的虚拟 `13_lingxi`，漏掉公开 `9_singer33`。此前 `0.4559639692` 的不同歌手 p95 **不是** 12 位公开歌手间的 p95。本次用同一离线模型、相同每人跨歌曲参考片，明确取 `spk01`～`spk12`、排除 `spk13`，重新计算 1056 对跨歌手余弦得到 p95 `0.4394697547`。两道门槛的严格者为 `<0.4394697547`；重要的是，12/12 配方对 `spk10` 和 `spk12` 的均值**均高于主人要求的 `<0.456`**，因此即便沿用旧数值也全部失败。每个配方实际检查 12 位公开歌手与四段虚拟验证参考；绝不以代理分数代替听感判断。详见 `reports/timbre_blend_stage3_recipe_screen.md` 和 `.json`。
- **阶段 4 未生成**：0 个配方通过，故无符合硬条件的 11 段盲听包或纯虚拟对照组；**阶段 5 未执行**。停止并等待主人决定是否调整配方探索策略；不自行变更阈值或宣称已有合规配方。阶段 2 原报告保留作为历史记录，口径问题由本条和新报告明确更正，不偷偷覆写历史数字。

### 2026-09-29 | 阶段 3 诊断：原声基线、纯公开歌手模型输出及未批准标准草案
- 按主人指示不改已生效门槛、不重训/重推已有配方、不烘焙。诊断一选私有 `sliced-v2` **未经模型转换的人声原始切片** 4 首歌（阶段 2 训练歌 00/01，验证歌 05/08）× 每首时长最长 3 段，12/12 与阶段 2 拷贝 WAV SHA-256 一致。这里不是本机未留存的整首 `raw/`；同歌不同段不冒充不同歌。使用同一个离线 CPU ECAPA、12 位公开歌手各 4 条跨歌曲参考片，原声对 `spk10` 均值 `0.471038`、`spk12` 均值 `0.468532`，均高于旧 `0.456`，说明其超线至少部分是固有相似／代理特性，不能全怪融合。
- 诊断二**复用**阶段 3 第一步已产的 C@7800 12 公开歌手 × 3 输入＝36 条纯音色 WAV，重新核验输出 SHA-256，不重新推理。目标歌手对本人参考片：144 次余弦均值 `0.528140`、p05 `0.285039`、p50 `0.535272`、p95 `0.725275`；对另外 11 位：1584 次余弦均值 `0.291992`、p05 `0.111001`、p50 `0.286744`、p95 `0.485992`。按每目标／参照歌手 3×4 等权平均后，不同歌手共 132 个均值，其 p95 为 `0.413682`；两种统计单位不同，均列在报告。
- 草案（**尚未获主人确认**）：对所有 12 位 Y 同时要求相似度相对原声 Y 增幅 `≤10%`（政策建议而非统计置信界限），且配方对 Y 均值严格 `<0.485992`（诊断二不同歌手单次余弦 p95；另披露更严格同统计单位 p95 `0.413682`）。只读取阶段 3 既有配方评分复评：**12/12 均失败，0 通过**；每个配方都因 `spk10` 超过草案绝对上限，即使不计相对增幅也不通过；相对上限还被 `spk03/04/05/07` 触犯。完整两项诊断与复评见 `reports/timbre_blend_stage3_diagnostics.{md,json}` 和 `reports/timbre_blend_stage3_standard_draft.json`。在此停下等主人决定，不暗改现行门槛、不生成盲听包或精简模型。
