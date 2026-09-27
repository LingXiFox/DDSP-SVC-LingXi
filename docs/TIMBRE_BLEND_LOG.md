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
