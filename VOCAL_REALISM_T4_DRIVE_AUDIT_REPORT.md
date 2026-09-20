# VOCAL_REALISM T4 + Drive Audit Report (FINAL)

> Branch: `feature/vocal-realism` @ `59af886` (pushed to origin).
> GPU: Google Colab Tesla T4 only. Local RTX 4090 never touched.
> Local machine: no torch/CUDA/baseline/pretrain/datasets; Git + Agent + Colab CLI only.
> Sessions `vocal-audit` and `vocal-assets` both terminated; no orphans.
> R3 Stereo NOT started.

## 0. Baseline waiver (per amendment)

```text
暂无可靠历史 6.3 checkpoint，因此未执行真实历史模型音频 A/B。
```

`models/baseline-6.3/` exists on Drive but is EMPTY. No random/foreign/fresh checkpoint
was misrepresented as a historical baseline. All compatibility claims below are
structural (bypass/zero-init/fresh-integration), not audio-quality claims.

## 1. Colab session / T4

- `nvidia-smi`: Tesla T4, 15360 MiB, driver 580.82.07.
- torch 2.11.0+cu128, cuda runtime 12.8, available=True, cap (7,5), VRAM 14.56 GB.

## 2. Drive layout (all under `/content/drive/MyDrive/DDSP-SVC-LingXi/`)

```text
repository/DDSP-SVC-LingXi @ 59af886 (persistent, GitHub -> Colab -> Drive)
pretrain/contentvec/pytorch_model.bin (361M)
pretrain/rmvpe/model.pt (352M)
pretrain/nsf_hifigan/model (55M) + config.json (607B)
pretrain/SHA256SUMS.txt
models/baseline-6.3/ (EMPTY, intentional)   models/realism/ (reserved)
data/realism-public-smoke/{train(12),val(4)} (synthetic audio + REAL preprocess outputs)
data/test/{test01,test02,test03}.wav (synthetic, fixed inputs)
data/private/ + data/realism-public/ (reserved, await real data)
experiments/realism-prior-smoke/{realism_final.pt, realism_latest.pt}
experiments/full-smoke/full_with_prior.pt
outputs/smoke_zeroinit_test01.wav (1,1,176640 ≈ 4.0s @44.1k)
```

SHA256:

```text
d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e  contentvec/pytorch_model.bin
19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2  rmvpe/model.pt
d6dd28909d2a1a2dcf74b3e3aa0b82b48695b87979fdf41561940aeecd85c67f  nsf_hifigan/model
983bb78f45f6790e573033bfab46163743e0182d564fc5eca6938743cca6a2ea  nsf_hifigan/config.json
```

Download path was public URL -> `/content/ddsp-assets` -> verify -> Drive. Work repo at
`/content/DDSP-SVC-LingXi` used Drive `pretrain` via symlink. Runtime ephemera discarded.

## 3. pytest + compile (§8 analog) — PASS on T4

```text
compileall ddsp reflow logger train_reflow.py train_realism.py -> COMPILE-OK
pytest (5 files incl. new test_realism_bypass.py) -> 17 passed
```

## 4. CUDA smoke (§9 analog) — PASS on T4

```text
steps=10:  initial_loss=0.412863 final_loss=0.061817 peak_cuda_mb=22.7
steps=100: initial_loss=0.412863 final_loss=0.000050 peak_cuda_mb=22.7
```

Zero-init identity + strict round-trip asserts inside; no NaN/Inf.
NOTE: `scripts/realism_smoke.py` needs `PYTHONPATH=<repo>` (lives in `scripts/`).

## 5. Disabled structural bypass (amendment A+B) — PASS, new regression tests

`tests/test_realism_bypass.py` (4 tests, in the 17 passed):

```text
disabled wrapper instantiates no adapter (realism is None)
disabled wrapper adds zero parameters (state_dict keys+values identical)
disabled forward == upstream forward bit-exact (torch.equal, reseeded noise)
enabled zero-init output == disabled output bit-exact
```

## 6. Zero-init identity (amendment C) — PASS (strict)

Adapter level (`torch.equal`): smoke asserts + bypass test.
Full-synth level: enabled-zero-init wrapper output == disabled output bit-exact.
Simulated-old-checkpoint load: 10 missing keys, all `realism`, 0 unexpected, identity holds.

## 7. Fresh full-model end-to-end (amendment D) — PASS

`scripts/fresh_model_audit.py` on T4, real config dims, real vocoder load:

```text
infer forward (real NSF-HiFiGAN): finite wav
train forward+backward (personalize stage): loss finite, grad finite, output_proj left zero-init
full ckpt strict reload: identical weights, 10 ddsp_model.realism.* keys present
old-ckpt simulation (strip realism keys, strict=False): no crash, identity holds
R2 matrix: timbre backbone 54914696/realism 0; personalize 0/199042; joint both>0
joint LRs: backbone 5e-5 < realism 3e-4
peak_cuda_mb=1158.9
FRESH MODEL AUDIT PASS
```

Realism adapter = 199,042 params (~0.36% of 54.9M backbone).

## 8. Public prior 120-step (R1, real pipeline, synthetic audio) — PASS

16 synthetic singing-like wavs (vibrato, gaps, 98–659 Hz) -> real T4 preprocess
(ContentVec + RMVPE + NSF-HiFiGAN mel) -> `train_realism.py`:

```text
train_items=12 valid_items=4, control-only
step=5 loss=8.25769 -> step=120 loss=1.81986, |df0|~9-11c (bounded), grad finite
validation total 4.36948 -> 3.63524
saved experiments/realism-prior-smoke/realism_final.pt (format lingxi-vocal-realism-v1)
```

No waveform/mel/speaker/formant in loss (static audit §12 held at runtime: loader only
yields units/f0/volume).

## 9. Prior injection (§14 analog) — PASS

```text
prior strict-load into adapter ok (256/256 output_proj params nonzero)
full-model constructor injection via realism.checkpoint ok
full checkpoint contains 10 ddsp_model.realism.* keys
full checkpoint strict reload: injected weights identical
```

## 10. Optimizer resume (§16 analog) — PASS

```text
resume_optimizer=false across personalize->joint: weights restored, no crash
resume_optimizer=true across layouts: loud ValueError (fail-loud, never silent)
```

## 11. Checkpoint self-consistency (amendment §9) — PASS

standalone realism ckpt + full ckpt save/reload/resume all verified; no historical-ckpt
claim made. Static compat audit: `strict=False` restores tolerate missing realism keys;
recommendation (not implemented): log missing/unexpected keys in `load_model` instead of
swallowing silently.

## 12. Audio — pipeline proof only, NO quality conclusion

`outputs/smoke_zeroinit_test01.wav` renders end-to-end (encoder->fresh zero-init model->
vocoder). Per amendment §6: fresh-model audio proves the pipeline runs; it says nothing
about Vocal Realism quality in either direction.

## 13. Bugs found / fixed (3 commits)

```text
b1d864f fix: load_model_vocoder never instantiated vocoder (NameError on ALL
        inference entries: main/gui/batch) — would have broken every A/B.
0af7216 test: disabled-path structural bypass regression tests.
b987071+59af886 test/fix: fresh full-model audit script + optimizer layout fix in script.
```

## 14. Environment gotchas (for next runs)

- `colab drivemount` / `drive.mount` needs one human browser consent per VM; afterwards
  headless `ls /content/drive` works.
- Kernel persists across `exec`: after `pip install -r requirements.txt` downgrades numpy
  (1.26.4), the IN-KERNEL numpy/scipy import breaks (`numpy.strings`). Always run work in
  fresh `subprocess` via `exec -f` runner scripts, never heavy imports in-kernel.
- Preprocess on T4: `-j 1`, ~2-3 it/s after model load; RMVPE+ContentVec fit in 15 GB.

## 15. Completion checklist (amendment §12)

```text
[x] Tesla T4 正常            [x] Drive 资产正常 (除 intentional-empty baseline)
[x] ContentVec/RMVPE/HiFiGAN 正常 (Drive + SHA256 + 实际加载推理)
[x] compileall / unit tests (17) / CUDA smoke
[x] disabled 结构旁路确认    [x] wrapper 与原 synthesizer 代码路径等价 (bit-exact)
[x] zero-init identity 严格成立 [x] unvoiced F0 始终为 0
[x] Fresh forward/backward/ckpt-save-reload/inference
[x] Public Prior 120-step    [x] 无 timbre leakage path
[x] personalize/joint freeze 正确 [x] optimizer stage transition 正确 (fail-loud)
[x] 无 NaN/Inf               [x] 无异常 OOM (peak 1159MB full / 22.7MB adapter)
[ ] 真实历史 6.3 音频 A/B — 明确未做 (OPTIONAL, 无可靠基线)
[ ] 真实人声数据验证 — 待主人提供 private/public 真实录音 preprocess 后重跑 §8/§15
```

## 16. Remaining work (needs real vocal recordings, not code)

1. 主人提供真实歌声 wav (private 目标 + public 多歌手) -> T4 preprocess 入 Drive.
2. 重跑 public prior (§8) + personalize/joint (§15) 于真实数据.
3. 真实数据上的 audio A/B (B/C/D/E) + 身份保持评估.
4. 届时 R0-R2 可判结案, 再议 R3 Mid/Side.
