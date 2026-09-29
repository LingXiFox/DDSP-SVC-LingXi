# Stage 2 C expanded blind-listening comparison

> Status: **Human gate 3 — waiting for blind listening.** This is not a passed gate.

## Training and model selection

- C resumed from the independently verified step 4600 with AdamW optimizer state reset; the resume is **not** an uninterrupted training trajectory. Only the spk13 embedding row was trainable. Checkpoints through 10000 are preserved. All 50 scheduled 200-step validations completed; the two-consecutive->2% virtual-loss stop condition did not trigger.
- Best C by virtual-validation DDSP loss: **step 7800 = 0.99295372**, checkpoint MD5 `733d9c6d44071d91a1c0b0f190db44db`. At 10000 the loss was 0.99673958 (MD5 `f2b62856e11612406be4ddfb9bf93fbc`). The last checkpoint is not the selected model.
- All logged C public-validation metrics match the fixed baseline exactly; every validation reports the same frozen-old-rows SHA-256 `dc1a76612a3531bf18fea47e61b6af3383a101cf1f6b0ce43ef2ba2539244bb1`. Direct CPU comparisons of C@7800 and C@10000 with the resume checkpoint confirm only the spk13 row changed.
- The validation-waveform `vocoder.infer` call alone used cuDNN-disabled execution after an isolated 4200 baseline comparison matched every public/virtual metric exactly. The underlying intermittent native crash was not proven fixed.

## Listening material

- 11 unmodified, unspliced OpenSinger source clips from Stage-1-withheld singers 10 (4 clips), 29 (3 clips), and 47 (4 clips), totaling **113.593 s**; every clip is at least 5 s. Source SHA-256 values are recorded in `reports/timbre_blend_stage2_c_resume_holdout_inputs.json`.
- Three arms: B@1k, C@3000, C@7800. Each original clip uses the **same inference seed and settings across all arms** (`spk_id=13`, Euler 50 steps, `t_start=0`, key shift 0, F0 shift 0, RMVPE 50–1100 Hz, threshold −60 dB). Output filenames are random; the arm key is stored separately outside the listening package.
- RMVPE confirms frames above B4 in the selected singer-29 excerpts. Long-pitch candidates are based on Praat run measurements; low-HNR singer-47 excerpts are **only breathiness candidates**, not perceptually confirmed breathy singing. High-register and breathy coverage for every singer are not established.

## Independent ECAPA cosine proxy

SpeechBrain ECAPA (speech-trained, fixed local revision) ran in an independent offline CPU environment. The OpenSinger references use cross-song clips for the same singer and cross-singer clips for the other baseline. Model outputs are compared with the same four virtual-singer validation clips; 11 outputs per arm × 4 references = 44 comparisons. p05/p50/p95 below are percentiles of **pairwise comparisons**, not independent confidence intervals.

| Pairing | Pairs | Mean | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| OpenSinger: same singer, different song | 48 | 0.5575 | 0.2926 | 0.5881 | 0.7449 |
| OpenSinger: different singers | 1056 | 0.2642 | 0.0885 | 0.2590 | 0.4560 |
| B@1k → virtual validation | 44 | 0.6603 | 0.5455 | 0.6695 | 0.7605 |
| C@3000 → virtual validation | 44 | 0.6127 | 0.5212 | 0.5985 | 0.7388 |
| C@7800 → virtual validation | 44 | 0.6166 | 0.5119 | 0.6040 | 0.7467 |

| Input singer / arm | Pairs | Mean | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| 10 / B@1k | 16 | 0.6971 | 0.6156 | 0.6854 | 0.7862 |
| 10 / C@3000 | 16 | 0.6385 | 0.5476 | 0.6319 | 0.7524 |
| 10 / C@7800 | 16 | 0.6482 | 0.5542 | 0.6491 | 0.7625 |
| 29 / B@1k | 12 | 0.5877 | 0.5333 | 0.5830 | 0.6533 |
| 29 / C@3000 | 12 | 0.5752 | 0.5081 | 0.5777 | 0.6448 |
| 29 / C@7800 | 12 | 0.5726 | 0.5148 | 0.5789 | 0.6392 |
| 47 / B@1k | 16 | 0.6779 | 0.5693 | 0.6903 | 0.7420 |
| 47 / C@3000 | 16 | 0.6152 | 0.5173 | 0.6328 | 0.7012 |
| 47 / C@7800 | 16 | 0.6179 | 0.5050 | 0.6379 | 0.7093 |

## Training-side context (not a causal A/B test)

| Arm | Virtual-validation DDSP | Public-validation DDSP | Change vs public baseline |
|---|---:|---:|---:|
| B@1k | 0.86000972 | 0.25701990 | +2.672% |
| C@3000 | 1.02055079 | 0.25033053 | 0% |
| C@7800 | 0.99295372 | 0.25033053 | 0% |

- B@1k and C checkpoints differ in initialization, training data/strategy, and number of steps. Do not attribute score differences to one factor. Public-validation songs were included in Stage-1 training, so unchanged public metrics test forgetting on that fixed set, **not generalization**.
- On these 11 singing inputs the ECAPA proxy ranks B@1k above both C arms on the aggregate mean. C@7800 improves slightly over C@3000 on that proxy. Neither proxy nor validation DDSP establishes perceived timbre or mechanical artifacts; **blind listening decides**.
- The inference environment intermittently failed during Python imports (one SciPy-related exception and one native SIGSEGV before model inference); a CPU-only import probe reproduced anomalous interpreter errors. One strictly bounded retry of only the missing last output succeeded. The environment root cause remains unresolved; all 33 final audio files passed format/length/finiteness/non-silence checks. No package installations or WSL restarts were performed.

## Local artifacts and gate

- Blind package: 11 original `inputs/`, 33 randomized `audio/` WAVs, `pairs.json`, and listening instructions. **No arm key in the package.** Keep the separately stored key unopened until choices are locked.
- Raw independent similarity: `reports/timbre_blend_stage2_c_resume_similarity.json`; fixed selection metadata: `reports/timbre_blend_stage2_c_resume_holdout_inputs.json`; training curve: `exp/timbre_blend_stage2_embedding_resume_4600/validation_history.jsonl`.
- **Human gate 3: pending.** Compare each three-output set against its input, mark preference and mechanical/breathy/high-pitch observations before opening the key. No claim that a preferred model has been selected by this report.
