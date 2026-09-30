#!/usr/bin/env python3
"""spk-41 (singer 41) F0 octave-error review - gate-1 obligation, read-only.

Pre-registered rule (docs/TIMBRE_BLEND_LOG.md, gate 1 + next-move plan):
compare the pipeline F0 (rmvpe, the f0 feature npy actually consumed by
training) against an independent second extractor (harvest / pyworld) on
the FINAL training slices, per file, on voiced frames only:
    ratio = harvest voiced median / rmvpe voiced median
  - majority (> 50%) of spk-41 files with ratio >= 1.5
      -> systematic octave-DOWN in the pipeline F0 -> STOP and report
         (user directive: do NOT preemptively "fix")
  - median ratio within [0.75, 1.33] AND < 20% of files with ratio >= 1.5
      -> genuine low-range voice -> continue
  - anything in between -> report the evidence, human judgment
Controls: the same measurement on 2 other singers (spk 10 / singer 36 and
spk 2 / singer 14, max 150 files each, deterministic stride) calibrates
the cross-extractor ratio distribution.

Inputs (after preprocess + audit):
  data/timbre_blend_stage1/{train,val}/audio/<spk_dir>/*.wav
  data/timbre_blend_stage1/{train,val}/f0/<spk_dir>/*.npy
Output: reports/timbre_blend_stage1_spk41_f0_review.json + printed table.
Exit 0 always; the verdict field carries CONTINUE / STOP / HUMAN_JUDGMENT.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path("data/timbre_blend_stage1").resolve()
TARGET = "11_singer41"
CONTROLS = {"10_singer36": 150, "2_singer14": 150}
HOP = 512
SR = 44100


def harvest_median(wav_path: str) -> float:
    import pyworld as pw
    x, sr = sf.read(wav_path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    f0, _ = pw.harvest(x, sr, frame_period=HOP / SR * 1000.0)
    v = f0[f0 > 1.0]
    return float(np.median(v)) if len(v) >= 5 else float("nan")


def rmvpe_median(npy_path: Path) -> float:
    f0 = np.load(str(npy_path))
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    v = f0[f0 > 1.0]
    return float(np.median(v)) if len(v) >= 5 else float("nan")


def collect(spk_dir: str, max_files: int):
    items = []
    for split in ("train", "val"):
        d = OUT / split / "audio" / spk_dir
        if not d.is_dir():
            continue
        for w in sorted(d.glob("*.wav")):
            npy = OUT / split / "f0" / spk_dir / (w.name + ".npy")
            items.append((spk_dir, split, str(w), str(npy)))
    if max_files and len(items) > max_files:
        stride = len(items) / max_files
        items = [items[int(i * stride)] for i in range(max_files)]
    return items


def measure(item):
    spk_dir, split, wav, npy = item
    npy_p = Path(npy)
    if not npy_p.exists():
        return {"spk_dir": spk_dir, "split": split, "wav": wav,
                "error": "f0_npy_missing"}
    h = harvest_median(wav)
    r = rmvpe_median(npy_p)
    rec = {"spk_dir": spk_dir, "split": split, "wav": wav,
           "harvest_med": round(h, 2) if h == h else None,
           "rmvpe_med": round(r, 2) if r == r else None}
    if h == h and r == r and r > 0:
        rec["ratio"] = round(h / r, 4)
    return rec


def summarize(spk_dir: str, recs):
    ratios = np.array([x["ratio"] for x in recs if "ratio" in x], dtype=float)
    if len(ratios) == 0:
        return {"n": len(recs), "n_ratio": 0}
    return {
        "n": len(recs),
        "n_ratio": int(len(ratios)),
        "n_error": sum(1 for x in recs if x.get("error")),
        "ratio_median": round(float(np.median(ratios)), 4),
        "ratio_p10": round(float(np.percentile(ratios, 10)), 4),
        "ratio_p90": round(float(np.percentile(ratios, 90)), 4),
        "frac_ratio_ge_1.5": round(float((ratios >= 1.5).mean()), 4),
        "frac_ratio_le_0.67": round(float((ratios <= 0.67).mean()), 4),
        "rmvpe_med_of_medians": round(float(np.median(
            [x["rmvpe_med"] for x in recs if x.get("rmvpe_med")])), 2),
        "harvest_med_of_medians": round(float(np.median(
            [x["harvest_med"] for x in recs if x.get("harvest_med")])), 2),
    }


def main():
    plan = [(TARGET, 0)] + list(CONTROLS.items())
    all_items = []
    for spk_dir, cap in plan:
        items = collect(spk_dir, cap)
        print(f"[*] {spk_dir}: {len(items)} files", flush=True)
        all_items.extend(items)
    with Pool(8) as pool:
        recs = pool.map(measure, all_items, chunksize=4)

    by_spk = {}
    for spk_dir, _ in plan:
        by_spk[spk_dir] = [x for x in recs if x["spk_dir"] == spk_dir]

    summary = {k: summarize(k, v) for k, v in by_spk.items()}
    t = summary[TARGET]
    if t.get("n_ratio", 0) == 0:
        verdict = "HUMAN_JUDGMENT (no measurable files)"
    elif t["frac_ratio_ge_1.5"] > 0.50:
        verdict = "STOP: systematic octave-down (majority ratio >= 1.5)"
    elif (0.75 <= t["ratio_median"] <= 1.33
          and t["frac_ratio_ge_1.5"] < 0.20):
        verdict = "CONTINUE: genuine low-range voice (no octave pathology)"
    else:
        verdict = "HUMAN_JUDGMENT: intermediate evidence"

    out = {"rule": "pre-registered gate-1 spk-41 review; see module docstring",
           "target": TARGET, "controls": list(CONTROLS),
           "summary": summary, "verdict": verdict,
           "per_file": recs}
    Path("reports/timbre_blend_stage1_spk41_f0_review.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'spk_dir':<14}{'n':>5}{'ratio_med':>10}{'p10':>7}{'p90':>7}"
          f"{'>=1.5':>8}{'<=0.67':>8}{'rmvpe_Hz':>9}{'harv_Hz':>9}")
    for k in [TARGET] + list(CONTROLS):
        s = summary[k]
        if s.get("n_ratio", 0) == 0:
            print(f"{k:<14}{s.get('n', 0):>5}  NO DATA")
            continue
        print(f"{k:<14}{s['n_ratio']:>5}{s['ratio_median']:>10}"
              f"{s['ratio_p10']:>7}{s['ratio_p90']:>7}"
              f"{s['frac_ratio_ge_1.5']:>8}{s['frac_ratio_le_0.67']:>8}"
              f"{s['rmvpe_med_of_medians']:>9}{s['harvest_med_of_medians']:>9}")
    print(f"\nVERDICT: {verdict}")
    print("detail: reports/timbre_blend_stage1_spk41_f0_review.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
