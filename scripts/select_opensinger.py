"""OpenSinger (female subset) singer statistics and selection tool.

Stage 1a of the timbre-blend pipeline. Scans a downloaded OpenSinger wav
tree (``<root>/<singer>_<song>/<singer>_<song>_<seg>.wav``), computes
per-file quality metrics, aggregates them per singer, writes a CSV report
and proposes a train / holdout-test singer split.

Per-file metrics
----------------
- duration_sec   : from the wav header
- clip_ratio     : fraction of samples with |x| >= 0.999 (full scale)
- snr_db         : energy-dynamic-range SNR proxy, p95(frame_db) -
                   p20(frame_db) over 25 ms frames with 10 ms hop.
                   Relative quality indicator, NOT a true SNR estimate.
- voiced_ratio   : fraction of RMVPE frames (hop 512 @ 44.1 kHz) with f0 > 0
- f0 histogram   : 96 quarter-tone bins over [60, 1000] Hz pooled per
                   singer for exact-enough p5 / median / p95
- octave_jumps   : adjacent voiced frame pairs with
                   |1200*log2(f0[i]/f0[i-1])| >= 1100 cents where both
                   sides are locally stable (neighbour diffs < 200 cents).
                   Reported per voiced minute.
- usable         : duration >= 2.0 s AND voiced_ratio >= 0.25 AND
                   clip_ratio <= 0.01 AND snr_db >= 15 AND
                   octave_jumps_per_voiced_min <= 30.

All thresholds and score weights below were fixed on 2026-09-27 BEFORE any
dataset statistics were inspected (see docs/TIMBRE_BLEND_LOG.md). They must
not be tuned after seeing results without logging the change and rationale.

Resumability: per-file results are appended to a JSONL cache; rerunning
skips files already present. Safe to interrupt and restart.

Usage:
  PYTHONPATH=<repo> python scripts/select_opensinger.py \
      --root ~/datasets/opensinger-womanraw \
      --out-csv reports/opensinger_singers.csv \
      --device cuda
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# --------------------------------------------------------------------------
# Fixed analysis constants (see module docstring)
# --------------------------------------------------------------------------
SAMPLE_RATE = 44100
HOP_SIZE = 512
F0_MIN = 65
F0_MAX = 800

MIN_DURATION_SEC = 2.0
MIN_VOICED_RATIO = 0.25
MAX_CLIP_RATIO = 0.01
MIN_SNR_DB = 15.0
MAX_OCT_JUMPS_PER_VOICED_MIN = 30.0
OCT_JUMP_CENTS = 1100.0
OCT_STABLE_CENTS = 200.0

PER_SINGER_CAP_MIN = 60.0  # plan section 12.5 initial target

# f0 histogram: 96 log-spaced bins over [60, 1000] Hz (~quarter tone)
F0_HIST_LO, F0_HIST_HI, F0_HIST_BINS = 60.0, 1000.0, 96
_F0_LOG_LO = math.log(F0_HIST_LO)
_F0_LOG_STEP = (math.log(F0_HIST_HI) - _F0_LOG_LO) / F0_HIST_BINS

# singer-level eligibility for the proposed split (fixed a priori)
MIN_USABLE_MIN_FOR_PROPOSAL = 10.0
MAX_FAILED_RATIO_FOR_PROPOSAL = 0.20

# quality score weights; negative weight = lower is better.
# range_fit: trapezoid membership of pooled f0 median in [100,150,350,450] Hz
# (common female pop singing range), 1.0 inside [150,350], 0.0 outside
# [100,450].
SCORE_WEIGHTS = {
    "snr_db_median": 0.25,
    "oct_jumps_per_voiced_min": -0.20,
    "usable_duration_min_capped60": 0.20,
    "voiced_ratio_mean": 0.15,
    "clip_ratio_mean": -0.10,
    "range_fit": 0.10,
}


def f0_to_hist(f0_voiced):
    """Quantize voiced f0 values into the fixed log-spaced histogram."""
    h = np.zeros(F0_HIST_BINS, dtype=np.int64)
    if len(f0_voiced) == 0:
        return h
    idx = np.floor((np.log(np.clip(f0_voiced, F0_HIST_LO, F0_HIST_HI)) - _F0_LOG_LO) / _F0_LOG_STEP).astype(np.int64)
    np.add.at(h, np.clip(idx, 0, F0_HIST_BINS - 1), 1)
    return h


def hist_percentiles(h, qs):
    """Percentiles from the pooled histogram (bin-center resolution)."""
    total = h.sum()
    if total == 0:
        return [float("nan")] * len(qs)
    cdf = np.cumsum(h) / total
    out = []
    for q in qs:
        b = int(np.searchsorted(cdf, q))
        b = min(b, F0_HIST_BINS - 1)
        out.append(round(math.exp(_F0_LOG_LO + (b + 0.5) * _F0_LOG_STEP), 1))
    return out


def frame_energy_db(x, sr, frame_ms=25, hop_ms=10):
    n = int(sr * frame_ms / 1000)
    h = max(1, int(sr * hop_ms / 1000))
    if len(x) < n:
        rms = math.sqrt(float(np.mean(x ** 2)) + 1e-12)
        return np.array([20 * math.log10(rms + 1e-12)])
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def count_octave_jumps(f0):
    """Count stable-to-stable >=1100-cent jumps between adjacent voiced frames."""
    idx = np.where(f0 > 0)[0]
    if len(idx) < 3:
        return 0
    adj = np.where(np.diff(idx) == 1)[0]  # positions in idx of adjacent pairs
    if len(adj) < 1:
        return 0
    d = 1200.0 * np.log2(f0[idx[adj + 1]] / f0[idx[adj]])
    jumps = 0
    for k in range(len(d)):
        if abs(d[k]) < OCT_JUMP_CENTS:
            continue
        prev_stable = (k == 0) or (abs(d[k - 1]) < OCT_STABLE_CENTS)
        next_stable = (k == len(d) - 1) or (abs(d[k + 1]) < OCT_STABLE_CENTS)
        if prev_stable and next_stable:
            jumps += 1
    return int(jumps)


def range_fit(f0_median):
    """Trapezoid membership over [100,150,350,450] Hz."""
    if not np.isfinite(f0_median):
        return 0.0
    if 150.0 <= f0_median <= 350.0:
        return 1.0
    if f0_median <= 100.0 or f0_median >= 450.0:
        return 0.0
    if f0_median < 150.0:
        return (f0_median - 100.0) / 50.0
    return (450.0 - f0_median) / 100.0


def analyze_file(path, f0_extractor, device):
    """Compute all per-file metrics. Returns a JSON-serializable dict."""
    rec = {"path": path, "error": None}
    try:
        info = sf.info(path)
        rec["duration_sec"] = round(float(info.duration), 3)
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
        if len(x) == 0:
            raise ValueError("empty audio")

        rec["clip_ratio"] = round(float(np.mean(np.abs(x) >= 0.999)), 6)
        edb = frame_energy_db(x, SAMPLE_RATE)
        rec["snr_db"] = round(float(np.percentile(edb, 95) - np.percentile(edb, 20)), 2)

        f0 = f0_extractor.extract(x, uv_interp=False, device=device)
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = f0 > 0
        n_voiced = int(voiced.sum())
        rec["n_frames"] = int(len(f0))
        rec["voiced_ratio"] = round(n_voiced / max(1, len(f0)), 4)
        voiced_min = n_voiced * HOP_SIZE / SAMPLE_RATE / 60.0
        jumps = count_octave_jumps(f0)
        rec["octave_jumps"] = jumps
        rec["oct_jumps_per_voiced_min"] = round(jumps / voiced_min, 2) if voiced_min > 1e-6 else 0.0
        rec["f0_hist"] = f0_to_hist(f0[voiced]).tolist()

        usable = (
            rec["duration_sec"] >= MIN_DURATION_SEC
            and rec["voiced_ratio"] >= MIN_VOICED_RATIO
            and rec["clip_ratio"] <= MAX_CLIP_RATIO
            and rec["snr_db"] >= MIN_SNR_DB
            and rec["oct_jumps_per_voiced_min"] <= MAX_OCT_JUMPS_PER_VOICED_MIN
        )
        rec["usable"] = bool(usable)
    except Exception as e:  # recorded, never silently swallowed
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["usable"] = False
    return rec


def discover_files(root, limit_per_singer=0):
    """Group wav files by singer id parsed from the top-level folder name."""
    root = Path(root)
    by_singer = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            p = Path(dirpath) / f
            rel = p.relative_to(root)
            m = re.match(r"^(\d+)", rel.parts[0])
            sid = int(m.group(1)) if m else -1
            by_singer[sid].append(str(rel))
    out = {}
    for sid, rels in sorted(by_singer.items()):
        rels.sort()
        if limit_per_singer and len(rels) > limit_per_singer:
            # evenly spaced subsample for deterministic coverage
            picks = np.linspace(0, len(rels) - 1, limit_per_singer).round().astype(int)
            rels = [rels[i] for i in sorted(set(picks))]
        out[sid] = rels
    return out


def load_cache(cache_path):
    done = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done[r["path"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated tail line from an interrupt
    return done


def aggregate(by_singer, records):
    rows = []
    for sid, rels in by_singer.items():
        recs = [records[r] for r in rels if r in records]
        failed = [r for r in recs if r.get("error")]
        ok = [r for r in recs if not r.get("error")]
        total_min = sum(r.get("duration_sec", 0.0) for r in recs) / 60.0
        usable = [r for r in ok if r.get("usable")]
        usable_min = sum(r["duration_sec"] for r in usable) / 60.0
        hist = np.zeros(F0_HIST_BINS, dtype=np.int64)
        for r in ok:
            hist += np.asarray(r.get("f0_hist", [0] * F0_HIST_BINS), dtype=np.int64)
        p5, med, p95 = hist_percentiles(hist, [0.05, 0.5, 0.95])
        voiced_frames = sum(r.get("n_frames", 0) * r.get("voiced_ratio", 0.0) for r in ok)
        voiced_min = voiced_frames * HOP_SIZE / SAMPLE_RATE / 60.0
        jumps = sum(r.get("octave_jumps", 0) for r in ok)
        row = {
            "singer_id": sid,
            "n_files": len(recs),
            "n_failed": len(failed),
            "total_duration_min": round(total_min, 2),
            "mean_file_sec": round(float(np.mean([r["duration_sec"] for r in ok])), 2) if ok else 0.0,
            "usable_files": len(usable),
            "usable_duration_min": round(usable_min, 2),
            "usable_duration_min_capped60": round(min(usable_min, PER_SINGER_CAP_MIN), 2),
            "snr_db_mean": round(float(np.mean([r["snr_db"] for r in ok])), 2) if ok else float("nan"),
            "snr_db_median": round(float(np.median([r["snr_db"] for r in ok])), 2) if ok else float("nan"),
            "voiced_ratio_mean": round(float(np.mean([r["voiced_ratio"] for r in ok])), 4) if ok else float("nan"),
            "oct_jumps_per_voiced_min": round(jumps / voiced_min, 2) if voiced_min > 1e-6 else 0.0,
            "clip_ratio_mean": round(float(np.mean([r["clip_ratio"] for r in ok])), 6) if ok else float("nan"),
            "f0_p5_hz": p5,
            "f0_median_hz": med,
            "f0_p95_hz": p95,
        }
        row["range_fit"] = round(range_fit(med), 3)
        rows.append(row)
    return rows


def rank_norm(values, higher_better):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    r = order / (n - 1)
    return r if higher_better else 1.0 - r


def score_and_propose(rows, n_train, n_holdout):
    scored = [r for r in rows if r["singer_id"] >= 0]
    eligible = [
        r for r in scored
        if r["usable_duration_min"] >= MIN_USABLE_MIN_FOR_PROPOSAL
        and r["n_files"] > 0
        and r["n_failed"] / r["n_files"] <= MAX_FAILED_RATIO_FOR_PROPOSAL
    ]
    if len(eligible) < 2:
        print(" [!] fewer than 2 eligible singers; cannot rank", file=sys.stderr)
        return rows, [], []
    missing = [k for k in SCORE_WEIGHTS if k not in eligible[0]]
    if missing:
        raise KeyError(f"score metric(s) missing from aggregated rows: {missing}")
    scores = np.zeros(len(eligible))
    for k, w in SCORE_WEIGHTS.items():
        scores += w * rank_norm([r[k] for r in eligible], higher_better=(w > 0))
    for r, s in zip(eligible, scores):
        r["quality_score"] = round(float(s), 4)
    for r in scored:
        r.setdefault("quality_score", float("nan"))
    order = sorted(eligible, key=lambda r: -r["quality_score"])
    train = order[:n_train]
    holdout = order[n_train:n_train + n_holdout]
    return rows, train, holdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="OpenSinger wav tree root")
    ap.add_argument("--out-csv", default="reports/opensinger_singers.csv")
    ap.add_argument("--out-proposal", default="reports/opensinger_proposal.json")
    ap.add_argument("--cache", default="reports/opensinger_file_metrics.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-per-singer", type=int, default=0,
                    help="max files per singer (0 = all, evenly spaced subsample)")
    ap.add_argument("--n-train", type=int, default=12, help="proposed train singers (8-15)")
    ap.add_argument("--n-holdout", type=int, default=3, help="proposed unseen test singers (2-3)")
    ap.add_argument("--seed", type=int, default=20260927, help="recorded for later stage splits")
    args = ap.parse_args()
    assert 8 <= args.n_train <= 15, "plan requires 8-15 train singers"
    assert 2 <= args.n_holdout <= 3, "plan requires 2-3 holdout singers"

    root = Path(os.path.expanduser(args.root)).resolve()
    if not root.is_dir():
        print(f" [x] root not found: {root}", file=sys.stderr)
        sys.exit(1)

    by_singer = discover_files(root, args.limit_per_singer)
    all_rels = [r for rels in by_singer.values() for r in rels]
    print(f" [*] {len(all_rels)} wav files across {len(by_singer)} singer folders under {root}")

    done = load_cache(args.cache)
    todo = [r for r in all_rels if r not in done]
    print(f" [*] cache: {len(done)} done, {len(todo)} to analyze")

    if todo:
        from ddsp.vocoder import F0_Extractor  # repo RMVPE wrapper
        f0x = F0_Extractor("rmvpe", SAMPLE_RATE, HOP_SIZE, F0_MIN, F0_MAX)
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "a", encoding="utf-8") as cf:
            for rel in tqdm(todo, desc="analyzing"):
                rec = analyze_file(str(root / rel), f0x, args.device)
                rec["path"] = rel
                m = re.match(r"^(\d+)", Path(rel).parts[0])
                rec["singer"] = int(m.group(1)) if m else -1
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cf.flush()
                done[rel] = rec

    rows = aggregate(by_singer, done)
    rows, train, holdout = score_and_propose(rows, args.n_train, args.n_holdout)
    rows.sort(key=lambda r: (-(r.get("quality_score") or -1), r["singer_id"]))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    cols = ["singer_id", "n_files", "n_failed", "total_duration_min", "mean_file_sec",
            "usable_files", "usable_duration_min", "usable_duration_min_capped60",
            "snr_db_mean", "snr_db_median", "voiced_ratio_mean",
            "oct_jumps_per_voiced_min", "clip_ratio_mean",
            "f0_p5_hz", "f0_median_hz", "f0_p95_hz", "range_fit", "quality_score"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f" [*] wrote {args.out_csv} ({len(rows)} singers)")

    proposal = {
        "seed": args.seed,
        "root": str(root),
        "n_files_analyzed": len(all_rels),
        "thresholds": {
            "min_duration_sec": MIN_DURATION_SEC,
            "min_voiced_ratio": MIN_VOICED_RATIO,
            "max_clip_ratio": MAX_CLIP_RATIO,
            "min_snr_db": MIN_SNR_DB,
            "max_oct_jumps_per_voiced_min": MAX_OCT_JUMPS_PER_VOICED_MIN,
            "oct_jump_cents": OCT_JUMP_CENTS,
            "oct_stable_cents": OCT_STABLE_CENTS,
            "per_singer_cap_min": PER_SINGER_CAP_MIN,
            "min_usable_min_for_proposal": MIN_USABLE_MIN_FOR_PROPOSAL,
            "max_failed_ratio_for_proposal": MAX_FAILED_RATIO_FOR_PROPOSAL,
        },
        "score_weights": SCORE_WEIGHTS,
        "proposed_train_singers": [r["singer_id"] for r in train],
        "proposed_holdout_test_singers": [r["singer_id"] for r in holdout],
    }
    with open(args.out_proposal, "w", encoding="utf-8") as f:
        json.dump(proposal, f, ensure_ascii=False, indent=2)
    print(f" [*] wrote {args.out_proposal}")

    print("\n === proposed TRAIN singers ===")
    for r in train:
        print(f"  spk {r['singer_id']:>3}: score={r['quality_score']:.3f} "
              f"usable={r['usable_duration_min']:.1f}min snr={r['snr_db_median']:.1f}dB "
              f"voiced={r['voiced_ratio_mean']:.2f} oct/min={r['oct_jumps_per_voiced_min']:.1f} "
              f"clip={r['clip_ratio_mean']:.4f} f0 p5/med/p95={r['f0_p5_hz']:.0f}/"
              f"{r['f0_median_hz']:.0f}/{r['f0_p95_hz']:.0f}Hz")
    print(" === proposed HOLDOUT (unseen test) singers ===")
    for r in holdout:
        print(f"  spk {r['singer_id']:>3}: score={r['quality_score']:.3f} "
              f"usable={r['usable_duration_min']:.1f}min f0 med={r['f0_median_hz']:.0f}Hz")
    total_capped = sum(r["usable_duration_min_capped60"] for r in train)
    if total_capped > 0:
        shares = sorted((r["usable_duration_min_capped60"] / total_capped for r in train), reverse=True)
        print(f" [*] capped usable total: {total_capped:.1f} min; largest singer share: {shares[0]*100:.1f}%")


if __name__ == "__main__":
    main()
