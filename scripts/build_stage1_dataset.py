#!/usr/bin/env python
"""Stage 1b: build the multi-singer training tree from official OpenSinger WomanRaw.

Pipeline (plan §12). Source data is READ-ONLY; every artifact is written
under --out-root. All processing parameters below were fixed BEFORE the
first full run and are recorded in docs/TIMBRE_BLEND_LOG.md; they must not
be tuned after seeing training results without logging the change.

Steps
-----
1. Read the gate-1 selection (reports/opensinger_selection_final.json).
2. Pre-filter source files with the stage-1a per-file metrics cache
   (same fixed thresholds, logic reused from scripts/select_opensinger.py).
3. Per singer (ascending id, single seeded rng stream => deterministic):
   shuffle usable files, accumulate until the 60 min pool cap, then split
   the pool 95/5 train/val AT SOURCE-FILE LEVEL (plan §11: stable,
   reproducible per original performance segment).
4. Loudness-normalize with the production static-gain scheme (user
   decision 2026-09-27, A+): MEASURE with ffmpeg loudnorm pass 1 only
   (integrated loudness + true peak), compute
       gain = min(TARGET_I - measured_I, TARGET_TP - measured_TP)
   and apply it as ONE constant multiplication (ffmpeg volume filter).
   TARGET_I=-23 LUFS, TARGET_TP=-1.5 dBTP; output 44100 Hz mono
   pcm_s16le wav. Compressor / limiter / dynamic loudnorm are FORBIDDEN
   anywhere in the pipeline. When the true-peak constraint binds
   (tp_limited), TP safety wins: final loudness stays below target and
   the file is recorded in the per-singer norm_tp_limited list.
   History: the first full run (2026-09-27) used two-pass loudnorm
   linear=true; 50/6041 files fell back to Dynamic normalization and
   were remediated with this scheme (scripts/remediate_static_gain.py,
   reports/timbre_blend_stage1_remediation.json). Acceptance evidence:
   scripts/verify_loudnorm.py v2 (static-normalization verifier, full
   tree).
5. Slice with the repo Slicer (slicer.py reused per plan §12.3):
   threshold=-40 dB, min_length=2000 ms (== model min training duration,
   configs data.duration=2 s), min_interval=300 ms, hop=20 ms,
   max_sil_kept=500 ms (deviates from the repo default 5000 ms, which is
   an inference-chunking setting; 500 ms trims edge silence for training).
   Only non-silence chunks (slice==False) of >= 2.0 s are kept.
6. Per-slice quality re-check on GPU with stage-1a analyze_file (plan
   §12.4). Slice-level criteria: duration >= 2 s, voiced_ratio >= 0.25,
   clip_ratio <= 0.01, oct_jumps <= 30/min — threshold VALUES identical
   to stage 1a. The SNR proxy is NOT applied at slice level: it is a
   source-recording property, systematically distorted by silence
   trimming (the p20 energy floor rises when quiet edges are removed;
   measured 18.8 dB full-file -> 13.7 dB trimmed on a clean sample,
   while static-gain normalization itself is exactly invariant).
   SNR >= 15 dB is enforced at source level by the stage-1a prefilter
   instead. Decision logged 2026-09-27 BEFORE any full run or training;
   slice snr_db values remain recorded in slice_metrics.jsonl for audit.
   Failing slices are removed from the output tree; every rejection is
   logged to <out_root>/rejected.jsonl.
7. Speaker ids: singers ascending -> spk_id 1..N (repo data loader parses
   the first _/- token of the audio subdir name, 1-based, see
   reflow/data_loaders.py). Directory names: "<spk_id>_singer<NN>".
   Mapping persisted to reports/timbre_blend_speakers.json (§12.6:
   downstream stages must not renumber).

Outputs
-------
<out_root>/{train,val}/audio/<dir>/*.wav   consumed by preprocess.py
<out_root>/normalized/<dir>/*.wav          static-gain normalized sources (audit)
<out_root>/manifest.json                   per-singer pool/train/val file lists
<out_root>/slice_metrics.jsonl             per-slice recheck cache (resumable)
<out_root>/rejected.jsonl                  every rejection with reason
reports/timbre_blend_speakers.json
reports/timbre_blend_stage1_build.json

Resumability: normalized files are skipped when present (tmp+rename write
pattern => a present file is complete); slice metrics are cached by output
relative path. Reruns reproduce the same plan given (seed, selection,
stage-1a cache).

Usage:
  PYTHONPATH=<repo> python scripts/build_stage1_dataset.py \
      --src-root ~/datasets/opensinger-official/OpenSinger/WomanRaw \
      --out-root data/timbre_blend_stage1 --device cuda
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slicer import Slicer  # repo slicer, reused per plan §12.3
from scripts.select_opensinger import (  # stage-1a metric logic, reused per §12.4
    F0_MAX,
    F0_MIN,
    HOP_SIZE,
    MAX_CLIP_RATIO,
    MAX_OCT_JUMPS_PER_VOICED_MIN,
    MIN_DURATION_SEC,
    MIN_SNR_DB,
    MIN_VOICED_RATIO,
    SAMPLE_RATE,
    analyze_file,
    load_cache,
)

# ---- fixed processing parameters (mirrored into the build report) ----
# Production normalization (user decision 2026-09-27, A+): loudnorm pass 1
# is a MEASUREMENT tool only; the applied gain is one explicit static
# multiplication (ffmpeg volume filter). No dynamic processing anywhere.
TARGET_I = -23.0       # target integrated loudness, LUFS (EBU R128)
TARGET_TP = -1.5       # true-peak ceiling, dBTP
MEASURE_LRA = 11.0     # loudnorm pass-1 filter hint (measurement only)
SLICER_KWARGS = dict(
    threshold=-40.0,
    min_length=2000,   # ms; == model min training duration (data.duration=2 s)
    min_interval=300,
    hop_size=20,
    max_sil_kept=500,  # repo default 5000 is an inference setting; see docstring
)
MIN_SLICE_SEC = 2.0    # == stage-1a MIN_DURATION_SEC


def parse_args(args=None, namespace=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selection", default="reports/opensinger_selection_final.json")
    ap.add_argument("--src-root", default="~/datasets/opensinger-official/OpenSinger/WomanRaw")
    ap.add_argument("--metrics-cache", default="reports/opensinger_file_metrics.jsonl")
    ap.add_argument("--out-root", default="data/timbre_blend_stage1")
    ap.add_argument("--speakers-out", default="reports/timbre_blend_speakers.json")
    ap.add_argument("--build-report", default="reports/timbre_blend_stage1_build.json")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--cap-min", type=float, default=60.0,
                    help="per-singer pool cap in minutes (plan §12.5)")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="per-singer validation fraction, source-file level (plan §11)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ffmpeg-jobs", type=int, default=8)
    ap.add_argument("--singers", default="",
                    help="comma list to restrict processing (smoke tests); "
                         "spk ids always follow the FULL selection")
    ap.add_argument("--limit-per-singer", type=int, default=0,
                    help="cap pool file count per singer (smoke tests)")
    return ap.parse_args(args=args, namespace=namespace)


# --------- loudness normalization (measure + explicit static gain) ---------

def measure_loudness(src):
    """Measure input loudness (loudnorm pass 1, MEASUREMENT ONLY).

    Deterministic; returns the measurement dict or None. The filter string
    is byte-identical to the original build's pass 1, so measurements stay
    comparable across the whole verification history.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
           "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={MEASURE_LRA}:print_format=json",
           "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        v = j.get(k)
        if v is None or str(v) == "-inf":
            return None
    return j


def compute_static_gain(meas):
    """gain = min(gain_loudness, gain_peak): TP-safe by construction.

    gain_loudness drives integrated loudness to TARGET_I; gain_peak caps
    the true peak at TARGET_TP. When the peak constraint binds
    (tp_limited=True), the final loudness stays BELOW target - allowed and
    recorded (user directive 2026-09-27: TP safety first, never dynamic
    processing to force the loudness target).
    """
    gain_loudness = TARGET_I - float(meas["input_i"])
    gain_peak = TARGET_TP - float(meas["input_tp"])
    return min(gain_loudness, gain_peak), gain_peak < gain_loudness


def apply_static_gain(src, dst, gain_db):
    """Apply ONE constant gain via the ffmpeg volume filter. tmp+rename.

    Pure sample-wise multiplication: no compressor, no limiter, no dynamic
    loudnorm (forbidden by the production data principle). Output true peak
    cannot exceed src true peak + gain; compute_static_gain keeps that
    <= TARGET_TP by construction.
    """
    tmp = dst.with_name(dst.name + ".tmp.wav")
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(src),
           "-af", f"volume={gain_db:.6f}dB",
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(tmp)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dst)
    return True


def normalize_one(rel, src, dst):
    """Measure (loudnorm pass 1) + explicit static gain (volume filter)."""
    if dst.exists() and dst.stat().st_size > 1024:
        return rel, "cached", None
    meas = measure_loudness(src)
    if meas is None:
        return rel, "measure_failed", None
    gain, tp_limited = compute_static_gain(meas)
    if not apply_static_gain(src, dst, gain):
        return rel, "apply_failed", None
    return rel, "ok", {"gain_db": round(gain, 6), "tp_limited": tp_limited}


# ---------------- slicing ----------------

def slice_normalized(norm_wav, out_dir, stem, slicer):
    """Slice one normalized wav; keep non-silence chunks >= MIN_SLICE_SEC."""
    x, sr = sf.read(str(norm_wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"unexpected sr {sr} in {norm_wav}")
    chunks = slicer.slice(x)
    written, n_short = [], 0
    for k in sorted(chunks, key=int):
        v = chunks[k]
        if v["slice"]:
            continue  # silence region
        a, b = (int(t) for t in v["split_time"].split(","))
        seg = x[a:b]
        if len(seg) < MIN_SLICE_SEC * SAMPLE_RATE:
            n_short += 1
            continue
        out = out_dir / f"{stem}__s{k}.wav"
        sf.write(str(out), seg, SAMPLE_RATE, subtype="PCM_16")
        written.append(out)
    return written, n_short


# ---------------- per-slice quality re-check (stage-1a logic) ----------------

def slice_usable(rec):
    """Stage-1a threshold values applied at slice level, minus the SNR proxy.

    The p95-p20 frame-energy SNR proxy measures a source-recording property
    and is systematically distorted on silence-trimmed slices (quiet edges
    removed => p20 floor rises => proxy drops; e.g. 18.8 dB full-file vs
    13.7 dB trimmed on a verified-clean sample). SNR >= MIN_SNR_DB is
    therefore enforced at SOURCE level by the stage-1a prefilter (every
    pool file passed it). See docstring + docs/TIMBRE_BLEND_LOG.md.
    """
    if rec.get("error"):
        return False
    return (rec.get("duration_sec", 0.0) >= MIN_DURATION_SEC
            and rec.get("voiced_ratio", 0.0) >= MIN_VOICED_RATIO
            and rec.get("clip_ratio", 1.0) <= MAX_CLIP_RATIO
            and rec.get("oct_jumps_per_voiced_min", 1e9) <= MAX_OCT_JUMPS_PER_VOICED_MIN)


def failed_criteria(rec):
    if rec.get("error"):
        return ["error"]
    f = []
    if rec.get("duration_sec", 0.0) < MIN_DURATION_SEC:
        f.append("duration")
    if rec.get("voiced_ratio", 0.0) < MIN_VOICED_RATIO:
        f.append("voiced_ratio")
    if rec.get("clip_ratio", 1.0) > MAX_CLIP_RATIO:
        f.append("clip_ratio")
    if rec.get("oct_jumps_per_voiced_min", 1e9) > MAX_OCT_JUMPS_PER_VOICED_MIN:
        f.append("octave_jumps")
    return f


def recheck_slices(slice_paths, out_root, slice_cache_path, f0x, device):
    """analyze_file every slice; remove + report failures. Cached, resumable."""
    cache = load_cache(str(slice_cache_path))
    removed = []
    with open(slice_cache_path, "a", encoding="utf-8") as cf:
        for p in tqdm(sorted(slice_paths), desc="slice recheck"):
            rel = str(p.relative_to(out_root))
            rec = cache.get(rel)
            if rec is None:
                rec = analyze_file(str(p), f0x, device)
                rec["path"] = rel
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cf.flush()
                cache[rel] = rec
            if not slice_usable(rec):
                p.unlink(missing_ok=True)
                removed.append((rel, ",".join(failed_criteria(rec)) or "unknown", rec))
    return removed


# ---------------- main ----------------

def main():
    args = parse_args()
    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    all_singers = sorted(sel["train_singers"])
    spk_of = {s: i + 1 for i, s in enumerate(all_singers)}  # 1-based, ascending: STABLE
    singers = all_singers
    if args.singers:
        want = {int(t) for t in args.singers.split(",")}
        singers = [s for s in all_singers if s in want]
        if not singers:
            sys.exit("[x] --singers matches none of the selected train singers")

    src_root = Path(os.path.expanduser(args.src_root)).resolve()
    out_root = Path(os.path.expanduser(args.out_root)).resolve()
    if not src_root.is_dir():
        sys.exit(f"[x] src root missing: {src_root}")
    if out_root == src_root or src_root in out_root.parents:
        sys.exit("[x] out_root must live outside the (read-only) source tree")

    cache = load_cache(args.metrics_cache)
    if not cache:
        sys.exit("[x] stage-1a metrics cache empty; run scripts/select_opensinger.py first")

    # usable source files per singer, straight from the stage-1a cache
    by_singer = {s: [] for s in singers}
    for rel, rec in cache.items():
        m = re.match(r"^(\d+)", rel)
        if not m:
            continue
        s = int(m.group(1))
        if s in by_singer and rec.get("usable") and not rec.get("error"):
            by_singer[s].append(rel)
    for s in singers:
        by_singer[s].sort()

    # ---- deterministic plan: pool cap + file-level 95/5 split ----
    rng = random.Random(args.seed)
    plan = {}
    for s in singers:  # ascending; one shared rng stream => reproducible
        files = list(by_singer[s])
        rng.shuffle(files)
        pool, acc = [], 0.0
        for rel in files:
            d = cache[rel].get("duration_sec", 0.0)
            if pool and acc + d > args.cap_min * 60.0:
                break
            pool.append(rel)
            acc += d
        if args.limit_per_singer and len(pool) > args.limit_per_singer:
            pool = sorted(pool)[: args.limit_per_singer]
            acc = sum(cache[r].get("duration_sec", 0.0) for r in pool)
        n_val = max(1, round(len(pool) * args.val_frac)) if pool else 0
        plan[s] = {"pool": pool, "pool_min": acc / 60.0,
                   "val": sorted(pool[:n_val]), "train": sorted(pool[n_val:])}
        print(f" [*] singer {s:>2} (spk {spk_of[s]:>2}): usable={len(files)} "
              f"pool={len(pool)} ({acc / 60.0:.1f} min) -> "
              f"train_src={len(plan[s]['train'])} val_src={len(plan[s]['val'])}")

    out_root.mkdir(parents=True, exist_ok=True)
    rej_path = out_root / "rejected.jsonl"
    slice_cache_path = out_root / "slice_metrics.jsonl"
    slicer = Slicer(sr=SAMPLE_RATE, **SLICER_KWARGS)

    def log_reject(singer, stage, item, detail):
        with open(rej_path, "a", encoding="utf-8") as rf:
            rf.write(json.dumps({"singer": singer, "stage": stage, "item": item,
                                 "detail": detail}, ensure_ascii=False) + "\n")

    from ddsp.vocoder import F0_Extractor  # lazy: pulls torch
    f0x = F0_Extractor("rmvpe", SAMPLE_RATE, HOP_SIZE, F0_MIN, F0_MAX)

    per_singer = []
    for s in singers:
        spk = spk_of[s]
        dir_name = f"{spk}_singer{s:02d}"
        norm_dir = out_root / "normalized" / dir_name
        norm_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for split in ("train", "val"):
            for rel in plan[s][split]:
                src = src_root / rel
                if not src.exists():
                    log_reject(s, "src_missing", rel, "file absent from source root")
                    continue
                tasks.append((split, rel, src, norm_dir / (Path(rel).stem + ".wav")))

        # 1) loudness normalization: measure (loudnorm pass 1) + explicit
        #    static gain (ffmpeg volume). Parallel ffmpeg.
        stats = defaultdict(int)
        tp_limited_files = []
        with ThreadPoolExecutor(args.ffmpeg_jobs) as ex:
            futs = {ex.submit(normalize_one, rel, src, dst): (split, rel, dst)
                    for split, rel, src, dst in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=f"static-gain spk{spk}", leave=False):
                split, rel, dst = futs[fut]
                rel_, status, info = fut.result()
                stats[f"norm_{status}"] += 1
                if info is not None and info["tp_limited"]:
                    tp_limited_files.append({"rel": rel, "gain_db": info["gain_db"]})
                if status.endswith("failed"):
                    log_reject(s, f"norm_{status}", rel,
                               "measure + static-gain normalization")
        stats["norm_tp_limited"] = len(tp_limited_files)
        if tp_limited_files:
            print(f" [!] spk {spk}: {len(tp_limited_files)} file(s) TP-limited "
                  f"(loudness below target, TP safety first; recorded in report)")

        # 2) slicing (repo Slicer) into the split audio trees
        singer_slices = []
        for split, rel, src, dst in tasks:
            if not dst.exists():
                continue
            audio_dir = out_root / split / "audio" / dir_name
            audio_dir.mkdir(parents=True, exist_ok=True)
            try:
                written, n_short = slice_normalized(dst, audio_dir, Path(rel).stem, slicer)
            except Exception as e:  # recorded, never silently swallowed
                log_reject(s, "slice_error", rel, f"{type(e).__name__}: {e}")
                stats["slice_error"] += 1
                continue
            singer_slices.extend(written)
            stats["slices_written"] += len(written)
            stats["slices_short"] += n_short
            if n_short:
                log_reject(s, "slice_short", rel,
                           f"{n_short} chunk(s) < {MIN_SLICE_SEC}s discarded")

        # 3) per-slice quality re-check on GPU; failures removed + logged
        removed = recheck_slices(singer_slices, out_root, slice_cache_path, f0x, args.device)
        for rel, reason, rec in removed:
            log_reject(s, "quality", rel,
                       {"failed": reason,
                        **{k: rec.get(k) for k in ("duration_sec", "voiced_ratio", "snr_db",
                                                   "clip_ratio", "oct_jumps_per_voiced_min")}})
        stats["slices_quality_removed"] = len(removed)

        # final per-split stats from disk (accepted slices only)
        row = {"singer": s, "spk_id": spk, "dir_name": dir_name,
               "n_usable_src": len(by_singer[s]), "n_pool": len(plan[s]["pool"]),
               "pool_min_stage1a": round(plan[s]["pool_min"], 2),
               "n_train_src": len(plan[s]["train"]), "n_val_src": len(plan[s]["val"]),
               "norm": {k: v for k, v in stats.items() if k.startswith("norm_")},
               "norm_tp_limited": tp_limited_files,
               "slices_short_discarded": stats["slices_short"],
               "slices_quality_removed": stats["slices_quality_removed"]}
        for split in ("train", "val"):
            d = out_root / split / "audio" / dir_name
            wavs = sorted(d.glob("*.wav")) if d.is_dir() else []
            dur = sum(sf.info(str(w)).duration for w in wavs)
            row[f"n_slices_{split}"] = len(wavs)
            row[f"{split}_min"] = round(dur / 60.0, 2)
        per_singer.append(row)
        print(f" [*] spk {spk:>2} singer {s:>2} done: train {row['n_slices_train']} slices "
              f"({row['train_min']} min), val {row['n_slices_val']} slices "
              f"({row['val_min']} min), rejected: short={row['slices_short_discarded']} "
              f"quality={row['slices_quality_removed']} norm={row['norm']}")

    # ---- speaker mapping (§12.6: stable, downstream must not renumber) ----
    speakers = {
        "n_spk": len(all_singers),
        "id_convention": "audio subdir first _/- token, 1-based (reflow/data_loaders.py)",
        "seed": args.seed,
        "mapping": [{"spk_id": spk_of[s], "opensinger_singer": s,
                     "dir_name": f"{spk_of[s]}_singer{s:02d}"} for s in all_singers],
        "holdout_singers": sel.get("holdout_test_singers", []),
        "holdout_note": "never processed into the training tree; unseen test inputs, "
                        "source audio stays in the official WomanRaw root",
    }
    Path(args.speakers_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.speakers_out).write_text(
        json.dumps(speakers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f" [*] wrote {args.speakers_out}")

    # ---- manifest (full file lists for audit) + build report ----
    manifest = {"seed": args.seed, "src_root": str(src_root),
                "per_singer": {str(s): {"spk_id": spk_of[s], **plan[s]} for s in singers}}
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    totals = {
        "train_slices": sum(r["n_slices_train"] for r in per_singer),
        "val_slices": sum(r["n_slices_val"] for r in per_singer),
        "train_min": round(sum(r["train_min"] for r in per_singer), 2),
        "val_min": round(sum(r["val_min"] for r in per_singer), 2),
    }
    report = {
        "seed": args.seed,
        "src_root": str(src_root),
        "out_root": str(out_root),
        "selection_file": args.selection,
        "stage1a_prefilter": "usable==true in reports/opensinger_file_metrics.jsonl "
                             "(fixed 2026-09-27 thresholds, unchanged)",
        "pool_cap_min_per_singer": args.cap_min,
        "val_frac": args.val_frac,
        "val_split_level": "source file (original performance segment)",
        "normalization": {
            "method": "measure-only (ffmpeg loudnorm pass 1) + explicit static "
                      "gain: gain = min(target_I - measured_I, target_TP - "
                      "measured_TP), applied via ffmpeg volume=<gain>dB",
            "target_I_lufs": TARGET_I, "target_TP_dbtp": TARGET_TP,
            "dynamic_processing": "forbidden: no compressor / limiter / dynamic "
                                  "loudnorm (user decision 2026-09-27, A+)",
            "tp_limited_policy": "TP safety first: final loudness may stay below "
                                 "target; files listed in per-singer norm_tp_limited",
            "output": f"{SAMPLE_RATE} Hz mono pcm_s16le wav"},
        "slicer": {"impl": "repo slicer.py Slicer (reused)", **SLICER_KWARGS,
                   "min_slice_sec": MIN_SLICE_SEC,
                   "note": "max_sil_kept=500 deviates from repo default 5000 "
                           "(inference setting) to trim edge silence for training"},
        "quality_recheck": {
            "slice_level": f"duration>={MIN_DURATION_SEC}s, voiced>={MIN_VOICED_RATIO}, "
                           f"clip<={MAX_CLIP_RATIO}, oct<={MAX_OCT_JUMPS_PER_VOICED_MIN}/min "
                           "(stage-1a threshold VALUES, GPU rmvpe)",
            "snr": f"enforced at source level via stage-1a prefilter (>={MIN_SNR_DB} dB); "
                   "slice-level SNR proxy distorted by silence trimming, "
                   "see docstring + log 2026-09-27; values kept in slice_metrics.jsonl"},
        "per_singer": per_singer,
        "totals": totals,
    }
    Path(args.build_report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f" [*] wrote {args.build_report}")

    print("\n === Stage 1b build summary ===")
    for r in per_singer:
        print(f"  spk {r['spk_id']:>2} singer {r['singer']:>2}: "
              f"train {r['n_slices_train']:>4} slices / {r['train_min']:>6.1f} min | "
              f"val {r['n_slices_val']:>3} slices / {r['val_min']:>5.1f} min | "
              f"rej short={r['slices_short_discarded']} qual={r['slices_quality_removed']}")
    print(f"  TOTAL: train {totals['train_slices']} slices / {totals['train_min']} min, "
          f"val {totals['val_slices']} slices / {totals['val_min']} min")


if __name__ == "__main__":
    main()
