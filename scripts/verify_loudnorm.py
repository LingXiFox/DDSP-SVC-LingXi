#!/usr/bin/env python3
"""Verify Stage 1b loudnorm actually applied static (linear) gain to every file.

Background
----------
The Stage 1b builder requested ffmpeg two-pass loudnorm with linear=true,
but discarded pass-2 stderr, so the reported "Normalization Type" was not
recorded during the build. loudnorm is deterministic: re-running the same
two passes on the same source reproduces the exact normalization decision.
This script provides the acceptance evidence for the full build (user
directive 2026-09-27):

Phase A (numpy, no ffmpeg): dynamics-preservation check comparing the
  source file with the normalized file actually on disk. The frame-energy
  spread proxy (p95 - p20 of 25 ms / 10 ms frame RMS dB, same utility as
  stage-1a) is exactly invariant under a pure static gain, and shrinks
  under dynamic compression. Also records the robust per-frame gain
  (median of frame dB difference on active frames) and its residual
  spread as supplementary evidence.

Phase B (ffmpeg, deterministic reproduction): for every pool source file,
  re-run pass 1 (measure) and pass 2 to null with the measured parameters
  and linear=true, and parse "Normalization Type:" from the pass-2
  summary (observed value for a successful linear pass: "Linear"). Also
  re-measure the normalized file on disk (pass 1) to obtain the achieved
  integrated loudness and LRA.

Pre-registered acceptance criteria (fixed 2026-09-27 BEFORE the full
verification run; see docs/TIMBRE_BLEND_LOG.md — do not tune post hoc).
A file passes only if ALL hold:
  1. reproduced pass-2 normalization type == "linear" (case-insensitive)
  2. |proxy(norm) - proxy(src)| <= 0.30 dB           (dynamics preserved)
  3. |input_lra(norm) - input_lra(src)| <= 0.50 LU   (loudness range preserved)
  4. |input_i(norm) - expected_i| <= 1.00 LUFS, where expected_i follows
     af_loudnorm.c (ffmpeg 8.0.1): files < 3 s are forced into LINEAR_MODE
     with a TP-capped gain (expected_i = min(I_target, i_src + TP - tp_src));
     otherwise expected_i = I_target (init()-linear applies gain
     I_target - measured_I and DISCARDS the user offset parameter; the
     dynamic feedback loop also targets I_target). The 1.0 LU tolerance
     covers loudnorm's known internal measurement jitter (~0.3-0.7 LU
     between its processing pass and an independent re-measurement,
     observed on probe files).

Mechanism classification (verified against ffmpeg 8.0.1 source AND
controlled experiments on smoke files, 2026-09-27):
  - short_file_rule:   whole file < 3 s -> filter_frame() forces LINEAR_MODE
                       regardless of measured_* sentinels
  - init_linear:       >= 3 s, init() grants linear (sentinels non-zero,
                       TP headroom OK, LRA <= target)
  - lra_zero_sentinel: >= 3 s with measured LRA == 0.00 -> collides with the
                       "not provided" sentinel (init() requires
                       measured_lra != 0) -> Dynamic fallback even though
                       nothing is wrong with the audio. Confirmed: the same
                       file fed with measured_LRA=2.5 reproduces as Linear.
  - tp_constraint:     the linear gain would push true peak above TP ceiling
  - thresh_sentinel / lra_above_target: other init() rejections
Any failing file -> listed in full in the JSON report and exit code 1.
Per user directive: if any dynamic fallback exists, STOP and report
before entering preprocess; the report carries per-file mechanism +
on-disk effect evidence (was the applied gain effectively constant
despite the Dynamic label?) for the decision.

Usage (repo root, remote WSL2):
  PYTHONPATH=. .venv/bin/python scripts/verify_loudnorm.py \
      --out-root data/timbre_blend_stage1 --workers 8 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.build_stage1_dataset import (
    LOUDNORM_I,
    LOUDNORM_LRA,
    LOUDNORM_TP,
    measure_loudness,
)
from scripts.select_opensinger import frame_energy_db

# ---- pre-registered thresholds (see module docstring) ----
DPROXY_MAX = 0.30   # dB
DLRA_MAX = 0.50     # LU
DI_MAX = 1.00       # LUFS
TYPE_EXPECTED = "linear"


def _proxy(db: np.ndarray) -> float:
    return float(np.percentile(db, 95) - np.percentile(db, 20))


def phase_a(src_path: Path, norm_path: Path) -> dict:
    """Numpy-only comparison of the file on disk against its source."""
    xs, sr_s = sf.read(str(src_path), dtype="float32")
    xn, sr_n = sf.read(str(norm_path), dtype="float32")
    if xs.ndim > 1:
        xs = xs.mean(axis=1)
    if xn.ndim > 1:
        xn = xn.mean(axis=1)
    db_s = frame_energy_db(xs, sr_s)
    db_n = frame_energy_db(xn, sr_n)
    proxy_s = _proxy(db_s)
    proxy_n = _proxy(db_n)
    # robust per-frame gain on active frames (within 50 dB of src peak);
    # supplementary evidence only — not an acceptance criterion, because
    # a constant filter delay would inflate the residual without any
    # dynamics change.
    m = min(len(db_s), len(db_n))
    act = db_s[:m] > (db_s[:m].max() - 50.0)
    if int(act.sum()) >= 10:
        diff = db_n[:m][act] - db_s[:m][act]
        gain = float(np.median(diff))
        dev = np.abs(diff - gain)
        rstd = float(1.4826 * np.median(dev))
        p99dev = float(np.percentile(dev, 99))
    else:
        gain, rstd, p99dev = float("nan"), float("nan"), float("nan")
    return {
        "dur_sec": round(len(xs) / sr_s, 3),
        "proxy_src": round(proxy_s, 4),
        "proxy_norm": round(proxy_n, 4),
        "dproxy": round(proxy_n - proxy_s, 4),
        "frame_gain_db": round(gain, 4) if gain == gain else None,
        "frame_gain_rstd": round(rstd, 4) if rstd == rstd else None,
        "frame_gain_p99dev": round(p99dev, 4) if p99dev == p99dev else None,
        "len_diff_samples": abs(len(xs) - len(xn)),
    }


def reproduce_pass2(src_path: Path, meas: dict) -> tuple:
    """Deterministic pass-2 rerun to null; returns (type, output_i, output_tp)."""
    mi = meas["input_i"]
    mtp = meas["input_tp"]
    mlra = meas["input_lra"]
    mth = meas["input_thresh"]
    off = meas["target_offset"]
    af = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={mi}:measured_TP={mtp}:measured_LRA={mlra}"
        f":measured_thresh={mth}:offset={off}:linear=true:print_format=summary"
    )
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src_path), "-af", af,
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        return None, None
    mt = re.search(r"Normalization Type:\s*(\S+)", p.stderr)
    mo = re.search(r"Output Integrated:\s*(-?[\d.]+)", p.stderr)
    mp = re.search(r"Output True Peak:\s*(-?[\d.]+)", p.stderr)
    return (
        mt.group(1) if mt else None,
        float(mo.group(1)) if mo else None,
        float(mp.group(1)) if mp else None,
    )


def verify_one(item: tuple, src_root: Path, out_root: Path) -> dict:
    singer, spk_id, dir_name, rel = item
    src = src_root / rel
    norm = out_root / "normalized" / dir_name / (Path(rel).stem + ".wav")
    rec = {
        "singer": singer,
        "spk_id": spk_id,
        "rel": rel,
        "norm_rel": str(norm.relative_to(out_root)),
    }
    failures = []
    if not src.exists():
        rec["failures"] = ["src_missing"]
        return rec
    if not norm.exists():
        rec["failures"] = ["norm_missing"]
        return rec

    rec.update(phase_a(src, norm))
    if abs(rec["dproxy"]) > DPROXY_MAX:
        failures.append("dproxy")

    meas_src = measure_loudness(src)
    meas_norm = measure_loudness(norm)
    if meas_src is None or meas_norm is None:
        failures.append("remeasure_failed")
        rec["failures"] = failures
        return rec

    ntype, out_i, out_tp = reproduce_pass2(src, meas_src)
    off = float(meas_src["target_offset"])
    i_src = float(meas_src["input_i"])
    i_norm = float(meas_norm["input_i"])
    lra_src = float(meas_src["input_lra"])
    lra_norm = float(meas_norm["input_lra"])
    tp_src = float(meas_src["input_tp"])
    tp_norm = float(meas_norm["input_tp"])
    thresh_src = float(meas_src["input_thresh"])
    dur = rec.get("dur_sec") or 0.0
    short_file = dur < 3.0
    lra_zero = abs(lra_src) < 1e-9
    # expected achieved loudness per af_loudnorm.c (ffmpeg 8.0.1), see docstring
    gain_tp_cap = LOUDNORM_TP - tp_src
    expected_i = min(LOUDNORM_I, i_src + gain_tp_cap) if short_file else LOUDNORM_I
    gain_lin = LOUDNORM_I - i_src  # gain init()-linear applies (offset discarded)
    tp_after_linear = tp_src + gain_lin
    tp_would_exceed = bool(tp_after_linear > LOUDNORM_TP)
    lra_above_target = bool(lra_src > LOUDNORM_LRA)
    thresh_sentinel = thresh_src == -70.0
    if ntype is None:
        mechanism = "type_parse_failed"
    elif ntype.lower() == TYPE_EXPECTED:
        mechanism = "short_file_rule" if short_file else "init_linear"
    elif lra_zero:
        mechanism = "lra_zero_sentinel"
    elif thresh_sentinel:
        mechanism = "thresh_sentinel"
    elif tp_would_exceed:
        mechanism = "tp_constraint"
    elif lra_above_target:
        mechanism = "lra_above_target"
    else:
        mechanism = "unexplained"
    rec.update({
        "type": ntype,
        "mechanism": mechanism,
        "pass2_output_i": out_i,
        "pass2_output_tp": out_tp,
        "target_offset": off,
        "dur_sec": dur,
        "short_file": short_file,
        "lra_zero": lra_zero,
        "input_i_src": i_src,
        "input_i_norm": i_norm,
        "input_lra_src": lra_src,
        "input_lra_norm": lra_norm,
        "input_tp_src": tp_src,
        "input_tp_norm": tp_norm,
        "expected_i": round(expected_i, 4),
        "tp_after_linear_gain": round(tp_after_linear, 4),
        "tp_would_exceed": tp_would_exceed,
        "di_err": round(i_norm - expected_i, 4),
        "dlra": round(lra_norm - lra_src, 4),
        "gain_applied_db": round(i_norm - i_src, 4),
    })
    if ntype is None:
        failures.append("type_parse_failed")
    elif ntype.lower() != TYPE_EXPECTED:
        failures.append("type=" + ntype)
    if abs(lra_norm - lra_src) > DLRA_MAX:
        failures.append("dlra")
    if abs(i_norm - expected_i) > DI_MAX:
        failures.append("di")
    rec["failures"] = failures
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="data/timbre_blend_stage1")
    ap.add_argument("--speakers", default="reports/timbre_blend_speakers.json")
    ap.add_argument("--jsonl", default=None,
                    help="default: <out_root>/loudnorm_verification.jsonl")
    ap.add_argument("--report",
                    default="reports/timbre_blend_stage1_loudnorm_verification.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="verify only the first N pool files (testing)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    manifest = json.loads((out_root / "manifest.json").read_text())
    src_root = Path(manifest["src_root"])
    speakers = json.loads(Path(args.speakers).read_text())
    dir_by_singer = {m["opensinger_singer"]: m["dir_name"]
                     for m in speakers["mapping"]}

    items = []
    for singer_s, blk in manifest["per_singer"].items():
        singer = int(singer_s)
        dir_name = dir_by_singer[singer]
        for rel in blk["pool"]:
            items.append((singer, blk["spk_id"], dir_name, rel))
    items.sort(key=lambda t: (t[0], t[3]))
    n_pool = len(items)

    # full-tree consistency (independent of --limit)
    expected = {out_root / "normalized" / d / (Path(r).stem + ".wav")
                for _, _, d, r in items}
    on_disk = set((out_root / "normalized").rglob("*.wav"))
    orphans = sorted(str(p.relative_to(out_root)) for p in on_disk - expected)
    missing = sorted(str(p.relative_to(out_root)) for p in expected - on_disk)

    if args.limit > 0:
        items = items[: args.limit]

    jsonl_path = Path(args.jsonl) if args.jsonl else out_root / "loudnorm_verification.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] pool files: {n_pool} | verifying: {len(items)} | workers: {args.workers}",
          flush=True)
    print(f"[*] normalized on disk: {len(on_disk)} | orphans: {len(orphans)} | missing: {len(missing)}",
          flush=True)
    if missing:
        for p in missing[:20]:
            print(f"    MISSING {p}", flush=True)

    t0 = time.time()
    recs = []
    with open(jsonl_path, "w") as fj, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify_one, it, src_root, out_root): it for it in items}
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            recs.append(rec)
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 500 == 0 or k == len(items):
                el = time.time() - t0
                print(f"[*] {k}/{len(items)} ({el:.0f}s, {k / max(el, 1e-9):.1f} f/s)",
                      flush=True)

    # ---- aggregate ----
    type_counts = Counter(r.get("type") for r in recs)
    fail_reasons = Counter(f for r in recs for f in r.get("failures", []))
    anomalies = [r for r in recs if r.get("failures")]
    anomalies.sort(key=lambda r: (r["spk_id"], r["rel"]))

    def _stats(key):
        vals = np.array([r[key] for r in recs
                         if isinstance(r.get(key), (int, float))], dtype=float)
        if len(vals) == 0:
            return None
        av = np.abs(vals)
        return {
            "n": int(len(vals)),
            "median": round(float(np.median(vals)), 4),
            "p99_abs": round(float(np.percentile(av, 99)), 4),
            "max_abs": round(float(av.max()), 4),
        }

    typed = [r for r in recs if r.get("type")]
    dyn = [r for r in typed if r["type"].lower() != TYPE_EXPECTED]
    lin = [r for r in typed if r["type"].lower() == TYPE_EXPECTED]

    def _maxabs(rs, key):
        vals = [abs(r[key]) for r in rs
                if isinstance(r.get(key), (int, float))]
        return round(max(vals), 4) if vals else None

    def _maxval(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return round(max(vals), 4) if vals else None

    mech_counts = Counter(r.get("mechanism") for r in recs)
    dynamic_analysis = {
        "n_dynamic": len(dyn),
        "mechanism_counts": dict(mech_counts),
        "dynamic_tp_would_exceed": sum(1 for r in dyn if r.get("tp_would_exceed")),
        "dynamic_lra_zero": sum(1 for r in dyn if r.get("lra_zero")),
        "dynamic_short_file": sum(1 for r in dyn if r.get("short_file")),
        "n_linear": len(lin),
        "linear_short_file": sum(1 for r in lin if r.get("short_file")),
        "linear_tp_would_exceed": sum(1 for r in lin if r.get("tp_would_exceed")),
        "dynamic_on_disk_evidence": {
            "dproxy_max_abs": _maxabs(dyn, "dproxy"),
            "frame_gain_rstd_max": _maxabs(dyn, "frame_gain_rstd"),
            "dlra_max_abs": _maxabs(dyn, "dlra"),
            "input_tp_norm_max": _maxval(dyn, "input_tp_norm"),
            "n_effectively_const_gain": sum(
                1 for r in dyn
                if isinstance(r.get("dproxy"), (int, float))
                and abs(r["dproxy"]) <= DPROXY_MAX
                and isinstance(r.get("frame_gain_rstd"), (int, float))
                and r["frame_gain_rstd"] <= 0.05),
        },
    }

    ok_tree = (not missing) and (not orphans) and n_pool == len(on_disk)
    verdict_pass = (not anomalies) and ok_tree and len(recs) == len(items)

    report = {
        "out_root": str(out_root),
        "src_root": str(src_root),
        "n_pool": n_pool,
        "n_verified": len(recs),
        "limit": args.limit,
        "normalized_on_disk": len(on_disk),
        "orphans": orphans[:50],
        "n_orphans": len(orphans),
        "missing": missing[:50],
        "n_missing": len(missing),
        "thresholds": {
            "type_expected": TYPE_EXPECTED,
            "dproxy_max_db": DPROXY_MAX,
            "dlra_max_lu": DLRA_MAX,
            "di_max_lufs": DI_MAX,
        },
        "type_counts": dict(type_counts),
        "fail_reason_counts": dict(fail_reasons),
        "n_anomalies": len(anomalies),
        "anomalies": anomalies[:200],
        "stats": {
            "dproxy": _stats("dproxy"),
            "dlra": _stats("dlra"),
            "di_err": _stats("di_err"),
            "gain_applied_db": _stats("gain_applied_db"),
            "frame_gain_rstd": _stats("frame_gain_rstd"),
            "frame_gain_p99dev": _stats("frame_gain_p99dev"),
            "len_diff_samples": _stats("len_diff_samples"),
        },
        "dynamic_analysis": dynamic_analysis,
        "elapsed_sec": round(time.time() - t0, 1),
        "verdict": "PASS" if verdict_pass else "FAIL",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print("\n=== loudnorm verification summary ===", flush=True)
    print(f"files verified: {len(recs)} / pool {n_pool} | on-disk {len(on_disk)}"
          f" | orphans {len(orphans)} | missing {len(missing)}", flush=True)
    print(f"pass-2 reproduced type counts: {dict(type_counts)}", flush=True)
    print(f"fail reasons: {dict(fail_reasons) if fail_reasons else 'none'}", flush=True)
    for key in ("dproxy", "dlra", "di_err", "frame_gain_rstd"):
        st = report["stats"][key]
        if st:
            print(f"{key}: median={st['median']} p99_abs={st['p99_abs']} max_abs={st['max_abs']}",
                  flush=True)
    da = dynamic_analysis
    print(f"dynamic analysis: n_linear={da['n_linear']} (short_file_rule={da['linear_short_file']}) "
          f"n_dynamic={da['n_dynamic']} | mechanisms: {da['mechanism_counts']}", flush=True)
    if da["n_dynamic"]:
        ev = da["dynamic_on_disk_evidence"]
        print(f"  dynamic files on-disk effect: dproxy_max={ev['dproxy_max_abs']} "
              f"rstd_max={ev['frame_gain_rstd_max']} dlra_max={ev['dlra_max_abs']} "
              f"tp_norm_max={ev['input_tp_norm_max']} "
              f"effectively_const_gain={ev['n_effectively_const_gain']}/{da['n_dynamic']}",
              flush=True)
    if anomalies:
        print(f"\nANOMALIES ({len(anomalies)}), first 50:", flush=True)
        for r in anomalies[:50]:
            print(f"  spk{r['spk_id']:>2} {r['rel']} -> {r['failures']}"
                  f" mech={r.get('mechanism')} dur={r.get('dur_sec')}"
                  f" dproxy={r.get('dproxy')} rstd={r.get('frame_gain_rstd')}"
                  f" dlra={r.get('dlra')} tp_src={r.get('input_tp_src')}"
                  f" tp_norm={r.get('input_tp_norm')}", flush=True)
    print(f"\nreport: {args.report}\njsonl:  {jsonl_path}", flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
