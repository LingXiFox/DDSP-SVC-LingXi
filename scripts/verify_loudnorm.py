#!/usr/bin/env python3
"""Static-normalization verifier (v2) for the Stage 1b tree.

Contract under test (production scheme, user decision 2026-09-27 A+):
every normalized file must equal its source times ONE constant gain,
    gain = min(TARGET_I - measured_I, TARGET_TP - measured_TP)
(measurements: ffmpeg loudnorm pass 1, deterministic), applied via the
ffmpeg volume filter. Compressor / limiter / dynamic loudnorm are
forbidden anywhere; when the TP constraint binds (tp_limited), missing
the loudness target is allowed and recorded.

History: v1 of this script audited the ORIGINAL build (two-pass loudnorm,
linear=true) by deterministic pass-2 reproduction and found 50/6041
Dynamic fallbacks -> verdict FAIL (report kept at
reports/timbre_blend_stage1_loudnorm_verification.json). Those 50 sources
were remediated with the static-gain scheme
(scripts/remediate_static_gain.py) and the builder switched to it
permanently. v2 verifies the FINAL tree against the static-gain contract;
the pass-2 reproduction is gone because loudnorm pass 2 is no longer part
of production.

Evidence per file:
  Phase A (numpy): frame-energy comparison src vs normalized-on-disk ->
    dproxy (p95-p20 proxy change; exactly 0 under a constant gain),
    frame_gain_db (robust median per-frame gain = the gain actually on
    disk), frame_gain_rstd (residual spread - any dynamic processing
    shows up here), len_diff_samples.
  Phase B (ffmpeg loudnorm pass 1, MEASUREMENT ONLY): fresh i/tp/lra of
    the source and of the normalized file; the expected gain is recomputed
    from the fresh source measurement (nothing is trusted from reports).

Pre-registered acceptance criteria (fixed 2026-09-27 BEFORE any v2 run;
see docs/TIMBRE_BLEND_LOG.md - do not tune post hoc). A file passes only
if ALL hold:
  1. frame_gain_rstd <= 0.05 dB       (gain constant across the file =>
                                       no dynamic compression anywhere)
  2. |dproxy| <= 0.30 dB              (dynamics proxy preserved)
  3. |lra_norm - lra_src| <= 0.50 LU  (loudness range preserved)
  4. |i_norm - (i_src + expected_gain)| <= 1.00 LUFS (achieved loudness
     matches the static-gain prediction; the tolerance absorbs loudnorm's
     known ~0.3-0.7 LU measurement jitter for the legacy short-file-rule
     population - v1 full-run evidence: Linear files di p99 = 0.14)
  5. tp_norm <= TARGET_TP + 0.30 dBTP (true-peak ceiling honored, 0.3 dB
     measurement margin)
Tree level: pool == on-disk == verified, 0 orphans, 0 missing.
Any failing file -> listed in full in the JSON report, exit code 1.

Usage (repo root, remote WSL2):
  PYTHONPATH=. .venv/bin/python scripts/verify_loudnorm.py \
      --out-root data/timbre_blend_stage1 --workers 8 \
      [--limit N] [--rels-file rels.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.build_stage1_dataset import (
    TARGET_I,
    TARGET_TP,
    compute_static_gain,
    measure_loudness,
)
from scripts.select_opensinger import frame_energy_db

# ---- pre-registered thresholds (see module docstring) ----
RSTD_MAX = 0.05     # dB, frame-gain residual spread (constant-gain proof)
DPROXY_MAX = 0.30   # dB
DLRA_MAX = 0.50     # LU
DI_MAX = 1.00       # LUFS
TP_MARGIN = 0.30    # dBTP measurement margin above the TARGET_TP ceiling


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
    # robust per-frame gain on active frames (within 50 dB of src peak)
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
    rstd = rec.get("frame_gain_rstd")
    if rstd is None:
        failures.append("rstd_unmeasurable")
    elif rstd > RSTD_MAX:
        failures.append("rstd")
    if abs(rec["dproxy"]) > DPROXY_MAX:
        failures.append("dproxy")

    meas_src = measure_loudness(src)
    meas_norm = measure_loudness(norm)
    if meas_src is None or meas_norm is None:
        failures.append("remeasure_failed")
        rec["failures"] = failures
        return rec

    i_src = float(meas_src["input_i"])
    tp_src = float(meas_src["input_tp"])
    lra_src = float(meas_src["input_lra"])
    i_norm = float(meas_norm["input_i"])
    tp_norm = float(meas_norm["input_tp"])
    lra_norm = float(meas_norm["input_lra"])
    expected_gain, tp_limited = compute_static_gain(meas_src)
    expected_i = i_src + expected_gain
    gain_err = (rec["frame_gain_db"] - expected_gain
                if rec.get("frame_gain_db") is not None else None)
    rec.update({
        "input_i_src": i_src,
        "input_tp_src": tp_src,
        "input_lra_src": lra_src,
        "input_i_norm": i_norm,
        "input_tp_norm": tp_norm,
        "input_lra_norm": lra_norm,
        "expected_gain_db": round(expected_gain, 4),
        "tp_limited": tp_limited,
        "expected_i": round(expected_i, 4),
        "di_err": round(i_norm - expected_i, 4),
        "dlra": round(lra_norm - lra_src, 4),
        "tp_err": round(tp_norm - (tp_src + expected_gain), 4),
        "gain_err": round(gain_err, 4) if gain_err is not None else None,
    })
    if abs(rec["dlra"]) > DLRA_MAX:
        failures.append("dlra")
    if abs(rec["di_err"]) > DI_MAX:
        failures.append("di")
    if tp_norm > TARGET_TP + TP_MARGIN:
        failures.append("tp_ceiling")
    rec["failures"] = failures
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="data/timbre_blend_stage1")
    ap.add_argument("--speakers", default="reports/timbre_blend_speakers.json")
    ap.add_argument("--jsonl", default=None,
                    help="default: <out_root>/static_norm_verification.jsonl")
    ap.add_argument("--report",
                    default="reports/timbre_blend_stage1_static_norm_verification.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="verify only the first N pool files (testing)")
    ap.add_argument("--rels-file", default=None,
                    help="restrict to the rels listed in this file, one per "
                         "line (testing; tree integrity still checks the "
                         "full pool)")
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

    # full-tree consistency (independent of --limit / --rels-file)
    expected = {out_root / "normalized" / d / (Path(r).stem + ".wav")
                for _, _, d, r in items}
    on_disk = set((out_root / "normalized").rglob("*.wav"))
    orphans = sorted(str(p.relative_to(out_root)) for p in on_disk - expected)
    missing = sorted(str(p.relative_to(out_root)) for p in expected - on_disk)

    if args.rels_file:
        want = {l.strip() for l in Path(args.rels_file).read_text().splitlines()
                if l.strip()}
        items = [it for it in items if it[3] in want]
        n_want = len(want)
        if len(items) != n_want:
            print(f"[!] rels-file: {n_want} rels listed, {len(items)} matched "
                  f"in the pool", flush=True)
    if args.limit > 0:
        items = items[: args.limit]
    partial = bool(args.limit or args.rels_file)

    jsonl_path = (Path(args.jsonl) if args.jsonl
                  else out_root / "static_norm_verification.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] pool files: {n_pool} | verifying: {len(items)} "
          f"| workers: {args.workers} | partial: {partial}", flush=True)
    print(f"[*] normalized on disk: {len(on_disk)} | orphans: {len(orphans)} "
          f"| missing: {len(missing)}", flush=True)
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
                print(f"[*] {k}/{len(items)} ({el:.0f}s, "
                      f"{k / max(el, 1e-9):.1f} f/s)", flush=True)

    # ---- aggregate ----
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

    def _maxval(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return round(max(vals), 4) if vals else None

    tpl = [r for r in recs if r.get("tp_limited")]
    ok_tree = (not missing) and (not orphans) and n_pool == len(on_disk)
    verdict_pass = ((not anomalies) and ok_tree and len(recs) == len(items)
                    and not partial)

    report = {
        "verifier": "v2 static-normalization (measure + explicit static gain "
                    "contract; loudnorm pass 2 no longer part of production)",
        "out_root": str(out_root),
        "src_root": str(src_root),
        "n_pool": n_pool,
        "n_verified": len(recs),
        "partial": partial,
        "normalized_on_disk": len(on_disk),
        "orphans": orphans[:50],
        "n_orphans": len(orphans),
        "missing": missing[:50],
        "n_missing": len(missing),
        "thresholds": {
            "rstd_max_db": RSTD_MAX,
            "dproxy_max_db": DPROXY_MAX,
            "dlra_max_lu": DLRA_MAX,
            "di_max_lufs": DI_MAX,
            "tp_ceiling_dbtp": round(TARGET_TP + TP_MARGIN, 4),
        },
        "fail_reason_counts": dict(fail_reasons),
        "n_anomalies": len(anomalies),
        "anomalies": anomalies[:200],
        "stats": {
            "dproxy": _stats("dproxy"),
            "dlra": _stats("dlra"),
            "di_err": _stats("di_err"),
            "tp_err": _stats("tp_err"),
            "gain_err": _stats("gain_err"),
            "frame_gain_db": _stats("frame_gain_db"),
            "frame_gain_rstd": _stats("frame_gain_rstd"),
            "frame_gain_p99dev": _stats("frame_gain_p99dev"),
            "len_diff_samples": _stats("len_diff_samples"),
        },
        "input_tp_norm_max": _maxval(recs, "input_tp_norm"),
        "tp_limited": {
            "n": len(tpl),
            "rels": [r["rel"] for r in tpl][:50],
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "verdict": "PASS" if verdict_pass else "FAIL",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print("\n=== static-normalization verification summary ===", flush=True)
    print(f"files verified: {len(recs)} / pool {n_pool} | on-disk {len(on_disk)}"
          f" | orphans {len(orphans)} | missing {len(missing)}"
          f" | partial: {partial}", flush=True)
    print(f"fail reasons: {dict(fail_reasons) if fail_reasons else 'none'}",
          flush=True)
    for key in ("dproxy", "dlra", "di_err", "tp_err", "gain_err",
                "frame_gain_rstd"):
        st = report["stats"][key]
        if st:
            print(f"{key}: median={st['median']} p99_abs={st['p99_abs']} "
                  f"max_abs={st['max_abs']}", flush=True)
    print(f"input_tp_norm max: {report['input_tp_norm_max']} dBTP "
          f"(ceiling {TARGET_TP}, criterion <= {TARGET_TP + TP_MARGIN})",
          flush=True)
    print(f"tp_limited files: {len(tpl)}", flush=True)
    if anomalies:
        print(f"\nANOMALIES ({len(anomalies)}), first 50:", flush=True)
        for r in anomalies[:50]:
            print(f"  spk{r['spk_id']:>2} {r['rel']} -> {r['failures']}"
                  f" rstd={r.get('frame_gain_rstd')} dproxy={r.get('dproxy')}"
                  f" dlra={r.get('dlra')} di_err={r.get('di_err')}"
                  f" tp_norm={r.get('input_tp_norm')}"
                  f" gain_err={r.get('gain_err')}", flush=True)
    print(f"\nreport: {args.report}\njsonl:  {jsonl_path}", flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
