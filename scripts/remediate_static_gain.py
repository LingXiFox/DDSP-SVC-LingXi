#!/usr/bin/env python3
"""A+ remediation: re-normalize the Dynamic-fallback sources with explicit
static gain and rebuild their slices (user decision 2026-09-27).

Background: the original Stage 1b build normalized with ffmpeg two-pass
loudnorm (linear=true). 50 of 6041 files fell back to Dynamic
normalization (35 lra_above_target = real dynamic compression, 15
lra_zero_sentinel = benign on disk but an ffmpeg fallback quirk).
verify_loudnorm.py v1 verdict: FAIL
(reports/timbre_blend_stage1_loudnorm_verification.json). User decision
A+: remediate ALL 50 with the new production normalization (measure-only
+ explicit static volume gain); keep the 5991 verified-Linear files
untouched (no full rebuild); preserve the original train/val source
split (no re-randomization).

Steps (per Dynamic source; the list is read from the v1 verification
jsonl, but ALL measurements are taken fresh - nothing is trusted):
  1. fresh loudnorm pass-1 measurement of the source (deterministic)
  2. gain = min(TARGET_I - measured_I, TARGET_TP - measured_TP)
     (builder compute_static_gain); applied via ffmpeg volume=<gain>dB
     (builder apply_static_gain) -> overwrites normalized/<dir>/<stem>.wav
     (tmp+rename). No compressor / limiter / dynamic loudnorm anywhere.
  3. post-write disk re-measurement (user directive: confirm on disk):
     tp_norm <= TARGET_TP + 0.3 dBTP AND |i_norm - (i_src + gain)| <= 1.0
     LU, else the file counts as remediation FAILED (listed, exit 1).
  4. delete the old slices <stem>__s*.wav from the source's ORIGINAL
     split (manifest.json per_singer train/val lists), re-slice the new
     normalized file with the same fixed Slicer parameters (builder
     slice_normalized).
  5. purge stale slice_metrics.jsonl entries for those stems (new slices
     can reuse identical names), re-run the GPU slice QC (builder
     recheck_slices = stage-1a analyze_file + the same fixed thresholds);
     failures removed + logged to rejected.jsonl (tagged remediation).
  6. write reports/timbre_blend_stage1_remediation.json: per-file
     before/after evidence, slice count/duration deltas, and final
     per-singer rows + tree totals recomputed FROM DISK. Its per_singer /
     totals supersede the historical build report for downstream tools
     (audit_preprocess_features.py --build-report accepts it directly).

Exit 0 only if every Dynamic file was remediated, all post-write checks
passed, and the slice rebuild completed.

Usage (repo root, remote WSL2):
  PYTHONPATH=. .venv/bin/python scripts/remediate_static_gain.py \
      [--dry-run] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slicer import Slicer
from scripts.build_stage1_dataset import (
    MIN_SLICE_SEC,
    SAMPLE_RATE,
    SLICER_KWARGS,
    TARGET_I,
    TARGET_TP,
    apply_static_gain,
    compute_static_gain,
    measure_loudness,
    recheck_slices,
    slice_normalized,
)
from scripts.select_opensinger import F0_MAX, F0_MIN, HOP_SIZE

# post-write disk checks, identical margins to the verifier v2 criteria
TP_POST_MARGIN = 0.30   # dBTP above TARGET_TP
DI_POST_MAX = 1.00      # LUFS


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="data/timbre_blend_stage1")
    ap.add_argument("--v1-jsonl", default=None,
                    help="default: <out_root>/loudnorm_verification.jsonl")
    ap.add_argument("--speakers", default="reports/timbre_blend_speakers.json")
    ap.add_argument("--report",
                    default="reports/timbre_blend_stage1_remediation.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: fresh measurements + gains, no disk changes")
    return ap.parse_args()


def slice_re(stem):
    """Exact matcher for slices produced from a source stem (<stem>__s<k>.wav)."""
    return re.compile(re.escape(stem) + r"__s\d+\.wav\Z")


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    src_root = Path(manifest["src_root"])
    speakers = json.loads(Path(args.speakers).read_text(encoding="utf-8"))
    dir_by_singer = {m["opensinger_singer"]: m["dir_name"]
                     for m in speakers["mapping"]}
    spk_by_singer = {m["opensinger_singer"]: m["spk_id"]
                     for m in speakers["mapping"]}

    # ---- Dynamic list from the v1 verification jsonl ----
    v1_path = (Path(args.v1_jsonl) if args.v1_jsonl
               else out_root / "loudnorm_verification.jsonl")
    v1 = [json.loads(l)
          for l in v1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    v1_by_rel = {r["rel"]: r for r in v1}
    dyn_rels = sorted(r["rel"] for r in v1
                      if (r.get("type") or "") and r["type"].lower() != "linear")
    if not dyn_rels:
        sys.exit(f"[x] no Dynamic records found in {v1_path}")
    print(f"[*] v1 records: {len(v1)} | Dynamic to remediate: {len(dyn_rels)}",
          flush=True)

    # ---- original split assignment (manifest = build-time plan; never re-randomized) ----
    split_of, singer_of = {}, {}
    for singer_s, blk in manifest["per_singer"].items():
        singer = int(singer_s)
        for rel in blk["train"]:
            split_of[rel], singer_of[rel] = "train", singer
        for rel in blk["val"]:
            split_of[rel], singer_of[rel] = "val", singer
    absent = [r for r in dyn_rels if r not in split_of]
    if absent:
        sys.exit(f"[x] {len(absent)} Dynamic rels absent from manifest splits, "
                 f"e.g. {absent[:3]}")

    # ---- plan: fresh measurements + gains (deterministic; nothing trusted) ----
    plan = []
    for rel in dyn_rels:
        singer = singer_of[rel]
        dir_name = dir_by_singer[singer]
        stem = Path(rel).stem
        src = src_root / rel
        norm = out_root / "normalized" / dir_name / (stem + ".wav")
        audio_dir = out_root / split_of[rel] / "audio" / dir_name
        if not src.exists():
            sys.exit(f"[x] source missing (read-only tree violated?): {src}")
        if not norm.exists():
            sys.exit(f"[x] normalized file missing: {norm}")
        meas = measure_loudness(src)
        if meas is None:
            sys.exit(f"[x] fresh pass-1 measurement failed: {src}")
        gain, tp_limited = compute_static_gain(meas)
        pat = slice_re(stem)
        old_slices = sorted(p for p in audio_dir.glob("*.wav")
                            if pat.fullmatch(p.name))
        old_dur = sum(sf.info(str(p)).duration for p in old_slices)
        plan.append({
            "rel": rel, "singer": singer, "spk_id": spk_by_singer[singer],
            "split": split_of[rel], "dir_name": dir_name, "stem": stem,
            "src": str(src), "norm": str(norm), "audio_dir": str(audio_dir),
            "mechanism_v1": v1_by_rel[rel].get("mechanism"),
            "i_src": float(meas["input_i"]),
            "tp_src": float(meas["input_tp"]),
            "lra_src": float(meas["input_lra"]),
            "gain_db": round(gain, 6), "tp_limited": tp_limited,
            "expected_i": round(float(meas["input_i"]) + gain, 4),
            "old_n_slices": len(old_slices),
            "old_slices_min": round(old_dur / 60.0, 4),
        })
    n_tpl = sum(1 for p in plan if p["tp_limited"])
    gains = [p["gain_db"] for p in plan]
    print(f"[*] plan: {len(plan)} files | gain {min(gains):+.2f} .. "
          f"{max(gains):+.2f} dB | tp_limited: {n_tpl} | old slices: "
          f"{sum(p['old_n_slices'] for p in plan)} "
          f"({sum(p['old_slices_min'] for p in plan):.2f} min)", flush=True)
    if args.dry_run:
        for p in plan:
            print(f"    spk{p['spk_id']:>2} {p['split']:<5} "
                  f"gain={p['gain_db']:+.2f}dB tp_limited={p['tp_limited']} "
                  f"old_slices={p['old_n_slices']} {p['rel']}", flush=True)
        print("[*] dry-run: no changes written", flush=True)
        return 0

    # ---- execute: static gain -> post-write check -> slice swap (per file) ----
    slicer = Slicer(sr=SAMPLE_RATE, **SLICER_KWARGS)
    rej_path = out_root / "rejected.jsonl"

    def log_reject(singer, stage, item, detail):
        with open(rej_path, "a", encoding="utf-8") as rf:
            rf.write(json.dumps({"singer": singer, "stage": stage, "item": item,
                                 "detail": detail}, ensure_ascii=False) + "\n")

    new_slice_paths = []
    for p in plan:
        src, norm = Path(p["src"]), Path(p["norm"])
        audio_dir, stem = Path(p["audio_dir"]), p["stem"]
        if not apply_static_gain(src, norm, p["gain_db"]):
            p["post_ok"] = False
            p["post_failures"] = ["apply_failed"]
            log_reject(p["singer"], "remediation_apply_failed", p["rel"],
                       "ffmpeg volume static gain")
            continue
        meas_norm = measure_loudness(norm)
        pf = []
        if meas_norm is None:
            pf.append("post_measure_failed")
        else:
            p["i_norm"] = float(meas_norm["input_i"])
            p["tp_norm"] = float(meas_norm["input_tp"])
            p["lra_norm"] = float(meas_norm["input_lra"])
            if p["tp_norm"] > TARGET_TP + TP_POST_MARGIN:
                pf.append("post_tp")
            if abs(p["i_norm"] - p["expected_i"]) > DI_POST_MAX:
                pf.append("post_di")
        p["post_ok"] = not pf
        p["post_failures"] = pf
        if pf:
            # normalized file is already swapped; old slices stay in place so
            # the tree remains self-consistent for review. Loud failure.
            log_reject(p["singer"], "remediation_post_check", p["rel"], pf)
            print(f" [!] POST-CHECK FAILED {p['rel']}: {pf}", flush=True)
            continue
        pat = slice_re(stem)
        for q in sorted(x for x in audio_dir.glob("*.wav") if pat.fullmatch(x.name)):
            q.unlink()
        try:
            written, n_short = slice_normalized(norm, audio_dir, stem, slicer)
        except Exception as e:
            p["post_ok"] = False
            p["post_failures"] = [f"slice_error: {type(e).__name__}: {e}"]
            log_reject(p["singer"], "slice_error", p["rel"],
                       f"remediation: {type(e).__name__}: {e}")
            continue
        p["new_n_written"] = len(written)
        p["new_n_short"] = n_short
        if n_short:
            log_reject(p["singer"], "slice_short", p["rel"],
                       f"{n_short} chunk(s) < {MIN_SLICE_SEC}s discarded "
                       f"(remediation)")
        new_slice_paths.extend(written)
        print(f" [*] spk{p['spk_id']:>2} {p['split']:<5} {p['rel']}: "
              f"gain={p['gain_db']:+.2f}dB i_norm={p['i_norm']:.2f} "
              f"tp_norm={p['tp_norm']:.2f} slices {p['old_n_slices']} -> "
              f"{len(written)} (short {n_short})", flush=True)

    n_post_fail = sum(1 for p in plan if not p.get("post_ok"))

    # ---- purge stale slice-metric cache entries for the remediated stems ----
    cache_path = out_root / "slice_metrics.jsonl"
    prefixes = tuple(f"{p['split']}/audio/{p['dir_name']}/{p['stem']}__s"
                     for p in plan)
    kept, purged = [], 0
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("path", "").startswith(prefixes):
            purged += 1
        else:
            kept.append(line)
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    tmp.rename(cache_path)
    print(f"[*] slice_metrics.jsonl: purged {purged} stale entries, "
          f"kept {len(kept)}", flush=True)

    # ---- GPU slice QC on the new slices (same fixed thresholds as build) ----
    removed = []
    if new_slice_paths:
        from ddsp.vocoder import F0_Extractor  # lazy: pulls torch
        f0x = F0_Extractor("rmvpe", SAMPLE_RATE, HOP_SIZE, F0_MIN, F0_MAX)
        removed = recheck_slices(new_slice_paths, out_root, cache_path, f0x,
                                 args.device)
    prefix_owner = [(f"{p['split']}/audio/{p['dir_name']}/{p['stem']}__s", p)
                    for p in plan]
    for slice_rel, reason, rec in removed:
        owner = next((p for pfx, p in prefix_owner if slice_rel.startswith(pfx)),
                     None)
        log_reject(owner["singer"] if owner else None, "quality", slice_rel,
                   {"failed": reason, "remediation": True,
                    **{k: rec.get(k) for k in
                       ("duration_sec", "voiced_ratio", "snr_db", "clip_ratio",
                        "oct_jumps_per_voiced_min")}})
        if owner is not None:
            owner.setdefault("qc_removed", []).append(
                {"slice": slice_rel, "failed": reason})

    # ---- final per-file census (after QC) ----
    for p in plan:
        audio_dir, pat = Path(p["audio_dir"]), slice_re(p["stem"])
        final = [q for q in audio_dir.glob("*.wav") if pat.fullmatch(q.name)]
        p["final_n_slices"] = len(final)
        p["final_slices_min"] = round(
            sum(sf.info(str(q)).duration for q in final) / 60.0, 4)
        p["qc_removed_n"] = len(p.get("qc_removed", []))
        expect_n = p.get("new_n_written", 0) - p["qc_removed_n"]
        if p.get("post_ok") and p["final_n_slices"] != expect_n:
            p["census_mismatch"] = True
            print(f" [!] census mismatch for {p['rel']}: on disk "
                  f"{p['final_n_slices']} != written {p.get('new_n_written')} "
                  f"- qc_removed {p['qc_removed_n']}", flush=True)

    # ---- final tree census (authoritative counts, from disk) ----
    per_singer = []
    for singer in sorted({int(k) for k in manifest["per_singer"]}):
        blk = manifest["per_singer"][str(singer)]
        dir_name = dir_by_singer[singer]
        row = {"singer": singer, "spk_id": blk["spk_id"], "dir_name": dir_name,
               "n_train_src": len(blk["train"]), "n_val_src": len(blk["val"])}
        for split in ("train", "val"):
            d = out_root / split / "audio" / dir_name
            wavs = sorted(d.glob("*.wav")) if d.is_dir() else []
            dur = sum(sf.info(str(w)).duration for w in wavs)
            row[f"n_slices_{split}"] = len(wavs)
            row[f"{split}_min"] = round(dur / 60.0, 2)
        per_singer.append(row)
    totals = {
        "train_slices": sum(r["n_slices_train"] for r in per_singer),
        "val_slices": sum(r["n_slices_val"] for r in per_singer),
        "train_min": round(sum(r["train_min"] for r in per_singer), 2),
        "val_min": round(sum(r["val_min"] for r in per_singer), 2),
    }

    old_n = sum(p["old_n_slices"] for p in plan)
    old_min = sum(p["old_slices_min"] for p in plan)
    final_n = sum(p.get("final_n_slices", 0) for p in plan)
    final_min = sum(p.get("final_slices_min", 0.0) for p in plan)
    n_mismatch = sum(1 for p in plan if p.get("census_mismatch"))
    report = {
        "decision": "A+ (user, 2026-09-27): remediate ALL Dynamic-fallback "
                    "sources with explicit static gain; no deletion; no "
                    "dynamic processing; original train/val source split "
                    "preserved",
        "method": {
            "measurement": "ffmpeg loudnorm pass 1, fresh per file "
                           "(deterministic; old report values not trusted)",
            "gain": "min(TARGET_I - measured_I, TARGET_TP - measured_TP)",
            "apply": "ffmpeg volume=<gain>dB, 44100 Hz mono pcm_s16le "
                     "(single constant multiplication; compressor / limiter "
                     "/ dynamic loudnorm forbidden)",
            "target_I_lufs": TARGET_I,
            "target_TP_dbtp": TARGET_TP,
            "post_check": f"on disk: tp_norm <= {TARGET_TP + TP_POST_MARGIN} "
                          f"dBTP and |i_norm - (i_src + gain)| <= "
                          f"{DI_POST_MAX} LU",
            "slicing": f"repo Slicer {SLICER_KWARGS}, min_slice_sec "
                       f"{MIN_SLICE_SEC} (identical to build)",
            "qc": "builder recheck_slices (stage-1a analyze_file, GPU rmvpe, "
                  "same fixed thresholds as build)",
        },
        "n_dynamic_input": len(plan),
        "n_remediated_ok": sum(1 for p in plan if p.get("post_ok")),
        "n_post_check_failed": n_post_fail,
        "post_check_failures": [p["rel"] for p in plan if not p.get("post_ok")],
        "n_census_mismatch": n_mismatch,
        "tp_limited": [{"rel": p["rel"], "gain_db": p["gain_db"]}
                       for p in plan if p["tp_limited"]],
        "slices": {
            "old_n": old_n, "old_min": round(old_min, 2),
            "new_written": sum(p.get("new_n_written", 0) for p in plan),
            "new_short_discarded": sum(p.get("new_n_short", 0) for p in plan),
            "qc_removed": len(removed),
            "final_n": final_n, "final_min": round(final_min, 2),
            "delta_n": final_n - old_n,
            "delta_min": round(final_min - old_min, 2),
        },
        "per_file": plan,
        "per_singer": per_singer,
        "totals": totals,
        "supersedes": "reports/timbre_blend_stage1_build.json per_singer/"
                      "totals (historical pre-remediation counts)",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    print("\n=== remediation summary ===", flush=True)
    print(f"files: {len(plan)} | ok: {report['n_remediated_ok']} | "
          f"post-check failed: {n_post_fail} | census mismatch: {n_mismatch}",
          flush=True)
    print(f"tp_limited: {len(report['tp_limited'])}", flush=True)
    print(f"slices: old {old_n} ({old_min:.2f} min) -> written "
          f"{report['slices']['new_written']} (short discarded "
          f"{report['slices']['new_short_discarded']}, QC removed "
          f"{len(removed)}) -> final {final_n} ({final_min:.2f} min)",
          flush=True)
    print(f"tree totals: train {totals['train_slices']} slices / "
          f"{totals['train_min']} min | val {totals['val_slices']} slices / "
          f"{totals['val_min']} min", flush=True)
    print(f"report: {args.report}", flush=True)
    ok = (n_post_fail == 0 and n_mismatch == 0
          and report["n_remediated_ok"] == len(plan))
    print(f"VERDICT: {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
