#!/usr/bin/env python3
"""Per-file feature completeness audit for a preprocessed DDSP-SVC tree.

preprocess.py swallows worker exceptions (prints "[Error] ..." and
continues), so its exit code 0 does NOT prove every audio file received
its features. User directive 2026-09-27: after preprocess, audit EVERY
audio file under {train,val}/audio/ for:

  - all 6 feature npys exist and are fully readable (np.load without
    mmap — catches truncated/corrupt files): units, f0, volume, mel,
    aug_mel, aug_vol at <root>/<feat>/<name_ext>.npy
    (reflow/data_loaders.py convention)
  - shape sanity: f0/volume/aug_vol 1-D; mel/aug_mel/units 2-D with
    axis 0 = frames (loader uses get_npy_shape(...)[0] and
    frame_len = min over the six features)
  - frame spread across the six features <= 2 frames (the loader
    tolerates off-by-one via min(); a larger spread indicates
    corruption -> FAIL)
  - pitch_aug_dict.npy exists, is a dict, and has a numeric entry for
    EVERY audio rel (loader does dict[name_ext] -> a missing key
    crashes training); train values within [-5, 5], val values all 0
  - skip/ directory empty (preprocess moves F0-failed audio there; the
    Stage 1b builder pre-verified every slice with RMVPE, so any moved
    file is an anomaly -> FAIL, listed in full)
  - orphan feature npys without audio (WARN, listed)
  - optional: audio count == expected slice count from the Stage 1b
    build report (--build-report); mismatch -> FAIL

Verdict PASS only with zero FAIL-level issues. Exit code 0 = PASS.

File traversal mirrors logger.utils.traverse_dir(is_pure=True,
is_ext=True): sorted relative posix paths with extension.

Usage (repo root, remote WSL2):
  PYTHONPATH=. .venv/bin/python scripts/audit_preprocess_features.py \
      --config configs/timbre_blend_stage1.yaml \
      --build-report reports/timbre_blend_stage1_build.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FEATURES = ["units", "f0", "volume", "mel", "aug_mel", "aug_vol"]
FEATS_1D = {"f0", "volume", "aug_vol"}
FEATS_2D = {"units", "mel", "aug_mel"}
FRAME_SPREAD_MAX = 2


def list_audio_rel(audio_root: Path, exts) -> list:
    exts = {e.lower().lstrip(".") for e in exts}
    out = []
    for p in sorted(audio_root.rglob("*")):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            out.append(p.relative_to(audio_root).as_posix())
    return out


def check_one(root: Path, rel: str) -> dict:
    rec = {"rel": rel, "issues": [], "frames": {}, "dtypes": {}}
    for feat in FEATURES:
        p = root / feat / (rel + ".npy")
        if not p.exists():
            rec["issues"].append(f"missing:{feat}")
            continue
        try:
            arr = np.load(str(p))  # full eager read: catches truncation
            shape = arr.shape
            dtype = str(arr.dtype)
            if feat in FEATS_1D:
                if arr.ndim != 1:
                    rec["issues"].append(f"shape:{feat}:{shape}")
                    n_frames = None
                else:
                    n_frames = int(shape[0])
            else:
                if arr.ndim != 2:
                    rec["issues"].append(f"shape:{feat}:{shape}")
                    n_frames = None
                else:
                    n_frames = int(shape[0])
            if n_frames is not None:
                rec["frames"][feat] = n_frames
            rec["dtypes"][feat] = dtype
            if feat == "f0" and arr.ndim == 1 and len(arr) > 0:
                f0 = arr.astype(np.float64)
                rec["f0_med"] = round(float(np.median(f0)), 2)
                rec["f0_min"] = round(float(f0.min()), 2)
                rec["f0_max"] = round(float(f0.max()), 2)
                if not np.all(np.isfinite(f0)):
                    rec["issues"].append("f0_nonfinite")
            del arr
        except Exception as e:  # noqa: BLE001 - any read error = unreadable
            rec["issues"].append(f"unreadable:{feat}:{type(e).__name__}")
    fr = [v for v in rec["frames"].values()]
    if len(fr) == len(FEATURES):
        spread = max(fr) - min(fr)
        rec["frame_spread"] = spread
        if spread > FRAME_SPREAD_MAX:
            rec["issues"].append(f"frame_spread={spread}")
    return rec


def audit_split(name: str, root: Path, exts, workers: int, expect_n=None) -> dict:
    root = Path(root)
    audio_root = root / "audio"
    print(f"\n=== split: {name} ({root}) ===", flush=True)
    if not audio_root.is_dir():
        return {"verdict": "FAIL", "fatal": f"no audio/ dir under {root}"}
    rels = list_audio_rel(audio_root, exts)
    print(f"[*] audio files: {len(rels)}", flush=True)

    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(check_one, root, rel): rel for rel in rels}
        for k, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if k % 1000 == 0 or k == len(rels):
                print(f"[*] {name}: checked {k}/{len(rels)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    recs.sort(key=lambda r: r["rel"])

    missing = Counter(i.split(":", 2)[1] for r in recs for i in r["issues"]
                      if i.startswith("missing:"))
    unreadable = Counter(i.split(":", 2)[1] for r in recs for i in r["issues"]
                         if i.startswith("unreadable:"))
    bad = [r for r in recs if r["issues"]]

    # pitch_aug_dict
    pa_issues = []
    pa = None
    pa_path = root / "pitch_aug_dict.npy"
    if not pa_path.exists():
        pa_issues.append("pitch_aug_dict.npy MISSING")
    else:
        try:
            pa = np.load(str(pa_path), allow_pickle=True).item()
            if not isinstance(pa, dict):
                pa_issues.append(f"pitch_aug_dict is {type(pa).__name__}, not dict")
                pa = None
        except Exception as e:  # noqa: BLE001
            pa_issues.append(f"pitch_aug_dict unreadable: {type(e).__name__}")
    missing_keys, extra_keys, bad_vals = [], [], []
    if pa is not None:
        rel_set = set(rels)
        missing_keys = sorted(rel_set - set(pa.keys()))
        extra_keys = sorted(set(pa.keys()) - rel_set)
        for rel in rels:
            v = pa.get(rel)
            if v is None:
                continue
            if not isinstance(v, (int, float, np.floating, np.integer)):
                bad_vals.append((rel, f"non-numeric:{type(v).__name__}"))
                continue
            v = float(v)
            if name == "val" and v != 0.0:
                bad_vals.append((rel, f"val keyshift={v}"))
            elif not (-5.0 <= v <= 5.0):
                bad_vals.append((rel, f"out of range:{v}"))

    # skip/ dir
    skip_root = root / "skip"
    skip_files = ([p.relative_to(skip_root).as_posix()
                   for p in sorted(skip_root.rglob("*")) if p.is_file()]
                  if skip_root.is_dir() else [])

    # orphan feature npys
    rel_set = set(rels)
    orphans = {}
    for feat in FEATURES:
        fdir = root / feat
        if not fdir.is_dir():
            orphans[feat] = ["<feature dir missing>"]
            continue
        have = {p.relative_to(fdir).as_posix()[: -len(".npy")]
                for p in fdir.rglob("*.npy")}
        orph = sorted(have - rel_set)
        if orph:
            orphans[feat] = orph

    # expected count cross-check
    expect_issue = None
    if expect_n is not None:
        if len(rels) + len(skip_files) != expect_n:
            expect_issue = (f"audio({len(rels)}) + skip({len(skip_files)}) "
                            f"!= expected {expect_n} from build report")

    # f0 medians per speaker (feeds the spk-41 review)
    f0_by_spk = defaultdict(list)
    for r in recs:
        if "f0_med" in r:
            spk = r["rel"].split("/", 1)[0]
            f0_by_spk[spk].append(r["f0_med"])
    f0_spk_stats = {}
    for spk in sorted(f0_by_spk, key=lambda s: int(s.split("_")[0]) if s.split("_")[0].isdigit() else 999):
        v = np.array(f0_by_spk[spk])
        f0_spk_stats[spk] = {
            "n": int(len(v)),
            "med": round(float(np.median(v)), 1),
            "p10": round(float(np.percentile(v, 10)), 1),
            "p90": round(float(np.percentile(v, 90)), 1),
        }

    fail = (bool(missing) or bool(unreadable) or bool(pa_issues)
            or bool(missing_keys) or bool(bad_vals) or bool(skip_files)
            or expect_issue is not None
            or any(i.startswith(("shape:", "frame_spread", "f0_nonfinite"))
                   for r in recs for i in r["issues"]))
    result = {
        "root": str(root),
        "n_audio": len(rels),
        "expected_n": expect_n,
        "expect_issue": expect_issue,
        "missing_counts": dict(missing),
        "unreadable_counts": dict(unreadable),
        "n_files_with_issues": len(bad),
        "bad_files": [{"rel": r["rel"], "issues": r["issues"]} for r in bad[:200]],
        "pitch_aug": {
            "issues": pa_issues,
            "n_keys": len(pa) if pa is not None else None,
            "missing_keys": missing_keys[:200],
            "n_missing_keys": len(missing_keys),
            "extra_keys": extra_keys[:200],
            "n_extra_keys": len(extra_keys),
            "bad_vals": bad_vals[:200],
            "n_bad_vals": len(bad_vals),
        },
        "skip_files": skip_files,
        "orphans": {k: v[:100] for k, v in orphans.items()},
        "f0_by_spk": f0_spk_stats,
        "elapsed_sec": round(time.time() - t0, 1),
        "verdict": "FAIL" if fail else "PASS",
    }

    print(f"[*] {name}: issues on {len(bad)} files | missing={dict(missing) or '{}'} "
          f"unreadable={dict(unreadable) or '{}'}", flush=True)
    print(f"[*] {name}: pitch_aug keys={len(pa) if pa is not None else 'N/A'} "
          f"missing={len(missing_keys)} extra={len(extra_keys)} bad_vals={len(bad_vals)} "
          f"| skip={len(skip_files)} | orphans="
          f"{ {k: len(v) for k, v in orphans.items() if v} or '{}' }", flush=True)
    if expect_issue:
        print(f"[!] {name}: {expect_issue}", flush=True)
    for r in bad[:20]:
        print(f"    BAD {r['rel']}: {r['issues']}", flush=True)
    print(f"[*] {name}: f0 medians by spk: "
          + ", ".join(f"{s}:{d['med']}Hz" for s, d in f0_spk_stats.items()), flush=True)
    print(f"[*] {name}: VERDICT {result['verdict']}", flush=True)

    # per-file jsonl for audit trail (written by caller)
    result["_recs"] = recs
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/timbre_blend_stage1.yaml")
    ap.add_argument("--build-report", default=None,
                    help="optional Stage 1b build report for expected slice counts")
    ap.add_argument("--report",
                    default="reports/timbre_blend_stage1_preprocess_audit.json")
    ap.add_argument("--jsonl", default=None,
                    help="default: <train_path>/../feature_audit.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from logger import utils as lutils  # repo config loader (same as preprocess.py)
    cfg = lutils.load_config(args.config)
    exts = list(cfg.data.extensions)
    splits = {"train": Path(cfg.data.train_path), "val": Path(cfg.data.valid_path)}

    expect = {}
    if args.build_report:
        br = json.loads(Path(args.build_report).read_text())
        expect["train"] = sum(s["n_slices_train"] for s in br["per_singer"])
        expect["val"] = sum(s["n_slices_val"] for s in br["per_singer"])
        print(f"[*] expected slice counts from build report: {expect}", flush=True)

    results = {}
    all_recs = []
    for name, root in splits.items():
        res = audit_split(name, root, exts, args.workers,
                          expect_n=expect.get(name))
        for r in res.pop("_recs", []):
            r["split"] = name
            all_recs.append(r)
        results[name] = res

    jsonl_path = Path(args.jsonl) if args.jsonl else \
        splits["train"].parent / "feature_audit.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    verdict = "PASS" if all(v.get("verdict") == "PASS" for v in results.values()) \
        else "FAIL"
    report = {
        "config": args.config,
        "splits": results,
        "jsonl": str(jsonl_path),
        "verdict": verdict,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\nreport: {args.report}\njsonl:  {jsonl_path}", flush=True)
    print(f"AUDIT VERDICT: {verdict}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
