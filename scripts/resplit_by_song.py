"""Build the Stage-2 data tree with a WHOLE-SONG train/val split (gate-2 fix #2).

Stage 1 split at source-file level, and an OpenSinger source file is one segment
of a song, so every val song also had segments in train (see
reports/split_leakage_before_stage1.json: 162/162 songs leaked). Songs are
re-grouped here with check_song_split_leakage.song_key and each whole song is
assigned to exactly one split, so nothing straddles the boundary.

Only wav audio is linked; the six feature dirs and pitch_aug_dict.npy are
generated afterwards by preprocess.py. Linking features would be unsafe: np.save
truncates in place, so a hardlink would let Stage-2 preprocessing corrupt the
shared Stage-1 tree.

Output: <out>/{train,val}/audio/<spk_dir>/*.wav
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_song_split_leakage import song_key  # noqa: E402

FEATURE_SUBDIRS = ("mel", "f0", "units", "volume", "aug_mel", "aug_vol")


def seconds(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def group_by_song(paths: list[Path]) -> dict:
    """{song_key: {"files": {name: Path}, "sec": float}}"""
    songs: dict = defaultdict(lambda: {"files": {}, "sec": 0.0})
    for wav in sorted(paths):
        key = song_key(wav.name)
        songs[key]["files"][wav.name] = wav
        songs[key]["sec"] += seconds(wav)
    return dict(songs)


def song_order(songs: dict, seed: int) -> list:
    """Deterministic order, independent of dict/insertion order."""
    return sorted(songs, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())


def pick_val(songs: dict, seed: int, *, val_sec: float = 0.0, val_count: int = 0) -> list:
    if val_count:
        return song_order(songs, seed)[:val_count]
    chosen, acc = [], 0.0
    for key in song_order(songs, seed):
        if acc >= val_sec:
            break
        chosen.append(key)
        acc += songs[key]["sec"]
    return chosen


def place(dst_root: Path, split: str, spk_dir: str, name: str, src: Path) -> None:
    out = dst_root / split / "audio" / spk_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        out.unlink()
    os.link(src, out)  # same filesystem: 415 min of audio costs no extra blocks


def write_assignment(summary: dict, spk_dir: str, songs: dict, val_songs: list,
                     dst_root: Path) -> None:
    rows, train_min, val_min, n_train, n_val = [], 0.0, 0.0, 0, 0
    for key in sorted(songs):
        split = "val" if key in val_songs else "train"
        for name, path in sorted(songs[key]["files"].items()):
            place(dst_root, split, spk_dir, name, path)
        if split == "val":
            val_min += songs[key]["sec"]
            n_val += len(songs[key]["files"])
        else:
            train_min += songs[key]["sec"]
            n_train += len(songs[key]["files"])
        rows.append({"song": key, "split": split,
                     "slices": len(songs[key]["files"]),
                     "sec": round(songs[key]["sec"], 2)})
    summary["speakers"][spk_dir] = {
        "songs": len(songs), "val_songs": len(val_songs),
        "slices_train": n_train, "slices_val": n_val,
        "train_minutes": round(train_min / 60, 2), "val_minutes": round(val_min / 60, 2),
        "song_assignment": rows,
    }
    summary["totals"]["train_minutes"] += train_min / 60
    summary["totals"]["val_minutes"] += val_min / 60
    print(f" [{spk_dir}] {len(songs)} songs -> val {len(val_songs)} songs / "
          f"{val_min / 60:.1f} min; train {n_train} slices / {train_min / 60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1-root", default="data/timbre_blend_stage1")
    ap.add_argument("--virtual-root", default="datasets/private-timbre/sliced-v2",
                    help="target singer's sliced tree (train/audio + val/audio)")
    ap.add_argument("--virtual-spk-dir", default="13_lingxi",
                    help="audio subdir name; leading int is the 1-based spk_id")
    ap.add_argument("--out-root", default="data/timbre_blend_stage2")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--public-val-frac", type=float, default=0.05,
                    help="per-singer val share of total minutes, taken in whole songs")
    ap.add_argument("--virtual-val-songs", type=int, default=2)
    ap.add_argument("--report", default="reports/timbre_blend_stage2_split.json")
    args = ap.parse_args()

    stage1 = Path(args.stage1_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    if out_root.exists():
        print(f" [x] refusing to touch existing tree {out_root}", file=sys.stderr)
        return 2
    out_root.mkdir(parents=True)

    summary = {
        "seed": args.seed,
        "public_val_frac": args.public_val_frac,
        "virtual_val_songs": args.virtual_val_songs,
        "song_key_rule": "check_song_split_leakage.song_key",
        "totals": {"train_minutes": 0.0, "val_minutes": 0.0},
        "speakers": {},
    }

    for spk_dir in sorted(p.name for p in (stage1 / "train" / "audio").iterdir()
                          if p.is_dir()):
        paths = [p for split in ("train", "val")
                 for p in (stage1 / split / "audio" / spk_dir).glob("*.wav")]
        if not paths:
            print(f" [!] {spk_dir}: no wavs, skipped", file=sys.stderr)
            continue
        songs = group_by_song(paths)
        val_songs = pick_val(songs, args.seed,
                             val_sec=sum(s["sec"] for s in songs.values()) * args.public_val_frac)
        write_assignment(summary, spk_dir, songs, val_songs, out_root)

    vroot = Path(args.virtual_root).expanduser()
    songs = group_by_song([p for split in ("train", "val")
                           for p in (vroot / split / "audio").rglob("*.wav")])
    if not songs:
        print(f" [x] no wavs under {vroot}", file=sys.stderr)
        return 2
    val_songs = pick_val(songs, args.seed, val_count=args.virtual_val_songs)
    write_assignment(summary, args.virtual_spk_dir, songs, val_songs, out_root)

    for split in ("train", "val"):
        for feat in FEATURE_SUBDIRS:
            (out_root / split / feat).mkdir(parents=True, exist_ok=True)

    summary["totals"]["train_minutes"] = round(summary["totals"]["train_minutes"], 2)
    summary["totals"]["val_minutes"] = round(summary["totals"]["val_minutes"], 2)
    summary["totals"]["audio_files"] = sum(1 for _ in out_root.rglob("*.wav"))

    rep = Path(args.report).expanduser()
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n totals: train {summary['totals']['train_minutes']} min / "
          f"val {summary['totals']['val_minutes']} min, "
          f"{summary['totals']['audio_files']} wav -> {rep}")
    print(" next: preprocess.py -c <stage2 config>, then check_song_split_leakage.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
