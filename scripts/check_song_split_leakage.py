"""Prove that a train/val tree shares no song across splits (exit 1 on any leak).

Background: OpenSinger lays out <singer>_<song>/<singer>_<song>_<seg>.wav, so one
"source file" is a segment of one song. Splitting at source-file level therefore
lets different segments of the SAME song land on opposite sides of the boundary.
This script maps slices back to their song and reports any cross-split overlap.

Song key: strip the trailing slice suffix, then the trailing segment index.
  36_一笑倾城_18__s0.wav -> 36_一笑倾城      r13_00_15s0.wav -> r13_00
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SLICE_SUFFIX = re.compile(r"_*s\d+$")   # "__s0" (stage-1 builder) or "s0" (virtual set)
SEGMENT_SUFFIX = re.compile(r"_\d+$")   # OpenSinger segment index / r13_<song>_<seg>


def song_key(name: str) -> str:
    stem = SLICE_SUFFIX.sub("", Path(name).stem)
    return SEGMENT_SUFFIX.sub("", stem)


def scan_split(root: Path, split: str) -> dict:
    """{(speaker_group, song_key): [slice rel paths]}

    Accepts both layouts: audio/<speaker_dir>/*.wav (multi-speaker trees) and a
    flat audio/*.wav (single-speaker trees), in which the group is "(flat)".
    """
    audio = root / split / "audio"
    songs = defaultdict(list)
    for wav in sorted(audio.rglob("*.wav")) if audio.is_dir() else []:
        rel = wav.relative_to(audio)
        group = rel.parts[0] if len(rel.parts) > 1 else "(flat)"
        songs[(group, song_key(wav.name))].append(str(wav.relative_to(root)))
    return dict(songs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="data root with train/audio + val/audio")
    ap.add_argument("--report", default="", help="also write the JSON report here")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    train, val = scan_split(root, "train"), scan_split(root, "val")
    train_slices = sorted(p for v in train.values() for p in v)
    val_slices = sorted(p for v in val.values() for p in v)
    shared = sorted(set(train) & set(val))
    slice_overlap = sorted({p.split('/audio/', 1)[1] for p in train_slices} &
                           {p.split('/audio/', 1)[1] for p in val_slices})

    by_singer = defaultdict(lambda: {"songs_train": 0, "songs_val": 0, "leaked_songs": 0})
    for spk, song in set(train) | set(val):
        row = by_singer[spk]
        if (spk, song) in train:
            row["songs_train"] += 1
        if (spk, song) in val:
            row["songs_val"] += 1
    for spk, song in shared:
        by_singer[spk]["leaked_songs"] += 1

    report = {
        "root": str(root),
        "splits": {
            "train": {"songs": len(train), "slices": len(train_slices)},
            "val": {"songs": len(val), "slices": len(val_slices)},
        },
        "songs_shared_across_splits": len(shared),
        "shared_song_detail": [
            {"speaker_dir": spk, "song": song,
             "train_slices": len(train[(spk, song)]), "val_slices": len(val[(spk, song)])}
            for spk, song in shared
        ],
        "identical_slice_paths_in_both_splits": len(slice_overlap),
        "per_speaker": {k: dict(v) for k, v in sorted(by_singer.items())},
        "verdict": "PASS" if train_slices and val_slices and not shared and not slice_overlap else "FAIL",
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        out = Path(args.report).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSPLIT-LEAKAGE VERDICT: {report['verdict']}", file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
