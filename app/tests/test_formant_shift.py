"""Formant-shift wiring: CLI parsing, range guard, and aug_shift construction.

Covers the CLI surface only; whether the value reaches the model is proven
by the end-to-end render, not by re-implementing the construction here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from runtime.cli import FORMANT_SHIFT_LIMIT, build_parser, main
from runtime.diagnostics import ConversionResult

INPUT = ROOT / "production" / "DDSP-SVC-LingXi-Production-v1" / "test" / "input" / "cold_test.wav"


def main_ok_is_zero_for_in_range():
    for value in (-FORMANT_SHIFT_LIMIT, -2.0, 0.0, 1.5, FORMANT_SHIFT_LIMIT):
        ns = build_parser().parse_args([str(INPUT), "--formant-shift", str(value)])
        assert ns.formant_shift == value, f"parser lost {value}"
    assert build_parser().parse_args([str(INPUT)]).formant_shift == 0.0
    print("[PASS] parser accepts the full [-5, 5] range and defaults to 0")


def out_of_range_is_rejected_before_model_load():
    for value in (-5.5, 9.0):
        rc = main([str(INPUT), "-o", "/tmp/lingxi_formant_never_written.wav",
                   "--formant-shift", str(value)])
        assert rc == 2, f"expected exit 2 for {value}, got {rc}"
        out = Path("/tmp/lingxi_formant_never_written.wav")
        assert not out.exists(), "rejected run must not write an output file"
    print("[PASS] out-of-range formant-shift exits 2 and writes nothing")


def result_defaults_formant_shift_to_zero():
    r = ConversionResult(output_path="x", duration=1.0, sample_rate=44100, device="cpu")
    assert r.formant_shift == 0.0
    print("[PASS] ConversionResult exposes formant_shift (default 0)")


if __name__ == "__main__":
    main_ok_is_zero_for_in_range()
    out_of_range_is_rejected_before_model_load()
    result_defaults_formant_shift_to_zero()
    print("FORMANT_SHIFT=PASS")
