"""Phase 4 CLI regression: CLI output vs direct API output.

Same input/realism/transpose/seed/device must give identical audio:
the CLI is a wrapper and must not change inference results.
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

from runtime.pipeline import LingXiSVCPipeline  # noqa: E402

SEED = 1234
INPUT = ROOT / "inputs" / "cli_reference_input.wav"
CLI_ON = ROOT / "outputs" / "cli_reference.wav"
API_ON = ROOT / "outputs" / "cli_api_compare.wav"
CLI_OFF = ROOT / "outputs" / "cli_reference_off.wav"
CLI_T0 = ROOT / "outputs" / "cli_transpose0.wav"
CLI_T2 = ROOT / "outputs" / "cli_transpose2.wav"


def _voiced_median_f0(path):
    data, sr = sf.read(str(path))
    f0 = librosa.pyin(np.asarray(data, dtype=np.float32), fmin=50, fmax=800, sr=sr)[0]
    voiced = f0[~np.isnan(f0)]
    assert voiced.size > 0
    return float(np.median(voiced))


def main():
    pipeline = LingXiSVCPipeline(device="mps")
    result = pipeline.convert(str(INPUT), str(API_ON), realism_strength=1.0,
                              transpose=0.0, seed=SEED)
    assert result.finite
    a, sr = sf.read(str(CLI_ON))
    b, _ = sf.read(str(API_ON))
    assert sr == 44100 and len(a) == len(b)
    diff = float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                               - np.asarray(b, dtype=np.float64))))
    print(f"CLI_VS_API_MAX_ABS_DIFF={diff}")
    assert diff == 0.0

    off, _ = sf.read(str(CLI_OFF))
    assert np.isfinite(np.asarray(off)).all() and np.any(off)
    print("REALISM_OFF_IDENTITY=PASS (asserted inside convert during CLI run)")

    t0, _ = sf.read(str(CLI_T0))
    t2, _ = sf.read(str(CLI_T2))
    assert len(t0) == len(t2), "transpose must not change duration"
    ratio = _voiced_median_f0(CLI_T2) / _voiced_median_f0(CLI_T0)
    print(f"TRANSPOSE_F0_RATIO={ratio:.4f} (expected {2 ** (2 / 12):.4f})")
    assert 1.05 < ratio < 1.20
    print("TRANSPOSE_CHECK=PASS")
    print("ALL_PHASE4_CLI_CHECKS=PASS")


if __name__ == "__main__":
    main()
