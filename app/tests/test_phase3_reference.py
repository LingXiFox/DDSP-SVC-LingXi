"""Phase 3 reference regression: pipeline.convert() on the bundle input.

Checks: model load once, features, Humanizer chain, synthesis, output
safety, seed repeatability, OFF identity, ON residual magnitudes.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from ddsp.vocoder import F0_Extractor, Volume_Extractor  # noqa: E402
from runtime.pipeline import LingXiSVCPipeline  # noqa: E402

SEED = 1234
INPUT = ROOT / "production" / "DDSP-SVC-LingXi-Production-v1" / "test" / "input" / "cold_test.wav"
PHASE2_MPS = ROOT / "outputs" / "reference" / "mac_reference_output_mps.wav"
OUT_ON = ROOT / "outputs" / "phase3_reference_on.wav"
OUT_ON_REPEAT = ROOT / "outputs" / "phase3_reference_on_repeat.wav"
OUT_OFF = ROOT / "outputs" / "phase3_reference_off.wav"


def _stats(path):
    data, sr = sf.read(str(path))
    data = np.asarray(data, dtype=np.float64)
    return {
        "sr": sr,
        "dur": len(data) / sr,
        "peak": float(np.max(np.abs(data))),
        "rms": float(np.sqrt(np.mean(data ** 2))),
        "finite": bool(np.isfinite(data).all()),
    }


def main():
    pipeline = LingXiSVCPipeline(device="mps")
    assert pipeline.model.ddsp_model.realism is not None
    print("[PASS] model load")

    r_on = pipeline.convert(str(INPUT), str(OUT_ON), realism_strength=1.0,
                            transpose=0.0, seed=SEED)
    assert r_on.finite and not r_on.clipping
    print("[PASS] ON convert", round(r_on.total_time, 1), "s RTF", round(r_on.rtf, 2))

    r_rep = pipeline.convert(str(INPUT), str(OUT_ON_REPEAT), realism_strength=1.0,
                             transpose=0.0, seed=SEED)
    a, _ = sf.read(str(OUT_ON))
    b, _ = sf.read(str(OUT_ON_REPEAT))
    seed_diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    print("seed repeat maxabsdiff:", seed_diff)
    assert seed_diff == 0.0, "same-seed repeats must be identical"
    print("[PASS] seed repeatability")

    r_off = pipeline.convert(str(INPUT), str(OUT_OFF), realism_strength=0.0,
                             transpose=0.0, seed=SEED)
    assert r_off.finite
    print("[PASS] OFF convert (control identity asserted inside convert)")

    # Full-file ON residual magnitudes (same metric family as Phase 2 probe).
    # NOTE: convert() leaves the shared strength at the last call's value,
    # so restore ON explicitly before measuring (each convert sets its own).
    pipeline.model.ddsp_model.realism_strength = 1.0
    from runtime.audio import load_audio

    loaded = load_audio(str(INPUT), pipeline.sample_rate)
    hop = pipeline.block_size
    f0 = F0_Extractor("rmvpe", pipeline.sample_rate, hop, 65, 800).extract(
        loaded.samples, uv_interp=True, device=pipeline.device)
    volume = Volume_Extractor(hop, pipeline.args.data.volume_smooth_size).extract(
        loaded.samples)
    f0_t = torch.from_numpy(f0).float().unsqueeze(-1).unsqueeze(0).to(pipeline.device)
    vol_t = torch.from_numpy(volume).float().unsqueeze(-1).unsqueeze(0).to(pipeline.device)
    units = pipeline.units_encoder.encode(
        torch.from_numpy(loaded.samples).float().unsqueeze(0).to(pipeline.device),
        pipeline.sample_rate, hop)
    n = min(units.size(1), f0_t.size(1), vol_t.size(1))
    with torch.no_grad():
        ad_f0, ad_vol = pipeline.model.adapt_controls(
            units[:, :n], f0_t[:, :n], vol_t[:, :n])
    voiced = (f0_t[:, :n, 0] > 0)
    cents = (1200 * torch.log2(
        ad_f0[:, :n, 0].clamp_min(1e-6) / f0_t[:, :n, 0].clamp_min(1e-6))).abs()
    vdb = (20 * (torch.log10(ad_vol[:, :n, 0].clamp_min(1e-9))
                 - torch.log10(vol_t[:, :n, 0].clamp_min(1e-9)))).abs()
    f0_res = float(cents[voiced].mean())
    vol_res = float(vdb[voiced].mean())
    print(f"F0_RESIDUAL_ON={f0_res:.2f}c VOLUME_RESIDUAL_ON={vol_res:.3f}dB")

    s_on, s_ref = _stats(OUT_ON), _stats(PHASE2_MPS)
    print("phase3:", {k: round(v, 4) if isinstance(v, float) else v for k, v in s_on.items()})
    print("phase2:", {k: round(v, 4) if isinstance(v, float) else v for k, v in s_ref.items()})
    assert s_on["sr"] == 44100 and abs(s_on["dur"] - 18.46) < 0.05
    assert abs(s_on["rms"] - s_ref["rms"]) / s_ref["rms"] < 0.15
    print("[PASS] phase2-vs-phase3 sanity")
    print("ALL_PHASE3_REFERENCE_CHECKS=PASS")


if __name__ == "__main__":
    main()
