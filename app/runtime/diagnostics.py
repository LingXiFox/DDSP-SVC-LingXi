"""Structured conversion diagnostics (data only, no CLI formatting)."""
from dataclasses import dataclass, field

import numpy as np

CLIP_THRESHOLD = 0.999


@dataclass
class ConversionResult:
    output_path: str
    duration: float
    sample_rate: int
    device: str
    module_devices: dict = field(default_factory=dict)
    realism_strength: float = 1.0
    transpose: float = 0.0
    seed: int = 0
    feature_time: float = 0.0
    synthesis_time: float = 0.0
    total_time: float = 0.0
    rtf: float = 0.0
    peak: float = 0.0
    peak_dbfs: float = float("-inf")
    clipping: bool = False
    finite: bool = False
    notes: list = field(default_factory=list)


def analyze_output(samples):
    """Safety checks on the rendered waveform. NaN/Inf is a hard error."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        raise RuntimeError("Synthesis produced empty audio.")
    if not np.isfinite(samples).all():
        raise RuntimeError("Synthesis produced NaN/Inf samples.")
    if not np.any(samples):
        raise RuntimeError("Synthesis produced all-zero audio.")
    peak = float(np.max(np.abs(samples)))
    clipping = bool(peak >= CLIP_THRESHOLD)
    peak_dbfs = float(20.0 * np.log10(peak)) if peak > 0 else float("-inf")
    return peak, peak_dbfs, clipping, True
