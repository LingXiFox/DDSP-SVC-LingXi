"""Audio loading and normalization for inference input."""
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf

SUPPORTED_SUFFIXES = (".wav", ".flac")


@dataclass
class LoadedAudio:
    samples: np.ndarray  # float32 mono at target sample rate
    sample_rate: int
    source_sample_rate: int
    source_channels: int
    notes: list = field(default_factory=list)


def load_audio(path, target_sample_rate):
    """Load WAV/FLAC, downmix to mono, resample to the model rate."""
    suffix = str(path).lower()
    if not suffix.endswith(SUPPORTED_SUFFIXES):
        raise ValueError(
            f"Unsupported input '{path}'. Expected one of {SUPPORTED_SUFFIXES}."
        )
    try:
        audio, source_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not read audio '{path}': {exc}") from exc
    notes = []
    channels = audio.shape[1]
    if channels > 1:
        audio = np.mean(audio, axis=1, keepdims=True)
        notes.append(f"stereo input downmixed to mono ({channels}ch)")
    samples = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    if source_sr != target_sample_rate:
        samples = librosa.resample(
            samples, orig_sr=source_sr, target_sr=target_sample_rate
        ).astype(np.float32)
        notes.append(f"resampled {source_sr}Hz -> {target_sample_rate}Hz")
    if samples.size == 0:
        raise ValueError(f"Input audio '{path}' is empty.")
    return LoadedAudio(
        samples=samples,
        sample_rate=int(target_sample_rate),
        source_sample_rate=int(source_sr),
        source_channels=int(channels),
        notes=notes,
    )
