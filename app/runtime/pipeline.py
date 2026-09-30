"""LingXi SVC Phase 3 pipeline: orchestration over the repo inference stack.

Math semantics mirror main_reflow.py exactly:
original F0 -> transpose (semitones) -> Humanizer -> DDSP/Reflow,
final NSF-HiFiGAN receives the adapted F0.
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

def _find_project_root():
    """Locate the single LingXi workspace root (repo + production + runtime).

    Resolution order: LINGXI_PROJECT_ROOT env var, then walk up from this
    file for a directory containing both production/ and ddsp/. Editable
    installs keep __file__ inside the workspace, so any CWD works.
    """
    env = os.environ.get("LINGXI_PROJECT_ROOT")
    if env:
        candidate = Path(env).resolve()
        if (candidate / "production").is_dir() and (candidate / "ddsp").is_dir():
            return candidate
        raise RuntimeError(
            f"LINGXI_PROJECT_ROOT={env} does not look like the LingXi workspace.")
    for parent in Path(__file__).resolve().parents:
        if (parent / "production").is_dir() and (parent / "ddsp").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate the LingXi workspace root. Set LINGXI_PROJECT_ROOT "
        "to the directory containing production/ and ddsp/.")


PROJECT_ROOT = _find_project_root()
APP_ROOT = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddsp.core import upsample
from ddsp.vocoder import F0_Extractor, Units_Encoder, Volume_Extractor
from reflow.vocoder import load_model_vocoder
from slicer import Slicer

from .audio import load_audio
from .device import module_device_report, resolve_device
from .diagnostics import ConversionResult, analyze_output
from .features import FeatureBundle, align_frames, check_finite, to_device_float32

DEFAULT_BUNDLE_NAME = "DDSP-SVC-LingXi-Production-v1"
REQUIRED_FILES = (
    "model/production_v1.pt",
    "model/config.yaml",
    "config/production.yaml",
    "assets/contentvec/pytorch_model.bin",
    "assets/nsf_hifigan/model",
    "assets/nsf_hifigan/config.json",
    "metadata/model.json",
)

# Verified-production inference constants (same defaults as main_reflow.py).
_SLICER_DB = -40
_SLICER_MIN_LEN = 5000
_MASK_THRESHOLD_DB = -60.0
_SPk_ID = 1
# aug_shift was trained as keyshift ~ U(-5, 5) semitones (preprocess.py), so
# the embedding has no support beyond that range.
FORMANT_SHIFT_LIMIT = 5.0


@contextlib.contextmanager
def _bundle_workdir(bundle_root):
    previous = os.getcwd()
    os.chdir(bundle_root)
    try:
        yield
    finally:
        os.chdir(previous)


def _cross_fade(a, b, idx):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    np.copyto(dst=result[:idx], src=a[:idx])
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[:fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


def _split(audio, sample_rate, hop_size):
    slicer = Slicer(sr=sample_rate, threshold=_SLICER_DB, min_length=_SLICER_MIN_LEN)
    chunks = dict(slicer.slice(audio))
    result = []
    for _, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                    start_frame,
                    audio[int(start_frame * hop_size): int(end_frame * hop_size)],
                ))
    return result


class LingXiSVCPipeline:
    """Single-load Production v1 inference pipeline."""

    def __init__(self, bundle_path=None, device="auto", verify_bundle=False):
        if bundle_path is None:
            bundle_path = PROJECT_ROOT / "production" / DEFAULT_BUNDLE_NAME
        self.bundle_root = Path(bundle_path).resolve()
        missing = [f for f in REQUIRED_FILES if not (self.bundle_root / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Production bundle incomplete at {self.bundle_root}. "
                f"Missing: {missing}"
            )
        if not (self.bundle_root / "assets" / "rmvpe" / "model.pt").exists():
            raise FileNotFoundError("Production bundle is missing assets/rmvpe/model.pt.")
        if verify_bundle:
            self.verify_bundle_checksums()
        with open(self.bundle_root / "config" / "production.yaml") as f:
            self.config = yaml.safe_load(f)
        with open(self.bundle_root / "metadata" / "model.json") as f:
            self.model_info = json.load(f)
        self.device = resolve_device(device)
        self.module_devices = module_device_report(self.device)
        # Model/vocoder construction reads bundle-relative asset paths,
        # so resolve everything while parked at the bundle root.
        with _bundle_workdir(self.bundle_root):
            self.model, self.vocoder, self.args = load_model_vocoder(
                str(self.bundle_root / "model" / "production_v1.pt"),
                device=self.device,
            )
        self.model.eval()
        if self.model.ddsp_model.realism is None:
            raise RuntimeError("Production bundle loaded without the Humanizer adapter.")
        encoder_ckpt = self.args.data.encoder_ckpt
        if not os.path.isabs(encoder_ckpt):
            encoder_ckpt = str(self.bundle_root / encoder_ckpt)
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=self.args.data.cnhubertsoft_gate,
            device=self.device,
        )
        self.sample_rate = int(self.args.data.sampling_rate)
        self.block_size = int(self.args.data.block_size)
        self.default_strength = float(self.args.model.realism.get("strength", 1.0))
        self.infer_step = int(self.args.infer.infer_step)
        self.method = str(self.args.infer.method)
        self.t_start = float(self.args.model.t_start or 0.0)

    def verify_bundle_checksums(self):
        import hashlib

        sums_file = self.bundle_root / "metadata" / "SHA256SUMS"
        failures = []
        with open(sums_file) as f:
            for line in f:
                expected, _, rel = line.strip().partition("  ")
                if not rel:
                    continue
                target = self.bundle_root / rel
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if digest != expected:
                    failures.append(rel)
        if failures:
            raise RuntimeError(f"Bundle checksum mismatch: {failures}")
        return True

    def convert(self, input_path, output_path=None, realism_strength=None,
                transpose=0.0, formant_shift=0.0, seed=1234):
        total_start = time.perf_counter()
        if realism_strength is None:
            realism_strength = self.default_strength
        realism_strength = float(realism_strength)
        if not 0.0 <= realism_strength <= 1.0:
            raise ValueError("realism_strength must be within [0.0, 1.0].")
        formant_shift = float(formant_shift)
        if not -FORMANT_SHIFT_LIMIT <= formant_shift <= FORMANT_SHIFT_LIMIT:
            raise ValueError(
                "formant_shift must be within "
                f"[{-FORMANT_SHIFT_LIMIT}, {FORMANT_SHIFT_LIMIT}]."
            )
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_name(input_path.stem + "_lingxi.wav")
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing {output_path}.")

        # load_audio normalizes to the model rate, so every extractor below runs
        # on the target-rate frame grid; hop_size is block_size, not a scaled
        # source-rate hop (upstream keeps the source rate and scales instead).
        loaded = load_audio(input_path, self.sample_rate)
        audio = loaded.samples
        work_sr = self.sample_rate
        hop_size = self.block_size
        win_size = int(self.args.data.volume_smooth_size)

        feature_start = time.perf_counter()
        with _bundle_workdir(self.bundle_root):
            pitch_extractor = F0_Extractor(
                self.args.data.f0_extractor, work_sr, hop_size,
                float(self.args.data.f0_min), float(self.args.data.f0_max))
            f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device)
            volume_extractor = Volume_Extractor(hop_size, win_size)
            volume = volume_extractor.extract(audio)
            mask = (volume > 10 ** (_MASK_THRESHOLD_DB / 20)).astype("float")
            mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
            mask = upsample(mask, self.block_size).squeeze(-1)
            volume_t = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
            f0_t = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
            units = self.units_encoder.encode(
                torch.from_numpy(audio).float().unsqueeze(0).to(self.device),
                work_sr, hop_size)
            units, f0_t, volume_t, frame_count = align_frames(units, f0_t, volume_t)
            units = check_finite("units", units)
            f0_t = check_finite("f0", f0_t)
            volume_t = check_finite("volume", volume_t)
            features = FeatureBundle(
                units=units, f0=f0_t, volume=volume_t, frame_count=frame_count)
        feature_time = time.perf_counter() - feature_start

        # Transpose BEFORE the Humanizer (verified production semantics).
        transposed_f0 = features.f0 * 2 ** (float(transpose) / 12)
        transposed_f0 = to_device_float32(transposed_f0, self.device)
        spk_id = torch.LongTensor(np.array([[_SPk_ID]])).to(self.device)
        aug_shift_t = torch.full(
            (1, 1), formant_shift, dtype=torch.float32, device=self.device)
        self.model.ddsp_model.realism_strength = realism_strength

        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(int(seed))

        synthesis_start = time.perf_counter()
        segments = _split(audio, work_sr, hop_size)
        if not segments:
            raise RuntimeError("Slicer produced no voiced segments.")
        result = np.zeros(0)
        current_length = 0
        with torch.no_grad(), _bundle_workdir(self.bundle_root):
            for start_frame, segment in segments:
                seg_input = torch.from_numpy(segment).float().unsqueeze(0).to(self.device)
                seg_units = self.units_encoder.encode(seg_input, work_sr, hop_size)
                width = seg_units.size(1)
                seg_f0 = transposed_f0[:, start_frame: start_frame + width, :]
                seg_volume = features.volume[:, start_frame: start_frame + width, :]
                seg_mel = self.model(
                    seg_units, seg_f0, seg_volume, spk_id=spk_id,
                    spk_mix_dict=None, aug_shift=aug_shift_t,
                    vocoder=self.vocoder, infer=True,
                    infer_step=self.infer_step, method=self.method,
                    t_start=self.t_start, use_tqdm=False)
                adapted_f0, adapted_volume = self.model.adapt_controls(
                    seg_units, seg_f0, seg_volume)
                adapted_f0 = check_finite("adapted_f0", adapted_f0)
                adapted_volume = check_finite("adapted_volume", adapted_volume)
                if realism_strength > 0:
                    if torch.equal(adapted_f0, seg_f0):
                        raise RuntimeError(
                            "Humanizer produced no F0 change at strength "
                            f"{realism_strength}.")
                else:
                    if not torch.equal(adapted_f0, seg_f0) or not torch.equal(
                            adapted_volume, seg_volume):
                        raise RuntimeError(
                            "OFF-strength controls are not identical to originals.")
                seg_output = self.vocoder.infer(seg_mel, adapted_f0)
                seg_output = seg_output * mask[
                    :, start_frame * self.block_size:
                    (start_frame + width) * self.block_size]
                seg_output = seg_output.squeeze().cpu().numpy()
                silent_length = round(start_frame * self.block_size) - current_length
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = _cross_fade(result, seg_output, current_length + silent_length)
                current_length = current_length + silent_length + len(seg_output)
        synthesis_time = time.perf_counter() - synthesis_start

        peak, peak_dbfs, clipping, finite = analyze_output(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), result, self.sample_rate)
        total_time = time.perf_counter() - total_start
        duration = len(result) / self.sample_rate
        notes = list(loaded.notes)
        if clipping:
            notes.append("WARNING: output clipping detected (>= 0.999).")
        return ConversionResult(
            output_path=str(output_path),
            duration=duration,
            sample_rate=self.sample_rate,
            device=self.device,
            module_devices=dict(self.module_devices),
            realism_strength=realism_strength,
            transpose=float(transpose),
            formant_shift=formant_shift,
            seed=int(seed),
            feature_time=feature_time,
            synthesis_time=synthesis_time,
            total_time=total_time,
            rtf=total_time / max(duration, 1e-6),
            peak=peak,
            peak_dbfs=peak_dbfs,
            clipping=clipping,
            finite=finite,
            notes=notes,
        )
