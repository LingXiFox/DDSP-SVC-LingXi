"""Device resolution for macOS local inference."""
import torch

VALID_DEVICES = ("auto", "mps", "cpu")


def resolve_device(requested="auto"):
    """Resolve a device string to a concrete torch device name.

    auto -> MPS when available, otherwise CPU. Never returns cuda:
    this runtime targets macOS local inference.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Unknown device '{requested}'. Expected one of {VALID_DEVICES}."
        )
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' was requested but is not available.")
    return requested


def module_device_report(device):
    """Phase 3 runs every module on the single resolved device."""
    modules = ("ContentVec", "RMVPE", "Humanizer", "DDSP", "Reflow", "Vocoder")
    return {name: device for name in modules}
