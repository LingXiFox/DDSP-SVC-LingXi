"""lingxi-selfcheck: system health check without running a conversion.

Verifies: bundle completeness + SHA256, torch/MPS status, and one full
model load (ContentVec / RMVPE / Production P / NSF-HiFiGAN).
"""
import torch

from .device import resolve_device
from .pipeline import REQUIRED_FILES, LingXiSVCPipeline, PROJECT_ROOT


def main(argv=None):
    print("[1/3] Production bundle files")
    bundle_root = PROJECT_ROOT / "production" / "DDSP-SVC-LingXi-Production-v1"
    missing = [f for f in REQUIRED_FILES if not (bundle_root / f).exists()]
    if missing:
        print(f"  FAIL, missing: {missing}")
        return 1
    print("  bundle files: PRESENT")

    print("[2/3] Compute backend")
    print(f"  torch: {torch.__version__}")
    print(f"  mps built: {torch.backends.mps.is_built()}")
    print(f"  mps available: {torch.backends.mps.is_available()}")
    print(f"  auto resolves to: {resolve_device('auto')}")

    print("[3/3] Model load + bundle SHA256 (single load, eval mode)")
    try:
        pipeline = LingXiSVCPipeline(bundle_path=bundle_root, device="auto")
        pipeline.verify_bundle_checksums()
        print("  bundle SHA256: PASS")
        print(f"  device: {pipeline.device}")
        for name, dev in pipeline.module_devices.items():
            print(f"  {name}: {dev}")
        realism = pipeline.model.ddsp_model.realism is not None
        print(f"  Humanizer adapter: {'PRESENT' if realism else 'MISSING'}")
        if not realism:
            print("SELFCHECK=FAIL")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc}")
        print("SELFCHECK=FAIL")
        return 1

    print("SELFCHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
