"""lingxi-svc command line interface.

Thin wrapper over LingXiSVCPipeline: no model math lives here.
"""
import argparse
import sys
import traceback
from pathlib import Path

from .device import VALID_DEVICES
from .pipeline import DEFAULT_BUNDLE_NAME, FORMANT_SHIFT_LIMIT, LingXiSVCPipeline

DEFAULT_SEED = 1234
SUPPORTED_SUFFIXES = (".wav", ".flac")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lingxi-svc",
        description=(
            "LingXi SVC Production v1 local offline voice conversion. "
            "Give it a dry vocal file, get back a converted WAV."
        ),
        epilog=(
            "examples:\n"
            "  lingxi-svc vocal.wav\n"
            "  lingxi-svc vocal.wav -o converted.wav --realism 0.85 --transpose 0\n"
            "  lingxi-svc male_stem.wav --transpose 12 --formant-shift 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="dry vocal input file (wav/flac)")
    parser.add_argument("-o", "--output", default=None,
                        help="output wav path (default: <input>_lingxi.wav next to input)")
    parser.add_argument("--realism", type=float, default=None,
                        help="Humanizer strength 0.0-1.0 "
                             "(default: production.yaml value, currently 1.0)")
    parser.add_argument("--transpose", type=float, default=0.0,
                        help="pitch shift in semitones, F0 only (default: 0)")
    parser.add_argument("--formant-shift", dest="formant_shift", type=float,
                        default=0.0, metavar="SEMITONES",
                        help="formant shift in semitones, within [-5, 5]; "
                             "brightens/darkens timbre without moving the pitch "
                             "(default: 0)")
    parser.add_argument("--device", default="auto", choices=list(VALID_DEVICES),
                        help="compute device (default: auto -> MPS if available, else CPU)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"deterministic seed (default: {DEFAULT_SEED})")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output file")
    parser.add_argument("--verbose", action="store_true",
                        help="show full traceback on failure")
    return parser


def _fail(message, verbose, exc=None, code=1):
    print(f"Error: {message}", file=sys.stderr)
    if verbose and exc is not None:
        traceback.print_exception(exc)
    return code


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = args.verbose

    input_path = Path(args.input)
    if not input_path.is_file():
        return _fail(f"input file not found: {args.input}", verbose, code=2)
    if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return _fail(
            f"unsupported input format '{input_path.suffix}'. "
            f"Expected one of {list(SUPPORTED_SUFFIXES)}.", verbose, code=2)
    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_lingxi.wav")
    else:
        output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _fail(f"output directory is not writable: {output_path.parent}",
                      verbose, exc)
    if output_path.exists() and not args.force:
        return _fail(f"output already exists: {output_path}\n"
                      f"Use --force to overwrite.", verbose, code=2)
    if args.realism is not None and not 0.0 <= args.realism <= 1.0:
        return _fail("realism must be within [0.0, 1.0].", verbose, code=2)
    if not -FORMANT_SHIFT_LIMIT <= args.formant_shift <= FORMANT_SHIFT_LIMIT:
        return _fail(
            f"formant-shift must be within [-{FORMANT_SHIFT_LIMIT:g}, "
            f"{FORMANT_SHIFT_LIMIT:g}] semitones.", verbose, code=2)

    print("LingXi SVC Production v1")
    print("Loading model...")
    try:
        pipeline = LingXiSVCPipeline(device=args.device)
    except FileNotFoundError as exc:
        return _fail(f"Production model could not be loaded: {exc}", verbose, exc)
    except RuntimeError as exc:
        message = str(exc)
        if "mps" in message.lower():
            message += " Try --device cpu."
        return _fail(message, verbose, exc)

    strength = args.realism if args.realism is not None else pipeline.default_strength
    print(f"Device: {pipeline.device.upper()}")
    print("Processing...")
    try:
        result = pipeline.convert(
            str(input_path), str(output_path),
            realism_strength=strength,
            transpose=args.transpose,
            formant_shift=args.formant_shift,
            seed=args.seed,
        )
    except FileExistsError as exc:
        return _fail(str(exc), verbose, exc, code=2)
    except Exception as exc:  # noqa: BLE001 - mapped to short UX below
        return _fail(f"conversion failed: {exc}", verbose, exc)

    print("Writing output...")
    print("Done.")
    print(f"Model: Production v1 ({DEFAULT_BUNDLE_NAME})")
    print(f"Device: {result.device.upper()}")
    print(f"Input: {input_path}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Realism: {result.realism_strength}")
    print(f"Transpose: {result.transpose}")
    print(f"Formant shift: {result.formant_shift}")
    print(f"Seed: {result.seed}")
    print()
    print(f"Processing time: {result.total_time:.1f}s")
    print(f"RTF: {result.rtf:.2f}")
    print()
    print(f"Peak: {result.peak_dbfs:.1f} dBFS")
    print(f"Clipping: {'YES - output limited by input level, check gain' if result.clipping else 'NO'}")
    print(f"Finite: {'PASS' if result.finite else 'FAIL'}")
    for note in result.notes:
        print(f"Note: {note}")
    print()
    print(f"Output:\n{result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
