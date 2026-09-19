#!/bin/bash
# LingXi SVC bootstrap: isolated venv + pinned runtime, no system pollution.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/environment/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -U pip wheel
"$VENV/bin/python" -m pip install -e "$ROOT/app"

echo "Done. Activate with:"
echo "  source $VENV/bin/activate"
echo "Then: lingxi-svc --help | lingxi-selfcheck"
